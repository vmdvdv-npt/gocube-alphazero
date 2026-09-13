from __future__ import annotations

import json
from pathlib import Path

import pytest

from gocube_golden.neural import Torus5GoldenGraphNetV2, instantiate_model_from_metadata
from gocube_golden.standard import (
    CURRENT_ALIAS,
    CURRENT_PRESET_ID,
    LegacyGoldenStandardError,
    build_run_metadata,
    build_torus5_model,
    config_fingerprint,
    load_universal_training_config,
    resolve_torus5_golden,
    validate_run_metadata,
    write_run_manifest,
)
from gocube_golden.torus9_contract import load_torus9_profile


ROOT = Path(__file__).resolve().parents[1]


def test_current_alias_resolves_to_one_concrete_v2_preset():
    catalog = json.loads((ROOT / "configs/gocube/golden_standards.json").read_text())
    current = [entry for entry in catalog["presets"] if entry["status"] == "current"]

    assert len(current) == 1
    assert current[0]["preset_id"] == CURRENT_PRESET_ID
    assert catalog["current_alias"] == CURRENT_ALIAS
    assert resolve_torus5_golden("current").preset_id == CURRENT_PRESET_ID
    assert resolve_torus5_golden(CURRENT_ALIAS).preset_id == CURRENT_PRESET_ID


def test_current_v2_pins_network_komi_and_board_identity():
    resolved = resolve_torus5_golden()
    board = resolved.config["board"]

    assert board["topology"]["width"] == 5
    assert board["topology"]["height"] == 5
    assert board["network"]["channels"] == 48
    assert board["network"]["hidden"] == 48
    assert board["network"]["blocks"] == 6
    assert board["rules"]["komi"] == 0.5
    assert "7.5" not in (ROOT / "configs/gocube/torus5_golden_v2.json").read_text()


def test_universal_settings_are_projected_from_current_torus9_without_board_capacity():
    universal = load_universal_training_config()
    torus9 = load_torus9_profile()
    settings = universal["settings"]

    assert universal["source"]["profile_id"] == torus9["profile_id"]
    assert settings["training"]["batch_size"] == torus9["training"]["batch_size"] == 64
    assert settings["training"]["optimizer_steps_per_iteration"] == 80
    assert settings["replay"]["generations"] == torus9["replay"]["generations"] == 3
    assert settings["self_play"]["simulations"] == torus9["self_play"]["simulations"] == 64
    assert settings["self_play"]["temperature_plies"] == [1, 8]
    assert settings["self_play"]["fast_search"] is False
    assert settings["arena"]["root_noise"] is False
    assert settings["inference"] == {
        "self_play": {"batch_size": 1, "coalescing": False},
        "arena": {"batch_size": 1, "coalescing": False},
    }
    assert "network" not in settings
    assert "topology" not in settings


def test_legacy_presets_are_blocked_by_default_but_remain_readable_with_override():
    with pytest.raises(LegacyGoldenStandardError, match="allow-legacy-config"):
        resolve_torus5_golden("gocube-torus5-golden-v1-legacy")

    legacy = resolve_torus5_golden(
        "gocube-torus5-golden-v1-legacy",
        allow_legacy_config=True,
    )
    assert legacy.is_legacy
    assert legacy.config["legacy_source"]["network"]["hidden"] == 64
    assert legacy.config["legacy_source"]["network"]["blocks"] == 4
    model = build_torus5_model(
        "gocube-torus5-golden-v1-legacy",
        allow_legacy_config=True,
    )
    assert model.hidden == 64
    assert model.blocks_count == 4


def test_current_model_is_48_by_6_and_checkpoint_metadata_can_reconstruct_it():
    model = build_torus5_model()
    assert isinstance(model, Torus5GoldenGraphNetV2)
    assert model.architecture_config["hidden"] == 48
    assert model.architecture_config["blocks"] == 6
    restored = instantiate_model_from_metadata(
        {"model_variant": "wdl", "architecture_config": model.architecture_config}
    )
    assert isinstance(restored, Torus5GoldenGraphNetV2)
    assert restored.architecture_config == model.architecture_config


def test_current_run_manifest_contains_concrete_version_full_config_and_git_identity(tmp_path):
    metadata = build_run_metadata(
        run_name="torus5-manifest-test",
        preset="current",
        argv=["python", "-m", "tools.torus5_golden"],
    )
    validate_run_metadata(metadata)

    assert metadata["requested_preset"] == "current"
    assert metadata["resolved_preset_id"] == CURRENT_PRESET_ID
    assert metadata["golden_standard"]["concrete_version"] == CURRENT_PRESET_ID
    assert metadata["network_channels"] == 48
    assert metadata["network_blocks"] == 6
    assert metadata["board_size"] == [5, 5]
    assert metadata["komi"] == 0.5
    assert len(metadata["git_sha"]) == 40
    assert len(metadata["git_tree_sha"]) == 40
    assert config_fingerprint(metadata["resolved_config"]) == metadata["resolved_config_sha256"]

    manifest_path = tmp_path / "run-manifest.json"
    written = write_run_manifest(
        manifest_path,
        run_name="torus5-manifest-test",
        preset="current",
        argv=["test"],
    )
    assert json.loads(manifest_path.read_text()) == written


def test_legacy_stage_runners_are_not_default_launch_paths():
    stage3 = (ROOT / "tools/torus_golden_stage3_train.py").read_text()
    stage4 = (ROOT / "tools/torus_golden_stage4.py").read_text()
    current_launcher = (ROOT / "tools/torus5_golden.py").read_text()

    assert "--allow-legacy-config" in stage3
    assert "--allow-legacy-config" in stage4
    assert 'parser.add_argument("--preset", default="current")' in current_launcher
    assert "torus_golden_stage3_train" not in current_launcher
    assert "torus_golden_stage4" not in current_launcher
