from __future__ import annotations

import json
from pathlib import Path

import pytest

from gocube_golden.torus9_contract import load_torus9_current_profile
from tools.arena_engine import ArenaExecutionConfig
from tools.arena_profiles.torus9 import Torus9ArenaProfile


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROFILE = ROOT / "configs/gocube/torus9_golden_current_v3.json"
REFERENCE_PROFILE_FINGERPRINT = "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775"
CURRENT_PROFILE_FINGERPRINT = "sha256:d3620fc36600d36753a4bb51a9810b7ea21fe82a43f70a43b6363485dc9684e3"
OBSERVATION_FINGERPRINT = "sha256:e5792b409199dfe2c25ac6f681e4ca29ed73cdf4b7d53a61df634f70d1fa415f"
TARGET_FINGERPRINT = "sha256:02ab244688534b271473302ab4edf00516b91d43fb91b8c9e592d2a8de63dfb5"
SELFPLAY_FINGERPRINT = "sha256:22a4e4dd37d70bd3d712b909476120b96385802ec874b99358fab256c4e3351f"


def test_exactly_one_torus9_profile_is_current():
    current = []
    for path in sorted((ROOT / "configs/gocube").glob("torus9*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "current":
            current.append(path.relative_to(ROOT).as_posix())
    assert current == ["configs/gocube/torus9_golden_current_v3.json"]


def test_current_torus9_scientific_contract_is_reference_locked():
    profile = load_torus9_current_profile()

    assert profile["profile_fingerprint"] == CURRENT_PROFILE_FINGERPRINT
    assert profile["rules"]["komi"] == 0.5
    assert profile["observation"]["shape"] == [6, 81]
    assert profile["observation"]["action_count"] == 82
    assert profile["observation"]["pass_index"] == 81
    assert profile["observation"]["fingerprint"] == OBSERVATION_FINGERPRINT
    assert profile["target"]["fingerprint"] == TARGET_FINGERPRINT

    assert profile["network"] == {
        "architecture_id": "GoldenGraphNetV2-Torus9",
        "hidden": 80,
        "blocks": 8,
        "input_channels": 6,
        "point_count": 81,
        "heads": {
            "policy": [82],
            "value": [3],
            "ownership": [81, 3],
            "score": [1],
        },
        "ownership": True,
        "score": True,
        "explicit_symmetry_augmentation": False,
    }

    self_play = profile["self_play"]
    assert self_play["fingerprint"] == SELFPLAY_FINGERPRINT
    assert {
        "games_per_iteration": self_play["games_per_iteration"],
        "mcts_simulations": self_play["mcts_simulations"],
        "cpuct": self_play["cpuct"],
        "fpu": self_play["fpu"],
        "root_noise": self_play["root_noise"],
        "dirichlet_epsilon": self_play["dirichlet_epsilon"],
        "dirichlet_alpha": self_play["dirichlet_alpha"],
        "fast_search": self_play["fast_search"],
        "resign": self_play["resign"],
        "watchdog": self_play["watchdog"],
        "komi": self_play["komi"],
    } == {
        "games_per_iteration": 64,
        "mcts_simulations": 64,
        "cpuct": 1.25,
        "fpu": 0.0,
        "root_noise": True,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.11,
        "fast_search": False,
        "resign": False,
        "watchdog": 500,
        "komi": 0.5,
    }

    assert profile["training"] == {
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 64,
        "optimizer_steps_per_iteration": 80,
        "samples_consumed_per_iteration": 5120,
        "lr_scheduler": None,
        "model_gating": False,
        "ownership_loss": "ON",
        "score_loss": "ON",
    }
    assert profile["replay"] == {
        "window": "rolling last 3 generations",
        "generations": 3,
        "cap": 20_000,
        "sampling": "deterministic / reproducible",
    }


def test_current_profile_rejects_noncanonical_komi(tmp_path: Path):
    profile = json.loads(CURRENT_PROFILE.read_text(encoding="utf-8"))
    profile["rules"]["komi"] = 15 / 2
    candidate = tmp_path / "invalid-torus9.json"
    candidate.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="komi"):
        load_torus9_current_profile(candidate, verify_fingerprint=False)


def test_retired_torus9_profiles_cannot_be_selected_by_current_launcher():
    launcher = (ROOT / "tools/continue_torus9_golden_m1_m100.py").read_text(encoding="utf-8")
    runtime = (ROOT / "tools/_frozen_continue_torus9_golden_m1_m100.py").read_text(encoding="utf-8")
    assert "_frozen_continue_torus9_golden_m1_m100.py" in launcher
    assert "torus9_golden_current_v3.json" in runtime
    for retired in (
        "torus9_golden_learning_v1.json",
        "torus9_ownership_ab_v1.json",
        "torus_golden_training_v1.json",
        "torus_golden_training_v2_data_rich.json",
    ):
        assert retired not in launcher
        assert retired not in runtime
    for retired_symbol in ("Coach", "SelfPlayAgent", "NNetWrapper"):
        assert retired_symbol not in launcher
        assert retired_symbol not in runtime


def test_current_arena_scientific_duplicate_is_validated_against_current_profile():
    profile = load_torus9_current_profile()
    arena = profile["arena"]
    adapter = Torus9ArenaProfile()
    contract = adapter.scientific_contract(
        ArenaExecutionConfig(device="cpu", strict_production=False)
    )
    assert contract["komi"] == arena["komi"] == 0.5
    assert contract["simulations"] == arena["mcts_simulations"]
    assert contract["cpuct"] == arena["cpuct"]
    assert contract["fpu"] == arena["fpu"]
    assert contract["noise"] == arena["noise"]
    assert contract["temperature"] == arena["temperature"]
    assert contract["fast_search"] == arena["fast_search"]
    assert contract["resign"] == arena["resign"]
    assert contract["watchdog"] == arena["watchdog"]
    assert contract["paired_starts_color_swap"] == arena["paired_starts_color_swap"]
    assert contract["technical_fail_closed"] is True

    # The adapter still duplicates these values internally. Until that adapter
    # is migrated to direct profile resolution, fail closed on any drift.
    source = (ROOT / "tools/arena_profiles/torus9.py").read_text(encoding="utf-8")
    assert f"simulations={arena['mcts_simulations']}" in source
    assert f"cpuct={arena['cpuct']}" in source
    assert f"fpu={arena['fpu']}" in source


def test_forbidden_historical_komi_literal_is_absent_from_current_path():
    paths = (
        "configs/gocube/torus9_golden_current_v3.json",
        "gocube_golden/torus9_contract.py",
        "gocube_golden/torus9.py",
        "tools/continue_torus9_golden_m1_m100.py",
        "tools/_frozen_continue_torus9_golden_m1_m100.py",
        "tools/arena_profiles/torus9.py",
    )
    for relative in paths:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "7.5" not in text, relative


def test_reference_fingerprint_is_provenance_only_not_current_content_hash():
    manifest = json.loads((ROOT / "docs/current-golden-boundary.json").read_text(encoding="utf-8"))
    assert manifest["reference"]["profile_fingerprint"] == REFERENCE_PROFILE_FINGERPRINT
    assert manifest["canonical_profile"]["content_fingerprint"] == CURRENT_PROFILE_FINGERPRINT
    assert REFERENCE_PROFILE_FINGERPRINT != CURRENT_PROFILE_FINGERPRINT
    assert manifest["behavior"]["scientific_semantics_changed"] is False
