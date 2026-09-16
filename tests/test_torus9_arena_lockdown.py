from __future__ import annotations

import inspect
import json
from pathlib import Path

import tools.arena as arena_cli
import tools.arena_engine as arena_engine
from tools.arena_profiles import available_profiles, get_profile
from tools.arena_profiles.torus9 import Torus9ArenaProfile


def test_repository_has_one_standalone_arena_engine():
    root = Path(__file__).resolve().parents[1]
    assert (root / "tools" / "arena.py").is_file()
    assert (root / "tools" / "arena_engine.py").is_file()
    assert not (root / "tools" / "torus9_arena.py").exists()
    assert not (root / "gocube_golden" / "arena.py").exists()
    assert not (root / "gocube_golden" / "arena_process.py").exists()


def test_engine_is_board_agnostic_and_torus9_is_only_a_profile():
    source = inspect.getsource(arena_engine)
    assert "gocube_golden.torus9" not in source
    assert "TORUS9_" not in source
    assert "build_torus9" not in source
    assert available_profiles() == ("torus9",)
    assert isinstance(get_profile("torus9"), Torus9ArenaProfile)


def test_profile_uses_canonical_sequential_puct_and_central_model_owner():
    profile = get_profile("torus9")
    profile_source = inspect.getsource(profile.worker_main)
    engine_source = inspect.getsource(arena_engine.run_arena)
    assert "SequentialPUCT" in profile_source
    assert "Torus9BatchedPUCT" not in profile_source
    assert "torus9_load_checkpoint" not in profile_source
    assert "profile.load_parent_model" in engine_source
    assert "dispatch_model_batch" in engine_source


def test_production_defaults_encode_current_arena_contract():
    config = arena_engine.ArenaExecutionConfig()
    config.validate_base()
    get_profile("torus9").validate_execution_config(config)
    assert config.games == 64
    assert config.workers == 16
    assert config.games_per_worker == 12
    assert config.inference_batch_rows == 64
    assert config.inference_batch_wait_ms == 4.0
    assert config.device == "cuda"
    assert config.strict_production is True


def test_legion_preset_is_one_explicit_performance_contract():
    root = Path(__file__).resolve().parents[1]
    preset = json.loads(
        (root / "configs" / "gocube" / "arena_torus9_legion_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert preset["workers"] == 16
    assert preset["games_per_worker"] == 12
    assert preset["configured_context_capacity"] == 192
    assert preset["inference_batch_rows"] == 64
    assert preset["inference_batch_wait_ms"] == 4.0
    assert preset["worker_local_inference_batch_wait_ms"] == 0.0
    assert preset["central_model_owner"] == "parent"
    assert preset["shared_memory"] is True
    assert preset["performance_gate"] == {
        "minimum_games": 64,
        "minimum_mean_inference_batch_rows": 16.0,
        "minimum_effective_cpu_cores": 8.0,
        "technical_games": 0,
        "fail_closed": True,
    }
    assert preset["selection_basis"]["confirmed_status"] == "PASS"


def test_debug_config_is_explicit_and_small_cpu():
    config = arena_engine.ArenaExecutionConfig(
        games=2,
        workers=1,
        games_per_worker=1,
        inference_batch_rows=1,
        inference_batch_wait_ms=0.0,
        device="cpu",
        strict_production=False,
    )
    config.validate_base()
    get_profile("torus9").validate_execution_config(config)


def test_only_one_normal_arena_cli_is_advertised():
    parser = arena_cli.build_parser()
    action = next(item for item in parser._actions if item.dest == "profile")
    assert action.default == "auto"
    assert action.choices == ("auto", "torus9")
