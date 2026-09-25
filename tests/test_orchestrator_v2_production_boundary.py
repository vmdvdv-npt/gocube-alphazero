from __future__ import annotations

import json
import os

import pytest

from gocube_golden.artifact_graph import EffectiveConfig
from gocube_golden.orchestrator_v2.execution_permit import (
    PERMIT_ENV,
    PERMIT_KEY_ENV,
    _child_execution_permit,
    _production_authority,
    require_child_execution_permit,
)
from gocube_golden.orchestrator_v2.operator_messages import format_arena_started, format_training_started
from gocube_golden.orchestrator_v2.production_generation import ProductionTrainOne
from gocube_golden.orchestrator_v2.run_spec import RunMode, RunSpecV2
from gocube_golden.orchestrator_v2.topology_binding import get_topology_binding
from gocube_golden.orchestrator_v2.version import mark_v2_process, require_v2_process
from tools.arena import run_arena
from tools.arena_engine import ArenaExecutionConfig
from tools.arena_profiles import get_profile


def _torus_config(*, komi: float = 5.5, simulations: int = 256) -> EffectiveConfig:
    return EffectiveConfig(
        topology="torus9",
        compatibility={"topology": "torus9", "rules": "positional-superko", "input_channels": 5},
        self_play={"komi": komi, "mcts_simulations": simulations, "games_per_iteration": 384},
        training={"learning_rate": 0.0001, "optimizer_steps": 160, "batch_size": 64},
        replay={"window": 2, "cap": 80000},
        execution={"total_active_contexts": 64},
        arena={"komi": komi, "mcts_simulations": simulations, "reference_gap": 1},
    )


def test_env_marker_and_mark_v2_process_do_not_authorize_production(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(PERMIT_ENV, raising=False)
    monkeypatch.delenv(PERMIT_KEY_ENV, raising=False)
    monkeypatch.setenv("AZ_ORCHESTRATOR_VERSION", "V2")
    with pytest.raises(RuntimeError, match="execution permit"):
        require_v2_process("test")
    mark_v2_process()
    with pytest.raises(RuntimeError, match="execution permit"):
        run_arena(candidate_path=tmp_path / "does-not-exist.pt")
    train_one = object.__new__(ProductionTrainOne)
    with pytest.raises(RuntimeError, match="execution permit"):
        train_one(parent=object(), config=object(), output_lineage=object())


def test_entrypoint_authority_can_mint_pid_bound_signed_child_permit(monkeypatch) -> None:
    monkeypatch.delenv(PERMIT_ENV, raising=False)
    monkeypatch.delenv(PERMIT_KEY_ENV, raising=False)
    with _production_authority(mode="arena", topology="torus9", run_id="eval-1", code_identity="abc123", launch_id="launch-1"):
        require_v2_process("test-entrypoint")
        with _child_execution_permit(action_type="arena", topology="torus9", run_id="eval-1", code_identity="abc123"):
            permit = require_child_execution_permit(
                "test-child",
                action_type="arena",
                topology="torus9",
                run_id="eval-1",
                code_identity="abc123",
                parent_pid=os.getpid(),
            )
            assert permit["launch_id"] == "launch-1"
            assert permit["action_type"] == "arena"
            assert permit["supervisor_pid"] == os.getpid()
            tampered = dict(os.environ)
            payload = json.loads(tampered[PERMIT_ENV])
            payload["run_id"] = "other"
            tampered[PERMIT_ENV] = json.dumps(payload)
            with pytest.raises(RuntimeError, match="signature mismatch"):
                require_child_execution_permit("tampered", environ=tampered, parent_pid=os.getpid())
    assert PERMIT_ENV not in os.environ
    assert PERMIT_KEY_ENV not in os.environ
    with pytest.raises(RuntimeError, match="execution permit"):
        require_v2_process("after-authority")


def test_torus_profile_accepts_run_owned_komi_sims_games_and_execution_preset() -> None:
    profile = get_profile("torus9|komi=5.5|simulations=256|5ch")
    assert profile.komi == 5.5
    assert profile.simulations == 256
    assert profile.observation_shape[0] == 5
    config = ArenaExecutionConfig(
        games=1024,
        workers=3,
        games_per_worker=5,
        inference_batch_rows=7,
        inference_batch_wait_ms=2.5,
        device="cpu",
        strict_production=True,
    )
    profile.validate_execution_config(config)


def test_real_invalid_arena_values_still_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite number"):
        get_profile("torus9|komi=nan|simulations=256|5ch")
    with pytest.raises(ValueError, match="positive integer"):
        get_profile("torus9|komi=1.5|simulations=0|5ch")
    profile = get_profile("torus9|komi=1.5|simulations=256|5ch")
    with pytest.raises(ValueError, match="positive even"):
        profile.validate_execution_config(ArenaExecutionConfig(games=255))


def test_topology_binding_supports_torus_and_cube_profiles_without_golden_whitelist() -> None:
    torus = _torus_config()
    torus_binding = get_topology_binding("torus9")
    torus_profile = torus_binding.default_arena_profile(torus)
    assert "komi=5.5" in torus_profile
    assert "simulations=256" in torus_profile
    assert getattr(get_profile(torus_profile), "simulations") == 256

    cube = EffectiveConfig(
        topology="cube4",
        compatibility={"topology": "cube4", "rules": "positional-superko"},
        arena={"simulations": 256, "cpuct": 1.25, "fpu": 0.0, "watchdog": 500},
    )
    cube_binding = get_topology_binding("cube4")
    cube_profile_id = cube_binding.default_arena_profile(cube)
    cube_profile = get_profile(cube_profile_id)
    assert getattr(cube_profile, "size") == 4
    cube_profile.validate_execution_config(
        ArenaExecutionConfig(
            games=256,
            workers=3,
            games_per_worker=5,
            inference_batch_rows=7,
            inference_batch_wait_ms=2.5,
            device="cpu",
            strict_production=True,
        )
    )


def test_universal_run_spec_covers_all_production_modes() -> None:
    for mode in ("continuous", "arena", "evaluation", "experiment", "calibration"):
        spec = RunSpecV2.from_dict({"schema": "gocube-orchestrator-v2-run-spec-v1", "mode": mode, mode: {"id": "x"}})
        assert spec.mode is RunMode(mode)
        assert spec.payload["id"] == "x"
    with pytest.raises(ValueError, match="unsupported"):
        RunSpecV2.from_dict({"mode": "bespoke-launcher"})


def test_operator_start_messages_are_multiline_and_resolved() -> None:
    effective = _torus_config(komi=1.5, simulations=256)
    arena_config = ArenaExecutionConfig(games=512, workers=8, games_per_worker=8, inference_batch_rows=16)
    training = format_training_started(
        topology="torus9",
        lineage_id="new-komi",
        parent_label="M137",
        network="GoldenGraphNetV2-Torus9-M137-5CH",
        effective_config=effective,
        arena_cadence=5,
        arena_config=arena_config,
    )
    assert training.startswith("TRAINING STARTED — GoCube AlphaZero\n")
    assert "\nSelf-play:\n" in training
    assert "\nTraining:\n" in training
    assert "\nReplay:\n" in training
    assert "\nArena:\n" in training
    assert "LR=0.0001" in training
    assert "self-play MCTS=256 sims" in training
    assert "replay=2 generations / 80000 positions" in training
    assert ";" not in training

    arena = format_arena_started(
        topology="torus9",
        evaluation_id="eval-1",
        candidate="M137",
        reference="M137",
        profile="torus9|komi=1.5|simulations=256|5ch",
        komi=1.5,
        games=1024,
        simulations=256,
        seed=20260925,
        workers=8,
        contexts=64,
        batch_cap=32,
        wait_ms=1.0,
    )
    assert arena.startswith("ARENA STARTED — GoCube AlphaZero\n")
    assert "Games: 1024" in arena
    assert "Pairs: 512" in arena
    assert "MCTS: 256 sims" in arena
    assert "Workers: 8" in arena
    assert ";" not in arena
