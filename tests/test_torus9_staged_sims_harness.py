from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.torus9_staged_sims_driver import (
    TOTAL_GAMES,
    TOTAL_OPTIMIZER_STEPS,
    TOTAL_SAMPLE_EXPOSURES,
    optimizer_steps_for_games,
    validate_arm_budget,
)
from tools.arena_profiles.torus9 import PROFILE as TORUS9_ARENA_PROFILE
from tools.torus9_staged_sims_harness import (
    ARENA_GAMES,
    ARMS,
    arena_config,
    should_run_192,
    validate_equal_budget,
)


ROOT = Path(__file__).resolve().parents[1]


def test_staged_arms_have_identical_total_training_budget() -> None:
    validate_equal_budget()
    assert [(arm.games, arm.iterations, arm.optimizer_steps) for arm in ARMS] == [
        (64, 6, 80),
        (128, 3, 160),
        (192, 2, 240),
    ]
    for arm in ARMS:
        assert arm.total_games == TOTAL_GAMES == 384
        assert arm.total_optimizer_steps == TOTAL_OPTIMIZER_STEPS == 480
        assert arm.total_sample_exposures == TOTAL_SAMPLE_EXPOSURES == 30_720


def test_driver_budget_mapping_is_fail_closed() -> None:
    assert optimizer_steps_for_games(64) == 80
    assert optimizer_steps_for_games(128) == 160
    assert optimizer_steps_for_games(192) == 240
    validate_arm_budget(games=128, optimizer_steps=160)
    with pytest.raises(ValueError, match="requires 160 optimizer steps"):
        validate_arm_budget(games=128, optimizer_steps=80)
    with pytest.raises(ValueError, match="64, 128, or 192"):
        optimizer_steps_for_games(256)


def test_192_stage_requires_strict_128_arm_arena_win() -> None:
    assert should_run_192({"W/L/D": [65, 63, 0]}) is True
    assert should_run_192({"W/L/D": [64, 64, 0]}) is False
    assert should_run_192({"W/L/D": [63, 65, 0]}) is False
    assert should_run_192({"W/L/D": [60, 59, 9]}) is True
    with pytest.raises(ValueError, match="W/L/D"):
        should_run_192({"wins": 65, "losses": 63})


def test_strength_arena_is_128_games_with_current_execution_shape() -> None:
    config = arena_config()
    assert ARENA_GAMES == 128
    assert config.games == 128
    assert config.workers == 16
    assert config.games_per_worker == 4
    assert config.inference_batch_rows == 64
    assert config.inference_batch_wait_ms == 1.0
    assert config.device == "cuda"
    assert config.strict_production is True
    assert config.early_gate_enabled is False
    contract = TORUS9_ARENA_PROFILE.scientific_contract(config)
    assert contract["simulations"] == 64
    assert contract["noise"] is False
    assert contract["temperature"] == 0.0


@pytest.mark.parametrize(
    ("games", "steps"),
    [(64, 80), (128, 160), (192, 240)],
)
def test_committed_arm_specs_keep_plateau_settings_and_only_change_cadence(
    games: int, steps: int
) -> None:
    path = ROOT / "configs" / "gocube" / f"torus9_staged_cadence_{games}_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    generation = payload["generation"]["driver_config"]

    assert payload["profile_path"] == "configs/gocube/torus9_staged_cadence_profile_v1.json"
    assert payload["generation"]["command"][1] == "tools/torus9_staged_sims_driver.py"
    assert generation["games"] == games
    assert generation["optimizer_steps_per_iteration"] == steps
    assert generation["workers"] == 16
    assert generation["active_games_per_worker"] == 4
    assert generation["total_active_contexts"] == 64
    assert generation["inference_batch_cap"] == 64
    assert generation["inference_batch_wait_ms"] == 1
    assert payload["arena"]["enabled"] is False
    assert payload["arena"]["required"] is False
    assert payload["experiment"]["total_self_play_games"] == 384
    assert payload["experiment"]["total_optimizer_steps"] == 480
    assert payload["experiment"]["total_sample_exposures"] == 30_720


def test_staged_profile_pins_current_plateau_exit_scientific_settings() -> None:
    payload = json.loads(
        (ROOT / "configs" / "gocube" / "torus9_staged_cadence_profile_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["self_play"]["mcts_simulations"] == 128
    assert payload["training"]["learning_rate"] == pytest.approx(0.0003)
    assert payload["replay"]["generations"] == 6
    assert payload["replay"]["cap"] == 40_000
    assert "parent_generation" not in payload["experiment"]
