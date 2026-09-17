from __future__ import annotations

import pytest

from tools.arena_profiles.torus9 import PROFILE as TORUS9_ARENA_PROFILE
from tools.torus9_staged_sims_driver import validate_arm_budget
from tools.torus9_staged_sims_harness import (
    DEFAULT_EXPERIMENT_SPEC,
    arena_config,
    arms_from_spec,
    build_arm_run_spec,
    condition_met,
    evaluations_from_spec,
    load_experiment_spec,
    validate_equal_budget,
)


EXPERIMENT = load_experiment_spec(DEFAULT_EXPERIMENT_SPEC)
ARMS = arms_from_spec(EXPERIMENT)
EVALUATIONS = evaluations_from_spec(EXPERIMENT, set(ARMS))


def test_experiment_spec_owns_equal_budget_and_stage_shape() -> None:
    validate_equal_budget(EXPERIMENT)
    assert [
        (arm.arm_id, arm.games, arm.iterations, arm.optimizer_steps)
        for arm in ARMS.values()
    ] == [
        ("g64", 64, 6, 80),
        ("g128", 128, 3, 160),
        ("g192", 192, 2, 240),
    ]
    for arm in ARMS.values():
        assert arm.total_games == 384
        assert arm.total_optimizer_steps == 480
        assert arm.total_sample_exposures == 30_720

    stages = EXPERIMENT["stages"]
    assert stages[0]["arms"] == ["g64", "g128"]
    assert stages[0]["evaluations"] == ["g128-vs-g64"]
    assert stages[1]["arms"] == ["g192"]
    assert stages[1]["evaluations"] == ["g192-vs-g128"]


def test_experiment_driver_accepts_run_spec_owned_positive_budgets() -> None:
    validate_arm_budget(games=64, optimizer_steps=80)
    validate_arm_budget(games=96, optimizer_steps=120)
    validate_arm_budget(games=192, optimizer_steps=240)
    with pytest.raises(ValueError, match="games_per_iteration"):
        validate_arm_budget(games=0, optimizer_steps=80)
    with pytest.raises(ValueError, match="optimizer_steps_per_iteration"):
        validate_arm_budget(games=64, optimizer_steps=0)


def test_stage_condition_is_config_driven_candidate_win() -> None:
    condition = EXPERIMENT["stages"][1]["run_if"]
    assert condition == {
        "evaluation": "g128-vs-g64",
        "metric": "wins_minus_losses",
        "operator": ">",
        "value": 0,
    }
    assert condition_met(
        condition, {"g128-vs-g64": {"W/L/D": [65, 63, 0]}}
    ) is True
    assert condition_met(
        condition, {"g128-vs-g64": {"W/L/D": [64, 64, 0]}}
    ) is False
    assert condition_met(
        condition, {"g128-vs-g64": {"W/L/D": [63, 65, 0]}}
    ) is False


def test_strength_arena_settings_come_from_experiment_spec() -> None:
    evaluation = EVALUATIONS["g128-vs-g64"]
    config = arena_config(EXPERIMENT, evaluation)
    assert evaluation.games == 128
    assert config.games == 128
    assert config.workers == 16
    assert config.games_per_worker == 4
    assert config.inference_batch_rows == 64
    assert config.inference_batch_wait_ms == 1.0
    assert config.device == "cuda"
    assert config.strict_production is True
    assert config.early_gate_enabled is False

    contract = TORUS9_ARENA_PROFILE.scientific_contract(config)
    for key, expected in EXPERIMENT["arena_scientific_contract"].items():
        assert contract[key] == expected


@pytest.mark.parametrize("arm_id", ["g64", "g128", "g192"])
def test_arm_run_specs_are_materialized_from_one_template(arm_id: str) -> None:
    arm = ARMS[arm_id]
    run_spec = build_arm_run_spec(EXPERIMENT, arm)
    payload = run_spec.payload
    generation = payload["generation"]["driver_config"]

    assert payload["profile_path"] == EXPERIMENT["profile_path"]
    assert payload["generation"]["command"][1] == "tools/torus9_staged_sims_driver.py"
    assert generation["games"] == arm.games
    assert generation["optimizer_steps_per_iteration"] == arm.optimizer_steps
    assert generation["workers"] == 16
    assert generation["active_games_per_worker"] == 4
    assert generation["total_active_contexts"] == 64
    assert generation["inference_batch_cap"] == 64
    assert generation["inference_batch_wait_ms"] == 1
    assert payload["arena"]["enabled"] is False
    assert payload["arena"]["required"] is False
    assert payload["experiment"]["arm_id"] == arm_id
    assert (
        payload["experiment"]["experiment_spec_path"]
        == "configs/gocube/torus9_staged_cadence_experiment_v1.json"
    )
    assert payload["experiment"]["total_self_play_games"] == 384
    assert payload["experiment"]["total_optimizer_steps"] == 480
    assert payload["experiment"]["total_sample_exposures"] == 30_720


def test_experiment_spec_pins_current_plateau_exit_settings() -> None:
    fixed = EXPERIMENT["fixed_training"]
    assert fixed["self_play_mcts_simulations"] == 128
    assert fixed["learning_rate"] == pytest.approx(0.0003)
    assert fixed["replay_generations"] == 6
    assert fixed["replay_cap"] == 40_000
    assert fixed["network_hidden"] == 80
    assert fixed["network_blocks"] == 8
    assert fixed["batch_size"] == 64
