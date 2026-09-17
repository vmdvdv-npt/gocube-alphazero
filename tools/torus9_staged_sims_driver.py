#!/usr/bin/env python3
"""Experiment-only Torus9 driver for the staged 64/128/192 cadence harness.

The normal production driver and Golden profile remain unchanged.  This wrapper
reuses the production lifecycle/replay/checkpoint path, but opts into the
already-proven proportional cadence adapter (80/160/240 Adam steps) and permits
only the three equal-budget workloads declared by the staged experiment.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from tools import torus9_run_driver as _base
from tools.torus9_nightly_diagnostics import CadenceTrainingAdapter


BUDGETS: dict[int, int] = {64: 80, 128: 160, 192: 240}
BATCH_SIZE = 64
TOTAL_GAMES = 384
TOTAL_OPTIMIZER_STEPS = 480
TOTAL_SAMPLE_EXPOSURES = 30_720


def optimizer_steps_for_games(games: int) -> int:
    try:
        return BUDGETS[int(games)]
    except KeyError as exc:
        raise ValueError("Staged Torus9 cadence must be one of 64, 128, or 192 games") from exc


def validate_arm_budget(*, games: int, optimizer_steps: int) -> None:
    expected = optimizer_steps_for_games(games)
    if int(optimizer_steps) != expected:
        raise ValueError(
            f"Staged Torus9 cadence {games} games requires {expected} optimizer steps, "
            f"got {optimizer_steps}"
        )


def _validate_staged_scientific_bindings(
    profile: Mapping[str, object], config: Mapping[str, object]
) -> None:
    """Validate every fixed experiment setting while allowing cadence batch size.

    ``games_per_iteration`` is deliberately an experiment execution/budget knob
    here, just as in the earlier equal-budget cadence harness.  Search, LR,
    replay scope, architecture, seeds and execution preset remain pinned.
    """
    self_play = _base._mapping(profile.get("self_play"), "profile.self_play")
    training = _base._mapping(profile.get("training"), "profile.training")
    replay = _base._mapping(profile.get("replay"), "profile.replay")
    network = _base._mapping(profile.get("network"), "profile.network")

    games = int(config["games"])
    steps = int(config.get("optimizer_steps_per_iteration", 0))
    validate_arm_budget(games=games, optimizer_steps=steps)

    if int(self_play["mcts_simulations"]) != 128:
        raise ValueError("Staged Torus9 harness requires exactly 128 self-play simulations")
    if float(training["learning_rate"]) != 0.0003:
        raise ValueError("Staged Torus9 harness requires LR=3e-4")
    if int(training["batch_size"]) != BATCH_SIZE:
        raise ValueError("Staged Torus9 harness requires batch size 64")
    if int(replay["generations"]) != 6 or int(replay["cap"]) != 40_000:
        raise ValueError("Staged Torus9 harness requires replay=6 generations / 40k positions")
    if int(network["hidden"]) != 80 or int(network["blocks"]) != 8:
        raise ValueError("Staged Torus9 harness requires the same 80x8 network")

    expected_execution = {
        "workers": 16,
        "active_games_per_worker": 4,
        "total_active_contexts": 64,
        "inference_batch_cap": 64,
        "inference_batch_wait_ms": 1.0,
        "coalescing": True,
    }
    for key, expected in expected_execution.items():
        actual = config.get(key)
        if isinstance(expected, float):
            if float(actual) != expected:
                raise ValueError(f"Staged Torus9 execution drift: {key}={actual!r}")
        elif actual != expected:
            raise ValueError(f"Staged Torus9 execution drift: {key}={actual!r}")

    seeds = _base._mapping(profile.get("seeds"), "profile.seeds")
    for config_key, profile_key in (
        ("model_init_seed", "model_init_seed"),
        ("selfplay_master_seed", "selfplay_master_seed"),
        ("training_master_seed", "training_master_seed"),
    ):
        if int(config[config_key]) != int(seeds[profile_key]):
            raise ValueError(
                f"run-spec {config_key} disagrees with the referenced scientific profile"
            )


def _adapter_type(optimizer_steps: int):
    class StagedCadenceTrainingAdapter(CadenceTrainingAdapter):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(
                *args,
                optimizer_steps=int(optimizer_steps),
                **kwargs,
            )

    StagedCadenceTrainingAdapter.__name__ = (
        f"StagedCadenceTrainingAdapter{int(optimizer_steps)}"
    )
    return StagedCadenceTrainingAdapter


def configure_from_run_spec() -> int:
    spec = _base._load_run_spec()
    config = _base._generation_config(spec)
    games = int(config["games"])
    steps = int(config.get("optimizer_steps_per_iteration", 0))
    validate_arm_budget(games=games, optimizer_steps=steps)

    # Local process-only overrides.  The normal driver module on disk and
    # Golden profile stay untouched; every child process reconstructs these
    # bindings from its immutable lineage-owned run spec.
    _base.Torus9TrainingAdapter = _adapter_type(steps)
    _base.TORUS9_OPTIMIZER_STEPS_PER_ITERATION = steps
    _base._validate_scientific_bindings = _validate_staged_scientific_bindings
    return steps


def main(argv: Sequence[str] | None = None) -> int:
    configure_from_run_spec()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
