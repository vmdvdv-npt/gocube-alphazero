#!/usr/bin/env python3
"""Experiment-only Torus9 driver for config-driven cadence studies.

The immutable run spec supplies games-per-iteration and optimizer-step budget.
The harness owns cross-arm budget comparability; this driver keeps the current
Torus9 scientific/execution boundaries and applies the declared Adam work.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.torus9 import Torus9TrainingAdapter
from training_engine import value_fingerprint
from tools import torus9_run_driver as _base


BATCH_SIZE = 64


def validate_arm_budget(*, games: int, optimizer_steps: int) -> None:
    if int(games) <= 0:
        raise ValueError("Experiment games_per_iteration must be positive")
    if int(optimizer_steps) <= 0:
        raise ValueError("Experiment optimizer_steps_per_iteration must be positive")


class ExperimentCadenceTrainingAdapter(Torus9TrainingAdapter):
    """Training adapter whose per-iteration Adam budget comes from the run spec."""

    def __init__(self, *, optimizer_steps: int, **kwargs: object) -> None:
        if int(optimizer_steps) <= 0:
            raise ValueError("Experiment optimizer step budget must be positive")
        super().__init__(**kwargs)
        self.cadence_optimizer_steps = int(optimizer_steps)

    def create_state(self, *args: object, **kwargs: object):
        state = super().create_state(*args, **kwargs)  # type: ignore[arg-type]
        state.adapter_state.optimizer_steps_per_iteration = self.cadence_optimizer_steps
        return state

    def train(
        self,
        state: Any,
        rows: Sequence[Mapping[str, object]],
        seed: int,
    ) -> Mapping[str, object]:
        self.validate_state(state)
        trainer = state.adapter_state
        count = self.cadence_optimizer_steps * BATCH_SIZE
        indices = trainer._sample_indices(len(rows), seed=int(seed), count=count)

        def training_progress(completed: int, total: int) -> None:
            self._report_progress(
                "training",
                completed,
                total,
                "optimizer_steps",
                "optimizer",
            )

        metrics = dict(
            trainer.train_fixed_budget(
                rows,
                seed=int(seed),
                validate_samples=False,
                timing=self._diagnostic_timing,
                progress_callback=training_progress,
            )
        )
        if self._diagnostic_timing is not None:
            metrics["stage_timing"] = dict(self._diagnostic_timing)
        metrics.update(
            {
                "training_seed": int(seed),
                "cadence_optimizer_steps_per_iteration": self.cadence_optimizer_steps,
                "sampled_replay_row_ids": tuple(
                    str(rows[index].get("replay_row_id", index)) for index in indices
                ),
            }
        )
        metrics["sampled_row_ids_fingerprint"] = value_fingerprint(
            metrics["sampled_replay_row_ids"]
        )
        if metrics.get("optimizer_steps") != self.cadence_optimizer_steps:
            raise ValueError("Experiment optimizer step budget drift")
        if metrics.get("samples_consumed") != count:
            raise ValueError("Experiment sample exposure budget drift")
        return metrics

    def prepare_checkpoint(
        self,
        state: Any,
        context: Any,
        training_metrics: Mapping[str, object],
    ) -> Mapping[str, object]:
        metadata = dict(super().prepare_checkpoint(state, context, training_metrics))
        metadata["optimizer_steps_per_iteration"] = self.cadence_optimizer_steps
        metadata["samples_consumed_per_iteration"] = (
            self.cadence_optimizer_steps * BATCH_SIZE
        )
        contract = dict(metadata["scientific_contract"])  # type: ignore[arg-type]
        contract.update(
            {
                "optimizer_steps": self.cadence_optimizer_steps,
                "samples_consumed": self.cadence_optimizer_steps * BATCH_SIZE,
            }
        )
        metadata["scientific_contract"] = contract
        metadata["cadence_experiment"] = {
            "status": "controlled run-spec-owned arm",
            "canonical_profile_unchanged": True,
            "optimizer": "Adam",
            "batch_size": BATCH_SIZE,
            "steps_per_iteration": self.cadence_optimizer_steps,
            "sample_draws_per_iteration": self.cadence_optimizer_steps * BATCH_SIZE,
        }
        return metadata


def _validate_experiment_bindings(
    profile: Mapping[str, object], config: Mapping[str, object]
) -> None:
    self_play = _base._mapping(profile.get("self_play"), "profile.self_play")
    training = _base._mapping(profile.get("training"), "profile.training")
    replay = _base._mapping(profile.get("replay"), "profile.replay")
    network = _base._mapping(profile.get("network"), "profile.network")

    games = int(config["games"])
    steps = int(config.get("optimizer_steps_per_iteration", 0))
    validate_arm_budget(games=games, optimizer_steps=steps)

    if int(self_play["mcts_simulations"]) <= 0:
        raise ValueError("Experiment self-play simulations must be positive")
    if float(training["learning_rate"]) <= 0.0:
        raise ValueError("Experiment learning rate must be positive")
    if int(training["batch_size"]) != BATCH_SIZE:
        raise ValueError("Torus9 experiment driver requires batch size 64")
    if int(replay["generations"]) <= 0 or int(replay["cap"]) <= 0:
        raise ValueError("Experiment replay window/cap must be positive")
    if int(network["hidden"]) != 80 or int(network["blocks"]) != 8:
        raise ValueError("Torus9 experiment driver requires the current 80x8 network")

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
                raise ValueError(f"Torus9 experiment execution drift: {key}={actual!r}")
        elif actual != expected:
            raise ValueError(f"Torus9 experiment execution drift: {key}={actual!r}")

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
    class BoundExperimentCadenceTrainingAdapter(ExperimentCadenceTrainingAdapter):
        def __init__(self, *args: object, **kwargs: object) -> None:
            if args:
                raise TypeError("Experiment adapter accepts keyword construction only")
            super().__init__(
                optimizer_steps=int(optimizer_steps),
                **kwargs,
            )

    BoundExperimentCadenceTrainingAdapter.__name__ = (
        f"ExperimentCadenceTrainingAdapter{int(optimizer_steps)}"
    )
    return BoundExperimentCadenceTrainingAdapter


def configure_from_run_spec() -> _base.DriverBindings:
    spec = _base._load_run_spec()
    config = _base._generation_config(spec)
    games = int(config["games"])
    steps = int(config.get("optimizer_steps_per_iteration", 0))
    validate_arm_budget(games=games, optimizer_steps=steps)

    return _base.DriverBindings(
        training_adapter_factory=_adapter_type(steps),
        optimizer_steps_per_iteration=steps,
        scientific_validator=_validate_experiment_bindings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    bindings = configure_from_run_spec()
    return _base.main(argv, bindings=bindings)


if __name__ == "__main__":
    raise SystemExit(main())
