from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


GOCUBE_KOMI = 0.5


def require_gocube_komi(value: object, *, context: str = "GoCube") -> float:
    """Validate the single supported GoCube komi and return its canonical value."""

    komi = float(value)
    if not math.isclose(komi, GOCUBE_KOMI, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{context} requires komi {GOCUBE_KOMI}, got {value!r}")
    return GOCUBE_KOMI


@dataclass(frozen=True)
class Cube4ProductionContract:
    """Fixed Cube-4 production settings shared by training and sweep tooling.

    Sweep axes belong elsewhere. This object contains only values that must not
    drift between launchers, validation, reporting, and Arena orchestration.
    """

    topology: str = "cube"
    size: int = 4
    komi: float = GOCUBE_KOMI
    workers: int = 16
    regular_sims: int = 50
    fast_sims: int = 20
    fast_probability: float = 0.25
    games_per_iteration: int = 256
    train_batch_size: int = 1024
    train_samples_per_new_sample: float = 1.0
    arena_sims: int = 50

    def validate_checkpoint_args(self, args: Mapping[str, object]) -> None:
        require_gocube_komi(args.get("gocube_komi"), context="Cube-4 production checkpoint")
        checks = (
            ("gocube_topology", self.topology),
            ("gocube_size", self.size),
            ("numMCTSSims", self.regular_sims),
            ("numFastSims", self.fast_sims),
            ("arenaMCTSSims", self.arena_sims),
            ("train_batch_size", self.train_batch_size),
            ("workers", self.workers),
        )
        for key, expected in checks:
            actual = args.get(key)
            if actual != expected:
                raise ValueError(
                    f"Cube-4 production checkpoint requires {key}={expected!r}, got {actual!r}"
                )
        for key, expected in (
            ("probFastSim", self.fast_probability),
            ("gocube_train_samples_per_new_sample", self.train_samples_per_new_sample),
        ):
            try:
                matches = math.isclose(float(args.get(key)), float(expected), rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(
                    f"Cube-4 production checkpoint requires {key}={expected!r}, got {args.get(key)!r}"
                )


CUBE4_PRODUCTION = Cube4ProductionContract()
