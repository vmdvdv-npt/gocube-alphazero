from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .komi_policy import (
    GOCUBE_DEFAULT_KOMI,
    LEGACY_FORBIDDEN_KOMI,
    LegacyKomiError,
    validate_gocube_komi,
)


# Backward-compatible name. It means "current default/baseline", not
# "the only globally valid GoCube komi".
GOCUBE_KOMI = GOCUBE_DEFAULT_KOMI


def require_gocube_komi(value: object, *, context: str = "GoCube") -> float:
    """Backward-compatible komi validator.

    Historically this function enforced ``komi == 0.5`` globally. It now
    follows the project-wide komi policy: 0.5 is the default baseline, explicit
    finite alternatives are allowed, and the legacy value 7.5 fails closed.
    """

    return validate_gocube_komi(value, context=context)


@dataclass(frozen=True)
class Cube4ProductionContract:
    """Frozen Cube-4 baseline settings shared by legacy production tooling.

    This object is an experiment/path-specific reproducibility contract. Its
    0.5 komi pin does not define the global GoCube/Torus komi policy. New
    reference/research paths should use :func:`validate_gocube_komi` and record
    their explicit komi in rules/checkpoint identity.
    """

    topology: str = "cube"
    size: int = 4
    komi: float = GOCUBE_DEFAULT_KOMI
    workers: int = 16
    regular_sims: int = 50
    fast_sims: int = 20
    fast_probability: float = 0.25
    games_per_iteration: int = 256
    train_batch_size: int = 1024
    train_samples_per_new_sample: float = 1.0
    arena_sims: int = 50

    def validate_checkpoint_args(self, args: Mapping[str, object]) -> None:
        actual_komi = validate_gocube_komi(
            args.get("gocube_komi"),
            context="Cube-4 frozen production checkpoint",
        )
        if not math.isclose(
            actual_komi, self.komi, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "Cube-4 frozen production checkpoint requires its historical "
                f"baseline komi {self.komi}, got {actual_komi!r}. This is an "
                "experiment-specific reproducibility pin, not the global GoCube "
                "komi policy."
            )
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
                matches = math.isclose(
                    float(args.get(key)),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(
                    f"Cube-4 production checkpoint requires {key}={expected!r}, got {args.get(key)!r}"
                )


CUBE4_PRODUCTION = Cube4ProductionContract()
