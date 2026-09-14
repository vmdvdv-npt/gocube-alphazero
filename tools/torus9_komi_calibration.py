#!/usr/bin/env python3
"""Read-only Torus9 komi calibration API with frozen Arena execution.

Analysis helpers remain importable for tests and offline reports. The historical
M8-vs-M8 Arena diagnostic is reachable only by invoking this script with the
explicit frozen-Arena override.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from gocube_golden.arena_policy import ArenaPolicyError, FROZEN_ARENA_OVERRIDE_FLAG


_FROZEN_MODULE = "tools._frozen_torus9_komi_calibration"


def _frozen_module():
    return importlib.import_module(_FROZEN_MODULE)


# Deliberately exported read-only/offline analysis surface.
_frozen = _frozen_module()
Trajectory = _frozen.Trajectory
bootstrap_estimates = _frozen.bootstrap_estimates
crossing_interval = _frozen.crossing_interval
crossing_point = _frozen.crossing_point
sweep_trajectories = _frozen.sweep_trajectories


def main(*_args: Any, **_kwargs: Any) -> int:
    raise ArenaPolicyError(
        "Programmatic execution of the historical Torus9 komi calibration is "
        "frozen because it can launch the old process Arena. Invoke this script "
        f"with {FROZEN_ARENA_OVERRIDE_FLAG} only for deliberate reproduction."
    )


def _cli() -> int:
    if FROZEN_ARENA_OVERRIDE_FLAG not in sys.argv[1:]:
        raise SystemExit(
            "LOCKED: this historical calibration can launch a frozen Torus9 Arena. "
            f"Add {FROZEN_ARENA_OVERRIDE_FLAG} only for deliberate reproduction."
        )
    sys.argv = [arg for arg in sys.argv if arg != FROZEN_ARENA_OVERRIDE_FLAG]
    return int(_frozen.main())


if __name__ == "__main__":
    raise SystemExit(_cli())
