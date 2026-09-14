#!/usr/bin/env python3
"""Read-only Torus9 komi calibration API with frozen Arena execution.

Offline analysis helpers stay importable. Historical Arena execution requires
``--allow-frozen-arena``.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from gocube_golden.arena_policy import ArenaPolicyError, FROZEN_ARENA_OVERRIDE_ENV, FROZEN_ARENA_OVERRIDE_FLAG

_FROZEN_MODULE = "tools._frozen_torus9_komi_calibration"
_frozen = importlib.import_module(_FROZEN_MODULE)
Trajectory = _frozen.Trajectory
bootstrap_estimates = _frozen.bootstrap_estimates
crossing_interval = _frozen.crossing_interval
crossing_point = _frozen.crossing_point
sweep_trajectories = _frozen.sweep_trajectories


def main(*_args: Any, **_kwargs: Any) -> int:
    raise ArenaPolicyError(
        "Programmatic historical Torus9 komi Arena execution is frozen; use "
        f"{FROZEN_ARENA_OVERRIDE_FLAG} only for deliberate reproduction."
    )


def _cli() -> int:
    if FROZEN_ARENA_OVERRIDE_FLAG not in sys.argv[1:]:
        raise SystemExit(f"LOCKED: historical komi calibration Arena. Add {FROZEN_ARENA_OVERRIDE_FLAG} only for deliberate reproduction.")
    previous = os.environ.get(FROZEN_ARENA_OVERRIDE_ENV)
    os.environ[FROZEN_ARENA_OVERRIDE_ENV] = "1"
    sys.argv = [arg for arg in sys.argv if arg != FROZEN_ARENA_OVERRIDE_FLAG]
    try:
        return int(_frozen.main())
    finally:
        if previous is None:
            os.environ.pop(FROZEN_ARENA_OVERRIDE_ENV, None)
        else:
            os.environ[FROZEN_ARENA_OVERRIDE_ENV] = previous


if __name__ == "__main__":
    raise SystemExit(_cli())
