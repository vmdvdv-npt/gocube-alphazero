#!/usr/bin/env python3
"""Fail-closed front door for the historical Torus9 alpha/score A/B experiment.

The preserved implementation contains the deprecated single-process
logical-lane Arena. It is available only through an explicit historical
reproduction command.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from gocube_golden.arena_policy import ArenaPolicyError, FROZEN_ARENA_OVERRIDE_FLAG


_FROZEN_MODULE = "tools._frozen_torus9_alpha_score_ab"
_BLOCKED = frozenset({"run_experiment", "cli"})


def _frozen_module():
    return importlib.import_module(_FROZEN_MODULE)


def _locked(name: str) -> ArenaPolicyError:
    return ArenaPolicyError(
        f"{name} belongs to the frozen Torus9 alpha/score A/B runner. "
        f"Run this script with {FROZEN_ARENA_OVERRIDE_FLAG} only for deliberate "
        "historical reproduction."
    )


def run_experiment(*_args: Any, **_kwargs: Any):
    raise _locked("run_experiment")


def cli(*_args: Any, **_kwargs: Any):
    raise _locked("cli")


def __getattr__(name: str):
    if name in _BLOCKED:
        raise _locked(name)
    return getattr(_frozen_module(), name)


def main() -> int:
    if FROZEN_ARENA_OVERRIDE_FLAG not in sys.argv[1:]:
        raise SystemExit(
            "LOCKED: this historical alpha/score A/B uses the frozen logical-lane "
            f"Torus9 Arena. Add {FROZEN_ARENA_OVERRIDE_FLAG} only for deliberate reproduction."
        )
    sys.argv = [arg for arg in sys.argv if arg != FROZEN_ARENA_OVERRIDE_FLAG]
    return int(_frozen_module().cli())


if __name__ == "__main__":
    raise SystemExit(main())
