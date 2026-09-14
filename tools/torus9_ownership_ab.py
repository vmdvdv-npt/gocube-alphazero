#!/usr/bin/env python3
"""Fail-closed front door for the historical Torus9 ownership A/B experiment.

Historical reproduction requires ``--allow-frozen-arena``.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from gocube_golden.arena_policy import ArenaPolicyError, FROZEN_ARENA_OVERRIDE_ENV, FROZEN_ARENA_OVERRIDE_FLAG

_FROZEN_MODULE = "tools._frozen_torus9_ownership_ab"
_BLOCKED = frozenset({"run_experiment", "cli"})


def _frozen_module():
    return importlib.import_module(_FROZEN_MODULE)


def _locked(name: str) -> ArenaPolicyError:
    return ArenaPolicyError(f"{name} is frozen; use {FROZEN_ARENA_OVERRIDE_FLAG} only for historical reproduction")


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
        raise SystemExit(f"LOCKED: historical ownership A/B. Add {FROZEN_ARENA_OVERRIDE_FLAG} only for deliberate reproduction.")
    previous = os.environ.get(FROZEN_ARENA_OVERRIDE_ENV)
    os.environ[FROZEN_ARENA_OVERRIDE_ENV] = "1"
    sys.argv = [arg for arg in sys.argv if arg != FROZEN_ARENA_OVERRIDE_FLAG]
    try:
        return int(_frozen_module().cli())
    finally:
        if previous is None:
            os.environ.pop(FROZEN_ARENA_OVERRIDE_ENV, None)
        else:
            os.environ[FROZEN_ARENA_OVERRIDE_ENV] = previous


if __name__ == "__main__":
    raise SystemExit(main())
