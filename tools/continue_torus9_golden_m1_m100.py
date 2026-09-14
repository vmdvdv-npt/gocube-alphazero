#!/usr/bin/env python3
"""Fail-closed front door for the frozen Torus9 M1->M100 continuation."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import runpy
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.arena_policy import (
    ArenaPolicyError,
    FROZEN_ARENA_OVERRIDE_ENV,
    FROZEN_ARENA_OVERRIDE_FLAG,
)

_FROZEN_MODULE = "tools._frozen_continue_torus9_golden_m1_m100"
_BLOCKED_IMPORT_NAMES = frozenset({"run", "_backfill_missing_m5_arena"})


def _frozen_module():
    return importlib.import_module(_FROZEN_MODULE)


def _locked(name: str) -> ArenaPolicyError:
    return ArenaPolicyError(
        f"{name} belongs to the frozen Torus9 Arena runner. Invoke this script "
        f"explicitly with {FROZEN_ARENA_OVERRIDE_FLAG} only for historical reproduction."
    )


def run(*_args: Any, **_kwargs: Any) -> None:
    raise _locked("run")


def _backfill_missing_m5_arena(*_args: Any, **_kwargs: Any) -> None:
    raise _locked("_backfill_missing_m5_arena")


def request_stop() -> None:
    _frozen_module().request_stop()


def __getattr__(name: str):
    if name in _BLOCKED_IMPORT_NAMES:
        raise _locked(name)
    return getattr(_frozen_module(), name)


def _dispatch_frozen() -> None:
    previous = os.environ.get(FROZEN_ARENA_OVERRIDE_ENV)
    os.environ[FROZEN_ARENA_OVERRIDE_ENV] = "1"
    sys.argv = [argument for argument in sys.argv if argument != FROZEN_ARENA_OVERRIDE_FLAG]
    try:
        runpy.run_module(_FROZEN_MODULE, run_name="__main__")
    finally:
        if previous is None:
            os.environ.pop(FROZEN_ARENA_OVERRIDE_ENV, None)
        else:
            os.environ[FROZEN_ARENA_OVERRIDE_ENV] = previous


def main() -> None:
    arguments = sys.argv[1:]
    if "--request-stop" in arguments:
        _frozen_module().request_stop()
        return
    if "--help" in arguments and FROZEN_ARENA_OVERRIDE_FLAG not in arguments:
        print(
            "Torus9 continuation is FROZEN.\n"
            f"Historical reproduction only: {FROZEN_ARENA_OVERRIDE_FLAG}\n"
            "Safe stop remains available: --request-stop"
        )
        return
    if FROZEN_ARENA_OVERRIDE_FLAG not in arguments:
        raise SystemExit(
            "LOCKED: historical Torus9 Arena runner. Current production Arena is "
            "tools/torus9_arena.py. For deliberate reproduction only, add "
            f"{FROZEN_ARENA_OVERRIDE_FLAG}."
        )
    _dispatch_frozen()


if __name__ == "__main__":
    main()
