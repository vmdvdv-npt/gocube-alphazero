#!/usr/bin/env python3
"""Fail-closed front door for the frozen Torus9 M1->M100 continuation.

The historical implementation is retained only for deliberate reproduction.
Normal imports may use its read-only helpers, but execution entry points are
blocked unless the operator invokes this script with ``--allow-frozen-arena``.
``--request-stop`` always remains available.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import runpy
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.arena_policy import (
    ArenaPolicyError,
    FROZEN_ARENA_OVERRIDE_FLAG,
)


_FROZEN_MODULE = "tools._frozen_continue_torus9_golden_m1_m100"
_BLOCKED_IMPORT_NAMES = frozenset({"run", "_backfill_missing_m5_arena"})


def _frozen_module():
    return importlib.import_module(_FROZEN_MODULE)


def _locked(name: str) -> ArenaPolicyError:
    return ArenaPolicyError(
        f"{name} belongs to the frozen logical-lane Torus9 Arena runner. "
        f"Invoke this script explicitly with {FROZEN_ARENA_OVERRIDE_FLAG} "
        "only for deliberate historical reproduction."
    )


def run(*_args: Any, **_kwargs: Any) -> None:
    """Programmatic execution is intentionally disabled."""
    raise _locked("run")


def _backfill_missing_m5_arena(*_args: Any, **_kwargs: Any) -> None:
    """Programmatic frozen-Arena backfill is intentionally disabled."""
    raise _locked("_backfill_missing_m5_arena")


def request_stop() -> None:
    """Stopping an already-running frozen job must never require an override."""
    _frozen_module().request_stop()


def __getattr__(name: str):
    if name in _BLOCKED_IMPORT_NAMES:
        raise _locked(name)
    return getattr(_frozen_module(), name)


def _dispatch_frozen() -> None:
    sys.argv = [argument for argument in sys.argv if argument != FROZEN_ARENA_OVERRIDE_FLAG]
    runpy.run_module(_FROZEN_MODULE, run_name="__main__")


def main() -> None:
    arguments = sys.argv[1:]
    if "--request-stop" in arguments:
        _frozen_module().request_stop()
        return
    if "--help" in arguments and FROZEN_ARENA_OVERRIDE_FLAG not in arguments:
        print(
            "Torus9 continuation is LOCKED because its Arena uses degraded "
            "single-process logical lanes.\n"
            f"Historical reproduction only: {FROZEN_ARENA_OVERRIDE_FLAG}\n"
            "Safe stop remains available: --request-stop"
        )
        return
    if FROZEN_ARENA_OVERRIDE_FLAG not in arguments:
        raise SystemExit(
            "LOCKED: this continuation runner contains the frozen logical-lane "
            "Torus9 Arena. Current Golden production is disabled until the "
            "16-OS-process + one-central-CUDA-broker Arena passes the M18 gate. "
            f"For deliberate historical reproduction only, add "
            f"{FROZEN_ARENA_OVERRIDE_FLAG}."
        )
    _dispatch_frozen()


if __name__ == "__main__":
    main()
