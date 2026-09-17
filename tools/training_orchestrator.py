#!/usr/bin/env python3
"""Production training CLI with fail-open Telegram observability."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import training_orchestrator_core as _core
from tools.training_orchestrator_core import *  # noqa: F401,F403
from gocube_golden.operator_policy import install_operator_policy
from gocube_golden.telegram_notifier import (
    TelegramError,
    flush_all,
    install,
    notify_stop_requested,
    telegram_test,
)

# Detached supervisors must re-enter this wrapper instead of bypassing Telegram.
_core.__file__ = __file__


def __getattr__(name: str) -> object:
    return getattr(_core, name)


def _cmd_stop_with_telegram(args: object) -> int:
    run = _core._from_lineage(args, terminal=False)
    payload = run.request_soft_stop(args.minutes, reason="terminal-command")
    notify_stop_requested(run.paths, payload)
    print(
        "Soft-stop requested. The active safe unit will not be hard-killed and no new "
        f"generation/Arena will start afterward. Target window ends at {payload['target_deadline_at']}."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "telegram-test":
        if len(values) != 1:
            raise SystemExit("usage: training_orchestrator.py telegram-test")
        try:
            telegram_test()
        except TelegramError as exc:
            raise SystemExit(f"Telegram test failed: {exc}") from None
        print("Telegram test message sent.")
        return 0

    # Operator policy is process-local and deliberately does not mutate the
    # immutable scientific run-spec/fingerprint.
    install_operator_policy()
    install()
    _core._cmd_stop = _cmd_stop_with_telegram
    try:
        return int(_core.main(values))
    finally:
        # Final stop/crash/Arena notifications are queued during the run.
        flush_all()


if __name__ == "__main__":
    raise SystemExit(main())
