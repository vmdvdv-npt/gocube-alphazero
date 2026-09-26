"""Canonical storage helpers for komi calibration ledgers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..experiment.storage import ScenarioStateStore, ScenarioStorageError


def ledger_for(state: Mapping[str, object], candidates: tuple[float, ...]) -> dict[str, dict[str, object]]:
    raw = state.get("candidate_batches")
    if raw is None:
        return {f"{komi:g}": {} for komi in candidates}
    if not isinstance(raw, Mapping):
        raise ScenarioStorageError("komi calibration batch ledger is malformed")
    ledger: dict[str, dict[str, object]] = {}
    for komi in candidates:
        key = f"{komi:g}"
        bucket = raw.get(key, {})
        if not isinstance(bucket, Mapping):
            raise ScenarioStorageError(f"komi {key} batch ledger is malformed")
        ledger[key] = dict(bucket)
    return ledger


__all__ = ["ScenarioStateStore", "ScenarioStorageError", "ledger_for"]
