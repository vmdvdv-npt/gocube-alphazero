"""Compatibility-preserving storage adapter for experiment scenarios.

The first extraction keeps the existing ``state.json`` location and shape.  A
scenario decision is written before the next expensive action; projections can
be rebuilt from this canonical state and the referenced Arena evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from ...process_supervision import atomic_write_text
from ...provenance import canonical_json, sha256_fingerprint


class ScenarioStorageError(RuntimeError):
    """Canonical scenario state is missing, malformed, or inconsistent."""


class ScenarioStateStore:
    """Read/write one owner's small canonical scenario state."""

    def __init__(self, root: str | Path, *, filename: str = "state.json") -> None:
        self.root = Path(root).resolve()
        self.path = self.root / filename

    def read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScenarioStorageError(f"cannot read canonical scenario state: {self.path}") from exc
        if not isinstance(payload, Mapping):
            raise ScenarioStorageError(f"canonical scenario state is not an object: {self.path}")
        return dict(payload)

    def write(self, payload: Mapping[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, canonical_json(dict(payload)) + "\n")

    def exists(self) -> bool:
        return self.path.is_file()


def plan_fingerprint(plan: Mapping[str, object]) -> str:
    return sha256_fingerprint(dict(plan))


def validate_plan_identity(
    state: Mapping[str, object],
    *,
    scenario_id: str,
    plan_fingerprint_value: str,
) -> None:
    if state.get("experiment_id", state.get("calibration_id")) != scenario_id:
        raise ScenarioStorageError("scenario state owner id does not match the plan")
    stored = state.get("config_fingerprint", state.get("plan_fingerprint"))
    if stored != plan_fingerprint_value:
        raise ScenarioStorageError("scenario plan fingerprint changed during resume")


__all__ = ["ScenarioStateStore", "ScenarioStorageError", "plan_fingerprint", "validate_plan_identity"]
