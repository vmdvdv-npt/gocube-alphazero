"""Durable tuning state and canonical path resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json

from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from .contracts import Plan, PERFORMANCE_TUNING_SCHEMA


STATE_SCHEMA = f"{PERFORMANCE_TUNING_SCHEMA}-state-v1"


def tuning_root(owner_root: str | Path, tuning_id: str) -> Path:
    """Resolve the one canonical standalone location under its owner."""

    root = Path(owner_root).resolve()
    if not tuning_id or "/" in tuning_id or "\\" in tuning_id or tuning_id in {".", ".."}:
        raise ValueError("tuning_id must be one safe path component")
    return root / "runtime" / "tuning" / tuning_id


def tuning_state_path(owner_root: str | Path, tuning_id: str) -> Path:
    return tuning_root(owner_root, tuning_id) / "state.json"


def tuning_report_path(owner_root: str | Path, tuning_id: str) -> Path:
    return tuning_root(owner_root, tuning_id) / "performance-tuning-v1.json"


def selected_profile_path(owner_root: str | Path, tuning_id: str) -> Path:
    return tuning_root(owner_root, tuning_id) / "selected-profile.json"


def _read(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read tuning state: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"tuning state must be an object: {path}")
    return payload


class TuningStateStore:
    """Small storage adapter; report files never become the authority."""

    def __init__(self, path: str | Path, *, owner_identity: Mapping[str, object] | None = None) -> None:
        self.path = Path(path).resolve()
        self.owner_identity = dict(owner_identity or {})

    @classmethod
    def for_owner(cls, owner_root: str | Path, plan: Plan) -> "TuningStateStore":
        return cls(
            tuning_state_path(owner_root, plan.tuning_id),
            owner_identity={
                "owner_id": plan.owner_id,
                "owner_root": str(Path(owner_root).resolve()),
                "topology": plan.topology,
            },
        )

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self, plan: Plan) -> dict[str, object]:
        state = _read(self.path)
        if state.get("schema_version", state.get("schema")) != STATE_SCHEMA:
            raise RuntimeError("unsupported performance tuning state schema")
        if state.get("plan_fingerprint") != plan.plan_fingerprint:
            raise RuntimeError("performance tuning plan fingerprint conflicts with durable state")
        if state.get("tuning_id") != plan.tuning_id:
            raise RuntimeError("performance tuning id conflicts with durable state")
        owner = state.get("owner")
        if self.owner_identity and isinstance(owner, Mapping):
            for key, expected in self.owner_identity.items():
                if expected is not None and owner.get(key) != expected:
                    raise RuntimeError(f"performance tuning owner identity conflicts for {key}")
        return state

    def save(self, state: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, canonical_json(dict(state)) + "\n")

    def create(self, plan: Plan) -> dict[str, object]:
        if self.exists():
            return self.load(plan)
        actions = [
            {
                "action_id": action_id,
                "mode": mode.to_dict(),
                "role": role,
                "status": "PENDING",
                "execution_profile": mode.execution_overrides,
            }
            for action_id, mode, role in plan.action_profiles()
        ]
        owner = {
            "owner_id": plan.owner_id,
            "owner_root": plan.owner_root,
            "topology": plan.topology,
        }
        owner.update(self.owner_identity)
        return {
            "schema_version": STATE_SCHEMA,
            "schema": STATE_SCHEMA,
            "plan_fingerprint": plan.plan_fingerprint,
            "tuning_id": plan.tuning_id,
            "owner": owner,
            "parent_checkpoint": plan.parent_checkpoint.to_dict(),  # type: ignore[union-attr]
            "current_checkpoint": plan.parent_checkpoint.to_dict(),  # type: ignore[union-attr]
            "planned_actions": actions,
            "observations": [],
            "failed_modes": [],
            "next_index": 0,
            "next_decision": None,
            "fallback_intent": None,
            "selected_profile": None,
            "status": "RUNNING",
            "stop_reason": None,
        }


def state_from_legacy_sweep(raw: Mapping[str, object], *, plan_fingerprint: str, tuning_id: str, owner: Mapping[str, object], parent_checkpoint: Mapping[str, object], current_checkpoint: Mapping[str, object] | None = None) -> dict[str, object]:
    """Read old ``performance_sweep`` without rewriting unknown fields."""

    observations = raw.get("observations", [])
    failures = raw.get("failed_modes", [])
    return {
        "schema_version": STATE_SCHEMA,
        "schema": STATE_SCHEMA,
        "plan_fingerprint": plan_fingerprint,
        "tuning_id": tuning_id,
        "owner": dict(owner),
        "parent_checkpoint": dict(parent_checkpoint),
        "current_checkpoint": dict(current_checkpoint or parent_checkpoint),
        "planned_actions": [],
        "observations": list(observations) if isinstance(observations, list) else [],
        "failed_modes": list(failures) if isinstance(failures, list) else [],
        "next_index": int(raw.get("next_mode_index", 0) or 0),
        "next_decision": None,
        "fallback_intent": (
            {"pending": True, "generation": raw.get("retry_baseline_generation")}
            if raw.get("retry_baseline_generation") is not None
            else None
        ),
        "selected_profile": None,
        "status": str(raw.get("status", "RUNNING")),
        "stop_reason": None,
        "legacy_performance_sweep": dict(raw),
    }


__all__ = [
    "STATE_SCHEMA", "TuningStateStore", "selected_profile_path", "state_from_legacy_sweep",
    "tuning_report_path", "tuning_root", "tuning_state_path",
]
