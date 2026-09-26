"""Derived, reproducible performance-tuning reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from .contracts import Plan, SelectedProfile


REPORT_SCHEMA = "gocube-performance-tuning-report-v1"
LEGACY_REPORT_SCHEMA = "gocube-orchestrator-v2-self-play-concurrency-sweep-v1-report-v1"


def build_report(plan: Plan, state: Mapping[str, object]) -> dict[str, object]:
    observations = state.get("observations", [])
    failures = state.get("failed_modes", [])
    selected = state.get("selected_profile")
    if not isinstance(observations, list):
        observations = []
    if not isinstance(failures, list):
        failures = []
    return {
        "schema": REPORT_SCHEMA,
        "tuning_id": plan.tuning_id,
        "plan_fingerprint": plan.plan_fingerprint,
        "topology": plan.topology,
        "parent_checkpoint": plan.parent_checkpoint.to_dict(),  # type: ignore[union-attr]
        "status": state.get("status", "RUNNING"),
        "source": "observations_and_decisions",
        "plan": plan.to_dict(),
        "selected_profile": selected,
        "observations": [dict(item) for item in observations if isinstance(item, Mapping)],
        "failed_modes": [dict(item) for item in failures if isinstance(item, Mapping)],
        "decision": state.get("next_decision"),
        "limitations": [
            "games_per_hour and self-play wall time are generation self-play metrics; they do not represent the full generation lifecycle.",
            "Sequential measurements may use evolving checkpoints; this report is not a same-checkpoint A/B benchmark unless the plan explicitly provides that topology.",
            "Execution scheduling differences do not imply bit-for-bit self-play or weight identity across parallelism profiles.",
        ],
        "telemetry_sources": {
            "games": {"source": "generation summary.orchestrator_selfplay.games", "unit": "games", "missing": "unstable", "aggregation": "expected-count check"},
            "games_per_hour": {"source": "generation summary.orchestrator_selfplay.games_per_hour", "unit": "games/hour", "missing": "unstable", "aggregation": "arithmetic mean among stable observations"},
            "selfplay_wall_time_sec": {"source": "generation summary.orchestrator_selfplay.selfplay_time_sec", "unit": "seconds", "missing": "unstable", "aggregation": "arithmetic mean among stable observations"},
            "cycle_wall_time_sec": {"source": "adjacent persisted generation request mtimes", "unit": "seconds", "missing": "not applicable to self-play rating", "aggregation": "diagnostic only"},
            "execution": {"source": "generation summary.orchestrator_selfplay.execution", "unit": "integer topology values", "missing": "unstable", "aggregation": "must equal planned profile"},
            "gpu_utilization_percent": {"source": "summary metrics/inference/timing first finite value", "unit": "percent", "missing": "allowed diagnostic", "aggregation": "not used for selection"},
            "gpu_power_w": {"source": "summary metrics/inference/timing first finite value", "unit": "watts", "missing": "allowed diagnostic", "aggregation": "not used for selection"},
            "cpu_utilization_percent": {"source": "summary metrics/inference/timing first finite value", "unit": "percent", "missing": "allowed diagnostic", "aggregation": "not used for selection"},
        },
    }


def write_report(path: str | Path, plan: Plan, state: Mapping[str, object]) -> dict[str, object]:
    report = build_report(plan, state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, canonical_json(report) + "\n")
    return report


def selected_profile_payload(value: SelectedProfile | Mapping[str, object]) -> dict[str, object]:
    profile = value if isinstance(value, SelectedProfile) else SelectedProfile.from_dict(value)
    return profile.to_dict()


__all__ = ["LEGACY_REPORT_SCHEMA", "REPORT_SCHEMA", "build_report", "selected_profile_payload", "write_report"]
