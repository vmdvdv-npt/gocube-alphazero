"""Deterministic, bounded human presentation of structured operator facts."""
from __future__ import annotations

from typing import Any, Mapping

from .events import OperatorEvent


DEFAULT_MESSAGE_BUDGET = 3900


def _value(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list, tuple)):
        return None
    return str(value)


def _line(lines: list[str], label: str, value: object) -> None:
    text = _value(value)
    if text is not None:
        lines.append(f"{label}: {text}")


def _ref(value: object) -> str | None:
    if isinstance(value, Mapping):
        lineage = value.get("lineage_id", value.get("lineage"))
        checkpoint = value.get("checkpoint_id", value.get("checkpoint"))
        if lineage is not None and checkpoint is not None:
            return f"{lineage}/{checkpoint}"
        if value.get("label") is not None:
            return str(value["label"])
    return _value(value)


def _format_arena(event: OperatorEvent) -> str:
    payload = event.payload
    title = "ARENA COMPLETED" if event.event_type == "ARENA_COMPLETED" else event.event_type.replace("_", " ")
    lines = [f"{title} — GoCube AlphaZero", ""]
    _line(lines, "Topology", event.topology)
    _line(lines, "Owner", f"{event.owner_type}/{event.owner_id}")
    _line(lines, "Action", event.action_id)
    _line(lines, "Evaluation", payload.get("evaluation_id", event.action_id))
    _line(lines, "Candidate", _ref(payload.get("candidate")))
    _line(lines, "Reference", _ref(payload.get("reference")))
    _line(lines, "Candidate lineage", payload.get("candidate_lineage"))
    _line(lines, "Reference lineage", payload.get("reference_lineage"))
    if payload.get("wld") is not None:
        wld = payload.get("wld")
        if isinstance(wld, (list, tuple)) and len(wld) == 3:
            _line(lines, "W/L/D", "/".join(str(item) for item in wld))
    _line(lines, "Validity", payload.get("validity"))
    _line(lines, "Execution commit", event.execution_code_commit)
    _line(lines, "Report", payload.get("evaluation_report", payload.get("report_ref")))
    for label, key in (
        ("Games", "games"),
        ("Valid games", "valid_games"),
        ("Technical games", "technical_games"),
        ("Performance", "performance_status"),
        ("Mean batch", "inference_mean_batch_rows"),
    ):
        _line(lines, label, payload.get(key))
    if event.event_type == "ARENA_FAILED":
        _line(lines, "Error", payload.get("error_code", payload.get("error")))
    return "\n".join(lines)


def format_event(event: OperatorEvent, *, max_chars: int = DEFAULT_MESSAGE_BUDGET) -> str:
    """Render an event without inventing missing measurements.

    If the bounded message budget is exceeded, the durable event and its
    evidence remain complete; only the chat presentation is shortened.
    """
    if event.event_type in {"ARENA_STARTED", "ARENA_COMPLETED", "ARENA_FAILED"}:
        text = _format_arena(event)
    else:
        title = event.event_type.replace("_", " ")
        lines = [f"{title} — GoCube AlphaZero", ""]
        _line(lines, "Topology", event.topology)
        _line(lines, "Owner", f"{event.owner_type}/{event.owner_id}")
        _line(lines, "Action", event.action_id)
        for key, value in event.payload.items():
            if key in {"stack_trace", "traceback", "token", "url"}:
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, (str, int, float, bool)):
                _line(lines, label, value)
        _line(lines, "Evidence", event.evidence_refs[0].get("ref") if event.evidence_refs else None)
        text = "\n".join(lines)
    limit = max(128, int(max_chars))
    if len(text) <= limit:
        return text
    suffix = "\n[details truncated; see local evidence]"
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


__all__ = ["DEFAULT_MESSAGE_BUDGET", "format_event"]
