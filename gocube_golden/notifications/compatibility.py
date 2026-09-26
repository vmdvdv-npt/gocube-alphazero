"""Read-only compatibility helpers for the PR215 Telegram files.

Migration intentionally does not guess that a legacy key is a new event ID.
Only an explicit evidence mapping can suppress a new delivery.  Original
outbox/receipt files are never removed by these helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..process_supervision import atomic_write_json
from .dispatcher import LegacyNotifierEventSink
from .events import OperatorEvent, event_id_for
from .store import NotificationStore


@dataclass(frozen=True)
class LegacyEnvelope:
    key: str
    text: str
    source: Path
    delivered: bool = False


def _legacy_root(root: str | Path) -> Path:
    value = Path(root).resolve()
    # Callers may pass the owner root or the PR215 ``runtime`` directory.
    return value / "runtime" if (value / "runtime" / "telegram-outbox").is_dir() or (value / "runtime" / "telegram-notifications.jsonl").is_file() else value


def read_legacy_receipts(root: str | Path) -> dict[str, Mapping[str, Any]]:
    path = _legacy_root(root) / "telegram-notifications.jsonl"
    result: dict[str, Mapping[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return result
    for line in lines:
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping) and isinstance(value.get("key"), str):
            result[str(value["key"])] = dict(value)
    return result


def read_legacy_outbox(root: str | Path) -> Iterator[LegacyEnvelope]:
    base = _legacy_root(root) / "telegram-outbox"
    receipts = read_legacy_receipts(root)
    if not base.is_dir():
        return
    for path in sorted(base.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping) or not isinstance(value.get("key"), str) or not isinstance(value.get("text"), str):
            continue
        key = str(value["key"])
        yield LegacyEnvelope(key=key, text=str(value["text"]), source=path, delivered=key in receipts)


def legacy_key_event_id(old_key: str, *, evidence: Mapping[str, Any] | None = None) -> str | None:
    """Return a mapping only when the caller supplied explicit evidence."""
    if not evidence:
        return None
    identity = dict(evidence)
    identity["legacy_key"] = str(old_key)
    return event_id_for("DELIVERY_DEGRADED", identity)


def migrate_legacy_storage(root: str | Path, store: NotificationStore, *, mapping: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, int]:
    mapping = mapping or {}
    receipts = read_legacy_receipts(root)
    migrated = 0
    delivered = 0
    pending = 0
    unknown = 0
    for envelope in read_legacy_outbox(root):
        evidence = mapping.get(envelope.key)
        if evidence is None:
            unknown += 1
            store._diagnose({"kind": "legacy_notification_unmapped", "key": envelope.key, "source": str(envelope.source)})
            evidence = {}
        event = OperatorEvent.create(
            "DELIVERY_DEGRADED",
            topology=str(evidence.get("topology", "unknown")),
            owner_type=str(evidence.get("owner_type", "workflow")),
            owner_id=str(evidence.get("owner_id", envelope.key)),
            action_id=str(evidence.get("action_id", envelope.key)),
            payload={"legacy_envelope": True, "legacy_key": envelope.key, "text": envelope.text},
            evidence_refs=[{"legacy_key": envelope.key, "source": str(envelope.source)}],
            producer_version="legacy-compatibility",
            identity={"legacy_key": envelope.key, "legacy_envelope": True, **dict(evidence)},
        )
        stored = store.publish(event)
        migrated += 1
        if envelope.key in receipts:
            delivered += 1
            # A successful old receipt is a durable fact; no new Telegram send.
            old = receipts[envelope.key]
            state = store.read_delivery(stored.event_id)
            if state is not None and state.status != "DELIVERED":
                from .store import DeliveryState

                store.write_delivery(DeliveryState(event_id=stored.event_id, status="DELIVERED", delivered_at=str(old.get("at", "legacy")), receipt={"legacy": True, "key": envelope.key}))
        else:
            pending += 1
    marker = store.root / "legacy-migration.json"
    atomic_write_json(marker, {"schema": "gocube-legacy-notification-migration-v1", "checked_at": store._now(), "migrated": migrated, "delivered": delivered, "pending": pending, "unknown": unknown})
    return {"migrated": migrated, "delivered": delivered, "pending": pending, "unknown": unknown}


__all__ = [
    "LegacyEnvelope", "LegacyNotifierEventSink", "legacy_key_event_id",
    "migrate_legacy_storage", "read_legacy_outbox", "read_legacy_receipts",
]
