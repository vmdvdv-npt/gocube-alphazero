"""Atomic local persistence for operator events and delivery state."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

from ..process_supervision import atomic_write_json, atomic_write_text
from .events import OperatorEvent, utc_now


DELIVERY_STATUSES = frozenset(
    {"PENDING", "RETRY_WAIT", "DELIVERED", "BLOCKED_CONFIGURATION", "DELIVERY_UNCERTAIN"}
)


def _safe_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create without replacing a concurrent first writer.

    ``atomic_write_json`` is the normal replacement primitive.  Event identity
    additionally needs no-clobber semantics, so a complete private file is
    linked into place; ``link`` is atomic and fails if another process won.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class DeliveryState:
    event_id: str
    status: str = "PENDING"
    attempts: int = 0
    next_attempt_at: str | None = None
    last_error_code: str | None = None
    last_attempt_at: str | None = None
    delivered_at: str | None = None
    receipt: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in DELIVERY_STATUSES:
            raise ValueError(f"unknown delivery status: {self.status}")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "status": self.status,
            "attempts": self.attempts,
            "next_attempt_at": self.next_attempt_at,
            "last_error_code": self.last_error_code,
            "last_attempt_at": self.last_attempt_at,
            "delivered_at": self.delivered_at,
            "receipt": None if self.receipt is None else dict(self.receipt),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeliveryState":
        return cls(
            event_id=str(value["event_id"]),
            status=str(value.get("status", "PENDING")),
            attempts=int(value.get("attempts", 0)),
            next_attempt_at=None if value.get("next_attempt_at") is None else str(value["next_attempt_at"]),
            last_error_code=None if value.get("last_error_code") is None else str(value["last_error_code"]),
            last_attempt_at=None if value.get("last_attempt_at") is None else str(value["last_attempt_at"]),
            delivered_at=None if value.get("delivered_at") is None else str(value["delivered_at"]),
            receipt=value.get("receipt") if isinstance(value.get("receipt"), Mapping) else None,
        )


class NotificationStore:
    """Own one stable notification root.

    The dispatcher lock is held only while a dispatcher drains one root.  The
    orchestrator never acquires it while an engine is executing.  Event files
    and delivery files use the same unique-temp-file atomic writer as process
    supervision, so a crash leaves either the old complete file or the new
    complete file.
    """

    def __init__(self, owner_root: str | Path, *, now=utc_now) -> None:
        self.owner_root = Path(owner_root).resolve()
        self.root = self.owner_root / "notifications"
        self.events_dir = self.root / "events"
        self.delivery_dir = self.root / "delivery"
        self.lock_path = self.root / "dispatcher.lock"
        self.diagnostics_path = self.root / "diagnostics.jsonl"
        self._now = now

    @staticmethod
    def event_hash(event_id: str) -> str:
        return _safe_hash(event_id)

    def event_path(self, event_id: str) -> Path:
        return self.events_dir / f"{self.event_hash(event_id)}.json"

    def delivery_path(self, event_id: str) -> Path:
        return self.delivery_dir / f"{self.event_hash(event_id)}.json"

    def _diagnose(self, payload: Mapping[str, Any]) -> None:
        try:
            self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            # Diagnostics must not turn a completed computation into a failed
            # computation.  The primary event/evidence remains authoritative.
            return

    def _quarantine(self, path: Path, reason: str) -> None:
        quarantine = path.with_name(f"{path.name}.corrupt.{int(time.time() * 1000000)}")
        try:
            os.replace(path, quarantine)
        except OSError:
            return
        self._diagnose({"kind": "corrupt_notification_file", "path": str(path), "quarantine": str(quarantine), "reason": reason, "at": self._now()})

    def publish(self, event: OperatorEvent) -> OperatorEvent:
        path = self.event_path(event.event_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            try:
                existing = OperatorEvent.from_dict(_read_object(path))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                self._quarantine(path, exc.__class__.__name__)
            else:
                if existing.identity_dict() != event.identity_dict():
                    self._diagnose({
                        "kind": "event_integrity_conflict",
                        "event_id": event.event_id,
                        "at": self._now(),
                        "first_evidence_refs": list(existing.evidence_refs),
                        "second_evidence_refs": list(event.evidence_refs),
                    })
                self.ensure_delivery(existing.event_id)
                return existing
        try:
            _atomic_create_json(path, event.to_dict())
            self.ensure_delivery(event.event_id)
        except FileExistsError:
            # Another process published the same logical fact.  Re-enter the
            # immutable comparison path without replacing its evidence.
            return self.publish(event)
        except (OSError, TypeError, ValueError) as exc:
            self._diagnose({"kind": "event_persist_failed", "event_id": event.event_id, "at": self._now(), "error_code": exc.__class__.__name__})
            raise
        return event

    def read_event(self, event_id: str) -> OperatorEvent | None:
        path = self.event_path(event_id)
        try:
            return OperatorEvent.from_dict(_read_object(path)) if path.is_file() else None
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            self._quarantine(path, exc.__class__.__name__)
            return None

    def iter_events(self) -> Iterator[OperatorEvent]:
        if not self.events_dir.is_dir():
            return
        for path in sorted(self.events_dir.glob("*.json")):
            try:
                yield OperatorEvent.from_dict(_read_object(path))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                self._quarantine(path, exc.__class__.__name__)

    def ensure_delivery(self, event_id: str) -> DeliveryState:
        existing = self.read_delivery(event_id)
        if existing is not None:
            return existing
        state = DeliveryState(event_id=event_id)
        try:
            atomic_write_json(self.delivery_path(event_id), state.to_dict())
        except OSError as exc:
            self._diagnose({"kind": "delivery_state_persist_failed", "event_id": event_id, "at": self._now(), "error_code": exc.__class__.__name__})
            raise
        return state

    def read_delivery(self, event_id: str) -> DeliveryState | None:
        path = self.delivery_path(event_id)
        if not path.is_file():
            return None
        try:
            return DeliveryState.from_dict(_read_object(path))
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._quarantine(path, exc.__class__.__name__)
            return None

    def write_delivery(self, state: DeliveryState) -> DeliveryState:
        try:
            atomic_write_json(self.delivery_path(state.event_id), state.to_dict())
        except OSError as exc:
            self._diagnose({"kind": "delivery_state_persist_failed", "event_id": state.event_id, "at": self._now(), "error_code": exc.__class__.__name__})
            raise
        return state

    def pending_events(self) -> Iterator[tuple[OperatorEvent, DeliveryState]]:
        for event in self.iter_events():
            state = self.read_delivery(event.event_id) or self.ensure_delivery(event.event_id)
            if state.status != "DELIVERED":
                yield event, state

    @contextmanager
    def sender_lock(self, *, timeout: float = 0.0) -> Iterator[bool]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        acquired = False
        deadline = time.monotonic() + max(0.0, timeout)
        try:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(min(0.01, deadline - time.monotonic()))
            yield acquired
        finally:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


__all__ = ["DELIVERY_STATUSES", "DeliveryState", "NotificationStore"]
