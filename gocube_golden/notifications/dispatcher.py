"""Event publication, durable delivery, retries, and test sinks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
import time
import weakref
from typing import Any, Callable, Mapping, Protocol

from .events import EventType, OperatorEvent, utc_now
from .formatter import DEFAULT_MESSAGE_BUDGET, format_event
from .store import DeliveryState, NotificationStore
from .telegram import TelegramTransport, TelegramTransportError


class EventSink(Protocol):
    def publish(self, event: OperatorEvent) -> OperatorEvent: ...
    def reconcile_completed(self, action_ref: object, verified_result: object) -> OperatorEvent | None: ...


@dataclass(frozen=True)
class DeliveryPolicy:
    base_delay_seconds: float = 30.0
    max_delay_seconds: float = 3600.0
    max_message_chars: int = DEFAULT_MESSAGE_BUDGET
    sender_lock_timeout_seconds: float = 0.0
    max_attempts: int = 20

    def delay(self, attempts: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(self.max_delay_seconds, max(0.0, float(retry_after)))
        exponent = max(0, int(attempts) - 1)
        return min(self.max_delay_seconds, self.base_delay_seconds * (2 ** min(exponent, 16)))


def _now_seconds() -> float:
    return time.time()


def _iso_seconds(value: str | None) -> float | None:
    if not value:
        return None


def _safe_receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"ok": True}
    result: dict[str, Any] = {}
    for key in ("ok", "message_id", "date", "chat_id", "legacy"):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            if item is not None:
                result[key] = item
    return result or {"ok": True}
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


class NullEventSink:
    """Explicitly disabled publication for callers that do not want storage."""

    structured = False

    def publish(self, event: OperatorEvent) -> OperatorEvent:
        return event

    def reconcile_completed(self, action_ref: object, verified_result: object) -> None:
        return None


class RecordingEventSink:
    def __init__(self) -> None:
        self.structured = True
        self.events: list[OperatorEvent] = []

    def publish(self, event: OperatorEvent) -> OperatorEvent:
        for existing in self.events:
            if existing.event_id == event.event_id:
                return existing
        self.events.append(event)
        return event

    def reconcile_completed(self, action_ref: object, verified_result: object) -> OperatorEvent:
        payload = dict(verified_result) if isinstance(verified_result, Mapping) else {"result": str(verified_result)}
        event = _event_from_reconciliation(action_ref, payload)
        return self.publish(event)


def _event_from_reconciliation(action_ref: object, result: Mapping[str, Any]) -> OperatorEvent:
    if "event_id" in result and "event_type" in result and "schema_version" in result:
        return OperatorEvent.from_dict(result)
    action = str(action_ref.get("action_id", "action")) if isinstance(action_ref, Mapping) else str(action_ref)
    topology = str(result.get("topology", "unknown"))
    owner_type = str(result.get("owner_type", "evaluation"))
    owner_id = str(result.get("owner_id", result.get("evaluation_id", action)))
    event_type = str(result.get("event_type", "ARENA_COMPLETED"))
    event_identity = result.get("identity") if isinstance(result.get("identity"), Mapping) else {
        "action_id": action,
        "evaluation_identity": result.get("evaluation_identity", result.get("evaluation_fingerprint", action)),
    }
    return OperatorEvent.create(
        event_type,
        topology=topology,
        owner_type=owner_type,
        owner_id=owner_id,
        action_id=action,
        payload=result,
        evidence_refs=result.get("evidence_refs"),
        launch_id=None if result.get("launch_id") is None else str(result["launch_id"]),
        attempt=None if result.get("attempt") is None else int(result["attempt"]),
        correlation_id=None if result.get("correlation_id") is None else str(result["correlation_id"]),
        producer_version=str(result.get("producer_version", "unknown")),
        execution_code_commit=None if result.get("execution_code_commit") is None else str(result["execution_code_commit"]),
        identity=event_identity,
    )


class LegacyNotifierEventSink:
    """Bridge old ``send_now(key, text)`` consumers to structured events.

    This is deliberately one-way and only exists during the compatibility
    window.  V2 coordinators depend on this sink's ``publish`` method, never on
    Telegram or message keys.
    """

    def __init__(self, notifier: object) -> None:
        self.notifier = notifier
        self.supports_starts = bool(getattr(notifier, "supports_structured_starts", False))
        self.structured = False

    def publish(self, event: OperatorEvent) -> OperatorEvent:
        sender = getattr(self.notifier, "send_now", None)
        if not callable(sender):
            return event
        try:
            legacy_message = event.payload.get("legacy_message") if isinstance(event.payload, Mapping) else None
            legacy_event = event.payload.get("legacy_event") if isinstance(event.payload, Mapping) else None
            text = f"{legacy_event} — {legacy_message}" if legacy_message and legacy_event else format_event(event)
            sender(event.event_id, text)
        except Exception:
            # Legacy notification failures are fail-open at this boundary.
            pass
        return event

    def reconcile_completed(self, action_ref: object, verified_result: object) -> OperatorEvent:
        return self.publish(_event_from_reconciliation(action_ref, dict(verified_result) if isinstance(verified_result, Mapping) else {"result": str(verified_result)}))


def coerce_event_sink(value: object | None) -> EventSink:
    if value is None:
        return NullEventSink()
    if callable(getattr(value, "publish", None)) and callable(getattr(value, "reconcile_completed", None)):
        return value  # type: ignore[return-value]
    if callable(getattr(value, "send_now", None)):
        return LegacyNotifierEventSink(value)
    return NullEventSink()


def operator_event(
    name: str,
    *,
    topology: str,
    owner_type: str,
    owner_id: str,
    action_id: str,
    message: str | None = None,
    payload: Mapping[str, Any] | None = None,
    evidence_refs: object = None,
    correlation_id: str | None = None,
    producer_version: str = "orchestrator-v2",
    execution_code_commit: str | None = None,
) -> OperatorEvent:
    """Create one typed event for legacy operator labels.

    The ``legacy_message`` field is presentation compatibility only.  New
    formatters use structured payloads; the old adapter may render this field
    as the historical ``EVENT — message`` text during migration.
    """
    upper = str(name).upper()
    if upper in EventType.__members__:
        kind = EventType[upper]
    elif upper in {"START", "TRAINING_STARTED"}:
        kind = EventType.TRAINING_STARTED
    elif upper == "GENERATION_STARTED":
        kind = EventType.GENERATION_STARTED
    elif upper in {"COMPLETED", "RUN_COMPLETED", "EXPERIMENT_COMPLETED"}:
        kind = EventType.EXPERIMENT_COMPLETED if upper == "EXPERIMENT_COMPLETED" else EventType.RUN_COMPLETED
    elif upper in {"SOFT_STOP_REQUESTED", "STOP_REQUESTED"}:
        kind = EventType.STOP_REQUESTED
    elif upper in {"SOFT_STOPPED", "RUN_STOPPED"}:
        kind = EventType.RUN_STOPPED
    elif upper.startswith("EXPERIMENT") or upper.endswith("_COMPLETED") or upper.endswith("_DECIDED"):
        kind = EventType.EXPERIMENT_STAGE_DECIDED
    elif (upper.startswith("KOMI") or upper.startswith("CALIBRATION")) and ("START" in upper or "SCHEDULE" in upper):
        kind = EventType.CALIBRATION_STARTED
    elif upper.startswith("KOMI") or upper.startswith("CALIBRATION"):
        kind = EventType.CALIBRATION_DECIDED
    elif upper in {"CRITICAL", "RUN_FAILED", "FAILED"}:
        kind = EventType.RUN_FAILED
    else:
        kind = EventType.RUN_FAILED
    data = dict(payload or {})
    if message is not None:
        data.setdefault("legacy_message", str(message))
    data.setdefault("legacy_event", upper)
    return OperatorEvent.create(
        kind,
        topology=topology,
        owner_type=owner_type,
        owner_id=owner_id,
        action_id=action_id,
        payload=data,
        evidence_refs=evidence_refs,
        correlation_id=correlation_id,
        producer_version=producer_version,
        execution_code_commit=execution_code_commit,
        identity={"event": upper, "owner_id": owner_id, "action_id": action_id, "correlation_id": correlation_id},
    )


_dispatchers: weakref.WeakSet["NotificationDispatcher"] = weakref.WeakSet()


class NotificationDispatcher:
    """Single-root at-least-once dispatcher.

    ``publish`` durably stores the event before a background worker is woken.
    The worker and explicit ``flush`` share ``dispatcher.lock``; the lock is
    never acquired by an engine-running coordinator.
    """

    def __init__(
        self,
        store: NotificationStore,
        *,
        transport: TelegramTransport | object | None = None,
        formatter: Callable[[OperatorEvent], str] = format_event,
        policy: DeliveryPolicy | None = None,
        enabled: bool = True,
        background: bool = True,
        now: Callable[[], float] = _now_seconds,
        utc_clock: Callable[[], str] = utc_now,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self.transport = transport
        self.formatter = formatter
        self.policy = policy or DeliveryPolicy()
        self.enabled = bool(enabled)
        self.background = bool(background)
        self._now = now
        self._utc_clock = utc_clock
        self._sleep = sleeper
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._close_lock = threading.Lock()
        self._closed = False
        self.thread: threading.Thread | None = None
        self.logger = logging.getLogger(__name__)
        self.structured = True
        _dispatchers.add(self)
        if self.background and self.enabled and any(state.status != "BLOCKED_CONFIGURATION" for _event, state in self.store.pending_events()):
            self._ensure_worker()

    def publish(self, event: OperatorEvent) -> OperatorEvent:
        with self._close_lock:
            if self._closed:
                raise RuntimeError("notification dispatcher is closed")
        stored = self.store.publish(event)
        if not self.enabled or self.transport is None:
            current = self.store.read_delivery(stored.event_id)
            if current is None or current.status != "DELIVERED":
                self._mark_blocked(stored.event_id, "DISABLED_CONFIGURATION")
        elif self.background:
            self._ensure_worker()
            self._wake.set()
        return stored

    def reconcile_completed(self, action_ref: object, verified_result: object) -> OperatorEvent | None:
        result = dict(verified_result) if isinstance(verified_result, Mapping) else {"result": str(verified_result)}
        return self.publish(_event_from_reconciliation(action_ref, result))

    def _ensure_worker(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._worker, name="gocube-notifications", daemon=True)
        self.thread.start()

    def _mark_blocked(self, event_id: str, code: str) -> None:
        state = self.store.read_delivery(event_id) or DeliveryState(event_id=event_id)
        try:
            self.store.write_delivery(DeliveryState(**{**state.to_dict(), "status": "BLOCKED_CONFIGURATION", "last_error_code": code, "last_attempt_at": self._utc_clock()}))
        except OSError:
            self.store._diagnose({"kind": "delivery_state_unwritten", "event_id": event_id, "error_code": code, "at": self._utc_clock()})

    def requeue_blocked(self, event_id: str | None = None) -> int:
        changed = 0
        for event, state in self.store.pending_events():
            if state.status == "BLOCKED_CONFIGURATION" and (event_id is None or event.event_id == event_id):
                self.store.write_delivery(DeliveryState(event_id=event.event_id))
                changed += 1
        if changed and self.background and self.enabled and self.transport is not None:
            self._ensure_worker()
            self._wake.set()
        return changed

    def _due(self, state: DeliveryState) -> bool:
        when = _iso_seconds(state.next_attempt_at)
        return when is None or when <= self._now()

    def _attempt(self, event: OperatorEvent, state: DeliveryState, deadline: float | None = None) -> None:
        attempts = state.attempts + 1
        attempted = DeliveryState(event_id=event.event_id, status=state.status, attempts=attempts, next_attempt_at=state.next_attempt_at, last_error_code=state.last_error_code, last_attempt_at=self._utc_clock(), delivered_at=state.delivered_at, receipt=state.receipt)
        try:
            self.store.write_delivery(attempted)
        except OSError:
            return
        try:
            text = self.formatter(event)
        except Exception as exc:
            try:
                self.store.write_delivery(DeliveryState(event_id=event.event_id, status="BLOCKED_CONFIGURATION", attempts=attempts, last_error_code="FORMAT_ERROR", last_attempt_at=attempted.last_attempt_at))
            except OSError:
                self.store._diagnose({"kind": "delivery_state_unwritten", "event_id": event.event_id, "error_code": "FORMAT_ERROR", "at": self._utc_clock()})
            return
        try:
            result: dict[str, Any] = {}
            failure: list[Exception] = []

            def send_once() -> None:
                try:
                    value = self.transport.send(text)  # type: ignore[union-attr]
                    result["receipt"] = value
                except Exception as exc:  # transport boundary, reclassified below
                    failure.append(exc)

            sender = threading.Thread(target=send_once, name="gocube-notification-attempt", daemon=True)
            sender.start()
            sender.join(timeout=None if deadline is None else max(0.0, deadline - self._now()))
            if sender.is_alive():
                next_at = self._now() + self.policy.delay(attempts)
                self.store.write_delivery(DeliveryState(event_id=event.event_id, status="RETRY_WAIT", attempts=attempts, next_attempt_at=datetime.fromtimestamp(next_at, timezone.utc).isoformat(), last_error_code="TRANSPORT_TIMEOUT", last_attempt_at=attempted.last_attempt_at))
                return
            if failure:
                raise failure[0]
            receipt = result.get("receipt", {"ok": True})
        except TelegramTransportError as exc:
            if exc.retryable and attempts < self.policy.max_attempts:
                next_at = self._now() + self.policy.delay(attempts, exc.retry_after)
                next_state = DeliveryState(event_id=event.event_id, status="RETRY_WAIT", attempts=attempts, next_attempt_at=datetime.fromtimestamp(next_at, timezone.utc).isoformat(), last_error_code=exc.code, last_attempt_at=attempted.last_attempt_at)
            else:
                next_state = DeliveryState(event_id=event.event_id, status="BLOCKED_CONFIGURATION", attempts=attempts, last_error_code=exc.code, last_attempt_at=attempted.last_attempt_at)
            try:
                self.store.write_delivery(next_state)
            except OSError:
                self.store._diagnose({"kind": "delivery_state_unwritten", "event_id": event.event_id, "error_code": exc.code, "at": self._utc_clock()})
        except Exception as exc:
            next_at = self._now() + self.policy.delay(attempts)
            try:
                status = "RETRY_WAIT" if attempts < self.policy.max_attempts else "BLOCKED_CONFIGURATION"
                self.store.write_delivery(DeliveryState(event_id=event.event_id, status=status, attempts=attempts, next_attempt_at=None if status != "RETRY_WAIT" else datetime.fromtimestamp(next_at, timezone.utc).isoformat(), last_error_code=exc.__class__.__name__ if attempts < self.policy.max_attempts else "RETRY_EXHAUSTED", last_attempt_at=attempted.last_attempt_at))
            except OSError:
                self.store._diagnose({"kind": "delivery_state_unwritten", "event_id": event.event_id, "error_code": exc.__class__.__name__, "at": self._utc_clock()})
        else:
            try:
                self.store.write_delivery(DeliveryState(event_id=event.event_id, status="DELIVERED", attempts=attempts, last_attempt_at=attempted.last_attempt_at, delivered_at=self._utc_clock(), receipt=_safe_receipt(receipt)))
            except OSError:
                # Telegram may have accepted the message.  Keeping the state
                # pending is intentional and gives at-least-once recovery.
                self.store._diagnose({"kind": "receipt_persist_failed", "event_id": event.event_id, "at": self._utc_clock()})

    def _drain(self, deadline: float) -> None:
        remaining = max(0.0, deadline - self._now())
        with self.store.sender_lock(timeout=min(self.policy.sender_lock_timeout_seconds, remaining)) as acquired:
            if not acquired:
                return
            for event, state in list(self.store.pending_events()):
                if self._now() >= deadline:
                    return
                if state.status == "BLOCKED_CONFIGURATION" or not self._due(state):
                    continue
                self._attempt(event, state, deadline)

    def _worker(self) -> None:
        while not self._stop.is_set():
            self._drain(self._now() + 0.5)
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            if not any(state.status != "BLOCKED_CONFIGURATION" for _event, state in self.store.pending_events()):
                return

    def flush(self, timeout: float = 7.0) -> None:
        deadline = self._now() + max(0.0, float(timeout))
        if self.enabled and self.transport is not None:
            self._drain(deadline)

    def close(self, timeout: float = 7.0) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        deadline = self._now() + max(0.0, float(timeout))
        self._stop.set()
        self._wake.set()
        if self.thread is not None:
            self.thread.join(timeout=max(0.0, deadline - self._now()))
        if self._now() < deadline:
            self.flush(deadline - self._now())


def flush_all(timeout: float = 7.0) -> None:
    values = list(_dispatchers)
    deadline = _now_seconds() + max(0.0, float(timeout))
    for dispatcher in values:
        remaining = max(0.0, deadline - _now_seconds())
        dispatcher.flush(remaining)


__all__ = [
    "DeliveryPolicy", "EventSink", "LegacyNotifierEventSink", "NotificationDispatcher",
    "NullEventSink", "RecordingEventSink", "coerce_event_sink", "flush_all",
]
