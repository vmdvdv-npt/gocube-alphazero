"""Structured operator events and durable notification delivery."""
from __future__ import annotations
from .compatibility import (
    LegacyEnvelope,
    legacy_key_event_id,
    migrate_legacy_storage,
    read_legacy_outbox,
    read_legacy_receipts,
)
from .dispatcher import (
    DeliveryPolicy,
    EventSink,
    LegacyNotifierEventSink,
    NotificationDispatcher,
    NullEventSink,
    RecordingEventSink,
    coerce_event_sink,
    flush_all,
    operator_event,
)
from .events import (
    EVENT_TYPES,
    EventType,
    OperatorEvent,
    OPERATOR_EVENT_SCHEMA,
    OWNER_TYPES,
    VALIDITIES,
    event_id_for,
)
from .formatter import DEFAULT_MESSAGE_BUDGET, format_event
from .store import DELIVERY_STATUSES, DeliveryState, NotificationStore
from .telegram import TelegramTransport, TelegramTransportError
from pathlib import Path


def create_telegram_dispatcher(owner_root: str | Path) -> NotificationDispatcher:
    """Compose the one production dispatcher for an owner root.

    Missing Telegram configuration disables network delivery while retaining
    a bounded, inspectable local event/delivery state.
    """
    from .telegram import load_config

    config = load_config()
    transport = None if config is None else TelegramTransport(*config)
    return NotificationDispatcher(
        NotificationStore(owner_root),
        transport=transport,
        enabled=config is not None,
    )

__all__ = [
    "DEFAULT_MESSAGE_BUDGET", "DELIVERY_STATUSES", "DeliveryPolicy", "DeliveryState",
    "EVENT_TYPES", "EventSink", "EventType", "LegacyEnvelope", "LegacyNotifierEventSink",
    "NotificationDispatcher", "NotificationStore", "NullEventSink", "OPERATOR_EVENT_SCHEMA",
    "OperatorEvent", "OWNER_TYPES", "RecordingEventSink", "TelegramTransport",
    "TelegramTransportError", "VALIDITIES", "coerce_event_sink", "event_id_for",
    "flush_all", "format_event", "legacy_key_event_id", "migrate_legacy_storage", "operator_event",
    "read_legacy_outbox", "read_legacy_receipts", "create_telegram_dispatcher",
]
