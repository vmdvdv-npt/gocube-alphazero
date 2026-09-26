from __future__ import annotations

from pathlib import Path
import json
import threading
import time

from gocube_golden.notifications import (
    DeliveryPolicy,
    NotificationDispatcher,
    NotificationStore,
    OperatorEvent,
    RecordingEventSink,
    TelegramTransportError,
    format_event,
    migrate_legacy_storage,
)
from gocube_golden.orchestrator_v2._arena_runner_core import ArenaRunner as CoreArenaRunner


def arena_event(*, recorded_at: str = "2026-09-26T00:00:00+00:00", wld=(8, 8, 0), evidence=None):
    return OperatorEvent.create(
        "ARENA_COMPLETED",
        topology="torus9",
        owner_type="evaluation",
        owner_id="evaluation-1",
        action_id="evaluation-1",
        recorded_at=recorded_at,
        occurred_at="2026-09-26T00:00:00+00:00",
        payload={
            "evaluation_id": "evaluation-1",
            "candidate": "M95",
            "reference": "M90",
            "candidate_lineage": "lineage-a",
            "reference_lineage": "lineage-a",
            "wld": list(wld),
            "validity": "VALID",
            "games": 16,
        },
        evidence_refs=evidence or [{"ref": "arena/result.json", "sha256": "sha256:" + "a" * 64}],
        producer_version="test",
        execution_code_commit="commit-a",
        identity={"evaluation_identity": "sha256:" + "b" * 64},
    )


def test_event_id_ignores_recording_time_and_formatter_does_not_invent_values():
    first = arena_event(recorded_at="2026-09-26T00:00:00+00:00")
    second = arena_event(recorded_at="2026-09-27T00:00:00+00:00")
    assert first.event_id == second.event_id
    text = format_event(first)
    assert "W/L/D: 8/8/0" in text
    assert "Synthetic" not in text
    assert "Performance:" not in text


def test_concurrent_publish_keeps_one_immutable_event(tmp_path: Path):
    store = NotificationStore(tmp_path)
    events = [arena_event(recorded_at=f"2026-09-26T00:00:{index:02d}+00:00") for index in range(2)]
    errors = []

    def publish(event):
        try:
            store.publish(event)
        except Exception as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(event,)) for event in events]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    stored = list(store.iter_events())
    assert len(stored) == 1
    assert stored[0].event_id == events[0].event_id


def test_same_id_with_different_result_is_diagnosed_without_overwrite(tmp_path: Path):
    store = NotificationStore(tmp_path)
    first = store.publish(arena_event())
    conflicting = arena_event(wld=(7, 9, 0), evidence=[{"ref": "other/result.json"}])
    assert store.publish(conflicting).payload["wld"] == [8, 8, 0]
    diagnostics = (tmp_path / "notifications" / "diagnostics.jsonl").read_text()
    assert "event_integrity_conflict" in diagnostics
    assert "arena/result.json" in diagnostics
    assert "other/result.json" in diagnostics
    assert first.event_id == conflicting.event_id


class FakeTransport:
    def __init__(self, failures=0, block=False):
        self.failures = failures
        self.block = block
        self.calls = 0

    def send(self, text):
        self.calls += 1
        if self.block:
            time.sleep(0.2)
        if self.calls <= self.failures:
            raise TelegramTransportError("HTTP_503", retryable=True)
        return {"ok": True, "message_id": self.calls}


def test_receipt_prevents_resume_http_and_retry_does_not_recompute(tmp_path: Path):
    store = NotificationStore(tmp_path)
    transport = FakeTransport()
    dispatcher = NotificationDispatcher(store, transport=transport, background=False, policy=DeliveryPolicy(base_delay_seconds=0))
    event = dispatcher.publish(arena_event())
    dispatcher.flush(1)
    assert transport.calls == 1
    assert store.read_delivery(event.event_id).status == "DELIVERED"
    resumed = NotificationDispatcher(store, transport=transport, background=False, policy=DeliveryPolicy(base_delay_seconds=0))
    resumed.flush(1)
    assert transport.calls == 1


def test_retry_wait_survives_restart_and_does_not_create_second_event(tmp_path: Path):
    store = NotificationStore(tmp_path)
    failing = FakeTransport(failures=1)
    first = NotificationDispatcher(store, transport=failing, background=False, policy=DeliveryPolicy(base_delay_seconds=0))
    event = first.publish(arena_event())
    first.flush(1)
    assert store.read_delivery(event.event_id).status == "RETRY_WAIT"
    succeeding = FakeTransport()
    second = NotificationDispatcher(store, transport=succeeding, background=False, policy=DeliveryPolicy(base_delay_seconds=0))
    second.flush(1)
    assert succeeding.calls == 1
    assert len(list(store.iter_events())) == 1


def test_flush_budget_leaves_slow_transport_pending(tmp_path: Path):
    store = NotificationStore(tmp_path)
    transport = FakeTransport(block=True)
    dispatcher = NotificationDispatcher(store, transport=transport, background=False, policy=DeliveryPolicy(base_delay_seconds=0))
    event = dispatcher.publish(arena_event())
    started = time.monotonic()
    dispatcher.flush(0.03)
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    assert store.read_delivery(event.event_id).status != "DELIVERED"


def test_corrupt_one_event_does_not_stop_other_delivery(tmp_path: Path):
    store = NotificationStore(tmp_path)
    first = store.publish(arena_event())
    second = store.publish(arena_event(evidence=[{"ref": "second.json"}], recorded_at="2026-09-26T01:00:00+00:00"))
    # Different evidence is not identity; make a distinct logical event.
    assert first.event_id == second.event_id
    # Publish an explicit second identity for the continuation check.
    second = OperatorEvent.create(
        "RUN_COMPLETED", topology="torus9", owner_type="lineage", owner_id="lineage-a", action_id="lineage-a:done", payload={"state": "COMPLETED"}, producer_version="test", identity={"action": "done"}
    )
    store.publish(second)
    paths = list((tmp_path / "notifications" / "events").glob("*.json"))
    paths[0].write_text("{broken", encoding="utf-8")
    transport = FakeTransport()
    dispatcher = NotificationDispatcher(store, transport=transport, background=False, policy=DeliveryPolicy(base_delay_seconds=0))
    dispatcher.flush(1)
    assert transport.calls == 1


def test_recording_sink_reconcile_is_idempotent():
    sink = RecordingEventSink()
    result = {"topology": "torus9", "evaluation_id": "eval", "evaluation_fingerprint": "sha256:" + "a" * 64, "wld": [1, 0, 0], "validity": "VALID"}
    first = sink.reconcile_completed("eval", result)
    second = sink.reconcile_completed("eval", result)
    assert first.event_id == second.event_id
    assert len(sink.events) == 1


def test_arena_reclamation_keeps_new_notification_root(tmp_path: Path):
    output = tmp_path / "evaluation"
    (output / "notifications" / "events").mkdir(parents=True)
    (output / "notifications" / "events" / "event.json").write_text("{}", encoding="utf-8")
    (output / "notifications" / "delivery").mkdir()
    (output / "notifications" / "delivery" / "state.json").write_text("{}", encoding="utf-8")
    CoreArenaRunner._clear_incomplete_evaluation(output)
    assert (output / "notifications" / "events" / "event.json").is_file()
    assert (output / "notifications" / "delivery" / "state.json").is_file()


def test_legacy_pending_and_receipt_are_read_without_mass_resend(tmp_path: Path):
    runtime = tmp_path / "runtime"
    outbox = runtime / "telegram-outbox"
    outbox.mkdir(parents=True)
    (outbox / "a.json").write_text(json.dumps({"key": "arena-complete:old", "text": "old result"}), encoding="utf-8")
    (outbox / "b.json").write_text(json.dumps({"key": "unknown", "text": "leave me diagnosable"}), encoding="utf-8")
    (runtime / "telegram-notifications.jsonl").write_text('{broken\n{"key":"arena-complete:old","at":"2026-09-26T00:00:00+00:00"}\n', encoding="utf-8")
    store = NotificationStore(tmp_path)
    result = migrate_legacy_storage(
        tmp_path,
        store,
        mapping={"arena-complete:old": {"topology": "torus9", "owner_type": "evaluation", "owner_id": "old", "action_id": "old"}},
    )
    assert result == {"migrated": 2, "delivered": 1, "pending": 1, "unknown": 1}
    states = [store.read_delivery(event.event_id).status for event in store.iter_events()]
    assert states.count("DELIVERED") == 1
    assert states.count("PENDING") == 1
    assert (tmp_path / "notifications" / "legacy-migration.json").is_file()
    assert (outbox / "a.json").is_file()
