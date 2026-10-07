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
    def __init__(self, failures=0, block=False, error="HTTP_503"):
        self.error = error
        self.failures = failures
        self.block = block
        self.calls = 0

    def send(self, text):
        self.calls += 1
        if self.block:
            time.sleep(0.2)
        if self.calls <= self.failures:
            raise TelegramTransportError(self.error, retryable=True)
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
    failing = FakeTransport(failures=1, error="HTTP_429")
    first = NotificationDispatcher(store, transport=failing, background=False, policy=DeliveryPolicy(base_delay_seconds=0))
    event = first.publish(arena_event())
    first.flush(1)
    assert store.read_delivery(event.event_id).status == "RETRY_WAIT"
    succeeding = FakeTransport()
    second = NotificationDispatcher(store, transport=succeeding, background=False, policy=DeliveryPolicy(base_delay_seconds=0))
    second.flush(1)
    assert succeeding.calls == 1
    assert len(list(store.iter_events())) == 1


def test_arena_started_ambiguous_delivery_is_not_retried(tmp_path: Path):
    store = NotificationStore(tmp_path)
    event = OperatorEvent.create(
        "ARENA_STARTED",
        topology="torus9",
        owner_type="evaluation",
        owner_id="evaluation-start",
        action_id="evaluation-start",
        payload={
            "evaluation_id": "evaluation-start",
            "candidate": "M137 5CH",
            "reference": "M137 5CH",
            "games": 1024,
        },
        producer_version="test",
        execution_code_commit="commit-a",
        identity={"evaluation_identity": "sha256:" + "c" * 64, "phase": "started"},
    )
    transport = FakeTransport(failures=1)
    dispatcher = NotificationDispatcher(
        store,
        transport=transport,
        background=False,
        policy=DeliveryPolicy(base_delay_seconds=0),
    )

    stored = dispatcher.publish(event)
    dispatcher.flush(1)

    state = store.read_delivery(stored.event_id)
    assert transport.calls == 1
    assert state is not None
    assert state.status == "DELIVERY_UNCERTAIN"
    assert state.attempts == 1

    resumed = NotificationDispatcher(
        store,
        transport=FakeTransport(),
        background=False,
        policy=DeliveryPolicy(base_delay_seconds=0),
    )
    resumed.flush(1)
    assert resumed.store.read_delivery(stored.event_id).status == "DELIVERY_UNCERTAIN"


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
    (output / "runtime").mkdir()
    (output / "runtime" / "execution-intent.json").write_text("{}", encoding="utf-8")
    (output / "runtime" / "incident-evidence-20260926.json").write_text("{}", encoding="utf-8")
    (output / "runtime" / "unrelated.json").write_text("{}", encoding="utf-8")
    CoreArenaRunner._clear_incomplete_evaluation(output)
    assert (output / "notifications" / "events" / "event.json").is_file()
    assert (output / "runtime" / "execution-intent.json").is_file()
    assert (output / "runtime" / "incident-evidence-20260926.json").is_file()
    assert not (output / "runtime" / "unrelated.json").exists()
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


def test_slow_send_is_not_repeated_by_flush_or_new_dispatcher(tmp_path):
    import threading
    started, release = threading.Event(), threading.Event()
    class SlowTransport:
        calls = 0
        def send(self, text):
            self.calls += 1
            started.set()
            assert release.wait(5)
            return {'ok': True, 'message_id': 123}
    transport = SlowTransport()
    store = NotificationStore(tmp_path)
    first = NotificationDispatcher(store, transport=transport, background=False)
    event = first.publish(arena_event())
    first.flush(.02)
    assert started.wait(1)
    assert store.read_delivery(event.event_id).status == 'DELIVERY_UNCERTAIN'
    second = NotificationDispatcher(NotificationStore(tmp_path), transport=transport, background=False)
    second.publish(event)
    second.flush(.02)
    assert transport.calls == 1
    release.set()
    for _ in range(100):
        if store.read_delivery(event.event_id).status == 'DELIVERED':
            break
        time.sleep(.01)
    assert store.read_delivery(event.event_id).receipt['message_id'] == 123
    second.flush(1)
    assert transport.calls == 1


def test_retry_after_is_respected(tmp_path):
    from datetime import datetime, timezone
    from gocube_golden.notifications.store import DeliveryState
    clock = [1000.0]
    store = NotificationStore(tmp_path)
    transport = FakeTransport()
    dispatcher = NotificationDispatcher(store, transport=transport, background=False, now=lambda: clock[0])
    event = dispatcher.publish(arena_event())
    store.write_delivery(DeliveryState(event_id=event.event_id, status='RETRY_WAIT',
        next_attempt_at=datetime.fromtimestamp(1060, timezone.utc).isoformat()))
    dispatcher.flush(1)
    assert transport.calls == 0
    clock[0] = 1061
    dispatcher.flush(1)
    assert transport.calls == 1


def test_ambiguous_result_and_corrupt_receipt_are_never_retried(tmp_path):
    store = NotificationStore(tmp_path)
    transport = FakeTransport(failures=1)
    dispatcher = NotificationDispatcher(store, transport=transport, background=False)
    event = dispatcher.publish(arena_event())
    dispatcher.flush(1)
    dispatcher.flush(1)
    assert transport.calls == 1
    assert store.read_delivery(event.event_id).status == 'DELIVERY_UNCERTAIN'
    store.delivery_path(event.event_id).write_text('{broken')
    dispatcher.publish(event)
    dispatcher.flush(1)
    dispatcher.flush(1)
    assert transport.calls == 1


def test_delivery_initialization_cannot_overwrite_concurrent_receipt(tmp_path, monkeypatch):
    from gocube_golden.notifications.store import DeliveryState
    store = NotificationStore(tmp_path)
    event = store.publish(arena_event())
    store.write_delivery(DeliveryState(event_id=event.event_id, status='DELIVERED'))
    read = store.read_delivery
    calls = []
    def stale_read(event_id):
        calls.append(event_id)
        return None if len(calls) == 1 else read(event_id)
    monkeypatch.setattr(store, 'read_delivery', stale_read)
    assert store.ensure_delivery(event.event_id).status == 'DELIVERED'
    assert read(event.event_id).status == 'DELIVERED'


def test_crash_after_http_begins_does_not_resend_on_restart(tmp_path):
    import subprocess
    import sys
    import os
    script = tmp_path / 'crash.py'
    script.write_text('''import os, sys
from pathlib import Path
from gocube_golden.notifications.dispatcher import NotificationDispatcher
from gocube_golden.notifications.store import NotificationStore
class Transport:
    def send(self, text):
        Path(sys.argv[1], 'http-started').write_text('yes')
        os._exit(17)
d = NotificationDispatcher(NotificationStore(sys.argv[1]), transport=Transport(), background=False)
d.flush(5)
''')
    store = NotificationStore(tmp_path)
    store.publish(arena_event())
    result = subprocess.run([sys.executable, str(script), str(tmp_path)], env=dict(os.environ, PYTHONPATH=str(Path.cwd())), timeout=20)
    assert result.returncode == 17
    assert (tmp_path / 'http-started').exists()
    transport = FakeTransport()
    resumed = NotificationDispatcher(store, transport=transport, background=False)
    resumed.flush(1)
    assert transport.calls == 0


def test_late_rate_limit_restarts_background_delivery(tmp_path):
    import threading
    sent = threading.Event()
    class Transport:
        calls = 0
        def send(self, text):
            self.calls += 1
            if self.calls == 1:
                time.sleep(.7)
                raise TelegramTransportError('HTTP_429', retryable=True, retry_after=.01)
            sent.set()
            return {'ok': True}
    transport = Transport()
    dispatcher = NotificationDispatcher(NotificationStore(tmp_path), transport=transport)
    dispatcher.publish(arena_event())
    assert sent.wait(5)
    dispatcher.close()
    assert transport.calls == 2


def test_arena_score_immediately_follows_completed_title():
    lines = format_event(arena_event()).splitlines()
    assert lines[:2] == ["🟢 ARENA COMPLETED — GoCube AlphaZero", "W/L/D: 8/8/0"]
    assert sum(line.startswith("W/L/D:") for line in lines) == 1


def test_telegram_request_bolds_sections_and_iterations_and_escapes_values():
    from gocube_golden.notifications.telegram import TelegramTransport

    bodies = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return b'{"ok":true,"result":{"message_id":1}}'

    def opener(request, **_kwargs):
        bodies.append(json.loads(request.data))
        return Response()

    transport = TelegramTransport("test-token", "test-chat", opener=opener)
    transport.send("🎬 TRAINING STARTED — GoCube AlphaZero\nParent: M201\nSelf-play:\nTraining:\nReplay:\nArena:\nNetwork: Golden-M201-5CH\nNote: x < y & <b>literal</b>")
    body = bodies[0]
    assert body["parse_mode"] == "HTML"
    assert "Parent: <b>M201</b>" in body["text"]
    for section in ("Self-play", "Training", "Replay", "Arena"):
        assert f"<b>{section}:</b>" in body["text"]
    assert "Golden-<b>M201</b>-5CH" in body["text"]
    assert "x &lt; y &amp; &lt;b&gt;literal&lt;/b&gt;" in body["text"]
    transport.send("ARENA STARTED\nCandidate: M202\nReference: M201")
    assert "Candidate: <b>M202</b>" in bodies[1]["text"]
    assert "Reference: <b>M201</b>" in bodies[1]["text"]


def test_offline_training_message_hides_inherited_selfplay_and_disabled_arena():
    event = OperatorEvent.create('TRAINING_STARTED', topology='torus9', owner_type='experiment',
        owner_id='abcd', action_id='abcd:A', payload={
            'lineage_id': 'abcd-A', 'execution_mode': 'offline', 'parent': 'parent/M255',
            'self_play': {'games_per_iteration': 1536, 'search_mode': 'pcr',
                          'pcr': {'cheap_simulations': 100, 'full_simulations': 400, 'full_probability': .33}},
            'training': {'learning_rate': 2.5e-5, 'batch_size': 64, 'optimizer_steps': 2560},
            'replay': {'generations': 5}, 'arena': {'enabled': False, 'every_iterations': 5, 'games': 192},
            'stop_after_iterations': 5})
    text = format_event(event)
    assert 'Lineage: abcd-A' in text and 'Mode: offline replay' in text
    assert 'Disabled — 0 new games' in text
    assert 'Periodic Arena disabled' in text
    assert 'LR=2.5e-05' in text and 'Batch: 64' in text and 'Steps: 2560' in text
    assert 'games/generation' not in text and 'PCR' not in text and 'every 5' not in text
    assert 'Games: 192' not in text


def test_training_formatter_respects_disabled_arena_without_offline_mode():
    event = OperatorEvent.create('TRAINING_STARTED', topology='torus9', owner_type='lineage',
        owner_id='online', action_id='start', payload={'training': {'batch_size': 64},
        'self_play': {'games_per_iteration': 1536}, 'arena': {'enabled': False}})
    text = format_event(event)
    assert 'games/generation=1536' in text
    assert 'Arena cadence' not in text
