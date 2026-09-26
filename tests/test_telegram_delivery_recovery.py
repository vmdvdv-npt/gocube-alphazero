from types import SimpleNamespace
import threading

from gocube_golden import telegram_notifier as tg


def test_background_retry_delivers_without_another_training_event(tmp_path, monkeypatch):
    sent = threading.Event()
    attempts = []
    def transport(*args):
        attempts.append(args[-1])
        if len(attempts) == 1:
            raise tg.TelegramError("HTTP 503")
        sent.set()
    monkeypatch.setattr(tg, "load_config", lambda: ("fake-token", "fake-chat"))
    monkeypatch.setattr(tg, "_send", transport)
    monkeypatch.setattr(tg.TelegramNotifier, "retry_interval_seconds", 0.01)
    notifier = tg.TelegramNotifier(SimpleNamespace(runtime=tmp_path / "runtime", logs=tmp_path / "logs"))
    notifier.send_now("completed", "arena result")
    worker = notifier.thread
    assert sent.wait(timeout=2)
    if worker is not None:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert attempts == ["arena result", "arena result"]
    assert not list(notifier.outbox.glob("*.json"))


def test_async_enqueue_is_flushed_for_direct_instance(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(tg, "load_config", lambda: ("fake-token", "fake-chat"))
    monkeypatch.setattr(tg, "_send", lambda *args: sent.append(args[-1]))
    notifier = tg.TelegramNotifier(SimpleNamespace(runtime=tmp_path / "runtime", logs=tmp_path / "logs"))
    notifier.enqueue("completed", "arena result")
    worker = notifier.thread
    tg.flush_all(timeout=2)
    if worker is not None:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert sent == ["arena result"]
    assert not list(notifier.outbox.glob("*.json"))
