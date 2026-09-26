from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gocube_golden import telegram_notifier as tg


def paths(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "lineage-a"
    for name in ("runtime", "logs", "metrics"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "lineage_id": "lineage-a",
                "config_fingerprint": "sha256:test-config",
                "orchestrator": {"arena_generations": [37, 47]},
                "operator_tunables": {
                    "learning_rate": 0.0003,
                    "replay_generations": 6,
                    "replay_cap": 40000,
                    "replay_window": "rolling last 6 generations",
                    "self_play_mcts_simulations": 128,
                    "arena_every_generations": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "runtime/state.json").write_text(
        json.dumps(
            {
                "state": "RUNNING",
                "last_committed_generation": 47,
                "active_generation": 47,
                "active_phase": "arena",
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        root=root,
        manifest=root / "manifest.json",
        runtime=root / "runtime",
        runtime_state=root / "runtime/state.json",
        logs=root / "logs",
        metrics=root / "metrics",
    )


def test_filtering_and_start_tunables(tmp_path: Path) -> None:
    p = paths(tmp_path)
    start = tg.build_notification(p, "INFO", "Training orchestrator started", {})
    assert start is not None
    text = start[1]
    for expected in (
        "Training started",
        "LR: 0.0003",
        "Replay: 6 generations / 40000 positions",
        "Self-play MCTS: 128 sims",
        "Arena cadence: every 5 generations",
    ):
        assert expected in text
    assert (
        tg.build_notification(
            p,
            "WARNING",
            "Arena performance warning; Arena accepted",
            {"generation": 47},
        )
        is None
    )
    assert "WARNING / DEGRADED" in tg.build_notification(
        p, "WARNING", "Disk free space low", {}
    )[1]


def test_arena_score(tmp_path: Path) -> None:
    p = paths(tmp_path)
    arena = p.root / "arena/generation-0047"
    arena.mkdir(parents=True)
    (arena / "result.json").write_text(
        json.dumps(
            {
                "reference_generation": 37,
                "technical_games": 0,
                "invalid_games": 0,
                "metrics": {
                    "games": 64,
                    "wins": 33,
                    "losses": 31,
                    "draws": 0,
                    "inference_mean_batch_rows": 7.92,
                    "performance_status": "SEVERE_WARNING",
                },
            }
        ),
        encoding="utf-8",
    )
    text = tg.build_notification(
        p, "INFO", "Arena completed", {"generation": 47}
    )[1]
    for expected in (
        "M47 vs M37",
        "W/L/D: 33/31/0",
        "Valid: 64",
        "Technical: 0",
        "Mean batch: 7.92",
    ):
        assert expected in text


def test_critical_recovery(tmp_path: Path) -> None:
    p = paths(tmp_path)
    p.runtime_state.write_text(
        json.dumps({"state": "RECOVERY_REQUIRED", "last_committed_generation": 47}),
        encoding="utf-8",
    )
    text = tg.build_notification(p, "CRITICAL", "boom", {})[1]
    assert "CRITICAL / RECOVERY_REQUIRED" in text
    assert "M47" in text


def test_failure_is_fail_open_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = paths(tmp_path)
    monkeypatch.setattr(tg, "load_config", lambda: ("TOP_SECRET", "42"))
    monkeypatch.setattr(
        tg,
        "_send",
        lambda *_: (_ for _ in ()).throw(tg.TelegramError("HTTP 500")),
    )
    notifier = tg.TelegramNotifier(p)
    notifier.send_now("warning:test", "test")
    log = (p.logs / "telegram-notifier-errors.jsonl").read_text(encoding="utf-8")
    assert "HTTP 500" in log
    assert "TOP_SECRET" not in log


def test_dedupe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = paths(tmp_path)
    monkeypatch.setattr(tg, "load_config", lambda: ("token", "42"))
    sent: list[str] = []
    monkeypatch.setattr(tg, "_send", lambda _token, _chat, text: sent.append(text))
    notifier = tg.TelegramNotifier(p)
    notifier.send_now("arena:47:37", "first")
    notifier.send_now("arena:47:37", "duplicate")
    assert sent == ["first"]


@pytest.fixture(autouse=True)
def no_background_transport(monkeypatch):
    # Drive retries explicitly; never leave a daemon using restored real HTTP.
    monkeypatch.setattr(tg.TelegramNotifier, "_ensure_worker", lambda self: None)


def test_failed_delivery_survives_restart(tmp_path, monkeypatch):
    p = paths(tmp_path)
    monkeypatch.setattr(tg, "load_config", lambda: ("token", "42"))
    def unavailable(*args):
        raise tg.TelegramError("HTTP 503")
    monkeypatch.setattr(tg, "_send", unavailable)
    first = tg.TelegramNotifier(p)
    first.send_now("arena-complete:test", "W/L/D: 8/8/0")
    assert len(list(first.outbox.glob("*.json"))) == 1
    sent = []
    monkeypatch.setattr(tg, "_send", lambda *args: sent.append(args[-1]))
    second = tg.TelegramNotifier(p)
    second._retry_pending()
    second._retry_pending()
    assert sent == ["W/L/D: 8/8/0"]
    assert not list(second.outbox.glob("*.json"))


def test_delivery_receipt_disk_error_is_fail_open(tmp_path, monkeypatch):
    p = paths(tmp_path)
    monkeypatch.setattr(tg, "load_config", lambda: ("token", "42"))
    monkeypatch.setattr(tg, "_send", lambda *args: None)
    notifier = tg.TelegramNotifier(p)
    notifier.delivered.mkdir()  # Simulate an unwritable receipt destination.
    notifier.send_now("result", "done")
    assert list(notifier.outbox.glob("*.json"))
    assert notifier.errors.is_file()


def test_corrupt_receipt_line_does_not_hide_later_deliveries(tmp_path, monkeypatch):
    p = paths(tmp_path)
    monkeypatch.setattr(tg, "load_config", lambda: ("token", "42"))
    sent = []
    monkeypatch.setattr(tg, "_send", lambda *args: sent.append(args[-1]))
    notifier = tg.TelegramNotifier(p)
    notifier.delivered.write_text('broken\n{"key":"result"}\n')
    notifier.send_now("result", "duplicate")
    assert not sent


def test_flush_all_includes_directly_constructed_notifiers(tmp_path, monkeypatch):
    notifier = tg.TelegramNotifier(paths(tmp_path))
    calls = []
    monkeypatch.setattr(notifier, "flush", lambda timeout: calls.append(timeout))
    tg.flush_all()
    assert len(calls) == 1
