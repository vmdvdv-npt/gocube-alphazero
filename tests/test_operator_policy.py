from __future__ import annotations

from types import SimpleNamespace

from gocube_golden.operator_policy import (
    NO_PROGRESS_ALERT_SECONDS,
    _apply_model_degradation_gate,
    _integrity_failure,
)


class Events:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, dict[str, object]]] = []

    def emit(self, level: str, message: str, **details: object) -> None:
        self.rows.append((level, message, details))


class FakeRun:
    def __init__(self, wins: int, losses: int, draws: int = 0) -> None:
        self.events = Events()
        self.spec = SimpleNamespace(soft_stop=SimpleNamespace(minimum_minutes=30))
        self._metrics = {"wins": wins, "losses": losses, "draws": draws}
        self.stop_calls: list[tuple[int, str]] = []

    def _history_rows(self):
        return [{"kind": "arena", "generation": 47, "metrics": self._metrics}]

    def request_soft_stop(self, minutes: int, *, reason: str):
        self.stop_calls.append((minutes, reason))
        return {}


def test_model_degradation_stops_only_when_losses_exceed_wins():
    run = FakeRun(31, 33)
    assert _apply_model_degradation_gate(run, 47) is True
    assert run.stop_calls == [(30, "arena-model-degradation")]
    level, message, details = run.events.rows[-1]
    assert level == "CRITICAL"
    assert "31/33/0" in message
    assert details["losses"] == 33


def test_equal_or_winning_arena_does_not_stop():
    tied = FakeRun(32, 32)
    winning = FakeRun(33, 31)
    assert _apply_model_degradation_gate(tied, 47) is False
    assert _apply_model_degradation_gate(winning, 47) is False
    assert tied.stop_calls == []
    assert winning.stop_calls == []


def test_only_training_integrity_failures_are_classified_as_integrity():
    assert _integrity_failure(ValueError("checkpoint SHA256 mismatch"))
    assert _integrity_failure(ValueError("replay artifact hash mismatch"))
    assert _integrity_failure(ValueError("Generation contains technical/invalid games"))
    assert not _integrity_failure(RuntimeError("driver made no observable progress for 3600s"))
    assert not _integrity_failure(RuntimeError("worker health degraded: 15/16 alive"))
    assert not _integrity_failure(RuntimeError("Arena technical/invalid outcomes are fail-closed"))


def test_no_progress_alert_is_thirty_minutes():
    assert NO_PROGRESS_ALERT_SECONDS == 1800.0
