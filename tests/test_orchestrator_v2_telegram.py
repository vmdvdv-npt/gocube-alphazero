from __future__ import annotations

from pathlib import Path

import gocube_golden.orchestrator_v2.production_entrypoint as entrypoint


def _continuous_payload() -> dict[str, object]:
    return {
        "parent_checkpoint": {
            "topology": "torus9",
            "lineage_id": "parent",
            "checkpoint_id": "M0",
            "generation": 0,
            "path": "checkpoints/M0.pt",
            "sha256": "sha256:" + "a" * 64,
        },
        "lineage_id": "continuous",
        "effective_config": {
            "schema": "gocube-effective-config-v2",
            "version": 2,
            "topology": "torus9",
            "compatibility": {"topology": "torus9"},
            "self_play": {},
            "training": {},
            "replay": {},
            "arena": {},
            "execution": {},
            "supervision": {},
            "extensions": {},
        },
        "generations": 0,
        "arena_cadence": 5,
        "arena_config": {"games": 4, "workers": 1, "games_per_worker": 2, "inference_batch_rows": 2},
    }


def test_production_entrypoint_injects_notifier_and_flushes(monkeypatch, tmp_path: Path) -> None:
    created: list[object] = []
    flushed: list[bool] = []

    class FakeNotifier:
        def __init__(self, paths: object) -> None:
            created.append(paths)

    class FakeRunner:
        def __init__(self, config, *, resolver, notifier, **_kwargs) -> None:
            self.config = config
            self.resolver = resolver
            self.notifier = notifier

        def run(self):
            assert self.notifier is created_notifier
            return "done"

    created_notifier: object

    def make_notifier(paths: object) -> FakeNotifier:
        nonlocal created_notifier
        created_notifier = FakeNotifier(paths)
        return created_notifier

    monkeypatch.setattr(entrypoint, "TelegramNotifier", make_notifier)
    monkeypatch.setattr(entrypoint, "ContinuousTrainingRunnerV2", FakeRunner)
    monkeypatch.setattr(entrypoint, "flush_all", lambda: flushed.append(True))

    result = entrypoint.run_continuous_from_config(
        _continuous_payload(),
        runs_root=tmp_path / "runs",
    )

    assert result == "done"
    assert len(created) == 1
    assert flushed == [True]
    paths = created[0]
    assert paths.root == tmp_path / "runs" / "torus9" / "active" / "continuous"


def test_production_entrypoint_flushes_when_runner_fails(monkeypatch, tmp_path: Path) -> None:
    flushed: list[bool] = []

    class FakeRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self):
            raise RuntimeError("training failure")

    monkeypatch.setattr(entrypoint, "TelegramNotifier", lambda _paths: object())
    monkeypatch.setattr(entrypoint, "ContinuousTrainingRunnerV2", FakeRunner)
    monkeypatch.setattr(entrypoint, "flush_all", lambda: flushed.append(True))

    try:
        entrypoint.run_continuous_from_config(_continuous_payload(), runs_root=tmp_path / "runs")
    except RuntimeError as exc:
        assert str(exc) == "training failure"
    else:
        raise AssertionError("expected runner failure")

    assert flushed == [True]


def test_production_entrypoint_explicitly_allows_code_rollover(
    monkeypatch, tmp_path: Path
) -> None:
    captured: list[object] = []

    class FakeRunner:
        def __init__(self, config, *, resolver, notifier, **_kwargs) -> None:
            captured.append(config)

        def run(self):
            return "done"

    monkeypatch.setattr(entrypoint, "TelegramNotifier", lambda _paths: object())
    monkeypatch.setattr(entrypoint, "ContinuousTrainingRunnerV2", FakeRunner)
    monkeypatch.setattr(entrypoint, "flush_all", lambda: None)

    assert (
        entrypoint.run_continuous_from_config(
            _continuous_payload(),
            runs_root=tmp_path / "runs",
            allow_code_rollover=True,
        )
        == "done"
    )
    assert len(captured) == 1
    assert captured[0].allow_code_rollover is True
