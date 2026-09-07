from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import c4_overnight_experiment as runner
from tools.gocube_experiment_resume import (
    STATE_SCHEMA_VERSION,
    find_resumable_experiments,
    validate_sweep_overrides,
)


def _project_skeleton(tmp_path: Path) -> None:
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")


def _cli(experiment_id: str):
    return runner.parse_args(["--experiment-id", experiment_id, "--device", "cpu"])


def _write_resume_state(root: Path, experiment_id: str, *, status: str = "RUNNING", stamp: float = 1.0):
    path = root / "training_reports" / experiment_id / "experiment-state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema_version": STATE_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "status": status,
            "started_at_epoch": stamp,
            "last_update_epoch": stamp,
        }),
        encoding="utf-8",
    )
    return path


def test_zero_argument_cli_auto_resumes_unique_interrupted_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_resume_state(tmp_path, "c4-sweep-existing", status="INTERRUPTED", stamp=10.0)

    cli = runner.parse_args([])

    assert cli.experiment_id == "c4-sweep-existing"
    assert cli.gocube_auto_resumed is True


def test_zero_argument_cli_refuses_ambiguous_multiple_resumable_experiments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_resume_state(tmp_path, "first", status="RUNNING", stamp=10.0)
    _write_resume_state(tmp_path, "second", status="INTERRUPTED", stamp=20.0)

    assert find_resumable_experiments(tmp_path / "training_reports") == ["second", "first"]
    with pytest.raises(RuntimeError, match="Multiple resumable"):
        runner.parse_args([])


def test_new_experiment_flag_bypasses_auto_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_resume_state(tmp_path, "old", status="RUNNING", stamp=10.0)
    monkeypatch.setattr(runner.time, "strftime", lambda _fmt: "c4-sweep-new")

    cli = runner.parse_args(["--new-experiment"])

    assert cli.experiment_id == "c4-sweep-new"
    assert cli.gocube_auto_resumed is False


def test_experiment_init_loads_state_instead_of_overwriting_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    cli = _cli("resume-me")
    first = runner.Experiment(cli)
    original_started = first.state["started_at_epoch"]
    first.state["bootstrap"] = {"run": "keep-this", "iteration": 7}
    first.state["status"] = "INTERRUPTED"
    first._save_state()

    second = runner.Experiment(cli)

    assert second.state["bootstrap"] == {"run": "keep-this", "iteration": 7}
    assert second.state["started_at_epoch"] == original_started
    assert second.state["resume_count"] == 1
    assert second.state["status"] == "RUNNING"


def test_schema_two_state_fails_closed_instead_of_guessing_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    cli = _cli("old-state")
    state_path = tmp_path / "training_reports" / "old-state" / "experiment-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({
            "schema_version": 2,
            "experiment_id": "old-state",
            "status": "RUNNING",
            "fixed_contract": {},
            "launch_config": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not safely resumable"):
        runner.Experiment(cli)


def _cached_arena_payload(expected: dict[str, object]) -> dict[str, object]:
    games = int(expected["number_of_games"])
    black = games // 2
    white = games - black
    return {
        "schema_version": 3,
        "run_a": expected["run_a"],
        "iteration_a": expected["iteration_a"],
        "run_b": expected["run_b"],
        "iteration_b": expected["iteration_b"],
        "seed": expected["seed"],
        "workers": expected["workers"],
        "evaluation_mode": expected["evaluation_mode"],
        "heldout_suite": expected["heldout_suite"],
        "arena_inference_batch_wait_ms": expected["arena_inference_batch_wait_ms"],
        "number_of_games": games,
        "requested_games": games,
        "wins": games // 2,
        "losses": games - games // 2,
        "draws": 0,
        "no_results": 0,
        "win_rate": 0.5,
        "win_rate_ci95": [0.4, 0.6],
        "by_color": {
            "black": {"games": black, "wins": black, "losses": 0, "draws": 0, "no_results": 0},
            "white": {"games": white, "wins": 0, "losses": white, "draws": 0, "no_results": 0},
        },
        "games_per_second": 1.0,
        "arena_contract": {
            "search_sims": expected["arena_sims"],
            "komi": expected["komi"],
        },
    }


def test_cached_arena_artifact_is_reused_and_accounted_exactly_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    experiment = runner.Experiment(_cli("arena-resume"))
    expected = experiment._arena_expected(
        run_a="a", iteration_a=7, run_b="b", iteration_b=7,
        games=128, heldout=False, seed=123, workers=16, wait_ms=1.0,
    )
    output = experiment.results / "cached.json"
    output.write_text(json.dumps(_cached_arena_payload(expected)), encoding="utf-8")
    monkeypatch.setattr(
        experiment,
        "stream_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Arena must not rerun")),
    )

    first = experiment.arena(
        run_a="a", iteration_a=7, run_b="b", iteration_b=7,
        games=128, name="cached", seed=123,
    )
    total_after_first = experiment.state["totals"]["arena_games"]
    second = experiment.arena(
        run_a="a", iteration_a=7, run_b="b", iteration_b=7,
        games=128, name="cached", seed=123,
    )

    assert first == second
    assert total_after_first == 128
    assert experiment.state["totals"]["arena_games"] == 128
    assert "arena:cached" in experiment.state["completed_actions"]


def test_cached_arena_with_wrong_seed_fails_closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    experiment = runner.Experiment(_cli("arena-mismatch"))
    expected = experiment._arena_expected(
        run_a="a", iteration_a=7, run_b="b", iteration_b=7,
        games=128, heldout=False, seed=123, workers=16, wait_ms=1.0,
    )
    payload = _cached_arena_payload(expected)
    payload["seed"] = 999
    (experiment.results / "bad.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="seed"):
        experiment.arena(
            run_a="a", iteration_a=7, run_b="b", iteration_b=7,
            games=128, name="bad", seed=123,
        )


def test_interrupted_one_sided_clone_is_quarantined_then_rebuilt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    experiment = runner.Experiment(_cli("clone-resume"))
    parent = "parent"
    target = "clone-resume-p1-l-10p0"
    (tmp_path / "checkpoint" / parent).mkdir(parents=True)
    (tmp_path / "data" / parent).mkdir(parents=True)
    (tmp_path / "checkpoint" / parent / "iteration-0007.pkl").write_bytes(b"parent")
    (tmp_path / "data" / parent / "iteration-0007-data.pkl").write_bytes(b"parent-data")
    (tmp_path / "checkpoint" / target).mkdir(parents=True)
    (tmp_path / "checkpoint" / target / "half-copy").write_text("partial", encoding="utf-8")

    experiment._ensure_clone(
        parent_run=parent,
        parent_iteration=7,
        target_run=target,
        kind="parameter:P1",
    )

    assert (tmp_path / "checkpoint" / target / runner.PROVENANCE_FILENAME).is_file()
    assert (tmp_path / "data" / target / "iteration-0007-data.pkl").is_file()
    quarantines = experiment.state.get("quarantined_namespaces") or []
    assert quarantines
    assert quarantines[-1]["run"] == target


def test_sweep_override_resume_validation_catches_wrong_candidate_value():
    saved = {
        "gocube_chosen_move_temperature_halflife": 19.0,
        "gocube_root_dirichlet_noise_weight": 0.25,
        "probFastSim": 0.25,
        "gocube_train_samples_per_new_sample": 1.0,
        "gocube_replay_window_iters": 8,
    }
    validate_sweep_overrides(saved, {"--replay-window-iters": 8})
    with pytest.raises(RuntimeError, match="replay-window-iters"):
        validate_sweep_overrides(saved, {"--replay-window-iters": 16})


def test_restore_completed_stage_resumes_from_saved_champion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    experiment = runner.Experiment(_cli("stage-resume"))
    experiment.state["parameters"] = [{
        "id": "P1",
        "parent": {"run": "bootstrap", "iteration": 7},
        "winner": {"decision": "IMPROVED"},
        "champion_after": {
            "run": "p1-winner",
            "iteration": 9,
            "sweep_overrides": {"--chosen-move-temperature-halflife": 19.0},
        },
    }]
    monkeypatch.setattr(experiment, "_validate_checkpoint", lambda *args, **kwargs: None)

    run, iteration, overrides, status, reason = experiment._restore_completed_stages("bootstrap")

    assert run == "p1-winner"
    assert iteration == 9
    assert overrides == {"--chosen-move-temperature-halflife": 19.0}
    assert status is None
    assert reason is None
