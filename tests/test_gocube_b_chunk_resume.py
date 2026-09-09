from __future__ import annotations

from pathlib import Path

from tools import gocube_b_experiment as runner


def _args(*extra: str):
    return runner.parse_args(["--treatment", "B0", "--dry-run", *extra])


def _child_iterations(command: list[str]) -> int:
    return int(command[command.index("--iterations") + 1])


def test_b_launcher_accepts_and_forwards_existing_run_resume():
    args = _args("--allow-existing-run")
    command = runner.training_command(args, python="python")

    assert args.allow_existing_run is True
    assert "--allow-existing-run" in command


def test_fresh_operational_chunk_uses_chunk_size_as_child_ceiling(tmp_path):
    args = _args("--run-name", "night-b0", "--chunk-iterations", "16")

    ceiling = runner.resolve_iteration_ceiling(args, repo=tmp_path)
    command = runner.training_command(
        args,
        python="python",
        iteration_ceiling=ceiling,
    )

    assert ceiling == 16
    assert _child_iterations(command) == 16
    assert "--allow-existing-run" not in command


def test_resumed_operational_chunk_advances_absolute_child_ceiling(monkeypatch, tmp_path):
    args = _args(
        "--run-name",
        "night-b0",
        "--chunk-iterations",
        "16",
        "--allow-existing-run",
    )
    seen = []

    def fake_last_valid(path: Path):
        seen.append(path)
        return 16, []

    monkeypatch.setattr(runner, "find_last_valid_contiguous_checkpoint", fake_last_valid)

    ceiling = runner.resolve_iteration_ceiling(args, repo=tmp_path)
    command = runner.training_command(
        args,
        python="python",
        iteration_ceiling=ceiling,
    )

    assert seen == [tmp_path / "checkpoint" / "night-b0"]
    assert ceiling == 32
    assert _child_iterations(command) == 32
    assert "--allow-existing-run" in command


def test_resumed_chunk_requires_a_valid_checkpoint(monkeypatch, tmp_path):
    args = _args(
        "--run-name",
        "night-b0",
        "--chunk-iterations",
        "16",
        "--allow-existing-run",
    )
    monkeypatch.setattr(
        runner,
        "find_last_valid_contiguous_checkpoint",
        lambda _path: (None, []),
    )

    try:
        runner.resolve_iteration_ceiling(args, repo=tmp_path)
    except RuntimeError as exc:
        assert "requires a valid existing checkpoint" in str(exc)
    else:
        raise AssertionError("missing checkpoint must fail closed")


def test_completed_chunk_normalizes_only_known_post_chunk_ceiling_failure(monkeypatch, tmp_path):
    args = _args("--run-name", "night-b0", "--chunk-iterations", "16")
    monkeypatch.setattr(
        runner,
        "find_last_valid_contiguous_checkpoint",
        lambda _path: (16, []),
    )
    stderr = (
        "Traceback (most recent call last):\n"
        "RuntimeError: Scientific sample target was not reached before the iteration "
        "safety ceiling: cumulative_new_samples=40000000, current=382684\n"
    )

    assert runner.normalize_chunk_returncode(
        1,
        stderr,
        args=args,
        iteration_ceiling=16,
        repo=tmp_path,
    ) == 0


def test_chunk_does_not_mask_unrelated_child_failure(monkeypatch, tmp_path):
    args = _args("--run-name", "night-b0", "--chunk-iterations", "16")
    monkeypatch.setattr(
        runner,
        "find_last_valid_contiguous_checkpoint",
        lambda _path: (16, []),
    )

    assert runner.normalize_chunk_returncode(
        1,
        "RuntimeError: CUDA out of memory\n",
        args=args,
        iteration_ceiling=16,
        repo=tmp_path,
    ) == 1


def test_chunk_does_not_normalize_ceiling_failure_without_committed_boundary(monkeypatch, tmp_path):
    args = _args("--run-name", "night-b0", "--chunk-iterations", "16")
    monkeypatch.setattr(
        runner,
        "find_last_valid_contiguous_checkpoint",
        lambda _path: (15, []),
    )

    assert runner.normalize_chunk_returncode(
        1,
        runner._ITERATION_CEILING_ERROR,
        args=args,
        iteration_ceiling=16,
        repo=tmp_path,
    ) == 1


def test_non_chunk_run_preserves_scientific_safety_ceiling_failure(monkeypatch, tmp_path):
    args = _args("--run-name", "scientific-b0", "--iterations", "16")
    monkeypatch.setattr(
        runner,
        "find_last_valid_contiguous_checkpoint",
        lambda _path: (16, []),
    )

    assert runner.resolve_iteration_ceiling(args, repo=tmp_path) == 16
    assert runner.normalize_chunk_returncode(
        1,
        runner._ITERATION_CEILING_ERROR,
        args=args,
        iteration_ceiling=16,
        repo=tmp_path,
    ) == 1
