from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Callable, Mapping

import pytest

from gocube_golden.run_spec import StrictRunSpec
from tools import torus9_staged_sims_harness_impl as harness
from tools.torus9_staged_sims_driver import (
    BATCH_SIZE,
    ExperimentCadenceTrainingAdapter,
)


ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env_without_pythonpath() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def test_staged_harness_direct_help_bootstraps_repository_root() -> None:
    result = subprocess.run(
        [sys.executable, "tools/torus9_staged_sims_harness.py", "--help"],
        cwd=ROOT,
        env=_subprocess_env_without_pythonpath(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "ModuleNotFoundError" not in result.stderr


def test_staged_driver_direct_help_bootstraps_repository_root(tmp_path: Path) -> None:
    spec = harness.load_experiment_spec()
    arm = harness.arms_from_spec(spec)["g64"]
    materialized = harness.build_arm_run_spec(spec, arm)
    spec_path = tmp_path / "run-spec.json"
    spec_path.write_text(
        json.dumps(materialized.payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = StrictRunSpec.load(spec_path, repo_root=ROOT)
    env = _subprocess_env_without_pythonpath()
    env["AZ_RUN_SPEC_PATH"] = str(spec_path)
    env["AZ_RUN_SPEC_FINGERPRINT"] = reloaded.fingerprint

    result = subprocess.run(
        [
            sys.executable,
            "tools/torus9_staged_sims_driver.py",
            "generation",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "ModuleNotFoundError" not in result.stderr


def test_staged_training_reports_optimizer_progress_without_real_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer_steps = 160
    adapter = ExperimentCadenceTrainingAdapter(optimizer_steps=optimizer_steps)
    progress: list[tuple[str, int, int, str, str]] = []

    def record_progress(
        phase: str,
        *,
        completed: int,
        total: int,
        unit: str,
        subphase: str,
    ) -> None:
        progress.append((phase, completed, total, unit, subphase))

    adapter.set_progress_callback(record_progress)
    monkeypatch.setattr(adapter, "validate_state", lambda _state: None)

    class FakeTrainer:
        def _sample_indices(self, row_count: int, *, seed: int, count: int) -> list[int]:
            assert row_count == 1
            assert seed == 23
            assert count == optimizer_steps * BATCH_SIZE
            return [0] * count

        def train_fixed_budget(
            self,
            rows: object,
            *,
            seed: int,
            validate_samples: bool,
            timing: object,
            progress_callback: Callable[[int, int], None],
        ) -> dict[str, int]:
            del rows, timing
            assert seed == 23
            assert validate_samples is False
            for completed in (1, 80, optimizer_steps):
                progress_callback(completed, optimizer_steps)
            return {
                "optimizer_steps": optimizer_steps,
                "samples_consumed": optimizer_steps * BATCH_SIZE,
            }

    state = SimpleNamespace(adapter_state=FakeTrainer())
    metrics = adapter.train(
        state,
        ({"replay_row_id": "row-0"},),
        seed=23,
    )

    assert progress == [
        ("training", 1, optimizer_steps, "optimizer_steps", "optimizer"),
        ("training", 80, optimizer_steps, "optimizer_steps", "optimizer"),
        (
            "training",
            optimizer_steps,
            optimizer_steps,
            "optimizer_steps",
            "optimizer",
        ),
    ]
    assert metrics["optimizer_steps"] == optimizer_steps
    assert metrics["samples_consumed"] == optimizer_steps * BATCH_SIZE
    assert metrics["cadence_optimizer_steps_per_iteration"] == optimizer_steps


def _arena_case() -> tuple[
    dict[str, object],
    harness.Evaluation,
    dict[str, dict[str, object]],
    dict[str, object],
    str,
    str,
]:
    spec = harness.load_experiment_spec()
    arms = harness.arms_from_spec(spec)
    evaluation = harness.evaluations_from_spec(spec, set(arms))["g128-vs-g64"]
    arm_results = {
        "g128": {
            "lineage_id": "experiment-g128",
            "generation": 50,
            "sha256": "a" * 64,
        },
        "g64": {
            "lineage_id": "experiment-g64",
            "generation": 53,
            "sha256": "b" * 64,
        },
    }
    candidate = harness._checkpoint_reference(arm_results[evaluation.candidate])
    reference = harness._checkpoint_reference(arm_results[evaluation.reference])
    config = harness.arena_config(spec, evaluation)
    identity, fingerprint = harness._resolved_evaluation_identity(
        spec,
        evaluation,
        candidate,
        reference,
        config,
    )
    run_id = harness._evaluation_run_id(candidate, reference, fingerprint)
    return spec, evaluation, arm_results, identity, fingerprint, run_id


def _write_fake_committed_arena(
    *,
    candidate_path: Mapping[str, object],
    reference_path: Mapping[str, object],
    output_dir: Path,
    run_id: str,
    master_seed: int,
    config: object,
    wld: list[int] | None = None,
    **_kwargs: object,
) -> dict[str, object]:
    selected_wld = list(wld or [80, 48, 0])
    games = int(getattr(config, "games"))
    summary: dict[str, object] = {
        "games": games,
        "W/L/D": selected_wld,
        "telemetry": {
            "technical_games": 0,
            "performance_status": "HEALTHY",
            "performance_failures": [],
        },
    }
    provenance = {
        "candidate": {
            "lineage_id": candidate_path["lineage_id"],
            "generation": candidate_path["generation"],
            "artifact_sha256": candidate_path["artifact_sha256"],
        },
        "reference": {
            "lineage_id": reference_path["lineage_id"],
            "generation": reference_path["generation"],
            "artifact_sha256": reference_path["artifact_sha256"],
        },
        "profile": "torus9",
        "master_seed": master_seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary) + "\n",
        encoding="utf-8",
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id}) + "\n",
        encoding="utf-8",
    )
    return summary


def test_interrupted_matching_evaluation_is_rerun_and_partial_wld_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, evaluation, arm_results, _identity, _fingerprint, run_id = _arena_case()
    output = tmp_path / run_id
    monkeypatch.setattr(harness, "evaluation_dir", lambda _topology, _run_id: output)

    def interrupted_arena(*, output_dir: Path, **_kwargs: object) -> dict[str, object]:
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "games": 128,
                    "W/L/D": [0, 128, 0],
                    "telemetry": {
                        "technical_games": 0,
                        "performance_status": "HEALTHY",
                        "performance_failures": [],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "partial-only.txt").write_text("interrupted\n", encoding="utf-8")
        raise RuntimeError("simulated host interruption")

    monkeypatch.setattr(harness, "run_arena", interrupted_arena)
    with pytest.raises(RuntimeError, match="simulated host interruption"):
        harness._compare(spec, evaluation, arm_results)

    assert (output / harness.EVALUATION_IDENTITY_FILENAME).is_file()
    assert (output / "summary.json").is_file()
    assert not (output / "provenance.json").exists()
    assert not (output / "manifest.json").exists()

    calls: list[str] = []

    def rerun_arena(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["run_id"]))
        return _write_fake_committed_arena(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(harness, "run_arena", rerun_arena)
    result = harness._compare(spec, evaluation, arm_results)

    assert calls == [run_id]
    assert result["W/L/D"] == [80, 48, 0]
    assert not (output / "partial-only.txt").exists()
    persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert persisted["W/L/D"] == [80, 48, 0]


def test_complete_matching_evaluation_is_reused_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, evaluation, arm_results, _identity, _fingerprint, run_id = _arena_case()
    output = tmp_path / run_id
    monkeypatch.setattr(harness, "evaluation_dir", lambda _topology, _run_id: output)
    calls: list[str] = []

    def first_arena(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["run_id"]))
        return _write_fake_committed_arena(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(harness, "run_arena", first_arena)
    first = harness._compare(spec, evaluation, arm_results)

    def unexpected_arena(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("complete matching evaluation should have been reused")

    monkeypatch.setattr(harness, "run_arena", unexpected_arena)
    second = harness._compare(spec, evaluation, arm_results)

    assert calls == [run_id]
    assert first["W/L/D"] == [80, 48, 0]
    assert second["W/L/D"] == [80, 48, 0]


def test_mismatched_evaluation_identity_fails_closed_and_is_not_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, evaluation, arm_results, identity, _fingerprint, run_id = _arena_case()
    output = tmp_path / run_id
    monkeypatch.setattr(harness, "evaluation_dir", lambda _topology, _run_id: output)
    mismatched = dict(identity)
    mismatched["master_seed"] = int(identity["master_seed"]) + 1
    mismatched_fingerprint = harness._evaluation_fingerprint(mismatched)
    harness._write_evaluation_identity(
        output,
        run_id,
        mismatched,
        mismatched_fingerprint,
    )
    identity_before = (output / harness.EVALUATION_IDENTITY_FILENAME).read_text(
        encoding="utf-8"
    )

    def unexpected_arena(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("mismatched evaluation must fail before Arena")

    monkeypatch.setattr(harness, "run_arena", unexpected_arena)
    with pytest.raises(RuntimeError, match="evaluation identity/contract mismatch"):
        harness._compare(spec, evaluation, arm_results)

    assert output.is_dir()
    assert (
        output / harness.EVALUATION_IDENTITY_FILENAME
    ).read_text(encoding="utf-8") == identity_before
