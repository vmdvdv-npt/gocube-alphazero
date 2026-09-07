from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import c4_overnight_experiment as runner
from tools import gocube_production_preflight as preflight


def _valid_report(*, head: str = "a" * 40, branch: str = "main", selected_device: str = "cuda"):
    report = {
        "schema_version": preflight.PREFLIGHT_SCHEMA_VERSION,
        "checked_at_epoch": 1.0,
        "source": {
            "repo_path": str(preflight.EXPECTED_REPO),
            "git_root": str(preflight.EXPECTED_REPO),
            "branch": branch,
            "head_sha": head,
            "origin_url": "git@github.com:vmdvdv-npt/gocube-alphazero.git",
            "tracked_tree_clean": True,
            "tracked_status": "",
            "upstream": "origin/main" if branch == "main" else None,
            "upstream_sha": head if branch == "main" else None,
            "remote_main_sha": head if branch == "main" else None,
            "remote_verified": branch == "main",
        },
        "runtime": {
            "user": "codex",
            "python_executable": str(preflight.EXPECTED_REPO / ".venv" / "bin" / "python"),
            "python_prefix": str(preflight.EXPECTED_REPO / ".venv"),
            "python_version": "3.13.7",
            "python_implementation": "CPython",
            "torch_version": "2.8.0+cu129",
            "torch_cuda_version": "12.9",
            "cudnn_version": 91002,
            "pip_freeze": ["numpy==2.0.0", "torch==2.8.0"],
            "pip_freeze_sha256": "freeze-hash",
            "platform_system": "Linux",
            "platform_release": "6.14.0",
            "platform_machine": "x86_64",
            "os_id": "ubuntu",
            "os_version_id": "26.04",
        },
        "hardware": {
            "hostname": "Legion",
            "logical_cpu_count": 16,
            "affinity_cpu_count": 16,
            "physical_memory_bytes": 64 * 1024**3,
            "requested_device": "auto",
            "selected_device": selected_device,
            "cuda_available": selected_device == "cuda",
            "cuda_device_count": 1 if selected_device == "cuda" else 0,
            "gpu": {
                "index": 0,
                "name": "NVIDIA GeForce RTX 3060",
                "total_memory_bytes": 6 * 1024**3,
                "compute_capability": [8, 6],
                "driver_version": "581.57",
            } if selected_device == "cuda" else None,
        },
        "disk": {
            "filesystem_total_bytes": 500 * 1024**3,
            "filesystem_used_bytes": 100 * 1024**3,
            "filesystem_free_bytes": 400 * 1024**3,
            "minimum_free_reserve_bytes": 5 * 1024**3,
        },
        "supervision": {
            "supervised": True,
            "invocation_id": "invocation",
            "unit": "gocube-sweep-test",
            "linger_enabled": True,
            "linger_error": None,
        },
    }
    return preflight.attach_environment_fingerprint(report)


def test_new_preflight_requires_clean_current_remote_main():
    report = _valid_report()
    preflight.validate_new_production_preflight(report)

    dirty = copy.deepcopy(report)
    dirty["source"]["tracked_tree_clean"] = False
    with pytest.raises(RuntimeError, match="dirty"):
        preflight.validate_new_production_preflight(dirty)

    stale = copy.deepcopy(report)
    stale["source"]["remote_main_sha"] = "b" * 40
    with pytest.raises(RuntimeError, match="current GitHub origin/main"):
        preflight.validate_new_production_preflight(stale)


def test_new_preflight_requires_legion_cuda_profile():
    wrong_gpu = _valid_report()
    wrong_gpu["hardware"]["gpu"]["name"] = "NVIDIA GeForce RTX 4090"
    with pytest.raises(RuntimeError, match="Unexpected production GPU"):
        preflight.validate_new_production_preflight(wrong_gpu)

    cpu = _valid_report(selected_device="cpu")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        preflight.validate_new_production_preflight(cpu)

    restricted = _valid_report()
    restricted["hardware"]["affinity_cpu_count"] = 8
    with pytest.raises(RuntimeError, match="CPU affinity"):
        preflight.validate_new_production_preflight(restricted)


def test_resume_requires_exact_source_and_environment_fingerprint():
    baseline = _valid_report()
    current = copy.deepcopy(baseline)
    current["source"]["branch"] = "DETACHED"
    current["source"]["upstream"] = None
    current["source"]["upstream_sha"] = None
    current["source"]["remote_main_sha"] = None
    current["source"]["remote_verified"] = False
    preflight.validate_resume_production_preflight(baseline, current)

    changed_commit = copy.deepcopy(current)
    changed_commit["source"]["head_sha"] = "c" * 40
    preflight.attach_environment_fingerprint(changed_commit)
    with pytest.raises(RuntimeError, match="source commit changed"):
        preflight.validate_resume_production_preflight(baseline, changed_commit)

    changed_driver = copy.deepcopy(current)
    changed_driver["hardware"]["gpu"]["driver_version"] = "999.0"
    preflight.attach_environment_fingerprint(changed_driver)
    with pytest.raises(RuntimeError, match="environment changed"):
        preflight.validate_resume_production_preflight(baseline, changed_driver)


def test_systemd_command_is_logout_safe_and_restarts_only_abnormal_crashes(tmp_path):
    repo = preflight.EXPECTED_REPO
    cli = SimpleNamespace(
        experiment_id="c4-sweep-test",
        bootstrap_run="c7-run",
        device="auto",
        arena_batch_wait_ms=1.0,
        telemetry_interval=2.0,
        heldout_positions=16,
        health_gate_min_win_rate=0.45,
        benchmark_games=64,
        skip_performance_benchmark=False,
        seed=20260906,
    )
    unit, command = preflight.build_systemd_run_command(repo, cli)
    joined = " ".join(command)

    assert unit == "gocube-sweep-c4-sweep-test"
    assert "--user" in command
    assert "--collect" in command
    assert "--property=Restart=on-abnormal" in command
    assert "Restart=always" not in joined
    assert f"--setenv={preflight.SUPERVISED_ENV}=1" in command
    assert "--experiment-id c4-sweep-test" in joined
    assert "--bootstrap-run c7-run" in joined
    assert str(repo / ".venv" / "bin" / "python") in command
    assert str(repo / "tools" / "c4_overnight_experiment.py") in command


def test_apply_preflight_persists_baseline_and_session_checks(monkeypatch):
    report = _valid_report()

    class DummyExperiment:
        def __init__(self):
            self.state = {
                "status": "RUNNING",
                "resume_count": 0,
                "parameters": [],
                "completed_actions": {},
            }
            self.repo = preflight.EXPECTED_REPO
            self.cli = SimpleNamespace(device="auto")
            self.saved = 0

        def _save_state(self):
            self.saved += 1

    experiment = DummyExperiment()
    monkeypatch.setattr(
        preflight,
        "collect_production_preflight",
        lambda repo, cli, verify_remote: copy.deepcopy(report),
    )
    first = preflight.apply_production_preflight(experiment)
    assert first["source"]["head_sha"] == "a" * 40
    assert experiment.state["source_commit"] == "a" * 40
    assert experiment.state["environment_fingerprint_sha256"] == report["environment_fingerprint_sha256"]
    assert len(experiment.state["production_preflight"]["checks"]) == 1

    second = preflight.apply_production_preflight(experiment)
    assert second["environment_fingerprint_sha256"] == report["environment_fingerprint_sha256"]
    assert len(experiment.state["production_preflight"]["checks"]) == 2
    assert experiment.saved == 2


def test_legacy_in_progress_state_without_preflight_fails_closed(monkeypatch):
    class DummyExperiment:
        state = {
            "status": "RUNNING",
            "resume_count": 1,
            "bootstrap": {"run": "old", "iteration": 7},
            "parameters": [],
            "completed_actions": {},
        }
        repo = preflight.EXPECTED_REPO
        cli = SimpleNamespace(device="auto")

        def _save_state(self):
            pass

    with pytest.raises(RuntimeError, match="no production preflight provenance"):
        preflight.apply_production_preflight(DummyExperiment())


def test_canonical_main_delegates_unsupervised_launch(monkeypatch):
    cli = runner.parse_args(["--experiment-id", "delegated"])
    monkeypatch.setattr(runner, "parse_args", lambda argv=None: cli)
    monkeypatch.setattr(runner._preflight, "is_supervised_invocation", lambda: False)
    captured = {}

    def fake_launch(repo, launch_cli):
        captured["repo"] = repo
        captured["id"] = launch_cli.experiment_id
        return 0

    monkeypatch.setattr(runner._preflight, "launch_under_systemd", fake_launch)
    assert runner.main([]) == 0
    assert captured["id"] == "delegated"
