from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import c4_overnight_experiment as runner
from tools import gocube_experiment_storage as storage


def _project_skeleton(tmp_path: Path) -> None:
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")


def _cli(experiment_id: str):
    return runner.parse_args(["--experiment-id", experiment_id, "--device", "cpu"])


def test_canonical_runner_uses_storage_efficient_experiment():
    assert runner.Experiment is storage.StorageEfficientExperiment
    assert issubclass(runner.Experiment, storage._runner.Experiment)
    assert "7.5" not in Path(storage.__file__).read_text(encoding="utf-8")


def test_clone_history_hardlinks_pkl_copies_metadata_and_skips_future(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "records" / "iteration-0007").mkdir(parents=True)
    (source / "records" / "iteration-0008").mkdir(parents=True)
    history = source / "iteration-0007-data.pkl"
    future = source / "iteration-0008-data.pkl"
    metadata = source / "records" / "iteration-0007" / "manifest.json"
    future_metadata = source / "records" / "iteration-0008" / "manifest.json"
    history.write_bytes(b"historical tensor")
    future.write_bytes(b"future tensor")
    metadata.write_text('{"iteration": 7}\n', encoding="utf-8")
    future_metadata.write_text('{"iteration": 8}\n', encoding="utf-8")

    stats = storage.clone_history_cow(source, destination, max_iteration=7)

    linked = destination / history.name
    copied_metadata = destination / "records" / "iteration-0007" / "manifest.json"
    assert linked.is_file()
    assert os.stat(history).st_ino == os.stat(linked).st_ino
    assert os.stat(metadata).st_ino != os.stat(copied_metadata).st_ino
    assert not (destination / future.name).exists()
    assert not (destination / "records" / "iteration-0008").exists()
    copied_metadata.write_text('{"iteration": 7, "clone": true}\n', encoding="utf-8")
    assert metadata.read_text(encoding="utf-8") == '{"iteration": 7}\n'
    assert stats["hardlinked_files"] == 1
    assert stats["copied_files"] == 1
    assert stats["skipped_future_entries"] >= 1


def test_storage_preflight_accounts_for_parallel_candidate_peak(tmp_path, monkeypatch):
    repo = tmp_path
    checkpoint = repo / "checkpoint" / "parent"
    data = repo / "data" / "parent"
    checkpoint.mkdir(parents=True)
    data.mkdir(parents=True)
    (checkpoint / "iteration-0007.pkl").write_bytes(b"x" * 100)
    (data / "iteration-0007-data.pkl").write_bytes(b"y" * 200)

    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=1_000, free=5_000),
    )
    report = storage.storage_preflight_report(
        repo,
        run_name="parent",
        iteration=7,
        parallel_branches=4,
        new_iterations_per_branch=2,
        reserve_bytes=100,
        safety_factor=2.0,
    )

    assert report["iteration_footprint_bytes"] == 300
    assert report["estimated_new_unique_bytes"] == 2_400
    assert report["working_bytes"] == 4_800
    assert report["required_free_bytes"] == 4_900
    assert report["ok"] is True

    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=6_000, free=4_000),
    )
    report = storage.storage_preflight_report(
        repo,
        run_name="parent",
        iteration=7,
        parallel_branches=4,
        new_iterations_per_branch=2,
        reserve_bytes=100,
        safety_factor=2.0,
    )
    assert report["ok"] is False


def test_ensure_clone_uses_hardlinks_and_only_parent_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    experiment = runner.Experiment(_cli("cow-clone"))
    parent = "parent"
    target = "cow-clone-p1-l-10p0"
    checkpoint = tmp_path / "checkpoint" / parent
    data = tmp_path / "data" / parent
    checkpoint.mkdir(parents=True)
    data.mkdir(parents=True)
    parent_checkpoint = checkpoint / "iteration-0007.pkl"
    parent_tensor = data / "iteration-0007-data.pkl"
    parent_checkpoint.write_bytes(b"checkpoint")
    parent_tensor.write_bytes(b"tensor")
    (checkpoint / "iteration-0008.pkl").write_bytes(b"future")
    (checkpoint / "metadata.json").write_text("{}\n", encoding="utf-8")

    experiment._ensure_clone(
        parent_run=parent,
        parent_iteration=7,
        target_run=target,
        kind="parameter:P1",
    )

    cloned_checkpoint = tmp_path / "checkpoint" / target / "iteration-0007.pkl"
    cloned_tensor = tmp_path / "data" / target / "iteration-0007-data.pkl"
    assert os.stat(parent_checkpoint).st_ino == os.stat(cloned_checkpoint).st_ino
    assert os.stat(parent_tensor).st_ino == os.stat(cloned_tensor).st_ino
    assert not (tmp_path / "checkpoint" / target / "iteration-0008.pkl").exists()
    provenance = json.loads(
        (tmp_path / "checkpoint" / target / runner.PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    assert provenance["schema_version"] == 2
    assert provenance["clone_mode"] == storage.COW_CLONE_MODE
    assert provenance["max_parent_iteration"] == 7
    clone_record = experiment.state["storage"]["clones"][target]
    assert clone_record["hardlinked_bytes"] == len(b"checkpoint") + len(b"tensor")


def test_completed_stage_prunes_losers_but_keeps_champion_bootstrap_and_last_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    experiment = runner.Experiment(_cli("prune-stage"))
    bootstrap = "external-bootstrap"
    previous = "prune-stage-p1-m-19p0"
    winner = "prune-stage-p2-h-0p35"
    loser_l = "prune-stage-p2-l-0p15"
    loser_m = "prune-stage-p2-m-0p25"
    for run_name in (previous, winner, loser_l, loser_m):
        (tmp_path / "checkpoint" / run_name).mkdir(parents=True)
        (tmp_path / "data" / run_name).mkdir(parents=True)
        (tmp_path / "runs" / run_name).mkdir(parents=True)
        (tmp_path / "checkpoint" / run_name / "iteration-0009.pkl").write_bytes(b"x")
        (tmp_path / "data" / run_name / "iteration-0009-data.pkl").write_bytes(b"y")
        (tmp_path / "runs" / run_name / "events").write_bytes(b"z")

    experiment.state["bootstrap"] = {"run": bootstrap, "iteration": 7}
    experiment.state["parameters"] = [
        {
            "id": "P1",
            "parent": {"run": bootstrap, "iteration": 7},
            "winner": {"promoted": True},
            "candidates": [
                {"run": previous, "label": "M"},
                {"run": "prune-stage-p1-l-10p0", "label": "L"},
            ],
            "champion_after": {"run": previous, "iteration": 9, "sweep_overrides": {}},
        },
        {
            "id": "P2",
            "parent": {"run": previous, "iteration": 9},
            "winner": {"promoted": True},
            "candidates": [
                {"run": loser_l, "label": "L"},
                {"run": loser_m, "label": "M"},
                {"run": winner, "label": "H"},
            ],
            "champion_after": {"run": winner, "iteration": 11, "sweep_overrides": {}},
        },
    ]

    experiment._prune_completed_candidate_namespaces(bootstrap)

    assert (tmp_path / "checkpoint" / previous).exists()
    assert (tmp_path / "checkpoint" / winner).exists()
    assert not (tmp_path / "checkpoint" / loser_l).exists()
    assert not (tmp_path / "data" / loser_l).exists()
    assert not (tmp_path / "runs" / loser_l).exists()
    assert not (tmp_path / "checkpoint" / loser_m).exists()
    assert experiment.state["parameters"][1]["candidates"][0]["artifact_status"] == "pruned-after-evaluation"
    assert experiment.state["parameters"][1]["candidates"][2]["artifact_status"] == "retained"
    assert loser_l in experiment.state["storage"]["pruned_namespaces"]


def test_non_experiment_namespace_can_never_be_pruned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project_skeleton(tmp_path)
    experiment = runner.Experiment(_cli("safe-prune"))
    (tmp_path / "checkpoint" / "external-bootstrap").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="non-experiment namespace"):
        experiment._remove_owned_namespace("external-bootstrap", "must not delete")

    assert (tmp_path / "checkpoint" / "external-bootstrap").exists()
