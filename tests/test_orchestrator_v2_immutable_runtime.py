from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig
from gocube_golden.orchestrator_v2.immutable_runtime import (
    ImmutableRuntimeManager,
    RuntimeIntegrityError,
    execution_commit_from_lineage,
)
from gocube_golden.orchestrator_v2.production_generation import ProductionTrainOne
from gocube_golden.orchestrator_v2.torus9_production import Torus9ProductionLineage


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _temporary_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "runtime tests")
    (repo / "runtime_fixture.py").write_text("VALUE = 'A'\n", encoding="utf-8")
    _git(repo, "add", "runtime_fixture.py")
    _git(repo, "commit", "-m", "A")
    commit_a = _git(repo, "rev-parse", "HEAD")
    (repo / "runtime_fixture.py").write_text("VALUE = 'B'\n", encoding="utf-8")
    _git(repo, "commit", "-am", "B")
    commit_b = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", commit_a)
    return repo, commit_a, commit_b


def test_live_child_survives_developer_checkout_change(tmp_path: Path) -> None:
    repo, commit_a, commit_b = _temporary_repo(tmp_path)
    manager = ImmutableRuntimeManager(repo, runtime_root=tmp_path / "runtime")
    runtime = manager.ensure(commit_a)
    script = (
        "import json, os, time, runtime_fixture; "
        "first=runtime_fixture.VALUE; time.sleep(0.5); "
        "second=runtime_fixture.VALUE; "
        "print(json.dumps({'first': first, 'second': second, 'commit': os.environ['AZ_ORCHESTRATOR_RUNTIME_COMMIT']}), flush=True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=runtime.path,
        env=runtime.environment(),
        stdout=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.1)
    _git(repo, "checkout", "--detach", commit_b)
    (repo / "runtime_fixture.py").write_text("VALUE = 'DIRTY-DEVELOPER'\n", encoding="utf-8")
    output, _ = process.communicate(timeout=5)
    assert process.returncode == 0
    assert json.loads(output) == {
        "first": "A",
        "second": "A",
        "commit": commit_a,
    }


def test_resume_uses_last_execution_commit_after_repository_move(tmp_path: Path) -> None:
    repo, commit_a, commit_b = _temporary_repo(tmp_path)
    lineage = tmp_path / "runs" / "torus9" / "active" / "lineage"
    lineage.mkdir(parents=True)
    (lineage / "manifest.json").write_text(
        json.dumps(
            {
                "git_commit": commit_a,
                "lineage_initial_git_commit": commit_a,
                "execution_code_commit": commit_a,
            }
        ),
        encoding="utf-8",
    )
    _git(repo, "checkout", "--detach", commit_b)
    (repo / "runtime_fixture.py").write_text("VALUE = 'DIRTY-DEVELOPER'\n", encoding="utf-8")
    manager = ImmutableRuntimeManager(repo, runtime_root=tmp_path / "runtime")
    runtime = manager.ensure(execution_commit_from_lineage(lineage))
    assert runtime.commit == commit_a
    assert (runtime.path / "runtime_fixture.py").read_text(encoding="utf-8") == "VALUE = 'A'\n"


def _lineage_inputs(tmp_path: Path) -> tuple[SimpleNamespace, EffectiveConfig]:
    parent = SimpleNamespace(
        ref=CheckpointRef(
            "torus9",
            "parent",
            "M1",
            1,
            "checkpoints/M1.pt",
            "sha256:" + "a" * 64,
        )
    )
    config = EffectiveConfig("torus9", {"topology": "torus9"})
    return parent, config


def test_explicit_rollover_happens_at_lineage_boundary(tmp_path: Path) -> None:
    repo, commit_a, commit_b = _temporary_repo(tmp_path)
    parent, config = _lineage_inputs(tmp_path)
    factory = Torus9ProductionLineage(tmp_path / "runs", repo_root=repo)
    factory.prepare(
        topology="torus9",
        lineage_id="lineage",
        parent=parent,
        effective_config=config,
        experiment_id="experiment",
        arm_id="continuous",
    )
    _git(repo, "checkout", "--detach", commit_b)
    factory.prepare(
        topology="torus9",
        lineage_id="lineage",
        parent=parent,
        effective_config=config,
        experiment_id="experiment",
        arm_id="continuous",
        allow_code_rollover=True,
    )
    manifest = json.loads(
        (
            tmp_path
            / "runs"
            / "torus9"
            / "active"
            / "lineage"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["lineage_initial_git_commit"] == commit_a
    assert manifest["execution_code_commit"] == commit_b
    assert manifest["execution_rollover"] == "yes"
    assert manifest["last_code_rollover"]["from"] == commit_a
    assert manifest["last_code_rollover"]["to"] == commit_b
    assert manifest["code_revision_history"][-1]["reason"].startswith(
        "explicit code rollover"
    )


def test_common_production_lifecycle_uses_runtime_for_generation_and_arena(
    tmp_path: Path,
) -> None:
    from gocube_golden.orchestrator_v2._arena_runner_core import ArenaRunner

    generation = ProductionTrainOne(resolver=SimpleNamespace(runs_root=tmp_path / "runs"), repo_root=tmp_path)
    arena = ArenaRunner(engine=lambda **_kwargs: {})
    assert isinstance(generation.runtime_manager, ImmutableRuntimeManager)
    assert hasattr(arena, "_production_summary")
    assert arena._runtime_manager.__class__ is generation.runtime_manager.__class__


def test_unresolvable_execution_commit_fails_before_new_child(tmp_path: Path) -> None:
    repo, _commit_a, _commit_b = _temporary_repo(tmp_path)
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "manifest.json").write_text(
        json.dumps({"execution_code_commit": "f" * 40}),
        encoding="utf-8",
    )
    parent, config = _lineage_inputs(tmp_path)
    resolved_config = SimpleNamespace(config=config, ref=SimpleNamespace())
    output = SimpleNamespace(root=lineage, topology="torus9", lineage_id="lineage")
    resolver = SimpleNamespace(runs_root=tmp_path / "runs")
    with pytest.raises(RuntimeIntegrityError, match="cannot be resolved"):
        ProductionTrainOne(resolver=resolver, repo_root=repo)(
            parent=SimpleNamespace(ref=parent.ref, generation=1),
            config=resolved_config,
            output_lineage=output,
        )
    assert not (lineage / "runtime" / "requests").exists()
    assert not list(lineage.glob("generation-*.complete.json"))
