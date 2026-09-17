from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import gocube_golden.code_update_policy as policy_module
import gocube_golden.orchestrator as legacy_orchestrator
from gocube_golden.code_update_policy import CodeUpdateProvenancePolicy
from gocube_golden.orchestrator import HealthPolicy, OrchestratorSpec, SoftStopPolicy
from gocube_golden.production_orchestrator import (
    ChildLifecycleContext,
    NoOpChildLifecyclePolicy,
    SupervisionPolicy,
    UniversalProductionTrainingOrchestrator,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _spec(tmp_path: Path) -> OrchestratorSpec:
    return OrchestratorSpec(
        path=tmp_path / "orchestrator.json",
        payload={"generation": {"driver_config": {"games": 64}}},
        topology="torus9",
        profile_path=tmp_path / "profile.json",
        profile_payload={
            "training": {"learning_rate": 0.0003},
            "replay": {"generations": 6, "cap": 40000},
            "self_play": {"mcts_simulations": 128},
        },
        profile_fingerprint="sha256:profile",
        config_fingerprint="sha256:config",
        generation_command=("fake-child",),
        generation_resume_command=("fake-child", "--resume"),
        arena_command=("fake-arena",),
        arena_every_generations=10,
        arena_required=False,
        arena_preset_fingerprint=None,
        arena_startset_fingerprint=None,
        health=HealthPolicy(),
        soft_stop=SoftStopPolicy(),
        performance={},
        learning={},
    )


def _supervision() -> SupervisionPolicy:
    return SupervisionPolicy(
        startup_ack_timeout_seconds=1.0,
        progress_warning_seconds=2.0,
        progress_critical_seconds=3.0,
        critical_child_grace_seconds=0.0,
        restart_backoff_seconds=0.0,
        max_generation_restarts=1,
    )


def _orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lineage_id: str,
    policy: object = ...,
) -> UniversalProductionTrainingOrchestrator:
    monkeypatch.setattr(
        legacy_orchestrator,
        "active_lineage_dir",
        lambda topology, lineage: tmp_path / "runs" / topology / "active" / lineage,
    )
    kwargs: dict[str, object] = {
        "repo_root": tmp_path,
        "spec": _spec(tmp_path),
        "lineage_id": lineage_id,
        "terminal": False,
        "supervision": _supervision(),
    }
    if policy is not ...:
        kwargs["child_lifecycle_policy"] = policy
    return UniversalProductionTrainingOrchestrator(**kwargs)


class FakeRun:
    def __init__(self, root: Path, *, lineage_id: str = "lineage-a") -> None:
        self.repo_root = root
        self.lineage_id = lineage_id
        self.paths = SimpleNamespace(
            root=root / "runs" / "torus9" / "active" / lineage_id,
        )
        self.paths.manifest = self.paths.root / "manifest.json"
        self.spec = SimpleNamespace(
            topology="torus9",
            payload={"generation": {"driver_config": {"games": 64}}},
            profile_payload={"training": {"learning_rate": 0.0003}},
            profile_fingerprint="sha256:profile",
            config_fingerprint="sha256:config",
        )
        self.strict_run_spec = SimpleNamespace(
            payload=self.spec.payload,
            fingerprint="sha256:run-spec",
        )

    def _generation_tx_path(self, generation: int) -> Path:
        return self.paths.root / "runtime" / "generations" / f"generation-{generation:04d}.json"


def _prepare_fake_run(root: Path, *, commit: str = "old-commit") -> FakeRun:
    run = FakeRun(root)
    (root / ".git").mkdir(parents=True, exist_ok=True)
    run.paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    run.paths.manifest.write_text(
        json.dumps(
            {
                "lineage_id": run.lineage_id,
                "git_commit": commit,
                "config_fingerprint": "sha256:config",
            }
        ),
        encoding="utf-8",
    )
    return run


def _identity(commit: str, tree: str, *, clean: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        git_commit_sha=commit,
        git_tree_sha=tree,
        working_tree_clean=clean,
    )


def _context(
    run: FakeRun,
    *,
    generation: int = 18,
    phase: str = "generation",
    resume: bool = False,
) -> ChildLifecycleContext:
    return ChildLifecycleContext(
        orchestrator=run,  # type: ignore[arg-type]
        generation=generation,
        phase=phase,
        resume=resume,
        command=("fake-child",),
    )


def test_import_gocube_golden_does_not_install_code_update_monkeypatch() -> None:
    code = r'''
import inspect, sys
import gocube_golden
from gocube_golden.production_orchestrator import UniversalProductionTrainingOrchestrator as U
assert "gocube_golden.code_update_policy" not in sys.modules
assert U._run_child.__module__ == "gocube_golden.production_orchestrator"
assert U._run_child.__name__ == "_run_child"
source = inspect.getsource(U._run_child)
assert "child_lifecycle_policy.before_child_start" in source
assert "child_lifecycle_policy.after_child_finish" in source
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    source = (ROOT / "gocube_golden" / "code_update_policy.py").read_text(encoding="utf-8")
    forbidden = "UniversalProductionTrainingOrchestrator" + "._run_child ="
    assert forbidden not in source
    assert "install_code_update_policy" not in source


def test_base_orchestrator_without_policy_has_no_rollover_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _orchestrator(tmp_path, monkeypatch, lineage_id="base")
    assert isinstance(run.child_lifecycle_policy, NoOpChildLifecyclePolicy)


def test_explicit_policy_is_instance_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = CodeUpdateProvenancePolicy()
    run_a = _orchestrator(tmp_path, monkeypatch, lineage_id="a", policy=policy)
    run_b = _orchestrator(tmp_path, monkeypatch, lineage_id="b")
    assert run_a.child_lifecycle_policy is policy
    assert isinstance(run_b.child_lifecycle_policy, NoOpChildLifecyclePolicy)
    assert run_b.child_lifecycle_policy is not policy

    cli_source = (ROOT / "tools" / "training_orchestrator_core.py").read_text(encoding="utf-8")
    harness_source = (ROOT / "tools" / "torus9_staged_sims_harness.py").read_text(encoding="utf-8")
    assert "child_lifecycle_policy=CodeUpdateProvenancePolicy()" in cli_source
    assert "child_lifecycle_policy=CodeUpdateProvenancePolicy()" in harness_source


def test_dirty_tree_is_fail_closed_only_when_explicit_policy_is_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _prepare_fake_run(tmp_path)
    monkeypatch.setattr(
        policy_module,
        "capture_code_identity",
        lambda _root: _identity("dirty-commit", "dirty-tree", clean=False),
    )
    with pytest.raises(ValueError, match="dirty working tree"):
        CodeUpdateProvenancePolicy().before_child_start(_context(run))
    assert _read(run.paths.manifest)["git_commit"] == "old-commit"


def test_clean_rollover_preserves_lineage_and_initial_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _prepare_fake_run(tmp_path)
    monkeypatch.setattr(
        policy_module,
        "capture_code_identity",
        lambda _root: _identity("new-commit", "new-tree"),
    )
    policy = CodeUpdateProvenancePolicy()
    policy.before_child_start(_context(run, generation=18, phase="arena"))

    manifest = _read(run.paths.manifest)
    assert manifest["lineage_id"] == "lineage-a"
    assert manifest["lineage_initial_git_commit"] == "old-commit"
    assert manifest["git_commit"] == "new-commit"
    assert manifest["current_git_tree"] == "new-tree"
    history = manifest["code_revision_history"]
    assert history[-1]["git_commit_sha"] == "new-commit"
    assert history[-1]["git_tree_sha"] == "new-tree"
    assert history[-1]["first_seen_generation"] == 18
    assert history[-1]["phase"] == "arena"


def test_generation_provenance_preserves_identity_fingerprints_parameters_and_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _prepare_fake_run(tmp_path)
    identities = iter(
        [
            _identity("commit-a", "tree-a"),
            _identity("commit-b", "tree-b"),
        ]
    )
    monkeypatch.setattr(policy_module, "capture_code_identity", lambda _root: next(identities))
    policy = CodeUpdateProvenancePolicy()

    first = _context(run, generation=18, resume=False)
    policy.before_child_start(first)
    policy.after_child_finish(first, 9)

    tx_path = run._generation_tx_path(18)
    tx_path.parent.mkdir(parents=True, exist_ok=True)
    tx_path.write_text(json.dumps({"restart_attempts": 1}), encoding="utf-8")

    second = _context(run, generation=18, resume=True)
    policy.before_child_start(second)
    policy.after_child_finish(second, 0)

    path = run.paths.root / "provenance" / "generations" / "generation-0018.json"
    provenance = _read(path)
    assert provenance["schema"] == "gocube-training-generation-provenance-v1"
    assert provenance["lineage_id"] == "lineage-a"
    assert provenance["generation"] == 18
    assert provenance["status"] == "CHILD_COMPLETED"
    attempts = provenance["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["code"] == {
        "git_commit_sha": "commit-a",
        "git_tree_sha": "tree-a",
        "working_tree_clean": True,
    }
    assert attempts[0]["run_spec_fingerprint"] == "sha256:run-spec"
    assert attempts[0]["config_fingerprint"] == "sha256:config"
    assert attempts[0]["profile_fingerprint"] == "sha256:profile"
    assert attempts[0]["effective_parameters"]["profile"] == run.spec.profile_payload
    assert attempts[0]["effective_parameters"]["generation"] == run.spec.payload["generation"]
    assert attempts[0]["orchestrator_restart_attempts"] == 0
    assert attempts[0]["child_exit_code"] == 9
    assert attempts[1]["code"]["git_commit_sha"] == "commit-b"
    assert attempts[1]["orchestrator_restart_attempts"] == 1
    assert attempts[1]["child_exit_code"] == 0


def test_existing_pr136_manifest_and_provenance_shape_remains_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _prepare_fake_run(tmp_path, commit="commit-a")
    run.paths.manifest.write_text(
        json.dumps(
            {
                "lineage_id": run.lineage_id,
                "git_commit": "commit-a",
                "lineage_initial_git_commit": "initial-commit",
                "current_git_tree": "tree-a",
                "code_revision_history": [
                    {
                        "git_commit_sha": "commit-a",
                        "git_tree_sha": "tree-a",
                        "first_seen_generation": 17,
                        "phase": "generation",
                        "recorded_at": "2026-09-17T00:00:00+00:00",
                        "reason": "application-code update within same training lineage",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provenance_path = run.paths.root / "provenance" / "generations" / "generation-0018.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    previous_attempt = {
        "attempt_index": 0,
        "orchestrator_restart_attempts": 0,
        "recorded_at": "2026-09-17T00:00:00+00:00",
        "code": {
            "git_commit_sha": "commit-a",
            "git_tree_sha": "tree-a",
            "working_tree_clean": True,
        },
        "run_spec_fingerprint": "sha256:run-spec",
        "config_fingerprint": "sha256:config",
        "profile_fingerprint": "sha256:profile",
        "effective_parameters": {
            "profile": run.spec.profile_payload,
            "generation": run.spec.payload["generation"],
        },
    }
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "gocube-training-generation-provenance-v1",
                "lineage_id": run.lineage_id,
                "topology": "torus9",
                "generation": 18,
                "status": "CHILD_RUNNING",
                "latest_attempt": previous_attempt,
                "attempts": [previous_attempt],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        policy_module,
        "capture_code_identity",
        lambda _root: _identity("commit-b", "tree-b"),
    )
    policy = CodeUpdateProvenancePolicy()
    context = _context(run, generation=18, resume=True)
    policy.before_child_start(context)
    policy.after_child_finish(context, 0)

    manifest = _read(run.paths.manifest)
    assert manifest["lineage_initial_git_commit"] == "initial-commit"
    assert manifest["git_commit"] == "commit-b"
    assert len(manifest["code_revision_history"]) == 2
    provenance = _read(provenance_path)
    assert provenance["schema"] == "gocube-training-generation-provenance-v1"
    assert provenance["attempts"][0]["code"]["git_commit_sha"] == "commit-a"
    assert provenance["attempts"][1]["code"]["git_commit_sha"] == "commit-b"
