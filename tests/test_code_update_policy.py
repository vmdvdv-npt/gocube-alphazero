from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gocube_golden.code_update_policy as policy


class _Paths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = root / "manifest.json"


class _Run:
    def __init__(self, root: Path) -> None:
        self.repo_root = root
        self.paths = _Paths(root)
        self.lineage_id = "torus9-existing-lineage"
        self.spec = SimpleNamespace(
            topology="torus9",
            profile_fingerprint="sha256:profile",
            config_fingerprint="sha256:config",
            profile_payload={
                "training": {"learning_rate": 0.0003},
                "replay": {"generations": 6, "cap": 40000},
                "self_play": {"mcts_simulations": 128},
                "network": {"hidden": 80, "blocks": 8},
            },
            payload={},
        )
        self.strict_run_spec = SimpleNamespace(
            fingerprint="sha256:run-spec",
            payload={
                "generation": {
                    "driver_config": {
                        "games": 64,
                        "workers": 16,
                        "active_games_per_worker": 4,
                        "total_active_contexts": 64,
                        "inference_batch_cap": 64,
                        "inference_batch_wait_ms": 1,
                    }
                }
            },
        )

    def _generation_tx_path(self, generation: int) -> Path:
        return self.paths.root / "runtime" / "generations" / f"generation-{generation:04d}.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_application_update_advances_runtime_pin_without_new_lineage(tmp_path: Path) -> None:
    run = _Run(tmp_path)
    run.paths.manifest.write_text(
        json.dumps(
            {
                "lineage_id": run.lineage_id,
                "git_commit": "old-commit",
                "config_fingerprint": "sha256:config",
            }
        ),
        encoding="utf-8",
    )
    code = {
        "git_commit_sha": "new-commit",
        "git_tree_sha": "new-tree",
        "working_tree_clean": True,
    }

    policy._advance_manifest_code_pin(
        run,
        code=code,
        generation=60,
        phase="generation",
    )

    manifest = _read(run.paths.manifest)
    assert manifest["lineage_id"] == run.lineage_id
    assert manifest["lineage_initial_git_commit"] == "old-commit"
    assert manifest["git_commit"] == "new-commit"
    assert manifest["current_git_tree"] == "new-tree"
    assert manifest["code_revision_history"][-1]["first_seen_generation"] == 60


def test_generation_provenance_records_exact_code_and_effective_parameters(
    tmp_path: Path,
) -> None:
    run = _Run(tmp_path)
    tx = run._generation_tx_path(60)
    tx.parent.mkdir(parents=True)
    tx.write_text(
        json.dumps({"status": "RUNNING", "restart_attempts": 1}),
        encoding="utf-8",
    )
    code = {
        "git_commit_sha": "new-commit",
        "git_tree_sha": "new-tree",
        "working_tree_clean": True,
    }

    path = policy._record_generation_attempt(run, generation=60, code=code)
    payload = _read(path)
    latest = payload["latest_attempt"]

    assert payload["lineage_id"] == run.lineage_id
    assert payload["generation"] == 60
    assert latest["code"]["git_commit_sha"] == "new-commit"
    assert latest["run_spec_fingerprint"] == "sha256:run-spec"
    assert (
        latest["effective_parameters"]["profile"]["training"]["learning_rate"]
        == 0.0003
    )
    assert (
        latest["effective_parameters"]["profile"]["self_play"]["mcts_simulations"]
        == 128
    )
    assert (
        latest["effective_parameters"]["generation"]["driver_config"]["games"]
        == 64
    )
    assert latest["orchestrator_restart_attempts"] == 1


def test_generation_provenance_keeps_attempt_history_across_code_updates(
    tmp_path: Path,
) -> None:
    run = _Run(tmp_path)
    first = {
        "git_commit_sha": "commit-a",
        "git_tree_sha": "tree-a",
        "working_tree_clean": True,
    }
    second = {
        "git_commit_sha": "commit-b",
        "git_tree_sha": "tree-b",
        "working_tree_clean": True,
    }

    path = policy._record_generation_attempt(run, generation=60, code=first)
    policy._record_generation_attempt(run, generation=60, code=second)

    payload = _read(path)
    commits = [attempt["code"]["git_commit_sha"] for attempt in payload["attempts"]]
    assert commits == ["commit-a", "commit-b"]
    assert payload["latest_attempt"]["code"]["git_commit_sha"] == "commit-b"
