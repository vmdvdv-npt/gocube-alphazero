from __future__ import annotations

import json
from types import SimpleNamespace

import gocube_golden.orchestrator_v2.production_entrypoint as entrypoint
from tools.arena import _publish_evaluation_metadata


def test_standalone_arena_request_accepts_validated_execution_commit_override(
    monkeypatch,
) -> None:
    candidate = SimpleNamespace(
        topology="torus9",
        owner_root="runs/torus9/active/new_komi",
        effective_config=SimpleNamespace(config=object()),
    )
    candidate.ref = object()
    reference = SimpleNamespace(
        topology="torus9",
        owner_root="runs/torus9/active/new_komi",
        effective_config=SimpleNamespace(config=object()),
        ref=object(),
    )

    class Binding:
        def default_arena_profile(self, _config):
            return "torus9|komi=0.5|simulations=64|cpuct=1.25|fpu=0|watchdog=500|5ch"

        def validate_arena_profile(self, _profile, _config):
            return None

        def arena_startset(self, *, master_seed, games):
            return SimpleNamespace(id="starts", fingerprint="sha256:" + "a" * 64)

    class Resolver:
        def checkpoint(self, value):
            return candidate if value == "candidate" else reference

    monkeypatch.setattr(entrypoint, "get_topology_binding", lambda _topology: Binding())
    monkeypatch.setattr(entrypoint, "_torus_profile_with_search", lambda profile, _search: profile)
    monkeypatch.setattr(
        entrypoint,
        "ArenaRunRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        entrypoint,
        "resolve_execution_commit",
        lambda repo_root, value: (
            assert_value(repo_root, value)
        ),
    )
    monkeypatch.setattr(
        entrypoint,
        "execution_commit_from_lineage",
        lambda _root: "lineage-commit",
    )

    request = entrypoint._standalone_arena_request(
        {
            "candidate_checkpoint": "candidate",
            "reference_checkpoint": "reference",
            "execution_code_commit": "current-commit",
            "arena_config": {
                "games": 4,
                "workers": 1,
                "games_per_worker": 2,
                "inference_batch_rows": 2,
            },
        },
        Resolver(),
    )

    assert request.execution_code_commit == "resolved-current-commit"


def assert_value(repo_root, value):
    assert repo_root == entrypoint._repo_root()
    assert value == "current-commit"
    return "resolved-current-commit"


def test_arena_provenance_persists_execution_code_commit(tmp_path) -> None:
    checkpoint = {
        "topology": "torus9",
        "lineage_id": "new_komi",
        "checkpoint_id": "M137-5CH-bootstrap",
        "generation": 137,
        "sha256": "sha256:" + "a" * 64,
    }
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "manifest.json").write_text(
        '{"run_id":"evaluation"}\n', encoding="utf-8"
    )

    _publish_evaluation_metadata(
        output_dir=output,
        summary={
            "telemetry": {
                "technical_games": 0,
                "performance_status": "HEALTHY",
                "performance_failures": [],
            }
        },
        profile_id="torus9",
        master_seed=2026092601,
        run_id="evaluation",
        candidate_ref=checkpoint,
        reference_ref=checkpoint,
        evaluation_identity={
            "schema": "gocube-arena-evaluation-identity-v2",
            "candidate": checkpoint,
            "reference": checkpoint,
            "master_seed": 2026092601,
            "scientific_contract": {},
            "execution_contract": {},
            "workload": {},
        },
        evaluation_fingerprint="fingerprint",
        execution_code_commit="d8306b8f4897f5e67681e9cc65df58640074c090",
    )

    assert json.loads((output / "provenance.json").read_text())["execution_code_commit"] == (
        "d8306b8f4897f5e67681e9cc65df58640074c090"
    )
