from __future__ import annotations

from types import SimpleNamespace

import gocube_golden.orchestrator_v2.production_entrypoint as entrypoint


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
