from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gocube_golden.orchestrator_v2.execution_permit import _test_authority


# These pre-boundary regression tests intentionally exercise production internals
# directly. Give only those exact tests the private unit-test authority; boundary
# tests remain unauthorized and continue proving that env markers/direct calls
# fail closed.
_AUTHORIZED_INTERNAL_TESTS: dict[tuple[str, str], tuple[str, str]] = {
    (
        "test_orchestrator_v2_cube_stage8.py",
        "test_cube4_real_orchestrated_m0_to_m2_smoke",
    ): ("cube4", "cube4-stage8-smoke"),
    (
        "test_orchestrator_v2_cube_stage8_commit_recovery.py",
        "test_cube_commit_boundary_crash_is_reconciled_and_same_generation_retries",
    ): ("cube4", "cube4-stage8-commit-recovery"),
    (
        "test_orchestrator_v2_generation_runner.py",
        "test_torus9_production_path_forwards_exact_input_and_maps_committed_result",
    ): ("torus9", "child-lineage"),
    (
        "test_orchestrator_v2_immutable_runtime.py",
        "test_unresolvable_execution_commit_fails_before_new_child",
    ): ("torus9", "lineage"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_train_one_runs_one_generation_and_returns_immediate_child",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_train_one_acknowledges_stopped_execution_before_baseline_start",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_train_one_reuses_committed_child_without_supervisor",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_train_one_reconciles_committed_execution_for_next_generation",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_committed_reuse_does_not_delete_foreign_or_malformed_supervisor_intent",
    ): ("torus9", "child"),
    (
        "test_orchestrator_v2_production_generation.py",
        "test_committed_reuse_does_not_delete_foreign_or_malformed_supervisor_stop",
    ): ("torus9", "child"),
}


@pytest.fixture(autouse=True)
def _legacy_orchestrator_v2_internal_authority(request: pytest.FixtureRequest):
    filename = Path(str(request.fspath)).name
    test_name = request.node.name.split("[", 1)[0]
    identity = _AUTHORIZED_INTERNAL_TESTS.get((filename, test_name))
    if identity is None:
        yield
        return

    topology, run_id = identity

    if filename == "test_orchestrator_v2_production_generation.py":
        # Those unit tests predate immutable-runtime enforcement and replace the
        # supervisor with a fake. Supply the minimal committed-code fixture they
        # now need without weakening the production implementation.
        tmp_path = request.getfixturevalue("tmp_path")
        monkeypatch = request.getfixturevalue("monkeypatch")
        lineage_root = tmp_path / "lineage"
        lineage_root.mkdir(parents=True, exist_ok=True)
        (lineage_root / "manifest.json").write_text(
            '{"execution_code_commit":"test-commit"}\n',
            encoding="utf-8",
        )

        from gocube_golden.orchestrator_v2 import production_generation

        monkeypatch.setattr(
            production_generation,
            "execution_commit_from_lineage",
            lambda _root: "test-commit",
        )

        def _fake_ensure(_manager, commit: str):
            assert commit == "test-commit"
            return SimpleNamespace(
                commit=commit,
                path=tmp_path,
                environment=lambda base=None: dict(
                    os.environ if base is None else base
                ),
            )

        monkeypatch.setattr(
            production_generation.ImmutableRuntimeManager,
            "ensure",
            _fake_ensure,
        )

    with _test_authority(
        topology=topology,
        run_id=run_id,
        code_identity="test-commit",
    ):
        yield
