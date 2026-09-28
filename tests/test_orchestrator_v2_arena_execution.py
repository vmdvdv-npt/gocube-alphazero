from __future__ import annotations

import pytest

from gocube_golden.orchestrator_v2.execution_permit import (
    _test_authority,
    require_arena_execution,
    require_child_execution_permit,
)


def test_arena_execution_rejects_direct_unauthorized_call() -> None:
    with pytest.raises(RuntimeError, match="production execution requires"):
        require_arena_execution("test.direct-arena", topology="cube4")


def test_arena_execution_uses_live_orchestrator_authority_run_id() -> None:
    with _test_authority(
        topology="cube4",
        run_id="evaluation-cube4-001",
        code_identity="test-commit",
    ):
        assert (
            require_arena_execution("test.live-arena", topology="cube4")
            == "evaluation-cube4-001"
        )
        permit = require_child_execution_permit(
            "test.nested-arena",
            action_type="arena",
            topology="cube4",
            run_id="evaluation-cube4-001",
        )
        assert permit["run_id"] == "evaluation-cube4-001"
        assert permit["action_type"] == "arena"


def test_arena_execution_rejects_authority_topology_mismatch() -> None:
    with _test_authority(
        topology="cube4",
        run_id="evaluation-cube4-002",
        code_identity="test-commit",
    ):
        with pytest.raises(RuntimeError, match="topology mismatch"):
            require_arena_execution("test.wrong-topology", topology="cube5")
