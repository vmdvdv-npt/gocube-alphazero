from __future__ import annotations

import pytest

from gocube_golden.orchestrator_v2 import execution_permit
from gocube_golden.orchestrator_v2.execution_permit import (
    _child_execution_permit,
    _test_authority,
    require_arena_execution,
)


def test_arena_execution_rejects_direct_unauthorized_call() -> None:
    with pytest.raises(RuntimeError, match="production execution requires"):
        require_arena_execution("test.direct-arena", topology="cube4")


def test_arena_execution_rejects_live_authority_without_child_permit() -> None:
    with _test_authority(
        topology="cube4",
        run_id="evaluation-cube4-001",
        code_identity="test-commit",
    ):
        with pytest.raises(RuntimeError, match="execution permit"):
            require_arena_execution("test.live-arena", topology="cube4")


def test_arena_execution_uses_signed_child_permit_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _test_authority(
        topology="cube4",
        run_id="evaluation-cube4-002",
        code_identity="test-commit",
    ):
        monkeypatch.setattr(execution_permit.os, "getppid", execution_permit.os.getpid)
        with _child_execution_permit(
            action_type="arena",
            topology="cube4",
            run_id="evaluation-cube4-002",
            code_identity="test-commit",
        ):
            assert (
                require_arena_execution("test.child-arena", topology="cube4")
                == "evaluation-cube4-002"
            )


def test_arena_execution_rejects_permit_topology_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _test_authority(
        topology="cube4",
        run_id="evaluation-cube4-003",
        code_identity="test-commit",
    ):
        monkeypatch.setattr(execution_permit.os, "getppid", execution_permit.os.getpid)
        with _child_execution_permit(
            action_type="arena",
            topology="cube4",
            run_id="evaluation-cube4-003",
            code_identity="test-commit",
        ):
            with pytest.raises(RuntimeError, match="topology mismatch"):
                require_arena_execution("test.wrong-topology", topology="cube5")
