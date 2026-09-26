from __future__ import annotations

from pathlib import Path

import pytest

from gocube_golden.orchestrator_v2.execution_permit import (
    DEFAULT_PERMIT_TTL_SECONDS,
    PERMIT_ENV,
    _child_execution_permit,
    _production_authority,
)
import gocube_golden.orchestrator_v2.execution_permit as execution_permit
from gocube_golden.orchestrator_v2.run_spec import RunSpecV2
from gocube_golden.orchestrator_v2.workflow import WorkflowRunner, WorkflowSpec
from tools.arena_profiles import get_profile


def test_run_spec_validates_supervision_and_run_owned_search() -> None:
    spec = RunSpecV2.from_dict(
        {
            "mode": "arena",
            "arena": {
                "search": {"simulations": 256, "cpuct": 2.0, "fpu": -0.25},
                "supervision": {
                    "default": {"max_retries": 1},
                    "arena": {"progress_timeout_seconds": 90.0},
                },
            },
        }
    )
    assert spec.supervision["arena"]["progress_timeout_seconds"] == 90.0
    with pytest.raises(ValueError, match="unsupported"):
        RunSpecV2.from_dict(
            {"mode": "arena", "arena": {"search": {"historical_whitelist": True}}}
        )
    with pytest.raises(ValueError, match="unsupported"):
        RunSpecV2.from_dict(
            {"mode": "arena", "arena": {"supervision": {"arena": {"typo": 1}}}}
        )


def test_torus_search_parameters_are_evaluation_owned() -> None:
    profile = get_profile(
        "torus9|komi=2.5|simulations=256|cpuct=2.0|fpu=-0.25|watchdog=777|5ch"
    )
    contract = profile.scientific_contract(type("Config", (), {"games": 2})())
    assert contract["simulations"] == 256
    assert contract["cpuct"] == 2.0
    assert contract["fpu"] == -0.25
    assert contract["watchdog"] == 777


def test_permit_is_refreshed_for_each_attempt_after_legacy_ttl(monkeypatch) -> None:
    clock = iter((0.0, DEFAULT_PERMIT_TTL_SECONDS + 1.0))
    monkeypatch.setattr(execution_permit.time, "time", lambda: next(clock))
    with _production_authority(
        mode="arena", topology="torus9", run_id="evaluation", code_identity="commit"
    ):
        with _child_execution_permit(
            action_type="arena",
            topology="torus9",
            run_id="evaluation",
            code_identity="commit",
            attempt=1,
        ) as first:
            first_payload = dict(first)
            assert first_payload["attempt"] == 1
        with _child_execution_permit(
            action_type="arena",
            topology="torus9",
            run_id="evaluation",
            code_identity="commit",
            attempt=2,
        ) as second:
            assert second["attempt"] == 2
            assert second["issued_at"] > first_payload["issued_at"]
            assert PERMIT_ENV in execution_permit.os.environ


def test_workflow_commits_selection_and_does_not_repeat_completed_steps(tmp_path: Path) -> None:
    spec = WorkflowSpec.from_dict(
        {
            "workflow_id": "calibration-select-training",
            "topology": "torus9",
            "steps": [
                {
                    "step_id": "calibration",
                    "action": "calibration",
                    "config": {
                        "arms": [
                            {"komi": 0.5, "bias": 0.02},
                            {"komi": 1.5, "bias": 0.01},
                            {"komi": 2.5, "bias": 0.03},
                        ]
                    },
                },
                {
                    "step_id": "selection",
                    "action": "select",
                    "dependencies": ["calibration"],
                    "config": {
                        "source": {"$ref": "calibration.outputs.arms"},
                        "score": "bias",
                        "value": "komi",
                        "direction": "minimize",
                    },
                },
                {
                    "step_id": "training",
                    "action": "continuous_training",
                    "dependencies": ["selection"],
                    "config": {"komi": {"$ref": "selection.outputs.selected"}},
                },
            ],
        }
    )
    calls: list[object] = []

    def calibration(*, config, **_):
        return {"arms": config["arms"]}

    def training(*, config, **_):
        calls.append(config["komi"])
        return {"checkpoint": "reference-only"}

    first = WorkflowRunner(
        spec,
        root=tmp_path / "workflow",
        handlers={"calibration": calibration, "continuous_training": training},
    ).run()
    assert first["state"] == "COMPLETED"
    assert calls == [1.5]

    second = WorkflowRunner(
        spec,
        root=tmp_path / "workflow",
        handlers={"calibration": lambda **_: pytest.fail("completed step repeated"), "continuous_training": training},
    ).run()
    assert second["state"] == "COMPLETED"
    assert calls == [1.5]
