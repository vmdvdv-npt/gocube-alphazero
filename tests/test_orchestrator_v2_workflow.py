from __future__ import annotations

import json
import os
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
from gocube_golden.orchestrator_v2.workflow import (
    WorkflowControllerLease,
    WorkflowError,
    WorkflowRunner,
    WorkflowSpec,
)
from gocube_golden.orchestrator_v2.production_entrypoint import _continuous_config
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


def test_workflow_controller_lease_is_durable_and_cleans_up_its_own_token(tmp_path: Path) -> None:
    root = tmp_path / "workflow"
    with WorkflowControllerLease(root, "durable-workflow"):
        record = json.loads(
            (root / "runtime" / "controller.json").read_text(encoding="utf-8")
        )
        assert record["schema"] == "gocube-orchestrator-v2-controller-v1"
        assert record["workflow_id"] == "durable-workflow"
        assert record["pid"] == os.getpid()
        assert record["process_group"] == os.getpgid(os.getpid())
    assert not (root / "runtime" / "controller.json").exists()
    with pytest.raises(ValueError, match="unsupported"):
        RunSpecV2.from_dict(
            {"mode": "arena", "arena": {"supervision": {"arena": {"typo": 1}}}}
        )
    with pytest.raises(ValueError, match="only supported value"):
        RunSpecV2.from_dict(
            {"mode": "arena", "arena": {"search": {"temperature": 1.0}}}
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
                    "config": {
                        "komi": {"$ref": "selection.outputs.selected"},
                    },
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


def test_workflow_runs_ready_graph_until_complete_and_resolves_list_indexes(tmp_path: Path) -> None:
    spec = WorkflowSpec.from_dict(
        {
            "workflow_id": "unordered-list-refs",
            "topology": "torus9",
            "steps": [
                {
                    "step_id": "training",
                    "action": "continuous_training",
                    "dependencies": ["selection"],
                    "config": {
                        "komi": {"$ref": "selection.outputs.selected"},
                        "selected_result": {"$ref": "calibration.outputs.1"},
                    },
                },
                {
                    "step_id": "selection",
                    "action": "select",
                    "dependencies": ["calibration"],
                    "config": {
                        "source": {"$ref": "calibration.outputs"},
                        "score": "metrics.bias",
                        "value": "komi",
                    },
                },
                {
                    "step_id": "calibration",
                    "action": "calibration",
                    "config": {},
                },
            ],
        }
    )
    seen: list[object] = []
    result = WorkflowRunner(
        spec,
        root=tmp_path / "unordered",
        handlers={
            "calibration": lambda **_: [
                {"komi": 1.5, "validity": "VALID", "metrics": {"bias": 0.2}},
                {"komi": 2.5, "validity": "VALID", "metrics": {"bias": 0.1}},
            ],
            "continuous_training": lambda *, config, **_: seen.append(config) or {"done": True},
        },
    ).run()

    assert result["state"] == "COMPLETED"
    assert seen == [
        {
            "komi": 2.5,
            "selected_result": {
                "komi": 2.5,
                "validity": "VALID",
                "metrics": {"bias": 0.1},
            },
        }
    ]


def test_workflow_stop_is_durable_and_blocks_later_steps(tmp_path: Path) -> None:
    spec = WorkflowSpec.from_dict(
        {
            "workflow_id": "explicit-stop",
            "topology": "torus9",
            "steps": [
                {"step_id": "stop", "action": "stop", "config": {"reason": "operator"}},
                {"step_id": "later", "action": "calibration", "config": {}},
            ],
        }
    )
    calls: list[int] = []
    first = WorkflowRunner(
        spec,
        root=tmp_path / "stop",
        handlers={"calibration": lambda **_: calls.append(1)},
    ).run()
    second = WorkflowRunner(
        spec,
        root=tmp_path / "stop",
        handlers={"calibration": lambda **_: calls.append(2)},
    ).run()

    assert first["state"] == "STOPPED"
    assert second["state"] == "STOPPED"
    assert calls == []
    assert first["steps"]["later"]["status"] == "PENDING"


def test_workflow_durable_supervisor_stop_is_not_retried_or_continued(tmp_path: Path) -> None:
    spec = WorkflowSpec.from_dict(
        {
            "workflow_id": "supervisor-stop",
            "topology": "torus9",
            "steps": [
                {
                    "step_id": "arena",
                    "action": "arena",
                    "failure_policy": {
                        "on_technical_failure": "retry",
                        "max_retries": 3,
                    },
                    "config": {},
                },
                {"step_id": "later", "action": "calibration", "dependencies": ["arena"]},
            ],
        }
    )
    attempts: list[int] = []

    def stopped(**_kwargs):
        attempts.append(1)
        raise RuntimeError("Arena has a durable supervisor stop: supervisor-stop.json")

    result = WorkflowRunner(
        spec,
        root=tmp_path / "supervisor-stop",
        handlers={"arena": stopped, "calibration": lambda **_: pytest.fail("must not continue")},
    ).run()

    assert result["state"] == "STOPPED"
    assert result["steps"]["arena"]["attempts"] == 1
    assert result["steps"]["later"]["status"] == "PENDING"
    assert attempts == [1]


def test_workflow_rejects_invalid_calibration_result_for_selection(tmp_path: Path) -> None:
    spec = WorkflowSpec.from_dict(
        {
            "workflow_id": "invalid-selection",
            "topology": "torus9",
            "steps": [
                {"step_id": "calibration", "action": "calibration", "config": {}},
                {
                    "step_id": "selection",
                    "action": "select",
                    "dependencies": ["calibration"],
                    "config": {
                        "source": {"$ref": "calibration.outputs"},
                        "score": "metrics.bias",
                        "value": "komi",
                    },
                },
            ],
        }
    )
    with pytest.raises(WorkflowError, match="validity"):
        WorkflowRunner(
            spec,
            root=tmp_path / "invalid",
            handlers={
                "calibration": lambda **_: [
                    {"komi": 1.5, "validity": "INVALID", "metrics": {"bias": 0.0}}
                ]
            },
        ).run()


def test_selected_komi_is_applied_to_real_effective_config() -> None:
    config = _continuous_config(
        {
            "parent_checkpoint": {
                "topology": "torus9",
                "lineage_id": "parent",
                "checkpoint_id": "M0",
                "generation": 0,
                "path": "checkpoints/M0.pt",
                "sha256": "sha256:" + "1" * 64,
            },
            "lineage_id": "selected",
            "effective_config": {
                "topology": "torus9",
                "compatibility": {"topology": "torus9", "rules": {"komi": 0.5}},
                "self_play": {"komi": 0.5},
                "arena": {"komi": 0.5},
            },
            "komi": 2.5,
            "generations": 0,
            "arena_cadence": 1,
            "arena_config": {
                "games": 2,
                "workers": 1,
                "games_per_worker": 1,
                "inference_batch_rows": 1,
                "device": "cpu",
                "strict_production": False,
            },
        }
    )
    assert config.effective_config.self_play["komi"] == 2.5
    assert config.effective_config.arena["komi"] == 2.5
    assert config.effective_config.compatibility["rules"]["komi"] == 2.5
