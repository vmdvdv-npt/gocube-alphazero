from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gocube_golden.scenarios.calibration import CalibrationArm, CalibrationRunner
from gocube_golden.scenarios.contracts import (
    ActionOutcome,
    ActionRequest,
    ExecutionStatus,
    ScientificValidity,
)
from gocube_golden.scenarios.komi.policy import aggregate_candidate_batches, select_komi


def _batch(komi: float, batch: int, black: int, white: int) -> dict[str, object]:
    valid = black + white
    return {
        "komi": komi,
        "batch": batch,
        "evaluation_id": f"eval-{komi:g}-{batch}",
        "evaluation_fingerprint": f"fingerprint-{komi:g}-{batch}",
        "output_dir": f"evaluations/eval-{komi:g}-{batch}",
        "identity": {"startset": {"fingerprint": "startset-1"}},
        "validity": "VALID",
        "stats": {
            "games": valid,
            "valid_games": valid,
            "technical_games": 0,
            "invalid_games": 0,
            "black_wins": black,
            "white_wins": white,
            "draws": 0,
        },
    }


def test_experiment_winner_policy_preserves_strict_candidate_rule() -> None:
    from gocube_golden.orchestrator_v2.contracts import CheckpointRef
    from gocube_golden.scenarios.experiment.policy import WinnerRule

    candidate = CheckpointRef("torus9", "candidate", "M1", 1, "checkpoints/M1.pt", "sha256:" + "1" * 64)
    reference = CheckpointRef("torus9", "reference", "M1", 1, "checkpoints/M1.pt", "sha256:" + "2" * 64)
    rule = WinnerRule()
    assert rule.choose(candidate=candidate, reference=reference, wins=3, losses=2) == candidate
    assert rule.choose(candidate=candidate, reference=reference, wins=2, losses=3) == reference
    assert rule.choose(candidate=candidate, reference=reference, wins=2, losses=2) == reference


@pytest.mark.parametrize(
    ("difference", "requires_extension"),
    ((0.0099, True), (0.01, False), (0.0101, False)),
)
def test_komi_extension_boundary_is_strictly_below_threshold(difference: float, requires_extension: bool) -> None:
    decision = select_komi(
        {1.5: 0.20, 2.5: 0.20 + difference},
        ambiguity_threshold=0.01,
    )
    assert decision.requires_extension is requires_extension


def test_komi_policy_uses_cumulative_counts_and_exact_tie_break() -> None:
    initial = {1.5: 0.20, 2.5: 0.201}
    decision = select_komi(
        initial,
        cumulative_biases={1.5: 0.03, 2.5: 0.03},
    )
    assert decision.selected_komi == 1.5
    assert decision.cumulative_biases == {1.5: 0.03, 2.5: 0.03}


def test_batch_ledger_rejects_duplicate_or_missing_evidence() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        aggregate_candidate_batches([_batch(1.5, 1, 517, 507), _batch(1.5, 3, 517, 507)])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_candidate_batches([_batch(1.5, 1, 517, 507), _batch(1.5, 1, 517, 507)])


def test_batch_ledger_aggregates_counters_not_percentages() -> None:
    result = aggregate_candidate_batches(
        [_batch(1.5, 1, 517, 507), _batch(1.5, 2, 514, 510)],
        expected_komi=1.5,
        initial_games=1024,
        extension_games=1024,
    )
    stats = result["stats"]
    assert isinstance(stats, dict)
    assert stats["valid_games"] == 2048
    assert stats["black_wins"] == 1031
    assert stats["black_win_rate"] == pytest.approx(1031 / 2048)


def test_action_envelope_is_stable_and_invalid_scientific_result_is_terminal() -> None:
    request = ActionRequest.from_payload(
        owner="experiment-owner",
        scenario_id="exp-1",
        action_id="exp-1:stage1:arena",
        action_type="arena",
        request={"evaluation_id": "eval-1", "candidate": "B-final"},
        correlation_id="exp-1",
    )
    assert ActionRequest.from_dict(request.to_dict()).request_fingerprint == request.request_fingerprint
    outcome = ActionOutcome(
        action_id=request.action_id,
        request_fingerprint=request.request_fingerprint,
        execution_status=ExecutionStatus.COMPLETED,
        result_refs=("evaluations/eval-1/summary.json",),
        commit_evidence={"summary_sha256": "sha256:" + "a" * 64},
        scientific_validity=ScientificValidity.INVALID,
        reused=True,
    )
    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert outcome.scientific_validity is ScientificValidity.INVALID
    outcome.validate_against(request)
    with pytest.raises(ValueError, match="commit evidence"):
        ActionOutcome(
            action_id="a",
            request_fingerprint="fingerprint",
            execution_status="completed",
            result_refs=("result",),
            scientific_validity="VALID",
        )


def test_simple_calibration_preserves_arm_order_without_m137_policy() -> None:
    calls: list[str] = []
    runner = CalibrationRunner(
        [CalibrationArm("first", {"komi": 1.5}), CalibrationArm("second", {"komi": 2.5})],
        lambda arm, _index: calls.append(arm.arm_id) or arm.request["komi"],
    )
    assert runner.run() == [1.5, 2.5]
    assert calls == ["first", "second"]


def test_pure_policies_have_no_process_or_transport_imports() -> None:
    root = Path(__file__).parents[1] / "gocube_golden" / "scenarios"
    forbidden = {"subprocess", "socket", "requests", "urllib", "telegram", "http"}
    for path in (root / "experiment" / "policy.py", root / "komi" / "policy.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports |= {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not imports & forbidden
