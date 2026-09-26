from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig
from gocube_golden.performance_tuning import (
    FailureCategory,
    Mode,
    PerformanceTuningRunner,
    Plan,
    SelectedProfile,
    TuningExecutionError,
    assess_observation,
    select_profile,
    scientific_contract_fingerprint,
)


SHA = "sha256:" + "a" * 64


def _parent() -> CheckpointRef:
    return CheckpointRef("torus9", "parent", "M7", 7, "checkpoints/M7.pt", SHA)


def _plan(tmp_path: Path, *, budget: int | None = None) -> Plan:
    baseline = Mode("baseline", 4, 64)
    mode = Mode("six-by-96", 6, 96)
    return Plan(
        tuning_id="tuning-1",
        topology="torus9",
        parent_checkpoint=_parent(),
        baseline=baseline,
        modes=(mode,),
        workers=16,
        scientific_config_fingerprint="scientific-fingerprint",
        measurement_budget=budget,
        measurement_contract={"expected_games": 128},
        owner_id="owner-1",
        owner_root=tmp_path,
    )


def _metrics(mode: Mode, games_per_hour: float) -> dict[str, object]:
    return {
        "orchestrator_selfplay": {
            "games": 128,
            "moves": 100,
            "games_per_hour": games_per_hour,
            "selfplay_time_sec": 1.0,
            "execution": {
                "workers": 16,
                "active_games_per_worker": mode.active_games_per_worker,
                "total_active_contexts": mode.total_active_contexts,
            },
        }
    }


def test_policy_preserves_legacy_speed_then_wall_tie_break() -> None:
    plan = _plan(Path("/tmp"))
    baseline = assess_observation(_metrics(plan.baseline, 100.0), {"mode": plan.baseline, "generation": 8, "expected_games": 128, "workers": 16})
    mode = assess_observation(_metrics(plan.modes[0], 100.0), {"mode": plan.modes[0], "generation": 9, "expected_games": 128, "workers": 16})
    mode_slow = mode.__class__(
        action_id=mode.action_id,
        mode=mode.mode,
        generation=mode.generation,
        metrics={**mode.metrics, "selfplay_wall_time_sec": 2.0},
        stable=True,
        raw_metrics=mode.raw_metrics,
    )
    selected = select_profile(plan, [baseline, mode_slow])
    assert selected.mode.label == plan.baseline.label
    assert selected.source == "measured_selection"


def test_missing_nan_and_infinite_metrics_are_not_stable() -> None:
    plan = _plan(Path("/tmp"))
    observation = assess_observation(
        {"games": 128, "selfplay_time_sec": float("nan"), "games_per_hour": float("inf")},
        {"mode": plan.baseline, "generation": 8, "expected_games": 128, "workers": 16},
    )
    assert observation.stable is False
    assert "missing self-play wall time" in observation.stability_reasons
    assert "missing games per hour" in observation.stability_reasons


def test_selected_profile_rejects_wrong_scientific_contract() -> None:
    profile = SelectedProfile(
        profile_id="tuning-1:baseline",
        topology="torus9",
        mode=Mode("baseline", 4, 64),
        execution_overrides={"active_games_per_worker": 4, "total_active_contexts": 64},
        scientific_config_fingerprint="scientific-fingerprint",
        parent_checkpoint=_parent().to_dict(),
    )
    with pytest.raises(ValueError, match="scientific contract"):
        profile.validate_applicability(topology="torus9", scientific_config_fingerprint="other")


def test_standalone_runner_uses_budgeted_actions_and_exports_profile(tmp_path: Path) -> None:
    plan = _plan(tmp_path, budget=2)
    calls: list[tuple[str, int, str]] = []

    def execute_generation(*, parent_checkpoint, execution_profile, action_id):
        generation = int(parent_checkpoint["generation"]) + 1
        calls.append((execution_profile.label, generation, action_id))
        ref = CheckpointRef("torus9", "tuning-1", f"M{generation}", generation, f"checkpoints/M{generation}.pt", SHA)
        return SimpleNamespace(ref=ref, metrics=_metrics(execution_profile, 200.0 if execution_profile.label != "baseline" else 100.0))

    result = PerformanceTuningRunner(plan, execute_generation=execute_generation, owner_root=tmp_path).run()

    assert result.state == "COMPLETED"
    assert result.selected_profile is not None
    assert result.selected_profile.mode.label == "six-by-96"
    assert [item[0] for item in calls] == ["baseline", "six-by-96"]
    assert (tmp_path / "runtime" / "tuning" / "tuning-1" / "selected-profile.json").is_file()
    assert len(calls) <= 2


def test_runner_retries_failed_mode_at_baseline_once(tmp_path: Path) -> None:
    plan = _plan(tmp_path, budget=4)
    calls: list[str] = []
    failed = False

    def execute_generation(*, parent_checkpoint, execution_profile, action_id):
        nonlocal failed
        calls.append(execution_profile.label)
        if execution_profile.label == "six-by-96" and not failed:
            failed = True
            raise RuntimeError("production generation stopped after supervisor failure")
        generation = int(parent_checkpoint["generation"]) + 1
        ref = CheckpointRef("torus9", "tuning-1", f"M{generation}", generation, f"checkpoints/M{generation}.pt", SHA)
        return SimpleNamespace(ref=ref, metrics=_metrics(execution_profile, 100.0))

    result = PerformanceTuningRunner(plan, execute_generation=execute_generation, owner_root=tmp_path).run()

    assert result.selected_profile is not None
    assert calls == ["baseline", "six-by-96", "baseline"]
    assert result.selected_profile.source == "measured_selection"


def test_dry_run_does_not_create_tuning_state(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    result = PerformanceTuningRunner(plan, execute_generation=lambda **_: None, owner_root=tmp_path).run(dry_run=True)
    assert result.dry_run is True
    assert not (tmp_path / "runtime").exists()
