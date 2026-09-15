from __future__ import annotations

import builtins
import json

from gocube_golden.execution_reference import (
    LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE as REFERENCE,
    assess_legion_torus9_selfplay_execution,
    compare_legion_torus9_selfplay_performance,
    effective_active_context_ceiling,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_PROFILE_FINGERPRINT,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)


def _assess(**overrides):
    values = {
        "games": 64,
        "workers": 16,
        "active_games_per_worker": 4,
        "total_active_contexts": 64,
        "batch_cap": 64,
        "wait_ms": 1.0,
        "coalescing": True,
        "shared_memory": True,
        "central_inference_owner": "parent",
        "device": "cuda",
        "interactive": False,
    }
    values.update(overrides)
    return assess_legion_torus9_selfplay_execution(**values)


def test_validated_legion_preset_uses_effective_workload():
    assessment = _assess()

    assert assessment.status == "validated_recommended"
    assert assessment.severity == "none"
    assert assessment.effective_context_ceiling == 64
    assert assessment.issues == ()
    assert assessment.override_reason is None
    json.dumps(assessment.as_dict())


def test_32_games_with_64_requested_contexts_is_underfilled(capsys):
    assessment = _assess(games=32)

    assert effective_active_context_ceiling(
        games=32,
        workers=16,
        active_games_per_worker=4,
        total_active_contexts=64,
    ) == 32
    assert assessment.status == "non_recommended"
    assert assessment.severity == "obvious_underfill_or_path_deviation"
    assert assessment.effective_context_ceiling == 32
    assert any("games=32" in issue for issue in assessment.issues)
    assert any("effective active context ceiling=32" in issue for issue in assessment.issues)
    assert "PERFORMANCE ADVISORY" in capsys.readouterr().out


def test_interactive_reason_is_optional_and_recorded(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _prompt: "nightly underfill check")

    assessment = _assess(games=32, interactive=True)

    assert assessment.prompted is True
    assert assessment.override_reason == "nightly underfill check"


def test_noninteractive_advisory_never_prompts(monkeypatch):
    def fail_if_called(_prompt):
        raise AssertionError("non-interactive execution must not prompt")

    monkeypatch.setattr(builtins, "input", fail_if_called)

    assessment = _assess(games=32, interactive=False)

    assert assessment.prompted is False
    assert assessment.override_reason is None


def test_worker_and_batch_deviations_are_advisory():
    assessment = _assess(
        workers=8,
        active_games_per_worker=2,
        total_active_contexts=16,
        batch_cap=16,
        wait_ms=2.0,
    )

    assert assessment.status == "non_recommended"
    assert any("workers=8" in issue for issue in assessment.issues)
    assert any("batch_cap=16" in issue for issue in assessment.issues)
    assert any("wait_ms=2" in issue for issue in assessment.issues)


def test_postrun_comparison_is_diagnostic_and_underfill_is_not_comparable():
    ok = compare_legion_torus9_selfplay_performance(
        games=64,
        effective_context_ceiling=64,
        moves_per_sec=21.09873,
        mean_batch_rows=13.255,
        p95_batch_rows=39,
        max_batch_rows=60,
    )
    degraded = compare_legion_torus9_selfplay_performance(
        games=64,
        effective_context_ceiling=64,
        moves_per_sec=14.0,
        mean_batch_rows=9.9,
        p95_batch_rows=24,
        max_batch_rows=30,
    )
    underfilled = compare_legion_torus9_selfplay_performance(
        games=32,
        effective_context_ceiling=32,
        moves_per_sec=14.92959,
        mean_batch_rows=9.904,
        p95_batch_rows=24,
        max_batch_rows=30,
    )

    assert ok["status"] == "OK"
    assert ok["delta_pct"] == 0.0
    assert degraded["status"] == "PERFORMANCE_DEGRADED"
    assert underfilled["status"] == "NOT_COMPARABLE_UNDERFILLED"
    assert underfilled["delta_pct"] is None


def test_reference_is_outside_scientific_profile_identity():
    profile = load_torus9_current_profile()

    assert current_torus9_profile_fingerprint(profile) == TORUS9_CURRENT_PROFILE_FINGERPRINT
    assert REFERENCE.reference_moves_per_sec == 21.09873
    assert "fingerprint" not in REFERENCE.as_dict()
