from __future__ import annotations

from tools.torus9_post_learning_diagnostics import classify_truncation, kl_divergence, wilson_interval


def test_diagnostic_statistics_and_truncation_labels_are_stable():
    low, high = wilson_interval(61, 64)
    assert 0.87 < low < 0.88
    assert 0.98 < high < 0.99
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0

    pass_case = classify_truncation(
        last100_passes=20,
        last100_captures=0,
        last100_point_moves=80,
        point_repeats=0,
        superko_last100=50,
    )
    assert pass_case["primary"] == "PASS_AVOIDANCE"
    assert pass_case["rules_issue_indicated"] is False

    capture_case = classify_truncation(
        last100_passes=3,
        last100_captures=30,
        last100_point_moves=97,
        point_repeats=0,
        superko_last100=10,
    )
    assert capture_case["primary"] == "CAPTURE_CYCLE"


def test_point_repeat_takes_priority_over_capture_label():
    result = classify_truncation(
        last100_passes=0,
        last100_captures=40,
        last100_point_moves=100,
        point_repeats=1,
        superko_last100=100,
    )
    assert result["primary"] == "CAPTURE_CYCLE"
    assert result["labels"] == ["CAPTURE_CYCLE", "SUPERKO_DRIVEN_LOOP"]
