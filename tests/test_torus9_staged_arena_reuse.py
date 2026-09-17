from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.torus9_staged_sims_harness_impl import _existing_arena


CANDIDATE = {"artifact_sha256": "candidate-sha"}
REFERENCE = {"artifact_sha256": "reference-sha"}
GAMES = 128
WLD = [80, 48, 0]


def _write_existing_arena(output: Path, telemetry: object) -> None:
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps({"games": GAMES, "W/L/D": WLD, "telemetry": telemetry}) + "\n",
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "candidate": {"artifact_sha256": CANDIDATE["artifact_sha256"]},
                "reference": {"artifact_sha256": REFERENCE["artifact_sha256"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_existing_arena_reuses_healthy_completed_evaluation(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    _write_existing_arena(
        output,
        {
            "technical_games": 0,
            "performance_status": "HEALTHY",
            "performance_failures": [],
        },
    )

    result = _existing_arena(output, CANDIDATE, REFERENCE, GAMES)

    assert result is not None
    assert result["W/L/D"] == WLD


def test_existing_arena_rejects_critical_restart_result_before_wld_reuse(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    _write_existing_arena(
        output,
        {
            "technical_games": 0,
            "performance_status": "CRITICAL",
            "performance_failures": ["lane_occupancy"],
        },
    )

    with pytest.raises(RuntimeError, match="production validity/performance gates"):
        _existing_arena(output, CANDIDATE, REFERENCE, GAMES)

    persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert persisted["W/L/D"] == WLD
    assert persisted["telemetry"]["performance_status"] == "CRITICAL"


@pytest.mark.parametrize("status", ["WARNING", "SEVERE_WARNING"])
def test_existing_arena_reuses_warning_without_hard_failures(
    tmp_path: Path,
    status: str,
) -> None:
    output = tmp_path / "evaluation"
    _write_existing_arena(
        output,
        {
            "technical_games": 0,
            "performance_status": status,
            "performance_failures": [],
        },
    )

    result = _existing_arena(output, CANDIDATE, REFERENCE, GAMES)

    assert result is not None
    assert result["W/L/D"] == WLD


@pytest.mark.parametrize(
    "telemetry",
    [
        {"technical_games": 0, "performance_failures": []},
        {
            "technical_games": 0,
            "performance_status": None,
            "performance_failures": [],
        },
        {"technical_games": 0, "performance_status": "HEALTHY"},
        {
            "technical_games": 0,
            "performance_status": "HEALTHY",
            "performance_failures": {},
        },
    ],
)
def test_existing_arena_fails_closed_on_missing_or_malformed_performance_telemetry(
    tmp_path: Path,
    telemetry: object,
) -> None:
    output = tmp_path / "evaluation"
    _write_existing_arena(output, telemetry)

    with pytest.raises(RuntimeError, match="production validity/performance gates"):
        _existing_arena(output, CANDIDATE, REFERENCE, GAMES)
