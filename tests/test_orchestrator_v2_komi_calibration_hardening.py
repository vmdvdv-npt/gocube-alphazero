from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gocube_golden.orchestrator_v2.komi_calibration_production import (
    ProductionKomiCalibrationRunnerV2,
    aggregate_candidate_batches,
)


def _batch(
    *,
    komi: float,
    batch: int,
    black: int,
    white: int,
    fingerprint: str = "sha256:" + "1" * 64,
) -> dict[str, object]:
    valid = black + white
    rate = black / valid
    return {
        "komi": komi,
        "batch": batch,
        "evaluation_id": f"eval-{komi:g}-{batch}",
        "evaluation_fingerprint": "sha256:" + str(batch) * 64,
        "output_dir": f"/tmp/eval-{komi:g}-{batch}",
        "identity": {
            "startset": {"fingerprint": fingerprint},
            "candidate": {"checkpoint_id": "M137"},
            "reference": {"checkpoint_id": "M137"},
        },
        "summary": {
            "games": valid,
            "valid_games": valid,
            "black_wins": black,
            "white_wins": white,
            "draws": 0,
            "technical_games": 0,
            "invalid_games": 0,
        },
        "validity": "VALID",
        "stats": {
            "games": valid,
            "valid_games": valid,
            "technical_games": 0,
            "invalid_games": 0,
            "black_wins": black,
            "white_wins": white,
            "draws": 0,
            "black_win_rate": rate,
            "bias": abs(rate - 0.5),
            "raw_score_margin_histogram": [float(batch)],
        },
    }


def test_aggregate_candidate_batches_uses_all_2048_games() -> None:
    result = aggregate_candidate_batches(
        [
            _batch(komi=1.5, batch=1, black=517, white=507),
            _batch(komi=1.5, batch=2, black=514, white=510),
        ]
    )

    stats = result["stats"]
    assert isinstance(stats, dict)
    assert stats["valid_games"] == 2048
    assert stats["black_wins"] == 1031
    assert stats["white_wins"] == 1017
    assert stats["black_win_rate"] == pytest.approx(1031 / 2048)
    assert stats["bias"] == pytest.approx(abs(1031 / 2048 - 0.5))
    assert result["batch"] == 2
    assert len(result["batches"]) == 2


def test_extension_winner_uses_cumulative_batches_not_second_batch_only() -> None:
    runner = object.__new__(ProductionKomiCalibrationRunnerV2)
    runner.config = SimpleNamespace(
        ambiguity_threshold=0.01,
        initial_games=1024,
        extension_games=1024,
        wait_poll_seconds=0.1,
    )
    runner._persist = lambda state: None
    runner._transition = lambda state, stage, **kwargs: None

    state: dict[str, object] = {
        "parent_stop_acknowledged": True,
        "candidates": {
            "1.5": _batch(komi=1.5, batch=1, black=517, white=507),
            "2.5": _batch(komi=2.5, batch=1, black=516, white=508),
        },
        "candidate_batches": {
            "1.5": {
                "1": _batch(komi=1.5, batch=1, black=517, white=507),
                "2": _batch(komi=1.5, batch=2, black=514, white=510),
            },
            "2.5": {
                "1": _batch(komi=2.5, batch=1, black=516, white=508),
                "2": _batch(komi=2.5, batch=2, black=510, white=514),
            },
        },
    }

    selected = runner._select_or_extend(state, None)  # type: ignore[arg-type]

    assert selected == 2.5
    for key in ("1.5", "2.5"):
        aggregate = state["candidates"][key]
        assert aggregate["stats"]["valid_games"] == 2048


def test_parent_soft_stop_must_be_acknowledged_before_arena(tmp_path: Path) -> None:
    control = tmp_path / "control"
    runtime = tmp_path / "runtime"
    control.mkdir()
    runtime.mkdir()
    (control / "soft-stop.json").write_text("{}", encoding="utf-8")
    (runtime / "state.json").write_text(
        json.dumps({"state": "SOFT_STOPPED", "active_generation": None}),
        encoding="utf-8",
    )

    runner = object.__new__(ProductionKomiCalibrationRunnerV2)
    runner.config = SimpleNamespace(wait_poll_seconds=0.1)
    runner.sleeper = lambda _: None
    runner._persist = lambda state: None

    state: dict[str, object] = {}
    parent = SimpleNamespace(owner_root=tmp_path)
    runner._await_parent_soft_stop(state, parent)  # type: ignore[arg-type]

    assert state["parent_stop_acknowledged"] is True
    assert state["parent_stop_state"] == "SOFT_STOPPED"
