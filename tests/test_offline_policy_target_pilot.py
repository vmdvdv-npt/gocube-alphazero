from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gocube_golden.offline_policy_target_pilot import (
    _bootstrap_cluster,
    _checked_payload,
    _distribution_metrics,
    _select_positions,
    _split_games,
    _state_fingerprint,
)


def test_whole_game_split_is_deterministic_and_70_15_15() -> None:
    first = _split_games([f"g{i:04d}" for i in range(100)], seed=2026101207)
    second = _split_games([f"g{i:04d}" for i in range(100)], seed=2026101207)
    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 70, "validation": 15, "test": 15,
    }
    assert len({game for values in first.values() for game in values}) == 100


def test_position_selection_balances_phase_and_color_without_reusing_games() -> None:
    phases = ("opening", "early", "middle", "late")
    candidates = []
    games = {name: [] for name in ("train", "validation", "test")}
    for split, game_count in (("train", 80), ("validation", 40), ("test", 40)):
        for index in range(game_count):
            game_id = f"{split}-{index:03d}"
            games[split].append(game_id)
            phase = phases[index % len(phases)]
            side = "BLACK" if (index // 4) % 2 == 0 else "WHITE"
            candidates.append({"game_id": game_id, "phase": phase, "side": side, "ply": index + 1})
    selected = _select_positions(candidates, games, seed=17, counts={"train": 32, "validation": 16, "test": 16})
    for split, values in selected.items():
        assert len(values) == len({row["game_id"] for row in values})
        strata = {}
        for row in values:
            key = (row["phase"], row["side"])
            strata[key] = strata.get(key, 0) + 1
        assert max(strata.values()) - min(strata.values()) <= 1
    all_games = [row["game_id"] for values in selected.values() for row in values]
    assert len(all_games) == len(set(all_games))


def test_raw_policy_metrics_follow_training_softmax_and_legal_renormalization() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True, False]])
    result = _distribution_metrics(logits, target, legal)
    assert result["policy_ce_raw"] == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert result["policy_kl_raw"] == pytest.approx(result["policy_ce_raw"])
    assert result["policy_kl_legal_renormalized"] == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert result["illegal_policy_mass"] == pytest.approx(1 / 3)


def test_paired_game_cluster_bootstrap_is_fixed_and_negative_favors_b() -> None:
    values = {f"g{i}": -0.1 - i / 100 for i in range(20)}
    first = _bootstrap_cluster(values, seed=23, replicates=1000)
    second = _bootstrap_cluster(values, seed=23, replicates=1000)
    assert first == second
    assert first["mean"] < 0
    assert first["ci95"][1] < 0


def test_state_fingerprint_is_content_sensitive() -> None:
    assert _state_fingerprint({"stones": [0, 1], "history": [[0, 1]]}) == _state_fingerprint(
        {"history": [[0, 1]], "stones": [0, 1]}
    )
    assert _state_fingerprint({"stones": [0, 1]}) != _state_fingerprint({"stones": [0, 1, 0]})


def test_diagnostic_output_must_be_outside_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    inside = runs / "diagnostic"
    payload = {
        "schema": "gocube-d270-policy-target-pilot-v1",
        "runs_root": str(runs), "output": str(inside),
        "parent_checkpoint": str(runs / "torus9/active/az23-m266-m269-offline-lr-ab-20261010-v1-ab-m267-m269-lr-B/checkpoints/M269.pt"),
        "parent_checkpoint_sha256": "sha256:" + "0" * 64,
        "parent_metadata": str(runs / "parent.json"),
        "parent_provenance": str(runs / "parent-provenance.json"),
        "parent_rolling_replay": str(runs / "rolling.jsonl"),
        "source_fresh_replay": str(runs / "torus9/active/torus9-c2560-seed2-online-m262-arena1-s64-20261008-v1/replay/iter-270-fresh.json"),
        "source_fresh_sha256": "sha256:" + "1" * 64,
        "device": "cuda",
    }
    with pytest.raises(ValueError, match="outside runs_root"):
        _checked_payload(payload)
