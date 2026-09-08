from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alphazero.envs.gocube.b_evaluation import (
    B_BOOTSTRAP_REPLICATES,
    B_BOOTSTRAP_SEED,
    B_HELDOUT_DEPTH_SCHEDULE,
    B_HELDOUT_POSITION_IDS,
    B_HELDOUT_SUITE_ID,
    B_HELDOUT_SUITE_POSITION_COUNT,
    B_HELDOUT_SUITE_SHA256,
    B_STATISTICAL_METHOD_IDENTIFIER,
    authoritative_suite_game_class,
    canonical_json_bytes,
    extension_seed_decision,
    hierarchical_paired_bootstrap,
    outcome_score_b1,
    pair_score_b1,
    select_checkpoint_at_or_after,
    validate_frozen_suite,
    validate_pairing_invariants,
)
from tools.build_gocube_b_heldout_suite import build_payload
from tools import evaluate_gocube_b_experiment as b_evaluator
from tools.evaluate_gocube_b_experiment import play_evaluation_game
from tools.evaluate_gocube_checkpoints import reject_b_experiment_checkpoint


SUITE = Path(__file__).parents[1] / "evaluation" / "gocube-b-heldout-suite-v1.json"


def test_canonical_suite_is_frozen_and_reproducible():
    assert SUITE.is_file()
    payload, positions = validate_frozen_suite(SUITE)
    assert payload["suite_id"] == B_HELDOUT_SUITE_ID
    assert payload["position_count"] == B_HELDOUT_SUITE_POSITION_COUNT == 16
    assert payload["depth_schedule"] == list(B_HELDOUT_DEPTH_SCHEDULE)
    assert len(positions) == 16
    assert payload["komi"] == 0.5
    assert payload["topology"] == "cube"
    assert payload["size"] == 4
    assert __import__("hashlib").sha256(SUITE.read_bytes()).hexdigest() == B_HELDOUT_SUITE_SHA256
    assert canonical_json_bytes(build_payload()) == SUITE.read_bytes()
    assert [item[0]["position_id"] for item in positions] == list(B_HELDOUT_POSITION_IDS)
    assert all(not item[1].win_state().any() for item in positions)


def test_tampered_suite_and_other_sha_are_rejected(tmp_path):
    tampered = tmp_path / "tampered.json"
    raw = bytearray(SUITE.read_bytes())
    raw[-2] = ord(" ") if raw[-2] != ord(" ") else ord("\n")
    tampered.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_frozen_suite(tampered)

    alternate = tmp_path / "alternate.json"
    payload = json.loads(SUITE.read_text(encoding="utf-8"))
    payload["suite_id"] = "another-valid-looking-suite"
    alternate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_frozen_suite(alternate)


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("win", 1.0), ("draw", 0.5), ("NO_RESULT", 0.5), ("loss", 0.0)),
)
def test_outcome_and_pair_scoring(raw, expected):
    assert outcome_score_b1(raw) == expected
    assert pair_score_b1([raw, raw]) == expected
    assert np.mean([outcome_score_b1("win"), outcome_score_b1("NR"), outcome_score_b1("loss")]) == 0.5


@pytest.mark.parametrize(
    ("scores", "expected"),
    ((["win", "loss"], 0.5), (["win", "draw"], 0.75), (["NR", "loss"], 0.25),
     (["NR", "NR"], 0.5), (["win", "win"], 1.0)),
)
def test_pair_examples(scores, expected):
    assert pair_score_b1(scores) == expected


def _synthetic_games():
    games = []
    for position_id in B_HELDOUT_POSITION_IDS:
        for index, (b0, b1, outcome) in enumerate(
            (("black", "white", "win"), ("white", "black", "loss"))
        ):
            games.append(
                {
                    "training_seed": 0,
                    "position_id": position_id,
                    "pair_game_index": index,
                    "b0_color": b0,
                    "b1_color": b1,
                    "starting_semantic_state_fingerprint": "state-" + position_id,
                    "raw_outcome": outcome,
                    "b1_game_score": outcome_score_b1(outcome),
                }
            )
    return games


def test_pairing_invariants_reject_missing_extra_duplicate_and_unbalanced_games():
    games = _synthetic_games()
    grouped = validate_pairing_invariants(games, expected_position_ids=B_HELDOUT_POSITION_IDS)
    assert len(grouped) == 16
    with pytest.raises(ValueError, match="exactly two"):
        validate_pairing_invariants(games[:-1], expected_position_ids=B_HELDOUT_POSITION_IDS)
    duplicate = list(games)
    duplicate[1] = {**duplicate[1], "pair_game_index": 0}
    with pytest.raises(ValueError, match="pair_game_index"):
        validate_pairing_invariants(duplicate, expected_position_ids=B_HELDOUT_POSITION_IDS)
    unbalanced = list(games)
    unbalanced[1] = {**unbalanced[1], "b0_color": "black", "b1_color": "white"}
    with pytest.raises(ValueError, match="balanced"):
        validate_pairing_invariants(unbalanced, expected_position_ids=B_HELDOUT_POSITION_IDS)


def test_hierarchical_bootstrap_is_deterministic_and_paired():
    all_half = {seed: [0.5] * 16 for seed in (0, 1, 2)}
    result = hierarchical_paired_bootstrap(all_half)
    assert result["replicates"] == B_BOOTSTRAP_REPLICATES
    assert result["seed"] == B_BOOTSTRAP_SEED
    assert result["ci95_delta"] == [0.0, 0.0]
    assert hierarchical_paired_bootstrap(all_half) == result
    all_win = {seed: [1.0] * 16 for seed in (0, 1, 2)}
    assert hierarchical_paired_bootstrap(all_win)["ci95_delta"] == [0.5, 0.5]


def test_extension_criterion_includes_boundary_and_stop_case():
    bootstrap = {"ci95_delta": [0.01, 0.20]}
    approved = extension_seed_decision(
        {0: 0.0, 1: 0.10, 2: -0.10}, bootstrap, experiment_contract_sha256="a" * 64
    )
    assert approved["approved"] is True
    assert approved["criterion_evidence"]["variance_exceeded"] is True
    stopped = extension_seed_decision(
        {0: 0.01, 1: 0.02, 2: 0.03}, bootstrap, experiment_contract_sha256="a" * 64
    )
    assert stopped["approved"] is False
    assert stopped["decision"] == "stop_at_three"


def test_milestone_selects_first_checkpoint_by_cumulative_counter():
    candidates = [
        {"iteration": 7, "cumulative_new_samples": 9_800_000},
        {"iteration": 8, "cumulative_new_samples": 10_400_000},
    ]
    selected = select_checkpoint_at_or_after(candidates, 10_000_000)
    assert selected["iteration"] == 8
    assert selected["scientific_counter"] == 10_400_000
    assert selected["overshoot"] == 400_000


def test_b_evaluator_pins_50_sims_and_legacy_wilson_path_is_a_trap():
    with pytest.raises(SystemExit):
        b_evaluator.main(["--sims", "1"])
    with pytest.raises(RuntimeError, match="evaluate_gocube_b_experiment.py"):
        reject_b_experiment_checkpoint(
            {"args": {"gocube_experiment_contract_id": "gocube-b-experiment-contract-v1"}},
            "B0",
        )


def test_b4_pair_loop_replays_one_state_and_swaps_model_colors():
    _, positions = validate_frozen_suite(SUITE)
    position = positions[0][0]
    game_cls = authoritative_suite_game_class()

    class PassPlayer:
        def __init__(self):
            self.seen_start_fingerprints = []

        def reset(self):
            pass

        def update(self, state, action):
            pass

        def __call__(self, state):
            self.seen_start_fingerprints.append(state)
            return state.pass_action()

    players = [PassPlayer(), PassPlayer()]
    game_a = play_evaluation_game(
        players,
        game_cls,
        position,
        pair_game_index=0,
        training_seed=0,
        evaluation_seed=1,
        sample_milestone=40_000_000,
    )
    game_b = play_evaluation_game(
        players,
        game_cls,
        position,
        pair_game_index=1,
        training_seed=0,
        evaluation_seed=2,
        sample_milestone=40_000_000,
    )
    grouped = validate_pairing_invariants([game_a, game_b], expected_position_ids=[position["position_id"]])
    assert game_a["starting_semantic_state_fingerprint"] == game_b["starting_semantic_state_fingerprint"]
    assert (game_a["b0_color"], game_a["b1_color"]) == ("black", "white")
    assert (game_b["b0_color"], game_b["b1_color"]) == ("white", "black")
    assert grouped[position["position_id"]]["pair_score_b1"] == 0.5
