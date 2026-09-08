from __future__ import annotations

import json
from pathlib import Path

from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.katago_train import parse_args


BASELINE = Path(__file__).parent / "fixtures" / "gocube_production_profile_baseline.json"


def test_production_baseline_keeps_protected_architecture_and_hyperparameters():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    game_cls, args = build_hardened_training_args(parse_args([]))

    assert game_cls.__name__ == "G1DiversifiedPinnedCube4JapaneseGame"
    assert list(game_cls.observation_size()) == baseline["architecture"]["observation_shape"]
    assert game_cls.action_size() == baseline["architecture"]["action_size"]
    assert game_cls.KOMI == baseline["rules"]["komi"]
    assert game_cls.KATAGO_REFERENCE_COMMIT == baseline["rules"]["katago_reference_commit"]

    assert args.nnet_type == "graph"
    assert args.num_channels == baseline["architecture"]["num_channels"]
    assert args.depth == baseline["architecture"]["depth"]
    assert list(args.value_dense_layers) == baseline["architecture"]["value_dense_layers"]
    assert list(args.score_dense_layers) == baseline["architecture"]["score_dense_layers"]

    assert args.optimizer.__module__ + "." + args.optimizer.__name__ == baseline["optimizer_training"]["optimizer"]
    assert args.lr == baseline["optimizer_training"]["learning_rate"]
    assert args.train_batch_size == baseline["optimizer_training"]["batch_size"]
    assert dict(args.optimizer_args) == baseline["optimizer_training"]["optimizer_args"]
    assert args.numMCTSSims == baseline["search_selfplay"]["regular_sims"]
    assert args.numFastSims == baseline["search_selfplay"]["fast_sims"]
    assert args.arenaMCTSSims == baseline["search_selfplay"]["arena_sims"]
    assert args.probFastSim == baseline["search_selfplay"]["fast_probability"]
    assert args.gamesPerIteration == baseline["search_selfplay"]["games_per_iteration"]


def test_baseline_only_allows_explicit_contract_metadata_changes():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert set(baseline["allowed_changes"]) == {
        "replay_format_version",
        "training_target_semantics",
        "sample_clock_contract_version",
        "auxiliary_target_masks",
        "metadata_schema",
        "seed_contract",
        "finite_validation_contract",
    }
