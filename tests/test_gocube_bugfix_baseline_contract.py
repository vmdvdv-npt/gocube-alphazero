from __future__ import annotations

import json
from pathlib import Path

from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.katago_train import parse_args


BASELINE = Path(__file__).parent / "fixtures" / "gocube_production_profile_baseline.json"


def test_production_baseline_keeps_protected_architecture_and_hyperparameters():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    game_cls, args = build_hardened_training_args(parse_args(["--model-profile", "baseline"]))

    assert game_cls.__name__ == "BaselineDiversifiedPinnedCube4JapaneseGame"
    assert list(game_cls.observation_size()) == baseline["architecture"]["observation_shape"]
    assert game_cls.action_size() == baseline["architecture"]["action_size"]
    assert game_cls.KOMI == baseline["rules"]["komi"]
    assert game_cls.KATAGO_REFERENCE_COMMIT == baseline["rules"]["katago_reference_commit"]
    assert args.gocube_model_profile == baseline["model_profile"]
    assert args.gocube_network_architecture == baseline["architecture"]["network_architecture_id"]
    assert args.gocube_structural_feature_schema == baseline["architecture"]["structural_feature_schema"]
    assert args.gocube_structural_feature_channels == baseline["architecture"]["structural_feature_channels"]

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


def test_baseline_provenance_points_to_the_g1_base_commit():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["baseline_commit"] == "adb8d9ca138d461cc3324c44930458561a031710"
    assert baseline["model_profile"] == "baseline"


def test_production_selector_defaults_to_baseline_and_explicitly_selects_g1():
    default_cls, default_args = build_hardened_training_args(parse_args(["--smoke"]))
    baseline_cls, baseline_args = build_hardened_training_args(
        parse_args(["--model-profile", "baseline", "--smoke"])
    )
    g1_cls, g1_args = build_hardened_training_args(
        parse_args(["--model-profile", "g1", "--smoke"])
    )

    assert default_cls is baseline_cls
    assert default_args == baseline_args
    assert default_args.gocube_model_profile == "baseline"
    assert baseline_cls.observation_size() == (18, 96, 1)
    assert g1_cls.observation_size() == (20, 96, 1)

    profile_fields = {
        "gocube_model_profile",
        "gocube_network_architecture",
        "gocube_observation_schema",
        "gocube_structural_feature_schema",
        "gocube_structural_feature_channels",
    }
    assert {
        key
        for key in set(baseline_args) | set(g1_args)
        if baseline_args.get(key) != g1_args.get(key)
    } <= profile_fields
