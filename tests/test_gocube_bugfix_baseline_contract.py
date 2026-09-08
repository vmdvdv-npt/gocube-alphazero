from __future__ import annotations

import json
from pathlib import Path

from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.katago_train import parse_args
from alphazero.envs.gocube.integration.contract import resolve_model_contract


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


def test_architecture_reference_scope_is_explicit():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["architecture_reference_commit"] == (
        "adb8d9ca138d461cc3324c44930458561a031710"
    )
    assert baseline["reference_scope"] == "architecture-search-training-hyperparameters"
    assert "baseline_commit" not in baseline
    assert "rules_implementation_version" not in baseline["rules"]
    assert baseline["rules"]["rules_version"] == 3
    assert baseline["rules"]["komi"] == 0.5
    assert set(baseline["allowed_non_architecture_changes"]) == {
        "replay_format_version",
        "training_target_semantics",
        "rules_implementation_version",
        "sample_clock_contract_version",
        "auxiliary_target_masks",
        "metadata_schema",
        "seed_contract",
        "finite_validation_contract",
    }


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


def test_baseline_and_g1_share_current_rules_and_target_semantics():
    baseline_cls, baseline_args = build_hardened_training_args(
        parse_args(["--model-profile", "baseline", "--smoke"])
    )
    g1_cls, g1_args = build_hardened_training_args(
        parse_args(["--model-profile", "g1", "--smoke"])
    )

    for field in (
        "KATAGO_RULES_VERSION",
        "KATAGO_RULES_IMPLEMENTATION_VERSION",
        "KATAGO_REFERENCE_COMMIT",
        "KOMI",
        "TERMINAL_ADJUDICATOR_ID",
    ):
        assert getattr(baseline_cls, field, None) == getattr(g1_cls, field, None)
    assert baseline_cls.KOMI == g1_cls.KOMI == 0.5
    assert baseline_cls.rules_fingerprint() == g1_cls.rules_fingerprint()

    for field in (
        "gocube_value_target_semantics",
        "gocube_score_target_semantics",
        "gocube_ownership_target_semantics",
        "gocube_katago_search_contract",
        "gocube_katago_search_reference_commit",
        "gocube_katago_exploration_contract",
        "search_utility_mode",
        "gocube_train_samples_per_new_sample",
        "gocube_replay_window_iters",
        "gocube_cleanup_training_prob",
        "gocube_early_fork_game_prob",
        "gocube_fork_game_prob",
        "numMCTSSims",
        "numFastSims",
        "arenaMCTSSims",
        "lr",
        "train_batch_size",
        "depth",
        "num_channels",
        "value_dense_layers",
        "score_dense_layers",
        "optimizer_args",
    ):
        assert baseline_args[field] == g1_args[field], field

    baseline_contract = resolve_model_contract(baseline_cls, baseline_args)
    g1_contract = resolve_model_contract(g1_cls, g1_args)
    assert baseline_contract.topology_kind == g1_contract.topology_kind
    assert baseline_contract.topology_size == g1_contract.topology_size
    assert baseline_contract.point_count == g1_contract.point_count
    assert baseline_contract.action_size == g1_contract.action_size
    assert baseline_contract.action_schema == g1_contract.action_schema
    assert baseline_contract.point_order_fingerprint == g1_contract.point_order_fingerprint
    assert baseline_contract.adjacency_fingerprint == g1_contract.adjacency_fingerprint
    assert baseline_contract.topology_fingerprint == g1_contract.topology_fingerprint
    assert baseline_contract.targets_schema == g1_contract.targets_schema
    assert baseline_contract.output_heads == g1_contract.output_heads


def test_baseline_and_g1_differ_only_in_model_profile_contract_fields():
    baseline_cls, baseline_args = build_hardened_training_args(
        parse_args(["--model-profile", "baseline", "--smoke"])
    )
    g1_cls, g1_args = build_hardened_training_args(
        parse_args(["--model-profile", "g1", "--smoke"])
    )
    baseline_contract = resolve_model_contract(baseline_cls, baseline_args)
    g1_contract = resolve_model_contract(g1_cls, g1_args)

    assert {
        key
        for key in set(baseline_args) | set(g1_args)
        if baseline_args.get(key) != g1_args.get(key)
    } == {
        "gocube_model_profile",
        "gocube_network_architecture",
        "gocube_observation_schema",
        "gocube_structural_feature_schema",
        "gocube_structural_feature_channels",
    }
    assert set(baseline_contract.differences(g1_contract)) == {
        "game_class_id",
        "observation_schema",
        "observation_shape",
        "network_architecture_id",
        "network_architecture_fingerprint",
    }
