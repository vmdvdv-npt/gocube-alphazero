"""Cross-profile GoCube Arena keeps one semantic state and per-model inputs."""

from dataclasses import replace

import numpy as np
import pytest

from alphazero.envs.gocube.diversified_game import (
    diversified_baseline_pinned_game_class,
    diversified_pinned_game_class,
    diversified_structural_pinned_game_class,
)
from alphazero.envs.gocube.game import Cube4JapaneseGame
from alphazero.envs.gocube.integration.contract import (
    resolve_model_contract,
)
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.envs.gocube.observation import GoCubeObservationAdapter
from alphazero.envs.gocube.structural import structural_feature_matrix
from tools import gocube_checkpoint_arena as checkpoint_arena


BASELINE = diversified_baseline_pinned_game_class(Cube4JapaneseGame)
G1 = diversified_structural_pinned_game_class(Cube4JapaneseGame)
SEMANTIC = diversified_pinned_game_class(Cube4JapaneseGame)


def test_b0_and_b1_adapters_share_semantic_channels_and_keep_state_immutable():
    state = SEMANTIC()
    semantic_state = state.semantic_state
    valids_before = state.valid_moves().copy()

    baseline = GoCubeObservationAdapter(BASELINE).observation(state)
    g1 = GoCubeObservationAdapter(G1).observation(state)

    assert baseline.shape == (18, 96, 1)
    assert g1.shape == (20, 96, 1)
    np.testing.assert_array_equal(baseline, g1[:18])
    np.testing.assert_array_equal(
        g1[-2:], structural_feature_matrix(Cube4JapaneseGame.logical_topology())
    )
    np.testing.assert_array_equal(state.valid_moves(), valids_before)
    assert state.semantic_state is semantic_state


def test_g1_structural_channels_are_topology_only_and_not_checkpoint_color():
    state = SEMANTIC()
    black_start = GoCubeObservationAdapter(G1).observation(state)
    white_start = GoCubeObservationAdapter(G1).observation(
        SEMANTIC(replace(state.semantic_state, current_player=1))
    )

    np.testing.assert_array_equal(black_start[-2:], white_start[-2:])
    changed = np.flatnonzero(np.any(black_start != white_start, axis=(1, 2)))
    np.testing.assert_array_equal(changed, np.array([4]))


def test_profile_contract_equality_ignores_only_model_representation():
    baseline_cls, baseline_args = build_katago_training_args(
        parse_args(["--model-profile", "baseline", "--smoke"])
    )
    g1_cls, g1_args = build_katago_training_args(
        parse_args(["--model-profile", "g1", "--smoke"])
    )
    assert resolve_model_contract(baseline_cls, baseline_args).observation_shape == (18, 96, 1)
    assert resolve_model_contract(g1_cls, g1_args).observation_shape == (20, 96, 1)
    checkpoint_arena._require_same_contract(baseline_args, g1_args)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gocube_topology", "torus"),
        ("gocube_size", 3),
        ("gocube_rule_set", "chinese"),
    ),
)
def test_wrong_shared_game_semantics_fail_closed(field, value):
    _game_cls, baseline_args = build_katago_training_args(
        parse_args(["--model-profile", "baseline", "--smoke"])
    )
    broken = baseline_args.copy()
    setattr(broken, field, value)
    with pytest.raises(ValueError):
        checkpoint_arena._require_same_contract(baseline_args, broken)


@pytest.mark.parametrize("profile", ("baseline", "g1"))
def test_same_profile_comparisons_remain_compatible(profile):
    _game_cls, args = build_katago_training_args(
        parse_args(["--model-profile", profile, "--smoke"])
    )
    checkpoint_arena._require_same_contract(args, args.copy())


def test_wrong_structural_schema_fails_closed():
    _game_cls, args = build_katago_training_args(
        parse_args(["--model-profile", "g1", "--smoke"])
    )
    broken = args.copy()
    broken.gocube_structural_feature_schema = "wrong-structural-schema"
    with pytest.raises(ValueError, match="invalid saved model contract|saved model contract mismatch"):
        checkpoint_arena._require_same_contract(args, broken)
