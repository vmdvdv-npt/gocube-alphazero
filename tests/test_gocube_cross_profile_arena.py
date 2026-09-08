"""Cross-profile GoCube Arena keeps one semantic state and per-model inputs."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from alphazero.GenericPlayers import MCTSPlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.diversified_game import (
    diversified_baseline_pinned_game_class,
    diversified_pinned_game_class,
    diversified_structural_pinned_game_class,
)
from alphazero.envs.gocube.evaluation import prepare_evaluation_args
from alphazero.envs.gocube.game import Cube4JapaneseGame
from alphazero.envs.gocube.integration.catalog import CheckpointCatalog
from alphazero.envs.gocube.integration.contract import (
    resolve_model_contract,
)
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.integration.models import CheckpointModelLoader
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.envs.gocube.observation import GoCubeObservationAdapter
from alphazero.envs.gocube.reproducible_manifest import create_reproducible_manifest
from alphazero.envs.gocube.structural import structural_feature_matrix
from alphazero.search_contract import SearchOutput
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
    assert 4 in changed
    assert np.all(changed < 18)


@pytest.fixture(scope="module")
def real_cross_profile_checkpoints(tmp_path_factory):
    """Create and load real B0/B1 checkpoints for the Cython Arena smoke."""

    root = tmp_path_factory.mktemp("gocube-cross-profile-arena")
    checkpoint_ids = {}
    for profile in ("baseline", "g1"):
        model_game_cls, args = build_katago_training_args(
            parse_args(["--model-profile", profile, "--smoke"])
        )
        args = args.copy()
        args.cuda = False
        args.checkpoint = str(root)
        args.run_name = f"real-{profile}"
        run_dir = root / args.run_name
        run_dir.mkdir()

        model = NNetWrapper(model_game_cls, args)
        model.save_checkpoint(str(run_dir), "iteration-0000.pkl")
        create_reproducible_manifest(
            checkpoint_dir=root,
            run_name=args.run_name,
            game_cls=model_game_cls,
            args=args,
            argv=["cross-profile-arena-test", profile],
            allow_dirty_source=True,
        )
        ensure_training_manifest(str(root), args.run_name, model_game_cls, args)
        checkpoint_ids[profile] = f"{args.run_name}@0"

    catalog = CheckpointCatalog(str(root))
    loader = CheckpointModelLoader(catalog, device="cpu")
    descriptors = {
        profile: catalog.get(checkpoint_id)
        for profile, checkpoint_id in checkpoint_ids.items()
    }
    assert all(descriptor is not None for descriptor in descriptors.values())
    loaded = {
        profile: loader.load(descriptor.checkpoint_id)[1]
        for profile, descriptor in descriptors.items()
    }
    return descriptors, loaded


def test_real_b0_vs_b1_arena_uses_loaded_checkpoints_and_model_shapes(
    real_cross_profile_checkpoints,
):
    descriptors, loaded = real_cross_profile_checkpoints
    baseline = loaded["baseline"]
    g1 = loaded["g1"]

    assert baseline.game_cls is BASELINE
    assert g1.game_cls is G1
    assert tuple(baseline.game_cls.observation_size()) == (18, 96, 1)
    assert tuple(g1.game_cls.observation_size()) == (20, 96, 1)

    semantic_game_cls = checkpoint_arena._authoritative_game_class(
        resolve_model_contract(baseline.game_cls, baseline.args)
    )
    assert semantic_game_cls is SEMANTIC

    observed_shapes = {"baseline": [], "g1": []}
    for profile, model, expected_shape in (
        ("baseline", baseline, (18, 96, 1)),
        ("g1", g1, (20, 96, 1)),
    ):
        original_process_for_search = model.process_for_search

        def record_real_inference(
            batch,
            *,
            _profile=profile,
            _expected_shape=expected_shape,
            _original=original_process_for_search,
        ):
            shape = tuple(int(value) for value in batch.shape)
            observed_shapes[_profile].append(shape)
            assert shape[1:] == _expected_shape
            output = _original(batch)
            assert isinstance(output, SearchOutput)
            assert output.score is not None
            assert output.ownership is not None

            # Keep the checkpoint forward pass real while making this smoke
            # finish after the two-pass terminal.  The assertion above still
            # observes the tensor that reached the loaded network.
            policy = torch.zeros_like(output.policy)
            policy[:, semantic_game_cls.pass_action()] = 1.0
            return SearchOutput(
                policy=policy,
                value=output.value,
                score=output.score,
                ownership=output.ownership,
            )

        model.process_for_search = record_real_inference

    eval_args = prepare_evaluation_args(
        baseline.args,
        semantic_game_cls,
        sims=1,
    )
    eval_args.cuda = False
    eval_args.workers = 2
    eval_args.arena_batch_size = 1
    eval_args.use_draws_for_winrate = True
    eval_args.gocube_arena_seed = 20260908

    players = [
        MCTSPlayer(
            baseline,
            game_cls=semantic_game_cls,
            args=eval_args.copy(),
            observation_adapter=GoCubeObservationAdapter(baseline.game_cls),
        ),
        MCTSPlayer(
            g1,
            game_cls=semantic_game_cls,
            args=eval_args.copy(),
            observation_adapter=GoCubeObservationAdapter(g1.game_cls),
        ),
    ]

    summary = checkpoint_arena._coalesced_batched_summary(
        players,
        semantic_game_cls,
        eval_args,
        games=2,
        seed=20260908,
        wait_ms=0.1,
    )

    assert summary["games"] == 2
    assert summary["by_color"]["black"]["games"] == 1
    assert summary["by_color"]["white"]["games"] == 1
    assert observed_shapes["baseline"]
    assert observed_shapes["g1"]
    assert all(shape == (1, 18, 96, 1) for shape in observed_shapes["baseline"])
    assert all(shape == (1, 20, 96, 1) for shape in observed_shapes["g1"])
    assert descriptors["baseline"].model_contract["observationShape"] == [18, 96, 1]
    assert descriptors["g1"].model_contract["observationShape"] == [20, 96, 1]


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
