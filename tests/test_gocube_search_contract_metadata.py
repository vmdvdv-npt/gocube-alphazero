from argparse import Namespace

import pytest

from alphazero.NNetWrapper import NNetWrapper, _SEARCH_CONTRACT_ARG_KEYS
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest, load_run_manifest
from alphazero.envs.gocube.train import build_training_args
from alphazero.search_contract import GOCUBE_SEARCH_CONTRACT


def _cli(**overrides):
    values = dict(
        topology="cube",
        size=4,
        workers=1,
        sims=20,
        arena_sims=20,
        games_per_iteration=8,
        iterations=2,
        train_batch_size=64,
        train_steps_per_iteration=None,
        fast_game_prob=0.0,
        endgame_sample_weight=1,
        inference_batch_wait_ms=0.0,
        no_arena=True,
        model_gating=False,
        smoke=False,
        run_name="contract-test-run",
    )
    values.update(overrides)
    return Namespace(**values)


def _wrapper_with_runtime_contract(game_cls, args):
    wrapper = object.__new__(NNetWrapper)
    wrapper.game_cls = game_cls
    wrapper.args = args
    return wrapper


def test_run_manifest_round_trips_exact_search_contract_and_settings(tmp_path):
    game_cls, args = build_training_args(_cli())
    ensure_training_manifest(str(tmp_path), args.run_name, game_cls, args)
    manifest = load_run_manifest(str(tmp_path / args.run_name))

    assert manifest.search_contract == GOCUBE_SEARCH_CONTRACT
    assert manifest.search_settings is not None
    assert manifest.search_settings["search_utility_mode"] == args.search_utility_mode
    assert manifest.search_settings["gocube_dynamic_score_utility_factor"] == pytest.approx(
        args.gocube_dynamic_score_utility_factor
    )
    assert manifest.search_settings["gocube_root_ending_bonus_points"] == pytest.approx(
        args.gocube_root_ending_bonus_points
    )
    assert manifest.search_settings["gocube_cleanup2_weight"] == args.gocube_cleanup2_weight


def test_initialized_wrapper_accepts_exact_saved_search_contract():
    game_cls, args = build_training_args(_cli())
    wrapper = _wrapper_with_runtime_contract(game_cls, args)
    wrapper._validate_saved_contract(args)


def test_initialized_wrapper_rejects_changed_search_semantics():
    game_cls, args = build_training_args(_cli())
    wrapper = _wrapper_with_runtime_contract(game_cls, args)
    saved = args.copy()
    saved.gocube_dynamic_score_utility_factor = args.gocube_dynamic_score_utility_factor + 0.1

    with pytest.raises(ValueError, match="gocube_dynamic_score_utility_factor"):
        wrapper._validate_saved_contract(saved)


def test_new_runtime_rejects_legacy_v3_checkpoint_without_search_contract_by_default():
    game_cls, args = build_training_args(_cli())
    wrapper = _wrapper_with_runtime_contract(game_cls, args)
    saved = args.copy()
    for key in _SEARCH_CONTRACT_ARG_KEYS:
        saved.pop(key, None)

    with pytest.raises(ValueError, match="missing required GoCube V3 metadata"):
        wrapper._validate_saved_contract(saved)

    wrapper._validate_saved_contract(saved, allow_legacy_search_contract=True)


def test_standalone_checkpoint_reader_can_validate_historical_v3_metadata_before_loading_saved_args():
    game_cls, args = build_training_args(_cli())
    standalone = object.__new__(NNetWrapper)
    standalone.game_cls = game_cls

    # from_checkpoint() has no runtime args yet. It must still enforce the
    # immutable rules/topology contract, while allowing the checkpoint's own
    # saved search settings to become the evaluation settings afterwards.
    standalone._validate_saved_contract(args)
