from argparse import Namespace

import pytest

from alphazero.envs.gocube.validate_recovery import (
    DEFAULT_CHECKPOINT,
    DEFAULT_RUN_NAME,
    DEFAULT_TRAINING_RUN_NAME,
    EXPECTED_KOMI,
    build_recovery_args,
    build_training_fork_args,
)
from alphazero.utils import dotdict


def _cli(**overrides):
    values = dict(
        checkpoint=DEFAULT_CHECKPOINT,
        topology="cube",
        size=4,
        games=256,
        workers=16,
        sims=50,
        fast_sims=20,
        fast_prob=0.75,
        inference_batch_wait_ms=1.0,
        run_name=DEFAULT_RUN_NAME,
        output_root="diagnostics",
        record_games=False,
        device="cpu",
        prepare_training_run=False,
        training_run_name=DEFAULT_TRAINING_RUN_NAME,
    )
    values.update(overrides)
    return Namespace(**values)


def test_recovery_defaults_match_archived_cube4_experiment():
    game_cls, args = build_recovery_args(_cli())
    assert game_cls.topology_kind() == "cube"
    assert game_cls.board_size() == 4
    assert game_cls.KOMI == pytest.approx(EXPECTED_KOMI)
    assert args.gocube_komi == pytest.approx(0.5)
    assert args.gamesPerIteration == 256
    assert args.workers == 16
    assert args.process_batch_size == 16
    assert args.numMCTSSims == 50
    assert args.numFastSims == 20
    assert args.probFastSim == pytest.approx(0.75)
    assert args.gocube_search_audit_probability == pytest.approx(1.0)
    assert args.gocube_validation_only is True
    assert args.gocube_recording_enabled is False
    assert args.load_model is False
    assert args.compareWithBaseline is False
    assert args.compareWithPast is False
    assert args.model_gating is False


def test_recovery_rejects_invalid_fast_probability():
    with pytest.raises(ValueError, match="fast-prob"):
        build_recovery_args(_cli(fast_prob=1.1))


def test_training_fork_preserves_architecture_but_adopts_current_contract_and_run_metadata():
    _, current = build_recovery_args(_cli())
    saved = dotdict({
        "nnet_type": "graph",
        "num_channels": 48,
        "depth": 5,
        "value_dense_layers": [96, 48],
        "policy_dense_layers": [96],
        "score_dense_layers": [48],
        "optimizer": current.optimizer,
        "optimizer_args": current.optimizer_args,
        "scheduler": current.scheduler,
        "scheduler_args": current.scheduler_args,
        "lr": 0.003,
        "value_loss_weight": 1.25,
        "ownership_loss_weight": 0.4,
        "score_loss_weight": 0.6,
        "gocube_search_contract": None,
        "search_utility_mode": "legacy",
        "run_name": "archived-source",
    })

    fork = build_training_fork_args(saved, current, "clean-recovery-run")

    assert fork.nnet_type == "graph"
    assert fork.num_channels == 48
    assert fork.depth == 5
    assert fork.value_dense_layers == [96, 48]
    assert fork.policy_dense_layers == [96]
    assert fork.score_dense_layers == [48]
    assert fork.lr == pytest.approx(0.003)
    assert fork.gocube_search_contract == current.gocube_search_contract
    assert fork.search_utility_mode == current.search_utility_mode
    assert fork.gocube_komi == pytest.approx(0.5)
    assert fork.run_name == "clean-recovery-run"
    assert fork.checkpoint == "checkpoint"
    assert fork.data == "data"
    assert fork.load_model is True
    assert fork.gocube_validation_only is False
    assert fork.gocube_recording_enabled is True
    assert fork.gocube_record_root == "data/clean-recovery-run/records"
