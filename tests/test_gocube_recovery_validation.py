from argparse import Namespace

import pytest

from alphazero.envs.gocube.validate_recovery import (
    DEFAULT_CHECKPOINT,
    DEFAULT_RUN_NAME,
    EXPECTED_KOMI,
    build_recovery_args,
)


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
