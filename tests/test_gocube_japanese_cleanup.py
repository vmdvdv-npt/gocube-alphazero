import numpy as np

from alphazero.envs.gocube import (
    ENDGAME,
    JAPANESE_CLEANUP_ADJUDICATOR_V2,
    japanese_cleanup_adjudicate,
    normalized_score_target,
    ownership_target,
    state_from_point_ids,
    torus_topology,
)


def terminal_state(topology, *, black=(), white=(), captures=(0, 0), cleanup_stage=0):
    return state_from_point_ids(
        topology,
        black=black,
        white=white,
        current_player=0,
        turns=20,
        consecutive_passes=2,
        captures=captures,
        phase=ENDGAME,
        cleanup_stage=cleanup_stage,
    )


def test_unresolved_group_produces_no_result_not_fake_alive_reward():
    topology = torus_topology(9)
    scoring = terminal_state(topology, black=("0,0",))

    result = japanese_cleanup_adjudicate(
        scoring,
        scoring,
        topology,
        ruleset="japanese",
        komi=0.5,
    )

    assert result.adjudicator_id == JAPANESE_CLEANUP_ADJUDICATOR_V2
    assert result.no_result
    assert not result.training_valid
    assert result.score is None
    assert result.unresolved_count == 1
    assert result.classification[0].status == "unresolved"


def test_cleanup_capture_resolves_original_group_as_dead():
    topology = torus_topology(9)
    scoring = terminal_state(topology, black=("0,0",), captures=(2, 3))
    cleanup = terminal_state(
        topology,
        white=("8,0", "1,0", "0,8", "0,1"),
        captures=(2, 4),
        cleanup_stage=1,
    )

    result = japanese_cleanup_adjudicate(
        scoring,
        cleanup,
        topology,
        ruleset="japanese",
        komi=0.5,
    )

    assert result.training_valid
    assert not result.no_result
    assert result.classification[0].status == "dead"
    assert result.classification[0].source == "cleanup-captured"
    assert result.score.ruleset == "japanese"
    assert result.score.dead_stones.black == 1


def test_cleanup_moves_do_not_change_frozen_japanese_score_or_prisoners():
    topology = torus_topology(9)
    scoring = terminal_state(topology, black=("0,0",), captures=(0, 0))
    cleanup = terminal_state(
        topology,
        white=("8,0", "1,0", "0,8", "0,1", "4,4", "5,5"),
        captures=(0, 12),
        cleanup_stage=2,
    )

    result = japanese_cleanup_adjudicate(
        scoring,
        cleanup,
        topology,
        ruleset="japanese",
        komi=0.0,
    )

    assert result.training_valid
    assert result.score.captures == (0, 0)
    assert result.score.prisoners == (0, 1)
    assert result.score.white == 1.0
    assert result.score.black == 0.0


def test_auxiliary_targets_are_finite_and_topology_aligned():
    topology = torus_topology(9)
    scoring = terminal_state(topology, black=("0,0",))
    cleanup = terminal_state(
        topology,
        white=("8,0", "1,0", "0,8", "0,1"),
        cleanup_stage=1,
    )
    result = japanese_cleanup_adjudicate(
        scoring,
        cleanup,
        topology,
        ruleset="japanese",
        komi=0.5,
    )

    score = normalized_score_target(result, topology)
    ownership = ownership_target(result, scoring, topology)

    assert score.shape == (1,)
    assert np.isfinite(score).all()
    assert -1.0 <= score[0] <= 1.0
    assert ownership.shape == (81, 3)
    assert np.allclose(ownership.sum(axis=1), 1.0)
