from alphazero.envs.gocube.katago_v3 import (
    EMERGENCY_MOVE_CAP_BASE,
    EMERGENCY_MOVE_CAP_FACTOR,
    SCORED,
    v3_state_from_board,
)
from alphazero.envs.gocube.pinned_game import PinnedCube2JapaneseGame


def test_pinned_training_force_scores_current_board_at_move_cap():
    game_cls = PinnedCube2JapaneseGame
    topology = game_cls.logical_topology()
    move_cap = EMERGENCY_MOVE_CAP_BASE + EMERGENCY_MOVE_CAP_FACTOR * topology.point_count

    state = v3_state_from_board(
        topology,
        turns=move_cap - 1,
    )
    game = game_cls(state)

    # The move crosses GoCube's emergency move cap. Pinned self-play mirrors
    # KataGo GameRunner::endAndScoreGameNow(): current board is force-scored
    # and remains valid training data rather than becoming NO_RESULT.
    game.play_action(0)

    assert game.semantic_state.turns == move_cap
    assert game.terminal_kind == SCORED
    assert game.semantic_state.no_result_reason is None
    assert game.has_training_result()
    assert game.terminal_adjudication is not None
    assert game.terminal_adjudication.score is not None

    score_target, ownership_target, ownership_mask = game.training_targets()
    assert score_target.shape == (1,)
    assert ownership_target.shape == (topology.point_count, 3)
    assert ownership_mask.shape == (topology.point_count,)
