from alphazero.envs.gocube.katago_v3 import (
    EMERGENCY_MOVE_CAP_BASE,
    EMERGENCY_MOVE_CAP_FACTOR,
    EPISODE_MOVE_LIMIT,
    RESULT_PROVENANCE_RUNTIME,
    v3_state_from_board,
)
from alphazero.envs.gocube.pinned_game import PinnedCube2JapaneseGame


def test_formal_transition_ignores_episode_move_cap():
    game_cls = PinnedCube2JapaneseGame
    topology = game_cls.logical_topology()
    move_cap = EMERGENCY_MOVE_CAP_BASE + EMERGENCY_MOVE_CAP_FACTOR * topology.point_count

    state = v3_state_from_board(
        topology,
        turns=move_cap - 1,
    )
    game = game_cls(state)

    # The move crosses the runtime budget in the formal counter. That counter
    # must not create a terminal in the rule transition or in a search clone.
    game.play_action(0)

    assert game.semantic_state.turns == move_cap
    assert game.terminal_kind is None
    assert game.semantic_state.no_result_reason is None


def test_runner_force_scores_at_episode_move_cap_with_provenance():
    game_cls = PinnedCube2JapaneseGame
    topology = game_cls.logical_topology()
    move_cap = EMERGENCY_MOVE_CAP_BASE + EMERGENCY_MOVE_CAP_FACTOR * topology.point_count
    game = game_cls()
    # The runner count is deliberately independent from formal/history turns.
    game._pinned_episode_move_count = move_cap

    assert game.finalize_episode_due_to_runtime_limit() is True
    assert game.terminal_kind == "scored"
    assert game.termination_reason == EPISODE_MOVE_LIMIT
    assert game.result_provenance == RESULT_PROVENANCE_RUNTIME
    assert game.has_training_result()
    assert game.terminal_adjudication is not None
    assert game.terminal_adjudication.score is not None

    score_target, ownership_target, ownership_mask = game.training_targets()
    assert score_target.shape == (1,)
    assert ownership_target.shape == (topology.point_count, 3)
    assert ownership_mask.shape == (topology.point_count,)
