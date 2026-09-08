from dataclasses import replace
from queue import Queue
from threading import Event

import numpy as np
import torch

from alphazero.MCTS import MCTS
from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.envs.gocube import CLEANUP_1, CLEANUP_2, Cube4JapaneseGame, NO_RESULT, SCORED
from alphazero.envs.gocube.diversified_game import diversified_structural_pinned_game_class
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.envs.gocube.katago_v3 import _state_key, apply_v3_action, initial_v3_state
from alphazero.search_contract import SearchOutput


GAME = diversified_structural_pinned_game_class(Cube4JapaneseGame)
SEED = 20260907
SIMS = 2
ATOL = 1.0e-6


def _search_args():
    game_cls, args = build_katago_training_args(parse_args(["--model-profile", "g1"]))
    assert game_cls is GAME
    assert float(args.gocube_komi) == 0.5
    args._num_players = GAME.num_players() + GAME.has_draw()
    args.add_root_noise = False
    args.add_root_temp = False
    args.gocube_cleanup_training_prob = 0.0
    return args


class _DeterministicSearchNet:
    def __init__(self, game_cls, preferred_action):
        self.action_size = int(game_cls.action_size())
        self.point_count = int(game_cls.logical_topology().point_count)
        self.preferred_action = int(preferred_action)

    def _numpy_output(self):
        policy = np.zeros(self.action_size, dtype=np.float32)
        policy[self.preferred_action] = 1.0
        return SearchOutput(
            policy=policy,
            value=np.array([0.20, 0.70, 0.10], dtype=np.float32),
            score=np.array([0.125], dtype=np.float32),
            ownership=np.tile(
                np.array([[0.20, 0.30, 0.50]], dtype=np.float32),
                (self.point_count, 1),
            ),
        )

    def predict_for_search(self, observation):
        assert tuple(observation.shape) == tuple(GAME.observation_size())
        return self._numpy_output()

    def process_for_search(self, batch):
        rows = int(batch.shape[0])
        out = self._numpy_output()
        return SearchOutput(
            policy=torch.as_tensor(out.policy).view(1, -1).repeat(rows, 1),
            value=torch.as_tensor(out.value).view(1, -1).repeat(rows, 1),
            score=torch.as_tensor(out.score).view(1, -1).repeat(rows, 1),
            ownership=torch.as_tensor(out.ownership).view(1, self.point_count, 3).repeat(rows, 1, 1),
        )


class _PlayerRelativeWhiteWinNet:
    """Return a White-win target in the V3 player-to-move value encoding."""

    def __init__(self, game_cls, preferred_action):
        self.action_size = int(game_cls.action_size())
        self.point_count = int(game_cls.logical_topology().point_count)
        self.preferred_action = int(preferred_action)

    def _policy(self):
        policy = np.zeros(self.action_size, dtype=np.float32)
        policy[self.preferred_action] = 1.0
        return policy

    def _numpy_output(self, player):
        # V3 value is WIN/LOSS for the player to move. A White win is therefore
        # WIN when White moves and LOSS when Black moves.
        value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if int(player) == 0:
            value = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return SearchOutput(
            policy=self._policy(),
            value=value,
            score=np.array([0.0], dtype=np.float32),
            ownership=np.tile(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                (self.point_count, 1),
            ),
        )

    def predict_for_search(self, observation):
        player = 1 if float(observation[4, 0, 0]) < 0.0 else 0
        return self._numpy_output(player)

    def process_for_search(self, batch):
        rows = int(batch.shape[0])
        values = torch.zeros(rows, 3, dtype=torch.float32)
        white_to_move = batch[:, 4, 0, 0] < 0.0
        values[white_to_move, 0] = 1.0
        values[~white_to_move, 1] = 1.0
        return SearchOutput(
            policy=torch.as_tensor(self._policy()).view(1, -1).repeat(rows, 1),
            value=values,
            score=torch.zeros(rows, 1, dtype=torch.float32),
            ownership=torch.tensor(
                np.tile(
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    (self.point_count, 1),
                )
            ).view(1, self.point_count, 3).repeat(rows, 1, 1),
        )


def _new_agent(args, game, *, arena=False, batch_size=1):
    action_size = int(GAME.action_size())
    point_count = int(GAME.logical_topology().point_count)
    policy = torch.zeros(batch_size, action_size, dtype=torch.float32)
    value = torch.zeros(batch_size, GAME.num_players() + 1, dtype=torch.float32)
    score = torch.zeros(batch_size, 1, dtype=torch.float32)
    ownership = torch.zeros(batch_size, point_count, 3, dtype=torch.float32)
    batch_tensor = None if arena else torch.zeros(
        batch_size, *GAME.observation_size(), dtype=torch.float32
    )
    agent = SelfPlayAgent(
        0,
        GAME,
        Queue(),
        Event(),
        batch_tensor,
        policy,
        value,
        Queue(),
        Queue(),
        None,
        None,
        Event(),
        Event(),
        args,
        _is_arena=arena,
        score_tensor=score,
        ownership_tensor=ownership,
    )
    if game is not None:
        agent.games[0] = GAME(game.semantic_state)
        agent.mcts[0] = agent._get_mcts()
    return agent


def _run_single(game, net, args):
    np.random.seed(SEED)
    mcts = MCTS(args)
    mcts.search(game, net, SIMS, False, False)
    return mcts


def _run_batched(game, net, args):
    np.random.seed(SEED)
    agent = _new_agent(args, game)
    leaves = []
    for _ in range(SIMS):
        agent.generateBatch()
        leaf = agent.search_states[0]
        assert leaf is not None
        leaves.append(leaf)

        out = net.process_for_search(agent.batch_tensor)
        agent.policy_tensor.copy_(out.policy)
        agent.value_tensor.copy_(out.value)
        agent.score_tensor.copy_(out.score)
        agent.ownership_tensor.copy_(out.ownership)
        agent.batch_ready.set()
        agent.processBatch()
        assert agent.search_states[0] is None
    return agent.mcts[0], leaves


def _assert_searches_match(single, batched, game):
    np.testing.assert_array_equal(
        np.asarray(single.raw_counts(game)),
        np.asarray(batched.raw_counts(game)),
    )
    np.testing.assert_allclose(single._root.q, batched._root.q, rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(single._root.v, batched._root.v, rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(single._root.score_q, batched._root.score_q, rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(single._root.utility_sq, batched._root.utility_sq, rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(single._root.weight_sum, batched._root.weight_sum, rtol=0.0, atol=ATOL)

    single_telemetry = single.root_search_telemetry(game)
    batched_telemetry = batched.root_search_telemetry(game)
    np.testing.assert_allclose(
        single_telemetry["policy_training_target"],
        batched_telemetry["policy_training_target"],
        rtol=0.0,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        single.probs(game, 1.0),
        batched.probs(game, 1.0),
        rtol=0.0,
        atol=ATOL,
    )


def _after_passes(count):
    topology = GAME.logical_topology()
    state = initial_v3_state(topology)
    for _ in range(int(count)):
        state = apply_v3_action(state, topology.pass_action, topology)
    return state


def test_batched_second_pass_matches_iter7_empty_cube_reproduction():
    """ITER7 reproduction: empty Cube4, first PASS played, search chooses PASS again."""
    args = _search_args()
    state = _after_passes(1)
    assert state.phase == "main"
    assert state.turns == 1
    assert state.consecutive_passes == 1
    game = GAME(state)
    net = _DeterministicSearchNet(GAME, game.pass_action())

    single = _run_single(game, net, args)
    batched, leaves = _run_batched(game, net, args)

    second_pass_leaf = leaves[-1]
    assert second_pass_leaf.last_action == game.pass_action()
    assert second_pass_leaf.semantic_state.turns == 2
    assert second_pass_leaf.semantic_state.phase == CLEANUP_1
    assert second_pass_leaf.terminal_kind is None
    _assert_searches_match(single, batched, game)


def test_batched_scored_terminal_uses_leaf_exact_score_and_matches_single():
    args = _search_args()
    state = _after_passes(5)
    assert state.phase == CLEANUP_2
    assert state.consecutive_passes == 1
    assert state.terminal_kind is None
    game = GAME(state)
    net = _DeterministicSearchNet(GAME, game.pass_action())

    single = _run_single(game, net, args)
    batched, leaves = _run_batched(game, net, args)

    terminal_leaf = leaves[-1]
    assert terminal_leaf.last_action == game.pass_action()
    assert terminal_leaf.terminal_kind == SCORED
    assert terminal_leaf.terminal_adjudication is not None
    assert terminal_leaf.terminal_adjudication.score is not None
    score = terminal_leaf.terminal_adjudication.score
    assert np.isclose(float(score.white) - float(score.black), 0.5)
    _assert_searches_match(single, batched, game)


def test_batched_no_result_leaf_matches_single_search():
    args = _search_args()
    topology = GAME.logical_topology()
    base = initial_v3_state(topology)
    action = 0
    candidate = apply_v3_action(base, action, topology)
    repeated_key = _state_key(
        candidate.board,
        candidate.current_player,
        candidate.ko_recap_blocked,
    )
    state = replace(base, history_since_pass=(repeated_key, repeated_key))
    game = GAME(state)
    net = _DeterministicSearchNet(GAME, action)

    single = _run_single(game, net, args)
    batched, leaves = _run_batched(game, net, args)

    no_result_leaf = leaves[-1]
    assert no_result_leaf.terminal_kind == NO_RESULT
    assert no_result_leaf.semantic_state.no_result_reason == "cycle"
    assert no_result_leaf.terminal_adjudication is not None
    assert no_result_leaf.terminal_adjudication.score is None
    assert np.array_equal(no_result_leaf.win_state(), np.array([0, 0, 1], dtype=np.uint8))
    _assert_searches_match(single, batched, game)


def test_batched_cleanup_phase_transition_matches_single_search():
    args = _search_args()
    state = _after_passes(3)
    assert state.phase == CLEANUP_1
    assert state.consecutive_passes == 1
    game = GAME(state)
    net = _DeterministicSearchNet(GAME, game.pass_action())

    single = _run_single(game, net, args)
    batched, leaves = _run_batched(game, net, args)

    cleanup_leaf = leaves[-1]
    assert cleanup_leaf.last_action == game.pass_action()
    assert cleanup_leaf.semantic_state.phase == CLEANUP_2
    assert cleanup_leaf.terminal_kind is None
    _assert_searches_match(single, batched, game)


def test_batched_white_to_move_value_conversion_matches_single_search():
    args = _search_args()
    args.gocube_win_loss_utility_factor = 1.0
    args.gocube_static_score_utility_factor = 0.0
    args.gocube_dynamic_score_utility_factor = 0.0
    topology = GAME.logical_topology()
    state = replace(initial_v3_state(topology), current_player=1)
    game = GAME(state)
    net = _PlayerRelativeWhiteWinNet(GAME, 0)

    single = _run_single(game, net, args)
    batched, leaves = _run_batched(game, net, args)

    assert leaves[0].player == 1
    assert np.isclose(single._root.q, 1.0)
    assert np.isclose(batched._root.q, 1.0)
    _assert_searches_match(single, batched, game)


class _RecordingMCTS:
    def __init__(self):
        self.received = None

    def find_leaf(self, game):
        return game.clone()

    def search_observation(self, state):
        return state.observation()

    def process_search_results(self, state, value, policy, score, ownership, add_root_noise, add_root_temp):
        self.received = {
            "state": state,
            "value": np.array(value, copy=True),
            "policy": np.array(policy, copy=True),
            "score": np.array(score, copy=True),
            "ownership": np.array(ownership, copy=True),
        }


def test_arena_batch_redistribution_keeps_each_leaf_with_its_network_row():
    args = _search_args()
    batch_size = 6
    agent = _new_agent(args, None, arena=True, batch_size=batch_size)
    agent.player_to_index = [0, 1]

    player_by_slot = [1, 1, 0, 0, 1, 0]
    topology = GAME.logical_topology()
    agent.games = [
        GAME(replace(initial_v3_state(topology), current_player=player))
        for player in player_by_slot
    ]
    recorders = [_RecordingMCTS() for _ in range(batch_size)]
    agent.mcts = [(recorder, recorder) for recorder in recorders]

    agent.generateBatch()
    generated_leaves = list(agent.search_states)

    # Parent Arena concatenates network outputs by player index. For the slot
    # pattern above, rows are [2,3,5] for player 0 and [0,1,4] for player 1.
    assert agent.batch_indices == [3, 4, 0, 1, 5, 2]

    for row in range(batch_size):
        agent.policy_tensor[row].zero_()
        agent.policy_tensor[row, 0] = 100.0 + row
        agent.value_tensor[row] = torch.tensor(
            [200.0 + row, 300.0 + row, 400.0 + row], dtype=torch.float32
        )
        agent.score_tensor[row, 0] = 500.0 + row
        agent.ownership_tensor[row].zero_()
        agent.ownership_tensor[row, :, 0] = 600.0 + row

    agent.batch_ready.set()
    agent.processBatch()

    for slot, recorder in enumerate(recorders):
        row = agent.batch_indices[slot]
        assert recorder.received is not None
        assert recorder.received["state"] is generated_leaves[slot]
        assert recorder.received["value"][0] == 200.0 + row
        assert recorder.received["policy"][0] == 100.0 + row
        assert recorder.received["score"][0] == 500.0 + row
        assert recorder.received["ownership"][0, 0] == 600.0 + row
    assert agent.search_states == [None] * batch_size
