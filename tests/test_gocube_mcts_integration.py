import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

from alphazero.MCTS import MCTS
from alphazero.envs.gocube import Torus9ChineseGame
from alphazero.utils import dotdict


def mcts_args():
    return dotdict({
        "root_noise_frac": 0.0,
        "root_policy_temp": 1.0,
        "min_discount": 1.0,
        "fpu_reduction": 0.0,
        "cpuct": 1.25,
        # MCTS node terminal vectors include Black, White, and draw.
        "_num_players": 3,
    })


def test_cython_mcts_can_traverse_second_pass_and_backpropagate_terminal_result():
    game = Torus9ChineseGame()
    game.play_action(game.pass_action())
    assert not game.win_state().any()

    mcts = MCTS(mcts_args())

    # Expand the first-pass root and make PASS the only neural prior with mass.
    root_leaf = mcts.find_leaf(game)
    policy = np.zeros(game.action_size(), dtype=np.float32)
    policy[game.pass_action()] = 1.0
    value = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    mcts.process_results(root_leaf, value, policy, False, False)

    # The next traversal must choose PASS, which is the second pass. The GoCube
    # adapter adjudicates it synchronously, so MCTS sees a real terminal vector
    # rather than an impossible "no moves but nonterminal" state.
    terminal_leaf = mcts.find_leaf(game)
    assert terminal_leaf.turns == 2
    assert np.array_equal(
        terminal_leaf.win_state(),
        np.array([0, 1, 0], dtype=np.uint8),
    )
    assert not terminal_leaf.valid_moves().any()

    mcts.process_results(terminal_leaf, value, policy, False, False)

    counts = np.asarray(mcts.counts(game))
    assert counts[game.pass_action()] == 1
    probabilities = mcts.probs(game)
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[game.pass_action()] == 1.0


def test_root_noise_accepts_values_below_float32_range(monkeypatch):
    game = Torus9ChineseGame()
    mcts = MCTS(mcts_args())
    mcts.root_noise_frac = 0.1
    mcts.find_leaf(game)

    def subnormal_dirichlet(alpha):
        # Reproduce the production failure deterministically: the old code
        # cast these float64 subnormals to float32 while np.seterr(under='raise').
        noise = np.full(len(alpha), 5e-324, dtype=np.float64)
        noise[0] = 1.0
        return noise

    monkeypatch.setattr(np.random, "dirichlet", subnormal_dirichlet)

    # Must not raise FloatingPointError during the root-noise application.
    mcts._add_root_noise()

    priors = np.asarray([child.p for child in mcts._root._children], dtype=np.float64)
    assert np.isfinite(priors).all()
    assert priors.max() > 0
