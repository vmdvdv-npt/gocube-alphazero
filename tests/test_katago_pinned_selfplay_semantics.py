from types import SimpleNamespace

import numpy as np

import alphazero.envs.gocube.pinned_game as pinned_game_module
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.envs.gocube.pinned_game import PinnedCube4JapaneseGame
from alphazero.envs.gocube.selfplay_semantics import KATAGO_PINNED_SELFPLAY_DEFAULTS


def test_remaining_pinned_selfplay_defaults_match_katago():
    defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
    assert defaults["pass_alive_auto_end_probability"] == 0.98
    assert defaults["root_prune_useless_moves"] is True
    assert defaults["seki_fork_hack_probability"] == 0.02

    _, args = build_katago_training_args(parse_args([]))
    assert args.gocube_pass_alive_auto_end_probability == 0.98
    assert args.gocube_root_prune_useless_moves is True
    assert args.gocube_seki_fork_hack_probability == 0.02


def test_game_level_pass_alive_auto_end_is_not_used_inside_search_clone(monkeypatch):
    calls = []

    def fake_auto_end(state, topology):
        calls.append((state.turns, topology.point_count))
        return state

    monkeypatch.setattr(pinned_game_module, "maybe_pass_alive_early_terminal", fake_auto_end)

    game = PinnedCube4JapaneseGame()
    game.configure_pinned_selfplay(
        auto_end_pass_alive=True,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    game.play_action(0)
    assert len(calls) == 1

    search_clone = game.clone()
    action = int(np.flatnonzero(search_clone.valid_moves())[0])
    search_clone.play_action(action)
    assert len(calls) == 1

    manual_cleanup_game = PinnedCube4JapaneseGame()
    manual_cleanup_game.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    manual_cleanup_game.play_action(0)
    assert len(calls) == 1


def test_root_prune_useless_moves_is_root_only_and_uses_four_opponent_passes(monkeypatch):
    game = PinnedCube4JapaneseGame()
    game.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.0,
    )
    pass_action = game.pass_action()
    # Root player is Black. The last four White moves (offsets 0,2,4,6)
    # are all PASS, exactly matching KataGo's root condition.
    game._pinned_move_history = (
        (1, pass_action),
        (0, 10),
        (1, pass_action),
        (0, 11),
        (1, pass_action),
        (0, 12),
        (1, pass_action),
    )

    fake_analysis = SimpleNamespace(
        pass_alive_black_territory=(0,),
        pass_alive_white_territory=(1,),
        pass_alive_black_groups=((2,),),
        pass_alive_white_groups=((3,),),
    )
    monkeypatch.setattr(pinned_game_module, "pass_alive_analysis", lambda board, topology: fake_analysis)

    root = game.clone()
    valids = root.valid_moves()
    assert all(valids[p] == 0 for p in (0, 1, 2, 3))
    assert valids[pass_action] == 1

    root._pinned_at_search_root = False
    deeper_valids = root.valid_moves()
    assert all(deeper_valids[p] == 1 for p in (0, 1, 2, 3))


def test_seki_fork_pool_consumes_a_late_position_once():
    source = PinnedCube4JapaneseGame()
    source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.02,
    )
    source.play_action(0)
    candidate_state = source.semantic_state
    candidate_history = source._pinned_move_history

    pool = PinnedCube4JapaneseGame._seki_pool()
    pool.clear()
    pool.append((candidate_state, candidate_history))

    target = PinnedCube4JapaneseGame()
    target.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=1.0,
    )
    assert target.maybe_start_seki_fork(1.0) is True
    assert target.semantic_state == candidate_state
    assert target._pinned_move_history == candidate_history
    assert target.pinned_selfplay_config()["started_from_seki_fork"] is True
    assert pool == []
