"""Small, executable proofs for the GoCube learning pipeline.

These tests deliberately avoid a production-sized run.  They pin the cheap
experiments that distinguish a broken trainer from a broken search/evaluator:
the real production GraphNet must overfit a tiny tensor set, a saved model
must round-trip exactly, and the Arena must separate deterministic good/bad
players while balancing colors.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from alphazero.Arena import Arena
from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.diversified_game import (
    diversified_baseline_pinned_game_class,
    diversified_structural_pinned_game_class,
)
from alphazero.envs.gocube.game import Cube4JapaneseGame
from alphazero.envs.gocube.katago_train import build_katago_training_args, parse_args
from alphazero.envs.gocube.observation import GoCubeObservationAdapter
from alphazero.envs.gocube.structural import structural_feature_matrix
from alphazero.envs.gocube.katago_v3 import initial_v3_state
from alphazero.envs.gocube.integration.contract import resolve_model_contract
from alphazero.utils import dotdict


BASELINE = diversified_baseline_pinned_game_class(Cube4JapaneseGame)
G1 = diversified_structural_pinned_game_class(Cube4JapaneseGame)


def _production_args(profile: str = "baseline"):
    game_cls, args = build_katago_training_args(
        parse_args(["--model-profile", profile, "--smoke"])
    )
    return game_cls, args.copy()


def _tiny_network_args(profile: str = "baseline"):
    game_cls, args = _production_args(profile)
    # Keep the real production GraphNet and NNetWrapper code paths, but make
    # the diagnostic cheap enough for CI and deterministic on CPU.
    args.cuda = False
    args.num_channels = 16
    args.depth = 2
    args.value_dense_layers = [32]
    args.score_dense_layers = [16]
    args.gocube_auxiliary_targets = False
    args.optimizer = torch.optim.Adam
    args.optimizer_args = {}
    args.scheduler = torch.optim.lr_scheduler.StepLR
    args.scheduler_args = {"step_size": 10_000, "gamma": 1.0}
    args.lr = 0.05
    return game_cls, args


def _state_digest(network: NNetWrapper) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(network.nnet.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_production_contract_has_one_explicit_v3_semantic_path():
    for profile, expected_shape, expected_schema in (
        ("baseline", (18, 96, 1), "gocube-observation-v4-pass-would-end-phase"),
        ("g1", (20, 96, 1), "gocube-observation-v5-structural-features"),
    ):
        game_cls, args = _production_args(profile)
        contract = resolve_model_contract(game_cls, args).to_checkpoint_fields()

        assert game_cls.GOCUBE_V3 is True
        assert game_cls.KOMI == 0.5
        assert args.gocube_komi == 0.5
        assert args.search_utility_mode == "katago-pinned-f6bc4b19"
        assert args.gocube_katago_search_contract == "katago-pinned-search-v4"
        assert args.probFastSim == 0.0  # smoke never silently uses fast search
        assert game_cls.action_size() == 97
        assert game_cls.observation_size() == expected_shape
        assert game_cls.OBSERVATION_SCHEMA == expected_schema
        model_contract = contract["gocube_model_contract"]
        assert model_contract["observationShape"] == list(expected_shape)
        assert model_contract["actionSize"] == 97
        assert contract["gocube_targets_schema"]["value"] == "win-loss-noresult-s1-v2"
        assert contract["gocube_output_heads"] == 4


def test_observation_is_semantically_identical_between_game_and_arena_adapters():
    state = BASELINE(
        replace(initial_v3_state(Cube4JapaneseGame.logical_topology()), current_player=1)
    )
    semantic_state = state.semantic_state

    baseline = GoCubeObservationAdapter(BASELINE).observation(state)
    g1 = GoCubeObservationAdapter(G1).observation(state)

    assert baseline.shape == (18, 96, 1)
    assert g1.shape == (20, 96, 1)
    np.testing.assert_array_equal(baseline, state.observation())
    np.testing.assert_array_equal(g1[:18], baseline)
    np.testing.assert_array_equal(
        g1[-2:], structural_feature_matrix(Cube4JapaneseGame.logical_topology())
    )
    assert np.all(baseline[4] == -1.0)  # white-to-move perspective is explicit
    assert np.all(baseline[5] == 0.0)  # pass/end-phase is false at the initial state
    assert state.semantic_state is semantic_state


def test_real_trainer_overfits_tiny_dataset_and_optimizer_changes_parameters():
    game_cls, args = _tiny_network_args()
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    rows = 16
    observations = torch.tensor(
        rng.normal(size=(rows, *game_cls.observation_size())).astype(np.float32)
    )
    target_policy = torch.zeros(rows, game_cls.action_size())
    target_policy[torch.arange(rows), torch.arange(rows) % 2] = 1.0
    target_value = torch.zeros(rows, 3)
    target_value[torch.arange(rows), torch.arange(rows) % 2] = 1.0
    loader = DataLoader(
        TensorDataset(observations, target_policy, target_value),
        batch_size=4,
        shuffle=False,
    )

    network = NNetWrapper(game_cls, args)
    with torch.no_grad():
        before_policy, before_value = network.nnet(observations)
        before_loss = float(
            network.loss_pi(target_policy, before_policy)
            + network.loss_v(target_value, before_value)
        )
    before_digest = _state_digest(network)

    network.train(loader, 200)

    with torch.no_grad():
        after_policy, after_value = network.nnet(observations)
        after_loss = float(
            network.loss_pi(target_policy, after_policy)
            + network.loss_v(target_value, after_value)
        )
        policy_probability = torch.exp(after_policy).gather(
            1, target_policy.argmax(dim=1, keepdim=True)
        ).mean()
        value_probability = torch.exp(after_value).gather(
            1, target_value.argmax(dim=1, keepdim=True)
        ).mean()

    assert network.last_train_actual_steps == 200
    assert _state_digest(network) != before_digest
    # Keep this robust across CPU/PyTorch builds: the target probabilities
    # below prove memorization, while this aggregate loss check proves a large
    # reduction without requiring identical optimizer trajectories.
    assert after_loss < before_loss * 0.15
    assert float(policy_probability) > 0.95
    assert float(value_probability) > 0.95


def test_checkpoint_round_trip_preserves_trained_predictions(tmp_path: Path):
    game_cls, args = _tiny_network_args()
    torch.manual_seed(456)
    observations = torch.randn(4, *game_cls.observation_size())
    target_policy = torch.zeros(4, game_cls.action_size())
    target_policy[:, 0] = 1.0
    target_value = torch.zeros(4, 3)
    target_value[:, 1] = 1.0
    network = NNetWrapper(game_cls, args)
    network.train(
        DataLoader(TensorDataset(observations, target_policy, target_value), batch_size=4),
        8,
    )
    before_save = network.predict(observations[0].numpy())
    assert network.optimizer.state  # optimizer state is not silently discarded

    network.save_checkpoint(str(tmp_path), "tiny.pkl")
    restored = NNetWrapper(game_cls, args)
    restored.load_checkpoint(str(tmp_path), "tiny.pkl", device="cpu")
    after_load = restored.predict(observations[0].numpy())

    np.testing.assert_allclose(before_save[0], after_load[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(before_save[1], after_load[1], rtol=1e-6, atol=1e-6)


class _OneMoveArenaGame(GameState):
    """One-ply game: action 0 wins for the mover, action 1 loses."""

    def __init__(self, terminal=False, winner=None):
        super().__init__(np.zeros((1,), dtype=np.uint8))
        self.terminal = bool(terminal)
        self.winner = winner

    def __eq__(self, other):
        return (
            isinstance(other, _OneMoveArenaGame)
            and self.terminal == other.terminal
            and self.winner == other.winner
            and self.player == other.player
        )

    def clone(self):
        clone = type(self)(self.terminal, self.winner)
        clone._player = self._player
        clone._turns = self._turns
        clone.last_action = self.last_action
        return clone

    @staticmethod
    def action_size():
        return 2

    @staticmethod
    def observation_size():
        return (1, 1, 1)

    @staticmethod
    def num_players():
        return 2

    @staticmethod
    def has_draw():
        return True

    def valid_moves(self):
        return np.zeros(2, dtype=np.uint8) if self.terminal else np.ones(2, dtype=np.uint8)

    def play_action(self, action):
        if self.terminal or int(action) not in (0, 1):
            raise ValueError("illegal one-ply action")
        mover = self.player
        self.winner = mover if int(action) == 0 else 1 - mover
        self.terminal = True
        self.last_action = int(action)
        self._update_turn()

    def win_state(self):
        result = np.zeros(3, dtype=np.uint8)
        if self.terminal:
            result[int(self.winner)] = 1
        return result

    def observation(self):
        return np.asarray([[[float(self.player)]]], dtype=np.float32)


class _DeterministicPlayer(BasePlayer):
    def __init__(self, good, args):
        super().__init__(_OneMoveArenaGame, args)
        self.good = bool(good)

    def play(self, _state):
        return 0 if self.good else 1


def test_arena_distinguishes_good_and_bad_deterministic_models():
    args = dotdict(
        numMCTSSims=1,
        arenaMCTSSims=1,
        arenaTemp=0.0,
        use_draws_for_winrate=True,
    )
    good = _DeterministicPlayer(True, args)
    bad = _DeterministicPlayer(False, args)
    arena = Arena([good, bad], _OneMoveArenaGame, use_batched_mcts=False, args=args)

    wins, draws, _ = arena.play_games(8, shuffle_players=True)

    assert wins == [8, 0]
    assert draws == 0


def test_production_gating_is_explicitly_disabled_until_observational_arena_passes():
    with pytest.raises(ValueError, match="model gating is disabled"):
        build_katago_training_args(parse_args(["--model-gating"]))
