from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.envs.gocube.train import (
    GoCubeNNetWrapper,
    expected_saved_samples,
    resolve_train_steps,
    validate_tensor_row_counts,
)
from alphazero.utils import dotdict


class _NoPause:
    @staticmethod
    def is_set():
        return False


class _DecisionGameClass:
    @staticmethod
    def max_turns():
        return 20


class _DecisionGame:
    def __init__(self):
        self.turns = 0
        self.player = 0

    @staticmethod
    def action_size():
        return 2

    def clone(self):
        clone = _DecisionGame()
        clone.turns = self.turns
        clone.player = self.player
        return clone

    def play_action(self, _action):
        self.turns += 1
        self.player = 1 - self.player

    @staticmethod
    def win_state():
        return np.array([False, False, False], dtype=np.bool_)


class _DecisionMCTS:
    @staticmethod
    def probs(_game, _temp=None):
        return np.array([1.0, 0.0], dtype=np.float64)

    @staticmethod
    def update_root(_game, _action):
        return None


def _decision_agent(prob_fast):
    agent = SelfPlayAgent.__new__(SelfPlayAgent)
    agent.args = dotdict({
        "numMCTSSims": 100,
        "numFastSims": 20,
        "numWarmupSims": 5,
        "probFastSim": prob_fast,
        "temp_scaling_fn": lambda temp, _turns, _max_turns: temp,
        "arenaTemp": 0.0,
        "mctsResetThreshold": None,
    })
    agent._is_arena = False
    agent._is_warmup = False
    agent.fast = False
    agent.batch_size = 1
    agent.games = [_DecisionGame()]
    agent.game_cls = _DecisionGameClass
    agent.mcts = [_DecisionMCTS()]
    agent.histories = [[]]
    agent.temps = [1.0]
    agent.next_reset = [0]
    agent.pause_event = _NoPause()
    agent.telemetry = {
        "regular_decisions": mp.Value('q', 0),
        "fast_decisions": mp.Value('q', 0),
    }
    return agent


@pytest.mark.parametrize(
    ("prob_fast", "regular", "fast", "fraction"),
    [(0.0, 1, 0, 0.0), (1.0, 0, 1, 1.0)],
)
def test_realized_fast_telemetry_tracks_executed_decisions(prob_fast, regular, fast, fraction):
    agent = _decision_agent(prob_fast)
    agent._select_search_sims()
    agent.playMoves()
    regular_count = agent.telemetry["regular_decisions"].value
    fast_count = agent.telemetry["fast_decisions"].value
    total = regular_count + fast_count
    realized = fast_count / total if total else 0.0
    assert regular_count == regular
    assert fast_count == fast
    assert realized == fraction


def test_sample_accounting_weight_one_and_three():
    assert expected_saved_samples(10_000, 2_000, 1) == 10_000
    assert expected_saved_samples(10_000, 2_000, 3) == 14_000


def test_tensor_row_count_invariant_accepts_equal_rows_and_rejects_mismatch():
    tensors = [torch.zeros((7, 1)), torch.zeros((7, 3)), torch.zeros((7, 2))]
    assert validate_tensor_row_counts(tensors, expected=7) == 7
    with pytest.raises(ValueError, match="row-count mismatch"):
        validate_tensor_row_counts([torch.zeros((7, 1)), torch.zeros((6, 1))])


def test_auto_step_guard_never_returns_zero_for_nonempty_dataset():
    assert resolve_train_steps(
        dataset_size=100,
        sample_budget=100,
        batch_size=256,
        auto_train_steps=True,
    ) == 1


def test_fixed_step_override_uses_exact_requested_budget():
    assert resolve_train_steps(
        dataset_size=1000,
        sample_budget=1000,
        batch_size=256,
        auto_train_steps=False,
        fixed_steps=17,
    ) == 17


class _TinyGame:
    @staticmethod
    def action_size():
        return 2


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 5)

    def forward(self, boards):
        logits = self.linear(boards.float())
        return (
            torch.log_softmax(logits[:, :2], dim=1),
            torch.log_softmax(logits[:, 2:], dim=1),
        )


class _TinyWrapper(GoCubeNNetWrapper):
    def _load_nnet(self, _args):
        self.nnet = _TinyNet()


def _wrapper_args():
    return SimpleNamespace(
        optimizer=torch.optim.SGD,
        optimizer_args={},
        scheduler=torch.optim.lr_scheduler.MultiStepLR,
        scheduler_args={"milestones": [100], "gamma": 0.1},
        lr=0.01,
        cuda=False,
        nnet_type="tiny",
        gocube_auxiliary_targets=False,
        value_loss_weight=1.5,
    )


def _dataset(size):
    boards = torch.arange(size, dtype=torch.float32).view(-1, 1) / max(1, size)
    policy = torch.zeros((size, 2), dtype=torch.float32)
    policy[:, 0] = 1.0
    value = torch.zeros((size, 3), dtype=torch.float32)
    value[:, 0] = 1.0
    return TensorDataset(boards, policy, value)


def test_optimizer_accounting_counts_actual_steps_and_examples_seen():
    wrapper = _TinyWrapper(_TinyGame, _wrapper_args())
    loader = DataLoader(_dataset(1000), batch_size=256, shuffle=False)
    wrapper.train(loader, 3)
    assert wrapper.last_train_planned_steps == 3
    assert wrapper.last_train_actual_steps == 3
    assert wrapper.last_train_examples_seen == 768
    assert wrapper.last_train_learning_rate == pytest.approx(0.01)


def test_optimizer_accounting_uses_real_partial_batch_size():
    wrapper = _TinyWrapper(_TinyGame, _wrapper_args())
    loader = DataLoader(_dataset(530), batch_size=256, shuffle=False)
    wrapper.train(loader, 3)
    assert wrapper.last_train_actual_steps == 3
    assert wrapper.last_train_examples_seen == 530
