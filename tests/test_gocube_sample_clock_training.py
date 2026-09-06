from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alphazero.envs.gocube.sample_clock import (
    SampleBasedLRScheduler,
    SampleClockNNetWrapper,
    TRAINING_CONTRACT,
)
from alphazero.utils import dotdict


def test_sample_scheduler_warmup_milestones_and_resume_state():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = SampleBasedLRScheduler(
        optimizer,
        base_lr=0.1,
        warmup_samples=100,
        warmup_start_factor=0.1,
        milestones=(200, 400),
        gamma=0.1,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    scheduler.step_samples(50)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.055)
    scheduler.step_samples(50)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    scheduler.step_samples(100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    state = scheduler.state_dict()

    parameter2 = nn.Parameter(torch.tensor([1.0]))
    optimizer2 = torch.optim.SGD([parameter2], lr=0.1)
    restored = SampleBasedLRScheduler(
        optimizer2,
        base_lr=0.1,
        warmup_samples=100,
        warmup_start_factor=0.1,
        milestones=(200, 400),
        gamma=0.1,
    )
    restored.load_state_dict(state)
    assert restored.total_training_samples == 200
    assert restored.total_optimizer_updates == 3
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(0.01)
    restored.step_samples(200)
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(0.001)


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


class _TinySampleClockWrapper(SampleClockNNetWrapper):
    def _load_nnet(self, _args):
        self.nnet = _TinyNet()


def _args():
    return dotdict({
        "optimizer": torch.optim.SGD,
        "optimizer_args": {},
        "scheduler": torch.optim.lr_scheduler.MultiStepLR,
        "scheduler_args": {"milestones": [75, 125], "gamma": 0.1},
        "lr": 0.1,
        "cuda": False,
        "nnet_type": "tiny",
        "gocube_auxiliary_targets": False,
        "value_loss_weight": 1.5,
        "gocube_training_contract": TRAINING_CONTRACT,
        "gocube_lr_warmup_samples": 1000,
        "gocube_lr_warmup_start_factor": 0.1,
        "gocube_lr_milestone_samples": (2000, 4000),
        "gocube_lr_decay_gamma": 0.1,
        "gocube_gradient_clip_norm": 1e-6,
    })


def _dataset(size):
    boards = torch.arange(size, dtype=torch.float32).view(-1, 1) / max(1, size)
    policy = torch.zeros((size, 2), dtype=torch.float32)
    policy[:, 0] = 1.0
    value = torch.zeros((size, 3), dtype=torch.float32)
    value[:, 0] = 1.0
    return TensorDataset(boards, policy, value)


def test_sample_clock_counts_real_partial_batches_and_clips_gradients():
    wrapper = _TinySampleClockWrapper(_TinyGame, _args())
    loader = DataLoader(_dataset(530), batch_size=256, shuffle=False)
    wrapper.train(loader, 3)
    assert wrapper.last_train_actual_steps == 3
    assert wrapper.last_train_examples_seen == 530
    assert wrapper.total_training_samples == 530
    assert wrapper.total_optimizer_updates == 3
    assert wrapper.last_train_clipping_events == 3
    assert wrapper.last_train_clipping_frequency == pytest.approx(1.0)
    expected_lr = wrapper.scheduler.lr_for_samples(530)
    assert wrapper.last_train_learning_rate == pytest.approx(expected_lr)


def test_checkpoint_resume_preserves_sample_clock_and_does_not_restart_warmup(tmp_path):
    wrapper = _TinySampleClockWrapper(_TinyGame, _args())
    loader = DataLoader(_dataset(300), batch_size=128, shuffle=False)
    wrapper.train(loader, 3)
    before_samples = wrapper.total_training_samples
    before_lr = wrapper.last_train_learning_rate
    wrapper.save_checkpoint(str(tmp_path), "sample-clock.pkl")

    restored = _TinySampleClockWrapper(_TinyGame, _args())
    restored.load_checkpoint(str(tmp_path), "sample-clock.pkl")
    assert restored.total_training_samples == before_samples
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(before_lr)
    restored.train(DataLoader(_dataset(100), batch_size=100, shuffle=False), 1)
    assert restored.total_training_samples == before_samples + 100
    assert restored.total_optimizer_updates == 4


def test_old_checkpoint_contract_is_rejected_explicitly():
    wrapper = _TinySampleClockWrapper(_TinyGame, _args())
    with pytest.raises(ValueError, match="sample-clock training metadata"):
        wrapper._validate_saved_contract(dotdict({}))
