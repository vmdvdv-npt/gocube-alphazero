from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alphazero.envs.gocube.training_contract import (
    LR_CLOCK,
    TRAINING_CONTRACT,
    SampleClockLRScheduler,
    SampleClockNNetWrapper,
)
from alphazero.utils import dotdict


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


class _TinyWrapper(SampleClockNNetWrapper):
    def _load_nnet(self, _args):
        self.nnet = _TinyNet()


def _args(*, warmup_samples=8, milestones=(10,), gamma=0.5, clip_norm=1.0):
    return dotdict({
        "optimizer": torch.optim.SGD,
        "optimizer_args": {},
        "scheduler": SampleClockLRScheduler,
        "scheduler_args": {
            "milestones": milestones,
            "gamma": gamma,
            "warmup_samples": warmup_samples,
            "warmup_start_factor": 0.1,
        },
        "lr": 0.1,
        "cuda": False,
        "nnet_type": "tiny",
        "gocube_auxiliary_targets": False,
        "value_loss_weight": 1.5,
        "gocube_training_contract": TRAINING_CONTRACT,
        "gocube_lr_clock": LR_CLOCK,
        "gocube_lr_sample_milestones": milestones,
        "gocube_lr_gamma": gamma,
        "gocube_lr_warmup_samples": warmup_samples,
        "gocube_lr_warmup_start_factor": 0.1,
        "gocube_gradient_clip_norm": clip_norm,
    })


def _dataset(size):
    boards = torch.arange(size, dtype=torch.float32).view(-1, 1) + 1.0
    policy = torch.zeros((size, 2), dtype=torch.float32)
    policy[:, 0] = 1.0
    value = torch.zeros((size, 3), dtype=torch.float32)
    value[:, 0] = 1.0
    return TensorDataset(boards, policy, value)


def test_sample_clock_uses_actual_partial_batch_sizes_and_not_iterations():
    wrapper = _TinyWrapper(_TinyGame, _args())
    loader = DataLoader(_dataset(10), batch_size=4, shuffle=False)
    wrapper.train(loader, 3)

    assert wrapper.total_training_samples == 10
    assert wrapper.total_optimizer_updates == 3
    assert wrapper.last_train_examples_seen == 10
    assert wrapper.scheduler.current_samples == 10
    assert wrapper.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_linear_warmup_advances_smoothly_by_samples():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
    scheduler = SampleClockLRScheduler(
        optimizer,
        milestones=(),
        gamma=0.5,
        warmup_samples=100,
        warmup_start_factor=0.1,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    scheduler.step_samples(25)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0325)
    scheduler.step_samples(50)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.055)
    scheduler.step_samples(100)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_gradient_norm_is_logged_and_clipping_fires():
    torch.manual_seed(1)
    wrapper = _TinyWrapper(_TinyGame, _args(clip_norm=1e-6, milestones=()))
    loader = DataLoader(_dataset(8), batch_size=8, shuffle=False)
    wrapper.train(loader, 1)

    assert wrapper.last_train_gradient_norm > 1e-6
    assert wrapper.last_train_clip_checks == 1
    assert wrapper.last_train_clip_events == 1
    assert wrapper.last_train_clip_frequency == 1.0


def test_checkpoint_resume_preserves_sample_clock_and_does_not_restart_warmup(tmp_path):
    torch.manual_seed(2)
    wrapper = _TinyWrapper(_TinyGame, _args(warmup_samples=100, milestones=(200,)))
    wrapper.train(DataLoader(_dataset(40), batch_size=20, shuffle=False), 2)
    lr_before = wrapper.optimizer.param_groups[0]["lr"]
    wrapper.save_checkpoint(str(tmp_path), "sample-clock.pkl")

    resumed = _TinyWrapper(_TinyGame, _args(warmup_samples=100, milestones=(200,)))
    resumed.load_checkpoint(str(tmp_path), "sample-clock.pkl")
    assert resumed.total_training_samples == 40
    assert resumed.total_optimizer_updates == 2
    assert resumed.scheduler.current_samples == 40
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(lr_before)

    resumed.train(DataLoader(_dataset(20), batch_size=20, shuffle=False), 1)
    assert resumed.total_training_samples == 60
    assert resumed.total_optimizer_updates == 3
    assert resumed.optimizer.param_groups[0]["lr"] > lr_before


def test_checkpoint_without_training_contract_state_is_rejected(tmp_path):
    wrapper = _TinyWrapper(_TinyGame, _args())
    wrapper.save_checkpoint(str(tmp_path), "new.pkl")
    checkpoint = torch.load(tmp_path / "new.pkl")
    checkpoint.pop("training_state")
    torch.save(checkpoint, tmp_path / "legacy.pkl")

    target = _TinyWrapper(_TinyGame, _args())
    try:
        target.load_checkpoint(str(tmp_path), "legacy.pkl")
    except ValueError as exc:
        assert "predates the GoCube sample-clock training contract" in str(exc)
    else:
        raise AssertionError("legacy checkpoint was silently accepted")
