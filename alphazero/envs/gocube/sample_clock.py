from __future__ import annotations

import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn.utils import clip_grad_norm_

from alphazero.NNetWrapper import NNetWrapper, _optional_arg
from alphazero.pytorch_classification.utils import AverageMeter, Bar


TRAINING_CONTRACT = "gocube-sample-clock-v1"


@dataclass(frozen=True)
class SampleClockConfig:
    base_lr: float
    warmup_samples: int
    warmup_start_factor: float
    milestones: tuple[int, ...]
    gamma: float


class SampleBasedLRScheduler:
    """Small stateful scheduler whose only clock is optimizer-consumed samples."""

    def __init__(
        self,
        optimizer,
        *,
        base_lr: float,
        warmup_samples: int,
        warmup_start_factor: float,
        milestones: Iterable[int],
        gamma: float,
    ):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_samples = int(warmup_samples)
        self.warmup_start_factor = float(warmup_start_factor)
        self.milestones = tuple(sorted(int(x) for x in milestones))
        self.gamma = float(gamma)
        if not math.isfinite(self.base_lr) or self.base_lr <= 0.0:
            raise ValueError("base LR must be finite and positive")
        if self.warmup_samples < 0:
            raise ValueError("warmup samples must be non-negative")
        if not 0.0 < self.warmup_start_factor <= 1.0:
            raise ValueError("warmup start factor must be within (0,1]")
        if any(x <= 0 for x in self.milestones):
            raise ValueError("LR sample milestones must be positive")
        if len(set(self.milestones)) != len(self.milestones):
            raise ValueError("LR sample milestones must be unique")
        if self.warmup_samples and any(x <= self.warmup_samples for x in self.milestones):
            raise ValueError("LR milestones must be after the warmup window")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("LR gamma must be within (0,1]")

        self.total_training_samples = 0
        self.total_optimizer_updates = 0
        self.last_lr_change_samples = 0
        self.current_lr = self.lr_for_samples(0)
        self._apply_lr(self.current_lr)

    def lr_for_samples(self, samples: int) -> float:
        samples = max(0, int(samples))
        if self.warmup_samples > 0 and samples < self.warmup_samples:
            progress = samples / float(self.warmup_samples)
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        else:
            factor = 1.0
        decays = sum(1 for threshold in self.milestones if samples >= threshold)
        return self.base_lr * factor * (self.gamma ** decays)

    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def step_samples(self, batch_size: int) -> float:
        batch_size = int(batch_size)
        if batch_size < 0:
            raise ValueError("batch size cannot be negative")
        self.total_training_samples += batch_size
        self.total_optimizer_updates += 1 if batch_size > 0 else 0
        next_lr = self.lr_for_samples(self.total_training_samples)
        if not math.isclose(next_lr, self.current_lr, rel_tol=1e-12, abs_tol=0.0):
            self.current_lr = next_lr
            self.last_lr_change_samples = self.total_training_samples
            self._apply_lr(next_lr)
        return self.current_lr

    @property
    def samples_since_last_lr_change(self) -> int:
        return max(0, self.total_training_samples - self.last_lr_change_samples)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "training_contract": TRAINING_CONTRACT,
            "base_lr": self.base_lr,
            "warmup_samples": self.warmup_samples,
            "warmup_start_factor": self.warmup_start_factor,
            "milestones": self.milestones,
            "gamma": self.gamma,
            "total_training_samples": self.total_training_samples,
            "total_optimizer_updates": self.total_optimizer_updates,
            "last_lr_change_samples": self.last_lr_change_samples,
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("training_contract") != TRAINING_CONTRACT:
            raise ValueError("Checkpoint scheduler state does not use gocube-sample-clock-v1")
        expected = SampleClockConfig(
            self.base_lr,
            self.warmup_samples,
            self.warmup_start_factor,
            self.milestones,
            self.gamma,
        )
        saved = SampleClockConfig(
            float(state["base_lr"]),
            int(state["warmup_samples"]),
            float(state["warmup_start_factor"]),
            tuple(int(x) for x in state["milestones"]),
            float(state["gamma"]),
        )
        if saved != expected:
            raise ValueError(f"Checkpoint sample-clock config mismatch: saved={saved}, expected={expected}")
        self.total_training_samples = int(state["total_training_samples"])
        self.total_optimizer_updates = int(state["total_optimizer_updates"])
        self.last_lr_change_samples = int(state["last_lr_change_samples"])
        self.current_lr = float(state["current_lr"])
        expected_lr = self.lr_for_samples(self.total_training_samples)
        if not math.isclose(self.current_lr, expected_lr, rel_tol=1e-10, abs_tol=0.0):
            raise ValueError(
                f"Checkpoint LR does not match sample clock: saved={self.current_lr}, expected={expected_lr}"
            )
        self._apply_lr(self.current_lr)


class SampleClockNNetWrapper(NNetWrapper):
    """Pinned GoCube trainer with sample-clock LR, warmup, and global grad clipping."""

    def __init__(self, game_cls, args):
        super().__init__(game_cls, args)
        contract = _optional_arg(args, "gocube_training_contract", None)
        if contract != TRAINING_CONTRACT:
            raise ValueError(f"SampleClockNNetWrapper requires {TRAINING_CONTRACT}, got {contract!r}")
        self.scheduler = SampleBasedLRScheduler(
            self.optimizer,
            base_lr=float(args.lr),
            warmup_samples=int(args.gocube_lr_warmup_samples),
            warmup_start_factor=float(args.gocube_lr_warmup_start_factor),
            milestones=tuple(args.gocube_lr_milestone_samples),
            gamma=float(args.gocube_lr_decay_gamma),
        )
        self.gradient_clip_norm = float(args.gocube_gradient_clip_norm)
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clipping norm must be finite and positive")
        self.last_train_gradient_norm = 0.0
        self.last_train_gradient_norm_max = 0.0
        self.last_train_clipping_events = 0
        self.last_train_clipping_frequency = 0.0
        self.last_train_samples_since_lr_change = self.scheduler.samples_since_last_lr_change

    def _checkpoint_contract(self):
        fields = super()._checkpoint_contract()
        for key in (
            "gocube_training_contract",
            "gocube_lr_warmup_samples",
            "gocube_lr_warmup_start_factor",
            "gocube_lr_milestone_samples",
            "gocube_lr_decay_gamma",
            "gocube_gradient_clip_norm",
        ):
            value = _optional_arg(self.args, key, None)
            if value is None:
                raise ValueError(f"Missing required training-contract field: {key}")
            fields[key] = value
        return fields

    def _validate_saved_contract(self, saved_args, allow_legacy_search_contract=False):
        super()._validate_saved_contract(
            saved_args, allow_legacy_search_contract=allow_legacy_search_contract
        )
        for key in (
            "gocube_training_contract",
            "gocube_lr_warmup_samples",
            "gocube_lr_warmup_start_factor",
            "gocube_lr_milestone_samples",
            "gocube_lr_decay_gamma",
            "gocube_gradient_clip_norm",
        ):
            present = key in saved_args if hasattr(saved_args, "__contains__") else hasattr(saved_args, key)
            if not present:
                raise ValueError(f"Checkpoint missing required sample-clock training metadata: {key}")

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', make_dirs=True):
        filepath = os.path.join(folder, filename)
        if make_dirs and not os.path.exists(folder):
            os.makedirs(folder)
        training_state = {
            "schema_version": 1,
            "training_contract": TRAINING_CONTRACT,
            "total_training_samples": self.total_training_samples,
            "total_optimizer_updates": self.total_optimizer_updates,
            "samples_since_lr_change": self.scheduler.samples_since_last_lr_change,
            "effective_lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sch_state': self.scheduler.state_dict(),
            'training_state': training_state,
            'args': self.args,
        }, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(
        self,
        folder='checkpoint',
        filename='checkpoint.pth.tar',
        use_saved_args=True,
        device=None,
        load_training_state=True,
        allow_legacy_search_contract=False,
    ):
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath, map_location=device) if device else torch.load(filepath)
        if load_training_state:
            training_state = checkpoint.get("training_state")
            if not isinstance(training_state, dict) or training_state.get("training_contract") != TRAINING_CONTRACT:
                raise ValueError(
                    "Checkpoint predates gocube-sample-clock-v1 and cannot be resumed under the new training contract"
                )
        result = super().load_checkpoint(
            folder=folder,
            filename=filename,
            use_saved_args=use_saved_args,
            device=device,
            load_training_state=load_training_state,
            allow_legacy_search_contract=allow_legacy_search_contract,
        )
        if load_training_state:
            training_state = checkpoint["training_state"]
            if int(training_state["total_training_samples"]) != self.total_training_samples:
                raise ValueError("Checkpoint training-sample counter disagrees with scheduler state")
            if int(training_state["total_optimizer_updates"]) != self.total_optimizer_updates:
                raise ValueError("Checkpoint optimizer-update counter disagrees with scheduler state")
        return result

    @property
    def total_training_samples(self) -> int:
        return int(self.scheduler.total_training_samples)

    @property
    def total_optimizer_updates(self) -> int:
        return int(self.scheduler.total_optimizer_updates)

    def train(self, batches, train_steps):
        self.total_steps = train_steps
        self.current_step = 0
        self.last_train_planned_steps = int(train_steps)
        self.last_train_actual_steps = 0
        self.last_train_examples_seen = 0
        self.last_train_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.last_train_gradient_norm = 0.0
        self.last_train_gradient_norm_max = 0.0
        self.last_train_clipping_events = 0
        self.last_train_clipping_frequency = 0.0
        self.last_train_samples_since_lr_change = self.scheduler.samples_since_last_lr_change
        if train_steps <= 0:
            return self.l_pi, self.l_v

        self.nnet.train()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        ownership_losses = AverageMeter()
        score_losses = AverageMeter()
        auxiliary = bool(getattr(self.args, "gocube_auxiliary_targets", False))
        gradient_norm_sum = 0.0
        gradient_norm_max = 0.0
        clipping_events = 0
        bar = Bar("Training Net", max=train_steps)

        while self.current_step < train_steps and not self.stop_train.is_set():
            for batch in batches:
                if self.current_step == train_steps or self.stop_train.is_set():
                    break
                while self.pause_train.is_set():
                    time.sleep(0.1)
                start = time.time()
                self.current_step += 1
                ownership_mask = None
                if auxiliary:
                    if len(batch) == 6:
                        boards, target_pis, target_vs, target_scores, target_ownership, ownership_mask = batch
                    elif len(batch) == 5:
                        boards, target_pis, target_vs, target_scores, target_ownership = batch
                    else:
                        raise ValueError(f"Expected 5 or 6 auxiliary tensors, got {len(batch)}")
                else:
                    boards, target_pis, target_vs = batch

                if self.args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()
                    if auxiliary:
                        target_scores = target_scores.contiguous().cuda()
                        target_ownership = target_ownership.contiguous().cuda()
                        if ownership_mask is not None:
                            ownership_mask = ownership_mask.contiguous().cuda()
                data_time.update(time.time() - start)

                outputs = self.nnet(boards)
                out_pi, out_v = outputs[0], outputs[1]
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                if auxiliary:
                    out_ownership, out_score = outputs[2], outputs[3]
                    l_ownership = self.loss_ownership(target_ownership, out_ownership, ownership_mask)
                    l_score = self.loss_score(target_scores, out_score)
                    total_loss = total_loss + l_ownership + l_score
                    ownership_losses.update(l_ownership.item(), boards.size(0))
                    score_losses.update(l_score.item(), boards.size(0))
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = float(clip_grad_norm_(self.nnet.parameters(), self.gradient_clip_norm))
                gradient_norm_sum += grad_norm
                gradient_norm_max = max(gradient_norm_max, grad_norm)
                if math.isfinite(grad_norm) and grad_norm > self.gradient_clip_norm:
                    clipping_events += 1
                self.optimizer.step()

                actual_batch_size = int(boards.size(0))
                self.scheduler.step_samples(actual_batch_size)
                self.last_train_actual_steps += 1
                self.last_train_examples_seen += actual_batch_size
                self.last_train_learning_rate = float(self.optimizer.param_groups[0]["lr"])
                batch_time.update(time.time() - start)
                self.l_pi = pi_losses.avg
                self.l_v = v_losses.avg
                self.l_ownership = ownership_losses.avg if auxiliary else 0
                self.l_score = score_losses.avg if auxiliary else 0
                self.l_total = self.l_pi + self.l_v + self.l_ownership + self.l_score
                self.step_time = data_time.avg + batch_time.avg
                self.elapsed_time = bar.elapsed_td
                self.eta = bar.eta_td
                bar.suffix = (
                    f"({self.current_step}/{train_steps}) Data: {data_time.avg:.3f}s | "
                    f"Batch: {batch_time.avg:.3f}s | LR: {self.last_train_learning_rate:g} | "
                    f"GNorm: {grad_norm:.3f} | Total: {bar.elapsed_td} | ETA: {bar.eta_td} | "
                    f"Loss_pi: {pi_losses.avg:.4f} | Loss_v: {v_losses.avg:.3f}"
                )
                if auxiliary:
                    bar.suffix += (
                        f" | Loss_owner: {ownership_losses.avg:.3f} | Loss_score: {score_losses.avg:.3f}"
                    )
                bar.next()

        steps = self.last_train_actual_steps
        self.last_train_gradient_norm = gradient_norm_sum / steps if steps else 0.0
        self.last_train_gradient_norm_max = gradient_norm_max
        self.last_train_clipping_events = clipping_events
        self.last_train_clipping_frequency = clipping_events / steps if steps else 0.0
        self.last_train_samples_since_lr_change = self.scheduler.samples_since_last_lr_change
        bar.update()
        bar.finish()
        print()
        return pi_losses.avg, v_losses.avg
