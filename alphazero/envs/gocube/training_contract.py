from __future__ import annotations

import os
import pickle
import time
from typing import Iterable

import torch

from alphazero.NNetWrapper import NNetWrapper
from alphazero.pytorch_classification.utils import AverageMeter, Bar


TRAINING_CONTRACT = "gocube-sample-clock-v1"
LR_CLOCK = "optimizer-training-samples"


class SampleClockLRScheduler:
    """LR scheduler driven only by samples actually consumed by optimizer steps."""

    def __init__(
        self,
        optimizer,
        *,
        milestones=(),
        gamma=0.1,
        warmup_samples=0,
        warmup_start_factor=0.05,
        verbose=False,
    ):
        self.optimizer = optimizer
        self.milestones = tuple(sorted(int(value) for value in milestones))
        if any(value <= 0 for value in self.milestones):
            raise ValueError("sample LR milestones must be positive")
        if float(gamma) <= 0.0:
            raise ValueError("sample LR gamma must be positive")
        self.gamma = float(gamma)
        self.warmup_samples = int(warmup_samples)
        if self.warmup_samples < 0:
            raise ValueError("warmup_samples must be non-negative")
        self.warmup_start_factor = float(warmup_start_factor)
        if not 0.0 < self.warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be within (0,1]")
        self.verbose = bool(verbose)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.current_samples = 0
        self.last_lr_change_samples = 0
        self._set_lrs(self._lrs_for_samples(0))

    def _warmup_factor(self, samples: int) -> float:
        if self.warmup_samples <= 0 or samples >= self.warmup_samples:
            return 1.0
        progress = float(samples) / float(self.warmup_samples)
        return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress

    def _decay_factor(self, samples: int) -> float:
        count = sum(samples >= threshold for threshold in self.milestones)
        return self.gamma ** count

    def _lrs_for_samples(self, samples: int) -> list[float]:
        scale = self._warmup_factor(samples) * self._decay_factor(samples)
        return [base * scale for base in self.base_lrs]

    def _set_lrs(self, lrs: Iterable[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = float(lr)

    def step_samples(self, samples: int) -> None:
        samples = int(samples)
        if samples < self.current_samples:
            raise ValueError("sample clock cannot move backwards")
        previous = [float(group["lr"]) for group in self.optimizer.param_groups]
        updated = self._lrs_for_samples(samples)
        if any(abs(a - b) > 1e-15 for a, b in zip(previous, updated)):
            self.last_lr_change_samples = samples
        self.current_samples = samples
        self._set_lrs(updated)

    def step(self, _metric=None):
        raise RuntimeError("SampleClockLRScheduler must be advanced with step_samples(), not iteration step()")

    def state_dict(self) -> dict:
        return {
            "contract": TRAINING_CONTRACT,
            "clock": LR_CLOCK,
            "milestones": self.milestones,
            "gamma": self.gamma,
            "warmup_samples": self.warmup_samples,
            "warmup_start_factor": self.warmup_start_factor,
            "base_lrs": list(self.base_lrs),
            "current_samples": self.current_samples,
            "last_lr_change_samples": self.last_lr_change_samples,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("contract") != TRAINING_CONTRACT or state.get("clock") != LR_CLOCK:
            raise ValueError("checkpoint scheduler state is not a GoCube sample-clock scheduler")
        expected = (
            self.milestones,
            self.gamma,
            self.warmup_samples,
            self.warmup_start_factor,
        )
        actual = (
            tuple(int(v) for v in state.get("milestones", ())),
            float(state.get("gamma")),
            int(state.get("warmup_samples")),
            float(state.get("warmup_start_factor")),
        )
        if actual != expected:
            raise ValueError(f"sample-clock scheduler configuration mismatch: saved={actual!r}, expected={expected!r}")
        saved_base_lrs = [float(v) for v in state.get("base_lrs", ())]
        if len(saved_base_lrs) != len(self.base_lrs):
            raise ValueError("sample-clock scheduler param-group count mismatch")
        self.base_lrs = saved_base_lrs
        self.current_samples = int(state.get("current_samples", 0))
        self.last_lr_change_samples = int(state.get("last_lr_change_samples", 0))
        self._set_lrs(self._lrs_for_samples(self.current_samples))


class SampleClockNNetWrapper(NNetWrapper):
    """GoCube pinned trainer with persistent sample clock, warmup, and grad clipping."""

    def __init__(self, game_cls, args):
        super().__init__(game_cls, args)
        self.total_training_samples = 0
        self.total_optimizer_updates = 0
        self.last_train_gradient_norm = 0.0
        self.last_train_gradient_norm_avg = 0.0
        self.last_train_clip_events = 0
        self.last_train_clip_checks = 0
        self.last_train_clip_frequency = 0.0
        self.scheduler.step_samples(0)
        self.last_train_learning_rate = float(self.optimizer.param_groups[0]["lr"])

    def _checkpoint_contract(self):
        fields = super()._checkpoint_contract()
        if getattr(self, "args", None) is None:
            return fields
        fields.update({
            "gocube_training_contract": getattr(self.args, "gocube_training_contract"),
            "gocube_lr_clock": getattr(self.args, "gocube_lr_clock"),
            "gocube_lr_sample_milestones": tuple(int(v) for v in getattr(self.args, "gocube_lr_sample_milestones")),
            "gocube_lr_gamma": float(getattr(self.args, "gocube_lr_gamma")),
            "gocube_lr_warmup_samples": int(getattr(self.args, "gocube_lr_warmup_samples")),
            "gocube_lr_warmup_start_factor": float(getattr(self.args, "gocube_lr_warmup_start_factor")),
            "gocube_gradient_clip_norm": float(getattr(self.args, "gocube_gradient_clip_norm")),
        })
        return fields

    @property
    def samples_since_lr_change(self) -> int:
        return max(0, int(self.total_training_samples) - int(self.scheduler.last_lr_change_samples))

    def _reset_last_train_metrics(self, train_steps: int) -> None:
        self.total_steps = train_steps
        self.current_step = 0
        self.last_train_planned_steps = int(train_steps)
        self.last_train_actual_steps = 0
        self.last_train_examples_seen = 0
        self.last_train_gradient_norm = 0.0
        self.last_train_gradient_norm_avg = 0.0
        self.last_train_clip_events = 0
        self.last_train_clip_checks = 0
        self.last_train_clip_frequency = 0.0
        self.scheduler.step_samples(self.total_training_samples)
        self.last_train_learning_rate = float(self.optimizer.param_groups[0]["lr"])

    def train(self, batches, train_steps):
        self._reset_last_train_metrics(train_steps)
        if train_steps <= 0:
            return self.l_pi, self.l_v

        self.nnet.train()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        ownership_losses = AverageMeter()
        score_losses = AverageMeter()
        gradient_norms = AverageMeter()
        auxiliary = bool(getattr(self.args, "gocube_auxiliary_targets", False))
        clip_norm = float(getattr(self.args, "gocube_gradient_clip_norm"))
        if clip_norm <= 0.0:
            raise ValueError("gocube_gradient_clip_norm must be positive")

        if self.verbose:
            print(f"Current sample-clock LR: {self.optimizer.param_groups[0]['lr']}")
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

                batch_samples = int(boards.size(0))
                self.scheduler.step_samples(self.total_training_samples)
                effective_lr = float(self.optimizer.param_groups[0]["lr"])
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
                    ownership_losses.update(l_ownership.item(), batch_samples)
                    score_losses.update(l_score.item(), batch_samples)

                pi_losses.update(l_pi.item(), batch_samples)
                v_losses.update(l_v.item(), batch_samples)
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), clip_norm))
                clipped = bool(grad_norm > clip_norm)
                self.optimizer.step()

                self.total_training_samples += batch_samples
                self.total_optimizer_updates += 1
                self.scheduler.step_samples(self.total_training_samples)
                self.last_train_actual_steps += 1
                self.last_train_examples_seen += batch_samples
                self.last_train_clip_checks += 1
                self.last_train_clip_events += int(clipped)
                gradient_norms.update(grad_norm)
                self.last_train_gradient_norm = grad_norm
                self.last_train_gradient_norm_avg = gradient_norms.avg
                self.last_train_clip_frequency = self.last_train_clip_events / self.last_train_clip_checks
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
                suffix = (
                    f"({self.current_step}/{train_steps}) LR: {effective_lr:.3g} | Samples: {self.total_training_samples} | "
                    f"Grad: {grad_norm:.3g} | Clip: {self.last_train_clip_frequency:.1%} | "
                    f"Loss_pi: {pi_losses.avg:.4f} | Loss_v: {v_losses.avg:.3f}"
                )
                if auxiliary:
                    suffix += f" | Loss_owner: {ownership_losses.avg:.3f} | Loss_score: {score_losses.avg:.3f}"
                bar.suffix = suffix
                bar.next()

        bar.update()
        bar.finish()
        print()
        return pi_losses.avg, v_losses.avg

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar", make_dirs=True):
        filepath = os.path.join(folder, filename)
        if make_dirs and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({
            "state_dict": self.nnet.state_dict(),
            "opt_state": self.optimizer.state_dict(),
            "sch_state": self.scheduler.state_dict(),
            "training_state": {
                "contract": TRAINING_CONTRACT,
                "total_training_samples": int(self.total_training_samples),
                "total_optimizer_updates": int(self.total_optimizer_updates),
            },
            "args": self.args,
        }, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, *args, load_training_state=True, **kwargs):
        folder = kwargs.get("folder", args[0] if len(args) > 0 else "checkpoint")
        filename = kwargs.get("filename", args[1] if len(args) > 1 else "checkpoint.pth.tar")
        device = kwargs.get("device")
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath, map_location=device) if device else torch.load(filepath)
        if load_training_state:
            training_state = checkpoint.get("training_state")
            if not isinstance(training_state, dict) or training_state.get("contract") != TRAINING_CONTRACT:
                raise ValueError(
                    "Checkpoint predates the GoCube sample-clock training contract; "
                    "refusing to infer total_training_samples from iteration number."
                )
        result = super().load_checkpoint(*args, load_training_state=load_training_state, **kwargs)
        if load_training_state:
            training_state = checkpoint["training_state"]
            self.total_training_samples = int(training_state["total_training_samples"])
            self.total_optimizer_updates = int(training_state["total_optimizer_updates"])
            if int(self.scheduler.current_samples) != self.total_training_samples:
                raise ValueError("checkpoint scheduler/sample counter mismatch")
            self.last_train_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        return result
