"""Shared parent-side Arena inference adapter.

Arena profiles provide only the model-specific forward boundary.  Tensor
transport, inference mode, normalization, validation, CPU return transport and
timing are owned here through ``BatchedPolicyWDLInferenceOwner``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch

from gocube_golden.inference import BatchedPolicyWDLInferenceOwner


@dataclass(frozen=True)
class ArenaInferenceBatch:
    policy: torch.Tensor
    wdl: torch.Tensor
    timing: dict[str, float]
    owner: BatchedPolicyWDLInferenceOwner


_OWNER_CACHE: dict[
    tuple[int, str, tuple[int, ...], int, int, str],
    BatchedPolicyWDLInferenceOwner,
] = {}


def _owner_for(
    model: torch.nn.Module,
    *,
    device: torch.device,
    observation_shape: tuple[int, ...],
    policy_size: int,
    wdl_size: int,
    forward_key: str,
    forward_policy_wdl_logits: Callable[[torch.nn.Module, torch.Tensor], object],
) -> BatchedPolicyWDLInferenceOwner:
    key = (
        id(model),
        str(device),
        tuple(int(value) for value in observation_shape),
        int(policy_size),
        int(wdl_size),
        str(forward_key),
    )
    owner = _OWNER_CACHE.get(key)
    if owner is None:
        owner = BatchedPolicyWDLInferenceOwner(
            model,
            device=device,
            expected_observation_shape=tuple(int(value) for value in observation_shape),
            expected_policy_size=int(policy_size),
            wdl_size=int(wdl_size),
            forward_policy_wdl_logits=lambda batch: forward_policy_wdl_logits(model, batch),
        )
        _OWNER_CACHE[key] = owner
    elif owner.model is not model:
        raise RuntimeError("Arena inference owner cache model identity collision")
    return owner


def infer_policy_wdl_batch(
    model: torch.nn.Module,
    cpu_batch: torch.Tensor,
    device: torch.device,
    *,
    observation_shape: tuple[int, ...],
    policy_size: int,
    wdl_size: int,
    forward_key: str,
    forward_policy_wdl_logits: Callable[[torch.nn.Module, torch.Tensor], object],
) -> ArenaInferenceBatch:
    """Run one Arena batch through the canonical parent-side inference owner."""

    device = torch.device(device)
    owner = _owner_for(
        model,
        device=device,
        observation_shape=observation_shape,
        policy_size=policy_size,
        wdl_size=wdl_size,
        forward_key=forward_key,
        forward_policy_wdl_logits=forward_policy_wdl_logits,
    )
    result = owner.evaluate_shared_batch(cpu_batch)

    d2h_started = time.perf_counter()
    policy = result.policy.detach().to("cpu", non_blocking=device.type == "cuda")
    wdl = result.wdl.detach().to("cpu", non_blocking=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    d2h_finished = time.perf_counter()

    h2d_started = result.h2d_started_at
    h2d_finished = result.h2d_finished_at
    forward_started = result.forward_started_at
    forward_finished = result.forward_finished_at
    timing = {
        "h2d_ms": max(0.0, float(h2d_finished) - float(h2d_started)) * 1000.0,
        "forward_ms": max(0.0, float(forward_finished) - float(forward_started)) * 1000.0,
        "d2h_ms": max(0.0, d2h_finished - d2h_started) * 1000.0,
    }
    return ArenaInferenceBatch(policy=policy, wdl=wdl, timing=timing, owner=owner)


def clear_arena_inference_owner_cache() -> None:
    """Test-only cache reset; production owns at most the models in this process."""

    _OWNER_CACHE.clear()


__all__ = [
    "ArenaInferenceBatch",
    "clear_arena_inference_owner_cache",
    "infer_policy_wdl_batch",
]
