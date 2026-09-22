"""Shared parent-side policy/WDL inference owner.

Topology adapters validate their model identity before constructing this
component.  The owner then performs the common tensor transport, forward,
normalization, shape and finite-value checks exactly once.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import torch

from .selfplay_engine import SharedInferenceResult


def _extract_policy_wdl(value: object) -> tuple[Any, Any]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    policy = getattr(value, "policy_logits", None)
    wdl = getattr(value, "wdl_logits", None)
    if policy is not None and wdl is not None:
        return policy, wdl
    raise ValueError("Policy/WDL forward callback must return a pair or logits object")


class BatchedPolicyWDLInferenceOwner:
    """One generic owner for the policy/WDL heads used during search."""

    def __init__(
        self,
        model: object,
        *,
        device: str | torch.device,
        expected_observation_shape: tuple[int, ...],
        expected_policy_size: int,
        wdl_size: int = 3,
        forward_policy_wdl_logits: Callable[[Any], object] | None = None,
    ) -> None:
        if not expected_observation_shape or any(int(value) <= 0 for value in expected_observation_shape):
            raise ValueError("Inference owner observation shape must be positive")
        if int(expected_policy_size) <= 0 or int(wdl_size) <= 0:
            raise ValueError("Inference owner output dimensions must be positive")
        self.model = model
        self.device = torch.device(device)
        self.expected_observation_shape = tuple(int(value) for value in expected_observation_shape)
        self.expected_policy_size = int(expected_policy_size)
        self.wdl_size = int(wdl_size)
        self._forward_policy_wdl_logits = forward_policy_wdl_logits or self._default_forward
        if not callable(self._forward_policy_wdl_logits):
            raise ValueError("Inference owner forward callback must be callable")
        to = getattr(self.model, "to", None)
        if callable(to):
            to(self.device)
        eval_method = getattr(self.model, "eval", None)
        if callable(eval_method):
            eval_method()

    def _default_forward(self, batch: Any) -> object:
        forward = getattr(self.model, "forward", None)
        if not callable(forward):
            raise ValueError("Inference owner model has no forward method")
        return forward(batch)

    def evaluate_shared_batch(self, observations: Any) -> SharedInferenceResult:
        if not isinstance(observations, torch.Tensor):
            raise ValueError("Shared observations must be a torch.Tensor")
        if observations.ndim != len(self.expected_observation_shape) + 1:
            raise ValueError("Shared observations must include exactly one batch dimension")
        if tuple(observations.shape[1:]) != self.expected_observation_shape:
            raise ValueError(
                "Shared observation shape drift: "
                f"expected [batch,{','.join(str(value) for value in self.expected_observation_shape)}]"
            )
        if observations.dtype != torch.float32 or not bool(torch.isfinite(observations).all()):
            raise ValueError("Shared observations must be finite float32")
        rows = int(observations.shape[0])
        if rows <= 0:
            raise ValueError("Shared inference batch must be non-empty")

        h2d_started = time.perf_counter()
        device_observations = observations.to(self.device, non_blocking=self.device.type == "cuda")
        h2d_finished = time.perf_counter()
        forward_started = time.perf_counter()
        with torch.inference_mode():
            raw_policy, raw_wdl = _extract_policy_wdl(self._forward_policy_wdl_logits(device_observations))
            policies = torch.softmax(raw_policy, dim=1)
            wdls = torch.softmax(raw_wdl, dim=1)
        forward_finished = time.perf_counter()

        if tuple(policies.shape) != (rows, self.expected_policy_size):
            raise ValueError("Central policy head shape drift")
        if tuple(wdls.shape) != (rows, self.wdl_size):
            raise ValueError("Central WDL head shape drift")
        outputs = torch.cat((policies, wdls), dim=1)
        if not bool(torch.isfinite(outputs).all()) or bool((outputs < 0.0).any()):
            raise ValueError("Central inference produced invalid probabilities")
        # Softmax should guarantee this, but the explicit check catches custom
        # kernels and dtype/device regressions at the common boundary.
        if any(not math.isfinite(float(value)) for value in outputs.detach().to("cpu").flatten().tolist()):
            raise ValueError("Central inference produced non-finite probabilities")
        return SharedInferenceResult(
            policy=policies,
            wdl=wdls,
            h2d_started_at=h2d_started,
            h2d_finished_at=h2d_finished,
            forward_started_at=forward_started,
            forward_finished_at=forward_finished,
        )


__all__ = ["BatchedPolicyWDLInferenceOwner"]
