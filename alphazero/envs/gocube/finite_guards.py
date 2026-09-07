from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class NonFiniteTrainingError(RuntimeError):
    """Raised when a value that would affect training is NaN or infinite."""


def ensure_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise NonFiniteTrainingError(f"Non-finite {name} detected")


def ensure_finite_outputs(outputs: Sequence[torch.Tensor]) -> None:
    names = ("policy", "value", "ownership", "score")
    for name, output in zip(names, outputs):
        ensure_finite_tensor(output, f"network {name} output")


def ensure_finite_losses(losses: Mapping[str, torch.Tensor]) -> None:
    for name, loss in losses.items():
        ensure_finite_tensor(loss, f"{name} loss")


def ensure_finite_gradients(module: torch.nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            ensure_finite_tensor(parameter.grad, f"gradient for {name}")


def ensure_finite_parameters(module: torch.nn.Module) -> None:
    for name, parameter in module.named_parameters():
        ensure_finite_tensor(parameter, f"parameter {name}")


def ensure_finite_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    for index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                ensure_finite_tensor(value, f"optimizer state {index}.{name}")
