from types import SimpleNamespace

import pytest
import torch

from alphazero.envs.gocube.finite_guards import (
    NonFiniteTrainingError,
    ensure_finite_gradients,
    ensure_finite_losses,
    ensure_finite_optimizer_state,
    ensure_finite_outputs,
    ensure_finite_parameters,
)


def test_finite_guards_reject_nonfinite_network_outputs_and_losses():
    with pytest.raises(NonFiniteTrainingError, match="network value output"):
        ensure_finite_outputs((torch.zeros(1, 2), torch.tensor([[float("nan")]])))
    with pytest.raises(NonFiniteTrainingError, match="score loss"):
        ensure_finite_losses({"score": torch.tensor(float("inf"))})


def test_finite_guards_cover_gradients_parameters_and_optimizer_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(NonFiniteTrainingError, match="gradient"):
        ensure_finite_gradients(SimpleNamespace(named_parameters=lambda: [("p", parameter)]))

    parameter.grad = None
    parameter.data.fill_(float("inf"))
    with pytest.raises(NonFiniteTrainingError, match="parameter"):
        ensure_finite_parameters(SimpleNamespace(named_parameters=lambda: [("p", parameter)]))

    parameter.data.fill_(1.0)
    loss = parameter.square().sum()
    loss.backward()
    optimizer.step()
    optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(NonFiniteTrainingError, match="optimizer state"):
        ensure_finite_optimizer_state(optimizer)
