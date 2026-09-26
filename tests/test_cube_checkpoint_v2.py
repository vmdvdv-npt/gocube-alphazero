from __future__ import annotations

import pytest
import torch

from gocube_golden.cube_checkpoint_v2 import _validate_adam_state


def _primed_two_parameter_adam() -> tuple[torch.optim.Adam, list[torch.nn.Parameter]]:
    parameters = [
        torch.nn.Parameter(torch.tensor(1.0)),
        torch.nn.Parameter(torch.tensor(2.0)),
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    loss = sum(parameter.square() for parameter in parameters)
    loss.backward()
    optimizer.step()
    return optimizer, parameters


def test_cube_checkpoint_rejects_missing_adam_state_for_trainable_parameter() -> None:
    optimizer, parameters = _primed_two_parameter_adam()
    del optimizer.state[parameters[1]]

    with pytest.raises(ValueError, match="missing"):
        _validate_adam_state(optimizer, updates=1)


def test_cube_checkpoint_rejects_adam_step_mismatch() -> None:
    optimizer, parameters = _primed_two_parameter_adam()
    optimizer.state[parameters[0]]["step"] = torch.tensor(0.0)

    with pytest.raises(ValueError, match="step.*counter"):
        _validate_adam_state(optimizer, updates=1)
