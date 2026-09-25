import torch

from gocube_golden.torus9_m137_5ch import convert_m137_model
from gocube_golden.torus9_monolith import Torus9CurrentGraphNet
from gocube_golden.torus9_new_komi import (
    NEW_KOMI_LINEAGE_ID,
    NEW_KOMI_OPTIMIZER_CONVERSION,
    NEW_KOMI_REPLAY_POLICY,
    migrate_m137_adam,
)


def _prime_adam(model: Torus9CurrentGraphNet) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    observation = torch.zeros((2, 6, 81), dtype=torch.float32)
    observation[:, 5, :].fill_(0.5)
    policy, value, ownership, score = model.forward_auxiliary(observation)
    loss = (
        policy.square().mean()
        + value.square().mean()
        + ownership.square().mean()
        + score.square().mean()
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return optimizer


def test_new_komi_identity_is_explicit_and_fresh_history() -> None:
    assert NEW_KOMI_LINEAGE_ID == "new_komi"
    assert NEW_KOMI_REPLAY_POLICY == "fresh-only-no-parent-history"
    assert "reset-folded-bias-moments" in NEW_KOMI_OPTIMIZER_CONVERSION


def test_m137_adam_migration_preserves_exact_states_and_resets_only_folded_bias() -> None:
    torch.manual_seed(137)
    source = Torus9CurrentGraphNet()
    source_optimizer = _prime_adam(source)
    source_state = source_optimizer.state_dict()
    target = convert_m137_model(source)

    target_optimizer, report = migrate_m137_adam(
        source_model=source,
        target_model=target,
        source_optimizer_state=source_state,
    )
    target_state = target_optimizer.state_dict()

    source_names = [name for name, _ in source.named_parameters()]
    source_ids = source_state["param_groups"][0]["params"]
    target_ids = target_state["param_groups"][0]["params"]
    source_by_name = {
        name: source_state["state"][parameter_id]
        for name, parameter_id in zip(source_names, source_ids)
    }
    target_by_name = {
        name: target_state["state"][parameter_id]
        for name, parameter_id in zip(source_names, target_ids)
    }

    source_weight = source_by_name["input_projection.weight"]
    target_weight = target_by_name["input_projection.weight"]
    assert torch.equal(target_weight["exp_avg"], source_weight["exp_avg"][:, :5])
    assert torch.equal(target_weight["exp_avg_sq"], source_weight["exp_avg_sq"][:, :5])
    assert int(target_weight["step"].item()) == int(source_weight["step"].item())

    source_bias = source_by_name["input_projection.bias"]
    target_bias = target_by_name["input_projection.bias"]
    assert torch.count_nonzero(target_bias["exp_avg"]).item() == 0
    assert torch.count_nonzero(target_bias["exp_avg_sq"]).item() == 0
    assert int(target_bias["step"].item()) == int(source_bias["step"].item())

    unchanged_name = next(
        name
        for name in source_names
        if name not in {"input_projection.weight", "input_projection.bias"}
    )
    for key, value in source_by_name[unchanged_name].items():
        migrated = target_by_name[unchanged_name][key]
        if torch.is_tensor(value):
            assert torch.equal(migrated, value)
        else:
            assert migrated == value

    steps = {
        int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
        for state in target_state["state"].values()
        if "step" in state
    }
    assert len(steps) == 1
    assert report["reset_parameter_moments"] == ["input_projection.bias"]
    assert report["cropped_parameter_states"] == ["input_projection.weight"]
