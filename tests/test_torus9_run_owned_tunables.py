from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from gocube_golden import (
    Torus9CurrentGraphNet,
    Torus9SelfPlaySearchContract,
    Torus9TrainingAdapter,
    load_torus9_current_profile,
)
from gocube_golden.run_spec import StrictRunSpec
from gocube_golden.torus9_contract import (
    current_torus9_content_fingerprint,
    profile_fingerprint,
    validate_torus9_current_profile,
)


def _run_profile(*, lr: float, replay_generations: int, replay_cap: int, simulations: int):
    profile = deepcopy(load_torus9_current_profile())
    profile["experiment"] = {"kind": "test-run-owned-tunables"}
    profile["training"]["learning_rate"] = lr
    profile["replay"]["window"] = f"rolling last {replay_generations} generations"
    profile["replay"]["generations"] = replay_generations
    profile["replay"]["cap"] = replay_cap
    profile["self_play"]["mcts_simulations"] = simulations
    profile["content_fingerprint"] = current_torus9_content_fingerprint(profile)
    profile["profile_fingerprint"] = profile_fingerprint(profile)
    return profile


@pytest.mark.parametrize(
    ("lr", "generations", "cap", "simulations"),
    [
        (0.0007, 2, 12345, 37),
        (0.00005, 9, 75000, 256),
        (0.003, 17, 150000, 511),
    ],
)
def test_run_owned_tunables_are_not_compared_with_golden(
    lr: float,
    generations: int,
    cap: int,
    simulations: int,
) -> None:
    profile = _run_profile(
        lr=lr,
        replay_generations=generations,
        replay_cap=cap,
        simulations=simulations,
    )
    validate_torus9_current_profile(profile)

    contract = Torus9SelfPlaySearchContract(simulations=simulations)
    contract.validate()
    assert contract.settings.simulations == simulations

    adapter = Torus9TrainingAdapter(profile=profile)
    state = adapter.create_state(Torus9CurrentGraphNet(), run_id="run-owned-test")
    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(lr)
    assert state.rolling_replay.generations == generations
    assert state.rolling_replay.maximum_positions == cap


def test_parent_adam_state_keeps_moments_but_run_lr_wins() -> None:
    profile = _run_profile(
        lr=0.0003,
        replay_generations=6,
        replay_cap=40000,
        simulations=128,
    )
    model = Torus9CurrentGraphNet()
    adapter = Torus9TrainingAdapter(profile=profile)
    state = adapter.create_state(model, run_id="adam-lr-override")

    parent_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    parent_optimizer.step()
    state.optimizer.load_state_dict(parent_optimizer.state_dict())

    first_parameter = next(iter(model.parameters()))
    moment_before = state.optimizer.state[first_parameter]["exp_avg"].detach().clone()
    step_before = int(state.optimizer.state[first_parameter]["step"].item())
    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)

    adapter._apply_effective_lr(state.optimizer)

    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.0003)
    assert int(state.optimizer.state[first_parameter]["step"].item()) == step_before
    assert torch.equal(state.optimizer.state[first_parameter]["exp_avg"], moment_before)


def test_torus9_run_spec_ignores_stale_expected_profile_fingerprint(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "configs" / "gocube" / "torus9_plateau_exit_production_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["expected_profile_fingerprint"] = "sha256:" + "0" * 64
    spec_path = tmp_path / "run-spec.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    spec = StrictRunSpec.load(spec_path, repo_root=repo_root)
    assert spec.orchestrator_spec.profile_payload["training"]["learning_rate"] == pytest.approx(0.0003)
    assert spec.orchestrator_spec.profile_payload["replay"]["generations"] == 6
    assert spec.orchestrator_spec.profile_payload["self_play"]["mcts_simulations"] == 128
    assert spec.orchestrator_spec.arena_every_generations == 5


def test_invalid_shapes_still_fail_without_golden_comparison() -> None:
    with pytest.raises(ValueError, match="positive"):
        Torus9SelfPlaySearchContract(simulations=0).validate()
