from __future__ import annotations

import pytest

from gocube_golden.cube_training import (
    CUBE_ACTION_COUNT,
    CUBE_WATCHDOG,
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    cube_post_action_termination,
    cube_state_identity,
    cube_z_target,
    validate_cube_policy_target,
)
from gocube_golden.cube_neural import build_cube_observation, build_cube_action_mask
from gocube_golden.cube_topology import CUBE4_TOPOLOGY
from gocube_golden.cube_training import cube_initial_state
from gocube_golden.state import BLACK, WHITE


def test_cube_contract_freezes_search_exploration_and_watchdog():
    contract = DEFAULT_CUBE_SELFPLAY_CONTRACT
    contract.validate()
    assert contract.simulations == 64
    assert contract.cpuct == 1.25
    assert contract.fpu == 0.0
    assert contract.dirichlet_epsilon == 0.25
    assert contract.dirichlet_alpha == 0.30
    assert contract.temperature_plies == (1, 8)
    assert contract.temperature_after == 0.0
    assert contract.watchdog == 20 * CUBE4_TOPOLOGY.point_count == CUBE_WATCHDOG == 1920


def test_double_pass_takes_precedence_over_action_1920_watchdog():
    state = cube_initial_state()
    state = __import__("gocube_golden.rules", fromlist=["apply_action"]).apply_action(state, "PASS").after
    state = __import__("gocube_golden.rules", fromlist=["apply_action"]).apply_action(state, "PASS").after
    assert cube_post_action_termination(state, CUBE_WATCHDOG) == "DOUBLE_PASS"


def test_wdl_targets_are_strictly_side_to_move_perspective():
    assert cube_z_target("BLACK", BLACK) == (1.0, 0.0, 0.0)
    assert cube_z_target("BLACK", WHITE) == (0.0, 0.0, 1.0)
    assert cube_z_target("WHITE", WHITE) == (1.0, 0.0, 0.0)
    assert cube_z_target("WHITE", BLACK) == (0.0, 0.0, 1.0)
    assert cube_z_target("DRAW", BLACK) == (0.0, 1.0, 0.0)


def test_policy_target_requires_97_actions_and_zero_illegal_mass():
    state = cube_initial_state()
    visits = [0] * CUBE_ACTION_COUNT
    visits[96] = 64
    policy = [0.0] * CUBE_ACTION_COUNT
    policy[96] = 1.0
    validate_cube_policy_target(state, policy, visits)
    with pytest.raises(ValueError):
        validate_cube_policy_target(state, policy[:-1], visits)


def test_replay_audit_rejects_observation_value_drift():
    from dataclasses import replace

    from gocube_golden.cube_training import CubeTrainingSample, CUBE_TARGET_FINGERPRINT
    from gocube_golden.cube_neural import CUBE_OBSERVATION_FINGERPRINT

    state = cube_initial_state()
    policy = [0.0] * CUBE_ACTION_COUNT
    policy[96] = 1.0
    visits = [0] * CUBE_ACTION_COUNT
    visits[96] = 64
    observation = build_cube_observation(state)
    sample = CubeTrainingSample(
        run_id="test-run",
        game_id="test-game",
        ply=1,
        state=cube_state_identity(state),
        side_to_move=state.side_to_move.name,
        observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
        legal_action_mask=build_cube_action_mask(state),
        root_visits=tuple(visits),
        pi=tuple(policy),
        z=(0.0, 1.0, 0.0),
        model_hash="sha256:test",
        selfplay_contract_fingerprint=DEFAULT_CUBE_SELFPLAY_CONTRACT.fingerprint,
        observation_fingerprint=CUBE_OBSERVATION_FINGERPRINT,
        target_fingerprint=CUBE_TARGET_FINGERPRINT,
    )
    sample.validate()
    tampered = replace(sample, observation=tuple(tuple(1.0 if value == 0.0 else value for value in row) for row in sample.observation))
    with pytest.raises(ValueError, match="observation values drift"):
        tampered.validate()
