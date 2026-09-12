from __future__ import annotations

import math

import pytest

import gocube_golden as g


def occupied_zero_state():
    return g.apply_action(g.initial_state(), 0).after


def policy_vector(state, fill=0.0):
    return [fill] * (state.topology.point_count + 1)


class FixedPolicyEvaluator:
    def __init__(self, policy):
        self.policy = policy

    def evaluate(self, state):
        return g.Evaluation(policy=self.policy, wdl=(0.5, 0.0, 0.5))


def run_policy(state, policy, *, simulations=8):
    return g.SequentialPUCT(g.SearchSettings(simulations=simulations)).search(
        state, FixedPolicyEvaluator(policy), seed=0
    )


def test_nan_in_legal_slot_is_rejected_before_masking():
    state = occupied_zero_state()
    policy = policy_vector(state)
    policy[1] = math.nan
    with pytest.raises(g.SearchError, match="Invalid policy weight"):
        run_policy(state, policy)


def test_nan_only_in_illegal_slot_is_rejected_before_masking():
    state = occupied_zero_state()
    policy = policy_vector(state)
    policy[0] = math.nan
    with pytest.raises(g.SearchError, match="Invalid policy weight"):
        run_policy(state, policy)


def test_positive_infinity_in_illegal_slot_is_rejected():
    state = occupied_zero_state()
    policy = policy_vector(state)
    policy[0] = math.inf
    with pytest.raises(g.SearchError, match="Invalid policy weight"):
        run_policy(state, policy)


def test_negative_infinity_is_rejected():
    state = occupied_zero_state()
    policy = policy_vector(state)
    policy[1] = -math.inf
    with pytest.raises(g.SearchError, match="Invalid policy weight"):
        run_policy(state, policy)


def test_negative_value_in_illegal_slot_is_rejected():
    state = occupied_zero_state()
    policy = policy_vector(state)
    policy[0] = -0.25
    with pytest.raises(g.SearchError, match="Invalid policy weight"):
        run_policy(state, policy)


def test_non_numeric_element_is_rejected():
    state = occupied_zero_state()
    policy = policy_vector(state)
    policy[0] = "not-a-number"
    with pytest.raises(g.SearchError, match="Non-numeric policy weight"):
        run_policy(state, policy)


@pytest.mark.parametrize("delta", [-1, 1])
def test_policy_sequence_requires_exact_action_size(delta):
    state = occupied_zero_state()
    size = state.topology.point_count + 1 + delta
    policy = [0.0] * size
    with pytest.raises(g.SearchError, match="policy length"):
        run_policy(state, policy)


def test_policy_mapping_requires_exact_action_space_and_validates_illegal_slots():
    state = occupied_zero_state()
    full = {action: 0.0 for action in g.GoldenSearchAdapter().action_space(state)}
    malformed = dict(full)
    malformed[0] = math.nan
    with pytest.raises(g.SearchError, match="Invalid policy weight"):
        run_policy(state, malformed)

    missing = dict(full)
    missing.pop(g.PASS)
    with pytest.raises(g.SearchError, match="policy length"):
        run_policy(state, missing)


def test_valid_zero_vector_uses_deterministic_uniform_legal_fallback():
    state = occupied_zero_state()
    policy = policy_vector(state)
    one = run_policy(state, policy, simulations=32)
    two = run_policy(state, policy, simulations=32)
    assert one.action in one.legal_actions
    assert one.root_visits[0] == 0
    assert one.root_visits == two.root_visits
    assert one.action == two.action
    assert sum(one.pi) == pytest.approx(1.0)


def test_finite_mass_on_illegal_action_is_masked_not_rejected():
    state = occupied_zero_state()
    policy = policy_vector(state)
    policy[0] = 1e300
    result = run_policy(state, policy, simulations=32)
    assert 0 not in result.legal_actions
    assert result.root_visits[0] == 0
    assert result.action in result.legal_actions
    assert sum(result.pi) == pytest.approx(1.0)
