from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gocube_golden.cube_neural import (
    GoldenCubeGraphNetV1,
    GoldenCubeNeuralEvaluator,
    SelfPlayCubeRootNoiseEvaluator,
    build_cube_action_mask,
    build_cube_observation,
)
from gocube_golden.cube_topology import CUBE4_TOPOLOGY
from gocube_golden.cube_training import (
    CUBE_ACTION_COUNT,
    CUBE_TARGET_FINGERPRINT,
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    CubeTrainingSample,
    cube_initial_state,
    cube_state_identity,
)
from gocube_golden.diagnostics import operation_stats
from gocube_golden.rules import (
    IllegalMoveError,
    LegalActionContext,
    apply_action,
    legal_actions,
    prepare_legal_actions,
    reference_apply_action,
    reference_legal_actions,
)
from gocube_golden.search import Evaluation, SequentialPUCT
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, EMPTY, WHITE, research_state_from_stones


def _synthetic_state(history_length: int = 1):
    history = []
    for index in range(history_length):
        board = [0] * 96
        for bit in range(10):
            if ((index + 1) >> bit) & 1:
                board[bit] = 1
        for point in range(24):
            if board[point] == 0 and (point + index) % 7 < 3:
                board[point] = 2 if point % 2 else 1
        history.append(tuple(board))
    return research_state_from_stones(
        history[-1],
        side_to_move=BLACK if history_length % 2 else WHITE,
        topology=CUBE4_TOPOLOGY,
        superko_history=tuple(history),
    )


def test_precomputed_legality_is_exact_and_reused_by_observation():
    state = _synthetic_state(25)
    context = prepare_legal_actions(state)
    assert context.actions == legal_actions(state)
    assert context.action_mask == build_cube_action_mask(state)
    assert torch.equal(
        build_cube_observation(state),
        build_cube_observation(state, legal_context=context),
    )
    assert torch.equal(
        build_cube_observation(state),
        build_cube_observation(state, legal_action_mask=context.action_mask),
    )


def test_trusted_transition_matches_fully_validated_reference_for_every_action():
    states = [cube_initial_state(), _synthetic_state(1), _synthetic_state(100)]
    for state in states:
        assert legal_actions(state) == reference_legal_actions(state)
        for action in tuple(range(96)) + ("PASS",):
            try:
                actual = apply_action(state, action)
            except IllegalMoveError as actual_error:
                with pytest.raises(IllegalMoveError) as expected_error:
                    reference_apply_action(state, action)
                assert actual_error.reason == expected_error.value.reason
                continue
            expected = reference_apply_action(state, action)
            assert actual.action == expected.action
            assert actual.captured == expected.captured
            assert actual.after.stones == expected.after.stones
            assert actual.after.side_to_move == expected.after.side_to_move
            assert actual.after.superko_history == expected.after.superko_history
            assert actual.after.state_key == expected.after.state_key


class _ReferenceAdapter:
    def prepare_legal_actions(self, state):
        actions = reference_legal_actions(state)
        mask = [False] * 97
        for action in actions:
            mask[96 if action == "PASS" else int(action)] = True
        return LegalActionContext(state.state_key, actions, tuple(mask))

    def apply_action(self, state, action):
        return reference_apply_action(state, action).after

    def is_terminal(self, state):
        return state.is_terminal

    def terminal_utility(self, state):
        return GoldenSearchAdapter().terminal_utility(state)

    def action_index(self, state, action):
        return 96 if action == "PASS" else int(action)

    def action_space(self, state):
        return tuple(range(96)) + ("PASS",)


class _ReferenceEvaluator:
    """Reference evaluator with the old one-scan observation boundary."""

    def __init__(self, model):
        self.model = model

    def evaluate(self, state):
        actions = reference_legal_actions(state)
        observation = build_cube_observation(state, legal_actions=actions)
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observation.unsqueeze(0))
            policy = torch.softmax(policy_logits[0], dim=0)
            value = torch.softmax(value_logits[0], dim=0)
        return Evaluation(
            policy=tuple(float(item) for item in policy),
            wdl=tuple(float(item) for item in value),
        )


def test_optimized_search_matches_reference_search_exactly():
    torch.manual_seed(2026091401)
    state = _synthetic_state(10)
    model = GoldenCubeGraphNetV1()
    optimized = SequentialPUCT(DEFAULT_CUBE_SELFPLAY_CONTRACT.puct_settings).search(
        state, GoldenCubeNeuralEvaluator(model), seed=123
    )
    reference = SequentialPUCT(
        DEFAULT_CUBE_SELFPLAY_CONTRACT.puct_settings, adapter=_ReferenceAdapter()
    ).search(state, _ReferenceEvaluator(model), seed=123)
    assert optimized.action == reference.action
    assert optimized.legal_actions == reference.legal_actions
    assert optimized.legal_action_mask == reference.legal_action_mask
    assert optimized.root_visits == reference.root_visits
    assert optimized.pi == reference.pi
    assert optimized.root_q == reference.root_q


def test_search_has_one_legality_calculation_per_expanded_node_and_reuses_root_noise():
    state = cube_initial_state()
    model = GoldenCubeGraphNetV1()
    evaluator = GoldenCubeNeuralEvaluator(model)
    wrapped = SelfPlayCubeRootNoiseEvaluator(evaluator, state, seed=123)
    with operation_stats() as stats:
        result = SequentialPUCT(DEFAULT_CUBE_SELFPLAY_CONTRACT.puct_settings).search(
            state, wrapped, seed=123
        )
    counters = stats.to_dict()
    assert result.legal_action_mask == build_cube_action_mask(
        state, legal_action_mask=result.legal_action_mask
    )
    assert counters["leaf_expansions"] == result.evaluator_calls == 65
    assert counters["legal_calculations"] == counters["leaf_expansions"]
    assert counters["legal_actions_calls"] == 0
    assert counters["root_noise_legal_reuses"] == 1
    assert counters["root_noise_legal_scans"] == 0
    assert counters["full_history_validations"] == 0
    assert counters["trusted_state_constructions"] == 64
    assert counters["apply_action_calls"] == 64


def test_long_history_membership_index_is_exact():
    state = _synthetic_state(200)
    assert state.superko_membership == frozenset(state.superko_history)
    assert all((position in state.superko_membership) == (position in state.superko_history) for position in state.superko_history)


def _sample_for_audit() -> CubeTrainingSample:
    state = cube_initial_state()
    observation = build_cube_observation(state)
    visits = [0] * CUBE_ACTION_COUNT
    visits[96] = 64
    policy = [0.0] * CUBE_ACTION_COUNT
    policy[96] = 1.0
    return CubeTrainingSample(
        run_id="audit",
        game_id="audit-game",
        ply=1,
        state=cube_state_identity(state),
        side_to_move=state.side_to_move.name,
        observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
        legal_action_mask=build_cube_action_mask(state),
        root_visits=tuple(visits),
        pi=tuple(policy),
        z=(0.0, 1.0, 0.0),
        model_hash="sha256:model",
        selfplay_contract_fingerprint=DEFAULT_CUBE_SELFPLAY_CONTRACT.fingerprint,
        target_fingerprint=CUBE_TARGET_FINGERPRINT,
    )


def test_deep_evidence_audit_rejects_corrupted_history_mask_pi_z_and_model_hash():
    sample = _sample_for_audit()
    sample.validate()

    corrupted_state = dict(sample.state)
    corrupted_state["superko_history"] = [list(sample.state["superko_history"][0])] * 2
    with pytest.raises(ValueError):
        replace(sample, state=corrupted_state).validate()

    corrupted_mask = list(sample.legal_action_mask)
    corrupted_mask[0] = False
    with pytest.raises(ValueError):
        replace(sample, legal_action_mask=tuple(corrupted_mask)).validate()

    corrupted_pi = list(sample.pi)
    corrupted_pi[0] = 0.5
    with pytest.raises(ValueError):
        replace(sample, pi=tuple(corrupted_pi)).validate()

    with pytest.raises(ValueError, match="z target"):
        replace(sample, z=(1.0, 1.0, 0.0)).validate()

    with pytest.raises(ValueError, match="model hash"):
        sample.validate(expected_model_hash="sha256:other")
