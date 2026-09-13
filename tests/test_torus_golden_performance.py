from __future__ import annotations

from dataclasses import dataclass
import random

import pytest
import torch

import gocube_golden as g
from gocube_golden.diagnostics import operation_stats
from gocube_golden.neural import (
    GoldenGraphNetV1,
    GoldenNeuralEvaluator,
    SelfPlayRootNoiseEvaluator,
    build_action_mask,
    build_observation,
)
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
from gocube_golden.state import BLACK, EMPTY, PASS, WHITE, GoldenState, initial_state, research_state_from_stones
from gocube_golden.training import (
    DEFAULT_SELFPLAY_CONTRACT,
    GoldenTrainer,
    GoldenTrainingSample,
    sample_action_from_visits,
    state_identity,
)


def _synthetic_state(history_length: int = 1) -> GoldenState:
    history: list[tuple[int, ...]] = []
    for index in range(history_length):
        board = [int(EMPTY)] * 25
        for bit in range(5):
            if ((index + 1) >> bit) & 1:
                board[bit] = int(BLACK)
        for point in range(25):
            if board[point] == int(EMPTY) and (point + index) % 7 < 2:
                board[point] = int(WHITE if point % 2 else BLACK)
        history.append(tuple(board))
    return research_state_from_stones(
        history[-1],
        side_to_move=BLACK if history_length % 2 else WHITE,
        superko_history=tuple(history),
    )


CORPUS = (
    initial_state(),
    _synthetic_state(8),
    _synthetic_state(24),
    _synthetic_state(120),
)


class _ReferenceAdapter:
    """The pre-optimization boundary: every speculative operation is validated."""

    def prepare_legal_actions(self, state: GoldenState) -> LegalActionContext:
        actions = reference_legal_actions(state)
        mask = [False] * 26
        for action in actions:
            mask[25 if action == PASS else int(action)] = True
        return LegalActionContext(state.state_key, actions, tuple(mask))

    def apply_action(self, state: GoldenState, action: int | str) -> GoldenState:
        return reference_apply_action(state, action).after

    def is_terminal(self, state: GoldenState) -> bool:
        return state.is_terminal

    def terminal_utility(self, state: GoldenState) -> float:
        return GoldenSearchAdapter().terminal_utility(state)

    def action_index(self, state: GoldenState, action: int | str) -> int:
        return 25 if action == PASS else int(action)

    def action_space(self, state: GoldenState) -> tuple[int | str, ...]:
        return tuple(range(25)) + (PASS,)


class _ReferenceEvaluator:
    """Reference NN boundary with the old legality-aware observation call."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()
        self.nn_evaluations = 0

    def evaluate(self, state: GoldenState) -> Evaluation:
        actions = reference_legal_actions(state)
        observation = build_observation(state, legal_actions=actions)
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observation.unsqueeze(0))
            policy = torch.softmax(policy_logits[0], dim=0)
            wdl = torch.softmax(value_logits[0], dim=0)
        self.nn_evaluations += 1
        return Evaluation(
            policy=tuple(float(value) for value in policy),
            wdl=tuple(float(value) for value in wdl),
        )


def _assert_transition_equivalent(state: GoldenState, action: int | str) -> None:
    try:
        actual = apply_action(state, action)
    except IllegalMoveError as actual_error:
        with pytest.raises(IllegalMoveError) as expected_error:
            reference_apply_action(state, action)
        assert actual_error.reason == expected_error.value.reason
        return
    expected = reference_apply_action(state, action)
    assert actual.action == expected.action
    assert actual.captured == expected.captured
    assert actual.after.stones == expected.after.stones
    assert actual.after.side_to_move == expected.after.side_to_move
    assert actual.after.consecutive_passes == expected.after.consecutive_passes
    assert actual.after.superko_history == expected.after.superko_history
    assert actual.after.is_terminal == expected.after.is_terminal
    assert actual.after.state_key == expected.after.state_key
    if actual.after.is_terminal:
        assert g.result_from_terminal(actual.after) == g.result_from_terminal(expected.after)


def test_torus_rules_and_semantics_match_reference_for_all_corpus_actions() -> None:
    for state in CORPUS:
        assert legal_actions(state) == reference_legal_actions(state)
        assert build_action_mask(state) == tuple(
            action in legal_actions(state) for action in tuple(range(25)) + (PASS,)
        )
        for action in tuple(range(25)) + (PASS,):
            _assert_transition_equivalent(state, action)


def test_prepared_legality_is_exact_and_reused_by_observation_and_evaluator() -> None:
    state = _synthetic_state(12)
    context = prepare_legal_actions(state)
    assert context.actions == legal_actions(state)
    assert context.action_mask == build_action_mask(state, legal_context=context)
    assert torch.equal(build_observation(state), build_observation(state, legal_context=context))

    torch.manual_seed(2026091201)
    model = GoldenGraphNetV1()
    evaluator = GoldenNeuralEvaluator(model)
    with operation_stats() as ordinary_stats:
        ordinary = evaluator.evaluate(state)
    with operation_stats() as prepared_stats:
        prepared = evaluator.evaluate_prepared(state, context)
    assert ordinary == prepared
    assert ordinary_stats.legal_calculations == 1
    assert prepared_stats.legal_calculations == 0
    assert prepared_stats.legal_actions_calls == 0
    assert prepared_stats.observation_builds == 1


def test_prepared_context_rejects_a_different_state() -> None:
    context = prepare_legal_actions(initial_state())
    with pytest.raises(ValueError, match="another state"):
        build_observation(_synthetic_state(2), legal_context=context)


def test_optimized_search_matches_reference_search_exactly() -> None:
    torch.manual_seed(2026091202)
    state = _synthetic_state(10)
    model = GoldenGraphNetV1()
    settings = DEFAULT_SELFPLAY_CONTRACT.puct_settings
    optimized = SequentialPUCT(settings).search(
        state, GoldenNeuralEvaluator(model), seed=123
    )
    reference = SequentialPUCT(
        settings, adapter=_ReferenceAdapter()
    ).search(state, _ReferenceEvaluator(model), seed=123)
    assert optimized.action == reference.action
    assert optimized.legal_actions == reference.legal_actions
    assert optimized.legal_action_mask == reference.legal_action_mask
    assert optimized.root_visits == reference.root_visits
    assert optimized.pi == reference.pi
    assert optimized.root_q == reference.root_q
    assert optimized.evaluator_calls == reference.evaluator_calls
    assert optimized.simulations == reference.simulations


def test_search_has_one_legality_calculation_per_expanded_node_and_reuses_root_noise() -> None:
    state = initial_state()
    torch.manual_seed(2026091203)
    model = GoldenGraphNetV1()
    evaluator = GoldenNeuralEvaluator(model)
    wrapped = SelfPlayRootNoiseEvaluator(evaluator, state, seed=123)
    with operation_stats() as stats:
        result = SequentialPUCT(DEFAULT_SELFPLAY_CONTRACT.puct_settings).search(
            state, wrapped, seed=123
        )
    counters = stats.to_dict()
    assert result.legal_action_mask == build_action_mask(
        state, legal_action_mask=result.legal_action_mask
    )
    assert counters["leaf_expansions"] == result.evaluator_calls
    assert counters["legal_calculations"] == counters["leaf_expansions"]
    assert counters["legal_actions_calls"] == 0
    assert counters["root_noise_legal_reuses"] == 1
    assert counters["root_noise_legal_scans"] == 0
    assert counters["full_history_validations"] == 0
    assert counters["trusted_state_constructions"] == counters["apply_action_calls"]
    assert counters["trusted_state_constructions"] > 0


def test_long_history_membership_index_is_exact() -> None:
    state = _synthetic_state(200)
    assert state.superko_membership == frozenset(state.superko_history)
    assert all(
        (position in state.superko_membership) == (position in state.superko_history)
        for position in state.superko_history
    )


@dataclass(frozen=True)
class _TraceRow:
    before: tuple[object, ...]
    action: int | str
    pi: tuple[float, ...]
    captures: tuple[int, ...]
    after: tuple[object, ...]


def _play_trace(
    model: torch.nn.Module,
    *,
    adapter: GoldenSearchAdapter | _ReferenceAdapter,
    reference: bool,
    seed: int,
    plies: int = 128,
) -> tuple[_TraceRow, ...]:
    state = initial_state()
    evaluator = _ReferenceEvaluator(model) if reference else GoldenNeuralEvaluator(model)
    rng = random.Random(seed)
    rows: list[_TraceRow] = []
    for ply in range(plies):
        search_seed = seed + ply * 1009
        wrapped = SelfPlayRootNoiseEvaluator(
            evaluator, state, seed=search_seed + 17
        )
        result = SequentialPUCT(
            DEFAULT_SELFPLAY_CONTRACT.puct_settings, adapter=adapter
        ).search(state, wrapped, seed=search_seed)
        action = sample_action_from_visits(result, temperature=0.0, rng=rng)
        if reference:
            transition = reference_apply_action(state, action)
        else:
            transition = apply_action(state, action)
        rows.append(_TraceRow(
            before=state.state_key,
            action=action,
            pi=result.pi,
            captures=transition.captured,
            after=transition.after.state_key,
        ))
        state = transition.after
        if state.is_terminal:
            break
    assert state.is_terminal, "deterministic trace did not reach a terminal state"
    return tuple(rows)


@pytest.mark.parametrize("trained", (False, True), ids=("M0", "trained"))
def test_full_game_deterministic_trace_is_identical_for_m0_and_trained_snapshots(trained: bool) -> None:
    torch.manual_seed(2026091210)
    model = GoldenGraphNetV1()
    if trained:
        state = initial_state()
        context = prepare_legal_actions(state)
        observation = build_observation(state, legal_context=context)
        sample = GoldenTrainingSample(
            run_id="parity-training",
            game_id="parity-training-game",
            ply=1,
            state=state_identity(state),
            side_to_move=state.side_to_move.name,
            observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
            legal_action_mask=context.action_mask,
            root_visits=tuple([1] * 26),
            pi=tuple([1.0 / 26.0] * 26),
            z=(1.0, 0.0, 0.0),
            model_hash="sha256:" + "0" * 64,
            selfplay_contract_fingerprint=DEFAULT_SELFPLAY_CONTRACT.fingerprint,
        )
        GoldenTrainer(model).train([sample], updates=2, batch_size=1, seed=17)
    optimized = _play_trace(
        model, adapter=GoldenSearchAdapter(), reference=False, seed=4001
    )
    reference = _play_trace(
        model, adapter=_ReferenceAdapter(), reference=True, seed=4001
    )
    assert optimized == reference
