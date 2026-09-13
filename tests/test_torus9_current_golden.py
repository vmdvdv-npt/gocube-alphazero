from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

import gocube_golden as g
from gocube_golden.arena_contract import SearchSettings
from gocube_golden.provenance import derive_seed
from gocube_golden.search import SequentialPUCT
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_DIRICHLET_ALPHA,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    current_torus9_selfplay_contract_fingerprint,
    load_torus9_current_profile,
)


def _states() -> tuple[g.GoldenState, ...]:
    first = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    second = g.apply_action(first, 0).after
    third = g.apply_action(second, 1).after
    return first, second, third


def _prepared(states: tuple[g.GoldenState, ...]):
    return tuple((state, g.prepare_legal_actions(state)) for state in states)


def _assert_evaluations_close(left, right) -> None:
    assert len(left) == len(right)
    for expected, actual in zip(left, right):
        assert torch.allclose(torch.tensor(expected.policy), torch.tensor(actual.policy), atol=1e-7, rtol=1e-6)
        assert torch.allclose(torch.tensor(expected.wdl), torch.tensor(actual.wdl), atol=1e-7, rtol=1e-6)


def test_current_profile_is_golden_source_and_excludes_legacy_defaults():
    profile = load_torus9_current_profile()
    assert profile["profile_id"] == TORUS9_CURRENT_PROFILE_ID
    assert profile["network"]["hidden"] == 80
    assert profile["network"]["blocks"] == 8
    assert profile["network"]["ownership"] is True
    assert profile["network"]["score"] is True
    assert profile["self_play"]["dirichlet_alpha"] == 0.11
    assert profile["rules"]["komi"] == 0.5
    assert profile["rules"]["legacy_komi_sentinel"] == 7.5
    assert profile["rules"]["legacy_komi_sentinel_policy"] == "reject-fail-closed"
    assert profile["training"]["batch_size"] == 64
    assert profile["training"]["optimizer_steps_per_iteration"] == 80
    assert profile["training"]["samples_consumed_per_iteration"] == 5120
    assert profile["replay"]["generations"] == 3
    assert profile["replay"]["cap"] == 20_000


def test_current_inference_batching_is_equivalent_for_different_groupings():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model = g.Torus9CurrentGraphNet().eval()
    states = _states()
    contexts = [context for _, context in _prepared(states)]
    direct = g.Torus9NeuralEvaluator(model)
    expected = tuple(direct.evaluate_prepared(state, context) for state, context in zip(states, contexts))

    grouped = g.Torus9NeuralEvaluator(model)
    grouped_a = grouped.evaluate_prepared_batch(states[:1], contexts[:1])
    grouped_b = grouped.evaluate_prepared_batch(states[1:], contexts[1:])
    _assert_evaluations_close(expected, grouped_a + grouped_b)
    assert grouped.inference_batch_rows == [1, 2]


def test_coordinator_routes_rows_without_cross_tree_or_order_mixup():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2027)
        model = g.Torus9CurrentGraphNet().eval()
    states = _states()
    prepared = _prepared(states)
    expected_evaluator = g.Torus9NeuralEvaluator(model)
    expected = tuple(expected_evaluator.evaluate_prepared(state, context) for state, context in prepared)

    coordinator = g.Torus9InferenceCoordinator(model, batch_cap=4, wait_ms=10.0)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(coordinator.evaluate_prepared, state, context) for state, context in prepared]
            actual = tuple(future.result() for future in futures)
        _assert_evaluations_close(expected, actual)
        telemetry = coordinator.telemetry
    finally:
        coordinator.close()
    assert telemetry["total_rows"] == 3
    assert telemetry["forward_calls"] >= 1
    assert telemetry["max_batch_rows"] <= 4


def test_batching_does_not_change_mcts_results_or_semantic_seeds():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2028)
        model = g.Torus9CurrentGraphNet().eval()
    states = _states()[:2]
    settings = SearchSettings(simulations=3, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)
    individual = tuple(
        SequentialPUCT(settings, adapter=g.GoldenSearchAdapter()).search(
            state, g.Torus9NeuralEvaluator(model), seed=derive_seed(77, index)
        )
        for index, state in enumerate(states)
    )
    grouped = g.Torus9BatchedPUCT(settings, adapter=g.GoldenSearchAdapter(), max_batch_rows=8).search(
        states,
        (g.Torus9NeuralEvaluator(model), g.Torus9NeuralEvaluator(model)),
        seeds=(derive_seed(77, 0), derive_seed(77, 1)),
    )
    assert [result.root_visits for result in grouped] == [result.root_visits for result in individual]
    assert [result.action for result in grouped] == [result.action for result in individual]

    contract = g.Torus9SelfPlaySearchContract(
        contract_id="torus9-golden-current-selfplay-search-v1",
        dirichlet_alpha=TORUS9_CURRENT_DIRICHLET_ALPHA,
    )
    contract.validate()
    assert contract.fingerprint == current_torus9_selfplay_contract_fingerprint(0.11)
    assert current_torus9_selfplay_contract_fingerprint(0.11) == current_torus9_selfplay_contract_fingerprint(0.11)
    assert TORUS9_CURRENT_TARGET_FINGERPRINT == load_torus9_current_profile()["target"]["fingerprint"]
    seeds = [derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, "run", f"game-{index}", "game") for index in range(4)]
    assert seeds == [derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, "run", f"game-{index}", "game") for index in range(4)]
