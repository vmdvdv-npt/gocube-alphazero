from __future__ import annotations

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
    current_torus9_content_fingerprint,
    current_torus9_profile_fingerprint,
    current_torus9_selfplay_contract_fingerprint,
    load_torus9_current_profile,
)


REFERENCE_PROFILE_FINGERPRINT = "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775"
CURRENT_CONTENT_FINGERPRINT = "sha256:7e97c50e1697641fb8f5b9a3566144f0a58c105e3b688940f42e7b6154fb0831"


def _states():
    first = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    second = g.apply_action(first, 0).after
    third = g.apply_action(second, 1).after
    return first, second, third


def test_current_profile_is_reference_locked():
    profile = load_torus9_current_profile()
    assert profile["profile_id"] == TORUS9_CURRENT_PROFILE_ID
    assert current_torus9_profile_fingerprint(profile) == REFERENCE_PROFILE_FINGERPRINT
    assert current_torus9_content_fingerprint(profile) == CURRENT_CONTENT_FINGERPRINT
    assert profile["network"]["hidden"] == 80
    assert profile["network"]["blocks"] == 8
    assert profile["network"]["ownership"] is True
    assert profile["network"]["score"] is True
    assert profile["self_play"]["dirichlet_alpha"] == 0.11
    assert profile["rules"]["komi"] == 0.5
    assert profile["training"]["model_gating"] is False


def test_current_neural_evaluator_batching_preserves_row_order():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model = g.Torus9CurrentGraphNet().eval()
    states = _states()
    prepared = tuple((state, g.prepare_legal_actions(state)) for state in states)
    evaluator = g.Torus9NeuralEvaluator(model)
    individual = tuple(evaluator.evaluate_prepared(state, context) for state, context in prepared)
    grouped = evaluator.evaluate_prepared_batch(
        [state for state, _ in prepared], [context for _, context in prepared]
    )
    assert len(grouped) == len(individual)
    for expected, actual in zip(individual, grouped):
        assert torch.allclose(torch.tensor(expected.policy), torch.tensor(actual.policy), atol=1e-7)
        assert torch.allclose(torch.tensor(expected.wdl), torch.tensor(actual.wdl), atol=1e-7)


def test_sequential_puct_is_the_only_torus9_search_core():
    state = _states()[0]
    model = g.Torus9CurrentGraphNet().eval()
    settings = SearchSettings(simulations=2, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)
    result = SequentialPUCT(settings, adapter=g.GoldenSearchAdapter()).search(
        state, g.Torus9NeuralEvaluator(model), seed=derive_seed(77, 0)
    )
    assert result.action in g.legal_actions(state)
    assert len(result.root_visits) == 82


def test_current_contract_and_seeds_are_deterministic():
    contract = g.Torus9SelfPlaySearchContract(
        contract_id="torus9-golden-current-selfplay-search-v1",
        dirichlet_alpha=TORUS9_CURRENT_DIRICHLET_ALPHA,
    )
    contract.validate()
    assert contract.fingerprint == current_torus9_selfplay_contract_fingerprint(0.11)
    assert TORUS9_CURRENT_TARGET_FINGERPRINT == load_torus9_current_profile()["target"]["fingerprint"]
    seeds = [
        derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, "run", f"game-{index}", "game")
        for index in range(4)
    ]
    assert seeds == [
        derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, "run", f"game-{index}", "game")
        for index in range(4)
    ]
