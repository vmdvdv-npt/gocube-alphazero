from __future__ import annotations

import copy

import torch

import gocube_golden as g
from gocube_golden.arena_contract import SearchSettings
from gocube_golden.search import SequentialPUCT
from gocube_golden.torus9_contract import (
    TORUS9_OBSERVATION_FINGERPRINT,
    TORUS9_SELFPLAY_CONTRACT_FINGERPRINT,
    TORUS9_TARGET_CONTRACT_ID,
    TORUS9_TARGET_FINGERPRINT,
)


def _sample(sample_id: str) -> dict[str, object]:
    state = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    final = g.apply_action(g.apply_action(state, g.PASS).after, g.PASS).after
    context = g.prepare_legal_actions(state)
    observation = g.build_torus9_observation(state, legal_context=context)
    return {
        "run_id": "ownership-ab-test",
        "game_id": "game-00",
        "ply": 1,
        "state": g.torus9_state_identity(state),
        "side_to_move": "BLACK",
        "observation": observation.tolist(),
        "legal_action_mask": list(context.action_mask),
        "root_visits": [1] * 82,
        "pi": [1.0 / 82.0] * 82,
        "z": [0.0, 1.0, 0.0],
        "model_hash": "sha256:" + "0" * 64,
        "selfplay_contract_fingerprint": TORUS9_SELFPLAY_CONTRACT_FINGERPRINT,
        "observation_fingerprint": TORUS9_OBSERVATION_FINGERPRINT,
        "target_contract_id": TORUS9_TARGET_CONTRACT_ID,
        "target_fingerprint": TORUS9_TARGET_FINGERPRINT,
        "ownership_target": list(g.torus9_ownership_target(final, g.BLACK)),
        "ownership_target_contract_id": "golden-ownership-final-state-side-to-move-v1",
        "auxiliary_target_source": "golden-referee-final-state-v1",
        "source_generation": 9,
        "replay_row_id": sample_id,
    }


def test_ownership_ab_trains_same_fixed_budget_but_only_b_moves_ownership_head():
    torch.set_num_threads(1)
    rows = [_sample("row-0"), _sample("row-1")]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        initial = g.Torus9OwnershipGraphNet(hidden=4, blocks=1)
    left = copy.deepcopy(initial)
    right = copy.deepcopy(initial)
    left_trainer = g.Torus9OwnershipTrainer(left, ownership_loss_enabled=False)
    right_trainer = g.Torus9OwnershipTrainer(right, ownership_loss_enabled=True)
    left_metrics = left_trainer.train_fixed_budget(rows, seed=17)
    right_metrics = right_trainer.train_fixed_budget(rows, seed=17)

    assert left_metrics["optimizer_steps"] == right_metrics["optimizer_steps"] == 80
    assert left_metrics["samples_consumed"] == right_metrics["samples_consumed"] == 5120
    assert left_metrics["ownership_loss_weight"] == 0.0
    assert right_metrics["ownership_loss_weight"] == 1.0
    assert all(torch.equal(initial.state_dict()[name], left.state_dict()[name]) for name in initial.state_dict() if name.startswith("ownership_head."))
    assert any(not torch.equal(initial.state_dict()[name], right.state_dict()[name]) for name in initial.state_dict() if name.startswith("ownership_head."))
    assert any(not torch.equal(left.state_dict()[name], right.state_dict()[name]) for name in left.state_dict() if not name.startswith("ownership_head."))


def test_batched_torus9_puct_matches_sequential_root_visits_and_records_rows():
    torch.set_num_threads(1)
    state_a = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    state_b = g.apply_action(state_a, 0).after
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = g.Torus9GraphNet(hidden=4, blocks=1).eval()
    settings = SearchSettings(simulations=3, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)
    sequential_evaluator = g.Torus9NeuralEvaluator(model)
    sequential = [
        SequentialPUCT(settings, adapter=g.GoldenSearchAdapter()).search(state, sequential_evaluator, seed=3)
        for state in (state_a, state_b)
    ]
    batched_evaluator = g.Torus9NeuralEvaluator(model)
    batched = g.Torus9BatchedPUCT(settings, adapter=g.GoldenSearchAdapter(), max_batch_rows=8)
    parallel = batched.search((state_a, state_b), (batched_evaluator, batched_evaluator), seeds=(3, 3))
    assert [result.root_visits for result in parallel] == [result.root_visits for result in sequential]
    assert batched.inference_batch_rows[0] == 2
    assert min(batched.inference_batch_rows) == 2
    assert batched_evaluator.nn_evaluations == sum(batched.inference_batch_rows)
