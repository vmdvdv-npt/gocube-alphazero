from __future__ import annotations

import copy

import torch

import gocube_golden as g


def _observation(stones: list[int]) -> torch.Tensor:
    state = g.research_state_from_stones(stones, topology=g.TORUS_9X9, side_to_move=g.BLACK, komi=0.5)
    return g.build_torus9_observation(state, legal_context=g.prepare_legal_actions(state))


def _sample(sample_id: str, value: list[float] | None = None) -> dict[str, object]:
    return {
        "observation": torch.zeros((6, 81), dtype=torch.float32).tolist(),
        "pi": (value or [1.0] + [0.0] * 81),
        "z": [1.0, 0.0, 0.0],
        "source_generation": 1,
        "replay_row_id": sample_id,
    }


def test_torus_diameters_and_eight_block_distant_dependency():
    assert g.graph_diameter(g.TORUS_5X5) == 4
    assert g.graph_diameter(g.TORUS_9X9) == 8
    assert g.TORUS9_BLOCKS == 8
    empty = [g.EMPTY] * 81
    distant = empty.copy()
    distant[0] = g.BLACK
    torch.manual_seed(20260914)
    four = g.Torus9GraphNet(blocks=4).eval()
    eight = g.Torus9GraphNet(blocks=8).eval()
    pair = torch.stack((_observation(empty), _observation(distant)))
    with torch.inference_mode():
        four_policy, _ = four(pair)
        eight_policy, _ = eight(pair)
    assert torch.equal(four_policy[0, 40], four_policy[1, 40])
    assert torch.equal(four_policy[0, 41], four_policy[1, 41])
    assert not torch.equal(eight_policy[0, 40], eight_policy[1, 40]) or not torch.equal(eight_policy[0, 41], eight_policy[1, 41])
    assert four.architecture_config["architecture_id"] != eight.architecture_config["architecture_id"] or four.blocks_count != eight.blocks_count


def test_rolling_replay_is_recent_three_generations_capped_and_provenanced():
    replay = g.Torus9RollingReplay(generations=3, maximum_positions=5)
    replay.append_generation(1, [_sample(f"g1-{i}") for i in range(3)])
    replay.append_generation(2, [_sample(f"g2-{i}") for i in range(3)])
    metrics = replay.append_generation(3, [_sample(f"g3-{i}") for i in range(2)])
    assert metrics["generations_represented"] == [2, 3]
    assert metrics["rolling_buffer_positions"] == 5
    assert metrics["eviction_count"] == 2
    assert metrics["total_evictions"] == 3
    assert [row["source_generation"] for row in replay.rows] == [2, 2, 2, 3, 3]
    assert [row["replay_row_id"] for row in replay.rows] == ["g2-0", "g2-1", "g2-2", "g3-0", "g3-1"]


def test_fixed_training_is_80_full_batches_deterministic_and_continuous():
    torch.set_num_threads(1)
    rows = [_sample("a"), _sample("b", [0.0, 1.0] + [0.0] * 80)]
    torch.manual_seed(42)
    left = g.Torus9GraphNet(hidden=4)
    right = copy.deepcopy(left)
    left_trainer = g.Torus9Trainer(left)
    right_trainer = g.Torus9Trainer(right)
    left_metrics = left_trainer.train_fixed_budget(rows, seed=17)
    right_metrics = right_trainer.train_fixed_budget(rows, seed=17)
    assert left_metrics["optimizer_steps"] == 80
    assert left_metrics["samples_consumed"] == 5120
    assert left_metrics["batch_sizes"] == [64] * 80
    assert left_metrics["reused_sample_rows"] > 0
    assert left_metrics["adam_step_before"] == 0
    assert left_metrics["adam_step_after"] == 80
    assert right_metrics["adam_step_after"] == 80
    assert all(torch.equal(a, b) for a, b in zip(left.state_dict().values(), right.state_dict().values()))
    left_metrics_2 = left_trainer.train_fixed_budget(rows, seed=18)
    assert left_metrics_2["adam_step_before"] == 80
    assert left_metrics_2["adam_step_after"] == 160


def test_eight_block_tiny_global_and_local_supervised_fixtures_reduce_loss():
    torch.set_num_threads(1)
    empty = [g.EMPTY] * 81
    marked = empty.copy()
    marked[0] = g.BLACK
    observations = torch.stack((_observation(empty), _observation(marked)))
    targets = torch.zeros((2, 82), dtype=torch.float32)
    targets[0, 40] = 1.0
    targets[1, 41] = 1.0
    torch.manual_seed(91)
    model = g.Torus9GraphNet(hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with torch.no_grad():
        initial = -(targets * torch.log_softmax(model(observations)[0], dim=1)).sum(dim=1).mean()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(observations)
        loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
    final = -(targets * torch.log_softmax(model(observations)[0], dim=1)).sum(dim=1).mean()
    assert float(final) < float(initial)

    local_target = torch.zeros((1, 82), dtype=torch.float32)
    local_target[0, 1] = 1.0
    local_optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    with torch.no_grad():
        local_initial = -(local_target * torch.log_softmax(model(observations[:1])[0], dim=1)).sum()
    for _ in range(20):
        local_optimizer.zero_grad(set_to_none=True)
        local_logits, _ = model(observations[:1])
        local_loss = -(local_target * torch.log_softmax(local_logits, dim=1)).sum()
        local_loss.backward()
        local_optimizer.step()
    local_final = -(local_target * torch.log_softmax(model(observations[:1])[0], dim=1)).sum()
    assert float(local_final) < float(local_initial)
