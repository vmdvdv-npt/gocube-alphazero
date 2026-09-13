from __future__ import annotations

import torch

import gocube_golden as g


def _observation(stones: list[int]) -> torch.Tensor:
    state = g.research_state_from_stones(stones, topology=g.TORUS_9X9, side_to_move=g.BLACK)
    return g.build_torus9_observation(state, legal_context=g.prepare_legal_actions(state))


def _distance(topology: object, source: int, target: int) -> int:
    frontier = [(source, 0)]
    reached = {source}
    while frontier:
        point, distance = frontier.pop(0)
        if point == target:
            return distance
        for neighbor in topology.neighbors(point):  # type: ignore[attr-defined]
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append((neighbor, distance + 1))
    raise AssertionError("target is unreachable")


def test_four_blocks_have_radius_four_and_preserve_distant_point_logits():
    assert _distance(g.TORUS_5X5, 0, 12) == 4
    assert _distance(g.TORUS_9X9, 3, 40) == 5
    assert _distance(g.TORUS_9X9, 0, 40) == 8

    stones = [g.EMPTY] * 81
    distant = stones.copy()
    distant[3] = g.BLACK
    torch.manual_seed(20260914)
    model = g.Torus9GraphNet(blocks=4).eval()
    with torch.inference_mode():
        logits, _ = model(torch.stack((_observation(stones), _observation(distant))))

    assert torch.equal(logits[0, 40], logits[1, 40])
    assert torch.equal(logits[0, 41], logits[1, 41])

