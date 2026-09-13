from __future__ import annotations

import torch

import gocube_golden as g
from gocube_golden.arena_contract import SearchSettings
from gocube_golden.search import SequentialPUCT
from gocube_golden.torus9 import Torus9NeuralEvaluator, graph_diameter


def test_torus9_topology_is_a_connected_four_regular_wrapped_graph():
    topology = g.TORUS_9X9
    assert topology.point_count == 81
    assert topology.width == topology.height == 9
    assert topology.coordinate_to_point(0, 0) == 0
    assert topology.coordinate_to_point(8, 8) == 80
    assert topology.point_to_coordinate(40) == (4, 4)
    assert all(len(row) == 4 and len(set(row)) == 4 for row in topology.adjacency)
    assert all(point not in row for point, row in enumerate(topology.adjacency))
    assert all(point in topology.adjacency[neighbor] for point, row in enumerate(topology.adjacency) for neighbor in row)
    reached = {0}
    frontier = [0]
    while frontier:
        point = frontier.pop()
        for neighbor in topology.neighbors(point):
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    assert reached == set(range(81))
    assert set(topology.neighbors(0)) == {8, 1, 72, 9}
    assert set(topology.neighbors(40)) == {31, 41, 49, 39}


def test_torus9_rules_pass_capture_suicide_superko_and_exact_komi():
    initial = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    assert len(g.legal_actions(initial)) == 82
    assert g.PASS in g.legal_actions(initial)
    after_pass = g.apply_action(initial, g.PASS).after
    assert after_pass.consecutive_passes == 1
    terminal = g.apply_action(after_pass, g.PASS).after
    assert terminal.is_terminal
    assert g.result_from_terminal(terminal).komi == 0.5
    # A board position with a surrounded empty point rejects suicide.
    stones = [g.EMPTY] * 81
    for point in (1, 9, 11, 19):
        stones[point] = g.WHITE
    state = g.research_state_from_stones(stones, topology=g.TORUS_9X9, komi=0.5)
    assert 10 not in g.legal_actions(state)
    capture_stones = [g.EMPTY] * 81
    capture_stones[10] = g.WHITE
    for point in (1, 9, 11):
        capture_stones[point] = g.BLACK
    capture_state = g.research_state_from_stones(capture_stones, topology=g.TORUS_9X9, komi=0.5)
    assert g.apply_action(capture_state, 19).captured == (10,)
    # Positional superko is checked against the complete board history.
    blocked = list(stones)
    blocked[10] = g.BLACK
    state_with_history = g.research_state_from_stones(
        stones,
        topology=g.TORUS_9X9,
        komi=0.5,
        superko_history=(tuple(stones), tuple(blocked)),
    )
    assert state_with_history.rules_fingerprint == g.TORUS9_RULES_FINGERPRINT


def test_torus9_observation_network_and_search_shapes():
    state = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    bundle = g.build_torus9_observation_bundle(state)
    assert tuple(bundle.tensor.shape) == (6, 81)
    assert bundle.tensor.dtype == torch.float32
    assert len(bundle.action_mask) == 82 and bundle.action_mask[81]
    assert torch.all(bundle.tensor[5] == 0.5)
    model = g.Torus9GraphNet()
    assert sum(parameter.numel() for parameter in model.parameters()) == 105925
    assert model.blocks_count == 8
    policy, value = model(bundle.tensor.unsqueeze(0))
    assert tuple(policy.shape) == (1, 82)
    assert tuple(value.shape) == (1, 3)
    evaluator = Torus9NeuralEvaluator(model)
    result = SequentialPUCT(SearchSettings(simulations=2, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)).search(
        state, evaluator, seed=9
    )
    assert len(result.root_visits) == 82
    assert result.action in g.legal_actions(state)


def test_torus9_profile_and_contract_proof():
    profile = g.load_torus9_profile()
    assert profile["topology"]["point_count"] == 81
    assert profile["observation"]["action_count"] == 82
    assert profile["observation"]["pass_index"] == 81
    assert profile["rules"]["komi"] == 0.5
    assert g.torus9_contract_proof()["status"] == "PASS"
