from __future__ import annotations

import torch

from gocube_golden.cube_neural import (
    CUBE_ACTION_COUNT,
    CUBE_OBSERVATION_CHANNEL_COUNT,
    CUBE_OBSERVATION_FINGERPRINT,
    CornerContextBlock,
    GoldenCubeGraphNetV1,
    build_cube_observation,
    cube_count_parameters,
)
from gocube_golden.cube_topology import CUBE4_TOPOLOGY, FACE_CORNER, FACE_INTERIOR
from gocube_golden.cube_training import cube_initial_state, cube_save_checkpoint, cube_load_checkpoint
from gocube_golden.state import BLACK, EMPTY, research_state_from_stones


def test_cube_observation_has_dynamic_and_explicit_geometry_channels():
    state = cube_initial_state()
    observation = build_cube_observation(state)
    assert tuple(observation.shape) == (15, 96)
    assert CUBE_OBSERVATION_CHANNEL_COUNT == 15
    assert CUBE_OBSERVATION_FINGERPRINT.startswith("sha256:")
    corner = next(point for point in range(96) if CUBE4_TOPOLOGY.geometry(point).geometry_class == FACE_CORNER)
    interior = next(point for point in range(96) if CUBE4_TOPOLOGY.geometry(point).geometry_class == FACE_INTERIOR)
    assert observation[8, corner] == 1.0 and observation[8, interior] == 0.0
    assert observation[6, interior] == 1.0 and observation[6, corner] == 0.0
    assert observation[9, corner] == 1.0
    assert observation[13, corner] == 1.0
    assert observation[14, corner] == 0.5


def test_geometry_sensitivity_is_present_before_first_graph_block():
    state = cube_initial_state()
    observation = build_cube_observation(state)
    corner = next(point for point in range(96) if CUBE4_TOPOLOGY.geometry(point).geometry_class == FACE_CORNER)
    interior = next(point for point in range(96) if CUBE4_TOPOLOGY.geometry(point).geometry_class == FACE_INTERIOR)
    static_corner = observation[6:, corner]
    static_interior = observation[6:, interior]
    assert not torch.equal(static_corner, static_interior)


def test_corner_context_propagates_input_from_one_incident_cell_to_other_two():
    block = CornerContextBlock(8, CUBE4_TOPOLOGY)
    nodes_a = torch.zeros((1, 96, 8), dtype=torch.float32)
    nodes_b = nodes_a.clone()
    a, b, c = CUBE4_TOPOLOGY.physical_corners[0]
    nodes_b[0, a, 0] = 1.0
    output_a = block(nodes_a)
    output_b = block(nodes_b)
    assert not torch.equal(output_a[0, b], output_b[0, b])
    assert not torch.equal(output_a[0, c], output_b[0, c])
    assert CUBE4_TOPOLOGY.adjacency == tuple(tuple(row) for row in CUBE4_TOPOLOGY.adjacency)


def test_corner_context_does_not_add_synthetic_game_edges():
    before = CUBE4_TOPOLOGY.adjacency
    block = CornerContextBlock(8, CUBE4_TOPOLOGY)
    block(torch.zeros((1, 96, 8), dtype=torch.float32))
    assert CUBE4_TOPOLOGY.adjacency == before


def test_corner_influence_reaches_a_game_neighbor_after_context_stage():
    model = GoldenCubeGraphNetV1()
    corner = CUBE4_TOPOLOGY.physical_corners[0][0]
    neighbor = next(point for point in CUBE4_TOPOLOGY.adjacency[corner] if point not in CUBE4_TOPOLOGY.physical_corners[0])
    empty = cube_initial_state()
    stones = [EMPTY] * 96
    stones[corner] = BLACK
    occupied = research_state_from_stones(stones, topology=CUBE4_TOPOLOGY)
    with torch.no_grad():
        base = model.input_projection(build_cube_observation(empty).unsqueeze(0).transpose(1, 2))
        changed = model.input_projection(build_cube_observation(occupied).unsqueeze(0).transpose(1, 2))
        base = model.corner_context_blocks[0](model.blocks[0](base))
        changed = model.corner_context_blocks[0](model.blocks[0](changed))
        base = model.blocks[1](base)
        changed = model.blocks[1](changed)
    assert not torch.equal(base[0, neighbor], changed[0, neighbor])


def test_cube_network_contract_forward_backward_and_parameter_count():
    model = GoldenCubeGraphNetV1()
    observation = build_cube_observation(cube_initial_state()).unsqueeze(0)
    policy, value = model(observation)
    assert tuple(policy.shape) == (1, CUBE_ACTION_COUNT)
    assert tuple(value.shape) == (1, 3)
    assert cube_count_parameters(model) > 0
    loss = policy.square().mean() + value.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_cube_checkpoint_save_load_preserves_model_hash(tmp_path):
    model = GoldenCubeGraphNetV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "M0.pt"
    metadata = {"architecture_id": model.architecture_id, "parent_or_source_run_identity": "test"}
    saved = cube_save_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)
    restored = GoldenCubeGraphNetV1()
    loaded = cube_load_checkpoint(path, model=restored, optimizer=torch.optim.Adam(restored.parameters(), lr=1e-3), expected={"model_hash": saved["model_hash"]})
    assert loaded["model_hash"] == saved["model_hash"]
