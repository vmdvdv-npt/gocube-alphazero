from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from gocube_golden.cube_family import CROSS_FACE_SEAM, FACE_INTERIOR, cube_family_topology
from gocube_golden.cube_network_v2 import (
    ARCHITECTURE_FINGERPRINT,
    ARCHITECTURE_ID,
    BLOCKS,
    HIDDEN,
    CubeGraphNetV2,
    build_cube_model_from_metadata,
    cube_graphnet_v2_model_hash,
    cube_model_metadata,
    load_cube_model_bundle,
    load_cube_network_architecture_contract,
    save_cube_model_bundle,
    trainable_parameter_count,
)

EXPECTED_PARAMETER_COUNT = 844_937


def _model(n: int = 4) -> CubeGraphNetV2:
    return build_cube_model_from_metadata(cube_model_metadata(n))


def _permute_points(tensor: torch.Tensor, permutation: tuple[int, ...], dim: int) -> torch.Tensor:
    result = torch.empty_like(tensor)
    index = [slice(None)] * tensor.ndim
    index[dim] = torch.as_tensor(permutation, dtype=torch.long)
    result[tuple(index)] = tensor
    return result


def _module_grad_sum(module: nn.Module) -> float:
    return float(sum(parameter.grad.abs().sum().item() for parameter in module.parameters() if parameter.grad is not None))


def test_architecture_contract_and_historical_identity_are_distinct():
    contract = load_cube_network_architecture_contract()
    assert contract["architecture_id"] == ARCHITECTURE_ID == "gocube-cube-graphnet-v2"
    assert contract["architecture_fingerprint"] == ARCHITECTURE_FINGERPRINT
    assert contract["hidden"] == HIDDEN == 112
    assert contract["blocks"] == BLOCKS == 10
    assert contract["global_context"]["after_blocks"] == [3, 6, 9]
    assert ARCHITECTURE_ID != "GoldenCubeGraphNetV1"


@pytest.mark.parametrize("n", range(2, 8))
@pytest.mark.parametrize("batch", [1, 3])
def test_forward_contract_all_cube_sizes(n, batch):
    topology = cube_family_topology(n)
    model = _model(n)
    output = model(torch.randn(batch, 30, topology.point_count, dtype=torch.float32))
    assert output.policy_logits.shape == (batch, topology.point_count + 1)
    assert output.wdl_logits.shape == (batch, 3)
    assert output.ownership_logits.shape == (batch, topology.point_count, 3)
    assert output.score.shape == (batch, 1)
    assert all(bool(torch.isfinite(t).all()) for t in (output.policy_logits, output.wdl_logits, output.ownership_logits, output.score))


def test_parameter_count_is_real_and_size_independent():
    counts = {}
    for n in range(2, 8):
        model = _model(n)
        counts[n] = trainable_parameter_count(model)
        assert counts[n] == model.model_metadata["parameter_count"]
    assert set(counts.values()) == {EXPECTED_PARAMETER_COUNT}


@pytest.mark.parametrize("bad", [torch.zeros(1,29,96), torch.zeros(1,30,95), torch.zeros(30,96), torch.zeros(1,30,96,dtype=torch.float64)])
def test_forward_rejects_wrong_tensor_contract(bad):
    with pytest.raises(ValueError):
        _model(4)(bad)


def test_forward_rejects_other_topology_mixed_container_and_nonfinite():
    model = _model(4)
    with pytest.raises(ValueError): model(torch.zeros(1,30,cube_family_topology(3).point_count))
    with pytest.raises(ValueError): model([torch.zeros(30,54), torch.zeros(30,96)])
    bad = torch.zeros(1,30,96); bad[0,0,0] = float("nan")
    with pytest.raises(ValueError): model(bad)


def test_backward_connects_every_stage_and_head():
    torch.manual_seed(7)
    model = _model(4)
    output = model(torch.randn(2,30,96))
    (output.policy_logits.square().mean() + output.wdl_logits.square().mean() + output.ownership_logits.square().mean() + output.score.square().mean()).backward()
    groups = {
        "input": model.input_projection,
        "same": model.blocks[0].message.same_face_transform,
        "seam": model.blocks[0].message.cross_face_seam_transform,
        "corner": model.corner_contexts[0],
        "global": model.global_contexts["3"],
        "point_policy": model.point_policy_head,
        "pass": model.pass_head,
        "wdl": model.wdl_head,
        "ownership": model.ownership_head,
        "score": model.score_head,
    }
    for name, module in groups.items(): assert _module_grad_sum(module) > 0.0, name


def test_relation_categories_use_different_trainable_paths():
    model = _model(4); layer = model.blocks[0].message; topology = cube_family_topology(4)
    assert layer.same_face_transform is not layer.cross_face_seam_transform
    with torch.no_grad():
        layer.self_transform.weight.zero_(); layer.self_transform.bias.zero_()
        layer.same_face_transform.weight.copy_(torch.eye(HIDDEN)); layer.same_face_transform.bias.zero_()
        layer.cross_face_seam_transform.weight.copy_(2*torch.eye(HIDDEN)); layer.cross_face_seam_transform.bias.zero_()
    nodes = torch.ones(1,topology.point_count,HIDDEN); first = layer(nodes)
    with torch.no_grad(): layer.cross_face_seam_transform.weight.copy_(3*torch.eye(HIDDEN))
    second = layer(nodes)
    interior = next(p.point_id for p in topology.points if p.geometry_class == FACE_INTERIOR)
    seam_point = next(point for point, relations in enumerate(topology.relation_types) if CROSS_FACE_SEAM in relations)
    assert torch.equal(first[:,interior], second[:,interior])
    assert not torch.equal(first[:,seam_point], second[:,seam_point])


def test_corner_context_is_unordered_local_to_eight_stage2_corners():
    torch.manual_seed(3); model = _model(4); topology = cube_family_topology(4); block = model.corner_contexts[0]
    assert block.corner_groups.shape == (8,3); before_adjacency = topology.adjacency
    nodes = torch.randn(1,topology.point_count,HIDDEN); corner = tuple(int(x) for x in block.corner_groups[0].tolist())
    swapped = nodes.clone(); swapped[:,corner[0]], swapped[:,corner[1]] = nodes[:,corner[1]].clone(), nodes[:,corner[0]].clone()
    assert torch.allclose(block.pooled_context(nodes)[:,0], block.pooled_context(swapped)[:,0])
    output = block(nodes); corner_points = {p for group in topology.physical_corners for p in group}
    non_corner = next(p for p in range(topology.point_count) if p not in corner_points)
    assert torch.equal(output[:,non_corner], nodes[:,non_corner]); assert topology.adjacency == before_adjacency


def _distance(topology, start):
    distance = {start:0}; queue = [start]
    for point in queue:
        for neighbor in topology.neighbors(point):
            if neighbor not in distance: distance[neighbor] = distance[point]+1; queue.append(neighbor)
    return distance


def test_global_context_transmits_nonlocal_information_after_block3():
    torch.manual_seed(5); model = _model(7); topology = cube_family_topology(7)
    local = next(p.point_id for p in topology.points if p.geometry_class == FACE_INTERIOR); distances = _distance(topology, local)
    remote = next(p.point_id for p in topology.points if p.geometry_class == FACE_INTERIOR and distances[p.point_id] > 3)
    a = torch.zeros(1,topology.point_count,HIDDEN); b = a.clone(); b[:,remote,0] = 4.0
    for index in range(3):
        a = model.blocks[index](a); b = model.blocks[index](b)
        if index < 2: a = model.corner_contexts[index](a); b = model.corner_contexts[index](b)
    assert torch.allclose(a[:,local], b[:,local], atol=1e-7, rtol=0.0)
    assert not torch.allclose(model.global_contexts["3"](a)[:,local], model.global_contexts["3"](b)[:,local], atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("n", range(2,8))
def test_all_24_stage2_rotations_are_equivariant(n):
    torch.manual_seed(100+n); topology = cube_family_topology(n); model = _model(n).eval(); observation = torch.randn(1,30,topology.point_count); base = model(observation)
    rotated = model(torch.cat([_permute_points(observation,r.point_permutation,2) for r in topology.rotations], dim=0)); assert len(topology.rotations) == 24
    for index, rotation in enumerate(topology.rotations):
        expected_policy = _permute_points(base.policy_logits[:,:-1], rotation.point_permutation,1)[0]
        expected_ownership = _permute_points(base.ownership_logits, rotation.point_permutation,1)[0]
        assert torch.allclose(rotated.policy_logits[index,:-1], expected_policy, atol=1e-5, rtol=1e-5)
        assert torch.allclose(rotated.ownership_logits[index], expected_ownership, atol=1e-5, rtol=1e-5)
        assert torch.allclose(rotated.policy_logits[index,-1], base.policy_logits[0,-1], atol=1e-5, rtol=1e-5)
        assert torch.allclose(rotated.wdl_logits[index], base.wdl_logits[0], atol=1e-5, rtol=1e-5)
        assert torch.allclose(rotated.score[index], base.score[0], atol=1e-5, rtol=1e-5)


def test_inference_path_does_not_call_auxiliary_heads():
    class Explode(nn.Module):
        def forward(self,*_args,**_kwargs): raise AssertionError("auxiliary head was called")
    model = _model(4); model.ownership_head = Explode(); model.score_head = Explode()
    result = model.infer_policy_wdl(torch.randn(2,30,96)); assert result.policy_logits.shape == (2,97); assert result.wdl_logits.shape == (2,3)
    with pytest.raises(AssertionError): model(torch.randn(1,30,96))


def test_model_only_save_load_roundtrip_and_concrete_hash(tmp_path):
    torch.manual_seed(11); metadata = cube_model_metadata(4); model = build_cube_model_from_metadata(metadata).eval(); observation = torch.randn(2,30,96); before = model(observation)
    paths = save_cube_model_bundle(model, tmp_path/"bundle"); loaded = load_cube_model_bundle(tmp_path/"bundle").eval(); after = loaded(observation)
    assert loaded.model_metadata == metadata; assert cube_graphnet_v2_model_hash(loaded) == paths["model_hash"].read_text().strip()
    assert torch.equal(before.policy_logits, after.policy_logits); assert torch.equal(before.wdl_logits, after.wdl_logits); assert torch.equal(before.ownership_logits, after.ownership_logits); assert torch.equal(before.score, after.score)


@pytest.mark.parametrize(("field","value"), [("architecture_id","unknown-cube-net"),("architecture_id","GoldenCubeGraphNetV1"),("architecture_fingerprint","sha256:bad"),("hidden",80),("blocks",8),("size",5),("point_count",95),("action_count",96),("topology_id","wrong"),("game_graph_fingerprint","sha256:wrong"),("geometry_fingerprint","sha256:wrong"),("observation_schema_id","old"),("observation_schema_fingerprint","sha256:old"),("concrete_observation_fingerprint","sha256:old")])
def test_metadata_compatibility_is_fail_closed(field,value):
    metadata = cube_model_metadata(4); metadata[field] = value
    with pytest.raises(ValueError): build_cube_model_from_metadata(metadata)


def test_missing_head_and_bad_contract_fingerprint_fail_closed():
    metadata = cube_model_metadata(4); del metadata["heads"]["ownership"]
    with pytest.raises(ValueError): build_cube_model_from_metadata(metadata)
    metadata = cube_model_metadata(4); metadata["model_contract_fingerprint"] = "sha256:wrong"
    with pytest.raises(ValueError): build_cube_model_from_metadata(metadata)


def test_bundle_rejects_bad_metadata_before_loading_weights(tmp_path, monkeypatch):
    model = _model(4); paths = save_cube_model_bundle(model, tmp_path/"bundle"); metadata = json.loads(paths["metadata"].read_text()); metadata["hidden"] = 80; paths["metadata"].write_text(json.dumps(metadata)); called = False
    def forbidden_load(*_args,**_kwargs):
        nonlocal called; called = True; raise AssertionError("weights should not be touched")
    monkeypatch.setattr(torch,"load",forbidden_load)
    with pytest.raises(ValueError): load_cube_model_bundle(tmp_path/"bundle")
    assert called is False
