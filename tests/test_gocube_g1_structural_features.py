"""G1 graph-structural features, symmetry, expressivity, and fingerprints."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.core import Topology, cube_topology, torus_topology
from alphazero.envs.gocube.diversified_game import (
    diversified_pinned_game_class,
    diversified_structural_pinned_game_class,
)
from alphazero.envs.gocube.game import (
    Cube2JapaneseGame,
    Cube3JapaneseGame,
    Cube4JapaneseGame,
    Cube5JapaneseGame,
    Cube6JapaneseGame,
    Cube7JapaneseGame,
    Torus9JapaneseGame,
)
from alphazero.envs.gocube.integration.catalog import CheckpointDescriptor
from alphazero.envs.gocube.integration.contract import (
    ContractError,
    resolve_model_contract,
    resolve_game_class_from_contract,
)
from alphazero.envs.gocube.integration.errors import CheckpointMetadataInvalid
from alphazero.envs.gocube.integration.models import CheckpointModelLoader
from alphazero.envs.gocube.atomic_io import (
    REPLAY_TARGET_PROVENANCE_SUFFIX,
    REPLAY_TENSOR_SUFFIXES,
    write_replay_marker,
)
from alphazero.envs.gocube.hardened_train import HardenedKataGoSearchCoach
from alphazero.envs.gocube.katago_train import parse_args
from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.network import GraphNet
from alphazero.envs.gocube.pinned_game import structural_pinned_game_class
from alphazero.envs.gocube.rotations import cube_rotations, permute_point_axis
from alphazero.envs.gocube.structural import (
    structural_feature_matrix,
    structural_feature_metadata,
)


def _independent_triangles(topology):
    adjacency = [set(row) for row in topology.neighbors_by_index]
    result = set()
    for first in range(topology.point_count):
        for second in adjacency[first]:
            for third in adjacency[first].intersection(adjacency[second]):
                result.add(tuple(sorted((first, second, third))))
    return tuple(sorted(triangle for triangle in result if len(set(triangle)) == 3))


def _reindexed_topology(topology, new_order):
    inverse = {old: new for new, old in enumerate(new_order)}
    point_ids = tuple(topology.point_ids[old] for old in new_order)
    neighbors = tuple(
        tuple(sorted(inverse[neighbor] for neighbor in topology.neighbor_indices(old)))
        for old in new_order
    )
    return Topology(
        topology.kind,
        topology.size,
        point_ids,
        neighbors,
        {point_id: index for index, point_id in enumerate(point_ids)},
    )


def _rewired_topology(topology):
    adjacency = [set(row) for row in topology.neighbors_by_index]
    for first in range(topology.point_count):
        for second in sorted(adjacency[first]):
            if second <= first:
                continue
            for third in range(first + 1, topology.point_count):
                for fourth in sorted(adjacency[third]):
                    if fourth <= third:
                        continue
                    if len({first, second, third, fourth}) != 4:
                        continue
                    if any(candidate in adjacency[source] for source, candidate in (
                        (first, third), (first, fourth), (second, third), (second, fourth)
                    )):
                        continue
                    changed = [set(row) for row in adjacency]
                    changed[first].remove(second)
                    changed[second].remove(first)
                    changed[third].remove(fourth)
                    changed[fourth].remove(third)
                    changed[first].add(third)
                    changed[third].add(first)
                    changed[second].add(fourth)
                    changed[fourth].add(second)
                    return replace(topology, neighbors_by_index=tuple(
                        tuple(sorted(row)) for row in changed
                    ))
    raise AssertionError("Could not create a valid degree-preserving graph rewiring")


def _small_graph_args(*, auxiliary=False, depth=2):
    return SimpleNamespace(
        num_channels=16,
        depth=depth,
        value_dense_layers=[16],
        score_dense_layers=[8],
        gocube_auxiliary_targets=auxiliary,
    )


def _g1_args(tmp_path):
    game_cls, args = build_hardened_training_args(
        parse_args(["--model-profile", "g1", "--smoke"])
    )
    args = args.copy()
    args.cuda = False
    args.checkpoint = str(tmp_path)
    args.run_name = "g1-loader"
    return game_cls, args


def test_cube_triangle_detection_is_graph_only_and_deduplicated():
    topology = cube_topology(4)
    triangles = _independent_triangles(topology)
    assert len(triangles) == 8
    assert triangles == tuple(sorted(set(triangles)))
    assert all(
        all(second in topology.neighbor_indices(first) for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[0], triangle[2]),
        ))
        for triangle in triangles
    )


def test_cube4_has_eight_vertex_triangles_and_24_triangle_points():
    metadata = structural_feature_metadata(cube_topology(4))
    assert len(metadata.triangles) == 8
    assert int(metadata.matrix[0, :, 0].sum()) == 24


@pytest.mark.parametrize("size", range(2, 8))
def test_cube_distance_features_are_bfs_normalized(size):
    topology = cube_topology(size)
    metadata = structural_feature_metadata(topology)
    matrix = structural_feature_matrix(topology)
    membership = matrix[0, :, 0]
    distances = matrix[1, :, 0]
    assert matrix.shape == (2, 6 * size * size, 1)
    assert np.isfinite(matrix).all()
    assert np.all(distances[membership == 1.0] == 0.0)
    assert distances.min() == 0.0
    assert distances.max() == 0.0 if size == 2 else distances.max() == 1.0
    assert metadata.max_distance == 2 * ((size - 1) // 2)


def test_no_triangle_topology_has_zero_features_and_cached_results_are_isolated():
    topology = torus_topology(9)
    first = structural_feature_matrix(topology)
    second = structural_feature_matrix(torus_topology(9))
    assert first.shape == (2, 81, 1)
    assert np.array_equal(first, np.zeros_like(first))
    assert np.array_equal(first, second)
    assert structural_feature_metadata(topology) is structural_feature_metadata(torus_topology(9))


def test_features_are_state_independent_and_follow_canonical_point_order():
    game_cls = structural_pinned_game_class(Cube4JapaneseGame)
    empty = game_cls().observation()[-2:]
    occupied_game = game_cls()
    occupied_game.play_action(0)
    occupied = occupied_game.observation()[-2:]
    np.testing.assert_array_equal(empty, occupied)
    np.testing.assert_array_equal(empty, structural_feature_matrix(Cube4JapaneseGame.logical_topology()))


def test_point_order_fingerprint_changes_for_reindexed_topology():
    from alphazero.envs.gocube.integration.contract import (
        adjacency_fingerprint,
        point_order_fingerprint,
        topology_fingerprint,
    )

    original = cube_topology(4)
    reordered = _reindexed_topology(original, (1, 0, *range(2, original.point_count)))
    assert reordered.kind == original.kind
    assert reordered.size == original.size
    assert reordered.point_count == original.point_count
    assert set(reordered.point_ids) == set(original.point_ids)
    for original_point in range(original.point_count):
        original_id = original.point_id(original_point)
        reordered_point = reordered.point_index(original_id)
        assert {
            reordered.point_id(neighbor)
            for neighbor in reordered.neighbor_indices(reordered_point)
        } == set(original.neighbor_ids(original_id))
    assert point_order_fingerprint(original) != point_order_fingerprint(reordered)
    assert topology_fingerprint(original) != topology_fingerprint(reordered)
    assert adjacency_fingerprint(original) != adjacency_fingerprint(reordered)


def test_adjacency_fingerprint_changes_for_valid_modified_graph():
    from alphazero.envs.gocube.integration.contract import (
        adjacency_fingerprint,
        point_order_fingerprint,
        topology_fingerprint,
    )

    original = cube_topology(4)
    modified = _rewired_topology(original)
    assert modified.kind == original.kind
    assert modified.size == original.size
    assert modified.point_ids == original.point_ids
    assert modified.point_count == original.point_count
    assert all(len(row) == 4 and len(set(row)) == 4 for row in modified.neighbors_by_index)
    assert all(
        point in modified.neighbor_indices(neighbor)
        for point, row in enumerate(modified.neighbors_by_index)
        for neighbor in row
    )
    assert point_order_fingerprint(original) == point_order_fingerprint(modified)
    assert adjacency_fingerprint(original) != adjacency_fingerprint(modified)
    assert topology_fingerprint(original) != topology_fingerprint(modified)


class _Catalog:
    def __init__(self, descriptor):
        self.descriptor = descriptor

    def get(self, checkpoint_id):
        return self.descriptor if checkpoint_id == self.descriptor.checkpoint_id else None


@pytest.mark.parametrize("field", ("pointOrderFingerprint", "adjacencyFingerprint"))
def test_model_contract_rejects_real_topology_fingerprint_mismatch(tmp_path, field):
    game_cls, args = _g1_args(tmp_path)
    model = NNetWrapper(game_cls, args)
    checkpoint_dir = tmp_path / "g1-loader"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "iteration-0000.pkl"
    model.save_checkpoint(str(checkpoint_dir), checkpoint_path.name)
    original = resolve_model_contract(game_cls, args)
    topology = cube_topology(4)
    changed = _reindexed_topology(topology, (1, 0, *range(2, topology.point_count))) if field == "pointOrderFingerprint" else _rewired_topology(topology)
    from alphazero.envs.gocube.integration.contract import adjacency_fingerprint, point_order_fingerprint, topology_fingerprint
    broken = original.to_dict()
    broken[field] = point_order_fingerprint(changed) if field == "pointOrderFingerprint" else adjacency_fingerprint(changed)
    broken["topologyFingerprint"] = topology_fingerprint(changed)
    descriptor = CheckpointDescriptor(
        checkpoint_id="g1-loader@0",
        run_name="g1-loader",
        iteration=0,
        topology="cube",
        size=4,
        rule_set="japanese",
        komi=0.5,
        terminal_adjudicator=game_cls.TERMINAL_ADJUDICATOR_ID,
        path=str(checkpoint_path),
        model_contract=broken,
    )
    with pytest.raises(CheckpointMetadataInvalid, match=("point_order_fingerprint" if field == "pointOrderFingerprint" else "adjacency_fingerprint")):
        CheckpointModelLoader(_Catalog(descriptor), device="cpu").load(descriptor.checkpoint_id)


def test_old_graph_architecture_is_uniform_on_empty_cube():
    from alphazero.envs.gocube.network import GraphNet

    game_cls = Cube4JapaneseGame
    observations = torch.from_numpy(game_cls().observation()[None])
    for seed in (0, 1, 42):
        torch.manual_seed(seed)
        network = GraphNet(game_cls, _small_graph_args())
        with torch.no_grad():
            policy, _ = network(observations)
        point_logits = policy[0, : game_cls.logical_topology().point_count]
        assert float((point_logits - point_logits[0]).abs().max()) <= 1e-7


def test_triangle_classification_reaches_100_percent_with_g1_network():
    game_cls = diversified_structural_pinned_game_class(Cube4JapaneseGame)
    args = _small_graph_args(depth=6)
    torch.manual_seed(0)
    network = GraphNet(game_cls, args)
    network.train()
    observations = torch.from_numpy(game_cls().observation()[None])
    target = torch.from_numpy(
        structural_feature_matrix(game_cls.logical_topology())[0, :, 0][None]
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    for _ in range(40):
        policy, _ = network(observations)
        point_scores = policy[:, :-1] - policy[:, -1:]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(point_scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        policy, _ = network(observations)
    predicted = (policy[:, :-1] - policy[:, -1:] > 0).to(target.dtype)
    assert torch.equal(predicted, target)


def test_g1_production_contract_has_new_identity_and_observation():
    from alphazero.envs.gocube.diversified_game import diversified_pinned_game_class

    game_cls, args = build_hardened_training_args(
        parse_args(["--model-profile", "g1", "--smoke"])
    )
    contract = resolve_model_contract(game_cls, args)
    old_cls = diversified_pinned_game_class(Cube4JapaneseGame)
    old_args = args.copy()
    old_args.gocube_network_architecture = "gocube-graph-v1"
    old_args.gocube_observation_schema = old_cls.OBSERVATION_SCHEMA
    old_args.pop("gocube_structural_feature_schema", None)
    old_args.pop("gocube_structural_feature_channels", None)
    old_contract = resolve_model_contract(old_cls, old_args)
    assert game_cls.observation_size() == (20, 96, 1)
    assert contract.observation_schema == "gocube-observation-v5-structural-features"
    assert contract.network_architecture_id == "gocube-graph-structural-v1"
    assert contract.network_architecture_fingerprint
    assert contract.network_architecture_id != "gocube-graph-v1"
    assert contract.network_architecture_fingerprint != old_contract.network_architecture_fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gocube_network_architecture", "gocube-graph-v1"),
        ("gocube_structural_feature_schema", "gocube-structural-features-other-v1"),
        ("gocube_structural_feature_channels", 0),
    ),
)
def test_resolve_model_contract_rejects_game_class_argument_conflicts(
    tmp_path, field, value
):
    game_cls, args = _g1_args(tmp_path)
    broken = args.copy()
    broken[field] = value
    with pytest.raises(ContractError, match=field):
        resolve_model_contract(game_cls, broken)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gocube_network_architecture", "gocube-graph-v1"),
        ("gocube_structural_feature_schema", "gocube-structural-features-other-v1"),
        ("gocube_structural_feature_channels", 0),
    ),
)
def test_real_loader_rejects_game_class_argument_conflicts(
    tmp_path, field, value
):
    game_cls, args = _g1_args(tmp_path)
    model = NNetWrapper(game_cls, args)
    checkpoint_dir = tmp_path / "g1-loader"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "iteration-0000.pkl"
    model.save_checkpoint(str(checkpoint_dir), checkpoint_path.name)

    payload = torch.load(checkpoint_path, map_location="cpu")
    saved_args = payload["args"].copy()
    saved_args.pop("gocube_model_contract", None)
    saved_args[field] = value
    payload["args"] = saved_args
    torch.save(payload, checkpoint_path)

    contract = resolve_model_contract(game_cls, args)
    descriptor = _descriptor_for(checkpoint_path, contract, game_cls)
    with pytest.raises(CheckpointMetadataInvalid, match=field):
        CheckpointModelLoader(_Catalog(descriptor), device="cpu").load(
            descriptor.checkpoint_id
        )


def test_historical_v4_checkpoint_round_trips_through_loader(tmp_path):
    historical_cls = diversified_pinned_game_class(Cube4JapaneseGame)
    _production_cls, production_args = build_hardened_training_args(
        parse_args(["--model-profile", "baseline", "--smoke"])
    )
    args = production_args.copy()
    args.gocube_observation_schema = historical_cls.OBSERVATION_SCHEMA
    for key in (
        "gocube_model_profile",
        "gocube_structural_feature_schema",
        "gocube_structural_feature_channels",
    ):
        args.pop(key, None)

    model = NNetWrapper(historical_cls, args)
    checkpoint_path = tmp_path / "historical-v4.pkl"
    model.save_checkpoint(str(tmp_path), checkpoint_path.name)
    contract = resolve_model_contract(historical_cls, args)
    descriptor = _descriptor_for(checkpoint_path, contract, historical_cls)

    game = historical_cls()
    observation = game.observation()
    _, loaded = CheckpointModelLoader(
        _Catalog(descriptor), device="cpu"
    ).load(descriptor.checkpoint_id)
    assert loaded.game_cls is historical_cls
    assert historical_cls.OBSERVATION_SCHEMA == "gocube-observation-v4-pass-would-end-phase"
    assert observation.shape == (18, 96, 1)
    direct = model.predict_for_search(observation)
    reloaded = loaded.predict_for_search(observation)
    for field in ("policy", "value", "ownership", "score"):
        np.testing.assert_allclose(
            getattr(direct, field), getattr(reloaded, field), rtol=1e-6, atol=1e-7
        )


def test_24_cube_rotations_preserve_adjacency_and_structural_features():
    topology = cube_topology(4)
    features = structural_feature_matrix(topology)
    rotations = cube_rotations(topology)
    assert len(rotations) == 24
    for permutation in rotations:
        assert sorted(permutation.permutation) == list(range(topology.point_count))
        for source, neighbors in enumerate(topology.neighbors_by_index):
            assert {
                permutation.apply_point(neighbor) for neighbor in neighbors
            } == set(topology.neighbor_indices(permutation.apply_point(source)))
        np.testing.assert_array_equal(
            permute_point_axis(features, permutation, axis=1), features
        )
        inverse = [0] * topology.point_count
        for source, target in enumerate(permutation.permutation):
            inverse[target] = source
        rotated_topology = _reindexed_topology(topology, tuple(inverse))
        np.testing.assert_array_equal(
            structural_feature_matrix(rotated_topology),
            permute_point_axis(features, permutation, axis=1),
        )


def test_g1_network_equivariance_for_all_cube_rotations():
    game_cls = diversified_structural_pinned_game_class(Cube4JapaneseGame)
    network = GraphNet(game_cls, _small_graph_args(auxiliary=True, depth=2))
    network.eval()
    game = game_cls()
    game.play_action(0)
    game.play_action(17)
    observation = torch.from_numpy(game.observation()[None])
    topology = game_cls.logical_topology()
    with torch.no_grad():
        direct = network(observation)
        for permutation in cube_rotations(topology):
            rotated_observation = torch.from_numpy(
                permute_point_axis(game.observation(), permutation, axis=1)[None]
            )
            rotated = network(rotated_observation)
            np.testing.assert_allclose(
                rotated[0][0, :-1].numpy(),
                permute_point_axis(direct[0][0, :-1].numpy(), permutation, axis=0),
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(rotated[0][0, -1].numpy(), direct[0][0, -1].numpy(), atol=1e-6)
            np.testing.assert_allclose(rotated[1].numpy(), direct[1].numpy(), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(
                rotated[2][0].numpy(),
                permute_point_axis(direct[2][0].numpy(), permutation, axis=0),
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(rotated[3].numpy(), direct[3].numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "base_game_cls",
    (
        Cube2JapaneseGame,
        Cube3JapaneseGame,
        Cube4JapaneseGame,
        Cube5JapaneseGame,
        Cube6JapaneseGame,
        Cube7JapaneseGame,
        Torus9JapaneseGame,
    ),
)
def test_g1_cube2_to_cube7_and_torus9_forward_smoke(base_game_cls):
    game_cls = diversified_structural_pinned_game_class(base_game_cls)
    game = game_cls()
    features = game.observation()[-2:]
    network = GraphNet(game_cls, _small_graph_args(auxiliary=True, depth=1))
    with torch.no_grad():
        outputs = network(torch.from_numpy(game.observation()[None]))
    assert game.observation().shape == game_cls.observation_size()
    assert features.shape == (2, game_cls.logical_topology().point_count, 1)
    assert np.isfinite(features).all()
    assert all(torch.isfinite(output).all() for output in outputs)


def _descriptor_for(path, contract, game_cls):
    return CheckpointDescriptor(
        checkpoint_id="g1-compat@0",
        run_name="g1-compat",
        iteration=0,
        topology="cube",
        size=4,
        rule_set="japanese",
        komi=0.5,
        terminal_adjudicator=game_cls.TERMINAL_ADJUDICATOR_ID,
        path=str(path),
        model_contract=contract.to_dict(),
    )


def test_old_checkpoint_rejected_by_g1_contract_boundary(tmp_path):
    from alphazero.envs.gocube.diversified_game import diversified_pinned_game_class

    g1_cls, g1_args = _g1_args(tmp_path)
    old_cls = diversified_pinned_game_class(Cube4JapaneseGame)
    old_args = g1_args.copy()
    old_args.gocube_network_architecture = "gocube-graph-v1"
    old_args.gocube_observation_schema = old_cls.OBSERVATION_SCHEMA
    for key in (
        "gocube_structural_feature_schema",
        "gocube_structural_feature_channels",
    ):
        old_args.pop(key, None)
    old_model = NNetWrapper(old_cls, old_args)
    checkpoint_path = tmp_path / "old.pkl"
    old_model.save_checkpoint(str(tmp_path), checkpoint_path.name)

    g1_contract = resolve_model_contract(g1_cls, g1_args)
    descriptor = _descriptor_for(checkpoint_path, g1_contract, g1_cls)
    with pytest.raises(CheckpointMetadataInvalid, match="game_class_id|observation_schema|network_architecture"):
        CheckpointModelLoader(_Catalog(descriptor), device="cpu").load(descriptor.checkpoint_id)


def test_new_checkpoint_rejected_by_historical_architecture_boundary(tmp_path):
    from alphazero.envs.gocube.diversified_game import diversified_pinned_game_class

    g1_cls, g1_args = _g1_args(tmp_path)
    g1_model = NNetWrapper(g1_cls, g1_args)
    checkpoint_path = tmp_path / "new.pkl"
    g1_model.save_checkpoint(str(tmp_path), checkpoint_path.name)

    old_cls = diversified_pinned_game_class(Cube4JapaneseGame)
    old_args = g1_args.copy()
    old_args.gocube_network_architecture = "gocube-graph-v1"
    old_args.gocube_observation_schema = old_cls.OBSERVATION_SCHEMA
    for key in (
        "gocube_structural_feature_schema",
        "gocube_structural_feature_channels",
    ):
        old_args.pop(key, None)
    old_contract = resolve_model_contract(old_cls, old_args)
    descriptor = _descriptor_for(checkpoint_path, old_contract, old_cls)
    with pytest.raises(CheckpointMetadataInvalid, match="game_class_id|observation_schema|network_architecture"):
        CheckpointModelLoader(_Catalog(descriptor), device="cpu").load(descriptor.checkpoint_id)


def test_historical_replay_observation_shape_is_rejected_by_g1_loader(tmp_path):
    game_cls, _args = _g1_args(tmp_path)
    replay_dir = tmp_path / "replay" / "g1-replay"
    replay_dir.mkdir(parents=True)
    base = replay_dir / "iteration-0001"
    tensors = (
        torch.zeros(1, 18, 96, 1),
        torch.zeros(1, 97),
        torch.zeros(1, 3),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.zeros(1, 96, 3),
        torch.ones(1, 96),
    )
    for suffix, tensor in zip(REPLAY_TENSOR_SUFFIXES, tensors):
        torch.save(tensor, str(base) + suffix)
    torch.save(
        torch.ones(1, dtype=torch.uint8),
        str(base) + REPLAY_TARGET_PROVENANCE_SUFFIX,
    )
    write_replay_marker(str(base), iteration=1, row_count=1)

    coach = object.__new__(HardenedKataGoSearchCoach)
    coach.game_cls = game_cls
    coach.args = SimpleNamespace(data=str(tmp_path / "replay"), run_name="g1-replay")
    coach._replay_iterations = lambda _iteration: (1,)
    with pytest.raises(ValueError, match="observation schema/shape mismatch"):
        coach._load_replay_datasets(1)
