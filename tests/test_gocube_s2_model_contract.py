"""S2 model-contract integration and fail-closed compatibility tests."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.evaluate import resolve_game_class
from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.integration.catalog import CheckpointCatalog, CheckpointDescriptor
from alphazero.envs.gocube.integration.errors import CheckpointMetadataInvalid
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.integration.models import CheckpointModelLoader
from alphazero.envs.gocube.integration.contract import resolve_model_contract
from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService
from alphazero.envs.gocube.katago_train import parse_args
from alphazero.envs.gocube.reproducible_manifest import create_reproducible_manifest


class _Catalog:
    def __init__(self, item):
        self.item = item

    def get(self, checkpoint_id):
        return self.item if checkpoint_id == self.item.checkpoint_id else None


@pytest.fixture(scope="module")
def real_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("gocube-s2")
    game_cls, args = build_hardened_training_args(parse_args([]))
    args = args.copy()
    args.cuda = False
    args.checkpoint = str(root)
    args.run_name = "audit"
    run_dir = root / args.run_name
    run_dir.mkdir()

    model = NNetWrapper(game_cls, args)
    model.save_checkpoint(str(run_dir), "iteration-0000.pkl")
    create_reproducible_manifest(
        checkpoint_dir=root,
        run_name=args.run_name,
        game_cls=game_cls,
        args=args,
        argv=["s2-test"],
        allow_dirty_source=True,
    )
    manifest = ensure_training_manifest(str(root), args.run_name, game_cls, args)
    catalog = CheckpointCatalog(str(root))
    descriptor = catalog.get("audit@0")
    assert descriptor is not None
    return root, game_cls, args, model, manifest, descriptor


def test_real_training_checkpoint_round_trips_all_four_heads_and_both_perspectives(real_checkpoint):
    root, game_cls, _args, model, manifest, descriptor = real_checkpoint
    assert manifest.version == 4
    assert manifest.observation_schema == "gocube-observation-v4-pass-would-end-phase"
    assert tuple(manifest.model_contract["observationShape"]) == (18, 96, 1)
    assert descriptor.model_contract["pointOrderFingerprint"]
    assert descriptor.model_contract["adjacencyFingerprint"]

    game = game_cls()
    observations = [game.observation()]
    game.play_action(0)
    observations.append(game.observation())

    _, loaded = CheckpointModelLoader(CheckpointCatalog(str(root)), device="cpu").load("audit@0")
    assert loaded.game_cls is game_cls
    for observation in observations:
        direct = model.predict_for_search(observation)
        reloaded = loaded.predict_for_search(observation)
        assert observation.shape == (18, 96, 1)
        assert direct.policy.shape == reloaded.policy.shape == (97,)
        assert direct.value.shape == reloaded.value.shape == (3,)
        assert direct.ownership.shape == reloaded.ownership.shape == (96, 3)
        assert direct.score.shape == reloaded.score.shape == (1,)
        for field in ("policy", "value", "ownership", "score"):
            np.testing.assert_allclose(getattr(direct, field), getattr(reloaded, field), rtol=1e-6, atol=1e-7)


def test_catalog_loader_and_generator_use_the_same_resolved_contract(real_checkpoint):
    root, game_cls, _args, _model, _manifest, descriptor = real_checkpoint

    class PassPlayer:
        def __init__(self, _model, game_cls, args):
            self.game_cls = game_cls
            self.args = args
            self.calls = 0

        def reset(self):
            pass

        def __call__(self, state):
            self.calls += 1
            if self.calls == 1:
                return int(np.flatnonzero(state.valid_moves())[0])
            return self.game_cls.pass_action()

        def update(self, _state, _action):
            pass

    from alphazero.envs.gocube.integration.generation import GameGenerator

    loader = CheckpointModelLoader(CheckpointCatalog(str(root)), device="cpu")
    service = GoCubeAlphaZeroService(
        str(root),
        catalog=CheckpointCatalog(str(root)),
        loader=loader,
        generator=GameGenerator(player_factory=PassPlayer),
    )
    response = service.generate_game({
        "protocolVersion": 1,
        "blackCheckpointId": descriptor.checkpoint_id,
        "whiteCheckpointId": descriptor.checkpoint_id,
        "mctsSims": 1,
    })
    colors = [move["color"] for move in response["game"]["moves"]]
    assert "black" in colors and "white" in colors
    assert response["game"]["result"]["adjudicatorId"] == descriptor.terminal_adjudicator
    assert resolve_game_class(str(root), "audit", "cube", 4) is game_cls


@pytest.mark.parametrize(
    "field,value,error_field",
    [
        ("topologyKind", "torus", "gocube_topology"),
        ("topologySize", 5, "gocube_size"),
        ("pointOrderFingerprint", "reordered-points", "point_order_fingerprint"),
        ("adjacencyFingerprint", "changed-adjacency", "adjacency_fingerprint"),
        ("observationSchema", "gocube-observation-v3", "gocube_observation_schema"),
        ("networkArchitectureId", "gocube-resnet-v1", "gocube_network_architecture"),
        ("targetsSchema", {"value": "wrong-targets", "score": "wrong", "ownership": "wrong"}, "targets_schema"),
    ],
)
def test_incompatible_contract_is_rejected_at_loader_boundary(real_checkpoint, field, value, error_field):
    _root, _game_cls, _args, _model, _manifest, descriptor = real_checkpoint
    contract = copy.deepcopy(descriptor.model_contract)
    contract[field] = value
    broken = replace(descriptor, model_contract=contract)
    with pytest.raises(CheckpointMetadataInvalid, match=error_field):
        CheckpointModelLoader(_Catalog(broken), device="cpu").load(descriptor.checkpoint_id)


def test_conflicting_rich_and_compact_manifests_fail_closed(real_checkpoint):
    root, _game_cls, _args, _model, _manifest, descriptor = real_checkpoint
    effective_path = root / "audit" / "effective-config.json"
    payload = json.loads(effective_path.read_text(encoding="utf-8"))
    payload["model_contract"]["actionSchema"] = "wrong-action-schema"
    effective_path.write_text(json.dumps(payload), encoding="utf-8")

    catalog = CheckpointCatalog(str(root))
    broken = catalog.get(descriptor.checkpoint_id)
    assert broken is not None
    with pytest.raises(CheckpointMetadataInvalid, match="Conflicting GoCube model contract"):
        CheckpointModelLoader(catalog, device="cpu").load(descriptor.checkpoint_id)


def test_fingerprints_are_deterministic_and_geometry_sensitive(real_checkpoint):
    _root, game_cls, args, _model, _manifest, _descriptor = real_checkpoint
    first = resolve_model_contract(game_cls, args)
    second = resolve_model_contract(game_cls, args)
    assert first.point_order_fingerprint == second.point_order_fingerprint
    assert first.adjacency_fingerprint == second.adjacency_fingerprint
    assert first.topology_fingerprint == second.topology_fingerprint
    assert first.point_order_fingerprint != first.adjacency_fingerprint
