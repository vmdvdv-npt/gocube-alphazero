"""Contract-driven semantic game restoration regressions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from alphazero.envs.gocube.diversified_game import (
    diversified_baseline_pinned_game_class,
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
    Torus13JapaneseGame,
    Torus19JapaneseGame,
)
from alphazero.envs.gocube.integration.contract import (
    ContractError,
    ResolvedGoCubeContract,
    resolve_game_class_from_contract,
    resolve_model_contract,
    resolve_model_contract_from_metadata,
    resolve_semantic_game_class_from_contract,
)
from alphazero.envs.gocube.pinned_game import pinned_game_class


BASE_GAMES = (
    Cube2JapaneseGame,
    Cube3JapaneseGame,
    Cube4JapaneseGame,
    Cube5JapaneseGame,
    Cube6JapaneseGame,
    Cube7JapaneseGame,
    Torus9JapaneseGame,
    Torus13JapaneseGame,
    Torus19JapaneseGame,
)


@pytest.mark.parametrize("base_game_cls", BASE_GAMES)
@pytest.mark.parametrize(
    "variant_factory",
    (
        lambda base: base,
        pinned_game_class,
        diversified_pinned_game_class,
    ),
    ids=("plain", "pinned", "diversified-pinned"),
)
def test_checkpoint_contract_resolves_exact_semantic_variant(base_game_cls, variant_factory):
    game_cls = variant_factory(base_game_cls)
    expected = resolve_model_contract(game_cls)
    saved = expected.to_checkpoint_fields()

    restored_contract = resolve_model_contract_from_metadata(saved)
    restored_model_cls = resolve_game_class_from_contract(restored_contract)
    restored_semantic_cls = resolve_semantic_game_class_from_contract(restored_contract)

    assert restored_model_cls is game_cls
    assert restored_semantic_cls is game_cls
    assert restored_contract.to_dict() == expected.to_dict()
    assert restored_contract.semantic_game_variant == expected.semantic_game_variant
    assert restored_contract.rules_fingerprint == expected.rules_fingerprint
    assert restored_contract.action_schema == expected.action_schema
    assert restored_contract.action_size == expected.action_size
    assert restored_contract.topology_kind == expected.topology_kind
    assert restored_contract.topology_size == expected.topology_size
    assert restored_contract.point_count == expected.point_count
    assert restored_contract.point_order_fingerprint == expected.point_order_fingerprint
    assert restored_contract.adjacency_fingerprint == expected.adjacency_fingerprint
    assert restored_contract.topology_fingerprint == expected.topology_fingerprint
    assert restored_contract.terminal_adjudicator_id == expected.terminal_adjudicator_id
    assert restored_contract.komi == 0.5


@pytest.mark.parametrize(
    "game_cls_factory",
    (diversified_baseline_pinned_game_class, diversified_structural_pinned_game_class),
    ids=("baseline", "g1"),
)
def test_production_profiles_keep_diversified_semantic_identity(game_cls_factory):
    game_cls = game_cls_factory(Cube4JapaneseGame)
    contract = resolve_model_contract(game_cls)

    assert contract.semantic_game_variant == "diversified_pinned"
    assert resolve_semantic_game_class_from_contract(contract) is diversified_pinned_game_class(
        Cube4JapaneseGame
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("topologyKind", "torus"),
        ("topologySize", 7),
        ("rulesFingerprint", "wrong-rules"),
        ("adjacencyFingerprint", "wrong-adjacency"),
        ("pointOrderFingerprint", "wrong-point-order"),
        ("actionSchema", "wrong-action-schema"),
        ("semanticGameVariant", "diversified_pinned"),
    ),
)
def test_contract_conflicts_fail_closed(field, value):
    contract = resolve_model_contract(Cube3JapaneseGame)
    broken = deepcopy(contract.to_dict())
    broken[field] = value

    with pytest.raises(ContractError):
        restored = ResolvedGoCubeContract.from_dict(broken)
        resolve_game_class_from_contract(restored)


def test_diversified_checkpoint_cannot_be_reinterpreted_as_plain():
    contract = resolve_model_contract(diversified_pinned_game_class(Cube3JapaneseGame))
    broken = deepcopy(contract.to_dict())
    broken["semanticGameVariant"] = "plain"

    with pytest.raises(ContractError, match="semantic_game_variant|semantic game"):
        resolve_game_class_from_contract(ResolvedGoCubeContract.from_dict(broken))


def test_plain_checkpoint_cannot_be_reinterpreted_as_diversified():
    contract = resolve_model_contract(Cube3JapaneseGame)
    broken = deepcopy(contract.to_dict())
    broken["semanticGameVariant"] = "diversified_pinned"

    with pytest.raises(ContractError, match="semantic_game_variant|semantic game"):
        resolve_game_class_from_contract(ResolvedGoCubeContract.from_dict(broken))


def test_komi_is_strictly_zero_point_five_in_contract():
    contract = resolve_model_contract(Cube3JapaneseGame)
    broken = deepcopy(contract.to_dict())
    broken["komi"] = 7.5

    with pytest.raises(ContractError, match="komi 0.5"):
        ResolvedGoCubeContract.from_dict(broken)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gocube_model_profile", "g1"),
        ("gocube_network_architecture", "wrong-network"),
    ),
)
def test_model_profile_contract_conflicts_fail_closed(field, value):
    game_cls = diversified_baseline_pinned_game_class(Cube4JapaneseGame)
    args = SimpleNamespace(
        gocube_model_profile="baseline",
        gocube_network_architecture="gocube-graph-v1",
    )
    setattr(args, field, value)

    with pytest.raises(ContractError, match=field):
        resolve_model_contract(game_cls, args)


def test_plain_training_checkpoint_uses_official_arena_resolution(tmp_path):
    """Reproduce the plain train.py checkpoint path without metadata edits."""

    from alphazero.NNetWrapper import NNetWrapper
    from alphazero.envs.gocube.train import build_training_args
    from tools import gocube_checkpoint_arena as checkpoint_arena

    cli = SimpleNamespace(
        topology="cube",
        size=3,
        workers=1,
        sims=1,
        arena_sims=1,
        games_per_iteration=1,
        iterations=1,
        train_batch_size=1,
        train_steps_per_iteration=1,
        fast_game_prob=0.0,
        endgame_sample_weight=1,
        inference_batch_wait_ms=0.0,
        no_arena=True,
        model_gating=False,
        smoke=True,
        run_name="plain-contract-e2e",
    )
    game_cls, args = build_training_args(cli)
    args = args.copy()
    args.checkpoint = str(tmp_path)
    args.run_name = "plain-contract-e2e"
    args.cuda = False

    model = NNetWrapper(game_cls, args)
    model.save_checkpoint(str(tmp_path / args.run_name), "iteration-0000.pkl")
    payload = checkpoint_arena._load_payload(
        tmp_path / args.run_name / "iteration-0000.pkl"
    )

    contract, model_game_cls = checkpoint_arena._resolve_checkpoint_contract(
        payload["args"], "plain train.py checkpoint"
    )
    semantic_game_cls = checkpoint_arena._authoritative_game_class(contract)

    assert model_game_cls is game_cls
    assert contract.semantic_game_variant == "plain"
    assert semantic_game_cls is Cube3JapaneseGame
    loaded = checkpoint_arena._load_network(
        model_game_cls,
        tmp_path / args.run_name / "iteration-0000.pkl",
        "cpu",
    )
    assert loaded.game_cls is Cube3JapaneseGame
