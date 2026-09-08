"""Cross-stage invariants for the post-S1/S2/S3/G1/H1/M1 F0 freeze."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

from alphazero.MCTS import MCTS
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube import (
    CYCLE,
    FORMAL_PASS,
    NO_RESULT,
    RESULT_PROVENANCE_FORMAL,
    RESULT_PROVENANCE_RULE_NO_RESULT,
    RESULT_PROVENANCE_RUNTIME,
    SCORED,
    Cube4JapaneseGame,
    cube_topology,
    final_v3_score,
    initial_v3_state,
    is_simple_ko_state,
    torus_topology,
    v3_state_from_board,
)
from alphazero.envs.gocube.atomic_io import (
    REPLAY_TARGET_PROVENANCE_SUFFIX,
    REPLAY_TENSOR_SUFFIXES,
    atomic_torch_save,
    load_replay_marker,
    load_replay_target_provenance,
    write_replay_marker,
)
from alphazero.envs.gocube.contract_versions import (
    REPLAY_FORMAT_VERSION,
    TARGET_PROVENANCE_ENCODING,
    TARGET_PROVENANCE_SEMANTICS,
    TERMINATION_CONTRACT,
    TRAINING_CONTRACT_VERSION,
)
from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.integration.catalog import CheckpointCatalog
from alphazero.envs.gocube.integration.errors import CheckpointMetadataInvalid
from alphazero.envs.gocube.integration.manifest import ensure_training_manifest
from alphazero.envs.gocube.integration.models import CheckpointModelLoader
from alphazero.envs.gocube.integration.contract import resolve_model_contract
from alphazero.envs.gocube.katago_train import parse_args
from alphazero.envs.gocube.katago_v3 import (
    EMERGENCY_MOVE_CAP_BASE,
    EMERGENCY_MOVE_CAP_FACTOR,
    _cycle_check_and_record,
)
from alphazero.envs.gocube.pinned_game import PinnedCube4JapaneseGame
from alphazero.envs.gocube.production_contract import CUBE4_PRODUCTION
from alphazero.envs.gocube.replay_provenance import (
    encode_target_provenance,
    provenance_codes_to_semantics,
)
from alphazero.envs.gocube.structural import (
    STRUCTURAL_FEATURE_CHANNELS,
    STRUCTURAL_FEATURE_SCHEMA,
    structural_feature_matrix,
    structural_feature_metadata,
)
from alphazero.search_contract import KATAGO_REFERENCE_COMMIT, KATAGO_SEARCH_CONTRACT, SearchOutput
from tests.support.h1_probe import (
    V3_STATE_AUDIT_FIELDS,
    find_observation_collisions,
    v3_immediate_semantic_signature,
)
from tests.test_gocube_h1_state_audit import _all_h1_samples, _observation_for_state


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "tests" / "fixtures" / "gocube_f0_integrated_contract.json"
PROTECTED_BUDGETS_PATH = ROOT / "tests" / "fixtures" / "gocube_production_profile_baseline.json"


def _profile_contract(profile: str, *, topology: str = "cube", size: int = 4):
    cli = parse_args([
        "--model-profile", profile,
        "--topology", topology,
        "--size", str(size),
        "--smoke",
        "--no-arena",
    ])
    game_cls, args = build_hardened_training_args(cli)
    args = args.copy()
    args.cuda = False
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    return game_cls, args, resolve_model_contract(game_cls, args)


def test_f0_snapshot_matches_current_baseline_and_g1_contracts():
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert snapshot["contract_id"] == "gocube-f0-integrated-freeze-v1"
    assert snapshot["integration_base_commit"] == (
        "efe31b7eefacbdd09abf4454a84ab66ffc1c9054"
    )
    assert snapshot["katago_reference_commit"] == KATAGO_REFERENCE_COMMIT
    assert snapshot["search_contract"] == KATAGO_SEARCH_CONTRACT == "katago-pinned-search-v4"
    assert snapshot["replay_format_version"] == REPLAY_FORMAT_VERSION == 4
    assert snapshot["training_contract_version"] == TRAINING_CONTRACT_VERSION == 3
    assert snapshot["termination_contract"] == TERMINATION_CONTRACT
    assert snapshot["target_provenance_semantics"] == TARGET_PROVENANCE_SEMANTICS
    assert snapshot["target_provenance_encoding"] == TARGET_PROVENANCE_ENCODING
    assert snapshot["komi"] == 0.5
    assert snapshot["replay"]["tensor_count"] == len(REPLAY_TENSOR_SUFFIXES) == 7
    assert snapshot["replay"]["target_provenance_sidecar"] == REPLAY_TARGET_PROVENANCE_SUFFIX

    contracts = {}
    for profile in ("baseline", "g1"):
        game_cls, _args, contract = _profile_contract(profile)
        contracts[profile] = contract
        expected = snapshot["profiles"][profile]
        assert expected["model_profile"] == profile
        assert expected["network_architecture_id"] == contract.network_architecture_id
        assert expected["network_architecture_fingerprint"] == contract.network_architecture_fingerprint
        assert expected["observation_schema"] == contract.observation_schema
        assert expected["observation_shape"] == list(contract.observation_shape)
        assert expected["action_schema"] == contract.action_schema
        assert expected["action_size"] == contract.action_size
        assert expected["point_count"] == contract.point_count
        assert expected["topology"] == contract.topology_kind
        assert expected["size"] == contract.topology_size
        assert expected["topology_fingerprint"] == contract.topology_fingerprint
        assert expected["point_order_fingerprint"] == contract.point_order_fingerprint
        assert expected["adjacency_fingerprint"] == contract.adjacency_fingerprint
        assert expected["rules_fingerprint"] == contract.rules_fingerprint
        assert expected["rules_implementation"] == contract.rules_implementation
        assert expected["terminal_adjudicator"] == contract.terminal_adjudicator_id
        assert expected["komi"] == contract.komi == game_cls.KOMI
        assert contract.search_contract_id == snapshot["search_contract"]

    assert contracts["baseline"].rules_fingerprint == contracts["g1"].rules_fingerprint
    assert contracts["baseline"].topology_fingerprint == contracts["g1"].topology_fingerprint
    assert contracts["baseline"].action_schema == contracts["g1"].action_schema
    assert contracts["baseline"].action_size == contracts["g1"].action_size
    assert contracts["baseline"].network_architecture_fingerprint != contracts["g1"].network_architecture_fingerprint
    assert contracts["baseline"].observation_schema != contracts["g1"].observation_schema
    assert contracts["baseline"].observation_shape == (18, 96, 1)
    assert contracts["g1"].observation_shape == (20, 96, 1)
    assert set(contracts["baseline"].differences(contracts["g1"])) <= {
        "game_class_id",
        "observation_schema",
        "observation_shape",
        "network_architecture_id",
        "network_architecture_fingerprint",
    }


def test_f0_protected_search_budgets_match_canonical_fixture_and_launch_args():
    protected = json.loads(PROTECTED_BUDGETS_PATH.read_text(encoding="utf-8"))["search_selfplay"]
    expected = CUBE4_PRODUCTION
    assert protected["regular_sims"] == expected.regular_sims == 50
    assert protected["fast_sims"] == expected.fast_sims == 20
    assert protected["arena_sims"] == expected.arena_sims == 50
    for profile in ("baseline", "g1"):
        _game_cls, args, _contract = _profile_contract(profile)
        assert args.numMCTSSims == expected.regular_sims
        assert args.numFastSims == expected.fast_sims
        assert args.arenaMCTSSims == expected.arena_sims
        assert args.gocube_komi == 0.5


def test_f0_v3_state_inventory_includes_s3_fields():
    from dataclasses import fields

    from alphazero.envs.gocube.katago_v3 import V3State

    assert set(V3_STATE_AUDIT_FIELDS) == {field.name for field in fields(V3State)}
    assert {"termination_reason", "result_provenance"} <= set(V3_STATE_AUDIT_FIELDS)
    assert len(fields(V3State)) == 27


@pytest.mark.parametrize("profile", ("baseline", "g1"))
def test_f0_h1_post_merge_probe_has_no_bounded_semantic_collisions(profile):
    samples = _all_h1_samples()
    report = find_observation_collisions(
        samples,
        lambda state: _observation_for_state(state, profile),
        lambda state: v3_immediate_semantic_signature(
            state, cube_topology(2) if len(state.board) == 24 else cube_topology(4)
        ),
    )
    assert report.samples_examined == 267
    assert report.observation_groups == 263
    assert len(report.collisions) == 3
    assert len(report.semantic_collisions) == 0


def test_f0_structural_and_rules_invariants_cover_cube4_and_torus9():
    cube = cube_topology(4)
    metadata = structural_feature_metadata(cube)
    assert len(metadata.triangles) == 8
    assert int(metadata.matrix[0, :, 0].sum()) == 24
    assert structural_feature_matrix(cube).shape == (STRUCTURAL_FEATURE_CHANNELS, 96, 1)

    torus = structural_feature_matrix(torus_topology(9))
    assert torus.shape == (STRUCTURAL_FEATURE_CHANNELS, 81, 1)
    assert np.array_equal(torus, np.zeros_like(torus))
    assert STRUCTURAL_FEATURE_SCHEMA == "gocube-structural-features-v1"

    baseline, _args, baseline_contract = _profile_contract("baseline")
    g1, _args, g1_contract = _profile_contract("g1")
    assert baseline.KOMI == g1.KOMI == 0.5
    assert baseline_contract.rules_fingerprint == g1_contract.rules_fingerprint
    assert baseline_contract.action_schema == g1_contract.action_schema
    assert baseline_contract.action_size == g1_contract.action_size


class _Catalog:
    def __init__(self, descriptor):
        self.descriptor = descriptor

    def get(self, checkpoint_id):
        return self.descriptor if checkpoint_id == self.descriptor.checkpoint_id else None


@pytest.fixture(params=("baseline", "g1"))
def current_profile_checkpoint(request, tmp_path):
    profile = request.param
    root = tmp_path / "checkpoint-root"
    run_name = f"f0-{profile}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    game_cls, args, _contract = _profile_contract(profile)
    args.checkpoint = str(root)
    args.run_name = run_name
    model = NNetWrapper(game_cls, args)
    model.save_checkpoint(str(run_dir), "iteration-0000.pkl")
    from alphazero.envs.gocube.reproducible_manifest import create_reproducible_manifest

    create_reproducible_manifest(
        checkpoint_dir=root,
        run_name=run_name,
        game_cls=game_cls,
        args=args,
        argv=["f0-test", profile],
        allow_dirty_source=True,
    )
    manifest = ensure_training_manifest(str(root), run_name, game_cls, args)
    descriptor = CheckpointCatalog(str(root)).get(f"{run_name}@0")
    assert descriptor is not None
    return root, game_cls, args, model, manifest, descriptor


def test_f0_current_profiles_round_trip_through_s2_resolver_and_search(current_profile_checkpoint):
    root, game_cls, args, model, manifest, descriptor = current_profile_checkpoint
    expected = resolve_model_contract(game_cls, args)
    assert manifest.model_contract == expected.to_dict()
    assert descriptor.model_contract == expected.to_dict()

    _loaded_descriptor, loaded = CheckpointModelLoader(
        CheckpointCatalog(str(root)), device="cpu"
    ).load(descriptor.checkpoint_id)
    assert loaded.game_cls is game_cls
    output = loaded.predict_for_search(game_cls().observation())
    assert output.policy.shape == (game_cls.action_size(),)
    assert output.value.shape == (3,)
    assert output.score.shape == (1,)
    assert output.ownership.shape == (game_cls.logical_topology().point_count, 3)
    assert np.isfinite(output.policy).all()
    assert np.isfinite(output.value).all()
    assert np.isfinite(output.score).all()
    assert np.isfinite(output.ownership).all()


def test_f0_torus9_baseline_forward_predict_and_mcts_smoke():
    game_cls, args, _contract = _profile_contract("baseline", topology="torus", size=9)
    game = game_cls()
    model = NNetWrapper(game_cls, args)
    output = model.predict_for_search(game.observation())
    assert output.policy.shape == (game_cls.action_size(),)
    assert output.value.shape == (3,)
    assert output.score.shape == (1,)
    assert output.ownership.shape == (81, 3)
    assert all(
        np.isfinite(values).all()
        for values in (output.policy, output.value, output.score, output.ownership)
    )
    MCTS(args).search(game, model, 1, False, False)


def test_f0_historical_v3_metadata_is_rejected_by_current_s2_boundary(current_profile_checkpoint):
    root, _game_cls, _args, _model, _manifest, descriptor = current_profile_checkpoint
    historical = dict(descriptor.model_contract)
    historical["searchContractId"] = "katago-pinned-search-v3"
    broken = replace(descriptor, model_contract=historical)
    with pytest.raises(CheckpointMetadataInvalid, match="search_contract"):
        CheckpointModelLoader(_Catalog(broken), device="cpu").load(descriptor.checkpoint_id)


class _ControlledWhiteWinNetwork:
    def __init__(self, action_size, point_count):
        self.action_size = action_size
        self.point_count = point_count

    def predict_for_search(self, _observation):
        return SearchOutput(
            policy=np.full(self.action_size, 1.0 / self.action_size, dtype=np.float32),
            value=np.array([0.8, 0.1, 0.1], dtype=np.float32),
            score=np.array([0.0], dtype=np.float32),
            ownership=np.tile(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (self.point_count, 1)
            ),
        )


@pytest.mark.parametrize("profile", ("baseline", "g1"))
@pytest.mark.parametrize(
    ("side_to_move", "expected_utility"),
    ((0, -0.7), (1, 0.7)),
    ids=("black-to-move", "white-to-move"),
)
def test_f0_production_mcts_converts_player_relative_value_once(
    profile, side_to_move, expected_utility
):
    game_cls, args, _contract = _profile_contract(profile)
    game = game_cls(
        replace(initial_v3_state(game_cls.logical_topology()), current_player=side_to_move)
    )
    mcts = MCTS(args)
    mcts.search(
        game,
        _ControlledWhiteWinNetwork(game.action_size(), game.logical_topology().point_count),
        1,
        False,
        False,
    )
    assert np.isclose(mcts._root.q, expected_utility)


def test_f0_s1_scorer_identity_and_m1_exact_ko_remain_current():
    from gocube_reference_topology import rectangular_test_topology
    from tests.support.fixtures import cube_verification_fixtures

    rectangular = rectangular_test_topology(5, 5)
    black = tuple(point for point in range(rectangular.point_count) if point not in (6, 7, 18))
    score = final_v3_score(
        v3_state_from_board(rectangular, black=black, white=(7,)), rectangular, 0.5
    )[0]
    assert score.white - score.black == -3.5
    assert score.winner == "black"

    from alphazero.envs.gocube.katago_v3 import apply_v3_action

    topology = cube_topology(4)
    for fixture_id, expected in (
        ("cube4_false_simple_ko_001", False),
        ("cube4_true_simple_ko_001", True),
    ):
        fixture = next(item for item in cube_verification_fixtures() if item.id == fixture_id)
        board = np.asarray(fixture.board(topology.index_by_id))
        state = v3_state_from_board(
            topology,
            black=np.flatnonzero(board == 1),
            white=np.flatnonzero(board == 2),
            current_player=0,
        )
        state = apply_v3_action(state, topology.point_index(fixture.actions[0]), topology)
        assert is_simple_ko_state(state, topology) is expected


def test_f0_provenance_round_trip_keeps_formal_cycle_and_runtime_rows_ordered(tmp_path):
    topology = Cube4JapaneseGame.logical_topology()
    formal = PinnedCube4JapaneseGame()
    for _ in range(6):
        formal.play_action(formal.pass_action())
    runtime = PinnedCube4JapaneseGame()
    for action in (0, 1, 2):
        runtime.play_action(action)
    assert runtime.finalize_episode_due_to_runtime_limit(3)
    initial = initial_v3_state(topology)
    key = initial.history_since_pass[0]
    cycle = PinnedCube4JapaneseGame(
        _cycle_check_and_record(replace(initial, history_since_pass=(key, key, key)), after_pass=False)
    )
    games = (formal, cycle, runtime)
    expected = (RESULT_PROVENANCE_FORMAL, RESULT_PROVENANCE_RULE_NO_RESULT, RESULT_PROVENANCE_RUNTIME)
    codes = torch.tensor(
        [encode_target_provenance(game.training_target_bundle().result_provenance) for game in games],
        dtype=torch.uint8,
    )
    base = tmp_path / "iteration-0001"
    for suffix in REPLAY_TENSOR_SUFFIXES:
        atomic_torch_save(torch.zeros((3, 1), dtype=torch.float32), str(base) + suffix)
    atomic_torch_save(codes, str(base) + REPLAY_TARGET_PROVENANCE_SUFFIX)
    write_replay_marker(str(base), iteration=1, row_count=3)
    marker = load_replay_marker(str(base))
    restored = load_replay_target_provenance(str(base), marker=marker)
    assert marker["replay_format_version"] == 4
    assert provenance_codes_to_semantics(restored) == expected
    assert [game.termination_reason for game in games] == [FORMAL_PASS, CYCLE, "episode_move_limit"]
    assert [game.terminal_kind for game in games] == [SCORED, NO_RESULT, SCORED]


def test_f0_runner_limit_is_not_a_formal_search_transition():
    topology = cube_topology(4)
    limit = EMERGENCY_MOVE_CAP_BASE + EMERGENCY_MOVE_CAP_FACTOR * topology.point_count
    game_cls, _args, _contract = _profile_contract("baseline")
    state = v3_state_from_board(topology, turns=limit - 1)
    game = game_cls(state)
    game._pinned_is_search_clone = True
    game.play_action(0)
    assert game.terminal_kind is None
    assert game.termination_reason is None
