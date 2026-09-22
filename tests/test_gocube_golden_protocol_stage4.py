from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphazero.envs.gocube.integration.catalog import CheckpointCatalog, CheckpointDescriptor
from alphazero.envs.gocube.integration.golden_generation import replay_protocol_game, replay_protocol_moves, serialize_golden_result
from alphazero.envs.gocube.integration.golden_mapping import GoldenActionMappingError, mapping_for
from alphazero.envs.gocube.integration.golden_models import GoldenCheckpointLoader
from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService, _compatible
from gocube_golden.state import PASS


def test_golden_protocol_mapping_is_bijective_and_adjacency_exact():
    mapping = mapping_for("torus", 9)
    assert mapping.point_count == 81
    assert len(set(mapping.golden_to_protocol)) == 81
    assert len(set(mapping.protocol_to_golden)) == 81
    assert mapping.proof()["adjacency_exact"] is True
    assert mapping.golden_action_to_protocol(PASS) == {"type": "pass"}
    assert mapping.protocol_action_to_golden({"type": "pass"}) == PASS
    for index in range(81):
        protocol_index = mapping.golden_point_to_protocol_index(index)
        point_id = mapping.protocol_point_ids[protocol_index]
        assert mapping.protocol_point_id_to_golden_index(point_id) == index


def test_golden_capture_mapping_round_trips_point_ids():
    mapping = mapping_for("torus", 9)
    captured = (0, mapping.point_count - 1)
    point_ids = mapping.captured_point_ids(captured)
    assert [mapping.protocol_point_id_to_golden_index(item) for item in point_ids] == list(captured)
    with pytest.raises(GoldenActionMappingError):
        mapping.captured_point_ids((mapping.point_count,))


def test_golden_protocol_passes_round_trip_through_canonical_terminal_result():
    moves = [
        {"moveNumber": 1, "color": "black", "action": {"type": "pass"}, "captured": []},
        {"moveNumber": 2, "color": "white", "action": {"type": "pass"}, "captured": []},
    ]
    state, captures = replay_protocol_moves(topology="torus", size=9, moves=moves)
    game = {"topology": "torus", "size": 9, "moves": moves, "result": serialize_golden_result(state, captures=captures)}
    assert state.is_terminal
    assert captures == (0, 0)
    assert replay_protocol_game(game) == state


def _descriptor(checkpoint_id: str, *, profile_id: str = "gocube-torus9-golden-v3"):
    run_name, iteration = checkpoint_id.rsplit("@", 1)
    return CheckpointDescriptor(
        checkpoint_id=checkpoint_id, run_name=run_name, iteration=int(iteration), topology="torus", size=9,
        rule_set="chinese", komi=0.5, terminal_adjudicator="golden-graph-area-v1",
        path=str(Path("/tmp") / f"{run_name}-{iteration}.pt"), profile_id=profile_id,
        profile_fingerprint="sha256:" + "1" * 64, architecture_id="GoldenGraphNetV2-Torus9",
        rules_fingerprint="sha256:" + "2" * 64, observation_fingerprint="sha256:" + "3" * 64,
        target_fingerprint="sha256:" + "4" * 64,
    )


def test_golden_descriptors_require_matching_scientific_identity():
    golden = _descriptor("same-run@17")
    different_profile = _descriptor("same-run@17", profile_id="other-golden-profile")
    assert _compatible(golden, golden)
    assert _compatible(golden, different_profile)


REAL_RUNS_ROOT = Path(__file__).parents[1] / "runs"


@pytest.mark.skipif(
    not (REAL_RUNS_ROOT / "torus9/active/torus9-post-ab-m80-g128-20260918-v1/checkpoints/M93.pt").is_file(),
    reason="real Torus9 M93 artifact is not checked out",
)
def test_real_publication_catalog_separates_training_provenance_from_serving_contract():
    catalog = CheckpointCatalog(str(REAL_RUNS_ROOT))
    expected = {
        "torus9-golden-v3-production-20260917-m17@47",
        "torus9-golden-v3-plateau-exit-m47-lr3e4-r6-40k-s128-20260917-v2@54",
        "torus9-golden-v3-plateau-exit-m54-lr3e4-r6-40k-s128-20260917-v3@80",
        "torus9-staged-cadence-m80-20260918-v1-g128@83",
        "torus9-post-ab-m80-g128-20260918-v1@88",
        "torus9-post-ab-m80-g128-20260918-v1@93",
    }
    descriptors = {item.checkpoint_id: item for item in catalog.list()}
    assert expected <= descriptors.keys()
    assert all(descriptors[item].published for item in expected)
    assert len({descriptors[item].profile_fingerprint for item in expected}) == 3
    assert _compatible(descriptors[min(expected)], descriptors["torus9-post-ab-m80-g128-20260918-v1@93"])


def test_lineage_identity_uses_manifest_status_and_legacy_defaults_active(tmp_path: Path):
    checkpoint = tmp_path / "torus9" / "archive" / "old-run" / "checkpoints" / "M3.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    manifest = checkpoint.parent.parent / "manifest.json"
    manifest.write_text(json.dumps({"lineage_id": "old-run", "status": "DISCARDED"}), encoding="utf-8")
    assert CheckpointCatalog._lineage_identity_for_checkpoint(str(checkpoint)) == ("old-run", "DISCARDED")
    assert CheckpointCatalog._lineage_identity_for_checkpoint(str(tmp_path / "legacy" / "checkpoints" / "M1.pt")) == (None, "ACTIVE")


def test_golden_integration_modules_have_no_legacy_execution_imports():
    root = Path(__file__).parents[1] / "alphazero/envs/gocube/integration"
    forbidden = ("alphazero.NNetWrapper", "alphazero.GenericPlayers", "SelfPlayAgent", "Coach")
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path.name


def test_pickle_artifacts_are_unsupported_and_hidden_from_catalog(tmp_path: Path):
    (tmp_path / "legacy" / "iteration-0000.pkl").parent.mkdir()
    (tmp_path / "legacy" / "iteration-0000.pkl").write_bytes(b"legacy")
    assert CheckpointCatalog(str(tmp_path)).list() == []
    assert CheckpointCatalog(str(tmp_path)).get("legacy@0") is None


M17_PATH = Path(__file__).parents[1] / "runs/torus9-golden-v3-active/torus9-golden-v3-20260914-run03/checkpoints/M17.pt"


@pytest.mark.skipif(not M17_PATH.is_file(), reason="immutable local M17 artifact is not checked out")
def test_real_m17_golden_loader_and_protocol_round_trip_are_read_only():
    checkpoint_id = "torus9-golden-v3-20260914-run03@17"
    run_root = M17_PATH.parents[3]
    metadata_path = M17_PATH.with_suffix(".metadata.json")
    metadata_before = metadata_path.read_bytes()
    artifact_stat_before = M17_PATH.stat()
    m18_before = tuple(sorted(M17_PATH.parent.glob("M18*")))
    catalog = CheckpointCatalog(str(run_root))
    descriptor = catalog.get(checkpoint_id)
    assert descriptor is not None
    assert descriptor.profile_id == "gocube-torus9-golden-v3"
    assert descriptor.komi == 0.5
    assert descriptor.to_api() == {
        "id": checkpoint_id, "runName": "torus9-golden-v3-20260914-run03", "iteration": 17,
        "topology": "torus", "size": 9, "ruleSet": "chinese", "komi": 0.5,
        "terminalAdjudicator": "golden-graph-area-v1", "lineageStatus": "ACTIVE",
    }
    loader = GoldenCheckpointLoader(catalog, device="cpu")
    _, first = loader.load(checkpoint_id)
    _, second = loader.load(checkpoint_id)
    assert first.network is not None
    assert first is not second
    assert first.metadata["model_hash"] == json.loads(metadata_before)["model_hash"]
    assert not first.network.training
    service = GoCubeAlphaZeroService(str(run_root), catalog=catalog, loader=loader)
    response = service.generate_game({"protocolVersion": 1, "blackCheckpointId": checkpoint_id, "whiteCheckpointId": checkpoint_id, "mctsSims": 1})
    game = response["game"]
    assert game["mctsSims"] == 1
    assert game["komi"] == 0.5
    assert len(game["moves"]) > 0
    assert sum(move["action"]["type"] == "pass" for move in game["moves"]) == 2
    assert replay_protocol_game(game).is_terminal
    assert metadata_path.read_bytes() == metadata_before
    artifact_stat_after = M17_PATH.stat()
    assert (artifact_stat_after.st_size, artifact_stat_after.st_mtime_ns) == (artifact_stat_before.st_size, artifact_stat_before.st_mtime_ns)
    assert tuple(sorted(M17_PATH.parent.glob("M18*"))) == m18_before
