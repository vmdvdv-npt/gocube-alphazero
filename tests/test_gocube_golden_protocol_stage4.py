import json
from pathlib import Path

import pytest

from alphazero.envs.gocube.integration.catalog import CheckpointCatalog, CheckpointDescriptor
from alphazero.envs.gocube.integration.golden_generation import (
    replay_protocol_game,
    replay_protocol_moves,
    serialize_golden_result,
)
from alphazero.envs.gocube.integration.golden_mapping import (
    GoldenActionMappingError,
    mapping_for,
)
from alphazero.envs.gocube.integration.service import _compatible
from alphazero.envs.gocube.integration.models import CheckpointModelLoader
from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService
from gocube_golden.state import PASS


@pytest.mark.parametrize(
    ("topology", "size", "point_count"),
    [("cube", 4, 96), ("torus", 9, 81)],
)
def test_golden_protocol_mapping_is_bijective_and_adjacency_exact(
    topology, size, point_count
):
    mapping = mapping_for(topology, size)

    assert mapping.point_count == point_count
    assert len(set(mapping.golden_to_protocol)) == point_count
    assert len(set(mapping.protocol_to_golden)) == point_count
    assert mapping.golden_to_protocol[mapping.protocol_to_golden[0]] == 0
    assert mapping.proof()["adjacency_exact"] is True
    assert mapping.golden_action_to_protocol(PASS) == {"type": "pass"}
    assert mapping.protocol_action_to_golden({"type": "pass"}) == PASS

    for golden_index in range(point_count):
        protocol_index = mapping.golden_point_to_protocol_index(golden_index)
        point_id = mapping.protocol_point_ids[protocol_index]
        assert mapping.protocol_point_id_to_golden_index(point_id) == golden_index


@pytest.mark.parametrize(("topology", "size"), [("cube", 4), ("torus", 9)])
def test_golden_capture_mapping_round_trips_point_ids(topology, size):
    mapping = mapping_for(topology, size)
    captured = (0, mapping.point_count - 1)
    point_ids = mapping.captured_point_ids(captured)

    assert [mapping.protocol_point_id_to_golden_index(point_id) for point_id in point_ids] == list(captured)
    with pytest.raises(GoldenActionMappingError):
        mapping.captured_point_ids((mapping.point_count,))


def test_golden_protocol_passes_round_trip_through_canonical_terminal_result():
    moves = [
        {"moveNumber": 1, "color": "black", "action": {"type": "pass"}, "captured": []},
        {"moveNumber": 2, "color": "white", "action": {"type": "pass"}, "captured": []},
    ]
    state, captures = replay_protocol_moves(topology="torus", size=9, moves=moves)
    game = {
        "topology": "torus",
        "size": 9,
        "moves": moves,
        "result": serialize_golden_result(state, captures=captures),
    }

    assert state.is_terminal
    assert captures == (0, 0)
    assert replay_protocol_game(game) == state


def _descriptor(checkpoint_id: str, *, backend_kind: str, checkpoint_format: str):
    run_name, iteration = checkpoint_id.rsplit("@", 1)
    return CheckpointDescriptor(
        checkpoint_id=checkpoint_id,
        run_name=run_name,
        iteration=int(iteration),
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        terminal_adjudicator=(
            "golden-graph-area-v1" if backend_kind == "golden" else "gocube-conservative-area-v1"
        ),
        path=str(Path("/tmp") / f"{run_name}-{iteration}"),
        backend_kind=backend_kind,
        checkpoint_format=checkpoint_format,
        profile_id="gocube-torus9-golden-v3" if backend_kind == "golden" else None,
        profile_fingerprint="sha256:" + "1" * 64 if backend_kind == "golden" else None,
        architecture_id="GoldenGraphNetV2-Torus9" if backend_kind == "golden" else None,
        rules_fingerprint="sha256:" + "2" * 64 if backend_kind == "golden" else None,
        observation_fingerprint="sha256:" + "3" * 64 if backend_kind == "golden" else None,
        target_fingerprint="sha256:" + "4" * 64 if backend_kind == "golden" else None,
    )


def test_golden_and_legacy_descriptors_are_never_compatible():
    golden = _descriptor("same-run@17", backend_kind="golden", checkpoint_format="golden_pt")
    legacy = _descriptor("same-run@17", backend_kind="legacy_nnet", checkpoint_format="legacy_pickle")

    assert not _compatible(golden, legacy)
    assert not _compatible(legacy, golden)


def test_golden_integration_modules_have_no_legacy_execution_imports():
    root = Path(__file__).parents[1] / "alphazero/envs/gocube/integration"
    forbidden = (
        "alphazero.NNetWrapper",
        "alphazero.GenericPlayers",
        "SelfPlayAgent",
        "Coach",
    )
    for filename in ("golden_mapping.py", "golden_models.py", "golden_generation.py"):
        source = (root / filename).read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), filename


M17_PATH = (
    Path(__file__).parents[1]
    / "runs/torus9-golden-v3-active/torus9-golden-v3-20260914-run03/checkpoints/M17.pt"
)


@pytest.mark.skipif(not M17_PATH.is_file(), reason="immutable local M17 artifact is not checked out")
def test_real_m17_service_path_is_read_only_and_protocol_round_trip_is_exact():
    checkpoint_id = "torus9-golden-v3-20260914-run03@17"
    run_root = M17_PATH.parents[3]
    metadata_path = M17_PATH.with_suffix(".metadata.json")
    metadata_before = metadata_path.read_bytes()
    artifact_stat_before = M17_PATH.stat()
    m18_before = tuple(sorted(M17_PATH.parent.glob("M18*")))

    catalog = CheckpointCatalog(str(run_root))
    descriptor = catalog.get(checkpoint_id)
    assert descriptor is not None
    assert descriptor.backend_kind == "golden"
    assert descriptor.komi == 0.5
    assert catalog.get(checkpoint_id).to_api() == {
        "id": checkpoint_id,
        "runName": "torus9-golden-v3-20260914-run03",
        "iteration": 17,
        "topology": "torus",
        "size": 9,
        "ruleSet": "chinese",
        "komi": 0.5,
        "terminalAdjudicator": "golden-graph-area-v1",
    }

    loader = CheckpointModelLoader(catalog, device="cpu")
    _, first = loader.load(checkpoint_id)
    _, second = loader.load(checkpoint_id)
    assert first is second
    assert first.backend_kind == "golden"
    assert first.metadata["model_hash"] == json.loads(metadata_before.decode())["model_hash"]
    assert not first.network.training

    service = GoCubeAlphaZeroService(
        str(run_root), catalog=catalog, loader=loader
    )
    response = service.generate_game({
        "protocolVersion": 1,
        "blackCheckpointId": checkpoint_id,
        "whiteCheckpointId": checkpoint_id,
        "mctsSims": 1,
    })
    game = response["game"]
    assert game["mctsSims"] == 1
    assert game["komi"] == 0.5
    assert len(game["moves"]) > 0
    assert sum(move["action"]["type"] == "pass" for move in game["moves"]) == 2
    assert replay_protocol_game(game).is_terminal

    assert metadata_path.read_bytes() == metadata_before
    artifact_stat_after = M17_PATH.stat()
    assert (artifact_stat_after.st_size, artifact_stat_after.st_mtime_ns) == (
        artifact_stat_before.st_size,
        artifact_stat_before.st_mtime_ns,
    )
    assert tuple(sorted(M17_PATH.parent.glob("M18*"))) == m18_before
