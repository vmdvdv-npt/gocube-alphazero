from __future__ import annotations

import json
from pathlib import Path

import torch

from gocube_golden.neural import model_hash
from gocube_golden.provenance import file_sha256, sha256_fingerprint
from gocube_golden.torus9 import TORUS9_TOPOLOGY_FINGERPRINT, generate_torus9_evaluation_starts
from gocube_golden.torus9_m137_5ch import (
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    Torus9M137FiveChannelGraphNet,
)
from tools.arena_engine import ArenaExecutionConfig, CheckpointIdentity
from tools.arena_profiles import get_profile
from tools.arena_profiles.torus9 import _load_frozen_starts


def _checkpoint(tmp_path: Path) -> Path:
    model = Torus9M137FiveChannelGraphNet()
    path = tmp_path / "M137-5CH-bootstrap.pt"
    metadata = {
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "architecture_config": model.architecture_config,
        "observation_shape": [5, 81],
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "converted_model_hash": model_hash(model),
    }
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "metadata": metadata,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
        },
        path,
    )
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _startset(tmp_path: Path) -> Path:
    row = generate_torus9_evaluation_starts(master_seed=20260925, accepted_per_stratum=1, komi=0.5)[0]
    body = {
        "schema": "torus9-frozen-startset-v1",
        "master_seed": 20260925,
        "pair_count": 1,
        "pairs": [
            {
                "pair_index": 0,
                "pair_id": "pair-0000",
                "start_id": row["start_id"],
                "trace": row["trace"],
                "state": row["state"],
            }
        ],
    }
    payload = {**body, "fingerprint": sha256_fingerprint(body)}
    path = tmp_path / "frozen-startset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_5ch_profile_loads_checkpoint_and_uses_256_simulations(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path)
    profile = get_profile("torus9-komi-calibration|1.5|simulations=256|5ch")
    identity = profile.load_identity(path)

    assert profile.observation_shape == (5, 81)
    assert profile.simulations == 256
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert identity.model_hash == metadata["converted_model_hash"]
    assert profile.scientific_contract(ArenaExecutionConfig(games=2, strict_production=False))["simulations"] == 256
    loaded = profile.load_parent_model(identity, torch.device("cpu"))
    assert tuple(loaded.input_projection.weight.shape) == (80, 5)


def test_frozen_startset_rebinds_only_rules_komi_and_preserves_pair_id(tmp_path: Path) -> None:
    path = _startset(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload["fingerprint"]
    low = _load_frozen_starts(
        path,
        master_seed=20260925,
        pair_indices=[0],
        expected_fingerprint=fingerprint,
        komi=0.5,
    )[0]
    high = _load_frozen_starts(
        path,
        master_seed=20260925,
        pair_indices=[0],
        expected_fingerprint=fingerprint,
        komi=2.5,
    )[0]

    assert low["pair_id"] == high["pair_id"] == "pair-0000"
    assert low["trace"] == high["trace"]
    assert low["state"]["stones"] == high["state"]["stones"]
    assert low["state"]["superko_history"] == high["state"]["superko_history"]
    assert low["state"]["komi"] == 0.5
    assert high["state"]["komi"] == 2.5
    assert low["state"]["rules_fingerprint"] != high["state"]["rules_fingerprint"]


def test_task_corpus_reuses_pair_and_game_ids_across_komi(tmp_path: Path) -> None:
    startset = _startset(tmp_path)
    fingerprint = json.loads(startset.read_text(encoding="utf-8"))["fingerprint"]
    identity = CheckpointIdentity(
        path=tmp_path / "checkpoint.pt",
        model_hash="sha256:" + "1" * 64,
        artifact_sha256="sha256:" + "2" * 64,
        architecture_config={"architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID, "input_channels": 5},
        metadata={},
    )
    rows = []
    for komi in (0.5, 2.5):
        profile = get_profile(f"torus9-komi-calibration|{komi:g}|simulations=256|5ch")
        tasks, pairs = profile.build_tasks(
            run_id="run",
            comparison="comparison",
            candidate=identity,
            reference=identity,
            master_seed=20260925,
            games=2,
            workers=16,
            workload={
                "startset_path": str(startset),
                "startset_fingerprint": fingerprint,
                "pair_indices": [0],
            },
        )
        rows.append((tasks, pairs))

    assert rows[0][1] == rows[1][1] == 1
    assert [task["pair_id"] for task in rows[0][0]] == [task["pair_id"] for task in rows[1][0]]
    assert [task["game_id"] for task in rows[0][0]] == [task["game_id"] for task in rows[1][0]]
    assert rows[0][0][0]["state"]["stones"] == rows[1][0][0]["state"]["stones"]
    assert rows[0][0][0]["state"]["komi"] == 0.5
    assert rows[1][0][0]["state"]["komi"] == 2.5
