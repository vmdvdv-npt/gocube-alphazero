"""Cube V2 replay codec plus deterministic sampling over the shared rolling primitive."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import torch

from .cube_family import cube_family_topology
from .cube_selfplay_contract import (
    CUBE_SELFPLAY_SEMANTICS_FINGERPRINT,
    CUBE_TARGET_CONTRACT_ID,
    CUBE_TARGET_FINGERPRINT,
)
from .cube_training_contract_v2 import REPLAY_SCHEMA
from .cube_training_targets import CubeTrainingSampleV2

OWNERSHIP_TO_CLASS = {"OWN": 0, "OPPONENT": 1, "NEUTRAL": 2}


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CubeReplayCodecV2:
    schema = REPLAY_SCHEMA

    def __init__(self, *, size: int, game_fingerprint: str, observation_fingerprint: str) -> None:
        self.size = int(size)
        self.topology = cube_family_topology(self.size)
        self.game_fingerprint = str(game_fingerprint)
        self.observation_fingerprint = str(observation_fingerprint)

    def encode(self, sample: CubeTrainingSampleV2) -> dict[str, object]:
        if not isinstance(sample, CubeTrainingSampleV2):
            raise TypeError("Cube replay codec accepts CubeTrainingSampleV2")
        row: dict[str, object] = {
            "replay_schema": REPLAY_SCHEMA,
            "observation": sample.observation.detach().cpu().tolist(),
            "policy": list(sample.policy),
            "wdl": list(sample.wdl),
            "ownership": list(sample.ownership),
            "score_exact": float(sample.score_exact),
            "score_normalized": float(sample.score_normalized),
            "legal_action_mask": list(sample.legal_action_mask),
            "selected_action": int(sample.selected_action),
            "ply": int(sample.ply),
            "side_to_move": sample.side_to_move,
            "game_id": sample.game_id,
            "size": int(sample.size),
            "game_fingerprint": sample.game_identity_fingerprint,
            "observation_fingerprint": sample.observation_fingerprint,
            "model_hash": sample.model_hash,
            "selfplay_semantics_fingerprint": sample.selfplay_semantics_fingerprint,
            "search_config_fingerprint": sample.search_config_fingerprint,
            "target_contract_id": sample.target_contract_id,
            "target_contract_fingerprint": sample.target_contract_fingerprint,
        }
        row["sample_fingerprint"] = _fingerprint(row)
        self.validate(row)
        return row

    @staticmethod
    def content_fingerprint(row: Mapping[str, object]) -> str:
        payload = dict(row)
        payload.pop("sample_fingerprint", None)
        payload.pop("source_generation", None)
        payload.pop("replay_row_id", None)
        return _fingerprint(payload)

    def validate(self, row: Mapping[str, object]) -> None:
        if not isinstance(row, Mapping) or row.get("replay_schema") != REPLAY_SCHEMA:
            raise ValueError("Cube replay schema drift")
        if int(row.get("size", -1)) != self.size:
            raise ValueError("Cube replay size drift")
        if row.get("game_fingerprint") != self.game_fingerprint:
            raise ValueError("Cube replay game fingerprint drift")
        if row.get("observation_fingerprint") != self.observation_fingerprint:
            raise ValueError("Cube replay observation fingerprint drift")
        if row.get("selfplay_semantics_fingerprint") != CUBE_SELFPLAY_SEMANTICS_FINGERPRINT:
            raise ValueError("Cube replay self-play semantics fingerprint drift")
        if row.get("target_contract_id") != CUBE_TARGET_CONTRACT_ID:
            raise ValueError("Cube replay target contract id drift")
        if row.get("target_contract_fingerprint") != CUBE_TARGET_FINGERPRINT:
            raise ValueError("Cube replay target fingerprint drift")
        if row.get("sample_fingerprint") != self.content_fingerprint(row):
            raise ValueError("Cube replay sample fingerprint mismatch")

        observation = torch.as_tensor(row.get("observation"), dtype=torch.float32)
        if tuple(observation.shape) != (30, self.topology.point_count):
            raise ValueError("Cube replay observation shape drift")
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("Cube replay observation contains NaN/Inf")

        policy = torch.as_tensor(row.get("policy"), dtype=torch.float32)
        legal = tuple(bool(value) for value in row.get("legal_action_mask", ()))
        if tuple(policy.shape) != (self.topology.action_count,) or len(legal) != self.topology.action_count:
            raise ValueError("Cube replay policy/action shape drift")
        if not bool(torch.isfinite(policy).all()) or bool((policy < 0).any()):
            raise ValueError("Cube replay policy target is invalid")
        if not math.isclose(float(policy.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Cube replay policy target is not normalized")
        if any(not legal[index] and float(value) != 0.0 for index, value in enumerate(policy.tolist())):
            raise ValueError("Cube replay policy assigns mass to an illegal action")

        wdl = tuple(int(value) for value in row.get("wdl", ()))
        if wdl not in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            raise ValueError("Cube replay WDL target is invalid")
        ownership = tuple(str(value) for value in row.get("ownership", ()))
        if len(ownership) != self.topology.point_count or any(value not in OWNERSHIP_TO_CLASS for value in ownership):
            raise ValueError("Cube replay ownership target is invalid")
        for key in ("score_exact", "score_normalized"):
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Cube replay {key} is invalid") from exc
            if not math.isfinite(value):
                raise ValueError(f"Cube replay {key} contains NaN/Inf")
        action = row.get("selected_action")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < self.topology.action_count:
            raise ValueError("Cube replay selected action is invalid")
        if not legal[action]:
            raise ValueError("Cube replay selected action is illegal")
        ply = row.get("ply")
        if isinstance(ply, bool) or not isinstance(ply, int) or ply <= 0:
            raise ValueError("Cube replay ply is invalid")
        if row.get("side_to_move") not in ("BLACK", "WHITE"):
            raise ValueError("Cube replay side-to-move is invalid")
        if not isinstance(row.get("game_id"), str) or not row.get("game_id"):
            raise ValueError("Cube replay game provenance is incomplete")


def deterministic_sample_indices(
    replay_size: int,
    *,
    count: int,
    sampling_seed: int,
    sampling_counter: int,
    training_seed: int,
) -> tuple[int, ...]:
    if replay_size <= 0 or count <= 0:
        raise ValueError("Cube replay sampling requires positive size/count")
    material = f"cube-replay-v2:{sampling_seed}:{sampling_counter}:{training_seed}".encode("ascii")
    seed = (int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 2)) + 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if replay_size >= count:
        values = torch.randperm(replay_size, generator=generator)[:count]
    else:
        values = torch.randint(replay_size, (count,), generator=generator)
    return tuple(int(value) for value in values.tolist())


def collate_cube_rows(
    rows: Sequence[Mapping[str, object]],
    indices: Sequence[int],
    *,
    point_count: int,
    action_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    selected = [rows[index] for index in indices]
    batch = {
        "observation": torch.tensor([row["observation"] for row in selected], dtype=torch.float32, device=device),
        "policy": torch.tensor([row["policy"] for row in selected], dtype=torch.float32, device=device),
        "wdl": torch.tensor([row["wdl"] for row in selected], dtype=torch.float32, device=device),
        "ownership": torch.tensor(
            [[OWNERSHIP_TO_CLASS[str(value)] for value in row["ownership"]] for row in selected],
            dtype=torch.long,
            device=device,
        ),
        "score": torch.tensor([[float(row["score_normalized"])] for row in selected], dtype=torch.float32, device=device),
    }
    expected = {
        "observation": (len(indices), 30, point_count),
        "policy": (len(indices), action_count),
        "wdl": (len(indices), 3),
        "ownership": (len(indices), point_count),
        "score": (len(indices), 1),
    }
    if {key: tuple(value.shape) for key, value in batch.items()} != expected:
        raise ValueError("Cube batch tensor shape drift")
    for key in ("observation", "policy", "wdl", "score"):
        if not bool(torch.isfinite(batch[key]).all()):
            raise ValueError(f"Cube batch {key} contains NaN/Inf")
    return batch


__all__ = [
    "CubeReplayCodecV2",
    "OWNERSHIP_TO_CLASS",
    "collate_cube_rows",
    "deterministic_sample_indices",
]
