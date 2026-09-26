"""Machine-readable scientific/training contract and run-owned Cube V2 knobs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from .cube_network_v2 import ARCHITECTURE_FINGERPRINT, ARCHITECTURE_ID
from .cube_selfplay_contract import CUBE_TARGET_CONTRACT_ID, CUBE_TARGET_FINGERPRINT

TRAINING_CONTRACT_ID = "gocube-cube-training-v2"
TRAINING_CONTRACT_SCHEMA_VERSION = 2
TRAINING_CONTRACT_PATH = Path("configs/gocube/cube_training_v2.json")
REPLAY_SCHEMA = "gocube-cube-replay-v2"
CHECKPOINT_SCHEMA = "gocube-cube-checkpoint-v2"
OPTIMIZER_FAMILY = "Adam"


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def training_contract_fingerprint(contract: Mapping[str, object]) -> str:
    payload = deepcopy(dict(contract))
    payload.pop("contract_fingerprint", None)
    return _fingerprint(payload)


def validate_cube_training_contract(contract: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(contract, Mapping):
        raise ValueError("Cube training contract must be a mapping")
    if contract.get("schema_version") != TRAINING_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Cube training contract schema version drift")
    if contract.get("contract_id") != TRAINING_CONTRACT_ID:
        raise ValueError("Cube training contract id drift")
    if contract.get("scope") != "scientific-training-semantics":
        raise ValueError("Cube training contract scope drift")
    if contract.get("required_architecture") != {
        "id": ARCHITECTURE_ID,
        "fingerprint": ARCHITECTURE_FINGERPRINT,
    }:
        raise ValueError("Cube training architecture contract drift")
    if contract.get("target_contract") != {
        "id": CUBE_TARGET_CONTRACT_ID,
        "fingerprint": CUBE_TARGET_FINGERPRINT,
    }:
        raise ValueError("Cube training target contract drift")
    if contract.get("optimizer_family") != OPTIMIZER_FAMILY:
        raise ValueError("Cube training optimizer family drift")
    if contract.get("weight_decay") != 0:
        raise ValueError("Cube training weight-decay contract drift")
    if contract.get("score_normalization") != "margin_stm/(P+abs(komi))":
        raise ValueError("Cube training score-normalization contract drift")
    if contract.get("auxiliary_heads") != {"ownership": True, "score": True}:
        raise ValueError("Cube training auxiliary-head contract drift")
    if contract.get("replay_schema") != REPLAY_SCHEMA:
        raise ValueError("Cube training replay schema drift")
    if contract.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Cube training checkpoint schema drift")
    run_owned = contract.get("run_owned_fields")
    if not isinstance(run_owned, list) or set(run_owned) != {
        "learning_rate",
        "batch_size",
        "optimizer_steps",
        "replay_generations",
        "replay_cap",
    }:
        raise ValueError("Cube training run-owned field contract drift")
    if contract.get("contract_fingerprint") != training_contract_fingerprint(contract):
        raise ValueError("Cube training contract fingerprint mismatch")
    return contract


def load_cube_training_contract(path: str | Path | None = None) -> dict[str, object]:
    target = Path(path) if path is not None else Path(__file__).resolve().parents[1] / TRAINING_CONTRACT_PATH
    with target.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    validate_cube_training_contract(contract)
    return contract


@dataclass(frozen=True)
class CubeTrainingConfig:
    """Run-owned values; validation is structural, never a Golden whitelist."""

    learning_rate: float
    batch_size: int
    optimizer_steps: int
    replay_generations: int
    replay_cap: int | None = None
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        lr = self.learning_rate
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(float(lr)) or float(lr) <= 0:
            raise ValueError("Cube learning_rate must be positive and finite")
        for name, value in (
            ("batch_size", self.batch_size),
            ("optimizer_steps", self.optimizer_steps),
            ("replay_generations", self.replay_generations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Cube {name} must be a positive integer")
        if self.replay_cap is not None and (
            isinstance(self.replay_cap, bool)
            or not isinstance(self.replay_cap, int)
            or self.replay_cap <= 0
        ):
            raise ValueError("Cube replay_cap must be None or a positive integer")
        if float(self.weight_decay) != 0.0:
            raise ValueError("Cube Stage-6 Adam baseline requires weight_decay=0")

    def identity_payload(self) -> dict[str, object]:
        return {
            "learning_rate": float(self.learning_rate),
            "batch_size": int(self.batch_size),
            "optimizer_steps": int(self.optimizer_steps),
            "replay_generations": int(self.replay_generations),
            "replay_cap": self.replay_cap,
            "weight_decay": 0.0,
        }

    @classmethod
    def from_identity_payload(cls, value: Mapping[str, object]) -> "CubeTrainingConfig":
        if not isinstance(value, Mapping):
            raise ValueError("Cube checkpoint concrete_training_config is missing")
        required = (
            "learning_rate",
            "batch_size",
            "optimizer_steps",
            "replay_generations",
        )
        if any(key not in value for key in required):
            raise ValueError("Cube checkpoint concrete_training_config is incomplete")
        if type(value["batch_size"]) is not int or type(value["optimizer_steps"]) is not int or type(value["replay_generations"]) is not int:
            raise ValueError("Cube checkpoint concrete_training_config integer field is invalid")
        if value.get("replay_cap") is not None and type(value["replay_cap"]) is not int:
            raise ValueError("Cube checkpoint concrete_training_config replay_cap is invalid")
        return cls(
            learning_rate=value["learning_rate"],  # type: ignore[arg-type]
            batch_size=value["batch_size"],  # type: ignore[arg-type]
            optimizer_steps=value["optimizer_steps"],  # type: ignore[arg-type]
            replay_generations=value["replay_generations"],  # type: ignore[arg-type]
            replay_cap=value.get("replay_cap"),  # type: ignore[arg-type]
            weight_decay=value.get("weight_decay", 0.0),  # type: ignore[arg-type]
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.identity_payload())


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CubeTrainingConfig",
    "OPTIMIZER_FAMILY",
    "REPLAY_SCHEMA",
    "TRAINING_CONTRACT_ID",
    "load_cube_training_contract",
    "training_contract_fingerprint",
    "validate_cube_training_contract",
]
