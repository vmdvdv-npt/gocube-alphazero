"""Versioned Cube V2 self-play and target contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from .arena_contract import SEARCH_IMPLEMENTATION_ID, SearchSettings
from .cube_game_contract_v2 import (
    FORMAL_DOUBLE_PASS,
    project_ownership,
    project_score,
    project_wdl,
    validate_cube_size,
)
from .cube_network_v2 import ARCHITECTURE_FINGERPRINT, ARCHITECTURE_ID
from .cube_observation_v2 import SCHEMA_FINGERPRINT
from .search import SearchResult
from .state import PASS


CUBE_SELFPLAY_CONTRACT_ID = "gocube-cube-selfplay-v2"
CUBE_SELFPLAY_CONTRACT_VERSION = 1
CUBE_TARGET_CONTRACT_ID = "gocube-cube-training-targets-v2"
CUBE_TARGET_CONTRACT_VERSION = 1
CUBE_SEMANTICS_ID = "gocube-common-selfplay-semantics-v1"
CUBE_TECHNICAL_RESULT_POLICY = "exclude-from-training-and-formal-statistics"
CUBE_DEFAULT_TECHNICAL_MOVE_LIMIT = 1920
CUBE_SELFPLAY_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "configs" / "gocube" / "cube_selfplay_targets_v2.json"


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cube_selfplay_semantics_identity() -> dict[str, object]:
    return {
        "semantics_id": CUBE_SEMANTICS_ID,
        "version": 1,
        "puct_implementation": SEARCH_IMPLEMENTATION_ID,
        "root_noise": "root-only-legal-normalized-dirichlet-v1",
        "temperature": "root-visits-power-sampling-v1;zero=max-canonical-tie",
        "policy_timing": "root-visits-before-selected-move",
        "wdl_perspective": "side-to-move",
        "ownership_perspective": "side-to-move",
        "score_perspective": "side-to-move",
        "technical_results": CUBE_TECHNICAL_RESULT_POLICY,
        "pass_representation": "internal-PASS;durable-canonical-P",
        "formal_completion": FORMAL_DOUBLE_PASS,
        "resign": "disabled-by-reference-contract",
        "search_history": "immutable-position-wrapper-bounded-observation-context",
    }


CUBE_SELFPLAY_SEMANTICS_FINGERPRINT = _fingerprint(cube_selfplay_semantics_identity())


def cube_target_contract_identity() -> dict[str, object]:
    return {
        "contract_id": CUBE_TARGET_CONTRACT_ID,
        "version": CUBE_TARGET_CONTRACT_VERSION,
        "policy": "saved-root-pi",
        "policy_shape": "P+1",
        "policy_illegal": 0.0,
        "wdl": ["WIN", "DRAW", "LOSS"],
        "wdl_perspective": "side-to-move",
        "ownership": ["OWN", "OPPONENT", "NEUTRAL"],
        "ownership_perspective": "side-to-move",
        "score": "exact-final-margin-stm-and-margin-stm-divided-by-P-plus-abs-komi",
        "technical_records": "zero-samples",
    }


CUBE_TARGET_FINGERPRINT = _fingerprint(cube_target_contract_identity())


def validate_cube_selfplay_contract(payload: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("Cube self-play machine contract must be a mapping")
    if payload.get("contract_id") != CUBE_SELFPLAY_CONTRACT_ID or payload.get("contract_version") != CUBE_SELFPLAY_CONTRACT_VERSION:
        raise ValueError("Cube self-play machine contract identity drift")
    candidate = dict(payload)
    supplied = candidate.pop("contract_fingerprint", None)
    if supplied != _fingerprint(candidate):
        raise ValueError("Cube self-play machine contract fingerprint mismatch")
    semantics = payload.get("semantics")
    if not isinstance(semantics, Mapping) or semantics.get("fingerprint") != CUBE_SELFPLAY_SEMANTICS_FINGERPRINT:
        raise ValueError("Cube self-play semantics fingerprint drift")
    target = payload.get("target")
    if not isinstance(target, Mapping) or target.get("contract_id") != CUBE_TARGET_CONTRACT_ID or target.get("fingerprint") != CUBE_TARGET_FINGERPRINT:
        raise ValueError("Cube self-play target contract fingerprint drift")
    required_observation = payload.get("required_observation")
    if not isinstance(required_observation, Mapping) or required_observation.get("schema_fingerprint") != SCHEMA_FINGERPRINT:
        raise ValueError("Cube self-play observation schema fingerprint drift")
    required_network = payload.get("required_network")
    if (
        not isinstance(required_network, Mapping)
        or required_network.get("architecture_id") != ARCHITECTURE_ID
        or required_network.get("architecture_fingerprint") != ARCHITECTURE_FINGERPRINT
        or required_network.get("policy") != "P+1"
        or required_network.get("wdl") != 3
    ):
        raise ValueError("Cube self-play network identity drift")
    return payload


def load_cube_selfplay_contract(path: str | Path = CUBE_SELFPLAY_CONTRACT_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(validate_cube_selfplay_contract(payload))


@dataclass(frozen=True)
class CubeSelfPlaySearchContract:
    """Scientific settings only; process/execution settings are elsewhere."""

    contract_id: str = CUBE_SELFPLAY_CONTRACT_ID
    simulations: int = 64
    cpuct: float = 1.25
    fpu: float = 0.0
    root_noise: bool = True
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.11
    temperature_plies: tuple[int, int] = (1, 8)
    temperature_after: float = 0.0
    resign: bool = False
    technical_move_limit: int = CUBE_DEFAULT_TECHNICAL_MOVE_LIMIT
    komi: float = 0.5

    @property
    def temperature_until_ply(self) -> int:
        return int(self.temperature_plies[1])

    @property
    def epsilon(self) -> float:
        """Short compatibility alias for the root-noise mixing weight."""

        return float(self.dirichlet_epsilon)

    @property
    def alpha(self) -> float:
        """Short compatibility alias for the Dirichlet concentration."""

        return float(self.dirichlet_alpha)

    @property
    def watchdog(self) -> int:
        """Historical spelling retained as a read-only compatibility alias."""

        return int(self.technical_move_limit)

    @property
    def move_limit(self) -> int:
        """Short compatibility alias for the technical safety limit."""

        return int(self.technical_move_limit)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.concrete_search_config_identity())

    @property
    def puct_settings(self) -> SearchSettings:
        # Root noise and move temperature are scientific self-play policy
        # transforms, not Arena settings and therefore never enter PUCT.
        return SearchSettings(
            simulations=int(self.simulations),
            cpuct=float(self.cpuct),
            fpu=float(self.fpu),
            deterministic_tie_break=True,
        )

    @property
    def settings(self) -> SearchSettings:
        """Compatibility spelling shared with the Torus self-play contract."""

        return self.puct_settings

    def concrete_search_config_identity(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "contract_version": CUBE_SELFPLAY_CONTRACT_VERSION,
            "search_implementation_id": SEARCH_IMPLEMENTATION_ID,
            "simulations": int(self.simulations),
            "cpuct": float(self.cpuct),
            "fpu": float(self.fpu),
            "root_noise": bool(self.root_noise),
            "dirichlet_epsilon": float(self.dirichlet_epsilon),
            "dirichlet_alpha": float(self.dirichlet_alpha),
            "temperature_plies": list(self.temperature_plies),
            "temperature_after": float(self.temperature_after),
            "resign": bool(self.resign),
            "technical_move_limit": int(self.technical_move_limit),
            "komi": float(self.komi),
        }

    def validate(self, *, canonical: bool = True) -> None:
        del canonical  # simulations are intentionally run-owned in Stage 5.
        if self.contract_id != CUBE_SELFPLAY_CONTRACT_ID:
            raise ValueError("Cube self-play contract id drift")
        if isinstance(self.simulations, bool) or not isinstance(self.simulations, int) or self.simulations <= 0:
            raise ValueError("Cube self-play simulations must be a positive integer")
        if not math.isfinite(float(self.cpuct)) or float(self.cpuct) <= 0.0:
            raise ValueError("Cube self-play cpuct must be positive and finite")
        if not math.isfinite(float(self.fpu)):
            raise ValueError("Cube self-play FPU must be finite")
        if type(self.root_noise) is not bool or type(self.resign) is not bool:
            raise ValueError("Cube self-play boolean settings are invalid")
        if not math.isfinite(float(self.dirichlet_epsilon)) or not 0.0 <= float(self.dirichlet_epsilon) <= 1.0:
            raise ValueError("Cube self-play Dirichlet epsilon is invalid")
        if not math.isfinite(float(self.dirichlet_alpha)) or float(self.dirichlet_alpha) <= 0.0:
            raise ValueError("Cube self-play Dirichlet alpha is invalid")
        if (
            not isinstance(self.temperature_plies, tuple)
            or len(self.temperature_plies) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.temperature_plies)
            or self.temperature_plies[0] > self.temperature_plies[1]
        ):
            raise ValueError("Cube self-play temperature schedule is invalid")
        if not math.isfinite(float(self.temperature_after)) or float(self.temperature_after) < 0.0:
            raise ValueError("Cube self-play temperature is invalid")
        if isinstance(self.technical_move_limit, bool) or not isinstance(self.technical_move_limit, int) or self.technical_move_limit <= 0:
            raise ValueError("Cube technical move limit must be positive")
        if not math.isfinite(float(self.komi)) or float(self.komi) != 0.5:
            raise ValueError("Cube self-play requires komi 0.5")


DEFAULT_CUBE_SELFPLAY_CONTRACT = CubeSelfPlaySearchContract()


def cube_action_index(action: int | str, *, point_count: int | None = None) -> int:
    if action == PASS:
        if point_count is None:
            raise ValueError("Cube PASS action indexing requires point_count")
        return int(point_count)
    if isinstance(action, bool) or not isinstance(action, int):
        raise ValueError(f"Invalid Cube action {action!r}")
    if point_count is not None and not 0 <= action <= point_count:
        raise ValueError(f"Cube action {action!r} is outside the canonical action space")
    return int(action)


def sample_cube_action_from_visits(
    result: SearchResult,
    *,
    temperature: float,
    rng: random.Random,
    point_count: int,
) -> int | str:
    from .selfplay_policy import sample_action_from_search_result

    return sample_action_from_search_result(
        result,
        temperature=temperature,
        rng=rng,
        action_index=lambda action: cube_action_index(action, point_count=point_count),
    )  # type: ignore[return-value]


def project_cube_wdl(winner: object, perspective: object) -> tuple[int, int, int]:
    return project_wdl(winner, perspective)


def project_cube_ownership(absolute_ownership: Sequence[object], perspective: object) -> tuple[str, ...]:
    return project_ownership(absolute_ownership, perspective)


def project_cube_score(margin_black: object, perspective: object) -> float:
    return project_score(margin_black, perspective)


__all__ = [
    "CUBE_DEFAULT_TECHNICAL_MOVE_LIMIT",
    "CUBE_SEMANTICS_ID",
    "CUBE_SELFPLAY_CONTRACT_ID",
    "CUBE_SELFPLAY_SEMANTICS_FINGERPRINT",
    "CUBE_SELFPLAY_CONTRACT_PATH",
    "CUBE_TARGET_CONTRACT_ID",
    "CUBE_TARGET_FINGERPRINT",
    "DEFAULT_CUBE_SELFPLAY_CONTRACT",
    "CubeSelfPlaySearchContract",
    "cube_action_index",
    "cube_selfplay_semantics_identity",
    "cube_target_contract_identity",
    "load_cube_selfplay_contract",
    "project_cube_ownership",
    "project_cube_score",
    "project_cube_wdl",
    "sample_cube_action_from_visits",
    "validate_cube_selfplay_contract",
]
