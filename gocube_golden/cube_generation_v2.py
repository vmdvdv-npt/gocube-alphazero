"""Cube V2 one-generation request/config contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from generation_driver import GenerationRequest
from tools.arena_engine import ArenaExecutionConfig
from .cube_arena_contract_v2 import CUBE_ARENA_CONTRACT_ID, CubeArenaSearchConfig
from .cube_game_contract_v2 import validate_cube_size
from .cube_network_v2 import ARCHITECTURE_FINGERPRINT, ARCHITECTURE_ID
from .cube_selfplay_contract import CUBE_SELFPLAY_SEMANTICS_FINGERPRINT, CubeSelfPlaySearchContract
from .cube_selfplay_v2 import CubeSelfPlayExecutionConfig, CubeSelfPlayGameRecord
from .cube_training_contract_v2 import CubeTrainingConfig
from .cube_training_v2 import CubeTrainingAdapter

@dataclass(frozen=True)
class CubeSelfPlayPlan:
    games: int
    search: CubeSelfPlaySearchContract

    def validate(self) -> None:
        if isinstance(self.games, bool) or not isinstance(self.games, int) or self.games <= 0:
            raise ValueError("Cube generation self-play games must be a positive integer")
        if not isinstance(self.search, CubeSelfPlaySearchContract):
            raise TypeError("Cube generation requires CubeSelfPlaySearchContract")
        self.search.validate()


@dataclass(frozen=True)
class CubeArenaRequest:
    reference_checkpoint: Mapping[str, object]
    search_config: CubeArenaSearchConfig
    execution_config: ArenaExecutionConfig
    seed: int

    def validate(self) -> None:
        if not isinstance(self.reference_checkpoint, Mapping):
            raise TypeError("Cube Arena request reference_checkpoint must be a mapping")
        self.search_config.validate()
        self.execution_config.validate_base()
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed <= 0:
            raise ValueError("Cube Arena request seed must be a positive explicit integer")


@dataclass(frozen=True)
class CubeGenerationRequest:
    size: int
    generation: int
    parent_checkpoint: Mapping[str, object]
    selfplay_plan: CubeSelfPlayPlan
    selfplay_execution_config: CubeSelfPlayExecutionConfig
    training_config: CubeTrainingConfig
    arena_request: CubeArenaRequest | None
    seed: int
    lineage_id: str
    lineage_dir: Path
    profile_identity: Mapping[str, object]

    def validate(self) -> None:
        validate_cube_size(self.size)
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation <= 0:
            raise ValueError("Cube generation must be a positive integer")
        if not isinstance(self.parent_checkpoint, Mapping):
            raise TypeError("Cube generation parent checkpoint is required")
        self.selfplay_plan.validate()
        self.selfplay_execution_config.validate()
        if not isinstance(self.training_config, CubeTrainingConfig):
            raise TypeError("Cube generation requires CubeTrainingConfig")
        if self.arena_request is not None:
            self.arena_request.validate()
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed <= 0:
            raise ValueError("Cube generation seed must be a positive explicit integer")
        if not isinstance(self.lineage_id, str) or not self.lineage_id.strip():
            raise ValueError("Cube generation lineage_id is required")
        if not isinstance(self.lineage_dir, Path):
            raise TypeError("Cube generation lineage_dir must be a pathlib.Path")
        if not isinstance(self.profile_identity, Mapping) or not self.profile_identity:
            raise ValueError("Cube generation profile identity is required")

    def to_common(self) -> GenerationRequest:
        self.validate()
        return GenerationRequest(
            topology=f"cube{self.size}",
            profile_identity=dict(self.profile_identity),
            generation=self.generation,
            parent_checkpoint=dict(self.parent_checkpoint),
            selfplay_scientific_config=self.selfplay_plan,
            selfplay_execution_config=self.selfplay_execution_config,
            training_config=self.training_config,
            arena_request=self.arena_request,
            seed=self.seed,
            lineage_id=self.lineage_id,
            lineage_dir=self.lineage_dir,
        )


@dataclass(frozen=True)
class CubeSelfPlayStageResult:
    records: tuple[CubeSelfPlayGameRecord, ...]
    telemetry: Mapping[str, object]
    execution_activity: Mapping[str, object]


def build_cube_generation_profile_identity(
    *,
    training_adapter: CubeTrainingAdapter,
    selfplay_plan: CubeSelfPlayPlan,
    selfplay_execution_config: CubeSelfPlayExecutionConfig,
    arena_request: CubeArenaRequest | None,
) -> dict[str, object]:
    """Build the explicit compatibility identity carried by a generation request."""

    selfplay_plan.validate()
    selfplay_execution_config.validate()
    if arena_request is not None:
        arena_request.validate()
    return {
        "size": training_adapter.size,
        "game": {
            "identity": dict(training_adapter.game_identity),
            "fingerprint": training_adapter.game_fingerprint,
        },
        "observation": {
            "fingerprint": training_adapter.observation_fingerprint,
        },
        "architecture": {
            "id": ARCHITECTURE_ID,
            "fingerprint": ARCHITECTURE_FINGERPRINT,
        },
        "selfplay": {
            "semantics_fingerprint": CUBE_SELFPLAY_SEMANTICS_FINGERPRINT,
            "search_config": selfplay_plan.search.concrete_search_config_identity(),
            "search_config_fingerprint": selfplay_plan.search.fingerprint,
            "games": selfplay_plan.games,
        },
        "training": {
            "semantics_fingerprint": training_adapter.training_semantics_fingerprint,
            "concrete_config": training_adapter.config.identity_payload(),
            "concrete_config_fingerprint": training_adapter.concrete_training_config_fingerprint,
        },
        "arena": (
            None
            if arena_request is None
            else {
                "contract_id": CUBE_ARENA_CONTRACT_ID,
                "search_config": arena_request.search_config.identity_payload(),
                "search_config_fingerprint": arena_request.search_config.fingerprint,
            }
        ),
        "execution": {
            "selfplay": asdict(selfplay_execution_config),
            "arena": None if arena_request is None else asdict(arena_request.execution_config),
        },
    }


def build_cube_generation_request(
    *,
    training_adapter: CubeTrainingAdapter,
    generation: int,
    parent_checkpoint: Mapping[str, object],
    selfplay_plan: CubeSelfPlayPlan,
    selfplay_execution_config: CubeSelfPlayExecutionConfig,
    training_config: CubeTrainingConfig,
    arena_request: CubeArenaRequest | None,
    seed: int,
    lineage_id: str,
    lineage_dir: str | Path,
) -> CubeGenerationRequest:
    profile_identity = build_cube_generation_profile_identity(
        training_adapter=training_adapter,
        selfplay_plan=selfplay_plan,
        selfplay_execution_config=selfplay_execution_config,
        arena_request=arena_request,
    )
    return CubeGenerationRequest(
        size=training_adapter.size,
        generation=int(generation),
        parent_checkpoint=dict(parent_checkpoint),
        selfplay_plan=selfplay_plan,
        selfplay_execution_config=selfplay_execution_config,
        training_config=training_config,
        arena_request=arena_request,
        seed=int(seed),
        lineage_id=str(lineage_id),
        lineage_dir=Path(lineage_dir),
        profile_identity=profile_identity,
    )


__all__ = [
    "CubeArenaRequest", "CubeGenerationRequest", "CubeSelfPlayPlan",
    "CubeSelfPlayStageResult", "build_cube_generation_profile_identity",
    "build_cube_generation_request",
]
