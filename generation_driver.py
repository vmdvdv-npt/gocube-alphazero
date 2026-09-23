"""Topology-neutral one-generation coordination.

The driver coordinates one explicit generation request. Scientific semantics,
game/model code, execution engines, lifecycle scheduling, and retry policy stay
behind adapters/callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Mapping, Protocol

GENERATION_REQUEST_SCHEMA_VERSION = 1
GENERATION_RESULT_SCHEMA_VERSION = 1


class GenerationStage(str, Enum):
    SELFPLAY_COMPLETE = "SELFPLAY_COMPLETE"
    TRAINING_COMPLETE = "TRAINING_COMPLETE"
    CHECKPOINT_COMMITTED = "CHECKPOINT_COMMITTED"
    ARENA_COMPLETE = "ARENA_COMPLETE"
    GENERATION_COMPLETE = "GENERATION_COMPLETE"


class GenerationCompletion(str, Enum):
    COMPLETE = "complete"
    FAILED = "failed"
    ARENA_FAILED = "arena_failed"


@dataclass(frozen=True)
class GenerationRequest:
    topology: str
    profile_identity: Mapping[str, object]
    generation: int
    parent_checkpoint: Mapping[str, object] | None
    selfplay_scientific_config: object
    selfplay_execution_config: object
    training_config: object
    arena_request: object | None
    seed: int
    lineage_id: str
    lineage_dir: Path
    schema_version: int = GENERATION_REQUEST_SCHEMA_VERSION

    def validate_core(self) -> None:
        if self.schema_version != GENERATION_REQUEST_SCHEMA_VERSION:
            raise ValueError("Generation request schema version is incompatible")
        if not isinstance(self.topology, str) or not self.topology.strip():
            raise ValueError("Generation request topology is required")
        if not isinstance(self.lineage_id, str) or not self.lineage_id.strip():
            raise ValueError("Generation request lineage_id is required")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation <= 0:
            raise ValueError("Generation request generation must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed <= 0:
            raise ValueError("Generation request seed must be a positive explicit integer")
        if not isinstance(self.profile_identity, Mapping) or not self.profile_identity:
            raise ValueError("Generation request profile_identity is required")
        if self.parent_checkpoint is not None and not isinstance(self.parent_checkpoint, Mapping):
            raise TypeError("Generation parent_checkpoint must be a mapping or None")
        if not isinstance(self.lineage_dir, Path):
            raise TypeError("Generation lineage_dir must be a pathlib.Path")


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    parent_checkpoint: Mapping[str, object] | None
    selfplay_result_summary: Mapping[str, object]
    formal_games: int
    technical_games: int
    samples_generated: int
    replay_state: Mapping[str, object]
    training_metrics: Mapping[str, object]
    checkpoint_reference: Mapping[str, object] | None
    arena_result: Mapping[str, object] | None
    timing_breakdown: Mapping[str, float]
    warnings: tuple[str, ...]
    completion_status: str
    stages: tuple[str, ...]
    error: str | None = None
    schema_version: int = GENERATION_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "parent_checkpoint": None if self.parent_checkpoint is None else dict(self.parent_checkpoint),
            "selfplay_result_summary": dict(self.selfplay_result_summary),
            "formal_games": self.formal_games,
            "technical_games": self.technical_games,
            "samples_generated": self.samples_generated,
            "replay_state": dict(self.replay_state),
            "training_metrics": dict(self.training_metrics),
            "checkpoint_reference": None if self.checkpoint_reference is None else dict(self.checkpoint_reference),
            "arena_result": None if self.arena_result is None else dict(self.arena_result),
            "timing_breakdown": dict(self.timing_breakdown),
            "warnings": list(self.warnings),
            "completion_status": self.completion_status,
            "stages": list(self.stages),
            "error": self.error,
        }


class GenerationAdapter(Protocol):
    def committed_generation_exists(self, request: GenerationRequest) -> bool: ...
    def validate_request(self, request: GenerationRequest) -> None: ...
    def run_selfplay(self, request: GenerationRequest) -> object: ...
    def validate_selfplay_result(self, request: GenerationRequest, result: object) -> None: ...
    def summarize_selfplay(self, request: GenerationRequest, result: object) -> Mapping[str, object]: ...
    def run_training(self, request: GenerationRequest, selfplay_result: object) -> object: ...
    def summarize_training(self, request: GenerationRequest, result: object) -> Mapping[str, object]: ...
    def run_arena(self, request: GenerationRequest, training_result: object) -> object: ...
    def summarize_arena(self, request: GenerationRequest, result: object) -> Mapping[str, object]: ...
    def validate_result(self, request: GenerationRequest, result: GenerationResult) -> None: ...
    def persist_result(self, request: GenerationRequest, result: GenerationResult) -> None: ...


class GenerationDriver:
    """Execute exactly one explicitly requested generation."""

    def __init__(self, adapter: GenerationAdapter) -> None:
        self.adapter = adapter

    @staticmethod
    def _error(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"

    def _finish(self, request: GenerationRequest, result: GenerationResult) -> GenerationResult:
        self.adapter.validate_result(request, result)
        self.adapter.persist_result(request, result)
        return result

    def run(self, request: GenerationRequest) -> GenerationResult:
        request.validate_core()
        if self.adapter.committed_generation_exists(request):
            raise FileExistsError(
                f"Generation M{request.generation} already has published artifacts; refusing overwrite"
            )
        self.adapter.validate_request(request)

        started = time.perf_counter()
        timing: dict[str, float] = {}
        stages: list[str] = []
        warnings: list[str] = []
        selfplay_summary: Mapping[str, object] = {}
        training_summary: Mapping[str, object] = {}
        checkpoint_reference: Mapping[str, object] | None = None
        arena_summary: Mapping[str, object] | None = None

        selfplay_started = time.perf_counter()
        try:
            selfplay_result = self.adapter.run_selfplay(request)
            self.adapter.validate_selfplay_result(request, selfplay_result)
            selfplay_summary = dict(self.adapter.summarize_selfplay(request, selfplay_result))
            stages.append(GenerationStage.SELFPLAY_COMPLETE.value)
        except Exception as exc:
            timing["selfplay_wall_time_sec"] = time.perf_counter() - selfplay_started
            timing["generation_wall_time_sec"] = time.perf_counter() - started
            return self._finish(
                request,
                GenerationResult(
                    generation=request.generation,
                    parent_checkpoint=request.parent_checkpoint,
                    selfplay_result_summary=selfplay_summary,
                    formal_games=int(selfplay_summary.get("formal_games", 0)),
                    technical_games=int(selfplay_summary.get("technical_games", 0)),
                    samples_generated=0,
                    replay_state={},
                    training_metrics={},
                    checkpoint_reference=None,
                    arena_result=None,
                    timing_breakdown=timing,
                    warnings=(),
                    completion_status=GenerationCompletion.FAILED.value,
                    stages=tuple(stages),
                    error=self._error(exc),
                ),
            )
        timing["selfplay_wall_time_sec"] = time.perf_counter() - selfplay_started

        training_started = time.perf_counter()
        try:
            training_result = self.adapter.run_training(request, selfplay_result)
            training_summary = dict(self.adapter.summarize_training(request, training_result))
            checkpoint = training_summary.get("checkpoint_reference")
            if not isinstance(checkpoint, Mapping):
                raise RuntimeError("Generation adapter did not return a committed checkpoint reference")
            checkpoint_reference = dict(checkpoint)
            stages.extend(
                (
                    GenerationStage.TRAINING_COMPLETE.value,
                    GenerationStage.CHECKPOINT_COMMITTED.value,
                )
            )
        except Exception as exc:
            timing["training_wall_time_sec"] = time.perf_counter() - training_started
            timing["generation_wall_time_sec"] = time.perf_counter() - started
            return self._finish(
                request,
                GenerationResult(
                    generation=request.generation,
                    parent_checkpoint=request.parent_checkpoint,
                    selfplay_result_summary=selfplay_summary,
                    formal_games=int(selfplay_summary.get("formal_games", 0)),
                    technical_games=int(selfplay_summary.get("technical_games", 0)),
                    samples_generated=0,
                    replay_state={},
                    training_metrics={},
                    checkpoint_reference=None,
                    arena_result=None,
                    timing_breakdown=timing,
                    warnings=(),
                    completion_status=GenerationCompletion.FAILED.value,
                    stages=tuple(stages),
                    error=self._error(exc),
                ),
            )
        timing["training_wall_time_sec"] = time.perf_counter() - training_started

        arena_error: str | None = None
        if request.arena_request is not None:
            arena_started = time.perf_counter()
            try:
                arena_result = self.adapter.run_arena(request, training_result)
                arena_summary = dict(self.adapter.summarize_arena(request, arena_result))
                stages.append(GenerationStage.ARENA_COMPLETE.value)
            except Exception as exc:
                arena_error = self._error(exc)
                warnings.append(arena_error)
            timing["arena_wall_time_sec"] = time.perf_counter() - arena_started

        samples_generated = int(training_summary.get("samples_generated", 0))
        replay_state = training_summary.get("replay_state", {})
        training_metrics = training_summary.get("training_metrics", {})
        if not isinstance(replay_state, Mapping) or not isinstance(training_metrics, Mapping):
            raise RuntimeError("Generation adapter returned malformed training summary")

        if arena_error is None:
            stages.append(GenerationStage.GENERATION_COMPLETE.value)
            status = GenerationCompletion.COMPLETE.value
        else:
            status = GenerationCompletion.ARENA_FAILED.value

        timing["generation_wall_time_sec"] = time.perf_counter() - started
        result = GenerationResult(
            generation=request.generation,
            parent_checkpoint=request.parent_checkpoint,
            selfplay_result_summary=selfplay_summary,
            formal_games=int(selfplay_summary.get("formal_games", 0)),
            technical_games=int(selfplay_summary.get("technical_games", 0)),
            samples_generated=samples_generated,
            replay_state=dict(replay_state),
            training_metrics=dict(training_metrics),
            checkpoint_reference=checkpoint_reference,
            arena_result=arena_summary,
            timing_breakdown=timing,
            warnings=tuple(warnings),
            completion_status=status,
            stages=tuple(stages),
            error=arena_error,
        )
        return self._finish(request, result)


__all__ = [
    "GENERATION_REQUEST_SCHEMA_VERSION",
    "GENERATION_RESULT_SCHEMA_VERSION",
    "GenerationAdapter",
    "GenerationCompletion",
    "GenerationDriver",
    "GenerationRequest",
    "GenerationResult",
    "GenerationStage",
]
