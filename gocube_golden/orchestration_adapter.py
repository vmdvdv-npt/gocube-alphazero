"""Typed game-independent boundary between the training supervisor and adapters.

The orchestrator never imports Cube/Torus rules.  A concrete adapter owns the
scientific phases and publishes durable results through the environment/path
contract.  The process adapter below is the current production transport; the
protocol is intentionally broader so an in-process adapter can implement the
same lifecycle without changing the supervisor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class AdapterContext:
    lineage_id: str
    topology: str
    run_root: Path
    profile_path: Path
    profile_fingerprint: str
    generation: int
    resume: bool


@dataclass(frozen=True)
class ProcessAdapterCommands:
    generation: tuple[str, ...]
    resume_generation: tuple[str, ...]
    arena: tuple[str, ...] | None

    @classmethod
    def from_sequences(
        cls,
        generation: Sequence[str],
        resume_generation: Sequence[str],
        arena: Sequence[str] | None,
    ) -> "ProcessAdapterCommands":
        return cls(
            generation=tuple(str(value) for value in generation),
            resume_generation=tuple(str(value) for value in resume_generation),
            arena=(tuple(str(value) for value in arena) if arena is not None else None),
        )


@runtime_checkable
class TrainingAdapter(Protocol):
    """Scientific lifecycle owned by a topology/profile adapter."""

    adapter_id: str

    def prepare(self, context: AdapterContext) -> None: ...

    def run_self_play(self, context: AdapterContext) -> Mapping[str, object]: ...

    def update_replay(self, context: AdapterContext) -> Mapping[str, object]: ...

    def train(self, context: AdapterContext) -> Mapping[str, object]: ...

    def save_checkpoint(self, context: AdapterContext) -> Mapping[str, object]: ...

    def validate_checkpoint(self, context: AdapterContext) -> Mapping[str, object]: ...

    def run_arena(self, context: AdapterContext) -> Mapping[str, object]: ...

    def collect_metrics(self, context: AdapterContext) -> Mapping[str, object]: ...

    def publish_report(self, context: AdapterContext) -> Mapping[str, object]: ...

    def resume(self, context: AdapterContext) -> Mapping[str, object]: ...

    def cleanup(self, context: AdapterContext) -> None: ...


__all__ = ["AdapterContext", "ProcessAdapterCommands", "TrainingAdapter"]
