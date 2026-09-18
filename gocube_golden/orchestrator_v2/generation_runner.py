"""One-shot V2 generation boundary.

The V2 resolver owns discovery.  The production generation path owns
self-play, replay/training, artifact publication, and the existing atomic
commit/catalog protocol.  ``GenerationRunner`` only connects those two
boundaries: it forwards an already-resolved input unchanged and accepts a
result only after the production path has reported a committed generation.

No ancestry or replay discovery belongs here.  In particular, this module
must not call ``parent()``, ``ancestor()``, or ``replay_window()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from ..artifact_catalog import sha256_file
from .artifact_resolver import ResolvedArtifact, ResolvedCheckpointNode, ResolvedEffectiveConfig
from .contracts import ArtifactRef, CheckpointRef


def _safe_component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


@dataclass(frozen=True)
class OutputLineage:
    """The already-created lineage which owns the child generation output."""

    topology: str
    lineage_id: str
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology", _safe_component(self.topology, "topology"))
        object.__setattr__(self, "lineage_id", _safe_component(self.lineage_id, "lineage_id"))
        object.__setattr__(self, "root", Path(self.root).resolve())


@dataclass(frozen=True)
class ResolvedGenerationInput:
    """All inputs required for exactly one generation.

    The values are deliberately resolved objects, not lookup keys.  The
    runner does not normalize or replace them so the production path sees the
    exact parent, replay artifacts, and effective config selected by the
    caller/resolver, including cross-lineage references.
    """

    parent_checkpoint: ResolvedCheckpointNode
    replay_artifacts: Sequence[ResolvedArtifact]
    generation: int
    effective_config: ResolvedEffectiveConfig
    output_lineage: OutputLineage

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")


@dataclass(frozen=True)
class GenerationExecutionResult:
    """Proof returned by the production path after one-shot execution.

    A failed or incomplete path returns ``committed=False`` and no result can
    be promoted by ``GenerationRunner``.  The production path is responsible
    for writing the checkpoint, commit marker, catalog entry, and other
    generation artifacts before returning a committed result.
    """

    generation: int
    committed: bool
    checkpoint: CheckpointRef | None = None
    commit_artifact: ArtifactRef | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Small successful result exposed by ``GenerationRunner.run``."""

    generation: int
    checkpoint: CheckpointRef
    commit_artifact: ArtifactRef

    @property
    def committed_checkpoint(self) -> CheckpointRef:
        """Explicit alias for callers that prefer the full result wording."""
        return self.checkpoint

    @property
    def commit_marker(self) -> ArtifactRef:
        """Compatibility alias for the authoritative commit artifact."""
        return self.commit_artifact


class GenerationNotCommitted(RuntimeError):
    """The production path did not produce an authoritative commit."""


class ProductionGenerationPath(Protocol):
    """Existing production generation pipeline used by the runner."""

    def run_generation(
        self, resolved_input: ResolvedGenerationInput
    ) -> GenerationExecutionResult: ...


class GenerationRunner:
    """Thin one-shot wrapper around an existing production generation path."""

    def __init__(self, production_path: ProductionGenerationPath) -> None:
        self.production_path = production_path

    def run(self, resolved_input: ResolvedGenerationInput) -> GenerationResult:
        """Execute one generation and return only an atomically committed result.

        ``resolved_input`` is passed as the same object to the production
        path.  This is intentional: the runner must not discover ancestry,
        select a replay window, or reconstruct configuration.
        """
        execution = self.production_path.run_generation(resolved_input)
        if not isinstance(execution, GenerationExecutionResult):
            raise TypeError("production path returned an invalid generation result")
        if not execution.committed:
            raise GenerationNotCommitted(
                f"Generation {resolved_input.generation} did not reach commit"
            )
        if execution.generation != resolved_input.generation:
            raise GenerationNotCommitted(
                "Production generation result number does not match resolved input"
            )
        checkpoint = execution.checkpoint
        commit_artifact = execution.commit_artifact
        if checkpoint is None or commit_artifact is None:
            raise GenerationNotCommitted(
                "Committed generation result must include checkpoint and commit artifact"
            )

        output = resolved_input.output_lineage
        if checkpoint.topology != output.topology or checkpoint.lineage_id != output.lineage_id:
            raise GenerationNotCommitted(
                "Committed child checkpoint is not owned by the output lineage"
            )
        if checkpoint.generation != resolved_input.generation:
            raise GenerationNotCommitted(
                "Committed child checkpoint generation does not match resolved input"
            )

        self._verify_owned_artifact(output, checkpoint.path, checkpoint.sha256, "checkpoint")
        self._verify_owned_artifact(output, commit_artifact.path, commit_artifact.sha256, "commit artifact")
        return GenerationResult(
            generation=execution.generation,
            checkpoint=checkpoint,
            commit_artifact=commit_artifact,
        )

    @staticmethod
    def _verify_owned_artifact(
        output: OutputLineage,
        relative_path: str,
        expected_sha256: str,
        label: str,
    ) -> None:
        path = (output.root / relative_path).resolve()
        if output.root not in path.parents:
            raise GenerationNotCommitted(f"Committed {label} escapes output lineage")
        if not path.is_file():
            raise GenerationNotCommitted(f"Committed {label} is missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise GenerationNotCommitted(
                f"Committed {label} hash mismatch: expected {expected_sha256}, got {actual}"
            )


__all__ = [
    "GenerationExecutionResult",
    "GenerationNotCommitted",
    "GenerationResult",
    "GenerationRunner",
    "OutputLineage",
    "ProductionGenerationPath",
    "ResolvedGenerationInput",
]
