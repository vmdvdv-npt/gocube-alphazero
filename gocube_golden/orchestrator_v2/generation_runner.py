"""One-shot V2 generation boundary.

The resolver owns discovery; topology production paths own self-play,
replay/training and artifact publication. GenerationRunner only connects those
boundaries and enforces the V2 execution capability for real production paths.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..artifact_catalog import sha256_file
from .artifact_resolver import ResolvedCheckpointNode, ResolvedEffectiveConfig
from .contracts import ArtifactRef, CheckpointRef
from .execution_permit import active_authority, require_child_execution_permit


def _safe_component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


@dataclass(frozen=True)
class OutputLineage:
    topology: str
    lineage_id: str
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology", _safe_component(self.topology, "topology"))
        object.__setattr__(self, "lineage_id", _safe_component(self.lineage_id, "lineage_id"))
        object.__setattr__(self, "root", Path(self.root).resolve())


@dataclass(frozen=True)
class ResolvedGenerationInput:
    parent_checkpoint: ResolvedCheckpointNode
    generation: int
    effective_config: ResolvedEffectiveConfig
    output_lineage: OutputLineage
    execution_overrides: Mapping[str, object] | None = None
    action_id: str | None = None
    scientific_contract_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if self.execution_overrides is not None:
            if not isinstance(self.execution_overrides, Mapping):
                raise ValueError("execution_overrides must be an object")
            allowed = {"active_games_per_worker", "total_active_contexts"}
            unknown = set(self.execution_overrides) - allowed
            if unknown:
                raise ValueError("execution_overrides contains unsupported fields: " + ", ".join(sorted(unknown)))
            for key in allowed:
                if key in self.execution_overrides:
                    value = self.execution_overrides[key]
                    if type(value) is not int or value <= 0:
                        raise ValueError(f"execution_overrides.{key} must be positive")
            object.__setattr__(self, "execution_overrides", dict(self.execution_overrides))
        if self.action_id is not None:
            if not isinstance(self.action_id, str) or not self.action_id.strip():
                raise ValueError("action_id must be a non-empty string")
        if self.scientific_contract_fingerprint is not None:
            if not isinstance(self.scientific_contract_fingerprint, str) or not self.scientific_contract_fingerprint:
                raise ValueError("scientific_contract_fingerprint must be a non-empty string")


@dataclass(frozen=True)
class GenerationExecutionResult:
    generation: int
    committed: bool
    checkpoint: CheckpointRef | None = None
    commit_artifact: ArtifactRef | None = None
    fresh_replay: ArtifactRef | None = None


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    checkpoint: CheckpointRef
    commit_artifact: ArtifactRef

    @property
    def committed_checkpoint(self) -> CheckpointRef:
        return self.checkpoint

    @property
    def commit_marker(self) -> ArtifactRef:
        return self.commit_artifact


class GenerationNotCommitted(RuntimeError):
    pass


class ProductionGenerationPath(Protocol):
    def run_generation(self, resolved_input: ResolvedGenerationInput) -> GenerationExecutionResult: ...


def _is_real_production_path(value: object) -> bool:
    return type(value).__module__ in {
        "gocube_golden.orchestrator_v2.torus9_production",
        "gocube_golden.orchestrator_v2.cube_production",
        "gocube_golden.orchestrator_v2.cube_production_recovery",
    }


class GenerationRunner:
    def __init__(self, production_path: ProductionGenerationPath) -> None:
        self.production_path = production_path

    def run(self, resolved_input: ResolvedGenerationInput) -> GenerationResult:
        if _is_real_production_path(self.production_path) and active_authority() is None:
            require_child_execution_permit(
                "gocube_golden.orchestrator_v2.GenerationRunner",
                action_type="generation",
                topology=resolved_input.output_lineage.topology,
                run_id=resolved_input.output_lineage.lineage_id,
            )
        execution = self.production_path.run_generation(resolved_input)
        if not isinstance(execution, GenerationExecutionResult):
            raise TypeError("production path returned an invalid generation result")
        if not execution.committed:
            raise GenerationNotCommitted(f"Generation {resolved_input.generation} did not reach commit")
        if execution.generation != resolved_input.generation:
            raise GenerationNotCommitted("Production generation result number does not match resolved input")
        checkpoint = execution.checkpoint
        commit_artifact = execution.commit_artifact
        if checkpoint is None or commit_artifact is None:
            raise GenerationNotCommitted("Committed generation result must include checkpoint and commit artifact")
        output = resolved_input.output_lineage
        if checkpoint.topology != output.topology or checkpoint.lineage_id != output.lineage_id:
            raise GenerationNotCommitted("Committed child checkpoint is not owned by the output lineage")
        if checkpoint.generation != resolved_input.generation:
            raise GenerationNotCommitted("Committed child checkpoint generation does not match resolved input")
        self._verify_owned_artifact(output, checkpoint.path, checkpoint.sha256, "checkpoint")
        self._verify_owned_artifact(output, commit_artifact.path, commit_artifact.sha256, "commit artifact")
        return GenerationResult(generation=execution.generation, checkpoint=checkpoint, commit_artifact=commit_artifact)

    @staticmethod
    def _verify_owned_artifact(output: OutputLineage, relative_path: str, expected_sha256: str, label: str) -> None:
        path = (output.root / relative_path).resolve()
        if output.root not in path.parents:
            raise GenerationNotCommitted(f"Committed {label} escapes output lineage")
        if not path.is_file():
            raise GenerationNotCommitted(f"Committed {label} is missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise GenerationNotCommitted(f"Committed {label} hash mismatch: expected {expected_sha256}, got {actual}")


__all__ = [
    "GenerationExecutionResult", "GenerationNotCommitted", "GenerationResult", "GenerationRunner",
    "OutputLineage", "ProductionGenerationPath", "ResolvedGenerationInput",
]
