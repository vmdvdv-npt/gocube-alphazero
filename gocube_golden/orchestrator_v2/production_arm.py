"""Production A/B arm execution for Orchestrator V2.

This module is the production seam between the coordinator and one arm.  It
owns only the durable arm loop and process boundary.  ArtifactResolver
resolves the parent graph, SupervisorV2 owns child supervision, and the
existing Torus9 driver owns training-state restore, self-play, training,
checkpoint bytes, and the atomic generation commit.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys

from ..artifact_catalog import sha256_file
from ..artifact_graph import validate_generation_commit
from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from .artifact_resolver import (
    ArtifactResolver,
    ResolvedArtifact,
    ResolvedCheckpointNode,
    ResolvedEffectiveConfig,
)
from .contracts import CheckpointRef, EffectiveConfig, EffectiveConfigRef
from .experiment_runner import ArmExecutionRequest, ArmExecutionResult
from .generation_runner import OutputLineage, ResolvedGenerationInput
from .supervisor import SupervisorPolicy, SupervisorStatus, SupervisorV2
from .torus9_production import (
    Torus9ProductionGenerationPath,
    Torus9ProductionLineage,
)


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, canonical_json(dict(payload)) + "\n")


def _relative_to(root: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} escapes its owner root") from exc


def _serialize_resolved_input(
    resolved: ResolvedGenerationInput,
    *,
    runs_root: Path,
    result_path: Path,
) -> dict[str, object]:
    """Serialize resolved identities, never lookup instructions."""
    effective = resolved.effective_config
    return {
        "schema": "gocube-orchestrator-v2-generation-input-v1",
        "runs_root": str(runs_root),
        "result_path": str(result_path),
        "generation": resolved.generation,
        "parent_checkpoint": resolved.parent_checkpoint.ref.to_dict(),
        "effective_config": {
            "ref": effective.ref.to_dict(),
            "path": str(effective.path),
            "owner_root": str(effective.artifact.owner_root),
            "owner_topology": effective.artifact.owner_topology,
            "owner_lineage_id": effective.artifact.owner_lineage_id,
            "owner_status": effective.artifact.owner_status,
            "identity": dict(effective.artifact.identity or {}),
            "config": effective.config.to_dict(),
        },
        "output_lineage": {
            "topology": resolved.output_lineage.topology,
            "lineage_id": resolved.output_lineage.lineage_id,
            "root": str(resolved.output_lineage.root),
        },
    }


def _deserialize_resolved_input(payload: Mapping[str, object]) -> ResolvedGenerationInput:
    """Rehydrate the exact resolved identities handed to the child."""
    parent_ref = CheckpointRef.from_dict(payload["parent_checkpoint"])  # type: ignore[arg-type]
    runs_root = Path(str(payload["runs_root"])).resolve()
    resolver = ArtifactResolver(runs_root)
    parent = resolver.checkpoint(parent_ref)

    raw_effective = payload.get("effective_config")
    if not isinstance(raw_effective, Mapping):
        raise ValueError("generation input effective_config must be an object")
    effective_ref = EffectiveConfigRef.from_dict(raw_effective["ref"])  # type: ignore[arg-type]
    effective_path = Path(str(raw_effective["path"])).resolve()
    effective_owner_root = Path(str(raw_effective["owner_root"])).resolve()
    if _relative_to(effective_owner_root, effective_path, "effective config") != effective_ref.artifact.path:
        raise ValueError("serialized effective config path does not match its reference")
    if not effective_path.is_file() or sha256_file(effective_path) != effective_ref.artifact.sha256:
        raise ValueError(f"serialized effective config failed integrity check: {effective_path}")
    config = EffectiveConfig.from_dict(raw_effective["config"])  # type: ignore[arg-type]
    if config.fingerprint != effective_ref.fingerprint:
        raise ValueError("serialized effective config fingerprint mismatch")
    raw_identity = raw_effective.get("identity")
    effective_artifact = ResolvedArtifact(
        ref=effective_ref.artifact,
        path=effective_path,
        owner_root=effective_owner_root,
        owner_topology=str(raw_effective["owner_topology"]),
        owner_lineage_id=str(raw_effective["owner_lineage_id"]),
        owner_status=None if raw_effective.get("owner_status") is None else str(raw_effective["owner_status"]),
        identity=dict(raw_identity) if isinstance(raw_identity, Mapping) else None,
    )
    effective = ResolvedEffectiveConfig(ref=effective_ref, artifact=effective_artifact, config=config)

    raw_output = payload.get("output_lineage")
    if not isinstance(raw_output, Mapping):
        raise ValueError("generation input output_lineage must be an object")
    return ResolvedGenerationInput(
        parent_checkpoint=parent,
        generation=int(payload["generation"]),
        effective_config=effective,
        output_lineage=OutputLineage(
            str(raw_output["topology"]),
            str(raw_output["lineage_id"]),
            Path(str(raw_output["root"])),
        ),
    )


class ProductionArmExecutionPath:
    """Run one configured arm through SupervisorV2 and the real child path."""

    def __init__(
        self,
        *,
        resolver: ArtifactResolver | None = None,
        repo_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        supervisor_policy: SupervisorPolicy | None = None,
    ) -> None:
        self.resolver = resolver or ArtifactResolver()
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.python_executable = str(python_executable or sys.executable)
        self.supervisor_policy = supervisor_policy
        self.lineage_factory = Torus9ProductionLineage(
            self.resolver.runs_root,
            repo_root=self.repo_root,
        )

    def _effective_supervisor_policy(self) -> SupervisorPolicy:
        return self.supervisor_policy or SupervisorPolicy()

    def run_arm(self, request: ArmExecutionRequest) -> ArmExecutionResult:
        if request.topology != "torus9":
            raise ValueError("ProductionArmExecutionPath currently supports topology=torus9 only")
        lineage_id = request.arm.lineage_id or f"{request.experiment_id}-{request.arm.arm_id}"
        root, effective = self.lineage_factory.prepare(
            topology=request.topology,
            lineage_id=lineage_id,
            parent=request.common_parent,
            effective_config=request.arm.effective_config,
            experiment_id=request.experiment_id,
            arm_id=request.arm.arm_id,
        )
        output = OutputLineage(request.topology, lineage_id, root)
        current = request.common_parent
        target = request.common_parent.generation + request.arm.generations

        while current.generation < target:
            generation = current.generation + 1
            node_path = root / "metadata" / "checkpoints" / f"M{generation}.json"
            marker = root / f"generation-{generation:02d}.complete.json"
            if marker.is_file():
                try:
                    validate_generation_commit(
                        root=root,
                        lineage_id=lineage_id,
                        generation=generation,
                    )
                except (OSError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"generation M{generation} has a completion marker without valid graph evidence"
                    ) from exc
                if not node_path.is_file():
                    raise RuntimeError(f"committed generation M{generation} has no CheckpointNode")
                node = _read_json(node_path)
                checkpoint = self.resolver.checkpoint(node["checkpoint"])  # type: ignore[arg-type]
                if checkpoint.node.parent != current.ref:
                    raise ValueError(f"resumed generation {generation} does not descend from current parent")
                current = checkpoint
                continue

            resolved = ResolvedGenerationInput(
                parent_checkpoint=current,
                generation=generation,
                effective_config=effective,
                output_lineage=output,
            )
            current = self._run_supervised_generation(
                resolved,
                root,
            )

        return ArmExecutionResult(final_checkpoint=current)

    def _run_supervised_generation(
        self,
        resolved: ResolvedGenerationInput,
        root: Path,
    ) -> ResolvedCheckpointNode:
        generation = resolved.generation
        input_path = root / "runtime" / "v2-inputs" / f"generation-{generation:04d}.json"
        result_path = root / "runtime" / "results" / f"v2-generation-{generation:04d}.json"
        _write_json(
            input_path,
            _serialize_resolved_input(
                resolved,
                runs_root=self.resolver.runs_root,
                result_path=result_path,
            ),
        )
        child_command = [
            self.python_executable,
            "-m",
            "gocube_golden.orchestrator_v2.generation_child",
            "--request",
            str(input_path),
            "--result",
            str(result_path),
        ]
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(self.repo_root)
            if not existing_pythonpath
            else str(self.repo_root) + os.pathsep + existing_pythonpath
        )
        env["AZ_GENERATION_RESULT_PATH"] = str(
            root / "runtime" / "results" / f"generation-{generation:04d}.json"
        )
        supervisor = SupervisorV2(
            root,
            lineage_id=resolved.output_lineage.lineage_id,
            initial_committed_generation=resolved.parent_checkpoint.generation,
            command=child_command,
            cwd=self.repo_root,
            env=env,
            policy=self._effective_supervisor_policy(),
        )
        result = supervisor.run_once()
        if result.status is not SupervisorStatus.COMMITTED:
            raise RuntimeError(
                f"production generation M{generation} stopped: "
                f"{root / 'runtime' / 'supervisor-stop.json'}"
            )
        if not result_path.is_file():
            raise RuntimeError(f"production generation child returned no result: {result_path}")
        child_result = _read_json(result_path)
        checkpoint = self.resolver.checkpoint(child_result["checkpoint"])  # type: ignore[arg-type]
        if checkpoint.node.parent != resolved.parent_checkpoint.ref:
            raise ValueError(f"production generation M{generation} has the wrong immediate parent")
        if checkpoint.generation != generation:
            raise ValueError(f"production generation child returned the wrong generation: {checkpoint.ref}")
        return checkpoint


__all__ = ["ProductionArmExecutionPath"]
