"""Production one-generation boundary for Orchestrator V2.

This module owns the production-specific process boundary around the already
strictly one-generation ``GenerationRunner``.  The coordinator supplies a
resolved parent, a prepared output lineage, and an effective-config artifact;
the child process receives only immutable references and resolves the full
objects locally.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys

from ..artifact_graph import validate_generation_commit
from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from .artifact_resolver import ArtifactResolver, ResolvedCheckpointNode, ResolvedEffectiveConfig
from .contracts import CheckpointRef, EffectiveConfigRef
from .generation_runner import GenerationRunner, OutputLineage, ResolvedGenerationInput
from .supervisor import SupervisorPolicy, SupervisorV2
from .torus9_production import Torus9ProductionGenerationPath


TRAIN_ONE_REQUEST_SCHEMA = "gocube-orchestrator-v2-train-one-request-v1"
TRAIN_ONE_RESULT_SCHEMA = "gocube-orchestrator-v2-train-one-result-v1"


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, canonical_json(dict(payload)) + "\n")


def _request_payload(
    *,
    resolver: ArtifactResolver,
    parent: ResolvedCheckpointNode,
    config: ResolvedEffectiveConfig,
    output_lineage: OutputLineage,
    execution_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the small immutable-ref request used by the worker process."""
    payload: dict[str, object] = {
        "schema": TRAIN_ONE_REQUEST_SCHEMA,
        "runs_root": str(resolver.runs_root),
        "parent_checkpoint": parent.ref.to_dict(),
        "effective_config": config.ref.to_dict(),
        "output_lineage": {
            "topology": output_lineage.topology,
            "lineage_id": output_lineage.lineage_id,
            "root": str(output_lineage.root),
        },
    }
    if execution_overrides is not None:
        payload["execution_overrides"] = dict(execution_overrides)
    return payload


def _validate_child(
    child: ResolvedCheckpointNode,
    *,
    parent: ResolvedCheckpointNode,
    config: ResolvedEffectiveConfig,
    output_lineage: OutputLineage,
    generation: int,
) -> ResolvedCheckpointNode:
    if child.node.parent != parent.ref:
        raise RuntimeError(f"generation M{generation} does not descend from the supplied parent")
    if child.generation != generation:
        raise RuntimeError(f"generation child has generation {child.generation}, expected {generation}")
    if child.topology != output_lineage.topology or child.lineage_id != output_lineage.lineage_id:
        raise RuntimeError("generation child is owned by the wrong output lineage")
    if child.effective_config.ref != config.ref:
        raise RuntimeError("generation child uses a different effective config")
    return child


def _reuse_committed_child(
    resolver: ArtifactResolver,
    *,
    parent: ResolvedCheckpointNode,
    config: ResolvedEffectiveConfig,
    output_lineage: OutputLineage,
    generation: int,
) -> ResolvedCheckpointNode | None:
    """Reuse only a fully committed generation; partial evidence is retried."""
    marker = output_lineage.root / f"generation-{generation:02d}.complete.json"
    if not marker.is_file():
        return None
    node = validate_generation_commit(
        root=output_lineage.root,
        lineage_id=output_lineage.lineage_id,
        generation=generation,
    )
    child = resolver.checkpoint(node.checkpoint)
    return _validate_child(
        child,
        parent=parent,
        config=config,
        output_lineage=output_lineage,
        generation=generation,
    )


class ProductionTrainOne:
    """Run or reuse exactly one committed production generation."""

    def __init__(
        self,
        *,
        resolver: ArtifactResolver,
        repo_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        supervisor_policy: SupervisorPolicy | None = None,
    ) -> None:
        self.resolver = resolver
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.python_executable = str(python_executable or sys.executable)
        self.supervisor_policy = supervisor_policy

    def __call__(
        self,
        *,
        parent: ResolvedCheckpointNode,
        config: ResolvedEffectiveConfig,
        output_lineage: OutputLineage,
        execution_overrides: Mapping[str, object] | None = None,
    ) -> ResolvedCheckpointNode:
        if output_lineage.topology != "torus9":
            raise ValueError("production train_one currently supports topology=torus9 only")
        generation = parent.generation + 1
        reused = _reuse_committed_child(
            self.resolver,
            parent=parent,
            config=config,
            output_lineage=output_lineage,
            generation=generation,
        )
        if reused is not None:
            return reused

        result_path = output_lineage.root / "runtime" / "results" / f"train-one-{generation:04d}.json"
        request_path = output_lineage.root / "runtime" / "requests" / f"train-one-{generation:04d}.json"
        _write_json(
            request_path,
            _request_payload(
                resolver=self.resolver,
                parent=parent,
                config=config,
                output_lineage=output_lineage,
                execution_overrides=execution_overrides,
            ),
        )
        heartbeat_path = output_lineage.root / "runtime" / "heartbeats" / f"generation-{generation:04d}.json"
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(self.repo_root)
            if not existing_pythonpath
            else str(self.repo_root) + os.pathsep + existing_pythonpath
        )
        env["AZ_DRIVER_HEARTBEAT_PATH"] = str(heartbeat_path)
        env["AZ_GENERATION_RESULT_PATH"] = str(
            output_lineage.root / "runtime" / "results" / f"generation-{generation:04d}.json"
        )
        supervisor = SupervisorV2(
            output_lineage.root,
            execution_id=f"{output_lineage.lineage_id}:generation:{generation}",
            liveness_path=heartbeat_path,
            progress_path=heartbeat_path,
            command=[
                self.python_executable,
                "-m",
                "gocube_golden.orchestrator_v2.generation_child",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            cwd=self.repo_root,
            env=env,
            policy=self.supervisor_policy,
        )
        result = supervisor.run_once()
        if not result.success:
            raise RuntimeError(
                f"production generation M{generation} stopped: "
                f"{result.reason or output_lineage.root / 'runtime' / 'supervisor-stop.json'}"
            )

        # The graph/commit boundary is authoritative.  The worker result is
        # intentionally only a transport hint, never a second recovery record.
        validate_generation_commit(
            root=output_lineage.root,
            lineage_id=output_lineage.lineage_id,
            generation=generation,
        )
        child_payload = _read_json(result_path)
        if child_payload.get("schema") != TRAIN_ONE_RESULT_SCHEMA:
            raise RuntimeError("generation worker returned an invalid result schema")
        raw_child = child_payload.get("checkpoint")
        if not isinstance(raw_child, Mapping):
            raise RuntimeError("generation worker returned no child checkpoint ref")
        child = self.resolver.checkpoint(CheckpointRef.from_dict(raw_child))
        return _validate_child(
            child,
            parent=parent,
            config=config,
            output_lineage=output_lineage,
            generation=generation,
        )


def run_generation_worker(request_path: str | Path, result_path: str | Path) -> None:
    """Resolve a ref-only request and execute one production generation."""
    payload = _read_json(Path(request_path).resolve())
    if payload.get("schema") != TRAIN_ONE_REQUEST_SCHEMA:
        raise ValueError("unsupported train_one request schema")
    raw_lineage = payload.get("output_lineage")
    if not isinstance(raw_lineage, Mapping):
        raise ValueError("train-one output_lineage must be an object")
    output = OutputLineage(
        str(raw_lineage["topology"]),
        str(raw_lineage["lineage_id"]),
        Path(str(raw_lineage["root"])),
    )
    resolver = ArtifactResolver(Path(str(payload["runs_root"])).resolve())
    parent = resolver.checkpoint(CheckpointRef.from_dict(payload["parent_checkpoint"]))  # type: ignore[arg-type]
    config = resolver.effective_config(
        EffectiveConfigRef.from_dict(payload["effective_config"]),  # type: ignore[arg-type]
        owner_root=output.root,
        topology=output.topology,
        lineage_id=output.lineage_id,
        owner_status="ACTIVE",
    )
    raw_overrides = payload.get("execution_overrides")
    if raw_overrides is not None and not isinstance(raw_overrides, Mapping):
        raise ValueError("train-one execution_overrides must be an object")
    resolved = ResolvedGenerationInput(
        parent_checkpoint=parent,
        generation=parent.generation + 1,
        effective_config=config,
        output_lineage=output,
        execution_overrides=None if raw_overrides is None else dict(raw_overrides),
    )
    result = GenerationRunner(Torus9ProductionGenerationPath()).run(resolved)
    _write_json(
        Path(result_path).resolve(),
        {
            "schema": TRAIN_ONE_RESULT_SCHEMA,
            "generation": result.generation,
            "checkpoint": result.checkpoint.to_dict(),
        },
    )


__all__ = [
    "ProductionTrainOne",
    "TRAIN_ONE_REQUEST_SCHEMA",
    "TRAIN_ONE_RESULT_SCHEMA",
    "run_generation_worker",
]
