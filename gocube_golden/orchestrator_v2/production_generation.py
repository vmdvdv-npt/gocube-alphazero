"""Production one-generation boundary for Orchestrator V2.

The coordinator supplies resolved immutable references; the worker reopens
those references and chooses a topology-specific production bridge only at the
composition boundary. Production execution is capability-gated before any new
generation-owned request/artifact is written.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..artifact_graph import validate_generation_commit
from ..process_supervision import atomic_write_text
from ..process_supervision import start_owned_child
from ..provenance import canonical_json
from .artifact_resolver import ArtifactResolver, ResolvedCheckpointNode, ResolvedEffectiveConfig
from .contracts import CheckpointRef, EffectiveConfigRef
from .execution_permit import _child_execution_permit
from .generation_runner import GenerationRunner, OutputLineage, ResolvedGenerationInput
from .immutable_runtime import ImmutableRuntimeManager, execution_commit_from_lineage, validate_runtime_head
from .supervisor import SupervisorPolicy, SupervisorV2
from .topology_binding import get_topology_binding, production_path_for
from .version import require_v2_process

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


def _record_post_commit_validation_timing(output_root: Path, generation: int, elapsed: float) -> None:
    result_path = output_root / "runtime" / "results" / f"generation-{generation:04d}.json"
    if not result_path.is_file():
        return
    payload = dict(_read_json(result_path))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return
    timing = metrics.get("timing")
    if not isinstance(timing, dict):
        timing = {}
        metrics["timing"] = timing
    timing["post_commit_validation_wall_time_sec"] = float(elapsed)
    _write_json(result_path, payload)


def _request_payload(*, resolver: ArtifactResolver, parent: ResolvedCheckpointNode, config: ResolvedEffectiveConfig, output_lineage: OutputLineage, execution_overrides: Mapping[str, object] | None = None, execution_code_commit: str | None = None) -> dict[str, object]:
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
    if execution_code_commit is not None:
        payload["execution_code_commit"] = execution_code_commit
    if execution_overrides is not None:
        payload["execution_overrides"] = dict(execution_overrides)
    return payload


def _validate_child(child: ResolvedCheckpointNode, *, parent: ResolvedCheckpointNode, config: ResolvedEffectiveConfig, output_lineage: OutputLineage, generation: int) -> ResolvedCheckpointNode:
    if child.node.parent != parent.ref:
        raise RuntimeError(f"generation M{generation} does not descend from the supplied parent")
    if child.generation != generation:
        raise RuntimeError(f"generation child has generation {child.generation}, expected {generation}")
    if child.topology != output_lineage.topology or child.lineage_id != output_lineage.lineage_id:
        raise RuntimeError("generation child is owned by the wrong output lineage")
    if child.effective_config.ref != config.ref:
        raise RuntimeError("generation child uses a different effective config")
    return child


def _reuse_committed_child(resolver: ArtifactResolver, *, parent: ResolvedCheckpointNode, config: ResolvedEffectiveConfig, output_lineage: OutputLineage, generation: int) -> ResolvedCheckpointNode | None:
    marker = output_lineage.root / f"generation-{generation:02d}.complete.json"
    if not marker.is_file():
        return None
    node = validate_generation_commit(root=output_lineage.root, lineage_id=output_lineage.lineage_id, generation=generation)
    child = resolver.checkpoint(node.checkpoint)
    return _validate_child(child, parent=parent, config=config, output_lineage=output_lineage, generation=generation)


class ProductionTrainOne:
    """Run or reuse exactly one committed production generation."""

    def __init__(self, *, resolver: ArtifactResolver, repo_root: str | Path | None = None, python_executable: str | Path | None = None, supervisor_policy: SupervisorPolicy | None = None) -> None:
        self.resolver = resolver
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
        self.python_executable = str(python_executable or sys.executable)
        self.supervisor_policy = supervisor_policy
        self.runtime_manager = ImmutableRuntimeManager(self.repo_root)

    def __call__(self, *, parent: ResolvedCheckpointNode, config: ResolvedEffectiveConfig, output_lineage: OutputLineage, execution_overrides: Mapping[str, object] | None = None, acknowledge_stopped_execution: bool = False) -> ResolvedCheckpointNode:
        require_v2_process("gocube_golden.orchestrator_v2.ProductionTrainOne")
        get_topology_binding(output_lineage.topology)
        if parent.ref.topology != output_lineage.topology or config.config.topology != output_lineage.topology:
            raise ValueError("production train_one topology identities disagree")
        if type(acknowledge_stopped_execution) is not bool:
            raise TypeError("acknowledge_stopped_execution must be a boolean")
        generation = parent.generation + 1
        runtime = None
        execution_commit: str | None = None
        if (output_lineage.root / "manifest.json").is_file():
            execution_commit = execution_commit_from_lineage(output_lineage.root)
            runtime = self.runtime_manager.ensure(execution_commit)
        reused = _reuse_committed_child(self.resolver, parent=parent, config=config, output_lineage=output_lineage, generation=generation)
        if reused is not None:
            heartbeat_path = output_lineage.root / "runtime" / "heartbeats" / f"generation-{generation:04d}.json"
            reconciliation = SupervisorV2(
                output_lineage.root,
                execution_id=f"{output_lineage.lineage_id}:generation:{generation}",
                liveness_path=heartbeat_path,
                progress_path=heartbeat_path,
                policy=self.supervisor_policy,
            ).reconcile_completed_execution()
            if not reconciliation.success:
                raise RuntimeError(f"committed generation M{generation} supervisor reconciliation failed: {reconciliation.reason or 'unknown reason'}")
            return reused
        if runtime is None or execution_commit is None:
            raise RuntimeError("production generation requires a committed lineage manifest/code identity")

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
                execution_code_commit=runtime.commit,
            ),
        )
        heartbeat_path = output_lineage.root / "runtime" / "heartbeats" / f"generation-{generation:04d}.json"
        command = [
            self.python_executable,
            "-m",
            "gocube_golden.orchestrator_v2.generation_child",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]

        def launch(child_request):
            # A permit belongs to one actual child attempt.  It is minted in
            # the launcher so a retry after a long generation gets a fresh
            # TTL, while a supervisor restart can still reattach to the
            # existing child without changing its capability.
            env = runtime.environment(os.environ)
            env["AZ_DRIVER_HEARTBEAT_PATH"] = str(heartbeat_path)
            env["AZ_GENERATION_RESULT_PATH"] = str(
                output_lineage.root / "runtime" / "results" / f"generation-{generation:04d}.json"
            )
            with _child_execution_permit(
                action_type="generation",
                topology=output_lineage.topology,
                run_id=output_lineage.lineage_id,
                code_identity=runtime.commit,
                attempt=int(child_request.attempt),
            ) as permit:
                env["AZ_V2_EXECUTION_PERMIT"] = canonical_json(dict(permit))
                env["AZ_V2_EXECUTION_PERMIT_KEY"] = os.environ["AZ_V2_EXECUTION_PERMIT_KEY"]
                return start_owned_child(
                    command,
                    cwd=runtime.path,
                    env=env,
                    popen=subprocess.Popen,
                )

        supervisor = SupervisorV2(
            output_lineage.root,
            execution_id=f"{output_lineage.lineage_id}:generation:{generation}",
            liveness_path=heartbeat_path,
            progress_path=heartbeat_path,
            launcher=launch,
            command=command,
            cwd=runtime.path,
            policy=self.supervisor_policy,
        )
        if acknowledge_stopped_execution:
            acknowledgement = supervisor.acknowledge_stopped_execution()
            if not acknowledgement.success:
                raise RuntimeError(f"could not acknowledge stopped execution: {acknowledgement.reason or 'unknown reason'}")
        result = supervisor.run_once()
        if not result.success:
            raise RuntimeError(f"production generation M{generation} stopped: {result.reason or output_lineage.root / 'runtime' / 'supervisor-stop.json'}")

        post_commit_started = time.perf_counter()
        validate_generation_commit(
            root=output_lineage.root,
            lineage_id=output_lineage.lineage_id,
            generation=generation,
            reuse_committed_rolling_replay_identity=True,
        )
        _record_post_commit_validation_timing(output_lineage.root, generation, time.perf_counter() - post_commit_started)
        child_payload = _read_json(result_path)
        if child_payload.get("schema") != TRAIN_ONE_RESULT_SCHEMA:
            raise RuntimeError("generation worker returned an invalid result schema")
        raw_child = child_payload.get("checkpoint")
        if not isinstance(raw_child, Mapping):
            raise RuntimeError("generation worker returned no child checkpoint ref")
        child = self.resolver.checkpoint(CheckpointRef.from_dict(raw_child))
        return _validate_child(child, parent=parent, config=config, output_lineage=output_lineage, generation=generation)


def run_generation_worker(request_path: str | Path, result_path: str | Path) -> None:
    payload = _read_json(Path(request_path).resolve())
    if payload.get("schema") != TRAIN_ONE_REQUEST_SCHEMA:
        raise ValueError("unsupported train_one request schema")
    execution_commit = payload.get("execution_code_commit")
    if not isinstance(execution_commit, str) or not execution_commit:
        raise ValueError("train-one request is missing execution_code_commit")
    validate_runtime_head(Path.cwd(), execution_commit)
    raw_lineage = payload.get("output_lineage")
    if not isinstance(raw_lineage, Mapping):
        raise ValueError("train-one output_lineage must be an object")
    output = OutputLineage(str(raw_lineage["topology"]), str(raw_lineage["lineage_id"]), Path(str(raw_lineage["root"])))
    get_topology_binding(output.topology)
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
    result = GenerationRunner(production_path_for(output.topology)).run(resolved)
    _write_json(
        Path(result_path).resolve(),
        {
            "schema": TRAIN_ONE_RESULT_SCHEMA,
            "generation": result.generation,
            "checkpoint": result.checkpoint.to_dict(),
            "execution_code_commit": execution_commit,
        },
    )


__all__ = ["ProductionTrainOne", "TRAIN_ONE_REQUEST_SCHEMA", "TRAIN_ONE_RESULT_SCHEMA", "run_generation_worker"]
