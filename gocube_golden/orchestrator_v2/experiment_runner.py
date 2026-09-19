"""Small, resume-safe A/B orchestration boundary for Orchestrator V2.

The runner owns only experiment sequencing and durable references:

``parent -> arm A generations -> arm B generations -> one Arena -> STOP``

Artifact lookup, one-generation execution, process supervision, and Arena
execution remain delegated to the existing V2 components.  In particular, an
arm manifest records the common parent reference, but the parent checkpoint is
never copied into either arm lineage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .. import run_storage
from ..artifact_catalog import sha256_file
from ..process_supervision import atomic_write_text
from ..provenance import canonical_json, sha256_fingerprint
from .arena_runner import ArenaRunRequest, ArenaRunResult, ArenaRunnerV2, torus9_startset_ref
from .artifact_resolver import (
    ArtifactResolver,
    ResolvedArtifact,
    ResolvedCheckpointNode,
    ResolvedEffectiveConfig,
    checkpoint_node_path,
)
from .contracts import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    EvaluationIdentity,
    StartsetRef,
)
from .generation_runner import (
    GenerationResult,
    GenerationRunner,
    OutputLineage,
    ResolvedGenerationInput,
)
from .supervisor import SupervisorPolicy, SupervisorV2

from tools.arena_engine import ArenaExecutionConfig


EXPERIMENT_RUNNER_SCHEMA = "gocube-experiment-runner-v2"
EXPERIMENT_STATE_SCHEMA = "gocube-experiment-runner-state-v2"
_COMPONENT_RE = re.compile(r"^[^/\\]+$")


class ExperimentRunnerError(RuntimeError):
    """A persisted experiment cannot be resumed safely."""


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or not _COMPONENT_RE.fullmatch(text):
        raise ValueError(f"{label} must be one safe path component")
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_text(path, canonical_json(dict(payload)) + "\n")


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentRunnerError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ExperimentRunnerError(f"JSON artifact is not an object: {path}")
    return value


def _config_from_value(value: EffectiveConfig | Mapping[str, object]) -> EffectiveConfig:
    if isinstance(value, EffectiveConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("arm config must be EffectiveConfig or an object")
    payload = dict(value)
    # Accept a normal config object without forcing callers to repeat the
    # versioned envelope; the persisted artifact is always the strict V2 form.
    if "schema" not in payload:
        topology = str(payload.get("topology", ""))
        payload = {
            "schema": "gocube-effective-config-v2",
            "version": 2,
            "topology": topology,
            "compatibility": payload.get("compatibility", {"topology": topology}),
            "self_play": payload.get("self_play", {}),
            "training": payload.get("training", {}),
            "replay": payload.get("replay", {}),
            "execution": payload.get("execution", {}),
            "arena": payload.get("arena", {}),
            "supervision": payload.get("supervision", {}),
            "extensions": payload.get("extensions", {}),
        }
    return EffectiveConfig.from_dict(payload)


@dataclass(frozen=True)
class ExperimentArmConfig:
    """One independent arm and its run-owned effective configuration."""

    arm_id: str
    generations: int
    config: EffectiveConfig | Mapping[str, object]
    lineage_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _component(self.arm_id, "arm_id"))
        if type(self.generations) is not int or self.generations < 0:
            raise ValueError("arm generations must be a non-negative integer")
        effective = _config_from_value(self.config)
        object.__setattr__(self, "config", effective)
        if self.lineage_id is not None:
            object.__setattr__(self, "lineage_id", _component(self.lineage_id, "lineage_id"))

    @property
    def effective_config(self) -> EffectiveConfig:
        return self.config  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExperimentArmConfig":
        if not isinstance(value, Mapping):
            raise ValueError("experiment arm must be an object")
        raw_config = value.get("config", value.get("effective_config"))
        if not isinstance(raw_config, Mapping) and isinstance(value.get("config"), EffectiveConfig):
            raw_config = value["config"]
        if raw_config is None:
            raise ValueError("experiment arm config is required")
        return cls(
            arm_id=str(value.get("arm_id", value.get("id", ""))),
            generations=int(value.get("generations", value.get("iterations", 0))),
            config=raw_config,  # type: ignore[arg-type]
            lineage_id=None if value.get("lineage_id") is None else str(value["lineage_id"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "generations": self.generations,
            "lineage_id": self.lineage_id,
            "config": self.effective_config.to_dict(),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Ordinary config for exactly two arms and one final A-vs-B Arena."""

    experiment_id: str
    topology: str
    parent: CheckpointRef | Mapping[str, object]
    arms: Sequence[ExperimentArmConfig]
    arena_config: ArenaExecutionConfig
    arena_master_seed: int
    arena_startset: StartsetRef | None = None
    arena_profile: str = "torus9"
    arena_scientific_contract: Mapping[str, object] | None = None
    arena_execution_contract: Mapping[str, object] | None = None
    arena_workload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _component(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "topology", _component(self.topology, "topology"))
        parent = (
            self.parent
            if isinstance(self.parent, CheckpointRef)
            else CheckpointRef.from_dict(self.parent)
        )
        if parent.topology != self.topology:
            raise ValueError("experiment parent topology does not match experiment topology")
        object.__setattr__(self, "parent", parent)
        arms = tuple(self.arms)
        if len(arms) != 2:
            raise ValueError("ExperimentRunner V2 requires exactly two arms")
        if {arm.arm_id for arm in arms} != {"A", "B"}:
            raise ValueError("ExperimentRunner V2 arm ids must be exactly A and B")
        for arm in arms:
            if arm.effective_config.topology != self.topology:
                raise ValueError(f"arm {arm.arm_id} config topology does not match experiment")
        object.__setattr__(self, "arms", arms)
        self.arena_config.validate_base()
        if isinstance(self.arena_master_seed, bool):
            raise ValueError("arena_master_seed must be an integer")
        object.__setattr__(self, "arena_master_seed", int(self.arena_master_seed))
        if not isinstance(self.arena_workload, Mapping):
            raise ValueError("arena_workload must be an object")

    @property
    def arm_a(self) -> ExperimentArmConfig:
        return next(arm for arm in self.arms if arm.arm_id == "A")

    @property
    def arm_b(self) -> ExperimentArmConfig:
        return next(arm for arm in self.arms if arm.arm_id == "B")

    def to_dict(self) -> dict[str, object]:
        startset = None if self.arena_startset is None else self.arena_startset.to_dict()
        return {
            "schema": EXPERIMENT_RUNNER_SCHEMA,
            "experiment_id": self.experiment_id,
            "topology": self.topology,
            "parent": self.parent.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
            "arena": {
                "config": asdict(self.arena_config),
                "master_seed": self.arena_master_seed,
                "startset": startset,
                "profile": self.arena_profile,
                "scientific_contract": dict(self.arena_scientific_contract or {}),
                "execution_contract": dict(self.arena_execution_contract or {}),
                "workload": dict(self.arena_workload),
            },
        }

    @property
    def fingerprint(self) -> str:
        return sha256_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExperimentConfig":
        """Build the runner from a normal JSON-like experiment config."""
        if not isinstance(value, Mapping):
            raise ValueError("experiment config must be an object")
        raw_arms = value.get("arms")
        if not isinstance(raw_arms, Sequence) or isinstance(raw_arms, (str, bytes)):
            raise ValueError("experiment arms must be a list")
        raw_arena = value.get("arena")
        if not isinstance(raw_arena, Mapping):
            raise ValueError("experiment arena config must be an object")
        raw_execution = raw_arena.get("config", raw_arena.get("execution"))
        if not isinstance(raw_execution, Mapping):
            raise ValueError("experiment arena execution config is required")
        startset_value = raw_arena.get("startset")
        startset = (
            None
            if startset_value is None
            else StartsetRef.from_dict(startset_value)  # type: ignore[arg-type]
        )
        return cls(
            experiment_id=str(value.get("experiment_id", value.get("id", ""))),
            topology=str(value.get("topology", "")),
            parent=value.get("parent", value.get("parent_checkpoint", {})),  # type: ignore[arg-type]
            arms=tuple(ExperimentArmConfig.from_dict(raw) for raw in raw_arms),  # type: ignore[arg-type]
            arena_config=ArenaExecutionConfig(**dict(raw_execution)),
            arena_master_seed=int(raw_arena.get("master_seed", 0)),
            arena_startset=startset,
            arena_profile=str(raw_arena.get("profile", "torus9")),
            arena_scientific_contract=(
                None
                if raw_arena.get("scientific_contract") is None
                else dict(raw_arena["scientific_contract"])  # type: ignore[arg-type]
            ),
            arena_execution_contract=(
                None
                if raw_arena.get("execution_contract") is None
                else dict(raw_arena["execution_contract"])  # type: ignore[arg-type]
            ),
            arena_workload=dict(raw_arena.get("workload", {})),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ExperimentRunResult:
    state: str
    final_checkpoints: Mapping[str, ResolvedCheckpointNode]
    arena: ArenaRunResult


SupervisorFactory = Callable[[Path, str], SupervisorV2]


class ExperimentRunnerV2:
    """Execute and resume one two-arm experiment without choosing a winner."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        generation_runner: GenerationRunner,
        arena_runner: ArenaRunnerV2,
        resolver: ArtifactResolver | None = None,
        supervisor_factory: SupervisorFactory | None = None,
        supervisor_policy: SupervisorPolicy | None = None,
        experiment_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.generation_runner = generation_runner
        self.arena_runner = arena_runner
        self.resolver = resolver or ArtifactResolver()
        self.supervisor_policy = supervisor_policy
        self.supervisor_factory = supervisor_factory
        self.experiment_root = (
            Path(experiment_root).resolve()
            if experiment_root is not None
            else self.resolver.runs_root / config.topology / "experiments" / config.experiment_id
        )

    @property
    def state_path(self) -> Path:
        return self.experiment_root / "state.json"

    @property
    def config_path(self) -> Path:
        return self.experiment_root / "config.json"

    def run(self) -> ExperimentRunResult:
        parent = self.resolver.checkpoint(self.config.parent)
        state = self._load_or_create_state(parent)
        if state["state"] == "STOPPED":
            return self._result_from_stopped_state(state)

        arm_roots: dict[str, Path] = {}
        for arm in self.config.arms:
            arm_roots[arm.arm_id] = self._ensure_arm_lineage(arm, parent)

        final: dict[str, ResolvedCheckpointNode] = {}
        for arm in self.config.arms:
            final[arm.arm_id] = self._run_arm(arm, arm_roots[arm.arm_id], parent, state)

        state["state"] = "ARENA"
        state["updated_at"] = _utc_now()
        _write_json(self.state_path, state)

        arena_result = self._run_arena(final)
        state["state"] = "STOPPED"
        state["updated_at"] = _utc_now()
        state["arena_result"] = self._arena_state(arena_result)
        state["stop_reason"] = "final A-vs-B Arena completed"
        _write_json(self.state_path, state)
        return ExperimentRunResult("STOPPED", final, arena_result)

    def _load_or_create_state(self, parent: ResolvedCheckpointNode) -> dict[str, Any]:
        if self.state_path.is_file():
            state = dict(_read_json(self.state_path))
            if not self.config_path.is_file() or dict(_read_json(self.config_path)) != self.config.to_dict():
                raise ExperimentRunnerError("experiment config artifact was modified or removed")
            if state.get("schema") != EXPERIMENT_STATE_SCHEMA:
                raise ExperimentRunnerError("unsupported experiment state schema")
            if state.get("experiment_id") != self.config.experiment_id:
                raise ExperimentRunnerError("experiment state id mismatch")
            if state.get("config_fingerprint") != self.config.fingerprint:
                raise ExperimentRunnerError("experiment config changed during resume")
            if state.get("parent") != parent.ref.to_dict():
                raise ExperimentRunnerError("experiment parent changed during resume")
            if state.get("state") not in {"RUNNING", "ARENA", "STOPPED"}:
                raise ExperimentRunnerError("experiment state is malformed")
            return state

        self.experiment_root.mkdir(parents=True, exist_ok=True)
        _write_json(self.config_path, self.config.to_dict())
        arms = {
            arm.arm_id: {
                "lineage_id": self._lineage_id(arm),
                "config_fingerprint": arm.effective_config.fingerprint,
                "target_generation": parent.generation + arm.generations,
                "last_checkpoint": parent.ref.to_dict(),
            }
            for arm in self.config.arms
        }
        state: dict[str, Any] = {
            "schema": EXPERIMENT_STATE_SCHEMA,
            "version": 2,
            "experiment_id": self.config.experiment_id,
            "topology": self.config.topology,
            "config_fingerprint": self.config.fingerprint,
            "parent": parent.ref.to_dict(),
            "state": "RUNNING",
            "arms": arms,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        _write_json(self.state_path, state)
        return state

    def _lineage_id(self, arm: ExperimentArmConfig) -> str:
        return arm.lineage_id or f"{self.config.experiment_id}-{arm.arm_id}"

    def _arm_root(self, arm: ExperimentArmConfig) -> Path:
        return (
            self.resolver.runs_root
            / self.config.topology
            / run_storage.ACTIVE
            / self._lineage_id(arm)
        ).resolve()

    def _ensure_arm_lineage(self, arm: ExperimentArmConfig, parent: ResolvedCheckpointNode) -> Path:
        root = self._arm_root(arm)
        manifest = {
            "lineage_id": self._lineage_id(arm),
            "topology": self.config.topology,
            "status": "ACTIVE",
            "parent_checkpoint": parent.ref.to_dict(),
            "git_commit": "experiment-runner-v2",
            "config_fingerprint": arm.effective_config.fingerprint,
            "created_at": _utc_now(),
            "checkpoint_hashes": {},
            "experiment": {
                "id": self.config.experiment_id,
                "arm": arm.arm_id,
                "schema": EXPERIMENT_RUNNER_SCHEMA,
            },
        }
        if root.exists():
            existing = _read_json(root / "manifest.json")
            if existing.get("lineage_id") != manifest["lineage_id"]:
                raise ExperimentRunnerError(f"arm lineage id mismatch: {root}")
            if existing.get("topology") != self.config.topology:
                raise ExperimentRunnerError(f"arm lineage topology mismatch: {root}")
            if existing.get("parent_checkpoint") != parent.ref.to_dict():
                raise ExperimentRunnerError(f"arm {arm.arm_id} does not reference the common parent")
            if existing.get("config_fingerprint") != arm.effective_config.fingerprint:
                raise ExperimentRunnerError(f"arm {arm.arm_id} config changed during resume")
        else:
            run_storage.ensure_lineage_layout(
                root,
                manifest=manifest,
                extra_directories=("runtime", "metadata"),
            )
        (root / "metadata" / "config").mkdir(parents=True, exist_ok=True)
        (root / "metadata" / "provenance").mkdir(parents=True, exist_ok=True)
        (root / "metadata" / "checkpoints").mkdir(parents=True, exist_ok=True)
        self._ensure_effective_config(root, arm)
        return root

    def _ensure_effective_config(self, root: Path, arm: ExperimentArmConfig) -> EffectiveConfigRef:
        path = root / "metadata" / "config" / "effective.json"
        content = canonical_json(arm.effective_config.to_dict()) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ExperimentRunnerError(f"arm effective config was modified: {path}")
        if not path.exists():
            atomic_write_text(path, content)
        artifact = ArtifactRef("metadata/config/effective.json", sha256_file(path))
        return EffectiveConfigRef(artifact, arm.effective_config.fingerprint)

    def _resolved_effective_config(self, root: Path, arm: ExperimentArmConfig) -> ResolvedEffectiveConfig:
        ref = self._ensure_effective_config(root, arm)
        artifact = ResolvedArtifact(
            ref=ref.artifact,
            path=root / ref.artifact.path,
            owner_root=root,
            owner_topology=self.config.topology,
            owner_lineage_id=self._lineage_id(arm),
            owner_status="ACTIVE",
        )
        return ResolvedEffectiveConfig(ref=ref, artifact=artifact, config=arm.effective_config)

    def _run_arm(
        self,
        arm: ExperimentArmConfig,
        root: Path,
        parent: ResolvedCheckpointNode,
        state: dict[str, Any],
    ) -> ResolvedCheckpointNode:
        record = state["arms"][arm.arm_id]
        raw_current = record.get("last_checkpoint")
        if not isinstance(raw_current, Mapping):
            raise ExperimentRunnerError(f"arm {arm.arm_id} last checkpoint is malformed")
        current = self.resolver.checkpoint(raw_current)
        target = parent.generation + arm.generations
        if current.generation < parent.generation or current.generation > target:
            raise ExperimentRunnerError(f"arm {arm.arm_id} current generation is outside its target")

        while current.generation < target:
            generation = current.generation + 1
            existing = self._existing_child(root, arm, current, generation)
            if existing is not None:
                current = existing
            else:
                recovered = self._recover_committed_child(root, arm, current, generation)
                if recovered is not None:
                    current = recovered
                else:
                    replay_count = self._replay_count(arm.effective_config)
                    replay = self.resolver.replay_window(current, replay_count)
                    resolved_input = ResolvedGenerationInput(
                        parent_checkpoint=current,
                        replay_artifacts=replay,
                        generation=generation,
                        effective_config=self._resolved_effective_config(root, arm),
                        output_lineage=OutputLineage(self.config.topology, self._lineage_id(arm), root),
                    )
                    supervisor = self._supervisor(root, arm)
                    raw_result = supervisor.run_callable(
                        generation,
                        lambda: self.generation_runner.run(resolved_input),
                    )
                    if not isinstance(raw_result, GenerationResult):
                        raise ExperimentRunnerError("SupervisorV2 did not return a GenerationResult")
                    current = self._publish_child_node(root, arm, current, raw_result)
            record["last_checkpoint"] = current.ref.to_dict()
            record["last_generation"] = current.generation
            state["updated_at"] = _utc_now()
            _write_json(self.state_path, state)
        return current

    def _supervisor(self, root: Path, arm: ExperimentArmConfig) -> SupervisorV2:
        if self.supervisor_factory is not None:
            supervisor = self.supervisor_factory(root, self._lineage_id(arm))
        else:
            supervisor = SupervisorV2(
                root,
                lineage_id=self._lineage_id(arm),
                policy=self.supervisor_policy,
            )
        if not hasattr(supervisor, "run_callable"):
            raise TypeError("ExperimentRunner requires a SupervisorV2 with run_callable")
        return supervisor

    @staticmethod
    def _replay_count(config: EffectiveConfig) -> int:
        value = config.replay.get("generations", 0)
        if type(value) is not int or value < 0:
            raise ValueError("arm replay.generations must be a non-negative integer")
        return value

    def _existing_child(
        self,
        root: Path,
        arm: ExperimentArmConfig,
        current: ResolvedCheckpointNode,
        generation: int,
    ) -> ResolvedCheckpointNode | None:
        path = root / "metadata" / "checkpoints" / f"M{generation}.json"
        if not path.is_file():
            return None
        try:
            node = CheckpointNode.from_dict(_read_json(path))
        except (TypeError, ValueError) as exc:
            raise ExperimentRunnerError(f"invalid arm checkpoint node: {path}") from exc
        if node.genesis or node.parent != current.ref:
            raise ExperimentRunnerError(
                f"arm {arm.arm_id} checkpoint M{generation} has the wrong immediate parent"
            )
        if node.checkpoint.lineage_id != self._lineage_id(arm):
            raise ExperimentRunnerError(f"arm {arm.arm_id} checkpoint has the wrong lineage")
        return self.resolver.checkpoint(node.checkpoint)

    def _recover_committed_child(
        self,
        root: Path,
        arm: ExperimentArmConfig,
        current: ResolvedCheckpointNode,
        generation: int,
    ) -> ResolvedCheckpointNode | None:
        marker = next(
            (
                candidate
                for candidate in (
                    root / f"generation-{generation:02d}.complete.json",
                    root / f"generation-{generation:04d}.complete.json",
                )
                if candidate.is_file()
            ),
            None,
        )
        checkpoint_path = root / "checkpoints" / f"M{generation}.pt"
        if marker is None:
            return None
        if not checkpoint_path.is_file():
            raise ExperimentRunnerError(
                f"commit marker exists without checkpoint for {self._lineage_id(arm)} M{generation}"
            )
        checkpoint = CheckpointRef(
            self.config.topology,
            self._lineage_id(arm),
            f"M{generation}",
            generation,
            f"checkpoints/M{generation}.pt",
            sha256_file(checkpoint_path),
        )
        commit = ArtifactRef(
            marker.relative_to(root).as_posix(),
            sha256_file(marker),
        )
        return self._publish_child_node(
            root,
            arm,
            current,
            GenerationResult(generation, checkpoint, commit),
        )

    def _fresh_replay(self, root: Path, generation: int) -> ArtifactRef:
        candidates = (
            root / "replay" / f"iter-{generation:02d}-fresh.jsonl",
            root / "replay" / f"iter-{generation:04d}-fresh.jsonl",
            root / "replay" / f"iter-{generation}-fresh.jsonl",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise ExperimentRunnerError(
                f"committed generation M{generation} has no generation-specific fresh replay"
            )
        return ArtifactRef(path.relative_to(root).as_posix(), sha256_file(path))

    def _publish_child_node(
        self,
        root: Path,
        arm: ExperimentArmConfig,
        parent: ResolvedCheckpointNode,
        result: GenerationResult,
    ) -> ResolvedCheckpointNode:
        if result.generation != parent.generation + 1:
            raise ExperimentRunnerError("generation result is not an immediate child")
        if result.checkpoint.lineage_id != self._lineage_id(arm):
            raise ExperimentRunnerError("generation result belongs to a different arm lineage")
        config_ref = self._ensure_effective_config(root, arm)
        provenance_path = root / "metadata" / "provenance" / f"M{result.generation}.json"
        provenance = {
            "schema": "gocube-experiment-checkpoint-provenance-v2",
            "experiment_id": self.config.experiment_id,
            "arm_id": arm.arm_id,
            "checkpoint": result.checkpoint.to_dict(),
            "parent": parent.ref.to_dict(),
            "effective_config": config_ref.to_dict(),
            "generation_commit": result.commit_artifact.to_dict(),
        }
        _write_json(provenance_path, provenance)
        node = CheckpointNode(
            checkpoint=result.checkpoint,
            genesis=False,
            parent=parent.ref,
            fresh_replay=self._fresh_replay(root, result.generation),
            effective_config=config_ref,
            provenance=ArtifactRef(
                provenance_path.relative_to(root).as_posix(),
                sha256_file(provenance_path),
            ),
        )
        node_path = checkpoint_node_path(root, result.checkpoint)
        if node_path.exists():
            existing = CheckpointNode.from_dict(_read_json(node_path))
            if existing != node:
                raise ExperimentRunnerError(f"checkpoint node already exists with different identity: {node_path}")
        else:
            _write_json(node_path, node.to_dict())
        manifest_path = root / "manifest.json"
        manifest = dict(_read_json(manifest_path))
        hashes = dict(manifest.get("checkpoint_hashes", {}))
        hashes[result.checkpoint.path] = result.checkpoint.sha256
        manifest["checkpoint_hashes"] = hashes
        _write_json(manifest_path, manifest)
        return self.resolver.checkpoint(result.checkpoint)

    def _run_arena(self, final: Mapping[str, ResolvedCheckpointNode]) -> ArenaRunResult:
        startset = self.config.arena_startset
        if startset is None:
            if self.config.topology != "torus9":
                raise ValueError("arena_startset is required for non-torus9 experiments")
            startset = torus9_startset_ref(
                master_seed=self.config.arena_master_seed,
                games=self.config.arena_config.games,
            )
        request = ArenaRunRequest(
            candidate=final["A"],
            reference=final["B"],
            master_seed=self.config.arena_master_seed,
            startset=startset,
            config=self.config.arena_config,
            profile=self.config.arena_profile,
            scientific_contract=self.config.arena_scientific_contract,
            execution_contract=self.config.arena_execution_contract,
            workload=self.config.arena_workload,
            candidate_label="A-final",
            reference_label="B-final",
            comparison="A-final-vs-B-final",
        )
        return self.arena_runner.run(request)

    @staticmethod
    def _arena_state(result: ArenaRunResult) -> dict[str, object]:
        return {
            "evaluation_id": result.evaluation_id,
            "evaluation_fingerprint": result.evaluation_fingerprint,
            "output_dir": str(result.output_dir),
            "identity": result.identity.to_dict(),
            "summary": dict(result.summary),
            "validity": result.validity,
        }

    def _result_from_stopped_state(self, state: Mapping[str, object]) -> ExperimentRunResult:
        raw_arms = state.get("arms")
        raw_arena = state.get("arena_result")
        if not isinstance(raw_arms, Mapping) or not isinstance(raw_arena, Mapping):
            raise ExperimentRunnerError("STOPPED experiment state lacks final evidence")
        final: dict[str, ResolvedCheckpointNode] = {}
        for arm in self.config.arms:
            record = raw_arms.get(arm.arm_id)
            if not isinstance(record, Mapping) or not isinstance(record.get("last_checkpoint"), Mapping):
                raise ExperimentRunnerError(f"STOPPED state lacks final checkpoint for arm {arm.arm_id}")
            final[arm.arm_id] = self.resolver.checkpoint(record["last_checkpoint"])
        try:
            identity = EvaluationIdentity.from_dict(raw_arena["identity"])  # type: ignore[arg-type]
            result = ArenaRunResult(
                evaluation_id=str(raw_arena["evaluation_id"]),
                evaluation_fingerprint=str(raw_arena["evaluation_fingerprint"]),
                output_dir=Path(str(raw_arena["output_dir"])),
                identity=identity,
                summary=dict(raw_arena["summary"]),  # type: ignore[arg-type]
                validity=str(raw_arena["validity"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentRunnerError("STOPPED Arena evidence is malformed") from exc
        return ExperimentRunResult("STOPPED", final, result)


ExperimentRunner = ExperimentRunnerV2


__all__ = [
    "EXPERIMENT_RUNNER_SCHEMA",
    "EXPERIMENT_STATE_SCHEMA",
    "ExperimentArmConfig",
    "ExperimentConfig",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentRunnerError",
    "ExperimentRunnerV2",
]
