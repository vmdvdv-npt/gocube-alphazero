"""Coordinator-only A/B orchestration boundary for Orchestrator V2.

The runner owns the experiment sequence and durable coordination state only:

    common parent -> arm A -> arm B -> one Arena -> STOP

An injected arm execution path owns the already-proven generation path:
ArtifactResolver, SupervisorV2, GenerationRunner, and the production child
that atomically commits a generation.  This module neither creates lineage
artifacts nor publishes checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol

from ..process_supervision import atomic_write_text
from ..provenance import canonical_json, sha256_fingerprint
from .arena_runner import ArenaRunRequest, ArenaRunResult, ArenaRunnerV2, torus9_startset_ref
from .artifact_resolver import ArtifactResolver, ResolvedCheckpointNode
from .contracts import CheckpointRef, EffectiveConfig, EvaluationIdentity, StartsetRef

from tools.arena_engine import ArenaExecutionConfig


EXPERIMENT_RUNNER_SCHEMA = "gocube-experiment-runner-v2"
EXPERIMENT_STATE_SCHEMA = "gocube-experiment-runner-state-v2"


class ExperimentRunnerError(RuntimeError):
    """A persisted experiment cannot be resumed safely."""


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


def _config_from_value(value: EffectiveConfig | Mapping[str, object]) -> EffectiveConfig:
    if isinstance(value, EffectiveConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("arm config must be EffectiveConfig or an object")
    payload = dict(value)
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
    """One arm and its ordinary run-owned effective configuration."""

    arm_id: str
    generations: int
    config: EffectiveConfig | Mapping[str, object]
    lineage_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _component(self.arm_id, "arm_id"))
        if type(self.generations) is not int or self.generations < 0:
            raise ValueError("arm generations must be a non-negative integer")
        object.__setattr__(self, "config", _config_from_value(self.config))
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
        parent = self.parent if isinstance(self.parent, CheckpointRef) else CheckpointRef.from_dict(self.parent)
        if parent.topology != self.topology:
            raise ValueError("experiment parent topology does not match experiment topology")
        object.__setattr__(self, "parent", parent)
        arms = tuple(self.arms)
        if len(arms) != 2 or {arm.arm_id for arm in arms} != {"A", "B"}:
            raise ValueError("ExperimentRunner V2 requires exactly the A and B arms")
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
        return {
            "schema": EXPERIMENT_RUNNER_SCHEMA,
            "experiment_id": self.experiment_id,
            "topology": self.topology,
            "parent": self.parent.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
            "arena": {
                "config": asdict(self.arena_config),
                "master_seed": self.arena_master_seed,
                "startset": None if self.arena_startset is None else self.arena_startset.to_dict(),
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
        raw_startset = raw_arena.get("startset")
        return cls(
            experiment_id=str(value.get("experiment_id", value.get("id", ""))),
            topology=str(value.get("topology", "")),
            parent=value.get("parent", value.get("parent_checkpoint", {})),  # type: ignore[arg-type]
            arms=tuple(ExperimentArmConfig.from_dict(raw) for raw in raw_arms),  # type: ignore[arg-type]
            arena_config=ArenaExecutionConfig(**dict(raw_execution)),
            arena_master_seed=int(raw_arena.get("master_seed", 0)),
            arena_startset=None if raw_startset is None else StartsetRef.from_dict(raw_startset),  # type: ignore[arg-type]
            arena_profile=str(raw_arena.get("profile", "torus9")),
            arena_scientific_contract=(
                None if raw_arena.get("scientific_contract") is None
                else dict(raw_arena["scientific_contract"])  # type: ignore[arg-type]
            ),
            arena_execution_contract=(
                None if raw_arena.get("execution_contract") is None
                else dict(raw_arena["execution_contract"])  # type: ignore[arg-type]
            ),
            arena_workload=dict(raw_arena.get("workload", {})),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ArmExecutionRequest:
    """One request handed to an already-proven production arm execution path."""

    experiment_id: str
    topology: str
    arm: ExperimentArmConfig
    common_parent: ResolvedCheckpointNode


@dataclass(frozen=True)
class ArmExecutionResult:
    """The only artifact returned by an arm execution path to the coordinator."""

    final_checkpoint: ResolvedCheckpointNode


class ArmExecutionPath(Protocol):
    """Production seam for the existing V2 generation execution path.

    A production implementation is expected to bind the common parent and
    arm config to ArtifactResolver, SupervisorV2, GenerationRunner, and the
    production child.  The coordinator does not implement or alter that path.
    """

    def run_arm(self, request: ArmExecutionRequest) -> ArmExecutionResult:
        """Execute the configured generations and return the resolved final."""


@dataclass(frozen=True)
class ExperimentRunResult:
    state: str
    final_checkpoints: Mapping[str, ResolvedCheckpointNode]
    arena: ArenaRunResult


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, canonical_json(dict(payload)) + "\n")


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentRunnerError(f"cannot read experiment state: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ExperimentRunnerError(f"experiment state is not an object: {path}")
    return payload


class ExperimentRunnerV2:
    """Coordinate two independent arms and one final Arena, then stop."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        arm_execution_path: ArmExecutionPath,
        arena_runner: ArenaRunnerV2,
        resolver: ArtifactResolver | None = None,
        experiment_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.arm_execution_path = arm_execution_path
        self.arena_runner = arena_runner
        self.resolver = resolver or ArtifactResolver()
        self.experiment_root = (
            Path(experiment_root).resolve()
            if experiment_root is not None
            else self.resolver.runs_root / config.topology / "experiments" / config.experiment_id
        )

    @property
    def state_path(self) -> Path:
        return self.experiment_root / "state.json"

    def run(self) -> ExperimentRunResult:
        parent = self.resolver.checkpoint(self.config.parent)
        state = self._load_or_create_state(parent)
        if state["state"] == "STOPPED":
            return self._result_from_stopped_state(state)

        final: dict[str, ResolvedCheckpointNode] = {}
        for arm in (self.config.arm_a, self.config.arm_b):
            record = state["arms"][arm.arm_id]
            raw_final = record.get("final_checkpoint")
            if isinstance(raw_final, Mapping):
                checkpoint = self.resolver.checkpoint(raw_final)
            else:
                result = self.arm_execution_path.run_arm(
                    ArmExecutionRequest(
                        experiment_id=self.config.experiment_id,
                        topology=self.config.topology,
                        arm=arm,
                        common_parent=parent,
                    )
                )
                if not isinstance(result, ArmExecutionResult):
                    raise ExperimentRunnerError("arm execution path returned an invalid result")
                checkpoint = result.final_checkpoint
                self._validate_final(arm, parent, checkpoint)
                record["final_checkpoint"] = checkpoint.ref.to_dict()
                state["updated_at"] = self._now()
                _write_json(self.state_path, state)
            self._validate_final(arm, parent, checkpoint)
            final[arm.arm_id] = checkpoint

        if final["A"].lineage_id == final["B"].lineage_id:
            raise ExperimentRunnerError("A and B final checkpoints must have independent lineages")

        state["state"] = "ARENA"
        state["updated_at"] = self._now()
        _write_json(self.state_path, state)
        arena_result = self._run_arena(final)

        state["state"] = "STOPPED"
        state["updated_at"] = self._now()
        state["arena_result"] = self._arena_state(arena_result)
        state["stop_reason"] = "final A-vs-B Arena completed"
        _write_json(self.state_path, state)
        return ExperimentRunResult("STOPPED", final, arena_result)

    def _load_or_create_state(self, parent: ResolvedCheckpointNode) -> dict[str, Any]:
        if self.state_path.is_file():
            state = dict(_read_json(self.state_path))
            if state.get("schema") != EXPERIMENT_STATE_SCHEMA:
                raise ExperimentRunnerError("unsupported experiment state schema")
            if state.get("experiment_id") != self.config.experiment_id:
                raise ExperimentRunnerError("experiment state id mismatch")
            if state.get("topology") != self.config.topology:
                raise ExperimentRunnerError("experiment state topology mismatch")
            if state.get("config_fingerprint") != self.config.fingerprint:
                raise ExperimentRunnerError("experiment config changed during resume")
            if state.get("parent") != parent.ref.to_dict():
                raise ExperimentRunnerError("experiment parent changed during resume")
            if state.get("state") not in {"RUNNING", "ARENA", "STOPPED"}:
                raise ExperimentRunnerError("experiment state is malformed")
            arms = state.get("arms")
            if not isinstance(arms, Mapping) or set(arms) != {"A", "B"}:
                raise ExperimentRunnerError("experiment state arms are malformed")
            state["arms"] = {key: dict(value) for key, value in arms.items() if isinstance(value, Mapping)}
            if set(state["arms"]) != {"A", "B"}:
                raise ExperimentRunnerError("experiment state arm records are malformed")
            return state

        state: dict[str, Any] = {
            "schema": EXPERIMENT_STATE_SCHEMA,
            "version": 2,
            "experiment_id": self.config.experiment_id,
            "topology": self.config.topology,
            "config_fingerprint": self.config.fingerprint,
            "parent": parent.ref.to_dict(),
            "state": "RUNNING",
            "arms": {
                arm.arm_id: {
                    "config_fingerprint": arm.effective_config.fingerprint,
                    "target_generation": parent.generation + arm.generations,
                }
                for arm in self.config.arms
            },
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        _write_json(self.state_path, state)
        return state

    def _validate_final(
        self,
        arm: ExperimentArmConfig,
        parent: ResolvedCheckpointNode,
        final: ResolvedCheckpointNode,
    ) -> None:
        if not isinstance(final, ResolvedCheckpointNode):
            raise ExperimentRunnerError(f"arm {arm.arm_id} returned an unresolved checkpoint")
        if final.topology != self.config.topology:
            raise ExperimentRunnerError(f"arm {arm.arm_id} final topology does not match experiment")
        expected_generation = parent.generation + arm.generations
        if final.generation != expected_generation:
            raise ExperimentRunnerError(
                f"arm {arm.arm_id} final generation is {final.generation}, "
                f"expected {expected_generation}"
            )
        if final.lineage_id == parent.lineage_id:
            raise ExperimentRunnerError(f"arm {arm.arm_id} final reuses the common parent lineage")
        if arm.lineage_id is not None and final.lineage_id != arm.lineage_id:
            raise ExperimentRunnerError(f"arm {arm.arm_id} final has the wrong configured lineage")
        try:
            ancestor = self.resolver.ancestor(final, arm.generations)
        except Exception as exc:
            raise ExperimentRunnerError(
                f"arm {arm.arm_id} final does not resolve to the explicit common parent"
            ) from exc
        if ancestor.ref != parent.ref:
            raise ExperimentRunnerError(f"arm {arm.arm_id} final has the wrong explicit parent")

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
            candidate=final["B"],
            reference=final["A"],
            master_seed=self.config.arena_master_seed,
            startset=startset,
            config=self.config.arena_config,
            profile=self.config.arena_profile,
            scientific_contract=self.config.arena_scientific_contract,
            execution_contract=self.config.arena_execution_contract,
            workload=self.config.arena_workload,
            candidate_label="B-final",
            reference_label="A-final",
            comparison="B-final-vs-A-final",
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
        parent = self.resolver.checkpoint(self.config.parent)
        final: dict[str, ResolvedCheckpointNode] = {}
        for arm in (self.config.arm_a, self.config.arm_b):
            record = raw_arms.get(arm.arm_id)
            if not isinstance(record, Mapping) or not isinstance(record.get("final_checkpoint"), Mapping):
                raise ExperimentRunnerError(f"STOPPED state lacks final checkpoint for arm {arm.arm_id}")
            checkpoint = self.resolver.checkpoint(record["final_checkpoint"])
            self._validate_final(arm, parent, checkpoint)
            final[arm.arm_id] = checkpoint
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

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()


ExperimentRunner = ExperimentRunnerV2


__all__ = [
    "EXPERIMENT_RUNNER_SCHEMA",
    "EXPERIMENT_STATE_SCHEMA",
    "ArmExecutionPath",
    "ArmExecutionRequest",
    "ArmExecutionResult",
    "ExperimentArmConfig",
    "ExperimentConfig",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentRunnerError",
    "ExperimentRunnerV2",
]
