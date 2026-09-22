"""Coordinator for one ordinary, resumable Orchestrator V2 lineage.

The continuous runner deliberately owns only sequencing and durable operator
state.  A generation is still executed by ``ProductionTrainOne`` and an
Arena is still executed by ``ArenaRunnerV2``.  No self-play, replay,
checkpoint, or Arena decision logic belongs here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import uuid

from ..artifact_graph import CheckpointRef, EffectiveConfig
from ..process_supervision import atomic_write_text
from ..provenance import canonical_json, sha256_fingerprint
from ..run_storage import ACTIVE
from .arena_runner import ArenaRunRequest, ArenaRunResult, ArenaRunnerV2, torus9_startset_ref
from .artifact_resolver import ArtifactResolver, ResolvedCheckpointNode, ResolvedEffectiveConfig
from .contracts import EvaluationIdentity, StartsetRef
from .experiment_runner import LineageFactory, TrainOne
from .generation_runner import OutputLineage
from .production_generation import ProductionTrainOne
from .torus9_production import Torus9ProductionLineage
from .version import ORCHESTRATOR_VERSION

from tools.arena_engine import ArenaExecutionConfig, DEFAULT_MASTER_SEED


CONTINUOUS_TRAINING_SCHEMA = "gocube-continuous-training-v2"


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


def _effective_config(
    value: EffectiveConfig | ResolvedEffectiveConfig | Mapping[str, object],
) -> EffectiveConfig:
    if isinstance(value, EffectiveConfig):
        return value
    if isinstance(value, ResolvedEffectiveConfig):
        return value.config
    if not isinstance(value, Mapping):
        raise TypeError("effective_config must be an EffectiveConfig or an object")
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


def _arena_config(
    value: ArenaExecutionConfig | Mapping[str, object],
) -> tuple[ArenaExecutionConfig, int | None]:
    if isinstance(value, ArenaExecutionConfig):
        return value, None
    if not isinstance(value, Mapping):
        raise TypeError("arena_config must be an ArenaExecutionConfig or an object")

    outer = dict(value)
    reference_gap = outer.get("reference_gap")
    raw = outer.get("config", outer.get("execution", outer))
    if not isinstance(raw, Mapping):
        raise ValueError("arena_config execution must be an object")
    fields = set(ArenaExecutionConfig.__dataclass_fields__)
    kwargs = {key: raw[key] for key in fields if key in raw}
    config = ArenaExecutionConfig(**kwargs)
    if reference_gap is None:
        reference_gap = raw.get("reference_gap")
    if reference_gap is None:
        return config, None
    if type(reference_gap) is not int or reference_gap <= 0:
        raise ValueError("Arena reference_gap must be a positive integer")
    return config, int(reference_gap)


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not an object: {path}")
    return payload


def _write_object(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, canonical_json(dict(payload)) + "\n")


@dataclass(frozen=True)
class ContinuousTrainingConfig:
    """Immutable operator input for one continuous lineage."""

    parent_checkpoint: CheckpointRef | Mapping[str, object]
    lineage_id: str
    effective_config: EffectiveConfig | ResolvedEffectiveConfig | Mapping[str, object]
    generations: int | None
    arena_cadence: int
    arena_config: ArenaExecutionConfig | Mapping[str, object]
    arena_master_seed: int = DEFAULT_MASTER_SEED
    arena_startset: StartsetRef | Mapping[str, object] | None = None
    arena_profile: str = "torus9"
    arena_scientific_contract: Mapping[str, object] | None = None
    arena_execution_contract: Mapping[str, object] | None = None
    arena_workload: Mapping[str, object] = field(default_factory=dict)
    arena_reference_gap: int | None = None
    allow_code_rollover: bool = False

    def __post_init__(self) -> None:
        parent = (
            self.parent_checkpoint
            if isinstance(self.parent_checkpoint, CheckpointRef)
            else CheckpointRef.from_dict(self.parent_checkpoint)
        )
        config = _effective_config(self.effective_config)
        arena_config, inferred_gap = _arena_config(self.arena_config)
        gap = self.arena_reference_gap if self.arena_reference_gap is not None else inferred_gap
        if gap is None:
            raw_gap = config.arena.get("reference_gap")
            gap = raw_gap if raw_gap is not None else 1
        object.__setattr__(self, "parent_checkpoint", parent)
        object.__setattr__(self, "lineage_id", _component(self.lineage_id, "lineage_id"))
        object.__setattr__(self, "effective_config", config)
        object.__setattr__(self, "arena_config", arena_config)
        if config.topology != parent.topology:
            raise ValueError("effective_config topology does not match parent checkpoint")
        if config.topology != "torus9":
            raise ValueError("ContinuousTrainingRunnerV2 currently supports topology=torus9 only")
        if self.generations is not None and (type(self.generations) is not int or self.generations < 0):
            raise ValueError("generations must be a non-negative integer or None")
        _positive_int(self.arena_cadence, "arena_cadence")
        if type(gap) is not int or gap <= 0:
            raise ValueError("arena_reference_gap must be a positive integer")
        object.__setattr__(self, "arena_reference_gap", gap)
        if isinstance(self.arena_master_seed, bool):
            raise ValueError("arena_master_seed must be an integer")
        object.__setattr__(self, "arena_master_seed", int(self.arena_master_seed))
        if not isinstance(self.arena_profile, str) or not self.arena_profile:
            raise ValueError("arena_profile must be a non-empty string")
        if self.arena_scientific_contract is not None and not isinstance(
            self.arena_scientific_contract, Mapping
        ):
            raise ValueError("arena_scientific_contract must be an object")
        if self.arena_execution_contract is not None and not isinstance(
            self.arena_execution_contract, Mapping
        ):
            raise ValueError("arena_execution_contract must be an object")
        if not isinstance(self.arena_workload, Mapping):
            raise ValueError("arena_workload must be an object")
        if type(self.allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        startset = self.arena_startset
        if startset is None:
            startset = torus9_startset_ref(
                master_seed=int(self.arena_master_seed),
                games=int(arena_config.games),
            )
        elif isinstance(startset, Mapping):
            startset = StartsetRef.from_dict(startset)
        if not isinstance(startset, StartsetRef):
            raise TypeError("arena_startset must be a StartsetRef or an object")
        object.__setattr__(self, "arena_startset", startset)

    @property
    def topology(self) -> str:
        return self.parent_checkpoint.topology  # type: ignore[union-attr]

    @property
    def target_generation(self) -> int | None:
        if self.generations is None:
            return None
        return self.parent_checkpoint.generation + self.generations  # type: ignore[union-attr]


@dataclass(frozen=True)
class ContinuousTrainingResult:
    state: str
    original_parent: ResolvedCheckpointNode
    final_checkpoint: ResolvedCheckpointNode
    committed_generations: tuple[ResolvedCheckpointNode, ...]
    arenas: tuple[ArenaRunResult, ...]
    lineage_root: Path
    soft_stop_requested: bool

    @property
    def current_checkpoint(self) -> ResolvedCheckpointNode:
        return self.final_checkpoint

    @property
    def last_committed_generation(self) -> int:
        return self.final_checkpoint.generation


class ContinuousTrainingRunnerV2:
    """Run one stable Torus9 lineage through repeated ``train_one`` calls."""

    def __init__(
        self,
        config: ContinuousTrainingConfig | None = None,
        *,
        parent_checkpoint: CheckpointRef | Mapping[str, object] | None = None,
        parent: CheckpointRef | Mapping[str, object] | None = None,
        lineage_id: str | None = None,
        effective_config: EffectiveConfig | ResolvedEffectiveConfig | Mapping[str, object] | None = None,
        generations: int | None = None,
        arena_cadence: int | None = None,
        arena_config: ArenaExecutionConfig | Mapping[str, object] | None = None,
        arena_master_seed: int = DEFAULT_MASTER_SEED,
        arena_startset: StartsetRef | Mapping[str, object] | None = None,
        arena_profile: str = "torus9",
        arena_scientific_contract: Mapping[str, object] | None = None,
        arena_execution_contract: Mapping[str, object] | None = None,
        arena_workload: Mapping[str, object] | None = None,
        arena_reference_gap: int | None = None,
        allow_code_rollover: bool | None = None,
        resolver: ArtifactResolver | None = None,
        arena_runner: ArenaRunnerV2 | None = None,
        lineage_factory: LineageFactory | None = None,
        train_one: TrainOne | None = None,
        reporter: Callable[..., None] | None = None,
        notifier: object | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if config is None:
            selected_parent = parent_checkpoint if parent_checkpoint is not None else parent
            if selected_parent is None or lineage_id is None or effective_config is None:
                raise TypeError(
                    "parent_checkpoint, lineage_id, effective_config, generations, "
                    "arena_cadence, and arena_config are required"
                )
            if arena_cadence is None or arena_config is None:
                raise TypeError("arena_cadence and arena_config are required")
            config = ContinuousTrainingConfig(
                parent_checkpoint=selected_parent,
                lineage_id=lineage_id,
                effective_config=effective_config,
                generations=generations,
                arena_cadence=arena_cadence,
                arena_config=arena_config,
                arena_master_seed=arena_master_seed,
                arena_startset=arena_startset,
                arena_profile=arena_profile,
                arena_scientific_contract=arena_scientific_contract,
                arena_execution_contract=arena_execution_contract,
                arena_workload={} if arena_workload is None else arena_workload,
                arena_reference_gap=arena_reference_gap,
                allow_code_rollover=False if allow_code_rollover is None else allow_code_rollover,
            )
        else:
            direct_override = (
                parent_checkpoint,
                parent,
                lineage_id,
                effective_config,
                arena_cadence,
                arena_config,
            )
            if any(value is not None for value in direct_override):
                raise TypeError("pass either config or direct ContinuousTrainingRunnerV2 inputs, not both")
            if allow_code_rollover is not None:
                if type(allow_code_rollover) is not bool:
                    raise TypeError("allow_code_rollover must be a boolean")
                config = replace(config, allow_code_rollover=allow_code_rollover)

        self.config = config
        self.resolver = resolver or ArtifactResolver()
        self.arena_runner = arena_runner or ArenaRunnerV2()
        self.lineage_factory = lineage_factory or Torus9ProductionLineage(self.resolver.runs_root)
        self.train_one = train_one or ProductionTrainOne(resolver=self.resolver)
        self.reporter = reporter
        self._notifier = notifier
        self.logger = logger or logging.getLogger(__name__)
        self._lineage_root: Path | None = None
        self._resolved_config: ResolvedEffectiveConfig | None = None

    @property
    def lineage_root(self) -> Path:
        if self._lineage_root is not None:
            return self._lineage_root
        return (
            self.resolver.runs_root
            / self.config.topology
            / ACTIVE
            / self.config.lineage_id
        ).resolve()

    @property
    def state_path(self) -> Path:
        return self.lineage_root / "runtime" / "state.json"

    @property
    def soft_stop_path(self) -> Path:
        return self.lineage_root / "control" / "soft-stop.json"

    def request_soft_stop(self, *, reason: str = "operator") -> dict[str, object]:
        """Durably request a safe stop after the current generation boundary."""
        if not self.lineage_root.joinpath("manifest.json").is_file():
            raise RuntimeError("continuous lineage has not been prepared")
        payload = {
            "schema": f"{CONTINUOUS_TRAINING_SCHEMA}-soft-stop-v1",
            "requested_at": _now(),
            "requested_by": str(reason),
            "mode": "finish-current-generation-no-next-generation",
        }
        _write_object(self.soft_stop_path, payload)
        if self.state_path.is_file():
            state = _read_object(self.state_path, "continuous training state")
            state["soft_stop_requested"] = True
            self._persist_state(state)
        self._notify_operator(
            "SOFT_STOP_REQUESTED",
            "A soft stop was requested; the active generation will finish before the safe boundary.",
            key_suffix=f"soft-stop-requested:{payload['requested_at']}",
        )
        return payload

    def resume(self) -> ContinuousTrainingResult:
        """Explicit convenience entrypoint which consumes a soft-stop request."""
        self.soft_stop_path.unlink(missing_ok=True)
        return self.run()

    def run(self) -> ContinuousTrainingResult:
        self._launch_id = uuid.uuid4().hex
        original_parent = self.resolver.checkpoint(self.config.parent_checkpoint)
        prepare_args: dict[str, object] = {
            "topology": self.config.topology,
            "lineage_id": self.config.lineage_id,
            "parent": original_parent,
            "effective_config": self.config.effective_config,
            "experiment_id": f"continuous-{self.config.lineage_id}",
            "arm_id": "continuous",
        }
        if self.config.allow_code_rollover:
            prepare_args["allow_code_rollover"] = True
        root, resolved_config = self.lineage_factory.prepare(**prepare_args)
        self._lineage_root = Path(root).resolve()
        expected_root = (
            self.resolver.runs_root
            / self.config.topology
            / ACTIVE
            / self.config.lineage_id
        ).resolve()
        if self._lineage_root != expected_root:
            raise RuntimeError("continuous lineage factory returned a non-canonical active lineage")
        if resolved_config.fingerprint != self.config.effective_config.fingerprint:
            raise RuntimeError("lineage resolved a different effective config")
        self._resolved_config = resolved_config
        output_lineage = OutputLineage(self.config.topology, self.config.lineage_id, self._lineage_root)
        self._ensure_runtime_layout()
        self._persist_operator_metadata(original_parent)
        state, current = self._load_or_create_state(original_parent)

        # A stopped lineage is resumable by a new launch.  A RUNNING lineage
        # with a stop request remains stopped at its next safe boundary.
        if state.get("state") == "SOFT_STOPPED":
            self.soft_stop_path.unlink(missing_ok=True)
            state["soft_stop_requested"] = False
            state["state"] = "RUNNING"
            self._persist_state(state)

        self._report_start(original_parent, resolved_config)
        committed: list[ResolvedCheckpointNode] = []
        arenas = self._stored_arenas(state)

        if state.get("state") == "COMPLETED":
            self._notify_operator(
                "COMPLETED",
                f"Continuous training already completed at M{current.generation}.",
                key_suffix=f"completed:{current.generation}",
            )
            return self._result(state, original_parent, current, committed, arenas)

        while True:
            target = self.config.target_generation
            if self.soft_stop_path.is_file():
                state.update(
                    {
                        "state": "SOFT_STOPPED",
                        "soft_stop_requested": True,
                        "active_generation": None,
                        "active_phase": None,
                    }
                )
                self._persist_state(state)
                self._report(
                    "soft_stop",
                    "Soft stop reached a safe generation boundary; lineage remains resumable",
                    generation=current.generation,
                )
                self._notify_operator(
                    "SOFT_STOPPED",
                    f"Soft stop completed at M{current.generation}; lineage remains resumable.",
                    key_suffix=f"soft-stopped:{current.generation}",
                )
                return self._result(state, original_parent, current, committed, arenas)

            # A completed generation can be followed by a process interruption
            # before Arena/state tail publication.  Re-enter the existing Arena
            # identity/reuse boundary before deciding whether to start another
            # generation.
            if self._arena_due(original_parent, current.generation) and not self._arena_recorded(
                state, current.generation
            ):
                arena = self._run_arena(original_parent, current)
                arenas.append(arena)
                self._record_arena(state, arena, current.generation)

            if target is not None and current.generation >= target:
                state.update(
                    {
                        "state": "COMPLETED",
                        "soft_stop_requested": False,
                        "active_generation": None,
                        "active_phase": None,
                    }
                )
                self._persist_state(state)
                self._notify_operator(
                    "COMPLETED",
                    f"Continuous training completed at M{current.generation}.",
                    key_suffix=f"completed:{current.generation}",
                )
                return self._result(state, original_parent, current, committed, arenas)

            next_generation = current.generation + 1
            state.update(
                {
                    "state": "RUNNING",
                    "active_generation": next_generation,
                    "active_phase": "generation",
                    "soft_stop_requested": False,
                }
            )
            self._persist_state(state)
            previous = current
            try:
                current = self.train_one(
                    parent=current,
                    config=resolved_config,
                    output_lineage=output_lineage,
                )
            except BaseException as exc:
                self._notify_operator(
                    "CRITICAL",
                    f"Generation M{next_generation} failed: {exc.__class__.__name__}.",
                    key_suffix=f"critical:generation:{next_generation}",
                )
                raise
            self._validate_child(previous, current, resolved_config, next_generation)
            committed.append(current)
            state.update(
                {
                    "current_checkpoint": current.ref.to_dict(),
                    "last_committed_generation": current.generation,
                    "active_generation": None,
                    "active_phase": None,
                }
            )
            self._persist_state(state)
            next_arena = self._next_arena_generation(original_parent, current.generation)
            self._report(
                "generation_committed",
                f"M{current.generation} committed checkpoint {current.ref.sha256}; "
                f"next Arena M{next_arena}",
                generation=current.generation,
                checkpoint_sha256=current.ref.sha256,
                next_arena_generation=next_arena,
            )

    def _ensure_runtime_layout(self) -> None:
        for name in ("runtime", "control", "logs", "metrics", "arena"):
            (self.lineage_root / name).mkdir(parents=True, exist_ok=True)

    def _persist_operator_metadata(self, parent: ResolvedCheckpointNode) -> None:
        manifest_path = self.lineage_root / "manifest.json"
        if not manifest_path.is_file():
            return
        manifest = _read_object(manifest_path, "lineage manifest")
        config = self.config.effective_config
        replay = config.replay
        self_play = config.self_play
        training = config.training
        operator_tunables = {
            "learning_rate": training.get("learning_rate"),
            "replay_generations": replay.get("generations", replay.get("window")),
            "replay_cap": replay.get("cap"),
            "self_play_mcts_simulations": self_play.get(
                "mcts_simulations", self_play.get("simulations")
            ),
            "games_per_generation": self_play.get(
                "games_per_iteration", self_play.get("games")
            ),
            "arena_every_generations": self.config.arena_cadence,
        }
        continuous = {
            "schema": CONTINUOUS_TRAINING_SCHEMA,
            "parent_checkpoint": parent.ref.to_dict(),
            "arena_cadence": self.config.arena_cadence,
            "arena_reference_gap": self.config.arena_reference_gap,
            "target_generation": self.config.target_generation,
        }
        if manifest.get("continuous_training") not in (None, continuous):
            previous = manifest.get("continuous_training")
            if isinstance(previous, Mapping):
                for key in ("parent_checkpoint", "arena_cadence", "arena_reference_gap"):
                    if previous.get(key) != continuous[key]:
                        raise RuntimeError(f"continuous lineage {key} changed during resume")
        manifest["operator_tunables"] = operator_tunables
        manifest["continuous_training"] = continuous
        _write_object(manifest_path, manifest)

    def _load_or_create_state(
        self, parent: ResolvedCheckpointNode
    ) -> tuple[dict[str, object], ResolvedCheckpointNode]:
        if self.state_path.is_file():
            state = _read_object(self.state_path, "continuous training state")
            if state.get("schema") != CONTINUOUS_TRAINING_SCHEMA:
                raise RuntimeError("unsupported continuous training state schema")
            self._validate_state(state, parent)
            if state.get("orchestrator_version") is None:
                state["orchestrator_version"] = ORCHESTRATOR_VERSION
                self._persist_state(state)
            raw_current = state.get("current_checkpoint")
            current = parent if not isinstance(raw_current, Mapping) else self.resolver.checkpoint(raw_current)
            if current.topology != self.config.topology or current.lineage_id not in {
                parent.lineage_id,
                self.config.lineage_id,
            }:
                raise RuntimeError("continuous state current checkpoint has an unexpected owner")
            return state, current

        state: dict[str, object] = {
            "schema": CONTINUOUS_TRAINING_SCHEMA,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "lineage_id": self.config.lineage_id,
            "topology": self.config.topology,
            "state": "RUNNING",
            "parent_checkpoint": parent.ref.to_dict(),
            "current_checkpoint": parent.ref.to_dict(),
            "last_committed_generation": parent.generation,
            "target_generation": self.config.target_generation,
            "requested_generations": self.config.generations,
            "effective_config_fingerprint": self.config.effective_config.fingerprint,
            "arena_cadence": self.config.arena_cadence,
            "arena_reference_gap": self.config.arena_reference_gap,
            "arena_config_fingerprint": self._arena_config_fingerprint(),
            "arena_generations": [],
            "arena_results": [],
            "soft_stop_requested": False,
            "created_at": _now(),
            "updated_at": _now(),
            "active_generation": None,
            "active_phase": None,
        }
        self._persist_state(state)
        return state, parent

    def _validate_state(self, state: Mapping[str, object], parent: ResolvedCheckpointNode) -> None:
        state_version = state.get("orchestrator_version")
        if state_version not in (None, ORCHESTRATOR_VERSION):
            raise RuntimeError(f"continuous state uses unsupported orchestrator version: {state_version!r}")
        expected = {
            "lineage_id": self.config.lineage_id,
            "topology": self.config.topology,
            "parent_checkpoint": parent.ref.to_dict(),
            "effective_config_fingerprint": self.config.effective_config.fingerprint,
            "arena_cadence": self.config.arena_cadence,
            "arena_reference_gap": self.config.arena_reference_gap,
            "arena_config_fingerprint": self._arena_config_fingerprint(),
            "target_generation": self.config.target_generation,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"continuous state {key} changed during resume")

    def _persist_state(self, state: dict[str, object]) -> None:
        state["updated_at"] = _now()
        _write_object(self.state_path, state)

    def _arena_config_fingerprint(self) -> str:
        return sha256_fingerprint(asdict(self.config.arena_config))

    def _arena_due(self, parent: ResolvedCheckpointNode, generation: int) -> bool:
        distance = int(generation) - parent.generation
        return distance > 0 and distance % self.config.arena_cadence == 0

    def _next_arena_generation(self, parent: ResolvedCheckpointNode, generation: int) -> int:
        distance = max(0, int(generation) - parent.generation)
        next_distance = ((distance // self.config.arena_cadence) + 1) * self.config.arena_cadence
        return parent.generation + next_distance

    @staticmethod
    def _validate_child(
        previous: ResolvedCheckpointNode,
        child: ResolvedCheckpointNode,
        config: ResolvedEffectiveConfig,
        generation: int,
    ) -> None:
        if not isinstance(child, ResolvedCheckpointNode):
            raise RuntimeError("train_one returned an unresolved checkpoint")
        if child.generation != generation:
            raise RuntimeError(
                f"train_one returned generation {child.generation}, expected {generation}"
            )
        if child.node.parent != previous.ref:
            raise RuntimeError("train_one returned a child with the wrong parent")
        if child.topology != "torus9" or child.lineage_id != config.artifact.lineage_id:
            raise RuntimeError("train_one returned a child owned by the wrong lineage")
        if child.effective_config.ref != config.ref:
            raise RuntimeError("train_one returned a child with the wrong effective config")

    def _arena_reference(
        self,
        parent: ResolvedCheckpointNode,
        current: ResolvedCheckpointNode,
    ) -> ResolvedCheckpointNode:
        gap = int(self.config.arena_reference_gap or 1)
        distance = current.generation - parent.generation
        if gap <= distance:
            return self.resolver.ancestor(current, gap)
        # An operator may choose a cadence shorter than the reference gap. The
        # first due Arena still compares against the declared run parent; the
        # existing Arena boundary remains responsible for its semantics.
        return parent

    def _run_arena(
        self,
        parent: ResolvedCheckpointNode,
        current: ResolvedCheckpointNode,
    ) -> ArenaRunResult:
        reference = self._arena_reference(parent, current)
        same_lineage = current.lineage_id == reference.lineage_id
        output_dir = None
        if same_lineage:
            output_dir = (
                self.lineage_root
                / "arena"
                / f"generation-{current.generation:04d}"
            )
        request = ArenaRunRequest(
            candidate=current,
            reference=reference,
            master_seed=int(self.config.arena_master_seed),
            startset=self.config.arena_startset,  # type: ignore[arg-type]
            config=self.config.arena_config,
            profile=self.config.arena_profile,
            workload=self.config.arena_workload,
            scientific_contract=self.config.arena_scientific_contract,
            execution_contract=self.config.arena_execution_contract,
            candidate_label=f"M{current.generation}",
            reference_label=f"M{reference.generation}",
            comparison=f"M{current.generation}-vs-M{reference.generation}",
            output_dir=output_dir,
        )
        try:
            result = self.arena_runner.run(request)
        except BaseException as exc:
            self._notify_operator(
                "CRITICAL",
                f"Arena execution failed for M{current.generation}: {exc.__class__.__name__}.",
                key_suffix=f"critical:arena:{current.generation}",
            )
            raise
        if not isinstance(result, ArenaRunResult):
            raise RuntimeError("ArenaRunnerV2 returned an invalid Arena result")
        if same_lineage:
            self._write_lineage_arena_projection(current, reference, result)
        self._report(
            "arena_completed",
            f"Arena M{current.generation} vs M{reference.generation}: "
            f"{result.validity} W/L/D={result.wld[0]}/{result.wld[1]}/{result.wld[2]}",
            generation=current.generation,
            reference_generation=reference.generation,
            validity=result.validity,
            wld=list(result.wld),
        )
        self._notify_operator(
            "ARENA_COMPLETED",
            f"Arena completed for M{current.generation} vs M{reference.generation}: "
            f"{result.validity} W/L/D={result.wld[0]}/{result.wld[1]}/{result.wld[2]}.",
            key_suffix=f"arena:{current.generation}:{result.evaluation_id}",
        )
        return result

    def _write_lineage_arena_projection(
        self,
        current: ResolvedCheckpointNode,
        reference: ResolvedCheckpointNode,
        result: ArenaRunResult,
    ) -> None:
        summary = dict(result.summary)
        wld = list(result.wld)
        payload: dict[str, object] = {
            "schema": f"{CONTINUOUS_TRAINING_SCHEMA}-arena-result-v1",
            "generation": current.generation,
            "reference_generation": reference.generation,
            "candidate_checkpoint": current.ref.to_dict(),
            "reference_checkpoint": reference.ref.to_dict(),
            "evaluation_id": result.evaluation_id,
            "evaluation_fingerprint": result.evaluation_fingerprint,
            "validity": result.validity,
            "W/L/D": wld,
            "summary": summary,
            "output_dir": str(result.output_dir),
        }
        metrics = summary.get("metrics")
        if isinstance(metrics, Mapping):
            payload["metrics"] = dict(metrics)
            payload["technical_games"] = int(metrics.get("technical_games", 0) or 0)
            payload["invalid_games"] = int(metrics.get("invalid_games", 0) or 0)
        else:
            payload["metrics"] = {
                "games": int(summary.get("games", self.config.arena_config.games)),
                "wins": wld[0],
                "losses": wld[1],
                "draws": wld[2],
            }
            payload["technical_games"] = int(summary.get("technical_games", 0) or 0)
            payload["invalid_games"] = int(summary.get("invalid_games", 0) or 0)
        projection = self.lineage_root / "arena" / f"generation-{current.generation:04d}" / "result.json"
        _write_object(projection, payload)

    @staticmethod
    def _arena_record(result: ArenaRunResult, generation: int) -> dict[str, object]:
        return {
            "generation": generation,
            "evaluation_id": result.evaluation_id,
            "evaluation_fingerprint": result.evaluation_fingerprint,
            "output_dir": str(result.output_dir),
            "identity": result.identity.to_dict(),
            "summary": dict(result.summary),
            "validity": result.validity,
        }

    def _record_arena(self, state: dict[str, object], result: ArenaRunResult, generation: int) -> None:
        raw_results = state.get("arena_results")
        records = list(raw_results) if isinstance(raw_results, list) else []
        records = [
            item
            for item in records
            if not isinstance(item, Mapping) or int(item.get("generation", -1)) != generation
        ]
        records.append(self._arena_record(result, generation))
        records.sort(key=lambda item: int(item.get("generation", -1)) if isinstance(item, Mapping) else -1)
        state["arena_results"] = records
        state["arena_generations"] = [
            int(item["generation"])
            for item in records
            if isinstance(item, Mapping) and item.get("generation") is not None
        ]
        state["last_arena_generation"] = generation
        self._persist_state(state)

    @staticmethod
    def _arena_recorded(state: Mapping[str, object], generation: int) -> bool:
        raw = state.get("arena_generations")
        return isinstance(raw, list) and generation in {int(value) for value in raw}

    @staticmethod
    def _stored_arenas(state: Mapping[str, object]) -> list[ArenaRunResult]:
        raw = state.get("arena_results")
        if not isinstance(raw, list):
            return []
        result: list[ArenaRunResult] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            try:
                identity = item["identity"]
                if not isinstance(identity, Mapping):
                    continue
                result.append(
                    ArenaRunResult(
                        evaluation_id=str(item["evaluation_id"]),
                        evaluation_fingerprint=str(item["evaluation_fingerprint"]),
                        output_dir=Path(str(item["output_dir"])),
                        identity=EvaluationIdentity.from_dict(identity),
                        summary=dict(item.get("summary", {})),
                        validity=str(item.get("validity", "INVALID")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def _result(
        self,
        state: Mapping[str, object],
        parent: ResolvedCheckpointNode,
        current: ResolvedCheckpointNode,
        committed: list[ResolvedCheckpointNode],
        arenas: list[ArenaRunResult],
    ) -> ContinuousTrainingResult:
        return ContinuousTrainingResult(
            state=str(state.get("state", "RUNNING")),
            original_parent=parent,
            final_checkpoint=current,
            committed_generations=tuple(committed),
            arenas=tuple(arenas),
            lineage_root=self.lineage_root,
            soft_stop_requested=bool(state.get("soft_stop_requested", False)),
        )

    def _report_start(
        self,
        parent: ResolvedCheckpointNode,
        config: ResolvedEffectiveConfig,
    ) -> None:
        effective = config.config
        replay = effective.replay
        self_play = effective.self_play
        training = effective.training
        message = (
            "Training started — GoCube AlphaZero; "
            f"parent={parent.topology}/{parent.lineage_id}/{parent.checkpoint_id}; "
            f"lineage={self.config.lineage_id}; "
            f"LR={training.get('learning_rate', '-')}; "
            f"replay={replay.get('generations', replay.get('window', '-'))} generations / "
            f"{replay.get('cap', '-')} positions; "
            f"self-play MCTS={self_play.get('mcts_simulations', self_play.get('simulations', '-'))} sims; "
            f"games/generation={self_play.get('games_per_iteration', self_play.get('games', '-'))}; "
            f"Arena cadence=every {self.config.arena_cadence} generations"
        )
        details = {
            "parent": parent.ref.to_dict(),
            "lineage_id": self.config.lineage_id,
            "learning_rate": training.get("learning_rate"),
            "replay_generations": replay.get("generations", replay.get("window")),
            "replay_cap": replay.get("cap"),
            "self_play_mcts_simulations": self_play.get(
                "mcts_simulations", self_play.get("simulations")
            ),
            "games_per_generation": self_play.get(
                "games_per_iteration", self_play.get("games")
            ),
            "arena_cadence": self.config.arena_cadence,
        }
        self._report("started", message, **details)
        self._notify_operator(
            "START",
            message,
            key_suffix=f"start:{self._launch_id}",
        )

    def _report(self, event: str, message: str, **details: object) -> None:
        self.logger.info(message)
        if self.reporter is None:
            return
        try:
            self.reporter(message, details)
        except TypeError:
            self.reporter(message)

    def _notify_operator(self, event: str, message: str, *, key_suffix: str) -> None:
        """Send one injected operator event without making Telegram a dependency.

        The production entrypoint owns construction of the notifier.  Keeping
        this boundary duck-typed also lets tests inject a recorder or a
        fail-open transport without importing or contacting Telegram.
        """
        if self._notifier is None:
            return
        try:
            self._notifier.send_now(
                f"continuous:{self.config.lineage_id}:{key_suffix}",
                f"{event} — {message}",
            )
        except BaseException:
            # Telegram is fail-open observability; it cannot affect training.
            self.logger.warning("operator notification failed", exc_info=True)


__all__ = [
    "CONTINUOUS_TRAINING_SCHEMA",
    "ContinuousTrainingConfig",
    "ContinuousTrainingResult",
    "ContinuousTrainingRunnerV2",
]
