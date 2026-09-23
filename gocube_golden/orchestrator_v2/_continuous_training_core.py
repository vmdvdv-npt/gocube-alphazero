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
import math
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


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


CONCURRENCY_SWEEP_SCHEMA = "gocube-orchestrator-v2-self-play-concurrency-sweep-v1"


@dataclass(frozen=True)
class SelfPlayConcurrencyMode:
    """One execution-only self-play concurrency setting."""

    label: str
    active_games_per_worker: int
    total_active_contexts: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _component(self.label, "concurrency mode label"))
        _positive_int(self.active_games_per_worker, "active_games_per_worker")
        _positive_int(self.total_active_contexts, "total_active_contexts")

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "active_games_per_worker": self.active_games_per_worker,
            "total_active_contexts": self.total_active_contexts,
        }


@dataclass(frozen=True)
class SelfPlayConcurrencySweep:
    """Durable operator schedule for an execution-only concurrency sweep."""

    start_after_generation: int
    workers: int
    baseline: SelfPlayConcurrencyMode
    modes: tuple[SelfPlayConcurrencyMode, ...]
    continue_after_sweep: bool = True

    def __post_init__(self) -> None:
        _nonnegative_int(self.start_after_generation, "sweep start_after_generation")
        _positive_int(self.workers, "sweep workers")
        if not self.modes:
            raise ValueError("self-play concurrency sweep requires at least one test mode")
        labels = {self.baseline.label}
        for mode in self.modes:
            if mode.label in labels:
                raise ValueError(f"duplicate self-play concurrency mode: {mode.label}")
            labels.add(mode.label)
        if type(self.continue_after_sweep) is not bool:
            raise ValueError("sweep continue_after_sweep must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CONCURRENCY_SWEEP_SCHEMA,
            "start_after_generation": self.start_after_generation,
            "workers": self.workers,
            "baseline": self.baseline.to_dict(),
            "modes": [mode.to_dict() for mode in self.modes],
            "continue_after_sweep": self.continue_after_sweep,
        }

    def mode_for_index(self, index: int) -> SelfPlayConcurrencyMode:
        return self.modes[index]


def _concurrency_mode(value: object, label: str) -> SelfPlayConcurrencyMode:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    allowed = {"label", "active_games_per_worker", "total_active_contexts"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {', '.join(sorted(unknown))}")
    missing = sorted(allowed - set(value))
    if missing:
        raise ValueError(f"{label} is missing explicit fields: {', '.join(missing)}")
    return SelfPlayConcurrencyMode(
        label=str(value["label"]),
        active_games_per_worker=value["active_games_per_worker"],  # type: ignore[arg-type]
        total_active_contexts=value["total_active_contexts"],  # type: ignore[arg-type]
    )


def _concurrency_sweep(
    value: SelfPlayConcurrencySweep | Mapping[str, object] | None,
    config: EffectiveConfig,
    parent_generation: int,
) -> SelfPlayConcurrencySweep | None:
    if value is None:
        return None
    if isinstance(value, SelfPlayConcurrencySweep):
        sweep = value
    else:
        if not isinstance(value, Mapping):
            raise ValueError("self_play_concurrency_sweep must be an object")
        allowed = {
            "schema",
            "start_after_generation",
            "workers",
            "baseline",
            "modes",
            "continue_after_sweep",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "self_play_concurrency_sweep contains unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        if value.get("schema") != CONCURRENCY_SWEEP_SCHEMA:
            raise ValueError("self_play_concurrency_sweep has an unsupported schema")
        raw_modes = value.get("modes")
        if not isinstance(raw_modes, list):
            raise ValueError("self_play_concurrency_sweep.modes must be a list")
        sweep = SelfPlayConcurrencySweep(
            start_after_generation=_nonnegative_int(
                value.get("start_after_generation"),
                "sweep start_after_generation",
            ),
            workers=_positive_int(value.get("workers"), "sweep workers"),
            baseline=_concurrency_mode(value.get("baseline"), "sweep baseline"),
            modes=tuple(
                _concurrency_mode(item, f"sweep modes[{index}]")
                for index, item in enumerate(raw_modes)
            ),
            continue_after_sweep=value.get("continue_after_sweep", True),  # type: ignore[arg-type]
        )

    if sweep.start_after_generation < parent_generation:
        raise ValueError("self-play concurrency sweep cannot start before the lineage parent")
    execution = config.execution
    configured_workers = _positive_int(execution.get("workers"), "effective execution workers")
    if sweep.workers != configured_workers:
        raise ValueError(
            "self-play concurrency sweep may not change worker count: "
            f"configured={configured_workers}, sweep={sweep.workers}"
        )
    configured_contexts = int(
        execution.get("active_contexts", execution.get("total_active_contexts"))
    )
    configured_active = int(
        execution.get(
            "active_games_per_worker",
            max(1, (configured_contexts + configured_workers - 1) // configured_workers),
        )
    )
    if (
        sweep.baseline.active_games_per_worker != configured_active
        or sweep.baseline.total_active_contexts != configured_contexts
    ):
        raise ValueError(
            "self-play concurrency sweep baseline must match the effective production execution"
        )
    for mode in (sweep.baseline, *sweep.modes):
        if mode.total_active_contexts > sweep.workers * mode.active_games_per_worker:
            raise ValueError(
                f"self-play concurrency mode {mode.label} exceeds worker lane capacity"
            )
    return sweep


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
    self_play_concurrency_sweep: SelfPlayConcurrencySweep | Mapping[str, object] | None = None

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
        object.__setattr__(
            self,
            "self_play_concurrency_sweep",
            _concurrency_sweep(self.self_play_concurrency_sweep, config, parent.generation),
        )
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
        self_play_concurrency_sweep: SelfPlayConcurrencySweep | Mapping[str, object] | None = None,
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
                self_play_concurrency_sweep=self_play_concurrency_sweep,
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

        self._initialize_concurrency_sweep(state, current)

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
            execution_mode = self._next_sweep_mode(state, next_generation)
            state.update(
                {
                    "state": "RUNNING",
                    "active_generation": next_generation,
                    "active_phase": "generation",
                    "soft_stop_requested": False,
                    "active_execution_mode": (
                        execution_mode.to_dict() if execution_mode is not None else None
                    ),
                }
            )
            self._persist_state(state)
            previous = current
            try:
                current = self._call_train_one(
                    parent=current,
                    config=resolved_config,
                    output_lineage=output_lineage,
                    execution_mode=execution_mode,
                    acknowledge_stopped_execution=self._is_baseline_recovery(
                        state,
                        generation=next_generation,
                        execution_mode=execution_mode,
                    ),
                )
            except BaseException as exc:
                if self._handle_sweep_failure(
                    state,
                    generation=next_generation,
                    execution_mode=execution_mode,
                    error=exc,
                ):
                    state.update(
                        {
                            "active_generation": None,
                            "active_phase": None,
                            "active_execution_mode": None,
                        }
                    )
                    self._persist_state(state)
                    continue
                self._notify_operator(
                    "CRITICAL",
                    f"Generation M{next_generation} failed: {exc.__class__.__name__}.",
                    key_suffix=f"critical:generation:{next_generation}",
                )
                raise
            self._validate_child(previous, current, resolved_config, next_generation)
            committed.append(current)
            self._record_sweep_observation(state, current, execution_mode)
            state.update(
                {
                    "current_checkpoint": current.ref.to_dict(),
                    "last_committed_generation": current.generation,
                    "active_generation": None,
                    "active_phase": None,
                    "last_execution_mode": (
                        execution_mode.to_dict() if execution_mode is not None else None
                    ),
                    "active_execution_mode": None,
                }
            )
            self._persist_state(state)
            self._write_sweep_report(state)
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
        if self.config.self_play_concurrency_sweep is not None:
            continuous["self_play_concurrency_sweep"] = (
                self.config.self_play_concurrency_sweep.to_dict()
            )
        if manifest.get("continuous_training") not in (None, continuous):
            previous = manifest.get("continuous_training")
            if isinstance(previous, Mapping):
                for key in ("parent_checkpoint", "arena_cadence", "arena_reference_gap"):
                    if previous.get(key) != continuous[key]:
                        raise RuntimeError(f"continuous lineage {key} changed during resume")
                previous_sweep = previous.get("self_play_concurrency_sweep")
                current_sweep = continuous.get("self_play_concurrency_sweep")
                if previous_sweep is not None and previous_sweep != current_sweep:
                    raise RuntimeError("continuous lineage self-play concurrency sweep changed during resume")
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
            if self.config.self_play_concurrency_sweep is not None and "performance_sweep" not in state:
                state["performance_sweep"] = self._new_sweep_state()
                self._persist_state(state)
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
            "active_execution_mode": None,
            "last_execution_mode": None,
        }
        if self.config.self_play_concurrency_sweep is not None:
            state["performance_sweep"] = self._new_sweep_state()
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
        configured_sweep = self.config.self_play_concurrency_sweep
        persisted_sweep = state.get("performance_sweep")
        if configured_sweep is None:
            if persisted_sweep is not None:
                raise RuntimeError("continuous state contains an unexpected self-play sweep")
        elif isinstance(persisted_sweep, Mapping):
            if persisted_sweep.get("config") != configured_sweep.to_dict():
                raise RuntimeError("continuous state self-play concurrency sweep changed during resume")
        elif persisted_sweep is not None:
            raise RuntimeError("continuous state self-play sweep is malformed")

    def _persist_state(self, state: dict[str, object]) -> None:
        state["updated_at"] = _now()
        _write_object(self.state_path, state)

    def _new_sweep_state(self) -> dict[str, object]:
        sweep = self.config.self_play_concurrency_sweep
        if sweep is None:
            raise RuntimeError("cannot create a concurrency sweep state without a sweep")
        return {
            "schema": f"{CONCURRENCY_SWEEP_SCHEMA}-state-v1",
            "config": sweep.to_dict(),
            "status": "RUNNING",
            "observations": [],
            "failed_modes": [],
            "next_mode_index": 0,
            "retry_baseline_generation": None,
            "selected_mode": None,
            "selection_generation": None,
        }

    def _sweep_state(self, state: Mapping[str, object]) -> dict[str, object] | None:
        if self.config.self_play_concurrency_sweep is None:
            return None
        raw = state.get("performance_sweep")
        if not isinstance(raw, dict):
            raise RuntimeError("continuous state is missing its self-play sweep state")
        if raw.get("config") != self.config.self_play_concurrency_sweep.to_dict():
            raise RuntimeError("continuous state self-play sweep config drifted")
        return raw

    def _initialize_concurrency_sweep(
        self,
        state: dict[str, object],
        current: ResolvedCheckpointNode,
    ) -> None:
        sweep = self.config.self_play_concurrency_sweep
        if sweep is None:
            return
        raw = self._sweep_state(state)
        if raw is None:
            return
        observations = raw.get("observations")
        if not isinstance(observations, list):
            raise RuntimeError("continuous state self-play sweep observations are malformed")
        # Baseline evidence is deliberately bounded to the three most recent
        # committed generations at the sweep boundary; it never scans replay.
        anchor = min(current.generation, sweep.start_after_generation)
        existing = {
            int(item.get("generation"))
            for item in observations
            if isinstance(item, Mapping) and item.get("generation") is not None
        }
        for generation in range(anchor, max(-1, anchor - 3), -1):
            if generation in existing:
                continue
            observation = self._load_sweep_observation(
                generation,
                sweep.baseline,
                role="baseline",
            )
            if observation is not None:
                observations.append(observation)
                existing.add(generation)
        observations.sort(key=lambda item: int(item.get("generation", -1)))
        self._refresh_sweep_cycles(observations)
        self._maybe_select_sweep_mode(raw, current.generation)
        self._persist_state(state)
        self._write_sweep_report(state)

    def _next_sweep_mode(
        self,
        state: Mapping[str, object],
        generation: int,
    ) -> SelfPlayConcurrencyMode | None:
        sweep = self.config.self_play_concurrency_sweep
        if sweep is None:
            return None
        raw = self._sweep_state(state)
        if raw is None:
            return None
        if raw.get("retry_baseline_generation") == generation:
            return sweep.baseline
        if generation <= sweep.start_after_generation:
            return sweep.baseline
        raw_selected = raw.get("selected_mode")
        next_index = int(raw.get("next_mode_index", 0))
        if next_index < len(sweep.modes):
            return sweep.mode_for_index(next_index)
        if isinstance(raw_selected, Mapping):
            return _concurrency_mode(raw_selected, "persisted selected mode")
        # This is only a short-lived fallback while the report is being
        # reconstructed after an interrupted sweep.
        return sweep.baseline

    def _call_train_one(
        self,
        *,
        parent: ResolvedCheckpointNode,
        config: ResolvedEffectiveConfig,
        output_lineage: OutputLineage,
        execution_mode: SelfPlayConcurrencyMode | None,
        acknowledge_stopped_execution: bool = False,
    ) -> ResolvedCheckpointNode:
        if execution_mode is not None and isinstance(self.train_one, ProductionTrainOne):
            return self.train_one(
                parent=parent,
                config=config,
                output_lineage=output_lineage,
                execution_overrides={
                    "active_games_per_worker": execution_mode.active_games_per_worker,
                    "total_active_contexts": execution_mode.total_active_contexts,
                },
                acknowledge_stopped_execution=acknowledge_stopped_execution,
            )
        return self.train_one(
            parent=parent,
            config=config,
            output_lineage=output_lineage,
        )

    def _is_baseline_recovery(
        self,
        state: Mapping[str, object],
        *,
        generation: int,
        execution_mode: SelfPlayConcurrencyMode | None,
    ) -> bool:
        """Return true only for the durable same-generation sweep fallback."""
        sweep = self.config.self_play_concurrency_sweep
        if sweep is None or execution_mode is None:
            return False
        if execution_mode.label != sweep.baseline.label:
            return False
        raw = self._sweep_state(state)
        return raw is not None and raw.get("retry_baseline_generation") == generation

    def _handle_sweep_failure(
        self,
        state: dict[str, object],
        *,
        generation: int,
        execution_mode: SelfPlayConcurrencyMode | None,
        error: BaseException,
    ) -> bool:
        sweep = self.config.self_play_concurrency_sweep
        if sweep is None or execution_mode is None:
            return False
        if execution_mode.label == sweep.baseline.label or not self._recoverable_sweep_failure(error):
            return False
        raw = self._sweep_state(state)
        if raw is None:
            return False
        failed_modes = raw.setdefault("failed_modes", [])
        if not isinstance(failed_modes, list):
            raise RuntimeError("continuous state self-play sweep failures are malformed")
        failed_modes[:] = [
            item
            for item in failed_modes
            if not isinstance(item, Mapping) or item.get("label") != execution_mode.label
        ]
        failed_modes.append(
            {
                "label": execution_mode.label,
                "generation": generation,
                "exception": error.__class__.__name__,
                "message": str(error)[:1000],
                "recorded_at": _now(),
            }
        )
        labels = [mode.label for mode in sweep.modes]
        if execution_mode.label in labels:
            raw["next_mode_index"] = max(
                int(raw.get("next_mode_index", 0)), labels.index(execution_mode.label) + 1
            )
        raw["retry_baseline_generation"] = generation
        self._maybe_select_sweep_mode(raw, generation)
        state["performance_sweep"] = raw
        self._write_sweep_report(state)
        self._notify_operator(
            "WARNING",
            f"Self-play concurrency mode {execution_mode.label} failed at M{generation}; "
            "retrying the same generation at the stable baseline and continuing the sweep.",
            key_suffix=f"sweep-failure:{generation}:{execution_mode.label}",
        )
        return True

    @staticmethod
    def _recoverable_sweep_failure(error: BaseException) -> bool:
        if isinstance(error, (MemoryError, OSError, TimeoutError, ChildProcessError)):
            return True
        if not isinstance(error, RuntimeError):
            return False
        text = str(error).lower()
        return any(
            token in text
            for token in (
                "production generation",
                "supervisor",
                "out of memory",
                "oom",
                "cuda",
                "technical",
                "timeout",
                "timed out",
                "stopped",
            )
        )

    def _summary_path(self, generation: int) -> Path:
        return self.lineage_root / f"iter-{generation:02d}-summary.json"

    def _request_path(self, generation: int) -> Path:
        return self.lineage_root / "runtime" / "requests" / f"train-one-{generation:04d}.json"

    @staticmethod
    def _numeric(value: object, default: float | None = None) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    def _load_sweep_observation(
        self,
        generation: int,
        mode: SelfPlayConcurrencyMode,
        *,
        role: str,
    ) -> dict[str, object] | None:
        summary_path = self._summary_path(generation)
        if not summary_path.is_file():
            return None
        try:
            summary = _read_object(summary_path, f"generation M{generation} summary")
        except RuntimeError as exc:
            self.logger.warning("Cannot read sweep summary for M%s: %s", generation, exc)
            return None
        metrics = summary.get("orchestrator_selfplay")
        if not isinstance(metrics, Mapping):
            return None
        inference = metrics.get("inference")
        inference = inference if isinstance(inference, Mapping) else {}
        timing = metrics.get("timing")
        timing = timing if isinstance(timing, Mapping) else {}
        games = int(metrics.get("games", 0) or 0)
        moves = int(metrics.get("moves", 0) or 0)
        wall = self._numeric(metrics.get("selfplay_time_sec"), 0.0) or 0.0
        games_per_hour = self._numeric(metrics.get("games_per_hour"), 0.0) or 0.0
        mean_batch = self._numeric(inference.get("mean_batch_rows"), 0.0) or 0.0
        mcts = int(
            self.config.effective_config.self_play.get(
                "mcts_simulations",
                self.config.effective_config.self_play.get("simulations", 0),
            )
            or 0
        )
        expected_games = int(
            self.config.effective_config.self_play.get(
                "games_per_iteration",
                self.config.effective_config.self_play.get("games", 0),
            )
            or 0
        )
        technical = int(metrics.get("technical_games", 0) or 0)
        invalid = int(metrics.get("invalid_games", 0) or 0)
        reasons: list[str] = []
        actual_execution: dict[str, object] | None = None
        sweep = self.config.self_play_concurrency_sweep
        if sweep is not None:
            expected_execution = {
                "workers": sweep.workers,
                "active_games_per_worker": mode.active_games_per_worker,
                "total_active_contexts": mode.total_active_contexts,
            }
            raw_execution = metrics.get("execution")
            if isinstance(raw_execution, Mapping):
                actual_execution = {
                    key: raw_execution.get(key)
                    for key in expected_execution
                }
                if any(
                    type(actual_execution[key]) is not int or actual_execution[key] <= 0
                    for key in expected_execution
                ):
                    reasons.append("self-play execution values are malformed")
                elif actual_execution != expected_execution:
                    reasons.append(
                        "self-play execution values do not match the selected mode: "
                        f"actual={actual_execution!r}, expected={expected_execution!r}"
                    )
            else:
                reasons.append("self-play execution values are missing")
        stall_value = None
        for source in (metrics, timing, inference):
            for key in ("stall_count", "stalls", "execution_stalls"):
                if key in source:
                    stall_value = int(source[key] or 0)
                    break
            if stall_value is not None:
                break
        if games != expected_games:
            reasons.append(f"games={games}, expected={expected_games}")
        if wall <= 0.0:
            reasons.append("missing self-play wall time")
        if technical:
            reasons.append(f"technical_games={technical}")
        if invalid:
            reasons.append(f"invalid_games={invalid}")
        if stall_value not in (None, 0):
            reasons.append(f"stalls={stall_value}")
        request_path = self._request_path(generation)
        start_epoch = request_path.stat().st_mtime if request_path.is_file() else None
        observation: dict[str, object] = {
            "generation": generation,
            "role": role,
            "mode": mode.to_dict(),
            "actual_execution": actual_execution,
            "workers": self.config.self_play_concurrency_sweep.workers  # type: ignore[union-attr]
            if self.config.self_play_concurrency_sweep is not None
            else None,
            "selfplay_wall_time_sec": wall,
            "cycle_wall_time_sec": None,
            "games_per_hour": games_per_hour,
            "mean_game_length": (moves / games if games else 0.0),
            "mcts_simulations": mcts,
            "mcts_simulations_per_sec_equivalent": (
                moves * mcts / wall if wall > 0.0 else 0.0
            ),
            "moves_per_sec": self._numeric(metrics.get("moves_per_sec"), 0.0) or 0.0,
            "mean_inference_batch": mean_batch,
            "gpu_utilization_percent": self._first_numeric(
                (metrics, inference, timing),
                ("gpu_utilization_percent", "gpu_utilization"),
            ),
            "gpu_power_w": self._first_numeric(
                (metrics, inference, timing),
                ("gpu_power_w", "gpu_power", "power_w"),
            ),
            "cpu_utilization_percent": self._first_numeric(
                (metrics, inference, timing),
                ("cpu_utilization_percent", "cpu_utilization"),
            ),
            "technical_games": technical,
            "invalid_games": invalid,
            "stalls": stall_value,
            "stable": not reasons,
            "stability_reasons": reasons,
            "request_path": str(request_path.relative_to(self.lineage_root))
            if request_path.is_relative_to(self.lineage_root)
            else str(request_path),
            "request_start_at": (
                datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
                if start_epoch is not None
                else None
            ),
            "timing": {
                key: timing[key]
                for key in (
                    "restore_previous_state_wall_time_sec",
                    "self_play_wall_time_sec",
                    "replay_file_load_wall_time_sec",
                    "optimizer_wall_time_sec",
                )
                if key in timing
            },
        }
        return observation

    @staticmethod
    def _first_numeric(
        sources: tuple[Mapping[str, object], ...], keys: tuple[str, ...]
    ) -> float | None:
        for source in sources:
            for key in keys:
                if key in source:
                    try:
                        value = float(source[key])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        return value
        return None

    def _refresh_sweep_cycles(self, observations: list[object]) -> None:
        for item in observations:
            if not isinstance(item, dict):
                continue
            try:
                generation = int(item["generation"])
            except (KeyError, TypeError, ValueError):
                continue
            start = self._request_path(generation)
            following = self._request_path(generation + 1)
            if not start.is_file() or not following.is_file():
                continue
            elapsed = following.stat().st_mtime - start.stat().st_mtime
            if elapsed >= 0.0:
                item["cycle_wall_time_sec"] = elapsed
                item["cycle_end_at"] = datetime.fromtimestamp(
                    following.stat().st_mtime, timezone.utc
                ).isoformat()

    def _record_sweep_observation(
        self,
        state: dict[str, object],
        current: ResolvedCheckpointNode,
        execution_mode: SelfPlayConcurrencyMode | None,
    ) -> None:
        sweep = self.config.self_play_concurrency_sweep
        if sweep is None or execution_mode is None:
            return
        raw = self._sweep_state(state)
        if raw is None:
            return
        role = "baseline" if execution_mode.label == sweep.baseline.label else "sweep"
        if current.generation > sweep.start_after_generation and role == "baseline":
            role = "fallback"
        observation = self._load_sweep_observation(current.generation, execution_mode, role=role)
        observations = raw.setdefault("observations", [])
        if not isinstance(observations, list):
            raise RuntimeError("continuous state self-play sweep observations are malformed")
        observations[:] = [
            item
            for item in observations
            if not isinstance(item, Mapping) or int(item.get("generation", -1)) != current.generation
        ]
        if observation is not None:
            observations.append(observation)
        observations.sort(key=lambda item: int(item.get("generation", -1)))
        if (
            current.generation > sweep.start_after_generation
            and execution_mode.label in {mode.label for mode in sweep.modes}
        ):
            labels = [mode.label for mode in sweep.modes]
            index = labels.index(execution_mode.label)
            raw["next_mode_index"] = max(int(raw.get("next_mode_index", 0)), index + 1)
        if raw.get("retry_baseline_generation") == current.generation:
            raw["retry_baseline_generation"] = None
        self._refresh_sweep_cycles(observations)
        self._maybe_select_sweep_mode(raw, current.generation)
        state["performance_sweep"] = raw

    def _maybe_select_sweep_mode(self, raw: dict[str, object], generation: int) -> None:
        sweep = self.config.self_play_concurrency_sweep
        if sweep is None:
            return
        observations = raw.get("observations")
        failures = raw.get("failed_modes")
        if not isinstance(observations, list) or not isinstance(failures, list):
            raise RuntimeError("continuous state self-play sweep records are malformed")
        candidate_labels = {mode.label for mode in sweep.modes}
        observed_labels = {
            item.get("mode", {}).get("label")
            for item in observations
            if isinstance(item, Mapping)
            and isinstance(item.get("mode"), Mapping)
            and item.get("mode", {}).get("label") in candidate_labels
        }
        failed_labels = {
            item.get("label")
            for item in failures
            if isinstance(item, Mapping) and item.get("label") in candidate_labels
        }
        if not candidate_labels.issubset(observed_labels | failed_labels):
            return
        modes = {sweep.baseline.label: sweep.baseline, **{mode.label: mode for mode in sweep.modes}}
        stable: list[tuple[float, float, SelfPlayConcurrencyMode]] = []
        for label, mode in modes.items():
            if label in failed_labels:
                continue
            values = [
                item
                for item in observations
                if isinstance(item, Mapping)
                and isinstance(item.get("mode"), Mapping)
                and item["mode"].get("label") == label
                and item.get("stable") is True
            ]
            if not values:
                continue
            hours = [float(item.get("games_per_hour", 0.0) or 0.0) for item in values]
            walls = [float(item.get("selfplay_wall_time_sec", 0.0) or 0.0) for item in values]
            stable.append((sum(hours) / len(hours), sum(walls) / len(walls), mode))
        if not stable:
            selected = sweep.baseline
        else:
            selected = max(stable, key=lambda item: (item[0], -item[1]))[2]
        previous = raw.get("selected_mode")
        raw["selected_mode"] = selected.to_dict()
        raw["status"] = "SELECTED"
        raw["selection_generation"] = generation
        if previous != raw["selected_mode"]:
            self._report(
                "concurrency_sweep_selected",
                f"Self-play concurrency sweep selected {selected.label} after M{generation}",
                generation=generation,
                contexts=selected.total_active_contexts,
                active_games_per_worker=selected.active_games_per_worker,
            )

    def _write_sweep_report(self, state: Mapping[str, object]) -> None:
        sweep = self.config.self_play_concurrency_sweep
        if sweep is None:
            return
        raw = state.get("performance_sweep")
        if not isinstance(raw, Mapping):
            return
        observations = raw.get("observations")
        observations = observations if isinstance(observations, list) else []
        baseline_values = [
            item
            for item in observations
            if isinstance(item, Mapping)
            and isinstance(item.get("mode"), Mapping)
            and item["mode"].get("label") == sweep.baseline.label
            and item.get("stable") is True
        ]
        baseline_wall = (
            sum(float(item.get("selfplay_wall_time_sec", 0.0) or 0.0) for item in baseline_values)
            / len(baseline_values)
            if baseline_values
            else None
        )
        baseline_hours = (
            sum(float(item.get("games_per_hour", 0.0) or 0.0) for item in baseline_values)
            / len(baseline_values)
            if baseline_values
            else None
        )
        report_observations: list[dict[str, object]] = []
        for item in observations:
            if not isinstance(item, Mapping):
                continue
            copied = dict(item)
            wall = self._numeric(copied.get("selfplay_wall_time_sec"))
            hours = self._numeric(copied.get("games_per_hour"))
            copied["change_to_baseline_percent"] = (
                (baseline_wall - wall) / baseline_wall * 100.0
                if baseline_wall and wall is not None
                else None
            )
            copied["games_per_hour_change_to_baseline_percent"] = (
                (hours - baseline_hours) / baseline_hours * 100.0
                if baseline_hours and hours is not None
                else None
            )
            report_observations.append(copied)
        effective = self.config.effective_config
        selfplay = effective.self_play
        training = effective.training
        execution = effective.execution
        report = {
            "schema": f"{CONCURRENCY_SWEEP_SCHEMA}-report-v1",
            "lineage_id": self.config.lineage_id,
            "sweep": sweep.to_dict(),
            "contract": {
                "games_per_generation": selfplay.get(
                    "games_per_iteration", selfplay.get("games")
                ),
                "mcts_simulations": selfplay.get(
                    "mcts_simulations", selfplay.get("simulations")
                ),
                "inference_batch_cap": execution.get("inference_batch_cap"),
                "inference_batch_wait_ms": execution.get("inference_batch_wait_ms"),
                "optimizer": training.get("optimizer"),
                "learning_rate": training.get("learning_rate"),
                "optimizer_steps": training.get(
                    "optimizer_steps", training.get("optimizer_steps_per_iteration")
                ),
                "replay_generations": effective.replay.get(
                    "generations", effective.replay.get("window")
                ),
                "replay_cap": effective.replay.get("cap"),
                "arena_cadence": self.config.arena_cadence,
                "arena_games": self.config.arena_config.games,
                "arena_diagnostic_only": True,
                "arena_gating": False,
            },
            "status": raw.get("status", "RUNNING"),
            "selected_mode": raw.get("selected_mode"),
            "selection_generation": raw.get("selection_generation"),
            "baseline_reference": {
                "observations": len(baseline_values),
                "mean_selfplay_wall_time_sec": baseline_wall,
                "mean_games_per_hour": baseline_hours,
            },
            "observations": report_observations,
            "failed_modes": [dict(item) for item in raw.get("failed_modes", []) if isinstance(item, Mapping)],
        }
        try:
            _write_object(self.lineage_root / "metrics" / "self-play-concurrency-sweep-v1.json", report)
        except (OSError, TypeError, ValueError):
            self.logger.warning("Could not persist self-play concurrency sweep report", exc_info=True)

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
    "CONCURRENCY_SWEEP_SCHEMA",
    "CONTINUOUS_TRAINING_SCHEMA",
    "ContinuousTrainingConfig",
    "ContinuousTrainingResult",
    "ContinuousTrainingRunnerV2",
    "SelfPlayConcurrencyMode",
    "SelfPlayConcurrencySweep",
]
