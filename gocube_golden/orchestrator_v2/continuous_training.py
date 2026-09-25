"""Topology-neutral public continuous-training API for Orchestrator V2."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import logging

from ..artifact_graph import CheckpointRef, EffectiveConfig
from . import _continuous_training_core as _core
from .arena_runner import ArenaRunnerV2
from .artifact_resolver import ArtifactResolver, ResolvedCheckpointNode, ResolvedEffectiveConfig
from .contracts import StartsetRef
from .experiment_runner import LineageFactory, TrainOne
from .generation_runner import OutputLineage
from .operator_messages import format_action_started, format_training_started
from .production_generation import ProductionTrainOne
from .production_lineage import ProductionLineage
from .topology_binding import get_topology_binding
from tools.arena_engine import ArenaExecutionConfig, DEFAULT_MASTER_SEED

CONTINUOUS_TRAINING_SCHEMA = _core.CONTINUOUS_TRAINING_SCHEMA
CONCURRENCY_SWEEP_SCHEMA = _core.CONCURRENCY_SWEEP_SCHEMA
SelfPlayConcurrencyMode = _core.SelfPlayConcurrencyMode
SelfPlayConcurrencySweep = _core.SelfPlayConcurrencySweep
ContinuousTrainingResult = _core.ContinuousTrainingResult


def _is_real_telegram(notifier: object | None) -> bool:
    return notifier is not None and notifier.__class__.__module__ == "gocube_golden.telegram_notifier"


@dataclass(frozen=True)
class ContinuousTrainingConfig:
    parent_checkpoint: CheckpointRef | Mapping[str, object]
    lineage_id: str
    effective_config: EffectiveConfig | ResolvedEffectiveConfig | Mapping[str, object]
    generations: int | None
    arena_cadence: int
    arena_config: ArenaExecutionConfig | Mapping[str, object]
    arena_master_seed: int = DEFAULT_MASTER_SEED
    arena_startset: StartsetRef | Mapping[str, object] | None = None
    arena_profile: str | None = None
    arena_scientific_contract: Mapping[str, object] | None = None
    arena_execution_contract: Mapping[str, object] | None = None
    arena_workload: Mapping[str, object] = field(default_factory=dict)
    arena_reference_gap: int | None = None
    allow_code_rollover: bool = False
    self_play_concurrency_sweep: SelfPlayConcurrencySweep | Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        parent = self.parent_checkpoint if isinstance(self.parent_checkpoint, CheckpointRef) else CheckpointRef.from_dict(self.parent_checkpoint)
        config = _core._effective_config(self.effective_config)
        binding = get_topology_binding(config.topology)
        arena_config, inferred_gap = _core._arena_config(self.arena_config)
        gap = self.arena_reference_gap if self.arena_reference_gap is not None else inferred_gap
        if gap is None:
            raw_gap = config.arena.get("reference_gap")
            gap = raw_gap if raw_gap is not None else 1
        object.__setattr__(self, "parent_checkpoint", parent)
        object.__setattr__(self, "lineage_id", _core._component(self.lineage_id, "lineage_id"))
        object.__setattr__(self, "effective_config", config)
        object.__setattr__(self, "arena_config", arena_config)
        object.__setattr__(self, "self_play_concurrency_sweep", _core._concurrency_sweep(self.self_play_concurrency_sweep, config, parent.generation))
        if config.topology != parent.topology:
            raise ValueError("effective_config topology does not match parent checkpoint")
        if self.generations is not None and (type(self.generations) is not int or self.generations < 0):
            raise ValueError("generations must be a non-negative integer or None")
        _core._positive_int(self.arena_cadence, "arena_cadence")
        if type(gap) is not int or gap <= 0:
            raise ValueError("arena_reference_gap must be a positive integer")
        object.__setattr__(self, "arena_reference_gap", gap)
        if isinstance(self.arena_master_seed, bool):
            raise ValueError("arena_master_seed must be an integer")
        object.__setattr__(self, "arena_master_seed", int(self.arena_master_seed))
        profile = self.arena_profile
        if profile is None:
            profile = binding.default_arena_profile(config)
        else:
            profile = binding.validate_arena_profile(profile, config)
        object.__setattr__(self, "arena_profile", profile)
        if self.arena_scientific_contract is not None and not isinstance(self.arena_scientific_contract, Mapping):
            raise ValueError("arena_scientific_contract must be an object")
        if self.arena_execution_contract is not None and not isinstance(self.arena_execution_contract, Mapping):
            raise ValueError("arena_execution_contract must be an object")
        if not isinstance(self.arena_workload, Mapping):
            raise ValueError("arena_workload must be an object")
        if type(self.allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        startset = self.arena_startset
        if startset is None:
            startset = binding.arena_startset(master_seed=int(self.arena_master_seed), games=int(arena_config.games))
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


class ContinuousTrainingRunnerV2(_core.ContinuousTrainingRunnerV2):
    """One durable lifecycle for Torus9 and supported Cube topologies."""

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
        arena_profile: str | None = None,
        arena_scientific_contract: Mapping[str, object] | None = None,
        arena_execution_contract: Mapping[str, object] | None = None,
        arena_workload: Mapping[str, object] | None = None,
        arena_reference_gap: int | None = None,
        allow_code_rollover: bool | None = None,
        self_play_concurrency_sweep: SelfPlayConcurrencySweep | Mapping[str, object] | None = None,
        resolver: ArtifactResolver | None = None,
        arena_runner=None,
        lineage_factory: LineageFactory | None = None,
        train_one: TrainOne | None = None,
        reporter: Callable[..., None] | None = None,
        notifier: object | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if config is None:
            selected_parent = parent_checkpoint if parent_checkpoint is not None else parent
            if selected_parent is None or lineage_id is None or effective_config is None:
                raise TypeError("parent_checkpoint, lineage_id, effective_config, generations, arena_cadence, and arena_config are required")
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
            direct_override = (parent_checkpoint, parent, lineage_id, effective_config, arena_cadence, arena_config)
            if any(value is not None for value in direct_override):
                raise TypeError("pass either config or direct ContinuousTrainingRunnerV2 inputs, not both")
            if allow_code_rollover is not None:
                if type(allow_code_rollover) is not bool:
                    raise TypeError("allow_code_rollover must be a boolean")
                config = replace(config, allow_code_rollover=allow_code_rollover)
        selected_resolver = resolver or ArtifactResolver()
        selected_train_one = train_one or ProductionTrainOne(resolver=selected_resolver)
        selected_arena_runner = arena_runner or ArenaRunnerV2(notifier=notifier)
        if notifier is not None and hasattr(selected_arena_runner, "notifier"):
            setattr(selected_arena_runner, "notifier", notifier)
        super().__init__(
            config=config,  # type: ignore[arg-type]
            resolver=selected_resolver,
            arena_runner=selected_arena_runner,
            lineage_factory=lineage_factory or ProductionLineage(selected_resolver.runs_root),
            train_one=selected_train_one,
            reporter=reporter,
            notifier=notifier,
            logger=logger,
        )

    @staticmethod
    def _validate_child(previous: ResolvedCheckpointNode, child: ResolvedCheckpointNode, config: ResolvedEffectiveConfig, generation: int) -> None:
        if not isinstance(child, ResolvedCheckpointNode):
            raise RuntimeError("train_one returned an unresolved checkpoint")
        if child.generation != generation:
            raise RuntimeError(f"train_one returned generation {child.generation}, expected {generation}")
        if child.node.parent != previous.ref:
            raise RuntimeError("train_one returned a child with the wrong parent")
        if child.topology != config.topology or child.lineage_id != config.artifact.owner_lineage_id:
            raise RuntimeError("train_one returned a child owned by the wrong lineage")
        if child.effective_config.ref != config.ref:
            raise RuntimeError("train_one returned a child with the wrong effective config")

    def _call_train_one(
        self,
        *,
        parent: ResolvedCheckpointNode,
        config: ResolvedEffectiveConfig,
        output_lineage: OutputLineage,
        execution_mode: SelfPlayConcurrencyMode | None,
        acknowledge_stopped_execution: bool = False,
    ) -> ResolvedCheckpointNode:
        generation = parent.generation + 1
        if _is_real_telegram(self._notifier):
            self._notify_operator(
                "GENERATION_STARTED",
                format_action_started(
                    "GENERATION STARTED",
                    topology=self.config.topology,
                    lineage=self.config.lineage_id,
                    generation=f"M{generation}",
                    execution_mode=None if execution_mode is None else execution_mode.label,
                ),
                key_suffix=f"generation-start:{generation}",
            )
        return super()._call_train_one(
            parent=parent,
            config=config,
            output_lineage=output_lineage,
            execution_mode=execution_mode,
            acknowledge_stopped_execution=acknowledge_stopped_execution,
        )

    def _report_start(self, parent: ResolvedCheckpointNode, config: ResolvedEffectiveConfig) -> None:
        effective = config.config
        compatibility = effective.compatibility
        network = compatibility.get("network") or compatibility.get("architecture") or compatibility.get("architecture_id")
        message = format_training_started(
            topology=self.config.topology,
            lineage_id=self.config.lineage_id,
            parent_label=parent.checkpoint_id,
            network=network,
            effective_config=effective,
            arena_cadence=self.config.arena_cadence,
            arena_config=self.config.arena_config,
        )
        details = {
            "topology": self.config.topology,
            "network": network,
            "parent": parent.ref.to_dict(),
            "lineage_id": self.config.lineage_id,
            "resolved_effective_config": effective.to_dict(),
            "arena_cadence": self.config.arena_cadence,
        }
        self._report("started", message, **details)
        self._notify_operator("START", message, key_suffix=f"start:{self._launch_id}")


__all__ = [
    "ArenaRunnerV2", "CONCURRENCY_SWEEP_SCHEMA", "CONTINUOUS_TRAINING_SCHEMA",
    "ContinuousTrainingConfig", "ContinuousTrainingResult", "ContinuousTrainingRunnerV2",
    "ProductionLineage", "SelfPlayConcurrencyMode", "SelfPlayConcurrencySweep",
]
