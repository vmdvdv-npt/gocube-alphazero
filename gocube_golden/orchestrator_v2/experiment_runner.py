"""Coordinator-only one- or two-stage orchestration for Orchestrator V2.

The runner owns sequencing and durable decisions only:

    common parent -> A/B -> Arena -> optional winner-rooted control/C -> Arena -> STOP

The runner owns the arm generation loop.  One-generation execution, process
supervision, commit/recovery, and graph resolution live below this boundary.
This module does not train, build replay, publish checkpoints, create commit
markers, or run Arena games.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Protocol
import uuid

from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from .arena_runner import ArenaRunRequest, ArenaRunResult, ArenaRunnerV2, torus9_startset_ref
from .artifact_resolver import ArtifactResolver, ResolvedCheckpointNode, ResolvedEffectiveConfig
from .contracts import CheckpointRef, EvaluationIdentity, StartsetRef
from .experiment_plan import (
    EXPERIMENT_WINNER_RULE,
    ExperimentArmConfig,
    ExperimentConfig,
    ExperimentStage2Config,
    Stage2Config,
    WinnerDecision,
    WinnerRule,
    WinnerRuleName,
)
from .generation_runner import OutputLineage
from .production_generation import ProductionTrainOne
from .torus9_production import Torus9ProductionLineage

from tools.arena_engine import ArenaExecutionConfig


EXPERIMENT_RUNNER_SCHEMA = "gocube-experiment-runner-v2"
EXPERIMENT_STATE_SCHEMA = "gocube-experiment-runner-state-v2"


class ExperimentRunnerError(RuntimeError):
    """A persisted experiment cannot be resumed safely."""


class LineageFactory(Protocol):
    """Prepare or resume one independent output lineage."""

    def prepare(
        self,
        *,
        topology: str,
        lineage_id: str,
        parent: ResolvedCheckpointNode,
        effective_config: object,
        experiment_id: str,
        arm_id: str,
    ) -> tuple[Path, ResolvedEffectiveConfig]: ...


class TrainOne(Protocol):
    """Strictly one-generation boundary used by the coordinator loop."""

    def __call__(
        self,
        *,
        parent: ResolvedCheckpointNode,
        config: ResolvedEffectiveConfig,
        output_lineage: OutputLineage,
    ) -> ResolvedCheckpointNode: ...


@dataclass(frozen=True)
class ExperimentRunResult:
    """Resolved result and durable evidence for a completed experiment."""

    state: str
    final_checkpoints: Mapping[str, ResolvedCheckpointNode]
    # ``arena`` remains the compatibility alias for the final Arena result.
    arena: ArenaRunResult
    original_parent: ResolvedCheckpointNode | None = None
    stage1_arena: ArenaRunResult | None = None
    stage1_winner: ResolvedCheckpointNode | None = None
    stage2_parent: ResolvedCheckpointNode | None = None
    stage2_arena: ArenaRunResult | None = None
    final_winner: ResolvedCheckpointNode | None = None
    decisions: Mapping[int, WinnerDecision] = field(default_factory=dict)


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
    """Coordinate one or two independent stages, then stop."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        arena_runner: ArenaRunnerV2,
        resolver: ArtifactResolver | None = None,
        experiment_root: str | Path | None = None,
        lineage_factory: LineageFactory | None = None,
        train_one: TrainOne | None = None,
        notifier: object | None = None,
    ) -> None:
        self.config = config
        self.arena_runner = arena_runner
        self.resolver = resolver or ArtifactResolver()
        self.lineage_factory = lineage_factory or Torus9ProductionLineage(self.resolver.runs_root)
        self.train_one = train_one or ProductionTrainOne(resolver=self.resolver)
        self._notifier = notifier
        self.logger = logging.getLogger(__name__)
        self.experiment_root = (
            Path(experiment_root).resolve()
            if experiment_root is not None
            else self.resolver.runs_root / config.topology / "experiments" / config.experiment_id
        )

    @property
    def state_path(self) -> Path:
        return self.experiment_root / "state.json"

    def run(self) -> ExperimentRunResult:
        self._launch_id = uuid.uuid4().hex
        original_parent = self.resolver.checkpoint(self.config.parent)
        state = self._load_or_create_state(original_parent)
        if state["state"] == "STOPPED":
            result = self._result_from_stopped_state(state)
            winner = result.final_winner
            self._notify_operator(
                "EXPERIMENT_COMPLETED",
                "Experiment is already completed.",
                key_suffix=f"completed:{winner.ref.sha256 if winner is not None else 'stopped'}",
            )
            return result
        if state["state"] == "ARENA_INVALID":
            stage = state.get("invalid_stage", "unknown")
            raise ExperimentRunnerError(f"stage {stage} Arena is invalid; no winner or Stage 2 is allowed")

        self._notify_operator(
            "EXPERIMENT_STARTED",
            f"Experiment {self.config.experiment_id} started from "
            f"{original_parent.topology}/{original_parent.lineage_id}/{original_parent.checkpoint_id}.",
            key_suffix=f"started:{self._launch_id}",
        )

        stage1_arms = {"A": self.config.arm_a, "B": self.config.arm_b}
        stage1_final = self._run_arms(
            state,
            stage_key="stage1",
            arms=stage1_arms,
            parent=original_parent,
        )
        if stage1_final["A"].lineage_id == stage1_final["B"].lineage_id:
            raise ExperimentRunnerError("A and B final checkpoints must have independent lineages")

        state["state"] = "STAGE1_ARENA"
        state["updated_at"] = self._now()
        _write_json(self.state_path, state)
        stage1_arena, stage1_decision = self._stage_arena(
            state,
            stage=1,
            candidate=stage1_final["B"],
            reference=stage1_final["A"],
            arena_config=self.config.arena_config,
            arena_master_seed=self.config.arena_master_seed,
            arena_startset=self.config.arena_startset,
            arena_profile=self.config.arena_profile,
            arena_scientific_contract=self.config.arena_scientific_contract,
            arena_execution_contract=self.config.arena_execution_contract,
            arena_workload=self.config.arena_workload,
            winner_rule=self.config.winner_rule,
            candidate_label="B-final",
            reference_label="A-final",
            comparison="B-final-vs-A-final",
        )
        stage1_winner = self.resolver.checkpoint(stage1_decision.winner)

        if self.config.stage2 is None:
            self._stop(
                state,
                final_winner=stage1_winner,
                reason="final A-vs-B Arena completed",
            )
            return self._result(
                original_parent=original_parent,
                stage1_final=stage1_final,
                stage1_arena=stage1_arena,
                stage1_winner=stage1_winner,
                stage1_decision=stage1_decision,
            )

        stage2 = self.config.stage2
        stage2_parent = stage1_winner
        stage2_arms = self._stage2_arms(stage2, stage2_parent, stage1_final)
        self._notify_operator(
            "STAGE2_STARTED",
            f"Stage 2 started from winner {stage2_parent.ref.lineage_id}/"
            f"{stage2_parent.ref.checkpoint_id}.",
            key_suffix=f"stage2-started:{stage2_parent.ref.sha256}",
        )
        stage2_state = state["stage2"]
        if not isinstance(stage2_state, dict):
            raise ExperimentRunnerError("experiment Stage 2 state is malformed")
        stored_parent = stage2_state.get("parent")
        if stored_parent is not None and stored_parent != stage2_parent.ref.to_dict():
            raise ExperimentRunnerError("Stage 2 parent changed during resume")
        stage2_state["parent"] = stage2_parent.ref.to_dict()
        state["state"] = "STAGE2_RUNNING"
        state["updated_at"] = self._now()
        _write_json(self.state_path, state)

        stage2_final = self._run_arms(
            state,
            stage_key="stage2",
            arms=stage2_arms,
            parent=stage2_parent,
        )
        if stage2_final["control"].lineage_id == stage2_final["C"].lineage_id:
            raise ExperimentRunnerError("control and C final checkpoints must have independent lineages")
        if {stage2_final["control"].lineage_id, stage2_final["C"].lineage_id}.intersection(
            {stage1_final["A"].lineage_id, stage1_final["B"].lineage_id}
        ):
            raise ExperimentRunnerError("Stage 2 arms must use independent lineages from Stage 1")

        state["state"] = "STAGE2_ARENA"
        state["updated_at"] = self._now()
        _write_json(self.state_path, state)
        stage2_arena, stage2_decision = self._stage_arena(
            state,
            stage=2,
            candidate=stage2_final["C"],
            reference=stage2_final["control"],
            arena_config=stage2.arena_config,
            arena_master_seed=stage2.arena_master_seed,
            arena_startset=stage2.arena_startset,
            arena_profile=stage2.arena_profile,
            arena_scientific_contract=stage2.arena_scientific_contract,
            arena_execution_contract=stage2.arena_execution_contract,
            arena_workload=stage2.arena_workload,
            winner_rule=stage2.winner_rule,
            candidate_label="C-final",
            reference_label="control-final",
            comparison="C-final-vs-control-final",
        )
        final_winner = self.resolver.checkpoint(stage2_decision.winner)
        self._stop(state, final_winner=final_winner, reason="final C-vs-control Arena completed")

        all_final = dict(stage1_final)
        all_final.update(stage2_final)
        return ExperimentRunResult(
            state="STOPPED",
            final_checkpoints=all_final,
            arena=stage2_arena,
            original_parent=original_parent,
            stage1_arena=stage1_arena,
            stage1_winner=stage1_winner,
            stage2_parent=stage2_parent,
            stage2_arena=stage2_arena,
            final_winner=final_winner,
            decisions={1: stage1_decision, 2: stage2_decision},
        )

    def _run_arms(
        self,
        state: dict[str, Any],
        *,
        stage_key: str,
        arms: Mapping[str, ExperimentArmConfig],
        parent: ResolvedCheckpointNode,
    ) -> dict[str, ResolvedCheckpointNode]:
        stage = state.get(stage_key)
        if not isinstance(stage, dict):
            raise ExperimentRunnerError(f"experiment {stage_key} state is malformed")
        raw_arms = stage.get("arms")
        if not isinstance(raw_arms, dict):
            raise ExperimentRunnerError(f"experiment {stage_key} arm state is malformed")

        final: dict[str, ResolvedCheckpointNode] = {}
        for arm_id, arm in arms.items():
            raw_record = raw_arms.get(arm_id)
            if raw_record is None:
                raw_record = {
                    "config_fingerprint": arm.effective_config.fingerprint,
                    "target_generation": parent.generation + arm.generations,
                }
                raw_arms[arm_id] = raw_record
            if not isinstance(raw_record, dict):
                raise ExperimentRunnerError(f"experiment {stage_key} arm {arm_id} state is malformed")
            if raw_record.get("config_fingerprint") != arm.effective_config.fingerprint:
                raise ExperimentRunnerError(f"experiment {stage_key} arm {arm_id} config changed during resume")
            if raw_record.get("target_generation") != parent.generation + arm.generations:
                raise ExperimentRunnerError(f"experiment {stage_key} arm {arm_id} budget changed during resume")

            raw_final = raw_record.get("final_checkpoint")
            if isinstance(raw_final, Mapping):
                checkpoint = self.resolver.checkpoint(raw_final)
            else:
                lineage_id = arm.lineage_id or f"{self.config.experiment_id}-{arm.arm_id}"
                root, effective_config = self.lineage_factory.prepare(
                    topology=self.config.topology,
                    lineage_id=lineage_id,
                    parent=parent,
                    effective_config=arm.effective_config,
                    experiment_id=self.config.experiment_id,
                    arm_id=arm.arm_id,
                )
                if effective_config.fingerprint != arm.effective_config.fingerprint:
                    raise ExperimentRunnerError(
                        f"arm {arm.arm_id} lineage resolved a different effective config"
                    )
                output_lineage = OutputLineage(self.config.topology, lineage_id, root)
                checkpoint = parent
                for _ in range(arm.generations):
                    previous = checkpoint
                    next_generation = previous.generation + 1
                    try:
                        checkpoint = self.train_one(
                            parent=checkpoint,
                            config=effective_config,
                            output_lineage=output_lineage,
                        )
                    except BaseException as exc:
                        self._notify_operator(
                            "CRITICAL",
                            f"Arm {arm.arm_id} generation M{next_generation} failed: "
                            f"{exc.__class__.__name__}.",
                            key_suffix=(
                                f"critical:train:{stage_key}:{arm.arm_id}:"
                                f"{next_generation}:{exc.__class__.__name__}"
                            ),
                        )
                        raise
                    if not isinstance(checkpoint, ResolvedCheckpointNode):
                        raise ExperimentRunnerError(
                            f"train_one returned an unresolved checkpoint for arm {arm.arm_id}"
                        )
                    if checkpoint.generation != previous.generation + 1:
                        raise ExperimentRunnerError(
                            f"arm {arm.arm_id} train_one did not return the immediate child"
                        )
                    if checkpoint.node.parent != previous.ref:
                        raise ExperimentRunnerError(
                            f"arm {arm.arm_id} train_one returned a child with the wrong parent"
                        )
                self._validate_final(arm, parent, checkpoint)
                raw_record["final_checkpoint"] = checkpoint.ref.to_dict()
                state["updated_at"] = self._now()
                _write_json(self.state_path, state)
            self._validate_final(arm, parent, checkpoint)
            final[arm_id] = checkpoint
            event = (
                f"{arm_id}_COMPLETED"
                if stage_key == "stage1"
                else f"STAGE2_{arm_id.upper()}_COMPLETED"
            )
            self._notify_operator(
                event,
                f"{event.replace('_', ' ').title()}: "
                f"{checkpoint.ref.lineage_id}/{checkpoint.ref.checkpoint_id}.",
                key_suffix=f"{event.lower()}:{checkpoint.ref.sha256}",
            )

        return final

    def _stage2_arms(
        self,
        stage2: ExperimentStage2Config,
        parent: ResolvedCheckpointNode,
        stage1_final: Mapping[str, ResolvedCheckpointNode],
    ) -> dict[str, ExperimentArmConfig]:
        stage1_lineages = {checkpoint.lineage_id for checkpoint in stage1_final.values()}
        c_lineage = stage2.c.lineage_id or f"{self.config.experiment_id}-stage2-C"
        control_lineage = f"{self.config.experiment_id}-stage2-control"
        if c_lineage in stage1_lineages or control_lineage in stage1_lineages:
            raise ExperimentRunnerError("Stage 2 lineage collides with a Stage 1 lineage")
        control = ExperimentArmConfig(
            "control",
            stage2.control_generations,
            parent.effective_config.config,
            lineage_id=control_lineage,
        )
        candidate = ExperimentArmConfig(
            "C",
            stage2.c.generations,
            stage2.c.effective_config,
            lineage_id=c_lineage,
        )
        return {"control": control, "C": candidate}

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
                f"arm {arm.arm_id} final generation is {final.generation}, expected {expected_generation}"
            )
        if final.lineage_id == parent.lineage_id:
            raise ExperimentRunnerError(f"arm {arm.arm_id} final reuses the common parent lineage")
        if arm.lineage_id is not None and final.lineage_id != arm.lineage_id:
            raise ExperimentRunnerError(f"arm {arm.arm_id} final has the wrong configured lineage")
        if final.effective_config.fingerprint != arm.effective_config.fingerprint:
            raise ExperimentRunnerError(f"arm {arm.arm_id} final has the wrong effective config")
        try:
            ancestor = self.resolver.ancestor(final, arm.generations)
        except Exception as exc:
            raise ExperimentRunnerError(
                f"arm {arm.arm_id} final does not resolve to the explicit common parent"
            ) from exc
        if ancestor.ref != parent.ref:
            raise ExperimentRunnerError(f"arm {arm.arm_id} final has the wrong explicit parent")

    def _stage_arena(
        self,
        state: dict[str, Any],
        *,
        stage: int,
        candidate: ResolvedCheckpointNode,
        reference: ResolvedCheckpointNode,
        arena_config: ArenaExecutionConfig,
        arena_master_seed: int,
        arena_startset: StartsetRef | None,
        arena_profile: str,
        arena_scientific_contract: Mapping[str, object] | None,
        arena_execution_contract: Mapping[str, object] | None,
        arena_workload: Mapping[str, object],
        winner_rule: WinnerRule,
        candidate_label: str,
        reference_label: str,
        comparison: str,
    ) -> tuple[ArenaRunResult, WinnerDecision]:
        stage_key = f"stage{stage}"
        stage_state = state.get(stage_key)
        if not isinstance(stage_state, dict):
            raise ExperimentRunnerError(f"experiment {stage_key} state is malformed")
        raw_arena = stage_state.get("arena")
        raw_decision = stage_state.get("winner")
        if isinstance(raw_arena, Mapping):
            # Re-enter the existing ArenaRunner identity/reuse path.  A
            # persisted coordinator record is evidence, not a second Arena
            # validator and not permission to bypass evaluation integrity.
            arena_result = self._run_arena(
                candidate=candidate,
                reference=reference,
                arena_config=arena_config,
                arena_master_seed=arena_master_seed,
                arena_startset=arena_startset,
                arena_profile=arena_profile,
                arena_scientific_contract=arena_scientific_contract,
                arena_execution_contract=arena_execution_contract,
                arena_workload=arena_workload,
                candidate_label=candidate_label,
                reference_label=reference_label,
                comparison=comparison,
            )
            self._validate_arena_result(arena_result, candidate, reference, stage)
        else:
            arena_result = self._run_arena(
                candidate=candidate,
                reference=reference,
                arena_config=arena_config,
                arena_master_seed=arena_master_seed,
                arena_startset=arena_startset,
                arena_profile=arena_profile,
                arena_scientific_contract=arena_scientific_contract,
                arena_execution_contract=arena_execution_contract,
                arena_workload=arena_workload,
                candidate_label=candidate_label,
                reference_label=reference_label,
                comparison=comparison,
            )
            stage_state["arena"] = self._arena_state(arena_result)
            state["updated_at"] = self._now()
            _write_json(self.state_path, state)

        if arena_result.validity != "VALID":
            state["state"] = "ARENA_INVALID"
            state["invalid_stage"] = stage
            state["stop_reason"] = f"stage {stage} Arena validity={arena_result.validity}"
            state["updated_at"] = self._now()
            _write_json(self.state_path, state)
            self._notify_operator(
                "CRITICAL",
                f"Stage {stage} Arena is {arena_result.validity}; no winner was recorded.",
                key_suffix=f"critical:arena:{stage}:{arena_result.evaluation_id}",
            )
            raise ExperimentRunnerError(
                f"stage {stage} Arena is {arena_result.validity}; no scientific winner was recorded"
            )

        if isinstance(raw_decision, Mapping):
            decision = WinnerDecision.from_dict(raw_decision)
            self._validate_decision(decision, arena_result, candidate, reference, winner_rule, stage)
        else:
            try:
                wins, losses, draws = arena_result.wld
            except (TypeError, ValueError) as exc:
                raise ExperimentRunnerError(f"stage {stage} valid Arena has malformed W/L/D") from exc
            decision = WinnerDecision(
                stage=stage,
                evaluation_id=arena_result.evaluation_id,
                evaluation_fingerprint=arena_result.evaluation_fingerprint,
                candidate=candidate.ref,
                reference=reference.ref,
                wins=wins,
                losses=losses,
                draws=draws,
                winner_rule=winner_rule,
                winner=winner_rule.choose(
                    candidate=candidate.ref,
                    reference=reference.ref,
                    wins=wins,
                    losses=losses,
                ),
            )
            stage_state["winner"] = decision.to_dict()
            state["updated_at"] = self._now()
            _write_json(self.state_path, state)
        winner_label = candidate_label if decision.winner == candidate.ref else reference_label
        self._notify_operator(
            f"STAGE{stage}_ARENA_COMPLETED",
            f"Stage {stage} Arena completed: W/L/D={decision.wins}/{decision.losses}/{decision.draws}; "
            f"winner={winner_label} ({decision.winner.lineage_id}/{decision.winner.checkpoint_id}).",
            key_suffix=f"stage{stage}-arena:{arena_result.evaluation_id}",
        )
        return arena_result, decision

    def _run_arena(
        self,
        *,
        candidate: ResolvedCheckpointNode,
        reference: ResolvedCheckpointNode,
        arena_config: ArenaExecutionConfig,
        arena_master_seed: int,
        arena_startset: StartsetRef | None,
        arena_profile: str,
        arena_scientific_contract: Mapping[str, object] | None,
        arena_execution_contract: Mapping[str, object] | None,
        arena_workload: Mapping[str, object],
        candidate_label: str,
        reference_label: str,
        comparison: str,
    ) -> ArenaRunResult:
        startset = arena_startset
        if startset is None:
            if self.config.topology != "torus9":
                raise ValueError("arena_startset is required for non-torus9 experiments")
            startset = torus9_startset_ref(master_seed=arena_master_seed, games=arena_config.games)
        request = ArenaRunRequest(
            candidate=candidate,
            reference=reference,
            master_seed=arena_master_seed,
            startset=startset,
            config=arena_config,
            profile=arena_profile,
            scientific_contract=arena_scientific_contract,
            execution_contract=arena_execution_contract,
            workload=arena_workload,
            candidate_label=candidate_label,
            reference_label=reference_label,
            comparison=comparison,
        )
        try:
            return self.arena_runner.run(request)
        except BaseException as exc:
            self._notify_operator(
                "CRITICAL",
                f"Stage Arena execution failed: {exc.__class__.__name__}.",
                key_suffix=f"critical:arena-execution:{candidate.ref.sha256}:{reference.ref.sha256}",
            )
            raise

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

    @staticmethod
    def _arena_result_from_state(raw: Mapping[str, object]) -> ArenaRunResult:
        try:
            identity = EvaluationIdentity.from_dict(raw["identity"])  # type: ignore[arg-type]
            summary = raw["summary"]
            if not isinstance(summary, Mapping):
                raise TypeError("summary is not an object")
            return ArenaRunResult(
                evaluation_id=str(raw["evaluation_id"]),
                evaluation_fingerprint=str(raw["evaluation_fingerprint"]),
                output_dir=Path(str(raw["output_dir"])),
                identity=identity,
                summary=dict(summary),
                validity=str(raw["validity"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentRunnerError("persisted Arena evidence is malformed") from exc

    @staticmethod
    def _validate_arena_result(
        result: ArenaRunResult,
        candidate: ResolvedCheckpointNode,
        reference: ResolvedCheckpointNode,
        stage: int,
    ) -> None:
        if result.identity.candidate != candidate.ref or result.identity.reference != reference.ref:
            raise ExperimentRunnerError(f"persisted stage {stage} Arena checkpoints do not match the stage")
        if result.evaluation_id != result.output_dir.name:
            raise ExperimentRunnerError(f"persisted stage {stage} Arena evaluation id is malformed")

    @staticmethod
    def _validate_decision(
        decision: WinnerDecision,
        arena: ArenaRunResult,
        candidate: ResolvedCheckpointNode,
        reference: ResolvedCheckpointNode,
        winner_rule: WinnerRule,
        stage: int,
    ) -> None:
        if decision.stage != stage:
            raise ExperimentRunnerError(f"persisted stage {stage} winner decision has the wrong stage")
        if decision.evaluation_id != arena.evaluation_id or decision.evaluation_fingerprint != arena.evaluation_fingerprint:
            raise ExperimentRunnerError(f"persisted stage {stage} winner decision has the wrong evaluation")
        if decision.candidate != candidate.ref or decision.reference != reference.ref:
            raise ExperimentRunnerError(f"persisted stage {stage} winner decision has the wrong checkpoints")
        if decision.winner_rule != winner_rule:
            raise ExperimentRunnerError(f"persisted stage {stage} winner rule changed during resume")
        if decision.winner != winner_rule.choose(
            candidate=candidate.ref,
            reference=reference.ref,
            wins=decision.wins,
            losses=decision.losses,
        ):
            raise ExperimentRunnerError(f"persisted stage {stage} winner decision is inconsistent")

    def _stop(
        self,
        state: dict[str, Any],
        *,
        final_winner: ResolvedCheckpointNode,
        reason: str,
    ) -> None:
        state["state"] = "STOPPED"
        state["final_winner"] = final_winner.ref.to_dict()
        state["updated_at"] = self._now()
        state["stop_reason"] = reason
        _write_json(self.state_path, state)
        self._notify_operator(
            "EXPERIMENT_COMPLETED",
            f"Experiment {self.config.experiment_id} completed; winner="
            f"{final_winner.ref.lineage_id}/{final_winner.ref.checkpoint_id}.",
            key_suffix=f"completed:{final_winner.ref.sha256}",
        )

    def _load_or_create_state(self, parent: ResolvedCheckpointNode) -> dict[str, Any]:
        if self.state_path.is_file():
            raw_state = dict(_read_json(self.state_path))
            if self._is_legacy_state(raw_state):
                return self._migrate_legacy_state(raw_state, parent)
            state = raw_state
            if state.get("schema") != EXPERIMENT_STATE_SCHEMA:
                raise ExperimentRunnerError("unsupported experiment state schema")
            if state.get("experiment_id") != self.config.experiment_id:
                raise ExperimentRunnerError("experiment state id mismatch")
            if state.get("topology") != self.config.topology:
                raise ExperimentRunnerError("experiment state topology mismatch")
            if state.get("config_fingerprint") != self.config.fingerprint:
                raise ExperimentRunnerError("experiment config changed during resume")
            if state.get("original_parent") != parent.ref.to_dict():
                raise ExperimentRunnerError("experiment parent changed during resume")
            if state.get("state") not in {
                "RUNNING",
                "STAGE1_ARENA",
                "STAGE2_RUNNING",
                "STAGE2_ARENA",
                "ARENA_INVALID",
                "STOPPED",
            }:
                raise ExperimentRunnerError("experiment state is malformed")
            raw_stage2 = state.get("stage2")
            if not isinstance(raw_stage2, Mapping) or raw_stage2.get("enabled") != self.config.stage2_enabled:
                raise ExperimentRunnerError("experiment Stage 2 enablement changed during resume")
            self._validate_state_shape(state)
            return state

        now = self._now()
        state: dict[str, Any] = {
            "schema": EXPERIMENT_STATE_SCHEMA,
            "version": 3,
            "experiment_id": self.config.experiment_id,
            "topology": self.config.topology,
            "config_fingerprint": self.config.fingerprint,
            "original_parent": parent.ref.to_dict(),
            # Readable compatibility alias; Stage 2 never uses it as parent.
            "parent": parent.ref.to_dict(),
            "state": "RUNNING",
            "stage1": {
                "arms": {
                    arm.arm_id: {
                        "config_fingerprint": arm.effective_config.fingerprint,
                        "target_generation": parent.generation + arm.generations,
                    }
                    for arm in self.config.arms
                },
                "arena": None,
                "winner": None,
            },
            "stage2": {
                "enabled": self.config.stage2_enabled,
                "parent": None,
                "arms": {},
                "arena": None,
                "winner": None,
            },
            "created_at": now,
            "updated_at": now,
        }
        _write_json(self.state_path, state)
        return state

    @staticmethod
    def _is_legacy_state(state: Mapping[str, object]) -> bool:
        """Recognize the exact top-level shape persisted by PR #155."""
        return (
            state.get("schema") == EXPERIMENT_STATE_SCHEMA
            and state.get("version") == 2
            and "arms" in state
            and "stage1" not in state
        )

    def _migrate_legacy_state(
        self,
        legacy: Mapping[str, object],
        parent: ResolvedCheckpointNode,
    ) -> dict[str, Any]:
        """Atomically convert a PR #155 A/B state into the v3 shape.

        Legacy state never had Stage 2, so it is accepted only for a current
        config with Stage 2 disabled and only when its old config fingerprint
        matches exactly.  A legacy STOPPED state must contain a valid Arena
        and complete A/B checkpoint evidence; otherwise it is not upgraded to
        a scientific decision.
        """
        if self.config.stage2 is not None:
            raise ExperimentRunnerError("legacy A/B state cannot be resumed with Stage 2 enabled")
        if legacy.get("config_fingerprint") != self.config.legacy_fingerprint:
            raise ExperimentRunnerError("legacy experiment config changed during resume")
        if legacy.get("parent") != parent.ref.to_dict():
            raise ExperimentRunnerError("legacy experiment parent changed during resume")
        if legacy.get("experiment_id") != self.config.experiment_id:
            raise ExperimentRunnerError("legacy experiment state id mismatch")
        if legacy.get("topology") != self.config.topology:
            raise ExperimentRunnerError("legacy experiment state topology mismatch")

        raw_arms = legacy.get("arms")
        if not isinstance(raw_arms, Mapping) or set(raw_arms) != {"A", "B"}:
            raise ExperimentRunnerError("legacy experiment state arms are malformed")
        arms = {
            arm_id: dict(record)
            for arm_id, record in raw_arms.items()
            if isinstance(record, Mapping)
        }
        if set(arms) != {"A", "B"}:
            raise ExperimentRunnerError("legacy experiment state arm records are malformed")

        legacy_state = legacy.get("state")
        if legacy_state not in {"RUNNING", "ARENA", "STOPPED"}:
            raise ExperimentRunnerError("legacy experiment state is malformed")
        state_name = "STAGE1_ARENA" if legacy_state == "ARENA" else str(legacy_state)
        stage1_arena = legacy.get("arena_result")
        decision: WinnerDecision | None = None

        complete_finals = all(
            isinstance(arms[arm_id].get("final_checkpoint"), Mapping)
            for arm_id in ("A", "B")
        )
        if legacy_state == "STOPPED" and not complete_finals:
            raise ExperimentRunnerError("legacy STOPPED state lacks final A/B checkpoints")

        final: dict[str, ResolvedCheckpointNode] = {}
        if complete_finals:
            for arm in (self.config.arm_a, self.config.arm_b):
                checkpoint = self.resolver.checkpoint(arms[arm.arm_id]["final_checkpoint"])
                self._validate_final(arm, parent, checkpoint)
                final[arm.arm_id] = checkpoint
            if final["A"].lineage_id == final["B"].lineage_id:
                raise ExperimentRunnerError("legacy A/B final checkpoints share a lineage")

        if stage1_arena is not None:
            if not isinstance(stage1_arena, Mapping):
                raise ExperimentRunnerError("legacy Arena evidence is malformed")
            arena_result = self._arena_result_from_state(stage1_arena)
            if arena_result.validity != "VALID":
                state_name = "ARENA_INVALID"
            elif complete_finals:
                self._validate_arena_result(arena_result, final["B"], final["A"], 1)
                try:
                    wins, losses, draws = arena_result.wld
                except (TypeError, ValueError) as exc:
                    raise ExperimentRunnerError("legacy valid Arena has malformed W/L/D") from exc
                decision = WinnerDecision(
                    stage=1,
                    evaluation_id=arena_result.evaluation_id,
                    evaluation_fingerprint=arena_result.evaluation_fingerprint,
                    candidate=final["B"].ref,
                    reference=final["A"].ref,
                    wins=wins,
                    losses=losses,
                    draws=draws,
                    winner_rule=self.config.winner_rule,
                    winner=self.config.winner_rule.choose(
                        candidate=final["B"].ref,
                        reference=final["A"].ref,
                        wins=wins,
                        losses=losses,
                    ),
                )

        if legacy_state == "STOPPED":
            if state_name == "ARENA_INVALID":
                # Never preserve the old runner's STOPPED claim for an
                # invalid Arena; it was not a scientific decision.
                state_name = "ARENA_INVALID"
            elif decision is None:
                raise ExperimentRunnerError("legacy STOPPED state lacks valid Arena winner evidence")

        now = self._now()
        migrated: dict[str, Any] = {
            "schema": EXPERIMENT_STATE_SCHEMA,
            "version": 3,
            "experiment_id": self.config.experiment_id,
            "topology": self.config.topology,
            "config_fingerprint": self.config.fingerprint,
            "original_parent": parent.ref.to_dict(),
            "parent": parent.ref.to_dict(),
            "state": state_name,
            "stage1": {
                "arms": arms,
                "arena": None if stage1_arena is None else dict(stage1_arena),
                "winner": None if decision is None else decision.to_dict(),
            },
            "stage2": {
                "enabled": False,
                "parent": None,
                "arms": {},
                "arena": None,
                "winner": None,
            },
            "created_at": legacy.get("created_at", now),
            "updated_at": now,
            "legacy_migration": {
                "from_schema": EXPERIMENT_STATE_SCHEMA,
                "from_version": 2,
                "legacy_config_fingerprint": legacy["config_fingerprint"],
            },
        }
        if state_name == "ARENA_INVALID":
            migrated["invalid_stage"] = 1
            migrated["stop_reason"] = "legacy Stage 1 Arena was invalid"
        elif state_name == "STOPPED":
            assert decision is not None
            migrated["final_winner"] = decision.winner.to_dict()
            migrated["stop_reason"] = legacy.get("stop_reason", "migrated legacy final A-vs-B Arena")
        _write_json(self.state_path, migrated)
        return migrated

    @staticmethod
    def _validate_state_shape(state: Mapping[str, object]) -> None:
        for key in ("stage1", "stage2"):
            stage = state.get(key)
            if not isinstance(stage, Mapping):
                raise ExperimentRunnerError(f"experiment {key} state is malformed")
            if not isinstance(stage.get("arms"), Mapping):
                raise ExperimentRunnerError(f"experiment {key} arm state is malformed")

    def _result(
        self,
        *,
        original_parent: ResolvedCheckpointNode,
        stage1_final: Mapping[str, ResolvedCheckpointNode],
        stage1_arena: ArenaRunResult,
        stage1_winner: ResolvedCheckpointNode,
        stage1_decision: WinnerDecision,
    ) -> ExperimentRunResult:
        return ExperimentRunResult(
            state="STOPPED",
            final_checkpoints=dict(stage1_final),
            arena=stage1_arena,
            original_parent=original_parent,
            stage1_arena=stage1_arena,
            stage1_winner=stage1_winner,
            final_winner=stage1_winner,
            decisions={1: stage1_decision},
        )

    def _result_from_stopped_state(self, state: Mapping[str, object]) -> ExperimentRunResult:
        self._validate_state_shape(state)
        original_parent = self.resolver.checkpoint(self.config.parent)
        stage1_state = state["stage1"]
        assert isinstance(stage1_state, Mapping)
        stage1_arms_state = stage1_state["arms"]
        assert isinstance(stage1_arms_state, Mapping)
        stage1_final: dict[str, ResolvedCheckpointNode] = {}
        for arm in (self.config.arm_a, self.config.arm_b):
            record = stage1_arms_state.get(arm.arm_id)
            if not isinstance(record, Mapping) or not isinstance(record.get("final_checkpoint"), Mapping):
                raise ExperimentRunnerError(f"STOPPED state lacks final checkpoint for arm {arm.arm_id}")
            checkpoint = self.resolver.checkpoint(record["final_checkpoint"])  # type: ignore[arg-type]
            self._validate_final(arm, original_parent, checkpoint)
            stage1_final[arm.arm_id] = checkpoint
        stage1_arena = self._arena_result_from_stopped_stage(stage1_state)
        stage1_decision = self._decision_from_stopped_stage(
            stage1_state,
            stage=1,
            arena=stage1_arena,
            candidate=stage1_final["B"],
            reference=stage1_final["A"],
            winner_rule=self.config.winner_rule,
        )
        stage1_winner = self.resolver.checkpoint(stage1_decision.winner)

        if self.config.stage2 is None:
            return self._result(
                original_parent=original_parent,
                stage1_final=stage1_final,
                stage1_arena=stage1_arena,
                stage1_winner=stage1_winner,
                stage1_decision=stage1_decision,
            )

        stage2 = self.config.stage2
        stage2_parent = self.resolver.checkpoint(stage1_decision.winner)
        stage2_arms = self._stage2_arms(stage2, stage2_parent, stage1_final)
        stage2_state = state["stage2"]
        assert isinstance(stage2_state, Mapping)
        if stage2_state.get("parent") != stage2_parent.ref.to_dict():
            raise ExperimentRunnerError("STOPPED state Stage 2 parent is not Stage 1 winner")
        raw_stage2_arms = stage2_state.get("arms")
        if not isinstance(raw_stage2_arms, Mapping):
            raise ExperimentRunnerError("STOPPED state lacks Stage 2 arm evidence")
        stage2_final: dict[str, ResolvedCheckpointNode] = {}
        for arm_id, arm in stage2_arms.items():
            record = raw_stage2_arms.get(arm_id)
            if not isinstance(record, Mapping) or not isinstance(record.get("final_checkpoint"), Mapping):
                raise ExperimentRunnerError(f"STOPPED state lacks final checkpoint for Stage 2 arm {arm_id}")
            checkpoint = self.resolver.checkpoint(record["final_checkpoint"])  # type: ignore[arg-type]
            self._validate_final(arm, stage2_parent, checkpoint)
            stage2_final[arm_id] = checkpoint
        stage2_arena = self._arena_result_from_stopped_stage(stage2_state)
        stage2_decision = self._decision_from_stopped_stage(
            stage2_state,
            stage=2,
            arena=stage2_arena,
            candidate=stage2_final["C"],
            reference=stage2_final["control"],
            winner_rule=stage2.winner_rule,
        )
        final_winner = self.resolver.checkpoint(stage2_decision.winner)
        all_final = dict(stage1_final)
        all_final.update(stage2_final)
        return ExperimentRunResult(
            state="STOPPED",
            final_checkpoints=all_final,
            arena=stage2_arena,
            original_parent=original_parent,
            stage1_arena=stage1_arena,
            stage1_winner=stage1_winner,
            stage2_parent=stage2_parent,
            stage2_arena=stage2_arena,
            final_winner=final_winner,
            decisions={1: stage1_decision, 2: stage2_decision},
        )

    @staticmethod
    def _arena_result_from_stopped_stage(stage: Mapping[str, object]) -> ArenaRunResult:
        raw_arena = stage.get("arena")
        if not isinstance(raw_arena, Mapping):
            raise ExperimentRunnerError("STOPPED state lacks Arena evidence")
        return ExperimentRunnerV2._arena_result_from_state(raw_arena)

    @staticmethod
    def _decision_from_stopped_stage(
        stage_state: Mapping[str, object],
        *,
        stage: int,
        arena: ArenaRunResult,
        candidate: ResolvedCheckpointNode,
        reference: ResolvedCheckpointNode,
        winner_rule: WinnerRule,
    ) -> WinnerDecision:
        raw_decision = stage_state.get("winner")
        if not isinstance(raw_decision, Mapping):
            raise ExperimentRunnerError(f"STOPPED state lacks Stage {stage} winner evidence")
        decision = WinnerDecision.from_dict(raw_decision)
        ExperimentRunnerV2._validate_arena_result(arena, candidate, reference, stage)
        ExperimentRunnerV2._validate_decision(decision, arena, candidate, reference, winner_rule, stage)
        if arena.validity != "VALID":
            raise ExperimentRunnerError(f"STOPPED state has invalid Stage {stage} Arena")
        return decision

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def _notify_operator(self, event: str, message: str, *, key_suffix: str) -> None:
        """Report an injected operator event; observability stays fail-open."""
        if self._notifier is None:
            return
        try:
            self._notifier.send_now(
                f"experiment:{self.config.experiment_id}:{key_suffix}",
                f"{event} — {message}",
            )
        except BaseException:
            self.logger.warning("operator notification failed", exc_info=True)


ExperimentRunner = ExperimentRunnerV2


__all__ = [
    "EXPERIMENT_RUNNER_SCHEMA",
    "EXPERIMENT_STATE_SCHEMA",
    "EXPERIMENT_WINNER_RULE",
    "ExperimentArmConfig",
    "ExperimentConfig",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentRunnerError",
    "ExperimentRunnerV2",
    "ExperimentStage2Config",
    "LineageFactory",
    "Stage2Config",
    "TrainOne",
    "WinnerDecision",
    "WinnerRule",
    "WinnerRuleName",
]
