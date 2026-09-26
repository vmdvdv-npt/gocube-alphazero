"""Topology-neutral public ArenaRunner V2 facade."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
from typing import Mapping

from . import _arena_runner_core as _core
from .contracts import EvaluationIdentity
from .execution_permit import _child_execution_permit
from .immutable_runtime import execution_commit_from_lineage
from ..notifications import EventType, OperatorEvent, coerce_event_sink
from .version import require_v2_process
from .supervisor import SupervisorPolicy
from tools.arena_profiles import get_profile

ARENA_RESULT_PROVENANCE_SCHEMA = _core.ARENA_RESULT_PROVENANCE_SCHEMA
ArenaRunRequest = _core.ArenaRunRequest
ArenaRunResult = _core.ArenaRunResult
torus9_startset_ref = _core.torus9_startset_ref
evaluation_dir = _core.evaluation_dir


def _common_wld_result(result: ArenaRunResult) -> ArenaRunResult:
    summary = dict(result.summary)
    values = summary.get("W/L/D")
    if not (isinstance(values, (list, tuple)) and len(values) == 3):
        values = (
            summary.get("candidate_wins"),
            summary.get("reference_wins"),
            summary.get("draws"),
        )
        if not all(type(value) is int and value >= 0 for value in values):
            return result
        summary["W/L/D"] = [int(values[0]), int(values[1]), int(values[2])]
    else:
        values = tuple(int(value) for value in values)
    valid_games = int(summary.get("valid_games", summary.get("games_valid", sum(values))))
    metrics = summary.get("metrics")
    metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
    metrics.setdefault("candidate_wins", int(values[0]))
    metrics.setdefault("reference_wins", int(values[1]))
    metrics.setdefault("draws", int(values[2]))
    metrics.setdefault("valid_games", valid_games)
    if valid_games > 0:
        metrics.setdefault("candidate_win_rate", int(values[0]) / valid_games)
    summary["metrics"] = metrics
    return replace(result, summary=summary)


class ArenaRunner(_core.ArenaRunner):
    """Run one V2 Arena evaluation with profile-owned scientific semantics."""

    def __init__(
        self,
        engine=None,
        *,
        notifier: object | None = None,
        event_sink: object | None = None,
        supervisor_policy: SupervisorPolicy | None = None,
    ) -> None:
        super().__init__(engine=engine, supervisor_policy=supervisor_policy)
        self.event_sink = coerce_event_sink(event_sink if event_sink is not None else notifier)
        # Kept as a compatibility attribute for callers that used to share a
        # notifier instance with ContinuousTrainingRunnerV2.  V2 itself uses
        # event_sink and never inspects the notifier class or module.
        self.notifier = notifier

    @staticmethod
    def _identity(request: ArenaRunRequest) -> EvaluationIdentity:
        profile = get_profile(request.profile)
        scientific = dict(request.scientific_contract or profile.scientific_contract(request.config))
        if request.candidate.topology.startswith("cube"):
            scientific.update(
                {
                    "startset_id": request.startset.id,
                    "startset_fingerprint": request.startset.fingerprint,
                    "startset_master_seed": int(request.master_seed),
                    "startset_pairs": int(request.config.games) // 2,
                    "startset_paired_colors": True,
                    "startset_empty_control": True,
                }
            )
        execution = dict(
            request.execution_contract
            or {
                "engine": "process-central-inference-v1",
                "workers": int(request.config.workers),
                "contexts": int(request.config.workers) * int(request.config.games_per_worker),
                "inference_batch_cap": int(request.config.inference_batch_rows),
                "inference_batch_wait_ms": float(request.config.inference_batch_wait_ms),
                "monitoring_acceptance": bool(request.config.monitoring_acceptance),
                "config": asdict(request.config),
            }
        )
        workload = dict(request.workload)
        workload.setdefault("pairs", int(request.config.games) // 2)
        workload.setdefault("paired_starts", True)
        workload.setdefault("color_swap", True)
        workload.setdefault("startset_fingerprint", request.startset.fingerprint)
        workload.setdefault("startset_id", request.startset.id)
        return EvaluationIdentity(
            candidate=request.candidate.ref,
            reference=request.reference.ref,
            games=int(request.config.games),
            master_seed=int(request.master_seed),
            startset=request.startset,
            scientific_contract=scientific,
            execution_contract=execution,
            workload=workload,
        )

    def _evaluation_id(self, request: ArenaRunRequest) -> str:
        identity = self._identity(request)
        fingerprint = _core.evaluation_fingerprint(identity.to_dict())
        return _core.evaluation_id(
            candidate_lineage_id=request.candidate.lineage_id,
            candidate_generation=request.candidate.generation,
            reference_lineage_id=request.reference.lineage_id,
            reference_generation=request.reference.generation,
            fingerprint=fingerprint,
        )

    def _notify_start(self, request: ArenaRunRequest, evaluation_id: str) -> None:
        # Old injected send_now fakes predate structured start events.  The
        # real compatibility facade opts in explicitly; structured sinks
        # receive the event without any transport/class inspection.
        if hasattr(self.event_sink, "notifier") and not getattr(self.event_sink, "supports_starts", False):
            return
        profile = get_profile(request.profile)
        scientific = dict(request.scientific_contract or {})
        simulations = getattr(profile, "simulations", scientific.get("simulations"))
        komi = getattr(profile, "komi", scientific.get("komi"))
        event = OperatorEvent.create(
            EventType.ARENA_STARTED,
            topology=request.candidate.topology,
            owner_type="evaluation",
            owner_id=evaluation_id,
            action_id=evaluation_id,
            payload={
                "evaluation_id": evaluation_id,
                "candidate": request.candidate_label or request.candidate.checkpoint_id,
                "reference": request.reference_label or request.reference.checkpoint_id,
                "candidate_lineage": request.candidate.lineage_id,
                "reference_lineage": request.reference.lineage_id,
                "profile": request.profile,
                "komi": komi,
                "games": int(request.config.games),
                "simulations": simulations,
                "seed": int(request.master_seed),
                "workers": int(request.config.workers),
                "contexts": int(request.config.workers) * int(request.config.games_per_worker),
                "batch_cap": int(request.config.inference_batch_rows),
                "wait_ms": float(request.config.inference_batch_wait_ms),
            },
            evidence_refs=[{"ref": str(request.candidate.ref.path), "kind": "candidate"}, {"ref": str(request.reference.ref.path), "kind": "reference"}],
            producer_version="orchestrator-v2",
            execution_code_commit=request.execution_code_commit,
            identity={"evaluation_identity": self._identity(request).to_dict(), "phase": "started"},
        )
        try:
            self.event_sink.publish(event)
        except Exception:
            pass

    def _notify_completed(self, request: ArenaRunRequest, result: ArenaRunResult) -> None:
        summary = dict(result.summary)
        safe_summary = {
            key: value
            for key, value in summary.items()
            if key in {
                "games", "valid_games", "technical_games", "invalid_games",
                "performance_status", "inference_mean_batch_rows", "evaluation_report",
            }
            and isinstance(value, (str, int, float, bool))
        }
        candidate = request.candidate_label or request.candidate.checkpoint_id
        reference = request.reference_label or request.reference.checkpoint_id
        event = OperatorEvent.create(
            EventType.ARENA_COMPLETED,
            topology=request.candidate.topology,
            owner_type="evaluation",
            owner_id=result.evaluation_id,
            action_id=result.evaluation_id,
            payload={
                "evaluation_id": result.evaluation_id,
                "evaluation_fingerprint": result.evaluation_fingerprint,
                "candidate": candidate,
                "reference": reference,
                "candidate_lineage": request.candidate.lineage_id,
                "reference_lineage": request.reference.lineage_id,
                "wld": list(result.wld),
                "validity": result.validity,
                "evaluation_report": str(result.output_dir / "result.json"),
                **safe_summary,
            },
            evidence_refs=[
                {"ref": str(result.output_dir / "result.json"), "kind": "arena-result", "evaluation_fingerprint": result.evaluation_fingerprint},
                {"ref": str(result.output_dir / "provenance.json"), "kind": "provenance"},
            ],
            producer_version="orchestrator-v2",
            execution_code_commit=result.execution_code_commit,
            identity={"evaluation_identity": result.identity.to_dict(), "phase": "completed"},
        )
        try:
            self.event_sink.reconcile_completed(
                {"action_id": result.evaluation_id, "event_type": EventType.ARENA_COMPLETED.value},
                event.to_dict(),
            )
        except Exception:
            pass  # Notification failures cannot invalidate completed Arena work.

    def run(self, request: ArenaRunRequest) -> ArenaRunResult:
        def execute() -> ArenaRunResult:
            if evaluation_dir is _core.evaluation_dir:
                return _common_wld_result(super(ArenaRunner, self).run(request))
            original = _core.evaluation_dir
            _core.evaluation_dir = evaluation_dir
            try:
                return _common_wld_result(super(ArenaRunner, self).run(request))
            finally:
                _core.evaluation_dir = original

        permit = nullcontext()
        if self.engine is _core.production_arena:
            require_v2_process("gocube_golden.orchestrator_v2.ArenaRunnerV2")
            evaluation_id = self._evaluation_id(request)
            code_identity = request.execution_code_commit or execution_commit_from_lineage(request.candidate.owner_root)
            self._notify_start(request, evaluation_id)
            permit = _child_execution_permit(
                action_type="arena",
                topology=request.candidate.topology,
                run_id=evaluation_id,
                code_identity=code_identity,
            )
        try:
            with permit:
                result = execute()
        except Exception as exc:
            evaluation_id = self._evaluation_id(request)
            try:
                self.event_sink.publish(
                    OperatorEvent.create(
                        EventType.ARENA_FAILED,
                        topology=request.candidate.topology,
                        owner_type="evaluation",
                        owner_id=evaluation_id,
                        action_id=evaluation_id,
                        payload={
                            "evaluation_id": evaluation_id,
                            "candidate": request.candidate_label or request.candidate.checkpoint_id,
                            "reference": request.reference_label or request.reference.checkpoint_id,
                            "candidate_lineage": request.candidate.lineage_id,
                            "reference_lineage": request.reference.lineage_id,
                            "error_code": exc.__class__.__name__,
                        },
                        evidence_refs=[{"ref": str(request.output_dir) if request.output_dir else "evaluation-output", "kind": "execution"}],
                        producer_version="orchestrator-v2",
                        execution_code_commit=request.execution_code_commit,
                        identity={"evaluation_identity": self._identity(request).to_dict(), "phase": "failed"},
                    )
                )
            except Exception:
                pass
            raise
        self._notify_completed(request, result)
        return result


ArenaRunnerV2 = ArenaRunner

__all__ = [
    "ARENA_RESULT_PROVENANCE_SCHEMA", "ArenaRunRequest", "ArenaRunResult", "ArenaRunner",
    "ArenaRunnerV2", "evaluation_dir", "torus9_startset_ref",
]
