"""Topology-neutral public ArenaRunner V2 facade.

The execution/storage implementation remains shared; this facade removes the
historical Torus scientific default and resolves scientific semantics through
the common Arena profile registry.
"""
from __future__ import annotations

from dataclasses import asdict

from . import _arena_runner_core as _core
from .contracts import EvaluationIdentity
from tools.arena_profiles import get_profile

ARENA_RESULT_PROVENANCE_SCHEMA = _core.ARENA_RESULT_PROVENANCE_SCHEMA
ArenaRunRequest = _core.ArenaRunRequest
ArenaRunResult = _core.ArenaRunResult
torus9_startset_ref = _core.torus9_startset_ref

# Preserve the historical module-level patch seam used by synthetic tests and
# callers that redirect cross-lineage evaluation storage.  Production keeps
# the original function, so the core module is not mutated in normal runs.
evaluation_dir = _core.evaluation_dir


class ArenaRunner(_core.ArenaRunner):
    """Run one V2 Arena evaluation with profile-owned scientific semantics."""

    @staticmethod
    def _identity(request: ArenaRunRequest) -> EvaluationIdentity:
        profile = get_profile(request.profile)
        scientific = dict(
            request.scientific_contract
            or profile.scientific_contract(request.config)
        )
        execution = dict(
            request.execution_contract
            or {
                "engine": "process-central-inference-v1",
                "workers": int(request.config.workers),
                "contexts": int(request.config.workers)
                * int(request.config.games_per_worker),
                "inference_batch_cap": int(request.config.inference_batch_rows),
                "inference_batch_wait_ms": float(
                    request.config.inference_batch_wait_ms
                ),
                "monitoring_acceptance": bool(
                    request.config.monitoring_acceptance
                ),
                "config": asdict(request.config),
            }
        )
        workload = dict(request.workload)
        workload.setdefault("pairs", int(request.config.games) // 2)
        workload.setdefault("paired_starts", True)
        workload.setdefault("color_swap", True)
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

    def run(self, request: ArenaRunRequest) -> ArenaRunResult:
        # `_arena_runner_core` is the unchanged Stage-7 execution/storage
        # implementation. Only propagate a deliberately overridden public
        # evaluation_dir seam (normally used by tests); production takes the
        # fast path without touching module globals.
        if evaluation_dir is _core.evaluation_dir:
            return super().run(request)
        original = _core.evaluation_dir
        _core.evaluation_dir = evaluation_dir
        try:
            return super().run(request)
        finally:
            _core.evaluation_dir = original


ArenaRunnerV2 = ArenaRunner


__all__ = [
    "ARENA_RESULT_PROVENANCE_SCHEMA",
    "ArenaRunRequest",
    "ArenaRunResult",
    "ArenaRunner",
    "ArenaRunnerV2",
    "evaluation_dir",
    "torus9_startset_ref",
]
