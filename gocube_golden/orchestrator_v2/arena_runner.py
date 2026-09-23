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


ArenaRunnerV2 = ArenaRunner


__all__ = [
    "ARENA_RESULT_PROVENANCE_SCHEMA",
    "ArenaRunRequest",
    "ArenaRunResult",
    "ArenaRunner",
    "ArenaRunnerV2",
    "torus9_startset_ref",
]
