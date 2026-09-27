"""Cube production path with pre-fence generation recovery.

Scientific execution stays in ``cube_production``.  This composition wrapper
only reconciles artifacts left by an interrupted, uncommitted generation
before the same generation is attempted again.
"""
from __future__ import annotations

from .cube_production import CubeProductionGenerationPath as _CubeProductionGenerationPath
from .generation_recovery import reconcile_uncommitted_generation
from .generation_runner import GenerationExecutionResult, ResolvedGenerationInput


class CubeProductionGenerationPath(_CubeProductionGenerationPath):
    """Run Cube production after removing only stale pre-fence evidence."""

    def run_generation(
        self, resolved: ResolvedGenerationInput
    ) -> GenerationExecutionResult:
        from .execution_permit import require_engine_execution
        require_engine_execution('gocube_golden/orchestrator_v2/cube_production_recovery.py', action="training", topology=None)
        reconcile_uncommitted_generation(
            root=resolved.output_lineage.root,
            lineage_id=resolved.output_lineage.lineage_id,
            generation=resolved.generation,
        )
        return super().run_generation(resolved)


__all__ = ["CubeProductionGenerationPath"]
