"""Single Torus9 Arena policy facade.

This tools-layer module is the only supported entry point for future Torus9
production Arena execution. The historical process Arena is exposed only as an
explicit foundation/reproduction path until central GPU brokering is restored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from tools.torus9_arena_policy import (
    CANONICAL_TORUS9_ARENA_ENGINE,
    CANONICAL_TORUS9_ARENA_SOURCE_COMMIT,
    CANONICAL_TORUS9_KOMI,
    CANONICAL_TORUS9_WORKERS,
    ArenaPolicyError,
    require_current_torus9_production_ready,
)


def run_torus9_process_foundation(
    *,
    run_id: str,
    comparison: str,
    candidate_path: Path,
    reference_path: Path,
    candidate_label: str,
    reference_label: str,
    starts: Sequence[Mapping[str, object]],
    master_seed: int,
    output_dir: Path,
    workers: int = CANONICAL_TORUS9_WORKERS,
    device: str | torch.device = "cpu",
    acknowledge_foundation_only: bool = False,
) -> dict[str, object]:
    """Run the historical real-process Torus9 Arena for controlled reproduction."""
    if not acknowledge_foundation_only:
        raise ArenaPolicyError(
            "Historical Torus9 process Arena requires "
            "acknowledge_foundation_only=True."
        )
    if workers != CANONICAL_TORUS9_WORKERS:
        raise ArenaPolicyError(
            f"Process foundation requires exactly {CANONICAL_TORUS9_WORKERS} OS workers."
        )
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise ArenaPolicyError(
            "Process foundation is CPU-only: per-worker CUDA model copies are forbidden. "
            "Current production must use one central CUDA inference broker."
        )

    from gocube_golden.torus9 import run_torus9_arena

    summary = run_torus9_arena(
        run_id=run_id,
        comparison=comparison,
        candidate_path=candidate_path,
        reference_path=reference_path,
        candidate_label=candidate_label,
        reference_label=reference_label,
        starts=starts,
        master_seed=master_seed,
        output_dir=output_dir,
        workers=workers,
        device=resolved,
    )
    enriched = dict(summary)
    enriched.update(
        {
            "arena_engine": CANONICAL_TORUS9_ARENA_ENGINE,
            "arena_foundation_source_commit": CANONICAL_TORUS9_ARENA_SOURCE_COMMIT,
            "arena_role": "FOUNDATION_ONLY_NOT_CURRENT_GOLDEN_PRODUCTION",
            "workers": CANONICAL_TORUS9_WORKERS,
            "komi": CANONICAL_TORUS9_KOMI,
        }
    )
    return enriched


def run_current_torus9_production_arena(**_: object) -> dict[str, object]:
    """The sole production API. It remains deliberately fail-closed."""
    require_current_torus9_production_ready()
    raise AssertionError("unreachable until a production Arena implementation is approved")
