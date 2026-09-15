"""Execution-only performance references and advisory diagnostics.

This module is deliberately independent from the scientific Torus9 profile.
The values describe a validated Legion execution path; they are not part of
model/search/rules/target/replay semantics or any scientific fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys


LEGION_SELFPLAY_PERFORMANCE_DEGRADED_DELTA_PCT = -15.0


@dataclass(frozen=True)
class LegionSelfPlayPerformanceReference:
    """Validated Legion/Torus9 execution recommendation and provenance."""

    recommended_workers: int
    recommended_active_games_per_worker: int
    recommended_total_active_contexts: int
    recommended_min_games: int
    recommended_batch_cap: int
    recommended_wait_ms: float
    recommended_shared_memory: bool
    recommended_central_inference_owner: str
    recommended_device: str
    reference_moves_per_sec: float
    reference_mean_batch_rows: float
    reference_p95_batch_rows: float
    reference_commit: str
    reference_date: str
    reference_description: str

    def as_dict(self) -> dict[str, object]:
        return {
            "recommended_workers": self.recommended_workers,
            "recommended_active_games_per_worker": self.recommended_active_games_per_worker,
            "recommended_total_active_contexts": self.recommended_total_active_contexts,
            "recommended_min_games": self.recommended_min_games,
            "recommended_batch_cap": self.recommended_batch_cap,
            "recommended_wait_ms": self.recommended_wait_ms,
            "recommended_shared_memory": self.recommended_shared_memory,
            "recommended_central_inference_owner": self.recommended_central_inference_owner,
            "recommended_device": self.recommended_device,
            "reference_moves_per_sec": self.reference_moves_per_sec,
            "reference_mean_batch_rows": self.reference_mean_batch_rows,
            "reference_p95_batch_rows": self.reference_p95_batch_rows,
            "reference_commit": self.reference_commit,
            "reference_date": self.reference_date,
            "reference_description": self.reference_description,
        }


LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE = LegionSelfPlayPerformanceReference(
    recommended_workers=16,
    recommended_active_games_per_worker=4,
    recommended_total_active_contexts=64,
    recommended_min_games=64,
    recommended_batch_cap=64,
    recommended_wait_ms=1.0,
    recommended_shared_memory=True,
    recommended_central_inference_owner="parent",
    recommended_device="cuda",
    reference_moves_per_sec=21.09873,
    reference_mean_batch_rows=13.255,
    reference_p95_batch_rows=39.0,
    reference_commit="85e37cd28e203345984c58ea2f56db818cab0c2d",
    reference_date="2026-09-15",
    reference_description=(
        "PR #102 performance validation; current-main reproduction "
        "20.931 -> 21.09873 moves/s"
    ),
)


LEGION_TORUS9_SELFPLAY_UNDERFILLED_REFERENCE = {
    "games": 32,
    "requested_total_active_contexts": 64,
    "effective_context_ceiling": 32,
    "moves_per_sec": 14.92959,
    "mean_batch_rows": 9.904,
    "p50_batch_rows": 8.0,
    "p95_batch_rows": 24.0,
    "max_batch_rows": 30,
    "description": "Known Stage 7 underfilled 32-game workload",
}


@dataclass(frozen=True)
class LegionSelfPlayExecutionAssessment:
    """Effective configuration classification emitted before self-play."""

    reference: LegionSelfPlayPerformanceReference
    workers: int
    active_games_per_worker: int
    games: int
    requested_total_active_contexts: int | None
    effective_context_ceiling: int
    batch_cap: int
    wait_ms: float
    coalescing: bool
    shared_memory: bool
    central_inference_owner: str
    device: str
    status: str
    severity: str
    issues: tuple[str, ...]
    override_reason: str | None
    prompted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference.as_dict(),
            "workers": self.workers,
            "active_games_per_worker": self.active_games_per_worker,
            "games": self.games,
            "requested_total_active_contexts": self.requested_total_active_contexts,
            "effective_context_ceiling": self.effective_context_ceiling,
            "batch_cap": self.batch_cap,
            "wait_ms": self.wait_ms,
            "coalescing": self.coalescing,
            "shared_memory": self.shared_memory,
            "central_inference_owner": self.central_inference_owner,
            "device": self.device,
            "status": self.status,
            "severity": self.severity,
            "issues": list(self.issues),
            "override_reason": self.override_reason,
            "prompted": self.prompted,
            "known_underfilled_reference": dict(LEGION_TORUS9_SELFPLAY_UNDERFILLED_REFERENCE),
        }


def effective_active_context_ceiling(
    *,
    games: int,
    workers: int,
    active_games_per_worker: int,
    total_active_contexts: int | None,
) -> int:
    """Return the contexts the supplied workload can actually populate."""
    if games < 0 or workers < 0 or active_games_per_worker < 0:
        raise ValueError("Execution counts cannot be negative")
    worker_capacity = workers * active_games_per_worker
    ceiling = min(games, worker_capacity)
    if total_active_contexts is not None:
        if total_active_contexts < 0:
            raise ValueError("Total active contexts cannot be negative")
        ceiling = min(ceiling, total_active_contexts)
    return int(ceiling)


def _is_cuda(device: str) -> bool:
    return str(device).lower().startswith("cuda")


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _issues(
    *,
    reference: LegionSelfPlayPerformanceReference,
    workers: int,
    active_games_per_worker: int,
    games: int,
    requested_total_active_contexts: int | None,
    effective_context_ceiling: int,
    batch_cap: int,
    wait_ms: float,
    coalescing: bool,
    shared_memory: bool,
    central_inference_owner: str,
    device: str,
) -> tuple[tuple[str, ...], str]:
    issues: list[str] = []
    obvious: list[str] = []
    if games < reference.recommended_min_games:
        message = f"games={games} < recommended minimum {reference.recommended_min_games}"
        issues.append(message)
        obvious.append(message)
    if effective_context_ceiling < reference.recommended_total_active_contexts:
        message = (
            f"effective active context ceiling={effective_context_ceiling} < "
            f"recommended {reference.recommended_total_active_contexts}"
        )
        issues.append(message)
        obvious.append(message)
    if workers < reference.recommended_workers:
        message = f"workers={workers} < recommended {reference.recommended_workers}"
        issues.append(message)
        obvious.append(message)
    if active_games_per_worker < reference.recommended_active_games_per_worker:
        message = (
            f"active_games_per_worker={active_games_per_worker} < "
            f"recommended {reference.recommended_active_games_per_worker}"
        )
        issues.append(message)
        obvious.append(message)
    if not coalescing:
        message = "batching/coalescing is disabled"
        issues.append(message)
        obvious.append(message)
    if not shared_memory:
        message = "shared-memory execution is disabled"
        issues.append(message)
        obvious.append(message)
    if central_inference_owner != reference.recommended_central_inference_owner:
        message = (
            f"central inference owner={central_inference_owner!r}, expected "
            f"{reference.recommended_central_inference_owner!r}"
        )
        issues.append(message)
        obvious.append(message)
    if not _is_cuda(device):
        message = f"device={device!r} is not CUDA"
        issues.append(message)
        obvious.append(message)
    if batch_cap != reference.recommended_batch_cap:
        issues.append(
            f"batch_cap={batch_cap} differs from validated {reference.recommended_batch_cap}"
        )
    if not _same_float(wait_ms, reference.recommended_wait_ms):
        issues.append(
            f"wait_ms={wait_ms:g} differs from validated {reference.recommended_wait_ms:g}"
        )
    if requested_total_active_contexts is not None and requested_total_active_contexts != reference.recommended_total_active_contexts:
        issues.append(
            f"requested total_active_contexts={requested_total_active_contexts} differs from "
            f"validated {reference.recommended_total_active_contexts}"
        )
    if workers != reference.recommended_workers and workers > reference.recommended_workers:
        issues.append(f"workers={workers} is outside the validated Legion preset")
    if active_games_per_worker != reference.recommended_active_games_per_worker and active_games_per_worker > reference.recommended_active_games_per_worker:
        issues.append(
            f"active_games_per_worker={active_games_per_worker} is outside the validated Legion preset"
        )
    severity = "obvious_underfill_or_path_deviation" if obvious else ("unvalidated_tuning" if issues else "none")
    return tuple(issues), severity


def _interactive_tty() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def format_legion_selfplay_advisory(assessment: LegionSelfPlayExecutionAssessment) -> str:
    """Format the visible, non-blocking execution advisory."""
    reference = assessment.reference
    lines = [
        "PERFORMANCE ADVISORY — Legion Torus9",
        "This run differs from the validated Legion Self-play execution preset.",
        "",
        "Validated reference:",
        f"{reference.recommended_workers} workers × {reference.recommended_active_games_per_worker} active games",
        f"{reference.recommended_min_games} games / {reference.recommended_total_active_contexts} active contexts",
        f"batch cap {reference.recommended_batch_cap}",
        f"wait {reference.recommended_wait_ms:g} ms",
        "CUDA + shared memory + central parent inference",
        "",
        f"Reference: {reference.reference_moves_per_sec:.5f} moves/s",
        "",
        "Current run:",
        f"games={assessment.games}",
        f"effective active context ceiling={assessment.effective_context_ceiling}",
        f"workers={assessment.workers}",
        f"active_games_per_worker={assessment.active_games_per_worker}",
        f"batch_cap={assessment.batch_cap}, wait_ms={assessment.wait_ms:g}",
        f"device={assessment.device}, shared_memory={assessment.shared_memory}, central_owner={assessment.central_inference_owner}",
    ]
    if assessment.effective_context_ceiling < reference.recommended_total_active_contexts:
        lines.extend([
            "",
            f"Known {LEGION_TORUS9_SELFPLAY_UNDERFILLED_REFERENCE['effective_context_ceiling']}-context reference: "
            f"{LEGION_TORUS9_SELFPLAY_UNDERFILLED_REFERENCE['moves_per_sec']:.5f} moves/s",
        ])
    if assessment.issues:
        lines.extend(["", "Differences:", *(f"- {issue}" for issue in assessment.issues)])
    return "\n".join(lines)


def assess_legion_torus9_selfplay_execution(
    *,
    games: int,
    workers: int,
    active_games_per_worker: int,
    total_active_contexts: int | None,
    batch_cap: int,
    wait_ms: float,
    coalescing: bool,
    shared_memory: bool = True,
    central_inference_owner: str = "parent",
    device: str = "cuda",
    execution_override_reason: str | None = None,
    interactive: bool | None = None,
) -> LegionSelfPlayExecutionAssessment:
    """Classify effective Torus9 execution and optionally ask for a reason.

    The question is advisory only.  Enter, EOF, unavailable stdin, or any
    other input condition continues the run and records a null reason.
    """
    reference = LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE
    effective = effective_active_context_ceiling(
        games=games,
        workers=workers,
        active_games_per_worker=active_games_per_worker,
        total_active_contexts=total_active_contexts,
    )
    issues, severity = _issues(
        reference=reference,
        workers=workers,
        active_games_per_worker=active_games_per_worker,
        games=games,
        requested_total_active_contexts=total_active_contexts,
        effective_context_ceiling=effective,
        batch_cap=batch_cap,
        wait_ms=float(wait_ms),
        coalescing=coalescing,
        shared_memory=shared_memory,
        central_inference_owner=central_inference_owner,
        device=str(device),
    )
    status = "validated_recommended" if not issues else "non_recommended"
    assessment = LegionSelfPlayExecutionAssessment(
        reference=reference,
        workers=int(workers),
        active_games_per_worker=int(active_games_per_worker),
        games=int(games),
        requested_total_active_contexts=(
            None if total_active_contexts is None else int(total_active_contexts)
        ),
        effective_context_ceiling=effective,
        batch_cap=int(batch_cap),
        wait_ms=float(wait_ms),
        coalescing=bool(coalescing),
        shared_memory=bool(shared_memory),
        central_inference_owner=str(central_inference_owner),
        device=str(device),
        status=status,
        severity=severity,
        issues=issues,
        override_reason=None,
        prompted=False,
    )
    if status == "validated_recommended":
        print("Legion execution preset: VALIDATED RECOMMENDED")
        print(f"reference: {reference.reference_moves_per_sec:.5f} moves/s")
        return assessment

    print(format_legion_selfplay_advisory(assessment))
    should_prompt = _interactive_tty() if interactive is None else bool(interactive)
    reason = execution_override_reason.strip() if execution_override_reason else None
    prompted = False
    if should_prompt and execution_override_reason is None:
        prompted = True
        try:
            entered = input("Why are you using a non-recommended Legion execution configuration? [Enter to continue] ")
        except (EOFError, OSError):
            entered = ""
        reason = entered.strip() or None
    return LegionSelfPlayExecutionAssessment(
        reference=reference,
        workers=assessment.workers,
        active_games_per_worker=assessment.active_games_per_worker,
        games=assessment.games,
        requested_total_active_contexts=assessment.requested_total_active_contexts,
        effective_context_ceiling=assessment.effective_context_ceiling,
        batch_cap=assessment.batch_cap,
        wait_ms=assessment.wait_ms,
        coalescing=assessment.coalescing,
        shared_memory=assessment.shared_memory,
        central_inference_owner=assessment.central_inference_owner,
        device=assessment.device,
        status=assessment.status,
        severity=assessment.severity,
        issues=assessment.issues,
        override_reason=reason,
        prompted=prompted,
    )


def compare_legion_torus9_selfplay_performance(
    *,
    games: int,
    effective_context_ceiling: int,
    moves_per_sec: float,
    mean_batch_rows: float,
    p95_batch_rows: float,
    max_batch_rows: int,
    reference: LegionSelfPlayPerformanceReference = LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE,
) -> dict[str, object]:
    """Return diagnostic-only post-run comparison against the reference."""
    comparable = (
        int(games) >= reference.recommended_min_games
        and int(effective_context_ceiling) >= reference.recommended_total_active_contexts
    )
    if not comparable:
        return {
            "status": "NOT_COMPARABLE_UNDERFILLED",
            "reference_moves_per_sec": reference.reference_moves_per_sec,
            "actual_moves_per_sec": float(moves_per_sec),
            "delta_pct": None,
            "reference_mean_batch_rows": reference.reference_mean_batch_rows,
            "actual_mean_batch_rows": float(mean_batch_rows),
            "reference_p95_batch_rows": reference.reference_p95_batch_rows,
            "actual_p95_batch_rows": float(p95_batch_rows),
            "actual_max_batch_rows": int(max_batch_rows),
            "reason": "Full reference comparison requires at least 64 games and 64 effective contexts",
        }
    delta_pct = 100.0 * (float(moves_per_sec) / reference.reference_moves_per_sec - 1.0)
    status = (
        "PERFORMANCE_DEGRADED"
        if delta_pct <= LEGION_SELFPLAY_PERFORMANCE_DEGRADED_DELTA_PCT
        else "OK"
    )
    return {
        "status": status,
        "reference_moves_per_sec": reference.reference_moves_per_sec,
        "actual_moves_per_sec": float(moves_per_sec),
        "delta_pct": delta_pct,
        "reference_mean_batch_rows": reference.reference_mean_batch_rows,
        "actual_mean_batch_rows": float(mean_batch_rows),
        "reference_p95_batch_rows": reference.reference_p95_batch_rows,
        "actual_p95_batch_rows": float(p95_batch_rows),
        "actual_max_batch_rows": int(max_batch_rows),
        "degraded_threshold_delta_pct": LEGION_SELFPLAY_PERFORMANCE_DEGRADED_DELTA_PCT,
    }


__all__ = [
    "LEGION_SELFPLAY_PERFORMANCE_DEGRADED_DELTA_PCT",
    "LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE",
    "LEGION_TORUS9_SELFPLAY_UNDERFILLED_REFERENCE",
    "LegionSelfPlayExecutionAssessment",
    "LegionSelfPlayPerformanceReference",
    "assess_legion_torus9_selfplay_execution",
    "compare_legion_torus9_selfplay_performance",
    "effective_active_context_ceiling",
    "format_legion_selfplay_advisory",
]
