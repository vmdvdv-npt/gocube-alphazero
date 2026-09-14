"""Fail-closed policy for Torus 9x9 Arena execution.

The repository historically accumulated several Arena engines.  This module
declares one process-parallel foundation and prevents the currently degraded
logical-lane executor from being treated as production.
"""

from __future__ import annotations

from dataclasses import dataclass


class ArenaPolicyError(RuntimeError):
    """Raised when an Arena engine is used outside its declared role."""


CANONICAL_TORUS9_ARENA_ENGINE = "torus9-process-parallel-foundation-v1"
CANONICAL_TORUS9_ARENA_SOURCE_COMMIT = "9723bb5ac8eb28d55d21d607e3673da5bd894315"
CANONICAL_GOLDEN_PROCESS_PROOF_PR = 83
CANONICAL_TORUS9_WORKERS = 16
CANONICAL_TORUS9_KOMI = 0.5
CURRENT_TORUS9_PRODUCTION_ARENA_READY = False

FROZEN_ARENA_OVERRIDE_FLAG = "--allow-frozen-arena"


@dataclass(frozen=True)
class ArenaEnginePolicy:
    symbol: str
    role: str
    production_allowed: bool
    reason: str


TORUS9_ARENA_ENGINES = (
    ArenaEnginePolicy(
        symbol="gocube_golden.torus9.run_torus9_arena",
        role="canonical-process-foundation",
        production_allowed=False,
        reason=(
            "Historical Torus9 16-process foundation. Reuse its process execution "
            "model, but current 80x8 production remains closed until one central "
            "CUDA inference broker and parity/performance gates are proven."
        ),
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.torus9.run_torus9_batched_arena",
        role="frozen-degraded-logical-lane-executor",
        production_allowed=False,
        reason="Single-process logical lanes regressed real CPU MCTS parallelism.",
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.torus9.Torus9BatchedPUCT",
        role="reference-only-search-equivalence",
        production_allowed=False,
        reason="Useful for semantic parity tests, not a production Arena executor.",
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.arena.SequentialGoldenArena",
        role="reference-oracle",
        production_allowed=False,
        reason="Sequential correctness oracle only.",
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.arena_process.ProcessParallelGoldenArena",
        role="proven-process-reference",
        production_allowed=False,
        reason="PR83 process proof; current Torus9 still needs central GPU brokering.",
    ),
)


def require_current_torus9_production_ready() -> None:
    """Fail closed until the process+central-broker Arena is proven."""
    if not CURRENT_TORUS9_PRODUCTION_ARENA_READY:
        raise ArenaPolicyError(
            "Torus9 production Arena is LOCKED. The canonical foundation is the "
            "historical 16-OS-process Arena, but current 80x8 production must not "
            "start until 16 CPU MCTS workers + one central CUDA inference broker "
            "pass semantic parity, Legion utilization, and batching gates."
        )


def require_frozen_override(enabled: bool, *, engine: str) -> None:
    """Require an explicit operator acknowledgement for frozen executors."""
    if not enabled:
        raise ArenaPolicyError(
            f"Arena engine {engine!r} is frozen. Re-run only for deliberate "
            f"reproduction with {FROZEN_ARENA_OVERRIDE_FLAG}; never use it as "
            "current Golden production evidence."
        )
