"""Fail-closed Torus 9x9 Arena policy declarations."""

from __future__ import annotations

from dataclasses import dataclass
import os


class ArenaPolicyError(RuntimeError):
    """Raised when an Arena engine is used outside its declared role."""


CANONICAL_TORUS9_ARENA_ENGINE = "torus9-process-central-inference-v1"
CANONICAL_TORUS9_ARENA_SOURCE_COMMIT = "9723bb5ac8eb28d55d21d607e3673da5bd894315"
CANONICAL_GOLDEN_PROCESS_PROOF_PR = 83
CANONICAL_TORUS9_WORKERS = 16
CANONICAL_TORUS9_KOMI = 0.5
CURRENT_TORUS9_PRODUCTION_ARENA_READY = True
FROZEN_ARENA_OVERRIDE_FLAG = "--allow-frozen-arena"
FROZEN_ARENA_OVERRIDE_ENV = "GOCUBE_ALLOW_FROZEN_TORUS9_ARENA"


@dataclass(frozen=True)
class ArenaEnginePolicy:
    symbol: str
    role: str
    production_allowed: bool
    reason: str


TORUS9_ARENA_ENGINES = (
    ArenaEnginePolicy(
        symbol="tools.torus9_arena.run_arena",
        role="canonical-production",
        production_allowed=True,
        reason="16 OS CPU search workers with one parent inference broker/model owner.",
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.torus9.run_torus9_arena",
        role="frozen-process-foundation",
        production_allowed=False,
        reason="Historical process executor duplicated models per worker.",
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.torus9.run_torus9_batched_arena",
        role="frozen-degraded-logical-lane-executor",
        production_allowed=False,
        reason="Single-process logical lanes regressed real CPU search parallelism.",
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.torus9.Torus9BatchedPUCT",
        role="internal-search-primitive",
        production_allowed=False,
        reason="Internal deterministic batched search primitive; not an Arena executor.",
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.arena.SequentialGoldenArena",
        role="reference-oracle",
        production_allowed=False,
        reason="Sequential correctness oracle only.",
    ),
    ArenaEnginePolicy(
        symbol="gocube_golden.arena_process.ProcessParallelGoldenArena",
        role="process-reference",
        production_allowed=False,
        reason="Historical process proof/reference only for Torus9 production.",
    ),
)


def require_current_torus9_production_ready() -> None:
    if not CURRENT_TORUS9_PRODUCTION_ARENA_READY:
        raise ArenaPolicyError("Torus9 production Arena is LOCKED")


def frozen_override_enabled() -> bool:
    return os.environ.get(FROZEN_ARENA_OVERRIDE_ENV) == "1"


def require_frozen_override(enabled: bool | None = None, *, engine: str) -> None:
    allowed = frozen_override_enabled() if enabled is None else bool(enabled)
    if not allowed:
        raise ArenaPolicyError(
            f"Arena engine {engine!r} is frozen. Use {FROZEN_ARENA_OVERRIDE_FLAG} "
            "through an explicit historical reproduction command."
        )
