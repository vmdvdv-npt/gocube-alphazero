"""Fail-closed Arena policy declarations.

There is one production execution engine for all boards. Game-specific profiles
supply semantics only. Historical Torus9 executors remain frozen for
reproduction and cannot be selected as production engines.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


class ArenaPolicyError(RuntimeError):
    """Raised when an Arena engine is used outside its declared role."""


CANONICAL_ARENA_ENGINE = "process-central-inference-v1"
CANONICAL_ARENA_SYMBOL = "tools.arena.run_arena"
CANONICAL_ARENA_WORKERS = 16
CURRENT_PRODUCTION_ARENA_READY = True

# Active Torus9 scientific constants remain explicit and fail-closed.
CANONICAL_TORUS9_KOMI = 0.5
CANONICAL_TORUS9_ARENA_SOURCE_COMMIT = "9723bb5ac8eb28d55d21d607e3673da5bd894315"
CANONICAL_GOLDEN_PROCESS_PROOF_PR = 83

# Backward aliases for code that only reads policy constants.
CANONICAL_TORUS9_ARENA_ENGINE = CANONICAL_ARENA_ENGINE
CANONICAL_TORUS9_WORKERS = CANONICAL_ARENA_WORKERS
CURRENT_TORUS9_PRODUCTION_ARENA_READY = CURRENT_PRODUCTION_ARENA_READY

FROZEN_ARENA_OVERRIDE_FLAG = "--allow-frozen-arena"
FROZEN_ARENA_OVERRIDE_ENV = "GOCUBE_ALLOW_FROZEN_TORUS9_ARENA"


@dataclass(frozen=True)
class ArenaEnginePolicy:
    symbol: str
    role: str
    production_allowed: bool
    reason: str


ARENA_ENGINES = (
    ArenaEnginePolicy(
        symbol=CANONICAL_ARENA_SYMBOL,
        role="canonical-production",
        production_allowed=True,
        reason=(
            "One board-agnostic OS-process search engine with one central "
            "inference broker; game semantics are profile adapters."
        ),
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
        reason="Deterministic batched search primitive; not an Arena executor.",
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
        reason="Historical process proof/reference only; not production.",
    ),
)

# Compatibility name; it no longer means a separate Torus9 production engine set.
TORUS9_ARENA_ENGINES = ARENA_ENGINES


def require_current_production_ready() -> None:
    if not CURRENT_PRODUCTION_ARENA_READY:
        raise ArenaPolicyError("Production Arena is LOCKED")


def require_current_torus9_production_ready() -> None:
    require_current_production_ready()


def frozen_override_enabled() -> bool:
    return os.environ.get(FROZEN_ARENA_OVERRIDE_ENV) == "1"


def require_frozen_override(enabled: bool | None = None, *, engine: str) -> None:
    allowed = frozen_override_enabled() if enabled is None else bool(enabled)
    if not allowed:
        raise ArenaPolicyError(
            f"Arena engine {engine!r} is frozen. Use {FROZEN_ARENA_OVERRIDE_FLAG} "
            "through an explicit historical reproduction command."
        )
