"""Opt-in operation counters for Golden rules/search performance diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Iterator


@dataclass
class GoldenOperationStats:
    """Counters are deliberately opt-in so production self-play pays no cost."""

    legal_actions_calls: int = 0
    legal_calculations: int = 0
    action_probe_calls: int = 0
    successful_action_probes: int = 0
    apply_action_calls: int = 0
    validated_state_constructions: int = 0
    trusted_state_constructions: int = 0
    full_history_validations: int = 0
    leaf_expansions: int = 0
    nn_forwards: int = 0
    observation_builds: int = 0
    action_mask_builds: int = 0
    root_noise_legal_reuses: int = 0
    root_noise_legal_scans: int = 0
    searches: int = 0
    simulations: int = 0

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


_ACTIVE_STATS: ContextVar[GoldenOperationStats | None] = ContextVar(
    "golden_operation_stats", default=None
)


def increment(counter: str, amount: int = 1) -> None:
    stats = _ACTIVE_STATS.get()
    if stats is not None:
        setattr(stats, counter, int(getattr(stats, counter)) + int(amount))


@contextmanager
def operation_stats() -> Iterator[GoldenOperationStats]:
    """Collect counters for the enclosed operation and restore the prior scope."""

    stats = GoldenOperationStats()
    token = _ACTIVE_STATS.set(stats)
    try:
        yield stats
    finally:
        _ACTIVE_STATS.reset(token)
