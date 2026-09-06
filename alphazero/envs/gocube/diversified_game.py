from __future__ import annotations

import math

import numpy as np

from .katago_v3 import terminal_from_state
from .pinned_game import (
    PinnedCube2JapaneseGame,
    PinnedCube3JapaneseGame,
    PinnedCube4JapaneseGame,
    PinnedCube5JapaneseGame,
    PinnedCube6JapaneseGame,
    PinnedCube7JapaneseGame,
    PinnedTorus9JapaneseGame,
    PinnedTorus13JapaneseGame,
    PinnedTorus19JapaneseGame,
)
from .game import (
    Cube2JapaneseGame,
    Cube3JapaneseGame,
    Cube4JapaneseGame,
    Cube5JapaneseGame,
    Cube6JapaneseGame,
    Cube7JapaneseGame,
    Torus9JapaneseGame,
    Torus13JapaneseGame,
    Torus19JapaneseGame,
)


PLAIN_FORK_POOL_CAPACITY = 1000


def sample_plain_fork_kind(rng, *, early_prob: float, ordinary_prob: float) -> str | None:
    """Mirror KataGo's early-first, ordinary-second fork Bernoulli ordering."""
    if rng.random_sample() < float(early_prob):
        return "early_fork"
    if rng.random_sample() < float(ordinary_prob):
        return "ordinary_fork"
    return None


def sample_fork_depth(rng, *, kind: str, move_count: int, point_count: int,
                      early_expected_move_prop: float) -> int:
    move_count = max(0, int(move_count))
    if move_count == 0:
        return 0
    if kind == "early_fork":
        depth = int(math.floor(
            rng.exponential() * float(early_expected_move_prop) * float(point_count)
        ))
    elif kind == "ordinary_fork":
        depth = int(rng.randint(0, move_count))
    else:
        raise ValueError(f"Unknown fork kind: {kind!r}")
    return max(0, min(depth, move_count - 1))


class _DiversifiedStartMixin:
    _PLAIN_FORK_POOL = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diverse_early_fork_prob = 0.0
        self._diverse_ordinary_fork_prob = 0.0
        self._diverse_early_expected_move_prop = 0.0
        self._diverse_plain_fork_pool_capacity = PLAIN_FORK_POOL_CAPACITY
        self._diverse_started_from_plain_fork = False
        self._diverse_suppress_plain_fork_generation = False
        self._diverse_train_state_history = (self._state,)
        self._diverse_training_history_offset = len(self._pinned_move_history)

    @classmethod
    def _plain_fork_pool(cls):
        if "_PLAIN_FORK_POOL" not in cls.__dict__:
            cls._PLAIN_FORK_POOL = []
        return cls._PLAIN_FORK_POOL

    def clone(self):
        clone = super().clone()
        clone._diverse_early_fork_prob = self._diverse_early_fork_prob
        clone._diverse_ordinary_fork_prob = self._diverse_ordinary_fork_prob
        clone._diverse_early_expected_move_prop = self._diverse_early_expected_move_prop
        clone._diverse_plain_fork_pool_capacity = self._diverse_plain_fork_pool_capacity
        clone._diverse_started_from_plain_fork = self._diverse_started_from_plain_fork
        clone._diverse_suppress_plain_fork_generation = self._diverse_suppress_plain_fork_generation
        clone._diverse_train_state_history = self._diverse_train_state_history
        clone._diverse_training_history_offset = self._diverse_training_history_offset
        return clone

    def configure_diversification(
        self,
        *,
        early_fork_prob: float,
        ordinary_fork_prob: float,
        early_expected_move_prop: float,
        pool_capacity: int = PLAIN_FORK_POOL_CAPACITY,
    ) -> None:
        early_fork_prob = float(early_fork_prob)
        ordinary_fork_prob = float(ordinary_fork_prob)
        early_expected_move_prop = float(early_expected_move_prop)
        pool_capacity = int(pool_capacity)
        if not 0.0 <= early_fork_prob <= 1.0:
            raise ValueError("early fork probability must be within [0,1]")
        if not 0.0 <= ordinary_fork_prob <= 1.0:
            raise ValueError("ordinary fork probability must be within [0,1]")
        if early_expected_move_prop < 0.0:
            raise ValueError("early fork expected move proportion must be non-negative")
        if pool_capacity < 1:
            raise ValueError("plain fork pool capacity must be positive")
        self._diverse_early_fork_prob = early_fork_prob
        self._diverse_ordinary_fork_prob = ordinary_fork_prob
        self._diverse_early_expected_move_prop = early_expected_move_prop
        self._diverse_plain_fork_pool_capacity = pool_capacity

    def set_plain_fork_generation_suppressed(self, suppressed: bool) -> None:
        self._diverse_suppress_plain_fork_generation = bool(suppressed)

    def mark_diversified_training_start(self) -> None:
        self._diverse_train_state_history = (self._state,)
        self._diverse_training_history_offset = len(self._pinned_move_history)
        self._diverse_suppress_plain_fork_generation = False

    def play_action(self, action: int) -> None:
        was_search_clone = bool(getattr(self, "_pinned_is_search_clone", False))
        suppress = bool(getattr(self, "_diverse_suppress_plain_fork_generation", False))
        super().play_action(action)
        if was_search_clone or suppress:
            return
        self._diverse_train_state_history = self._diverse_train_state_history + (self._state,)
        if self._state.terminal_kind is not None:
            self._maybe_store_plain_fork()

    def _maybe_store_plain_fork(self) -> None:
        if (
            self._diverse_suppress_plain_fork_generation
            or self._diverse_started_from_plain_fork
            or getattr(self, "_pinned_started_from_seki_fork", False)
            or getattr(self, "_pinned_start_phase", None) != "main"
            or getattr(self._state, "terminal_kind", None) is None
            or self._diverse_early_fork_prob <= 0.0 and self._diverse_ordinary_fork_prob <= 0.0
        ):
            return
        move_count = len(self._diverse_train_state_history) - 1
        if move_count <= 0:
            return
        kind = sample_plain_fork_kind(
            np.random,
            early_prob=self._diverse_early_fork_prob,
            ordinary_prob=self._diverse_ordinary_fork_prob,
        )
        if kind is None:
            return
        depth = sample_fork_depth(
            np.random,
            kind=kind,
            move_count=move_count,
            point_count=int(self.logical_topology().point_count),
            early_expected_move_prop=self._diverse_early_expected_move_prop,
        )
        # KataGo replays to the selected move. GoCube's V3State already contains
        # the complete ko/pass/cycle history, so retain that state directly.
        while depth > 0 and self._diverse_train_state_history[depth].phase != "main":
            depth -= 1
        candidate_state = self._diverse_train_state_history[depth]
        if candidate_state.terminal_kind is not None or candidate_state.phase != "main":
            return
        absolute_history_len = self._diverse_training_history_offset + depth
        candidate_history = tuple(self._pinned_move_history[:absolute_history_len])
        entry = (kind, candidate_state, candidate_history, int(depth))
        pool = type(self)._plain_fork_pool()
        capacity = self._diverse_plain_fork_pool_capacity
        if len(pool) < capacity:
            pool.append(entry)
        else:
            pool[int(np.random.randint(0, len(pool)))] = entry

    def maybe_start_plain_fork(self):
        pool = type(self)._plain_fork_pool()
        if not pool:
            return None
        index = int(np.random.randint(0, len(pool)))
        kind, candidate_state, candidate_history, depth = pool.pop(index)
        self._state = candidate_state
        self._terminal = terminal_from_state(candidate_state, self.logical_topology(), self.KOMI)
        self._sync_framework_fields()
        self.last_action = candidate_history[-1][1] if candidate_history else None
        self._pinned_move_history = tuple(candidate_history)
        self._pinned_state_history = (candidate_state,)
        self._pinned_start_phase = candidate_state.phase
        self._pinned_started_from_seki_fork = False
        self._diverse_started_from_plain_fork = True
        self._diverse_suppress_plain_fork_generation = True
        self._diverse_train_state_history = (candidate_state,)
        self._diverse_training_history_offset = len(candidate_history)
        return {"mode": kind, "fork_depth": int(depth)}


class DiversifiedPinnedTorus9JapaneseGame(_DiversifiedStartMixin, PinnedTorus9JapaneseGame): pass
class DiversifiedPinnedTorus13JapaneseGame(_DiversifiedStartMixin, PinnedTorus13JapaneseGame): pass
class DiversifiedPinnedTorus19JapaneseGame(_DiversifiedStartMixin, PinnedTorus19JapaneseGame): pass
class DiversifiedPinnedCube2JapaneseGame(_DiversifiedStartMixin, PinnedCube2JapaneseGame): pass
class DiversifiedPinnedCube3JapaneseGame(_DiversifiedStartMixin, PinnedCube3JapaneseGame): pass
class DiversifiedPinnedCube4JapaneseGame(_DiversifiedStartMixin, PinnedCube4JapaneseGame): pass
class DiversifiedPinnedCube5JapaneseGame(_DiversifiedStartMixin, PinnedCube5JapaneseGame): pass
class DiversifiedPinnedCube6JapaneseGame(_DiversifiedStartMixin, PinnedCube6JapaneseGame): pass
class DiversifiedPinnedCube7JapaneseGame(_DiversifiedStartMixin, PinnedCube7JapaneseGame): pass


_DIVERSIFIED_BY_BASE = {
    Torus9JapaneseGame: DiversifiedPinnedTorus9JapaneseGame,
    Torus13JapaneseGame: DiversifiedPinnedTorus13JapaneseGame,
    Torus19JapaneseGame: DiversifiedPinnedTorus19JapaneseGame,
    Cube2JapaneseGame: DiversifiedPinnedCube2JapaneseGame,
    Cube3JapaneseGame: DiversifiedPinnedCube3JapaneseGame,
    Cube4JapaneseGame: DiversifiedPinnedCube4JapaneseGame,
    Cube5JapaneseGame: DiversifiedPinnedCube5JapaneseGame,
    Cube6JapaneseGame: DiversifiedPinnedCube6JapaneseGame,
    Cube7JapaneseGame: DiversifiedPinnedCube7JapaneseGame,
}


def diversified_pinned_game_class(base_game_cls):
    try:
        return _DIVERSIFIED_BY_BASE[base_game_cls]
    except KeyError as exc:
        raise ValueError(f"No diversified pinned KataGo wrapper for {base_game_cls!r}") from exc
