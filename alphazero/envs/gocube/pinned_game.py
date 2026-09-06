from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import numpy as np

from alphazero.Game import GameState

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
from .katago_v3 import (
    CLEANUP_2,
    NO_RESULT,
    SCORED,
    apply_v3_action,
    independent_life_analysis,
    maybe_pass_alive_early_terminal,
    pass_alive_analysis,
    terminal_from_state,
)
from .selfplay_semantics import (
    KATAGO_PINNED_SELFPLAY_DEFAULTS,
    PASS_WOULD_END_PHASE_CHANNEL,
    PINNED_OBSERVATION_SCHEMA,
    apply_pass_would_end_phase_feature,
    sample_early_fork_depth,
    sample_plain_fork_kind,
)


# =============================================================================
# ПРИНЦИПЫ ОБУЧЕНИЯ ДЕТАЛЬНО СКОПИРОВАНЫ С KATAGO.
#
# Источник истины для этой ветки — pinned KataGo commit
# f6bc4b19a1686caa2d088b56251e8c11c8be6d51. Search/self-play/endgame
# semantics ниже переносились по upstream-механике, а не придумывались как
# локальные эвристики GoCube. Отличия допускаются только там, где Cube/Torus
# topology или текущий NN contract физически требуют адаптации, и такие
# отличия должны быть явно задокументированы.
# =============================================================================


class _PinnedPassWouldEndPhaseMixin:
    OBSERVATION_FEATURES = PASS_WOULD_END_PHASE_CHANNEL + 1
    OBSERVATION_SCHEMA = PINNED_OBSERVATION_SCHEMA
    _SEKI_FORK_POOL = []
    _PLAIN_FORK_POOL = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
        self._pinned_auto_end_pass_alive = True
        self._pinned_root_prune_useless_moves = False
        self._pinned_selfplay_semantics = False
        self._pinned_seki_fork_hack_prob = 0.0
        self._pinned_is_search_clone = False
        self._pinned_at_search_root = False
        self._pinned_started_from_seki_fork = False
        self._pinned_started_from_plain_fork = False
        self._pinned_early_fork_game_prob = defaults["early_fork_game_probability"]
        self._pinned_early_fork_expected_move_prop = defaults["early_fork_expected_move_prop"]
        self._pinned_fork_game_prob = defaults["fork_game_probability"]
        self._pinned_fork_game_min_choices = defaults["fork_game_min_choices"]
        self._pinned_early_fork_game_max_choices = defaults["early_fork_game_max_choices"]
        self._pinned_fork_game_max_choices = defaults["fork_game_max_choices"]
        self._pinned_start_phase = self._state.phase
        self._pinned_move_history = ()
        self._pinned_state_history = (self._state,)

    def observation(self):
        return apply_pass_would_end_phase_feature(self, super().observation())

    @classmethod
    def rules_fingerprint(cls) -> str:
        base = super().rules_fingerprint()
        payload = (
            f"{base}|observation={PINNED_OBSERVATION_SCHEMA}"
            f"|passWouldEndPhaseChannel={PASS_WOULD_END_PHASE_CHANNEL}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _seki_pool(cls):
        if "_SEKI_FORK_POOL" not in cls.__dict__:
            cls._SEKI_FORK_POOL = []
        return cls._SEKI_FORK_POOL

    @classmethod
    def _plain_fork_pool(cls):
        if "_PLAIN_FORK_POOL" not in cls.__dict__:
            cls._PLAIN_FORK_POOL = []
        return cls._PLAIN_FORK_POOL

    def clone(self):
        clone = super().clone()
        for name in (
            "_pinned_auto_end_pass_alive",
            "_pinned_root_prune_useless_moves",
            "_pinned_selfplay_semantics",
            "_pinned_seki_fork_hack_prob",
            "_pinned_started_from_seki_fork",
            "_pinned_started_from_plain_fork",
            "_pinned_early_fork_game_prob",
            "_pinned_early_fork_expected_move_prop",
            "_pinned_fork_game_prob",
            "_pinned_fork_game_min_choices",
            "_pinned_early_fork_game_max_choices",
            "_pinned_fork_game_max_choices",
            "_pinned_start_phase",
            "_pinned_move_history",
            "_pinned_state_history",
        ):
            setattr(clone, name, getattr(self, name))
        clone._pinned_is_search_clone = True
        clone._pinned_at_search_root = True
        return clone

    def configure_pinned_selfplay(
        self,
        *,
        auto_end_pass_alive: bool,
        root_prune_useless_moves: bool,
        seki_fork_hack_prob: float,
        started_from_seki_fork: bool | None = None,
        started_from_plain_fork: bool | None = None,
        early_fork_game_prob: float | None = None,
        early_fork_expected_move_prop: float | None = None,
        fork_game_prob: float | None = None,
        fork_game_min_choices: int | None = None,
        early_fork_game_max_choices: int | None = None,
        fork_game_max_choices: int | None = None,
    ) -> None:
        defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
        self._pinned_selfplay_semantics = True
        self._pinned_auto_end_pass_alive = bool(auto_end_pass_alive)
        self._pinned_root_prune_useless_moves = bool(root_prune_useless_moves)
        self._pinned_seki_fork_hack_prob = float(seki_fork_hack_prob)
        self._pinned_early_fork_game_prob = float(
            defaults["early_fork_game_probability"] if early_fork_game_prob is None else early_fork_game_prob
        )
        self._pinned_early_fork_expected_move_prop = float(
            defaults["early_fork_expected_move_prop"] if early_fork_expected_move_prop is None else early_fork_expected_move_prop
        )
        self._pinned_fork_game_prob = float(
            defaults["fork_game_probability"] if fork_game_prob is None else fork_game_prob
        )
        self._pinned_fork_game_min_choices = int(
            defaults["fork_game_min_choices"] if fork_game_min_choices is None else fork_game_min_choices
        )
        self._pinned_early_fork_game_max_choices = int(
            defaults["early_fork_game_max_choices"] if early_fork_game_max_choices is None else early_fork_game_max_choices
        )
        self._pinned_fork_game_max_choices = int(
            defaults["fork_game_max_choices"] if fork_game_max_choices is None else fork_game_max_choices
        )
        self._pinned_is_search_clone = False
        self._pinned_at_search_root = False
        if started_from_seki_fork is not None:
            self._pinned_started_from_seki_fork = bool(started_from_seki_fork)
        if started_from_plain_fork is not None:
            self._pinned_started_from_plain_fork = bool(started_from_plain_fork)

    def pinned_selfplay_config(self) -> dict[str, object]:
        return {
            "auto_end_pass_alive": self._pinned_auto_end_pass_alive,
            "root_prune_useless_moves": self._pinned_root_prune_useless_moves,
            "seki_fork_hack_prob": self._pinned_seki_fork_hack_prob,
            "started_from_seki_fork": self._pinned_started_from_seki_fork,
            "started_from_plain_fork": self._pinned_started_from_plain_fork,
            "early_fork_game_prob": self._pinned_early_fork_game_prob,
            "early_fork_expected_move_prop": self._pinned_early_fork_expected_move_prop,
            "fork_game_prob": self._pinned_fork_game_prob,
            "fork_game_min_choices": self._pinned_fork_game_min_choices,
            "early_fork_game_max_choices": self._pinned_early_fork_game_max_choices,
            "fork_game_max_choices": self._pinned_fork_game_max_choices,
        }

    def _last_four_opponent_moves_are_passes(self) -> bool:
        history = self._pinned_move_history
        if len(history) < 7:
            return False
        opponent = 1 - int(self.player)
        pass_action = int(self.pass_action())
        last = len(history) - 1
        for offset in (0, 2, 4, 6):
            player, action = history[last - offset]
            if int(player) != opponent or int(action) != pass_action:
                return False
        return True

    def _root_pruned_valid_moves(self, valids: np.ndarray) -> np.ndarray:
        if (
            not self._pinned_root_prune_useless_moves
            or not self._pinned_is_search_clone
            or not self._pinned_at_search_root
            or not self._last_four_opponent_moves_are_passes()
        ):
            return valids

        analysis = pass_alive_analysis(self._state.board, self.logical_topology())
        safe = set(analysis.pass_alive_black_territory)
        safe.update(analysis.pass_alive_white_territory)
        for group in analysis.pass_alive_black_groups + analysis.pass_alive_white_groups:
            safe.update(group)

        result = np.asarray(valids, dtype=np.uint8).copy()
        for point in safe:
            result[int(point)] = 0
        result[int(self.pass_action())] = valids[int(self.pass_action())]
        return result

    def valid_moves(self) -> np.ndarray:
        return self._root_pruned_valid_moves(super().valid_moves())

    def play_action(self, action: int) -> None:
        player_before = int(self.player)
        GameState.play_action(self, action)
        state = apply_v3_action(self._state, int(action), self.logical_topology())

        if state.terminal_kind == NO_RESULT and state.no_result_reason == "move-cap":
            state = replace(
                state,
                phase=SCORED,
                terminal_kind=SCORED,
                no_result_reason=None,
            )

        if not self._pinned_is_search_clone and self._pinned_auto_end_pass_alive:
            state = maybe_pass_alive_early_terminal(state, self.logical_topology())
        self._state = state
        self._terminal = terminal_from_state(state, self.logical_topology(), self.KOMI)
        self._sync_framework_fields()
        self._pinned_move_history = self._pinned_move_history + ((player_before, int(action)),)
        self._pinned_at_search_root = False

        if not self._pinned_is_search_clone:
            self._pinned_state_history = self._pinned_state_history + (self._state,)
            self._maybe_store_seki_forks()
            self._maybe_store_plain_fork()

    def _has_unowned_final_spot(self) -> bool:
        analysis = independent_life_analysis(self._state.board, self.logical_topology())
        return bool(analysis.dame or analysis.seki)

    def _maybe_store_plain_fork(self) -> None:
        if (
            not self._pinned_selfplay_semantics
            or self._pinned_start_phase != "main"
            or self._pinned_started_from_seki_fork
            or self._pinned_started_from_plain_fork
            or self._state.terminal_kind is None
        ):
            return
        move_count = len(self._pinned_move_history)
        if move_count <= 0:
            return

        mode = sample_plain_fork_kind(
            np.random, self._pinned_early_fork_game_prob, self._pinned_fork_game_prob
        )
        if mode is None:
            return

        if mode == "early":
            move_idx = sample_early_fork_depth(
                np.random, int(self.logical_topology().point_count), self._pinned_early_fork_expected_move_prop
            )
        else:
            move_idx = int(np.random.randint(0, move_count))
        move_idx = max(0, min(move_count - 1, move_idx))
        candidate_state = self._pinned_state_history[move_idx]
        if candidate_state.terminal_kind is not None:
            return
        type(self)._plain_fork_pool().append((
            candidate_state,
            self._pinned_move_history[:move_idx],
            mode,
            move_idx,
        ))

    def maybe_start_plain_fork(self):
        pool = type(self)._plain_fork_pool()
        if not pool:
            return None
        index = int(np.random.randint(0, len(pool)))
        candidate_state, candidate_history, mode, depth = pool.pop(index)
        self._state = candidate_state
        self._terminal = terminal_from_state(candidate_state, self.logical_topology(), self.KOMI)
        self._sync_framework_fields()
        self.last_action = candidate_history[-1][1] if candidate_history else None
        self._pinned_move_history = tuple(candidate_history)
        self._pinned_state_history = (candidate_state,)
        self._pinned_start_phase = candidate_state.phase
        self._pinned_started_from_plain_fork = True
        self._pinned_is_search_clone = False
        self._pinned_at_search_root = False
        return {"mode": mode, "depth": int(depth)}

    def _maybe_store_seki_forks(self) -> None:
        defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
        if (
            not self._pinned_selfplay_semantics
            or self._pinned_seki_fork_hack_prob <= 0.0
            or self._pinned_started_from_seki_fork
            or self._pinned_started_from_plain_fork
            or self._pinned_start_phase == CLEANUP_2
            or self._state.terminal_kind != SCORED
            or not self._has_unowned_final_spot()
        ):
            return

        move_count = len(self._pinned_move_history)
        if move_count <= 0:
            return
        pool = type(self)._seki_pool()
        capacity = int(defaults["seki_fork_pool_capacity"])
        candidates = int(defaults["seki_fork_candidates_per_game"])
        tail_scale = float(defaults["seki_fork_tail_scale"])

        for _ in range(candidates):
            move_idx = int(math.floor(
                move_count * (1.0 - tail_scale * np.random.exponential()) - 1.0
            ))
            move_idx = max(0, min(move_count, move_idx))
            candidate_state = self._pinned_state_history[move_idx]
            if candidate_state.terminal_kind is not None:
                continue
            candidate = (candidate_state, self._pinned_move_history[:move_idx])
            if len(pool) < capacity:
                pool.append(candidate)
            else:
                pool[int(np.random.randint(0, len(pool)))] = candidate

    def maybe_start_seki_fork(self, probability: float) -> bool:
        probability = float(probability)
        pool = type(self)._seki_pool()
        if probability <= 0.0 or not pool or np.random.random_sample() >= probability:
            return False

        index = int(np.random.randint(0, len(pool)))
        candidate_state, candidate_history = pool.pop(index)
        self._state = candidate_state
        self._terminal = terminal_from_state(candidate_state, self.logical_topology(), self.KOMI)
        self._sync_framework_fields()
        self.last_action = candidate_history[-1][1] if candidate_history else None
        self._pinned_move_history = tuple(candidate_history)
        self._pinned_state_history = (candidate_state,)
        self._pinned_start_phase = candidate_state.phase
        self._pinned_started_from_seki_fork = True
        self._pinned_is_search_clone = False
        self._pinned_at_search_root = False
        return True


class PinnedTorus9JapaneseGame(_PinnedPassWouldEndPhaseMixin, Torus9JapaneseGame): pass
class PinnedTorus13JapaneseGame(_PinnedPassWouldEndPhaseMixin, Torus13JapaneseGame): pass
class PinnedTorus19JapaneseGame(_PinnedPassWouldEndPhaseMixin, Torus19JapaneseGame): pass
class PinnedCube2JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube2JapaneseGame): pass
class PinnedCube3JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube3JapaneseGame): pass
class PinnedCube4JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube4JapaneseGame): pass
class PinnedCube5JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube5JapaneseGame): pass
class PinnedCube6JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube6JapaneseGame): pass
class PinnedCube7JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube7JapaneseGame): pass


_PINNED_BY_BASE = {
    Torus9JapaneseGame: PinnedTorus9JapaneseGame,
    Torus13JapaneseGame: PinnedTorus13JapaneseGame,
    Torus19JapaneseGame: PinnedTorus19JapaneseGame,
    Cube2JapaneseGame: PinnedCube2JapaneseGame,
    Cube3JapaneseGame: PinnedCube3JapaneseGame,
    Cube4JapaneseGame: PinnedCube4JapaneseGame,
    Cube5JapaneseGame: PinnedCube5JapaneseGame,
    Cube6JapaneseGame: PinnedCube6JapaneseGame,
    Cube7JapaneseGame: PinnedCube7JapaneseGame,
}


def pinned_game_class(base_game_cls):
    try:
        return _PINNED_BY_BASE[base_game_cls]
    except KeyError as exc:
        raise ValueError(f"No pinned KataGo V3 wrapper for {base_game_cls!r}") from exc
