from __future__ import annotations

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
    _v3_observation_from_state,
)
from .katago_v3 import (
    CLEANUP_2,
    SCORED,
    apply_v3_action,
    independent_life_analysis,
    maybe_pass_alive_early_terminal,
    pass_alive_analysis,
    terminal_from_state,
    episode_move_limit as production_episode_move_limit,
)
from .selfplay_semantics import (
    KATAGO_PINNED_SELFPLAY_DEFAULTS,
    PASS_WOULD_END_PHASE_CHANNEL,
    PINNED_OBSERVATION_SCHEMA,
    finalize_episode_due_to_runtime_limit,
    pass_would_end_phase,
    resolve_episode_move_limit,
)
from .structural import (
    STRUCTURAL_FEATURE_CHANNELS,
    STRUCTURAL_FEATURE_SCHEMA,
    structural_feature_matrix,
)


G1_OBSERVATION_SCHEMA = "gocube-observation-v5-structural-features"
G1_NETWORK_ARCHITECTURE_ID = "gocube-graph-structural-v1"
BASELINE_NETWORK_ARCHITECTURE_ID = "gocube-graph-v1"


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
    GOCUBE_SEMANTIC_GAME_VARIANT = "pinned"
    OBSERVATION_FEATURES = PASS_WOULD_END_PHASE_CHANNEL + 1
    PINNED_OBSERVATION_FEATURES = PASS_WOULD_END_PHASE_CHANNEL + 1
    OBSERVATION_SCHEMA = PINNED_OBSERVATION_SCHEMA
    _SEKI_FORK_POOL = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pinned_auto_end_pass_alive = True
        self._pinned_root_prune_useless_moves = False
        self._pinned_selfplay_semantics = False
        self._pinned_seki_fork_hack_prob = 0.0
        self._pinned_is_search_clone = False
        self._pinned_at_search_root = False
        self._pinned_started_from_seki_fork = False
        self._pinned_start_phase = self._state.phase
        self._pinned_move_history = ()
        self._pinned_state_history = (self._state,)
        self._pinned_state_history_offset = 0
        self._pinned_episode_move_count = 0
        self._pinned_episode_type = "ordinary"
        self._pinned_episode_limit_override = None

    def observation(self):
        return type(self).observation_from_semantic_state(self._state)

    @classmethod
    def observation_from_semantic_state(cls, state):
        """Build the pinned 18-plane view from semantic state only."""

        result = np.zeros(
            (cls.PINNED_OBSERVATION_FEATURES, cls.logical_topology().point_count, 1),
            dtype=np.float32,
        )
        result[:PASS_WOULD_END_PHASE_CHANNEL] = _v3_observation_from_state(
            state, cls.logical_topology()
        )
        result[PASS_WOULD_END_PHASE_CHANNEL] = (
            1.0 if pass_would_end_phase(state) else 0.0
        )
        return result

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
        # Keep pools topology-specific within each self-play worker process.
        if "_SEKI_FORK_POOL" not in cls.__dict__:
            cls._SEKI_FORK_POOL = []
        return cls._SEKI_FORK_POOL

    def clone(self):
        clone = super().clone()
        clone._pinned_auto_end_pass_alive = self._pinned_auto_end_pass_alive
        clone._pinned_root_prune_useless_moves = self._pinned_root_prune_useless_moves
        clone._pinned_selfplay_semantics = self._pinned_selfplay_semantics
        clone._pinned_seki_fork_hack_prob = self._pinned_seki_fork_hack_prob
        clone._pinned_started_from_seki_fork = self._pinned_started_from_seki_fork
        clone._pinned_start_phase = self._pinned_start_phase
        clone._pinned_move_history = self._pinned_move_history
        clone._pinned_state_history = self._pinned_state_history
        clone._pinned_state_history_offset = self._pinned_state_history_offset
        clone._pinned_episode_move_count = self._pinned_episode_move_count
        clone._pinned_episode_type = self._pinned_episode_type
        clone._pinned_episode_limit_override = self._pinned_episode_limit_override
        # MCTS clones are search states. KataGo does not call the game-level
        # all-pass-alive auto-terminal inside search. Root pruning is also root-only.
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
    ) -> None:
        self._pinned_selfplay_semantics = True
        self._pinned_auto_end_pass_alive = bool(auto_end_pass_alive)
        self._pinned_root_prune_useless_moves = bool(root_prune_useless_moves)
        self._pinned_seki_fork_hack_prob = float(seki_fork_hack_prob)
        self._pinned_is_search_clone = False
        self._pinned_at_search_root = False
        if started_from_seki_fork is not None:
            self._pinned_started_from_seki_fork = bool(started_from_seki_fork)

    def pinned_selfplay_config(self) -> dict[str, object]:
        return {
            "auto_end_pass_alive": self._pinned_auto_end_pass_alive,
            "root_prune_useless_moves": self._pinned_root_prune_useless_moves,
            "seki_fork_hack_prob": self._pinned_seki_fork_hack_prob,
            "started_from_seki_fork": self._pinned_started_from_seki_fork,
        }

    @property
    def episode_move_count(self) -> int:
        """Number of actions in the current runner episode.

        This is separate from ``V3State.turns`` because a fork can preserve
        formal history and cleanup training deliberately rebases it.
        """

        return int(self._pinned_episode_move_count)

    @property
    def episode_type(self) -> str:
        return str(self._pinned_episode_type)

    @property
    def episode_move_limit(self) -> int:
        if self._pinned_episode_limit_override is not None:
            return int(self._pinned_episode_limit_override)
        return production_episode_move_limit(self.logical_topology())

    def _reset_episode_runtime(self, *, episode_type: str | None = None) -> None:
        self._pinned_episode_move_count = 0
        self._pinned_episode_limit_override = None
        if episode_type is not None:
            self._pinned_episode_type = str(episode_type)

    def mark_synthetic_cleanup_episode(self) -> None:
        self._reset_episode_runtime(episode_type="synthetic_cleanup")

    def finalize_episode_due_to_runtime_limit(self, episode_limit: int | None = None) -> bool:
        """Apply runner force-score semantics to this real game only."""

        if self._pinned_is_search_clone:
            return False
        if episode_limit is not None:
            self._pinned_episode_limit_override = resolve_episode_move_limit(
                self.logical_topology(), episode_limit
            )
        next_state = finalize_episode_due_to_runtime_limit(
            self._state,
            self.logical_topology(),
            episode_move_count=self.episode_move_count,
            episode_limit=episode_limit,
        )
        if next_state is self._state:
            return False
        self._state = next_state
        self._terminal = terminal_from_state(next_state, self.logical_topology(), self.KOMI)
        self._sync_framework_fields()
        self._pinned_at_search_root = False
        return True

    def _assert_pinned_history_alignment(self) -> None:
        """Check the compact state-history segment used by fork sampling.

        The check is intentionally limited to the real-game history path; MCTS
        clones do not append to this history and should not pay for a deep
        consistency check on every search action.
        """

        offset = int(self._pinned_state_history_offset)
        move_history_length = len(self._pinned_move_history)
        state_history_length = len(self._pinned_state_history)
        if offset < 0 or offset > move_history_length:
            raise RuntimeError(
                "Invalid pinned state-history offset: "
                f"offset={offset}, state_history_length={state_history_length}, "
                f"move_history_length={move_history_length}"
            )
        expected_states = move_history_length - offset + 1
        if state_history_length != expected_states:
            raise RuntimeError(
                "Pinned state/move history length mismatch: "
                f"offset={offset}, state_history_length={state_history_length}, "
                f"move_history_length={move_history_length}, expected_state_history_length={expected_states}"
            )

    def _pinned_state_for_history_len(self, absolute_history_len):
        """Return the saved V3 state for an absolute move-history prefix length."""

        absolute_history_len = int(absolute_history_len)
        offset = int(self._pinned_state_history_offset)
        state_history_length = len(self._pinned_state_history)
        move_history_length = len(self._pinned_move_history)
        local_state_index = absolute_history_len - offset
        if (
            absolute_history_len < 0
            or absolute_history_len > move_history_length
            or local_state_index < 0
            or local_state_index >= state_history_length
        ):
            raise RuntimeError(
                "Pinned state-history lookup is out of range: "
                f"requested_absolute_history_len={absolute_history_len}, "
                f"state_history_offset={offset}, "
                f"state_history_length={state_history_length}, "
                f"move_history_length={move_history_length}, "
                f"local_state_index={local_state_index}"
            )
        return self._pinned_state_history[local_state_index]

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
        # PASS is never pruned by KataGo's rootPruneUselessMoves condition.
        result[int(self.pass_action())] = valids[int(self.pass_action())]
        return result

    def valid_moves(self) -> np.ndarray:
        return self._root_pruned_valid_moves(super().valid_moves())

    def play_action(self, action: int) -> None:
        player_before = int(self.player)
        GameState.play_action(self, action)
        state = apply_v3_action(self._state, int(action), self.logical_topology())

        if not self._pinned_is_search_clone and self._pinned_auto_end_pass_alive:
            state = maybe_pass_alive_early_terminal(state, self.logical_topology())
        self._state = state
        self._terminal = terminal_from_state(state, self.logical_topology(), self.KOMI)
        self._sync_framework_fields()
        self._pinned_move_history = self._pinned_move_history + ((player_before, int(action)),)
        self._pinned_at_search_root = False
        if not self._pinned_is_search_clone:
            self._pinned_episode_move_count += 1

        if not self._pinned_is_search_clone:
            self._pinned_state_history = self._pinned_state_history + (self._state,)
            self._assert_pinned_history_alignment()
            self._maybe_store_seki_forks()

    def _has_unowned_final_spot(self) -> bool:
        analysis = independent_life_analysis(self._state.board, self.logical_topology())
        return bool(analysis.dame or analysis.seki)

    def _maybe_store_seki_forks(self) -> None:
        defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
        if (
            not self._pinned_selfplay_semantics
            or self._pinned_seki_fork_hack_prob <= 0.0
            or self._pinned_started_from_seki_fork
            or self._pinned_start_phase == CLEANUP_2
            or self._state.terminal_kind != SCORED
            or not self._has_unowned_final_spot()
        ):
            return

        self._assert_pinned_history_alignment()
        segment_start = int(self._pinned_state_history_offset)
        segment_end = len(self._pinned_move_history)
        segment_move_count = segment_end - segment_start
        if segment_move_count != len(self._pinned_state_history) - 1:
            raise RuntimeError(
                "Seki fork state-history segment mismatch: "
                f"segment_start={segment_start}, segment_end={segment_end}, "
                f"segment_move_count={segment_move_count}, "
                f"state_history_length={len(self._pinned_state_history)}"
            )
        if segment_move_count <= 0:
            return
        pool = type(self)._seki_pool()
        capacity = int(defaults["seki_fork_pool_capacity"])
        candidates = int(defaults["seki_fork_candidates_per_game"])
        tail_scale = float(defaults["seki_fork_tail_scale"])

        for _ in range(candidates):
            local_state_index = int(math.floor(
                segment_move_count * (1.0 - tail_scale * np.random.exponential()) - 1.0
            ))
            local_state_index = max(0, min(segment_move_count, local_state_index))
            absolute_history_len = segment_start + local_state_index
            candidate_state = self._pinned_state_for_history_len(absolute_history_len)
            if candidate_state.terminal_kind is not None:
                continue
            candidate_history = self._pinned_move_history[:absolute_history_len]
            candidate = (candidate_state, candidate_history)
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
        self._pinned_state_history_offset = len(candidate_history)
        self._reset_episode_runtime(episode_type="fork")
        self._assert_pinned_history_alignment()
        self._pinned_start_phase = candidate_state.phase
        self._pinned_started_from_seki_fork = True
        self._pinned_is_search_clone = False
        self._pinned_at_search_root = False
        assert self.semantic_state == candidate_state
        assert self.player == candidate_state.current_player
        assert self.last_action == (candidate_history[-1][1] if candidate_history else None)
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


class _StructuralObservationMixin:
    """Append topology-only structural channels after pinned V3 channels."""

    STRUCTURAL_FEATURE_SCHEMA = STRUCTURAL_FEATURE_SCHEMA
    STRUCTURAL_FEATURE_CHANNELS = STRUCTURAL_FEATURE_CHANNELS

    @classmethod
    def observation_size(cls) -> tuple[int, int, int]:
        return (
            int(cls._G1_BASE_OBSERVATION_FEATURES) + STRUCTURAL_FEATURE_CHANNELS,
            cls.logical_topology().point_count,
            1,
        )

    def observation(self):
        return type(self).observation_from_semantic_state(self._state)

    @classmethod
    def observation_from_semantic_state(cls, state):
        result = super().observation_from_semantic_state(state)
        structural = structural_feature_matrix(cls.logical_topology())
        expected_shape = (STRUCTURAL_FEATURE_CHANNELS, cls.logical_topology().point_count, 1)
        if structural.shape != expected_shape or result.shape[0] != int(
            cls._G1_BASE_OBSERVATION_FEATURES
        ):
            raise RuntimeError("G1 structural observation shape mismatch")
        return np.concatenate((result, structural), axis=0)


_STRUCTURAL_PINNED_BY_BASE = {}


def structural_pinned_game_class(base_game_cls):
    """Return the stable pinned V3 class carrying the G1 channels."""

    try:
        return _STRUCTURAL_PINNED_BY_BASE[base_game_cls]
    except KeyError:
        pinned = pinned_game_class(base_game_cls)
        topology = base_game_cls.logical_topology()
        name = f"G1Pinned{base_game_cls.__name__}"
        result = type(
            name,
            (_StructuralObservationMixin, pinned),
            {
                "__module__": __name__,
                "_G1_BASE_OBSERVATION_FEATURES": int(pinned.OBSERVATION_FEATURES),
                "OBSERVATION_FEATURES": int(pinned.OBSERVATION_FEATURES) + STRUCTURAL_FEATURE_CHANNELS,
                "OBSERVATION_SCHEMA": G1_OBSERVATION_SCHEMA,
                "GOCUBE_NETWORK_ARCHITECTURE_ID": G1_NETWORK_ARCHITECTURE_ID,
                "GOCUBE_MODEL_PROFILE": "g1",
                "GOCUBE_GAME_CLASS_ID": (
                    f"gocube-g1-pinned-{topology.kind}-{int(topology.size)}"
                ),
            },
        )
        # Dynamic classes are intentionally cached, but they also need a
        # module-level binding so multiprocessing can pickle game instances.
        globals()[name] = result
        _STRUCTURAL_PINNED_BY_BASE[base_game_cls] = result
        return result


# Explicitly named aliases make the new contract discoverable without making
# historical pinned classes change their observation schema.
g1_pinned_game_class = structural_pinned_game_class

# Materialize the supported variants at import time so they are importable in
# multiprocessing spawn mode as well as fork mode.
for _base_game_cls in _PINNED_BY_BASE:
    structural_pinned_game_class(_base_game_cls)
