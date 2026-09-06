from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np

from .katago_v3 import (
    CLEANUP_1,
    CLEANUP_2,
    MAIN,
    V3State,
    _board_key,
    _state_key,
)


KATAGO_CLEANUP_TRAINING_DEFAULTS = {
    "probability": 0.04,
    "prelude_area_prop": 0.25,
    "prelude_gamma_shape": 1.0,
    "policy_temperature": 2.0 / 3.0,
}

PINNED_OBSERVATION_SCHEMA = "gocube-observation-v4-pass-would-end-phase"
PASS_WOULD_END_PHASE_CHANNEL = 17


def pass_would_end_phase(state: V3State) -> bool:
    """Return whether PASS would end the current V3 phase/game.

    This mirrors the actual V3 PASS transition instead of approximating it as
    merely ``consecutive_passes == 1``. A repeated same-player pass state can
    also end a phase under the pinned cleanup rules.
    """

    if state.terminal_kind is not None or state.phase not in (MAIN, CLEANUP_1, CLEANUP_2):
        return False
    pre_key = _state_key(state.board, state.current_player, state.ko_recap_blocked)
    pass_states = state.black_pass_states if state.current_player == 0 else state.white_pass_states
    repeated_pass_state = pre_key in pass_states
    return state.consecutive_passes + 1 >= 2 or repeated_pass_state


def apply_pass_would_end_phase_feature(game: Any, observation: np.ndarray) -> np.ndarray:
    """Fill KataGo's dedicated passWouldEndPhase input plane.

    GoCube represents global inputs as constant spatial planes. Pinned training
    expands V3 from 17 to 18 planes and reserves plane 17 for this exact bit;
    the old consecutive-pass plane 5 remains unchanged.
    """

    state = getattr(game, "semantic_state", None)
    if not isinstance(state, V3State):
        return observation
    result = np.asarray(observation).copy()
    if result.ndim < 1 or result.shape[0] <= PASS_WOULD_END_PHASE_CHANNEL:
        raise ValueError(
            "Pinned V3 observation is missing the dedicated passWouldEndPhase plane"
        )
    result[PASS_WOULD_END_PHASE_CHANNEL] = 1.0 if pass_would_end_phase(state) else 0.0
    return result


def install_pinned_observation_contract(game_cls):
    """Expand the pinned pilot observation from 17 to 18 planes, idempotently."""

    if getattr(game_cls, "_PINNED_PASS_WOULD_END_PHASE_INSTALLED", False):
        return game_cls

    original_features = int(game_cls.OBSERVATION_FEATURES)
    if original_features != PASS_WOULD_END_PHASE_CHANNEL:
        raise ValueError(
            f"Pinned passWouldEndPhase contract expected {PASS_WOULD_END_PHASE_CHANNEL} V3 planes, "
            f"got {original_features}"
        )

    original_observation = game_cls.observation
    original_rules_fingerprint = game_cls.rules_fingerprint

    def observation(self):
        # observation_size() reads the class feature count dynamically, so after
        # installing the contract the original V3 encoder allocates 18 planes
        # and leaves the new final plane free for the exact pass-end bit.
        result = original_observation(self)
        return apply_pass_would_end_phase_feature(self, result)

    def rules_fingerprint(cls):
        base = original_rules_fingerprint()
        payload = (
            f"{base}|observation={PINNED_OBSERVATION_SCHEMA}"
            f"|passWouldEndPhaseChannel={PASS_WOULD_END_PHASE_CHANNEL}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    game_cls.OBSERVATION_FEATURES = PASS_WOULD_END_PHASE_CHANNEL + 1
    game_cls.OBSERVATION_SCHEMA = PINNED_OBSERVATION_SCHEMA
    game_cls.observation = observation
    game_cls.rules_fingerprint = classmethod(rules_fingerprint)
    game_cls._PINNED_PASS_WOULD_END_PHASE_INSTALLED = True
    return game_cls


def rebase_cleanup_training_state(state: V3State, target_phase: str) -> V3State:
    """Rebase a MAIN position as a synthetic KataGo cleanup-training start.

    KataGo's cleanup-training path policy-initializes a board and then clears
    BoardHistory into encore phase 1 or 2. GoCube mirrors that by preserving the
    board and player to move while clearing history-dependent state. Captures are
    retained because KataGo stores them on Board, while GoCube stores their
    equivalent count on V3State.
    """

    if target_phase not in (CLEANUP_1, CLEANUP_2):
        raise ValueError(f"cleanup training phase must be {CLEANUP_1!r} or {CLEANUP_2!r}")
    if state.terminal_kind is not None or state.phase != MAIN:
        raise ValueError("cleanup training can only rebase a non-terminal MAIN position")

    key = _state_key(state.board, state.current_player, ())
    second_start = _board_key(state.board) if target_phase == CLEANUP_2 else None
    return replace(
        state,
        turns=0,
        consecutive_passes=0,
        captures=state.captures,
        previous_board=None,
        phase=target_phase,
        ko_recap_blocked=(),
        phase_history=(key,),
        history_since_pass=(key,),
        black_pass_states=(),
        white_pass_states=(),
        ko_capture_history=(),
        second_cleanup_start_colors=second_start,
        cleanup2_moves=(0, 0),
        main_moves=(0, 0),
        cleanup1_moves=(0, 0),
        terminal_kind=None,
        no_result_reason=None,
        pass_alive_early_end=False,
        entered_cleanup1=True,
        entered_cleanup2=target_phase == CLEANUP_2,
        cleanup_captures=0,
        ko_unblock_actions=0,
    )
