from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .katago_v3 import CLEANUP_1, CLEANUP_2, MAIN, V3State


PINNED_OBSERVATION_SCHEMA = "gocube-observation-v4-pass-would-end-phase"
PASS_WOULD_END_PHASE_CHANNEL = 17
CLEANUP_TRAINING_PROB = 0.04
CLEANUP_TRAINING_BOARD_PROP = 0.25
CLEANUP_TRAINING_POLICY_TEMPERATURE = 2.0 / 3.0


def _board_key(board: np.ndarray) -> bytes:
    return bytes(np.asarray(board, dtype=np.uint8).reshape(-1).tolist())


def _state_key(board: np.ndarray, player: int, blocked) -> bytes:
    mask = bytearray(len(board))
    for point in blocked:
        mask[int(point)] = 1
    return bytes((int(player),)) + _board_key(board) + bytes(mask)


def pass_would_end_phase(state: V3State) -> bool:
    """Exact V3 equivalent of KataGo BoardHistory::passWouldEndPhase().

    V3 phase transitions occur either after two consecutive ending passes or
    after a player passes again from the same phase state (the spight-like
    territory-scoring rule). Keep this predicate identical to _pass().
    """

    if state.terminal_kind is not None or state.phase not in (MAIN, CLEANUP_1, CLEANUP_2):
        return False
    pre_key = _state_key(state.board, state.current_player, state.ko_recap_blocked)
    pass_states = state.black_pass_states if state.current_player == 0 else state.white_pass_states
    return state.consecutive_passes + 1 >= 2 or pre_key in pass_states


def _pinned_rules_fingerprint(original_fingerprint: str) -> str:
    payload = (
        f"{original_fingerprint}|observation={PINNED_OBSERVATION_SCHEMA}"
        f"|passWouldEndPhaseChannel={PASS_WOULD_END_PHASE_CHANNEL}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def install_pinned_observation_contract(game_cls):
    """Install KataGo's passWouldEndPhase feature for the pinned pilot class.

    GoCube represents global features as constant spatial planes. The existing
    V3 observation has 17 planes, so the exact pass-phase bit is appended as
    plane 17. The installer is idempotent because build_katago_training_args()
    may be called more than once in one process.
    """

    if getattr(game_cls, "_PINNED_PASS_WOULD_END_PHASE_INSTALLED", False):
        return game_cls

    original_observation = game_cls.observation
    original_rules_fingerprint = game_cls.rules_fingerprint
    original_features = int(game_cls.OBSERVATION_FEATURES)
    if original_features != PASS_WOULD_END_PHASE_CHANNEL:
        raise ValueError(
            f"Pinned passWouldEndPhase contract expected {PASS_WOULD_END_PHASE_CHANNEL} V3 planes, "
            f"got {original_features}"
        )

    def observation(self):
        result = original_observation(self)
        if result.shape[0] != PASS_WOULD_END_PHASE_CHANNEL + 1:
            raise RuntimeError(
                "Pinned observation allocation did not include passWouldEndPhase channel"
            )
        result[PASS_WOULD_END_PHASE_CHANNEL, :, 0] = (
            1.0 if pass_would_end_phase(self.semantic_state) else 0.0
        )
        return result

    @classmethod
    def rules_fingerprint(cls):
        return _pinned_rules_fingerprint(original_rules_fingerprint())

    game_cls.OBSERVATION_FEATURES = PASS_WOULD_END_PHASE_CHANNEL + 1
    game_cls.OBSERVATION_SCHEMA = PINNED_OBSERVATION_SCHEMA
    game_cls.observation = observation
    game_cls.rules_fingerprint = rules_fingerprint
    game_cls._PINNED_PASS_WOULD_END_PHASE_INSTALLED = True
    return game_cls


def cleanup_training_game(game: Any, phase: str):
    """Rebase an initialized position into KataGo-style encore training.

    Mirrors BoardHistory::clear(board, pla, rules, encorePhase): move/pass/ko
    history is cleared, the current board/player are retained, and entering the
    second encore snapshots the starting colors. Captures are retained because
    GoCube stores them on V3State rather than on the board object.
    """

    if phase not in (CLEANUP_1, CLEANUP_2):
        raise ValueError(f"cleanup training phase must be cleanup1 or cleanup2, got {phase!r}")

    state = game.semantic_state
    topology = game.logical_topology()
    board = np.asarray(state.board, dtype=np.uint8).reshape(-1).copy()
    if board.shape != (topology.point_count,):
        raise ValueError(f"Unexpected board shape {board.shape}")
    board.flags.writeable = False
    key = _state_key(board, state.current_player, ())
    second_start = _board_key(board) if phase == CLEANUP_2 else None

    rebased = V3State(
        board=board,
        current_player=state.current_player,
        turns=0,
        consecutive_passes=0,
        captures=state.captures,
        previous_board=None,
        phase=phase,
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
        entered_cleanup2=phase == CLEANUP_2,
        cleanup_captures=0,
        ko_unblock_actions=0,
    )
    return game.__class__(_state=rebased)


def sample_cleanup_initialization_moves(point_count: int) -> int:
    """Pinned KataGo policy-init move count for gamma shape 1.0.

    selfplay8b20 leaves policyInitGammaShape unspecified, whose default is 1.0.
    KataGo therefore draws an exponential count with mean 25% of board area for
    the special cleanup-training games.
    """

    mean = float(point_count) * CLEANUP_TRAINING_BOARD_PROP
    return int(np.floor(np.random.exponential(mean)))


def sample_cleanup_phase() -> str:
    return CLEANUP_1 if np.random.randint(0, 2) == 0 else CLEANUP_2


def choose_policy_initialization_move(game: Any, policy: Any) -> int:
    """Choose a legal pure-policy initialization move at KataGo temperature 2/3."""

    probs = np.asarray(policy, dtype=np.float64).reshape(-1)
    valids = np.asarray(game.valid_moves(), dtype=np.float64).reshape(-1)
    if probs.size != valids.size:
        raise ValueError(f"policy/valid size mismatch: {probs.size} vs {valids.size}")

    weights = np.where(valids > 0, np.maximum(probs, 0.0), 0.0)
    positive = weights > 0
    if positive.any():
        weights[positive] = np.power(
            weights[positive], 1.0 / CLEANUP_TRAINING_POLICY_TEMPERATURE
        )
    total = float(weights.sum())
    if total <= 0.0:
        weights = np.where(valids > 0, 1.0, 0.0)
        total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("cleanup policy initialization found no legal move")
    weights /= total
    return int(np.random.choice(weights.size, p=weights))
