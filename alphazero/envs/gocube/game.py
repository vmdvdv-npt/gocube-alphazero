from __future__ import annotations

from typing import ClassVar

import numpy as np

from alphazero.Game import GameState

from .core import (
    BLACK,
    CLEANUP,
    EMPTY,
    ENDGAME,
    PLAYING,
    WHITE,
    GoState,
    Topology,
    apply_action,
    cube_topology,
    initial_state,
    torus_topology,
    valid_moves as core_valid_moves,
)
from .terminal import (
    CONSERVATIVE_AREA_ADJUDICATOR_V1,
    JAPANESE_CLEANUP_ADJUDICATOR_V2,
    TerminalAdjudication,
    conservative_area_adjudicate,
    japanese_cleanup_adjudicate,
    normalized_score_target,
    ownership_target,
)

NUM_PLAYERS = 2
LEGACY_OBSERVATION_FEATURES = 8
OBSERVATION_FEATURES = 10
DEFAULT_KOMI = 7.5
MAX_CLEANUP_STAGES = 2


class GoGame(GameState):
    """AlphaZero adapter for Japanese-like Go on Torus/Cube topology.

    New training uses Japanese territory scoring. Two passes freeze the scoring
    position and, when Benson/dead/seki proof is incomplete, enter up to two
    service cleanup stages. Cleanup stones never alter the frozen Japanese score.
    An unresolved result after the final cleanup stage is a no-result and is not
    a valid training target.
    """

    TOPOLOGY: ClassVar[Topology]
    RULESET: ClassVar[str] = "japanese"
    KOMI: ClassVar[float] = DEFAULT_KOMI
    TERMINAL_ADJUDICATOR_ID: ClassVar[str] = JAPANESE_CLEANUP_ADJUDICATOR_V2
    OBSERVATION_FEATURES: ClassVar[int] = OBSERVATION_FEATURES

    def __init__(
        self,
        _state: GoState | None = None,
        _terminal: TerminalAdjudication | None = None,
        _scoring_state: GoState | None = None,
    ):
        topology = self.logical_topology()
        state = _state if _state is not None else initial_state(topology)
        super().__init__(state.board)
        self._state = state
        self._terminal = _terminal
        self._scoring_state = _scoring_state
        self._sync_framework_fields()

    @classmethod
    def logical_topology(cls) -> Topology:
        topology = getattr(cls, "TOPOLOGY", None)
        if topology is None:
            raise TypeError("GoGame must be used through a configured concrete subclass")
        return topology

    @classmethod
    def topology_kind(cls) -> str:
        return cls.logical_topology().kind

    @classmethod
    def board_size(cls) -> int:
        return cls.logical_topology().size

    @classmethod
    def action_size(cls) -> int:
        return cls.logical_topology().action_size

    @classmethod
    def observation_size(cls) -> tuple[int, int, int]:
        return cls.OBSERVATION_FEATURES, cls.logical_topology().point_count, 1

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @classmethod
    def graph_neighbors(cls) -> tuple[tuple[int, ...], ...]:
        return cls.logical_topology().neighbors_by_index

    @classmethod
    def pass_action(cls) -> int:
        return cls.logical_topology().pass_action

    @classmethod
    def action_for_point_id(cls, point_id: str) -> int:
        return cls.logical_topology().point_index(point_id)

    @classmethod
    def point_id_for_action(cls, action: int) -> str | None:
        if action == cls.pass_action():
            return None
        if action < 0 or action >= cls.logical_topology().point_count:
            raise ValueError(f"Invalid action: {action}")
        return cls.logical_topology().point_id(action)

    @property
    def semantic_state(self) -> GoState:
        return self._state

    @property
    def scoring_state(self) -> GoState | None:
        return self._scoring_state

    @property
    def terminal_adjudication(self) -> TerminalAdjudication | None:
        return self._terminal

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self._state == other._state
            and self._terminal == other._terminal
            and self._scoring_state == other._scoring_state
        )

    def clone(self) -> "GoGame":
        clone = self.__class__(self._state, self._terminal, self._scoring_state)
        clone.last_action = self.last_action
        return clone

    def valid_moves(self) -> np.ndarray:
        return core_valid_moves(self._state, self.logical_topology())

    @staticmethod
    def _freeze_scoring_state(state: GoState) -> GoState:
        return GoState(
            board=state.board,
            current_player=state.current_player,
            turns=state.turns,
            consecutive_passes=2,
            captures=state.captures,
            previous_board=state.previous_board,
            phase=ENDGAME,
            cleanup_stage=0,
        )

    @staticmethod
    def _restart_cleanup(state: GoState) -> GoState:
        return GoState(
            board=state.board,
            current_player=state.current_player,
            turns=state.turns,
            consecutive_passes=0,
            captures=state.captures,
            previous_board=state.previous_board,
            phase=CLEANUP,
            cleanup_stage=state.cleanup_stage + 1,
        )

    def _japanese_adjudication(self, cleanup_state: GoState) -> TerminalAdjudication:
        if self._scoring_state is None:
            raise RuntimeError("Japanese cleanup reached adjudication without frozen scoring state")
        return japanese_cleanup_adjudicate(
            self._scoring_state,
            cleanup_state,
            self.logical_topology(),
            ruleset=self.RULESET,
            komi=self.KOMI,
        )

    def play_action(self, action: int) -> None:
        super().play_action(action)
        previous_phase = self._state.phase
        japanese = self.TERMINAL_ADJUDICATOR_ID == JAPANESE_CLEANUP_ADJUDICATOR_V2
        self._state = apply_action(
            self._state,
            int(action),
            self.logical_topology(),
            territory_cleanup=japanese,
        )
        self._terminal = None

        if self.TERMINAL_ADJUDICATOR_ID == CONSERVATIVE_AREA_ADJUDICATOR_V1:
            if self._state.phase == ENDGAME:
                self._terminal = conservative_area_adjudicate(
                    self._state,
                    self.logical_topology(),
                    ruleset=self.RULESET,
                    komi=self.KOMI,
                )
        elif japanese:
            if previous_phase == PLAYING and self._state.phase == CLEANUP:
                self._scoring_state = self._freeze_scoring_state(self._state)
                immediate = self._japanese_adjudication(self._scoring_state)
                if immediate.training_valid:
                    self._terminal = immediate
                    self._state = self._freeze_scoring_state(self._state)
            elif previous_phase == CLEANUP and self._state.phase == ENDGAME:
                adjudication = self._japanese_adjudication(self._state)
                if adjudication.training_valid or self._state.cleanup_stage >= MAX_CLEANUP_STAGES:
                    self._terminal = adjudication
                else:
                    self._state = self._restart_cleanup(self._state)
        else:
            raise ValueError(f"Unsupported terminal adjudicator: {self.TERMINAL_ADJUDICATOR_ID}")

        self._sync_framework_fields()

    def win_state(self) -> np.ndarray:
        result = np.zeros(NUM_PLAYERS + 1, dtype=np.uint8)
        if self._terminal is None:
            return result

        winner = self._terminal.winner
        if winner == "black":
            result[0] = 1
        elif winner == "white":
            result[1] = 1
        elif winner == "draw":
            result[2] = 1
        else:
            raise ValueError(f"Unknown terminal winner: {winner}")
        return result

    def has_training_result(self) -> bool:
        return self._terminal is not None and self._terminal.training_valid

    def training_targets(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.has_training_result() or self._terminal is None or self._scoring_state is None:
            raise ValueError("Training targets require a valid Japanese terminal result")
        return (
            normalized_score_target(self._terminal, self.logical_topology()),
            ownership_target(self._terminal, self._scoring_state, self.logical_topology()),
        )

    def is_endgame_training_state(self) -> bool:
        return self._state.phase == CLEANUP or (
            self._state.phase == PLAYING and self._state.consecutive_passes == 1
        )

    def observation(self) -> np.ndarray:
        state = self._state
        point_count = self.logical_topology().point_count
        observation = np.zeros(self.observation_size(), dtype=np.float32)

        board = state.board
        observation[0, :, 0] = board == BLACK
        observation[1, :, 0] = board == WHITE

        if state.previous_board is not None:
            observation[2, :, 0] = state.previous_board == BLACK
            observation[3, :, 0] = state.previous_board == WHITE

        observation[4, :, 0] = 1.0 if state.current_player == 0 else -1.0
        observation[5, :, 0] = 1.0 if state.consecutive_passes == 1 else 0.0
        observation[6, :, 0] = state.captures[0] / point_count
        observation[7, :, 0] = state.captures[1] / point_count
        if self.OBSERVATION_FEATURES > LEGACY_OBSERVATION_FEATURES:
            observation[8, :, 0] = 1.0 if state.phase == CLEANUP else 0.0
            observation[9, :, 0] = state.cleanup_stage / MAX_CLEANUP_STAGES
        return observation

    def _sync_framework_fields(self) -> None:
        self._board = self._state.board
        self._player = self._state.current_player
        self._turns = self._state.turns


class _LegacyChineseGoGame(GoGame):
    RULESET = "chinese"
    TERMINAL_ADJUDICATOR_ID = CONSERVATIVE_AREA_ADJUDICATOR_V1
    OBSERVATION_FEATURES = LEGACY_OBSERVATION_FEATURES

    def training_targets(self):
        raise ValueError("Legacy Chinese V1 checkpoints do not have auxiliary GoCube targets")


class Torus9JapaneseGame(GoGame):
    TOPOLOGY = torus_topology(9)


class Torus13JapaneseGame(GoGame):
    TOPOLOGY = torus_topology(13)


class Torus19JapaneseGame(GoGame):
    TOPOLOGY = torus_topology(19)


class Cube2JapaneseGame(GoGame):
    TOPOLOGY = cube_topology(2)


class Cube3JapaneseGame(GoGame):
    TOPOLOGY = cube_topology(3)


class Cube4JapaneseGame(GoGame):
    TOPOLOGY = cube_topology(4)


class Cube5JapaneseGame(GoGame):
    TOPOLOGY = cube_topology(5)


class Cube6JapaneseGame(GoGame):
    TOPOLOGY = cube_topology(6)


class Cube7JapaneseGame(GoGame):
    TOPOLOGY = cube_topology(7)


class Torus9ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = torus_topology(9)


class Torus13ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = torus_topology(13)


class Torus19ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = torus_topology(19)


class Cube2ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = cube_topology(2)


class Cube3ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = cube_topology(3)


class Cube4ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = cube_topology(4)


class Cube5ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = cube_topology(5)


class Cube6ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = cube_topology(6)


class Cube7ChineseGame(_LegacyChineseGoGame):
    TOPOLOGY = cube_topology(7)


SUPPORTED_JAPANESE_GAMES = {
    ("torus", 9): Torus9JapaneseGame,
    ("torus", 13): Torus13JapaneseGame,
    ("torus", 19): Torus19JapaneseGame,
    ("cube", 2): Cube2JapaneseGame,
    ("cube", 3): Cube3JapaneseGame,
    ("cube", 4): Cube4JapaneseGame,
    ("cube", 5): Cube5JapaneseGame,
    ("cube", 6): Cube6JapaneseGame,
    ("cube", 7): Cube7JapaneseGame,
}

SUPPORTED_CHINESE_GAMES = {
    ("torus", 9): Torus9ChineseGame,
    ("torus", 13): Torus13ChineseGame,
    ("torus", 19): Torus19ChineseGame,
    ("cube", 2): Cube2ChineseGame,
    ("cube", 3): Cube3ChineseGame,
    ("cube", 4): Cube4ChineseGame,
    ("cube", 5): Cube5ChineseGame,
    ("cube", 6): Cube6ChineseGame,
    ("cube", 7): Cube7ChineseGame,
}


def game_class(topology_kind: str, size: int, rule_set: str = "japanese"):
    if rule_set == "japanese":
        games = SUPPORTED_JAPANESE_GAMES
    elif rule_set == "chinese":
        games = SUPPORTED_CHINESE_GAMES
    else:
        raise ValueError(f"Unsupported GoCube ruleset: {rule_set!r}")
    try:
        return games[(topology_kind, size)]
    except KeyError as exc:
        raise ValueError(
            f"No enabled {rule_set} self-play game for topology={topology_kind!r}, size={size}"
        ) from exc
