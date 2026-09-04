from __future__ import annotations

from typing import ClassVar

import numpy as np

from alphazero.Game import GameState

from .core import (
    BLACK,
    EMPTY,
    ENDGAME,
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
    TerminalAdjudication,
    conservative_area_adjudicate,
)

NUM_PLAYERS = 2
OBSERVATION_FEATURES = 8
DEFAULT_KOMI = 7.5


class GoGame(GameState):
    """AlphaZero-framework adapter for one immutable GoCube environment config.

    Concrete subclasses bind topology/size/komi statically because the upstream
    framework requires static action/observation sizes and constructs games with
    a zero-argument constructor in self-play workers.
    """

    TOPOLOGY: ClassVar[Topology]
    RULESET: ClassVar[str] = "chinese"
    KOMI: ClassVar[float] = DEFAULT_KOMI
    TERMINAL_ADJUDICATOR_ID: ClassVar[str] = CONSERVATIVE_AREA_ADJUDICATOR_V1

    def __init__(
        self,
        _state: GoState | None = None,
        _terminal: TerminalAdjudication | None = None,
    ):
        topology = self.logical_topology()
        state = _state if _state is not None else initial_state(topology)
        super().__init__(state.board)
        self._state = state
        self._terminal = _terminal
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
        # Graph observations use the framework's C x W x H tensor contract with
        # W = number of logical points and a singleton H dimension.
        return OBSERVATION_FEATURES, cls.logical_topology().point_count, 1

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
    def terminal_adjudication(self) -> TerminalAdjudication | None:
        return self._terminal

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self._state == other._state
            and self._terminal == other._terminal
        )

    def clone(self) -> "GoGame":
        clone = self.__class__(self._state, self._terminal)
        clone.last_action = self.last_action
        return clone

    def valid_moves(self) -> np.ndarray:
        return core_valid_moves(self._state, self.logical_topology())

    def play_action(self, action: int) -> None:
        super().play_action(action)
        self._state = apply_action(self._state, int(action), self.logical_topology())
        self._terminal = None

        if self._state.phase == ENDGAME:
            if self.TERMINAL_ADJUDICATOR_ID != CONSERVATIVE_AREA_ADJUDICATOR_V1:
                raise ValueError(
                    f"Unsupported terminal adjudicator: {self.TERMINAL_ADJUDICATOR_ID}"
                )
            self._terminal = conservative_area_adjudicate(
                self._state,
                self.logical_topology(),
                ruleset=self.RULESET,
                komi=self.KOMI,
            )

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

        # A signed constant plane preserves side-to-move without privileging one
        # point in the graph representation.
        observation[4, :, 0] = 1.0 if state.current_player == 0 else -1.0
        observation[5, :, 0] = 1.0 if state.consecutive_passes == 1 else 0.0
        observation[6, :, 0] = state.captures[0] / point_count
        observation[7, :, 0] = state.captures[1] / point_count
        return observation

    def _sync_framework_fields(self) -> None:
        self._board = self._state.board
        self._player = self._state.current_player
        self._turns = self._state.turns


class Torus9ChineseGame(GoGame):
    TOPOLOGY = torus_topology(9)


class Torus13ChineseGame(GoGame):
    TOPOLOGY = torus_topology(13)


class Torus19ChineseGame(GoGame):
    TOPOLOGY = torus_topology(19)


class Cube2ChineseGame(GoGame):
    TOPOLOGY = cube_topology(2)


class Cube3ChineseGame(GoGame):
    TOPOLOGY = cube_topology(3)


class Cube4ChineseGame(GoGame):
    TOPOLOGY = cube_topology(4)


class Cube5ChineseGame(GoGame):
    TOPOLOGY = cube_topology(5)


class Cube6ChineseGame(GoGame):
    TOPOLOGY = cube_topology(6)


class Cube7ChineseGame(GoGame):
    TOPOLOGY = cube_topology(7)


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


def game_class(topology_kind: str, size: int):
    try:
        return SUPPORTED_CHINESE_GAMES[(topology_kind, size)]
    except KeyError as exc:
        raise ValueError(
            f"No enabled Chinese self-play game for topology={topology_kind!r}, size={size}"
        ) from exc
