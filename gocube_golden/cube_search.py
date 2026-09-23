"""Cube V2 search-position and rules adapter.

The Golden ``GoldenState`` remains the only rules state.  ``CubeSearchPosition``
adds the bounded real-move observation context required by CubeGraphNetV2 and
is otherwise transparent to the shared PUCT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cube_family import CubeFamilyTopology
from .cube_game_contract_v2 import (
    action_index_to_rules_action,
    rules_action_to_action_index,
)
from .cube_observation_v2 import (
    CubeObservationContext,
    advance_cube_observation_context,
)
from .rules import LegalActionContext
from .search_adapter import GoldenSearchAdapter
from .state import GoldenState, PASS


@dataclass(frozen=True)
class CubeSearchPosition:
    """Immutable rules state plus the bounded history seen by the network."""

    game_state: GoldenState
    observation_context: CubeObservationContext

    def __post_init__(self) -> None:
        if not isinstance(self.game_state, GoldenState):
            raise TypeError("CubeSearchPosition requires a GoldenState")
        if not isinstance(self.game_state.topology, CubeFamilyTopology):
            raise ValueError("CubeSearchPosition requires CubeFamilyTopology")
        if not isinstance(self.observation_context, CubeObservationContext):
            raise TypeError("CubeSearchPosition requires CubeObservationContext")
        if self.observation_context.size != self.game_state.topology.size:
            raise ValueError("Cube search position/context size mismatch")
        if self.observation_context.current_board != tuple(int(value) for value in self.game_state.stones):
            raise ValueError("Cube search position/context board mismatch")

    @property
    def state(self) -> GoldenState:
        """Compatibility alias for adapters that call the rules state ``state``."""

        return self.game_state

    @property
    def topology(self) -> CubeFamilyTopology:
        return self.game_state.topology

    @property
    def is_terminal(self) -> bool:
        return self.game_state.is_terminal

    @property
    def state_key(self) -> tuple[object, ...]:
        context = self.observation_context
        return (
            "cube-search-position-v2",
            self.game_state.state_key,
            context.observation_schema_id,
            context.observation_schema_fingerprint,
            context.concrete_observation_fingerprint,
            context.topology_fingerprint,
            context.geometry_fingerprint,
            context.current_board,
            context.previous_boards,
            context.previous_action,
        )


class CubeSearchAdapter:
    """Cube rules/history adapter layered over ``GoldenSearchAdapter``."""

    def __init__(self, rules_adapter: GoldenSearchAdapter | None = None) -> None:
        self.rules = rules_adapter or GoldenSearchAdapter()

    @staticmethod
    def _require_position(position: CubeSearchPosition) -> CubeSearchPosition:
        if not isinstance(position, CubeSearchPosition):
            raise TypeError("CubeSearchAdapter requires CubeSearchPosition")
        return position

    def legal_actions(self, position: CubeSearchPosition) -> tuple[int | str, ...]:
        position = self._require_position(position)
        return self.rules.legal_actions(position.game_state)

    def prepare_legal_actions(self, position: CubeSearchPosition) -> LegalActionContext:
        position = self._require_position(position)
        return self.rules.prepare_legal_actions(position.game_state)

    def action_index(self, position: CubeSearchPosition, action: int | str) -> int:
        position = self._require_position(position)
        return rules_action_to_action_index(action, position.topology.size)

    def action_space(self, position: CubeSearchPosition) -> tuple[int | str, ...]:
        position = self._require_position(position)
        return self.rules.action_space(position.game_state)

    def canonical_action(self, position: CubeSearchPosition, action: int | str) -> int:
        return self.action_index(position, action)

    def rules_action(self, position: CubeSearchPosition, action: int | str) -> int | str:
        return action_index_to_rules_action(self.action_index(position, action), position.topology.size)

    def apply_action(self, position: CubeSearchPosition, action: int | str) -> CubeSearchPosition:
        position = self._require_position(position)
        canonical = self.action_index(position, action)
        rules_action = action_index_to_rules_action(canonical, position.topology.size)
        resulting_state = self.rules.apply_action(position.game_state, rules_action)
        resulting_context = advance_cube_observation_context(
            position.observation_context,
            canonical,
            resulting_state,
        )
        return CubeSearchPosition(resulting_state, resulting_context)

    def is_terminal(self, position: CubeSearchPosition) -> bool:
        return self._require_position(position).game_state.is_terminal

    def terminal_result(self, position: CubeSearchPosition):
        position = self._require_position(position)
        return self.rules.terminal_result(position.game_state)

    def terminal_utility(self, position: CubeSearchPosition) -> float:
        position = self._require_position(position)
        return self.rules.terminal_utility(position.game_state)


__all__ = ["CubeSearchAdapter", "CubeSearchPosition"]
