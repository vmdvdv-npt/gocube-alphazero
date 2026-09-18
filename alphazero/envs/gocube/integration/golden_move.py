"""Stateless Golden single-move runtime shared by game generation and serving."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

from gocube_golden.arena_contract import SEARCH_IMPLEMENTATION_ID, SearchSettings
from gocube_golden.cube_training import cube_initial_state
from gocube_golden.rules import apply_action
from gocube_golden.search import SearchResult, SequentialPUCT
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, PASS, GoldenState, initial_state
from gocube_golden.topology import TORUS_9X9

from .catalog import CheckpointDescriptor, serving_contract
from .errors import (
    CheckpointIncompatible,
    GenerationFailed,
    InvalidMoveHistory,
    InvalidRequest,
    TerminalPosition,
)
from .golden_mapping import GoldenActionMappingError, GoldenProtocolMapping, mapping_for
from .golden_models import GoldenPlayableModel


GOLDEN_PROTOCOL_RULESET = "chinese"
GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID = "golden-interactive-deterministic-v1"


@dataclass(frozen=True)
class GoldenPositionContract:
    topology: str
    size: int
    rule_set: str
    komi: float

    def __post_init__(self) -> None:
        if not isinstance(self.topology, str) or not self.topology:
            raise InvalidRequest("topology must be a non-empty string")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1:
            raise InvalidRequest("size must be an integer >= 1")
        if not isinstance(self.rule_set, str) or not self.rule_set:
            raise InvalidRequest("ruleSet must be a non-empty string")
        if isinstance(self.komi, bool):
            raise InvalidRequest("komi must be a finite number")
        try:
            komi = float(self.komi)
        except (TypeError, ValueError) as exc:
            raise InvalidRequest("komi must be a finite number") from exc
        if not math.isfinite(komi):
            raise InvalidRequest("komi must be a finite number")
        object.__setattr__(self, "komi", komi)
        if self.rule_set != GOLDEN_PROTOCOL_RULESET:
            raise InvalidRequest(
                f"Unsupported ruleSet {self.rule_set!r}; expected {GOLDEN_PROTOCOL_RULESET!r}"
            )


@dataclass(frozen=True)
class GoldenInteractiveSearchContract:
    profile_id: str = GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID
    cpuct: float = 1.25
    fpu: float = 0.0
    root_noise: bool = False
    fast_search: bool = False
    resign: bool = False
    root_policy_temperature: bool = False
    move_temperature: float = 0.0
    deterministic_tie_break: bool = True

    def settings(self, mcts_sims: int) -> SearchSettings:
        if isinstance(mcts_sims, bool) or not isinstance(mcts_sims, int) or mcts_sims < 1:
            raise InvalidRequest("mctsSims must be an integer >= 1")
        return SearchSettings(
            simulations=mcts_sims,
            cpuct=self.cpuct,
            fpu=self.fpu,
            root_noise=self.root_noise,
            fast_search=self.fast_search,
            resign=self.resign,
            root_policy_temperature=self.root_policy_temperature,
            move_temperature=self.move_temperature,
            deterministic_tie_break=self.deterministic_tie_break,
        )


INTERACTIVE_SEARCH_CONTRACT = GoldenInteractiveSearchContract()


@dataclass(frozen=True)
class MoveSelection:
    action: int | str
    legal_actions: tuple[int | str, ...]
    simulations: int
    implementation_id: str
    search_profile_id: str


def initial_state_for_position(position: GoldenPositionContract) -> GoldenState:
    if position.topology == "cube" and position.size == 4:
        return cube_initial_state(komi=position.komi)
    if position.topology == "torus" and position.size == 9:
        return initial_state(topology=TORUS_9X9, komi=position.komi)
    raise InvalidRequest(
        f"No Golden initial state is registered for {position.topology} size {position.size}"
    )


def mapping_for_position(position: GoldenPositionContract) -> GoldenProtocolMapping:
    try:
        return mapping_for(position.topology, position.size)
    except (GoldenActionMappingError, ValueError) as exc:
        raise InvalidRequest(f"Action mapping failure: {exc}") from exc


def _replay_actions(
    *,
    position: GoldenPositionContract,
    moves: Sequence[Mapping[str, object]],
) -> tuple[GoldenState, tuple[int, int], tuple[tuple[str, ...], ...]]:
    if isinstance(moves, (str, bytes, bytearray)) or not isinstance(moves, Sequence):
        raise InvalidMoveHistory("Action history must be an ordered sequence")
    mapping = mapping_for_position(position)
    state = initial_state_for_position(position)
    captures = [0, 0]
    captured_by_move: list[tuple[str, ...]] = []

    for expected_number, move in enumerate(moves, start=1):
        if state.is_terminal:
            raise InvalidMoveHistory("Action history contains a move after terminal state")
        if not isinstance(move, Mapping):
            raise InvalidMoveHistory("History move must be an object")

        move_number = move.get("moveNumber")
        if (
            isinstance(move_number, bool)
            or not isinstance(move_number, int)
            or move_number != expected_number
        ):
            raise InvalidMoveHistory("History moveNumber is not sequential")

        expected_color = "black" if state.side_to_move == BLACK else "white"
        if move.get("color") != expected_color:
            raise InvalidMoveHistory("History move color does not match Golden side_to_move")

        if "action" not in move:
            raise InvalidMoveHistory("History move is missing action")
        try:
            action = mapping.protocol_action_to_golden(move["action"])
            current_color = state.side_to_move
            transition = apply_action(state, action)
            captured_ids = tuple(mapping.captured_point_ids(transition.captured))
        except (GoldenActionMappingError, KeyError, TypeError, ValueError) as exc:
            raise InvalidMoveHistory(
                f"History move {expected_number} is illegal or unmappable: {exc}"
            ) from exc

        if action != PASS:
            captures[0 if current_color == BLACK else 1] += len(transition.captured)
        captured_by_move.append(captured_ids)
        state = transition.after

    return state, (captures[0], captures[1]), tuple(captured_by_move)


def replay_action_history(
    *,
    topology: str,
    size: int,
    rule_set: str,
    komi: float,
    moves: Sequence[Mapping[str, object]],
) -> GoldenState:
    """Reconstruct a canonical GoldenState from ordered interactive actions."""

    position = GoldenPositionContract(
        topology=topology,
        size=size,
        rule_set=rule_set,
        komi=komi,
    )
    state, _captures, _captured_by_move = _replay_actions(position=position, moves=moves)
    return state


def replay_protocol_moves_strict(
    *,
    topology: str,
    size: int,
    rule_set: str,
    komi: float,
    moves: Sequence[Mapping[str, object]],
) -> tuple[GoldenState, tuple[int, int]]:
    """Generated-game replay wrapper that additionally verifies `captured`."""

    position = GoldenPositionContract(
        topology=topology,
        size=size,
        rule_set=rule_set,
        komi=komi,
    )
    state, captures, captured_by_move = _replay_actions(position=position, moves=moves)
    for expected_number, (move, expected_captured) in enumerate(
        zip(moves, captured_by_move), start=1
    ):
        captured = move.get("captured")
        if not isinstance(captured, list) or tuple(captured) != expected_captured:
            raise InvalidMoveHistory(
                f"Protocol captures disagree with Golden transition at move {expected_number}"
            )
    return state, captures


def validate_checkpoint_position_compatibility(
    *,
    position: GoldenPositionContract,
    state: GoldenState,
    descriptor: CheckpointDescriptor,
    mapping: GoldenProtocolMapping | None = None,
) -> GoldenProtocolMapping:
    """Fail closed before model loading/search on any serving-contract mismatch."""

    checks = {
        "topology": (descriptor.topology, position.topology),
        "size": (descriptor.size, position.size),
        "ruleSet": (descriptor.rule_set, position.rule_set),
        "komi": (descriptor.komi, position.komi),
    }
    for field, (actual, requested) in checks.items():
        if actual != requested:
            raise CheckpointIncompatible(
                f"Golden checkpoint {descriptor.checkpoint_id} is incompatible for {field}: "
                f"checkpoint={actual!r}, requested={requested!r}"
            )

    resolved_mapping = mapping or mapping_for_position(position)
    for field in (
        "architecture_id",
        "rules_fingerprint",
        "observation_fingerprint",
        "target_fingerprint",
    ):
        value = getattr(descriptor, field)
        if not isinstance(value, str) or not value:
            raise CheckpointIncompatible(
                f"Golden checkpoint {descriptor.checkpoint_id} is missing {field}"
            )
    contract = serving_contract(descriptor)
    expected_contract = {
        "topology": position.topology,
        "size": position.size,
        "rule_set": position.rule_set,
        "komi": position.komi,
        "architecture_id": descriptor.architecture_id,
        "rules_fingerprint": descriptor.rules_fingerprint,
        "observation_fingerprint": descriptor.observation_fingerprint,
        "target_fingerprint": descriptor.target_fingerprint,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise CheckpointIncompatible(
                f"Golden checkpoint {descriptor.checkpoint_id} serving contract mismatch for {key}"
            )

    if contract.get("topology_fingerprint") != resolved_mapping.golden_topology_fingerprint:
        raise CheckpointIncompatible(
            f"Golden checkpoint {descriptor.checkpoint_id} topology fingerprint "
            "does not match the Golden mapping"
        )
    if descriptor.rules_fingerprint != state.rules_fingerprint:
        raise CheckpointIncompatible(
            f"Golden checkpoint {descriptor.checkpoint_id} rules fingerprint "
            "does not match the replayed Golden state"
        )
    if state.topology.fingerprint != resolved_mapping.golden_topology_fingerprint:
        raise CheckpointIncompatible("Replayed Golden state does not match requested topology mapping")
    if float(state.komi) != position.komi:
        raise CheckpointIncompatible("Replayed Golden state komi does not match requested position")
    return resolved_mapping


def validate_loaded_model(
    model: object,
    descriptor: CheckpointDescriptor,
    mapping: GoldenProtocolMapping,
) -> GoldenPlayableModel:
    if not isinstance(model, GoldenPlayableModel):
        raise GenerationFailed(
            f"Golden checkpoint {descriptor.checkpoint_id} did not load as GoldenPlayableModel"
        )
    if model.descriptor.checkpoint_id != descriptor.checkpoint_id:
        raise CheckpointIncompatible(
            f"Golden model descriptor does not match requested checkpoint {descriptor.checkpoint_id}"
        )
    metadata = model.metadata
    profile_value = metadata.get("profile_id", metadata.get("training_profile_id"))
    if descriptor.profile_id is not None and profile_value != descriptor.profile_id:
        raise CheckpointIncompatible(
            f"Golden model {descriptor.checkpoint_id} metadata mismatch for profile_id"
        )
    profile_fingerprint = metadata.get(
        "profile_fingerprint", metadata.get("training_profile_fingerprint")
    )
    if (
        descriptor.profile_fingerprint is not None
        and profile_fingerprint != descriptor.profile_fingerprint
    ):
        raise CheckpointIncompatible(
            f"Golden model {descriptor.checkpoint_id} metadata mismatch for profile_fingerprint"
        )
    expected = {
        "architecture_id": descriptor.architecture_id,
        "rules_fingerprint": descriptor.rules_fingerprint,
        "observation_fingerprint": descriptor.observation_fingerprint,
        "target_fingerprint": descriptor.target_fingerprint,
        "komi": descriptor.komi,
    }
    for key, value in expected.items():
        if value is not None and metadata.get(key) != value:
            raise CheckpointIncompatible(
                f"Golden model {descriptor.checkpoint_id} metadata mismatch for {key}"
            )
    topology_fp = metadata.get("topology_fingerprint")
    if topology_fp != mapping.golden_topology_fingerprint:
        raise CheckpointIncompatible(
            f"Golden model {descriptor.checkpoint_id} topology fingerprint does not match mapping"
        )
    if float(metadata.get("komi", math.nan)) != float(descriptor.komi):
        raise CheckpointIncompatible(
            f"Golden model {descriptor.checkpoint_id} komi does not match descriptor"
        )
    return model


class GoldenMoveSelector:
    """Select exactly one legal action through the deterministic Golden search path."""

    def __init__(
        self,
        *,
        mapping_resolver: Callable[[str, int], GoldenProtocolMapping] = mapping_for,
        search_factory=SequentialPUCT,
        adapter_factory=GoldenSearchAdapter,
        search_contract: GoldenInteractiveSearchContract = INTERACTIVE_SEARCH_CONTRACT,
    ) -> None:
        self.mapping_resolver = mapping_resolver
        self.search_factory = search_factory
        self.adapter_factory = adapter_factory
        self.search_contract = search_contract

    def select_move(
        self,
        *,
        state: GoldenState,
        descriptor: CheckpointDescriptor,
        model: GoldenPlayableModel,
        mcts_sims: int,
    ) -> MoveSelection:
        if state.is_terminal:
            raise TerminalPosition("Cannot select a move from a terminal Golden position")

        try:
            mapping = self.mapping_resolver(descriptor.topology, descriptor.size)
        except (GoldenActionMappingError, ValueError) as exc:
            raise CheckpointIncompatible(f"Action mapping failure: {exc}") from exc

        position = GoldenPositionContract(
            topology=descriptor.topology,
            size=descriptor.size,
            rule_set=descriptor.rule_set,
            komi=descriptor.komi,
        )
        validate_checkpoint_position_compatibility(
            position=position,
            state=state,
            descriptor=descriptor,
            mapping=mapping,
        )
        playable = validate_loaded_model(model, descriptor, mapping)
        settings = self.search_contract.settings(mcts_sims)

        try:
            result: SearchResult = self.search_factory(
                settings,
                adapter=self.adapter_factory(),
            ).search(state, playable.evaluator, seed=0)
        except (CheckpointIncompatible, GenerationFailed, TerminalPosition):
            raise
        except Exception as exc:
            raise GenerationFailed(f"Golden search failed: {exc}") from exc

        if result.simulations != mcts_sims:
            raise GenerationFailed("Golden SequentialPUCT did not execute the requested mctsSims")
        if result.implementation_id != SEARCH_IMPLEMENTATION_ID:
            raise GenerationFailed("Golden search implementation provenance drifted")
        if result.action not in result.legal_actions:
            raise GenerationFailed("Golden search selected an illegal action")

        return MoveSelection(
            action=result.action,
            legal_actions=tuple(result.legal_actions),
            simulations=result.simulations,
            implementation_id=result.implementation_id,
            search_profile_id=self.search_contract.profile_id,
        )


__all__ = [
    "GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID",
    "GOLDEN_PROTOCOL_RULESET",
    "GoldenInteractiveSearchContract",
    "GoldenMoveSelector",
    "GoldenPositionContract",
    "INTERACTIVE_SEARCH_CONTRACT",
    "MoveSelection",
    "initial_state_for_position",
    "mapping_for_position",
    "replay_action_history",
    "replay_protocol_moves_strict",
    "validate_checkpoint_position_compatibility",
    "validate_loaded_model",
]
