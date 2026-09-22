"""Versioned Cube-family observation schema for Stage 3 (cube2..cube7).

This module is intentionally independent from the historical Cube4 neural
observation. It defines only the deterministic CPU input contract and bounded
real-move history needed by later network/self-play/training stages.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Mapping, Sequence

import torch

from .cube_family import (
    FACE_CORNER,
    FACE_EDGE,
    FACE_INTERIOR,
    GEOMETRY_SCHEMA_ID,
    CubeFamilyTopology,
    CubeRotation,
    cube_family_topology,
)
from .cube_game_contract_v2 import SUPPORTED_SIZES, validate_cube_size
from .rules import LegalActionContext, prepare_legal_actions
from .state import BLACK, EMPTY, WHITE, GoldenState, Stone, opponent

SCHEMA_VERSION = 2
SCHEMA_ID = "gocube-cube-observation-v2"
FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"
HISTORY_DEPTH = 4
LAYOUT = "[channels,points]"
DTYPE = "float32"
CONTEXT_SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "gocube" / "cube_observation_v2.json"

CHANNELS = (
    "own_stones",
    "opponent_stones",
    "side_to_move_is_black",
    "previous_action_was_pass",
    "legal_point_mask",
    "komi_stm_normalized",
    "previous_move_point",
    "own_liberties_1",
    "own_liberties_2",
    "own_liberties_3_plus",
    "opponent_liberties_1",
    "opponent_liberties_2",
    "opponent_liberties_3_plus",
    "history_1_own",
    "history_1_opponent",
    "history_2_own",
    "history_2_opponent",
    "history_3_own",
    "history_3_opponent",
    "history_4_own",
    "history_4_opponent",
    "is_face_interior",
    "is_face_edge",
    "is_face_corner",
    "corner_distance_family_scaled",
    "seam_distance_family_scaled",
    "corner_distance_topology_relative",
    "seam_distance_topology_relative",
    "cross_face_neighbor_count_scaled",
    "face_size_scaled",
)
CHANNEL_COUNT = len(CHANNELS)
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}

# Derived from Stage-2 CubeFamilyTopology across cube2..cube7. The schema
# validator independently recomputes them to catch geometry drift.
CORNER_DISTANCE_FAMILY_SCALE = 6.0
SEAM_DISTANCE_FAMILY_SCALE = 3.0


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _schema_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_id": SCHEMA_ID,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "channel_order": list(CHANNELS),
        "channel_count": CHANNEL_COUNT,
        "history_depth": HISTORY_DEPTH,
        "layout": LAYOUT,
        "dtype": DTYPE,
        "geometry_dependency": {
            "source": "CubeFamilyTopology",
            "schema_id": GEOMETRY_SCHEMA_ID,
            "supported_sizes": list(SUPPORTED_SIZES),
        },
        "perspective": {
            "stone_planes": "current-side-to-move-relative",
            "liberty_planes": "current-side-to-move-relative",
            "history_planes": "current-side-to-move-relative",
            "side_to_move_is_black": {"BLACK": 1.0, "WHITE": 0.0},
            "komi": "WHITE:+komi;BLACK:-komi",
        },
        "normalization": {
            "komi_stm_normalized": "komi_stm/(P+abs(komi))",
            "corner_distance_family_scale": CORNER_DISTANCE_FAMILY_SCALE,
            "seam_distance_family_scale": SEAM_DISTANCE_FAMILY_SCALE,
            "corner_distance_topology_relative": (
                "corner_distance/max_corner_distance_of_this_topology;zero-if-max=0"
            ),
            "seam_distance_topology_relative": (
                "seam_distance/max_seam_distance_of_this_topology;zero-if-max=0"
            ),
            "cross_face_neighbor_count_scaled": "num_cross_face_neighbors/2.0",
            "face_size_scaled": "n/7.0",
        },
        "history": {
            "depth": HISTORY_DEPTH,
            "real_move_history_separate_from_superko": True,
            "pass_is_real_action": True,
            "pass_appends_superko_history": False,
            "missing_positions": "empty-board",
            "previous_action_space": "point:0..P-1;PASS:P;none:null",
        },
        "compatibility": {
            "historical_cube_observation_compatible": False,
            "torus_observation_compatible": False,
            "requires_matching_schema_fingerprint": True,
            "requires_matching_game_graph_fingerprint": True,
            "requires_matching_geometry_fingerprint": True,
            "requires_matching_size": True,
        },
    }


_SCHEMA_PAYLOAD = _schema_payload()
SCHEMA_FINGERPRINT = _fingerprint(_SCHEMA_PAYLOAD)


def cube_observation_schema() -> dict[str, object]:
    result = copy.deepcopy(_SCHEMA_PAYLOAD)
    result["schema_fingerprint"] = SCHEMA_FINGERPRINT
    return result


def _family_distance_scales() -> tuple[float, float]:
    corner = max(
        point.corner_distance
        for size in SUPPORTED_SIZES
        for point in cube_family_topology(size).points
    )
    seam = max(
        point.seam_distance
        for size in SUPPORTED_SIZES
        for point in cube_family_topology(size).points
    )
    return float(corner), float(seam)


def validate_cube_observation_schema(schema: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(schema, Mapping):
        raise ValueError("Cube observation schema must be a mapping")
    candidate = copy.deepcopy(dict(schema))
    supplied_fingerprint = candidate.pop("schema_fingerprint", None)
    if candidate != _SCHEMA_PAYLOAD:
        raise ValueError("Cube observation schema does not match the canonical Stage-3 contract")
    if supplied_fingerprint != SCHEMA_FINGERPRINT:
        raise ValueError("Cube observation schema fingerprint mismatch")
    if CHANNEL_COUNT != len(CHANNELS):
        raise RuntimeError("Cube observation channel count drift")
    actual_scales = _family_distance_scales()
    expected_scales = (CORNER_DISTANCE_FAMILY_SCALE, SEAM_DISTANCE_FAMILY_SCALE)
    if actual_scales != expected_scales:
        raise ValueError(
            f"Cube family distance scale drift: expected {expected_scales}, got {actual_scales}"
        )
    return schema


def load_cube_observation_schema(path: Path | str = SCHEMA_PATH) -> Mapping[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_cube_observation_schema(payload)


def _require_cube_state(state: GoldenState) -> CubeFamilyTopology:
    if not isinstance(state, GoldenState) or not isinstance(state.topology, CubeFamilyTopology):
        raise ValueError("Cube observation v2 requires a Stage-2 CubeFamilyTopology state")
    return state.topology


def concrete_observation_identity(topology: CubeFamilyTopology) -> dict[str, object]:
    if not isinstance(topology, CubeFamilyTopology):
        raise ValueError("Concrete Cube observation identity requires CubeFamilyTopology")
    payload = {
        "schema_id": SCHEMA_ID,
        "schema_fingerprint": SCHEMA_FINGERPRINT,
        "size": topology.size,
        "point_count": topology.point_count,
        "topology_id": topology.topology_id,
        "game_graph_fingerprint": topology.fingerprint,
        "geometry_fingerprint": topology.geometry_fingerprint,
    }
    result = dict(payload)
    result["concrete_observation_fingerprint"] = _fingerprint(payload)
    return result


def validate_concrete_observation_identity(
    identity: Mapping[str, object], topology: CubeFamilyTopology
) -> Mapping[str, object]:
    if not isinstance(identity, Mapping):
        raise ValueError("Concrete Cube observation identity must be a mapping")
    expected = concrete_observation_identity(topology)
    if dict(identity) != expected:
        raise ValueError("Concrete Cube observation identity mismatch")
    return identity


def _normalize_board(values: Sequence[Stone | int], point_count: int) -> tuple[int, ...]:
    if len(values) != point_count:
        raise ValueError(f"Cube observation history board must contain {point_count} points")
    board = tuple(int(value) for value in values)
    if any(value not in (0, 1, 2) for value in board):
        raise ValueError("Cube observation history contains an invalid stone value")
    return board


@dataclass(frozen=True)
class CubeObservationContext:
    size: int
    topology_id: str
    topology_fingerprint: str
    geometry_fingerprint: str
    observation_schema_id: str
    observation_schema_fingerprint: str
    concrete_observation_fingerprint: str
    current_board: tuple[int, ...]
    previous_boards: tuple[tuple[int, ...], ...]
    previous_action: int | None

    def __post_init__(self) -> None:
        size = validate_cube_size(self.size)
        topology = cube_family_topology(size)
        identity = concrete_observation_identity(topology)
        expected_identity = (
            identity["topology_id"],
            identity["game_graph_fingerprint"],
            identity["geometry_fingerprint"],
            identity["schema_id"],
            identity["schema_fingerprint"],
            identity["concrete_observation_fingerprint"],
        )
        actual_identity = (
            self.topology_id,
            self.topology_fingerprint,
            self.geometry_fingerprint,
            self.observation_schema_id,
            self.observation_schema_fingerprint,
            self.concrete_observation_fingerprint,
        )
        if actual_identity != expected_identity:
            raise ValueError("Cube observation context identity mismatch")
        object.__setattr__(
            self, "current_board", _normalize_board(self.current_board, topology.point_count)
        )
        if len(self.previous_boards) != HISTORY_DEPTH:
            raise ValueError(f"Cube observation context requires exactly {HISTORY_DEPTH} history boards")
        object.__setattr__(
            self,
            "previous_boards",
            tuple(_normalize_board(board, topology.point_count) for board in self.previous_boards),
        )
        action = self.previous_action
        if action is not None and (
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action <= topology.pass_action
        ):
            raise ValueError("Cube observation context previous action is outside canonical action space")

    @property
    def topology(self) -> CubeFamilyTopology:
        return cube_family_topology(self.size)


def make_cube_observation_context(
    state: GoldenState,
    *,
    previous_boards: Sequence[Sequence[Stone | int]] = (),
    previous_action: int | None = None,
) -> CubeObservationContext:
    topology = _require_cube_state(state)
    if len(previous_boards) > HISTORY_DEPTH:
        raise ValueError("Cube observation context accepts at most four previous boards")
    empty = tuple(0 for _ in range(topology.point_count))
    normalized = [_normalize_board(board, topology.point_count) for board in previous_boards]
    normalized.extend(empty for _ in range(HISTORY_DEPTH - len(normalized)))
    identity = concrete_observation_identity(topology)
    return CubeObservationContext(
        size=topology.size,
        topology_id=topology.topology_id,
        topology_fingerprint=topology.fingerprint,
        geometry_fingerprint=topology.geometry_fingerprint,
        observation_schema_id=SCHEMA_ID,
        observation_schema_fingerprint=SCHEMA_FINGERPRINT,
        concrete_observation_fingerprint=str(identity["concrete_observation_fingerprint"]),
        current_board=tuple(int(stone) for stone in state.stones),
        previous_boards=tuple(normalized),
        previous_action=previous_action,
    )


def initial_cube_observation_context(state: GoldenState) -> CubeObservationContext:
    _require_cube_state(state)
    if any(stone != EMPTY for stone in state.stones) or state.side_to_move != BLACK:
        raise ValueError("Initial Cube observation context requires the empty BLACK-to-move state")
    if state.consecutive_passes != 0:
        raise ValueError("Initial Cube observation context requires zero consecutive passes")
    return make_cube_observation_context(state)


def _context_identity(context: CubeObservationContext, topology: CubeFamilyTopology) -> dict[str, object]:
    return {
        "schema_id": context.observation_schema_id,
        "schema_fingerprint": context.observation_schema_fingerprint,
        "size": context.size,
        "point_count": topology.point_count,
        "topology_id": context.topology_id,
        "game_graph_fingerprint": context.topology_fingerprint,
        "geometry_fingerprint": context.geometry_fingerprint,
        "concrete_observation_fingerprint": context.concrete_observation_fingerprint,
    }


def _assert_context_matches_state(
    state: GoldenState, context: CubeObservationContext
) -> CubeFamilyTopology:
    topology = _require_cube_state(state)
    if context.size != topology.size:
        raise ValueError("Cube observation context size does not match state")
    validate_concrete_observation_identity(_context_identity(context, topology), topology)
    current = tuple(int(stone) for stone in state.stones)
    if context.current_board != current:
        raise ValueError("Cube observation context current board does not match state")
    return topology


def advance_cube_observation_context(
    context: CubeObservationContext,
    real_action: int,
    resulting_state: GoldenState,
) -> CubeObservationContext:
    topology = _require_cube_state(resulting_state)
    if context.size != topology.size:
        raise ValueError("Cube observation transition crosses topology sizes")
    validate_concrete_observation_identity(_context_identity(context, topology), topology)
    if (
        isinstance(real_action, bool)
        or not isinstance(real_action, int)
        or not 0 <= real_action <= topology.pass_action
    ):
        raise ValueError("Real Cube action must use canonical point/PASS indexing")
    resulting_board = tuple(int(stone) for stone in resulting_state.stones)
    if real_action == topology.pass_action:
        if resulting_board != context.current_board:
            raise ValueError("PASS cannot change the Cube board arrangement")
    else:
        if context.current_board[real_action] != int(EMPTY):
            raise ValueError("Real point action was not empty in the observation context")
        mover = opponent(resulting_state.side_to_move)
        if resulting_board[real_action] != int(mover):
            raise ValueError("Resulting Cube state does not contain the mover at the real action")
    identity = concrete_observation_identity(topology)
    return CubeObservationContext(
        size=topology.size,
        topology_id=topology.topology_id,
        topology_fingerprint=topology.fingerprint,
        geometry_fingerprint=topology.geometry_fingerprint,
        observation_schema_id=SCHEMA_ID,
        observation_schema_fingerprint=SCHEMA_FINGERPRINT,
        concrete_observation_fingerprint=str(identity["concrete_observation_fingerprint"]),
        current_board=resulting_board,
        previous_boards=(context.current_board,) + context.previous_boards[: HISTORY_DEPTH - 1],
        previous_action=real_action,
    )


def serialize_cube_observation_context(context: CubeObservationContext) -> dict[str, object]:
    return {
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "size": context.size,
        "topology_id": context.topology_id,
        "topology_fingerprint": context.topology_fingerprint,
        "geometry_fingerprint": context.geometry_fingerprint,
        "observation_schema_id": context.observation_schema_id,
        "observation_schema_fingerprint": context.observation_schema_fingerprint,
        "concrete_observation_fingerprint": context.concrete_observation_fingerprint,
        "current_board": list(context.current_board),
        "previous_boards": [list(board) for board in context.previous_boards],
        "previous_action": context.previous_action,
    }


def deserialize_cube_observation_context(payload: Mapping[str, object]) -> CubeObservationContext:
    if not isinstance(payload, Mapping) or payload.get("context_schema_version") != CONTEXT_SCHEMA_VERSION:
        raise ValueError("Invalid Cube observation context schema")
    size = validate_cube_size(payload.get("size"))
    topology = cube_family_topology(size)
    boards = payload.get("previous_boards")
    current = payload.get("current_board")
    if not isinstance(boards, list) or not isinstance(current, list):
        raise ValueError("Serialized Cube observation context board payload is invalid")
    return CubeObservationContext(
        size=size,
        topology_id=payload.get("topology_id"),  # type: ignore[arg-type]
        topology_fingerprint=payload.get("topology_fingerprint"),  # type: ignore[arg-type]
        geometry_fingerprint=payload.get("geometry_fingerprint"),  # type: ignore[arg-type]
        observation_schema_id=payload.get("observation_schema_id"),  # type: ignore[arg-type]
        observation_schema_fingerprint=payload.get("observation_schema_fingerprint"),  # type: ignore[arg-type]
        concrete_observation_fingerprint=payload.get("concrete_observation_fingerprint"),  # type: ignore[arg-type]
        current_board=_normalize_board(current, topology.point_count),
        previous_boards=tuple(_normalize_board(board, topology.point_count) for board in boards),
        previous_action=payload.get("previous_action"),  # type: ignore[arg-type]
    )


def rotate_cube_observation_context(
    context: CubeObservationContext, rotation: CubeRotation
) -> CubeObservationContext:
    topology = context.topology
    if len(rotation.point_permutation) != topology.point_count:
        raise ValueError("Cube observation context rotation/topology mismatch")
    action = context.previous_action
    rotated_action = None if action is None else rotation.action(action)
    return CubeObservationContext(
        size=context.size,
        topology_id=context.topology_id,
        topology_fingerprint=context.topology_fingerprint,
        geometry_fingerprint=context.geometry_fingerprint,
        observation_schema_id=context.observation_schema_id,
        observation_schema_fingerprint=context.observation_schema_fingerprint,
        concrete_observation_fingerprint=context.concrete_observation_fingerprint,
        current_board=tuple(rotation.permute_points(context.current_board)),
        previous_boards=tuple(
            tuple(rotation.permute_points(board)) for board in context.previous_boards
        ),
        previous_action=rotated_action,
    )


_STATIC_GEOMETRY_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _static_geometry_planes(topology: CubeFamilyTopology) -> torch.Tensor:
    key = (topology.fingerprint, topology.geometry_fingerprint)
    cached = _STATIC_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    planes = torch.zeros((9, topology.point_count), dtype=torch.float32)
    max_corner = max(point.corner_distance for point in topology.points)
    max_seam = max(point.seam_distance for point in topology.points)
    class_channel = {FACE_INTERIOR: 0, FACE_EDGE: 1, FACE_CORNER: 2}
    for point in topology.points:
        index = point.point_id
        planes[class_channel[point.geometry_class], index] = 1.0
        planes[3, index] = float(point.corner_distance) / CORNER_DISTANCE_FAMILY_SCALE
        planes[4, index] = float(point.seam_distance) / SEAM_DISTANCE_FAMILY_SCALE
        planes[5, index] = 0.0 if max_corner == 0 else float(point.corner_distance) / max_corner
        planes[6, index] = 0.0 if max_seam == 0 else float(point.seam_distance) / max_seam
        planes[7, index] = float(point.num_cross_face_neighbors) / 2.0
        planes[8, index] = float(topology.size) / 7.0
    if not bool(torch.isfinite(planes).all()):
        raise RuntimeError("Static Cube observation geometry contains NaN or Inf")
    _STATIC_GEOMETRY_CACHE[key] = planes
    return planes


def _validate_destination(destination: torch.Tensor, topology: CubeFamilyTopology) -> None:
    if not isinstance(destination, torch.Tensor):
        raise ValueError("Cube observation destination must be a torch.Tensor")
    if tuple(destination.shape) != (CHANNEL_COUNT, topology.point_count):
        raise ValueError(
            f"Cube observation destination must have shape [{CHANNEL_COUNT},{topology.point_count}]"
        )
    if destination.dtype != torch.float32:
        raise ValueError("Cube observation destination must have dtype float32")
    if destination.device.type != "cpu":
        raise ValueError("Cube observation destination must be CPU-resident")
    if not destination.is_contiguous():
        raise ValueError("Cube observation destination must use contiguous [channels,points] layout")


def _write_stones(
    destination: torch.Tensor,
    channel_own: int,
    channel_opponent: int,
    board: Sequence[int],
    side_to_move: Stone,
) -> None:
    own = int(side_to_move)
    other = int(opponent(side_to_move))
    for point, stone in enumerate(board):
        destination[channel_own, point] = float(stone == own)
        destination[channel_opponent, point] = float(stone == other)


def _write_liberties(destination: torch.Tensor, state: GoldenState) -> None:
    topology = state.topology
    visited: set[int] = set()
    own = state.side_to_move
    for point, stone in enumerate(state.stones):
        if stone == EMPTY or point in visited:
            continue
        color = stone
        group = {point}
        stack = [point]
        visited.add(point)
        liberties: set[int] = set()
        while stack:
            current = stack.pop()
            for neighbor in topology.neighbors(current):
                neighbor_stone = state.stones[neighbor]
                if neighbor_stone == EMPTY:
                    liberties.add(neighbor)
                elif neighbor_stone == color and neighbor not in visited:
                    visited.add(neighbor)
                    group.add(neighbor)
                    stack.append(neighbor)
        count = len(liberties)
        if count <= 0:
            raise ValueError("Cube observation encountered a stone group with zero liberties")
        bucket = 0 if count == 1 else 1 if count == 2 else 2
        base = (
            CHANNEL_INDEX["own_liberties_1"]
            if color == own
            else CHANNEL_INDEX["opponent_liberties_1"]
        )
        channel = base + bucket
        for member in group:
            destination[channel, member] = 1.0


def write_cube_observation(
    destination: torch.Tensor,
    state: GoldenState,
    history_context: CubeObservationContext,
    schema: Mapping[str, object] | None = None,
    legal_context: LegalActionContext | None = None,
) -> torch.Tensor:
    topology = _assert_context_matches_state(state, history_context)
    if state.is_terminal:
        raise ValueError("Terminal Cube states must never be sent to the network for move selection")
    if schema is not None:
        validate_cube_observation_schema(schema)
    if history_context.observation_schema_fingerprint != SCHEMA_FINGERPRINT:
        raise ValueError("Cube observation history context uses a different schema fingerprint")
    _validate_destination(destination, topology)

    destination.zero_()
    current_board = tuple(int(stone) for stone in state.stones)
    _write_stones(
        destination,
        CHANNEL_INDEX["own_stones"],
        CHANNEL_INDEX["opponent_stones"],
        current_board,
        state.side_to_move,
    )
    destination[CHANNEL_INDEX["side_to_move_is_black"]].fill_(
        1.0 if state.side_to_move == BLACK else 0.0
    )

    previous_action = history_context.previous_action
    if previous_action is not None:
        if previous_action == topology.pass_action:
            destination[CHANNEL_INDEX["previous_action_was_pass"]].fill_(1.0)
        else:
            destination[CHANNEL_INDEX["previous_move_point"], previous_action] = 1.0

    if legal_context is None:
        legality = prepare_legal_actions(state)
    else:
        # Search already paid for the exact legality calculation at this leaf.
        # Keep the assertion here so a prepared context can never silently be
        # reused for another rules state.
        legal_context.assert_compatible(state)
        legality = legal_context
    if len(legality.action_mask) != topology.action_count:
        raise ValueError("Cube legal-action context has the wrong action-mask shape")
    destination[CHANNEL_INDEX["legal_point_mask"]].copy_(
        torch.as_tensor(legality.action_mask[: topology.point_count], dtype=torch.float32)
    )

    komi_stm = state.komi if state.side_to_move == WHITE else -state.komi
    komi_normalized = komi_stm / (topology.point_count + abs(state.komi))
    destination[CHANNEL_INDEX["komi_stm_normalized"]].fill_(float(komi_normalized))

    _write_liberties(destination, state)

    for history_index, board in enumerate(history_context.previous_boards, start=1):
        _write_stones(
            destination,
            CHANNEL_INDEX[f"history_{history_index}_own"],
            CHANNEL_INDEX[f"history_{history_index}_opponent"],
            board,
            state.side_to_move,
        )

    destination[CHANNEL_INDEX["is_face_interior"] : CHANNEL_COUNT].copy_(
        _static_geometry_planes(topology)
    )
    if not bool(torch.isfinite(destination).all()):
        raise ValueError("Cube observation contains NaN or Inf")
    return destination


def build_cube_observation(
    state: GoldenState,
    history_context: CubeObservationContext,
    schema: Mapping[str, object] | None = None,
    legal_context: LegalActionContext | None = None,
) -> torch.Tensor:
    topology = _assert_context_matches_state(state, history_context)
    destination = torch.empty((CHANNEL_COUNT, topology.point_count), dtype=torch.float32)
    return write_cube_observation(
        destination,
        state,
        history_context,
        schema,
        legal_context=legal_context,
    )


__all__ = [
    "CHANNELS",
    "CHANNEL_COUNT",
    "CHANNEL_INDEX",
    "CONTEXT_SCHEMA_VERSION",
    "CORNER_DISTANCE_FAMILY_SCALE",
    "CubeObservationContext",
    "DTYPE",
    "HISTORY_DEPTH",
    "LAYOUT",
    "SCHEMA_FINGERPRINT",
    "SCHEMA_ID",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "SEAM_DISTANCE_FAMILY_SCALE",
    "advance_cube_observation_context",
    "build_cube_observation",
    "concrete_observation_identity",
    "cube_observation_schema",
    "deserialize_cube_observation_context",
    "initial_cube_observation_context",
    "load_cube_observation_schema",
    "make_cube_observation_context",
    "rotate_cube_observation_context",
    "serialize_cube_observation_context",
    "validate_concrete_observation_identity",
    "validate_cube_observation_schema",
    "write_cube_observation",
]
