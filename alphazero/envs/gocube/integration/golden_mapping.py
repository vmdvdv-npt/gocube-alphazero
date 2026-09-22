"""Explicit Golden-to-Protocol point/action mappings.

The Golden engines use dense integer point indices while Protocol V1 uses the
stable GoCube ``PointId`` strings. This production bridge currently exposes
only the active Torus9 serving mapping. Cube V2 serving will be integrated by
a later stage rather than falling back to the retired Cube4/V1 runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from alphazero.envs.gocube.core import Topology, torus_topology

from gocube_golden.topology import TORUS_9X9
from gocube_golden.state import PASS


class GoldenActionMappingError(ValueError):
    """Raised when a supported Golden topology cannot cross Protocol V1."""


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GoldenProtocolMapping:
    """A proven bijection for one supported Golden topology."""

    topology_kind: str
    size: int
    golden_topology_id: str
    golden_topology_fingerprint: str
    golden_point_ids: tuple[str, ...]
    protocol_point_ids: tuple[str, ...]
    golden_to_protocol: tuple[int, ...]
    protocol_to_golden: tuple[int, ...]
    golden_neighbors: tuple[tuple[int, ...], ...]
    protocol_neighbors: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        count = len(self.golden_point_ids)
        if count == 0 or len(self.protocol_point_ids) != count:
            raise GoldenActionMappingError("Golden/Protocol point counts differ")
        if len(self.golden_to_protocol) != count or len(self.protocol_to_golden) != count:
            raise GoldenActionMappingError("Golden/Protocol mapping lengths differ")
        if tuple(sorted(self.golden_to_protocol)) != tuple(range(count)):
            raise GoldenActionMappingError("Golden-to-Protocol mapping is not bijective")
        if tuple(sorted(self.protocol_to_golden)) != tuple(range(count)):
            raise GoldenActionMappingError("Protocol-to-Golden mapping is not bijective")
        for golden, protocol in enumerate(self.golden_to_protocol):
            if self.protocol_to_golden[protocol] != golden:
                raise GoldenActionMappingError("Golden/Protocol reverse mapping is not exact")

        if len(self.golden_neighbors) != count or len(self.protocol_neighbors) != count:
            raise GoldenActionMappingError("Golden/Protocol adjacency lengths differ")
        for golden_point, neighbors in enumerate(self.golden_neighbors):
            protocol_point = self.golden_to_protocol[golden_point]
            mapped = {self.golden_to_protocol[neighbor] for neighbor in neighbors}
            expected = set(self.protocol_neighbors[protocol_point])
            if mapped != expected:
                point_id = self.golden_point_ids[golden_point]
                raise GoldenActionMappingError(
                    f"Adjacency mismatch for {self.topology_kind} point {point_id!r}: "
                    f"golden={sorted(mapped)!r}, protocol={sorted(expected)!r}"
                )

    @property
    def point_count(self) -> int:
        return len(self.golden_point_ids)

    @property
    def golden_pass_action(self) -> str:
        return PASS

    @property
    def protocol_pass_action(self) -> int:
        return self.point_count

    @property
    def point_order_fingerprint(self) -> str:
        return _fingerprint({"pointIds": list(self.golden_point_ids)})

    @property
    def adjacency_fingerprint(self) -> str:
        return _fingerprint({"neighborsByIndex": [list(row) for row in self.golden_neighbors]})

    def golden_point_to_protocol_index(self, point: int) -> int:
        self._check_golden_point(point, self.point_count)
        return self.golden_to_protocol[point]

    def protocol_point_to_golden_index(self, point: int) -> int:
        if isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < self.point_count:
            raise GoldenActionMappingError(
                f"Protocol point index is outside 0..{self.point_count - 1}: {point!r}"
            )
        return self.protocol_to_golden[point]

    def golden_point_id_to_protocol_index(self, point_id: str) -> int:
        try:
            golden = self.golden_point_ids.index(point_id)
        except ValueError as exc:
            raise GoldenActionMappingError(f"Unknown Golden PointId {point_id!r}") from exc
        return self.golden_point_to_protocol_index(golden)

    def protocol_point_id_to_golden_index(self, point_id: str) -> int:
        try:
            protocol = self.protocol_point_ids.index(point_id)
        except ValueError as exc:
            raise GoldenActionMappingError(f"Unknown Protocol PointId {point_id!r}") from exc
        return self.protocol_point_to_golden_index(protocol)

    def golden_action_to_protocol(self, action: int | str) -> dict[str, object]:
        if action == PASS:
            return {"type": "pass"}
        if isinstance(action, bool) or not isinstance(action, int):
            raise GoldenActionMappingError(f"Invalid Golden action {action!r}")
        protocol_index = self.golden_point_to_protocol_index(action)
        return {"type": "place", "pointId": self.protocol_point_ids[protocol_index]}

    def protocol_action_to_golden(self, action: Mapping[str, object]) -> int | str:
        if not isinstance(action, Mapping):
            raise GoldenActionMappingError("Protocol action must be an object")
        action_type = action.get("type")
        if action_type == "pass":
            if set(action) != {"type"}:
                raise GoldenActionMappingError("Protocol pass action has unexpected fields")
            return PASS
        if action_type == "place":
            if set(action) != {"type", "pointId"} or not isinstance(action.get("pointId"), str):
                raise GoldenActionMappingError(
                    "Protocol place action must contain only type and pointId"
                )
            return self.protocol_point_id_to_golden_index(str(action["pointId"]))
        raise GoldenActionMappingError(f"Unsupported Protocol action type {action_type!r}")

    def captured_point_ids(self, captured: Sequence[int]) -> list[str]:
        result = []
        seen: set[int] = set()
        for point in captured:
            self._check_golden_point(point, self.point_count)
            if point in seen:
                raise GoldenActionMappingError(f"Duplicate captured Golden point {point}")
            seen.add(point)
            protocol_index = self.golden_point_to_protocol_index(point)
            result.append(self.protocol_point_ids[protocol_index])
        return result

    def protocol_board(self, stones: Sequence[object]) -> dict[str, list[str]]:
        if len(stones) != self.point_count:
            raise GoldenActionMappingError("Golden board length does not match mapping")
        board = {"black": [], "white": []}
        for golden_point, stone in enumerate(stones):
            value = int(stone)
            if value not in (0, 1, 2):
                raise GoldenActionMappingError(f"Invalid Golden stone value {stone!r}")
            if value == 0:
                continue
            protocol_index = self.golden_point_to_protocol_index(golden_point)
            board["black" if value == 1 else "white"].append(
                self.protocol_point_ids[protocol_index]
            )
        return board

    def proof(self) -> dict[str, object]:
        return {
            "topology_kind": self.topology_kind,
            "size": self.size,
            "golden_topology_id": self.golden_topology_id,
            "golden_topology_fingerprint": self.golden_topology_fingerprint,
            "point_count": self.point_count,
            "action_count": self.point_count + 1,
            "pass": {"golden": PASS, "protocol": self.protocol_pass_action},
            "point_order": {
                "golden": list(self.golden_point_ids),
                "protocol": list(self.protocol_point_ids),
            },
            "golden_to_protocol": list(self.golden_to_protocol),
            "protocol_to_golden": list(self.protocol_to_golden),
            "adjacency_exact": True,
            "golden_point_order_fingerprint": self.point_order_fingerprint,
            "golden_adjacency_fingerprint": self.adjacency_fingerprint,
        }

    @staticmethod
    def _check_golden_point(point: object, count: int | None = None) -> None:
        limit = count if count is not None else 0
        if (
            isinstance(point, bool)
            or not isinstance(point, int)
            or point < 0
            or (count is not None and point >= limit)
        ):
            raise GoldenActionMappingError(f"Golden point index is invalid: {point!r}")


def _golden_point_ids(topology_kind: str) -> tuple[str, ...]:
    if topology_kind == "torus":
        return tuple(f"{x},{y}" for y in range(9) for x in range(9))
    raise GoldenActionMappingError(f"Unsupported Golden topology {topology_kind!r}")


def _build_mapping(topology_kind: str, size: int) -> GoldenProtocolMapping:
    if topology_kind == "torus" and size == 9:
        golden = TORUS_9X9
        protocol: Topology = torus_topology(9)
    else:
        raise GoldenActionMappingError(
            f"No Golden/Protocol mapping is registered for {topology_kind} size {size}"
        )

    golden_ids = _golden_point_ids(topology_kind)
    protocol_ids = tuple(protocol.point_ids)
    if len(golden_ids) != golden.point_count:
        raise GoldenActionMappingError("Golden topology point identity count is inconsistent")
    if set(golden_ids) != set(protocol_ids):
        missing = sorted(set(golden_ids) - set(protocol_ids))
        extra = sorted(set(protocol_ids) - set(golden_ids))
        raise GoldenActionMappingError(
            f"PointId sets differ: missing_from_protocol={missing!r}, extra_in_protocol={extra!r}"
        )

    protocol_index_by_id = {point_id: index for index, point_id in enumerate(protocol_ids)}
    golden_to_protocol = tuple(protocol_index_by_id[point_id] for point_id in golden_ids)
    protocol_to_golden_list = [0] * len(golden_to_protocol)
    for golden_index, protocol_index in enumerate(golden_to_protocol):
        protocol_to_golden_list[protocol_index] = golden_index

    return GoldenProtocolMapping(
        topology_kind=topology_kind,
        size=size,
        golden_topology_id=golden.topology_id,
        golden_topology_fingerprint=golden.fingerprint,
        golden_point_ids=golden_ids,
        protocol_point_ids=protocol_ids,
        golden_to_protocol=golden_to_protocol,
        protocol_to_golden=tuple(protocol_to_golden_list),
        golden_neighbors=tuple(tuple(row) for row in golden.adjacency),
        protocol_neighbors=tuple(tuple(row) for row in protocol.neighbors_by_index),
    )


def mapping_for(topology_kind: str, size: int) -> GoldenProtocolMapping:
    """Return the registered, independently audited mapping."""

    return _build_mapping(str(topology_kind), int(size))


def torus9_mapping() -> GoldenProtocolMapping:
    return mapping_for("torus", 9)


torus_point_mapping = torus9_mapping


__all__ = [
    "GoldenActionMappingError",
    "GoldenProtocolMapping",
    "mapping_for",
    "torus9_mapping",
    "torus_point_mapping",
]
