from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .core import BLACK, EMPTY, WHITE, GoState, Topology, stone_groups

ALIVE_ALGORITHM = "benson-pass-alive-v1"
AUTOMATIC_DEAD_ALGORITHM = "sealed-single-liberty-dead-v1"
AUTOMATIC_SEKI_ALGORITHM = "closed-mutual-two-liberties-seki-v1"


@dataclass(frozen=True)
class EndgameGroupProposal:
    points: tuple[int, ...]
    status: str
    source: str | None = None
    evidence: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _GroupInfo:
    key: str
    points: tuple[int, ...]
    color: int
    liberties: tuple[int, ...]


@dataclass(frozen=True)
class _EmptyRegion:
    key: str
    points: tuple[int, ...]
    boundary_groups: tuple[str, ...]
    vital_groups: tuple[str, ...]


@dataclass(frozen=True)
class _DeadCandidate:
    group_key: str
    points: tuple[int, ...]
    color: int
    liberty: int


@dataclass(frozen=True)
class _SekiCandidate:
    group_keys: tuple[str, str]
    shared_liberties: tuple[int, int]


def _canonical_points(topology: Topology, points: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(points, key=topology.point_id))


def _point_ids(topology: Topology, points: Iterable[int]) -> tuple[str, ...]:
    return tuple(topology.point_id(point) for point in _canonical_points(topology, points))


def _group_key(topology: Topology, points: Iterable[int]) -> str:
    # Match JavaScript JSON.stringify(canonicalizeEndgameGroup(points)).
    return json.dumps(_point_ids(topology, points), separators=(",", ":"))


def _sorted_indices(topology: Topology, points: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(points, key=topology.point_id))


def _validate_requested_groups(
    state: GoState,
    topology: Topology,
    requested_groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    expected = {
        frozenset(group): _canonical_points(topology, group)
        for group in stone_groups(state, topology)
    }
    seen_groups: set[frozenset[int]] = set()
    seen_points: set[int] = set()
    validated: list[tuple[int, ...]] = []

    for requested in requested_groups:
        group = frozenset(requested)
        if not group:
            raise ValueError("Requested endgame group must contain at least one point")
        canonical = expected.get(group)
        if canonical is None:
            raise ValueError("Requested endgame group must be a complete logical stone group")
        if group in seen_groups:
            raise ValueError("Duplicate endgame group requested for analysis")
        if any(point in seen_points for point in group):
            raise ValueError("Requested endgame groups overlap")
        seen_groups.add(group)
        seen_points.update(group)
        validated.append(canonical)

    return tuple(validated)


def _baseline(
    state: GoState,
    topology: Topology,
    groups: Sequence[Sequence[int]],
) -> tuple[EndgameGroupProposal, ...]:
    validated = _validate_requested_groups(state, topology, groups)
    return tuple(
        EndgameGroupProposal(points=points, status="unresolved")
        for points in sorted(validated, key=lambda points: _group_key(topology, points))
    )


def _index_groups(
    state: GoState,
    topology: Topology,
    baseline: Sequence[EndgameGroupProposal],
) -> tuple[dict[str, _GroupInfo], dict[int, str], bool]:
    by_key: dict[str, _GroupInfo] = {}
    point_owner: dict[int, str] = {}

    for proposal in baseline:
        key = _group_key(topology, proposal.points)
        first = proposal.points[0]
        color = int(state.board[first])
        if color not in (BLACK, WHITE):
            raise ValueError("Validated endgame group lost stone occupancy")

        liberties: set[int] = set()
        for point in proposal.points:
            for neighbor in topology.neighbor_indices(point):
                if int(state.board[neighbor]) == EMPTY:
                    liberties.add(neighbor)

        info = _GroupInfo(
            key=key,
            points=proposal.points,
            color=color,
            liberties=_sorted_indices(topology, liberties),
        )
        by_key[key] = info
        for point in proposal.points:
            point_owner[point] = key

    complete = True
    for point in range(topology.point_count):
        if int(state.board[point]) in (BLACK, WHITE) and point not in point_owner:
            complete = False
            break

    return by_key, point_owner, complete


def _collect_empty_regions(
    state: GoState,
    topology: Topology,
    point_owner: Mapping[int, str],
) -> tuple[_EmptyRegion, ...]:
    visited: set[int] = set()
    regions: list[_EmptyRegion] = []

    starts = sorted(range(topology.point_count), key=topology.point_id)
    for start in starts:
        if start in visited or int(state.board[start]) != EMPTY:
            continue

        pending = [start]
        points: list[int] = []
        visited.add(start)
        while pending:
            point = pending.pop()
            points.append(point)
            for neighbor in topology.neighbor_indices(point):
                if int(state.board[neighbor]) != EMPTY or neighbor in visited:
                    continue
                visited.add(neighbor)
                pending.append(neighbor)

        canonical_points = _canonical_points(topology, points)
        boundary_groups: set[str] = set()
        for point in canonical_points:
            for neighbor in topology.neighbor_indices(point):
                owner = point_owner.get(neighbor)
                if owner is not None:
                    boundary_groups.add(owner)

        vital_groups = set(boundary_groups)
        for point in canonical_points:
            adjacent_groups = {
                owner
                for neighbor in topology.neighbor_indices(point)
                if (owner := point_owner.get(neighbor)) is not None
            }
            vital_groups.intersection_update(adjacent_groups)

        key = json.dumps(_point_ids(topology, canonical_points), separators=(",", ":"))
        regions.append(
            _EmptyRegion(
                key=key,
                points=canonical_points,
                boundary_groups=tuple(sorted(boundary_groups)),
                vital_groups=tuple(sorted(vital_groups)),
            )
        )

    return tuple(sorted(regions, key=lambda region: region.key))


def _prove_pass_alive(
    color: int,
    groups: Mapping[str, _GroupInfo],
    regions: Sequence[_EmptyRegion],
) -> dict[str, tuple[_EmptyRegion, ...]]:
    remaining_groups = {
        group.key for group in groups.values() if group.color == color
    }
    candidate_regions = {
        region.key: region
        for region in regions
        if region.boundary_groups
        and all(groups.get(group_key) is not None and groups[group_key].color == color
                for group_key in region.boundary_groups)
    }
    remaining_regions = set(candidate_regions)

    while True:
        groups_to_remove: list[str] = []
        for group_key in remaining_groups:
            vital_region_count = sum(
                group_key in candidate_regions[region_key].vital_groups
                for region_key in remaining_regions
            )
            if vital_region_count < 2:
                groups_to_remove.append(group_key)
        for group_key in groups_to_remove:
            remaining_groups.remove(group_key)

        regions_to_remove = [
            region_key
            for region_key in remaining_regions
            if any(
                group_key not in remaining_groups
                for group_key in candidate_regions[region_key].boundary_groups
            )
        ]
        for region_key in regions_to_remove:
            remaining_regions.remove(region_key)

        if not groups_to_remove and not regions_to_remove:
            break

    proofs: dict[str, tuple[_EmptyRegion, ...]] = {}
    for group_key in sorted(remaining_groups):
        vital_regions = tuple(
            sorted(
                (
                    candidate_regions[region_key]
                    for region_key in remaining_regions
                    if group_key in candidate_regions[region_key].vital_groups
                ),
                key=lambda region: region.key,
            )
        )
        if len(vital_regions) >= 2:
            proofs[group_key] = vital_regions
    return proofs


def _generate_dead_candidates(
    groups: Mapping[str, _GroupInfo],
    pass_alive_group_keys: set[str],
) -> tuple[_DeadCandidate, ...]:
    return tuple(
        _DeadCandidate(
            group_key=group.key,
            points=group.points,
            color=group.color,
            liberty=group.liberties[0],
        )
        for group in sorted(groups.values(), key=lambda item: item.key)
        if group.key not in pass_alive_group_keys and len(group.liberties) == 1
    )


def _verify_dead_candidate(
    candidate: _DeadCandidate,
    *,
    state: GoState,
    topology: Topology,
    groups: Mapping[str, _GroupInfo],
    point_owner: Mapping[int, str],
    pass_alive_group_keys: set[str],
) -> Mapping[str, object] | None:
    group = groups.get(candidate.group_key)
    if group is None or len(group.liberties) != 1 or group.liberties[0] != candidate.liberty:
        return None

    candidate_points = set(candidate.points)
    opponent = WHITE if candidate.color == BLACK else BLACK
    boundary_opponent_keys: set[str] = set()

    def inspect_occupied_boundary(point: int) -> bool:
        occupancy = int(state.board[point])
        if occupancy == EMPTY:
            return True
        owner = point_owner.get(point)
        if owner is None:
            return False
        if occupancy == candidate.color:
            if owner != candidate.group_key and point not in candidate_points:
                return False
            return True
        if occupancy == opponent:
            boundary_opponent_keys.add(owner)
        return True

    for point in candidate.points:
        for neighbor in topology.neighbor_indices(point):
            if neighbor == candidate.liberty or neighbor in candidate_points:
                continue
            if not inspect_occupied_boundary(neighbor):
                return None

    for neighbor in topology.neighbor_indices(candidate.liberty):
        if neighbor in candidate_points:
            continue
        if int(state.board[neighbor]) == EMPTY:
            return None
        if not inspect_occupied_boundary(neighbor):
            return None

    if any(group_key not in pass_alive_group_keys for group_key in boundary_opponent_keys):
        return None

    boundary_alive_groups = tuple(
        _point_ids(topology, groups[group_key].points)
        for group_key in sorted(boundary_opponent_keys)
        if group_key in groups
    )
    return {
        "algorithm": AUTOMATIC_DEAD_ALGORITHM,
        "candidate": "single-liberty",
        "proof": "sealed-liberty-with-pass-alive-boundary",
        "liberty": topology.point_id(candidate.liberty),
        "boundaryAliveGroups": boundary_alive_groups,
    }


def _same_two_liberties(
    first: _GroupInfo,
    second: _GroupInfo,
    topology: Topology,
) -> tuple[int, int] | None:
    if len(first.liberties) != 2 or len(second.liberties) != 2:
        return None
    first_liberties = _sorted_indices(topology, first.liberties)
    second_liberties = _sorted_indices(topology, second.liberties)
    if first_liberties != second_liberties:
        return None
    return first_liberties[0], first_liberties[1]


def _generate_seki_candidates(
    groups: Mapping[str, _GroupInfo],
    excluded_group_keys: set[str],
    topology: Topology,
) -> tuple[_SekiCandidate, ...]:
    eligible = sorted(
        (
            group
            for group in groups.values()
            if group.key not in excluded_group_keys and len(group.liberties) == 2
        ),
        key=lambda group: group.key,
    )
    candidates: list[_SekiCandidate] = []
    for first_index, first in enumerate(eligible):
        for second in eligible[first_index + 1:]:
            if first.color == second.color:
                continue
            shared = _same_two_liberties(first, second, topology)
            if shared is None:
                continue
            candidates.append(_SekiCandidate((first.key, second.key), shared))
    return tuple(candidates)


def _verify_seki_candidate(
    candidate: _SekiCandidate,
    *,
    state: GoState,
    topology: Topology,
    groups: Mapping[str, _GroupInfo],
    point_owner: Mapping[int, str],
) -> Mapping[str, object] | None:
    first_key, second_key = candidate.group_keys
    first = groups.get(first_key)
    second = groups.get(second_key)
    if first is None or second is None or first.color == second.color:
        return None
    actual = _same_two_liberties(first, second, topology)
    if actual is None or actual != candidate.shared_liberties:
        return None

    pair_keys = set(candidate.group_keys)
    shared_liberties = set(candidate.shared_liberties)
    for liberty in candidate.shared_liberties:
        if int(state.board[liberty]) != EMPTY:
            return None
        adjacent_owners: set[str] = set()
        for neighbor in topology.neighbor_indices(liberty):
            if int(state.board[neighbor]) == EMPTY:
                if neighbor not in shared_liberties:
                    return None
                continue
            owner = point_owner.get(neighbor)
            if owner is None or owner not in pair_keys:
                return None
            adjacent_owners.add(owner)
        if first_key not in adjacent_owners or second_key not in adjacent_owners:
            return None

    return {
        "algorithm": AUTOMATIC_SEKI_ALGORITHM,
        "candidate": "two-shared-liberties",
        "proof": "closed-mutual-capture",
        "sharedLiberties": tuple(topology.point_id(point) for point in candidate.shared_liberties),
        "groups": (
            _point_ids(topology, first.points),
            _point_ids(topology, second.points),
        ),
    }


def assisted_endgame_proposal(
    state: GoState,
    topology: Topology,
    groups: Sequence[Sequence[int]] | None = None,
) -> tuple[EndgameGroupProposal, ...]:
    """Port of GoCube AssistedEndgameClassifier at Compatibility V1 source anchor.

    This is Stage A only: it deliberately preserves unresolved groups and is not
    a total AlphaZero terminal adjudicator.
    """
    requested_groups = tuple(stone_groups(state, topology) if groups is None else groups)
    baseline = _baseline(state, topology, requested_groups)
    group_index, point_owner, complete = _index_groups(state, topology, baseline)

    # Match GoCube: automatic proof is disabled for a partial analysis context.
    if not complete:
        return baseline

    regions = _collect_empty_regions(state, topology, point_owner)
    alive_proofs: dict[str, tuple[_EmptyRegion, ...]] = {}
    for color in (BLACK, WHITE):
        alive_proofs.update(_prove_pass_alive(color, group_index, regions))

    pass_alive_group_keys = set(alive_proofs)
    dead_proofs: dict[str, Mapping[str, object]] = {}
    for candidate in _generate_dead_candidates(group_index, pass_alive_group_keys):
        proof = _verify_dead_candidate(
            candidate,
            state=state,
            topology=topology,
            groups=group_index,
            point_owner=point_owner,
            pass_alive_group_keys=pass_alive_group_keys,
        )
        if proof is not None:
            dead_proofs[candidate.group_key] = proof

    already_resolved = set(pass_alive_group_keys) | set(dead_proofs)
    seki_proofs: dict[str, Mapping[str, object]] = {}
    for candidate in _generate_seki_candidates(group_index, already_resolved, topology):
        proof = _verify_seki_candidate(
            candidate,
            state=state,
            topology=topology,
            groups=group_index,
            point_owner=point_owner,
        )
        if proof is None:
            continue
        for group_key in candidate.group_keys:
            seki_proofs[group_key] = proof

    result: list[EndgameGroupProposal] = []
    for proposal in baseline:
        group_key = _group_key(topology, proposal.points)
        vital_regions = alive_proofs.get(group_key)
        if vital_regions is not None:
            result.append(
                EndgameGroupProposal(
                    points=proposal.points,
                    status="alive",
                    source="automatic",
                    evidence={
                        "algorithm": ALIVE_ALGORITHM,
                        "proof": "two-vital-regions",
                        "vitalRegions": tuple(
                            _point_ids(topology, region.points) for region in vital_regions
                        ),
                    },
                )
            )
            continue
        dead_proof = dead_proofs.get(group_key)
        if dead_proof is not None:
            result.append(
                EndgameGroupProposal(
                    points=proposal.points,
                    status="dead",
                    source="automatic",
                    evidence=dead_proof,
                )
            )
            continue
        seki_proof = seki_proofs.get(group_key)
        if seki_proof is not None:
            result.append(
                EndgameGroupProposal(
                    points=proposal.points,
                    status="seki",
                    source="automatic",
                    evidence=seki_proof,
                )
            )
            continue
        result.append(proposal)

    return tuple(result)


def proposal_point_ids(
    proposal: EndgameGroupProposal,
    topology: Topology,
) -> tuple[str, ...]:
    return _point_ids(topology, proposal.points)
