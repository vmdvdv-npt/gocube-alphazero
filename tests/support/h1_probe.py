"""Observation-collision probes for H1, independent of any observation schema."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


def stable_observation_bytes(observation: Any) -> bytes:
    """Encode an observation without Python's process-randomized ``hash``."""

    if hasattr(observation, "tobytes"):
        shape = tuple(int(dimension) for dimension in getattr(observation, "shape", ()))
        dtype = str(getattr(observation, "dtype", "unknown"))
        payload = observation.tobytes(order="C")
        return json.dumps({"shape": shape, "dtype": dtype}, separators=(",", ":")).encode() + b"\0" + payload
    return json.dumps(observation, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def stable_observation_digest(observation: Any) -> str:
    return hashlib.sha256(stable_observation_bytes(observation)).hexdigest()


@dataclass(frozen=True)
class ReachableSample:
    sample_id: str
    state: Any
    actions: tuple[Any, ...] = ()
    category: str = "ordinary"
    seed: int | None = None


@dataclass(frozen=True)
class ObservationCollision:
    observation_digest: str
    sample_ids: tuple[str, ...]
    categories: tuple[str, ...]
    semantic_signatures: tuple[Any, ...]
    semantic_collision: bool


@dataclass(frozen=True)
class ObservationProbeReport:
    samples_examined: int
    observation_groups: int
    collisions: tuple[ObservationCollision, ...]

    @property
    def semantic_collisions(self) -> tuple[ObservationCollision, ...]:
        return tuple(collision for collision in self.collisions if collision.semantic_collision)


def find_observation_collisions(
    samples: Iterable[ReachableSample],
    observation_builder: Callable[[Any], Any],
    semantic_signature: Callable[[Any], Any],
) -> ObservationProbeReport:
    """Group reachable samples by canonical observation and compare semantics."""

    buckets: dict[str, list[tuple[ReachableSample, Any]]] = defaultdict(list)
    sample_count = 0
    for sample in samples:
        sample_count += 1
        observation = observation_builder(sample.state)
        buckets[stable_observation_digest(observation)].append((sample, semantic_signature(sample.state)))
    collisions = []
    for digest, bucket in sorted(buckets.items()):
        if len(bucket) < 2:
            continue
        signatures = tuple(item[1] for item in bucket)
        collisions.append(
            ObservationCollision(
                observation_digest=digest,
                sample_ids=tuple(item[0].sample_id for item in bucket),
                categories=tuple(item[0].category for item in bucket),
                semantic_signatures=signatures,
                semantic_collision=len({repr(signature) for signature in signatures}) > 1,
            )
        )
    return ObservationProbeReport(sample_count, len(buckets), tuple(collisions))


def bounded_reachable_states(
    initial_state: Any,
    *,
    legal_actions: Callable[[Any], Iterable[Any]],
    transition: Callable[[Any, Any], Any],
    state_key: Callable[[Any], Any],
    max_depth: int = 6,
    max_states: int = 2_000,
    category: str = "ordinary",
    seed: int | None = None,
) -> tuple[ReachableSample, ...]:
    """Enumerate only callback-approved legal trajectories with fixed limits."""

    if max_depth < 0 or max_states < 1:
        raise ValueError("Reachability limits must be non-negative depth and positive state count")
    queue: list[tuple[Any, tuple[Any, ...], int]] = [(initial_state, (), 0)]
    seen = {repr(state_key(initial_state))}
    samples: list[ReachableSample] = []
    while queue and len(samples) < max_states:
        state, actions, depth = queue.pop(0)
        samples.append(ReachableSample(f"{category}:{len(samples):05d}", state, actions, category, seed))
        if depth >= max_depth:
            continue
        for action in sorted(legal_actions(state), key=repr):
            child = transition(state, action)
            key = repr(state_key(child))
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, actions + (action,), depth + 1))
    return tuple(samples)


def classify_search_result(report: ObservationProbeReport, *, searched_categories: Sequence[str]) -> dict[str, Any]:
    """Produce the bounded-search wording used by the H1 findings report."""

    return {
        "found": [collision.observation_digest for collision in report.semantic_collisions],
        "same_observation_non_ambiguous": [
            collision.observation_digest
            for collision in report.collisions
            if not collision.semantic_collision
        ],
        "not_found_within_search": list(searched_categories),
        "bounded": True,
    }
