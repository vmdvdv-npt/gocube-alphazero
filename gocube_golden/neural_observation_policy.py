"""Project-wide neural-observation safety invariants.

Rules komi belongs to the referee/rules/scoring contract. It must never be
encoded directly or through a derived feature in a neural observation tensor.
This policy is topology-neutral and applies to Torus, Cube, and future games.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence


NO_KOMI_NEURAL_OBSERVATION_POLICY_ID = "no-komi-neural-observation-channel-v1"


def normalize_neural_observation_channel(channel: object) -> str:
    return str(channel).strip().lower().replace("-", "_").replace(" ", "_")


def assert_no_komi_neural_observation_channels(
    channels: Sequence[object],
    *,
    context: str = "neural observation",
) -> tuple[str, ...]:
    """Reject neural channel declarations that expose komi or komi-derived data."""
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
        raise ValueError(f"{context} channels must be an ordered sequence")
    normalized = tuple(normalize_neural_observation_channel(channel) for channel in channels)
    forbidden = [
        str(raw)
        for raw, name in zip(channels, normalized)
        if "komi" in name
    ]
    if forbidden:
        raise ValueError(
            f"{context} violates project-wide policy {NO_KOMI_NEURAL_OBSERVATION_POLICY_ID}, "
            "which forbids komi as a neural observation channel and also forbids komi-derived "
            f"neural features; offending channels: {forbidden}. "
            "Komi belongs only to referee/rules/scoring."
        )
    return normalized


def assert_no_komi_neural_observation_schema(
    schema: Mapping[str, object],
    *,
    context: str = "neural observation schema",
) -> tuple[str, ...]:
    """Validate the common channel-list fields used by current/future schemas."""
    if not isinstance(schema, Mapping):
        raise ValueError(f"{context} must be a mapping")
    for key in ("channel_order", "observation_channels", "channels"):
        value = schema.get(key)
        if value is None:
            continue
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"{context} {key} must be an ordered sequence")
        return assert_no_komi_neural_observation_channels(value, context=context)
    raise ValueError(f"{context} does not declare neural observation channels")


__all__ = [
    "NO_KOMI_NEURAL_OBSERVATION_POLICY_ID",
    "assert_no_komi_neural_observation_channels",
    "assert_no_komi_neural_observation_schema",
    "normalize_neural_observation_channel",
]
