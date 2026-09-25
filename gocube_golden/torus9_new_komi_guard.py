"""Fail-closed observation policy for the Torus9 ``new_komi`` training line.

Actual rules komi belongs to the referee/game contract.  It must never be
encoded as a neural-network observation channel again.  The historical M137
6-channel model is permitted only as a read-only conversion source.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .torus9_m137_5ch import (
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    M137_FIVE_CHANNEL_CHANNELS,
)


NO_KOMI_OBSERVATION_POLICY_ID = "torus9-no-komi-observation-channel-v1"


def _normalized_channels(channels: Sequence[object]) -> tuple[str, ...]:
    return tuple(
        str(channel).strip().lower().replace("-", "_").replace(" ", "_")
        for channel in channels
    )


def assert_new_komi_training_observation_contract(
    *,
    architecture_id: object,
    input_channels: object,
    observation_channels: Sequence[object],
) -> dict[str, object]:
    """Reject any trainable Torus9 contract that can feed komi to the network."""
    channels = _normalized_channels(observation_channels)
    if "komi" in channels:
        raise ValueError(
            "Torus9 training forbids komi as a neural observation channel; "
            "komi belongs only to the referee/rules contract"
        )
    if int(input_channels) != 5 or len(channels) != 5:
        raise ValueError(
            "Torus9 new_komi training requires exactly 5 neural observation channels; "
            "legacy 6-channel networks are conversion-source only"
        )
    expected_channels = _normalized_channels(M137_FIVE_CHANNEL_CHANNELS)
    if channels != expected_channels:
        raise ValueError(
            "Torus9 new_komi training observation channels differ from the canonical 5CH contract"
        )
    if str(architecture_id) != M137_FIVE_CHANNEL_ARCHITECTURE_ID:
        raise ValueError(
            "Torus9 new_komi training requires the canonical M137-derived 5CH architecture; "
            "legacy architecture selection is forbidden"
        )
    return {
        "policy_id": NO_KOMI_OBSERVATION_POLICY_ID,
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "input_channels": 5,
        "observation_channels": list(M137_FIVE_CHANNEL_CHANNELS),
        "komi_channel": False,
    }


def assert_new_komi_training_model(model: Any) -> dict[str, object]:
    """Validate a model before it can be used by the ``new_komi`` training path."""
    config = getattr(model, "architecture_config", None)
    if not isinstance(config, Mapping):
        raise ValueError("Torus9 training model has no architecture_config mapping")
    projection = getattr(model, "input_projection", None)
    in_features = getattr(projection, "in_features", None)
    if in_features is None:
        raise ValueError("Torus9 training model has no input projection width")
    channels = config.get("observation_channels", M137_FIVE_CHANNEL_CHANNELS)
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
        raise ValueError("Torus9 training observation_channels must be an ordered sequence")
    declared_input_channels = config.get("input_channels", in_features)
    if int(declared_input_channels) != int(in_features):
        raise ValueError("Torus9 training model input-channel metadata disagrees with the network")
    return assert_new_komi_training_observation_contract(
        architecture_id=config.get("architecture_id"),
        input_channels=in_features,
        observation_channels=channels,
    )


def assert_new_komi_training_checkpoint_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate the bootstrap checkpoint before any future training binding uses it."""
    architecture = metadata.get("architecture_config")
    if not isinstance(architecture, Mapping):
        raise ValueError("Torus9 training checkpoint has no architecture_config mapping")
    shape = metadata.get("observation_shape")
    if (
        isinstance(shape, (str, bytes))
        or not isinstance(shape, Sequence)
        or len(shape) != 2
        or int(shape[0]) != 5
    ):
        raise ValueError(
            "Torus9 new_komi training checkpoint must declare observation shape [5,81]; "
            "legacy 6-channel checkpoints are forbidden"
        )
    channels = metadata.get(
        "observation_channels",
        architecture.get("observation_channels", M137_FIVE_CHANNEL_CHANNELS),
    )
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
        raise ValueError("Torus9 training checkpoint observation_channels must be an ordered sequence")
    return assert_new_komi_training_observation_contract(
        architecture_id=metadata.get("architecture_id", architecture.get("architecture_id")),
        input_channels=architecture.get("input_channels", shape[0]),
        observation_channels=channels,
    )


__all__ = [
    "NO_KOMI_OBSERVATION_POLICY_ID",
    "assert_new_komi_training_checkpoint_metadata",
    "assert_new_komi_training_model",
    "assert_new_komi_training_observation_contract",
]
