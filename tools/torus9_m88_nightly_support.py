#!/usr/bin/env python3
"""Experiment-only profile validation for the M88 nightly A/B campaign."""
from __future__ import annotations

import json
from typing import Any, Mapping

from gocube_golden import torus9_contract as _contract


EXPERIMENT_KIND = "torus9-m88-nightly-ab-v1"
ALLOWED_LEARNING_RATES = frozenset({0.0001, 0.0003, 0.001})
_INSTALLED = False


def _validate_nightly_profile(profile: Mapping[str, Any]) -> None:
    normalized = json.loads(json.dumps(profile))
    normalized.pop("experiment", None)
    normalized["profile_fingerprint"] = _contract.TORUS9_CURRENT_PROFILE_FINGERPRINT
    normalized["content_fingerprint"] = _contract.TORUS9_CURRENT_CONTENT_FINGERPRINT
    normalized["self_play"]["mcts_simulations"] = 64
    normalized["self_play"]["fingerprint"] = (
        _contract.current_torus9_selfplay_contract_fingerprint(
            float(normalized["self_play"]["dirichlet_alpha"])
        )
    )
    normalized["training"]["learning_rate"] = 0.001
    normalized["replay"]["window"] = "rolling last 3 generations"
    normalized["replay"]["generations"] = _contract.TORUS9_ROLLING_GENERATIONS
    normalized["replay"]["cap"] = _contract.TORUS9_MAX_REPLAY_POSITIONS
    _contract._validate_current_torus9_profile(normalized)

    self_play = profile.get("self_play", {})
    training = profile.get("training", {})
    replay = profile.get("replay", {})
    if not isinstance(self_play, Mapping) or not isinstance(training, Mapping) or not isinstance(replay, Mapping):
        raise ValueError("M88 nightly profile sections are malformed")
    if int(self_play.get("mcts_simulations", 0)) != 128:
        raise ValueError("M88 nightly self-play must use exactly 128 simulations")
    learning_rate = float(training.get("learning_rate", 0.0))
    if learning_rate not in ALLOWED_LEARNING_RATES:
        raise ValueError(
            "M88 nightly learning rate must be one of "
            + ", ".join(str(value) for value in sorted(ALLOWED_LEARNING_RATES))
        )
    if int(replay.get("generations", 0)) != 6 or int(replay.get("cap", 0)) != 40000:
        raise ValueError("M88 nightly replay must be rolling-6/cap-40000")


def install_nightly_profile_validation() -> None:
    """Allow only the committed M88 nightly experiment profile family."""
    global _INSTALLED
    if _INSTALLED:
        return
    original = _contract._validate_experimental_torus9_profile

    def validate(profile: Mapping[str, Any]) -> None:
        marker = profile.get("experiment")
        if isinstance(marker, Mapping) and marker.get("kind") == EXPERIMENT_KIND:
            _validate_nightly_profile(profile)
            return
        original(profile)

    _contract._validate_experimental_torus9_profile = validate
    _INSTALLED = True


__all__ = [
    "ALLOWED_LEARNING_RATES",
    "EXPERIMENT_KIND",
    "install_nightly_profile_validation",
]
