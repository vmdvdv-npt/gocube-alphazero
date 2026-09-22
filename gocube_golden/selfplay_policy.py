"""Topology-neutral self-play policy mechanics.

The rules/search adapters own action meaning.  This module only owns the two
pieces of self-play policy semantics that are identical for every topology:
root-only Dirichlet mixing and action selection from root visits.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Mapping, Sequence

from .search import Evaluation, SearchPosition, SearchResult


ActionIndex = Callable[[object], int]


def _action_index(action: object, action_index: ActionIndex | Mapping[object, int] | None) -> int:
    if callable(action_index):
        return int(action_index(action))
    if isinstance(action_index, Mapping):
        try:
            return int(action_index[action])
        except KeyError as exc:
            raise ValueError(f"Action is missing from the canonical index mapping: {action!r}") from exc
    if isinstance(action, bool) or not isinstance(action, int):
        raise ValueError("A canonical action-index callback is required for non-integer actions")
    return int(action)


def _dirichlet_noise(count: int, alpha: float, generator: Any) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("Dirichlet noise requires at least one legal action")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("Dirichlet alpha must be positive and finite")

    # Torch's generator is used by the existing Golden fixtures.  Keep the
    # import local so the pure action sampler remains usable without torch.
    if generator is not None and generator.__class__.__module__.split(".", 1)[0] == "torch":
        try:
            import torch

            values = torch._standard_gamma(  # type: ignore[attr-defined]
                torch.full((count,), float(alpha), dtype=torch.float64),
                generator=generator,
            )
            total = float(values.sum())
            if total <= 0.0 or not math.isfinite(total):
                raise ValueError("Dirichlet generator produced an invalid sample")
            return tuple(float(value) / total for value in values.tolist())
        except (ImportError, TypeError, RuntimeError, ValueError) as exc:
            raise ValueError("Unable to draw deterministic Dirichlet noise") from exc

    sampler = getattr(generator, "gammavariate", None)
    if not callable(sampler):
        raise ValueError("Dirichlet noise requires a torch.Generator or gammavariate-capable RNG")
    values = [float(sampler(float(alpha), 1.0)) for _ in range(count)]
    total = sum(values)
    if total <= 0.0 or not math.isfinite(total) or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Dirichlet generator produced an invalid sample")
    return tuple(value / total for value in values)


def apply_root_dirichlet_noise(
    policy: Sequence[float],
    legal_actions: Sequence[object],
    *,
    action_index: ActionIndex | Mapping[object, int] | None = None,
    epsilon: float,
    alpha: float,
    generator: Any,
) -> tuple[float, ...]:
    """Return one legal-only root policy after deterministic Dirichlet mixing.

    ``policy`` remains the caller-owned full action vector.  The returned
    vector has zero probability for every illegal action and is normalized over
    ``legal_actions``.  The helper is intentionally root-agnostic: callers
    decide when it is invoked and provide the action indexing boundary.
    """

    if not math.isfinite(float(epsilon)) or not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("Dirichlet epsilon must be finite and in [0,1]")
    if not legal_actions:
        raise ValueError("Root Dirichlet noise requires at least one legal action")
    try:
        full = tuple(float(value) for value in policy)
    except (TypeError, ValueError) as exc:
        raise ValueError("Root policy must be a numeric sequence") from exc
    if not full or any(not math.isfinite(value) or value < 0.0 for value in full):
        raise ValueError("Root policy must contain finite non-negative values")

    indices = tuple(_action_index(action, action_index) for action in legal_actions)
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(full) for index in indices):
        raise ValueError("Legal actions do not map to a unique policy vector")
    prior_values = tuple(full[index] for index in indices)
    prior_total = sum(prior_values)
    if prior_total <= 0.0:
        prior = tuple(1.0 / len(indices) for _ in indices)
    else:
        prior = tuple(value / prior_total for value in prior_values)
    noise = _dirichlet_noise(len(indices), float(alpha), generator)
    mixed = tuple(
        (1.0 - float(epsilon)) * prior_value + float(epsilon) * noise_value
        for prior_value, noise_value in zip(prior, noise)
    )
    result = [0.0] * len(full)
    for index, value in zip(indices, mixed):
        result[index] = float(value)
    return tuple(result)


class RootDirichletNoiseTransform:
    """Callable evaluation transform that applies noise at one root only."""

    def __init__(
        self,
        root_position: SearchPosition,
        *,
        action_index: ActionIndex | Mapping[object, int] | None = None,
        epsilon: float,
        alpha: float,
        generator: Any,
    ) -> None:
        self.root_state_key = root_position.state_key
        self.action_index = action_index
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.generator = generator

    def __call__(self, base: Evaluation, state: SearchPosition, legal_context: Any) -> Evaluation:
        if state.state_key != self.root_state_key:
            return base
        rules_state = getattr(state, "game_state", state)
        assert_compatible = getattr(legal_context, "assert_compatible", None)
        if callable(assert_compatible):
            assert_compatible(rules_state)
        base_policy = base.policy
        if isinstance(base_policy, Mapping):
            if not callable(self.action_index):
                raise ValueError("Mapping root policies require a canonical action-index callback")
            indexed = {
                int(self.action_index(action)): float(value)
                for action, value in base_policy.items()
            }
            if not indexed:
                raise ValueError("Root policy mapping is empty")
            full_policy = tuple(indexed.get(index, 0.0) for index in range(max(indexed) + 1))
        else:
            full_policy = base_policy
        mixed = apply_root_dirichlet_noise(
            full_policy,
            legal_context.actions,
            action_index=self.action_index,
            epsilon=self.epsilon,
            alpha=self.alpha,
            generator=self.generator,
        )
        return Evaluation(policy=mixed, wdl=base.wdl)


def sample_action_from_search_result(
    result: SearchResult,
    *,
    temperature: float,
    rng: random.Random,
    action_index: ActionIndex | Mapping[object, int] | None = None,
) -> object:
    """Select a self-play action from root visits without changing ``pi``."""

    if not result.legal_actions:
        raise ValueError("Search result has no legal actions")
    if len(result.root_visits) == 0:
        raise ValueError("Search result has no root visits")
    indices = tuple(_action_index(action, action_index) for action in result.legal_actions)
    if any(index < 0 or index >= len(result.root_visits) for index in indices):
        raise ValueError("Search result action index is outside root visits")

    if float(temperature) <= 0.0:
        maximum = max(result.root_visits[index] for index in indices)
        return min(
            (action for action, index in zip(result.legal_actions, indices) if result.root_visits[index] == maximum),
            key=lambda action: _action_index(action, action_index),
        )

    if not math.isfinite(float(temperature)):
        raise ValueError("Temperature must be finite")
    weights = [float(result.root_visits[index]) ** (1.0 / float(temperature)) for index in indices]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return rng.choice(tuple(result.legal_actions))
    threshold = rng.random() * total
    for action, weight in zip(result.legal_actions, weights):
        threshold -= weight
        if threshold <= 0.0:
            return action
    return result.legal_actions[-1]


__all__ = [
    "RootDirichletNoiseTransform",
    "apply_root_dirichlet_noise",
    "sample_action_from_search_result",
]
