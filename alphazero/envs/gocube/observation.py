"""Pure model-specific observation adapters for GoCube evaluation.

An adapter owns only immutable model metadata.  It never owns a ``GameState``
and never mutates the authoritative state passed to it.  This is the boundary
used by cross-profile Arena: the game object supplies one semantic timeline,
while each network gets the tensor shape and feature planes recorded in its
own checkpoint contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class GoCubeObservationAdapter:
    """Adapt one authoritative GoCube state to one saved model profile."""

    __slots__ = ("model_game_cls", "_observation_shape")

    def __init__(self, model_game_cls):
        if not hasattr(model_game_cls, "observation_from_semantic_state"):
            raise TypeError(
                "GoCube model game class must expose a pure "
                "observation_from_semantic_state() adapter"
            )
        shape = tuple(int(value) for value in model_game_cls.observation_size())
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError(f"Invalid GoCube model observation shape: {shape!r}")
        self.model_game_cls = model_game_cls
        self._observation_shape = shape

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return self._observation_shape

    def observation_size(self) -> tuple[int, int, int]:
        """Return the immutable shape expected by the associated network."""

        return self._observation_shape

    def observation(self, authoritative_state: Any) -> np.ndarray:
        semantic_state = getattr(authoritative_state, "semantic_state", authoritative_state)
        point_count = self._observation_shape[1]
        board = getattr(semantic_state, "board", None)
        if board is None or np.asarray(board).shape != (point_count,):
            raise ValueError(
                "Authoritative GoCube state topology does not match model observation topology"
            )
        result = np.asarray(
            self.model_game_cls.observation_from_semantic_state(semantic_state),
            dtype=np.float32,
        )
        if tuple(result.shape) != self._observation_shape:
            raise RuntimeError(
                "GoCube observation adapter returned an unexpected shape: "
                f"got={tuple(result.shape)!r}, expected={self._observation_shape!r}"
            )
        if not np.isfinite(result).all():
            raise ValueError("GoCube observation adapter returned non-finite values")
        return result

    def __call__(self, authoritative_state: Any) -> np.ndarray:
        return self.observation(authoritative_state)


def model_observation_adapter(model_game_cls) -> GoCubeObservationAdapter:
    """Construct an immutable adapter for a checkpoint's concrete game class."""

    return GoCubeObservationAdapter(model_game_cls)


# A descriptive alias for callers that prefer the profile terminology.
ModelObservationAdapter = GoCubeObservationAdapter
