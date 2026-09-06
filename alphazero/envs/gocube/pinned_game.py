from __future__ import annotations

import hashlib

from .game import (
    Cube2JapaneseGame,
    Cube3JapaneseGame,
    Cube4JapaneseGame,
    Cube5JapaneseGame,
    Cube6JapaneseGame,
    Cube7JapaneseGame,
    Torus9JapaneseGame,
    Torus13JapaneseGame,
    Torus19JapaneseGame,
)
from .selfplay_semantics import (
    PASS_WOULD_END_PHASE_CHANNEL,
    PINNED_OBSERVATION_SCHEMA,
    apply_pass_would_end_phase_feature,
)


class _PinnedPassWouldEndPhaseMixin:
    OBSERVATION_FEATURES = PASS_WOULD_END_PHASE_CHANNEL + 1
    OBSERVATION_SCHEMA = PINNED_OBSERVATION_SCHEMA

    def observation(self):
        # Base GoGame.observation() allocates using self.observation_size(), so
        # the subclass feature count makes the existing 17 V3 planes land in an
        # 18-plane tensor and leaves the final plane for passWouldEndPhase.
        return apply_pass_would_end_phase_feature(self, super().observation())

    @classmethod
    def rules_fingerprint(cls) -> str:
        base = super().rules_fingerprint()
        payload = (
            f"{base}|observation={PINNED_OBSERVATION_SCHEMA}"
            f"|passWouldEndPhaseChannel={PASS_WOULD_END_PHASE_CHANNEL}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PinnedTorus9JapaneseGame(_PinnedPassWouldEndPhaseMixin, Torus9JapaneseGame): pass
class PinnedTorus13JapaneseGame(_PinnedPassWouldEndPhaseMixin, Torus13JapaneseGame): pass
class PinnedTorus19JapaneseGame(_PinnedPassWouldEndPhaseMixin, Torus19JapaneseGame): pass
class PinnedCube2JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube2JapaneseGame): pass
class PinnedCube3JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube3JapaneseGame): pass
class PinnedCube4JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube4JapaneseGame): pass
class PinnedCube5JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube5JapaneseGame): pass
class PinnedCube6JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube6JapaneseGame): pass
class PinnedCube7JapaneseGame(_PinnedPassWouldEndPhaseMixin, Cube7JapaneseGame): pass


_PINNED_BY_BASE = {
    Torus9JapaneseGame: PinnedTorus9JapaneseGame,
    Torus13JapaneseGame: PinnedTorus13JapaneseGame,
    Torus19JapaneseGame: PinnedTorus19JapaneseGame,
    Cube2JapaneseGame: PinnedCube2JapaneseGame,
    Cube3JapaneseGame: PinnedCube3JapaneseGame,
    Cube4JapaneseGame: PinnedCube4JapaneseGame,
    Cube5JapaneseGame: PinnedCube5JapaneseGame,
    Cube6JapaneseGame: PinnedCube6JapaneseGame,
    Cube7JapaneseGame: PinnedCube7JapaneseGame,
}


def pinned_game_class(base_game_cls):
    try:
        return _PINNED_BY_BASE[base_game_cls]
    except KeyError as exc:
        raise ValueError(f"No pinned KataGo V3 wrapper for {base_game_cls!r}") from exc
