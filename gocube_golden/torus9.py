"""Current Torus9 public boundary.

The pre-extraction scientific/replay/training implementation remains in
``torus9_monolith`` for this narrow Stage-2 extraction. Current self-play
execution is overridden by the standalone universal ``SelfPlayEngine`` plus a
Torus9 scientific adapter. Training is intentionally not refactored here.
"""
from __future__ import annotations

from . import torus9_monolith as _implementation


for _name in dir(_implementation):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_implementation, _name)

from .torus9_selfplay import (  # noqa: E402
    Torus9CentralInferenceOwner,
    Torus9SelfPlayAdapter,
    Torus9SelfPlayWorkerContext,
    run_torus9_selfplay_games,
    torus9_game_seed,
)

__all__ = [name for name in globals() if not name.startswith("_")]
