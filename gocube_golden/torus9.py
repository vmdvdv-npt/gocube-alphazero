"""Current Torus9 public boundary.

Self-play and training execution are exposed through their standalone engines;
the historical monolith remains available as a compatibility/reference module.
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

from .torus9_training import (  # noqa: E402
    Torus9OwnershipScoreTrainer,
    Torus9OwnershipTrainer,
    Torus9RollingReplay,
    Torus9Trainer,
    Torus9TrainingAdapter,
    TrainingState,
    run_torus9_training_iteration,
    torus9_checkpoint_metadata,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    torus9_save_checkpoint,
    validate_torus9_replay_sample,
)

__all__ = [name for name in globals() if not name.startswith("_")]
