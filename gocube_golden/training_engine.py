"""Package facade for the game-independent training execution engine."""

from training_engine import (
    CheckpointContext,
    TrainingAdapter,
    TrainingEngine,
    TrainingIterationResult,
    TrainingState,
    canonical_json,
    sequence_fingerprint,
    value_fingerprint,
)

__all__ = [
    "CheckpointContext",
    "TrainingAdapter",
    "TrainingEngine",
    "TrainingIterationResult",
    "TrainingState",
    "canonical_json",
    "sequence_fingerprint",
    "value_fingerprint",
]
