"""Compatibility facade for process self-play execution.

The implementation lives at repository level so the Golden scientific package
keeps its source dependency boundary free of process-runtime imports.
"""
from selfplay_engine import (
    GameFinished,
    InferenceNeed,
    InferenceClient,
    InferenceTransportError,
    SharedInferenceResult,
    SharedMemorySpec,
    SelfPlayEngine,
    SelfPlayEngineConfig,
    SelfPlayEngineError,
)

__all__ = [
    "InferenceClient",
    "InferenceTransportError",
    "InferenceNeed",
    "GameFinished",
    "SharedMemorySpec",
    "SharedInferenceResult",
    "SelfPlayEngine",
    "SelfPlayEngineConfig",
    "SelfPlayEngineError",
]
