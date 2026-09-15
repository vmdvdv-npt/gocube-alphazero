"""Compatibility facade for process self-play execution.

The implementation lives at repository level so the Golden scientific package
keeps its source dependency boundary free of process-runtime imports.
"""
from selfplay_engine import (
    InferenceClient,
    InferenceTransportError,
    SelfPlayEngine,
    SelfPlayEngineConfig,
    SelfPlayEngineError,
)

__all__ = [
    "InferenceClient",
    "InferenceTransportError",
    "SelfPlayEngine",
    "SelfPlayEngineConfig",
    "SelfPlayEngineError",
]
