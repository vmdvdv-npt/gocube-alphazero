"""Topology-neutral facade over the existing deterministic rolling replay primitive.

Stage 6 deliberately reuses the proven Torus rolling-window/cap implementation
instead of creating a parallel Cube replay buffer.
"""
from __future__ import annotations

from .torus9_monolith import Torus9RollingReplay as RollingGenerationReplay

__all__ = ["RollingGenerationReplay"]
