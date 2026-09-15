"""Protocol topology helpers for the Golden GoCube integration.

Game rules, neural inference, search, self-play and training live in
``gocube_golden``. This package intentionally exposes only the small product
topology bridge used to prove the Protocol V1 point mapping.
"""

from .core import CUBE_FACES, Topology, cube_topology, torus_topology

__all__ = [
    "CUBE_FACES",
    "Topology",
    "cube_topology",
    "torus_topology",
]
