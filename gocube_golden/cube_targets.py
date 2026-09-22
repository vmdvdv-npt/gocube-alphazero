"""Short compatibility facade for the Cube V2 pure target builder."""

from .cube_training_targets import (
    CubeTrainingSampleV2,
    build_cube_training_samples,
    build_cube_training_targets,
)

__all__ = ["CubeTrainingSampleV2", "build_cube_training_samples", "build_cube_training_targets"]
