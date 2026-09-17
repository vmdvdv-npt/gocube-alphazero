"""Current Torus9 Golden surface.

Only scientific primitives and the current self-play/training adapters are
exported here.  Arena execution belongs to ``tools.arena``; historical
Torus9 runners are not part of the public Golden runtime.
"""

from __future__ import annotations

# Install the run-owned profile resolver before self-play/training modules bind
# their loader references.  Golden remains the invariant/reference layer; LR,
# replay, self-play simulations and Arena cadence are operator-owned.
from .torus9_run_owned import (
    RunOwnedTorus9SelfPlaySearchContract,
    install_profile_loader,
    install_selfplay_boundary,
)

install_profile_loader()

from .torus9_monolith import (
    TORUS9_OBSERVATION_CHANNELS,
    TORUS9_TOPOLOGY_FINGERPRINT,
    TORUS9_TOPOLOGY_ID,
    Torus9CurrentGraphNet,
    Torus9GraphNet,
    Torus9NeuralEvaluator,
    Torus9Observation,
    Torus9OwnershipGraphNet,
    Torus9OwnershipScoreGraphNet,
    Torus9RootNoiseEvaluator,
    Torus9SelfPlayGameRecord,
    Torus9SelfPlayPosition,
    _sample_action,
    build_torus9_observation,
    build_torus9_observation_bundle,
    build_torus9_observation_into,
    generate_torus9_evaluation_starts,
    graph_diameter,
    graph_distance,
    summarize_torus9_arena,
    torus9_build_ownership_replay_samples,
    torus9_build_ownership_score_replay_samples,
    torus9_build_replay_samples,
    torus9_checkpoint_info,
    torus9_checkpoint_metadata,
    torus9_contract_proof,
    torus9_first_move_statistics,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    torus9_ownership_target,
    torus9_restore_optimizer_state,
    torus9_save_checkpoint,
    torus9_score_target,
    torus9_state_from_identity,
    torus9_state_identity,
    torus9_z_target,
    validate_torus9_replay_sample,
    write_json,
    write_jsonl,
)
from . import torus9_selfplay as _torus9_selfplay

install_selfplay_boundary(_torus9_selfplay)

from .torus9_selfplay import (
    Torus9CentralInferenceOwner,
    Torus9SelfPlayAdapter,
    Torus9SelfPlayWorkerContext,
    run_torus9_selfplay_games,
    torus9_game_seed,
)
from .torus9_training import (
    Torus9OwnershipScoreTrainer,
    Torus9OwnershipTrainer,
    Torus9RollingReplay,
    TrainingState,
)
from .torus9_parallel_validation import Torus9TrainingAdapter
from . import torus9_training as _torus9_training

# Public name used by the production driver.  Only the simulation count differs
# from the frozen search-family contract; the actual value is profile/run-owned.
Torus9SelfPlaySearchContract = RunOwnedTorus9SelfPlaySearchContract


def run_torus9_training_iteration(*, adapter=None, **kwargs):
    """Current Torus9 training front door with the run-owned adapter by default."""
    selected = adapter or Torus9TrainingAdapter()
    return _torus9_training.run_torus9_training_iteration(adapter=selected, **kwargs)


__all__ = [
    "TORUS9_OBSERVATION_CHANNELS",
    "TORUS9_TOPOLOGY_FINGERPRINT",
    "TORUS9_TOPOLOGY_ID",
    "Torus9CentralInferenceOwner",
    "Torus9CurrentGraphNet",
    "Torus9GraphNet",
    "Torus9NeuralEvaluator",
    "Torus9Observation",
    "Torus9OwnershipGraphNet",
    "Torus9OwnershipScoreGraphNet",
    "Torus9OwnershipScoreTrainer",
    "Torus9OwnershipTrainer",
    "Torus9RootNoiseEvaluator",
    "Torus9RollingReplay",
    "Torus9SelfPlayAdapter",
    "Torus9SelfPlayGameRecord",
    "Torus9SelfPlayPosition",
    "Torus9SelfPlaySearchContract",
    "Torus9SelfPlayWorkerContext",
    "Torus9TrainingAdapter",
    "TrainingState",
    "_sample_action",
    "build_torus9_observation",
    "build_torus9_observation_bundle",
    "build_torus9_observation_into",
    "generate_torus9_evaluation_starts",
    "graph_diameter",
    "graph_distance",
    "run_torus9_selfplay_games",
    "run_torus9_training_iteration",
    "summarize_torus9_arena",
    "torus9_build_ownership_replay_samples",
    "torus9_build_ownership_score_replay_samples",
    "torus9_build_replay_samples",
    "torus9_checkpoint_info",
    "torus9_checkpoint_metadata",
    "torus9_contract_proof",
    "torus9_first_move_statistics",
    "torus9_game_seed",
    "torus9_load_checkpoint",
    "torus9_model_from_metadata",
    "torus9_ownership_target",
    "torus9_restore_optimizer_state",
    "torus9_save_checkpoint",
    "torus9_score_target",
    "torus9_state_from_identity",
    "torus9_state_identity",
    "torus9_z_target",
    "validate_torus9_replay_sample",
    "write_json",
    "write_jsonl",
]
