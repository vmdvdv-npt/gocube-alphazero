"""Versioned GoCube training contracts.

These identifiers are part of the on-disk compatibility contract.  Keep them
in one module so checkpoint, replay, manifest, and documentation code cannot
silently drift apart.
"""

SAMPLE_CLOCK_CONTRACT = "sample-clock-v2"
SAMPLE_CLOCK_STATE_VERSION = 2

REPLAY_FORMAT_VERSION = 2
VALUE_TARGET_SEMANTICS = "win-loss-noresult-v1"
SCORE_TARGET_SEMANTICS = "normalized-score-with-applicability-mask-v1"
OWNERSHIP_TARGET_SEMANTICS = "formal-v3-with-point-mask-v1"
TRAINING_CONTRACT_VERSION = 2

SEED_DERIVATION_CONTRACT = "gocube-seed-derivation-v1"
DEFAULT_MASTER_SEED = 0
