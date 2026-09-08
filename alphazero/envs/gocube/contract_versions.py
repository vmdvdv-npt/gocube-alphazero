"""Versioned GoCube training contracts.

These identifiers are part of the on-disk compatibility contract.  Keep them
in one module so checkpoint, replay, manifest, and documentation code cannot
silently drift apart.
"""

SAMPLE_CLOCK_CONTRACT = "sample-clock-v2"
SAMPLE_CLOCK_STATE_VERSION = 2

# S1 changes the meaning of terminal score/value/ownership labels for setup
# and synthetic-cleanup states. S3 adds the required per-row provenance
# sidecar. Old tensors and old run manifests must fail closed instead of being
# interpreted under the current contract.
REPLAY_FORMAT_VERSION = 4
VALUE_TARGET_SEMANTICS = "win-loss-noresult-s1-v2"
SCORE_TARGET_SEMANTICS = "normalized-score-with-applicability-mask-s1-v2"
OWNERSHIP_TARGET_SEMANTICS = "formal-v3-s1-with-point-mask-v2"
TRAINING_CONTRACT_VERSION = 3

SCORE_INITIALIZATION_CONTRACT = "katago-boardhistory-clear-v1"
TARGET_PROVENANCE_SEMANTICS = "formal-runtime-result-provenance-v1"
TARGET_PROVENANCE_ENCODING = "gocube-target-provenance-encoding-v1"
TERMINATION_CONTRACT = "gocube-termination-provenance-v1"

SEED_DERIVATION_CONTRACT = "gocube-seed-derivation-v1"
DEFAULT_MASTER_SEED = 0
