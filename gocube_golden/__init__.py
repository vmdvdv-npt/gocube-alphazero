"""Current Golden rules, search, adapters, and checkpoint primitives.

Production execution is deliberately explicit: profile adapters feed the
shared ``SelfPlayEngine``/``TrainingEngine`` and the standalone Arena engine.
Retired generic runners are not re-exported from this package.
"""

from .arena_contract import SearchSettings
from .cube_neural import (
    CUBE_ACTION_COUNT,
    GoldenCubeGraphNetV1,
    GoldenCubeNeuralEvaluator,
    build_cube_action_mask,
    build_cube_observation,
    cube_model_hash,
)
from .cube_selfplay import CubeSelfPlayAdapter, run_cube_selfplay_games_shared
from .cube_topology import CUBE4_TOPOLOGY
from .cube_training import (
    CUBE_SELFPLAY_CONTRACT_ID,
    CUBE_TARGET_CONTRACT_ID,
    CUBE_TARGET_FINGERPRINT,
    CubeSelfPlayGameRecord,
    CubeSelfPlayPosition,
    CubeSelfPlaySearchContract,
    CubeTrainingSample,
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    cube_initial_state,
    cube_state_from_identity,
    cube_state_identity,
    run_cube_selfplay_games,
)
from .cube_training_adapter import CubeTrainingAdapter, run_cube_training_iteration
from .neural import GoldenGraphNetV1, build_observation, model_hash
from .provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256
from .result import Winner
from .rules import (
    IllegalMoveError,
    LegalActionContext,
    apply_action,
    legal_actions,
    prepare_legal_actions,
)
from .scoring import GoldenScore, Ownership, score_terminal
from .search import (
    Evaluation,
    SearchResult,
    SequentialPUCT,
    SequentialPUCTSession,
)
from .search_adapter import GoldenSearchAdapter
from .state import (
    BASELINE_KOMI,
    BLACK,
    EMPTY,
    PASS,
    RULES_PROFILE_ID,
    STAGE0_RULES_FINGERPRINT,
    WHITE,
    GoldenState,
    Stone,
    board_key,
    initial_state,
    opponent,
    research_state_from_stones,
    rules_fingerprint_for,
    state_from_stones,
    validate_komi,
)
from .topology import (
    TORUS_5X5,
    TORUS_5X5_TOPOLOGY_FINGERPRINT,
    TORUS_5X5_TOPOLOGY_ID,
    TORUS_9X9,
    TORUS_9X9_TOPOLOGY_FINGERPRINT,
    TORUS_9X9_TOPOLOGY_ID,
    GoldenTopology,
    permute_topology,
    research_topology,
    torus_5x5,
    torus_9x9,
)
from .torus9 import (
    TORUS9_TOPOLOGY_FINGERPRINT,
    TORUS9_TOPOLOGY_ID,
    Torus9CurrentGraphNet,
    Torus9NeuralEvaluator,
    Torus9OwnershipScoreTrainer,
    Torus9OwnershipTrainer,
    Torus9RollingReplay,
    Torus9SelfPlayAdapter,
    Torus9SelfPlayGameRecord,
    Torus9SelfPlayPosition,
    Torus9SelfPlaySearchContract,
    Torus9TrainingAdapter,
    run_torus9_selfplay_games,
    run_torus9_training_iteration,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
)
from .torus9_contract import (
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_PROFILE_FINGERPRINT,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_KOMI,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)


__all__ = [name for name in globals() if not name.startswith("_")]
