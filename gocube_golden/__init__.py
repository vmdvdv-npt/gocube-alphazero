"""Current Golden rules, search, adapters, and checkpoint primitives.

Production execution is deliberately explicit: profile adapters feed the
shared ``SelfPlayEngine``/``TrainingEngine`` and the standalone Arena engine.
Retired generic runners are not re-exported from this package.

Cube Stage 1-4 V2 modules stay explicit imports until the production pipeline
is built; historical Cube4/V1 runtime is intentionally absent from package API.
"""

from .arena_contract import SearchSettings
from .execution_reference import (
    LEGION_SELFPLAY_PERFORMANCE_DEGRADED_DELTA_PCT,
    LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE,
    LEGION_TORUS9_SELFPLAY_UNDERFILLED_REFERENCE,
    LegionSelfPlayExecutionAssessment,
    LegionSelfPlayPerformanceReference,
    assess_legion_torus9_selfplay_execution,
    compare_legion_torus9_selfplay_performance,
    effective_active_context_ceiling,
    format_legion_selfplay_advisory,
)
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
    SearchPosition,
    SequentialPUCT,
    SequentialPUCTSession,
)
from .search_adapter import GoldenSearchAdapter
from .selfplay_engine import (
    CooperativeSelfPlayAdapter,
    CooperativeSelfPlayResult,
    run_cooperative_selfplay,
)
from .selfplay_policy import (
    RootDirichletNoiseTransform,
    apply_root_dirichlet_noise,
    sample_action_from_search_result,
)
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
from .torus9_run_owned import (
    install_run_spec_policy,
    install_telegram_start_notification,
)


# Existing run-owned tuning and Telegram presentation policies remain
# process-local. Application-code rollover/provenance is deliberately not
# installed here; it is an explicit child-lifecycle policy per orchestrator.
install_run_spec_policy()
install_telegram_start_notification()


__all__ = [name for name in globals() if not name.startswith("_")]
