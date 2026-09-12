"Independent Golden Stage-1 referee plus Stage-2 sequential Arena/search proof line."

from .demo import DEMO_ACTIONS, demonstration_text
from .replay import ReplayReport, replay
from .arena import (
    ActionEvidence,
    ArenaSummary,
    GameRecord,
    MappedResult,
    PairSummary,
    PlayerSlot,
    SequentialGoldenArena,
    TECHNICAL_TERMINATIONS,
    TerminationReason,
    map_absolute_result,
    pair_schedule_from_records,
    post_action_termination,
    recompute_summary,
    validate_game_record,
    validate_pair_records,
    validate_run_evidence,
    write_records_jsonl,
    write_run_evidence,
)
from .arena_contract import (
    ARENA_CONTRACT_ID,
    DEFAULT_ARENA_CONTRACT,
    GOLDEN_MOVE_LIMIT,
    SEARCH_CONTRACT_FINGERPRINT,
    SEARCH_IMPLEMENTATION_ID,
    SEARCH_PATH,
    GoldenArenaContract,
    SearchSettings,
    compute_search_contract_fingerprint,
    reject_checkpoint_arena_overrides,
)
from .experiment_profile import (
    EXPERIMENT_FINGERPRINT,
    PROFILE_ID as EXPERIMENT_PROFILE_ID,
    SEED_DERIVATION_ID,
    load_profile as load_experiment_profile,
)
from .players import BadPlayer, GoodPlayer, PlayerContext, SearchPlayer, TracePlayer
from .provenance import (
    CodeIdentity,
    PlayerIdentity,
    RunManifest,
    capture_code_identity,
    checkpoint_player_identity,
    derive_game_seeds,
    evaluator_player_identity,
    infer_player_identity,
    validate_run_manifest,
)
from .result import DOUBLE_PASS, GoldenResult, Winner, result_from_terminal
from .rules import (
    IllegalMoveError,
    IllegalMoveReason,
    Transition,
    apply_action,
    group_from_board,
    legal_actions,
    liberties_from_board,
)
from .scoring import GoldenScore, Ownership, score_terminal
from .search import (
    INTERNAL_Q_CONVENTION,
    SEARCH_IMPLEMENTATION_FINGERPRINT,
    WDL_SEMANTICS,
    Evaluation,
    ExactSolveResult,
    SearchError,
    SearchResult,
    SequentialPUCT,
    SolveStatus,
    solve_exact,
    wdl_to_side_to_move_utility,
)
from .search_adapter import GoldenSearchAdapter, GoldenSearchBoundaryError
from .serialization import (
    GoldenSerializationError,
    game_record_from_dict,
    game_record_from_json,
    game_record_to_dict,
    game_record_to_json,
    load_and_validate_game_record,
    read_records_jsonl,
)
from .state import (
    BASELINE_KOMI,
    BLACK,
    EMPTY,
    LEGACY_FORBIDDEN_KOMI,
    LIVE_HISTORY,
    PASS,
    RULES_PROFILE_ID,
    STAGE0_RULES_FINGERPRINT,
    SYNTHETIC_HISTORY,
    WHITE,
    BoardKey,
    GoldenState,
    LegacyKomiError,
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
    GoldenTopology,
    permute_topology,
    research_topology,
    torus_5x5,
)

__all__ = [name for name in globals() if not name.startswith("_")]
