"Independent Golden Stage-1 referee plus Stage-2 sequential Arena/search proof line."

from .demo import DEMO_ACTIONS, demonstration_text
from .replay import ReplayReport, replay
from .arena import (
    ActionEvidence, ArenaSummary, GameRecord, MappedResult, PairSummary, PlayerSlot,
    SequentialGoldenArena, TerminationReason, map_absolute_result, post_action_termination,
    recompute_summary, validate_game_record, write_records_jsonl,
)
from .arena_contract import (
    ARENA_CONTRACT_ID, DEFAULT_ARENA_CONTRACT, GOLDEN_MOVE_LIMIT,
    SEARCH_IMPLEMENTATION_ID, SEARCH_PATH, GoldenArenaContract, SearchSettings,
    reject_checkpoint_arena_overrides,
)
from .players import BadPlayer, GoodPlayer, PlayerContext, SearchPlayer, TracePlayer
from .result import DOUBLE_PASS, GoldenResult, Winner, result_from_terminal
from .rules import (
    IllegalMoveError, IllegalMoveReason, Transition, apply_action, group_from_board,
    legal_actions, liberties_from_board,
)
from .scoring import GoldenScore, Ownership, score_terminal
from .search import (
    INTERNAL_Q_CONVENTION, SEARCH_IMPLEMENTATION_FINGERPRINT, WDL_SEMANTICS,
    Evaluation, ExactSolveResult, SearchError, SearchResult, SequentialPUCT,
    SolveStatus, solve_exact, wdl_to_side_to_move_utility,
)
from .search_adapter import GoldenSearchAdapter, GoldenSearchBoundaryError
from .state import (
    BASELINE_KOMI, BLACK, EMPTY, LEGACY_FORBIDDEN_KOMI, LIVE_HISTORY, PASS,
    RULES_PROFILE_ID, STAGE0_RULES_FINGERPRINT, SYNTHETIC_HISTORY, WHITE, BoardKey,
    GoldenState, LegacyKomiError, Stone, board_key, initial_state, opponent,
    research_state_from_stones, rules_fingerprint_for, state_from_stones, validate_komi,
)
from .topology import (
    TORUS_5X5, TORUS_5X5_TOPOLOGY_FINGERPRINT, TORUS_5X5_TOPOLOGY_ID,
    GoldenTopology, permute_topology, research_topology, torus_5x5,
)

__all__ = [name for name in globals() if not name.startswith("_")]
