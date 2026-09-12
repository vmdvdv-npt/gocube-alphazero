"""Independent, deliberately small Golden rules oracle for graph-area-v1."""

from .demo import DEMO_ACTIONS, demonstration_text
from .replay import ReplayReport, replay
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
from .state import (
    BASELINE_KOMI,
    BLACK,
    EMPTY,
    LEGACY_FORBIDDEN_KOMI,
    PASS,
    RULES_PROFILE_ID,
    STAGE0_RULES_FINGERPRINT,
    WHITE,
    BoardKey,
    GoldenState,
    LegacyKomiError,
    Stone,
    board_key,
    initial_state,
    opponent,
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
