"""Isolated reader and semantic helpers for Cube game contract v2.

Importing this module does not register a production profile and does not import
training, Arena, orchestration, or neural-network code.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import sha256_fingerprint
from .state import PASS

SCHEMA_VERSION = 2
CONTRACT_ID = "gocube-cube-game-contract-v2"
GAME_FAMILY = "cube"
SUPPORTED_SIZES = (2, 3, 4, 5, 6, 7)
INITIAL_KOMI = 0.5
FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"
CONTRACT_PATH = Path("configs/gocube/cube_game_contract_v2.json")

FORMAL_DOUBLE_PASS = "FORMAL_DOUBLE_PASS"
TECHNICAL_REASONS = frozenset(("MOVE_LIMIT", "TIMEOUT", "WORKER_ERROR"))
PERSPECTIVES = frozenset(("BLACK", "WHITE"))
ABSOLUTE_OWNERSHIP = frozenset(("BLACK", "WHITE", "NEUTRAL"))

_EXPECTED_SEMANTICS: tuple[tuple[tuple[str, ...], object], ...] = (
    (("scope",), "game-semantics-only"),
    (("topology", "faces"), 6),
    (("topology", "points_per_face_formula"), "n*n"),
    (("topology", "point_count_formula"), "6*n*n"),
    (("topology", "distinct_points_across_faces"), True),
    (("topology", "physical_vertex_is_game_point"), False),
    (("topology", "graph", "undirected"), True),
    (("topology", "graph", "connected"), True),
    (("topology", "graph", "self_loops"), False),
    (("topology", "graph", "duplicate_neighbors"), False),
    (("topology", "graph", "degree"), 4),
    (("topology", "graph", "relation_types"), ["SAME_FACE", "CROSS_FACE_SEAM"]),
    (("topology", "graph", "relation_types_have_equal_rule_semantics"), True),
    (("topology", "seams", "physical_seam_count"), 12),
    (("topology", "seams", "pairs_per_seam_formula"), "n"),
    (("topology", "seams", "undirected_cross_face_edges_formula"), "12*n"),
    (("topology", "physical_corners", "count"), 8),
    (("topology", "physical_corners", "points_per_corner"), 3),
    (("topology", "physical_corners", "game_edge_shape"), "triangle"),
    (("topology", "physical_corners", "corner_context_adds_game_edges"), False),
    (("topology", "point_classes", "corner_points"), "24"),
    (("topology", "point_classes", "edge_non_corner_points"), "24*(n-2)"),
    (("topology", "point_classes", "face_interior_points"), "6*(n-2)*(n-2)"),
    (("topology", "point_classes", "cube2_all_points_are_corners"), True),
    (("topology", "stage1_executable_geometry"), "cube4"),
    (("topology", "all_sizes_generator_deferred_to_stage"), 2),
    (("topology", "all_24_rotations_n2_to_n7_deferred_to_stage"), 2),
    (("actions", "point_indices"), "0..P-1"),
    (("actions", "pass_index"), "P"),
    (("actions", "action_count_formula"), "P+1"),
    (("actions", "resign_in_action_space"), False),
    (("actions", "rules_core_internal_pass_sentinel"), "PASS"),
    (("actions", "canonical_to_rules_core_mapping"), "P<->PASS"),
    (("actions", "action_count_is_not_game_length_limit"), True),
    (("rules", "initial_board"), "empty"),
    (("rules", "first_player"), "BLACK"),
    (("rules", "turn_order"), "alternate"),
    (("rules", "occupied_move"), "illegal"),
    (("rules", "capture_order"), "place-capture-opponent-then-check-own-suicide"),
    (("rules", "suicide"), "forbidden"),
    (("rules", "liberties"), "unique-empty-neighbors-of-whole-group"),
    (("rules", "cross_face_seams_participate_in_groups_and_liberties"), True),
    (("rules", "captured_points_may_be_reused_if_legal"), True),
    (("rules", "ko"), "positional-superko"),
    (("rules", "superko_compares"), "stone-arrangement-only"),
    (("rules", "side_to_move_part_of_superko_comparison"), False),
    (("rules", "initial_board_in_superko_history"), True),
    (("rules", "pass", "legal_in_any_nonterminal_state"), True),
    (("rules", "pass", "exempt_from_superko"), True),
    (("rules", "pass", "appends_board_to_superko_history"), False),
    (("rules", "pass", "toggles_player"), True),
    (("rules", "pass", "increments_consecutive_passes"), True),
    (("rules", "pass", "point_move_resets_consecutive_passes"), True),
    (("rules", "terminal", "reason"), "two-consecutive-passes"),
    (("rules", "terminal", "moves_after_terminal"), "illegal"),
    (("rules", "terminal", "terminal_state_sent_to_network_for_move_selection"), False),
    (("rules", "scoring", "method"), "exact-graph-area"),
    (("rules", "scoring", "stones_count_for_owner"), True),
    (("rules", "scoring", "empty_component_single_color_boundary"), "owner"),
    (("rules", "scoring", "empty_component_mixed_or_no_color_boundary"), "neutral"),
    (("rules", "scoring", "prisoner_points"), False),
    (("rules", "scoring", "automatic_dead_stone_removal"), False),
    (("rules", "scoring", "cleanup"), False),
    (("rules", "scoring", "automatic_life_based_termination"), False),
    (("rules", "margin_black"), "black_area-white_area-komi"),
    (("rules", "winner_from_margin"), {"positive": "BLACK", "negative": "WHITE", "zero": "DRAW"}),
    (("state", "real_move_history_separate_from_superko_history"), True),
    (("state", "superko_history_contains_pass"), False),
    (("results", "policy", "meaning"), "root-MCTS-visit-distribution-before-selected-move"),
    (("results", "policy", "length"), "P+1"),
    (("results", "policy", "illegal_action_target_probability"), 0.0),
    (("results", "wdl", "order"), ["WIN", "DRAW", "LOSS"]),
    (("results", "wdl", "perspective"), "side_to_move_of_training_position"),
    (("results", "wdl", "not_terminal_side_to_move"), True),
    (("results", "ownership", "labels"), ["OWN", "OPPONENT", "NEUTRAL"]),
    (("results", "ownership", "perspective"), "side_to_move_of_training_position"),
    (("results", "ownership", "meaning"), "actual-final-graph-area-ownership"),
    (("results", "ownership", "life_estimate"), False),
    (("results", "score", "meaning"), "final-komi-adjusted-margin"),
    (("results", "score", "perspective"), "side_to_move_of_training_position"),
    (("results", "score", "store_exact_margin"), True),
    (("results", "score", "loss_normalization_deferred"), True),
    (("results", "projection", "absolute_black_white_result_computed_once"), True),
    (("results", "projection", "project_per_saved_training_position"), True),
    (("results", "game_length", "limited_by_action_count"), False),
    (("results", "game_length", "captures_can_make_game_longer_than_point_count"), True),
    (("results", "technical_termination", "is_game_rule"), False),
    (("results", "technical_termination", "watchdog_limit"), None),
    (("results", "technical_termination", "historical_1920_inherited"), False),
    (("results", "technical_termination", "reasons"), ["MOVE_LIMIT", "TIMEOUT", "WORKER_ERROR"]),
    (("results", "technical_termination", "converted_to_wdl_score_or_ownership"), False),
    (("results", "technical_termination", "included_in_training_replay"), False),
    (("results", "technical_termination", "included_in_formal_win_statistics"), False),
    (("results", "technical_termination", "formal_double_pass_on_last_allowed_move_has_priority"), True),
    (("results", "technical_termination", "stage1_implementation"), "pure-result-classification-only"),
    (("identity", "family_contract_fingerprint_is_concrete_game_identity"), False),
    (("identity", "size_topology_and_komi_required"), True),
    (("identity", "full_point_mapping_and_adjacency_fingerprint_deferred_to_stage"), 2),
    (("identity", "human_readable_names_are_compatibility_check"), False),
    (("identity", "historical_cube_model_identity_is_future_architecture_identity"), False),
    (("verification_scope", "rules_executable_size"), 4),
    (("verification_scope", "sizes_2_to_7_semantics_contract_only"), True),
    (("verification_scope", "do_not_claim_all_sizes_geometry_verified_before_stage2"), True),
    (("verification_scope", "production_integration"), False),
    (("verification_scope", "training_launch_config"), False),
)

_IDENTITY_COMPONENTS = [
    "game_family",
    "size",
    "point_count",
    "action_count",
    "topology_id",
    "topology_fingerprint",
    "rules_fingerprint",
    "komi",
    "family_contract_fingerprint",
]


def validate_cube_size(size: object) -> int:
    if type(size) is not int or size not in SUPPORTED_SIZES:
        raise ValueError(f"Cube game contract v2 requires integer size 2..7, got {size!r}")
    return size


def point_count_for_size(size: object) -> int:
    n = validate_cube_size(size)
    return 6 * n * n


def action_count_for_size(size: object) -> int:
    return point_count_for_size(size) + 1


def action_index_to_rules_action(action: object, size: object) -> int | str:
    point_count = point_count_for_size(size)
    if type(action) is not int or not 0 <= action <= point_count:
        raise ValueError(f"Cube action must be an integer in 0..{point_count}, got {action!r}")
    return PASS if action == point_count else action


def rules_action_to_action_index(action: object, size: object) -> int:
    point_count = point_count_for_size(size)
    if action == PASS:
        return point_count
    if type(action) is not int or not 0 <= action < point_count:
        raise ValueError(
            f"Golden Cube rule action must be PointId 0..{point_count - 1} or PASS, got {action!r}"
        )
    return action


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(contract))
    payload.pop("contract_fingerprint", None)
    return sha256_fingerprint(payload)


def _at(contract: Mapping[str, Any], path: tuple[str, ...]) -> object:
    value: object = contract
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Cube contract missing semantic field {'.'.join(path)}")
        value = value[key]
    return value


def _profiles(contract: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = contract.get("profiles")
    if not isinstance(raw, list):
        raise ValueError("Cube contract profiles must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for profile in raw:
        if not isinstance(profile, Mapping):
            raise ValueError("Cube contract profile entries must be mappings")
        n = validate_cube_size(profile.get("size"))
        if n in result:
            raise ValueError(f"Duplicate Cube profile size {n}")
        result[n] = profile
    return result


def validate_contract(
    contract: Mapping[str, Any], *, verify_fingerprint: bool = True
) -> Mapping[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError("Cube game contract must be a mapping")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Cube contract schema_version drift")
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError("Cube contract_id drift")
    if contract.get("game_family") != GAME_FAMILY:
        raise ValueError("Cube game_family drift")
    if contract.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM:
        raise ValueError("Cube fingerprint algorithm drift")
    if contract.get("supported_sizes") != list(SUPPORTED_SIZES):
        raise ValueError("Cube supported_sizes drift")
    komi = contract.get("initial_komi")
    if (
        isinstance(komi, bool)
        or not isinstance(komi, (int, float))
        or not math.isfinite(float(komi))
        or float(komi) != INITIAL_KOMI
    ):
        raise ValueError("Cube initial_komi must be exactly 0.5")

    profiles = _profiles(contract)
    if set(profiles) != set(SUPPORTED_SIZES):
        raise ValueError("Cube profiles must cover sizes 2..7 exactly")
    for n in SUPPORTED_SIZES:
        expected = {
            "profile": f"cube{n}",
            "size": n,
            "point_count": point_count_for_size(n),
            "action_count": action_count_for_size(n),
            "initial_komi": INITIAL_KOMI,
        }
        if dict(profiles[n]) != expected:
            raise ValueError(f"cube{n} profile semantic drift")

    for path, expected in _EXPECTED_SEMANTICS:
        actual = _at(contract, path)
        if actual != expected:
            raise ValueError(f"Cube semantic drift at {'.'.join(path)}: {actual!r}")

    if _at(contract, ("identity", "concrete_game_identity_components")) != _IDENTITY_COMPONENTS:
        raise ValueError("Concrete game identity component drift")

    if verify_fingerprint:
        reported = contract.get("contract_fingerprint")
        if not isinstance(reported, str):
            raise ValueError("Cube contract_fingerprint missing")
        if reported != contract_fingerprint(contract):
            raise ValueError("Cube contract fingerprint mismatch")
    return contract


def load_contract(
    path: str | Path | None = None, *, verify_fingerprint: bool = True
) -> dict[str, Any]:
    target = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / CONTRACT_PATH
    )
    with target.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    validate_contract(contract, verify_fingerprint=verify_fingerprint)
    return contract


def profile_for_size(contract: Mapping[str, Any], size: object) -> Mapping[str, Any]:
    validate_contract(contract)
    return _profiles(contract)[validate_cube_size(size)]


def concrete_game_identity(
    contract: Mapping[str, Any],
    size: object,
    *,
    topology_id: str,
    topology_fingerprint: str,
    rules_fingerprint: str,
    komi: float | int | None = None,
) -> dict[str, object]:
    validate_contract(contract)
    n = validate_cube_size(size)
    resolved_komi = INITIAL_KOMI if komi is None else komi
    if (
        isinstance(resolved_komi, bool)
        or not isinstance(resolved_komi, (int, float))
        or not math.isfinite(float(resolved_komi))
    ):
        raise ValueError("Concrete Cube identity requires finite numeric komi")
    for label, value in (
        ("topology_id", topology_id),
        ("topology_fingerprint", topology_fingerprint),
        ("rules_fingerprint", rules_fingerprint),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Concrete Cube identity requires non-empty {label}")
    return {
        "game_family": GAME_FAMILY,
        "size": n,
        "point_count": point_count_for_size(n),
        "action_count": action_count_for_size(n),
        "topology_id": topology_id,
        "topology_fingerprint": topology_fingerprint,
        "rules_fingerprint": rules_fingerprint,
        "komi": float(resolved_komi),
        "family_contract_fingerprint": contract["contract_fingerprint"],
    }


def concrete_game_fingerprint(identity: Mapping[str, object]) -> str:
    return sha256_fingerprint(dict(identity))


def _perspective(value: object) -> str:
    if hasattr(value, "name") and getattr(value, "name") in PERSPECTIVES:
        value = getattr(value, "name")
    if value not in PERSPECTIVES:
        raise ValueError(f"Perspective must be BLACK or WHITE, got {value!r}")
    return str(value)


def project_wdl(winner: object, perspective: object) -> tuple[int, int, int]:
    side = _perspective(perspective)
    if hasattr(winner, "name") and getattr(winner, "name") in PERSPECTIVES:
        winner = getattr(winner, "name")
    if winner == "DRAW":
        return (0, 1, 0)
    if winner not in PERSPECTIVES:
        raise ValueError(f"Winner must be BLACK, WHITE, or DRAW, got {winner!r}")
    return (1, 0, 0) if winner == side else (0, 0, 1)


def project_ownership(
    absolute_ownership: Sequence[object], perspective: object
) -> tuple[str, ...]:
    side = _perspective(perspective)
    result: list[str] = []
    for raw in absolute_ownership:
        if hasattr(raw, "value") and getattr(raw, "value") in ABSOLUTE_OWNERSHIP:
            raw = getattr(raw, "value")
        if raw not in ABSOLUTE_OWNERSHIP:
            raise ValueError(f"Invalid absolute ownership {raw!r}")
        result.append(
            "NEUTRAL" if raw == "NEUTRAL" else "OWN" if raw == side else "OPPONENT"
        )
    return tuple(result)


def project_score(margin_black: object, perspective: object) -> float:
    if (
        isinstance(margin_black, bool)
        or not isinstance(margin_black, (int, float))
        or not math.isfinite(float(margin_black))
    ):
        raise ValueError("Score projection requires finite numeric black margin")
    margin = float(margin_black)
    return margin if _perspective(perspective) == "BLACK" else -margin


def classify_completion(
    *, formal_double_pass: bool, technical_reason: str | None = None
) -> str | None:
    if type(formal_double_pass) is not bool:
        raise ValueError("formal_double_pass must be bool")
    if technical_reason is not None and technical_reason not in TECHNICAL_REASONS:
        raise ValueError(f"Unknown Cube technical termination {technical_reason!r}")
    if formal_double_pass:
        return FORMAL_DOUBLE_PASS
    return f"TECHNICAL_{technical_reason}" if technical_reason is not None else None


def completion_is_formal_result(completion: str | None) -> bool:
    return completion == FORMAL_DOUBLE_PASS


__all__ = [
    "FORMAL_DOUBLE_PASS",
    "action_count_for_size",
    "action_index_to_rules_action",
    "classify_completion",
    "completion_is_formal_result",
    "concrete_game_fingerprint",
    "concrete_game_identity",
    "contract_fingerprint",
    "load_contract",
    "point_count_for_size",
    "profile_for_size",
    "project_ownership",
    "project_score",
    "project_wdl",
    "rules_action_to_action_index",
    "validate_contract",
    "validate_cube_size",
]
