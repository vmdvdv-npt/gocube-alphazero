from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PROFILE_ID = "gocube-torus-golden-unified-v2"
SCHEMA_VERSION = 2
PROFILE_RELATIVE_PATH = Path("configs/gocube/torus_golden_v2.json")
FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"
EXPERIMENT_FINGERPRINT = "sha256:5f2347fdb7d230454030a31d8eb828e25f1bbb20c4fdae829b9742d0de3f0d61"

HISTORICAL_PROFILE_ID = "gocube-torus-golden-graph-area-v1"
HISTORICAL_EXPERIMENT_FINGERPRINT = (
    "sha256:1f5a1f821fc2e3dc52c957da9ffe1fc9313fb5b02bd81fd2f452233b57bd576e"
)
RULES_PROFILE_ID = "graph-area-v1"
RULES_FINGERPRINT = "sha256:8eac3337443a70893fa5ad359580f7ba92b18958e06f0d775c29f08791796842"
TOPOLOGY_FINGERPRINT = "sha256:b4097c32d4ab5034b84300fa41f951b353a5fcf0d8226e83922889b5552289ef"
OBSERVATION_SCHEMA_ID = "gocube-torus-golden-observation-v1"
OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_FINGERPRINT = "sha256:d6e3aecc89f7df84f6e758423da4e3fe9269abeca644070db0be9261b30c6361"
TARGET_CONTRACT_ID = "gocube-wdl-side-to-move-v1"
TARGET_CONTRACT_VERSION = 1
TARGET_FINGERPRINT = "sha256:6dffad74c4f832741f9cd522aa407f3e88d6f6159ebe56c3dc226a52b4145e41"
BASELINE_KOMI = 0.5

ARENA_CONTRACT_ID = "golden-arena-search-v1"
SEARCH_IMPLEMENTATION_ID = "golden-sequential-puct-v1"
SEARCH_IMPLEMENTATION_FINGERPRINT = (
    "sha256:2f226a0d4ae69c08a7e07ec74bd4cfb560c730a59da8bb3d3b6824a0fa0c6760"
)
SEARCH_CONTRACT_FINGERPRINT = (
    "sha256:c13d3159e123865f4f091dc667ce94458ec9e6002e076ab82d78cffe8c46df1c"
)
SEED_DERIVATION_ID = "golden-arena-seed-v1"
MOVE_LIMIT = 500
POINT_ID_ORDER_IDENTITY = "row-major-yx:point_id=y*width+x"
VALUE_HEAD_SEMANTICS = "side-to-move:[WIN,DRAW,LOSS]"
NETWORK_HEADS_AND_SHAPES = {"policy": [26], "value": [3]}
TECHNICAL_TERMINATIONS = frozenset(
    {"TRUNCATED_MOVE_LIMIT", "ERROR_ILLEGAL_PLAYER_ACTION", "ERROR_PLAYER_EXCEPTION", "ERROR_SEARCH"}
)
RULE_OUTCOMES = frozenset({"BLACK", "WHITE", "DRAW"})
CHECKPOINT_REQUIRED_METADATA = (
    "rules_profile_id",
    "rules_fingerprint",
    "topology_fingerprint",
    "board_size",
    "point_id_order_identity",
    "komi",
    "observation_schema_id",
    "observation_schema_version",
    "observation_fingerprint",
    "target_contract_id",
    "target_contract_version",
    "target_fingerprint",
    "value_head_semantics",
    "network_heads_and_shapes",
    "parent_or_source_run_identity",
    "model_hash",
)


class GoldenExperimentProfileError(ValueError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _semantic_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(profile[key])
        for key in (
            "profile_id",
            "schema_version",
            "fingerprint_algorithm",
            "topology",
            "rules",
            "observation",
            "targets",
            "results",
            "golden_arena",
            "checkpoint_identity",
            "reproducibility",
        )
    }


def compute_experiment_fingerprint(profile: Mapping[str, Any]) -> str:
    return fingerprint(_semantic_payload(profile))


def compute_search_contract_fingerprint(profile: Mapping[str, Any]) -> str:
    arena = profile["golden_arena"]
    return fingerprint(
        {
            "arena_contract_id": arena["arena_contract_id"],
            "search_implementation_id": arena["search_implementation_id"],
            "search_settings": arena["search_settings"],
            "move_limit": arena["move_limit"],
            "execution": {
                "batching": arena["execution"]["batching"],
                "workers": arena["execution"]["workers"],
                "production_arena_reuse": arena["execution"]["production_arena_reuse"],
            },
        }
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoldenExperimentProfileError(message)


def validate_profile(profile: Mapping[str, Any], *, verify_fingerprint: bool = True) -> None:
    _require(profile.get("profile_id") == PROFILE_ID, "Unexpected Golden experiment profile_id")
    _require(profile.get("schema_version") == SCHEMA_VERSION, "Unsupported Golden profile schema")
    _require(profile.get("fingerprint_algorithm") == FINGERPRINT_ALGORITHM, "Unexpected fingerprint algorithm")

    historical = profile["historical_parent"]
    _require(historical.get("profile_id") == HISTORICAL_PROFILE_ID, "Historical parent profile drift")
    _require(historical.get("experiment_fingerprint") == HISTORICAL_EXPERIMENT_FINGERPRINT, "Historical v1 fingerprint was rewritten")

    topology = profile["topology"]
    _require(topology.get("kind") == "standalone-research-torus", "Golden Torus must stay standalone")
    _require(topology.get("width") == 5 and topology.get("height") == 5 and topology.get("point_count") == 25, "Golden topology must be 5x5/25 points")
    _require(topology.get("point_order") == POINT_ID_ORDER_IDENTITY, "Golden PointId order drift")
    _require(topology.get("production_torus_factory") is False, "Production Torus factory reuse is forbidden")
    _require(topology.get("fingerprint") == TOPOLOGY_FINGERPRINT, "Golden topology fingerprint drift")

    rules = profile["rules"]
    _require(rules.get("profile_id") == RULES_PROFILE_ID, "Golden rules profile drift")
    _require(rules.get("fingerprint") == RULES_FINGERPRINT, "Golden rules fingerprint drift")
    _require(rules.get("komi") == BASELINE_KOMI, "Golden komi must be exactly 0.5")
    _require(rules.get("ko") == "positional-superko", "Golden ko semantics drift")
    _require(rules.get("suicide") == "forbidden", "Golden suicide semantics drift")
    _require(rules.get("placement_order") == ["place", "capture-opponent-zero-liberties", "reject-own-zero-liberties"], "Golden capture-before-suicide order drift")
    _require(rules["pass"].get("terminal_after_consecutive_passes") == 2, "Golden terminal must be exactly two consecutive passes")
    _require(rules["pass"].get("legal_while_nonterminal") is True, "PASS must stay legal while live")
    _require(rules["scoring"].get("method") == "exact-graph-area", "Golden scoring must remain exact graph-area")
    _require(rules["scoring"].get("heuristic_dead_stone_cleanup") is False, "Golden scoring forbids heuristic dead-stone cleanup")

    observation = profile["observation"]
    _require(observation.get("schema_id") == OBSERVATION_SCHEMA_ID, "Observation contract drift")
    _require(observation.get("schema_version") == OBSERVATION_SCHEMA_VERSION, "Observation version drift")
    _require(observation.get("fingerprint") == OBSERVATION_FINGERPRINT, "Observation fingerprint drift")
    _require(observation.get("legal_action_mask_length") == 26, "Golden action mask must include PASS")

    targets = profile["targets"]
    _require(targets.get("contract_id") == TARGET_CONTRACT_ID, "Target contract drift")
    _require(targets.get("contract_version") == TARGET_CONTRACT_VERSION, "Target version drift")
    _require(targets.get("fingerprint") == TARGET_FINGERPRINT, "Target fingerprint drift")
    value = targets["value"]
    _require(value.get("vector") == ["WIN", "DRAW", "LOSS"], "Value vector must be [WIN,DRAW,LOSS]")
    _require(value.get("perspective") == "side-to-move", "Value perspective must be side-to-move")
    _require(value.get("utility") == "P(WIN)-P(LOSS)", "Value utility drift")
    _require(value.get("child_to_parent_sign_flips") == 1, "Exactly one child->parent sign flip is required")
    _require(targets["policy"].get("fast_search") is False, "Golden target policy forbids fast search")

    results = profile["results"]
    _require(results.get("formal_termination") == ["DOUBLE_PASS"], "Formal termination taxonomy drift")
    _require(set(results.get("rule_level", ())) == RULE_OUTCOMES, "Rule result taxonomy drift")
    _require(set(results.get("technical", ())) == TECHNICAL_TERMINATIONS, "Technical taxonomy drift")
    _require(results.get("technical_are_training_targets") is False, "Technical outcomes cannot be WDL targets")

    arena = profile["golden_arena"]
    _require(arena.get("arena_contract_id") == ARENA_CONTRACT_ID, "Arena contract id drift")
    _require(arena.get("search_implementation_id") == SEARCH_IMPLEMENTATION_ID, "Search implementation id drift")
    _require(arena.get("search_implementation_fingerprint") == SEARCH_IMPLEMENTATION_FINGERPRINT, "Search implementation fingerprint drift")
    _require(compute_search_contract_fingerprint(profile) == SEARCH_CONTRACT_FINGERPRINT, "Golden Arena search contract fingerprint drift")
    _require(arena.get("search_contract_fingerprint") == SEARCH_CONTRACT_FINGERPRINT, "Persisted Golden Arena search fingerprint drift")
    settings = arena["search_settings"]
    _require(settings.get("simulations") == 64, "Golden Arena simulations must be 64")
    _require(settings.get("cpuct") == 1.25 and settings.get("fpu") == 0.0, "Golden PUCT settings drift")
    for key in ("root_noise", "fast_search", "resign", "root_policy_temperature"):
        _require(settings.get(key) is False, f"Golden Arena {key} must be false")
    _require(settings.get("move_temperature") == 0.0, "Golden Arena move temperature must be zero")
    _require(settings.get("deterministic_tie_break") is True, "Golden tie break must be deterministic")
    _require(arena.get("move_limit") == MOVE_LIMIT, "Golden move limit must be 500")
    execution = arena["execution"]
    _require(execution.get("sequential") is True, "Golden reference Arena must be sequential")
    _require(execution.get("batching") is False and execution.get("workers") is False, "Golden reference Arena must stay single-process/sequential")
    _require(execution.get("production_arena_reuse") is False, "Golden reference Arena must not reuse production Arena")

    checkpoint = profile["checkpoint_identity"]
    _require(checkpoint.get("mismatch_policy") == "fail-closed-no-fallback-default-or-coercion", "Checkpoint mismatch policy drift")
    _require(tuple(checkpoint.get("required_metadata", ())) == CHECKPOINT_REQUIRED_METADATA, "Checkpoint required metadata drift")
    _require(checkpoint.get("board_size") == [5, 5], "Checkpoint board_size contract drift")
    _require(checkpoint.get("point_id_order_identity") == POINT_ID_ORDER_IDENTITY, "Checkpoint PointId order drift")
    _require(checkpoint.get("observation_schema_version") == OBSERVATION_SCHEMA_VERSION, "Checkpoint observation version drift")
    _require(checkpoint.get("target_contract_version") == TARGET_CONTRACT_VERSION, "Checkpoint target version drift")
    _require(checkpoint.get("value_head_semantics") == VALUE_HEAD_SEMANTICS, "Checkpoint value semantics drift")
    _require(checkpoint.get("network_heads_and_shapes") == NETWORK_HEADS_AND_SHAPES, "Checkpoint head/shape contract drift")

    reproducibility = profile["reproducibility"]
    _require(reproducibility.get("seed_derivation_id") == SEED_DERIVATION_ID, "Golden seed derivation contract drift")
    _require(reproducibility.get("master_seed_persisted") is True, "Master seed must be persisted")
    _require(reproducibility.get("derived_seeds_recomputed_by_validator") is True, "Derived seeds must be independently recomputed")
    _require(reproducibility.get("pair_schedule_persisted_and_recomputed") is True, "Pair schedule must be independently recomputed")
    _require(reproducibility.get("canonical_evidence_requires_clean_git_tree") is True, "Canonical evidence must require a clean git tree")

    if verify_fingerprint:
        expected = compute_experiment_fingerprint(profile)
        _require(profile.get("experiment_fingerprint") == expected == EXPERIMENT_FINGERPRINT, f"Golden experiment fingerprint mismatch: expected {expected}, got {profile.get('experiment_fingerprint')}")


def load_profile(path: str | Path | None = None, *, verify_fingerprint: bool = True) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else _repo_root() / PROFILE_RELATIVE_PATH
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    validate_profile(profile, verify_fingerprint=verify_fingerprint)
    return profile
