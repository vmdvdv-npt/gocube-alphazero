from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .komi_policy import validate_gocube_komi


PROFILE_ID = "gocube-torus-golden-graph-area-v1"
RULES_PROFILE_ID = "graph-area-v1"
OBSERVATION_SCHEMA_ID = "gocube-torus-golden-observation-v1"
TARGET_CONTRACT_ID = "gocube-wdl-side-to-move-v1"
BASE_BRANCH = "codex/torus-rebuild-v1"
BASE_SHA = "c7a0fab2c708cbc9785f914ec6004fffeb7bb2a7"
PROFILE_RELATIVE_PATH = Path("configs/gocube/torus_golden_v1.json")
FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"
BASELINE_KOMI = 0.5
TECHNICAL_TERMINATIONS = frozenset({"TRUNCATED_MOVE_LIMIT", "ERROR"})
RULE_OUTCOMES = frozenset({"BLACK", "WHITE", "DRAW"})


class ExperimentContractError(ValueError):
    """Raised when the Stage-0 Torus passport is incomplete or inconsistent."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: object) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def torus_neighbors(width: int, height: int) -> list[list[int]]:
    if width < 3 or height < 3:
        raise ExperimentContractError(
            "Stage-0 topology identity requires dimensions >= 3 so N/E/S/W are distinct."
        )
    neighbors: list[list[int]] = []
    for y in range(height):
        for x in range(width):
            north = ((y - 1) % height) * width + x
            east = y * width + ((x + 1) % width)
            south = ((y + 1) % height) * width + x
            west = y * width + ((x - 1) % width)
            neighbors.append([north, east, south, west])
    return neighbors


def topology_identity_payload(topology: Mapping[str, Any]) -> dict[str, Any]:
    width = int(topology["width"])
    height = int(topology["height"])
    return {
        "topology_id": topology["topology_id"],
        "kind": topology["kind"],
        "width": width,
        "height": height,
        "number_of_points": int(topology["number_of_points"]),
        "wrap_x": bool(topology["wrap_x"]),
        "wrap_y": bool(topology["wrap_y"]),
        "undirected": bool(topology["undirected"]),
        "degree": int(topology["degree"]),
        "self_loops": bool(topology["self_loops"]),
        "duplicate_neighbors": bool(topology["duplicate_neighbors"]),
        "point_order": topology["point_order"],
        "neighbor_order": topology["neighbor_order"],
        "neighbors_by_point_id": torus_neighbors(width, height),
    }


def compute_topology_fingerprint(profile: Mapping[str, Any]) -> str:
    return fingerprint(topology_identity_payload(profile["topology"]))


def _without_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("fingerprint", None)
    return payload


def compute_rules_fingerprint(profile: Mapping[str, Any]) -> str:
    rules = _without_fingerprint(profile["rules"])
    payload = {
        "rules": rules,
        "topology_fingerprint": compute_topology_fingerprint(profile),
        "komi": validate_gocube_komi(
            rules["komi"], context=f"{PROFILE_ID} rules fingerprint"
        ),
    }
    return fingerprint(payload)


def compute_observation_fingerprint(profile: Mapping[str, Any]) -> str:
    return fingerprint(_without_fingerprint(profile["observation"]))


def compute_target_fingerprint(profile: Mapping[str, Any]) -> str:
    return fingerprint(_without_fingerprint(profile["targets"]))


def compute_experiment_fingerprint(profile: Mapping[str, Any]) -> str:
    payload = {
        "profile_id": profile["profile_id"],
        "schema_version": profile["schema_version"],
        "fingerprint_algorithm": profile["fingerprint_algorithm"],
        "base": profile["base"],
        "topology_fingerprint": compute_topology_fingerprint(profile),
        "rules_fingerprint": compute_rules_fingerprint(profile),
        "observation_fingerprint": compute_observation_fingerprint(profile),
        "target_fingerprint": compute_target_fingerprint(profile),
        "results": profile["results"],
        "watchdog": profile["watchdog"],
        "search_contracts": profile["search_contracts"],
        "network_contract": profile["network_contract"],
        "checkpoint_identity": profile["checkpoint_identity"],
        "reproducibility": profile["reproducibility"],
        "legacy_inheritance": profile["legacy_inheritance"],
    }
    return fingerprint(payload)


def computed_fingerprints(profile: Mapping[str, Any]) -> dict[str, str]:
    return {
        "topology": compute_topology_fingerprint(profile),
        "rules": compute_rules_fingerprint(profile),
        "observation": compute_observation_fingerprint(profile),
        "targets": compute_target_fingerprint(profile),
        "experiment": compute_experiment_fingerprint(profile),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentContractError(message)


def validate_profile(profile: Mapping[str, Any], *, verify_fingerprints: bool = True) -> None:
    _require(profile.get("profile_id") == PROFILE_ID, "Unexpected experiment profile_id")
    _require(profile.get("schema_version") == 1, "Unsupported experiment schema_version")
    _require(profile.get("fingerprint_algorithm") == FINGERPRINT_ALGORITHM, "Unexpected fingerprint algorithm")

    base = profile["base"]
    _require(base.get("branch") == BASE_BRANCH, "Experiment base branch drifted")
    _require(base.get("git_sha") == BASE_SHA, "Experiment base SHA drifted")

    topology = profile["topology"]
    _require(topology.get("kind") == "torus", "Stage-0 proof topology must be torus")
    _require(topology.get("width") == 5 and topology.get("height") == 5, "Stage-0 proof must be Torus 5x5")
    _require(topology.get("number_of_points") == 25, "Torus 5x5 must expose 25 PointIds")
    _require(topology.get("degree") == 4, "Each Torus 5x5 point must have degree four")
    _require(topology.get("wrap_x") is True and topology.get("wrap_y") is True, "Torus must wrap on both coordinates")
    _require(topology.get("undirected") is True, "Topology must be an undirected graph")
    _require(topology.get("self_loops") is False, "Self-loops are forbidden")
    _require(topology.get("duplicate_neighbors") is False, "Duplicate neighbors are forbidden")
    _require(topology.get("point_order") == "row-major-yx:point_id=y*width+x", "PointId order drifted")
    _require(topology.get("neighbor_order") == ["N", "E", "S", "W"], "Neighbor order drifted")

    adjacency = torus_neighbors(5, 5)
    for point_id, neighbors in enumerate(adjacency):
        _require(len(neighbors) == 4 and len(set(neighbors)) == 4, f"Point {point_id} lacks four distinct neighbors")
        _require(point_id not in neighbors, f"Point {point_id} has a self-loop")
        for neighbor in neighbors:
            _require(point_id in adjacency[neighbor], f"Edge {point_id}<->{neighbor} is not undirected")

    rules = profile["rules"]
    _require(rules.get("profile_id") == RULES_PROFILE_ID, "Unexpected rules profile")
    _require(rules.get("suicide") == "forbidden", "Suicide must be forbidden")
    _require(rules.get("ko") == "positional-superko", "Ko policy must be positional superko")
    _require(rules.get("initial_position_in_superko_history") is True, "Initial placement must enter superko history")
    _require(rules.get("full_state_includes_superko_history") is True, "Full state must include superko history")
    _require(rules["pass"]["terminal_after_consecutive_passes"] == 2, "Exactly two consecutive passes must end the game")
    _require(rules["pass"]["ordinary_move_resets_count"] is True, "Ordinary moves must reset consecutive-pass count")
    komi = validate_gocube_komi(rules.get("komi"), context=f"{PROFILE_ID} profile")
    _require(komi == BASELINE_KOMI, "The first proof profile must pin baseline komi 0.5")

    results = profile["results"]
    _require(set(results["rule_level"]) == RULE_OUTCOMES, "Rule-level result vocabulary drifted")
    _require(set(results["technical"]) == TECHNICAL_TERMINATIONS, "Technical result vocabulary drifted")
    _require(results.get("technical_are_training_targets") is False, "Technical termination must never become a WDL target")

    watchdog = profile["watchdog"]
    _require(watchdog.get("action_cap") == 500, "Torus 5x5 watchdog must be 500 actions")
    _require(watchdog.get("actions_per_point") == 20, "Watchdog formula must be 20 actions per point")
    _require(watchdog.get("pass_counts_as_action") is True, "PASS must count toward watchdog")
    _require(watchdog.get("second_pass_terminal_precedes_truncation") is True, "Formal terminal semantics must precede truncation at the cap")

    targets = profile["targets"]
    _require(targets.get("contract_id") == TARGET_CONTRACT_ID, "Unexpected target contract")
    value = targets["value"]
    _require(value.get("vector") == ["WIN", "DRAW", "LOSS"], "Value vector must be [WIN,DRAW,LOSS]")
    _require(value.get("perspective") == "side-to-move-replay-position", "Value perspective must be side-to-move replay position")
    _require(value.get("utility") == "P(WIN)-P(LOSS)", "WDL utility definition drifted")
    _require(value.get("legacy_wlnr_compatible") is False, "Legacy [WIN,LOSS,NO_RESULT] must be explicitly incompatible")
    policy = targets["policy"]
    _require(policy.get("formula") == "N(a)/sum(N(legal_actions))", "Policy target must be normalized root visits")
    _require(policy.get("illegal_action_mass") == 0, "Illegal actions must have zero policy mass")
    _require(policy.get("requires_positive_visit_sum") is True, "Root visits must have positive sum")
    _require(policy.get("fast_search") is False, "Fast search is forbidden in baseline policy targets")
    _require(policy.get("forced_playout_pruning") is False, "Forced-playout pruning is forbidden")
    _require(policy.get("lcb_transform") is False, "LCB transformations are forbidden")

    observation = profile["observation"]
    _require(observation.get("schema_id") == OBSERVATION_SCHEMA_ID, "Unexpected observation schema")
    names = [channel["name"] for channel in observation["point_channels"]]
    _require(names == ["own_stones", "opponent_stones", "side_to_move_color", "previous_pass", "legal_point_mask", "komi"], "Observation point-channel order drifted")
    _require(observation["legal_action_mask"]["length"] == 26, "Torus 5x5 legal action mask must include 25 points plus PASS")
    _require(observation.get("nn_observation_contains_full_superko_history") is False, "NN observation intentionally does not expose full superko history")
    _require(observation.get("rules_search_state_contains_full_superko_history") is True, "Rules/search state must contain full superko history")

    search = profile["search_contracts"]
    _require(search["self_play"]["contract_id"] != search["golden_arena"]["contract_id"], "Self-play and Golden Arena must have separate contract IDs")
    arena = search["golden_arena"]
    _require(arena.get("root_noise") is False, "Golden Arena root noise must be OFF")
    _require(arena.get("fast_search") is False, "Golden Arena fast search must be OFF")
    _require(arena.get("move_temperature") == 0, "Golden Arena move temperature must be zero")
    _require(arena.get("root_policy_temperature") is False, "Golden Arena root policy temperature must be OFF")
    _require(arena.get("resign") is False, "Golden Arena resign must be OFF")
    _require(arena.get("search_settings_source") == "arena-contract-not-checkpoint", "Checkpoint must not control Arena search settings")

    _require(profile["legacy_inheritance"].get("japanese_v3_defaults") is False, "Japanese-V3 defaults must not be inherited")
    _require(profile["legacy_inheritance"].get("implicit_defaults") is False, "Implicit defaults are forbidden")

    if verify_fingerprints:
        expected = computed_fingerprints(profile)
        actual = {
            "topology": topology["fingerprint"],
            "rules": rules["fingerprint"],
            "observation": observation["fingerprint"],
            "targets": targets["fingerprint"],
            "experiment": profile["experiment_fingerprint"],
        }
        _require(actual == expected, f"Experiment fingerprint mismatch: expected {expected}, got {actual}")


def load_profile(path: str | Path | None = None, *, verify_fingerprints: bool = True) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else _repo_root() / PROFILE_RELATIVE_PATH
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    validate_profile(profile, verify_fingerprints=verify_fingerprints)
    return profile


def target_contract_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return compute_target_fingerprint(left) == compute_target_fingerprint(right)


def training_outcome_is_eligible(profile: Mapping[str, Any], outcome: str) -> bool:
    if outcome in profile["results"]["technical"]:
        return False
    return outcome in profile["results"]["rule_level"]
