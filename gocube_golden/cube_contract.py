"""Frozen semantic and experiment contract for the Cube 4x4 Golden proof."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .cube_arena import CUBE_ARENA_FINGERPRINT, CUBE_ARENA_SEARCH
from .cube_neural import (
    CUBE_ACTION_COUNT,
    CUBE_OBSERVATION_CHANNEL_COUNT,
    CUBE_OBSERVATION_CHANNELS,
    CUBE_OBSERVATION_FINGERPRINT,
    CUBE_OBSERVATION_SCHEMA_ID,
    CUBE_OBSERVATION_SCHEMA_VERSION,
    GoldenCubeGraphNetV1,
    cube_count_parameters,
)
from .cube_topology import (
    CUBE4_GEOMETRY_FINGERPRINT,
    CUBE4_TOPOLOGY,
    CUBE4_TOPOLOGY_FINGERPRINT,
    CUBE4_TOPOLOGY_ID,
    GEOMETRY_SCHEMA_ID,
)
from .cube_training import (
    CUBE_ARENA_CONTRACT_ID,
    CUBE_SELFPLAY_CONTRACT_ID,
    CUBE_TARGET_CONTRACT_ID,
    CUBE_TARGET_FINGERPRINT,
    CUBE_WATCHDOG,
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    cube_initial_state,
)

CUBE_PROFILE_ID = "gocube-cube4-golden-training-v1"
CUBE_PROFILE_SCHEMA_VERSION = 1
CUBE_PROFILE_PATH = Path("configs/gocube/cube4_golden_training_v1.json")
BASELINE_KOMI = 0.5


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def profile_fingerprint(profile: Mapping[str, Any]) -> str:
    payload = {key: copy.deepcopy(value) for key, value in profile.items() if key not in {"profile_fingerprint", "config_sha256"}}
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _target_payload() -> dict[str, object]:
    return {
        "contract_id": CUBE_TARGET_CONTRACT_ID,
        "contract_version": 1,
        "value_vector": ["WIN", "DRAW", "LOSS"],
        "value_perspective": "side-to-move",
        "utility": "P(WIN)-P(LOSS)",
        "child_to_parent_sign_flips": 1,
        "technical_results_are_targets": False,
    }


def build_profile() -> dict[str, Any]:
    network = GoldenCubeGraphNetV1()
    return {
        "profile_id": CUBE_PROFILE_ID,
        "schema_version": CUBE_PROFILE_SCHEMA_VERSION,
        "fingerprint_algorithm": "sha256-canonical-json-v1",
        "topology": {
            "topology_id": CUBE4_TOPOLOGY_ID,
            "topology_version": 1,
            "kind": "cube-surface",
            "faces": 6,
            "face_size": [4, 4],
            "point_count": 96,
            "action_count": 97,
            "point_ordering_fingerprint": CUBE4_TOPOLOGY.point_ordering_fingerprint,
            "fingerprint": CUBE4_TOPOLOGY_FINGERPRINT,
            "independently_derived": True,
            "production_runtime_dependency": False,
        },
        "geometry": {
            "schema_id": GEOMETRY_SCHEMA_ID,
            "fingerprint": CUBE4_GEOMETRY_FINGERPRINT,
            "classes": ["FACE_INTERIOR", "FACE_EDGE", "FACE_CORNER"],
            "class_counts": {"FACE_INTERIOR": 24, "FACE_EDGE": 48, "FACE_CORNER": 24},
            "physical_corners": 8,
            "face_corner_cells_per_physical_corner": 3,
            "seams": 12,
            "cross_face_adjacency_pairs": 48,
            "relation_types": ["SAME_FACE", "CROSS_FACE_SEAM"],
            "corner_distance_buckets": ["corner_distance_0", "corner_distance_1", "corner_distance_2", "corner_distance_3_plus"],
        },
        "rules": {
            "rules_id": "graph-area-v1",
            "fingerprint": cube_initial_state().rules_fingerprint,
            "suicide": "forbidden",
            "ko": "positional-superko",
            "initial_board_in_history": True,
            "side_to_move_irrelevant_to_repetition": True,
            "pass_exempt_from_repetition": True,
            "pass_appends_board": False,
            "double_pass_terminal": True,
            "komi": BASELINE_KOMI,
            "scoring": "exact-graph-area",
            "dead_stone_removal": False,
            "cleanup": False,
            "resign_action": False,
        },
        "observation": {
            "schema_id": CUBE_OBSERVATION_SCHEMA_ID,
            "schema_version": CUBE_OBSERVATION_SCHEMA_VERSION,
            "fingerprint": CUBE_OBSERVATION_FINGERPRINT,
            "layout": "[channels,points]",
            "channels": list(CUBE_OBSERVATION_CHANNELS),
            "channel_count": CUBE_OBSERVATION_CHANNEL_COUNT,
            "point_count": 96,
            "action_mask_length": CUBE_ACTION_COUNT,
        },
        "target": _target_payload() | {"fingerprint": CUBE_TARGET_FINGERPRINT},
        "search": {
            "implementation_id": "golden-sequential-puct-v1",
            "simulations": CUBE_ARENA_SEARCH.simulations,
            "cpuct": CUBE_ARENA_SEARCH.cpuct,
            "fpu": CUBE_ARENA_SEARCH.fpu,
            "child_to_parent_sign_flips": 1,
            "terminal_value": "exact Golden graph-area result",
            "batching": False,
            "virtual_loss": False,
        },
        "self_play": {
            "contract_id": CUBE_SELFPLAY_CONTRACT_ID,
            "fingerprint": DEFAULT_CUBE_SELFPLAY_CONTRACT.fingerprint,
            "simulations": 64,
            "cpuct": 1.25,
            "fpu": 0.0,
            "root_noise": True,
            "dirichlet_epsilon": 0.25,
            "dirichlet_alpha": 0.30,
            "temperature_plies": [1, 8],
            "temperature_after": 0.0,
            "fast_search": False,
            "resign": False,
            "watchdog": CUBE_WATCHDOG,
            "komi": BASELINE_KOMI,
            "workers": 16,
            "process_isolated": True,
            "inference_batch_size": 1,
            "inference_coalescing": False,
        },
        "arena": {
            "contract_id": CUBE_ARENA_CONTRACT_ID,
            "fingerprint": CUBE_ARENA_FINGERPRINT,
            "simulations": 64,
            "cpuct": 1.25,
            "fpu": 0.0,
            "root_noise": False,
            "temperature": 0.0,
            "fast_search": False,
            "resign": False,
            "watchdog": CUBE_WATCHDOG,
            "inference_batch_size": 1,
            "inference_coalescing": False,
        },
        "network": {
            "architecture_id": network.architecture_id,
            "architecture_config": network.architecture_config,
            "parameter_count": cube_count_parameters(network),
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 64,
            "games_per_chunk": 128,
            "chunks": 4,
            "total_games": 512,
            "replay_policy": "cumulative",
            "sampling": "deterministic-uniform-without-replacement",
            "samples_per_new_position": 1.0,
            "prioritized_replay": False,
            "reanalysis": False,
            "eviction": False,
        },
        "evaluation": {
            "evaluation_id": "cube-golden-evaluation-v1",
            "prefix_lengths": [4, 8, 12, 16, 24, 32, 40, 48],
            "accepted_per_stratum": 8,
            "pairs": 64,
            "diagnostic_pairs": 16,
            "empty_board_control": True,
            "exact_semantic_dedupe": True,
            "symmetry_rejection": False,
            "confidence_interval": "95% Hoeffding bounded mean",
        },
        "seeds": {
            "model_init_seed": 2026091401,
            "selfplay_master_seed": 2026091402,
            "evaluation_seed": 2026091403,
            "arena_master_seed": 2026091404,
        },
    }


def _contains_forbidden_komi(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_forbidden_komi(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_forbidden_komi(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and abs(float(value) - 7.5) <= 1e-12


def validate_profile(profile: Mapping[str, Any], *, verify_fingerprint: bool = True) -> None:
    if _contains_forbidden_komi(profile):
        raise ValueError("Cube Golden profile contains forbidden legacy komi 7.5")
    if profile.get("profile_id") != CUBE_PROFILE_ID or profile.get("schema_version") != CUBE_PROFILE_SCHEMA_VERSION:
        raise ValueError("Cube Golden profile identity drift")
    if profile.get("topology", {}).get("fingerprint") != CUBE4_TOPOLOGY_FINGERPRINT:
        raise ValueError("Cube Golden topology fingerprint drift")
    if profile.get("topology", {}).get("point_count") != 96 or profile.get("topology", {}).get("action_count") != 97:
        raise ValueError("Cube Golden point/action count drift")
    if profile.get("geometry", {}).get("fingerprint") != CUBE4_GEOMETRY_FINGERPRINT:
        raise ValueError("Cube Golden geometry fingerprint drift")
    if profile.get("rules", {}).get("komi") != BASELINE_KOMI:
        raise ValueError("Cube Golden komi must be exactly 0.5")
    if profile.get("observation", {}).get("fingerprint") != CUBE_OBSERVATION_FINGERPRINT:
        raise ValueError("Cube Golden observation fingerprint drift")
    if profile.get("target", {}).get("fingerprint") != CUBE_TARGET_FINGERPRINT:
        raise ValueError("Cube Golden target fingerprint drift")
    if profile.get("network", {}).get("architecture_id") != "GoldenCubeGraphNetV1":
        raise ValueError("Cube Golden architecture identity drift")
    if profile.get("self_play", {}).get("watchdog") != CUBE_WATCHDOG:
        raise ValueError("Cube Golden self-play watchdog must be 1920")
    if profile.get("arena", {}).get("watchdog") != CUBE_WATCHDOG:
        raise ValueError("Cube Golden Arena watchdog must be 1920")
    if profile.get("training", {}).get("games_per_chunk") != 128 or profile.get("training", {}).get("chunks") != 4:
        raise ValueError("Cube Golden training schedule drift")
    if profile.get("training", {}).get("batch_size") != 64:
        raise ValueError("Cube Golden batch size must be 64")
    if profile.get("evaluation", {}).get("prefix_lengths") != [4, 8, 12, 16, 24, 32, 40, 48]:
        raise ValueError("Cube Golden evaluation strata drift")
    if verify_fingerprint and profile.get("profile_fingerprint") != profile_fingerprint(profile):
        raise ValueError("Cube Golden profile fingerprint mismatch")


def load_profile(path: str | Path | None = None, *, verify_fingerprint: bool = True) -> dict[str, Any]:
    target = Path(path) if path is not None else Path(__file__).resolve().parents[1] / CUBE_PROFILE_PATH
    with target.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    validate_profile(profile, verify_fingerprint=verify_fingerprint)
    return profile
