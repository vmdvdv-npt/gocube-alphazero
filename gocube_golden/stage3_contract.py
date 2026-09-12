"""Immutable Stage-3 training identity and fail-closed contract checks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .arena_contract import SEARCH_CONTRACT_FINGERPRINT, SEARCH_IMPLEMENTATION_ID
from .experiment_profile import (
    EXPERIMENT_FINGERPRINT,
    PROFILE_ID as GOLDEN_V2_PROFILE_ID,
    RULES_FINGERPRINT,
    RULES_PROFILE_ID,
    TOPOLOGY_FINGERPRINT,
)
from .neural import OBSERVATION_FINGERPRINT
from .state import BASELINE_KOMI, LEGACY_FORBIDDEN_KOMI


PROFILE_ID = "gocube-torus-golden-training-v1"
PROFILE_SCHEMA_VERSION = 1
TARGET_CONTRACT_ID = "gocube-wdl-side-to-move-v1"
TARGET_FINGERPRINT = "sha256:6dffad74c4f832741f9cd522aa407f3e88d6f6159ebe56c3dc226a52b4145e41"
PROFILE_PATH = Path("configs/gocube/torus_golden_training_v1.json")
SELFPLAY_CONTRACT_ID = "golden-selfplay-search-v1"
SELFPLAY_CONTRACT_FINGERPRINT = "sha256:e5295418527562d91bacddb10b44ac04a1faff0fa8d1ecbf31c982de0ed481d0"
ARENA_CONTRACT_ID = "golden-arena-search-v1"
ARCHITECTURE_ID = "GoldenGraphNetV1"
PASS_INDEX = 25


class Stage3ContractError(ValueError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def profile_fingerprint(profile: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(profile))
    payload.pop("profile_fingerprint", None)
    return fingerprint(payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage3ContractError(message)


def validate_profile(profile: Mapping[str, Any], *, verify_fingerprint: bool = True) -> None:
    _require(profile.get("profile_id") == PROFILE_ID, "Stage-3 profile id drift")
    _require(profile.get("schema_version") == PROFILE_SCHEMA_VERSION, "Stage-3 profile schema drift")
    _require(profile.get("fingerprint_algorithm") == "sha256-canonical-json-v1", "Stage-3 fingerprint algorithm drift")
    frozen = profile.get("frozen_identities")
    _require(isinstance(frozen, Mapping), "Stage-3 frozen identities are missing")
    _require(frozen.get("golden_v2_profile_id") == GOLDEN_V2_PROFILE_ID, "Stage-3 Golden v2 identity drift")
    _require(frozen.get("golden_v2_experiment_fingerprint") == EXPERIMENT_FINGERPRINT, "Stage-3 Golden v2 fingerprint drift")
    _require(frozen.get("rules_profile_id") == RULES_PROFILE_ID, "Stage-3 rules profile drift")
    _require(frozen.get("rules_fingerprint") == RULES_FINGERPRINT, "Stage-3 rules fingerprint drift")
    _require(frozen.get("topology_fingerprint") == TOPOLOGY_FINGERPRINT, "Stage-3 topology fingerprint drift")
    _require(profile.get("komi") == BASELINE_KOMI, "Stage-3 komi must be exactly 0.5")
    _require(profile.get("komi") != LEGACY_FORBIDDEN_KOMI, "Stage-3 rejects legacy komi 7.5")

    observation = profile.get("observation")
    _require(isinstance(observation, Mapping), "Stage-3 observation contract is missing")
    _require(observation.get("schema_id") == "gocube-torus-golden-training-observation-v1", "Stage-3 observation schema drift")
    _require(observation.get("layout") == "[channels,points]", "Stage-3 observation layout drift")
    _require(observation.get("channels") == ["own_stones", "opponent_stones", "side_to_move_color", "previous_pass", "legal_point_mask", "komi"], "Stage-3 observation channels drift")
    _require(observation.get("shape") == [6, 25], "Stage-3 observation shape drift")
    _require(observation.get("action_count") == 26 and observation.get("pass_index") == PASS_INDEX, "Stage-3 action space drift")
    _require(observation.get("fingerprint") == OBSERVATION_FINGERPRINT, "Stage-3 observation fingerprint drift")

    target = profile.get("target")
    _require(isinstance(target, Mapping), "Stage-3 target contract is missing")
    _require(target.get("contract_id") == TARGET_CONTRACT_ID, "Stage-3 target id drift")
    _require(target.get("fingerprint") == TARGET_FINGERPRINT, "Stage-3 target fingerprint drift")
    _require(target.get("value_vector") == ["WIN", "DRAW", "LOSS"], "Stage-3 value vector drift")
    _require(target.get("perspective") == "side-to-move", "Stage-3 value perspective drift")
    _require(target.get("policy_source") == "root-visits-before-action", "Stage-3 policy target drift")
    _require("NO_RESULT" not in canonical_json(target), "Legacy NO_RESULT target semantics are forbidden")

    network = profile.get("network")
    _require(isinstance(network, Mapping), "Stage-3 network contract is missing")
    _require(network.get("architecture_id") == ARCHITECTURE_ID, "Stage-3 network architecture drift")
    _require(network.get("hidden") == 64 and network.get("blocks") == 4, "Stage-3 network dimensions drift")
    _require(network.get("heads") == {"policy": [26], "value": [3]}, "Stage-3 network heads drift")

    selfplay = profile.get("self_play")
    _require(isinstance(selfplay, Mapping), "Stage-3 self-play contract is missing")
    _require(selfplay.get("contract_id") == SELFPLAY_CONTRACT_ID, "Stage-3 self-play contract id drift")
    _require(selfplay.get("fingerprint") == SELFPLAY_CONTRACT_FINGERPRINT, "Stage-3 self-play contract fingerprint drift")
    _require(selfplay.get("simulations") == 64 and selfplay.get("cpuct") == 1.25 and selfplay.get("fpu") == 0.0, "Stage-3 self-play PUCT drift")
    _require(selfplay.get("root_noise") is True and selfplay.get("dirichlet_epsilon") == 0.25 and selfplay.get("dirichlet_alpha") == 0.30, "Stage-3 self-play noise drift")
    _require(selfplay.get("temperature_plies") == [1, 8] and selfplay.get("temperature_after") == 0.0, "Stage-3 self-play temperature drift")
    _require(selfplay.get("watchdog") == 500 and selfplay.get("komi") == 0.5, "Stage-3 self-play watchdog/komi drift")

    arena = profile.get("arena")
    _require(isinstance(arena, Mapping), "Stage-3 Arena contract is missing")
    _require(arena.get("contract_id") == ARENA_CONTRACT_ID, "Stage-3 Arena contract drift")
    _require(arena.get("search_implementation_id") == SEARCH_IMPLEMENTATION_ID, "Stage-3 Arena implementation drift")
    _require(arena.get("search_contract_fingerprint") == SEARCH_CONTRACT_FINGERPRINT, "Stage-3 Arena search fingerprint drift")
    _require(arena.get("simulations") == 64 and arena.get("cpuct") == 1.25 and arena.get("fpu") == 0.0, "Stage-3 Arena PUCT drift")
    _require(arena.get("root_noise") is False and arena.get("fast_search") is False and arena.get("resign") is False and arena.get("move_temperature") == 0.0, "Stage-3 Arena must be noise-free")
    _require(arena.get("watchdog") == 500, "Stage-3 Arena watchdog drift")

    training = profile.get("training")
    _require(isinstance(training, Mapping), "Stage-3 training schedule is missing")
    _require(training.get("optimizer") == "Adam" and training.get("learning_rate") == 0.001 and training.get("weight_decay") == 0.0, "Stage-3 optimizer drift")
    _require(training.get("batch_size") == 128 and training.get("optimizer_updates_per_chunk") == 200, "Stage-3 update schedule drift")
    _require(training.get("games_per_chunk") == 16 and training.get("chunks") == 4, "Stage-3 canonical schedule drift")
    _require(training.get("replay_policy") == "cumulative", "Stage-3 replay policy drift")

    seeds = profile.get("seeds")
    _require(isinstance(seeds, Mapping), "Stage-3 seeds are missing")
    for key in ("model_init_seed", "selfplay_master_seed", "arena_master_seed"):
        _require(isinstance(seeds.get(key), int), f"Stage-3 {key} must be an integer")

    if verify_fingerprint:
        expected = profile_fingerprint(profile)
        _require(profile.get("profile_fingerprint") == expected, f"Stage-3 profile fingerprint mismatch: expected {expected}")


def load_profile(path: str | Path | None = None, *, verify_fingerprint: bool = True) -> dict[str, Any]:
    target = Path(path) if path is not None else Path(__file__).resolve().parents[1] / PROFILE_PATH
    with target.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    validate_profile(profile, verify_fingerprint=verify_fingerprint)
    return profile


def validate_checkpoint_metadata(metadata: Mapping[str, Any], *, profile: Mapping[str, Any] | None = None) -> None:
    """Validate the semantic checkpoint passport before loading parameters."""
    from .arena_contract import reject_checkpoint_arena_overrides

    reject_checkpoint_arena_overrides(metadata)
    active = profile or load_profile()
    required = (
        "checkpoint_schema_version",
        "architecture_id",
        "architecture_config",
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
        "training_profile_id",
        "training_profile_fingerprint",
        "parent_or_source_run_identity",
        "model_hash",
    )
    missing = [key for key in required if key not in metadata]
    _require(not missing, "Golden checkpoint metadata is incomplete: " + ", ".join(missing))
    _require(metadata.get("checkpoint_schema_version") == 1, "Golden checkpoint schema mismatch")
    _require(metadata.get("architecture_id") == active["network"]["architecture_id"], "Golden checkpoint architecture mismatch")
    _require(metadata.get("rules_profile_id") == active["frozen_identities"]["rules_profile_id"], "Golden checkpoint rules profile mismatch")
    _require(metadata.get("rules_fingerprint") == active["frozen_identities"]["rules_fingerprint"], "Golden checkpoint rules fingerprint mismatch")
    _require(metadata.get("topology_fingerprint") == active["frozen_identities"]["topology_fingerprint"], "Golden checkpoint topology fingerprint mismatch")
    _require(metadata.get("board_size") == [5, 5] and metadata.get("point_id_order_identity") == "row-major-yx:point_id=y*width+x", "Golden checkpoint board identity mismatch")
    _require(metadata.get("komi") == 0.5, "Golden checkpoint komi must be exactly 0.5")
    _require(metadata.get("observation_schema_id") == active["observation"]["schema_id"], "Golden checkpoint observation schema mismatch")
    _require(metadata.get("observation_schema_version") == active["observation"]["schema_version"], "Golden checkpoint observation version mismatch")
    _require(metadata.get("observation_fingerprint") == active["observation"]["fingerprint"], "Golden checkpoint observation fingerprint mismatch")
    _require(metadata.get("target_contract_id") == active["target"]["contract_id"], "Golden checkpoint target contract mismatch")
    _require(metadata.get("target_contract_version") == active["target"]["contract_version"], "Golden checkpoint target version mismatch")
    _require(metadata.get("target_fingerprint") == active["target"]["fingerprint"], "Golden checkpoint target fingerprint mismatch")
    _require(metadata.get("value_head_semantics") == "side-to-move:[WIN,DRAW,LOSS]", "Golden checkpoint value semantics mismatch")
    _require(metadata.get("network_heads_and_shapes") == {"policy": [26], "value": [3]}, "Golden checkpoint head shapes mismatch")
    _require(metadata.get("training_profile_id") == active["profile_id"], "Golden checkpoint training profile mismatch")
    _require(metadata.get("training_profile_fingerprint") == active["profile_fingerprint"], "Golden checkpoint training profile fingerprint mismatch")
    _require(isinstance(metadata.get("parent_or_source_run_identity"), str) and bool(metadata["parent_or_source_run_identity"]), "Golden checkpoint parent identity is missing")
    _require(bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(metadata.get("model_hash")))), "Golden checkpoint model_hash is malformed")
    if "artifact_sha256" in metadata:
        _require(bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(metadata["artifact_sha256"]))), "Golden checkpoint artifact hash is malformed")
