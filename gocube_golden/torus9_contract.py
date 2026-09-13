"""Machine-checked semantic contract for the standalone Golden Torus 9×9 line.

The existing Golden 5×5 profile remains frozen.  This module deliberately gives
9×9 a new identity, so a 5×5 checkpoint, replay row, rules fingerprint, or
observation cannot be accepted by the 9×9 pipeline by accident.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .arena_contract import resolve_arena_watchdog
from .state import BASELINE_KOMI, RULES_PROFILE_ID, rules_fingerprint_for
from .topology import TORUS_9X9


TORUS9_PROFILE_ID = "gocube-torus9-stable-learning-v2"
TORUS9_SCHEMA_VERSION = 1
TORUS9_OBSERVATION_SCHEMA_ID = "gocube-torus9-golden-observation-v1"
TORUS9_OBSERVATION_SCHEMA_VERSION = 1
TORUS9_TARGET_CONTRACT_ID = "gocube-torus9-wdl-side-to-move-v1"
TORUS9_TARGET_CONTRACT_VERSION = 1
TORUS9_SELFPLAY_CONTRACT_ID = "torus9-golden-selfplay-search-v1"
TORUS9_ARENA_CONTRACT_ID = "torus9-golden-arena-search-v2"
TORUS9_LEGACY_ARENA_CONTRACT_ID = "torus9-golden-arena-search-v1"
TORUS9_LEGACY_ARENA_CONTRACT_FINGERPRINT = "sha256:4466460601b5e2e036b67dcadcf29ef3b1c20001dcb4d5e0fec341582f07c04f"
TORUS9_SEARCH_IMPLEMENTATION_ID = "golden-sequential-puct-v1"
TORUS9_POINT_COUNT = 81
TORUS9_ACTION_COUNT = 82
TORUS9_PASS_INDEX = 81
TORUS9_KOMI = 0.5
TORUS9_HIDDEN = 64
TORUS9_BLOCKS = 8
TORUS9_ARCHITECTURE_ID = "GoldenGraphNetV2-Torus9-8Block"
TORUS9_WORKERS = 16
TORUS9_BATCH_SIZE = 64
TORUS9_OPTIMIZER_STEPS_PER_ITERATION = 80
TORUS9_TRAINING_SAMPLES_PER_ITERATION = TORUS9_BATCH_SIZE * TORUS9_OPTIMIZER_STEPS_PER_ITERATION
TORUS9_ROLLING_GENERATIONS = 3
TORUS9_MAX_REPLAY_POSITIONS = 20_000
TORUS9_MOVE_LIMIT = 500
TORUS9_ARENA_MOVE_LIMIT = resolve_arena_watchdog((9, 9))
TORUS9_RULES_FINGERPRINT = rules_fingerprint_for(TORUS_9X9, TORUS9_KOMI)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


TORUS9_OBSERVATION_FINGERPRINT = fingerprint(
    {
        "schema_id": TORUS9_OBSERVATION_SCHEMA_ID,
        "schema_version": TORUS9_OBSERVATION_SCHEMA_VERSION,
        "layout": "[channels,points]",
        "channels": [
            "own_stones",
            "opponent_stones",
            "side_to_move_color",
            "previous_pass",
            "legal_point_mask",
            "komi",
        ],
        "shape": [6, TORUS9_POINT_COUNT],
        "action_count": TORUS9_ACTION_COUNT,
        "pass_index": TORUS9_PASS_INDEX,
    }
)
TORUS9_TARGET_FINGERPRINT = fingerprint(
    {
        "contract_id": TORUS9_TARGET_CONTRACT_ID,
        "contract_version": TORUS9_TARGET_CONTRACT_VERSION,
        "value_vector": ["WIN", "DRAW", "LOSS"],
        "perspective": "side-to-move",
        "utility": "P(WIN)-P(LOSS)",
        "child_to_parent_sign_flips": 1,
        "policy_source": "root-visits-before-action",
        "technical_games": "excluded",
    }
)
TORUS9_SELFPLAY_CONTRACT_FINGERPRINT = fingerprint(
    {
        "contract_id": TORUS9_SELFPLAY_CONTRACT_ID,
        "search_implementation_id": TORUS9_SEARCH_IMPLEMENTATION_ID,
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
        "watchdog": TORUS9_MOVE_LIMIT,
        "komi": TORUS9_KOMI,
    }
)
TORUS9_ARENA_CONTRACT_FINGERPRINT = fingerprint(
    {
        "contract_id": TORUS9_ARENA_CONTRACT_ID,
        "search_implementation_id": TORUS9_SEARCH_IMPLEMENTATION_ID,
        "simulations": 64,
        "cpuct": 1.25,
        "fpu": 0.0,
        "root_noise": False,
        "temperature": 0.0,
        "fast_search": False,
        "resign": False,
        "deterministic_tie_break": True,
        "watchdog": TORUS9_ARENA_MOVE_LIMIT,
        "workers": TORUS9_WORKERS,
        "one_game_per_process": True,
        "technical_fail_closed": True,
        "komi": TORUS9_KOMI,
    }
)


def _profile_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if key != "profile_fingerprint"}


def profile_fingerprint(profile: Mapping[str, Any]) -> str:
    return fingerprint(_profile_payload(profile))


def validate_torus9_profile(profile: Mapping[str, Any], *, verify_fingerprint: bool = True) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(profile.get("profile_id") == TORUS9_PROFILE_ID, "Torus 9×9 profile id drift")
    require(profile.get("schema_version") == TORUS9_SCHEMA_VERSION, "Torus 9×9 schema version drift")
    topology = profile.get("topology", {})
    require(topology.get("topology_id") == TORUS_9X9.topology_id, "Torus 9×9 topology id drift")
    require(topology.get("width") == 9 and topology.get("height") == 9, "Torus 9×9 dimensions drift")
    require(topology.get("point_count") == TORUS9_POINT_COUNT, "Torus 9×9 point count drift")
    require(topology.get("fingerprint") == TORUS_9X9.fingerprint, "Torus 9×9 topology fingerprint drift")
    require(topology.get("point_order") == "row-major-yx:point_id=y*width+x", "Point order drift")
    require(topology.get("wrap_x") is True and topology.get("wrap_y") is True, "Torus wrapping drift")
    rules = profile.get("rules", {})
    require(rules.get("profile_id") == RULES_PROFILE_ID, "Torus 9×9 rules profile drift")
    require(rules.get("fingerprint") == TORUS9_RULES_FINGERPRINT, "Torus 9×9 rules fingerprint drift")
    require(rules.get("komi") == TORUS9_KOMI, "Torus 9×9 komi must be exactly 0.5")
    require(rules.get("ko") == "positional-superko", "Torus 9×9 ko semantics drift")
    require(rules.get("suicide") == "forbidden", "Torus 9×9 suicide semantics drift")
    require(rules.get("pass", {}).get("terminal_after_consecutive_passes") == 2, "Two-pass semantics drift")
    require(rules.get("scoring", {}).get("method") == "exact-graph-area", "Graph-area scoring drift")
    observation = profile.get("observation", {})
    require(observation.get("schema_id") == TORUS9_OBSERVATION_SCHEMA_ID, "Observation schema id drift")
    require(observation.get("schema_version") == TORUS9_OBSERVATION_SCHEMA_VERSION, "Observation schema version drift")
    require(observation.get("shape") == [6, TORUS9_POINT_COUNT], "Observation shape drift")
    require(observation.get("action_count") == TORUS9_ACTION_COUNT, "Action count drift")
    require(observation.get("pass_index") == TORUS9_PASS_INDEX, "PASS index drift")
    require(observation.get("fingerprint") == TORUS9_OBSERVATION_FINGERPRINT, "Observation fingerprint drift")
    target = profile.get("target", {})
    require(target.get("contract_id") == TORUS9_TARGET_CONTRACT_ID, "Target contract id drift")
    require(target.get("value_vector") == ["WIN", "DRAW", "LOSS"], "WDL vector drift")
    require(target.get("perspective") == "side-to-move", "WDL perspective drift")
    require(target.get("fingerprint") == TORUS9_TARGET_FINGERPRINT, "Target fingerprint drift")
    network = profile.get("network", {})
    require(network.get("architecture_id") == TORUS9_ARCHITECTURE_ID, "Torus 9×9 architecture identity drift")
    require(network.get("hidden") == TORUS9_HIDDEN and network.get("blocks") == TORUS9_BLOCKS, "Network capacity drift")
    require(network.get("heads") == {"policy": [TORUS9_ACTION_COUNT], "value": [3]}, "Network head shape drift")
    require(network.get("ownership") is False and network.get("score") is False, "Auxiliary heads are forbidden in canonical proof")
    selfplay = profile.get("self_play", {})
    require(selfplay.get("simulations") == 64 and selfplay.get("cpuct") == 1.25 and selfplay.get("fpu") == 0.0, "Self-play search drift")
    require(selfplay.get("root_noise") is True and selfplay.get("dirichlet_epsilon") == 0.25 and selfplay.get("dirichlet_alpha") == 0.30, "Self-play root noise drift")
    require(selfplay.get("workers") == TORUS9_WORKERS and selfplay.get("batch_size") == TORUS9_BATCH_SIZE, "Self-play execution drift")
    require(selfplay.get("komi") == TORUS9_KOMI and selfplay.get("fast_sims") is False, "Self-play komi/fast-sims drift")
    arena = profile.get("arena", {})
    require(arena.get("contract_id") == TORUS9_ARENA_CONTRACT_ID, "Arena contract id drift")
    require(arena.get("workers") == TORUS9_WORKERS and arena.get("temperature") == 0.0 and arena.get("noise") is False, "Arena execution drift")
    require(arena.get("watchdog") == TORUS9_ARENA_MOVE_LIMIT, "Arena watchdog scaling drift")
    require(arena.get("fingerprint") == TORUS9_ARENA_CONTRACT_FINGERPRINT, "Arena fingerprint drift")
    require(profile.get("canonical_games_per_iteration") == 64, "Canonical games/iteration must be fixed to 64")
    replay = profile.get("replay", {})
    require(replay.get("policy") == "rolling-recent-generations", "Torus 9×9 replay policy drift")
    require(replay.get("generations") == TORUS9_ROLLING_GENERATIONS, "Torus 9×9 rolling generation count drift")
    require(replay.get("maximum_positions") == TORUS9_MAX_REPLAY_POSITIONS, "Torus 9×9 replay cap drift")
    training = profile.get("training", {})
    require(training.get("optimizer") == "Adam" and training.get("learning_rate") == 0.001 and training.get("weight_decay") == 0.0, "Torus 9×9 optimizer drift")
    require(training.get("batch_size") == TORUS9_BATCH_SIZE, "Torus 9×9 batch size drift")
    require(training.get("optimizer_steps_per_iteration") == TORUS9_OPTIMIZER_STEPS_PER_ITERATION, "Torus 9×9 optimizer budget drift")
    require(training.get("samples_consumed_per_iteration") == TORUS9_TRAINING_SAMPLES_PER_ITERATION, "Torus 9×9 sample budget drift")
    if verify_fingerprint:
        expected = profile_fingerprint(profile)
        require(profile.get("profile_fingerprint") == expected, f"Torus 9×9 profile fingerprint mismatch: {expected}")


def load_torus9_profile(path: str | Path | None = None, *, verify_fingerprint: bool = True) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "configs/gocube/torus9_golden_learning_v1.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    validate_torus9_profile(profile, verify_fingerprint=verify_fingerprint)
    return profile
