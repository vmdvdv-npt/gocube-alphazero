"""Machine-checked semantic contract for the standalone Golden Torus 9×9 line.

The existing Golden 5×5 profile remains frozen.  This module deliberately gives
9×9 a new identity, so a 5×5 checkpoint, replay row, rules fingerprint, or
observation cannot be accepted by the 9×9 pipeline by accident.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .arena_contract import resolve_arena_watchdog
from .state import BASELINE_KOMI, RULES_PROFILE_ID, rules_fingerprint_for
from .topology import TORUS_9X9


TORUS9_OBSERVATION_SCHEMA_ID = "gocube-torus9-golden-observation-v1"
TORUS9_OBSERVATION_SCHEMA_VERSION = 1
TORUS9_TARGET_CONTRACT_ID = "gocube-torus9-wdl-side-to-move-v1"
TORUS9_TARGET_CONTRACT_VERSION = 1
TORUS9_ARENA_CONTRACT_ID = "torus9-golden-arena-search-v2"
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

# The v2 stable-learning profile above is retained for historical
# reproducibility.  New Torus9 work resolves this explicitly versioned
# current profile instead of inheriting any of those defaults.
TORUS9_CURRENT_PROFILE_ID = "gocube-torus9-golden-v3"
TORUS9_CURRENT_PROFILE_PATH = "configs/gocube/torus9_golden_current_v3.json"
TORUS9_CURRENT_SCHEMA_VERSION = 1
TORUS9_CURRENT_SELFPLAY_CONTRACT_ID = "torus9-golden-current-selfplay-search-v1"
TORUS9_CURRENT_ARCHITECTURE_ID = "GoldenGraphNetV2-Torus9"
TORUS9_CURRENT_HIDDEN = 80
TORUS9_CURRENT_BLOCKS = 8
TORUS9_CURRENT_OWNERSHIP = True
TORUS9_CURRENT_SCORE = True
TORUS9_CURRENT_DIRICHLET_ALPHA = 0.11
TORUS9_CURRENT_MODEL_INIT_SEED = 202609131001
TORUS9_CURRENT_SELFPLAY_MASTER_SEED = 202609131002
TORUS9_CURRENT_TRAINING_MASTER_SEED = 202609131003
TORUS9_CURRENT_ARENA_MASTER_SEED = 202609131004
TORUS9_CURRENT_EVALUATION_MASTER_SEED = 202609131005
TORUS9_CURRENT_PROFILE_FINGERPRINT = "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775"
TORUS9_CURRENT_CONTENT_FINGERPRINT = "sha256:7e97c50e1697641fb8f5b9a3566144f0a58c105e3b688940f42e7b6154fb0831"
# This is the immutable Golden training-lineage base.  Refactor commits are
# recorded separately in checkpoint code/tree provenance fields.
TORUS9_GOLDEN_LINEAGE_BASE_COMMIT = "53946d0c84fca5a6f81a387bfd399ea62e34b088"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


TORUS9_CURRENT_TARGET_FINGERPRINT = fingerprint({
    "contract_id": TORUS9_TARGET_CONTRACT_ID,
    "contract_version": TORUS9_TARGET_CONTRACT_VERSION,
    "value_vector": ["WIN", "DRAW", "LOSS"],
    "perspective": "side-to-move",
    "policy_source": "root-visits-before-chosen-move",
    "ownership_auxiliary": True,
    "score_auxiliary": True,
    "technical_outcomes": "exclude",
})


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


def current_torus9_selfplay_contract_fingerprint(
    alpha: float = TORUS9_CURRENT_DIRICHLET_ALPHA,
) -> str:
    if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("Current Torus 9×9 Dirichlet alpha must be positive and finite")
    return fingerprint(
        {
            "contract_id": TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
            "search_implementation_id": TORUS9_SEARCH_IMPLEMENTATION_ID,
            "simulations": 64,
            "cpuct": 1.25,
            "fpu": 0.0,
            "root_noise": True,
            "dirichlet_epsilon": 0.25,
            "dirichlet_alpha": float(alpha),
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


def _validate_current_torus9_profile(profile: Mapping[str, Any]) -> None:
    """Validate the current profile without weakening the preserved v2 one."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(profile.get("profile_id") == TORUS9_CURRENT_PROFILE_ID, "current profile: Torus 9×9 profile id drift")
    require(profile.get("schema_version") == TORUS9_CURRENT_SCHEMA_VERSION, "Current Torus 9×9 schema drift")
    topology = profile.get("topology", {})
    require(isinstance(topology, Mapping), "Current Torus topology section is malformed")
    require(topology.get("topology_id") == TORUS_9X9.topology_id, "Current Torus 9×9 topology drift")
    require(topology.get("width") == 9 and topology.get("height") == 9, "Current Torus 9×9 dimensions drift")
    require(topology.get("point_count") == TORUS9_POINT_COUNT, "Current Torus 9×9 point count drift")
    require(topology.get("wrap_x") is True and topology.get("wrap_y") is True, "Current Torus wrapping drift")
    require(topology.get("fingerprint") == TORUS_9X9.fingerprint, "Current Torus topology fingerprint drift")

    rules = profile.get("rules", {})
    require(isinstance(rules, Mapping), "Current Torus rules section is malformed")
    require(rules.get("profile_id") == RULES_PROFILE_ID, "Current Torus rules profile drift")
    require(rules.get("fingerprint") == TORUS9_RULES_FINGERPRINT, "Current Torus rules fingerprint drift")
    require(rules.get("komi") == TORUS9_KOMI, "Current Torus komi must be exactly 0.5")
    require(rules.get("ko") == "positional-superko", "Current Torus superko drift")
    require(rules.get("suicide") == "forbidden", "Current Torus suicide drift")
    require(rules.get("pass", {}).get("terminal_after_consecutive_passes") == 2, "Current Torus two-pass drift")
    require(rules.get("scoring", {}).get("method") == "exact-graph-area", "Current Torus scoring drift")
    require(rules.get("benson_auto_ending") is False, "Current automatic-ending setting drift")

    observation = profile.get("observation", {})
    require(isinstance(observation, Mapping), "Current Torus observation section is malformed")
    require(observation.get("shape") == [6, TORUS9_POINT_COUNT], "Current observation shape drift")
    require(observation.get("action_count") == TORUS9_ACTION_COUNT, "Current action count drift")
    require(observation.get("pass_index") == TORUS9_PASS_INDEX, "Current PASS index drift")
    require(observation.get("channels") == [
        "own_stones", "opponent_stones", "side_to_move_color", "previous_pass",
        "legal_point_mask", "komi",
    ], "Current observation channels drift")
    require(observation.get("fingerprint") == TORUS9_OBSERVATION_FINGERPRINT, "Current observation fingerprint drift")

    target = profile.get("target", {})
    require(isinstance(target, Mapping), "Current Torus target section is malformed")
    require(target.get("contract_id") == TORUS9_TARGET_CONTRACT_ID, "Current target contract drift")
    require(target.get("value_vector") == ["WIN", "DRAW", "LOSS"], "Current WDL vector drift")
    require(target.get("perspective") == "side-to-move", "Current WDL perspective drift")
    require(target.get("policy_source") == "root-visits-before-chosen-move", "Current policy target drift")
    require(target.get("ownership_auxiliary") is True and target.get("score_auxiliary") is True, "Current auxiliary targets must be ON")
    require(target.get("technical_outcomes") == "exclude", "Current technical target policy drift")
    require(target.get("fingerprint") == TORUS9_CURRENT_TARGET_FINGERPRINT, "Current target fingerprint drift")

    network = profile.get("network", {})
    require(isinstance(network, Mapping), "Current Torus network section is malformed")
    require(network.get("architecture_id") == TORUS9_CURRENT_ARCHITECTURE_ID, "Current Torus architecture drift")
    require(network.get("hidden") == TORUS9_CURRENT_HIDDEN and network.get("blocks") == TORUS9_CURRENT_BLOCKS, "Current Torus capacity drift")
    require(network.get("input_channels") == 6 and network.get("point_count") == TORUS9_POINT_COUNT, "Current Torus input drift")
    require(network.get("heads") == {"policy": [82], "value": [3], "ownership": [81, 3], "score": [1]}, "Current Torus heads drift")
    require(network.get("explicit_symmetry_augmentation") is False, "Current symmetry augmentation must be OFF")

    self_play = profile.get("self_play", {})
    require(isinstance(self_play, Mapping), "Current Torus self-play section is malformed")
    require(self_play.get("contract_id") == TORUS9_CURRENT_SELFPLAY_CONTRACT_ID, "Current self-play contract id drift")
    require(self_play.get("fingerprint") == current_torus9_selfplay_contract_fingerprint(TORUS9_CURRENT_DIRICHLET_ALPHA), "Current self-play fingerprint drift")
    require(self_play.get("games_per_iteration") == 64, "Current games/iteration drift")
    require(self_play.get("mcts_simulations") == 64 and self_play.get("cpuct") == 1.25 and self_play.get("fpu") == 0.0, "Current search drift")
    require(self_play.get("root_noise") is True and self_play.get("dirichlet_epsilon") == 0.25, "Current root noise drift")
    require(self_play.get("dirichlet_alpha") == TORUS9_CURRENT_DIRICHLET_ALPHA, "Current Dirichlet alpha must be 0.11")
    require(self_play.get("temperature") == "1.0 on plies 1–8, then 0", "Current temperature schedule drift")
    require(self_play.get("fast_search") is False and self_play.get("resign") is False, "Current self-play switches drift")
    require(self_play.get("watchdog") == TORUS9_MOVE_LIMIT and self_play.get("workers") == TORUS9_WORKERS, "Current self-play limits drift")
    require(self_play.get("existing_batch_size") == TORUS9_BATCH_SIZE and self_play.get("komi") == TORUS9_KOMI, "Current self-play batch/komi drift")

    training = profile.get("training", {})
    require(isinstance(training, Mapping), "Current Torus training section is malformed")
    require(training.get("optimizer") == "Adam" and training.get("learning_rate") == 0.001 and training.get("weight_decay") == 0.0, "Current optimizer drift")
    require(training.get("batch_size") == TORUS9_BATCH_SIZE, "Current training batch drift")
    require(training.get("optimizer_steps_per_iteration") == TORUS9_OPTIMIZER_STEPS_PER_ITERATION, "Current optimizer budget drift")
    require(training.get("samples_consumed_per_iteration") == TORUS9_TRAINING_SAMPLES_PER_ITERATION, "Current sample budget drift")
    require(training.get("lr_scheduler") is None and training.get("model_gating") is False, "Current scheduler/gating drift")

    replay = profile.get("replay", {})
    require(isinstance(replay, Mapping), "Current Torus replay section is malformed")
    require(replay.get("window") == "rolling last 3 generations", "Current replay window drift")
    require(replay.get("generations") == TORUS9_ROLLING_GENERATIONS and replay.get("cap") == TORUS9_MAX_REPLAY_POSITIONS, "Current replay cap drift")
    require(replay.get("sampling") == "deterministic / reproducible", "Current replay sampling drift")

    execution = profile.get("execution_sweep", {})
    require(isinstance(execution, Mapping), "Current Torus execution section is malformed")
    require(execution.get("baseline") == {"coalescing": False, "batch_cap": 1, "wait_ms": 0}, "Current baseline execution drift")
    require(
        execution.get("candidates") == [
            {"id": "cap8-wait1", "coalescing": True, "batch_cap": 8, "wait_ms": 1},
            {"id": "cap8-wait2", "coalescing": True, "batch_cap": 8, "wait_ms": 2},
            {"id": "cap16-wait1", "coalescing": True, "batch_cap": 16, "wait_ms": 1},
            {"id": "cap16-wait2", "coalescing": True, "batch_cap": 16, "wait_ms": 2},
            {"id": "cap16-wait4", "coalescing": True, "batch_cap": 16, "wait_ms": 4},
        ],
        "Current execution sweep candidates drift",
    )
    require(execution.get("iteration_schedule") == [
        "baseline", "baseline", "cap8-wait1", "cap8-wait2", "cap16-wait1", "cap16-wait2", "cap16-wait4",
    ], "Current execution sweep schedule drift")
    require(execution.get("warmup_iterations") == [1], "Current warm-up iteration drift")
    require(execution.get("selection_iterations") == [2, 3, 4, 5, 6, 7], "Current selection iterations drift")
    require(execution.get("benchmark_games") == 0, "Separate benchmark games are forbidden")

    seeds = profile.get("seeds", {})
    require(isinstance(seeds, Mapping), "Current Torus seeds section is malformed")
    for name in (
        "model_init_seed", "selfplay_master_seed", "training_master_seed",
        "arena_master_seed", "evaluation_master_seed",
    ):
        require(isinstance(seeds.get(name), int) and seeds[name] > 0, f"Current seed missing: {name}")
    require(seeds.get("model_init_seed") == TORUS9_CURRENT_MODEL_INIT_SEED, "Current model-init seed drift")
    require(seeds.get("selfplay_master_seed") == TORUS9_CURRENT_SELFPLAY_MASTER_SEED, "Current self-play seed drift")
    require(seeds.get("training_master_seed") == TORUS9_CURRENT_TRAINING_MASTER_SEED, "Current training seed drift")
    require(seeds.get("arena_master_seed") == TORUS9_CURRENT_ARENA_MASTER_SEED, "Current Arena seed drift")
    require(seeds.get("evaluation_master_seed") == TORUS9_CURRENT_EVALUATION_MASTER_SEED, "Current evaluation seed drift")


def current_torus9_content_fingerprint(profile: Mapping[str, Any]) -> str:
    return fingerprint({
        key: value
        for key, value in profile.items()
        if key not in {"profile_fingerprint", "content_fingerprint"}
    })


def current_torus9_profile_fingerprint(profile: Mapping[str, Any]) -> str:
    value = profile.get("profile_fingerprint")
    if value != TORUS9_CURRENT_PROFILE_FINGERPRINT:
        raise ValueError("Current Torus 9×9 profile lineage fingerprint drift")
    return TORUS9_CURRENT_PROFILE_FINGERPRINT


def validate_torus9_current_profile(
    profile: Mapping[str, Any],
    *,
    verify_fingerprint: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the actual current Torus9 profile payload.

    This is the shared validation boundary for production callers.  The
    embedded ``profile_fingerprint`` is checked only after the scientific
    sections and their derived fingerprints have been validated; it is never
    used as evidence that the JSON payload is canonical by itself.
    """
    if not isinstance(profile, Mapping):
        raise ValueError("Current Torus 9×9 profile must be a JSON object")
    _validate_current_torus9_profile(profile)
    source = profile.get("golden_source", {})
    if not isinstance(source, Mapping):
        raise ValueError("Current Torus Golden source section is malformed")
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    snapshot_path = root / str(source.get("snapshot_path", ""))
    if not snapshot_path.is_file():
        raise ValueError(f"Current Torus 9×9 Golden snapshot is missing: {snapshot_path}")
    snapshot_digest = "sha256:" + hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if source.get("snapshot_sha256") != snapshot_digest:
        raise ValueError("Current Torus 9×9 Golden snapshot fingerprint drift")
    if verify_fingerprint:
        current_torus9_profile_fingerprint(profile)
        expected_content = current_torus9_content_fingerprint(profile)
        if profile.get("content_fingerprint") != expected_content:
            raise ValueError(f"Current Torus 9×9 content fingerprint mismatch: {expected_content}")
        if expected_content != TORUS9_CURRENT_CONTENT_FINGERPRINT:
            raise ValueError("Current Torus 9×9 canonical content fingerprint drift")
    return dict(profile)


def load_torus9_current_profile(path: str | Path | None = None, *, verify_fingerprint: bool = True) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / TORUS9_CURRENT_PROFILE_PATH
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    return validate_torus9_current_profile(profile, verify_fingerprint=verify_fingerprint)
