from __future__ import annotations

from collections.abc import Mapping


PARAMETER_ORIGINS = {
    "gocube_katago_reference_commit": "katago_reference",
    "gocube_katago_search_reference_commit": "katago_reference",
    "gocube_katago_search_contract": "katago_reference",
    "gocube_katago_exploration_contract": "katago_reference",
    "gocube_cpuct_exploration": "katago_reference",
    "gocube_root_fpu_reduction": "katago_reference",
    "gocube_fpu_parent_weight_by_visited_policy": "katago_reference",
    "gocube_fpu_parent_weight_by_visited_policy_pow": "katago_reference",
    "gocube_root_ending_bonus_points": "katago_reference",
    "gocube_cleanup_training_prob": "katago_reference",
    "gocube_pass_alive_auto_end_probability": "katago_reference",
    "gocube_seki_fork_hack_probability": "katago_reference",
    "gocube_topology": "topology_adaptation",
    "gocube_size": "topology_adaptation",
    "gocube_rule_set": "framework",
    "gocube_komi": "framework",
    "gocube_rules_fingerprint": "topology_adaptation",
}


def parameter_origin(key: str) -> str:
    if key in PARAMETER_ORIGINS:
        return PARAMETER_ORIGINS[key]
    if key.startswith(("gocube_", "cpuct", "fpu_", "root_")):
        return "framework"
    if key in {"topology", "size", "point_count", "komi", "rules_fingerprint"}:
        return "topology_adaptation"
    if key in {"cuda", "python_executable", "python_version", "platform", "kernel", "machine"}:
        return "runtime"
    if key in {"workers", "process_batch_size", "numIters", "gamesPerIteration", "numMCTSSims", "arenaMCTSSims"}:
        return "experiment"
    return "framework"


def classify_parameters(config: Mapping[str, object]) -> dict[str, str]:
    result = {str(key): parameter_origin(str(key)) for key in config}
    missing = sorted(key for key, origin in result.items() if not origin)
    if missing:
        raise ValueError(f"Unclassified effective configuration keys: {missing}")
    return result
