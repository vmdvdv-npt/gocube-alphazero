from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


STATE_SCHEMA_VERSION = 3
AUTO_RESUME_STATUSES = frozenset({"RUNNING", "INTERRUPTED"})
TERMINAL_STATUSES = frozenset({
    "COMPLETE",
    "STOPPED_REGRESSION",
    "STOPPED_NO_IMPROVEMENT",
    "STOPPED_REGRESSION_FINAL",
})

_SWEEP_FLAG_TO_ARG_KEY = {
    "--chosen-move-temperature-halflife": "gocube_chosen_move_temperature_halflife",
    "--root-dirichlet-noise-weight": "gocube_root_dirichlet_noise_weight",
    "--fast-game-prob": "probFastSim",
    "--train-samples-per-new-sample": "gocube_train_samples_per_new_sample",
    "--replay-window-iters": "gocube_replay_window_iters",
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_resumable_experiments(reports_root: Path) -> list[str]:
    candidates: list[tuple[float, str]] = []
    if not reports_root.is_dir():
        return []
    for state_path in reports_root.glob("*/experiment-state.json"):
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if int(payload.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            continue
        if str(payload.get("status")) not in AUTO_RESUME_STATUSES:
            continue
        experiment_id = payload.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            continue
        stamp = float(payload.get("last_update_epoch", payload.get("started_at_epoch", 0.0)))
        candidates.append((stamp, experiment_id))
    candidates.sort(reverse=True)
    return [experiment_id for _, experiment_id in candidates]


def launch_config(cli) -> dict[str, object]:
    return {
        "bootstrap_run": cli.bootstrap_run,
        "device": str(cli.device),
        "arena_batch_wait_ms": float(cli.arena_batch_wait_ms),
        "heldout_positions": int(cli.heldout_positions),
        "health_gate_min_win_rate": float(cli.health_gate_min_win_rate),
        "benchmark_games": int(cli.benchmark_games),
        "skip_performance_benchmark": bool(cli.skip_performance_benchmark),
        "seed": int(cli.seed),
    }


def validate_loaded_state(
    payload: dict[str, Any],
    *,
    experiment_id: str,
    fixed_contract: dict[str, object],
    expected_launch_config: dict[str, object],
) -> None:
    if int(payload.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Experiment state schema {payload.get('schema_version')!r} is not safely resumable; "
            f"expected schema {STATE_SCHEMA_VERSION}. Start a new experiment instead."
        )
    if payload.get("experiment_id") != experiment_id:
        raise RuntimeError("Experiment state id does not match requested experiment id")
    if payload.get("fixed_contract") != fixed_contract:
        raise RuntimeError("Saved experiment fixed contract differs from the current Cube-4 contract")
    if payload.get("launch_config") != expected_launch_config:
        raise RuntimeError(
            "Saved experiment launch configuration differs from this resume command; "
            "resume with the original arguments or start a new experiment."
        )


def begin_action(state: dict[str, Any], key: str, kind: str, details: dict[str, object]) -> None:
    completed = state.setdefault("completed_actions", {})
    existing = completed.get(key)
    if existing is not None:
        if existing.get("kind") != kind or existing.get("details") != details:
            raise RuntimeError(f"Completed action {key} has incompatible resume metadata")
        return
    state["current_action"] = {
        "key": key,
        "kind": kind,
        "details": details,
        "started_at_epoch": time.time(),
    }


def complete_action_once(
    state: dict[str, Any],
    *,
    key: str,
    kind: str,
    details: dict[str, object],
    counter: str | None = None,
    amount: int = 0,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    completed = state.setdefault("completed_actions", {})
    existing = completed.get(key)
    if existing is not None:
        if existing.get("kind") != kind or existing.get("details") != details:
            raise RuntimeError(f"Completed action {key} has incompatible resume metadata")
        return existing
    record: dict[str, object] = {
        "kind": kind,
        "details": details,
        "completed_at_epoch": time.time(),
    }
    if metadata:
        record["metadata"] = metadata
    completed[key] = record
    if counter is not None and amount:
        totals = state.setdefault("totals", {})
        totals[counter] = int(totals.get(counter, 0)) + int(amount)
    current = state.get("current_action")
    if isinstance(current, dict) and current.get("key") == key:
        state.pop("current_action", None)
    return record


def validate_arena_payload(payload: dict[str, Any], expected: dict[str, object]) -> None:
    if int(payload.get("schema_version", -1)) != 3:
        raise RuntimeError("Cached Arena result has unsupported schema")
    exact_keys = (
        "run_a", "iteration_a", "run_b", "iteration_b", "seed", "workers",
        "evaluation_mode", "heldout_suite",
    )
    for key in exact_keys:
        if payload.get(key) != expected.get(key):
            raise RuntimeError(
                f"Cached Arena result mismatch for {key}: {payload.get(key)!r} != {expected.get(key)!r}"
            )
    if not math.isclose(
        float(payload.get("arena_inference_batch_wait_ms", float("nan"))),
        float(expected["arena_inference_batch_wait_ms"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Cached Arena result has a different inference batch wait")
    expected_games = int(expected["number_of_games"])
    if int(payload.get("number_of_games", -1)) != expected_games:
        raise RuntimeError("Cached Arena result has a different number of games")
    outcomes = sum(int(payload.get(key, 0)) for key in ("wins", "losses", "draws", "no_results"))
    if outcomes != expected_games:
        raise RuntimeError("Cached Arena result outcome count is inconsistent")
    by_color = payload.get("by_color") or {}
    color_games = sum(int((by_color.get(color) or {}).get("games", 0)) for color in ("black", "white"))
    if color_games != expected_games:
        raise RuntimeError("Cached Arena result color accounting is inconsistent")
    contract = payload.get("arena_contract") or {}
    if int(contract.get("search_sims", -1)) != int(expected["arena_sims"]):
        raise RuntimeError("Cached Arena result has a different search-sim contract")
    if not math.isclose(
        float(contract.get("komi", float("nan"))),
        float(expected["komi"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Cached Arena result has a different komi contract")


def validate_sweep_overrides(saved_args: dict[str, object], overrides: dict[str, float | int]) -> None:
    for flag, value in overrides.items():
        key = _SWEEP_FLAG_TO_ARG_KEY.get(str(flag))
        if key is None:
            raise RuntimeError(f"Unknown sweep override cannot be resume-validated: {flag}")
        saved = saved_args.get(key)
        if isinstance(value, (int, float)) and isinstance(saved, (int, float)):
            if not math.isclose(float(saved), float(value), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"Checkpoint sweep override mismatch for {flag}: {saved!r} != {value!r}")
        elif saved != value:
            raise RuntimeError(f"Checkpoint sweep override mismatch for {flag}: {saved!r} != {value!r}")
