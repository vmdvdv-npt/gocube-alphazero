"""Production Arena child launched from an immutable runtime worktree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from typing import Mapping

from tools.arena import run_arena
from tools.arena_engine import ArenaExecutionConfig
from gocube_golden.process_supervision import atomic_write_json

from .execution_permit import require_child_execution_permit
from .immutable_runtime import validate_runtime_head


def _read(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _clear_incomplete_output(output: Path) -> None:
    """Remove only retryable Arena products; identity remains immutable."""
    for name in (
        "games.jsonl",
        "summary.json",
        "manifest.json",
        "provenance.json",
        "telemetry.json",
        "startup-failure.json",
    ):
        (output / name).unlink(missing_ok=True)


def run_arena_worker(request_path: str | Path, result_path: str | Path) -> None:
    request = _read(Path(request_path).resolve())
    if request.get("schema") != "gocube-orchestrator-v2-arena-request-v1":
        raise ValueError("unsupported Arena child request schema")
    execution_commit = request.get("execution_code_commit")
    if not isinstance(execution_commit, str) or not execution_commit:
        raise ValueError("Arena child request is missing execution_code_commit")
    validate_runtime_head(Path.cwd(), execution_commit)
    raw_config = request.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Arena child config must be an object")
    raw_workload = request.get("workload")
    workload = raw_workload if isinstance(raw_workload, Mapping) else None
    evaluation_identity = request.get("evaluation_identity")
    topology = None
    if isinstance(evaluation_identity, Mapping):
        candidate = evaluation_identity.get("candidate")
        if isinstance(candidate, Mapping) and candidate.get("topology") is not None:
            topology = str(candidate["topology"])
    if not topology:
        raise ValueError("Arena child request has no candidate topology")
    require_child_execution_permit(
        "gocube_golden.orchestrator_v2.arena_child",
        action_type="arena",
        topology=topology,
        run_id=str(request["run_id"]),
        code_identity=execution_commit,
    )
    raw_liveness_path = request.get("liveness_path")
    raw_progress_path = request.get("progress_path")
    if not isinstance(raw_liveness_path, str) or not raw_liveness_path:
        raise ValueError("Arena child request is missing heartbeat paths")
    if not isinstance(raw_progress_path, str) or not raw_progress_path:
        raise ValueError("Arena child request is missing heartbeat paths")
    liveness_path = Path(raw_liveness_path).resolve()
    progress_path = Path(raw_progress_path).resolve()
    output_path = Path(str(request["output_dir"])).resolve()
    _clear_incomplete_output(output_path)
    stop_heartbeat = threading.Event()
    progress_lock = threading.Lock()
    started_at = time.time()
    progress_state = {
        "attempt": int(request.get("attempt", 1)),
        "completed_games": 0,
        "started_games": 0,
        "move_events": 0,
        "total_games": int(raw_config.get("games", 0)),
        "last_move_at": None,
        "inference_batches": 0,
        "inference_rows": 0,
        "active_contexts": 0,
        "progress_at": started_at,
    }
    last_progress_publish_at = 0.0

    def publish(*, progress: bool = False) -> None:
        now = time.time()
        with progress_lock:
            snapshot = dict(progress_state)
        progress_at = float(snapshot.get("progress_at", started_at))
        payload = {
            "schema": "gocube-orchestrator-v2-arena-heartbeat-v1",
            "liveness_at": now,
            "progress_at": now if progress else progress_at,
            "progress_token": f"moves:{snapshot['move_events']};games:{snapshot['completed_games']}/{snapshot['total_games']}",
            "progress": snapshot,
        }
        atomic_write_json(liveness_path, payload)
        if progress_path != liveness_path:
            atomic_write_json(progress_path, payload)

    def heartbeat_loop() -> None:
        while not stop_heartbeat.is_set():
            try:
                publish()
            except OSError:
                pass
            stop_heartbeat.wait(1.0)

    def progress_callback(completed: int, total: int) -> None:
        nonlocal last_progress_publish_at
        now = time.time()
        with progress_lock:
            progress_state["completed_games"] = int(completed)
            progress_state["total_games"] = int(total)
            progress_state["progress_at"] = now
        # Move-level progress keeps the supervisor from restarting a healthy
        # long game while bounding heartbeat file writes to about one per sec.
        if now - last_progress_publish_at >= 1.0 or int(completed) >= int(total):
            publish(progress=True)
            last_progress_publish_at = now

    def activity_callback(activity: Mapping[str, object]) -> None:
        with progress_lock:
            for key in (
                "started_games",
                "move_events",
                "inference_batches",
                "inference_rows",
                "active_contexts",
            ):
                if key in activity:
                    progress_state[key] = int(activity[key])
            if activity.get("last_move_at") is not None:
                progress_state["last_move_at"] = float(activity["last_move_at"])

    publish()
    heartbeat = threading.Thread(target=heartbeat_loop, name="arena-heartbeat", daemon=True)
    heartbeat.start()
    try:
        summary = run_arena(
            candidate_path=Path(str(request["candidate_path"])),
            reference_path=Path(str(request["reference_path"])),
            profile_name=str(request["profile_name"]),
            output_dir=output_path,
            candidate_label=str(request["candidate_label"]),
            reference_label=str(request["reference_label"]),
            run_id=str(request["run_id"]),
            comparison=str(request["comparison"]),
            master_seed=int(request["master_seed"]),
            config=ArenaExecutionConfig(**dict(raw_config)),
            expected_candidate_artifact_sha256=str(request["expected_candidate_artifact_sha256"]),
            expected_reference_artifact_sha256=str(request["expected_reference_artifact_sha256"]),
            evaluation_identity=evaluation_identity if isinstance(evaluation_identity, Mapping) else None,
            evaluation_fingerprint=str(request["evaluation_fingerprint"]),
            allowed_lineage_arena_root=None if request.get("allowed_lineage_arena_root") is None else Path(str(request["allowed_lineage_arena_root"])),
            allowed_evaluation_root=None if request.get("allowed_evaluation_root") is None else Path(str(request["allowed_evaluation_root"])),
            workload=workload,
            progress_callback=progress_callback,
            activity_callback=activity_callback,
            execution_code_commit=execution_commit,
        )
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=2.0)
        try:
            publish(progress=True)
        except OSError:
            pass
    result = Path(result_path).resolve()
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    run_arena_worker(args.request, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
