"""Supervised training-adapter preparation; the controller performs no inference."""
from __future__ import annotations

import argparse
from pathlib import Path

from .execution_permit import require_engine_execution, require_child_execution_permit
from .immutable_runtime import resolve_execution_commit, validate_runtime_head
from .production_generation import _read_json, _write_json
from .supervisor import SupervisorV2

REQUEST_SCHEMA = "gocube-replay-preparation-v1"


def prepare_replay_cache(adapter, *, spec, experiment_root, experiment_id, device):
    require_engine_execution("offline replay preparation", action="training")
    root = Path(experiment_root)
    request_path = root / "runtime" / "requests" / "replay-preparation.json"
    result_path = root / "runtime" / "results" / "replay-preparation.json"
    heartbeat = root / "runtime" / "heartbeats" / "replay-preparation.json"
    commit = resolve_execution_commit(adapter.repo_root, "HEAD")
    request = {"schema": REQUEST_SCHEMA, "spec": spec, "device": device,
               "experiment_id": experiment_id, "cache_root": str(root / "artifacts" / "policy-surprise"),
               "heartbeat": str(heartbeat), "execution_code_commit": commit}
    if request_path.exists():
        saved = dict(_read_json(request_path))
        if saved != request:
            raise ValueError("Replay preparation request changed during resume")
    else:
        _write_json(request_path, request)
    runtime = adapter.runtime_manager.ensure(commit)
    command = [adapter.python_executable, "-m", "gocube_golden.orchestrator_v2.replay_preparation",
               "--request", str(request_path), "--result", str(result_path)]

    def launch(child_request):
        return adapter.launch_replay_preparation(command=command, runtime=runtime,
            experiment_id=experiment_id, attempt=int(child_request.attempt))

    supervisor = SupervisorV2(root, execution_id=f"{experiment_id}:replay-preparation",
                              liveness_path=heartbeat, progress_path=heartbeat,
                              launcher=launch, command=command, cwd=runtime.path,
                              policy=adapter.supervisor_policy)
    from ..policy_surprise import load_cache

    def verified_cache():
        payload = _read_json(result_path)
        if payload.get("schema") != REQUEST_SCHEMA or payload.get("execution_code_commit") != commit:
            raise ValueError("Replay preparation result identity mismatch")
        cache = dict(payload["cache"])
        load_cache(cache, spec)
        return cache

    if result_path.exists():
        cache = verified_cache()
        result = supervisor.reconcile_completed_execution()
    else:
        result = supervisor.run_once()
        if not result.success:
            raise RuntimeError(f"Replay preparation stopped: {result.reason}")
        cache = verified_cache()
    if not result.success:
        raise RuntimeError(f"Replay preparation reconciliation failed: {result.reason}")
    return cache


def run_worker(request_path, result_path):
    require_engine_execution("offline replay preparation worker", action="training")
    request = _read_json(Path(request_path))
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("Unsupported replay preparation request")
    require_child_execution_permit("replay preparation", action_type="generation", topology="torus9",
                                   run_id=request["experiment_id"], code_identity=request["execution_code_commit"])
    validate_runtime_head(Path.cwd(), request["execution_code_commit"])
    from ..torus9_five_channel_training import prepare_policy_surprise_cache
    cache = prepare_policy_surprise_cache(request)
    _write_json(Path(result_path), {"schema": REQUEST_SCHEMA, "cache": cache,
                                  "execution_code_commit": request["execution_code_commit"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    run_worker(args.request, args.result)


if __name__ == "__main__":
    main()
