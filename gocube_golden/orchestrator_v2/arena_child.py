"""Production Arena child launched from an immutable runtime worktree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from tools.arena import run_arena
from tools.arena_engine import ArenaExecutionConfig

from .immutable_runtime import validate_runtime_head
from .version import mark_v2_process


def _read(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def run_arena_worker(request_path: str | Path, result_path: str | Path) -> None:
    request = _read(Path(request_path).resolve())
    if request.get("schema") != "gocube-orchestrator-v2-arena-request-v1":
        raise ValueError("unsupported Arena child request schema")
    execution_commit = request.get("execution_code_commit")
    if not isinstance(execution_commit, str) or not execution_commit:
        raise ValueError("Arena child request is missing execution_code_commit")
    validate_runtime_head(Path.cwd(), execution_commit)
    mark_v2_process()
    raw_config = request.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Arena child config must be an object")
    raw_workload = request.get("workload")
    workload = raw_workload if isinstance(raw_workload, Mapping) else None
    summary = run_arena(
        candidate_path=Path(str(request["candidate_path"])),
        reference_path=Path(str(request["reference_path"])),
        profile_name=str(request["profile_name"]),
        output_dir=Path(str(request["output_dir"])),
        candidate_label=str(request["candidate_label"]),
        reference_label=str(request["reference_label"]),
        run_id=str(request["run_id"]),
        comparison=str(request["comparison"]),
        master_seed=int(request["master_seed"]),
        config=ArenaExecutionConfig(**dict(raw_config)),
        expected_candidate_artifact_sha256=str(request["expected_candidate_artifact_sha256"]),
        expected_reference_artifact_sha256=str(request["expected_reference_artifact_sha256"]),
        evaluation_identity=request["evaluation_identity"] if isinstance(request.get("evaluation_identity"), Mapping) else None,
        evaluation_fingerprint=str(request["evaluation_fingerprint"]),
        allowed_lineage_arena_root=(
            None
            if request.get("allowed_lineage_arena_root") is None
            else Path(str(request["allowed_lineage_arena_root"]))
        ),
        allowed_evaluation_root=(
            None
            if request.get("allowed_evaluation_root") is None
            else Path(str(request["allowed_evaluation_root"]))
        ),
        workload=workload,
    )
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
