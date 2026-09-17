#!/usr/bin/env python3
"""Terminal CLI for immutable, detached production AlphaZero training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.code_update_policy import CodeUpdateProvenancePolicy
from gocube_golden.production_orchestrator import format_production_status
from gocube_golden.provenance import file_sha256
from gocube_golden.provenance import canonical_json
from gocube_golden.artifact_catalog import ARTIFACT_VALIDATION_SCHEMA
from gocube_golden.run_lifecycle import archive_lineage, discard_lineage
from gocube_golden.run_spec import (
    StrictProductionTrainingOrchestrator,
    StrictRunSpec,
    load_persisted_run_spec,
)


def _from_source(
    args: argparse.Namespace, *, terminal: bool = True
) -> StrictProductionTrainingOrchestrator:
    spec = StrictRunSpec.load(args.spec, repo_root=ROOT)
    return StrictProductionTrainingOrchestrator(
        repo_root=ROOT,
        run_spec=spec,
        lineage_id=args.lineage,
        terminal=terminal,
        child_lifecycle_policy=CodeUpdateProvenancePolicy(),
    )


def _from_lineage(
    args: argparse.Namespace, *, terminal: bool = True
) -> StrictProductionTrainingOrchestrator:
    spec = load_persisted_run_spec(repo_root=ROOT, lineage_id=args.lineage)
    return StrictProductionTrainingOrchestrator(
        repo_root=ROOT,
        run_spec=spec,
        lineage_id=args.lineage,
        terminal=terminal,
        child_lifecycle_policy=CodeUpdateProvenancePolicy(),
    )


def _parent_checkpoint(args: argparse.Namespace) -> dict[str, object] | None:
    values = (args.parent_lineage, args.parent_path, args.parent_sha256)
    extensions = (args.parent_generation, args.parent_replay_path)
    if not any(values):
        if any(value is not None for value in extensions):
            raise SystemExit(
                "Parent reference extensions require --parent-lineage, --parent-path and --parent-sha256"
            )
        return None
    if not all(values):
        raise SystemExit(
            "--parent-lineage, --parent-path and --parent-sha256 must be supplied together"
        )
    if args.parent_replay_path is None:
        raise SystemExit("--parent-replay-path is required for a continuation parent")
    checkpoint = Path(str(args.parent_path)).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Parent checkpoint does not exist: {checkpoint}")
    actual_checkpoint_sha256 = file_sha256(checkpoint)
    if actual_checkpoint_sha256 != str(args.parent_sha256):
        raise SystemExit(
            "--parent-sha256 does not match the referenced checkpoint: "
            f"expected {args.parent_sha256}, actual {actual_checkpoint_sha256}"
        )
    metadata_path = checkpoint.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise SystemExit(f"Parent checkpoint metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise SystemExit("Parent checkpoint metadata must be a JSON object")
    label = str(metadata.get("checkpoint_label", ""))
    inferred_generation = int(label[1:]) if label.startswith("M") and label[1:].isdigit() else None
    generation = args.parent_generation if args.parent_generation is not None else inferred_generation
    if generation is None or generation < 0:
        raise SystemExit("Parent checkpoint metadata must provide an M<number> label")
    if inferred_generation is not None and inferred_generation != generation:
        raise SystemExit("--parent-generation disagrees with parent checkpoint label")
    replay = Path(str(args.parent_replay_path)).resolve()
    if not replay.is_file():
        raise SystemExit(f"Parent replay does not exist: {replay}")
    replay_digest = hashlib.sha256()
    replay_rows = 0
    try:
        with replay.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                replay_digest.update(canonical_json(row).encode("utf-8"))
                replay_digest.update(b"\n")
                replay_rows += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Parent replay is not valid JSONL: {replay}") from exc
    expected_rows = metadata.get("valid_replay_positions", metadata.get("replay_row_count"))
    if expected_rows is not None and int(expected_rows) != replay_rows:
        raise SystemExit(
            f"Parent replay row count mismatch: metadata={expected_rows}, file={replay_rows}"
        )
    source_root = replay.parent.parent
    summary_path = source_root / f"iter-{generation:02d}-summary.json"
    total_evictions = 0
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        replay_summary = summary.get("replay") if isinstance(summary, dict) else None
        if isinstance(replay_summary, dict):
            total_evictions = int(replay_summary.get("total_evictions", 0))
    return {
        "lineage_id": args.parent_lineage,
        "label": label,
        "generation": int(generation),
        "path": str(checkpoint),
        "sha256": actual_checkpoint_sha256,
        "artifact_sha256": actual_checkpoint_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "model_hash": metadata.get("model_hash"),
        "replay_path": str(replay),
        "replay_sha256": file_sha256(replay),
        "replay_row_count": replay_rows,
        "replay_fingerprint": metadata.get(
            "replay_fingerprint", "sha256:" + replay_digest.hexdigest()
        ),
        "replay_validation_schema": ARTIFACT_VALIDATION_SCHEMA,
        "total_evictions": total_evictions,
    }


def _cmd_create(args: argparse.Namespace) -> int:
    run = _from_source(args)
    run.create(parent_checkpoint=_parent_checkpoint(args))
    print(format_production_status(run.status()))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    run = _from_source(args)
    if run.paths.root.exists():
        raise SystemExit("Lineage already exists; use resume and its persisted run-spec")
    run.create(parent_checkpoint=_parent_checkpoint(args))
    run.run(max_generations=args.max_generations)
    print(format_production_status(run.status()))
    return 0


def _terminate_supervisor_startup(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _spawn_supervisor(
    args: argparse.Namespace, run: StrictProductionTrainingOrchestrator
) -> int:
    log_path = run.paths.logs / "orchestrator-supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_supervise",
        "--lineage",
        args.lineage,
    ]
    if args.max_generations is not None:
        command.extend(["--max-generations", str(args.max_generations)])
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    deadline = time.monotonic() + run.supervision.startup_ack_timeout_seconds
    last_state: object = None
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise SystemExit(
                f"Orchestrator supervisor exited during startup with code {code}. Log: {log_path}"
            )
        try:
            status = run.status()
        except Exception:
            time.sleep(0.1)
            continue
        last_state = status.get("state")
        if last_state == "RECOVERY_REQUIRED":
            raise SystemExit(
                "Orchestrator entered RECOVERY_REQUIRED during startup. "
                f"Log: {log_path}"
            )
        heartbeat = status.get("heartbeat_age_sec")
        if last_state == "RUNNING" and isinstance(heartbeat, (int, float)):
            print(f"Started orchestrator PID {process.pid}")
            print(f"Lineage: {args.lineage}")
            print(f"Log: {log_path}")
            print(
                f"Status: {sys.executable} {Path(__file__).name} status --lineage {args.lineage} --watch"
            )
            return 0
        time.sleep(0.1)

    _terminate_supervisor_startup(process)
    raise SystemExit(
        "Detached supervisor did not publish a RUNNING heartbeat before the explicit "
        f"startup timeout (last state={last_state!r}). Log: {log_path}"
    )


def _cmd_start(args: argparse.Namespace) -> int:
    run = _from_source(args)
    if run.paths.root.exists():
        raise SystemExit("Lineage already exists; use resume and its persisted run-spec")
    run.create(parent_checkpoint=_parent_checkpoint(args))
    return _spawn_supervisor(args, run)


def _cmd_resume(args: argparse.Namespace) -> int:
    run = _from_lineage(args)
    run.prepare_resume()
    if args.foreground:
        run.run(max_generations=args.max_generations)
        print(format_production_status(run.status()))
        return 0
    return _spawn_supervisor(args, run)


def _cmd_supervise(args: argparse.Namespace) -> int:
    run = _from_lineage(args, terminal=True)
    run.run(max_generations=args.max_generations)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    run = _from_lineage(args, terminal=False)
    while True:
        status = run.status()
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(format_production_status(status))
            print(
                f"Run spec: {status['run_spec_path']} ({status['run_spec_fingerprint']})"
            )
        if not args.watch or status.get("state") in {
            "COMPLETED",
            "SOFT_STOPPED",
            "RECOVERY_REQUIRED",
        }:
            return 0
        print("-" * 72, flush=True)
        time.sleep(args.interval)


def _cmd_stop(args: argparse.Namespace) -> int:
    run = _from_lineage(args, terminal=False)
    payload = run.request_soft_stop(args.minutes, reason="terminal-command")
    print(
        "Soft-stop requested. The active safe unit will not be hard-killed and no new "
        f"generation/Arena will start afterward. Target window ends at {payload['target_deadline_at']}."
    )
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    run = _from_lineage(args, terminal=False)
    path = run.paths.logs / "orchestrator-supervisor.log"
    if not path.is_file():
        raise SystemExit(f"Supervisor log does not exist yet: {path}")
    position = 0
    while True:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if position == 0:
            selected = lines[-args.lines :]
        else:
            selected = lines[position:]
        for line in selected:
            print(line)
        position = len(lines)
        if not args.follow:
            return 0
        state = run.status().get("state")
        if state in {"COMPLETED", "SOFT_STOPPED", "RECOVERY_REQUIRED"}:
            return 0
        time.sleep(args.interval)


def _cmd_archive(args: argparse.Namespace) -> int:
    target = archive_lineage(repo_root=ROOT, lineage_id=args.lineage)
    print(f"Archived lineage by move: {target}")
    return 0


def _cmd_discard(args: argparse.Namespace) -> int:
    if args.confirm_lineage != args.lineage:
        raise SystemExit("--confirm-lineage must exactly match --lineage")
    record = discard_lineage(
        repo_root=ROOT,
        lineage_id=args.lineage,
        reason=args.reason,
        useful_result=args.useful_result,
    )
    print(f"Discarded heavy lineage artifacts; retained policy record: {record}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def new_run(command: argparse.ArgumentParser) -> None:
        command.add_argument("--spec", required=True, help="One-shot immutable run-spec JSON")
        command.add_argument("--lineage", required=True, help="Stable lineage id")

    def existing_run(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--lineage", required=True, help="Stable lineage id; spec is read from lineage"
        )

    def parent(command: argparse.ArgumentParser) -> None:
        command.add_argument("--parent-lineage")
        command.add_argument("--parent-path")
        command.add_argument("--parent-sha256")
        command.add_argument("--parent-generation", type=int)
        command.add_argument("--parent-replay-path")

    create = sub.add_parser("create", help="Create lineage and freeze the one-shot run spec")
    new_run(create)
    parent(create)
    create.set_defaults(func=_cmd_create)

    run = sub.add_parser("run", help="Create a new lineage and run it in foreground")
    new_run(run)
    parent(run)
    run.add_argument("--max-generations", type=int)
    run.set_defaults(func=_cmd_run)

    start = sub.add_parser("start", help="Create a new lineage and start detached supervisor")
    new_run(start)
    parent(start)
    start.add_argument("--max-generations", type=int)
    start.set_defaults(func=_cmd_start)

    resume = sub.add_parser(
        "resume", help="Resume using only the immutable lineage-owned run spec"
    )
    existing_run(resume)
    resume.add_argument("--max-generations", type=int)
    resume.add_argument("--foreground", action="store_true")
    resume.set_defaults(func=_cmd_resume)

    supervise = sub.add_parser("_supervise", help=argparse.SUPPRESS)
    existing_run(supervise)
    supervise.add_argument("--max-generations", type=int)
    supervise.set_defaults(func=_cmd_supervise)

    status = sub.add_parser("status", help="Show live health/progress/status")
    existing_run(status)
    status.add_argument("--json", action="store_true")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=5.0)
    status.set_defaults(func=_cmd_status)

    stop = sub.add_parser("stop", help="Request bounded soft stop at the next safe boundary")
    existing_run(stop)
    stop.add_argument("--minutes", type=int)
    stop.set_defaults(func=_cmd_stop)

    logs = sub.add_parser("logs", help="Read or follow the detached supervisor log")
    existing_run(logs)
    logs.add_argument("--lines", type=int, default=80)
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--interval", type=float, default=2.0)
    logs.set_defaults(func=_cmd_logs)

    archive = sub.add_parser("archive", help="Move a stopped lineage from active/ to archive/")
    existing_run(archive)
    archive.set_defaults(func=_cmd_archive)

    discard = sub.add_parser(
        "discard",
        help="Delete stopped heavy artifacts only after dependency check and policy record",
    )
    existing_run(discard)
    discard.add_argument("--reason", required=True)
    discard.add_argument("--useful-result", required=True)
    discard.add_argument("--confirm-lineage", required=True)
    discard.set_defaults(func=_cmd_discard)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "max_generations", None) is not None and args.max_generations <= 0:
        raise SystemExit("--max-generations must be positive")
    if getattr(args, "interval", 1.0) <= 0:
        raise SystemExit("--interval must be positive")
    if getattr(args, "lines", 1) <= 0:
        raise SystemExit("--lines must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
