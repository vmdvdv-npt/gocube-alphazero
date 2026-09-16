#!/usr/bin/env python3
"""CLI for immutable run-spec production training orchestration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.orchestrator import format_status
from gocube_golden.run_spec import (
    StrictProductionTrainingOrchestrator,
    StrictRunSpec,
    load_persisted_run_spec,
)


def _from_source(args: argparse.Namespace, *, terminal: bool = True) -> StrictProductionTrainingOrchestrator:
    spec = StrictRunSpec.load(args.spec, repo_root=ROOT)
    return StrictProductionTrainingOrchestrator(repo_root=ROOT, run_spec=spec, lineage_id=args.lineage, terminal=terminal)


def _from_lineage(args: argparse.Namespace, *, terminal: bool = True) -> StrictProductionTrainingOrchestrator:
    spec = load_persisted_run_spec(repo_root=ROOT, lineage_id=args.lineage)
    return StrictProductionTrainingOrchestrator(repo_root=ROOT, run_spec=spec, lineage_id=args.lineage, terminal=terminal)


def _parent_checkpoint(args: argparse.Namespace) -> dict[str, object] | None:
    values = (args.parent_lineage, args.parent_path, args.parent_sha256)
    if not any(values):
        return None
    if not all(values):
        raise SystemExit("--parent-lineage, --parent-path and --parent-sha256 must be supplied together")
    return {"lineage_id": args.parent_lineage, "path": args.parent_path, "sha256": args.parent_sha256}


def _cmd_create(args: argparse.Namespace) -> int:
    run = _from_source(args)
    run.create(parent_checkpoint=_parent_checkpoint(args))
    print(format_status(run.status()))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    run = _from_source(args)
    if run.paths.root.exists():
        raise SystemExit("Lineage already exists; use resume and the persisted run-spec")
    run.create(parent_checkpoint=_parent_checkpoint(args))
    run.run(max_generations=args.max_generations)
    print(format_status(run.status()))
    return 0


def _spawn_supervisor(args: argparse.Namespace, run: StrictProductionTrainingOrchestrator) -> int:
    log_path = run.paths.logs / "orchestrator-supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "_supervise", "--lineage", args.lineage]
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
    print(f"Started orchestrator PID {process.pid}")
    print(f"Log: {log_path}")
    print(f"Status: {sys.executable} {Path(__file__).name} status --lineage {args.lineage}")
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    run = _from_source(args)
    if run.paths.root.exists():
        raise SystemExit("Lineage already exists; use resume and the persisted run-spec")
    run.create(parent_checkpoint=_parent_checkpoint(args))
    return _spawn_supervisor(args, run)


def _cmd_resume(args: argparse.Namespace) -> int:
    run = _from_lineage(args)
    run.prepare_resume()
    if args.foreground:
        run.run(max_generations=args.max_generations)
        print(format_status(run.status()))
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
            print(format_status(status))
            print(f"Run spec: {status['run_spec_path']} ({status['run_spec_fingerprint']})")
        if not args.watch or status.get("state") in {"COMPLETED", "SOFT_STOPPED", "RECOVERY_REQUIRED"}:
            return 0
        print("-" * 72, flush=True)
        time.sleep(args.interval)


def _cmd_stop(args: argparse.Namespace) -> int:
    run = _from_lineage(args, terminal=False)
    payload = run.request_soft_stop(args.minutes, reason="terminal-command")
    print(
        "Soft-stop requested. Current work will not be hard-killed; "
        f"target safe-stop window ends at {payload['target_deadline_at']}."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def new_run(command: argparse.ArgumentParser) -> None:
        command.add_argument("--spec", required=True, help="One-shot immutable run-spec JSON")
        command.add_argument("--lineage", required=True, help="Stable lineage id")

    def existing_run(command: argparse.ArgumentParser) -> None:
        command.add_argument("--lineage", required=True, help="Stable lineage id; spec is read from lineage")

    def parent(command: argparse.ArgumentParser) -> None:
        command.add_argument("--parent-lineage")
        command.add_argument("--parent-path")
        command.add_argument("--parent-sha256")

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

    resume = sub.add_parser("resume", help="Resume using only the immutable lineage-owned run spec")
    existing_run(resume)
    resume.add_argument("--max-generations", type=int)
    resume.add_argument("--foreground", action="store_true")
    resume.set_defaults(func=_cmd_resume)

    supervise = sub.add_parser("_supervise", help=argparse.SUPPRESS)
    existing_run(supervise)
    supervise.add_argument("--max-generations", type=int)
    supervise.set_defaults(func=_cmd_supervise)

    status = sub.add_parser("status", help="Show status using the lineage-owned run spec")
    existing_run(status)
    status.add_argument("--json", action="store_true")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=5.0, help="Terminal refresh interval only")
    status.set_defaults(func=_cmd_status)

    stop = sub.add_parser("stop", help="Request soft stop using the lineage-owned policy")
    existing_run(stop)
    stop.add_argument("--minutes", type=int, help="Optional explicit window; otherwise run-spec default is used")
    stop.set_defaults(func=_cmd_stop)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "max_generations", None) is not None and args.max_generations <= 0:
        raise SystemExit("--max-generations must be positive")
    if getattr(args, "interval", 1.0) <= 0:
        raise SystemExit("--interval must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
