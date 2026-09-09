#!/usr/bin/env python3
"""Detached, fail-closed GoCube COMMON_PREP -> B4 -> B5 rerun for Legion.

This is bounded non-scientific preparation only. The full B experiment is never
started. B5 cannot start until the complete B4 artifact is written, validated,
and copied to the GitHub-report publish directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphazero.envs.gocube.b_evaluation import (
    B_HELDOUT_SUITE_PATH,
    B_HELDOUT_SUITE_POSITION_COUNT,
    B_HELDOUT_SUITE_SHA256,
    validate_frozen_suite,
)
from alphazero.envs.gocube.b_experiment_contract import (
    B05_DRY_RUN_DEFAULT_SETTINGS,
    B05_DRY_RUN_TARGET,
    B0_TREATMENT,
    B1_TREATMENT,
    preflight_b_experiment,
    validate_b_experiment_record,
)
from alphazero.envs.gocube.production_training import SampleBudgetTarget
from tools import analyze_gocube_b_evaluation, gocube_b05
from tools.evaluate_gocube_b05_dryrun import main as evaluate_b05_main
from tools.gocube_production_preflight import (
    B05_MIN_RAM_BYTES,
    SUPERVISED_ENV,
    UNIT_ENV,
    _assert_supervised_launch_ready,
    collect_production_preflight,
    is_supervised_invocation,
    validate_b05_production_preflight,
)
from tools.hardware_telemetry import HardwareTelemetry

REPORT_SCHEMA_VERSION = 1
DEFAULT_RUN_NAME = "gocube-b45-rerun-20260909"
PIPELINE_STAGE_ORDER = ("COMMON_PREP", "B4", "B5")
EVALUATION_MILESTONE = gocube_b05.EVALUATION_MILESTONE
B05_RUNTIME_NAMES = (
    "gocube-b05-b0",
    "gocube-b05-b1",
    *(f"gocube-b05-throughput-w{w}-r{r}"
      for w in (8, 16)
      for r in range(1, gocube_b05.THROUGHPUT_REPEATS + 1)),
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _markdown(report: Mapping[str, object]) -> str:
    stages = report.get("stages") if isinstance(report.get("stages"), Mapping) else {}
    lines = [
        "# GoCube B4 -> B5 preparation rerun", "",
        f"Status: **{report.get('status', 'UNKNOWN')}**", "",
        "Bounded preparation only; the full B experiment is not started.", "",
        "## Stages", "",
        *(f"- {stage}: `{stages.get(stage, 'NOT_RUN')}`" for stage in PIPELINE_STAGE_ORDER),
    ]
    failure = report.get("failure")
    if isinstance(failure, Mapping):
        lines += ["", "## Fail-closed reason", "", f"`{failure.get('type')}: {failure.get('message')}`"]
    return "\n".join(lines) + "\n"


def _write_report(root: Path, report: dict[str, object], b4_artifact: Path | None = None) -> None:
    report["report_schema_version"] = REPORT_SCHEMA_VERSION
    report["updated_at_epoch"] = time.time()
    md = _markdown(report)
    _atomic_json(root / "b45-report.json", report)
    (root / "b45-report.md").write_text(md, encoding="utf-8")
    publish = root / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    _atomic_json(publish / "b45-report.json", report)
    (publish / "b45-report.md").write_text(md, encoding="utf-8")
    published_b4 = publish / "b4-result.json"
    if b4_artifact is not None and b4_artifact.is_file():
        shutil.copy2(b4_artifact, published_b4)
    elif published_b4.exists():
        published_b4.unlink()


def _archive_previous_b05_runtime(repo: Path, root: Path) -> list[dict[str, str]]:
    # Outside active root: archived bytes must not contaminate B5 storage math.
    archive = repo / "training_reports" / "_b05_runtime_archive" / f"{root.name}-{int(time.time() * 1000)}"
    moved: list[dict[str, str]] = []
    for category in ("checkpoint", "data", "runs"):
        for name in B05_RUNTIME_NAMES:
            src = repo / category / name
            if not src.exists():
                continue
            dst = archive / category / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise RuntimeError(f"Archive destination already exists: {dst}")
            shutil.move(str(src), str(dst))
            moved.append({"source": str(src), "archive": str(dst)})
    return moved


def _validate_b4(evaluation: Mapping[str, object], record: Mapping[str, object], artifact: Path) -> dict[str, object]:
    games = evaluation.get("games")
    expected = B_HELDOUT_SUITE_POSITION_COUNT * 2
    if not isinstance(games, list) or len(games) != expected:
        raise RuntimeError(f"B4 dry-run did not complete exactly {expected} games")
    if evaluation.get("non_scientific_dry_run") is not True:
        raise RuntimeError("B4 artifact lacks non-scientific dry-run marker")
    if int(evaluation.get("training_seed", -1)) != 0:
        raise RuntimeError("B4 artifact has wrong training seed")
    if int(evaluation.get("scientific_milestone", -1)) != EVALUATION_MILESTONE:
        raise RuntimeError("B4 artifact has wrong milestone")
    contract = record.get("experiment_contract")
    if not isinstance(contract, Mapping) or not isinstance(contract.get("evaluation_milestones"), Mapping):
        raise RuntimeError("B4 experiment record lacks evaluation schedule")
    try:
        analyze_gocube_b_evaluation.validate_seed_evaluation(
            evaluation,
            evaluation_schedule=contract["evaluation_milestones"],
            experiment_contract_sha256=str(record["experiment_contract_sha256"]),
        )
    except ValueError as exc:
        rejection = str(exc)
    else:
        raise RuntimeError("Scientific B4 analyzer accepted a non-scientific B45 artifact")
    return {
        "status": "PASS", "artifact": str(artifact), "number_of_games": len(games),
        "position_count": evaluation.get("position_count"), "training_seed": 0,
        "sample_milestone": EVALUATION_MILESTONE, "seed_score_b1": evaluation.get("seed_score_b1"),
        "seed_delta": evaluation.get("seed_delta"), "non_scientific_dry_run": True,
        "scientific_analyzer_accepts": False, "scientific_analyzer_rejection": rejection,
    }


def _run_pipeline(repo: Path, args: argparse.Namespace) -> int:
    root = Path(args.report_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "RUNNING",
        "stage_order": list(PIPELINE_STAGE_ORDER),
        "stages": {stage: "NOT_RUN" for stage in PIPELINE_STAGE_ORDER},
        "request": {
            "run_name": args.run_name,
            "scientific_run": False,
            "full_b_experiment_started": False,
            "canonical_komi": 0.5,
            "dry_run_target": B05_DRY_RUN_TARGET,
            "dry_run_settings": dict(B05_DRY_RUN_DEFAULT_SETTINGS),
        },
    }
    telemetry = HardwareTelemetry(root / "hardware-telemetry.jsonl", interval_s=1.0)
    telemetry.start()
    active = "COMMON_PREP"
    b4_artifact: Path | None = None
    try:
        preflight = collect_production_preflight(repo, argparse.Namespace(device=args.device), verify_remote=True)
        validate_b05_production_preflight(preflight)
        ram = int((preflight.get("hardware") or {}).get("physical_memory_bytes", 0))
        if ram < B05_MIN_RAM_BYTES:
            raise RuntimeError("B45 RAM gate failed")
        report["preflight"] = preflight
        report["komi_audit"] = gocube_b05._komi_audit(repo)

        suite = B_HELDOUT_SUITE_PATH
        check = subprocess.run(
            [str(repo / ".venv/bin/python"), "tools/build_gocube_b_heldout_suite.py", "--check"],
            cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if check.returncode != 0:
            raise RuntimeError(f"Frozen suite check failed: {check.stdout.strip()}")
        suite_payload, positions = validate_frozen_suite(suite, expected_sha256=B_HELDOUT_SUITE_SHA256)
        if len(positions) != B_HELDOUT_SUITE_POSITION_COUNT or float(suite_payload.get("komi")) != 0.5:
            raise RuntimeError("Frozen B4 suite drifted in size or komi; expected komi 0.5")
        report["frozen_suite"] = {
            "path": str(suite), "sha256": gocube_b05._sha256_file(suite),
            "position_count": len(positions), "komi": suite_payload.get("komi"),
        }

        contract_path = root / "gocube-b45-experiment-contract.json"
        target = SampleBudgetTarget(SampleBudgetTarget.NEW_SAMPLES, B05_DRY_RUN_TARGET)
        preflight_b_experiment(
            repo=repo, contract_path=contract_path, heldout_suite_path=suite,
            scientific_target=target, dry_run_settings=B05_DRY_RUN_DEFAULT_SETTINGS,
        )
        record = validate_b_experiment_record(contract_path)
        report["contract_sha256"] = record["experiment_contract_sha256"]
        report["effective_configs"] = gocube_b05._manifest_diff(record)
        report["archived_previous_runtime"] = _archive_previous_b05_runtime(repo, root)
        if args.device != "cuda":
            raise RuntimeError("B45 workload requires CUDA; CPU is diagnostic-only")

        b0, b0_logs = gocube_b05._run_training(repo, root, suite, contract_path, B0_TREATMENT, telemetry)
        b1, b1_logs = gocube_b05._run_training(repo, root, suite, contract_path, B1_TREATMENT, telemetry)
        b0["log_paths"] = [str(p) for p in b0_logs]
        b1["log_paths"] = [str(p) for p in b1_logs]
        training = {B0_TREATMENT: b0, B1_TREATMENT: b1}
        report["training_runs"] = training
        report["stages"]["COMMON_PREP"] = "PASS"
        _write_report(root, report)

        active = "B4"
        report["stages"]["B4"] = "RUNNING"
        b4_artifact = root / "artifacts/b4-seed0-m128.json"
        b4_artifact.parent.mkdir(parents=True, exist_ok=True)
        b4_artifact.unlink(missing_ok=True)
        _write_report(root, report)
        evaluate_b05_main([
            "--b0-checkpoint", str(b0["checkpoint_metadata"]["path"]),
            "--b1-checkpoint", str(b1["checkpoint_metadata"]["path"]),
            "--suite", str(suite), "--experiment-contract", str(contract_path),
            "--training-seed", "0", "--sample-milestone", str(EVALUATION_MILESTONE),
            "--device", args.device, "--output", str(b4_artifact),
        ])
        report["b4"] = _validate_b4(_load_json(b4_artifact), record, b4_artifact)
        report["stages"]["B4"] = "PASS"
        _write_report(root, report, b4_artifact)

        active = "B5"
        report["stages"]["B5"] = "RUNNING"
        _write_report(root, report, b4_artifact)
        w8 = [gocube_b05._throughput_benchmark(repo, root, 8, r, telemetry)
              for r in range(1, gocube_b05.THROUGHPUT_REPEATS + 1)]
        w16 = [gocube_b05._throughput_benchmark(repo, root, 16, r, telemetry)
               for r in range(1, gocube_b05.THROUGHPUT_REPEATS + 1)]
        report["b5"] = {
            "status": "PASS",
            "throughput": gocube_b05._summarize_throughput(w8, w16),
            "inference_microbenchmark": gocube_b05._inference_microbenchmark(repo, root, args.device),
            "manifests_diff": gocube_b05._run_manifest_diff(repo, training),
            "storage_extrapolation": gocube_b05._storage_extrapolation(repo, root, training),
        }
        report["hardware_telemetry"] = telemetry.summary()
        report["stages"]["B5"] = "PASS"
        report["status"] = "PASS"
        _write_report(root, report, b4_artifact)
        return 0
    except Exception as exc:
        report["status"] = "BLOCKED"
        report["stages"][active] = "BLOCKED"
        report["failure"] = {"stage": active, "type": type(exc).__name__, "message": str(exc), "fail_closed": True}
        report["hardware_telemetry"] = telemetry.summary()
        _write_report(root, report, b4_artifact)
        print(f"B45 STOPPED FAIL-CLOSED in {active}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        telemetry.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--report-dir")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("invalid --run-name")
    args.report_dir = args.report_dir or f"training_reports/{args.run_name}"
    return args


def _systemd_command(repo: Path, args: argparse.Namespace) -> tuple[str, list[str]]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.run_name).strip("-.") or "b45"
    unit = f"gocube-b45-{safe}"[:120]
    inner = [
        str(repo / ".venv/bin/python"), str(repo / "tools/gocube_b45.py"),
        "--device", args.device, "--run-name", args.run_name, "--report-dir", args.report_dir,
    ]
    wrapped = [str(repo / "tools/run_with_github_reports.sh"), args.run_name, "--", *inner]
    return unit, [
        "systemd-run", "--user", "--collect", "--unit", unit,
        f"--working-directory={repo}", "--property=Restart=no",
        "--property=TimeoutStopSec=30s", "--property=KillMode=mixed",
        f"--setenv={SUPERVISED_ENV}=1", f"--setenv={UNIT_ENV}={unit}",
        "--setenv=PYTHONUNBUFFERED=1", *wrapped,
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path.cwd().resolve()
    if is_supervised_invocation():
        return _run_pipeline(repo, args)
    _assert_supervised_launch_ready(repo)
    unit, command = _systemd_command(repo, args)
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    bus = Path(env["XDG_RUNTIME_DIR"]) / "bus"
    if bus.exists():
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}")
    active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], cwd=repo, env=env, check=False)
    if active.returncode == 0:
        raise SystemExit(f"B45 service already active: {unit}")
    return int(subprocess.run(command, cwd=repo, env=env, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
