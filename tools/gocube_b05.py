#!/usr/bin/env python3
"""Legion B05 preflight and bounded GoCube B0/B1 integration proof.

This command is deliberately a review-branch tool.  It runs the production
preflight first and stops before any workload if the machine, source, frozen
suite, or effective configuration is not safe.  When the preflight passes it
executes only a tiny non-scientific B0/B1 proof; it never starts the 40M-sample
experiment and its artifacts are rejected by the scientific B4 analyzer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphazero.envs.gocube.b_evaluation import (
    B_HELDOUT_SUITE_POSITION_COUNT,
    B_HELDOUT_SUITE_PATH,
    B_HELDOUT_SUITE_SHA256,
    validate_frozen_suite,
)
from alphazero.envs.gocube.b_experiment_contract import (
    ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES,
    B05_DRY_RUN_DEFAULT_SETTINGS,
    B05_DRY_RUN_TARGET,
    B0_MODEL_PROFILE,
    B0_TREATMENT,
    B1_MODEL_PROFILE,
    B1_TREATMENT,
    diff_effective_configs,
    load_b_experiment_contract,
    preflight_b_experiment,
    validate_b0_b1_effective_configs,
    validate_b_experiment_record,
)
from alphazero.envs.gocube.hardened_train import (
    AtomicSampleClockNNetWrapper,
    build_hardened_training_args,
)
from alphazero.envs.gocube.integration.catalog import CheckpointCatalog
from alphazero.envs.gocube.integration.models import CheckpointModelLoader
from alphazero.envs.gocube.katago_train import parse_args as parse_training_args
from alphazero.envs.gocube.production_contract import CUBE4_PRODUCTION, GOCUBE_KOMI
from alphazero.envs.gocube.production_training import (
    CumulativeTrainingCounters,
    SampleBudgetTarget,
    load_training_progress,
)
from alphazero.envs.gocube.records import effective_parameter_snapshot
from tools import analyze_gocube_b_evaluation
from tools.evaluate_gocube_b05_dryrun import main as evaluate_b05_main
from tools.gocube_production_preflight import (
    B05_MIN_RAM_BYTES,
    EXPECTED_REPO,
    SUPERVISED_ENV,
    UNIT_ENV,
    _assert_supervised_launch_ready,
    collect_production_preflight,
    is_supervised_invocation,
    validate_b05_production_preflight,
)
from tools.hardware_telemetry import HardwareTelemetry, _read_nvidia_smi


REPORT_SCHEMA_VERSION = 1
DEFAULT_REPORT_DIR = "training_reports/gocube-b05-legion-preflight-dryrun"
THROUGHPUT_GAMES = 8
INFERENCE_BATCH_ROWS = 32
INFERENCE_REPEATS = 12
EVALUATION_MILESTONE = 128


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _background_snapshot(repo: Path) -> dict[str, object]:
    process = subprocess.run(
        [
            "ps",
            "-eo",
            "pid,ppid,user,%cpu,%mem,stat,etime,comm,args",
            "--sort=-%cpu",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lines = process.stdout.splitlines()
    return {
        "captured_at_epoch": time.time(),
        "process_exit_code": int(process.returncode),
        "top_processes": lines[:21],
        "gpu_counters": _read_nvidia_smi(),
        "load_average": list(os.getloadavg()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def _komi_audit(repo: Path) -> dict[str, object]:
    excluded = [
        ".git",
        ".venv",
        ".pytest_cache",
        ".cache",
        "__pycache__",
        "checkpoint",
        "data",
        "runs",
        "training_reports",
    ]
    rg = shutil.which("rg")
    if rg:
        command = [rg, "-n", "-I", "--hidden"]
        for directory in excluded:
            command.extend(["-g", f"!{directory}/**"])
        command.extend([r"7\.5", str(repo)])
    else:
        grep = shutil.which("grep") or "/usr/bin/grep"
        command = [grep, "-RIn", "--binary-files=without-match"]
        command.extend(f"--exclude-dir={directory}" for directory in excluded)
        command.extend([r"7\.5", str(repo)])
    result = subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Komi audit failed: {result.stderr.strip()}")
    matches = []
    applicable = []
    production_roots = ("alphazero/", "tools/", "evaluation/", "scripts/")
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        filename = line.split(":", 1)[0]
        try:
            relative = str(Path(filename).resolve().relative_to(repo.resolve()))
        except ValueError:
            relative = filename
        item = {"path": relative, "line": line}
        matches.append(item)
        if relative.startswith(production_roots) and not relative.startswith(("tests/", "docs/")):
            applicable.append(item)
    searched_literal = "7" + ".5"
    if applicable:
        raise RuntimeError(
            f"Applicable production literal {searched_literal} found; fail-closed Komi audit: "
            + ", ".join(item["path"] for item in applicable)
        )
    return {
        "searched_literal": searched_literal,
        "canonical_komi": GOCUBE_KOMI,
        "matches": matches,
        "applicable_production_matches": applicable,
        "historical_or_test_matches_only": True,
        "excluded_artifact_roots": excluded,
    }


def _run_command(
    command: list[str],
    *,
    repo: Path,
    log_path: Path,
    phase: str,
    telemetry: HardwareTelemetry,
) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    environment["PYTHONUNBUFFERED"] = "1"
    telemetry.set_phase(phase)
    telemetry.start()
    started = time.perf_counter()
    sample_time = None
    inference_batch = None
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n=== COMMAND ===\n" + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            match = re.search(r"Sample Time:\s*([0-9.]+)s", line)
            if match:
                sample_time = float(match.group(1))
            match = re.search(r"Infer Batch:\s*([0-9.]+)", line)
            if match:
                inference_batch = float(match.group(1))
        return_code = process.wait()
    wall = time.perf_counter() - started
    result = {
        "command": command,
        "log": str(log_path),
        "phase": phase,
        "return_code": int(return_code),
        "wall_time_seconds": wall,
        "sample_time_seconds": sample_time,
        "mean_inference_batch_rows": inference_batch,
    }
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
    return result


def _checkpoint_file(repo: Path, run_name: str, iteration: int | None = None) -> Path:
    root = repo / "checkpoint" / run_name
    if iteration is not None:
        path = root / f"iteration-{int(iteration):04d}.pkl"
        if not path.is_file():
            raise RuntimeError(f"Missing expected checkpoint: {path}")
        return path
    candidates = sorted(root.glob("iteration-*.pkl"))
    if not candidates:
        raise RuntimeError(f"No checkpoint was produced for {run_name}")
    return candidates[-1]


def _checkpoint_metadata(repo: Path, run_name: str) -> dict[str, object]:
    checkpoint = _checkpoint_file(repo, run_name)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("args"), Mapping):
        raise RuntimeError(f"Checkpoint lacks saved args: {checkpoint}")
    args = payload["args"]
    required = {
        "gocube_b05_dry_run": True,
        "gocube_komi": GOCUBE_KOMI,
        "gocube_topology": "cube",
        "gocube_size": 4,
        "train_batch_size": CUBE4_PRODUCTION.train_batch_size,
        "gocube_experiment_contract_id": "gocube-b-experiment-contract-v1",
    }
    for key, expected in required.items():
        actual = args.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"{run_name} checkpoint metadata drift for {key}: {actual!r}")
        elif actual != expected:
            raise RuntimeError(f"{run_name} checkpoint metadata drift for {key}: {actual!r}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError(f"{run_name} checkpoint has no state_dict")
    nonfinite = [
        key for key, value in state_dict.items()
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise RuntimeError(f"{run_name} checkpoint has non-finite tensors: {nonfinite[:3]}")
    return {
        "path": str(checkpoint),
        "sha256": _sha256_file(checkpoint),
        "iteration": int(re.search(r"iteration-(\d+)", checkpoint.name).group(1)),
        "profile": args.get("gocube_model_profile"),
        "args": effective_parameter_snapshot(args),
        "state_dict_tensor_count": len(state_dict),
        "state_dict_finite": True,
    }


def _production_loader_check(repo: Path, run_name: str, device: str) -> dict[str, object]:
    metadata = _checkpoint_metadata(repo, run_name)
    checkpoint_id = f"{run_name}@{metadata['iteration']}"
    catalog = CheckpointCatalog(str(repo / "checkpoint"))
    descriptor, model = CheckpointModelLoader(catalog, device=device).load(checkpoint_id)
    del model
    if descriptor.iteration != metadata["iteration"]:
        raise RuntimeError("Production checkpoint loader selected the wrong iteration")
    return {
        "checkpoint_id": checkpoint_id,
        "descriptor": descriptor.to_api(),
        "loader": "CheckpointModelLoader",
        "device": device,
    }


def _training_metrics(repo: Path, run_name: str, log_paths: list[Path]) -> dict[str, object]:
    progress = load_training_progress("data", run_name)
    if progress is None:
        raise RuntimeError(f"Missing training-progress.json for {run_name}")
    counters = CumulativeTrainingCounters.from_mapping(progress)
    checkpoint = _checkpoint_file(repo, run_name)
    records = sorted((repo / "data" / run_name / "records").glob("iteration-*/iteration-manifest.json"))
    if not records:
        raise RuntimeError(f"Missing iteration manifest for {run_name}")
    manifest = _json_load(records[-1])
    aggregate = manifest.get("aggregate_metrics", {})
    training = aggregate.get("training", {}) if isinstance(aggregate, Mapping) else {}
    sample = aggregate.get("sample_accounting", {}) if isinstance(aggregate, Mapping) else {}
    losses = []
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for pattern in (r"Loss_pi:\s*([0-9.eE+-]+)", r"Loss_v:\s*([0-9.eE+-]+)"):
            losses.extend(float(value) for value in re.findall(pattern, text))
    if not losses or not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"No finite optimizer loss evidence for {run_name}")
    return {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "progress": progress,
        "counters": counters.as_dict(),
        "latest_iteration": int(progress.get("latest_iteration", -1)),
        "iteration_manifest": str(records[-1]),
        "sample_accounting": dict(sample) if isinstance(sample, Mapping) else {},
        "training_metrics": dict(training) if isinstance(training, Mapping) else {},
        "finite_loss_values": losses,
        "checkpoint_bytes": int(checkpoint.stat().st_size),
    }


def _state_delta(repo: Path, run_name: str) -> dict[str, object]:
    first = _checkpoint_file(repo, run_name, 1)
    latest = _checkpoint_file(repo, run_name)
    if first == latest:
        return {"initial_checkpoint": str(first), "final_checkpoint": str(latest), "changed": False}
    def load(path: Path):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        return payload["state_dict"]
    left, right = load(first), load(latest)
    changed = any(not torch.equal(left[key], right[key]) for key in left if key in right)
    return {
        "initial_checkpoint": str(first),
        "final_checkpoint": str(latest),
        "changed": bool(changed),
    }


def _run_training(repo: Path, root: Path, suite: Path, contract: Path, treatment: str, telemetry: HardwareTelemetry) -> tuple[dict[str, object], list[Path]]:
    run_name = f"gocube-b05-{treatment.lower()}"
    if (repo / "checkpoint" / run_name).exists() or (repo / "data" / run_name).exists():
        raise RuntimeError(f"Refusing to overwrite existing B05 run namespace: {run_name}")
    logs = []
    base = [
        str(repo / ".venv" / "bin" / "python"),
        "tools/gocube_b_experiment.py",
        "--treatment", treatment,
        "--run-name", run_name,
        "--heldout-suite", str(suite),
        "--contract-path", str(contract),
        "--seed", "0",
        "--iterations", "1",
        "--b05-dry-run",
        "--b05-segment",
    ]
    first_log = root / "logs" / f"{treatment.lower()}-initial.log"
    _run_command(base, repo=repo, log_path=first_log, phase=f"{treatment}_INITIAL", telemetry=telemetry)
    logs.append(first_log)
    resume = base.copy()
    resume[resume.index("--iterations") + 1] = "2"
    resume.extend(["--allow-existing-run", "--b05-resume-segment"])
    resume_log = root / "logs" / f"{treatment.lower()}-resume.log"
    _run_command(resume, repo=repo, log_path=resume_log, phase=f"{treatment}_RESUME", telemetry=telemetry)
    logs.append(resume_log)
    metrics = _training_metrics(repo, run_name, logs)
    metadata = _checkpoint_metadata(repo, run_name)
    profile = B0_MODEL_PROFILE if treatment == B0_TREATMENT else B1_MODEL_PROFILE
    if metadata["profile"] != profile:
        raise RuntimeError(f"{treatment} checkpoint profile is {metadata['profile']!r}, expected {profile!r}")
    metrics["checkpoint_metadata"] = metadata
    metrics["production_loader"] = _production_loader_check(repo, run_name, "cuda")
    metrics["checkpoint_state_delta"] = _state_delta(repo, run_name)
    if not metrics["checkpoint_state_delta"]["changed"]:
        raise RuntimeError(f"{treatment} optimizer did not change checkpoint parameters")
    return metrics, logs


def _throughput_benchmark(repo: Path, root: Path, workers: int, telemetry: HardwareTelemetry) -> dict[str, object]:
    run_name = f"gocube-b05-throughput-w{workers}"
    if (repo / "checkpoint" / run_name).exists() or (repo / "data" / run_name).exists():
        raise RuntimeError(f"Refusing to overwrite throughput namespace: {run_name}")
    command = [
        str(repo / ".venv" / "bin" / "python"),
        "-m", "alphazero.envs.gocube.train",
        "--topology", "cube", "--size", "4",
        "--workers", str(workers), "--sims", "50", "--arena-sims", "50",
        "--games-per-iteration", str(THROUGHPUT_GAMES), "--iterations", "1",
        "--train-batch-size", "1024", "--train-steps-per-iteration", "1",
        "--fast-game-prob", "0.25", "--no-arena", "--run-name", run_name,
    ]
    log = root / "logs" / f"throughput-w{workers}.log"
    command_metrics = _run_command(command, repo=repo, log_path=log, phase=f"THROUGHPUT_W{workers}", telemetry=telemetry)
    records = sorted((repo / "data" / run_name / "records").glob("iteration-*/iteration-manifest.json"))
    if not records:
        raise RuntimeError(f"Throughput benchmark produced no iteration manifest: {run_name}")
    manifest = _json_load(records[-1])
    aggregate = manifest.get("aggregate_metrics", {})
    sample = aggregate.get("sample_accounting", {}) if isinstance(aggregate, Mapping) else {}
    games = int(sample.get("selfplay_games_completed", sample.get("games", 0)))
    positions = int(sample.get("positions_generated", sample.get("base_positions", 0)))
    if games <= 0 or positions <= 0:
        raise RuntimeError(f"Throughput benchmark has no positive game/position counters: {run_name}")
    wall = float(command_metrics["wall_time_seconds"])
    return {
        "workers": workers,
        "run_name": run_name,
        "command": command,
        "log": str(log),
        "games": games,
        "positions": positions,
        "wall_time_seconds": wall,
        "games_per_second": games / wall if wall else 0.0,
        "positions_per_second": positions / wall if wall else 0.0,
        "sample_time_seconds": command_metrics["sample_time_seconds"],
        "mean_inference_batch_rows": command_metrics["mean_inference_batch_rows"],
    }


def _inference_microbenchmark(repo: Path, root: Path, device: str) -> dict[str, object]:
    results = {}
    for profile_name in (B0_MODEL_PROFILE, B1_MODEL_PROFILE):
        cli = parse_training_args([
            "--model-profile", profile_name, "--topology", "cube", "--size", "4",
            "--workers", "2", "--sims", "50", "--arena-sims", "50",
            "--games-per-iteration", "4", "--train-batch-size", "1024",
            "--cumulative-new-samples-target", str(B05_DRY_RUN_TARGET),
            "--run-name", f"gocube-b05-bench-{profile_name}", "--no-arena",
            "--b05-dry-run", "--b05-config-resolution",
        ])
        game_cls, args = build_hardened_training_args(cli)
        args.cuda = device == "cuda"
        wrapper = AtomicSampleClockNNetWrapper(game_cls, args)
        observation = game_cls().observation()
        batch = torch.from_numpy(np.repeat(observation[None, ...], INFERENCE_BATCH_ROWS, axis=0)).float()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            wrapper.process_for_search(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        durations = []
        for _ in range(INFERENCE_REPEATS):
            started = time.perf_counter()
            wrapper.process_for_search(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device == "cuda" else None
        median = float(np.median(durations))
        results[profile_name] = {
            "profile": profile_name,
            "device": device,
            "batch_rows": INFERENCE_BATCH_ROWS,
            "repeats": INFERENCE_REPEATS,
            "median_seconds": median,
            "mean_seconds": float(np.mean(durations)),
            "rows_per_second_median": INFERENCE_BATCH_ROWS / median if median else 0.0,
            "cuda_peak_memory_mib": peak,
            "observation_shape": list(observation.shape),
        }
        del wrapper, batch
        if device == "cuda":
            torch.cuda.empty_cache()
    b0 = results[B0_MODEL_PROFILE]
    b1 = results[B1_MODEL_PROFILE]
    results["comparison"] = {
        "median_latency_ratio_b1_over_b0": b1["median_seconds"] / b0["median_seconds"],
        "median_latency_delta_percent": 100.0 * (b1["median_seconds"] / b0["median_seconds"] - 1.0),
        "peak_memory_delta_mib": (
            None if b0["cuda_peak_memory_mib"] is None else b1["cuda_peak_memory_mib"] - b0["cuda_peak_memory_mib"]
        ),
    }
    return results


def _storage_extrapolation(repo: Path, root: Path, runs: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    def tree_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())
    per_treatment = {}
    for treatment, details in runs.items():
        name = str(details["run_name"])
        paths = {
            "checkpoint": repo / "checkpoint" / name,
            "replay": repo / "data" / name,
            "logs": root / "logs",
        }
        per_treatment[treatment] = {
            key: tree_bytes(path) if key != "logs" else 0
            for key, path in paths.items()
        }
        per_treatment[treatment]["total"] = sum(per_treatment[treatment].values())
    pair_bytes = per_treatment[B0_TREATMENT]["total"] + per_treatment[B1_TREATMENT]["total"]
    usage = shutil.disk_usage(repo)
    reserve = 5 * 1024**3
    estimates = {}
    for replicate_count in (3, 5):
        estimate = pair_bytes * replicate_count
        estimates[f"{replicate_count}+{replicate_count}"] = {
            "replicate_count_per_treatment": replicate_count,
            "real_pair_artifact_bytes": pair_bytes,
            "estimated_new_bytes": estimate,
            "reserve_bytes": reserve,
            "required_free_bytes": estimate + reserve,
            "filesystem_free_bytes": int(usage.free),
            "safety_margin_bytes": int(usage.free) - estimate - reserve,
            "ok": int(usage.free) >= estimate + reserve,
        }
    if not all(bool(item["ok"]) for item in estimates.values()):
        raise RuntimeError("Storage extrapolation falls below the mandatory free-disk reserve")
    return {"per_treatment": per_treatment, "estimates": estimates}


def _manifest_diff(record: Mapping[str, object]) -> dict[str, object]:
    configs = record.get("effective_configs")
    if not isinstance(configs, Mapping) or not isinstance(configs.get("B0"), Mapping) or not isinstance(configs.get("B1"), Mapping):
        raise RuntimeError("B contract record lacks effective B0/B1 configs")
    differences = diff_effective_configs(configs["B0"], configs["B1"])
    validate_b0_b1_effective_configs(configs["B0"], configs["B1"])
    return {
        "effective_config_diff_outside_whitelist": differences,
        "allowed_effective_config_differences": list(ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES),
        "same_batch_size": configs["B0"].get("train_batch_size") == configs["B1"].get("train_batch_size") == 1024,
        "same_common_contract": not differences,
    }


def _run_manifest_diff(repo: Path, runs: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    artifacts = {}
    for treatment, details in runs.items():
        run_name = str(details["run_name"])
        run_root = repo / "checkpoint" / run_name
        artifacts[treatment] = {}
        for filename in ("run-manifest.json", "effective-config.json"):
            path = run_root / filename
            if not path.is_file():
                raise RuntimeError(f"Missing hardened run manifest artifact: {path}")
            artifacts[treatment][filename] = _json_load(path)
    effective_diff = diff_effective_configs(
        artifacts[B0_TREATMENT]["effective-config.json"],
        artifacts[B1_TREATMENT]["effective-config.json"],
    )
    return {
        "artifacts": {
            treatment: {filename: str(repo / "checkpoint" / str(runs[treatment]["run_name"]) / filename)
                        for filename in artifacts[treatment]}
            for treatment in artifacts
        },
        "effective_config_diff_outside_whitelist": effective_diff,
        "same_common_manifest_config": not effective_diff,
    }


def _render_markdown(report: Mapping[str, object]) -> str:
    status = report.get("status", "UNKNOWN")
    lines = [
        "# GoCube B05 Legion preflight + dry-run",
        "",
        f"Status: **{status}**",
        f"Source: `{(report.get('preflight') or {}).get('source', {}).get('head_sha', 'unknown')}`",
        "",
        "B05 is integration evidence only. It is not a pilot, a full scientific run, or a strength claim.",
        "",
        "## Gate results",
        "",
    ]
    for key, value in report.get("gates", {}).items():
        lines.append(f"- {key}: `{value}`")
    failure = report.get("failure")
    if failure:
        lines.extend(["", "## Fail-closed reason", "", f"`{failure.get('type')}: {failure.get('message')}`"])
    for title, key in (
        ("Komi audit", "komi_audit"),
        ("B05 RAM gate", "b05_ram_gate"),
        ("Frozen suite", "frozen_suite"),
        ("Effective configs", "effective_configs"),
        ("Run manifests diff", "manifests_diff"),
        ("Throughput", "throughput"),
        ("Inference microbenchmark", "inference_microbenchmark"),
        ("Training runs", "training_runs"),
        ("Cross-profile evaluation", "cross_profile_evaluation"),
        ("B4 analyzer boundary", "analyzer_boundary"),
        ("Storage extrapolation", "storage_extrapolation"),
        ("Hardware telemetry", "hardware_telemetry"),
        ("Background load", "background_load"),
    ):
        if key in report:
            lines.extend(["", f"## {title}", "", "```json", json.dumps(report[key], indent=2, sort_keys=True), "```"])
    return "\n".join(lines) + "\n"


def _write_report(root: Path, report: dict[str, object]) -> None:
    report["report_schema_version"] = REPORT_SCHEMA_VERSION
    report["updated_at_epoch"] = time.time()
    _atomic_json(root / "b05-report.json", report)
    (root / "b05-report.md").write_text(_render_markdown(report), encoding="utf-8")


def _run_pipeline(repo: Path, args: argparse.Namespace) -> int:
    root = Path(args.report_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "RUNNING",
        "request": {
            "scope": "B05 Legion preflight and tiny B0/B1 end-to-end dry-run",
            "scientific_run": False,
            "dry_run_target": B05_DRY_RUN_TARGET,
            "dry_run_settings": dict(B05_DRY_RUN_DEFAULT_SETTINGS),
        },
        "gates": {},
    }
    telemetry = HardwareTelemetry(root / "hardware-telemetry.jsonl", interval_s=1.0)
    telemetry.start()
    try:
        preflight = collect_production_preflight(repo, argparse.Namespace(device=args.device), verify_remote=True)
        report["preflight"] = preflight
        observed_ram_bytes = int((preflight.get("hardware") or {}).get("physical_memory_bytes", 0))
        report["b05_ram_gate"] = {
            "minimum_ram_bytes": B05_MIN_RAM_BYTES,
            "minimum_ram_gib": B05_MIN_RAM_BYTES / (1024.0**3),
            "observed_physical_memory_bytes": observed_ram_bytes,
            "observed_physical_memory_gib": observed_ram_bytes / (1024.0**3),
            "passed": observed_ram_bytes >= B05_MIN_RAM_BYTES,
        }
        report["background_load"] = _background_snapshot(repo)
        report["komi_audit"] = _komi_audit(repo)
        report["gates"]["komi_7_5_audit"] = "PASS"
        _write_report(root, report)
        validate_b05_production_preflight(preflight)
        report["gates"]["production_preflight"] = "PASS"

        suite = B_HELDOUT_SUITE_PATH
        suite_check = subprocess.run(
            [str(repo / ".venv" / "bin" / "python"), "tools/build_gocube_b_heldout_suite.py", "--check"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if suite_check.returncode != 0:
            raise RuntimeError(f"Frozen suite subprocess check failed: {suite_check.stdout.strip()}")
        suite_payload, positions = validate_frozen_suite(suite, expected_sha256=B_HELDOUT_SUITE_SHA256)
        if len(positions) != B_HELDOUT_SUITE_POSITION_COUNT:
            raise RuntimeError(f"Frozen suite position count mismatch: {len(positions)}")
        report["frozen_suite"] = {
            "path": str(suite),
            "sha256": _sha256_file(suite),
            "expected_sha256": B_HELDOUT_SUITE_SHA256,
            "position_count": len(positions),
            "komi": suite_payload.get("komi"),
            "rules_fingerprint": suite_payload.get("rules_fingerprint"),
            "subprocess_check": {
                "command": "tools/build_gocube_b_heldout_suite.py --check",
                "return_code": int(suite_check.returncode),
                "output": suite_check.stdout.strip(),
            },
        }
        report["gates"]["frozen_b4_suite"] = "PASS"
        contract_path = root / "gocube-b05-experiment-contract.json"
        target = SampleBudgetTarget(SampleBudgetTarget.NEW_SAMPLES, B05_DRY_RUN_TARGET)
        record = preflight_b_experiment(
            repo=repo,
            contract_path=contract_path,
            heldout_suite_path=suite,
            scientific_target=target,
            dry_run_settings=B05_DRY_RUN_DEFAULT_SETTINGS,
        )
        record = validate_b_experiment_record(contract_path)
        report["contract_path"] = str(contract_path)
        report["contract_sha256"] = record["experiment_contract_sha256"]
        report["effective_configs"] = _manifest_diff(record)
        report["effective_configs"]["record"] = record
        report["gates"]["separate_process_effective_configs"] = "PASS"
        _write_report(root, report)

        if args.device != "cuda":
            raise RuntimeError("B05 workload requires CUDA after preflight; CPU is diagnostic-only")
        b0, b0_logs = _run_training(repo, root, suite, contract_path, B0_TREATMENT, telemetry)
        b1, b1_logs = _run_training(repo, root, suite, contract_path, B1_TREATMENT, telemetry)
        report["training_runs"] = {B0_TREATMENT: b0, B1_TREATMENT: b1}
        report["gates"]["b0_b1_training_resume_checkpoint"] = "PASS"
        report["throughput"] = {
            "workers_8": _throughput_benchmark(repo, root, 8, telemetry),
            "workers_16": _throughput_benchmark(repo, root, 16, telemetry),
        }
        report["gates"]["throughput_8_vs_16"] = "PASS"
        report["inference_microbenchmark"] = _inference_microbenchmark(repo, root, args.device)
        report["gates"]["b0_b1_inference_microbenchmark"] = "PASS"
        report["manifests_diff"] = _run_manifest_diff(repo, report["training_runs"])
        report["gates"]["run_manifests_diff"] = "PASS"
        report["storage_extrapolation"] = _storage_extrapolation(repo, root, report["training_runs"])
        report["gates"]["storage_3plus3_5plus5"] = "PASS"

        evaluation_path = root / "artifacts" / "b05-seed0-m128.json"
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluate_b05_main([
            "--b0-checkpoint", str(b0["checkpoint_metadata"]["path"]),
            "--b1-checkpoint", str(b1["checkpoint_metadata"]["path"]),
            "--suite", str(suite), "--experiment-contract", str(contract_path),
            "--training-seed", "0", "--sample-milestone", str(EVALUATION_MILESTONE),
            "--device", args.device, "--output", str(evaluation_path),
        ])
        evaluation = _json_load(evaluation_path)
        report["cross_profile_evaluation"] = {
            "artifact": str(evaluation_path),
            "training_seed": evaluation.get("training_seed"),
            "scientific_milestone": evaluation.get("scientific_milestone"),
            "number_of_games": len(evaluation.get("games", [])),
            "position_count": evaluation.get("position_count"),
            "seed_score_b1": evaluation.get("seed_score_b1"),
            "seed_delta": evaluation.get("seed_delta"),
            "non_scientific_dry_run": evaluation.get("non_scientific_dry_run"),
        }
        try:
            analyze_gocube_b_evaluation.validate_seed_evaluation(
                evaluation,
                evaluation_schedule=record["experiment_contract"]["evaluation_milestones"],
                experiment_contract_sha256=str(record["experiment_contract_sha256"]),
            )
        except ValueError as exc:
            report["analyzer_boundary"] = {
                "b05_artifact": str(evaluation_path),
                "scientific_analyzer_accepts": False,
                "rejection": str(exc),
            }
        else:
            raise RuntimeError("Scientific B4 analyzer accepted a B05 dry-run artifact")
        report["gates"]["b4_analyzer_rejects_non_scientific_artifact"] = "PASS"
        report["hardware_telemetry"] = telemetry.summary()
        report["gates"]["report_and_telemetry"] = "PASS"
        report["status"] = "PASS"
        _write_report(root, report)
        return 0
    except Exception as exc:
        report["status"] = "BLOCKED"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "fail_closed": True,
        }
        report["hardware_telemetry"] = telemetry.summary()
        _write_report(root, report)
        print(f"B05 STOPPED FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Report: {root / 'b05-report.md'}", file=sys.stderr)
        return 1
    finally:
        telemetry.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


def _systemd_command(repo: Path, args: argparse.Namespace) -> tuple[str, list[str]]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(args.report_dir).name).strip("-.") or "b05"
    unit = f"gocube-b05-{safe}"[:120]
    command = [
        "systemd-run", "--user", "--wait", "--pipe", "--collect", "--unit", unit,
        f"--working-directory={repo}", "--property=Restart=no",
        "--property=TimeoutStopSec=30s", "--property=KillMode=mixed",
        f"--setenv={SUPERVISED_ENV}=1", f"--setenv={UNIT_ENV}={unit}",
        "--setenv=PYTHONUNBUFFERED=1", str(repo / ".venv" / "bin" / "python"),
        str(repo / "tools" / "gocube_b05.py"), "--device", str(args.device),
        "--report-dir", str(args.report_dir),
    ]
    return unit, command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path.cwd().resolve()
    if not is_supervised_invocation():
        _assert_supervised_launch_ready(repo)
        unit, command = _systemd_command(repo, args)
        environment = os.environ.copy()
        environment.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        bus = Path(environment["XDG_RUNTIME_DIR"]) / "bus"
        if bus.exists():
            environment.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}")
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            cwd=repo,
            env=environment,
            check=False,
        )
        if active.returncode == 0:
            raise SystemExit(f"B05 service already active: {unit}")
        return int(subprocess.run(command, cwd=repo, env=environment, check=False).returncode)
    return _run_pipeline(repo, args)


if __name__ == "__main__":
    raise SystemExit(main())
