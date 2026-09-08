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
    DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET,
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
from alphazero.envs.gocube.atomic_io import REPLAY_ARTIFACT_SUFFIXES
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
THROUGHPUT_GAMES = 64
THROUGHPUT_REPEATS = 3
THROUGHPUT_TARGET_GRACE_SECONDS = 120.0
INFERENCE_BATCH_ROWS = 32
INFERENCE_REPEATS = 12
EVALUATION_MILESTONE = 128
REPLAY_WINDOW_MAX_ITERATIONS = 20
CHECKPOINT_CADENCE_ITERATIONS = 1
STORAGE_OPERATIONAL_HEADROOM_BYTES = 1 * 1024**3
STORAGE_TRANSIENT_SAFETY_FACTOR = 2.0


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
    benchmark_target_games: int | None = None,
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
    target_games_elapsed = None
    last_target_games_elapsed = None
    replay_save_started_elapsed = None
    iteration_summary_elapsed = None
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
            observed_elapsed = time.perf_counter() - started
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            if benchmark_target_games is not None:
                target_marker = rf"\({int(benchmark_target_games)}/{int(benchmark_target_games)}\)"
                if re.search(target_marker, line):
                    if target_games_elapsed is None:
                        target_games_elapsed = observed_elapsed
                    last_target_games_elapsed = observed_elapsed
                if line.startswith("Saving ") and replay_save_started_elapsed is None:
                    replay_save_started_elapsed = observed_elapsed
                if line.startswith("=== V3 iteration ") and iteration_summary_elapsed is None:
                    iteration_summary_elapsed = observed_elapsed
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
    if benchmark_target_games is not None:
        if target_games_elapsed is None:
            raise RuntimeError(
                f"Benchmark command never reported target game count {benchmark_target_games}: "
                f"{' '.join(command)}"
            )
        result.update(
            {
                "benchmark_elapsed_seconds": float(target_games_elapsed),
                "target_games_elapsed_seconds": float(target_games_elapsed),
                "last_target_games_elapsed_seconds": (
                    float(last_target_games_elapsed)
                    if last_target_games_elapsed is not None
                    else float(target_games_elapsed)
                ),
                "legacy_worker_pool_drain_seconds": max(
                    0.0,
                    float(last_target_games_elapsed or target_games_elapsed) - float(target_games_elapsed),
                ),
                "replay_save_started_elapsed_seconds": (
                    float(replay_save_started_elapsed)
                    if replay_save_started_elapsed is not None
                    else None
                ),
                "iteration_summary_elapsed_seconds": (
                    float(iteration_summary_elapsed)
                    if iteration_summary_elapsed is not None
                    else None
                ),
                "runner_overhead_seconds": max(0.0, wall - float(target_games_elapsed)),
                "legacy_runner_grace_seconds": THROUGHPUT_TARGET_GRACE_SECONDS,
            }
        )
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


def _throughput_benchmark(
    repo: Path,
    root: Path,
    workers: int,
    repeat: int,
    telemetry: HardwareTelemetry,
) -> dict[str, object]:
    run_name = f"gocube-b05-throughput-w{workers}-r{repeat}"
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
    command_metrics = _run_command(
        command,
        repo=repo,
        log_path=log,
        phase=f"THROUGHPUT_W{workers}",
        telemetry=telemetry,
        benchmark_target_games=THROUGHPUT_GAMES,
    )
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
    benchmark_elapsed = float(command_metrics["benchmark_elapsed_seconds"])
    return {
        "workers": workers,
        "repeat": repeat,
        "run_name": run_name,
        "command": command,
        "log": str(log),
        "games": games,
        "positions": positions,
        "wall_time_seconds": wall,
        "games_per_second": games / benchmark_elapsed if benchmark_elapsed else 0.0,
        "positions_per_second": positions / benchmark_elapsed if benchmark_elapsed else 0.0,
        "sample_time_seconds": command_metrics["sample_time_seconds"],
        "mean_inference_batch_rows": command_metrics["mean_inference_batch_rows"],
        "benchmark_elapsed_seconds": benchmark_elapsed,
        "target_games_elapsed_seconds": command_metrics["target_games_elapsed_seconds"],
        "last_target_games_elapsed_seconds": command_metrics["last_target_games_elapsed_seconds"],
        "legacy_worker_pool_drain_seconds": command_metrics["legacy_worker_pool_drain_seconds"],
        "replay_save_started_elapsed_seconds": command_metrics["replay_save_started_elapsed_seconds"],
        "iteration_summary_elapsed_seconds": command_metrics["iteration_summary_elapsed_seconds"],
        "runner_overhead_seconds": command_metrics["runner_overhead_seconds"],
        "legacy_runner_grace_seconds": command_metrics["legacy_runner_grace_seconds"],
    }


def _summarize_throughput(
    workers_8: list[Mapping[str, object]],
    workers_16: list[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize paired worker runs without changing the production default."""

    if len(workers_8) != len(workers_16) or not workers_8:
        raise ValueError("throughput comparison requires equally sized non-empty repeat sets")
    if len(workers_8) < 2:
        raise ValueError("throughput comparison requires at least two repeats")

    def summarize(items: list[Mapping[str, object]], workers: int) -> dict[str, object]:
        rates = [float(item["positions_per_second"]) for item in items]
        games = [int(item["games"]) for item in items]
        if any(int(item["workers"]) != workers for item in items):
            raise ValueError(f"throughput repeat has the wrong worker count for w{workers}")
        if any(game_count < THROUGHPUT_GAMES for game_count in games):
            raise ValueError(
                f"throughput repeat did not complete the requested {THROUGHPUT_GAMES} games"
            )
        return {
            "workers": workers,
            "repeat_count": len(items),
            "games_per_repeat": THROUGHPUT_GAMES,
            "repeats": items,
            "positions_per_second": {
                "median": float(np.median(rates)),
                "mean": float(np.mean(rates)),
                "min": float(min(rates)),
                "max": float(max(rates)),
            },
        }

    summary_8 = summarize(workers_8, 8)
    summary_16 = summarize(workers_16, 16)
    rates_8 = [float(item["positions_per_second"]) for item in workers_8]
    rates_16 = [float(item["positions_per_second"]) for item in workers_16]
    paired_deltas = [
        100.0 * (right / left - 1.0) if left else 0.0
        for left, right in zip(rates_8, rates_16)
    ]
    median_8 = float(summary_8["positions_per_second"]["median"])
    median_16 = float(summary_16["positions_per_second"]["median"])
    slower_repeats = sum(delta < 0.0 for delta in paired_deltas)
    stable_regression = median_16 < median_8 and slower_repeats >= math.ceil(len(paired_deltas) / 2)
    return {
        "benchmark": {
            "games_per_repeat": THROUGHPUT_GAMES,
            "repeats": len(workers_8),
            "metric": "median positions/sec",
            "paired_repeat_indices": list(range(1, len(workers_8) + 1)),
        },
        "workers_8": summary_8,
        "workers_16": summary_16,
        "comparison": {
            "median_positions_per_second_workers_8": median_8,
            "median_positions_per_second_workers_16": median_16,
            "relative_delta_percent_16_vs_8": 100.0 * (median_16 / median_8 - 1.0)
            if median_8
            else 0.0,
            "paired_repeat_deltas_percent_16_vs_8": paired_deltas,
            "slower_repeat_count": slower_repeats,
            "stable_workers_16_regression": stable_regression,
            "canonical_workers": CUBE4_PRODUCTION.workers,
            "canonical_workers_changed": False,
        },
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


def _storage_extrapolation(
    repo: Path,
    root: Path,
    runs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Estimate the full B-run footprint from measured B05 artifacts.

    B05 is too small to extrapolate by multiplying its total directory size.
    The scientific run is sample-clocked, writes one checkpoint per iteration,
    and trains from a maximum 20-iteration replay window.  The report keeps
    both the contract-retained estimate and a conservative view of the current
    launcher, which does not delete old replay/record artifacts on disk.
    """

    def tree_bytes(path: Path, *, include=None) -> int:
        if not path.exists():
            return 0
        total = 0
        for item in path.rglob("*"):
            if not item.is_file() or item.is_symlink():
                continue
            if include is None or include(item):
                total += item.stat().st_size
        return total

    def files_bytes(paths) -> int:
        return sum(path.stat().st_size for path in paths if path.is_file())

    def artifact_measurement(treatment: str, details: Mapping[str, object]) -> dict[str, object]:
        name = str(details["run_name"])
        checkpoint_root = repo / "checkpoint" / name
        data_root = repo / "data" / name
        runs_root = repo / "runs" / name
        checkpoint_files = sorted(checkpoint_root.glob("iteration-*.pkl"))
        data_files = [path for path in data_root.rglob("*") if path.is_file()]
        replay_files = [
            path for path in data_files
            if path.name.endswith(tuple(REPLAY_ARTIFACT_SUFFIXES))
            or path.name.endswith("-complete.json")
        ]
        game_record_files = [
            path for path in data_files
            if "records" in path.parts and path.name.startswith("C4-") and path.suffix == ".json"
        ]
        iteration_manifest_files = [
            path for path in data_files if path.name == "iteration-manifest.json"
        ]
        progress_files = [path for path in data_files if path.name == "training-progress.json"]
        checkpoint_metadata_files = [
            path for path in checkpoint_root.iterdir()
            if path.is_file() and not re.fullmatch(r"iteration-\d+\.pkl", path.name)
        ] if checkpoint_root.exists() else []
        log_paths = [Path(path) for path in details.get("log_paths", [])]
        command_log_bytes = files_bytes(log_paths)
        tensorboard_log_bytes = tree_bytes(runs_root)
        report_bytes = tree_bytes(
            root,
            include=lambda path: "logs" not in path.relative_to(root).parts,
        )

        counters = details.get("counters")
        if not isinstance(counters, Mapping):
            raise RuntimeError(f"Missing B05 counters for storage measurement: {treatment}")
        measured_samples = int(counters.get("new_samples_accepted", 0))
        measured_games = int(counters.get("selfplay_games_completed", 0))
        measured_iterations = int(details.get("latest_iteration", 0))
        if measured_samples <= 0 or measured_games <= 0 or measured_iterations <= 0:
            raise RuntimeError(f"Invalid B05 counters for storage measurement: {treatment}")

        replay_bytes = files_bytes(replay_files)
        game_record_bytes = files_bytes(game_record_files)
        iteration_manifest_bytes = files_bytes(iteration_manifest_files)
        progress_bytes = files_bytes(progress_files)
        initial_checkpoint_bytes = next(
            (path.stat().st_size for path in checkpoint_files if path.name == "iteration-0000.pkl"),
            0,
        )
        steady_checkpoints = [
            path.stat().st_size for path in checkpoint_files if path.name != "iteration-0000.pkl"
        ]
        if not steady_checkpoints:
            raise RuntimeError(f"Missing post-bootstrap checkpoints for storage measurement: {treatment}")
        measured_checkpoint_bytes = float(np.mean(steady_checkpoints))
        measured_record_games = max(1, len(game_record_files))
        measured_iteration_manifest_count = max(1, len(iteration_manifest_files))
        measured_command_log_bytes = command_log_bytes
        measured_report_share_bytes = report_bytes // 2
        return {
            "run_name": name,
            "measured_samples": measured_samples,
            "measured_games": measured_games,
            "measured_iterations": measured_iterations,
            "measured_replay_bytes": replay_bytes,
            "measured_game_record_bytes": game_record_bytes,
            "measured_checkpoint_bytes": int(sum(path.stat().st_size for path in checkpoint_files)),
            "measured_initial_checkpoint_bytes": int(initial_checkpoint_bytes),
            "measured_checkpoint_metadata_bytes": files_bytes(checkpoint_metadata_files),
            "measured_iteration_manifest_bytes": iteration_manifest_bytes,
            "measured_progress_bytes": progress_bytes,
            "measured_command_log_bytes": measured_command_log_bytes,
            "measured_tensorboard_log_bytes": tensorboard_log_bytes,
            "measured_report_share_bytes": measured_report_share_bytes,
            "measured_tiny_total_bytes": int(
                replay_bytes
                + game_record_bytes
                + sum(path.stat().st_size for path in checkpoint_files)
                + files_bytes(checkpoint_metadata_files)
                + iteration_manifest_bytes
                + progress_bytes
                + command_log_bytes
                + tensorboard_log_bytes
                + measured_report_share_bytes
            ),
            "replay_bytes_per_sample": replay_bytes / measured_samples,
            "game_record_bytes_per_game": game_record_bytes / measured_record_games,
            "samples_per_iteration": measured_samples / measured_iterations,
            "samples_per_game": measured_samples / measured_games,
            "checkpoint_bytes_per_iteration": measured_checkpoint_bytes,
            "iteration_manifest_bytes_per_iteration": iteration_manifest_bytes / measured_iteration_manifest_count,
            "command_log_bytes_per_iteration": command_log_bytes / measured_iterations,
            "tensorboard_log_bytes_per_iteration": tensorboard_log_bytes / measured_iterations,
            "transient_iteration_bytes": int(
                math.ceil(
                    replay_bytes / measured_iterations
                    + game_record_bytes / measured_iterations
                    + measured_checkpoint_bytes
                    + iteration_manifest_bytes / measured_iteration_manifest_count
                    + command_log_bytes / measured_iterations
                    + tensorboard_log_bytes / measured_iterations
                )
            ),
        }

    measured = {
        treatment: artifact_measurement(treatment, details)
        for treatment, details in runs.items()
    }
    target = int(DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET)
    full_run = {}
    for treatment, item in measured.items():
        # B05 uses only eight games per iteration.  Scale the measured
        # samples/game to the production 256-game iteration before deriving
        # the number of iterations; using the tiny iteration total directly
        # would inflate checkpoints, manifests, and logs by 32x.
        production_samples_per_iteration = float(item["samples_per_game"]) * int(
            CUBE4_PRODUCTION.games_per_iteration
        )
        full_iterations = int(math.ceil(target / production_samples_per_iteration))
        window_iterations = min(REPLAY_WINDOW_MAX_ITERATIONS, full_iterations)
        window_samples = production_samples_per_iteration * window_iterations
        window_games = int(CUBE4_PRODUCTION.games_per_iteration) * window_iterations
        # The bootstrap checkpoint is explicitly separated from steady-state
        # checkpoints so cadence is visible in the report and formula.
        checkpoint_bytes = int(
            int(item["measured_initial_checkpoint_bytes"])
            + round(float(item["checkpoint_bytes_per_iteration"]) * full_iterations)
        )
        manifests_bytes = int(
            item["measured_checkpoint_metadata_bytes"]
            + item["measured_progress_bytes"]
            + round(float(item["iteration_manifest_bytes_per_iteration"]) * full_iterations)
        )
        logs_bytes = int(
            round(
                (
                    float(item["command_log_bytes_per_iteration"])
                    + float(item["tensorboard_log_bytes_per_iteration"])
                )
                * full_iterations
            )
        )
        retained = {
            "replay": round(float(item["replay_bytes_per_sample"]) * window_samples),
            "game_records": round(float(item["game_record_bytes_per_game"]) * window_games),
            "checkpoint": checkpoint_bytes,
            "manifests": manifests_bytes,
            "logs": logs_bytes,
            "reports": int(item["measured_report_share_bytes"]),
        }
        unpruned = dict(retained)
        unpruned["replay"] = round(float(item["replay_bytes_per_sample"]) * target)
        unpruned["game_records"] = round(
            float(item["game_record_bytes_per_game"])
            * (int(CUBE4_PRODUCTION.games_per_iteration) * full_iterations)
        )
        full_run[treatment] = {
            "scientific_target_new_samples": target,
            "games_per_iteration": int(CUBE4_PRODUCTION.games_per_iteration),
            "measured_samples_per_game": float(item["samples_per_game"]),
            "production_samples_per_iteration": production_samples_per_iteration,
            "checkpoint_cadence_iterations": CHECKPOINT_CADENCE_ITERATIONS,
            "estimated_iterations": full_iterations,
            "replay_window_iterations": window_iterations,
            "replay_window_samples": round(window_samples),
            "replay_window_games": window_games,
            "retained_policy_bytes": retained,
            "retained_policy_total_bytes": int(sum(retained.values())),
            "unpruned_current_launcher_bytes": unpruned,
            "unpruned_current_launcher_total_bytes": int(sum(unpruned.values())),
        }

    pair_retained = sum(int(item["retained_policy_total_bytes"]) for item in full_run.values())
    pair_unpruned = sum(int(item["unpruned_current_launcher_total_bytes"]) for item in full_run.values())
    measured_pair = sum(int(item["measured_tiny_total_bytes"]) for item in measured.values())
    pair_transient_iteration_bytes = sum(
        int(item["transient_iteration_bytes"]) for item in measured.values()
    )
    usage = shutil.disk_usage(repo)
    estimates = {}
    for replicate_count in (3, 5):
        retained_estimate = pair_retained * replicate_count
        unpruned_estimate = pair_unpruned * replicate_count
        # The old 5 GiB value came from the generic storage preflight and had
        # no B-specific peak or filesystem-behavior derivation.  For this
        # full-run estimate, reserve only the measured one-iteration write
        # set, doubled for transient files and scaled by parallel pairs, with
        # a 1 GiB operational floor.
        reserve = max(
            STORAGE_OPERATIONAL_HEADROOM_BYTES,
            int(
                math.ceil(
                    pair_transient_iteration_bytes
                    * replicate_count
                    * STORAGE_TRANSIENT_SAFETY_FACTOR
                )
            ),
        )
        estimates[f"{replicate_count}+{replicate_count}"] = {
            "replicate_count_per_treatment": replicate_count,
            "measured_tiny_pair_artifact_bytes": measured_pair,
            "estimated_new_bytes": retained_estimate,
            "estimated_retained_policy_bytes": retained_estimate,
            "estimated_unpruned_current_launcher_bytes": unpruned_estimate,
            "reserve_bytes": reserve,
            "reserve_basis": "measured_transient_iteration_bytes_x_replicates_x_2_with_1GiB_floor",
            "required_free_bytes": retained_estimate + reserve,
            "unpruned_required_free_bytes": unpruned_estimate + reserve,
            "filesystem_free_bytes": int(usage.free),
            "safety_margin_bytes": int(usage.free) - retained_estimate - reserve,
            "unpruned_safety_margin_bytes": int(usage.free) - unpruned_estimate - reserve,
            "ok": int(usage.free) >= retained_estimate + reserve,
            "unpruned_current_launcher_ok": int(usage.free) >= unpruned_estimate + reserve,
        }
    if not all(bool(item["ok"]) for item in estimates.values()):
        raise RuntimeError("Storage extrapolation falls below the mandatory free-disk reserve")
    return {
        "scientific_target_new_samples_per_treatment": target,
        "retention_policy": {
            "source": "gocube-b-experiment-contract-v1",
            "replay_window_max_iterations": REPLAY_WINDOW_MAX_ITERATIONS,
            "estimate_uses": "replay and game-record artifacts retained in the training-visible window",
            "current_launcher_note": (
                "The current launcher does not delete older replay/record files; "
                "unpruned_current_launcher_* is reported separately as a conservative warning."
            ),
        },
        "formula": {
            "estimated_iterations": "ceil(40,000,000 / (measured_samples_per_game * production_games_per_iteration))",
            "retained_replay": "measured_replay_bytes_per_sample * samples_per_iteration * min(20, estimated_iterations)",
            "retained_game_records": "measured_game_record_bytes_per_game * 256 * min(20, estimated_iterations)",
            "checkpoints": "bootstrap_checkpoint + steady_checkpoint_bytes_per_iteration * estimated_iterations",
            "manifests": "measured_static_manifests + measured_iteration_manifest_bytes_per_iteration * estimated_iterations",
            "logs": "measured command/tensorboard log bytes per iteration * estimated_iterations",
            "replicates": "pair_bytes * replicate_count_per_treatment",
            "reserve": "max(1 GiB, measured_transient_iteration_bytes_pair * replicate_count_per_treatment * 2)",
        },
        "reserve_policy": {
            "legacy_generic_preflight_reserve_bytes": 5 * 1024**3,
            "legacy_generic_preflight_source": "tools.gocube_experiment_storage.MIN_FREE_RESERVE_BYTES",
            "legacy_generic_preflight_justification": "generic production free-space floor; no B-specific peak derivation found",
            "used_for_full_run_estimate": False,
            "measured_transient_iteration_bytes_pair": pair_transient_iteration_bytes,
            "transient_safety_factor": STORAGE_TRANSIENT_SAFETY_FACTOR,
            "minimum_operational_headroom_bytes": STORAGE_OPERATIONAL_HEADROOM_BYTES,
            "basis": "measured replay/record/checkpoint/manifest/log write set, with 2x transient margin",
        },
        "measured_b05": measured,
        "full_run_per_treatment": full_run,
        "pair_totals": {
            "measured_tiny_pair_artifact_bytes": measured_pair,
            "retained_policy_bytes": pair_retained,
            "unpruned_current_launcher_bytes": pair_unpruned,
        },
        "estimates": estimates,
    }


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
        b0["log_paths"] = [str(path) for path in b0_logs]
        b1["log_paths"] = [str(path) for path in b1_logs]
        report["training_runs"] = {B0_TREATMENT: b0, B1_TREATMENT: b1}
        report["gates"]["b0_b1_training_resume_checkpoint"] = "PASS"
        throughput_8 = [
            _throughput_benchmark(repo, root, 8, repeat, telemetry)
            for repeat in range(1, THROUGHPUT_REPEATS + 1)
        ]
        throughput_16 = [
            _throughput_benchmark(repo, root, 16, repeat, telemetry)
            for repeat in range(1, THROUGHPUT_REPEATS + 1)
        ]
        report["throughput"] = _summarize_throughput(throughput_8, throughput_16)
        comparison = report["throughput"]["comparison"]
        all_throughput_runs = throughput_8 + throughput_16
        legacy_drain_runs = [
            item for item in all_throughput_runs
            if float(item.get("legacy_worker_pool_drain_seconds", 0.0))
            >= THROUGHPUT_TARGET_GRACE_SECONDS
        ]
        if legacy_drain_runs:
            report.setdefault("findings", []).append(
                {
                    "id": "legacy-runner-worker-pool-drain-after-target",
                    "severity": "finding",
                    "status": "OPEN",
                    "summary": (
                        "Legacy self-play runner keeps the worker pool alive well after "
                        "the target game count; B05 records the excess as runner overhead."
                    ),
                    "evidence": {
                        "grace_seconds": THROUGHPUT_TARGET_GRACE_SECONDS,
                        "affected_runs": [
                            {
                                "run_name": item["run_name"],
                                "benchmark_elapsed_seconds": item["benchmark_elapsed_seconds"],
                                "legacy_worker_pool_drain_seconds": item[
                                    "legacy_worker_pool_drain_seconds"
                                ],
                                "runner_overhead_seconds": item["runner_overhead_seconds"],
                            }
                            for item in legacy_drain_runs
                        ],
                    },
                    "action": "Fix or bound legacy runner shutdown separately; do not expand B4 scope.",
                }
            )
        if comparison["stable_workers_16_regression"]:
            report.setdefault("findings", []).append(
                {
                    "id": "throughput-workers-16-regression-before-pilot",
                    "severity": "finding",
                    "status": "OPEN",
                    "summary": (
                        "Canonical workers=16 is stably slower than workers=8 on the "
                        "64-game repeated benchmark; production workers remain unchanged."
                    ),
                    "evidence": {
                        "median_positions_per_second_workers_8": comparison[
                            "median_positions_per_second_workers_8"
                        ],
                        "median_positions_per_second_workers_16": comparison[
                            "median_positions_per_second_workers_16"
                        ],
                        "relative_delta_percent_16_vs_8": comparison[
                            "relative_delta_percent_16_vs_8"
                        ],
                        "slower_repeat_count": comparison["slower_repeat_count"],
                        "repeat_count": THROUGHPUT_REPEATS,
                    },
                    "action": "Review workers/batching before pilot; do not change canonical workers automatically.",
                }
            )
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
