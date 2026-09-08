#!/usr/bin/env python3
"""Single public entrypoint for the autonomous, crash-resumable Cube-4 P1..P5 sweep.

Normal production use is one command with no required arguments:

    .venv/bin/python tools/c4_overnight_experiment.py

If exactly one schema-v3 experiment is RUNNING/INTERRUPTED, the zero-argument
command resumes it. Otherwise a new experiment id is generated. Use
`--new-experiment` to force a new run or `--experiment-id ID` to resume a
specific FAILED/RUNNING experiment explicitly.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from alphazero.envs.gocube.production_contract import CUBE4_PRODUCTION
from alphazero.envs.gocube.production_training import (
    CumulativeTrainingCounters,
    SampleBudgetTarget,
    load_training_progress,
)
from tools._c4_overnight_runtime import *  # noqa: F401,F403
from tools import _c4_overnight_runtime as _impl
from tools.gocube_experiment_resume import (
    STATE_SCHEMA_VERSION,
    TERMINAL_STATUSES,
    atomic_json,
    begin_action,
    complete_action_once,
    find_resumable_experiments,
    launch_config,
    validate_arena_payload,
    validate_loaded_state,
    validate_sweep_overrides,
)
from tools.gocube_overnight_safety import (
    build_fresh_heldout_suite,
    extract_last_progress_metrics,
)
from tools.hardware_telemetry import HardwareTelemetry


CANONICAL_COMMAND = ".venv/bin/python tools/c4_overnight_experiment.py"
PROVENANCE_FILENAME = ".sweep-provenance.json"

WORKERS = _impl.WORKERS = CUBE4_PRODUCTION.workers
REGULAR_SIMS = _impl.REGULAR_SIMS = CUBE4_PRODUCTION.regular_sims
FAST_SIMS = _impl.FAST_SIMS = CUBE4_PRODUCTION.fast_sims
GAMES_PER_ITERATION = _impl.GAMES_PER_ITERATION = CUBE4_PRODUCTION.games_per_iteration
TRAIN_BATCH_SIZE = _impl.TRAIN_BATCH_SIZE = CUBE4_PRODUCTION.train_batch_size
ARENA_SIMS = _impl.ARENA_SIMS = CUBE4_PRODUCTION.arena_sims
EXPECTED_KOMI = _impl.EXPECTED_KOMI = CUBE4_PRODUCTION.komi
DEFAULT_CANDIDATE_NEW_SAMPLE_BUDGET = _impl.DEFAULT_CANDIDATE_NEW_SAMPLE_BUDGET

build_frozen_heldout_suite = build_fresh_heldout_suite


class SweepInterrupted(Exception):
    def __init__(self, signum: int):
        self.signum = int(signum)
        super().__init__(f"sweep interrupted by signal {self.signum}")


def _default_experiment_id() -> str:
    base = time.strftime("c4-sweep-%Y%m%d-%H%M%S")
    reports = Path.cwd() / "training_reports"
    if not (reports / base).exists():
        return base
    suffix = 2
    while (reports / f"{base}-{suffix}").exists():
        suffix += 1
    return f"{base}-{suffix}"


def _has_option(argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in argv)


def parse_args(argv=None):
    """Parse CLI, auto-resuming the unique interrupted experiment when safe."""

    raw = list(sys.argv[1:] if argv is None else argv)
    force_new = "--new-experiment" in raw
    raw = [token for token in raw if token != "--new-experiment"]
    explicit_id = _has_option(raw, "--experiment-id")
    if force_new and explicit_id:
        raise RuntimeError("--new-experiment cannot be combined with --experiment-id")
    auto_resumed = False
    if not explicit_id:
        if not force_new:
            resumable = find_resumable_experiments(Path.cwd() / "training_reports")
            if len(resumable) > 1:
                raise RuntimeError(
                    "Multiple resumable Cube-4 experiments exist; choose one with --experiment-id "
                    "or use --new-experiment."
                )
            if resumable:
                raw = ["--experiment-id", resumable[0], *raw]
                auto_resumed = True
        if not _has_option(raw, "--experiment-id"):
            raw = ["--experiment-id", _default_experiment_id(), *raw]
    cli = _impl.parse_args(raw)
    cli.gocube_auto_resumed = auto_resumed
    cli.gocube_force_new = force_new
    return cli


class Experiment(_impl.Experiment):
    """P1..P5 orchestration with state/artifact-level crash recovery."""

    def __init__(self, cli):
        self.cli = cli
        self.repo = Path.cwd().resolve()
        self.python = self.repo / ".venv" / "bin" / "python"
        if not self.python.is_file():
            raise RuntimeError(f"Missing project Python: {self.python}")
        self.root = self.repo / "training_reports" / cli.experiment_id
        self.logs = self.root / "logs"
        self.results = self.root / "arena"
        self.state_path = self.root / "experiment-state.json"
        self.report_path = self.root / "overnight-report.json"
        self.report_md_path = self.root / "overnight-report.md"
        self.heldout_path = self.root / "heldout-suite.json"
        self.telemetry = HardwareTelemetry(
            self.root / "hardware-telemetry.jsonl", interval_s=cli.telemetry_interval
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.selfplay_wait_ms = SELFPLAY_BATCH_WAIT_MS
        self.arena_wait_ms = float(cli.arena_batch_wait_ms)
        self.arena_workers = WORKERS
        self._already_terminal = False

        if cli.candidate_optimizer_examples_budget is not None:
            scientific_budget = {
                "kind": "cumulative_optimizer_examples",
                "increment": int(cli.candidate_optimizer_examples_budget),
            }
        else:
            scientific_budget = {
                "kind": "cumulative_new_samples",
                "increment": int(cli.candidate_new_samples_budget),
            }

        fixed_contract = {
            "workers": WORKERS,
            "regular_sims": REGULAR_SIMS,
            "fast_sims": FAST_SIMS,
            "games_per_iteration": GAMES_PER_ITERATION,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "initial_selfplay_inference_batch_wait_ms": SELFPLAY_BATCH_WAIT_MS,
            "arena_sims": ARENA_SIMS,
            "komi": EXPECTED_KOMI,
            "scientific_budget": scientific_budget,
        }
        expected_launch = launch_config(cli)
        now = time.time()
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            validate_loaded_state(
                payload,
                experiment_id=cli.experiment_id,
                fixed_contract=fixed_contract,
                expected_launch_config=expected_launch,
            )
            if bool(getattr(cli, "gocube_force_new", False)):
                raise RuntimeError(f"Experiment already exists: {cli.experiment_id}")
            if str(payload.get("status")) in TERMINAL_STATUSES:
                if not isinstance(payload.get("recommended_champion"), dict):
                    raise RuntimeError("Terminal experiment state is incomplete and cannot be trusted")
                self.state = payload
                self._already_terminal = True
            else:
                prior_start = payload.get("session_started_at_epoch")
                prior_update = payload.get("last_update_epoch")
                active = float(payload.get("active_runtime_seconds", 0.0))
                if isinstance(prior_start, (int, float)) and isinstance(prior_update, (int, float)):
                    active += max(0.0, float(prior_update) - float(prior_start))
                payload["active_runtime_seconds"] = active
                payload["session_started_at_epoch"] = now
                payload["resume_count"] = int(payload.get("resume_count", 0)) + 1
                if payload.get("error"):
                    payload.setdefault("error_history", []).append({
                        "error": payload["error"],
                        "recorded_at_epoch": now,
                    })
                    payload.pop("error", None)
                payload.pop("finished_at_epoch", None)
                payload["status"] = "RUNNING"
                self.state = payload
        else:
            self.state = {
                "schema_version": STATE_SCHEMA_VERSION,
                "experiment_id": cli.experiment_id,
                "status": "RUNNING",
                "started_at_epoch": now,
                "last_update_epoch": now,
                "session_started_at_epoch": now,
                "active_runtime_seconds": 0.0,
                "resume_count": 0,
                "fixed_contract": fixed_contract,
                "launch_config": expected_launch,
                "parameters": [],
                "completed_actions": {},
                "totals": {
                    "training_games": 0,
                    "benchmark_selfplay_games": 0,
                    "arena_games": 0,
                },
                "training_runs": {},
                "cumulative_counters": {
                    "selfplay_games_completed": 0,
                    "positions_generated": 0,
                    "saved_replay_samples": 0,
                    "new_samples_accepted": 0,
                    "optimizer_steps": 0,
                    "optimizer_examples_seen": 0,
                },
                "scientific_budget": scientific_budget,
            }
        self._restore_benchmark_selection()
        self._save_state()

    def _restore_benchmark_selection(self) -> None:
        benchmark = self.state.get("performance_benchmark")
        if not isinstance(benchmark, dict):
            return
        selected = benchmark.get("selected")
        if not isinstance(selected, dict):
            return
        self.selfplay_wait_ms = float(selected["selfplay_wait_ms"])
        self.arena_workers = min(WORKERS, int(selected["arena_workers"]))
        self.arena_wait_ms = float(selected["arena_wait_ms"])

    def _save_state(self) -> None:
        hardware = self.telemetry.summary()
        self.state["hardware"] = hardware
        self.state["bottlenecks"] = _impl.infer_bottlenecks(hardware)
        now = time.time()
        active = float(self.state.get("active_runtime_seconds", 0.0))
        session_start = self.state.get("session_started_at_epoch")
        if isinstance(session_start, (int, float)):
            active += max(0.0, now - float(session_start))
        self.state.setdefault("totals", {})["wall_time_seconds"] = active
        self.state["totals"]["calendar_elapsed_seconds"] = max(
            0.0, now - float(self.state["started_at_epoch"])
        )
        self.state["last_update_epoch"] = now
        _impl._atomic_json(self.state_path, self.state)
        _impl._atomic_json(self.report_path, self.state)
        _impl._atomic_text(self.report_md_path, _impl.render_markdown_report(self.state))

    def _close_session(self) -> None:
        start = self.state.get("session_started_at_epoch")
        if isinstance(start, (int, float)):
            self.state["active_runtime_seconds"] = float(
                self.state.get("active_runtime_seconds", 0.0)
            ) + max(0.0, time.time() - float(start))
        self.state["session_started_at_epoch"] = None

    def _begin_action(self, key: str, kind: str, details: dict[str, object]) -> None:
        begin_action(self.state, key, kind, details)
        self._save_state()

    def _complete_action(
        self,
        *,
        key: str,
        kind: str,
        details: dict[str, object],
        counter: str | None = None,
        amount: int = 0,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        record = complete_action_once(
            self.state,
            key=key,
            kind=kind,
            details=details,
            counter=counter,
            amount=amount,
            metadata=metadata,
        )
        self._save_state()
        return record

    def _completed_action(self, key: str):
        return (self.state.get("completed_actions") or {}).get(key)

    def stream_command(self, command: list[str], log_path: Path, phase: str):
        """Stream a child command and terminate it cleanly if the sweep is interrupted."""

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo)
        env["PYTHONUNBUFFERED"] = "1"
        self.telemetry.set_phase(phase)
        self.telemetry.start()
        sample_time = None
        infer_batch = None
        started = time.perf_counter()
        process = None
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as log:
                log.write("\n=== COMMAND ===\n" + " ".join(command) + "\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=self.repo,
                    env=env,
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
                    self._phase_from_line(line, phase)
                    line_sample_time, line_infer_batch = extract_last_progress_metrics(line)
                    if line_sample_time is not None:
                        sample_time = line_sample_time
                    if line_infer_batch is not None:
                        infer_batch = line_infer_batch
                return_code = process.wait()
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            raise
        wall = time.perf_counter() - started
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
        return _impl.CommandMetrics(wall, sample_time, infer_batch)

    def training_command(self, *args, **kwargs):
        """Ensure benchmark inference-wait overrides survive checkpoint resume."""

        resume = bool(kwargs.get("resume", False))
        inference_wait_ms = kwargs.get("inference_wait_ms")
        sweep_overrides = kwargs.get("sweep_overrides")
        run_name = kwargs.get("run_name")
        if resume and inference_wait_ms is not None and not sweep_overrides and run_name:
            checkpoint_dir = Path("checkpoint") / str(run_name)
            iterations = []
            for path in checkpoint_dir.glob("iteration-*.pkl"):
                try:
                    iterations.append(int(path.stem.rsplit("-", 1)[1]))
                except (IndexError, ValueError):
                    continue
            if not iterations:
                raise RuntimeError(f"Benchmark resume has no checkpoint: {run_name}")
            saved_args = _impl._load_checkpoint_args(str(run_name), max(iterations))
            halflife = saved_args.get("gocube_chosen_move_temperature_halflife")
            if (
                not isinstance(halflife, (int, float))
                or not math.isfinite(float(halflife))
                or float(halflife) <= 0
            ):
                raise RuntimeError("Benchmark resume checkpoint has invalid chosen-move halflife")
            kwargs["sweep_overrides"] = {
                "--chosen-move-temperature-halflife": float(halflife),
            }
        if not hasattr(self, "selfplay_wait_ms"):
            self.selfplay_wait_ms = _impl.SELFPLAY_BATCH_WAIT_MS
        return super().training_command(*args, **kwargs)

    def _checkpoint(self, run_name: str, iteration: int) -> Path:
        return self.repo / "checkpoint" / run_name / f"iteration-{int(iteration):04d}.pkl"

    def _validate_checkpoint(
        self,
        run_name: str,
        iteration: int,
        overrides: dict[str, float | int] | None = None,
    ) -> None:
        _impl.validate_production_checkpoint(run_name, iteration)
        if overrides:
            validate_sweep_overrides(_impl._load_checkpoint_args(run_name, iteration), overrides)

    def _quarantine_namespace(self, run_name: str, reason: str) -> None:
        stamp = f"{int(time.time())}"
        root = self.root / "quarantine" / f"{run_name}-{stamp}"
        suffix = 2
        while root.exists():
            root = self.root / "quarantine" / f"{run_name}-{stamp}-{suffix}"
            suffix += 1
        root.mkdir(parents=True, exist_ok=False)
        moved = []
        for category in ("checkpoint", "data"):
            for name in (run_name, f".{run_name}.partial"):
                source = self.repo / category / name
                if source.exists():
                    destination = root / category / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
                    moved.append(str(destination.relative_to(self.repo)))
        self.state.setdefault("quarantined_namespaces", []).append({
            "run": run_name,
            "reason": reason,
            "paths": moved,
            "at_epoch": time.time(),
        })
        self._save_state()

    def _ensure_clone(
        self,
        *,
        parent_run: str,
        parent_iteration: int,
        target_run: str,
        kind: str,
    ) -> None:
        checkpoint_dest = self.repo / "checkpoint" / target_run
        data_dest = self.repo / "data" / target_run
        checkpoint_partial = self.repo / "checkpoint" / f".{target_run}.partial"
        data_partial = self.repo / "data" / f".{target_run}.partial"
        expected = {
            "schema_version": 1,
            "experiment_id": self.cli.experiment_id,
            "kind": kind,
            "parent_run": parent_run,
            "parent_iteration": int(parent_iteration),
        }
        if checkpoint_partial.exists() or data_partial.exists():
            self._quarantine_namespace(target_run, "stale partial clone from interrupted copy")
        if checkpoint_dest.exists() != data_dest.exists():
            self._quarantine_namespace(target_run, "one-sided candidate namespace after interrupted clone")
        if checkpoint_dest.exists() and data_dest.exists():
            provenance_path = checkpoint_dest / PROVENANCE_FILENAME
            if not provenance_path.is_file():
                self._quarantine_namespace(target_run, "candidate namespace has no sweep provenance")
            else:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance != expected:
                    raise RuntimeError(f"Candidate namespace provenance mismatch: {target_run}")
                return
        checkpoint_source = self.repo / "checkpoint" / parent_run
        data_source = self.repo / "data" / parent_run
        if not checkpoint_source.is_dir() or not data_source.is_dir():
            raise FileNotFoundError(f"Parent run is incomplete: {parent_run}")
        shutil.copytree(checkpoint_source, checkpoint_partial, copy_function=shutil.copy2)
        shutil.copytree(data_source, data_partial, copy_function=shutil.copy2)
        manifest = checkpoint_partial / "gocube-run.json"
        if manifest.exists():
            manifest.unlink()
        atomic_json(checkpoint_partial / PROVENANCE_FILENAME, expected)
        os.replace(checkpoint_partial, checkpoint_dest)
        os.replace(data_partial, data_dest)

    def _arena_expected(
        self,
        *,
        run_a: str,
        iteration_a: int,
        run_b: str,
        iteration_b: int,
        games: int,
        heldout: bool,
        seed: int,
        workers: int,
        wait_ms: float,
    ) -> dict[str, object]:
        if heldout:
            suite = json.loads(self.heldout_path.read_text(encoding="utf-8"))
            number_of_games = 2 * len(suite["positions"])
            evaluation_mode = "heldout-paired"
            heldout_suite = str(self.heldout_path)
        else:
            number_of_games = int(games)
            evaluation_mode = "batched-coalesced"
            heldout_suite = None
        return {
            "run_a": run_a,
            "iteration_a": int(iteration_a),
            "run_b": run_b,
            "iteration_b": int(iteration_b),
            "seed": int(seed),
            "workers": int(workers),
            "evaluation_mode": evaluation_mode,
            "heldout_suite": heldout_suite,
            "arena_inference_batch_wait_ms": float(wait_ms),
            "number_of_games": int(number_of_games),
            "arena_sims": ARENA_SIMS,
            "komi": EXPECTED_KOMI,
        }

    def arena(
        self,
        *,
        run_a: str,
        iteration_a: int,
        run_b: str,
        iteration_b: int,
        games: int,
        name: str,
        heldout: bool = False,
        seed: int,
        workers: int | None = None,
        wait_ms: float | None = None,
        phase: str = "ARENA",
    ) -> dict[str, object]:
        worker_count = self.arena_workers if workers is None else int(workers)
        batch_wait = self.arena_wait_ms if wait_ms is None else float(wait_ms)
        output = self.results / f"{name}.json"
        expected = self._arena_expected(
            run_a=run_a,
            iteration_a=iteration_a,
            run_b=run_b,
            iteration_b=iteration_b,
            games=games,
            heldout=heldout,
            seed=seed,
            workers=worker_count,
            wait_ms=batch_wait,
        )
        key = f"arena:{name}"
        details = dict(expected)
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
            validate_arena_payload(payload, expected)
            self._complete_action(
                key=key,
                kind="arena",
                details=details,
                counter="arena_games",
                amount=int(payload["number_of_games"]),
                metadata={"output": str(output)},
            )
            return payload
        if self._completed_action(key) is not None:
            raise RuntimeError(f"Completed Arena action lost its result artifact: {name}")
        command = [
            str(self.python),
            "tools/gocube_checkpoint_arena.py",
            "--run-a", run_a,
            "--iteration-a", str(iteration_a),
            "--run-b", run_b,
            "--iteration-b", str(iteration_b),
            "--games", str(games),
            "--workers", str(worker_count),
            "--batched",
            "--device", self.cli.device,
            "--arena-inference-batch-wait-ms", str(batch_wait),
            "--seed", str(seed),
            "--output", str(output),
        ]
        if heldout:
            command.extend(["--heldout-suite", str(self.heldout_path)])
        self._begin_action(key, "arena", details)
        self.stream_command(command, self.logs / f"{name}.log", phase)
        payload = json.loads(output.read_text(encoding="utf-8"))
        validate_arena_payload(payload, expected)
        self._complete_action(
            key=key,
            kind="arena",
            details=details,
            counter="arena_games",
            amount=int(payload["number_of_games"]),
            metadata={"output": str(output)},
        )
        return payload

    def bootstrap(self) -> tuple[str, int]:
        if self.cli.bootstrap_run:
            self._validate_checkpoint(self.cli.bootstrap_run, BOOTSTRAP_ITERATION)
            return self.cli.bootstrap_run, BOOTSTRAP_ITERATION
        run_name = f"{self.cli.experiment_id}-bootstrap"
        target = self._checkpoint(run_name, BOOTSTRAP_ITERATION)
        key = "train:bootstrap"
        details = {"run": run_name, "target_iteration": BOOTSTRAP_ITERATION}
        checkpoint_dir = self.repo / "checkpoint" / run_name
        data_dir = self.repo / "data" / run_name
        if checkpoint_dir.exists() != data_dir.exists():
            self._quarantine_namespace(run_name, "one-sided interrupted bootstrap namespace")
        if target.is_file():
            self._validate_checkpoint(run_name, BOOTSTRAP_ITERATION)
            for iteration in range(1, BOOTSTRAP_ITERATION + 1):
                if not self._checkpoint(run_name, iteration).is_file():
                    raise RuntimeError(f"Bootstrap checkpoint missing: iteration {iteration}")
            if self._completed_action(key) is None:
                self._account_training_progress(
                    run_name=run_name,
                    before=None,
                    action_key=key,
                    kind="training",
                    details=details,
                )
            return run_name, BOOTSTRAP_ITERATION
        if self._completed_action(key) is not None:
            raise RuntimeError("Bootstrap action is complete but iteration-0007 is missing")
        resume = checkpoint_dir.exists() and data_dir.exists()
        before_progress = self.training_progress(run_name)
        self._begin_action(key, "training", details)
        self.stream_command(
            self.training_command(
                run_name=run_name,
                target_iteration=BOOTSTRAP_ITERATION,
                resume=resume,
            ),
            self.logs / "bootstrap.log",
            "SELFPLAY",
        )
        self._validate_checkpoint(run_name, BOOTSTRAP_ITERATION)
        for iteration in range(1, BOOTSTRAP_ITERATION + 1):
            if not self._checkpoint(run_name, iteration).is_file():
                raise RuntimeError(f"Bootstrap checkpoint missing: iteration {iteration}")
        self._account_training_progress(
            run_name=run_name,
            before=before_progress,
            action_key=key,
            kind="training",
            details=details,
        )
        return run_name, BOOTSTRAP_ITERATION

    def _metrics_from_log(self, log_path: Path, wall_time: float = 0.0):
        if not log_path.is_file():
            return _impl.CommandMetrics(float(wall_time), None, None)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        sample_time, infer_batch = extract_last_progress_metrics(text)
        return _impl.CommandMetrics(float(wall_time), sample_time, infer_batch)

    @staticmethod
    def _metrics_payload(metrics) -> dict[str, object]:
        return {
            "wall_time_seconds": float(metrics.wall_time_seconds),
            "sample_time_seconds": metrics.sample_time_seconds,
            "inference_batch_rows": metrics.inference_batch_rows,
        }

    def _benchmark_training(self, bootstrap_run: str, wait_ms: float):
        run_name = f"{self.cli.experiment_id}-bench-selfplay-{_impl._safe(wait_ms)}ms"
        parent_progress = self.training_progress(bootstrap_run)
        self._ensure_clone(
            parent_run=bootstrap_run,
            parent_iteration=BOOTSTRAP_ITERATION,
            target_run=run_name,
            kind="selfplay-benchmark",
        )
        target_iteration = BOOTSTRAP_ITERATION + 1
        target = self._checkpoint(run_name, target_iteration)
        key = f"benchmark-training:{_impl._safe(wait_ms)}ms"
        details = {
            "run": run_name,
            "target_iteration": target_iteration,
            "inference_wait_ms": float(wait_ms),
        }
        completed = self._completed_action(key)
        if completed is not None:
            if not target.is_file():
                raise RuntimeError(f"Completed benchmark training lost checkpoint: {run_name}")
            metadata = completed.get("metadata") or {}
            metrics_data = metadata.get("metrics") or {}
            return _impl.CommandMetrics(
                float(metrics_data.get("wall_time_seconds", 0.0)),
                metrics_data.get("sample_time_seconds"),
                metrics_data.get("inference_batch_rows"),
            )
        if target.is_file():
            self._validate_checkpoint(run_name, target_iteration)
            metrics = self._metrics_from_log(
                self.logs / f"benchmark-selfplay-{_impl._safe(wait_ms)}ms.log"
            )
            if metrics.sample_time_seconds is not None and metrics.sample_time_seconds > 0:
                after = self.training_progress(run_name)
                delta = self._training_progress_delta(parent_progress, after)
                self._complete_action(
                    key=key,
                    kind="benchmark-training",
                    details=details,
                    counter="benchmark_selfplay_games",
                    amount=delta["selfplay_games_completed"],
                    metadata={
                        "metrics": self._metrics_payload(metrics),
                        "training_progress": after,
                        "training_counter_delta": delta,
                    },
                )
                return metrics
            self._quarantine_namespace(
                run_name,
                "benchmark checkpoint survived but final performance metrics did not",
            )
            self._ensure_clone(
                parent_run=bootstrap_run,
                parent_iteration=BOOTSTRAP_ITERATION,
                target_run=run_name,
                kind="selfplay-benchmark",
            )
        before_progress = self.training_progress(run_name)
        self._begin_action(key, "benchmark-training", details)
        metrics = self.stream_command(
            self.training_command(
                run_name=run_name,
                target_iteration=target_iteration,
                resume=True,
                inference_wait_ms=wait_ms,
            ),
            self.logs / f"benchmark-selfplay-{_impl._safe(wait_ms)}ms.log",
            "BENCHMARK",
        )
        self._validate_checkpoint(run_name, target_iteration)
        after = self.training_progress(run_name)
        delta = self._training_progress_delta(before_progress, after)
        self._complete_action(
            key=key,
            kind="benchmark-training",
            details=details,
            counter="benchmark_selfplay_games",
            amount=delta["selfplay_games_completed"],
            metadata={
                "metrics": self._metrics_payload(metrics),
                "training_progress": after,
                "training_counter_delta": delta,
            },
        )
        return metrics

    def performance_benchmark(self, bootstrap_run: str) -> dict[str, object]:
        if self.cli.skip_performance_benchmark:
            selected = {
                "selfplay_wait_ms": SELFPLAY_BATCH_WAIT_MS,
                "arena_workers": WORKERS,
                "arena_wait_ms": float(self.cli.arena_batch_wait_ms),
            }
            self.selfplay_wait_ms = float(selected["selfplay_wait_ms"])
            self.arena_workers = int(selected["arena_workers"])
            self.arena_wait_ms = float(selected["arena_wait_ms"])
            return {"skipped": True, "selected": selected, "selfplay": [], "arena": []}

        selfplay_results = []
        for wait_ms in SELFPLAY_BENCHMARK_WAITS_MS:
            metrics = self._benchmark_training(bootstrap_run, float(wait_ms))
            if metrics.sample_time_seconds and metrics.sample_time_seconds > 0:
                games_per_second = 1.0 / metrics.sample_time_seconds
                source = "selfplay_sample_time"
            else:
                games_per_second = (
                    GAMES_PER_ITERATION / metrics.wall_time_seconds
                    if metrics.wall_time_seconds > 0 else 0.0
                )
                source = "command_wall_time_fallback"
            selfplay_results.append({
                "wait_ms": wait_ms,
                "games_per_second": games_per_second,
                "mean_inference_batch_rows": metrics.inference_batch_rows,
                "wall_time_seconds": metrics.wall_time_seconds,
                "metric_source": source,
                "stable": math.isfinite(games_per_second) and games_per_second > 0,
            })
        stable_selfplay = [item for item in selfplay_results if item["stable"]]
        if not stable_selfplay:
            raise RuntimeError("No stable self-play inference batching benchmark result")
        best_selfplay = max(
            stable_selfplay,
            key=lambda item: (item["games_per_second"], -abs(item["wait_ms"] - 1.0)),
        )
        self.selfplay_wait_ms = float(best_selfplay["wait_ms"])

        arena_results = []
        for workers in ARENA_BENCHMARK_WORKERS:
            for wait_ms in ARENA_BENCHMARK_WAITS_MS:
                name = f"benchmark-arena-w{workers}-{_impl._safe(wait_ms)}ms"
                result = self.arena(
                    run_a=bootstrap_run,
                    iteration_a=BOOTSTRAP_ITERATION,
                    run_b=bootstrap_run,
                    iteration_b=HEALTH_REFERENCE_ITERATION,
                    games=self.cli.benchmark_games,
                    name=name,
                    seed=self.cli.seed + workers * 100 + int(wait_ms * 10),
                    workers=workers,
                    wait_ms=wait_ms,
                    phase="BENCHMARK",
                )
                stable = (
                    int(result["number_of_games"]) == int(self.cli.benchmark_games)
                    and math.isfinite(float(result["games_per_second"]))
                    and float(result["games_per_second"]) > 0
                )
                arena_results.append({
                    "workers": workers,
                    "wait_ms": wait_ms,
                    "games_per_second": float(result["games_per_second"]),
                    "mean_inference_batch_rows": result.get("mean_inference_batch_rows"),
                    "cuda_peak_memory_mib": result.get("cuda_peak_memory_mib"),
                    "stable": stable,
                })
        stable_arena = [item for item in arena_results if item["stable"]]
        if not stable_arena:
            raise RuntimeError("No stable Arena batching benchmark result")
        best_arena = max(
            stable_arena,
            key=lambda item: (
                item["games_per_second"],
                item["workers"],
                -abs(item["wait_ms"] - 1.0),
            ),
        )
        self.arena_workers = min(WORKERS, int(best_arena["workers"]))
        self.arena_wait_ms = float(best_arena["wait_ms"])
        selected = {
            "selfplay_wait_ms": self.selfplay_wait_ms,
            "arena_workers": self.arena_workers,
            "arena_wait_ms": self.arena_wait_ms,
        }
        return {
            "skipped": False,
            "selected": selected,
            "selfplay": selfplay_results,
            "arena": arena_results,
        }

    def train_candidate(
        self,
        *,
        spec: dict[str, object],
        label: str,
        value: float | int,
        parent_run: str,
        parent_iteration: int,
        active_overrides: dict[str, float | int],
    ):
        run_name = (
            f"{self.cli.experiment_id}-{str(spec['id']).lower()}-"
            f"{label.lower()}-{_impl._safe(value)}"
        )
        self._ensure_clone(
            parent_run=parent_run,
            parent_iteration=parent_iteration,
            target_run=run_name,
            kind=f"parameter:{spec['id']}",
        )
        requested_target_iteration = int(parent_iteration) + CANDIDATE_ITERATIONS
        # This remains only a safety ceiling for a target-driven run.  The
        # scientific stop is the cumulative sample milestone below; the
        # number of games/iterations is not used as the budget.
        max_iteration = int(parent_iteration) + max(CANDIDATE_ITERATIONS, 1024)
        scientific_target = self._candidate_budget_target(parent_run)
        parent_progress = self.training_progress(parent_run)
        sweep_overrides = dict(active_overrides)
        sweep_overrides[str(spec["flag"])] = value
        key = f"training:{run_name}:{requested_target_iteration}"
        details = {
            "run": run_name,
            "parent_run": parent_run,
            "parent_iteration": int(parent_iteration),
            "target_iteration": requested_target_iteration,
            "max_iteration_safety_ceiling": max_iteration,
            "scientific_target": {
                "kind": scientific_target.kind,
                "target": int(scientific_target.target),
            },
            "sweep_overrides": sweep_overrides,
        }
        target = self._checkpoint(run_name, requested_target_iteration)
        if target.is_file():
            progress = self.training_progress(run_name)
            actual_iteration = int((progress or {}).get("latest_iteration", requested_target_iteration))
            self._validate_checkpoint(run_name, actual_iteration, sweep_overrides)
            if self._completed_action(key) is None:
                self._account_training_progress(
                    run_name=run_name,
                    before=parent_progress,
                    action_key=key,
                    kind="training",
                    details=details,
                )
            progress = self.training_progress(run_name)
        else:
            if self._completed_action(key) is not None:
                raise RuntimeError(f"Completed candidate training lost checkpoint: {run_name}")
            self._begin_action(key, "training", details)
            self.stream_command(
                self.training_command(
                    run_name=run_name,
                    target_iteration=max_iteration,
                    sweep_overrides=sweep_overrides,
                    resume=True,
                    scientific_target=scientific_target,
                ),
                self.logs / f"{spec['id']}-{label}-train.log",
                "SELFPLAY",
            )
            progress = self.training_progress(run_name)
            actual_iteration = int((progress or {}).get("latest_iteration", 0))
            if actual_iteration <= int(parent_iteration):
                raise RuntimeError(
                    f"Target-driven training produced no new checkpoint for {run_name}"
                )
            self._validate_checkpoint(run_name, actual_iteration, sweep_overrides)
            self._account_training_progress(
                run_name=run_name,
                before=parent_progress,
                action_key=key,
                kind="training",
                details=details,
            )
        if progress is None:
            raise RuntimeError(f"Training progress disappeared for {run_name}")
        budget = progress.get("budget") or {}
        if budget and not bool(budget.get("reached", False)):
            raise RuntimeError(
                f"Scientific target was not reached for {run_name}: {budget}"
            )
        actual_iteration = int(progress.get("latest_iteration", actual_iteration))
        target = self._checkpoint(run_name, actual_iteration)
        if not target.is_file():
            raise RuntimeError(f"Training progress points to missing checkpoint: {target}")
        screen = self.arena(
            run_a=run_name,
            iteration_a=actual_iteration,
            run_b=parent_run,
            iteration_b=parent_iteration,
            games=SCREEN_GAMES,
            name=f"{spec['id']}-{label}-screen",
            seed=self.cli.seed + requested_target_iteration * 100 + sum(ord(c) for c in label),
        )
        heldout = self.arena(
            run_a=run_name,
            iteration_a=actual_iteration,
            run_b=parent_run,
            iteration_b=parent_iteration,
            games=SCREEN_GAMES,
            name=f"{spec['id']}-{label}-heldout",
            heldout=True,
            seed=self.cli.seed + requested_target_iteration * 1000 + sum(ord(c) for c in label),
        )
        return _impl.Candidate(
            label,
            value,
            run_name,
            actual_iteration,
            screen,
            heldout,
            training_budget=progress,
        )

    def _restore_completed_stages(self, bootstrap_run: str):
        champion_run = bootstrap_run
        champion_iteration = BOOTSTRAP_ITERATION
        active_overrides: dict[str, float | int] = {}
        pending_status = None
        pending_reason = None
        records = self.state.get("parameters") or []
        if len(records) > len(PARAMETER_SPECS):
            raise RuntimeError("Experiment state contains too many parameter stages")
        for index, record in enumerate(records):
            spec = PARAMETER_SPECS[index]
            if record.get("id") != spec["id"]:
                raise RuntimeError("Experiment parameter-stage order is inconsistent")
            expected_parent = {"run": champion_run, "iteration": champion_iteration}
            if record.get("parent") != expected_parent:
                raise RuntimeError(f"Saved {spec['id']} parent does not match champion ancestry")
            champion_after = record.get("champion_after")
            if not isinstance(champion_after, dict):
                raise RuntimeError(f"Saved {spec['id']} has no champion_after")
            champion_run = str(champion_after["run"])
            champion_iteration = int(champion_after["iteration"])
            active_overrides = dict(champion_after.get("sweep_overrides") or {})
            self._validate_checkpoint(champion_run, champion_iteration, active_overrides)
            decision = str((record.get("winner") or {}).get("decision"))
            if decision != "IMPROVED":
                if index != len(records) - 1:
                    raise RuntimeError("Stages exist after a non-promoted parameter decision")
                pending_status = (
                    "STOPPED_REGRESSION" if decision == "REGRESSED"
                    else "STOPPED_NO_IMPROVEMENT"
                )
                pending_reason = (
                    f"{spec['id']} confirmed regression against its parent."
                    if decision == "REGRESSED"
                    else f"{spec['id']} produced no confidence-supported improvement after the maximum confirmation bracket."
                )
        return champion_run, champion_iteration, active_overrides, pending_status, pending_reason

    def run(self) -> None:
        if self._already_terminal:
            return

        old_handlers = {}

        def _interrupt(signum, _frame):
            raise SweepInterrupted(int(signum))

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _interrupt)

        self.telemetry.start()
        bootstrap_run = None
        try:
            self.state["status"] = "RUNNING"
            self._save_state()

            if isinstance(self.state.get("bootstrap"), dict):
                bootstrap_run = str(self.state["bootstrap"]["run"])
                bootstrap_iteration = int(self.state["bootstrap"]["iteration"])
                if bootstrap_iteration != BOOTSTRAP_ITERATION:
                    raise RuntimeError("Saved bootstrap iteration is incompatible")
                self._validate_checkpoint(bootstrap_run, bootstrap_iteration)
            else:
                bootstrap_run, bootstrap_iteration = self.bootstrap()
                self.state["bootstrap"] = {
                    "run": bootstrap_run,
                    "iteration": bootstrap_iteration,
                    "contract": _impl.validate_production_checkpoint(
                        bootstrap_run, bootstrap_iteration
                    ),
                }
                self._save_state()

            heldout = build_fresh_heldout_suite(
                run_name=bootstrap_run,
                iteration=bootstrap_iteration,
                output_path=self.heldout_path,
                positions=self.cli.heldout_positions,
                seed=self.cli.seed,
                device=self.cli.device,
            )
            self.state["heldout_suite"] = {
                "path": str(self.heldout_path),
                "positions": len(heldout["positions"]),
                "source_run": bootstrap_run,
                "source_iteration": bootstrap_iteration,
            }
            self._save_state()

            if not isinstance(self.state.get("health_gate"), dict):
                self.state["health_gate"] = self.health_gate(bootstrap_run)
                self._save_state()

            if isinstance(self.state.get("performance_benchmark"), dict):
                self._restore_benchmark_selection()
            else:
                self.state["performance_benchmark"] = self.performance_benchmark(bootstrap_run)
                self._restore_benchmark_selection()
                self._save_state()

            (
                champion_run,
                champion_iteration,
                active_overrides,
                pending_status,
                pending_reason,
            ) = self._restore_completed_stages(bootstrap_run)

            if pending_status is None:
                start_index = len(self.state.get("parameters") or [])
                for spec in PARAMETER_SPECS[start_index:]:
                    champion_run, champion_iteration, active_overrides, decision = self.run_parameter(
                        spec,
                        champion_run,
                        champion_iteration,
                        active_overrides,
                    )
                    if decision != "IMPROVED":
                        pending_status = (
                            "STOPPED_REGRESSION" if decision == "REGRESSED"
                            else "STOPPED_NO_IMPROVEMENT"
                        )
                        pending_reason = (
                            f"{spec['id']} confirmed regression against its parent."
                            if decision == "REGRESSED"
                            else f"{spec['id']} produced no confidence-supported improvement after the maximum confirmation bracket."
                        )
                        break
                else:
                    pending_status = "COMPLETE"

            champion = {
                "run": champion_run,
                "iteration": champion_iteration,
                "sweep_overrides": dict(active_overrides),
            }
            self.state["champion"] = champion
            self._save_state()

            if not isinstance(self.state.get("final_confirmation"), dict):
                self.state["final_confirmation"] = self.final_confirmation(
                    champion, bootstrap_run
                )
                self._save_state()

            recommended = champion
            original = self.state["final_confirmation"].get("champion_vs_original_c7")
            if isinstance(original, dict) and _impl.decision_from_arena(original["aggregate"]) == "REGRESSED":
                pending_status = "STOPPED_REGRESSION_FINAL"
                pending_reason = (
                    "Final champion regressed significantly against original C7; "
                    "recommend original C7."
                )
                recommended = {
                    "run": bootstrap_run,
                    "iteration": BOOTSTRAP_ITERATION,
                    "sweep_overrides": {},
                }
            self.state["recommended_champion"] = recommended
            self.state["resume_production_command"] = self._resume_command(recommended)
            self.state["status"] = pending_status or "COMPLETE"
            if pending_reason:
                self.state["stop_reason"] = pending_reason
            else:
                self.state.pop("stop_reason", None)
            self.state["finished_at_epoch"] = time.time()
            self._save_state()
        except SweepInterrupted as exc:
            self.state["status"] = "INTERRUPTED"
            self.state["interrupted_signal"] = exc.signum
            self.state["stop_reason"] = (
                f"Sweep interrupted by signal {exc.signum}; rerun the canonical command to resume."
            )
            self._save_state()
        except Exception as exc:
            self.state["status"] = "FAILED"
            self.state["error"] = f"{type(exc).__name__}: {exc}"
            self._save_state()
            raise
        finally:
            self.telemetry.stop()
            self._close_session()
            self.state["last_exit_at_epoch"] = time.time()
            self._save_state()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def main(argv=None) -> int:
    Experiment(parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
