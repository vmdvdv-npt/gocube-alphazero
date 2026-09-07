#!/usr/bin/env python3
"""Single public entrypoint for the autonomous Cube-4 P1..P5 sweep.

Normal production use is intentionally one command with no required arguments:

    .venv/bin/python tools/c4_overnight_experiment.py

The runner generates its experiment id automatically, bootstraps when no
compatible C7 run is supplied, performs the health gate and performance
benchmark, runs P1..P5, confirms the champion, and writes final reports.
Historical aliases and A-G/deadline orchestration are intentionally unsupported.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

from alphazero.envs.gocube.production_contract import CUBE4_PRODUCTION
from tools._c4_overnight_runtime import *  # noqa: F401,F403
from tools import _c4_overnight_runtime as _impl
from tools.gocube_overnight_safety import (
    build_fresh_heldout_suite,
    extract_last_progress_metrics,
)


CANONICAL_COMMAND = ".venv/bin/python tools/c4_overnight_experiment.py"

# Bind every fixed production value to the single authoritative contract before
# the private orchestration runtime is used.
WORKERS = _impl.WORKERS = CUBE4_PRODUCTION.workers
REGULAR_SIMS = _impl.REGULAR_SIMS = CUBE4_PRODUCTION.regular_sims
FAST_SIMS = _impl.FAST_SIMS = CUBE4_PRODUCTION.fast_sims
GAMES_PER_ITERATION = _impl.GAMES_PER_ITERATION = CUBE4_PRODUCTION.games_per_iteration
TRAIN_BATCH_SIZE = _impl.TRAIN_BATCH_SIZE = CUBE4_PRODUCTION.train_batch_size
ARENA_SIMS = _impl.ARENA_SIMS = CUBE4_PRODUCTION.arena_sims
EXPECTED_KOMI = _impl.EXPECTED_KOMI = CUBE4_PRODUCTION.komi

# Historical function name retained only as an internal API expected by the
# runtime. The implementation creates fresh evaluation-only rollouts and never
# samples training replay records.
build_frozen_heldout_suite = build_fresh_heldout_suite


def _default_experiment_id() -> str:
    return time.strftime("c4-sweep-%Y%m%d-%H%M%S")


def _has_option(argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in argv)


def parse_args(argv=None):
    """Parse the runtime CLI while making the normal zero-argument path valid."""

    raw = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(raw, "--experiment-id"):
        raw = ["--experiment-id", _default_experiment_id(), *raw]
    return _impl.parse_args(raw)


class Experiment(_impl.Experiment):
    """Current sweep plus benchmark-resume and fresh-heldout safeguards."""

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

    def stream_command(self, command: list[str], log_path: Path, phase: str):
        """Stream child output while preserving final CR-redrawn perf metrics."""

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo)
        env["PYTHONUNBUFFERED"] = "1"
        self.telemetry.set_phase(phase)
        self.telemetry.start()
        sample_time = None
        infer_batch = None
        started = time.perf_counter()
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
        wall = time.perf_counter() - started
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
        return _impl.CommandMetrics(wall, sample_time, infer_batch)

    def run(self):
        """Bind fresh heldout generation to the device selected by the CLI."""

        original_builder = _impl.build_frozen_heldout_suite

        def fresh_builder(**kwargs):
            return build_fresh_heldout_suite(**kwargs, device=self.cli.device)

        _impl.build_frozen_heldout_suite = fresh_builder
        try:
            return super().run()
        finally:
            _impl.build_frozen_heldout_suite = original_builder


def main(argv=None) -> int:
    Experiment(parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
