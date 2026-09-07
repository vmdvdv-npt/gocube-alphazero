#!/usr/bin/env python3
"""Canonical CLI for the current Cube-4 P1..P5 overnight sweep.

This module intentionally contains no historical A-G experiment surface. The
orchestration lives in ``c4_overnight_complete``; this entrypoint adds only the
small runtime-safety adapters that belong at the process boundary.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.c4_overnight_complete import *  # noqa: F401,F403
from tools import c4_overnight_complete as _impl
from tools.c4_preflight import preflight
from tools.gocube_overnight_safety import (
    build_fresh_heldout_suite,
    extract_last_progress_metrics,
)


# Public name retained because the implementation calls this helper. The
# behavior is evaluation-only fresh rollout generation, never training replay.
build_frozen_heldout_suite = build_fresh_heldout_suite


class Experiment(_impl.Experiment):
    """Current P1..P5 runner with process-boundary safety fixes."""

    def training_command(self, *args, **kwargs):
        """Keep an explicit inference-wait benchmark value across resume.

        Hardened checkpoint resume reloads saved args unless at least one sweep
        override is explicit. For a pure performance benchmark, re-state the
        checkpoint's existing chosen-move halflife so the safe explicit-resume
        path is selected without changing search semantics.
        """

        resume = bool(kwargs.get("resume", False))
        inference_wait_ms = kwargs.get("inference_wait_ms")
        sweep_overrides = kwargs.get("sweep_overrides")
        run_name = kwargs.get("run_name")
        if resume and inference_wait_ms is not None and not sweep_overrides and run_name:
            checkpoint_dir = Path("checkpoint") / str(run_name)
            iterations: list[int] = []
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
        """Stream a child process and retain final CR-redrawn perf metrics."""

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
            raise RuntimeError(
                f"Command failed with exit code {return_code}: {' '.join(command)}"
            )
        return _impl.CommandMetrics(wall, sample_time, infer_batch)

    def run(self):
        """Bind the implementation to fresh heldout generation for this device."""

        original_builder = _impl.build_frozen_heldout_suite

        def fresh_builder(**kwargs):
            return build_fresh_heldout_suite(**kwargs, device=self.cli.device)

        _impl.build_frozen_heldout_suite = fresh_builder
        try:
            return super().run()
        finally:
            _impl.build_frozen_heldout_suite = original_builder


def main(argv=None) -> int:
    cli = _impl.parse_args(argv)
    preflight(cli, _impl)
    Experiment(cli).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
