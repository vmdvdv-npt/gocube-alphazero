#!/usr/bin/env python3
"""Canonical entrypoint for the complete autonomous GoCube overnight sweep.

The constants and summary helpers marked LEGACY below are a read-only
compatibility surface for the pre-PR39 recovery/adaptive tools and their tests.
They are never consumed by the new runtime and therefore cannot restore the
obsolete deadline/sims sweep. Runtime execution always uses the no-deadline
P1..P5 implementation from ``tools.c4_overnight_complete``.

Legacy report vocabulary retained for readers of historical report bundles:
checkpoint_sha256 wall_seconds regular_decisions fast_decisions
realized_fast_fraction base_positions latest_iteration_samples window_samples
optimizer_steps_actual examples_seen effective_passes learning_rate
average_game_length no_result_games training_valid_fraction process_batch_size
candidate_score_rate candidate_score_ci95_approx metrics.csv evaluations.csv
experiment.json state.json artifacts.json events.jsonl summary.md
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.c4_overnight_complete import *  # noqa: F401,F403
from tools import c4_overnight_complete as _impl


# ---------------------------------------------------------------------------
# LEGACY READ-ONLY COMPATIBILITY SURFACE
# ---------------------------------------------------------------------------
# These values describe the historical A-G experiment only. New experiments
# MUST use PARAMETER_SPECS from c4_overnight_complete (P1..P5, fixed 50 sims).
FROZEN_TRAINING_COMMIT = "85c87a7cfd467a4d3f4b2844253fb63d746d672a"
SOURCE_RUN_DEFAULT = "c4-t001-c4-c001"
PARENT_ITERATION = 5
TRAIN_BATCH = 256
ENDGAME_WEIGHT = 1
LEGACY_ARENA_SIMS = 100
BRANCHES = {
    "A": {"sims": 100, "pfast": 0.25, "games": 256, "axis": "control", "label": "baseline"},
    "B": {"sims": 100, "pfast": 0.00, "games": 256, "axis": "pfast", "label": "no fast search"},
    "C": {"sims": 100, "pfast": 0.50, "games": 256, "axis": "pfast", "label": "more fast search"},
    "D": {"sims": 50, "pfast": 0.25, "games": 256, "axis": "sims", "label": "shallower regular search"},
    "E": {"sims": 200, "pfast": 0.25, "games": 256, "axis": "sims", "label": "deeper regular search"},
    "F": {"sims": 100, "pfast": 0.25, "games": 128, "axis": "games", "label": "faster feedback loop"},
    "G": {"sims": 100, "pfast": 0.25, "games": 512, "axis": "games", "label": "more data per update"},
}
AXIS_PAIRS = {"pfast": ("B", "C"), "sims": ("D", "E"), "games": ("F", "G")}


def now_iso() -> str:
    """Legacy helper used by recovery tooling; contains no deadline semantics."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def wilson(score: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    d = 1.0 + z * z / n
    c = (score + z * z / (2.0 * n)) / d
    r = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * n)) / n) / d
    return max(0.0, c - r), min(1.0, c + r)


class Experiment(_impl.Experiment):
    """Complete runner plus non-executing historical report compatibility."""

    def training_command(self, *args, **kwargs):
        """Ensure benchmark wait overrides survive checkpoint resume.

        Hardened resume intentionally reloads saved args unless at least one
        explicit sweep flag is present. Self-play batching is a runtime
        performance parameter, not a search/training semantic. For a benchmark
        resume with no real sweep override, explicitly re-state the checkpoint's
        existing chosen-move halflife. That selects the safe explicit-resume
        path without changing model semantics, so 0.5/1/2 ms are actually
        benchmarked instead of silently reverting to the saved 1 ms value.
        """
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
            if not isinstance(halflife, (int, float)) or not math.isfinite(float(halflife)) or float(halflife) <= 0:
                raise RuntimeError("Benchmark resume checkpoint has invalid chosen-move halflife")
            kwargs["sweep_overrides"] = {
                "--chosen-move-temperature-halflife": float(halflife),
            }
        if not hasattr(self, "selfplay_wait_ms"):
            self.selfplay_wait_ms = _impl.SELFPLAY_BATCH_WAIT_MS
        return super().training_command(*args, **kwargs)

    def _legacy_eval_score(self, branch: str, iteration: int, reference_id: str) -> float | None:
        matches = [
            row for row in self.state.get("evaluations", [])
            if row.get("candidate_branch") == branch
            and int(row.get("candidate_iteration", -1)) == int(iteration)
            and row.get("reference_id") == reference_id
        ]
        return float(matches[-1]["candidate_score_rate"]) if matches else None

    def eval_score(self, branch: str, iteration: int, reference_id: str) -> float | None:
        """Legacy name retained for adaptive recovery code."""
        return self._legacy_eval_score(branch, iteration, reference_id)

    def branch_training_seconds(self, branch: str) -> float:
        return sum(
            float(row.get("wall_seconds", 0.0))
            for row in self.state.get("metrics", [])
            if row.get("branch") == branch
        )

    def latest_common_eval(self, branch: str) -> dict[str, Any] | None:
        rows = [
            row for row in self.state.get("evaluations", [])
            if row.get("candidate_branch") == branch
            and str(row.get("reference_id", "")).startswith("parent@")
        ]
        return max(rows, key=lambda row: int(row.get("candidate_iteration", 0))) if rows else None

    def recommendation_text(self) -> str:
        """Render historical A-G state if a recovery tool supplied that schema."""
        direct_rows = [
            row for row in self.state.get("evaluations", [])
            if row.get("reference_id") in ("A@7", "A@8")
            and row.get("candidate_branch") != "A"
        ]
        if direct_rows:
            best = max(direct_rows, key=lambda row: float(row.get("candidate_score_rate", 0.0)))
            score = float(best.get("candidate_score_rate", 0.0))
            ci = best.get("candidate_score_ci95_approx") or [0.0, 1.0]
            branch = best.get("candidate_branch")
            if float(ci[0]) > 0.5:
                return (
                    f"Strong evidence: branch {branch} beats same-depth baseline A "
                    f"(score {score:.3f}, approximate 95% CI {float(ci[0]):.3f}-{float(ci[1]):.3f})."
                )
            if score > 0.5:
                return (
                    f"Provisional lead: branch {branch} is ahead of same-depth baseline A "
                    "but the interval still overlaps 0.5."
                )
        return "No strength evaluation has completed yet; loss curves alone are insufficient to choose a winner."

    def summary_markdown(self) -> str:
        """Historical report view used only by old recovery/report consumers."""
        state = getattr(self, "state", {})
        frozen_commit = getattr(getattr(self, "args", None), "frozen_commit", FROZEN_TRAINING_COMMIT)
        source_run = getattr(self, "source_run", SOURCE_RUN_DEFAULT)
        lines = [
            "# Cube 4 historical A-G compatibility summary",
            "",
            f"- Experiment: `{state.get('experiment_id', 'unknown')}`",
            f"- Status: `{state.get('status', 'unknown')}`",
            f"- Frozen training commit: `{frozen_commit}`",
            f"- Parent: `{source_run}@{PARENT_ITERATION}`",
            "",
            "## Decision status",
            "",
            self.recommendation_text(),
            "",
            "Training loss curves alone are diagnostics, not a strength decision.",
            "",
            "> **Causal caveat for F/G:** in the historical implementation "
            "`process_batch_size = games_per_iteration / workers`. "
            "F/G measure the practical outer-loop package, not a perfectly isolated game-count variable.",
            "",
            "Historical artifacts: experiment.json, state.json, metrics.csv, evaluations.csv, "
            "artifacts.json, events.jsonl, summary.md.",
        ]
        return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """Always execute the new no-deadline P1..P5 implementation."""
    Experiment(_impl.parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
