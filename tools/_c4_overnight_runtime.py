#!/usr/bin/env python3
"""Complete autonomous GoCube Cube-4 parameter sweep orchestration.

The runner has no wall-clock deadline. It benchmarks batching first, evaluates
with confidence intervals, stops only on an execution/contract failure or when
a parameter stage is confirmed as regressed/not improving, and continuously
writes resumable JSON plus a human-readable Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from alphazero.envs.gocube.production_contract import CUBE4_PRODUCTION
from alphazero.envs.gocube.production_training import (
    CumulativeTrainingCounters,
    SampleBudgetTarget,
    build_sample_budget_target,
    load_training_progress,
)
from tools.hardware_telemetry import HardwareTelemetry

WORKERS = CUBE4_PRODUCTION.workers
REGULAR_SIMS = CUBE4_PRODUCTION.regular_sims
FAST_SIMS = CUBE4_PRODUCTION.fast_sims
GAMES_PER_ITERATION = CUBE4_PRODUCTION.games_per_iteration
TRAIN_BATCH_SIZE = CUBE4_PRODUCTION.train_batch_size
SELFPLAY_BATCH_WAIT_MS = 1.0
ARENA_SIMS = CUBE4_PRODUCTION.arena_sims
EXPECTED_KOMI = CUBE4_PRODUCTION.komi
BOOTSTRAP_ITERATION = 7
HEALTH_REFERENCE_ITERATION = 4
CANDIDATE_ITERATIONS = 2
SCREEN_GAMES = 128
HEAD_TO_HEAD_GAMES = 256
HELDOUT_POSITIONS = 16
DEFAULT_SEED = 20260907
SELFPLAY_BENCHMARK_WAITS_MS = (0.5, 1.0, 2.0)
ARENA_BENCHMARK_WAITS_MS = (0.5, 1.0, 2.0)
ARENA_BENCHMARK_WORKERS = (4, 8, 16)
DEFAULT_BENCHMARK_GAMES = 64
# A generation chunk remains 256 games, but candidate stages are now defined
# by a sample milestone.  The default retains the old two-chunk scale only as
# a convenient numeric milestone; it is never used to discard games or rows.
DEFAULT_CANDIDATE_NEW_SAMPLE_BUDGET = 512

PARAMETER_SPECS = (
    {
        "id": "P1",
        "name": "temperature_halflife",
        "flag": "--chosen-move-temperature-halflife",
        "values": (10.0, 19.0, 32.0),
        "extensions": {"L": 5.0, "H": 48.0},
    },
    {
        "id": "P2",
        "name": "dirichlet_weight",
        "flag": "--root-dirichlet-noise-weight",
        "values": (0.15, 0.25, 0.35),
        "extensions": {"L": 0.05, "H": 0.45},
    },
    {
        "id": "P3",
        "name": "fast_search_probability",
        "flag": "--fast-game-prob",
        "values": (0.10, 0.25, 0.40),
        "extensions": {"L": 0.0, "H": 0.55},
    },
    {
        "id": "P4",
        "name": "train_samples_per_new_sample",
        "flag": "--train-samples-per-new-sample",
        "values": (0.5, 1.0, 2.0),
        "extensions": {"L": 0.25, "H": 4.0},
    },
    {
        "id": "P5",
        "name": "replay_window_iters",
        "flag": "--replay-window-iters",
        "values": (4, 8, 16),
        "extensions": {"L": 2, "H": 32},
    },
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _safe(value: object) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(char if char.isalnum() or char in "_-" else "-" for char in text)


def _checkpoint_path(run_name: str, iteration: int) -> Path:
    return Path("checkpoint") / run_name / f"iteration-{int(iteration):04d}.pkl"


def _load_checkpoint_args(run_name: str, iteration: int):
    path = _checkpoint_path(run_name, iteration)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload["args"]


def validate_production_checkpoint(run_name: str, iteration: int) -> dict[str, object]:
    args = _load_checkpoint_args(run_name, iteration)
    checks = {
        "komi": float(args.get("gocube_komi", float("nan"))),
        "regular_sims": int(args.get("numMCTSSims", -1)),
        "fast_sims": int(args.get("numFastSims", -1)),
        "fast_probability": float(args.get("probFastSim", float("nan"))),
        "arena_sims": int(args.get("arenaMCTSSims", -1)),
        "train_batch_size": int(args.get("train_batch_size", -1)),
        "train_samples_per_new_sample": float(
            args.get("gocube_train_samples_per_new_sample", float("nan"))
        ),
        "workers": int(args.get("workers", -1)),
        "topology": args.get("gocube_topology"),
        "size": int(args.get("gocube_size", -1)),
        "rules_fingerprint": args.get("gocube_rules_fingerprint"),
    }
    if not math.isclose(checks["komi"], EXPECTED_KOMI, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"GoCube contract requires komi 0.5, got {checks['komi']}")
    if checks["regular_sims"] != REGULAR_SIMS:
        raise RuntimeError(f"GoCube contract requires {REGULAR_SIMS} regular sims")
    if checks["fast_sims"] != FAST_SIMS:
        raise RuntimeError(f"GoCube contract requires {FAST_SIMS} fast sims")
    if not math.isclose(
        checks["fast_probability"], CUBE4_PRODUCTION.fast_probability,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError(
            "GoCube contract requires fast probability "
            f"{CUBE4_PRODUCTION.fast_probability}, got {checks['fast_probability']}"
        )
    if checks["arena_sims"] != ARENA_SIMS:
        raise RuntimeError(f"GoCube contract requires {ARENA_SIMS} Arena sims")
    if checks["train_batch_size"] != TRAIN_BATCH_SIZE:
        raise RuntimeError(
            f"GoCube contract requires training batch size {TRAIN_BATCH_SIZE}, "
            f"got {checks['train_batch_size']}"
        )
    if checks["workers"] != WORKERS:
        raise RuntimeError(f"GoCube contract requires {WORKERS} workers")
    if not math.isclose(
        checks["train_samples_per_new_sample"],
        CUBE4_PRODUCTION.train_samples_per_new_sample,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "GoCube contract requires train_samples_per_new_sample "
            f"{CUBE4_PRODUCTION.train_samples_per_new_sample}, "
            f"got {checks['train_samples_per_new_sample']}"
        )
    if checks["topology"] != "cube" or checks["size"] != 4:
        raise RuntimeError(f"Expected Cube 4x4 checkpoint, got {checks['topology']} {checks['size']}")
    return checks


def clone_run_namespace(parent_run: str, target_run: str) -> None:
    destinations = [Path("checkpoint") / target_run, Path("data") / target_run]
    if any(path.exists() for path in destinations):
        raise FileExistsError(f"Candidate namespace already exists: {target_run}")
    checkpoint_source = Path("checkpoint") / parent_run
    data_source = Path("data") / parent_run
    if not checkpoint_source.is_dir() or not data_source.is_dir():
        raise FileNotFoundError(f"Parent run is incomplete: {parent_run}")
    shutil.copytree(checkpoint_source, destinations[0], copy_function=shutil.copy2)
    shutil.copytree(data_source, destinations[1], copy_function=shutil.copy2)
    cloned_manifest = destinations[0] / "gocube-run.json"
    if cloned_manifest.exists():
        cloned_manifest.unlink()


def build_frozen_heldout_suite(*, run_name: str, iteration: int, output_path: Path,
                               positions: int = HELDOUT_POSITIONS, seed: int = DEFAULT_SEED) -> dict[str, object]:
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("source_run") != run_name or int(payload.get("source_iteration", -1)) != iteration:
            raise RuntimeError("Existing held-out suite was built from a different source")
        return payload
    checkpoint_args = _load_checkpoint_args(run_name, iteration)
    records_dir = Path("data") / run_name / "records" / f"iteration-{iteration:04d}"
    record_paths = sorted(path for path in records_dir.glob("*.json") if path.name != "iteration-manifest.json")
    if not record_paths:
        raise RuntimeError(f"No recorded self-play games available for held-out suite: {records_dir}")
    rng = random.Random(int(seed))
    rng.shuffle(record_paths)
    selected = []
    fractions = (0.25, 0.40, 0.55, 0.70)
    for record_path in record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        moves = record.get("moves")
        if not isinstance(moves, list) or len(moves) < 8:
            continue
        if isinstance(moves[0], dict) and moves[0].get("training_start") is not None:
            continue
        fraction = fractions[len(selected) % len(fractions)]
        prefix_len = min(len(moves) - 2, max(2, int(round(len(moves) * fraction))))
        prefix = moves[:prefix_len]
        actions = [int(move["action"]) for move in prefix if isinstance(move, dict) and "action" in move]
        if len(actions) != prefix_len:
            continue
        selected.append({
            "position_id": f"H{len(selected) + 1:03d}",
            "source_game_id": record.get("game_id"),
            "source_record": str(record_path),
            "prefix_length": prefix_len,
            "prefix_actions": actions,
            "prefix_moves": [move.get("move") for move in prefix],
        })
        if len(selected) >= int(positions):
            break
    if len(selected) < int(positions):
        raise RuntimeError(f"Could build only {len(selected)} held-out positions, requested {positions}")
    payload = {
        "schema_version": 1,
        "seed": int(seed),
        "source_run": run_name,
        "source_iteration": int(iteration),
        "source_checkpoint": str(_checkpoint_path(run_name, iteration)),
        "komi": EXPECTED_KOMI,
        "rules_fingerprint": checkpoint_args["gocube_rules_fingerprint"],
        "positions": selected,
    }
    _atomic_json(output_path, payload)
    return payload


def _wilson_interval(score: float, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    denominator = 1.0 + z * z / n
    center = (score + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * n)) / n) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def combine_arena_results(results: list[dict[str, object]]) -> dict[str, object]:
    wins = sum(int(result["wins"]) for result in results)
    losses = sum(int(result["losses"]) for result in results)
    draws = sum(int(result["draws"]) for result in results)
    no_results = sum(int(result["no_results"]) for result in results)
    scored = wins + losses + draws
    games = wins + losses + draws + no_results
    win_rate = (wins + 0.5 * draws) / scored if scored else 0.0
    wall = sum(float(result.get("wall_time_seconds", 0.0)) for result in results)
    by_color = {color: {key: 0 for key in ("games", "wins", "losses", "draws", "no_results")}
                for color in ("black", "white")}
    for result in results:
        for color in by_color:
            source = result.get("by_color", {}).get(color, {})
            for key in by_color[color]:
                by_color[color][key] += int(source.get(key, 0))
    weighted_keys = (
        "average_game_length", "score_margin_mean", "score_margin_mean_abs", "pass_count_mean",
        "entered_cleanup1_fraction", "entered_cleanup2_fraction", "pass_alive_early_end_fraction",
        "cleanup_moves_mean", "cleanup_captures_mean", "mean_inference_batch_rows",
    )
    aggregate: dict[str, object] = {
        "number_of_games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "no_results": no_results,
        "win_rate": win_rate,
        "win_rate_ci95": _wilson_interval(win_rate, scored),
        "by_color": by_color,
        "wall_time_seconds": wall,
        "games_per_second": games / wall if wall > 0 else 0.0,
    }
    for key in weighted_keys:
        pairs = [(float(result[key]), int(result.get("number_of_games", 0)))
                 for result in results if isinstance(result.get(key), (int, float))]
        weight = sum(count for _, count in pairs)
        aggregate[key] = sum(value * count for value, count in pairs) / weight if weight else None
    terminal_counts: dict[str, int] = {}
    for result in results:
        for kind, count in (result.get("terminal_kind_counts") or {}).items():
            terminal_counts[str(kind)] = terminal_counts.get(str(kind), 0) + int(count)
    aggregate["terminal_kind_counts"] = terminal_counts
    return aggregate


def decision_from_arena(result: dict[str, object]) -> str:
    low, high = [float(value) for value in result["win_rate_ci95"]]
    if low > 0.5:
        return "IMPROVED"
    if high < 0.5:
        return "REGRESSED"
    return "NO_IMPROVEMENT"


def infer_bottlenecks(hardware: dict[str, object]) -> dict[str, str]:
    output: dict[str, str] = {}
    phases = hardware.get("phases", {}) if isinstance(hardware, dict) else {}
    for phase, metrics in phases.items():
        if not isinstance(metrics, dict):
            continue
        cpu = (metrics.get("cpu_util_percent") or {}).get("mean")
        gpu = (metrics.get("gpu_util_percent") or {}).get("mean")
        if isinstance(cpu, (int, float)) and cpu >= 85 and (not isinstance(gpu, (int, float)) or gpu < 70):
            output[str(phase)] = "CPU-bound"
        elif isinstance(gpu, (int, float)) and gpu >= 85:
            output[str(phase)] = "GPU-bound"
        elif isinstance(cpu, (int, float)) or isinstance(gpu, (int, float)):
            output[str(phase)] = "mixed / no dominant saturation"
        else:
            output[str(phase)] = "insufficient telemetry"
    return output


def render_markdown_report(state: dict[str, object]) -> str:
    lines = ["# GoCube overnight parameter sweep", ""]
    lines.append(f"Status: **{state.get('status', 'RUNNING')}**")
    lines.append("")
    fixed = state.get("fixed_contract", {})
    lines.extend([
        "## Fixed contract", "",
        f"- Cube 4×4; komi **{fixed.get('komi', EXPECTED_KOMI)}**",
        f"- workers: {fixed.get('workers', WORKERS)}",
        f"- regular / fast sims: {fixed.get('regular_sims', REGULAR_SIMS)} / {fixed.get('fast_sims', FAST_SIMS)}",
        f"- games / iteration: {fixed.get('games_per_iteration', GAMES_PER_ITERATION)}",
        f"- train batch: {fixed.get('train_batch_size', TRAIN_BATCH_SIZE)}",
        "- Arena: deterministic 50 sims; fast/noise/root temperature OFF",
        "- no wall-clock deadline",
        "",
    ])
    benchmark = state.get("performance_benchmark")
    if isinstance(benchmark, dict):
        lines.extend(["## Performance benchmark", ""])
        selected = benchmark.get("selected", {})
        lines.append(f"- self-play inference wait: **{selected.get('selfplay_wait_ms')} ms**")
        lines.append(f"- Arena workers: **{selected.get('arena_workers')}**")
        lines.append(f"- Arena inference wait: **{selected.get('arena_wait_ms')} ms**")
        lines.append("")
    health = state.get("health_gate")
    if isinstance(health, dict):
        agg = health.get("aggregate", health)
        lines.extend(["## Bootstrap health gate", ""])
        lines.append(
            f"C7 vs C4: {agg.get('wins')}W/{agg.get('losses')}L/{agg.get('draws')}D, "
            f"win rate {float(agg.get('win_rate', 0.0)):.3f}, CI95 {agg.get('win_rate_ci95')}"
        )
        lines.append("")
    lines.extend(["## Parameter stages", ""])
    for stage in state.get("parameters", []):
        winner = stage.get("winner", {})
        lines.append(f"### {stage.get('id')} — {stage.get('name')}")
        lines.append("")
        lines.append(f"Chosen: **{winner.get('value')}**; decision: **{winner.get('decision')}**")
        lines.append("")
        for candidate in stage.get("candidates", []):
            screen = candidate.get("screen", {})
            lines.append(
                f"- {candidate.get('label')} = {candidate.get('value')}: "
                f"screen {float(screen.get('win_rate', 0.0)):.3f} "
                f"CI95 {screen.get('win_rate_ci95')}"
            )
            budget = candidate.get("training_budget") or {}
            counters = budget.get("counters") or budget.get("cumulative_counters") or {}
            if counters:
                lines.append(
                    "  training budget: "
                    f"games={counters.get('selfplay_games_completed', 0)}, "
                    f"positions={counters.get('positions_generated', 0)}, "
                    f"samples={counters.get('new_samples_accepted', 0)}, "
                    f"optimizer_steps={counters.get('optimizer_steps', 0)}, "
                    f"examples_seen={counters.get('optimizer_examples_seen', 0)}"
                )
            scientific = budget.get("budget")
            if isinstance(scientific, dict):
                lines.append(
                    "  scientific stop: "
                    f"{scientific.get('kind')} target={scientific.get('target')} "
                    f"after={scientific.get('after')} "
                    f"overshoot={scientific.get('overshoot', 0)} "
                    f"overshot={scientific.get('overshot', False)}"
                )
            metrics = budget.get("latest_iteration_metrics")
            if isinstance(metrics, dict):
                lines.append(
                    "  episode metrics: "
                    f"average_length={metrics.get('average_game_length', 0.0)}, "
                    f"no_results={metrics.get('no_result_games', 0)}, "
                    f"move_limit={metrics.get('episode_move_limit_games', 0)}"
                )
        if stage.get("justification"):
            lines.append(f"- Rationale: {stage['justification']}")
        lines.append("")
    champion = state.get("recommended_champion") or state.get("champion")
    if isinstance(champion, dict):
        lines.extend(["## Recommended champion", ""])
        lines.append(f"- run: `{champion.get('run')}`")
        lines.append(f"- iteration: `{champion.get('iteration')}`")
        lines.append(f"- sweep overrides: `{champion.get('sweep_overrides', {})}`")
        lines.append("")
    confirmation = state.get("final_confirmation")
    if isinstance(confirmation, dict):
        lines.extend(["## Final confirmation", ""])
        for key, value in confirmation.items():
            if isinstance(value, dict):
                agg = value.get("aggregate", value)
                lines.append(f"- {key}: win rate {agg.get('win_rate')}, CI95 {agg.get('win_rate_ci95')}")
        lines.append("")
    totals = state.get("totals", {})
    lines.extend(["## Totals", ""])
    for key in ("training_games", "benchmark_selfplay_games", "arena_games", "wall_time_seconds"):
        if key in totals:
            lines.append(f"- {key}: {totals[key]}")
    counters = state.get("cumulative_counters") or state.get("training_counters")
    if isinstance(counters, dict):
        lines.extend(["", "## Scientific training accounting", ""])
        for key in (
            "selfplay_games_completed",
            "positions_generated",
            "saved_replay_samples",
            "new_samples_accepted",
            "optimizer_steps",
            "optimizer_examples_seen",
        ):
            value = counters.get(key, counters.get(f"cumulative_{key}", 0))
            lines.append(f"- cumulative {key}: {value}")
    budget = state.get("scientific_budget")
    if isinstance(budget, dict):
        lines.extend(["", "Scientific stopping target:", ""])
        lines.append(
            f"- {budget.get('kind')}: target {budget.get('target')} "
            f"(per candidate increment {budget.get('increment')})"
        )
    lines.append("")
    if state.get("bottlenecks"):
        lines.extend(["## Resource bottlenecks", ""])
        for phase, text in state["bottlenecks"].items():
            lines.append(f"- {phase}: {text}")
        lines.append("")
    if state.get("resume_production_command"):
        lines.extend(["## Resume production", "", "```bash", str(state["resume_production_command"]), "```", ""])
    if state.get("stop_reason"):
        lines.extend(["## Stop reason", "", str(state["stop_reason"]), ""])
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class Candidate:
    label: str
    value: float | int
    run_name: str
    iteration: int
    screen: dict[str, object]
    heldout: dict[str, object]
    training_budget: dict[str, object] = field(default_factory=dict)


@dataclass
class CommandMetrics:
    wall_time_seconds: float
    sample_time_seconds: float | None
    inference_batch_rows: float | None


class Experiment:
    def __init__(self, cli: argparse.Namespace):
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
        self.telemetry = HardwareTelemetry(self.root / "hardware-telemetry.jsonl", interval_s=cli.telemetry_interval)
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.selfplay_wait_ms = SELFPLAY_BATCH_WAIT_MS
        self.arena_wait_ms = float(cli.arena_batch_wait_ms)
        self.arena_workers = WORKERS
        self.state: dict[str, Any] = {
            "schema_version": 2,
            "experiment_id": cli.experiment_id,
            "status": "RUNNING",
            "started_at_epoch": time.time(),
            "fixed_contract": {
                "workers": WORKERS,
                "regular_sims": REGULAR_SIMS,
                "fast_sims": FAST_SIMS,
                "games_per_iteration": GAMES_PER_ITERATION,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "initial_selfplay_inference_batch_wait_ms": SELFPLAY_BATCH_WAIT_MS,
                "arena_sims": ARENA_SIMS,
                "komi": EXPECTED_KOMI,
            },
            "parameters": [],
            "totals": {"training_games": 0, "benchmark_selfplay_games": 0, "arena_games": 0},
        }
        self._save_state()

    def _save_state(self) -> None:
        hardware = self.telemetry.summary()
        self.state["hardware"] = hardware
        self.state["bottlenecks"] = infer_bottlenecks(hardware)
        self.state["totals"]["wall_time_seconds"] = max(0.0, time.time() - float(self.state["started_at_epoch"]))
        _atomic_json(self.state_path, self.state)
        _atomic_json(self.report_path, self.state)
        _atomic_text(self.report_md_path, render_markdown_report(self.state))

    def _phase_from_line(self, line: str, default_phase: str) -> None:
        if "Generating Samples" in line:
            self.telemetry.set_phase("SELFPLAY" if default_phase != "BENCHMARK" else "BENCHMARK")
        elif "Training Net" in line:
            self.telemetry.set_phase("TRAIN" if default_phase != "BENCHMARK" else "BENCHMARK")
        elif "Arena" in line:
            self.telemetry.set_phase("ARENA" if default_phase != "BENCHMARK" else "BENCHMARK")
        elif self.telemetry.phase_name == "IDLE":
            self.telemetry.set_phase(default_phase)

    def training_progress(self, run_name: str) -> dict[str, object] | None:
        """Read the trainer-published cumulative sample clock for a run."""

        return load_training_progress("data", run_name)

    @staticmethod
    def _progress_counters(progress: dict[str, object] | None) -> CumulativeTrainingCounters:
        return CumulativeTrainingCounters.from_mapping(progress or {})

    def _training_progress_delta(
        self,
        before: dict[str, object] | None,
        after: dict[str, object] | None,
    ) -> dict[str, int]:
        if after is None:
            raise RuntimeError("Training command completed without training-progress.json")
        old = self._progress_counters(before)
        new = self._progress_counters(after)
        delta = {
            key: int(getattr(new, key)) - int(getattr(old, key))
            for key in (
                "selfplay_games_completed",
                "positions_generated",
                "saved_replay_samples",
                "new_samples_accepted",
                "optimizer_steps",
                "optimizer_examples_seen",
            )
        }
        if any(value < 0 for value in delta.values()):
            raise RuntimeError(f"Training progress moved backwards on resume: {delta}")
        return delta

    def _account_training_progress(
        self,
        *,
        run_name: str,
        before: dict[str, object] | None,
        action_key: str,
        kind: str,
        details: dict[str, object],
    ) -> dict[str, object]:
        after = self.training_progress(run_name)
        delta = self._training_progress_delta(before, after)
        metadata = {
            "training_progress": after,
            "training_counter_delta": delta,
        }
        self.state.setdefault("training_runs", {})[run_name] = after
        self._complete_action(
            key=action_key,
            kind=kind,
            details=details,
            counter="training_games",
            amount=delta["selfplay_games_completed"],
            metadata=metadata,
        )
        self.state["cumulative_counters"] = self._progress_counters(after).as_dict()
        return after

    def _candidate_budget_target(self, parent_run: str) -> SampleBudgetTarget:
        parent = self.training_progress(parent_run)
        counters = self._progress_counters(parent)
        optimizer_increment = getattr(self.cli, "candidate_optimizer_examples_budget", None)
        new_increment = getattr(
            self.cli,
            "candidate_new_samples_budget",
            DEFAULT_CANDIDATE_NEW_SAMPLE_BUDGET,
        )
        if optimizer_increment is not None:
            return SampleBudgetTarget(
                SampleBudgetTarget.OPTIMIZER_EXAMPLES,
                counters.optimizer_examples_seen + int(optimizer_increment),
            )
        return SampleBudgetTarget(
            SampleBudgetTarget.NEW_SAMPLES,
            counters.new_samples_accepted + int(new_increment),
        )

    def stream_command(self, command: list[str], log_path: Path, phase: str) -> CommandMetrics:
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
                command, cwd=self.repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace"
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                self._phase_from_line(line, phase)
                sample_match = re.search(r"Sample Time:\s*([0-9.]+)s", line)
                if sample_match:
                    sample_time = float(sample_match.group(1))
                batch_match = re.search(r"Infer Batch:\s*([0-9.]+)", line)
                if batch_match:
                    infer_batch = float(batch_match.group(1))
            return_code = process.wait()
        wall = time.perf_counter() - started
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
        return CommandMetrics(wall, sample_time, infer_batch)

    def training_command(self, *, run_name: str, target_iteration: int,
                         sweep_overrides: dict[str, float | int] | None = None,
                         resume: bool, inference_wait_ms: float | None = None,
                         scientific_target: SampleBudgetTarget | None = None) -> list[str]:
        wait_ms = self.selfplay_wait_ms if inference_wait_ms is None else float(inference_wait_ms)
        command = [
            str(self.python), "-m", "alphazero.envs.gocube.hardened_train",
            "--topology", "cube", "--size", "4", "--workers", str(WORKERS),
            "--sims", str(REGULAR_SIMS), "--arena-sims", str(ARENA_SIMS),
            "--games-per-iteration", str(GAMES_PER_ITERATION), "--iterations", str(int(target_iteration)),
            "--train-batch-size", str(TRAIN_BATCH_SIZE), "--inference-batch-wait-ms", str(wait_ms),
            "--endgame-sample-weight", "1", "--no-arena", "--run-name", run_name,
        ]
        if scientific_target is not None:
            if scientific_target.kind == SampleBudgetTarget.NEW_SAMPLES:
                command.extend(["--cumulative-new-samples-target", str(scientific_target.target)])
            else:
                command.extend([
                    "--cumulative-optimizer-examples-target",
                    str(scientific_target.target),
                ])
        for flag, value in (sweep_overrides or {}).items():
            command.extend([str(flag), str(value)])
        if resume:
            command.append("--allow-existing-run")
        return command

    def arena(self, *, run_a: str, iteration_a: int, run_b: str, iteration_b: int,
              games: int, name: str, heldout: bool = False, seed: int,
              workers: int | None = None, wait_ms: float | None = None,
              phase: str = "ARENA") -> dict[str, object]:
        worker_count = self.arena_workers if workers is None else int(workers)
        batch_wait = self.arena_wait_ms if wait_ms is None else float(wait_ms)
        output = self.results / f"{name}.json"
        command = [
            str(self.python), "tools/gocube_checkpoint_arena.py",
            "--run-a", run_a, "--iteration-a", str(iteration_a),
            "--run-b", run_b, "--iteration-b", str(iteration_b),
            "--games", str(games), "--workers", str(worker_count), "--batched",
            "--device", self.cli.device,
            "--arena-inference-batch-wait-ms", str(batch_wait),
            "--seed", str(seed), "--output", str(output),
        ]
        if heldout:
            command.extend(["--heldout-suite", str(self.heldout_path)])
        self.stream_command(command, self.logs / f"{name}.log", phase)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.state["totals"]["arena_games"] += int(result["number_of_games"])
        return result

    def bootstrap(self) -> tuple[str, int]:
        if self.cli.bootstrap_run:
            validate_production_checkpoint(self.cli.bootstrap_run, BOOTSTRAP_ITERATION)
            return self.cli.bootstrap_run, BOOTSTRAP_ITERATION
        run_name = f"{self.cli.experiment_id}-bootstrap"
        self.stream_command(
            self.training_command(run_name=run_name, target_iteration=BOOTSTRAP_ITERATION, resume=False),
            self.logs / "bootstrap.log", "SELFPLAY"
        )
        validate_production_checkpoint(run_name, BOOTSTRAP_ITERATION)
        for iteration in range(1, BOOTSTRAP_ITERATION + 1):
            if not _checkpoint_path(run_name, iteration).is_file():
                raise RuntimeError(f"Bootstrap checkpoint missing: iteration {iteration}")
        self.state["totals"]["training_games"] += BOOTSTRAP_ITERATION * GAMES_PER_ITERATION
        return run_name, BOOTSTRAP_ITERATION

    def health_gate(self, run_name: str) -> dict[str, object]:
        runs = [self.arena(
            run_a=run_name, iteration_a=BOOTSTRAP_ITERATION,
            run_b=run_name, iteration_b=HEALTH_REFERENCE_ITERATION,
            games=128, name="health-c7-vs-c4-128", seed=self.cli.seed
        )]
        aggregate = combine_arena_results(runs)
        low, high = aggregate["win_rate_ci95"]
        if low <= 0.5 <= high:
            runs.append(self.arena(
                run_a=run_name, iteration_a=BOOTSTRAP_ITERATION,
                run_b=run_name, iteration_b=HEALTH_REFERENCE_ITERATION,
                games=128, name="health-c7-vs-c4-plus128", seed=self.cli.seed + 1
            ))
            aggregate = combine_arena_results(runs)
        if float(aggregate["win_rate"]) < float(self.cli.health_gate_min_win_rate) or float(aggregate["win_rate_ci95"][1]) < 0.5:
            raise RuntimeError(
                f"Bootstrap health gate failed: C7 vs C4 win rate {aggregate['win_rate']:.3f}, "
                f"CI95={aggregate['win_rate_ci95']}"
            )
        return {"runs": runs, "aggregate": aggregate}

    def performance_benchmark(self, bootstrap_run: str) -> dict[str, object]:
        if self.cli.skip_performance_benchmark:
            selected = {"selfplay_wait_ms": SELFPLAY_BATCH_WAIT_MS, "arena_workers": WORKERS,
                        "arena_wait_ms": float(self.cli.arena_batch_wait_ms)}
            return {"skipped": True, "selected": selected, "selfplay": [], "arena": []}
        selfplay_results = []
        for index, wait_ms in enumerate(SELFPLAY_BENCHMARK_WAITS_MS):
            run_name = f"{self.cli.experiment_id}-bench-selfplay-{_safe(wait_ms)}ms"
            clone_run_namespace(bootstrap_run, run_name)
            metrics = self.stream_command(
                self.training_command(
                    run_name=run_name, target_iteration=BOOTSTRAP_ITERATION + 1,
                    resume=True, inference_wait_ms=wait_ms
                ),
                self.logs / f"benchmark-selfplay-{_safe(wait_ms)}ms.log", "BENCHMARK"
            )
            validate_production_checkpoint(run_name, BOOTSTRAP_ITERATION + 1)
            if metrics.sample_time_seconds and metrics.sample_time_seconds > 0:
                games_per_second = 1.0 / metrics.sample_time_seconds
                source = "selfplay_sample_time"
            else:
                games_per_second = GAMES_PER_ITERATION / metrics.wall_time_seconds if metrics.wall_time_seconds > 0 else 0.0
                source = "command_wall_time_fallback"
            selfplay_results.append({
                "wait_ms": wait_ms, "games_per_second": games_per_second,
                "mean_inference_batch_rows": metrics.inference_batch_rows,
                "wall_time_seconds": metrics.wall_time_seconds, "metric_source": source,
                "stable": math.isfinite(games_per_second) and games_per_second > 0,
            })
            self.state["totals"]["benchmark_selfplay_games"] += GAMES_PER_ITERATION
        stable_selfplay = [item for item in selfplay_results if item["stable"]]
        if not stable_selfplay:
            raise RuntimeError("No stable self-play inference batching benchmark result")
        best_selfplay = max(stable_selfplay, key=lambda item: (item["games_per_second"], -abs(item["wait_ms"] - 1.0)))
        self.selfplay_wait_ms = float(best_selfplay["wait_ms"])

        arena_results = []
        for workers in ARENA_BENCHMARK_WORKERS:
            for wait_ms in ARENA_BENCHMARK_WAITS_MS:
                name = f"benchmark-arena-w{workers}-{_safe(wait_ms)}ms"
                result = self.arena(
                    run_a=bootstrap_run, iteration_a=BOOTSTRAP_ITERATION,
                    run_b=bootstrap_run, iteration_b=HEALTH_REFERENCE_ITERATION,
                    games=self.cli.benchmark_games, name=name,
                    seed=self.cli.seed + workers * 100 + int(wait_ms * 10),
                    workers=workers, wait_ms=wait_ms, phase="BENCHMARK"
                )
                stable = (
                    int(result["number_of_games"]) == int(self.cli.benchmark_games)
                    and math.isfinite(float(result["games_per_second"]))
                    and float(result["games_per_second"]) > 0
                )
                arena_results.append({
                    "workers": workers, "wait_ms": wait_ms,
                    "games_per_second": float(result["games_per_second"]),
                    "mean_inference_batch_rows": result.get("mean_inference_batch_rows"),
                    "cuda_peak_memory_mib": result.get("cuda_peak_memory_mib"),
                    "stable": stable,
                })
        stable_arena = [item for item in arena_results if item["stable"]]
        if not stable_arena:
            raise RuntimeError("No stable Arena batching benchmark result")
        best_arena = max(stable_arena, key=lambda item: (item["games_per_second"], item["workers"], -abs(item["wait_ms"] - 1.0)))
        self.arena_workers = min(WORKERS, int(best_arena["workers"]))
        self.arena_wait_ms = float(best_arena["wait_ms"])
        selected = {
            "selfplay_wait_ms": self.selfplay_wait_ms,
            "arena_workers": self.arena_workers,
            "arena_wait_ms": self.arena_wait_ms,
        }
        return {"skipped": False, "selected": selected, "selfplay": selfplay_results, "arena": arena_results}

    def train_candidate(self, *, spec: dict[str, object], label: str, value: float | int,
                        parent_run: str, parent_iteration: int,
                        active_overrides: dict[str, float | int]) -> Candidate:
        run_name = f"{self.cli.experiment_id}-{str(spec['id']).lower()}-{label.lower()}-{_safe(value)}"
        clone_run_namespace(parent_run, run_name)
        target_iteration = int(parent_iteration) + CANDIDATE_ITERATIONS
        sweep_overrides = dict(active_overrides)
        sweep_overrides[str(spec["flag"])] = value
        self.stream_command(
            self.training_command(
                run_name=run_name, target_iteration=target_iteration,
                sweep_overrides=sweep_overrides, resume=True
            ),
            self.logs / f"{spec['id']}-{label}-train.log", "SELFPLAY"
        )
        self.state["totals"]["training_games"] += CANDIDATE_ITERATIONS * GAMES_PER_ITERATION
        validate_production_checkpoint(run_name, target_iteration)
        screen = self.arena(
            run_a=run_name, iteration_a=target_iteration,
            run_b=parent_run, iteration_b=parent_iteration,
            games=SCREEN_GAMES, name=f"{spec['id']}-{label}-screen",
            seed=self.cli.seed + target_iteration * 100 + sum(ord(c) for c in label)
        )
        heldout = self.arena(
            run_a=run_name, iteration_a=target_iteration,
            run_b=parent_run, iteration_b=parent_iteration,
            games=SCREEN_GAMES, name=f"{spec['id']}-{label}-heldout", heldout=True,
            seed=self.cli.seed + target_iteration * 1000 + sum(ord(c) for c in label)
        )
        return Candidate(label, value, run_name, target_iteration, screen, heldout)

    def arena_series(self, *, run_a: str, iteration_a: int, run_b: str, iteration_b: int,
                     name: str, seed: int, increments: tuple[int, ...],
                     repeat_if_uncertain: int | None = None) -> dict[str, object]:
        runs = []
        for index, games in enumerate(increments):
            runs.append(self.arena(
                run_a=run_a, iteration_a=iteration_a, run_b=run_b, iteration_b=iteration_b,
                games=games, name=f"{name}-{sum(increments[:index + 1])}", seed=seed + index
            ))
            aggregate = combine_arena_results(runs)
            low, high = aggregate["win_rate_ci95"]
            if low > 0.5 or high < 0.5:
                return {"runs": runs, "aggregate": aggregate}
        aggregate = combine_arena_results(runs)
        low, high = aggregate["win_rate_ci95"]
        if repeat_if_uncertain and low <= 0.5 <= high:
            runs.append(self.arena(
                run_a=run_a, iteration_a=iteration_a, run_b=run_b, iteration_b=iteration_b,
                games=repeat_if_uncertain, name=f"{name}-repeat-seed", seed=seed + len(runs) + 1000
            ))
            aggregate = combine_arena_results(runs)
        return {"runs": runs, "aggregate": aggregate}

    def head_to_head(self, spec_id: str, left: Candidate, right: Candidate) -> tuple[Candidate, dict[str, object], bool]:
        series = self.arena_series(
            run_a=left.run_name, iteration_a=left.iteration,
            run_b=right.run_name, iteration_b=right.iteration,
            name=f"{spec_id}-h2h-{left.label}-vs-{right.label}",
            seed=self.cli.seed + left.iteration * 10,
            increments=(HEAD_TO_HEAD_GAMES, 128, 128), repeat_if_uncertain=512
        )
        aggregate = series["aggregate"]
        winner = left if float(aggregate["win_rate"]) >= 0.5 else right
        low, high = aggregate["win_rate_ci95"]
        confident = low > 0.5 or high < 0.5
        return winner, series, confident

    def confirm_against_parent(self, candidate: Candidate, parent_run: str, parent_iteration: int,
                               spec_id: str) -> tuple[str, dict[str, object]]:
        series = self.arena_series(
            run_a=candidate.run_name, iteration_a=candidate.iteration,
            run_b=parent_run, iteration_b=parent_iteration,
            name=f"{spec_id}-winner-vs-parent", seed=self.cli.seed + candidate.iteration * 31,
            increments=(256, 128, 128)
        )
        decision = decision_from_arena(series["aggregate"])
        return decision, series

    def run_parameter(self, spec: dict[str, object], parent_run: str, parent_iteration: int,
                      active_overrides: dict[str, float | int]) -> tuple[str, int, dict[str, float | int], str]:
        candidates = [
            self.train_candidate(
                spec=spec, label=label, value=value, parent_run=parent_run,
                parent_iteration=parent_iteration, active_overrides=active_overrides
            )
            for label, value in zip(("L", "M", "H"), spec["values"])
        ]
        ranked = sorted(candidates, key=lambda c: (float(c.screen["win_rate"]), float(c.heldout["win_rate"])), reverse=True)
        winner, h2h, confident = self.head_to_head(str(spec["id"]), ranked[0], ranked[1])
        extension_record = None
        extensions = spec.get("extensions", {})
        if confident and winner.label in ("L", "H") and winner.label in extensions:
            extension = self.train_candidate(
                spec=spec, label="X", value=extensions[winner.label], parent_run=parent_run,
                parent_iteration=parent_iteration, active_overrides=active_overrides
            )
            candidates.append(extension)
            edge_winner, extension_h2h, _ = self.head_to_head(str(spec["id"]), winner, extension)
            extension_record = {
                "triggered_by": winner.label,
                "value": extension.value,
                "head_to_head": extension_h2h,
            }
            winner = edge_winner
        decision, confirmation = self.confirm_against_parent(winner, parent_run, parent_iteration, str(spec["id"]))
        promoted = decision == "IMPROVED"
        next_run = winner.run_name if promoted else parent_run
        next_iteration = winner.iteration if promoted else parent_iteration
        next_overrides = dict(active_overrides)
        if promoted:
            next_overrides[str(spec["flag"])] = winner.value
        justification = (
            "Promoted because the winner's confirmation Arena has CI95 entirely above 0.5."
            if decision == "IMPROVED" else
            "Stopped because the winner's confirmation Arena is significantly below 0.5."
            if decision == "REGRESSED" else
            "Stopped because no candidate demonstrated a confidence-supported improvement after extension to 512 games."
        )
        record = {
            "id": spec["id"], "name": spec["name"], "flag": spec["flag"],
            "grid": list(spec["values"]),
            "parent": {"run": parent_run, "iteration": parent_iteration},
            "candidates": [
                {"label": c.label, "value": c.value, "run": c.run_name, "iteration": c.iteration,
                 "screen": c.screen, "heldout": c.heldout,
                 "training_budget": c.training_budget} for c in candidates
            ],
            "head_to_head": h2h,
            "edge_extension": extension_record,
            "confirmation_vs_parent": confirmation,
            "winner": {"label": winner.label, "value": winner.value, "run": winner.run_name,
                       "iteration": winner.iteration, "decision": decision, "promoted": promoted},
            "justification": justification,
            "champion_after": {"run": next_run, "iteration": next_iteration,
                               "sweep_overrides": dict(next_overrides)},
        }
        self.state["parameters"].append(record)
        self.state["champion"] = record["champion_after"]
        self._save_state()
        return next_run, next_iteration, next_overrides, decision

    def _resume_command(self, champion: dict[str, object]) -> str:
        command = [
            str(self.python), "-m", "alphazero.envs.gocube.hardened_train",
            "--topology", "cube", "--size", "4", "--workers", str(WORKERS),
            "--sims", str(REGULAR_SIMS), "--arena-sims", str(ARENA_SIMS),
            "--games-per-iteration", str(GAMES_PER_ITERATION),
            "--iterations", str(int(champion["iteration"]) + 1),
            "--train-batch-size", str(TRAIN_BATCH_SIZE),
            "--inference-batch-wait-ms", str(self.selfplay_wait_ms),
            "--endgame-sample-weight", "1", "--no-arena",
            "--run-name", str(champion["run"]), "--allow-existing-run",
        ]
        for flag, value in champion.get("sweep_overrides", {}).items():
            command.extend([str(flag), str(value)])
        return " ".join(shlex.quote(token) for token in command)

    def final_confirmation(self, champion: dict[str, object], bootstrap_run: str) -> dict[str, object]:
        output: dict[str, object] = {}
        if champion["run"] != bootstrap_run or int(champion["iteration"]) != BOOTSTRAP_ITERATION:
            output["champion_vs_original_c7"] = self.arena_series(
                run_a=str(champion["run"]), iteration_a=int(champion["iteration"]),
                run_b=bootstrap_run, iteration_b=BOOTSTRAP_ITERATION,
                name="final-champion-vs-original-c7", seed=self.cli.seed + 90000,
                increments=(512,), repeat_if_uncertain=512
            )
            promoted_records = [stage for stage in self.state["parameters"] if stage["winner"]["promoted"]]
            if promoted_records:
                parent = promoted_records[-1]["parent"]
                output["champion_vs_last_parent"] = self.arena_series(
                    run_a=str(champion["run"]), iteration_a=int(champion["iteration"]),
                    run_b=str(parent["run"]), iteration_b=int(parent["iteration"]),
                    name="final-champion-vs-last-parent", seed=self.cli.seed + 91000,
                    increments=(512,), repeat_if_uncertain=512
                )
        return output

    def run(self) -> None:
        self.telemetry.start()
        bootstrap_run = None
        try:
            bootstrap_run, bootstrap_iteration = self.bootstrap()
            self.state["bootstrap"] = {
                "run": bootstrap_run, "iteration": bootstrap_iteration,
                "contract": validate_production_checkpoint(bootstrap_run, bootstrap_iteration),
            }
            heldout = build_frozen_heldout_suite(
                run_name=bootstrap_run, iteration=bootstrap_iteration,
                output_path=self.heldout_path, positions=self.cli.heldout_positions, seed=self.cli.seed
            )
            self.state["heldout_suite"] = {
                "path": str(self.heldout_path), "positions": len(heldout["positions"]),
                "source_run": bootstrap_run, "source_iteration": bootstrap_iteration,
            }
            self.state["health_gate"] = self.health_gate(bootstrap_run)
            self.state["performance_benchmark"] = self.performance_benchmark(bootstrap_run)
            self._save_state()

            champion_run, champion_iteration = bootstrap_run, bootstrap_iteration
            active_overrides: dict[str, float | int] = {}
            for spec in PARAMETER_SPECS:
                champion_run, champion_iteration, active_overrides, decision = self.run_parameter(
                    spec, champion_run, champion_iteration, active_overrides
                )
                if decision != "IMPROVED":
                    self.state["status"] = "STOPPED_REGRESSION" if decision == "REGRESSED" else "STOPPED_NO_IMPROVEMENT"
                    self.state["stop_reason"] = (
                        f"{spec['id']} confirmed regression against its parent."
                        if decision == "REGRESSED" else
                        f"{spec['id']} produced no confidence-supported improvement after the maximum confirmation bracket."
                    )
                    break
            else:
                self.state["status"] = "COMPLETE"

            champion = {"run": champion_run, "iteration": champion_iteration, "sweep_overrides": dict(active_overrides)}
            self.state["champion"] = champion
            self.state["final_confirmation"] = self.final_confirmation(champion, bootstrap_run)
            recommended = champion
            original = self.state["final_confirmation"].get("champion_vs_original_c7")
            if isinstance(original, dict) and decision_from_arena(original["aggregate"]) == "REGRESSED":
                self.state["status"] = "STOPPED_REGRESSION_FINAL"
                self.state["stop_reason"] = "Final champion regressed significantly against original C7; recommend original C7."
                recommended = {"run": bootstrap_run, "iteration": BOOTSTRAP_ITERATION, "sweep_overrides": {}}
            self.state["recommended_champion"] = recommended
            self.state["resume_production_command"] = self._resume_command(recommended)
        except Exception as exc:
            self.state["status"] = "FAILED"
            self.state["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.telemetry.stop()
            self.state["finished_at_epoch"] = time.time()
            self._save_state()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete adaptive P1..P5 GoCube overnight experiment")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--bootstrap-run", default=None,
                        help="Use an existing compatible run with iteration-0007 instead of bootstrapping from scratch.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--arena-batch-wait-ms", type=float, default=1.0,
                        help="Initial/fallback Arena wait; normal runs benchmark 0.5/1/2 ms before the sweep.")
    parser.add_argument("--telemetry-interval", type=float, default=2.0)
    parser.add_argument("--heldout-positions", type=int, default=HELDOUT_POSITIONS)
    parser.add_argument("--health-gate-min-win-rate", type=float, default=0.45)
    parser.add_argument("--benchmark-games", type=int, default=DEFAULT_BENCHMARK_GAMES)
    parser.add_argument("--skip-performance-benchmark", action="store_true")
    budget_targets = parser.add_mutually_exclusive_group()
    budget_targets.add_argument(
        "--candidate-new-samples-budget",
        "--candidate-sample-budget",
        dest="candidate_new_samples_budget",
        type=int,
        default=None,
        help="New accepted samples added per candidate stage.",
    )
    budget_targets.add_argument(
        "--candidate-optimizer-examples-budget",
        "--candidate-examples-budget",
        dest="candidate_optimizer_examples_budget",
        type=int,
        default=None,
        help="Optimizer examples added per candidate stage (alternative target clock).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cli = parser.parse_args(argv)
    if cli.candidate_optimizer_examples_budget is not None:
        # argparse defaults are not considered a seen member of a mutually
        # exclusive group. Clear the new-sample default when the alternate
        # optimizer-example clock was explicitly selected.
        cli.candidate_new_samples_budget = None
    elif cli.candidate_new_samples_budget is None:
        cli.candidate_new_samples_budget = DEFAULT_CANDIDATE_NEW_SAMPLE_BUDGET
    if cli.arena_batch_wait_ms < 0:
        parser.error("--arena-batch-wait-ms must be non-negative")
    if cli.telemetry_interval <= 0:
        parser.error("--telemetry-interval must be positive")
    if cli.heldout_positions < 1:
        parser.error("--heldout-positions must be positive")
    if cli.benchmark_games < 1:
        parser.error("--benchmark-games must be positive")
    if cli.candidate_new_samples_budget is not None and cli.candidate_new_samples_budget < 1:
        parser.error("--candidate-new-samples-budget must be positive")
    if cli.candidate_optimizer_examples_budget is not None and cli.candidate_optimizer_examples_budget < 1:
        parser.error("--candidate-optimizer-examples-budget must be positive")
    if not 0.0 <= cli.health_gate_min_win_rate <= 1.0:
        parser.error("--health-gate-min-win-rate must be within [0,1]")
    return cli


def main(argv=None) -> int:
    Experiment(parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
