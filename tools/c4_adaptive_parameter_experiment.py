#!/usr/bin/env python3
"""Adaptive five-parameter GoCube experiment runner.

This is intentionally an orchestration layer, not a new experiment framework.
It keeps production search/training semantics fixed, clones every candidate
from the same parent checkpoint, trains two 256-game iterations per candidate,
and uses fixed 50-sim observational Arena evaluation.

No wall-clock deadline is enforced. The runner stops only on an execution or
contract error, or after P1..P5 complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tools.hardware_telemetry import HardwareTelemetry


WORKERS = 16
REGULAR_SIMS = 50
FAST_SIMS = 20
GAMES_PER_ITERATION = 256
TRAIN_BATCH_SIZE = 1024
SELFPLAY_BATCH_WAIT_MS = 1.0
ARENA_SIMS = 50
EXPECTED_KOMI = 0.5
BOOTSTRAP_ITERATION = 7
HEALTH_REFERENCE_ITERATION = 4
CANDIDATE_ITERATIONS = 2
SCREEN_GAMES = 128
HEAD_TO_HEAD_GAMES = 256
HELDOUT_POSITIONS = 16
DEFAULT_SEED = 20260907

PARAMETER_SPECS = (
    {
        "id": "P1",
        "name": "temperature_halflife",
        "flag": "--chosen-move-temperature-halflife",
        "values": (9.5, 19.0, 38.0),
    },
    {
        "id": "P2",
        "name": "dirichlet_weight",
        "flag": "--root-dirichlet-noise-weight",
        "values": (0.15, 0.25, 0.35),
    },
    {
        "id": "P3",
        "name": "fast_search_probability",
        "flag": "--fast-game-prob",
        "values": (0.10, 0.25, 0.40),
    },
    {
        "id": "P4",
        "name": "train_samples_per_new_sample",
        "flag": "--train-samples-per-new-sample",
        "values": (0.75, 1.0, 1.5),
    },
    {
        "id": "P5",
        "name": "replay_window_iters",
        "flag": "--replay-window-iters",
        "values": (4, 8, 16),
    },
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    if checks["topology"] != "cube" or checks["size"] != 4:
        raise RuntimeError(f"Expected Cube 4x4 checkpoint, got {checks['topology']} {checks['size']}")
    return checks


def clone_run_namespace(parent_run: str, target_run: str) -> None:
    """Clone checkpoint/replay history without mutating the parent namespace."""
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


def build_frozen_heldout_suite(
    *,
    run_name: str,
    iteration: int,
    output_path: Path,
    positions: int = HELDOUT_POSITIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("source_run") != run_name or int(payload.get("source_iteration", -1)) != iteration:
            raise RuntimeError("Existing held-out suite was built from a different source")
        return payload

    checkpoint_args = _load_checkpoint_args(run_name, iteration)
    records_dir = Path("data") / run_name / "records" / f"iteration-{iteration:04d}"
    record_paths = sorted(
        path for path in records_dir.glob("*.json")
        if path.name != "iteration-manifest.json"
    )
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
        selected.append(
            {
                "position_id": f"H{len(selected) + 1:03d}",
                "source_game_id": record.get("game_id"),
                "source_record": str(record_path),
                "prefix_length": prefix_len,
                "prefix_actions": actions,
                "prefix_moves": [move.get("move") for move in prefix],
            }
        )
        if len(selected) >= int(positions):
            break

    if len(selected) < int(positions):
        raise RuntimeError(
            f"Could build only {len(selected)} held-out positions, requested {positions}"
        )

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


@dataclass
class Candidate:
    label: str
    value: float | int
    run_name: str
    iteration: int
    screen: dict[str, object]
    heldout: dict[str, object]


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
        self.heldout_path = self.root / "heldout-suite.json"
        self.telemetry = HardwareTelemetry(
            self.root / "hardware-telemetry.jsonl",
            interval_s=cli.telemetry_interval,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": cli.experiment_id,
            "started_at_epoch": time.time(),
            "fixed_contract": {
                "workers": WORKERS,
                "regular_sims": REGULAR_SIMS,
                "fast_sims": FAST_SIMS,
                "games_per_iteration": GAMES_PER_ITERATION,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "selfplay_inference_batch_wait_ms": SELFPLAY_BATCH_WAIT_MS,
                "arena_sims": ARENA_SIMS,
                "komi": EXPECTED_KOMI,
            },
            "parameters": [],
        }
        self._save_state()

    def _save_state(self) -> None:
        self.state["hardware"] = self.telemetry.summary()
        _atomic_json(self.state_path, self.state)
        _atomic_json(self.report_path, self.state)

    def _phase_from_line(self, line: str, default_phase: str) -> None:
        if "Generating Samples" in line:
            self.telemetry.set_phase("SELFPLAY")
        elif "Training Net" in line:
            self.telemetry.set_phase("TRAIN")
        elif "Arena" in line:
            self.telemetry.set_phase("ARENA")
        elif self.telemetry.phase_name == "IDLE":
            self.telemetry.set_phase(default_phase)

    def stream_command(self, command: list[str], log_path: Path, phase: str) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo)
        env["PYTHONUNBUFFERED"] = "1"
        self.telemetry.set_phase(phase)
        self.telemetry.start()
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write("\n=== COMMAND ===\n")
            log.write(" ".join(command) + "\n")
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
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")

    def training_command(
        self,
        *,
        run_name: str,
        target_iteration: int,
        sweep_overrides: dict[str, float | int] | None = None,
        resume: bool,
    ) -> list[str]:
        command = [
            str(self.python),
            "-m",
            "alphazero.envs.gocube.hardened_train",
            "--topology",
            "cube",
            "--size",
            "4",
            "--workers",
            str(WORKERS),
            "--sims",
            str(REGULAR_SIMS),
            "--arena-sims",
            str(ARENA_SIMS),
            "--games-per-iteration",
            str(GAMES_PER_ITERATION),
            "--iterations",
            str(int(target_iteration)),
            "--train-batch-size",
            str(TRAIN_BATCH_SIZE),
            "--inference-batch-wait-ms",
            str(SELFPLAY_BATCH_WAIT_MS),
            "--endgame-sample-weight",
            "1",
            "--no-arena",
            "--run-name",
            run_name,
        ]
        for flag, value in (sweep_overrides or {}).items():
            command.extend([str(flag), str(value)])
        if resume:
            command.append("--allow-existing-run")
        return command

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
    ) -> dict[str, object]:
        output = self.results / f"{name}.json"
        command = [
            str(self.python),
            "tools/gocube_checkpoint_arena.py",
            "--run-a",
            run_a,
            "--iteration-a",
            str(iteration_a),
            "--run-b",
            run_b,
            "--iteration-b",
            str(iteration_b),
            "--games",
            str(games),
            "--workers",
            str(WORKERS),
            "--batched",
            "--device",
            self.cli.device,
            "--arena-inference-batch-wait-ms",
            str(self.cli.arena_batch_wait_ms),
            "--seed",
            str(seed),
            "--output",
            str(output),
        ]
        if heldout:
            command.extend(["--heldout-suite", str(self.heldout_path)])
        self.stream_command(command, self.logs / f"{name}.log", "ARENA")
        return json.loads(output.read_text(encoding="utf-8"))

    def bootstrap(self) -> tuple[str, int]:
        if self.cli.bootstrap_run:
            run_name = self.cli.bootstrap_run
            validate_production_checkpoint(run_name, BOOTSTRAP_ITERATION)
            return run_name, BOOTSTRAP_ITERATION

        run_name = f"{self.cli.experiment_id}-bootstrap"
        command = self.training_command(
            run_name=run_name,
            target_iteration=BOOTSTRAP_ITERATION,
            sweep_overrides=None,
            resume=False,
        )
        self.stream_command(command, self.logs / "bootstrap.log", "SELFPLAY")
        validate_production_checkpoint(run_name, BOOTSTRAP_ITERATION)
        for iteration in range(1, BOOTSTRAP_ITERATION + 1):
            if not _checkpoint_path(run_name, iteration).is_file():
                raise RuntimeError(f"Bootstrap checkpoint missing: iteration {iteration}")
        return run_name, BOOTSTRAP_ITERATION

    def health_gate(self, run_name: str) -> dict[str, object]:
        result = self.arena(
            run_a=run_name,
            iteration_a=BOOTSTRAP_ITERATION,
            run_b=run_name,
            iteration_b=HEALTH_REFERENCE_ITERATION,
            games=SCREEN_GAMES,
            name="health-c7-vs-c4",
            heldout=False,
            seed=self.cli.seed,
        )
        if float(result["win_rate"]) < float(self.cli.health_gate_min_win_rate):
            raise RuntimeError(
                f"Bootstrap health gate failed: C7 vs C4 win rate {result['win_rate']:.3f} "
                f"< {self.cli.health_gate_min_win_rate:.3f}"
            )
        return result

    def train_candidate(
        self,
        *,
        spec: dict[str, object],
        label: str,
        value: float | int,
        parent_run: str,
        parent_iteration: int,
        active_overrides: dict[str, float | int],
    ) -> Candidate:
        run_name = (
            f"{self.cli.experiment_id}-{spec['id'].lower()}-"
            f"{label.lower()}-{_safe(value)}"
        )
        clone_run_namespace(parent_run, run_name)
        target_iteration = int(parent_iteration) + CANDIDATE_ITERATIONS
        sweep_overrides = dict(active_overrides)
        sweep_overrides[str(spec["flag"])] = value
        command = self.training_command(
            run_name=run_name,
            target_iteration=target_iteration,
            sweep_overrides=sweep_overrides,
            resume=True,
        )
        self.stream_command(
            command,
            self.logs / f"{spec['id']}-{label}-train.log",
            "SELFPLAY",
        )
        validate_production_checkpoint(run_name, target_iteration)
        screen = self.arena(
            run_a=run_name,
            iteration_a=target_iteration,
            run_b=parent_run,
            iteration_b=parent_iteration,
            games=SCREEN_GAMES,
            name=f"{spec['id']}-{label}-screen",
            heldout=False,
            seed=self.cli.seed + target_iteration * 100 + ord(label),
        )
        heldout = self.arena(
            run_a=run_name,
            iteration_a=target_iteration,
            run_b=parent_run,
            iteration_b=parent_iteration,
            games=SCREEN_GAMES,
            name=f"{spec['id']}-{label}-heldout",
            heldout=True,
            seed=self.cli.seed + target_iteration * 1000 + ord(label),
        )
        return Candidate(label, value, run_name, target_iteration, screen, heldout)

    def head_to_head(
        self,
        spec_id: str,
        left: Candidate,
        right: Candidate,
    ) -> tuple[Candidate, dict[str, object]]:
        games = HEAD_TO_HEAD_GAMES
        result = self.arena(
            run_a=left.run_name,
            iteration_a=left.iteration,
            run_b=right.run_name,
            iteration_b=right.iteration,
            games=games,
            name=f"{spec_id}-h2h-{left.label}-vs-{right.label}-{games}",
            heldout=False,
            seed=self.cli.seed + left.iteration * 10,
        )
        distance = abs(float(result["win_rate"]) - 0.5)
        if distance < 0.05:
            games = 384
            result = self.arena(
                run_a=left.run_name,
                iteration_a=left.iteration,
                run_b=right.run_name,
                iteration_b=right.iteration,
                games=games,
                name=f"{spec_id}-h2h-{left.label}-vs-{right.label}-{games}",
                heldout=False,
                seed=self.cli.seed + left.iteration * 10 + 1,
            )
            distance = abs(float(result["win_rate"]) - 0.5)
        if distance < 0.03:
            games = 512
            result = self.arena(
                run_a=left.run_name,
                iteration_a=left.iteration,
                run_b=right.run_name,
                iteration_b=right.iteration,
                games=games,
                name=f"{spec_id}-h2h-{left.label}-vs-{right.label}-{games}",
                heldout=False,
                seed=self.cli.seed + left.iteration * 10 + 2,
            )
        winner = left if float(result["win_rate"]) >= 0.5 else right
        return winner, result

    def run_parameter(
        self,
        spec: dict[str, object],
        parent_run: str,
        parent_iteration: int,
        active_overrides: dict[str, float | int],
    ) -> tuple[str, int, dict[str, float | int]]:
        candidates = []
        for label, value in zip(("L", "M", "H"), spec["values"]):
            candidates.append(
                self.train_candidate(
                    spec=spec,
                    label=label,
                    value=value,
                    parent_run=parent_run,
                    parent_iteration=parent_iteration,
                    active_overrides=active_overrides,
                )
            )

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                float(candidate.screen["win_rate"]),
                float(candidate.heldout["win_rate"]),
            ),
            reverse=True,
        )
        winner, h2h = self.head_to_head(str(spec["id"]), ranked[0], ranked[1])

        promotion_score = 0.75 * float(winner.screen["win_rate"]) + 0.25 * float(
            winner.heldout["win_rate"]
        )
        promoted = promotion_score >= 0.5
        next_run = winner.run_name if promoted else parent_run
        next_iteration = winner.iteration if promoted else parent_iteration
        next_overrides = dict(active_overrides)
        if promoted:
            next_overrides[str(spec["flag"])] = winner.value

        record = {
            "id": spec["id"],
            "name": spec["name"],
            "flag": spec["flag"],
            "grid": list(spec["values"]),
            "parent": {"run": parent_run, "iteration": parent_iteration},
            "candidates": [
                {
                    "label": candidate.label,
                    "value": candidate.value,
                    "run": candidate.run_name,
                    "iteration": candidate.iteration,
                    "screen": candidate.screen,
                    "heldout": candidate.heldout,
                }
                for candidate in candidates
            ],
            "head_to_head": h2h,
            "winner": {
                "label": winner.label,
                "value": winner.value,
                "run": winner.run_name,
                "iteration": winner.iteration,
                "promotion_score": promotion_score,
                "promoted": promoted,
            },
            "champion_after": {
                "run": next_run,
                "iteration": next_iteration,
                "sweep_overrides": dict(next_overrides),
            },
        }
        self.state["parameters"].append(record)
        self.state["champion"] = {
            "run": next_run,
            "iteration": next_iteration,
            "sweep_overrides": dict(next_overrides),
        }
        self._save_state()
        return next_run, next_iteration, next_overrides

    def run(self) -> None:
        self.telemetry.start()
        try:
            bootstrap_run, bootstrap_iteration = self.bootstrap()
            self.state["bootstrap"] = {
                "run": bootstrap_run,
                "iteration": bootstrap_iteration,
                "contract": validate_production_checkpoint(bootstrap_run, bootstrap_iteration),
            }
            heldout = build_frozen_heldout_suite(
                run_name=bootstrap_run,
                iteration=bootstrap_iteration,
                output_path=self.heldout_path,
                positions=self.cli.heldout_positions,
                seed=self.cli.seed,
            )
            self.state["heldout_suite"] = {
                "path": str(self.heldout_path),
                "positions": len(heldout["positions"]),
                "source_run": bootstrap_run,
                "source_iteration": bootstrap_iteration,
            }
            self.state["health_gate"] = self.health_gate(bootstrap_run)
            self._save_state()

            champion_run = bootstrap_run
            champion_iteration = bootstrap_iteration
            active_overrides: dict[str, float | int] = {}
            for spec in PARAMETER_SPECS:
                champion_run, champion_iteration, active_overrides = self.run_parameter(
                    spec,
                    champion_run,
                    champion_iteration,
                    active_overrides,
                )
            self.state["status"] = "COMPLETE"
            self.state["champion"] = {
                "run": champion_run,
                "iteration": champion_iteration,
                "sweep_overrides": dict(active_overrides),
            }
        except Exception as exc:
            self.state["status"] = "FAILED"
            self.state["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.telemetry.stop()
            self.state["finished_at_epoch"] = time.time()
            self._save_state()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive P1..P5 GoCube parameter experiment")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--bootstrap-run",
        default=None,
        help="Use an existing compatible run with iteration-0007 instead of bootstrapping from scratch.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--arena-batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--telemetry-interval", type=float, default=1.0)
    parser.add_argument("--heldout-positions", type=int, default=HELDOUT_POSITIONS)
    parser.add_argument("--health-gate-min-win-rate", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cli = parser.parse_args(argv)
    if cli.arena_batch_wait_ms < 0:
        parser.error("--arena-batch-wait-ms must be non-negative")
    if cli.telemetry_interval <= 0:
        parser.error("--telemetry-interval must be positive")
    if cli.heldout_positions < 1:
        parser.error("--heldout-positions must be positive")
    if not 0.0 <= cli.health_gate_min_win_rate <= 1.0:
        parser.error("--health-gate-min-win-rate must be within [0,1]")
    return cli


def main(argv=None) -> int:
    cli = parse_args(argv)
    Experiment(cli).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
