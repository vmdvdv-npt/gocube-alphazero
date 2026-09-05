#!/usr/bin/env python3
"""Resumable, deadline-aware Cube 4 hyperparameter experiment.

The training checkout is intentionally kept at the frozen source commit. This
orchestrator is expected to be copied from a newer tooling commit into /tmp and
executed with the frozen repository as its working directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


FROZEN_TRAINING_COMMIT = "85c87a7cfd467a4d3f4b2844253fb63d746d672a"
SOURCE_RUN_DEFAULT = "c4-t001-c4-c001"
WORKERS = 16
PARENT_ITERATION = 5
FAST_SIMS = 20
TRAIN_BATCH = 256
ENDGAME_WEIGHT = 1
ARENA_SIMS = 100
BASELINE = {"sims": 100, "pfast": 0.25, "games": 256}
BRANCHES = {
    "A": {"sims": 100, "pfast": 0.25, "games": 256, "axis": "control", "label": "baseline"},
    "B": {"sims": 100, "pfast": 0.00, "games": 256, "axis": "pfast", "label": "no fast search"},
    "C": {"sims": 100, "pfast": 0.50, "games": 256, "axis": "pfast", "label": "more fast search"},
    "D": {"sims": 50,  "pfast": 0.25, "games": 256, "axis": "sims", "label": "shallower regular search"},
    "E": {"sims": 200, "pfast": 0.25, "games": 256, "axis": "sims", "label": "deeper regular search"},
    "F": {"sims": 100, "pfast": 0.25, "games": 128, "axis": "games", "label": "faster feedback loop"},
    "G": {"sims": 100, "pfast": 0.25, "games": 512, "axis": "games", "label": "more data per update"},
}
AXIS_PAIRS = {"pfast": ("B", "C"), "sims": ("D", "E"), "games": ("F", "G")}
TENSOR_SUFFIXES = ("data", "policy", "value", "score", "ownership", "ownership-mask")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def wilson(score: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    d = 1.0 + z * z / n
    c = (score + z * z / (2.0 * n)) / d
    r = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * n)) / n) / d
    return max(0.0, c - r), min(1.0, c + r)


class Experiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = Path.cwd().resolve()
        self.python = self.repo / ".venv" / "bin" / "python"
        self.train_py = self.repo / "alphazero" / "envs" / "gocube" / "train.py"
        self.tool_dir = Path(__file__).resolve().parent
        self.evaluator = self.tool_dir / "evaluate_gocube_checkpoints.py"
        self.root = self.repo / "training_reports" / args.experiment_id
        self.publish = self.root / "publish"
        self.logs = self.root / "logs"
        self.state_path = self.root / "state-private.json"
        self.events_path = self.publish / "events.jsonl"
        self.source_run = args.source_run
        self.deadline_epoch = time.time() + args.max_hours * 3600.0
        self.state = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.publish.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.deadline_epoch = float(state["deadline_epoch"])
            return state
        tooling_commit = self.git("rev-parse", "origin/main", check=False).strip() or "unknown"
        state = {
            "schema_version": 1,
            "experiment_id": self.args.experiment_id,
            "status": "RUNNING",
            "started_at": now_iso(),
            "started_epoch": time.time(),
            "deadline_epoch": self.deadline_epoch,
            "deadline_at": datetime.fromtimestamp(self.deadline_epoch, timezone.utc).astimezone().isoformat(timespec="seconds"),
            "max_hours": self.args.max_hours,
            "reserve_minutes": self.args.reserve_minutes,
            "source_run": self.source_run,
            "frozen_training_commit": self.args.frozen_commit,
            "tooling_commit": tooling_commit,
            "workers": WORKERS,
            "parent_iteration": PARENT_ITERATION,
            "parent_checkpoint_sha256": None,
            "stages": {},
            "branches": {
                key: {
                    **value,
                    "run_name": f"{self.args.experiment_id}-{key.lower()}",
                    "status": "PENDING",
                    "latest_iteration": PARENT_ITERATION,
                }
                for key, value in BRANCHES.items()
            },
            "metrics": [],
            "evaluations": [],
            "artifacts": [],
            "selection": {},
            "notes": [
                "Primary practical objective: strength gain per wall-clock hour.",
                "Losses are diagnostics, not the primary ranking metric.",
                "F/G are an end-to-end outer-loop configuration test: with 16 workers, games/iteration also changes process_batch_size (8/16/32).",
            ],
        }
        self._save_state(state)
        return state

    def git(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(["git", *args], cwd=self.repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check and result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
        return result.stdout

    def event(self, event_type: str, **fields: Any) -> None:
        payload = {"time": now_iso(), "event": event_type, **fields}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"EXPERIMENT EVENT: {event_type} {fields}", flush=True)

    def _save_state(self, state: dict[str, Any] | None = None) -> None:
        if state is not None:
            self.state = state
        atomic_json(self.state_path, self.state)
        self.render_reports()

    def remaining_seconds(self) -> float:
        return self.deadline_epoch - time.time()

    def reserve_seconds(self) -> float:
        return self.args.reserve_minutes * 60.0

    def can_start(self, estimated_seconds: float, label: str) -> bool:
        remaining = self.remaining_seconds()
        allowed = remaining >= estimated_seconds + self.reserve_seconds()
        self.event(
            "deadline_check",
            label=label,
            remaining_seconds=round(remaining, 1),
            estimated_seconds=round(estimated_seconds, 1),
            reserve_seconds=round(self.reserve_seconds(), 1),
            allowed=allowed,
        )
        return allowed

    def validate_environment(self) -> None:
        if not self.python.is_file():
            raise RuntimeError(f"missing venv Python: {self.python}")
        if not self.evaluator.is_file():
            raise RuntimeError(f"missing evaluator next to runner: {self.evaluator}")
        head = self.git("rev-parse", "HEAD").strip()
        if head != self.args.frozen_commit:
            raise RuntimeError(f"training checkout HEAD {head} != frozen commit {self.args.frozen_commit}")
        if self.git("status", "--porcelain").strip():
            raise RuntimeError("tracked/untracked checkout is not clean; refuse overnight experiment")
        if os.cpu_count() is None or os.cpu_count() < WORKERS:
            raise RuntimeError(f"need at least {WORKERS} logical CPUs, found {os.cpu_count()}")
        self.event("environment_validated", head=head, cpu_count=os.cpu_count())

    def train_command(self, run_name: str, target_iteration: int, config: dict[str, Any]) -> list[str]:
        return [
            str(self.python), str(self.train_py),
            "--topology", "cube",
            "--size", "4",
            "--workers", str(WORKERS),
            "--sims", str(config["sims"]),
            "--arena-sims", str(ARENA_SIMS),
            "--games-per-iteration", str(config["games"]),
            "--iterations", str(target_iteration),
            "--train-batch-size", str(TRAIN_BATCH),
            "--fast-game-prob", str(config["pfast"]),
            "--endgame-sample-weight", str(ENDGAME_WEIGHT),
            "--no-arena",
            "--run-name", run_name,
        ]

    def stream_command(self, command: list[str], log_path: Path) -> tuple[int, float]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo)
        env["PYTHONUNBUFFERED"] = "1"
        started = time.time()
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== COMMAND {now_iso()} ===\n")
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
            return_code = process.wait()
        return return_code, time.time() - started

    def cleanup_uncommitted_iteration(self, run_name: str, iteration: int) -> None:
        checkpoint = self.checkpoint_path(run_name, iteration)
        data_dir = self.repo / "data" / run_name
        record_dir = data_dir / "records" / f"iteration-{iteration:04d}"
        tensor_paths = [data_dir / f"iteration-{iteration:04d}-{suffix}.pkl" for suffix in TENSOR_SUFFIXES]
        if checkpoint.exists():
            return
        removed = []
        for path in tensor_paths:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        if record_dir.exists():
            shutil.rmtree(record_dir)
            removed.append(str(record_dir))
        if removed:
            self.event("partial_iteration_cleaned", run_name=run_name, iteration=iteration, paths=removed)

    def checkpoint_path(self, run_name: str, iteration: int) -> Path:
        return self.repo / "checkpoint" / run_name / f"iteration-{iteration:04d}.pkl"

    def manifest_path(self, run_name: str, iteration: int) -> Path:
        return self.repo / "data" / run_name / "records" / f"iteration-{iteration:04d}" / "iteration-manifest.json"

    def tensor_path(self, run_name: str, iteration: int, suffix: str) -> Path:
        return self.repo / "data" / run_name / f"iteration-{iteration:04d}-{suffix}.pkl"

    def validate_completed_iteration(
        self,
        run_name: str,
        iteration: int,
        config: dict[str, Any],
        *,
        load_tensors: bool = True,
    ) -> dict[str, Any]:
        checkpoint = self.checkpoint_path(run_name, iteration)
        manifest_path = self.manifest_path(run_name, iteration)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f"missing checkpoint {checkpoint}")
        if not manifest_path.is_file():
            raise RuntimeError(f"missing manifest {manifest_path}")
        tensors = [self.tensor_path(run_name, iteration, suffix) for suffix in TENSOR_SUFFIXES]
        if not all(path.is_file() and path.stat().st_size > 0 for path in tensors):
            missing = [str(path) for path in tensors if not path.is_file() or path.stat().st_size == 0]
            raise RuntimeError(f"missing iteration tensors: {missing}")

        try:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu")
        required = {"state_dict", "opt_state", "sch_state", "args"}
        missing_keys = required.difference(checkpoint_payload)
        if missing_keys:
            raise RuntimeError(f"checkpoint missing keys {sorted(missing_keys)}")

        row_counts = []
        if load_tensors:
            for path in tensors:
                try:
                    tensor = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    tensor = torch.load(path, map_location="cpu")
                row_counts.append(int(tensor.size(0)))
            if len(set(row_counts)) != 1 or row_counts[0] <= 0:
                raise RuntimeError(f"tensor row mismatch: {row_counts}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_name") != run_name or int(manifest.get("iteration", -1)) != iteration:
            raise RuntimeError("iteration manifest identity mismatch")
        records = manifest.get("records", [])
        expected_games = int(config["games"])
        if len(records) != expected_games:
            raise RuntimeError(f"manifest records={len(records)}, expected={expected_games}")
        params = manifest.get("effective_iteration_parameters", {})
        expected_params = {
            "workers": WORKERS,
            "gamesPerIteration": expected_games,
            "process_batch_size": expected_games // WORKERS,
            "train_batch_size": TRAIN_BATCH,
            "numMCTSSims": int(config["sims"]),
            "probFastSim": float(config["pfast"]),
            "compareWithBaseline": False,
            "compareWithPast": False,
        }
        for key, expected in expected_params.items():
            actual = params.get(key)
            if actual != expected:
                raise RuntimeError(f"manifest {key}={actual!r}, expected={expected!r}")
        return {
            "checkpoint_sha256": sha256_file(checkpoint),
            "manifest": manifest,
            "tensor_rows": row_counts[0] if row_counts else None,
        }

    def parse_training_log(self, log_path: Path, iteration: int) -> dict[str, Any]:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        marker = f"=== V3 iteration {iteration} training summary ==="
        block = text.rsplit(marker, 1)[-1] if marker in text else text
        patterns: dict[str, tuple[str, Any]] = {
            "games": (r"Games:\s+(\d+)", int),
            "regular_decisions": (r"Regular decisions:\s+(\d+)", int),
            "fast_decisions": (r"Fast decisions:\s+(\d+)", int),
            "realized_fast_fraction": (r"Fast fraction:\s+([0-9.]+)%", lambda x: float(x) / 100.0),
            "base_positions": (r"Base positions:\s+(\d+)", int),
            "base_endgame_positions": (r"Base endgame:\s+(\d+)", int),
            "saved_samples": (r"Saved samples:\s+(\d+)", int),
            "history_datasets": (r"History datasets:\s+(\d+)", int),
            "window_samples": (r"Window samples:\s+(\d+)", int),
            "latest_iteration_samples": (r"Latest iteration samples:\s+(\d+)", int),
            "batch_size": (r"Batch size:\s+(\d+)", int),
            "optimizer_steps_planned": (r"Optimizer steps planned:\s+(\d+)", int),
            "optimizer_steps_actual": (r"Optimizer steps actual:\s+(\d+)", int),
            "examples_seen": (r"Examples seen:\s+(\d+)", int),
            "effective_passes": (r"Effective passes:\s+([0-9.]+)", float),
            "learning_rate": (r"LR:\s+([0-9.eE+-]+)", float),
        }
        metrics: dict[str, Any] = {}
        for key, (pattern, converter) in patterns.items():
            match = re.search(pattern, block)
            metrics[key] = converter(match.group(1)) if match else None
        infer = re.findall(r"Infer Batch:\s*([0-9.]+)", text)
        metrics["last_reported_inference_batch"] = float(infer[-1]) if infer else None
        losses = re.findall(
            r"Loss_pi:\s*([0-9.]+).*?Loss_v:\s*([0-9.]+)(?:.*?Loss_owner:\s*([0-9.]+).*?Loss_score:\s*([0-9.]+))?",
            text,
        )
        if losses:
            pi, value, owner, score = losses[-1]
            metrics.update({
                "loss_policy": float(pi),
                "loss_value": float(value),
                "loss_ownership": float(owner) if owner else None,
                "loss_score": float(score) if score else None,
            })
        return metrics

    def record_training_metrics(
        self,
        branch: str,
        run_name: str,
        iteration: int,
        config: dict[str, Any],
        elapsed: float,
        verified: dict[str, Any],
        log_path: Path,
        stage_id: str,
    ) -> None:
        parsed = self.parse_training_log(log_path, iteration)
        aggregate = verified["manifest"].get("aggregate_metrics", {})
        row = {
            "stage_id": stage_id,
            "branch": branch,
            "run_name": run_name,
            "iteration": iteration,
            "wall_seconds": elapsed,
            "sims": config["sims"],
            "fast_sims": FAST_SIMS,
            "pfast": config["pfast"],
            "games_per_iteration": config["games"],
            "workers": WORKERS,
            "process_batch_size": config["games"] // WORKERS,
            "train_batch_size": TRAIN_BATCH,
            "checkpoint_sha256": verified["checkpoint_sha256"],
            "tensor_rows": verified["tensor_rows"],
            "black_wins": aggregate.get("black_wins"),
            "white_wins": aggregate.get("white_wins"),
            "draws": aggregate.get("draws"),
            "average_game_length": aggregate.get("average_game_length"),
            "no_result_games": aggregate.get("terminal/no_result_games"),
            "training_valid_fraction": aggregate.get("terminal/training_valid_fraction"),
            **parsed,
        }
        self.state["metrics"] = [m for m in self.state["metrics"] if m.get("stage_id") != stage_id]
        self.state["metrics"].append(row)
        self.state["artifacts"].append({
            "kind": "checkpoint",
            "branch": branch,
            "iteration": iteration,
            "path": str(self.checkpoint_path(run_name, iteration).relative_to(self.repo)),
            "sha256": verified["checkpoint_sha256"],
        })
        self.publish_manifest(branch, run_name, iteration)

    def publish_manifest(self, branch: str, run_name: str, iteration: int) -> None:
        source = self.manifest_path(run_name, iteration)
        destination = self.publish / "manifests" / branch / f"iteration-{iteration:04d}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def publish_log_tail(self, stage_id: str, log_path: Path, lines: int = 400) -> None:
        if not log_path.is_file():
            return
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        atomic_text(self.publish / "logs" / f"{stage_id}.tail.log", "\n".join(content[-lines:]) + "\n")

    def run_training_stage(
        self,
        stage_id: str,
        branch: str,
        run_name: str,
        iteration: int,
        config: dict[str, Any],
        *,
        critical: bool,
    ) -> bool:
        existing = self.state["stages"].get(stage_id)
        log_path = self.logs / f"{stage_id}.log"
        if existing and existing.get("status") == "COMPLETED":
            self.validate_completed_iteration(run_name, iteration, config, load_tensors=False)
            return True

        if self.checkpoint_path(run_name, iteration).exists():
            try:
                verified = self.validate_completed_iteration(run_name, iteration, config)
                elapsed = float(existing.get("elapsed_seconds", 0.0)) if existing else 0.0
                self.record_training_metrics(branch, run_name, iteration, config, elapsed, verified, log_path, stage_id)
                self.state["stages"][stage_id] = {
                    "status": "COMPLETED", "recovered": True, "ended_at": now_iso(),
                    "elapsed_seconds": elapsed, "command": self.train_command(run_name, iteration, config),
                }
                self._save_state()
                return True
            except Exception as exc:
                raise RuntimeError(f"existing {run_name}@{iteration} is incomplete/corrupt: {exc}") from exc

        self.cleanup_uncommitted_iteration(run_name, iteration)
        command = self.train_command(run_name, iteration, config)
        self.state["stages"][stage_id] = {
            "status": "RUNNING", "started_at": now_iso(), "command": command,
            "branch": branch, "iteration": iteration,
        }
        self._save_state()
        self.event("stage_started", stage_id=stage_id, branch=branch, iteration=iteration)
        return_code, elapsed = self.stream_command(command, log_path)
        if return_code != 0:
            self.state["stages"][stage_id].update({
                "status": "FAILED", "ended_at": now_iso(), "exit_code": return_code, "elapsed_seconds": elapsed,
            })
            self.publish_log_tail(stage_id, log_path)
            self._save_state()
            self.event("stage_failed", stage_id=stage_id, exit_code=return_code)
            if critical:
                raise RuntimeError(f"critical stage {stage_id} failed with exit code {return_code}")
            return False

        try:
            verified = self.validate_completed_iteration(run_name, iteration, config)
        except Exception as exc:
            self.state["stages"][stage_id].update({
                "status": "FAILED_VALIDATION", "ended_at": now_iso(), "elapsed_seconds": elapsed, "error": str(exc),
            })
            self.publish_log_tail(stage_id, log_path)
            self._save_state()
            if critical:
                raise
            return False

        self.record_training_metrics(branch, run_name, iteration, config, elapsed, verified, log_path, stage_id)
        self.state["stages"][stage_id].update({
            "status": "COMPLETED", "ended_at": now_iso(), "exit_code": 0, "elapsed_seconds": elapsed,
            "checkpoint_sha256": verified["checkpoint_sha256"],
        })
        if branch in self.state["branches"]:
            self.state["branches"][branch]["latest_iteration"] = iteration
            self.state["branches"][branch]["status"] = "ACTIVE"
        self.publish_log_tail(stage_id, log_path)
        self._save_state()
        self.event("stage_completed", stage_id=stage_id, elapsed_seconds=round(elapsed, 1))
        return True

    def ensure_parent_five(self) -> None:
        for iteration in range(2, PARENT_ITERATION + 1):
            stage_id = f"canonical_iter_{iteration}"
            self.run_training_stage(
                stage_id,
                "PARENT",
                self.source_run,
                iteration,
                BASELINE,
                critical=True,
            )
        parent = self.checkpoint_path(self.source_run, PARENT_ITERATION)
        parent_sha = sha256_file(parent)
        if self.state.get("parent_checkpoint_sha256") not in (None, parent_sha):
            raise RuntimeError("parent iteration-5 checkpoint SHA changed across resume")
        self.state["parent_checkpoint_sha256"] = parent_sha
        atomic_json(self.publish / "parent5.json", {
            "run_name": self.source_run,
            "iteration": PARENT_ITERATION,
            "checkpoint": str(parent.relative_to(self.repo)),
            "sha256": parent_sha,
            "frozen_training_commit": self.args.frozen_commit,
        })
        self._save_state()
        self.event("parent_frozen", sha256=parent_sha)

    def fork_branch(self, branch: str) -> bool:
        info = self.state["branches"][branch]
        run_name = info["run_name"]
        checkpoint_dir = self.repo / "checkpoint" / run_name
        data_dir = self.repo / "data" / run_name
        provenance_path = checkpoint_dir / "fork-provenance.json"
        parent_sha = self.state["parent_checkpoint_sha256"]

        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("parent_checkpoint_sha256") != parent_sha:
                raise RuntimeError(f"branch {branch} provenance does not match frozen parent")
            return True
        if checkpoint_dir.exists() or data_dir.exists():
            raise RuntimeError(f"branch {branch} directories exist without provenance; refusing ambiguous fork")

        checkpoint_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        source_checkpoint_dir = self.repo / "checkpoint" / self.source_run
        source_data_dir = self.repo / "data" / self.source_run
        for iteration in range(0, PARENT_ITERATION + 1):
            source = source_checkpoint_dir / f"iteration-{iteration:04d}.pkl"
            shutil.copy2(source, checkpoint_dir / source.name)
        for iteration in range(1, PARENT_ITERATION + 1):
            for suffix in TENSOR_SUFFIXES:
                source = source_data_dir / f"iteration-{iteration:04d}-{suffix}.pkl"
                shutil.copy2(source, data_dir / source.name)

        run_manifest = json.loads((source_checkpoint_dir / "gocube-run.json").read_text(encoding="utf-8"))
        run_manifest["runName"] = run_name
        atomic_json(checkpoint_dir / "gocube-run.json", run_manifest)
        provenance = {
            "schema_version": 1,
            "experiment_id": self.args.experiment_id,
            "branch": branch,
            "branch_config": {key: info[key] for key in ("sims", "pfast", "games", "axis", "label")},
            "parent_run": self.source_run,
            "parent_iteration": PARENT_ITERATION,
            "parent_checkpoint_sha256": parent_sha,
            "frozen_training_commit": self.args.frozen_commit,
            "created_at": now_iso(),
        }
        atomic_json(provenance_path, provenance)
        if sha256_file(self.checkpoint_path(run_name, PARENT_ITERATION)) != parent_sha:
            raise RuntimeError(f"fork {branch} changed parent checkpoint bytes")
        info["status"] = "FORKED"
        self._save_state()
        self.event("branch_forked", branch=branch, run_name=run_name)
        return True

    def estimate_base_iteration_seconds(self) -> float:
        canonical = [
            float(row["wall_seconds"])
            for row in self.state["metrics"]
            if row.get("branch") == "PARENT" and int(row.get("iteration", 0)) >= 2 and float(row.get("wall_seconds", 0)) > 0
        ]
        if canonical:
            return sum(canonical) / len(canonical)
        return 1800.0

    def estimate_train_seconds(self, config: dict[str, Any]) -> float:
        baseline_effective = (1.0 - BASELINE["pfast"]) * BASELINE["sims"] + BASELINE["pfast"] * FAST_SIMS
        effective = (1.0 - config["pfast"]) * config["sims"] + config["pfast"] * FAST_SIMS
        factor = (effective / baseline_effective) * (config["games"] / BASELINE["games"])
        if config["games"] != 256:
            factor *= 1.15
        return max(300.0, self.estimate_base_iteration_seconds() * factor * 1.20)

    def observed_eval_seconds_per_game(self) -> float:
        values = [
            float(row["elapsed_seconds"]) / max(1, int(row["games_requested"]))
            for row in self.state["evaluations"]
            if float(row.get("elapsed_seconds", 0)) > 0 and int(row.get("games_requested", 0)) > 0
        ]
        return sum(values) / len(values) if values else 12.0

    def run_evaluation(
        self,
        stage_id: str,
        candidate_branch: str,
        candidate_iteration: int,
        reference_id: str,
        reference_path: Path,
        games: int,
        *,
        critical: bool = False,
    ) -> bool:
        existing = self.state["stages"].get(stage_id)
        output = self.root / "evaluations" / f"{stage_id}.json"
        log_path = self.logs / f"{stage_id}.log"
        if existing and existing.get("status") == "COMPLETED" and output.is_file():
            return True

        candidate_run = self.state["branches"][candidate_branch]["run_name"]
        candidate_path = self.checkpoint_path(candidate_run, candidate_iteration)
        command = [
            str(self.python), str(self.evaluator),
            "--candidate", str(candidate_path),
            "--reference", str(reference_path),
            "--output", str(output),
            "--candidate-id", f"{candidate_branch}@{candidate_iteration}",
            "--reference-id", reference_id,
            "--topology", "cube",
            "--size", "4",
            "--games", str(games),
            "--sims", str(ARENA_SIMS),
            "--seed", str(self.args.eval_seed + candidate_iteration * 100 + ord(candidate_branch)),
        ]
        self.state["stages"][stage_id] = {
            "status": "RUNNING", "started_at": now_iso(), "command": command,
            "candidate": candidate_branch, "candidate_iteration": candidate_iteration,
            "reference": reference_id, "games": games,
        }
        self._save_state()
        self.event("evaluation_started", stage_id=stage_id, games=games)
        return_code, elapsed = self.stream_command(command, log_path)
        if return_code != 0 or not output.is_file():
            self.state["stages"][stage_id].update({
                "status": "FAILED", "ended_at": now_iso(), "exit_code": return_code, "elapsed_seconds": elapsed,
            })
            self.publish_log_tail(stage_id, log_path)
            self._save_state()
            if critical:
                raise RuntimeError(f"evaluation {stage_id} failed")
            return False
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload.update({
            "stage_id": stage_id,
            "candidate_branch": candidate_branch,
            "candidate_iteration": candidate_iteration,
            "reference_id": reference_id,
        })
        self.state["evaluations"] = [e for e in self.state["evaluations"] if e.get("stage_id") != stage_id]
        self.state["evaluations"].append(payload)
        self.state["stages"][stage_id].update({
            "status": "COMPLETED", "ended_at": now_iso(), "exit_code": 0, "elapsed_seconds": elapsed,
        })
        self.publish_log_tail(stage_id, log_path)
        shutil.copy2(output, self.publish / "evaluations" / output.name)
        self._save_state()
        self.event("evaluation_completed", stage_id=stage_id, score=payload.get("candidate_score_rate"))
        return True

    def eval_score(self, branch: str, iteration: int, reference_id: str) -> float | None:
        matches = [
            row for row in self.state["evaluations"]
            if row.get("candidate_branch") == branch
            and int(row.get("candidate_iteration", -1)) == iteration
            and row.get("reference_id") == reference_id
        ]
        if not matches:
            return None
        return float(matches[-1]["candidate_score_rate"])

    def choose_axis_promotions(self) -> list[str]:
        selected = ["A"]
        details = {}
        for axis, pair in AXIS_PAIRS.items():
            scored = [(branch, self.eval_score(branch, 6, f"parent@{PARENT_ITERATION}")) for branch in pair]
            scored = [(b, s) for b, s in scored if s is not None]
            if scored:
                winner = max(scored, key=lambda item: item[1])[0]
                selected.append(winner)
                details[axis] = {"winner": winner, "scores": dict(scored)}
            else:
                details[axis] = {"winner": None, "scores": {}}
        self.state["selection"]["stage2"] = {"branches": selected, "axes": details, "selected_at": now_iso()}
        self._save_state()
        return selected

    def choose_stage3_challenger(self, promoted: list[str]) -> str | None:
        challengers = [branch for branch in promoted if branch != "A"]
        direct = []
        for branch in challengers:
            score = self.eval_score(branch, 7, "A@7")
            if score is not None:
                direct.append((branch, score))
        if direct:
            winner = max(direct, key=lambda item: item[1])[0]
        else:
            common = [(b, self.eval_score(b, 7, f"parent@{PARENT_ITERATION}")) for b in challengers]
            common = [(b, s) for b, s in common if s is not None]
            winner = max(common, key=lambda item: item[1])[0] if common else None
        self.state["selection"]["stage3"] = {"challenger": winner, "selected_at": now_iso()}
        self._save_state()
        return winner

    def run(self) -> None:
        self.validate_environment()
        self.ensure_parent_five()
        parent_path = self.checkpoint_path(self.source_run, PARENT_ITERATION)

        for branch in BRANCHES:
            self.fork_branch(branch)

        # Stage 1: every branch gets exactly one descendant iteration and a common-parent evaluation.
        for branch, config in BRANCHES.items():
            estimate = self.estimate_train_seconds(config)
            if not self.can_start(estimate, f"stage1 train {branch}@6"):
                self.state["branches"][branch]["status"] = "SKIPPED_DEADLINE"
                self._save_state()
                continue
            run_name = self.state["branches"][branch]["run_name"]
            ok = self.run_training_stage(f"stage1_train_{branch}_i6", branch, run_name, 6, config, critical=False)
            if not ok:
                self.state["branches"][branch]["status"] = "FAILED"
                self._save_state()
                continue
            eval_games = self.args.stage1_eval_games
            estimate_eval = self.observed_eval_seconds_per_game() * eval_games * 1.25
            if self.can_start(estimate_eval, f"stage1 eval {branch}@6"):
                self.run_evaluation(
                    f"stage1_eval_{branch}_i6_vs_parent5", branch, 6,
                    f"parent@{PARENT_ITERATION}", parent_path, eval_games,
                )

        promoted = self.choose_axis_promotions()

        # Stage 2: control + best direction on each independent axis.
        completed_stage2 = []
        for branch in promoted:
            if self.state["branches"][branch]["status"] in ("FAILED", "SKIPPED_DEADLINE"):
                continue
            config = BRANCHES[branch]
            estimate = self.estimate_train_seconds(config)
            if not self.can_start(estimate, f"stage2 train {branch}@7"):
                continue
            run_name = self.state["branches"][branch]["run_name"]
            if self.run_training_stage(f"stage2_train_{branch}_i7", branch, run_name, 7, config, critical=False):
                completed_stage2.append(branch)
                eval_games = self.args.stage2_eval_games
                estimate_eval = self.observed_eval_seconds_per_game() * eval_games * 1.25
                if self.can_start(estimate_eval, f"stage2 common-parent eval {branch}@7"):
                    self.run_evaluation(
                        f"stage2_eval_{branch}_i7_vs_parent5", branch, 7,
                        f"parent@{PARENT_ITERATION}", parent_path, eval_games,
                    )

        # Direct same-depth comparisons answer whether a tuned variant beats baseline A.
        if "A" in completed_stage2:
            a_path = self.checkpoint_path(self.state["branches"]["A"]["run_name"], 7)
            for branch in completed_stage2:
                if branch == "A":
                    continue
                games = self.args.direct_eval_games
                estimate_eval = self.observed_eval_seconds_per_game() * games * 1.25
                if not self.can_start(estimate_eval, f"direct eval {branch}@7 vs A@7"):
                    continue
                self.run_evaluation(
                    f"direct_eval_{branch}_i7_vs_A_i7", branch, 7, "A@7", a_path, games,
                )

        # Stage 3: if time remains, deepen baseline and the strongest challenger once more.
        challenger = self.choose_stage3_challenger(completed_stage2)
        finalists = [b for b in ("A", challenger) if b]
        completed_stage3 = []
        for branch in finalists:
            config = BRANCHES[branch]
            estimate = self.estimate_train_seconds(config)
            if not self.can_start(estimate, f"stage3 train {branch}@8"):
                continue
            run_name = self.state["branches"][branch]["run_name"]
            if self.run_training_stage(f"stage3_train_{branch}_i8", branch, run_name, 8, config, critical=False):
                completed_stage3.append(branch)

        if challenger and set(("A", challenger)).issubset(completed_stage3):
            games = self.args.final_eval_games
            estimate_eval = self.observed_eval_seconds_per_game() * games * 1.25
            if self.can_start(estimate_eval, f"final direct eval {challenger}@8 vs A@8"):
                a8 = self.checkpoint_path(self.state["branches"]["A"]["run_name"], 8)
                self.run_evaluation(
                    f"final_eval_{challenger}_i8_vs_A_i8", challenger, 8, "A@8", a8, games,
                )

        self.state["status"] = "DONE"
        self.state["ended_at"] = now_iso()
        self.state["ended_epoch"] = time.time()
        self._save_state()
        self.event("experiment_completed", remaining_seconds=round(self.remaining_seconds(), 1))

    def render_reports(self) -> None:
        self.publish.mkdir(parents=True, exist_ok=True)
        public_state = dict(self.state)
        atomic_json(self.publish / "state.json", public_state)
        design = {
            "schema_version": 1,
            "experiment_id": self.state["experiment_id"],
            "source_run": self.source_run,
            "frozen_training_commit": self.args.frozen_commit,
            "tooling_commit": self.state.get("tooling_commit"),
            "workers": WORKERS,
            "parent_iteration": PARENT_ITERATION,
            "fixed": {
                "fast_sims": FAST_SIMS,
                "train_batch": TRAIN_BATCH,
                "endgame_weight": ENDGAME_WEIGHT,
                "arena_sims": ARENA_SIMS,
                "arena_fast_probability": 0.0,
                "arena_root_noise": False,
                "arena_root_temperature": False,
                "arena_action_temperature": 0.0,
                "arena_batched": False,
                "model_gating": False,
            },
            "branches": BRANCHES,
            "axis_pairs": AXIS_PAIRS,
            "stage_policy": {
                "stage1": "all A-G: parent5 -> iteration6, then common-parent evaluation",
                "stage2": "A + best of B/C + best of D/E + best of F/G: -> iteration7",
                "direct": "stage2 challengers vs same-depth A@7",
                "stage3": "if time: A + strongest challenger -> iteration8, direct final",
            },
            "primary_metric": "evaluation strength per wall-clock hour",
            "caveat": "F/G also change process_batch_size because gamesPerIteration/workers is coupled in the current implementation.",
        }
        atomic_json(self.publish / "experiment.json", design)
        self.write_metrics_csv()
        self.write_evaluations_csv()
        atomic_json(self.publish / "artifacts.json", self.state.get("artifacts", []))
        atomic_text(self.publish / "summary.md", self.summary_markdown())

    def write_metrics_csv(self) -> None:
        rows = sorted(self.state.get("metrics", []), key=lambda r: (str(r.get("branch")), int(r.get("iteration", 0))))
        fields = [
            "stage_id", "branch", "run_name", "iteration", "wall_seconds", "sims", "fast_sims", "pfast",
            "games_per_iteration", "workers", "process_batch_size", "train_batch_size", "checkpoint_sha256",
            "tensor_rows", "regular_decisions", "fast_decisions", "realized_fast_fraction", "base_positions",
            "base_endgame_positions", "saved_samples", "history_datasets", "window_samples", "latest_iteration_samples",
            "optimizer_steps_planned", "optimizer_steps_actual", "examples_seen", "effective_passes", "learning_rate",
            "last_reported_inference_batch", "black_wins", "white_wins", "draws", "average_game_length",
            "no_result_games", "training_valid_fraction", "loss_policy", "loss_value", "loss_ownership", "loss_score",
        ]
        path = self.publish / "metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    def write_evaluations_csv(self) -> None:
        rows = self.state.get("evaluations", [])
        fields = [
            "stage_id", "candidate_branch", "candidate_iteration", "reference_id", "games_requested", "games_effective",
            "candidate_wins", "reference_wins", "draws", "no_results", "candidate_score_rate",
            "candidate_score_ci95_approx", "mcts_sims", "elapsed_seconds", "seconds_per_game", "seed",
        ]
        path = self.publish / "evaluations.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                copy = dict(row)
                copy["candidate_score_ci95_approx"] = json.dumps(copy.get("candidate_score_ci95_approx"))
                writer.writerow(copy)
        os.replace(temporary, path)

    def branch_training_seconds(self, branch: str) -> float:
        return sum(float(row.get("wall_seconds", 0)) for row in self.state.get("metrics", []) if row.get("branch") == branch)

    def latest_common_eval(self, branch: str) -> dict[str, Any] | None:
        rows = [
            row for row in self.state.get("evaluations", [])
            if row.get("candidate_branch") == branch and str(row.get("reference_id", "")).startswith("parent@")
        ]
        return max(rows, key=lambda r: int(r.get("candidate_iteration", 0))) if rows else None

    def recommendation_text(self) -> str:
        direct_rows = [
            row for row in self.state.get("evaluations", [])
            if row.get("reference_id") in ("A@7", "A@8") and row.get("candidate_branch") != "A"
        ]
        if direct_rows:
            best = max(direct_rows, key=lambda r: float(r.get("candidate_score_rate", 0)))
            score = float(best["candidate_score_rate"])
            ci = best.get("candidate_score_ci95_approx") or [0.0, 1.0]
            branch = best["candidate_branch"]
            if float(ci[0]) > 0.5:
                return f"Strong evidence: branch {branch} beats same-depth baseline A (score {score:.3f}, approximate 95% CI {ci[0]:.3f}-{ci[1]:.3f})."
            if score > 0.5:
                return f"Provisional lead: branch {branch} is ahead of same-depth baseline A (score {score:.3f}) but the interval still overlaps 0.5; repeat before locking the big-training default."
            return "No tuned branch has yet shown a direct same-depth advantage over baseline A; keep the current baseline provisionally."
        common = [(branch, self.latest_common_eval(branch)) for branch in BRANCHES]
        common = [(b, r) for b, r in common if r is not None]
        if common:
            branch, row = max(common, key=lambda item: float(item[1]["candidate_score_rate"]))
            return f"Only common-parent evidence is available so far; branch {branch} currently leads at score {float(row['candidate_score_rate']):.3f}. Treat this as provisional until a same-depth A comparison completes."
        return "No strength evaluation has completed yet; throughput/training diagnostics alone are insufficient to choose a winner."

    def summary_markdown(self) -> str:
        lines = [
            "# Cube 4 overnight hyperparameter experiment",
            "",
            f"- Experiment: `{self.state['experiment_id']}`",
            f"- Status: `{self.state.get('status')}`",
            f"- Started: {self.state.get('started_at')}",
            f"- Deadline: {self.state.get('deadline_at')}",
            f"- Frozen training commit: `{self.args.frozen_commit}`",
            f"- Parent: `{self.source_run}@{PARENT_ITERATION}`",
            f"- Parent checkpoint SHA256: `{self.state.get('parent_checkpoint_sha256') or 'pending'}`",
            f"- Workers: **{WORKERS}** everywhere",
            "",
            "## Decision status",
            "",
            self.recommendation_text(),
            "",
            "Primary ranking is based on fixed-Arena playing strength, with wall-clock cost shown alongside it. Training losses are diagnostics only.",
            "",
            "## Branch design",
            "",
            "| Branch | Regular sims | pFast | Games/iteration | Process batch | Purpose |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for branch, cfg in BRANCHES.items():
            lines.append(
                f"| {branch} | {cfg['sims']} | {cfg['pfast']:.2f} | {cfg['games']} | {cfg['games'] // WORKERS} | {cfg['label']} |"
            )
        lines += [
            "",
            "> **Causal caveat for F/G:** in the current implementation `process_batch_size = games_per_iteration / workers`. Therefore 128/256/512 games also means process batches 8/16/32. F/G measure the practical outer-loop package, not a perfectly isolated game-count variable.",
            "",
            "## Latest training state",
            "",
            "| Branch | Latest iter | Train wall | Samples (latest) | Opt steps | Realized fast | No-result | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for branch in BRANCHES:
            info = self.state["branches"][branch]
            rows = [r for r in self.state.get("metrics", []) if r.get("branch") == branch]
            latest = max(rows, key=lambda r: int(r.get("iteration", 0))) if rows else {}
            seconds = self.branch_training_seconds(branch)
            fast = latest.get("realized_fast_fraction")
            nr = latest.get("no_result_games")
            games = latest.get("games") or latest.get("games_per_iteration") or 0
            nr_rate = (float(nr) / float(games)) if nr is not None and games else None
            lines.append(
                f"| {branch} | {info.get('latest_iteration', PARENT_ITERATION)} | {seconds/3600:.2f} h | "
                f"{latest.get('latest_iteration_samples') or latest.get('tensor_rows') or '—'} | "
                f"{latest.get('optimizer_steps_actual') if latest else '—'} | "
                f"{fast:.3f}" if isinstance(fast, (int, float)) else "| —"
            )
            # Replace the partially built row above with a robust explicit row.
            lines[-1] = (
                f"| {branch} | {info.get('latest_iteration', PARENT_ITERATION)} | {seconds/3600:.2f} h | "
                f"{latest.get('latest_iteration_samples') or latest.get('tensor_rows') or '—'} | "
                f"{latest.get('optimizer_steps_actual', '—') if latest else '—'} | "
                f"{(f'{float(fast):.3f}' if fast is not None else '—')} | "
                f"{(f'{nr_rate:.3%}' if nr_rate is not None else '—')} | {info.get('status')} |"
            )
        lines += [
            "",
            "## Strength evaluations",
            "",
            "| Candidate | Reference | Games | W-L-D-NR | Score | Approx. 95% CI | Eval wall |",
            "|---|---|---:|---|---:|---|---:|",
        ]
        evaluations = sorted(
            self.state.get("evaluations", []),
            key=lambda r: (int(r.get("candidate_iteration", 0)), str(r.get("candidate_branch")), str(r.get("reference_id"))),
        )
        for row in evaluations:
            ci = row.get("candidate_score_ci95_approx") or [0.0, 1.0]
            lines.append(
                f"| {row.get('candidate_branch')}@{row.get('candidate_iteration')} | {row.get('reference_id')} | "
                f"{row.get('games_requested')} | {row.get('candidate_wins')}-{row.get('reference_wins')}-{row.get('draws')}-{row.get('no_results')} | "
                f"{float(row.get('candidate_score_rate', 0)):.3f} | {float(ci[0]):.3f}-{float(ci[1]):.3f} | "
                f"{float(row.get('elapsed_seconds', 0))/60:.1f} min |"
            )
        if not evaluations:
            lines.append("| — | — | — | — | — | — | — |")

        lines += [
            "",
            "## Axis readout after the first descendant iteration",
            "",
        ]
        for axis, pair in AXIS_PAIRS.items():
            scores = [(b, self.eval_score(b, 6, f"parent@{PARENT_ITERATION}")) for b in pair]
            printable = ", ".join(f"{b}={s:.3f}" if s is not None else f"{b}=pending" for b, s in scores)
            lines.append(f"- **{axis}:** {printable}.")
        lines += [
            "",
            "## Report completeness",
            "",
            "The GitHub report intentionally includes `experiment.json`, `state.json`, `metrics.csv`, `evaluations.csv`, `artifacts.json`, `events.jsonl`, compact iteration manifests, evaluation JSON files, and per-stage log tails. Together these preserve exact configuration/provenance, compute cost, self-play mix, data volume, optimizer work, terminal quality, checkpoint hashes, and fixed-Arena strength.",
            "",
            "A final choice should not be made from loss curves alone. Prefer direct same-depth comparison against A; if its interval overlaps 0.5, repeat the leading candidate with another self-play/evaluation seed before committing to long training.",
        ]
        return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-run", default=SOURCE_RUN_DEFAULT)
    parser.add_argument("--frozen-commit", default=FROZEN_TRAINING_COMMIT)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--reserve-minutes", type=float, default=35.0)
    parser.add_argument("--stage1-eval-games", type=int, default=24)
    parser.add_argument("--stage2-eval-games", type=int, default=32)
    parser.add_argument("--direct-eval-games", type=int, default=40)
    parser.add_argument("--final-eval-games", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=20260906)
    args = parser.parse_args()
    for name in ("stage1_eval_games", "stage2_eval_games", "direct_eval_games", "final_eval_games"):
        value = getattr(args, name)
        if value < 2 or value % 2:
            parser.error(f"--{name.replace('_', '-')} must be an even integer >= 2")
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    return args


def main() -> int:
    args = parse_args()
    experiment = Experiment(args)
    try:
        experiment.run()
        return 0
    except BaseException as exc:
        experiment.state["status"] = "FAILED"
        experiment.state["ended_at"] = now_iso()
        experiment.state["fatal_error"] = f"{type(exc).__name__}: {exc}"
        experiment._save_state()
        experiment.event("experiment_failed", error=experiment.state["fatal_error"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
