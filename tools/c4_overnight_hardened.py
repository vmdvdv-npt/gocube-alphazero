#!/usr/bin/env python3
"""Reliability wrapper for the Cube 4 overnight experiment.

This module deliberately leaves the frozen AlphaZero training checkout alone.
It wraps the orchestration layer with recovery for interruption windows that are
safe to replay from the previous committed iteration.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path.cwd().resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_PATH = Path(__file__).resolve().with_name("c4_overnight_experiment.py")
_spec = importlib.util.spec_from_file_location("c4_overnight_experiment_base", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load base overnight runner: {BASE_PATH}")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def _stamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")


class HardenedExperiment(base.Experiment):
    """Base experiment plus fail-closed, replay-safe recovery."""

    def __init__(self, args):
        if str(Path.cwd().resolve()) not in sys.path:
            sys.path.insert(0, str(Path.cwd().resolve()))
        super().__init__(args)
        (self.publish / "evaluations").mkdir(parents=True, exist_ok=True)
        (self.root / "quarantine").mkdir(parents=True, exist_ok=True)

    def _tooling_commit(self) -> str:
        explicit = os.environ.get("GOCUBE_NIGHT_TOOLING_COMMIT", "").strip()
        if explicit:
            return explicit
        name = self.tool_dir.name
        prefix = "gocube-night-tools-"
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
        return str(self.state.get("tooling_commit") or "unknown")

    def _prepare_attempt_state(self) -> None:
        previous = self.state.get("status")
        tooling = self._tooling_commit()
        history = list(self.state.get("tooling_history", []))
        if not history or history[-1] != tooling:
            history.append(tooling)
        self.state["tooling_history"] = history
        self.state["tooling_commit"] = tooling
        self.state["run_attempts"] = int(self.state.get("run_attempts", 0)) + 1
        self.state["last_attempt_started_at"] = base.now_iso()
        if previous != "RUNNING":
            self.state["last_resume_previous_status"] = previous
            self.state["last_resumed_at"] = base.now_iso()
            self.state["resume_count"] = int(self.state.get("resume_count", 0)) + 1
        self.state["status"] = "RUNNING"
        self.state.pop("fatal_error", None)
        self.state.pop("ended_at", None)
        self.state.pop("ended_epoch", None)
        self._save_state()
        self.event(
            "attempt_started",
            previous_status=previous,
            tooling_commit=tooling,
            attempt=self.state["run_attempts"],
        )

    def run(self) -> None:
        self._prepare_attempt_state()
        return super().run()

    def _later_checkpoint_exists(self, run_name: str, iteration: int) -> bool:
        directory = self.repo / "checkpoint" / run_name
        if not directory.is_dir():
            return False
        for path in directory.glob("iteration-*.pkl"):
            match = re.fullmatch(r"iteration-(\d+)\.pkl", path.name)
            if match and int(match.group(1)) > iteration:
                return True
        return False

    def _later_data_exists(self, run_name: str, iteration: int) -> bool:
        directory = self.repo / "data" / run_name
        if not directory.is_dir():
            return False
        for path in directory.glob("iteration-*-data.pkl"):
            match = re.fullmatch(r"iteration-(\d+)-data\.pkl", path.name)
            if match and int(match.group(1)) > iteration:
                return True
        records = directory / "records"
        if records.is_dir():
            for path in records.glob("iteration-*"):
                match = re.fullmatch(r"iteration-(\d+)", path.name)
                if match and int(match.group(1)) > iteration:
                    return True
        return False

    def _quarantine_iteration(self, stage_id: str, run_name: str, iteration: int, reason: str) -> None:
        destination = self.root / "quarantine" / f"{stage_id}-{_stamp()}-{os.getpid()}"
        destination.mkdir(parents=True, exist_ok=False)
        moved: list[str] = []

        checkpoint = self.checkpoint_path(run_name, iteration)
        if checkpoint.exists():
            target = destination / checkpoint.name
            shutil.move(str(checkpoint), str(target))
            moved.append(str(checkpoint.relative_to(self.repo)))

        for suffix in base.TENSOR_SUFFIXES:
            path = self.tensor_path(run_name, iteration, suffix)
            if path.exists():
                target = destination / path.name
                shutil.move(str(path), str(target))
                moved.append(str(path.relative_to(self.repo)))

        record_dir = self.repo / "data" / run_name / "records" / f"iteration-{iteration:04d}"
        if record_dir.exists():
            target = destination / "records"
            shutil.move(str(record_dir), str(target))
            moved.append(str(record_dir.relative_to(self.repo)))

        self.event(
            "iteration_quarantined",
            stage_id=stage_id,
            run_name=run_name,
            iteration=iteration,
            reason=reason,
            moved=moved,
            quarantine=str(destination.relative_to(self.repo)),
        )

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
        checkpoint = self.checkpoint_path(run_name, iteration)
        existing = self.state.get("stages", {}).get(stage_id)
        if checkpoint.exists():
            try:
                # Full validation here catches corrupt tensor files even when a
                # previous state already called the stage COMPLETED.
                self.validate_completed_iteration(run_name, iteration, config, load_tensors=True)
            except Exception as exc:
                replayable_state = existing and existing.get("status") in {
                    "RUNNING", "FAILED", "FAILED_VALIDATION"
                }
                has_later = self._later_checkpoint_exists(run_name, iteration) or self._later_data_exists(
                    run_name, iteration
                )
                if replayable_state and not has_later:
                    self._quarantine_iteration(stage_id, run_name, iteration, str(exc))
                    self.state["stages"][stage_id] = {
                        "status": "RECOVERING",
                        "recovery_reason": str(exc),
                        "recovered_at": base.now_iso(),
                        "branch": branch,
                        "iteration": iteration,
                    }
                    self._save_state()
                else:
                    raise RuntimeError(
                        f"cannot safely replay invalid {run_name}@{iteration}: {exc}"
                    ) from exc
        return super().run_training_stage(
            stage_id, branch, run_name, iteration, config, critical=critical
        )

    def _branch_has_descendants(self, run_name: str) -> bool:
        return self._later_checkpoint_exists(run_name, base.PARENT_ITERATION) or self._later_data_exists(
            run_name, base.PARENT_ITERATION
        )

    def _validate_existing_fork(self, branch: str) -> None:
        info = self.state["branches"][branch]
        run_name = info["run_name"]
        checkpoint_dir = self.repo / "checkpoint" / run_name
        data_dir = self.repo / "data" / run_name
        provenance_path = checkpoint_dir / "fork-provenance.json"
        if not provenance_path.is_file():
            raise RuntimeError("missing fork provenance")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("experiment_id") != self.args.experiment_id:
            raise RuntimeError("fork provenance experiment mismatch")
        if provenance.get("branch") != branch:
            raise RuntimeError("fork provenance branch mismatch")
        if provenance.get("parent_checkpoint_sha256") != self.state.get("parent_checkpoint_sha256"):
            raise RuntimeError("fork provenance parent SHA mismatch")

        manifest_path = checkpoint_dir / "gocube-run.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runName") != run_name:
            raise RuntimeError("fork run manifest name mismatch")

        for iteration in range(0, base.PARENT_ITERATION + 1):
            path = checkpoint_dir / f"iteration-{iteration:04d}.pkl"
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"fork missing checkpoint {path.name}")
        for iteration in range(1, base.PARENT_ITERATION + 1):
            for suffix in base.TENSOR_SUFFIXES:
                path = data_dir / f"iteration-{iteration:04d}-{suffix}.pkl"
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RuntimeError(f"fork missing tensor {path.name}")

        parent = checkpoint_dir / f"iteration-{base.PARENT_ITERATION:04d}.pkl"
        if base.sha256_file(parent) != self.state.get("parent_checkpoint_sha256"):
            raise RuntimeError("fork parent checkpoint bytes differ from frozen parent")

    def _quarantine_fork(self, branch: str, reason: str) -> None:
        info = self.state["branches"][branch]
        run_name = info["run_name"]
        destination = self.root / "quarantine" / f"fork-{branch}-{_stamp()}-{os.getpid()}"
        destination.mkdir(parents=True, exist_ok=False)
        for label, path in (
            ("checkpoint", self.repo / "checkpoint" / run_name),
            ("data", self.repo / "data" / run_name),
        ):
            if path.exists():
                shutil.move(str(path), str(destination / label))
        self.event(
            "fork_quarantined",
            branch=branch,
            run_name=run_name,
            reason=reason,
            quarantine=str(destination.relative_to(self.repo)),
        )

    def fork_branch(self, branch: str) -> bool:
        info = self.state["branches"][branch]
        run_name = info["run_name"]
        checkpoint_dir = self.repo / "checkpoint" / run_name
        data_dir = self.repo / "data" / run_name
        if checkpoint_dir.exists() or data_dir.exists():
            try:
                self._validate_existing_fork(branch)
                if info.get("status") == "PENDING":
                    info["status"] = "FORKED"
                    self._save_state()
                return True
            except Exception as exc:
                if self._branch_has_descendants(run_name):
                    raise RuntimeError(
                        f"branch {branch} has descendants and an invalid fork; refusing destructive recovery: {exc}"
                    ) from exc
                self._quarantine_fork(branch, str(exc))
                info["status"] = "PENDING"
                self._save_state()
        result = super().fork_branch(branch)
        self._validate_existing_fork(branch)
        return result

    def _validated_eval_payload(
        self,
        output: Path,
        candidate_path: Path,
        reference_path: Path,
        games: int,
    ) -> dict[str, Any]:
        payload = json.loads(output.read_text(encoding="utf-8"))
        if int(payload.get("games_requested", -1)) != int(games):
            raise RuntimeError("evaluation games_requested mismatch")
        if int(payload.get("mcts_sims", -1)) != int(base.ARENA_SIMS):
            raise RuntimeError("evaluation MCTS budget mismatch")
        if payload.get("candidate", {}).get("sha256") != base.sha256_file(candidate_path):
            raise RuntimeError("evaluation candidate SHA mismatch")
        if payload.get("reference", {}).get("sha256") != base.sha256_file(reference_path):
            raise RuntimeError("evaluation reference SHA mismatch")
        effective = int(payload.get("games_effective", -1))
        no_results = int(payload.get("no_results", -1))
        wins = int(payload.get("candidate_wins", -1)) + int(payload.get("reference_wins", -1))
        draws = int(payload.get("draws", -1))
        if effective <= 0 or effective + no_results != games or wins + draws != effective:
            raise RuntimeError("evaluation game accounting mismatch")
        score = float(payload.get("candidate_score_rate", -1.0))
        if not 0.0 <= score <= 1.0:
            raise RuntimeError("evaluation score outside [0,1]")
        return payload

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
        candidate_run = self.state["branches"][candidate_branch]["run_name"]
        candidate_path = self.checkpoint_path(candidate_run, candidate_iteration)
        output = self.root / "evaluations" / f"{stage_id}.json"
        existing = self.state.get("stages", {}).get(stage_id)
        if output.is_file():
            try:
                payload = self._validated_eval_payload(output, candidate_path, reference_path, games)
                payload.update({
                    "stage_id": stage_id,
                    "candidate_branch": candidate_branch,
                    "candidate_iteration": candidate_iteration,
                    "reference_id": reference_id,
                })
                self.state["evaluations"] = [
                    row for row in self.state.get("evaluations", []) if row.get("stage_id") != stage_id
                ]
                self.state["evaluations"].append(payload)
                self.state["stages"][stage_id] = {
                    **(existing or {}),
                    "status": "COMPLETED",
                    "recovered": True,
                    "ended_at": base.now_iso(),
                    "candidate": candidate_branch,
                    "candidate_iteration": candidate_iteration,
                    "reference": reference_id,
                    "games": games,
                }
                (self.publish / "evaluations").mkdir(parents=True, exist_ok=True)
                shutil.copy2(output, self.publish / "evaluations" / output.name)
                self._save_state()
                self.event("evaluation_recovered", stage_id=stage_id, score=payload.get("candidate_score_rate"))
                return True
            except Exception as exc:
                quarantine = self.root / "quarantine" / f"eval-{stage_id}-{_stamp()}-{os.getpid()}.json"
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), str(quarantine))
                self.event(
                    "evaluation_output_quarantined",
                    stage_id=stage_id,
                    reason=str(exc),
                    quarantine=str(quarantine.relative_to(self.repo)),
                )
        return super().run_evaluation(
            stage_id,
            candidate_branch,
            candidate_iteration,
            reference_id,
            reference_path,
            games,
            critical=critical,
        )


def main() -> int:
    args = base.parse_args()
    experiment = HardenedExperiment(args)
    try:
        experiment.run()
        return 0
    except BaseException as exc:
        experiment.state["status"] = "FAILED"
        experiment.state["ended_at"] = base.now_iso()
        experiment.state["fatal_error"] = f"{type(exc).__name__}: {exc}"
        experiment._save_state()
        experiment.event("experiment_failed", error=experiment.state["fatal_error"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
