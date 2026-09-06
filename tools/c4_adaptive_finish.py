#!/usr/bin/env python3
"""Adaptive sequential finisher for the active Cube 4 hyperparameter experiment.

The first 24-game common-parent Arenas are treated as screening only.  This
finisher reuses all completed work, identifies a provisional leader, then tries
to falsify it with direct same-depth matches against a conservative shortlist.
Matches are played in independent seed blocks and may stop early only on strong
sequential evidence.  If the provisional leader cannot be confirmed, execution
falls back to the original hardened Stage 2/3 plan rather than declaring a
winner from noisy screening scores.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path.cwd().resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HARDENED_PATH = Path(__file__).resolve().with_name("c4_overnight_hardened.py")
_spec = importlib.util.spec_from_file_location("c4_overnight_hardened_base", HARDENED_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load hardened overnight runner: {HARDENED_PATH}")
hardened = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hardened)
base = hardened.base


EARLY_Z = 2.5758293035489004  # 99% Wilson interval for repeated interim looks.
FINAL_Z = 1.959963984540054   # 95% Wilson interval at the configured cap.
SCREEN_TIE_DELTA = 0.10
DEFAULT_SCREEN_BLOCK = 24
DEFAULT_SCREEN_MAX = 72
DEFAULT_FINAL_BLOCK = 32
DEFAULT_FINAL_MAX = 128


def stable_seed(base_seed: int, stage_id: str) -> int:
    digest = hashlib.sha256(stage_id.encode("utf-8")).hexdigest()
    return int(base_seed) + 100_000 + (int(digest[:8], 16) % 900_000)


def wilson(score: float, n: int, z: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    d = 1.0 + z * z / n
    c = (score + z * z / (2.0 * n)) / d
    r = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * n)) / n) / d
    return max(0.0, c - r), min(1.0, c + r)


class AdaptiveExperiment(hardened.HardenedExperiment):
    """Hardened experiment plus leader-first sequential Arena decisions."""

    def _adaptive_state(self) -> dict[str, Any]:
        selection = self.state.setdefault("selection", {})
        adaptive = selection.setdefault(
            "adaptive",
            {
                "schema_version": 1,
                "mode": "leader-first-sequential",
                "screening_games_per_branch": int(self.args.stage1_eval_games),
                "screen_tie_delta": SCREEN_TIE_DELTA,
                "screen_block_games": int(self.args.adaptive_block_games),
                "screen_max_games_per_match": int(self.args.adaptive_max_games),
                "final_block_games": int(self.args.adaptive_final_block_games),
                "final_max_games": int(self.args.adaptive_final_max_games),
                "matches": {},
                "status": "PENDING",
            },
        )
        return adaptive

    def recommendation_text(self) -> str:
        adaptive = self.state.get("selection", {}).get("adaptive", {})
        winner = adaptive.get("winner")
        if winner and adaptive.get("status") == "CONFIRMED":
            final = adaptive.get("final_confirmation", {})
            score = final.get("score")
            ci = final.get("ci95")
            if score is not None and ci:
                return (
                    f"Adaptive winner confirmed: branch {winner} survived direct same-depth screening "
                    f"and the fresh-depth confirmation (score {float(score):.3f}, 95% CI "
                    f"{float(ci[0]):.3f}-{float(ci[1]):.3f})."
                )
            return f"Adaptive winner confirmed: branch {winner}."
        if adaptive.get("status") == "FALLBACK":
            reason = adaptive.get("fallback_reason", "adaptive confirmation was inconclusive")
            return f"Adaptive leader-first test did not settle the winner ({reason}); hardened fixed Stage 2/3 is continuing."
        return super().recommendation_text()

    def _stage1_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for branch in base.BRANCHES:
            score = self.eval_score(branch, 6, f"parent@{base.PARENT_ITERATION}")
            if score is not None:
                scores[branch] = float(score)
        return scores

    def _ensure_stage1(self, parent_path: Path) -> bool:
        """Finish only missing Stage-1 work; completed stages are recovered as-is."""
        complete = True
        for branch, config in base.BRANCHES.items():
            self.fork_branch(branch)
            run_name = self.state["branches"][branch]["run_name"]
            if not self.run_training_stage(
                f"stage1_train_{branch}_i6", branch, run_name, 6, config, critical=False
            ):
                complete = False
                continue
            if self.eval_score(branch, 6, f"parent@{base.PARENT_ITERATION}") is None:
                games = int(self.args.stage1_eval_games)
                estimate = self.observed_eval_seconds_per_game() * games * 1.25
                if not self.can_start(estimate, f"stage1 eval {branch}@6"):
                    complete = False
                    continue
                if not self.run_evaluation(
                    f"stage1_eval_{branch}_i6_vs_parent5",
                    branch,
                    6,
                    f"parent@{base.PARENT_ITERATION}",
                    parent_path,
                    games,
                ):
                    complete = False
        return complete and len(self._stage1_scores()) == len(base.BRANCHES)

    def _shortlist(self, scores: dict[str, float], leader: str) -> list[str]:
        """Keep A, the leader, each axis winner, and both sides of close axes."""
        keep = {"A", leader}
        for _axis, pair in base.AXIS_PAIRS.items():
            left, right = pair
            if left not in scores or right not in scores:
                continue
            winner = left if scores[left] >= scores[right] else right
            keep.add(winner)
            if abs(scores[left] - scores[right]) <= SCREEN_TIE_DELTA:
                keep.update(pair)
        return sorted(keep, key=lambda b: (-scores.get(b, -1.0), b))

    def _store_eval_payload(
        self,
        *,
        stage_id: str,
        candidate_branch: str,
        candidate_iteration: int,
        reference_branch: str,
        reference_iteration: int,
        games: int,
        output: Path,
        payload: dict[str, Any],
        elapsed: float,
        recovered: bool,
    ) -> None:
        reference_id = f"adaptive:{reference_branch}@{reference_iteration}"
        payload = dict(payload)
        payload.update(
            {
                "stage_id": stage_id,
                "candidate_branch": candidate_branch,
                "candidate_iteration": candidate_iteration,
                "reference_id": reference_id,
                "adaptive_reference_branch": reference_branch,
                "adaptive_reference_iteration": reference_iteration,
                "adaptive_block": True,
            }
        )
        self.state["evaluations"] = [
            row for row in self.state.get("evaluations", []) if row.get("stage_id") != stage_id
        ]
        self.state["evaluations"].append(payload)
        self.state["stages"][stage_id] = {
            "status": "COMPLETED",
            "recovered": recovered,
            "ended_at": base.now_iso(),
            "elapsed_seconds": elapsed,
            "candidate": candidate_branch,
            "candidate_iteration": candidate_iteration,
            "reference": reference_id,
            "games": games,
        }
        (self.publish / "evaluations").mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, self.publish / "evaluations" / output.name)
        self._save_state()

    def run_seeded_block(
        self,
        *,
        stage_id: str,
        candidate_branch: str,
        candidate_iteration: int,
        reference_branch: str,
        reference_iteration: int,
        games: int,
    ) -> bool:
        candidate_run = self.state["branches"][candidate_branch]["run_name"]
        reference_run = self.state["branches"][reference_branch]["run_name"]
        candidate_path = self.checkpoint_path(candidate_run, candidate_iteration)
        reference_path = self.checkpoint_path(reference_run, reference_iteration)
        output = self.root / "evaluations" / f"{stage_id}.json"
        log_path = self.logs / f"{stage_id}.log"
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.is_file():
            try:
                payload = self._validated_eval_payload(output, candidate_path, reference_path, games)
                elapsed = float(self.state.get("stages", {}).get(stage_id, {}).get("elapsed_seconds", 0.0))
                self._store_eval_payload(
                    stage_id=stage_id,
                    candidate_branch=candidate_branch,
                    candidate_iteration=candidate_iteration,
                    reference_branch=reference_branch,
                    reference_iteration=reference_iteration,
                    games=games,
                    output=output,
                    payload=payload,
                    elapsed=elapsed,
                    recovered=True,
                )
                self.event("adaptive_evaluation_recovered", stage_id=stage_id)
                return True
            except Exception as exc:
                quarantine = self.root / "quarantine" / f"adaptive-eval-{stage_id}-{int(time.time())}-{os.getpid()}.json"
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), str(quarantine))
                self.event(
                    "adaptive_evaluation_quarantined",
                    stage_id=stage_id,
                    reason=str(exc),
                    quarantine=str(quarantine.relative_to(self.repo)),
                )

        seed = stable_seed(self.args.eval_seed, stage_id)
        command = [
            str(self.python),
            str(self.evaluator),
            "--candidate", str(candidate_path),
            "--reference", str(reference_path),
            "--output", str(output),
            "--candidate-id", f"{candidate_branch}@{candidate_iteration}",
            "--reference-id", f"{reference_branch}@{reference_iteration}",
            "--topology", "cube",
            "--size", "4",
            "--games", str(games),
            "--sims", str(base.ARENA_SIMS),
            "--seed", str(seed),
        ]
        self.state["stages"][stage_id] = {
            "status": "RUNNING",
            "started_at": base.now_iso(),
            "command": command,
            "candidate": candidate_branch,
            "candidate_iteration": candidate_iteration,
            "reference": f"adaptive:{reference_branch}@{reference_iteration}",
            "games": games,
            "seed": seed,
        }
        self._save_state()
        self.event("adaptive_evaluation_started", stage_id=stage_id, games=games, seed=seed)
        return_code, elapsed = self.stream_command(command, log_path)
        if return_code != 0 or not output.is_file():
            self.state["stages"][stage_id].update(
                {"status": "FAILED", "ended_at": base.now_iso(), "exit_code": return_code, "elapsed_seconds": elapsed}
            )
            self.publish_log_tail(stage_id, log_path)
            self._save_state()
            self.event("adaptive_evaluation_failed", stage_id=stage_id, exit_code=return_code)
            return False

        payload = self._validated_eval_payload(output, candidate_path, reference_path, games)
        self.publish_log_tail(stage_id, log_path)
        self._store_eval_payload(
            stage_id=stage_id,
            candidate_branch=candidate_branch,
            candidate_iteration=candidate_iteration,
            reference_branch=reference_branch,
            reference_iteration=reference_iteration,
            games=games,
            output=output,
            payload=payload,
            elapsed=elapsed,
            recovered=False,
        )
        self.event(
            "adaptive_evaluation_completed",
            stage_id=stage_id,
            score=payload.get("candidate_score_rate"),
            elapsed_seconds=round(elapsed, 1),
        )
        return True

    def _aggregate_match(self, prefix: str) -> dict[str, Any]:
        rows = [
            row for row in self.state.get("evaluations", [])
            if row.get("adaptive_block") and str(row.get("stage_id", "")).startswith(prefix)
        ]
        rows.sort(key=lambda row: str(row.get("stage_id")))
        wins = sum(int(row.get("candidate_wins", 0)) for row in rows)
        losses = sum(int(row.get("reference_wins", 0)) for row in rows)
        draws = sum(int(row.get("draws", 0)) for row in rows)
        no_results = sum(int(row.get("no_results", 0)) for row in rows)
        requested = sum(int(row.get("games_requested", 0)) for row in rows)
        effective = wins + losses + draws
        score = (wins + 0.5 * draws) / effective if effective else 0.5
        ci99 = wilson(score, effective, EARLY_Z)
        ci95 = wilson(score, effective, FINAL_Z)
        return {
            "blocks": len(rows),
            "games_requested": requested,
            "games_effective": effective,
            "candidate_wins": wins,
            "reference_wins": losses,
            "draws": draws,
            "no_results": no_results,
            "score": score,
            "ci99": list(ci99),
            "ci95": list(ci95),
        }

    def adaptive_match(
        self,
        *,
        candidate: str,
        reference: str,
        iteration: int,
        block_games: int,
        max_games: int,
        label: str,
    ) -> dict[str, Any]:
        prefix = f"adaptive_{label}_{candidate}_i{iteration}_vs_{reference}_i{iteration}_b"
        key = f"{label}:{candidate}@{iteration}_vs_{reference}@{iteration}"
        adaptive = self._adaptive_state()
        matches = adaptive.setdefault("matches", {})

        aggregate = self._aggregate_match(prefix)
        while aggregate["games_requested"] < max_games:
            # Strong early stopping: 99% Wilson interval excludes 0.5.
            if aggregate["games_effective"] > 0:
                if float(aggregate["ci99"][0]) > 0.5:
                    aggregate["decision"] = "CANDIDATE"
                    aggregate["decision_basis"] = "99% sequential interval above 0.5"
                    break
                if float(aggregate["ci99"][1]) < 0.5:
                    aggregate["decision"] = "REFERENCE"
                    aggregate["decision_basis"] = "99% sequential interval below 0.5"
                    break

            block_index = aggregate["blocks"] + 1
            games = min(block_games, max_games - aggregate["games_requested"])
            estimate = self.observed_eval_seconds_per_game() * games * 1.25
            if not self.can_start(estimate, f"adaptive {candidate}@{iteration} vs {reference}@{iteration} block {block_index}"):
                aggregate["decision"] = "INCONCLUSIVE"
                aggregate["decision_basis"] = "deadline"
                break
            stage_id = f"{prefix}{block_index:02d}"
            if not self.run_seeded_block(
                stage_id=stage_id,
                candidate_branch=candidate,
                candidate_iteration=iteration,
                reference_branch=reference,
                reference_iteration=iteration,
                games=games,
            ):
                aggregate["decision"] = "INCONCLUSIVE"
                aggregate["decision_basis"] = "evaluation failure"
                break
            aggregate = self._aggregate_match(prefix)
            matches[key] = {**aggregate, "candidate": candidate, "reference": reference, "iteration": iteration}
            self._save_state()

        if "decision" not in aggregate:
            if aggregate["games_effective"] > 0 and float(aggregate["ci95"][0]) > 0.5:
                aggregate["decision"] = "CANDIDATE"
                aggregate["decision_basis"] = "95% interval above 0.5 at cap"
            elif aggregate["games_effective"] > 0 and float(aggregate["ci95"][1]) < 0.5:
                aggregate["decision"] = "REFERENCE"
                aggregate["decision_basis"] = "95% interval below 0.5 at cap"
            else:
                aggregate["decision"] = "INCONCLUSIVE"
                aggregate["decision_basis"] = "95% interval overlaps 0.5 at cap"

        matches[key] = {**aggregate, "candidate": candidate, "reference": reference, "iteration": iteration}
        self._save_state()
        self.event(
            "adaptive_match_decided",
            candidate=candidate,
            reference=reference,
            iteration=iteration,
            decision=aggregate["decision"],
            games=aggregate["games_effective"],
            score=round(float(aggregate["score"]), 4),
            ci95=[round(float(v), 4) for v in aggregate["ci95"]],
        )
        return aggregate

    def _fallback_to_hardened_plan(self, reason: str) -> None:
        adaptive = self._adaptive_state()
        adaptive["status"] = "FALLBACK"
        adaptive["fallback_reason"] = reason
        adaptive["fallback_at"] = base.now_iso()
        self._save_state()
        self.event("adaptive_fallback", reason=reason)
        # Run the original state machine. Completed Stage-1/adaptive artifacts are
        # preserved; its own Stage 2/3 stages remain resumable and hardened.
        base.Experiment.run(self)

    def run(self) -> None:
        self._prepare_attempt_state()
        self.validate_environment()
        self.ensure_parent_five()
        parent_path = self.checkpoint_path(self.source_run, base.PARENT_ITERATION)

        if not self._ensure_stage1(parent_path):
            self._fallback_to_hardened_plan("Stage 1 could not be completed for all seven branches")
            return

        scores = self._stage1_scores()
        leader = max(scores, key=lambda branch: (scores[branch], -ord(branch[0])))
        shortlist = self._shortlist(scores, leader)
        adaptive = self._adaptive_state()
        adaptive.update(
            {
                "status": "VALIDATING_SCREEN_LEADER",
                "stage1_scores": scores,
                "screen_leader": leader,
                "shortlist": shortlist,
                "screen_selected_at": base.now_iso(),
            }
        )
        self._save_state()
        self.event("adaptive_shortlist_selected", leader=leader, shortlist=shortlist, scores=scores)

        # Baseline first gives the fastest falsification of a misleading screen leader.
        opponents = [branch for branch in shortlist if branch != leader]
        opponents.sort(key=lambda branch: (0 if branch == "A" else 1, -scores.get(branch, -1.0), branch))
        for opponent in opponents:
            result = self.adaptive_match(
                candidate=leader,
                reference=opponent,
                iteration=6,
                block_games=int(self.args.adaptive_block_games),
                max_games=int(self.args.adaptive_max_games),
                label="screen",
            )
            if result["decision"] != "CANDIDATE":
                self._fallback_to_hardened_plan(
                    f"screen leader {leader} was not decisively better than {opponent}: {result['decision']}"
                )
                return

        adaptive["screen_leader_confirmed"] = True
        adaptive["screen_confirmed_at"] = base.now_iso()
        adaptive["status"] = "FRESH_DEPTH_CONFIRMATION"
        self._save_state()

        if leader == "A":
            adaptive["winner"] = "A"
            adaptive["status"] = "CONFIRMED"
            adaptive["confirmed_at"] = base.now_iso()
            self.state["status"] = "DONE"
            self.state["ended_at"] = base.now_iso()
            self.state["ended_epoch"] = time.time()
            self._save_state()
            self.event("adaptive_experiment_completed", winner="A", reason="baseline survived shortlist")
            return

        # One fresh descendant iteration checks that the result is not peculiar to
        # a single self-play seed/depth. Reuse any @7 checkpoints already produced
        # by the old runner before the switch.
        for branch in ("A", leader):
            run_name = self.state["branches"][branch]["run_name"]
            if not self.run_training_stage(
                f"adaptive_train_{branch}_i7",
                branch,
                run_name,
                7,
                base.BRANCHES[branch],
                critical=False,
            ):
                self._fallback_to_hardened_plan(f"could not train {branch}@7 for adaptive confirmation")
                return

        final = self.adaptive_match(
            candidate=leader,
            reference="A",
            iteration=7,
            block_games=int(self.args.adaptive_final_block_games),
            max_games=int(self.args.adaptive_final_max_games),
            label="fresh",
        )
        adaptive["final_confirmation"] = dict(final)
        if final["decision"] != "CANDIDATE":
            self._save_state()
            self._fallback_to_hardened_plan(
                f"{leader}@7 did not decisively beat A@7 in fresh-depth confirmation: {final['decision']}"
            )
            return

        adaptive["winner"] = leader
        adaptive["status"] = "CONFIRMED"
        adaptive["confirmed_at"] = base.now_iso()
        self.state["status"] = "DONE"
        self.state["ended_at"] = base.now_iso()
        self.state["ended_epoch"] = time.time()
        self._save_state()
        self.event(
            "adaptive_experiment_completed",
            winner=leader,
            final_games=final["games_effective"],
            final_score=round(float(final["score"]), 4),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-run", default=base.SOURCE_RUN_DEFAULT)
    parser.add_argument("--frozen-commit", default=base.FROZEN_TRAINING_COMMIT)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--reserve-minutes", type=float, default=35.0)
    parser.add_argument("--stage1-eval-games", type=int, default=24)
    parser.add_argument("--stage2-eval-games", type=int, default=32)
    parser.add_argument("--direct-eval-games", type=int, default=40)
    parser.add_argument("--final-eval-games", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=20260906)
    parser.add_argument("--adaptive-block-games", type=int, default=DEFAULT_SCREEN_BLOCK)
    parser.add_argument("--adaptive-max-games", type=int, default=DEFAULT_SCREEN_MAX)
    parser.add_argument("--adaptive-final-block-games", type=int, default=DEFAULT_FINAL_BLOCK)
    parser.add_argument("--adaptive-final-max-games", type=int, default=DEFAULT_FINAL_MAX)
    args = parser.parse_args()
    for name in (
        "stage1_eval_games",
        "stage2_eval_games",
        "direct_eval_games",
        "final_eval_games",
        "adaptive_block_games",
        "adaptive_max_games",
        "adaptive_final_block_games",
        "adaptive_final_max_games",
    ):
        value = int(getattr(args, name))
        if value < 2 or value % 2:
            parser.error(f"--{name.replace('_', '-')} must be an even integer >= 2")
    if args.adaptive_max_games % args.adaptive_block_games:
        parser.error("--adaptive-max-games must be divisible by --adaptive-block-games")
    if args.adaptive_final_max_games % args.adaptive_final_block_games:
        parser.error("--adaptive-final-max-games must be divisible by --adaptive-final-block-games")
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    return args


def main() -> int:
    args = parse_args()
    experiment = AdaptiveExperiment(args)
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
