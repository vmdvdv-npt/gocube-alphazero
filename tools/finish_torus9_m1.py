#!/usr/bin/env python3
"""Finish a previously completed Torus 9×9 M0→M1 boundary.

This is an explicit recovery tool for a run whose 64 self-play games and
generation-1 replay were durably written but whose training/checkpoint step
failed.  It never runs self-play, appends replay, or starts another
iteration.  The resulting manifest carries an explicit M1 resume contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from gocube_golden.provenance import CodeIdentity, derive_seed, file_sha256
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9OwnershipScoreTrainer,
    Torus9RollingReplay,
    torus9_load_checkpoint,
    write_json,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_MODEL_INIT_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_CURRENT_TRAINING_MASTER_SEED,
    TORUS9_WORKERS,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)
from tools.torus9_golden_learning import (
    BASE_COMMIT,
    _checkpoint,
    _contract,
    _execution_profile,
)


ROOT = Path(__file__).resolve().parents[1]
STOP_REASON = "manual checkpoint for baseline telemetry review"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _gpu_summary(path: Path) -> dict[str, object]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"GPU telemetry is empty: {path}")
    utilization = [float(row["gpu_util_pct"]) for row in rows]
    vram = [float(row["vram_used_mb"]) for row in rows]
    return {
        "samples": len(rows),
        "sample_path": str(path),
        "gpu_utilization_avg_pct": sum(utilization) / len(utilization),
        "gpu_utilization_peak_pct": max(utilization),
        "gpu_vram_peak_mb": max(vram),
        "sample_start_unix": rows[0].get("ts"),
        "sample_end_unix": rows[-1].get("ts"),
    }


def finish(run_root: Path, gpu_samples: Path) -> dict[str, object]:
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") or manifest.get("result"):
        raise RuntimeError("Refusing to mutate a run that already has a terminal status")
    if manifest.get("run_id") != run_root.name:
        raise ValueError("Run root/name does not match manifest run_id")
    if manifest.get("base_commit") != BASE_COMMIT:
        raise ValueError("Run base commit is not the required Torus 9×9 starting point")
    if manifest.get("device") != "cuda" or manifest.get("device_locked") is not True:
        raise ValueError("M1 recovery requires the locked CUDA run")
    if manifest.get("profile_id") != TORUS9_CURRENT_PROFILE_ID:
        raise ValueError("M1 recovery requires the current Golden profile")

    forbidden = [
        run_root / "checkpoints" / "M1.pt",
        run_root / "checkpoints" / "M2.pt",
        run_root / "selfplay" / "iter-02-games.jsonl",
        run_root / "replay" / "iter-02-fresh.jsonl",
        run_root / "replay" / "rolling-after-02.jsonl",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("M1 recovery refuses a run with M1/M2 or iteration-2 artifacts")

    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    if profile_fp != manifest.get("profile_fingerprint"):
        raise ValueError("Current Golden profile fingerprint drifted from the run manifest")
    games_path = run_root / "selfplay" / "iter-01-games.jsonl"
    replay_path = run_root / "replay" / "rolling-after-01.jsonl"
    fresh_path = run_root / "replay" / "iter-01-fresh.jsonl"
    m0_path = run_root / "checkpoints" / "M0.pt"
    for path in (games_path, fresh_path, replay_path, m0_path):
        if not path.exists():
            raise FileNotFoundError(path)

    games = _read_jsonl(games_path)
    replay_rows = _read_jsonl(replay_path)
    fresh_rows = _read_jsonl(fresh_path)
    if len(games) != 64 or len({str(row["game_id"]) for row in games}) != 64:
        raise ValueError("Persisted self-play does not contain exactly 64 unique games")
    if any(row.get("technical_termination") is not None for row in games):
        raise ValueError("M1 baseline contains a technical outcome")
    if len(replay_rows) != len(fresh_rows) or file_sha256(replay_path) != file_sha256(fresh_path):
        raise ValueError("Fresh and rolling generation-1 replay diverged")
    total_moves = sum(len(row["final_action_trace"]) for row in games)
    total_rows = sum(int(row["nn_evaluations"]) for row in games)
    if len(replay_rows) != total_moves or total_rows <= 0:
        raise ValueError("Persisted self-play/replay row counts are inconsistent")
    expected_game_seeds = {
        derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, run_root.name, str(row["game_id"]), "game")
        for row in games
    }
    if {int(row["game_seed"]) for row in games} != expected_game_seeds:
        raise ValueError("Persisted self-play game seeds do not match the Golden derivation")
    replay_hash_before = file_sha256(replay_path)

    model = Torus9CurrentGraphNet().to("cuda")
    trainer = Torus9OwnershipScoreTrainer(
        model,
        score_loss_enabled=True,
        learning_rate=float(profile["training"]["learning_rate"]),  # type: ignore[index]
        weight_decay=float(profile["training"]["weight_decay"]),  # type: ignore[index]
        optimizer_steps_per_iteration=int(profile["training"]["optimizer_steps_per_iteration"]),  # type: ignore[index]
    )
    m0_metadata = torus9_load_checkpoint(
        m0_path,
        model=model,
        expected={
            "checkpoint_label": "M0",
            "run_id": run_root.name,
            "profile_id": TORUS9_CURRENT_PROFILE_ID,
            "profile_fingerprint": profile_fp,
            "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
            "optimizer_updates": 0,
            "valid_replay_positions": 0,
        },
        device="cuda",
    )
    replay = Torus9RollingReplay(
        generations=int(profile["replay"]["generations"]),  # type: ignore[index]
        maximum_positions=int(profile["replay"]["cap"]),  # type: ignore[index]
    )
    replay_metrics = replay.append_generation(1, replay_rows)
    if list(replay.rows) != replay_rows:
        raise ValueError("Reconstructed replay state differs from persisted generation-1 replay")

    train_started = time.perf_counter()
    train_metrics = trainer.train_fixed_budget(
        list(replay.rows),
        seed=derive_seed(TORUS9_CURRENT_TRAINING_MASTER_SEED, run_root.name, "training", 1),
    )
    training_wall = time.perf_counter() - train_started
    if train_metrics.get("optimizer_steps") != 80 or train_metrics.get("samples_consumed") != 5120:
        raise AssertionError("Current Torus 9×9 fixed training budget drifted during M1 recovery")
    if train_metrics.get("batch_sizes") != [64] * 80:
        raise AssertionError("Current Torus 9×9 M1 recovery produced a partial batch")
    if train_metrics.get("ownership_loss_enabled") is not True or train_metrics.get("score_loss_enabled") is not True:
        raise AssertionError("Current Torus 9×9 auxiliary losses are not enabled")

    original_code = CodeIdentity(
        git_commit_sha=str(manifest["source_commit"]),
        git_tree_sha=str(manifest["source_tree"]),
        working_tree_clean=bool(manifest["source_worktree_clean"]),
    )
    checkpoint_path = run_root / "checkpoints" / "M1.pt"
    contract = _contract(profile)
    checkpoint_metadata = _checkpoint(
        checkpoint_path,
        model,
        trainer.optimizer,
        run_id=run_root.name,
        label="M1",
        parent="M0",
        code=original_code,
        profile_fp=profile_fp,
        contract=contract,
        completed_games=64,
        replay_positions=len(replay.rows),
        optimizer_updates=int(trainer.update_count),
        samples_consumed=int(trainer.samples_consumed),
        device="cuda",
    )
    write_json(run_root / "training" / "iter-01.json", train_metrics)

    gpu = _gpu_summary(gpu_samples)
    selfplay_wall = max(0.0, games_path.stat().st_mtime - m0_path.stat().st_mtime)
    batch_rows = 1
    execution = _execution_profile(profile, "baseline")
    row = {
        "iteration": 1,
        "label": "M1",
        "device": "cuda",
        "games": 64,
        "valid_games": 64,
        "technical_games": 0,
        "fresh_positions": len(fresh_rows),
        "total_moves": total_moves,
        "replay": replay_metrics,
        "average_ply": total_moves / 64.0,
        "median_game_length": statistics.median(len(row["final_action_trace"]) for row in games),
        "p95_game_length": sorted(len(row["final_action_trace"]) for row in games)[min(63, max(0, int(0.95 * 64) - 1))],
        "min_ply": min(len(row["final_action_trace"]) for row in games),
        "max_ply": max(len(row["final_action_trace"]) for row in games),
        "self_play_wall_time_sec": selfplay_wall,
        "wall_time_source": "M0 checkpoint mtime to iter-01 self-play artifact mtime",
        "games_per_sec": 64.0 / selfplay_wall if selfplay_wall else 0.0,
        "games_per_hour": 64.0 * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
        "moves_per_sec": total_moves / selfplay_wall if selfplay_wall else 0.0,
        "positions_per_hour": len(fresh_rows) * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
        "execution": execution,
        "inference": {
            "inference_calls": total_rows,
            "total_inference_rows": total_rows,
            "inference_rows_per_sec": total_rows / selfplay_wall if selfplay_wall else 0.0,
            "mean_inference_batch_rows": batch_rows,
            "median_inference_batch_rows": batch_rows,
            "p95_inference_batch_rows": batch_rows,
            "max_inference_batch_rows": batch_rows,
            "batch_rows_semantics": "coalescing OFF, one NN forward per request",
            "max_active_mcts_lanes": TORUS9_WORKERS,
            "max_active_inference_requests": TORUS9_WORKERS,
            "telemetry_source": "persisted self-play plus process-tree diagnostic",
            **gpu,
        },
        "training": train_metrics,
        "training_wall_time_sec": training_wall,
        "checkpoint": {
            "path": str(checkpoint_path),
            "metadata_path": str(checkpoint_path.with_suffix(".metadata.json")),
            "model_hash": checkpoint_metadata["model_hash"],
            "artifact_sha256": file_sha256(checkpoint_path),
            "optimizer_state_present": True,
        },
        "technical_outcomes_excluded": True,
        "replay_hash_before_training": replay_hash_before,
        "replay_hash_after_training": file_sha256(replay_path),
        "replay_changed_after_generation_1": replay_hash_before != file_sha256(replay_path),
    }
    if row["replay_changed_after_generation_1"] is not False:
        raise AssertionError("M1 recovery changed the persisted replay")
    write_json(run_root / "iter-01-summary.json", row)

    manifest.update({
        "status": "STOPPED_AFTER_M1",
        "result": "PAUSED_FOR_BASELINE_TELEMETRY_REVIEW",
        "stop_reason": STOP_REASON,
        "stopped_after_transition": "M0→M1",
        "next_transition_started": False,
        "resumable_from": "M1",
        "m1": {
            "checkpoint": str(checkpoint_path),
            "metadata": str(checkpoint_path.with_suffix(".metadata.json")),
            "optimizer_state_present": True,
            "optimizer_updates": int(trainer.update_count),
            "samples_consumed": int(trainer.samples_consumed),
            "completed_games": 64,
            "technical_games": 0,
            "replay_state": str(replay_path),
            "replay_positions": len(replay.rows),
            "replay_sha256": file_sha256(replay_path),
            "iteration_telemetry": str(run_root / "iter-01-summary.json"),
        },
        "resume_contract": {
            "from_checkpoint": str(checkpoint_path),
            "from_replay": str(replay_path),
            "from_iteration": "M1",
            "next_transition": "M1→M2",
            "next_iteration": 2,
            "automatic_resume": False,
            "requires_explicit_resume": True,
            "device": "cuda",
            "device_locked": True,
        },
        "replay_immutable_after_m1": True,
        "iteration_telemetry": str(run_root / "iter-01-summary.json"),
        "m1_recovery": {
            "selfplay_reused": True,
            "replay_reused_without_append": True,
            "training_metrics_schema_fix": "Torus9OwnershipScoreTrainer now reports ownership_loss_enabled",
        },
    })
    write_json(manifest_path, manifest)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    args = parser.parse_args()
    row = finish(args.run_root.resolve(), args.gpu_samples.resolve())
    print(json.dumps({
        "status": "STOPPED_AFTER_M1",
        "checkpoint": row["checkpoint"]["path"],
        "games": row["games"],
        "total_moves": row["total_moves"],
        "next_transition_started": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
