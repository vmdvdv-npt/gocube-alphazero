#!/usr/bin/env python3
"""Benchmark the Stage-2 Torus9 self-play execution path only.

This launcher is intentionally separate from the learning/continuation
runtime. It loads the immutable M17 checkpoint, runs one execution candidate,
and writes JSON telemetry to a caller-selected benchmark namespace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.provenance import capture_code_identity, file_sha256
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9SelfPlaySearchContract,
    run_torus9_selfplay_games,
    torus9_load_checkpoint,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_DIRICHLET_ALPHA,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)

from tools._frozen_continue_torus9_golden_m1_m100 import GpuSampler


DEFAULT_CHECKPOINT = ROOT / "runs" / "torus9-golden-v3-active" / "torus9-golden-v3-20260914-run03" / "checkpoints" / "M17.pt"
DEFAULT_RUN_ID = "torus9-stage2-execution-benchmark-20260915"


def _contract(profile: dict[str, object]) -> Torus9SelfPlaySearchContract:
    settings = profile["self_play"]
    return Torus9SelfPlaySearchContract(
        contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        simulations=int(settings["mcts_simulations"]),  # type: ignore[index]
        cpuct=float(settings["cpuct"]),  # type: ignore[index]
        fpu=float(settings["fpu"]),  # type: ignore[index]
        temperature_until_ply=int(settings["temperature_plies"][1]),  # type: ignore[index]
        temperature_after=float(settings["temperature_after"]),  # type: ignore[index]
        dirichlet_epsilon=float(settings["dirichlet_epsilon"]),  # type: ignore[index]
        dirichlet_alpha=float(settings["dirichlet_alpha"]),  # type: ignore[index]
        watchdog=int(settings["watchdog"]),  # type: ignore[index]
    )


def run_one(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Legion benchmark requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The execution characterization must run on CUDA")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.games <= 0 or args.workers <= 0 or args.active_games_per_worker <= 0:
        raise ValueError("games, workers and active_games_per_worker must be positive")
    if args.batch_cap <= 0 or args.wait_ms < 0:
        raise ValueError("batch_cap must be positive and wait_ms must be non-negative")

    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    model = Torus9CurrentGraphNet().to(device)
    checkpoint_meta = json.loads(checkpoint.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    loaded_meta = torus9_load_checkpoint(
        checkpoint,
        model=model,
        expected={"model_hash": checkpoint_meta["model_hash"]},
        device=device,
    )
    model.eval()
    contract = _contract(profile)
    contract.validate()
    code = capture_code_identity(ROOT)
    game_ids = tuple(f"{args.run_id}-game-{index:04d}" for index in range(args.games))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    gpu_sampler = GpuSampler(output.with_suffix(".gpu.jsonl"), interval_s=1.0)
    telemetry: dict[str, object] = {}
    started = time.perf_counter()
    gpu_sampler.start()
    try:
        records = run_torus9_selfplay_games(
            model,
            checkpoint_path=checkpoint,
            run_id=args.run_id,
            label="M17",
            artifact=file_sha256(checkpoint),
            master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
            profile_fp=profile_fp,
            profile_id=TORUS9_CURRENT_PROFILE_ID,
            game_ids=game_ids,
            workers=args.workers,
            code_identity=code,
            device=device,
            contract=contract,
            coalescing=True,
            inference_batch_cap=args.batch_cap,
            inference_batch_wait_ms=args.wait_ms,
            active_games_per_worker=args.active_games_per_worker,
            total_active_contexts=args.total_active_contexts,
            inference_telemetry=telemetry,
            execution_activity=telemetry,
        )
    finally:
        gpu_summary = gpu_sampler.stop()
    wall = time.perf_counter() - started

    if len(records) != len(game_ids) or {record.game_id for record in records} != set(game_ids):
        raise RuntimeError("benchmark returned an incomplete or duplicate game set")
    for record in records:
        record.validate()
    technical = sum(record.technical_termination is not None for record in records)
    rows = sum(record.nn_evaluations for record in records)
    if rows != int(telemetry.get("total_rows", -1)):
        raise RuntimeError("inference row count does not match record count")
    moves = sum(len(record.final_action_trace) for record in records)
    result = {
        "benchmark_schema": "torus9-selfplay-execution-benchmark-v1",
        "host": "Legion",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_model_hash": loaded_meta["model_hash"],
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "selfplay_contract_id": contract.contract_id,
        "selfplay_contract_fingerprint": contract.fingerprint,
        "komi": 0.5,
        "dirichlet_alpha": TORUS9_CURRENT_DIRICHLET_ALPHA,
        "scientific_semantics_changed": "NO",
        "config": {
            "run_id": args.run_id,
            "games": args.games,
            "workers": args.workers,
            "active_games_per_worker": args.active_games_per_worker,
            "total_active_contexts": args.total_active_contexts,
            "batch_cap": args.batch_cap,
            "wait_ms": args.wait_ms,
            "device": str(device),
            "worker_local_wait_ms": 0.0,
        },
        "metrics": {
            "wall_time_sec": wall,
            "games_per_sec": len(records) / wall if wall else 0.0,
            "moves_per_sec": moves / wall if wall else 0.0,
            "total_moves": moves,
            "total_rows": rows,
            "technical_games": technical,
            "record_ids": list(game_ids),
        },
        "telemetry": telemetry,
        "gpu": gpu_summary,
        "code": {
            "git_commit_sha": code.git_commit_sha,
            "git_tree_sha": code.git_tree_sha,
            "working_tree_clean": code.working_tree_clean,
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "games": len(records),
        "moves_per_sec": result["metrics"]["moves_per_sec"],  # type: ignore[index]
        "games_per_sec": result["metrics"]["games_per_sec"],  # type: ignore[index]
        "mean_batch": telemetry.get("mean_batch_rows"),
        "gpu_avg": gpu_summary.get("gpu_utilization_avg_pct"),
    }, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--active-games-per-worker", type=int, default=4)
    parser.add_argument("--total-active-contexts", type=int, default=None)
    parser.add_argument("--batch-cap", type=int, default=64)
    parser.add_argument("--wait-ms", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    run_one(parser.parse_args())


if __name__ == "__main__":
    main()
