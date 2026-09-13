#!/usr/bin/env python3
"""Run the current Torus 9×9 Golden Standard M0→M8 experiment.

The legacy v2 profile remains loadable through ``load_torus9_profile`` for
reproduction, but this launcher has no legacy-profile selection path. It
resolves the current profile explicitly, runs the performance sweep on the
same 64 self-play games that feed training, and starts every invocation in a
new active namespace with an empty replay.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time

import torch

from gocube_golden.provenance import capture_code_identity, derive_seed, file_sha256
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9OwnershipScoreTrainer,
    Torus9RollingReplay,
    Torus9SelfPlaySearchContract,
    run_torus9_selfplay_games,
    torus9_build_ownership_score_replay_samples,
    torus9_checkpoint_metadata,
    torus9_save_checkpoint,
    validate_torus9_replay_sample,
    write_json,
    write_jsonl,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_BLOCKS,
    TORUS9_CURRENT_DIRICHLET_ALPHA,
    TORUS9_CURRENT_HIDDEN,
    TORUS9_CURRENT_MODEL_INIT_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    TORUS9_CURRENT_TRAINING_MASTER_SEED,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_WORKERS,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "53946d0c84fca5a6f81a387bfd399ea62e34b088"
ACTIVE_NAMESPACE = ROOT / "runs" / "torus9-golden-v3-active"
DEFAULT_RUN_ID = "torus9-golden-v3-20260913-run01"


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _checkpoint(
    path: Path,
    model: Torus9CurrentGraphNet,
    optimizer: torch.optim.Optimizer | None,
    *,
    run_id: str,
    label: str,
    parent: str | None,
    code,
    profile_fp: str,
    contract: Torus9SelfPlaySearchContract,
    completed_games: int,
    replay_positions: int,
    optimizer_updates: int,
    samples_consumed: int,
) -> dict[str, object]:
    metadata = torus9_checkpoint_metadata(
        model=model,
        run_id=run_id,
        label=label,
        parent=parent,
        model_seed=TORUS9_CURRENT_MODEL_INIT_SEED,
        code=code,
        profile_fp=profile_fp,
        completed_games=completed_games,
        replay_positions=replay_positions,
        optimizer_updates=optimizer_updates,
        samples_consumed=samples_consumed,
        ownership_loss_enabled=True,
        score_loss_enabled=True,
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        target_fingerprint=TORUS9_CURRENT_TARGET_FINGERPRINT,
        selfplay_contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        selfplay_contract_fingerprint=contract.fingerprint,
        base_commit=BASE_COMMIT,
    )
    metadata["adam_step"] = optimizer_updates
    metadata["model_init_seed"] = TORUS9_CURRENT_MODEL_INIT_SEED
    metadata["execution_only_parameters"] = {
        "self_play_inference_batch_cap": "not checkpoint semantic",
        "self_play_inference_batch_wait_ms": "not checkpoint semantic",
    }
    return torus9_save_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)


def _compact_samples(samples: list[dict[str, object]], generation: int) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for position, sample in enumerate(samples):
        row = dict(sample)
        row["source_generation"] = int(generation)
        row["replay_row_id"] = f"M{generation}:{row['game_id']}:{row['ply']}:{position}"
        compact.append(row)
    return compact


def _execution_profile(profile: dict[str, object], name: str) -> dict[str, object]:
    sweep = profile["execution_sweep"]
    if name == "baseline":
        return {"id": "baseline", **dict(sweep["baseline"])}  # type: ignore[index]
    for candidate in sweep["candidates"]:  # type: ignore[index]
        if candidate["id"] == name:  # type: ignore[index]
            return dict(candidate)  # type: ignore[arg-type]
    raise ValueError(f"Unknown Torus 9×9 execution sweep candidate: {name}")


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


def _profile_comparison(profile: dict[str, object]) -> dict[str, object]:
    """Return the fail-closed source/profile comparison artifact."""
    network = profile["network"]
    self_play = profile["self_play"]
    training = profile["training"]
    replay = profile["replay"]
    checked = {
        "komi": profile["rules"]["komi"],  # type: ignore[index]
        "legacy_komi_sentinel": profile["rules"]["legacy_komi_sentinel"],  # type: ignore[index]
        "network": f"{network['hidden']}×{network['blocks']}",  # type: ignore[index]
        "ownership": network["ownership"],  # type: ignore[index]
        "score": network["score"],  # type: ignore[index]
        "dirichlet_alpha": self_play["dirichlet_alpha"],  # type: ignore[index]
        "self_play_games": self_play["games_per_iteration"],  # type: ignore[index]
        "self_play_batch_size": self_play["existing_batch_size"],  # type: ignore[index]
        "training_batch_size": training["batch_size"],  # type: ignore[index]
        "optimizer_steps": training["optimizer_steps_per_iteration"],  # type: ignore[index]
        "sample_exposures": training["samples_consumed_per_iteration"],  # type: ignore[index]
        "replay_generations": replay["generations"],  # type: ignore[index]
        "replay_cap": replay["cap"],  # type: ignore[index]
    }
    snapshot_path = ROOT / str(profile["golden_source"]["snapshot_path"])  # type: ignore[index]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_values = {(row[0], row[1]): row[2] for row in snapshot["rows"]}
    expected_snapshot_values = {
        ("Rules", "Topology"): "Torus 9×9, 81 points, wrap X/Y",
        ("Rules", "Scoring"): "Exact graph-area",
        ("Rules", "Superko"): "Positional superko",
        ("Rules", "Suicide"): "Forbidden",
        ("Rules", "Termination"): "Two passes",
        ("Rules", "Komi"): "0.5",
        ("Observation", "Tensor"): "6×81",
        ("Observation", "Channels"): "Own stones; opponent stones; side-to-move color; previous pass; legal mask; komi",
        ("Targets", "Action count / PASS"): "82 actions / PASS=81",
        ("Targets", "Value target"): "WDL, side-to-move perspective",
        ("Targets", "Policy target"): "Root visit distribution before chosen move",
        ("Targets", "Ownership auxiliary"): "ON",
        ("Targets", "Score auxiliary"): "ON",
        ("Targets", "Technical / invalid outcomes"): "Exclude",
        ("Network", "Architecture"): "GoldenGraphNetV2-Torus9",
        ("Network", "Hidden width"): "80",
        ("Network", "Blocks"): "8",
        ("Network", "Input channels"): "6",
        ("Network", "Policy head"): "82 actions",
        ("Network", "WDL head"): "3 classes",
        ("Network", "Ownership head"): "81×3",
        ("Network", "Explicit symmetry augmentation"): "OFF",
        ("Self-play", "Games / iteration"): "64",
        ("Self-play", "MCTS simulations"): "64",
        ("Self-play", "cpuct"): "1.25",
        ("Self-play", "FPU"): "0",
        ("Self-play", "Root noise"): "ON",
        ("Self-play", "Dirichlet epsilon"): "0.25",
        ("Self-play", "Dirichlet alpha"): "0.11",
        ("Self-play", "Temperature"): "1.0 on plies 1–8, then 0",
        ("Self-play", "Fast search"): "OFF",
        ("Self-play", "Resign"): "OFF",
        ("Self-play", "Watchdog"): "500 plies",
        ("Self-play", "Workers"): "16",
        ("Self-play", "Batch size"): "64",
        ("Self-play", "Komi"): "0.5",
        ("Training", "Optimizer"): "Adam",
        ("Training", "Learning rate"): "0.001",
        ("Training", "Weight decay"): "0",
        ("Training", "Train batch size"): "64",
        ("Training", "Optimizer steps / iteration"): "80",
        ("Training", "Sample exposures / iteration"): "5120",
        ("Training", "LR scheduler"): "None",
        ("Training", "Model gating"): "OFF / decoupled",
        ("Replay", "Window"): "Rolling last 3 generations",
        ("Replay", "Cap"): "20,000 positions",
        ("Replay", "Sampling"): "Deterministic / reproducible",
        ("Arena", "Minimum games"): "64",
        ("Arena", "MCTS simulations"): "64",
        ("Arena", "cpuct / FPU"): "1.25 / 0",
        ("Arena", "Noise / temperature / fast / resign"): "OFF / 0 / OFF / OFF",
        ("Arena", "Watchdog"): "1000 plies",
        ("Arena", "Workers / batched"): "16 / ON",
        ("Arena", "arena_batch_size"): "8",
        ("Arena", "inference_batch_wait_ms"): "6",
        ("Arena", "Paired starts / color swap"): "ON where applicable",
        ("Arena", "Technical outcomes"): "Fail-closed / excluded",
        ("Arena", "Gating"): "Decoupled",
        ("Arena", "Mean inference batch rows"): ">=16 target",
        ("Arena", "Komi"): "0.5",
    }
    mismatches = {
        f"{section}/{parameter}": {"expected": expected, "actual": snapshot_values.get((section, parameter))}
        for (section, parameter), expected in expected_snapshot_values.items()
        if snapshot_values.get((section, parameter)) != expected
    }
    if mismatches:
        raise ValueError(f"Golden Standart snapshot mismatch: {mismatches}")
    return {
        "status": "PASS",
        "source": profile["golden_source"],
        "profile_id": profile["profile_id"],
        "profile_fingerprint": current_torus9_profile_fingerprint(profile),
        "mismatches": mismatches,
        "checked": checked,
        "golden_standard_untouched": True,
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return float(ordered[index])


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _runtime_telemetry(*, wall_time: float, cpu_before: float, inference: dict[str, object]) -> dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = max(0.0, _process_cpu_seconds() - cpu_before)
    cpu_count = max(1, os.cpu_count() or 1)
    batch_rows = [float(value) for value in inference.get("batch_rows", [])]
    rows = int(inference.get("total_rows", 0))
    return {
        "inference_calls": int(inference.get("forward_calls", 0)),
        "total_inference_rows": rows,
        "inference_rows_per_sec": rows / wall_time if wall_time else 0.0,
        "mean_inference_batch_rows": float(inference.get("mean_batch_rows", 0.0)),
        "median_inference_batch_rows": statistics.median(batch_rows) if batch_rows else 0.0,
        "p95_inference_batch_rows": _p95(batch_rows),
        "max_inference_batch_rows": int(inference.get("max_batch_rows", 0)),
        "cpu_utilization_pct": 100.0 * cpu_seconds / (wall_time * cpu_count) if wall_time else 0.0,
        "gpu_utilization_pct": None,
        "gpu_vram_peak_mb": 0.0,
        "ram_peak_mb": float(usage.ru_maxrss) / 1024.0,
    }


def _iteration_record(
    *,
    iteration: int,
    execution: dict[str, object],
    records: tuple,
    fresh_positions: int,
    replay_metrics: dict[str, object],
    train_metrics: dict[str, object],
    selfplay_wall: float,
    cpu_before: float,
    inference_telemetry: dict[str, object],
    checkpoint: dict[str, object],
) -> dict[str, object]:
    plies = [float(len(record.final_action_trace)) for record in records]
    technical = sum(record.technical_termination is not None for record in records)
    total_moves = int(sum(plies))
    runtime = _runtime_telemetry(wall_time=selfplay_wall, cpu_before=cpu_before, inference=inference_telemetry)
    inference_telemetry.update(runtime)
    return {
        "iteration": iteration,
        "label": f"M{iteration}",
        "games": len(records),
        "valid_games": len(records) - technical,
        "technical_games": technical,
        "fresh_positions": fresh_positions,
        "total_moves": total_moves,
        "replay": replay_metrics,
        "average_ply": sum(plies) / len(plies) if plies else None,
        "median_game_length": statistics.median(plies) if plies else None,
        "p95_game_length": _p95(plies),
        "min_ply": min(plies) if plies else None,
        "max_ply": max(plies) if plies else None,
        "self_play_wall_time_sec": selfplay_wall,
        "games_per_sec": len(records) / selfplay_wall if selfplay_wall else 0.0,
        "games_per_hour": len(records) * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
        "moves_per_sec": total_moves / selfplay_wall if selfplay_wall else 0.0,
        "positions_per_hour": fresh_positions * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
        "execution": execution,
        "inference": inference_telemetry,
        "training": train_metrics,
        "checkpoint": checkpoint,
        "technical_outcomes_excluded": True,
    }


def _select_winner(rows: list[dict[str, object]]) -> dict[str, object]:
    baseline_rows = [row for row in rows if row["iteration"] == 2 and row["execution"]["id"] == "baseline"]  # type: ignore[index]
    candidate_rows = [row for row in rows if row["iteration"] in {3, 4, 5, 6, 7} and row["execution"]["coalescing"] is True]  # type: ignore[index]
    if len(baseline_rows) != 1:
        return {"status": "INCONCLUSIVE", "reason": "M1→M2 baseline is missing or duplicated", "candidates": {}}
    if not candidate_rows:
        return {"status": "INCONCLUSIVE", "reason": "no coalesced candidate observations", "candidates": {}}
    if any(row["technical_games"] or row["games"] != 64 for row in [baseline_rows[0], *candidate_rows]):
        return {"status": "INCONCLUSIVE", "reason": "baseline/candidate had missing games or technical outcomes", "candidates": {}}

    baseline = baseline_rows[0]
    candidates = {
        str(row["execution"]["id"]): {  # type: ignore[index]
            "iteration": row["iteration"],
            "batch_cap": row["execution"]["batch_cap"],  # type: ignore[index]
            "wait_ms": row["execution"]["wait_ms"],  # type: ignore[index]
            "moves_per_sec": row["moves_per_sec"],
            "inference_rows_per_sec": row["inference"]["inference_rows_per_sec"],  # type: ignore[index]
            "inference_calls": row["inference"]["inference_calls"],  # type: ignore[index]
            "mean_inference_batch_rows": row["inference"]["mean_inference_batch_rows"],  # type: ignore[index]
            "median_inference_batch_rows": row["inference"]["median_inference_batch_rows"],  # type: ignore[index]
            "p95_inference_batch_rows": row["inference"]["p95_inference_batch_rows"],  # type: ignore[index]
            "games_per_hour": row["games_per_hour"],
        }
        for row in candidate_rows
    }
    # Normalize on the real work performed (moves), then use inference rows and
    # batching as tie-breaks.  Within a 5% primary-metric band, choose the
    # simpler/lower-wait setting as required by the experiment protocol.
    ranked_by_work = sorted(candidates, key=lambda candidate: float(candidates[candidate]["moves_per_sec"]), reverse=True)
    fastest = float(candidates[ranked_by_work[0]]["moves_per_sec"])
    equivalent = [
        candidate for candidate in ranked_by_work
        if float(candidates[candidate]["moves_per_sec"]) >= fastest * 0.95
    ]
    winner = min(
        equivalent,
        key=lambda candidate: (
            float(candidates[candidate]["wait_ms"]),
            int(candidates[candidate]["batch_cap"]),
            -float(candidates[candidate]["inference_rows_per_sec"]),
        ),
    )
    runner_up = next((candidate for candidate in ranked_by_work if candidate != winner), None)
    winner_work = float(candidates[winner]["moves_per_sec"])
    runner_up_work = float(candidates[runner_up]["moves_per_sec"]) if runner_up else 0.0
    margin = (winner_work / runner_up_work - 1.0) if runner_up_work > 0.0 else float("inf")
    return {
        "status": "CONFIRMED",
        "winner": winner,
        "winner_candidate": winner,
        "runner_up": runner_up,
        "winner_margin_over_runner_up": margin,
        "candidates": candidates,
        "baseline_m1_m2": {
            "moves_per_sec": baseline["moves_per_sec"],
            "inference_rows_per_sec": baseline["inference"]["inference_rows_per_sec"],  # type: ignore[index]
            "games_per_hour": baseline["games_per_hour"],
        },
        "winner_speedup_vs_m1_m2_moves_per_sec": winner_work / float(baseline["moves_per_sec"]) - 1.0 if baseline["moves_per_sec"] else None,
        "criterion": "M1 warm-up excluded; select M2-M7 by moves/sec, use inference rows/sec and batch telemetry as tie-breaks, prefer lower wait/cap within 5%",
    }


def _confirmation(selection: dict[str, object], m8: dict[str, object]) -> dict[str, object]:
    if selection.get("status") != "CONFIRMED":
        return {"status": "INCONCLUSIVE", "reason": "winner selection was inconclusive"}
    winner = str(selection["winner"])
    winner_stats = selection["candidates"][winner]  # type: ignore[index]
    baseline = float(winner_stats["median_games_per_hour"])  # type: ignore[index]
    observed = float(m8["games_per_hour"])
    ratio = observed / baseline if baseline else 0.0
    confirmed = 0.85 <= ratio <= 1.15 and m8["execution"]["id"] == winner  # type: ignore[index]
    return {
        "status": "CONFIRMED" if confirmed else "INCONCLUSIVE",
        "winner": winner,
        "winner_m1_m7_median_games_per_hour": baseline,
        "m8_games_per_hour": observed,
        "m8_to_winner_median_ratio": ratio,
        "criterion": "M8 winner throughput is within ±15% of M1-M7 winner median",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    profile = load_torus9_current_profile()
    if args.workers != TORUS9_WORKERS:
        raise ValueError("Current Torus 9×9 Golden Standard is fixed to 16 workers")
    root = ACTIVE_NAMESPACE / args.run_id
    if root.exists():
        raise FileExistsError(f"Active Torus 9×9 run namespace already exists; refusing automatic resume: {root}")
    root.mkdir(parents=True)
    for name in ("checkpoints", "selfplay", "replay", "training"):
        (root / name).mkdir()

    code = capture_code_identity(ROOT)
    profile_fp = current_torus9_profile_fingerprint(profile)
    comparison = _profile_comparison(profile)
    write_json(root / "resolved-config.json", profile)
    write_json(root / "golden-profile-comparison.json", comparison)
    write_json(root / "replay" / "initial.json", {"positions": 0, "generations": [], "automatic_resume": False})

    seed_everything(TORUS9_CURRENT_MODEL_INIT_SEED)
    model = Torus9CurrentGraphNet().to(args.device)
    contract = _contract(profile)
    trainer = Torus9OwnershipScoreTrainer(
        model,
        score_loss_enabled=True,
        learning_rate=float(profile["training"]["learning_rate"]),  # type: ignore[index]
        weight_decay=float(profile["training"]["weight_decay"]),  # type: ignore[index]
        optimizer_steps_per_iteration=int(profile["training"]["optimizer_steps_per_iteration"]),  # type: ignore[index]
    )
    replay = Torus9RollingReplay(
        generations=int(profile["replay"]["generations"]),  # type: ignore[index]
        maximum_positions=int(profile["replay"]["cap"]),  # type: ignore[index]
    )
    m0_path = root / "checkpoints" / "M0.pt"
    m0_metadata = _checkpoint(
        m0_path,
        model,
        None,
        run_id=args.run_id,
        label="M0",
        parent=None,
        code=code,
        profile_fp=profile_fp,
        contract=contract,
        completed_games=0,
        replay_positions=0,
        optimizer_updates=0,
        samples_consumed=0,
    )
    manifest = {
        "manifest_schema": "torus9-golden-current-v3-run-v1",
        "run_id": args.run_id,
        "active_namespace": str(root),
        "base_commit": BASE_COMMIT,
        "source_commit": code.git_commit_sha,
        "source_tree": code.git_tree_sha,
        "source_worktree_clean": code.working_tree_clean,
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "golden_source": profile["golden_source"],
        "golden_profile_comparison": comparison,
        "m0": {
            "checkpoint": str(m0_path),
            "model_hash": m0_metadata["model_hash"],
            "model_init_seed": TORUS9_CURRENT_MODEL_INIT_SEED,
            "fresh_initialization": True,
            "replay_positions": 0,
            "first_post_training_checkpoint": "M1",
        },
        "automatic_resume": False,
        "golden_standard_untouched": True,
    }
    write_json(root / "manifest.json", manifest)

    schedule = list(profile["execution_sweep"]["iteration_schedule"])  # type: ignore[index]
    if len(schedule) != 7:
        raise ValueError("Current Torus 9×9 sweep must define exactly seven M1-M7 entries")
    iteration_rows: list[dict[str, object]] = []
    for iteration in range(1, 8):
        execution = _execution_profile(profile, str(schedule[iteration - 1]))
        previous_checkpoint = m0_path if iteration == 1 else root / "checkpoints" / f"M{iteration - 1}.pt"
        game_ids = [f"{args.run_id}-iter-{iteration:02d}-game-{index:04d}" for index in range(64)]
        telemetry: dict[str, object] = {}
        cpu_before = _process_cpu_seconds()
        started = time.perf_counter()
        records = run_torus9_selfplay_games(
            model,
            checkpoint_path=previous_checkpoint,
            run_id=args.run_id,
            label=f"M{iteration - 1}",
            artifact=file_sha256(previous_checkpoint),
            master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
            profile_fp=profile_fp,
            profile_id=TORUS9_CURRENT_PROFILE_ID,
            game_ids=game_ids,
            workers=args.workers,
            code_identity=code,
            device=args.device,
            contract=contract,
            coalescing=bool(execution["coalescing"]),
            inference_batch_cap=int(execution["batch_cap"]),
            inference_batch_wait_ms=float(execution["wait_ms"]),
            inference_telemetry=telemetry,
        )
        selfplay_wall = time.perf_counter() - started
        if len(records) != 64 or {record.game_id for record in records} != set(game_ids):
            raise AssertionError(f"M{iteration} did not return exactly the 64 requested game IDs")
        write_jsonl(root / "selfplay" / f"iter-{iteration:02d}-games.jsonl", [record.to_dict() for record in records])
        if any(record.game_seed != derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, args.run_id, record.game_id, "game") for record in records):
            raise AssertionError("Torus 9×9 game seed invariance drifted")
        if sum(record.nn_evaluations for record in records) != int(telemetry.get("total_rows", -1)):
            raise AssertionError("Torus 9×9 inference coordinator lost or duplicated rows")
        fresh: list[dict[str, object]] = []
        for record in records:
            record.validate()
            if record.technical_termination is not None:
                continue
            rows = list(torus9_build_ownership_score_replay_samples(record))
            for row in rows:
                validate_torus9_replay_sample(row, expected_target_fingerprint=TORUS9_CURRENT_TARGET_FINGERPRINT)
            fresh.extend(rows)
        if not fresh:
            raise RuntimeError(f"M{iteration} produced no formal replay positions")
        stamped = _compact_samples(fresh, iteration)
        write_jsonl(root / "replay" / f"iter-{iteration:02d}-fresh.jsonl", stamped)
        replay_metrics = replay.append_generation(iteration, stamped)
        write_jsonl(root / "replay" / f"rolling-after-{iteration:02d}.jsonl", list(replay.rows))
        train_metrics = trainer.train_fixed_budget(
            list(replay.rows),
            seed=derive_seed(TORUS9_CURRENT_TRAINING_MASTER_SEED, args.run_id, "training", iteration),
        )
        if train_metrics["optimizer_steps"] != 80 or train_metrics["samples_consumed"] != 5120 or train_metrics["batch_sizes"] != [64] * 80:
            raise AssertionError("Current Torus 9×9 fixed training budget drifted")
        if train_metrics["ownership_loss_enabled"] is not True or train_metrics["score_loss_enabled"] is not True:
            raise AssertionError("Current Torus 9×9 auxiliary losses are not enabled")
        if any(not math.isfinite(float(train_metrics[key])) for key in ("mean_policy_loss", "mean_value_loss", "mean_ownership_loss", "mean_score_loss_normalized", "mean_total_loss")):
            raise AssertionError("Current Torus 9×9 training loss is non-finite")
        checkpoint_path = root / "checkpoints" / f"M{iteration}.pt"
        checkpoint_metadata = _checkpoint(
            checkpoint_path,
            model,
            trainer.optimizer,
            run_id=args.run_id,
            label=f"M{iteration}",
            parent=f"M{iteration - 1}",
            code=code,
            profile_fp=profile_fp,
            contract=contract,
            completed_games=iteration * 64,
            replay_positions=len(replay.rows),
            optimizer_updates=int(trainer.update_count),
            samples_consumed=int(trainer.samples_consumed),
        )
        row = _iteration_record(
            iteration=iteration,
            execution=execution,
            records=records,
            fresh_positions=len(fresh),
            replay_metrics=replay_metrics,
            train_metrics=train_metrics,
            selfplay_wall=selfplay_wall,
            cpu_before=cpu_before,
            inference_telemetry=telemetry,
            checkpoint={"path": str(checkpoint_path), "model_hash": checkpoint_metadata["model_hash"], "artifact_sha256": file_sha256(checkpoint_path)},
        )
        iteration_rows.append(row)
        write_json(root / "training" / f"iter-{iteration:02d}.json", train_metrics)
        write_json(root / f"iter-{iteration:02d}-summary.json", row)

    selection = _select_winner(iteration_rows)
    write_json(root / "sweep-selection-m1-m7.json", selection)
    if selection.get("status") != "CONFIRMED":
        report = {
            "report_schema": "torus9-golden-current-v3-learning-v1",
            "run_id": args.run_id,
            "base_commit": BASE_COMMIT,
            "source_commit": code.git_commit_sha,
            "profile_id": TORUS9_CURRENT_PROFILE_ID,
            "profile_fingerprint": profile_fp,
            "golden_standard_untouched": True,
            "m0": manifest["m0"],
            "iterations": iteration_rows,
            "sweep_selection_m1_m7": selection,
            "m8_confirmation": {"status": "NOT_RUN", "reason": "M1-M7 sweep was inconclusive"},
            "result": "INCONCLUSIVE",
            "table_update": {"status": "NOT_ATTEMPTED", "reason": "INCONCLUSIVE sweep; Golden Standart remains unchanged"},
        }
        write_json(root / "final-report.json", report)
        write_json(ROOT / "docs" / "TORUS9_GOLDEN_CURRENT_V3_LEARNING_20260913.json", report)
        manifest["status"] = "INCONCLUSIVE"
        manifest["result"] = "INCONCLUSIVE"
        manifest["sweep_selection_m1_m7"] = selection
        write_json(root / "manifest.json", manifest)
        return report

    execution = _execution_profile(profile, str(selection["winner"]))
    previous_checkpoint = root / "checkpoints" / "M7.pt"
    game_ids = [f"{args.run_id}-iter-08-game-{index:04d}" for index in range(64)]
    telemetry: dict[str, object] = {}
    cpu_before = _process_cpu_seconds()
    started = time.perf_counter()
    records = run_torus9_selfplay_games(
        model,
        checkpoint_path=previous_checkpoint,
        run_id=args.run_id,
        label="M7",
        artifact=file_sha256(previous_checkpoint),
        master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
        profile_fp=profile_fp,
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        game_ids=game_ids,
        workers=args.workers,
        code_identity=code,
        device=args.device,
        contract=contract,
        coalescing=True,
        inference_batch_cap=int(execution["batch_cap"]),
        inference_batch_wait_ms=float(execution["wait_ms"]),
        inference_telemetry=telemetry,
    )
    selfplay_wall = time.perf_counter() - started
    if len(records) != 64 or {record.game_id for record in records} != set(game_ids):
        raise AssertionError("M8 did not return exactly the 64 requested game IDs")
    write_jsonl(root / "selfplay" / "iter-08-games.jsonl", [record.to_dict() for record in records])
    if any(record.game_seed != derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, args.run_id, record.game_id, "game") for record in records):
        raise AssertionError("Torus 9×9 M8 game seed invariance drifted")
    if sum(record.nn_evaluations for record in records) != int(telemetry.get("total_rows", -1)):
        raise AssertionError("Torus 9×9 M8 inference coordinator lost or duplicated rows")
    fresh = []
    for record in records:
        record.validate()
        if record.technical_termination is not None:
            continue
        rows = list(torus9_build_ownership_score_replay_samples(record))
        for row in rows:
            validate_torus9_replay_sample(row, expected_target_fingerprint=TORUS9_CURRENT_TARGET_FINGERPRINT)
        fresh.extend(rows)
    if not fresh:
        raise RuntimeError("M8 produced no formal replay positions")
    stamped = _compact_samples(fresh, 8)
    write_jsonl(root / "replay" / "iter-08-fresh.jsonl", stamped)
    replay_metrics = replay.append_generation(8, stamped)
    write_jsonl(root / "replay" / "rolling-after-08.jsonl", list(replay.rows))
    train_metrics = trainer.train_fixed_budget(
        list(replay.rows),
        seed=derive_seed(TORUS9_CURRENT_TRAINING_MASTER_SEED, args.run_id, "training", 8),
    )
    if train_metrics["optimizer_steps"] != 80 or train_metrics["samples_consumed"] != 5120 or train_metrics["batch_sizes"] != [64] * 80:
        raise AssertionError("Current Torus 9×9 M8 training budget drifted")
    if train_metrics["ownership_loss_enabled"] is not True or train_metrics["score_loss_enabled"] is not True:
        raise AssertionError("Current Torus 9×9 M8 auxiliary losses are not enabled")
    if any(not math.isfinite(float(train_metrics[key])) for key in ("mean_policy_loss", "mean_value_loss", "mean_ownership_loss", "mean_score_loss_normalized", "mean_total_loss")):
        raise AssertionError("Current Torus 9×9 M8 training loss is non-finite")
    checkpoint_path = root / "checkpoints" / "M8.pt"
    checkpoint_metadata = _checkpoint(
        checkpoint_path,
        model,
        trainer.optimizer,
        run_id=args.run_id,
        label="M8",
        parent="M7",
        code=code,
        profile_fp=profile_fp,
        contract=contract,
        completed_games=8 * 64,
        replay_positions=len(replay.rows),
        optimizer_updates=int(trainer.update_count),
        samples_consumed=int(trainer.samples_consumed),
    )
    m8_row = _iteration_record(
        iteration=8,
        execution=execution,
        records=records,
        fresh_positions=len(fresh),
        replay_metrics=replay_metrics,
        train_metrics=train_metrics,
        selfplay_wall=selfplay_wall,
        cpu_before=cpu_before,
        inference_telemetry=telemetry,
        checkpoint={"path": str(checkpoint_path), "model_hash": checkpoint_metadata["model_hash"], "artifact_sha256": file_sha256(checkpoint_path)},
    )
    iteration_rows.append(m8_row)
    confirmation = _confirmation(selection, m8_row)
    status = "CONFIRMED" if confirmation["status"] == "CONFIRMED" else "INCONCLUSIVE"
    report = {
        "report_schema": "torus9-golden-current-v3-learning-v1",
        "run_id": args.run_id,
        "base_commit": BASE_COMMIT,
        "source_commit": code.git_commit_sha,
        "source_tree": code.git_tree_sha,
        "source_worktree_clean": code.working_tree_clean,
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "golden_source": profile["golden_source"],
        "golden_standard_untouched": True,
        "m0": manifest["m0"],
        "iterations": iteration_rows,
        "sweep_selection_m1_m7": selection,
        "m8_confirmation": confirmation,
        "result": status,
        "table_update": {"status": "NOT_ATTEMPTED", "reason": "table write is permitted only after confirmed M8 winner review"},
        "scientific_contract": {
            "games_per_iteration": 64,
            "iterations": 8,
            "total_games": 512,
            "mcts_simulations": 64,
            "training_batch_size": 64,
            "self_play_existing_batch_size": 64,
            "optimizer_steps_per_iteration": 80,
            "samples_consumed_per_iteration": 5120,
            "replay_generations": 3,
            "replay_cap": 20000,
            "ownership": True,
            "score": True,
            "explicit_symmetry_augmentation": False,
            "komi": 0.5,
            "legacy_komi_sentinel": 7.5,
        },
    }
    write_json(root / "final-report.json", report)
    docs_json = ROOT / "docs" / "TORUS9_GOLDEN_CURRENT_V3_LEARNING_20260913.json"
    docs_md = ROOT / "docs" / "TORUS9_GOLDEN_CURRENT_V3_LEARNING_20260913.md"
    write_json(docs_json, report)
    lines = [
        "# Torus 9×9 current Golden Standard M0→M8",
        "",
        f"Result: **{status}**",
        f"Base commit: `{BASE_COMMIT}`",
        f"Run: `{args.run_id}`",
        "",
        "The source Golden Standart sheet was read only. No spreadsheet write was performed by this run.",
        "",
        "| Iteration | Execution | Games/hour | Mean batch rows | Fresh positions | Optimizer steps | Samples |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| M{row['iteration']} | {row['execution']['id']} | {row['games_per_hour']:.2f} | {row['inference']['mean_batch_rows']:.2f} | {row['fresh_positions']} | {row['training']['optimizer_steps']} | {row['training']['samples_consumed']} |"
        for row in iteration_rows
    )
    lines.extend([
        "",
        f"Sweep selection M1→M7: **{selection['status']}**, winner={selection.get('winner')}",
        f"M8 confirmation: **{confirmation['status']}**",
        "",
        "The two spreadsheet rows remain pending until the result is confirmed and explicitly reviewed.",
        "",
    ])
    docs_md.write_text("\n".join(lines), encoding="utf-8")
    report["artifacts"] = {"run_root": str(root), "run_report": str(root / "final-report.json"), "docs_json": str(docs_json), "docs_markdown": str(docs_md)}
    write_json(root / "final-report.json", report)
    write_json(docs_json, report)
    manifest["status"] = "COMPLETED"
    manifest["m8"] = {"checkpoint": str(checkpoint_path), "model_hash": checkpoint_metadata["model_hash"]}
    manifest["result"] = status
    manifest["sweep_winner"] = selection.get("winner")
    write_json(root / "manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=TORUS9_WORKERS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run(args)
    print(f"TORUS 9×9 CURRENT PROFILE: {report['profile_id']}")
    print(f"TORUS 9×9 M0→M8: {report['result']}")
    print(f"SWEEP WINNER: {report['sweep_selection_m1_m7'].get('winner')}")


if __name__ == "__main__":
    main()
