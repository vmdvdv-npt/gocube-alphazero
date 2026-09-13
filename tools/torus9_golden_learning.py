#!/usr/bin/env python3
"""Run the Torus9 stable-learning rerun and its paired Arena evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from gocube_golden.provenance import capture_code_identity, derive_seed, file_sha256
from gocube_golden.torus9 import (
    Torus9GraphNet,
    Torus9NeuralEvaluator,
    Torus9RollingReplay,
    Torus9Trainer,
    generate_torus9_evaluation_starts,
    run_torus9_arena,
    run_torus9_selfplay_games,
    torus9_build_replay_samples,
    torus9_checkpoint_info,
    torus9_checkpoint_metadata,
    torus9_contract_proof,
    torus9_load_checkpoint,
    torus9_save_checkpoint,
    torus9_state_from_identity,
    validate_torus9_replay_sample,
    write_json,
    write_jsonl,
)
from gocube_golden.torus9_contract import (
    TORUS9_BATCH_SIZE,
    TORUS9_HIDDEN,
    TORUS9_KOMI,
    TORUS9_MAX_REPLAY_POSITIONS,
    TORUS9_MOVE_LIMIT,
    TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    TORUS9_PROFILE_ID,
    TORUS9_ROLLING_GENERATIONS,
    TORUS9_WORKERS,
    load_torus9_profile,
    profile_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "torus9-stable-learning-20260913-v1"
DEFAULT_OLD_RUN = ROOT / "runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3"


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _checkpoint(path: Path, model: Torus9GraphNet, optimizer: torch.optim.Optimizer | None, *, run_id: str, label: str, parent: str | None, model_seed: int, code, profile_fp: str, games: int, positions: int, updates: int, consumed: int) -> dict[str, object]:
    metadata = torus9_checkpoint_metadata(
        model=model, run_id=run_id, label=label, parent=parent, model_seed=model_seed,
        code=code, profile_fp=profile_fp, completed_games=games, replay_positions=positions,
        optimizer_updates=updates, samples_consumed=consumed,
    )
    metadata["adam_step"] = updates
    return torus9_save_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)


def _compact_samples(samples: list[dict[str, object]], generation: int) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for position, row in enumerate(samples):
        compact.append({
            "observation": row["observation"], "pi": row["pi"], "z": row["z"],
            "run_id": row["run_id"], "game_id": row["game_id"], "ply": row["ply"],
            "model_hash": row["model_hash"], "source_generation": generation,
            "replay_row_id": f"M{generation}:{row['game_id']}:{row['ply']}:{position}",
        })
    return compact


def _cross_entropy(model: Torus9GraphNet, rows: list[dict[str, object]], device: str) -> dict[str, float | None]:
    if not rows:
        return {"policy_ce": None, "wdl_ce": None, "total_ce": None}
    model.eval()
    policy_total = value_total = 0.0
    with torch.inference_mode():
        for offset in range(0, len(rows), 512):
            batch = rows[offset:offset + 512]
            observations = torch.tensor([row["observation"] for row in batch], dtype=torch.float32, device=device)
            policies = torch.tensor([row["pi"] for row in batch], dtype=torch.float32, device=device)
            values = torch.tensor([row["z"] for row in batch], dtype=torch.float32, device=device)
            policy_logits, value_logits = model(observations)
            policy_total += float((-(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1)).sum().cpu())
            value_total += float((-(values * F.log_softmax(value_logits, dim=1)).sum(dim=1)).sum().cpu())
    n = float(len(rows))
    return {"policy_ce": policy_total / n, "wdl_ce": value_total / n, "total_ce": (policy_total + value_total) / n}


def _frozen_state_metrics(model: Torus9GraphNet, starts: list[dict[str, object]], device: str) -> dict[str, float | None]:
    evaluator = Torus9NeuralEvaluator(model, device=device)
    from gocube_golden.rules import legal_actions
    wdl_bias: list[float] = []
    entropy: list[float] = []
    pass_probability: list[float] = []
    for row in starts:
        state = torus9_state_from_identity(row["state"])
        evaluation = evaluator.evaluate(state)
        legal = [int(action) if action != "PASS" else 81 for action in legal_actions(state)]
        probabilities = [evaluation.policy[index] for index in legal]
        entropy.append(-sum(value * math.log(value) for value in probabilities if value > 0.0))
        pass_probability.append(evaluation.policy[81])
        wdl_bias.append(evaluation.wdl[0] - evaluation.wdl[2])
    return {
        "mean_wdl_bias": sum(wdl_bias) / len(wdl_bias) if wdl_bias else None,
        "mean_policy_entropy": sum(entropy) / len(entropy) if entropy else None,
        "mean_pass_probability": sum(pass_probability) / len(pass_probability) if pass_probability else None,
        "states": len(starts),
    }


def _load_model(path: Path, device: str) -> Torus9GraphNet:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    architecture = metadata["architecture_config"]
    model = Torus9GraphNet(
        hidden=int(architecture["hidden"]), blocks=int(architecture["blocks"]),
        architecture_id=str(architecture["architecture_id"]),
    ).to(device)
    torus9_load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]}, device=device)
    model.eval()
    return model


def _run_canonical(*, root: Path, run_id: str, model_seed: int, selfplay_seed: int, games_per_iteration: int, iterations: int, workers: int, device: str, code, profile_fp: str, frozen_starts: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, Path], dict[int, list[dict[str, object]]]]:
    checkpoint_dir = root / "canonical" / "checkpoints"
    selfplay_dir = root / "canonical" / "selfplay"
    replay_dir = root / "canonical" / "replay"
    training_dir = root / "canonical" / "training"
    for directory in (checkpoint_dir, selfplay_dir, replay_dir, training_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed_everything(model_seed)
    model = Torus9GraphNet(hidden=TORUS9_HIDDEN).to(device)
    trainer = Torus9Trainer(model)
    replay = Torus9RollingReplay()
    m0_path = checkpoint_dir / "M0.pt"
    m0_metadata = _checkpoint(
        m0_path, model, None, run_id=run_id, label="M0", parent=None, model_seed=model_seed,
        code=code, profile_fp=profile_fp, games=0, positions=0, updates=0, consumed=0,
    )
    checkpoints: dict[str, Path] = {"M0": m0_path}
    iteration_rows: list[dict[str, object]] = []
    iteration_records: dict[int, list[dict[str, object]]] = {}
    total_positions = 0
    total_selfplay_wall = 0.0
    previous_label = "M0"

    for iteration in range(1, iterations + 1):
        label = f"M{iteration}"
        game_ids = [f"canonical-iter-{iteration:02d}-game-{index:04d}" for index in range(games_per_iteration)]
        before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        selfplay_started = time.perf_counter()
        records = run_torus9_selfplay_games(
            model, checkpoint_path=checkpoints[previous_label], run_id=run_id, label=previous_label,
            artifact=file_sha256(checkpoints[previous_label]), master_seed=selfplay_seed,
            profile_fp=profile_fp, game_ids=game_ids, workers=workers, code_identity=code, device=device,
        )
        selfplay_wall = time.perf_counter() - selfplay_started
        after_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        total_selfplay_wall += selfplay_wall
        write_jsonl(selfplay_dir / f"iter-{iteration:02d}-games.jsonl", [record.to_dict() for record in records])
        technical = sum(record.technical_termination is not None for record in records)
        fresh: list[dict[str, object]] = []
        for record in records:
            record.validate()
            if record.technical_termination is not None:
                continue
            rows = list(torus9_build_replay_samples(record))
            for row in rows:
                validate_torus9_replay_sample(row)
            fresh.extend(rows)
        if not fresh:
            raise RuntimeError(f"{label} produced no formal replay positions; cannot continue training safely")
        stamped = _compact_samples(fresh, iteration)
        write_jsonl(replay_dir / f"iter-{iteration:02d}-fresh.jsonl", stamped)
        replay_metrics = replay.append_generation(iteration, stamped)
        write_jsonl(replay_dir / f"rolling-after-{iteration:02d}.jsonl", list(replay.rows))
        train_started = time.perf_counter()
        train_metrics = trainer.train_fixed_budget(replay.rows, seed=derive_seed(model_seed, run_id, "training", iteration))
        train_wall = time.perf_counter() - train_started
        if train_metrics["optimizer_steps"] != TORUS9_OPTIMIZER_STEPS_PER_ITERATION or train_metrics["samples_consumed"] != 5120:
            raise AssertionError("Torus 9×9 canonical fixed training budget drifted")
        if any(size != TORUS9_BATCH_SIZE for size in train_metrics["batch_sizes"]):
            raise AssertionError("Torus 9×9 canonical training emitted a partial batch")
        checkpoint_path = checkpoint_dir / f"{label}.pt"
        metadata = _checkpoint(
            checkpoint_path, model, trainer.optimizer, run_id=run_id, label=label,
            parent=previous_label, model_seed=model_seed, code=code, profile_fp=profile_fp,
            games=iteration * games_per_iteration, positions=len(replay.rows),
            updates=trainer.update_count, consumed=trainer.samples_consumed,
        )
        checkpoints[label] = checkpoint_path
        previous_label = label
        total_positions += len(fresh)
        plies = [len(record.final_action_trace) for record in records]
        winners = Counter(record.formal_result for record in records if record.formal_result is not None)
        child_cpu = max(0.0, after_children.ru_utime - before_children.ru_utime)
        generation_rows = {
            str(generation): [row for row in replay.rows if int(row["source_generation"]) == generation]
            for generation in replay_metrics["generations_represented"]
        }
        current_ce = _cross_entropy(model, generation_rows.get(str(iteration), []), device)
        old_rows = [row for row in replay.rows if int(row["source_generation"]) < iteration]
        old_ce = _cross_entropy(model, old_rows, device)
        fixed = _frozen_state_metrics(model, frozen_starts, device)
        iteration_records[iteration] = [
            {"game_id": record.game_id, "start_state": record.start_state, "final_action_trace": record.final_action_trace, "formal_result": record.formal_result, "technical_termination": record.technical_termination}
            for record in records
        ]
        row = {
            "iteration": iteration, "label": label, "games": len(records), "valid_games": len(records) - technical,
            "technical": technical, "positions": len(fresh), "positions_per_game": len(fresh) / len(records),
            "average_ply": sum(plies) / len(plies), "median_ply": sorted(plies)[len(plies) // 2],
            "min_ply": min(plies), "max_ply": max(plies), "winner_counts": dict(sorted(winners.items())),
            "pass_frequency": sum(record.final_action_trace.count("PASS") for record in records) / sum(plies),
            "self_play_wall_time_sec": selfplay_wall, "training_wall_time_sec": train_wall,
            "games_per_hour": len(records) * 3600.0 / selfplay_wall if selfplay_wall else None,
            "positions_per_hour": len(fresh) * 3600.0 / selfplay_wall if selfplay_wall else None,
            "child_cpu_time_sec": child_cpu, "child_cpu_utilization_percent": 100.0 * child_cpu / selfplay_wall if selfplay_wall else None,
            "parent_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "child_peak_rss_mb": after_children.ru_maxrss / 1024.0, "replay": replay_metrics,
            "training": train_metrics, "current_generation_ce": current_ce, "old_generation_ce": old_ce,
            "frozen_diagnostics": fixed,
            "checkpoint": {"path": str(checkpoint_path), "model_hash": metadata["model_hash"], "artifact_sha256": file_sha256(checkpoint_path)},
            "nan_inf": False,
        }
        iteration_rows.append(row)
        write_json(training_dir / f"iter-{iteration:02d}.json", train_metrics)
        write_json(root / "canonical" / f"iter-{iteration:02d}-summary.json", row)
        del records, fresh, stamped
        gc.collect()

    summary = {
        "lineage": "canonical", "run_id": run_id, "games_per_iteration": games_per_iteration,
        "iterations": iterations, "total_games": games_per_iteration * iterations,
        "total_positions": total_positions, "mean_positions_per_game": total_positions / (games_per_iteration * iterations),
        "total_selfplay_wall_time_sec": total_selfplay_wall, "checkpoint_labels": list(checkpoints),
        "checkpoint_hashes": {label: torus9_checkpoint_info(path)["model_hash"] for label, path in checkpoints.items()},
        "iteration_rows": iteration_rows, "optimizer_steps_per_iteration": TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
        "samples_consumed_per_iteration": 5120, "total_optimizer_steps": trainer.update_count,
        "total_samples_consumed": trainer.samples_consumed,
        "m0": {"model_hash": m0_metadata["model_hash"], "fresh_initialization": True, "optimizer_state": None},
    }
    write_json(root / "canonical" / "summary.json", summary)
    return summary, checkpoints, iteration_records


def _performance_preflight(*, root: Path, run_id: str, profile: dict[str, object], workers: int, device: str, code, profile_fp: str) -> dict[str, object]:
    preflight = root / "preflight"
    checkpoint = preflight / "M0-preflight.pt"
    seed = int(profile["seeds"]["canonical_model_seed"])
    seed_everything(seed)
    model = Torus9GraphNet().to(device)
    metadata = _checkpoint(checkpoint, model, None, run_id=run_id + "-preflight", label="M0-preflight", parent=None, model_seed=seed, code=code, profile_fp=profile_fp, games=0, positions=0, updates=0, consumed=0)
    started = time.perf_counter()
    records = run_torus9_selfplay_games(
        model, checkpoint_path=checkpoint, run_id=run_id + "-preflight", label="M0-preflight",
        artifact=file_sha256(checkpoint), master_seed=int(profile["seeds"]["canonical_selfplay_master_seed"]),
        profile_fp=profile_fp, game_ids=[f"preflight-game-{index:03d}" for index in range(16)],
        workers=workers, code_identity=code, device=device,
    )
    wall = time.perf_counter() - started
    technical = sum(record.technical_termination is not None for record in records)
    positions = sum(len(record.positions) for record in records)
    plies = [len(record.final_action_trace) for record in records]
    result = {
        "status": "PASS" if technical == 0 else "PASS_WITH_TECHNICAL_WARNINGS", "games": len(records),
        "valid_games": len(records) - technical, "technical": technical, "positions": positions,
        "average_ply": sum(plies) / len(plies) if plies else None, "min_ply": min(plies) if plies else None,
        "max_ply": max(plies) if plies else None, "wall_time_sec": wall,
        "games_per_hour": len(records) * 3600.0 / wall if wall else None,
        "positions_per_hour": positions * 3600.0 / wall if wall else None, "workers": workers,
        "checkpoint": {"path": str(checkpoint), "model_hash": metadata["model_hash"], "artifact_sha256": file_sha256(checkpoint)},
        "technical_count_is_not_draw": True,
    }
    write_json(preflight / "summary.json", result)
    write_jsonl(preflight / "games.jsonl", [record.to_dict() for record in records])
    return result


def _arena(*, root: Path, run_id: str, slug: str, candidate: Path, reference: Path, candidate_label: str, reference_label: str, starts: list[dict[str, object]], pairs: int, seed: int, workers: int, device: str) -> dict[str, object]:
    return run_torus9_arena(
        run_id=run_id, comparison=slug, candidate_path=candidate, reference_path=reference,
        candidate_label=candidate_label, reference_label=reference_label, starts=starts[:pairs],
        master_seed=seed, output_dir=root / "canonical" / "arena" / slug, workers=workers, device=device,
    )


def _old_baseline(old_run: Path, starts: list[dict[str, object]], device: str) -> dict[str, object]:
    report_path = old_run / "final-report.json"
    summary_path = old_run / "canonical" / "summary.json"
    if not report_path.exists() or not summary_path.exists():
        return {"status": "UNAVAILABLE", "path": str(old_run)}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fixed: dict[str, object] = {}
    for iteration in range(1, 9):
        checkpoint = old_run / "canonical" / "checkpoints" / f"M{iteration}.pt"
        if checkpoint.exists():
            fixed[str(iteration)] = _frozen_state_metrics(_load_model(checkpoint, device), starts, device)
    return {"status": "AVAILABLE", "path": str(old_run), "run_id": summary.get("run_id"), "source_sha": report.get("source_sha"), "arena": report.get("arena", {}), "iteration_rows": summary.get("iteration_rows", []), "checkpoint_hashes": summary.get("checkpoint_hashes", {}), "fixed_diagnostics": fixed}


def _trajectory(old: dict[str, object], new_summary: dict[str, object]) -> list[dict[str, object]]:
    old_rows = {int(row["iteration"]): row for row in old.get("iteration_rows", [])} if old.get("status") == "AVAILABLE" else {}
    new_rows = {int(row["iteration"]): row for row in new_summary["iteration_rows"]}
    old_fixed = old.get("fixed_diagnostics", {})
    result = []
    for iteration in range(1, 9):
        old_row = old_rows.get(iteration, {})
        new_row = new_rows[iteration]
        old_diag = old_fixed.get(str(iteration), {})
        result.append({"iteration": iteration, "old_positions": old_row.get("positions"), "new_positions": new_row["positions"], "old_average_ply": old_row.get("average_ply"), "new_average_ply": new_row["average_ply"], "old_wdl_bias": old_diag.get("mean_wdl_bias"), "new_wdl_bias": new_row["frozen_diagnostics"].get("mean_wdl_bias")})
    return result


def _stability_verdict(summary: dict[str, object]) -> dict[str, object]:
    rows = summary["iteration_rows"]
    all_budget = all(row["training"]["optimizer_steps"] == 80 and row["training"]["samples_consumed"] == 5120 and all(size == 64 for size in row["training"]["batch_sizes"]) and row["training"]["adam_step_after"] == row["iteration"] * 80 for row in rows)
    extreme_winners = []
    for row in rows:
        winners = row["winner_counts"]
        total = sum(winners.values())
        if total and max(winners.values()) / total > 0.9:
            extreme_winners.append(row["iteration"])
    biases = [float(row["frozen_diagnostics"]["mean_wdl_bias"]) for row in rows]
    m4 = rows[3]
    m2 = rows[1]
    m4_like = bool(m4["average_ply"] < 0.5 * m2["average_ply"] and max(m4["winner_counts"].values(), default=0) / max(1, sum(m4["winner_counts"].values())) > 0.8)
    oscillation = max(biases) - min(biases) > 1.0
    confirmed = bool(all_budget and not extreme_winners and not m4_like and not oscillation)
    return {"status": "CONFIRMED" if confirmed else "PARTIAL", "fixed_budget": all_budget, "extreme_winner_iterations": extreme_winners, "fixed_state_wdl_bias_range": [min(biases), max(biases)] if biases else None, "wdl_oscillation": oscillation, "m4_like_collapse": m4_like, "criterion": "80 full batch64 updates; no >90% winner iteration; no M4 short/black collapse; fixed-state WDL bias range <= 1.0"}


def _learning_verdict(arenas: dict[str, dict[str, object]]) -> tuple[str, str]:
    def score(name: str) -> float | None:
        return arenas.get(name, {}).get("mean_pair_score")  # type: ignore[return-value]
    m8_m0, m8_m1, m8_m4 = score("M8-vs-M0"), score("M8-vs-M1"), score("M8-vs-M4")
    if m8_m0 is not None and m8_m1 is not None and m8_m4 is not None and min(m8_m0, m8_m1, m8_m4) > 0.5:
        return "LEARNING CONFIRMED", "M8 has practical Arena improvement over M0, M1, and M4 on the declared paired protocol."
    if m8_m0 is not None and m8_m0 > 0.5:
        return "PARTIAL", "M8 improves over M0, but the complete M8 > M1 and M8 > M4 evidence threshold was not met."
    return "NOT CONFIRMED", "The declared M8 practical improvement threshold over M0 was not met."


def _markdown(report: dict[str, object]) -> str:
    canonical = report["canonical"]
    arenas = report["arena"]
    lines = [
        "TORUS 9×9 STABLE LEARNING RERUN", "", "SOURCE:", f"{report['source_sha']} (clean source commit)", "",
        "FIXES:", "8 graph blocks", "3-generation / 20k rolling replay", "80 full batch64 updates per iteration", "",
        "SELF-PLAY:", "8 × 64 = 512 games", "", f"OVERALL: {report['learning_verdict']}", "",
        f"TRAINING STABILITY: {report['training_stability']['status']}", "",
        f"OLD M8 vs NEW M8: {report['new_m8_vs_old_m8'].get('W/L/D', 'UNAVAILABLE')}", "",
        f"OLD FAILURE MODE REPRODUCED: {'YES' if report['training_stability']['m4_like_collapse'] or report['training_stability']['wdl_oscillation'] else 'NO'}",
        f"M4-LIKE COLLAPSE: {'YES' if report['training_stability']['m4_like_collapse'] else 'NO'}", "",
        "## Contract and run", "", f"Architecture: {report['contract']['architecture']['architecture_id']} / {report['contract']['architecture_fingerprint']}",
        f"Policy/value heads: [82] / [3]; PASS index: 81; komi: {TORUS9_KOMI}",
        f"Replay: {TORUS9_ROLLING_GENERATIONS} generations, cap {TORUS9_MAX_REPLAY_POSITIONS} positions",
        f"Optimizer: Adam lr=0.001 wd=0; {TORUS9_OPTIMIZER_STEPS_PER_ITERATION} × batch {TORUS9_BATCH_SIZE}", "",
        "## M0→M8 telemetry", "", "| ITER | GAMES | FRESH POS | REPLAY POS | AVG PLY | WINNERS | OPT STEPS | SAMPLES | UNIQUE / REUSED | WDL BIAS |", "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in canonical["iteration_rows"]:
        training = row["training"]
        lines.append(f"| M{row['iteration']} | {row['games']} | {row['positions']} | {row['replay']['rolling_buffer_positions']} | {row['average_ply']:.2f} | {row['winner_counts']} | {training['optimizer_steps']} | {training['samples_consumed']} | {training['unique_sample_rows']} / {training['reused_sample_rows']} | {row['frozen_diagnostics']['mean_wdl_bias']:.4f} |")
    lines += ["", "## Arena", "", "| COMPARISON | W/L/D | VALID PAIRS | TECHNICAL GAMES | MEAN PAIR SCORE | 95% INTERVAL |", "|---|---:|---:|---:|---:|---|"]
    for slug in ("M4-vs-M0", "M4-vs-M1", "M8-vs-M0", "M8-vs-M1", "M8-vs-M4", "M8-vs-M7", "NEW-M8-vs-OLD-M8"):
        row = arenas.get(slug, {})
        lines.append(f"| {slug} | {row.get('W/L/D')} | {row.get('pairs_valid')} | {row.get('technical_games')} | {row.get('mean_pair_score')} | {row.get('95_percent_hoeffding_interval')} |")
    lines += ["", "## Old/new trajectory", "", "| ITER | OLD POS | NEW POS | OLD AVG PLY | NEW AVG PLY | OLD WDL BIAS | NEW WDL BIAS |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["old_new_trajectory"]:
        lines.append(f"| M{row['iteration']} | {row['old_positions']} | {row['new_positions']} | {row['old_average_ply']} | {row['new_average_ply']:.2f} | {row['old_wdl_bias']} | {row['new_wdl_bias']:.4f} |")
    lines += ["", "## Final analysis", "", "1. 8-block network global point-policy coverage: CONFIRMED by the Torus5 diameter=4 / Torus9 diameter=8 dependency proof and canonical 8-block path.", "2. Fresh-generation feedback collapse: " + ("NOT REPRODUCED under the stability criteria." if report["training_stability"]["status"] == "CONFIRMED" else "PARTIAL; inspect the fixed-state and replay telemetry."), "3. Training-impulse wandering: " + ("NOT REPRODUCED; optimizer exposure is fixed at 80×64." if report["training_stability"]["fixed_budget"] else "FAILED."), "4. WDL oscillation: " + ("NO under the declared threshold." if not report["training_stability"]["wdl_oscillation"] else "YES."), f"5. NEW M8 stronger than OLD M8: {report['new_m8_vs_old_m8'].get('mean_pair_score')} mean paired score.", f"6. NEW M8 stronger than NEW M1: {arenas.get('M8-vs-M1', {}).get('mean_pair_score')}.", f"7. NEW M4 normal or degraded: {'NORMAL' if not report['training_stability']['m4_like_collapse'] else 'DEGRADED'}.", "8. Pipeline ready for next stage: " + ("YES, conditionally on the declared Arena evidence." if report["training_stability"]["status"] == "CONFIRMED" else "NOT YET; retain the diagnostic boundary."), "9. Remaining bottleneck: 64-simulation teacher/search target quality remains the known unmodified risk.", "10. Next step: better teacher/search diagnostics first; ownership, history, and other auxiliary changes remain out of this rerun.", "", "## Verification", "", f"Cheap contract: {report['contract']['status']}", f"Performance preflight: {report['performance_preflight']['status']}", f"Source commit: {report['source_sha']}", "Full pytest: run after report generation before final push.", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    profile = load_torus9_profile()
    code = capture_code_identity(ROOT)
    if not code.working_tree_clean:
        raise RuntimeError("Scientific run requires a clean source commit")
    root = ROOT / "runs" / "torus9-stable-learning-v2" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    contract = torus9_contract_proof()
    write_json(root / "contract-proof.json", contract)
    profile_fp = profile_fingerprint(profile)
    starts = list(generate_torus9_evaluation_starts(master_seed=int(profile["seeds"]["evaluation_master_seed"])))
    evaluation_manifest = {"contract_id": "torus9-frozen-evaluation-corpus-v1", "master_seed": profile["seeds"]["evaluation_master_seed"], "starts": len(starts), "stratification": {"prefix_lengths": [2, 4, 6, 8, 10, 12, 14, 16], "accepted_per_stratum": 8}, "exact_state_includes": ["stones", "side_to_move", "full positional-superko history", "consecutive_passes", "komi"], "color_swapped_pair_semantics": True, "fingerprint": starts[0]["corpus_fingerprint"], "created_before_results": True, "source_sha": code.git_commit_sha}
    write_jsonl(root / "evaluation" / "starts.jsonl", starts)
    write_json(root / "evaluation" / "manifest.json", evaluation_manifest)
    report: dict[str, object] = {"run_id": args.run_id, "source_sha": code.git_commit_sha, "source_tree": code.git_tree_sha, "source_worktree_clean": code.working_tree_clean, "profile_id": TORUS9_PROFILE_ID, "profile_fingerprint": profile_fp, "contract": contract, "profile": profile, "evaluation": evaluation_manifest, "fixes": {"graph_blocks": 8, "hidden": 64, "rolling_generations": 3, "maximum_replay_positions": 20000, "optimizer_steps": 80, "batch_size": 64, "samples_consumed": 5120}, "old_run": str(args.old_run)}
    report["performance_preflight"] = _performance_preflight(root=root, run_id=args.run_id, profile=profile, workers=args.workers, device=args.device, code=code, profile_fp=profile_fp)
    canonical_summary, checkpoints, iteration_records = _run_canonical(root=root, run_id=args.run_id, model_seed=int(profile["seeds"]["canonical_model_seed"]), selfplay_seed=int(profile["seeds"]["canonical_selfplay_master_seed"]), games_per_iteration=64, iterations=8, workers=args.workers, device=args.device, code=code, profile_fp=profile_fp, frozen_starts=starts)
    report["canonical"] = canonical_summary
    arena: dict[str, dict[str, object]] = {}
    arena["M4-vs-M0"] = _arena(root=root, run_id=args.run_id, slug="M4-vs-M0", candidate=checkpoints["M4"], reference=checkpoints["M0"], candidate_label="M4", reference_label="M0", starts=starts, pairs=32, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 40, workers=args.workers, device=args.device)
    arena["M4-vs-M1"] = _arena(root=root, run_id=args.run_id, slug="M4-vs-M1", candidate=checkpoints["M4"], reference=checkpoints["M1"], candidate_label="M4", reference_label="M1", starts=starts, pairs=16, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 41, workers=args.workers, device=args.device)
    arena["M8-vs-M0"] = _arena(root=root, run_id=args.run_id, slug="M8-vs-M0", candidate=checkpoints["M8"], reference=checkpoints["M0"], candidate_label="M8", reference_label="M0", starts=starts, pairs=64, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 80, workers=args.workers, device=args.device)
    arena["M8-vs-M1"] = _arena(root=root, run_id=args.run_id, slug="M8-vs-M1", candidate=checkpoints["M8"], reference=checkpoints["M1"], candidate_label="M8", reference_label="M1", starts=starts, pairs=64, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 81, workers=args.workers, device=args.device)
    arena["M8-vs-M4"] = _arena(root=root, run_id=args.run_id, slug="M8-vs-M4", candidate=checkpoints["M8"], reference=checkpoints["M4"], candidate_label="M8", reference_label="M4", starts=starts, pairs=64, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 82, workers=args.workers, device=args.device)
    arena["M8-vs-M7"] = _arena(root=root, run_id=args.run_id, slug="M8-vs-M7", candidate=checkpoints["M8"], reference=checkpoints["M7"], candidate_label="M8", reference_label="M7", starts=starts, pairs=16, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 83, workers=args.workers, device=args.device)
    old = _old_baseline(args.old_run, starts, args.device)
    report["old_baseline"] = old
    if old.get("status") == "AVAILABLE":
        old_m8 = args.old_run / "canonical" / "checkpoints" / "M8.pt"
        arena["NEW-M8-vs-OLD-M8"] = _arena(root=root, run_id=args.run_id, slug="NEW-M8-vs-OLD-M8", candidate=checkpoints["M8"], reference=old_m8, candidate_label="NEW-M8", reference_label="OLD-M8", starts=starts, pairs=32, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 90, workers=args.workers, device=args.device)
    else:
        arena["NEW-M8-vs-OLD-M8"] = {"status": "UNAVAILABLE", "reason": "old baseline artifacts are unavailable"}
    report["arena"] = arena
    report["new_m8_vs_old_m8"] = arena["NEW-M8-vs-OLD-M8"]
    report["old_new_trajectory"] = _trajectory(old, canonical_summary)
    report["training_stability"] = _stability_verdict(canonical_summary)
    report["learning_verdict"], report["verdict_reason"] = _learning_verdict(arena)
    report["performance"] = {"selfplay_games_per_hour": sum(float(row["games_per_hour"]) for row in canonical_summary["iteration_rows"]) / 8.0, "selfplay_positions_per_hour": sum(float(row["positions_per_hour"]) for row in canonical_summary["iteration_rows"]) / 8.0, "average_ply": sum(float(row["average_ply"]) for row in canonical_summary["iteration_rows"]) / 8.0, "peak_memory_mb": max(float(row["parent_peak_rss_mb"]) for row in canonical_summary["iteration_rows"]), "workers": args.workers}
    report_path = root / "final-report.json"
    write_json(report_path, report)
    docs_json = ROOT / "docs" / "TORUS9_STABLE_LEARNING_RERUN_20260913.json"
    docs_md = ROOT / "docs" / "TORUS9_STABLE_LEARNING_RERUN_20260913.md"
    docs_manifest = ROOT / "docs" / "TORUS9_STABLE_LEARNING_RERUN_20260913.manifest.json"
    write_json(docs_json, report)
    docs_md.parent.mkdir(parents=True, exist_ok=True)
    docs_md.write_text(_markdown(report), encoding="utf-8")
    manifest = {"manifest_schema": "torus9-stable-learning-rerun-v1", "run_id": args.run_id, "source_commit": code.git_commit_sha, "source_tree": code.git_tree_sha, "source_worktree_clean": code.working_tree_clean, "profile_id": TORUS9_PROFILE_ID, "profile_fingerprint": profile_fp, "architecture_id": contract["architecture"]["architecture_id"], "architecture_fingerprint": contract["architecture_fingerprint"], "old_run": str(args.old_run), "new_m0_fresh_initialization": True, "scientific_parameters": {"games_per_iteration": 64, "iterations": 8, "total_games": 512, "workers": 16, "komi": 0.5, "selfplay_simulations": 64, "arena_simulations": 64, "cpuct": 1.25, "fpu": 0.0, "root_noise": True, "arena_noise": False, "temperature": "1.0 plies 1..8, then 0", "move_limit": 500, "ownership": False, "score_head": False}, "replay": {"generations": 3, "maximum_positions": 20000, "eviction": "oldest generation then oldest rows at cap", "sampling": "deterministic permutation when enough rows, deterministic replacement otherwise"}, "training": {"optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0, "steps_per_iteration": 80, "batch_size": 64, "samples_per_iteration": 5120, "optimizer_continuation": True}, "artifacts": {"run_report": str(report_path), "run_report_sha256": file_sha256(report_path), "docs_json": str(docs_json), "docs_markdown": str(docs_md)}, "status": "RUN_COMPLETED_BEFORE_FINAL_TEST_COMMIT"}
    write_json(docs_manifest, manifest)
    report["manifest_path"] = str(docs_manifest)
    write_json(report_path, report)
    write_json(docs_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--old-run", type=Path, default=DEFAULT_OLD_RUN)
    parser.add_argument("--workers", type=int, default=TORUS9_WORKERS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.workers != TORUS9_WORKERS:
        raise SystemExit("The canonical Torus 9×9 protocol is frozen to 16 workers")
    report = run(args)
    print(f"TORUS 9×9 CONTRACT: {report['contract']['status']}")
    print(f"TORUS 9×9 TRAINING STABILITY: {report['training_stability']['status']}")
    print(f"TORUS 9×9 LEARNING: {report['learning_verdict']}")


if __name__ == "__main__":
    main()
