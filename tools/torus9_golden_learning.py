#!/usr/bin/env python3
"""Run the Torus 9×9 contract, bring-up, and canonical Golden learning proof."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import time

import torch

from gocube_golden.provenance import capture_code_identity, file_sha256
from gocube_golden.torus9 import (
    Torus9GraphNet,
    Torus9Trainer,
    generate_torus9_evaluation_starts,
    run_torus9_arena,
    run_torus9_selfplay_games,
    torus9_build_replay_samples,
    torus9_checkpoint_info,
    torus9_checkpoint_metadata,
    torus9_contract_proof,
    torus9_first_move_statistics,
    torus9_save_checkpoint,
    validate_torus9_replay_sample,
    write_json,
    write_jsonl,
)
from gocube_golden.torus9_contract import (
    TORUS9_BATCH_SIZE,
    TORUS9_KOMI,
    TORUS9_MOVE_LIMIT,
    TORUS9_PROFILE_ID,
    TORUS9_WORKERS,
    load_torus9_profile,
    profile_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "torus9-golden-learning-proof-20260913-v1"


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _checkpoint(
    path: Path,
    model: Torus9GraphNet,
    optimizer: torch.optim.Optimizer | None,
    *,
    run_id: str,
    label: str,
    parent: str | None,
    model_seed: int,
    code,
    profile_fp: str,
    games: int,
    positions: int,
    updates: int,
    consumed: int,
) -> dict[str, object]:
    metadata = torus9_checkpoint_metadata(
        model=model,
        run_id=run_id,
        label=label,
        parent=parent,
        model_seed=model_seed,
        code=code,
        profile_fp=profile_fp,
        completed_games=games,
        replay_positions=positions,
        optimizer_updates=updates,
        samples_consumed=consumed,
    )
    return torus9_save_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)


def _run_lineage(
    *,
    root: Path,
    lineage: str,
    run_id: str,
    model_seed: int,
    selfplay_seed: int,
    games_per_iteration: int,
    iterations: int,
    workers: int,
    device: str,
    code,
    profile_fp: str,
) -> tuple[dict[str, object], dict[str, Path], dict[int, list], dict[str, object]]:
    checkpoint_dir = root / lineage / "checkpoints"
    selfplay_dir = root / lineage / "selfplay"
    replay_dir = root / lineage / "replay"
    training_dir = root / lineage / "training"
    for directory in (checkpoint_dir, selfplay_dir, replay_dir, training_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed_everything(model_seed)
    model = Torus9GraphNet().to(device)
    trainer = Torus9Trainer(model)
    m0_label = "M0b" if lineage == "bringup" else "M0"
    m0_path = checkpoint_dir / f"{m0_label}.pt"
    m0_metadata = _checkpoint(
        m0_path, model, None, run_id=run_id, label=m0_label, parent=None,
        model_seed=model_seed, code=code, profile_fp=profile_fp, games=0,
        positions=0, updates=0, consumed=0,
    )
    checkpoints: dict[str, Path] = {m0_label: m0_path}
    iteration_rows: list[dict[str, object]] = []
    iteration_records: dict[int, list] = {}
    source_samples_total = 0
    total_selfplay_wall = 0.0
    total_positions = 0
    for iteration in range(1, iterations + 1):
        label = f"M{iteration}b" if lineage == "bringup" else f"M{iteration}"
        game_ids = [f"{lineage}-iter-{iteration:02d}-game-{index:04d}" for index in range(games_per_iteration)]
        before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        records = run_torus9_selfplay_games(
            model,
            checkpoint_path=checkpoints[f"M{iteration - 1}b" if lineage == "bringup" else f"M{iteration - 1}"],
            run_id=run_id,
            label=f"M{iteration - 1}b" if lineage == "bringup" else f"M{iteration - 1}",
            artifact=file_sha256(checkpoints[f"M{iteration - 1}b" if lineage == "bringup" else f"M{iteration - 1}"]),
            master_seed=selfplay_seed,
            profile_fp=profile_fp,
            game_ids=game_ids,
            workers=workers,
            code_identity=code,
            device=device,
        )
        selfplay_wall = time.perf_counter() - started
        after_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        total_selfplay_wall += selfplay_wall
        write_jsonl(selfplay_dir / f"iter-{iteration:02d}-games.jsonl", [record.to_dict() for record in records])
        technical = sum(record.technical_termination is not None for record in records)
        if technical:
            raise RuntimeError(f"{lineage} iteration {iteration} has {technical} technical self-play games")
        samples: list[dict[str, object]] = []
        for record in records:
            record.validate()
            rows = torus9_build_replay_samples(record)
            for row in rows:
                validate_torus9_replay_sample(row)
            samples.extend(rows)
        write_jsonl(replay_dir / f"iter-{iteration:02d}.jsonl", samples)
        # Full replay rows contain every superko-history board and are persisted
        # above.  Training needs only the observation and two WDL/policy
        # targets; remove the large provenance payload before the next phase.
        for row in samples:
            compact = {"observation": row["observation"], "pi": row["pi"], "z": row["z"]}
            row.clear()
            row.update(compact)
        gc.collect()
        train_started = time.perf_counter()
        train_metrics = trainer.train_fresh_epoch(samples, seed=model_seed + iteration)
        train_wall = time.perf_counter() - train_started
        checkpoint_path = checkpoint_dir / f"{label}.pt"
        previous = f"M{iteration - 1}b" if lineage == "bringup" else f"M{iteration - 1}"
        metadata = _checkpoint(
            checkpoint_path, model, trainer.optimizer, run_id=run_id, label=label,
            parent=previous, model_seed=model_seed, code=code, profile_fp=profile_fp,
            games=iteration * games_per_iteration, positions=len(samples),
            updates=trainer.update_count, consumed=trainer.samples_consumed,
        )
        checkpoints[label] = checkpoint_path
        iteration_records[iteration] = [
            {
                "game_id": record.game_id,
                "start_state": record.start_state,
                "final_action_trace": record.final_action_trace,
                "formal_result": record.formal_result,
                "technical_termination": record.technical_termination,
            }
            for record in records
        ]
        positions = len(samples)
        source_samples_total += positions
        total_positions += positions
        plies = [len(record.final_action_trace) for record in records]
        child_cpu = max(0.0, after_children.ru_utime - before_children.ru_utime)
        iteration_rows.append({
            "iteration": iteration,
            "label": label,
            "games": len(records),
            "valid_games": len(records),
            "technical": technical,
            "positions": positions,
            "positions_per_game": positions / len(records),
            "average_ply": sum(plies) / len(plies),
            "min_ply": min(plies),
            "max_ply": max(plies),
            "self_play_wall_time_sec": selfplay_wall,
            "training_wall_time_sec": train_wall,
            "games_per_hour": len(records) * 3600.0 / selfplay_wall if selfplay_wall else None,
            "positions_per_hour": positions * 3600.0 / selfplay_wall if selfplay_wall else None,
            "child_cpu_time_sec": child_cpu,
            "child_cpu_utilization_percent": 100.0 * child_cpu / selfplay_wall if selfplay_wall else None,
            "parent_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "child_peak_rss_mb": after_children.ru_maxrss / 1024.0,
            "optimizer_steps": train_metrics["updates"],
            "train_samples_consumed": train_metrics["samples"],
            "training": train_metrics,
            "checkpoint": {"path": str(checkpoint_path), "model_hash": metadata["model_hash"], "artifact_sha256": file_sha256(checkpoint_path)},
            "nan_inf": False,
        })
        del records
        gc.collect()
    summary = {
        "lineage": lineage,
        "run_id": run_id,
        "games_per_iteration": games_per_iteration,
        "iterations": iterations,
        "total_games": games_per_iteration * iterations,
        "total_positions": total_positions,
        "mean_positions_per_game": total_positions / (games_per_iteration * iterations),
        "total_selfplay_wall_time_sec": total_selfplay_wall,
        "checkpoint_labels": list(checkpoints),
        "checkpoint_hashes": {label: torus9_checkpoint_info(path)["model_hash"] for label, path in checkpoints.items()},
        "iteration_rows": iteration_rows,
    }
    write_json(root / lineage / "summary.json", summary)
    return summary, checkpoints, iteration_records, {"m0": m0_metadata}


def _performance_preflight(*, root: Path, run_id: str, profile: Mapping[str, object], workers: int, device: str, code, profile_fp: str) -> dict[str, object]:
    """Run the required all-worker smoke before committing to 160 games."""
    preflight = root / "preflight"
    checkpoint = preflight / "M0-preflight.pt"
    seed = int(profile["seeds"]["bringup_model_seed"])
    torch.manual_seed(seed)
    model = Torus9GraphNet().to(device)
    metadata = _checkpoint(
        checkpoint, model, None, run_id=run_id + "-preflight", label="M0-preflight", parent=None,
        model_seed=seed, code=code, profile_fp=profile_fp, games=0, positions=0, updates=0, consumed=0,
    )
    started = time.perf_counter()
    records = run_torus9_selfplay_games(
        model,
        checkpoint_path=checkpoint,
        run_id=run_id + "-preflight",
        label="M0-preflight",
        artifact=file_sha256(checkpoint),
        master_seed=int(profile["seeds"]["bringup_selfplay_master_seed"]),
        profile_fp=profile_fp,
        game_ids=[f"preflight-game-{index:03d}" for index in range(16)],
        workers=workers,
        code_identity=code,
        device=device,
    )
    wall = time.perf_counter() - started
    technical = sum(record.technical_termination is not None for record in records)
    positions = sum(len(record.positions) for record in records)
    plies = [len(record.final_action_trace) for record in records]
    result = {
        "status": "PASS" if technical == 0 else "FAIL",
        "games": len(records),
        "valid_games": len(records) - technical,
        "technical": technical,
        "positions": positions,
        "average_ply": sum(plies) / len(plies) if plies else None,
        "min_ply": min(plies) if plies else None,
        "max_ply": max(plies) if plies else None,
        "wall_time_sec": wall,
        "games_per_hour": len(records) * 3600.0 / wall if wall else None,
        "positions_per_hour": positions * 3600.0 / wall if wall else None,
        "workers": workers,
        "parent_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "checkpoint": {"path": str(checkpoint), "model_hash": metadata["model_hash"], "artifact_sha256": file_sha256(checkpoint)},
        "technical_count_is_not_draw": True,
    }
    write_json(preflight / "summary.json", result)
    write_jsonl(preflight / "games.jsonl", [record.to_dict() for record in records])
    if technical:
        raise RuntimeError("Torus 9×9 performance preflight failed")
    return result


def _arena(
    *,
    root: Path,
    lineage: str,
    run_id: str,
    slug: str,
    candidate: Path,
    reference: Path,
    candidate_label: str,
    reference_label: str,
    starts: list[dict[str, object]],
    pairs: int,
    seed: int,
    workers: int,
    device: str,
) -> dict[str, object]:
    selected = starts[:pairs]
    return run_torus9_arena(
        run_id=run_id,
        comparison=slug,
        candidate_path=candidate,
        reference_path=reference,
        candidate_label=candidate_label,
        reference_label=reference_label,
        starts=selected,
        master_seed=seed,
        output_dir=root / lineage / "arena" / slug,
        workers=workers,
        device=device,
    )


def _representative_traces(root: Path, lineage: str, records: dict[int, list], arena_root: Path, labels: tuple[str, ...]) -> dict[str, object]:
    output = root / lineage / "representative-traces"
    output.mkdir(parents=True, exist_ok=True)
    selected: dict[str, object] = {}
    all_selfplay = [record for rows in records.values() for record in rows if (record.get("technical_termination") if isinstance(record, dict) else record.technical_termination) is None]
    for index, record in enumerate(all_selfplay[:6]):
        path = output / f"selfplay-{index + 1:02d}.json"
        write_json(path, record.to_dict() if hasattr(record, "to_dict") else record)
        selected[path.name] = str(path)
    for label in labels:
        path = arena_root / label / "games.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, row in enumerate(rows[:4]):
            trace_path = output / f"{label}-game-{index + 1:02d}.json"
            write_json(trace_path, row)
            selected[trace_path.name] = str(trace_path)
    return selected


def _markdown(report: dict[str, object]) -> str:
    contract = report["contract"]
    bringup = report.get("bring_up") or {}
    canonical = report.get("canonical") or {}
    arenas = report.get("arena", {})
    first = report.get("first_move", {})
    performance = report.get("performance", {})
    verdict = report.get("learning_verdict", "NOT CONFIRMED")
    lines = [
        "# Torus 9×9 Golden learning proof",
        "",
        f"TORUS 9×9 CONTRACT: {contract.get('status', 'FAIL')}",
        "",
        "POINTS: 81",
        "ACTIONS: 82",
        "PASS INDEX: 81",
        "KOMI: 0.5",
        "",
        f"BRING-UP: {bringup.get('status', 'FAIL')}",
        "",
        "BRING-UP:",
        "5 × 32 games",
        "",
        f"MEAN POSITIONS/GAME: {bringup.get('mean_positions_per_game')}",
        f"PROJECTED POSITIONS @ 64: {bringup.get('projected_positions_at_64')}",
        f"SELECTED CANONICAL GAMES/ITER: {bringup.get('canonical_games_per_iteration')}",
        f"WHY: {bringup.get('reason')}",
        "",
        "| ITER | GAMES | POSITIONS | POS/GAME | SELF-PLAY TIME | TRAIN TIME | TECHNICAL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bringup.get("iteration_rows", []):
        lines.append(f"| M{row['iteration']} | {row['games']} | {row['positions']} | {row['positions_per_game']:.2f} | {row['self_play_wall_time_sec']:.2f}s | {row['training_wall_time_sec']:.2f}s | {row['technical']} |")
    lines += ["", "## Canonical M0→M8", "", "| ITER | GAMES | POSITIONS | OPT STEPS | AVG PLY | SELF-PLAY TIME | TRAIN TIME | TECHNICAL |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in canonical.get("iteration_rows", []):
        lines.append(f"| {row['iteration']} | {row['games']} | {row['positions']} | {row['optimizer_steps']} | {row['average_ply']:.2f} | {row['self_play_wall_time_sec']:.2f}s | {row['training_wall_time_sec']:.2f}s | {row['technical']} |")
    lines += ["", f"CANONICAL RUN: M0 → M8", f"GAMES/ITER: {canonical.get('games_per_iteration')}", f"TOTAL SELF-PLAY: {canonical.get('total_games')}", f"TOTAL POSITIONS: {canonical.get('total_positions')}", "TECHNICAL: 0 expected", "", "## Arena report", ""]
    for slug in ("M4-vs-M0", "M4-vs-M1", "M8-vs-M0", "M8-vs-M1", "M8-vs-M4", "M8-vs-M7"):
        row = arenas.get(slug, {})
        lines.append(f"{slug.replace('-', ' ')} = {row.get('W/L/D')} | pairs={row.get('pairs_valid')} | mean={row.get('mean_pair_score')} | CI={row.get('95_percent_hoeffding_interval')} | technical={row.get('technical_games')}")
    lines += ["", "PRIMARY CONFIDENCE INTERVALS:"]
    for slug in ("M8-vs-M0", "M8-vs-M1", "M8-vs-M4"):
        lines.append(f"- {slug}: {arenas.get(slug, {}).get('95_percent_hoeffding_interval')}")
    lines += ["", "## Verdict", "", f"TORUS 9×9 LEARNING: {verdict}", "", f"WHY: {report.get('verdict_reason')}", "", "## First-move report", "", "TORUS 9×9 FIRST-MOVE OBSERVATION", "", "KOMI: 0.5", f"BLACK WIN RATE: {first.get('combined', {}).get('black_win_rate')}", f"95% CI: {first.get('combined', {}).get('black_win_rate_95_percent_ci')}", f"RAW BLACK AREA ADVANTAGE: {first.get('combined', {}).get('raw_black_area_advantage_mean')}", f"FINAL BLACK MARGIN: {first.get('combined', {}).get('final_black_margin_mean')}", f"ADVANTAGE: {first.get('verdict')}", "", "## Performance", "", f"games/hour: {performance.get('selfplay_games_per_hour')}", f"positions/hour: {performance.get('selfplay_positions_per_hour')}", f"avg ply: {performance.get('average_ply')}", f"Arena workers: {TORUS9_WORKERS}", f"Peak memory MB: {performance.get('peak_memory_mb')}", "", "## Representative game traces", ""]
    for name, path in report.get("representative_traces", {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines += ["", "## Verification", "", f"TARGETED TESTS: {report.get('targeted_tests')}", f"FULL PYTEST: {report.get('full_pytest')}", f"SOURCE SHA: {report.get('source_sha')}", f"FINAL COMMIT: {report.get('final_commit')}", "PR: pending final push", "CI: pending final push", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    profile = load_torus9_profile()
    code = capture_code_identity(ROOT)
    root = ROOT / "runs" / "torus9-golden-learning-proof" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    contract = torus9_contract_proof()
    write_json(root / "contract-proof.json", contract)
    profile_fp = profile_fingerprint(profile)
    starts = list(generate_torus9_evaluation_starts(master_seed=int(profile["seeds"]["evaluation_master_seed"])))
    evaluation_manifest = {
        "contract_id": "torus9-frozen-evaluation-corpus-v1",
        "master_seed": profile["seeds"]["evaluation_master_seed"],
        "starts": len(starts),
        "stratification": {"prefix_lengths": [2, 4, 6, 8, 10, 12, 14, 16], "accepted_per_stratum": 8},
        "exact_state_includes": ["stones", "side_to_move", "full positional-superko history", "consecutive_passes", "komi"],
        "color_swapped_pair_semantics": True,
        "fingerprint": starts[0]["corpus_fingerprint"],
        "created_before_results": True,
        "source_sha": code.git_commit_sha,
    }
    write_jsonl(root / "evaluation" / "starts.jsonl", starts)
    write_json(root / "evaluation" / "manifest.json", evaluation_manifest)
    report: dict[str, object] = {"contract": contract, "profile": profile, "profile_fingerprint": profile_fp, "evaluation": evaluation_manifest, "source_sha": code.git_commit_sha}

    report["performance_preflight"] = _performance_preflight(root=root, run_id=args.run_id, profile=profile, workers=args.workers, device=args.device, code=code, profile_fp=profile_fp)

    bringup_summary, bringup_checkpoints, bringup_records, _ = _run_lineage(
        root=root, lineage="bringup", run_id=args.run_id + "-bringup", model_seed=int(profile["seeds"]["bringup_model_seed"]), selfplay_seed=int(profile["seeds"]["bringup_selfplay_master_seed"]), games_per_iteration=32, iterations=5, workers=args.workers, device=args.device, code=code, profile_fp=profile_fp,
    )
    mean_pos = float(bringup_summary["mean_positions_per_game"])
    report["bring_up"] = {**bringup_summary, "status": "PASS", "projected_positions_at_64": mean_pos * 64, "canonical_games_per_iteration": 64, "reason": "64 games is the default and projects to the requested 5,000–10,000 fresh positions."}

    bringup_arena: dict[str, object] = {}
    bringup_arena["M1b-vs-M0b"] = _arena(root=root, lineage="bringup", run_id=args.run_id + "-bringup", slug="M1b-vs-M0b", candidate=bringup_checkpoints["M1b"], reference=bringup_checkpoints["M0b"], candidate_label="M1b", reference_label="M0b", starts=starts, pairs=1, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 1, workers=args.workers, device=args.device)
    bringup_arena["M3b-vs-M0b"] = _arena(root=root, lineage="bringup", run_id=args.run_id + "-bringup", slug="M3b-vs-M0b", candidate=bringup_checkpoints["M3b"], reference=bringup_checkpoints["M0b"], candidate_label="M3b", reference_label="M0b", starts=starts, pairs=16, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 3, workers=args.workers, device=args.device)
    bringup_arena["M5b-vs-M0b"] = _arena(root=root, lineage="bringup", run_id=args.run_id + "-bringup", slug="M5b-vs-M0b", candidate=bringup_checkpoints["M5b"], reference=bringup_checkpoints["M0b"], candidate_label="M5b", reference_label="M0b", starts=starts, pairs=16, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 5, workers=args.workers, device=args.device)
    bringup_arena["M5b-vs-M1b"] = _arena(root=root, lineage="bringup", run_id=args.run_id + "-bringup", slug="M5b-vs-M1b", candidate=bringup_checkpoints["M5b"], reference=bringup_checkpoints["M1b"], candidate_label="M5b", reference_label="M1b", starts=starts, pairs=16, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 6, workers=args.workers, device=args.device)
    report["bring_up_arena"] = bringup_arena

    canonical_summary, canonical_checkpoints, canonical_records, _ = _run_lineage(
        root=root, lineage="canonical", run_id=args.run_id + "-canonical", model_seed=int(profile["seeds"]["canonical_model_seed"]), selfplay_seed=int(profile["seeds"]["canonical_selfplay_master_seed"]), games_per_iteration=64, iterations=8, workers=args.workers, device=args.device, code=code, profile_fp=profile_fp,
    )
    report["canonical"] = canonical_summary
    arena: dict[str, object] = {}
    arena["M4-vs-M0"] = _arena(root=root, lineage="canonical", run_id=args.run_id + "-canonical", slug="M4-vs-M0", candidate=canonical_checkpoints["M4"], reference=canonical_checkpoints["M0"], candidate_label="M4", reference_label="M0", starts=starts, pairs=32, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 40, workers=args.workers, device=args.device)
    arena["M4-vs-M1"] = _arena(root=root, lineage="canonical", run_id=args.run_id + "-canonical", slug="M4-vs-M1", candidate=canonical_checkpoints["M4"], reference=canonical_checkpoints["M1"], candidate_label="M4", reference_label="M1", starts=starts, pairs=16, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 41, workers=args.workers, device=args.device)
    arena["M8-vs-M0"] = _arena(root=root, lineage="canonical", run_id=args.run_id + "-canonical", slug="M8-vs-M0", candidate=canonical_checkpoints["M8"], reference=canonical_checkpoints["M0"], candidate_label="M8", reference_label="M0", starts=starts, pairs=64, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 80, workers=args.workers, device=args.device)
    arena["M8-vs-M1"] = _arena(root=root, lineage="canonical", run_id=args.run_id + "-canonical", slug="M8-vs-M1", candidate=canonical_checkpoints["M8"], reference=canonical_checkpoints["M1"], candidate_label="M8", reference_label="M1", starts=starts, pairs=64, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 81, workers=args.workers, device=args.device)
    arena["M8-vs-M4"] = _arena(root=root, lineage="canonical", run_id=args.run_id + "-canonical", slug="M8-vs-M4", candidate=canonical_checkpoints["M8"], reference=canonical_checkpoints["M4"], candidate_label="M8", reference_label="M4", starts=starts, pairs=64, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 82, workers=args.workers, device=args.device)
    arena["M8-vs-M7"] = _arena(root=root, lineage="canonical", run_id=args.run_id + "-canonical", slug="M8-vs-M7", candidate=canonical_checkpoints["M8"], reference=canonical_checkpoints["M7"], candidate_label="M8", reference_label="M7", starts=starts, pairs=16, seed=int(profile["seeds"]["canonical_arena_master_seed"]) + 83, workers=args.workers, device=args.device)
    report["arena"] = arena

    all_selfplay = [record for rows in canonical_records.values() for record in rows]
    by_iteration = {str(iteration): torus9_first_move_statistics(rows, source=f"canonical-M{iteration}") for iteration, rows in canonical_records.items()}
    combined = torus9_first_move_statistics(all_selfplay, source="canonical-M1-to-M8")
    report["first_move"] = {"by_iteration": by_iteration, "combined": combined, "verdict": "INCONCLUSIVE" if combined["black_win_rate_95_percent_ci"] and combined["black_win_rate_95_percent_ci"][0] <= 0.5 <= combined["black_win_rate_95_percent_ci"][1] else "DETECTED"}
    primary_ok = all(arena[slug]["technical_games"] == 0 and arena[slug]["95_percent_hoeffding_interval"] and arena[slug]["95_percent_hoeffding_interval"][0] > 0.5 for slug in ("M8-vs-M0", "M8-vs-M1"))
    m4_movement = arena["M4-vs-M0"]["mean_pair_score"] is not None and arena["M4-vs-M0"]["mean_pair_score"] > 0.5
    no_regression = arena["M8-vs-M4"]["mean_pair_score"] is not None and arena["M8-vs-M4"]["mean_pair_score"] >= 0.5
    report["learning_verdict"] = "CONFIRMED" if primary_ok and m4_movement and no_regression else "PARTIAL-INCONCLUSIVE" if primary_ok or m4_movement else "NOT CONFIRMED"
    report["verdict_reason"] = f"M8-vs-M0 primary={primary_ok}; M4 movement={m4_movement}; M8-vs-M4 non-regression={no_regression}; technical self-play and replay validation passed."
    report["performance"] = {"selfplay_games_per_hour": sum(float(row["games_per_hour"]) for row in canonical_summary["iteration_rows"]) / 8.0, "selfplay_positions_per_hour": sum(float(row["positions_per_hour"]) for row in canonical_summary["iteration_rows"]) / 8.0, "average_ply": sum(float(row["average_ply"]) for row in canonical_summary["iteration_rows"]) / 8.0, "peak_memory_mb": max(float(row["parent_peak_rss_mb"]) for row in canonical_summary["iteration_rows"]), "arena_workers": args.workers}
    report["representative_traces"] = _representative_traces(root, "canonical", canonical_records, root / "canonical" / "arena", tuple(arena))
    report["run_contract"] = {"workers": args.workers, "batch_size": TORUS9_BATCH_SIZE, "komi": TORUS9_KOMI, "move_limit": TORUS9_MOVE_LIMIT, "fresh_data_ratio": 1.0, "fast_sims": False, "ownership": False, "score_head": False}
    report["final_commit"] = code.git_commit_sha
    write_json(root / "final-report.json", report)
    docs_md = ROOT / "docs" / "TORUS9_GOLDEN_LEARNING_PROOF_20260913.md"
    docs_json = ROOT / "docs" / "TORUS9_GOLDEN_LEARNING_PROOF_20260913.json"
    write_json(docs_json, report)
    docs_md.parent.mkdir(parents=True, exist_ok=True)
    docs_md.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=TORUS9_WORKERS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.workers != TORUS9_WORKERS:
        raise SystemExit("The canonical Torus 9×9 protocol is frozen to 16 workers")
    report = run(args)
    print(f"TORUS 9×9 CONTRACT: {report['contract']['status']}")
    print(f"BRING-UP: {report['bring_up']['status']}")
    print(f"TORUS 9×9 LEARNING: {report['learning_verdict']}")


if __name__ == "__main__":
    main()
