#!/usr/bin/env python3
"""Run the frozen Torus Golden Arena process-parallel parity proof.

This runner performs evaluation only.  It never trains, regenerates replay,
or changes the existing search path.  Every parallel game is compared with a
same-commit sequential-oracle game before its summary is accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.arena import GameRecord, SequentialGoldenArena, write_records_jsonl
from gocube_golden.arena_process import (
    CheckpointPlayerSpec,
    PairTask,
    ProcessParallelGoldenArena,
)
from gocube_golden.neural import GoldenGraphNetV1, GoldenNeuralEvaluator
from gocube_golden.provenance import capture_code_identity, derive_seed, file_sha256
from gocube_golden.stage4 import (
    STAGE4_ARENA_MASTER_SEED,
    diagnostic_subset,
    load_frozen_starts,
    state_from_start_row,
    summarize_pair_records,
)
from gocube_golden.training import load_checkpoint
from tools.torus_golden_stage4 import checkpoint_info, make_checkpoint_player


FROZEN_RUN = ROOT / "runs" / "torus-golden-stage4" / "torus-golden-stage4-seed2-parity-v8"
EXPECTED = {
    "M1-M0": (91, 37, 0),
    "M4-M0": (124, 4, 0),
    "M4-M1": (113, 15, 0),
    "M2-M1": (25, 7, 0),
    "M3-M2": (19, 13, 0),
    "M4-M3": (18, 14, 0),
}
SPECS = (
    ("M1-M0", "M1", "M0", 64),
    ("M4-M0", "M4", "M0", 64),
    ("M4-M1", "M4", "M1", 64),
    ("M2-M1", "M2", "M1", 16),
    ("M3-M2", "M3", "M2", 16),
    ("M4-M3", "M4", "M3", 16),
)
SEMANTIC_FIELDS = (
    "pair_id",
    "game_id",
    "player_A_id",
    "player_B_id",
    "black_player",
    "white_player",
    "master_seed",
    "seed_game",
    "seed_A",
    "seed_B",
    "start_state_key",
    "start_history",
    "start_trace",
    "action_trace",
    "final_board",
    "absolute_rule_result",
    "mapped_result",
    "black_area",
    "white_area",
    "margin_black",
    "termination_reason",
    "error_details",
)


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cpu_seconds(usage: resource.struct_rusage) -> float:
    return float(usage.ru_utime + usage.ru_stime)


def _first_trace_divergence(left: GameRecord, right: GameRecord) -> dict[str, object] | None:
    for index, (a, b) in enumerate(zip(left.action_trace, right.action_trace)):
        if a != b:
            return {
                "trace_index": index,
                "ply": min(a.ply, b.ply),
                "sequential": _jsonable(asdict(a)),
                "parallel": _jsonable(asdict(b)),
            }
    if len(left.action_trace) != len(right.action_trace):
        index = min(len(left.action_trace), len(right.action_trace))
        return {
            "trace_index": index,
            "ply": index + 1,
            "sequential_trace_length": len(left.action_trace),
            "parallel_trace_length": len(right.action_trace),
        }
    return None


def _compare_records(sequential: Sequence[GameRecord], parallel: Sequence[GameRecord]) -> dict[str, object]:
    if len(sequential) != len(parallel):
        raise RuntimeError(f"Sequential/parallel game count mismatch: {len(sequential)} != {len(parallel)}")
    for index, (left, right) in enumerate(zip(sequential, parallel)):
        for field in SEMANTIC_FIELDS:
            if getattr(left, field) != getattr(right, field):
                return {
                    "identical": False,
                    "game_index": index,
                    "game_id": left.game_id,
                    "field": field,
                    "sequential": _jsonable(getattr(left, field)),
                    "parallel": _jsonable(getattr(right, field)),
                    "first_trace_divergence": _first_trace_divergence(left, right),
                }
    return {
        "identical": True,
        "games": len(sequential),
        "first_trace_divergence": None,
    }


def _make_sequential_players(
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    candidate_label: str,
    reference_label: str,
):
    device = torch.device("cpu")
    candidate_model = GoldenGraphNetV1().to(device)
    reference_model = GoldenGraphNetV1().to(device)
    load_checkpoint(
        Path(str(candidate["path"])),
        model=candidate_model,
        expected={"model_hash": candidate["model_hash"]},
        device=device,
    )
    load_checkpoint(
        Path(str(reference["path"])),
        model=reference_model,
        expected={"model_hash": reference["model_hash"]},
        device=device,
    )
    candidate_evaluator = GoldenNeuralEvaluator(candidate_model, device=device)
    reference_evaluator = GoldenNeuralEvaluator(reference_model, device=device)
    return (
        make_checkpoint_player(candidate_label, candidate, candidate_model, candidate_evaluator),
        make_checkpoint_player(reference_label, reference, reference_model, reference_evaluator),
    )


def _pairs(slug: str, starts: Sequence[Mapping[str, object]]) -> tuple[PairTask, ...]:
    return tuple(
        PairTask(
            pair_id=f"{slug}--{row['start_id']}",
            start_state=state_from_start_row(row),
            start_trace=tuple(int(action) for action in row["trace"]),
        )
        for row in starts
    )


def _one_comparison(
    *,
    slug: str,
    candidate_label: str,
    reference_label: str,
    pair_count: int,
    starts: Sequence[Mapping[str, object]],
    code,
    workers: int,
    run_dir: Path,
    require_expected_result: bool = True,
) -> dict[str, object]:
    checkpoint_dir = FROZEN_RUN / "checkpoints"
    candidate = checkpoint_info(checkpoint_dir / f"{candidate_label}.pt", label=candidate_label, device=torch.device("cpu"))
    reference = checkpoint_info(checkpoint_dir / f"{reference_label}.pt", label=reference_label, device=torch.device("cpu"))
    arena_seed = derive_seed(STAGE4_ARENA_MASTER_SEED, slug)
    run_id = f"golden-arena-process-parity-20260913-{slug.lower()}"
    pair_tasks = _pairs(slug, starts)
    sequential_A, sequential_B = _make_sequential_players(
        candidate,
        reference,
        candidate_label=candidate_label,
        reference_label=reference_label,
    )
    sequential = SequentialGoldenArena(
        master_seed=arena_seed,
        run_id=run_id,
        code_identity=code,
    )
    seq_started = time.perf_counter()
    sequential_records: list[GameRecord] = []
    for pair in pair_tasks:
        sequential_records.extend(
            sequential.play_pair(
                pair_id=pair.pair_id,
                player_A=sequential_A,
                player_B=sequential_B,
                start_state=pair.start_state,
                start_trace=pair.start_trace,
            )
        )
    sequential_wall = time.perf_counter() - seq_started
    sequential_records_tuple = tuple(sequential_records)

    candidate_spec = CheckpointPlayerSpec.from_checkpoint(
        candidate["path"],
        player_id=candidate_label,
        device="cpu",
        metadata=candidate["metadata"],
        artifact_sha256=candidate["artifact_sha256"],
    )
    reference_spec = CheckpointPlayerSpec.from_checkpoint(
        reference["path"],
        player_id=reference_label,
        device="cpu",
        metadata=reference["metadata"],
        artifact_sha256=reference["artifact_sha256"],
    )
    parallel = ProcessParallelGoldenArena(
        player_A=candidate_spec,
        player_B=reference_spec,
        workers=workers,
        master_seed=arena_seed,
        run_id=run_id,
        code_identity=code,
        mp_context="spawn",
    )
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parallel_started = time.perf_counter()
    parallel_records = parallel.play_pairs(pair_tasks)
    parallel_wall = time.perf_counter() - parallel_started
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    parity = _compare_records(sequential_records_tuple, parallel_records)
    if not parity["identical"]:
        raise RuntimeError(f"Arena {slug} is not bit-exact: {parity}")
    sequential_summary = summarize_pair_records(
        sequential_records_tuple,
        candidate_label=candidate_label,
        reference_label=reference_label,
        starts=starts,
    )
    parallel_summary = summarize_pair_records(
        parallel_records,
        candidate_label=candidate_label,
        reference_label=reference_label,
        starts=starts,
    )
    if require_expected_result and tuple(sequential_summary["W/L/D"]) != EXPECTED[slug]:
        _write_json(run_dir / f"{slug.lower()}-aggregate-drift.json", {
            "sequential": sequential_summary,
            "parallel": parallel_summary,
        })
        raise RuntimeError(f"Arena {slug} aggregate result drift: {parallel_summary}")
    if parallel_summary != sequential_summary:
        differing = {
            key: {"sequential": sequential_summary[key], "parallel": parallel_summary[key]}
            for key in sequential_summary
            if sequential_summary[key] != parallel_summary.get(key)
        }
        _write_json(run_dir / f"{slug.lower()}-summary-diff.json", differing)
        raise RuntimeError(f"Arena {slug} aggregate result drift: {parallel_summary}")
    if sequential_summary["technical"] != 0:
        raise RuntimeError(f"Arena {slug} contains technical games")
    comparison_dir = run_dir / slug.lower()
    write_records_jsonl(comparison_dir / "sequential-games.jsonl", sequential_records_tuple)
    write_records_jsonl(comparison_dir / "parallel-games.jsonl", parallel_records)
    _write_json(comparison_dir / "parity.json", parity)
    _write_json(comparison_dir / "sequential-summary.json", sequential_summary)
    _write_json(comparison_dir / "parallel-summary.json", parallel_summary)
    child_cpu = _cpu_seconds(child_after) - _cpu_seconds(child_before)
    return {
        "slug": slug,
        "candidate": candidate_label,
        "reference": reference_label,
        "pairs": pair_count,
        "games": pair_count * 2,
        "technical_games": 0,
        "expected_W/L/D": EXPECTED[slug],
        "actual_W/L/D": parallel_summary["W/L/D"],
        "sequential_wall_time_sec": sequential_wall,
        "parallel_wall_time_sec": parallel_wall,
        "speedup": sequential_wall / parallel_wall if parallel_wall else None,
        "parallel_child_cpu_time_sec": child_cpu,
        "effective_parallelism": child_cpu / parallel_wall if parallel_wall else None,
        "worker_utilization_percent": (
            100.0 * child_cpu / (parallel_wall * workers) if parallel_wall else None
        ),
        "parity": parity,
        "output_dir": str(comparison_dir),
    }


def _worker_count_invariance(
    *,
    starts: Sequence[Mapping[str, object]],
    code,
    output_dir: Path,
) -> dict[str, object]:
    """Compare several worker counts on the same two frozen start pairs."""

    checkpoint_dir = FROZEN_RUN / "checkpoints"
    candidate = checkpoint_info(checkpoint_dir / "M1.pt", label="M1", device=torch.device("cpu"))
    reference = checkpoint_info(checkpoint_dir / "M0.pt", label="M0", device=torch.device("cpu"))
    pair_tasks = _pairs("M1-M0", tuple(starts[:2]))
    arena_seed = derive_seed(STAGE4_ARENA_MASTER_SEED, "M1-M0")
    run_id = "golden-arena-worker-count-invariance-20260913"
    sequential_A, sequential_B = _make_sequential_players(
        candidate,
        reference,
        candidate_label="M1",
        reference_label="M0",
    )
    sequential = SequentialGoldenArena(
        master_seed=arena_seed,
        run_id=run_id,
        code_identity=code,
    )
    sequential_records: list[GameRecord] = []
    for pair in pair_tasks:
        sequential_records.extend(
            sequential.play_pair(
                pair_id=pair.pair_id,
                player_A=sequential_A,
                player_B=sequential_B,
                start_state=pair.start_state,
                start_trace=pair.start_trace,
            )
        )
    oracle = tuple(sequential_records)
    rows: dict[str, object] = {}
    for worker_count in (1, 2, 4, 8, 16):
        candidate_spec = CheckpointPlayerSpec.from_checkpoint(
            candidate["path"],
            player_id="M1",
            device="cpu",
            metadata=candidate["metadata"],
            artifact_sha256=candidate["artifact_sha256"],
        )
        reference_spec = CheckpointPlayerSpec.from_checkpoint(
            reference["path"],
            player_id="M0",
            device="cpu",
            metadata=reference["metadata"],
            artifact_sha256=reference["artifact_sha256"],
        )
        parallel = ProcessParallelGoldenArena(
            player_A=candidate_spec,
            player_B=reference_spec,
            workers=worker_count,
            master_seed=arena_seed,
            run_id=run_id,
            code_identity=code,
            mp_context="spawn",
        )
        started = time.perf_counter()
        records = parallel.play_pairs(pair_tasks)
        elapsed = time.perf_counter() - started
        parity = _compare_records(oracle, records)
        if not parity["identical"]:
            raise RuntimeError(
                f"Worker-count invariance failed for workers={worker_count}: {parity}"
            )
        worker_dir = output_dir / "worker-count-invariance" / f"workers-{worker_count}"
        write_records_jsonl(worker_dir / "parallel-games.jsonl", records)
        _write_json(worker_dir / "parity.json", parity)
        rows[str(worker_count)] = {
            "games": len(records),
            "identical_to_sequential": True,
            "parallel_wall_time_sec": elapsed,
            "first_trace_divergence": None,
        }
    return {
        "games": len(oracle),
        "pairs": len(pair_tasks),
        "worker_counts": rows,
        "all_identical": True,
    }


def run(
    *,
    workers: int,
    output_dir: Path,
    limit_pairs: int | None = None,
    only: str | None = None,
) -> dict[str, object]:
    if workers != 16 and limit_pairs is None:
        raise ValueError("The canonical Golden Arena proof requires workers=16")
    if limit_pairs is not None and limit_pairs <= 0:
        raise ValueError("limit_pairs must be positive")
    torch.set_num_threads(1)
    code = capture_code_identity()
    starts = load_frozen_starts(FROZEN_RUN)
    subset = diagnostic_subset(starts)
    full_proof = limit_pairs is None
    rows: list[dict[str, object]] = []
    selected_specs = tuple(item for item in SPECS if only is None or item[0] == only)
    if not selected_specs:
        raise ValueError(f"Unknown comparison slug: {only}")
    for slug, candidate, reference, pair_count in selected_specs:
        corpus = starts if pair_count == 64 else subset
        if limit_pairs is not None:
            corpus = corpus[:limit_pairs]
        rows.append(
            _one_comparison(
                slug=slug,
                candidate_label=candidate,
                reference_label=reference,
                pair_count=len(corpus),
                starts=corpus,
                code=code,
                workers=workers,
                run_dir=output_dir,
                require_expected_result=full_proof,
            )
        )
    worker_count_report = _worker_count_invariance(
        starts=starts,
        code=code,
        output_dir=output_dir,
    )
    sequential_wall = sum(float(row["sequential_wall_time_sec"]) for row in rows)
    parallel_wall = sum(float(row["parallel_wall_time_sec"]) for row in rows)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {
        "status": "PASS" if full_proof else "SMOKE_PASS",
        "source": {
            "git_commit": code.git_commit_sha,
            "git_tree": code.git_tree_sha,
            "git_worktree_clean": code.working_tree_clean,
            "frozen_run": str(FROZEN_RUN),
            "checkpoint_identity": "torus-golden-stage4-seed2-parity-v8",
            "training_run": False,
        },
        "architecture": {
            "executor": "ProcessParallelGoldenArena",
            "worker_model": "one OS process / one active Arena game",
            "single_game_path": "SequentialGoldenArena.play_game",
            "search_changes": False,
            "inference_batch_size": 1,
            "inference_batching": False,
            "max_workers": workers,
            "canonical_order": "pair input order -> A-black -> B-black",
            "completion_order_persisted": False,
        },
        "parity": {
            "sequential_vs_parallel_games": sum(int(row["games"]) for row in rows),
            "identical_games": sum(int(row["games"]) for row in rows),
            "divergent_games": 0,
            "first_divergent_ply": None,
            "pair_color_swap_parity": "PASS",
            "worker_count_invariance": worker_count_report,
            "technical_games": 0,
        },
        "comparisons": rows,
        "timing": {
            "sequential_wall_time_sec": sequential_wall,
            "parallel_16_wall_time_sec": parallel_wall,
            "speedup": sequential_wall / parallel_wall if parallel_wall else None,
            "parallel_child_cpu_time_sec": sum(float(row["parallel_child_cpu_time_sec"]) for row in rows),
            "effective_parallelism": (
                sum(float(row["parallel_child_cpu_time_sec"]) for row in rows) / parallel_wall
                if parallel_wall
                else None
            ),
            "worker_utilization_percent": (
                100.0
                * sum(float(row["parallel_child_cpu_time_sec"]) for row in rows)
                / (parallel_wall * workers)
                if parallel_wall
                else None
            ),
            "process_startup_load_overhead": "included in parallel wall time; persistent pool loads each checkpoint once per worker per comparison",
        },
        "memory": {
            "peak_parent_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
            "peak_child_rss_mb": float(child_usage.ru_maxrss) / 1024.0,
            "measurement": "POSIX getrusage; child value is peak of child processes",
        },
        "checkpoint_artifacts": {
            label: file_sha256(FROZEN_RUN / "checkpoints" / f"{label}.pt")
            for label in ("M0", "M1", "M2", "M3", "M4")
        },
    }
    _write_json(output_dir / "benchmark.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--limit-pairs",
        type=int,
        help="development smoke limit per comparison; omitting it runs the 480-game proof",
    )
    parser.add_argument("--only", choices=tuple(item[0] for item in SPECS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "golden-arena-process-parallel" / "torus-golden-arena-process-parity-20260913",
    )
    args = parser.parse_args()
    report = run(
        workers=args.workers,
        output_dir=args.output_dir,
        limit_pairs=args.limit_pairs,
        only=args.only,
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
