#!/usr/bin/env python3
"""Run the bounded Cube4 Stage-9 production qualification.

All checkpoints, replay, self-play records, and Arena output are created under
one temporary root.  The script emits a compact JSON report and never writes a
production ``runs/cube4/active`` lineage unless the caller explicitly points
``--temporary-root`` there (which is intentionally rejected by the defaults in
the launcher documentation).
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Mapping

import torch

from gocube_golden.cube_arena_contract_v2 import CubeArenaSearchConfig
from gocube_golden.cube_m0_publisher import publish_cube_m0
from gocube_golden.cube_selfplay_contract import CubeSelfPlaySearchContract
from gocube_golden.cube_training_contract_v2 import CubeTrainingConfig
from gocube_golden.cube_training_v2 import load_cube_checkpoint
from gocube_golden.provenance import capture_code_identity
from tools.arena_engine import ArenaExecutionConfig
from tools.hardware_telemetry import HardwareTelemetry
from gocube_golden.cube_arena_v2 import run_cube_arena


SIZE = 4
GAMES = 64
SIMULATIONS = 64
WORKERS = 16
ACTIVE_GAMES_PER_WORKER = 4
ACTIVE_CONTEXTS = 64
SELFPLAY_BATCH_CAP = 64
SELFPLAY_BATCH_WAIT_MS = 1.0
ARENA_GAMES = 64
ARENA_GAMES_PER_WORKER = 12
ARENA_BATCH_ROWS = 64
ARENA_BATCH_WAIT_MS = 4.0
DEFAULT_SEED = 2026092301


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("production config must be a JSON object")
    return payload


def _training_config(config: Mapping[str, object]) -> CubeTrainingConfig:
    training = config.get("training")
    replay = config.get("replay")
    if not isinstance(training, Mapping) or not isinstance(replay, Mapping):
        raise ValueError("production config training/replay blocks are required")
    return CubeTrainingConfig(
        learning_rate=float(training["learning_rate"]),
        batch_size=int(training["batch_size"]),
        optimizer_steps=int(training["optimizer_steps"]),
        replay_generations=int(replay["generations"]),
        replay_cap=int(replay["cap"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def _selfplay_contract(move_limit: int) -> CubeSelfPlaySearchContract:
    return CubeSelfPlaySearchContract(
        simulations=SIMULATIONS,
        cpuct=1.25,
        fpu=0.0,
        root_noise=True,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.11,
        temperature_plies=(1, 8),
        temperature_after=0.0,
        resign=False,
        technical_move_limit=int(move_limit),
        komi=0.5,
    )


def _selfplay_execution() -> object:
    from gocube_golden.cube_selfplay_v2 import CubeSelfPlayExecutionConfig

    return CubeSelfPlayExecutionConfig(
        workers=WORKERS,
        active_games_per_worker=ACTIVE_GAMES_PER_WORKER,
        total_active_contexts=ACTIVE_CONTEXTS,
        inference_batch_cap=SELFPLAY_BATCH_CAP,
        inference_batch_wait_ms=SELFPLAY_BATCH_WAIT_MS,
        device="cuda",
        process_start_method="spawn",
    )


def _phase_metric(
    summary: Mapping[str, object], metric: str
) -> Mapping[str, object] | None:
    phases = summary.get("phases")
    if not isinstance(phases, Mapping):
        return None
    phase = phases.get("SELFPLAY")
    if not isinstance(phase, Mapping):
        return None
    value = phase.get(metric)
    return value if isinstance(value, Mapping) else None


def _selfplay_acceptance(result: Mapping[str, object]) -> dict[str, bool]:
    return {
        "self_play_games": int(result.get("games", -1)) == GAMES,
        "self_play_formal_games": int(result.get("formal_games", -1)) == GAMES,
        "self_play_technical_games": int(result.get("technical_games", -1)) == 0,
        "self_play_invalid_records": int(result.get("invalid_records", -1)) == 0,
        "self_play_worker_errors": not bool(result.get("worker_errors")),
        "self_play_central_inference_fatal": not bool(
            result.get("central_inference_fatal")
        ),
    }


def _arena_acceptance(result: Mapping[str, object]) -> dict[str, bool]:
    requested = int(result.get("games_requested", -1))
    expected_colors = {
        "candidate_black": ARENA_GAMES // 2,
        "candidate_white": ARENA_GAMES // 2,
    }
    return {
        "arena_games_requested": requested == ARENA_GAMES,
        "arena_games_valid": int(result.get("games_valid", -1)) == requested == ARENA_GAMES,
        "arena_technical_games": int(result.get("technical_games", -1)) == 0,
        "arena_invalid_games": int(result.get("invalid_games", -1)) == 0,
        "paired_colors": result.get("paired_colors") == expected_colors,
    }


def _qualification_acceptance(
    selfplay: Mapping[str, object], arena: Mapping[str, object]
) -> dict[str, bool]:
    checks = {**_selfplay_acceptance(selfplay), **_arena_acceptance(arena)}
    return {"passed": all(checks.values()), **checks}


def _require_acceptance(checks: Mapping[str, bool], label: str) -> None:
    if all(bool(value) for value in checks.values()):
        return
    failed = sorted(key for key, value in checks.items() if not bool(value))
    raise RuntimeError(f"{label} failed acceptance checks: {', '.join(failed)}")


def _run_selfplay(
    *,
    publication,
    training_config: CubeTrainingConfig,
    root: Path,
    seed: int,
    move_limits: tuple[int, ...],
) -> dict[str, object]:
    execution = _selfplay_execution()
    checkpoint = publication.root / publication.checkpoint.path
    replay = publication.root / publication.rolling_replay.path
    adapter, state, _ = load_cube_checkpoint(
        checkpoint,
        config=training_config,
        replay_path=replay,
        map_location="cuda",
        expected_size=SIZE,
    )
    selected: dict[str, object] | None = None
    for index, move_limit in enumerate(move_limits):
        telemetry: dict[str, object] = {}
        hardware_path = root / f"selfplay-{move_limit}-hardware.jsonl"
        hardware = HardwareTelemetry(hardware_path, interval_s=1.0)
        hardware.set_phase("SELFPLAY")
        hardware.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        records = ()
        try:
            from gocube_golden.cube_selfplay_v2 import run_cube_selfplay_games

            records = run_cube_selfplay_games(
                state.model,
                tuple(f"qualification-g{game_index:04d}" for game_index in range(GAMES)),
                size=SIZE,
                run_id="cube4-stage9-qualification",
                model_checkpoint_label="M0",
                checkpoint_artifact_hash=publication.checkpoint.sha256,
                master_seed=seed,
                profile_id="cube4-v2",
                profile_fingerprint=adapter.game_fingerprint,
                contract=_selfplay_contract(move_limit),
                device="cuda",
                execution_config=execution,
                inference_telemetry=telemetry,
                execution_activity=telemetry,
            )
        finally:
            hardware.stop()
        elapsed = max(0.0, time.perf_counter() - started)
        hardware_summary = hardware.summary()
        reasons = Counter(
            str(record.technical_termination)
            for record in records
            if record.technical_termination is not None
        )
        formal = sum(record.formal_result is not None for record in records)
        moves = sum(len(record.final_action_trace) for record in records)
        for record in records:
            record.validate(deep=True)
        samples = tuple(adapter.build_samples(records))
        peak_allocated = (
            float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else None
        )
        peak_reserved = (
            float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else None
        )
        selected = {
            "move_limit": move_limit,
            "games": len(records),
            "formal_games": formal,
            "technical_games": len(records) - formal,
            "technical_reasons": dict(sorted(reasons.items())),
            "invalid_records": 0,
            "replay_positions": len(samples),
            "moves": moves,
            "moves_per_game": moves / len(records) if records else 0.0,
            "wall_time_sec": elapsed,
            "games_per_hour": len(records) / elapsed * 3600.0 if elapsed else 0.0,
            "moves_per_hour": moves / elapsed * 3600.0 if elapsed else 0.0,
            "inference_calls": int(telemetry.get("inference_forwards", 0)),
            "inference_rows": int(telemetry.get("inference_rows", 0)),
            "batching": {
                key: telemetry.get(key)
                for key in (
                    "mean_inference_batch_rows",
                    "p50_inference_batch_rows",
                    "p95_inference_batch_rows",
                    "max_inference_batch_rows",
                )
            },
            "cpu_execution": {
                key: telemetry.get(key)
                for key in (
                    "effective_cpu_cores",
                    "process_tree_effective_cpu_cores",
                    "peak_concurrent_search_contexts",
                )
            },
            "peak_vram_mib": {
                "torch_allocated": peak_allocated,
                "torch_reserved": peak_reserved,
                "hardware_max": _phase_metric(
                    hardware_summary, "gpu_memory_used_mib"
                ),
            },
            "gpu_utilization": _phase_metric(
                hardware_summary, "gpu_util_percent"
            ),
            "hardware_telemetry": hardware_summary,
            "worker_errors": telemetry.get("worker_failures", []),
            "central_inference_fatal": telemetry.get("central_inference_fatal"),
        }
        retry_move_limit = (
            reasons
            and set(reasons) == {"MOVE_LIMIT"}
            and index < len(move_limits) - 1
        )
        if retry_move_limit:
            continue
        break
    if selected is None:
        raise RuntimeError("Cube self-play qualification produced no result")
    _require_acceptance(_selfplay_acceptance(selected), "Cube self-play qualification")
    return selected


def _run_arena(*, publication, root: Path, seed: int) -> dict[str, object]:
    config = ArenaExecutionConfig(
        games=ARENA_GAMES,
        workers=WORKERS,
        games_per_worker=ARENA_GAMES_PER_WORKER,
        inference_batch_rows=ARENA_BATCH_ROWS,
        inference_batch_wait_ms=ARENA_BATCH_WAIT_MS,
        device="cuda",
        strict_production=True,
        monitoring_acceptance=True,
        min_effective_cpu_cores=0.0,
    )
    reference = publication.checkpoint_reference()
    result = run_cube_arena(
        size=SIZE,
        candidate_checkpoint=reference,
        reference_checkpoint=reference,
        output_dir=root / "arena-qualification",
        search_config=CubeArenaSearchConfig(
            simulations=SIMULATIONS,
            cpuct=1.25,
            fpu=0.0,
            watchdog=1200,
        ),
        execution_config=config,
        seed=seed,
        temporary_root=root,
    )
    summary = json.loads((Path(result.output_dir) / "summary.json").read_text())
    games = [
        json.loads(line)
        for line in (Path(result.output_dir) / "games.jsonl").read_text().splitlines()
        if line.strip()
    ]
    telemetry = summary.get("telemetry", {})
    if not isinstance(telemetry, Mapping):
        telemetry = {}
    colors = Counter(bool(row.get("candidate_black")) for row in games)
    return {
        "games_requested": result.games_requested,
        "games_valid": result.games_valid,
        "technical_games": result.technical,
        "invalid_games": result.invalid,
        "candidate_wins": result.wins_a,
        "reference_wins": result.wins_b,
        "draws": result.draws,
        "paired_colors": {"candidate_black": colors[True], "candidate_white": colors[False]},
        "wall_time_sec": telemetry.get("wall_time_sec"),
        "games_per_hour": (
            result.games_requested / float(telemetry["wall_time_sec"]) * 3600.0
            if float(telemetry.get("wall_time_sec", 0.0)) > 0.0
            else 0.0
        ),
        "batching": {
            key: telemetry.get(key)
            for key in (
                "mean_inference_batch_rows",
                "p50_inference_batch_rows",
                "p95_inference_batch_rows",
                "max_inference_batch_rows",
                "cross_worker_inference_calls",
            )
        },
        "effective_cpu_cores": telemetry.get("effective_cpu_cores"),
        "peak_vram_mib": telemetry.get("gpu_memory_used_mib"),
        "gpu_utilization": telemetry.get("gpu_utilization_percent"),
        "performance_status": telemetry.get("performance_status"),
        "performance_warnings": telemetry.get("performance_warnings", []),
        "output": str(result.output_dir),
    }


def run_qualification(*, config_path: Path, output_path: Path | None, temporary_root: Path | None) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Cube4 qualification requires CUDA")
    config = _load_config(config_path)
    code = capture_code_identity()
    seed = int(config.get("extensions", {}).get("master_seed", DEFAULT_SEED)) if isinstance(config.get("extensions"), Mapping) else DEFAULT_SEED
    with tempfile.TemporaryDirectory(prefix="cube4-stage9-qualification-", dir=temporary_root) as raw_root:
        root = Path(raw_root)
        publication = publish_cube_m0(
            size=SIZE,
            lineage_id="cube4-stage9-qualification-m0",
            effective_config=config,
            seed=seed,
            runs_root=root / "runs",
            repo_root=Path(__file__).resolve().parents[1],
            code_identity=code,
            require_clean_code=False,
        )
        selfplay = _run_selfplay(
            publication=publication,
            training_config=_training_config(config),
            root=root,
            seed=seed,
            move_limits=(600, 1000, 1600),
        )
        arena = _run_arena(publication=publication, root=root, seed=seed + 1)
        technical_acceptance = _qualification_acceptance(selfplay, arena)
        _require_acceptance(technical_acceptance, "Cube4 production qualification")
        report = {
            "schema": "gocube-cube4-production-qualification-v1",
            "status": "PASS",
            "commit": code.git_commit_sha,
            "source": {
                "git_commit_sha": code.git_commit_sha,
                "git_tree_sha": code.git_tree_sha,
                "working_tree_clean": code.working_tree_clean,
            },
            "machine": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
            },
            "self_play_config": {
                "games": GAMES,
                "simulations": SIMULATIONS,
                "workers": WORKERS,
                "active_games_per_worker": ACTIVE_GAMES_PER_WORKER,
                "total_active_contexts": ACTIVE_CONTEXTS,
                "inference_batch_cap": SELFPLAY_BATCH_CAP,
                "inference_batch_wait_ms": SELFPLAY_BATCH_WAIT_MS,
                "device": "cuda",
                "process_start_method": "spawn",
            },
            "arena_config": {
                "games": ARENA_GAMES,
                "simulations": SIMULATIONS,
                "workers": WORKERS,
                "games_per_worker": ARENA_GAMES_PER_WORKER,
                "inference_batch_rows": ARENA_BATCH_ROWS,
                "inference_batch_wait_ms": ARENA_BATCH_WAIT_MS,
                "device": "cuda",
                "monitoring_acceptance": True,
            },
            "self_play": selfplay,
            "arena": arena,
            "technical_acceptance": technical_acceptance,
        }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gocube/cube4_production_stage9_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args(argv)
    report = run_qualification(
        config_path=args.config,
        output_path=args.output,
        temporary_root=args.temporary_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
