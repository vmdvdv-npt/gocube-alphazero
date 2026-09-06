from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from argparse import Namespace
from pathlib import Path

import torch

from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.train import GoCubeCoach, build_training_args


DEFAULT_CHECKPOINT = "checkpoint/c4-komi05-d5-20260906-110926/iteration-0006.pkl"
DEFAULT_RUN_NAME = "c4-komi05-iter6-score-aware-recovery"
EXPECTED_KOMI = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run score-aware GoCube self-play validation from an archived checkpoint "
            "without training or modifying the source run."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--topology", choices=("cube", "torus"), default="cube")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--fast-sims", type=int, default=20)
    parser.add_argument("--fast-prob", type=float, default=0.75)
    parser.add_argument("--inference-batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-root", default="diagnostics")
    parser.add_argument("--record-games", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for recovery validation but torch.cuda.is_available() is false")
    return value


def build_recovery_args(cli) -> tuple[type, object]:
    if cli.games < 1:
        raise ValueError("games must be at least 1")
    if cli.workers < 1:
        raise ValueError("workers must be at least 1")
    if cli.sims < 1 or cli.fast_sims < 1:
        raise ValueError("search sims must be at least 1")
    if not 0.0 <= cli.fast_prob <= 1.0:
        raise ValueError("fast-prob must be between 0 and 1")

    training_cli = Namespace(
        topology=cli.topology,
        size=cli.size,
        workers=cli.workers,
        sims=cli.sims,
        arena_sims=cli.sims,
        games_per_iteration=cli.games,
        iterations=1,
        train_batch_size=256,
        train_steps_per_iteration=1,
        fast_game_prob=cli.fast_prob,
        endgame_sample_weight=1,
        inference_batch_wait_ms=cli.inference_batch_wait_ms,
        no_arena=True,
        model_gating=False,
        smoke=False,
        run_name=cli.run_name,
    )
    game_cls, args = build_training_args(training_cli)
    if not math.isclose(float(game_cls.KOMI), EXPECTED_KOMI, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Recovery validation requires GoCube komi {EXPECTED_KOMI}, got {game_cls.KOMI}")

    args.numFastSims = int(cli.fast_sims)
    args.probFastSim = float(cli.fast_prob)
    args.process_batch_size = max(1, math.ceil(cli.games / cli.workers))
    args.load_model = False
    args.numWarmupIters = 0
    args.startIter = 1
    args.numIters = 1
    args.compareWithBaseline = False
    args.compareWithPast = False
    args.model_gating = False
    args.gocube_validation_only = True
    args.gocube_recording_enabled = bool(cli.record_games)
    args.gocube_search_audit_probability = 1.0
    args.data = str(cli.output_root)
    return game_cls, args


def run_recovery_validation(cli) -> dict[str, object]:
    checkpoint = Path(cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Historical checkpoint not found: {checkpoint}")

    device = _resolve_device(cli.device)
    game_cls, args = build_recovery_args(cli)
    args.cuda = device == "cuda"

    source_stat = checkpoint.stat()
    source_hash = _sha256(checkpoint)
    network = NNetWrapper.from_checkpoint(
        game_cls,
        folder=str(checkpoint.parent),
        filename=checkpoint.name,
        use_saved_args=True,
        device=device,
        load_training_state=False,
        allow_legacy_search_contract=True,
    )

    saved_komi = getattr(network.args, "gocube_komi", EXPECTED_KOMI)
    if not math.isclose(float(saved_komi), EXPECTED_KOMI, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Checkpoint komi {saved_komi} does not match required recovery komi {EXPECTED_KOMI}"
        )

    probe = network.predict_for_search(game_cls().observation())
    if probe.score is None or probe.ownership is None:
        raise RuntimeError("Historical checkpoint does not expose score and ownership heads required by recovery")

    coach = GoCubeCoach(game_cls, network, args)
    iteration = 1
    killed = False
    try:
        coach.generateSelfPlayAgents()
        coach.processSelfPlayBatches(iteration)
        if coach.file_queue.qsize() != 0:
            raise RuntimeError("validation-only self-play unexpectedly produced training samples")

        coach._iteration_record_context = {
            "run_name": args.run_name,
            "iteration": iteration,
            "checkpoint": {
                "id": f"archived@{checkpoint.name}",
                "iteration": 6,
                "path": os.path.relpath(checkpoint, Path.cwd()),
                "model_role": "historical_recovery_source",
            },
            "parameters": {
                "topology": game_cls.topology_kind(),
                "size": game_cls.board_size(),
                "komi": float(game_cls.KOMI),
                "games": int(cli.games),
                "workers": int(cli.workers),
                "regular_sims": int(cli.sims),
                "fast_sims": int(cli.fast_sims),
                "fast_probability": float(cli.fast_prob),
                "search_contract": args.gocube_search_contract,
                "search_utility_mode": args.search_utility_mode,
                "validation_only": True,
            },
        }
        coach.processGameResults(iteration)
        guard = coach._selfplay_guard_result
        telemetry = dict(coach._iteration_telemetry)
        manifest_path = Path(coach._iteration_record_manifest_path).resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            iteration_manifest = json.load(handle)
    finally:
        if coach.agents:
            coach.killSelfPlayAgents()
            killed = True
        coach.writer.close()

    after_stat = checkpoint.stat()
    after_hash = _sha256(checkpoint)
    unchanged = (
        source_hash == after_hash
        and source_stat.st_size == after_stat.st_size
        and source_stat.st_mtime_ns == after_stat.st_mtime_ns
    )
    if not unchanged:
        raise RuntimeError("Historical checkpoint changed during validation")

    report = {
        "schema": "gocube-score-aware-recovery-v1",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": source_hash,
            "size_bytes": source_stat.st_size,
            "unchanged": unchanged,
        },
        "game": {
            "topology": game_cls.topology_kind(),
            "size": game_cls.board_size(),
            "komi": float(game_cls.KOMI),
            "rules_fingerprint": game_cls.rules_fingerprint(),
        },
        "self_play": {
            "games": int(cli.games),
            "workers": int(cli.workers),
            "regular_sims": int(cli.sims),
            "fast_sims": int(cli.fast_sims),
            "fast_probability": float(cli.fast_prob),
            "search_audit_probability": 1.0,
            "training_samples_generated": 0,
            "optimizer_steps": 0,
        },
        "search": {
            "contract": args.gocube_search_contract,
            "utility_mode": args.search_utility_mode,
        },
        "guard": {
            "status": guard.status,
            "training_allowed": guard.training_allowed,
            "warnings": list(guard.warnings),
            "fatal_reasons": list(guard.fatal_reasons),
            "metrics": guard.metrics,
        },
        "telemetry": telemetry,
        "iteration_manifest": str(manifest_path),
        "aggregate_metrics": iteration_manifest.get("aggregate_metrics", {}),
        "agents_joined": killed,
    }

    output_dir = Path(cli.output_root) / cli.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "recovery-report.json"
    temporary = report_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, report_path)
    report["report_path"] = str(report_path.resolve())
    return report


def main():
    cli = parse_args()
    report = run_recovery_validation(cli)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["guard"]["training_allowed"] else 2)


if __name__ == "__main__":
    main()
