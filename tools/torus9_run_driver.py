#!/usr/bin/env python3
"""Torus9 adapter driven exclusively by the immutable lineage run spec.

No worker count, batch/wait choice, Arena workload, seed, reference gap,
heartbeat period, or performance threshold is selected in this module.
Scientific Torus9 rules/search semantics remain owned by the referenced
canonical profile and the Torus9 Arena profile, not by the orchestrator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.run_spec import RUN_SPEC_SCHEMA, StrictRunSpec, run_spec_fingerprint
from gocube_golden.provenance import file_sha256
from gocube_golden.run_storage import evaluation_dir
from tools.arena_engine import ArenaExecutionConfig, run_arena as run_arena_engine
from tools.arena_profiles.torus9 import PROFILE as TORUS9_ARENA_PROFILE
import tools.torus9_orchestrator_driver as legacy


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _required(mapping: Mapping[str, object], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing explicit fields: {', '.join(missing)}")


def _positive_int(value: object, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_float(value: object, label: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _load_run_spec() -> StrictRunSpec:
    raw_path = os.environ.get("AZ_RUN_SPEC_PATH")
    expected = os.environ.get("AZ_RUN_SPEC_FINGERPRINT")
    if not raw_path or not expected:
        raise ValueError("Torus9 run driver requires AZ_RUN_SPEC_PATH and AZ_RUN_SPEC_FINGERPRINT")
    spec = StrictRunSpec.load(Path(raw_path), repo_root=ROOT)
    if spec.payload.get("schema") != RUN_SPEC_SCHEMA or spec.fingerprint != expected:
        raise ValueError("Immutable run-spec identity mismatch")
    if spec.orchestrator_spec.topology != "torus9":
        raise ValueError("Torus9 driver requires topology=torus9")
    return spec


def _generation_config(spec: StrictRunSpec) -> dict[str, object]:
    generation = _mapping(spec.payload["generation"], "generation")
    config = _mapping(generation["driver_config"], "generation.driver_config")
    _required(
        config,
        (
            "games",
            "device",
            "workers",
            "active_games_per_worker",
            "total_active_contexts",
            "inference_batch_cap",
            "inference_batch_wait_ms",
            "coalescing",
            "heartbeat_interval_seconds",
            "model_init_seed",
            "selfplay_master_seed",
            "training_master_seed",
        ),
        "generation.driver_config",
    )
    games = _positive_int(config["games"], "generation.games")
    workers = _positive_int(config["workers"], "generation.workers")
    active = _positive_int(config["active_games_per_worker"], "generation.active_games_per_worker")
    contexts = _positive_int(config["total_active_contexts"], "generation.total_active_contexts")
    cap = _positive_int(config["inference_batch_cap"], "generation.inference_batch_cap")
    wait = _nonnegative_float(config["inference_batch_wait_ms"], "generation.inference_batch_wait_ms")
    heartbeat = float(config["heartbeat_interval_seconds"])
    if heartbeat <= 0:
        raise ValueError("generation.heartbeat_interval_seconds must be positive")
    if contexts > workers * active:
        raise ValueError("generation.total_active_contexts exceeds configured lane capacity")
    if not isinstance(config["coalescing"], bool):
        raise ValueError("generation.coalescing must be boolean")
    device = str(config["device"]).strip()
    if not device:
        raise ValueError("generation.device must be explicit")
    return {
        **config,
        "games": games,
        "workers": workers,
        "active_games_per_worker": active,
        "total_active_contexts": contexts,
        "inference_batch_cap": cap,
        "inference_batch_wait_ms": wait,
        "heartbeat_interval_seconds": heartbeat,
        "device": device,
        "model_init_seed": int(config["model_init_seed"]),
        "selfplay_master_seed": int(config["selfplay_master_seed"]),
        "training_master_seed": int(config["training_master_seed"]),
    }


def _arena_config(spec: StrictRunSpec) -> tuple[dict[str, object], dict[str, object]]:
    arena = _mapping(spec.payload["arena"], "arena")
    if arena.get("enabled") is not True:
        raise ValueError("Arena driver invoked while arena.enabled is false")
    config = _mapping(arena["driver_config"], "arena.driver_config")
    startset = _mapping(arena["startset"], "arena.startset")
    _required(config, ("reference_gap", "games", "master_seed", "heartbeat_interval_seconds", "execution"), "arena.driver_config")
    execution = _mapping(config["execution"], "arena.driver_config.execution")
    _required(execution, ("workers", "games_per_worker", "inference_batch_rows", "inference_batch_wait_ms", "device", "strict_production"), "arena.driver_config.execution")
    games = _positive_int(config["games"], "arena.games")
    if games % 2:
        raise ValueError("arena.games must be even for paired color-swapped starts")
    reference_gap = _positive_int(config["reference_gap"], "arena.reference_gap")
    heartbeat = float(config["heartbeat_interval_seconds"])
    if heartbeat <= 0:
        raise ValueError("arena.heartbeat_interval_seconds must be positive")
    normalized_execution = {
        "workers": _positive_int(execution["workers"], "arena.execution.workers"),
        "games_per_worker": _positive_int(execution["games_per_worker"], "arena.execution.games_per_worker"),
        "inference_batch_rows": _positive_int(execution["inference_batch_rows"], "arena.execution.inference_batch_rows"),
        "inference_batch_wait_ms": _nonnegative_float(execution["inference_batch_wait_ms"], "arena.execution.inference_batch_wait_ms"),
        "device": str(execution["device"]).strip(),
        "strict_production": execution["strict_production"],
    }
    if not normalized_execution["device"]:
        raise ValueError("arena.execution.device must be explicit")
    if not isinstance(normalized_execution["strict_production"], bool):
        raise ValueError("arena.execution.strict_production must be boolean")
    _required(startset, ("master_seed", "pairs"), "arena.startset")
    master_seed = int(config["master_seed"])
    if int(startset["master_seed"]) != master_seed:
        raise ValueError("arena.startset.master_seed must match arena.driver_config.master_seed")
    if int(startset["pairs"]) != games // 2:
        raise ValueError("arena.startset.pairs must equal arena.games / 2")
    return (
        {**config, "reference_gap": reference_gap, "games": games, "master_seed": master_seed, "heartbeat_interval_seconds": heartbeat, "execution": normalized_execution},
        startset,
    )


def _validate_device(device: str) -> None:
    parsed = torch.device(device)
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run spec requested CUDA but CUDA is unavailable")


def run_generation(args: argparse.Namespace) -> dict[str, object]:
    spec = _load_run_spec()
    config = _generation_config(spec)
    _validate_device(str(config["device"]))

    profile = legacy._load_profile(Path(os.environ["AZ_PROFILE_PATH"]), os.environ["AZ_PROFILE_FINGERPRINT"])
    self_play = _mapping(profile.get("self_play"), "profile.self_play")
    actual_games = int(self_play.get("games_per_iteration", -1))
    if actual_games != int(config["games"]):
        raise ValueError(
            "Run-spec generation.games must match the referenced scientific profile; "
            "use a different explicitly fingerprinted profile when changing scientific workload"
        )

    original_validate = legacy._validate_execution
    original_heartbeat = legacy._Heartbeat
    original_selfplay = legacy.run_torus9_selfplay_games
    original_model_seed = legacy.TORUS9_CURRENT_MODEL_INIT_SEED
    original_selfplay_seed = legacy.TORUS9_CURRENT_SELFPLAY_MASTER_SEED
    original_training_seed = legacy.TORUS9_CURRENT_TRAINING_MASTER_SEED
    heartbeat_interval = float(config["heartbeat_interval_seconds"])

    class BoundHeartbeat(original_heartbeat):
        def __init__(self, path: Path, generation: int, interval: float | None = None) -> None:
            super().__init__(path, generation, interval=heartbeat_interval)

    def validate_execution(namespace: argparse.Namespace, profile_value: Mapping[str, object]) -> None:
        del profile_value
        for label, value in (
            ("workers", namespace.workers),
            ("active_games_per_worker", namespace.active_games_per_worker),
            ("total_active_contexts", namespace.total_active_contexts),
            ("batch_cap", namespace.batch_cap),
        ):
            _positive_int(value, f"generation.{label}")
        _nonnegative_float(namespace.wait_ms, "generation.wait_ms")
        _validate_device(str(namespace.device))

    def selfplay_bound(*call_args: object, **call_kwargs: object):
        call_kwargs["coalescing"] = bool(config["coalescing"])
        return original_selfplay(*call_args, **call_kwargs)

    try:
        legacy._validate_execution = validate_execution
        legacy._Heartbeat = BoundHeartbeat
        legacy.run_torus9_selfplay_games = selfplay_bound
        legacy.TORUS9_CURRENT_MODEL_INIT_SEED = int(config["model_init_seed"])
        legacy.TORUS9_CURRENT_SELFPLAY_MASTER_SEED = int(config["selfplay_master_seed"])
        legacy.TORUS9_CURRENT_TRAINING_MASTER_SEED = int(config["training_master_seed"])
        namespace = argparse.Namespace(
            generation=int(args.generation),
            device=str(config["device"]),
            workers=int(config["workers"]),
            active_games_per_worker=int(config["active_games_per_worker"]),
            total_active_contexts=int(config["total_active_contexts"]),
            batch_cap=int(config["inference_batch_cap"]),
            wait_ms=float(config["inference_batch_wait_ms"]),
            resume=bool(args.resume),
        )
        return legacy.run_generation(namespace)
    finally:
        legacy._validate_execution = original_validate
        legacy._Heartbeat = original_heartbeat
        legacy.run_torus9_selfplay_games = original_selfplay
        legacy.TORUS9_CURRENT_MODEL_INIT_SEED = original_model_seed
        legacy.TORUS9_CURRENT_SELFPLAY_MASTER_SEED = original_selfplay_seed
        legacy.TORUS9_CURRENT_TRAINING_MASTER_SEED = original_training_seed


def _validate_existing_arena(*, output: Path, summary: Mapping[str, object], candidate: Path, reference: Path, config: Mapping[str, object]) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Existing Arena output has no manifest")
    manifest = legacy._read_json(manifest_path)
    execution = summary.get("execution")
    expected_execution = config["execution"]
    if (
        summary.get("candidate_artifact_sha256") != file_sha256(candidate)
        or summary.get("reference_artifact_sha256") != file_sha256(reference)
        or int(summary.get("games", -1)) != int(config["games"])
        or summary.get("arena_profile") != "torus9"
        or int(manifest.get("master_seed", -1)) != int(config["master_seed"])
        or not isinstance(execution, Mapping)
        or dict(execution) != dict(expected_execution)  # type: ignore[arg-type]
    ):
        raise ValueError("Existing Arena output does not match immutable run-spec policy")


def run_arena(args: argparse.Namespace) -> dict[str, object]:
    spec = _load_run_spec()
    config, startset = _arena_config(spec)
    execution = _mapping(config["execution"], "arena.execution")
    _validate_device(str(execution["device"]))

    root, lineage_id, profile_path, expected_fingerprint, _code = legacy._environment(args.generation)
    legacy._load_profile(profile_path, expected_fingerprint)
    reference_gap = int(config["reference_gap"])
    if args.generation < reference_gap:
        raise ValueError("Arena generation/reference gap is invalid")

    reference_generation = int(args.generation) - reference_gap
    candidate = root / "checkpoints" / f"M{args.generation}.pt"
    reference = root / "checkpoints" / f"M{reference_generation}.pt"
    if not candidate.is_file() or not reference.is_file():
        raise FileNotFoundError("Arena candidate/reference checkpoint is missing")

    heartbeat_path = Path(os.environ["AZ_DRIVER_HEARTBEAT_PATH"])
    with legacy._Heartbeat(heartbeat_path, args.generation, interval=float(config["heartbeat_interval_seconds"])) as heartbeat:
        heartbeat.set_phase("arena-snapshot")
        before = legacy._training_snapshot(root)
        evaluation_id = f"{lineage_id}-periodic-M{args.generation:04d}-vs-M{reference_generation:04d}"
        output = evaluation_dir("torus9", evaluation_id)
        summary_path = output / "summary.json"
        if output.exists() and not summary_path.is_file():
            shutil.rmtree(output)

        if summary_path.is_file():
            summary = legacy._read_json(summary_path)
            _validate_existing_arena(output=output, summary=summary, candidate=candidate, reference=reference, config=config)
        else:
            heartbeat.set_phase("arena")
            arena_execution = ArenaExecutionConfig(
                games=int(config["games"]),
                workers=int(execution["workers"]),
                games_per_worker=int(execution["games_per_worker"]),
                inference_batch_rows=int(execution["inference_batch_rows"]),
                inference_batch_wait_ms=float(execution["inference_batch_wait_ms"]),
                device=str(execution["device"]),
                strict_production=bool(execution["strict_production"]),
            )
            summary = run_arena_engine(
                profile=TORUS9_ARENA_PROFILE,
                candidate_path=candidate,
                reference_path=reference,
                output_dir=output,
                candidate_label=f"M{args.generation}",
                reference_label=f"M{reference_generation}",
                run_id=evaluation_id,
                comparison=f"periodic-M{args.generation}-vs-M{reference_generation}",
                master_seed=int(config["master_seed"]),
                config=arena_execution,
            )

        heartbeat.set_phase("arena-verify")
        after = legacy._training_snapshot(root)
        games = int(summary["games"])
        wins = int(summary["wins"])
        draws = int(summary["draws"])
        telemetry = summary.get("telemetry")
        telemetry_map = telemetry if isinstance(telemetry, Mapping) else {}
        arena_block = _mapping(spec.payload["arena"], "arena")
        payload = {
            "schema": legacy.ARENA_RESULT_SCHEMA,
            "generation": int(args.generation),
            "status": "COMPLETED",
            "profile_fingerprint": expected_fingerprint,
            "technical_games": int(summary["technical_games"]),
            "invalid_games": 0,
            "training_mutated": before != after,
            "preset_fingerprint": run_spec_fingerprint(_mapping(arena_block["driver_config"], "arena.driver_config")),
            "startset_fingerprint": run_spec_fingerprint(startset),
            "evaluation_output": str(output),
            "candidate_generation": int(args.generation),
            "reference_generation": reference_generation,
            "metrics": {
                "games": games,
                "wins": wins,
                "losses": int(summary["losses"]),
                "draws": draws,
                "win_rate": ((wins + 0.5 * draws) / games if games else 0.0),
                "games_per_hour": float(telemetry_map.get("games_per_hour", 0.0)),
                "moves_per_sec": float(telemetry_map.get("moves_per_sec", 0.0)),
                "inference_mean_batch_rows": float(telemetry_map.get("mean_inference_batch_rows", 0.0)),
                "effective_cpu_cores": float(telemetry_map.get("effective_cpu_cores", 0.0)),
            },
        }
        legacy._atomic_json(Path(os.environ["AZ_ARENA_RESULT_PATH"]), payload)
        heartbeat.set_phase("completed")
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generation = sub.add_parser("generation")
    generation.add_argument("--generation", type=int, required=True)
    generation.add_argument("--resume", action="store_true")
    generation.set_defaults(func=run_generation)
    arena = sub.add_parser("arena")
    arena.add_argument("--generation", type=int, required=True)
    arena.set_defaults(func=run_arena)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generation <= 0:
        raise SystemExit("--generation must be positive")
    result = args.func(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
