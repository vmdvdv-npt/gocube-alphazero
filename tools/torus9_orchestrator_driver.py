#!/usr/bin/env python3
"""Production Torus9 adapter for the game-independent training orchestrator.

This file is intentionally Torus-specific.  It binds the generic supervisor to
the already-canonical Torus9 SelfPlayEngine, TrainingEngine and Arena engine.
Future Cube support belongs in another adapter; the supervisor itself must not
need to change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.execution_reference import LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE
from gocube_golden.provenance import capture_code_identity, derive_seed, file_sha256
from gocube_golden.run_storage import evaluation_dir
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9SelfPlaySearchContract,
    Torus9TrainingAdapter,
    run_torus9_selfplay_games,
    run_torus9_training_iteration,
    torus9_load_checkpoint,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_MODEL_INIT_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    TORUS9_CURRENT_TRAINING_MASTER_SEED,
    TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    current_torus9_profile_fingerprint,
)
from tools.arena_engine import ArenaExecutionConfig, run_arena as run_arena_engine
from tools.arena_profiles.torus9 import PROFILE as TORUS9_ARENA_PROFILE


GENERATION_RESULT_SCHEMA = "gocube-generation-driver-result-v1"
ARENA_RESULT_SCHEMA = "gocube-arena-driver-result-v1"
PERIODIC_ARENA_PRESET = {
    "schema": "torus9-periodic-arena-preset-v1",
    "profile": "torus9",
    "games": 64,
    "simulations": 64,
    "cpuct": 1.25,
    "fpu": 0.0,
    "noise": False,
    "temperature": 0.0,
    "fast_search": False,
    "resign": False,
    "watchdog": 1000,
    "paired_starts_color_swap": True,
    "master_seed": 202609131004,
    "execution": {
        "workers": 16,
        "games_per_worker": 12,
        "inference_batch_rows": 64,
        "inference_batch_wait_ms": 4.0,
        "device": "cuda",
        "strict_production": True,
    },
}
PERIODIC_ARENA_STARTSET = {
    "schema": "torus9-periodic-arena-startset-v1",
    "generator": "generate_torus9_evaluation_starts",
    "master_seed": 202609131004,
    "pairs": 32,
    "selection": "round-robin-across-8-strata",
}


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


PERIODIC_ARENA_PRESET_FINGERPRINT = _fingerprint(PERIODIC_ARENA_PRESET)
PERIODIC_ARENA_STARTSET_FINGERPRINT = _fingerprint(PERIODIC_ARENA_STARTSET)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


class _Heartbeat:
    def __init__(self, path: Path, generation: int, interval: float = 15.0) -> None:
        self.path = path
        self.generation = int(generation)
        self.interval = float(interval)
        self.phase = "startup"
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase)
        self.write()

    def fail(self, exc: BaseException) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        self.phase = "failed"
        self.write()

    def write(self) -> None:
        payload: dict[str, object] = {
            "at": time.time(),
            "pid": os.getpid(),
            "generation": self.generation,
            "phase": self.phase,
        }
        if self.error is not None:
            payload["errors"] = [self.error]
        _atomic_json(self.path, payload)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()

    def __enter__(self) -> "_Heartbeat":
        self.write()
        self._thread = threading.Thread(target=self._loop, name="torus9-orchestrator-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if exc is not None:
            self.fail(exc)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval))
        self.write()


def _environment(generation: int) -> tuple[Path, str, Path, str]:
    env_generation = int(os.environ["AZ_GENERATION"])
    if env_generation != int(generation):
        raise ValueError("Generation argument disagrees with AZ_GENERATION")
    if os.environ.get("AZ_TOPOLOGY") != "torus9":
        raise ValueError("Torus9 orchestrator driver requires AZ_TOPOLOGY=torus9")
    root = Path(os.environ["AZ_RUN_ROOT"]).resolve()
    lineage_id = os.environ["AZ_LINEAGE_ID"]
    profile_path = Path(os.environ["AZ_PROFILE_PATH"]).resolve()
    expected_fingerprint = os.environ["AZ_PROFILE_FINGERPRINT"]
    if not root.is_dir():
        raise ValueError(f"Lineage root does not exist: {root}")
    try:
        root.relative_to(ROOT / "runs" / "torus9" / "active")
    except ValueError as exc:
        raise ValueError("Torus9 driver refuses a run root outside canonical active storage") from exc
    return root, lineage_id, profile_path, expected_fingerprint


def _load_profile(profile_path: Path, expected_fingerprint: str) -> dict[str, object]:
    profile = _read_json(profile_path)
    if profile.get("profile_id") != TORUS9_CURRENT_PROFILE_ID:
        raise ValueError("Torus9 orchestrator received a non-current profile")
    actual = current_torus9_profile_fingerprint(profile)
    if actual != expected_fingerprint or profile.get("profile_fingerprint") != expected_fingerprint:
        raise ValueError("Torus9 canonical profile fingerprint drift")
    return profile


def _contract(profile: Mapping[str, object]) -> Torus9SelfPlaySearchContract:
    settings = profile["self_play"]
    if not isinstance(settings, Mapping):
        raise ValueError("Torus9 self_play profile is malformed")
    return Torus9SelfPlaySearchContract(
        contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
        simulations=int(settings["mcts_simulations"]),
        cpuct=float(settings["cpuct"]),
        fpu=float(settings["fpu"]),
        temperature_until_ply=int(settings["temperature_plies"][1]),
        temperature_after=float(settings["temperature_after"]),
        dirichlet_epsilon=float(settings["dirichlet_epsilon"]),
        dirichlet_alpha=float(settings["dirichlet_alpha"]),
        watchdog=int(settings["watchdog"]),
    )


def _validate_execution(args: argparse.Namespace, profile: Mapping[str, object]) -> None:
    reference = LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE
    expected = {
        "workers": reference.recommended_workers,
        "active_games_per_worker": reference.recommended_active_games_per_worker,
        "total_active_contexts": reference.recommended_total_active_contexts,
        "batch_cap": reference.recommended_batch_cap,
        "wait_ms": reference.recommended_wait_ms,
    }
    actual = {
        "workers": int(args.workers),
        "active_games_per_worker": int(args.active_games_per_worker),
        "total_active_contexts": int(args.total_active_contexts),
        "batch_cap": int(args.batch_cap),
        "wait_ms": float(args.wait_ms),
    }
    if actual != expected:
        raise ValueError(f"Production Torus9 execution preset drift: expected {expected}, got {actual}")
    self_play = profile["self_play"]
    if not isinstance(self_play, Mapping) or int(self_play["games_per_iteration"]) != 64:
        raise ValueError("Production Torus9 orchestrator requires exactly 64 self-play games per generation")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production Torus9 orchestrator requires CUDA")


def _seed_model() -> None:
    torch.manual_seed(TORUS9_CURRENT_MODEL_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORUS9_CURRENT_MODEL_INIT_SEED)


def _prepare_state(
    *,
    root: Path,
    lineage_id: str,
    generation: int,
    profile: Mapping[str, object],
    device: str,
    code_identity,
) -> tuple[Torus9TrainingAdapter, object, Path]:
    adapter = Torus9TrainingAdapter(
        profile=profile,
        code_identity=code_identity,
        base_commit=TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    )
    if generation == 1:
        _seed_model()
        model = Torus9CurrentGraphNet().to(device)
        state = adapter.create_state(model, run_id=lineage_id)
        m0 = root / "checkpoints" / "M0.pt"
        if not m0.exists():
            metadata = adapter.save_initial_checkpoint(
                m0,
                state,
                run_id=lineage_id,
                label="M0",
                completed_games=0,
                device=device,
                code_identity=code_identity,
            )
        else:
            metadata = _read_json(m0.with_suffix(".metadata.json"))
            torus9_load_checkpoint(
                m0,
                model=model,
                expected={
                    "model_hash": metadata["model_hash"],
                    "profile_id": TORUS9_CURRENT_PROFILE_ID,
                    "profile_fingerprint": adapter.profile_fingerprint,
                },
                device=device,
            )
        state.parent_checkpoint_identity = {
            "label": "M0",
            "path": str(m0),
            "metadata_path": str(m0.with_suffix(".metadata.json")),
            "model_hash": metadata["model_hash"],
            "artifact_sha256": file_sha256(m0),
        }
        return adapter, state, m0

    previous_checkpoint = root / "checkpoints" / f"M{generation - 1}.pt"
    previous_replay = root / "replay" / f"rolling-after-{generation - 1:02d}.jsonl"
    if not previous_checkpoint.is_file() or not previous_replay.is_file():
        raise FileNotFoundError(
            f"Cannot start M{generation}; previous committed checkpoint/replay is missing"
        )
    previous_summary = root / f"iter-{generation - 1:02d}-summary.json"
    evictions = 0
    if previous_summary.is_file():
        replay_metrics = _read_json(previous_summary).get("replay")
        if isinstance(replay_metrics, Mapping):
            evictions = int(replay_metrics.get("total_evictions", 0))
    state = adapter.load_state(
        previous_checkpoint,
        replay_path=previous_replay,
        device=device,
        total_evictions=evictions,
    )
    return adapter, state, previous_checkpoint


def _generation_paths(root: Path, generation: int) -> tuple[Path, ...]:
    return (
        root / "selfplay" / f"iter-{generation:02d}-games.jsonl",
        root / "replay" / f"iter-{generation:02d}-fresh.jsonl",
        root / "replay" / f"rolling-after-{generation:02d}.jsonl",
        root / "checkpoints" / f"M{generation}.pt",
        root / "checkpoints" / f"M{generation}.metadata.json",
        root / "training" / f"iter-{generation:02d}.json",
        root / f"iter-{generation:02d}-summary.json",
    )


def _cleanup_uncommitted_generation(root: Path, generation: int) -> None:
    marker = root / f"generation-{generation:02d}.complete.json"
    if marker.exists():
        return
    for path in _generation_paths(root, generation):
        path.unlink(missing_ok=True)
    for pattern in (
        f"replay/.iter-{generation:02d}.tmp-*",
        f"replay/.rolling-after-{generation:02d}.tmp*",
        f"checkpoints/.M{generation}.tmp*",
        f"training/.iter-{generation:02d}.tmp*",
        f".iter-{generation:02d}-summary.tmp*",
        f".generation-{generation:02d}.complete.tmp*",
    ):
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _artifact(root: Path, path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(root, path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _resume_state(
    *,
    root: Path,
    generation: int,
    checkpoint: Path,
    replay: Path,
    loaded_state,
) -> Path:
    path = root / "runtime" / "resume" / f"generation-{generation:04d}.json"
    payload = {
        "schema": "torus9-orchestrator-resume-v1",
        "generation": generation,
        "components": ["model", "optimizer", "replay", "generation", "rng"],
        "checkpoint": _artifact(root, checkpoint),
        "replay": _artifact(root, replay),
        "optimizer_updates": int(loaded_state.optimizer_updates),
        "samples_consumed": int(loaded_state.samples_consumed),
        "replay_last_generation": int(loaded_state.rolling_replay.last_generation),
        "rng": {
            "model_init_seed": TORUS9_CURRENT_MODEL_INIT_SEED,
            "selfplay_master_seed": TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
            "training_master_seed": TORUS9_CURRENT_TRAINING_MASTER_SEED,
            "training_seed": derive_seed(
                TORUS9_CURRENT_TRAINING_MASTER_SEED,
                os.environ["AZ_LINEAGE_ID"],
                "training",
                generation,
            ),
            "policy": "all game/search/training randomness is derived from explicit stable seeds",
        },
    }
    _atomic_json(path, payload)
    return path


def _generation_metrics(
    *,
    records,
    selfplay_wall: float,
    inference: Mapping[str, object],
    training: Mapping[str, object],
) -> dict[str, object]:
    moves = sum(len(record.final_action_trace) for record in records)
    training_wall = float(training.get("training_wall_time_sec", 0.0))
    optimizer_steps = int(training.get("optimizer_steps", 0))
    return {
        "games": len(records),
        "games_per_hour": (len(records) * 3600.0 / selfplay_wall if selfplay_wall else 0.0),
        "moves": moves,
        "moves_per_sec": (moves / selfplay_wall if selfplay_wall else 0.0),
        "selfplay_time_sec": selfplay_wall,
        "training_time_sec": training_wall,
        "optimizer_updates_per_sec": (
            optimizer_steps / training_wall if training_wall else 0.0
        ),
        "inference": {
            "mean_batch_rows": float(inference.get("mean_batch_rows", 0.0)),
            "p95_batch_rows": float(inference.get("p95_batch_rows", 0.0)),
            "max_batch_rows": int(inference.get("max_batch_rows", 0)),
            "rows_per_sec": float(inference.get("rows_per_sec", 0.0)),
        },
        "loss": {
            "policy": float(training.get("mean_policy_loss", 0.0)),
            "value": float(training.get("mean_value_loss", 0.0)),
            "ownership": float(training.get("mean_ownership_loss", 0.0)),
            "score": float(training.get("mean_score_loss_normalized", 0.0)),
            "total": float(training.get("mean_total_loss", 0.0)),
        },
        "learning": {
            "optimizer_updates_total": int(training.get("optimizer_updates_total", 0)),
            "samples_consumed_total": int(training.get("samples_consumed_total", 0)),
            "mean_parameter_delta": float(training.get("mean_parameter_delta", 0.0)),
            "mean_gradient_norm": float(training.get("mean_gradient_norm", 0.0)),
        },
    }


def _publish_generation_result(
    *,
    root: Path,
    generation: int,
    profile_fingerprint: str,
    selfplay_metrics: Mapping[str, object],
) -> dict[str, object]:
    checkpoint = root / "checkpoints" / f"M{generation}.pt"
    replay = root / "replay" / f"rolling-after-{generation:02d}.jsonl"
    summary_path = root / f"iter-{generation:02d}-summary.json"
    marker = root / f"generation-{generation:02d}.complete.json"
    selfplay_path = root / "selfplay" / f"iter-{generation:02d}-games.jsonl"
    adapter = Torus9TrainingAdapter(profile=_read_json(Path(os.environ["AZ_PROFILE_PATH"])))
    loaded = adapter.load_state(checkpoint, replay_path=replay, device="cpu")
    if int(loaded.current_generation) != generation:
        raise ValueError("Reloaded Torus9 checkpoint generation mismatch")
    resume = _resume_state(
        root=root,
        generation=generation,
        checkpoint=checkpoint,
        replay=replay,
        loaded_state=loaded,
    )
    summary = _read_json(summary_path)
    training = summary.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Training summary is missing training metrics")
    metrics = _generation_metrics(
        records=(),
        selfplay_wall=float(selfplay_metrics["selfplay_time_sec"]),
        inference=selfplay_metrics["inference"],
        training=training,
    )
    metrics.update(dict(selfplay_metrics))
    expected_selfplay_hash = selfplay_metrics.get("selfplay_artifact_sha256")
    if expected_selfplay_hash is not None and expected_selfplay_hash != file_sha256(selfplay_path):
        raise ValueError("Persisted self-play artifact hash disagrees with committed generation summary")
    artifacts = [
        _artifact(root, path)
        for path in (
            checkpoint,
            checkpoint.with_suffix(".metadata.json"),
            replay,
            root / "replay" / f"iter-{generation:02d}-fresh.jsonl",
            root / "training" / f"iter-{generation:02d}.json",
            summary_path,
            marker,
            selfplay_path,
            resume,
        )
    ]
    payload = {
        "schema": GENERATION_RESULT_SCHEMA,
        "generation": generation,
        "status": "COMPLETED",
        "profile_fingerprint": profile_fingerprint,
        "checkpoint_reload_verified": True,
        "checkpoint": {"path": _relative(root, checkpoint)},
        "replay": {"path": _relative(root, replay)},
        "resume_state": {
            "path": _relative(root, resume),
            "components": ["model", "optimizer", "replay", "generation", "rng"],
        },
        "artifacts": artifacts,
        "technical_games": int(selfplay_metrics.get("technical_games", 0)),
        "invalid_games": int(selfplay_metrics.get("invalid_games", 0)),
        "metrics": metrics,
    }
    result_path = Path(os.environ["AZ_GENERATION_RESULT_PATH"])
    _atomic_json(result_path, payload)
    return payload


def run_generation(args: argparse.Namespace) -> dict[str, object]:
    root, lineage_id, profile_path, expected_fingerprint = _environment(args.generation)
    profile = _load_profile(profile_path, expected_fingerprint)
    _validate_execution(args, profile)
    code = capture_code_identity(ROOT)
    heartbeat_path = Path(os.environ["AZ_DRIVER_HEARTBEAT_PATH"])
    marker = root / f"generation-{args.generation:02d}.complete.json"

    with _Heartbeat(heartbeat_path, args.generation) as heartbeat:
        manifest = _read_json(root / "manifest.json")
        if args.generation == 1 and manifest.get("parent_checkpoint") is not None:
            raise ValueError(
                "Current Torus9 orchestrator driver supports fresh M0 lineages only; "
                "it will not silently discard a cross-lineage parent checkpoint/replay state"
            )
        if marker.is_file():
            heartbeat.set_phase("recover-published-generation")
            summary = _read_json(root / f"iter-{args.generation:02d}-summary.json")
            persisted = summary.get("orchestrator_selfplay")
            if not isinstance(persisted, Mapping):
                raise ValueError("Committed generation lacks orchestrator self-play metrics")
            return _publish_generation_result(
                root=root,
                generation=args.generation,
                profile_fingerprint=expected_fingerprint,
                selfplay_metrics=persisted,
            )

        if args.resume:
            heartbeat.set_phase("cleanup-uncommitted-generation")
            _cleanup_uncommitted_generation(root, args.generation)
            Path(os.environ["AZ_GENERATION_RESULT_PATH"]).unlink(missing_ok=True)
            (root / "runtime" / "resume" / f"generation-{args.generation:04d}.json").unlink(missing_ok=True)
        elif any(path.exists() for path in _generation_paths(root, args.generation)):
            raise FileExistsError(
                "Generation has uncommitted artifacts; use the orchestrator resume path"
            )

        heartbeat.set_phase("load-previous-state")
        adapter, state, previous_checkpoint = _prepare_state(
            root=root,
            lineage_id=lineage_id,
            generation=args.generation,
            profile=profile,
            device=args.device,
            code_identity=code,
        )

        games = int(profile["self_play"]["games_per_iteration"])
        game_ids = [
            f"{lineage_id}-generation-{args.generation:04d}-game-{index:04d}"
            for index in range(games)
        ]
        inference: dict[str, object] = {}
        heartbeat.set_phase("self-play")
        started = time.perf_counter()
        records = run_torus9_selfplay_games(
            state.model,
            run_id=lineage_id,
            label=f"M{args.generation - 1}",
            artifact=file_sha256(previous_checkpoint),
            master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
            profile_fp=expected_fingerprint,
            game_ids=game_ids,
            workers=args.workers,
            code_identity=code,
            device=args.device,
            contract=_contract(profile),
            coalescing=True,
            inference_batch_cap=args.batch_cap,
            inference_batch_wait_ms=args.wait_ms,
            active_games_per_worker=args.active_games_per_worker,
            total_active_contexts=args.total_active_contexts,
            inference_telemetry=inference,
            execution_activity=inference,
        )
        selfplay_wall = time.perf_counter() - started
        if len(records) != games or {record.game_id for record in records} != set(game_ids):
            raise RuntimeError("Torus9 self-play did not return exactly the requested game set")
        technical_games = sum(record.technical_termination is not None for record in records)
        if technical_games:
            raise RuntimeError(f"Torus9 self-play produced {technical_games} technical games")
        for record in records:
            record.validate()
        selfplay_path = root / "selfplay" / f"iter-{args.generation:02d}-games.jsonl"
        _atomic_jsonl(selfplay_path, [record.to_dict() for record in records])
        moves = sum(len(record.final_action_trace) for record in records)
        selfplay_metrics = {
            "games": games,
            "technical_games": 0,
            "invalid_games": 0,
            "moves": moves,
            "games_per_hour": games * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
            "moves_per_sec": moves / selfplay_wall if selfplay_wall else 0.0,
            "selfplay_time_sec": selfplay_wall,
            "inference": {
                "mean_batch_rows": float(inference.get("mean_batch_rows", 0.0)),
                "p95_batch_rows": float(inference.get("p95_batch_rows", 0.0)),
                "max_batch_rows": int(inference.get("max_batch_rows", 0)),
                "rows_per_sec": float(inference.get("rows_per_sec", 0.0)),
            },
            "selfplay_artifact_path": _relative(root, selfplay_path),
            "selfplay_artifact_sha256": file_sha256(selfplay_path),
        }

        heartbeat.set_phase("training")
        run_torus9_training_iteration(
            state=state,
            generation=args.generation,
            output_dir=root,
            run_id=lineage_id,
            records=records,
            training_seed=derive_seed(
                TORUS9_CURRENT_TRAINING_MASTER_SEED,
                lineage_id,
                "training",
                args.generation,
            ),
            completed_games=args.generation * games,
            code_identity=code,
            device=args.device,
            adapter=adapter,
            summary_extra={"orchestrator_selfplay": selfplay_metrics},
        )

        heartbeat.set_phase("reload-verification")
        payload = _publish_generation_result(
            root=root,
            generation=args.generation,
            profile_fingerprint=expected_fingerprint,
            selfplay_metrics=selfplay_metrics,
        )
        heartbeat.set_phase("completed")
        return payload


def _training_snapshot(root: Path, generation: int) -> dict[str, str]:
    paths = list(sorted((root / "checkpoints").glob("M*.pt")))
    paths += list(sorted((root / "checkpoints").glob("M*.metadata.json")))
    rolling = root / "replay" / f"rolling-after-{generation:02d}.jsonl"
    if rolling.is_file():
        paths.append(rolling)
    return {_relative(root, path): file_sha256(path) for path in paths if path.is_file()}


def run_arena(args: argparse.Namespace) -> dict[str, object]:
    root, lineage_id, profile_path, expected_fingerprint = _environment(args.generation)
    _load_profile(profile_path, expected_fingerprint)
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production Torus9 periodic Arena requires CUDA")
    if args.reference_gap <= 0 or args.generation < args.reference_gap:
        raise ValueError("Periodic Arena generation/reference gap is invalid")
    reference_generation = args.generation - args.reference_gap
    candidate = root / "checkpoints" / f"M{args.generation}.pt"
    reference = root / "checkpoints" / f"M{reference_generation}.pt"
    if not candidate.is_file() or not reference.is_file():
        raise FileNotFoundError("Periodic Arena candidate/reference checkpoint is missing")

    heartbeat_path = Path(os.environ["AZ_DRIVER_HEARTBEAT_PATH"])
    with _Heartbeat(heartbeat_path, args.generation) as heartbeat:
        heartbeat.set_phase("arena-snapshot")
        before = _training_snapshot(root, args.generation)
        evaluation_id = (
            f"{lineage_id}-periodic-M{args.generation:04d}-vs-M{reference_generation:04d}"
        )
        output = evaluation_dir("torus9", evaluation_id)
        summary_path = output / "summary.json"
        if output.exists() and not summary_path.is_file():
            shutil.rmtree(output)

        if summary_path.is_file():
            summary = _read_json(summary_path)
            if (
                summary.get("candidate_artifact_sha256") != file_sha256(candidate)
                or summary.get("reference_artifact_sha256") != file_sha256(reference)
                or int(summary.get("games", -1)) != 64
                or summary.get("arena_profile") != "torus9"
            ):
                raise ValueError("Existing periodic Arena output does not match requested checkpoints/preset")
        else:
            heartbeat.set_phase("arena")
            config = ArenaExecutionConfig(
                games=64,
                workers=16,
                games_per_worker=12,
                inference_batch_rows=64,
                inference_batch_wait_ms=4.0,
                device=args.device,
                strict_production=True,
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
                master_seed=int(PERIODIC_ARENA_PRESET["master_seed"]),
                config=config,
            )

        heartbeat.set_phase("arena-verify")
        after = _training_snapshot(root, args.generation)
        training_mutated = before != after
        games = int(summary["games"])
        wins = int(summary["wins"])
        draws = int(summary["draws"])
        technical = int(summary["technical_games"])
        win_rate = (wins + 0.5 * draws) / games if games else 0.0
        telemetry = summary.get("telemetry")
        telemetry_map = telemetry if isinstance(telemetry, Mapping) else {}
        payload = {
            "schema": ARENA_RESULT_SCHEMA,
            "generation": args.generation,
            "status": "COMPLETED",
            "profile_fingerprint": expected_fingerprint,
            "technical_games": technical,
            "invalid_games": 0,
            "training_mutated": training_mutated,
            "preset_fingerprint": PERIODIC_ARENA_PRESET_FINGERPRINT,
            "startset_fingerprint": PERIODIC_ARENA_STARTSET_FINGERPRINT,
            "evaluation_output": str(output),
            "candidate_generation": args.generation,
            "reference_generation": reference_generation,
            "metrics": {
                "games": games,
                "wins": wins,
                "losses": int(summary["losses"]),
                "draws": draws,
                "win_rate": win_rate,
                "games_per_hour": float(telemetry_map.get("games_per_hour", 0.0)),
                "moves_per_sec": float(telemetry_map.get("moves_per_sec", 0.0)),
                "inference_mean_batch_rows": float(
                    telemetry_map.get("mean_inference_batch_rows", 0.0)
                ),
                "effective_cpu_cores": float(
                    telemetry_map.get("effective_cpu_cores", 0.0)
                ),
            },
        }
        _atomic_json(Path(os.environ["AZ_ARENA_RESULT_PATH"]), payload)
        heartbeat.set_phase("completed")
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generation = sub.add_parser("generation")
    generation.add_argument("--generation", type=int, required=True)
    generation.add_argument("--device", default="cuda")
    generation.add_argument("--workers", type=int, default=16)
    generation.add_argument("--active-games-per-worker", type=int, default=4)
    generation.add_argument("--total-active-contexts", type=int, default=64)
    generation.add_argument("--batch-cap", type=int, default=64)
    generation.add_argument("--wait-ms", type=float, default=1.0)
    generation.add_argument("--resume", action="store_true")
    generation.set_defaults(func=run_generation)

    arena = sub.add_parser("arena")
    arena.add_argument("--generation", type=int, required=True)
    arena.add_argument("--device", default="cuda")
    arena.add_argument("--reference-gap", type=int, default=5)
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
