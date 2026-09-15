#!/usr/bin/env python3
"""Continue the existing current Torus 9×9 Golden run from M1 to M100.

The run namespace is deliberately fixed and must already contain a complete
M1.  This driver owns only continuation orchestration and execution telemetry;
the current Golden profile and all semantic contracts remain read-only.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.provenance import CodeIdentity, derive_seed, file_sha256, capture_code_identity
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9OwnershipScoreTrainer,
    Torus9RollingReplay,
    run_torus9_batched_arena,
    run_torus9_selfplay_games,
    torus9_build_ownership_score_replay_samples,
    torus9_load_checkpoint,
    torus9_save_checkpoint,
    validate_torus9_replay_sample,
    write_json,
    write_jsonl,
)
from gocube_golden.torus9_contract import (
    TORUS9_ARENA_MOVE_LIMIT,
    TORUS9_CURRENT_ARENA_MASTER_SEED,
    TORUS9_CURRENT_MODEL_INIT_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_CURRENT_TRAINING_MASTER_SEED,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    TORUS9_WORKERS,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)
from tools.torus9_golden_learning import BASE_COMMIT, _contract
from tools._torus9_legacy_compat import checkpoint as _checkpoint


RUN_ID = "torus9-golden-v3-20260914-run03"
RUN_ROOT = ROOT / "runs" / "torus9-golden-v3-active" / RUN_ID
STATE_PATH = RUN_ROOT / "continuation-state.json"
STOP_PATH = RUN_ROOT / "stop-request.json"

CAP_SWEEP = (
    {"iteration": 2, "cap": 4, "wait_ms": 6.0, "phase": "cap_sweep"},
    {"iteration": 3, "cap": 8, "wait_ms": 6.0, "phase": "cap_sweep"},
    {"iteration": 4, "cap": 12, "wait_ms": 6.0, "phase": "cap_sweep"},
    {"iteration": 5, "cap": 16, "wait_ms": 6.0, "phase": "cap_sweep"},
)
WAIT_SWEEP = (
    {"iteration": 6, "wait_ms": 4.0, "phase": "wait_sweep"},
    {"iteration": 7, "wait_ms": 2.0, "phase": "wait_sweep"},
    {"iteration": 8, "wait_ms": 1.0, "phase": "wait_sweep"},
)


_STOP_REQUESTED = False


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def _canonical(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]


def _p50(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median([float(value) for value in values]))


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


_NVML_LIB: Any | None = None
_NVML_HANDLE: ctypes.c_void_p | None = None


def _nvml_sample() -> dict[str, object] | None:
    """Fallback GPU telemetry for environments without the nvidia-smi CLI."""
    global _NVML_LIB, _NVML_HANDLE
    try:
        if _NVML_LIB is None:
            class Utilization(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

            class Memory(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                ]

            library = ctypes.CDLL("libnvidia-ml.so.1")
            library.nvmlInit_v2.restype = ctypes.c_int
            library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            if library.nvmlInit_v2() != 0:
                return None
            _NVML_LIB = (library, Utilization, Memory)
        library, utilization_type, memory_type = _NVML_LIB
        if _NVML_HANDLE is None:
            handle = ctypes.c_void_p()
            if library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                return None
            _NVML_HANDLE = handle
        utilization = utilization_type()
        memory = memory_type()
        if library.nvmlDeviceGetUtilizationRates(_NVML_HANDLE, ctypes.byref(utilization)) != 0:
            return None
        if library.nvmlDeviceGetMemoryInfo(_NVML_HANDLE, ctypes.byref(memory)) != 0:
            return None
        return {
            "ts": time.time(),
            "gpu_util_pct": float(utilization.gpu),
            "vram_used_mb": float(memory.used) / (1024.0 * 1024.0),
        }
    except (OSError, AttributeError, TypeError, ValueError):
        return None


class GpuSampler:
    """Low-rate nvidia-smi sampler kept outside the training/search code."""

    def __init__(self, path: Path, interval_s: float = 2.0) -> None:
        self.path = path
        self.interval_s = float(interval_s)
        self.rows: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            first = result.stdout.strip().splitlines()[0]
            utilization, memory = [float(value.strip()) for value in first.split(",", 1)]
            self.rows.append({"ts": time.time(), "gpu_util_pct": utilization, "vram_used_mb": memory})
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            sample = _nvml_sample()
            if sample is not None:
                self.rows.append(sample)

    def _serve(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._sample_once()
        self._thread = threading.Thread(target=self._serve, name="torus9-gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        _atomic_write_jsonl(self.path, self.rows)
        utilization = [float(row["gpu_util_pct"]) for row in self.rows]
        vram = [float(row["vram_used_mb"]) for row in self.rows]
        return {
            "samples": len(self.rows),
            "sample_path": str(self.path),
            "gpu_utilization_avg_pct": sum(utilization) / len(utilization) if utilization else None,
            "gpu_utilization_peak_pct": max(utilization) if utilization else None,
            "gpu_vram_peak_mb": max(vram) if vram else None,
            "sample_start_unix": self.rows[0].get("ts") if self.rows else None,
            "sample_end_unix": self.rows[-1].get("ts") if self.rows else None,
        }


def _signal_stop(signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    _atomic_write(STOP_PATH, {"requested": True, "signal": int(signum), "ts": time.time(), "reason": "user-requested clean boundary stop"})


def request_stop() -> None:
    _atomic_write(STOP_PATH, {"requested": True, "ts": time.time(), "reason": "user-requested clean boundary stop"})


def _stop_requested() -> bool:
    return _STOP_REQUESTED or STOP_PATH.exists()


def _state_update(**updates: object) -> None:
    current = _read_json(STATE_PATH) if STATE_PATH.exists() else {}
    current.update(updates)
    _atomic_write(STATE_PATH, current)


def _execution(cap: int, wait_ms: float, *, phase: str) -> dict[str, object]:
    return {
        "coalescing": True,
        "self_play_inference_batch_cap": int(cap),
        "self_play_inference_batch_wait_ms": float(wait_ms),
        "batch_cap": int(cap),
        "wait_ms": float(wait_ms),
        "phase": phase,
        "execution_only": True,
    }


def _profile_and_contract() -> tuple[dict[str, object], str, object]:
    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    if profile_fp != "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775":
        raise RuntimeError("Current Torus 9×9 profile fingerprint drifted")
    return profile, profile_fp, __import__("tools.torus9_golden_learning", fromlist=["_contract"])._contract(profile)


def _validate_starting_run(profile_fp: str) -> dict[str, object]:
    if not RUN_ROOT.is_dir():
        raise FileNotFoundError(RUN_ROOT)
    manifest_path = RUN_ROOT / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("run_id") != RUN_ID or RUN_ROOT.name != RUN_ID:
        raise RuntimeError("Torus 9×9 continuation run identity mismatch")
    if manifest.get("base_commit") != BASE_COMMIT:
        raise RuntimeError("Torus 9×9 continuation base commit mismatch")
    if manifest.get("profile_id") != TORUS9_CURRENT_PROFILE_ID or manifest.get("profile_fingerprint") != profile_fp:
        raise RuntimeError("Torus 9×9 continuation profile mismatch")
    if manifest.get("device") != "cuda" or manifest.get("device_locked") is not True:
        raise RuntimeError("Torus 9×9 continuation requires locked CUDA")
    if manifest.get("status") not in {"STOPPED_AFTER_M1", "RUNNING", "STOP_REQUESTED"}:
        raise RuntimeError("Continuation requires a resumable prepared lineage")
    m1 = RUN_ROOT / "checkpoints" / "M1.pt"
    m1_meta_path = RUN_ROOT / "checkpoints" / "M1.metadata.json"
    replay = RUN_ROOT / "replay" / "rolling-after-01.jsonl"
    for path in (m1, m1_meta_path, replay, RUN_ROOT / "iter-01-summary.json", RUN_ROOT / "selfplay" / "iter-01-games.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = _read_json(m1_meta_path)
    expected = {
        "checkpoint_label": "M1",
        "run_id": RUN_ID,
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        "completed_games": 64,
        "optimizer_updates": 80,
        "train_samples_consumed": 5120,
        "device": "cuda",
        "device_locked": True,
        "ownership_loss_enabled": True,
        "score_loss_enabled": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"M1 metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    replay_rows = _read_jsonl(replay)
    if len(replay_rows) != int(metadata.get("valid_replay_positions", -1)):
        raise RuntimeError("M1 replay position count mismatch")
    if file_sha256(replay) != manifest.get("m1", {}).get("replay_sha256"):
        raise RuntimeError("M1 replay hash mismatch")
    if file_sha256(m1) != _read_json(RUN_ROOT / "iter-01-summary.json").get("checkpoint", {}).get("artifact_sha256"):
        raise RuntimeError("M1 checkpoint artifact hash mismatch")
    return manifest


def _rebuild_replay(last_generation: int, profile: Mapping[str, object]) -> Torus9RollingReplay:
    replay = Torus9RollingReplay(
        generations=int(profile["replay"]["generations"]),  # type: ignore[index]
        maximum_positions=int(profile["replay"]["cap"]),  # type: ignore[index]
    )
    for generation in range(1, last_generation + 1):
        path = RUN_ROOT / "replay" / f"iter-{generation:02d}-fresh.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Missing replay generation artifact: {path}")
        rows = _read_jsonl(path)
        for row in rows:
            validate_torus9_replay_sample(row, expected_target_fingerprint=TORUS9_CURRENT_TARGET_FINGERPRINT)
        replay.append_generation(generation, rows)
    expected_path = RUN_ROOT / "replay" / f"rolling-after-{last_generation:02d}.jsonl"
    if not expected_path.is_file() or list(replay.rows) != _read_jsonl(expected_path):
        raise RuntimeError("Reconstructed rolling replay differs from persisted state")
    return replay


def _complete_iterations() -> list[int]:
    complete: list[int] = [1]
    for iteration in range(2, 101):
        paths = (
            RUN_ROOT / "checkpoints" / f"M{iteration}.pt",
            RUN_ROOT / "checkpoints" / f"M{iteration}.metadata.json",
            RUN_ROOT / "selfplay" / f"iter-{iteration:02d}-games.jsonl",
            RUN_ROOT / "replay" / f"iter-{iteration:02d}-fresh.jsonl",
            RUN_ROOT / "replay" / f"rolling-after-{iteration:02d}.jsonl",
            RUN_ROOT / "training" / f"iter-{iteration:02d}.json",
            RUN_ROOT / f"iter-{iteration:02d}-summary.json",
        )
        if all(path.is_file() for path in paths):
            complete.append(iteration)
        else:
            break
    return complete


def _load_history(iterations: Sequence[int]) -> list[dict[str, object]]:
    rows = []
    for iteration in iterations:
        rows.append(_read_json(RUN_ROOT / f"iter-{iteration:02d}-summary.json"))
    return rows


def _validate_transition_result(iteration: int, records: Sequence[object], telemetry: Mapping[str, object]) -> None:
    game_ids = {f"{RUN_ID}-iter-{iteration:02d}-game-{index:04d}" for index in range(64)}
    actual_ids = {str(getattr(record, "game_id")) for record in records}
    if len(records) != 64 or actual_ids != game_ids:
        raise RuntimeError(f"M{iteration} did not complete exactly 64 canonical self-play games")
    if sum(int(getattr(record, "nn_evaluations")) for record in records) != int(telemetry.get("total_rows", -1)):
        raise RuntimeError(f"M{iteration} inference row accounting mismatch")
    for record in records:
        record.validate()
        expected_seed = derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, RUN_ID, record.game_id, "game")
        if int(record.game_seed) != expected_seed:
            raise RuntimeError(f"M{iteration} game seed drift for {record.game_id}")


def _run_iteration(
    *,
    iteration: int,
    cap: int,
    wait_ms: float,
    phase: str,
    profile: Mapping[str, object],
    profile_fp: str,
    contract: object,
    model: Torus9CurrentGraphNet,
    trainer: Torus9OwnershipScoreTrainer,
    replay: Torus9RollingReplay,
    code: CodeIdentity,
    manifest: dict[str, object],
) -> dict[str, object]:
    previous_checkpoint = RUN_ROOT / "checkpoints" / f"M{iteration - 1}.pt"
    if not previous_checkpoint.is_file():
        raise FileNotFoundError(previous_checkpoint)
    execution = _execution(cap, wait_ms, phase=phase)
    game_ids = [f"{RUN_ID}-iter-{iteration:02d}-game-{index:04d}" for index in range(64)]
    telemetry: dict[str, object] = {}
    gpu = GpuSampler(RUN_ROOT / "telemetry" / f"selfplay-M{iteration}-gpu.jsonl")
    cpu_before = _cpu_seconds()
    gpu.start()
    started = time.perf_counter()
    _state_update(
        status="RUNNING",
        current_atomic_unit=f"M{iteration - 1}→M{iteration}",
        current_iteration=iteration,
        next_transition=f"M{iteration - 1}→M{iteration}",
        stop_requested=_stop_requested(),
        scientific_validity="PENDING_ATOMIC_UNIT",
    )
    try:
        records = run_torus9_selfplay_games(
            model,
            checkpoint_path=previous_checkpoint,
            run_id=RUN_ID,
            label=f"M{iteration - 1}",
            artifact=file_sha256(previous_checkpoint),
            master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
            profile_fp=profile_fp,
            profile_id=TORUS9_CURRENT_PROFILE_ID,
            game_ids=game_ids,
            workers=TORUS9_WORKERS,
            code_identity=code,
            device="cuda",
            contract=contract,  # type: ignore[arg-type]
            coalescing=True,
            inference_batch_cap=int(cap),
            inference_batch_wait_ms=float(wait_ms),
            inference_telemetry=telemetry,
            execution_activity=telemetry,
        )
    finally:
        selfplay_wall = time.perf_counter() - started
        gpu_summary = gpu.stop()
    _validate_transition_result(iteration, records, telemetry)
    selfplay_path = RUN_ROOT / "selfplay" / f"iter-{iteration:02d}-games.jsonl"
    _atomic_write_jsonl(selfplay_path, [record.to_dict() for record in records])

    fresh: list[dict[str, object]] = []
    for record in records:
        if record.technical_termination is not None:
            continue
        rows = list(torus9_build_ownership_score_replay_samples(record))
        for row in rows:
            validate_torus9_replay_sample(row, expected_target_fingerprint=TORUS9_CURRENT_TARGET_FINGERPRINT)
        fresh.extend(rows)
    if not fresh:
        raise RuntimeError(f"M{iteration} produced no valid replay positions")
    stamped: list[dict[str, object]] = []
    for position, row in enumerate(fresh):
        current = dict(row)
        current["source_generation"] = iteration
        current["replay_row_id"] = f"M{iteration}:{current['game_id']}:{current['ply']}:{position}"
        stamped.append(current)
    _atomic_write_jsonl(RUN_ROOT / "replay" / f"iter-{iteration:02d}-fresh.jsonl", stamped)
    replay_metrics = replay.append_generation(iteration, stamped)
    _atomic_write_jsonl(RUN_ROOT / "replay" / f"rolling-after-{iteration:02d}.jsonl", list(replay.rows))

    train_started = time.perf_counter()
    train_metrics = trainer.train_fixed_budget(
        list(replay.rows),
        seed=derive_seed(TORUS9_CURRENT_TRAINING_MASTER_SEED, RUN_ID, "training", iteration),
    )
    training_wall = time.perf_counter() - train_started
    if train_metrics.get("optimizer_steps") != 80 or train_metrics.get("samples_consumed") != 5120 or train_metrics.get("batch_sizes") != [64] * 80:
        raise RuntimeError(f"M{iteration} fixed training budget drifted")
    if train_metrics.get("ownership_loss_enabled") is not True or train_metrics.get("score_loss_enabled") is not True:
        raise RuntimeError(f"M{iteration} auxiliary loss contract drifted")
    for key in ("mean_policy_loss", "mean_value_loss", "mean_ownership_loss", "mean_score_loss_normalized", "mean_total_loss"):
        if not math.isfinite(float(train_metrics[key])):
            raise RuntimeError(f"M{iteration} produced non-finite {key}")
    _atomic_write(RUN_ROOT / "training" / f"iter-{iteration:02d}.json", train_metrics)

    checkpoint_final = RUN_ROOT / "checkpoints" / f"M{iteration}.pt"
    checkpoint_tmp = RUN_ROOT / "checkpoints" / f".M{iteration}.tmp.pt"
    for path in (checkpoint_tmp, checkpoint_tmp.with_suffix(".metadata.json")):
        if path.exists():
            path.unlink()
    checkpoint_metadata = _checkpoint(
        checkpoint_tmp,
        model,
        trainer.optimizer,
        run_id=RUN_ID,
        label=f"M{iteration}",
        parent=f"M{iteration - 1}",
        code=code,
        profile_fp=profile_fp,
        contract=contract,  # type: ignore[arg-type]
        completed_games=iteration * 64,
        replay_positions=len(replay.rows),
        optimizer_updates=int(trainer.update_count),
        samples_consumed=int(trainer.samples_consumed),
        device="cuda",
    )
    checkpoint_tmp_meta = checkpoint_tmp.with_suffix(".metadata.json")
    checkpoint_final_meta = checkpoint_final.with_suffix(".metadata.json")
    os.replace(checkpoint_tmp, checkpoint_final)
    os.replace(checkpoint_tmp_meta, checkpoint_final_meta)

    plies = [len(record.final_action_trace) for record in records]
    technical = sum(record.technical_termination is not None for record in records)
    cpu_seconds = max(0.0, _cpu_seconds() - cpu_before)
    inference_rows = [float(value) for value in telemetry.get("batch_rows", [])]
    inference = {
        "inference_calls": int(telemetry.get("forward_calls", 0)),
        "total_inference_rows": int(telemetry.get("total_rows", 0)),
        "inference_rows_per_sec": int(telemetry.get("total_rows", 0)) / selfplay_wall if selfplay_wall else 0.0,
        "mean_inference_batch_rows": float(telemetry.get("mean_batch_rows", 0.0)),
        "median_inference_batch_rows": _p50(inference_rows),
        "p95_inference_batch_rows": _p95(inference_rows),
        "max_inference_batch_rows": int(telemetry.get("max_batch_rows", 0)),
        "batch_rows_semantics": "execution-only coalescing across independent self-play lanes",
        "max_active_mcts_lanes": int(telemetry.get("max_active_mcts_lanes", TORUS9_WORKERS)),
        "max_active_inference_requests": int(telemetry.get("max_active_inference_requests", TORUS9_WORKERS)),
        "cpu_utilization_pct": 100.0 * cpu_seconds / (selfplay_wall * max(1, os.cpu_count() or 1)) if selfplay_wall else 0.0,
        **gpu_summary,
    }
    row: dict[str, object] = {
        "iteration": iteration,
        "label": f"M{iteration}",
        "device": "cuda",
        "games": 64,
        "valid_games": 64 - technical,
        "technical_games": technical,
        "fresh_positions": len(fresh),
        "total_moves": sum(plies),
        "average_ply": sum(plies) / len(plies),
        "median_game_length": statistics.median(plies),
        "p95_game_length": _p95(plies),
        "min_ply": min(plies),
        "max_ply": max(plies),
        "self_play_wall_time_sec": selfplay_wall,
        "games_per_sec": 64.0 / selfplay_wall if selfplay_wall else 0.0,
        "games_per_hour": 64.0 * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
        "moves_per_sec": sum(plies) / selfplay_wall if selfplay_wall else 0.0,
        "positions_per_hour": len(fresh) * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
        "execution": execution,
        "replay": replay_metrics,
        "inference": inference,
        "training": train_metrics,
        "training_wall_time_sec": training_wall,
        "checkpoint": {
            "path": str(checkpoint_final),
            "metadata_path": str(checkpoint_final_meta),
            "model_hash": checkpoint_metadata["model_hash"],
            "artifact_sha256": file_sha256(checkpoint_final),
            "optimizer_state_present": True,
        },
        "technical_outcomes_excluded": True,
    }
    _atomic_write(RUN_ROOT / f"iter-{iteration:02d}-summary.json", row)
    manifest.update({
        "status": "RUNNING",
        "result": "IN_PROGRESS",
        "last_completed_iteration": iteration,
        "resumable_from": f"M{iteration}",
        "next_transition_started": False,
        "continuation_source_code": {
            "git_commit": code.git_commit_sha,
            "git_tree": code.git_tree_sha,
            "git_worktree_clean": code.working_tree_clean,
        },
        "iteration_telemetry": str(RUN_ROOT / f"iter-{iteration:02d}-summary.json"),
    })
    _atomic_write(RUN_ROOT / "manifest.json", manifest)
    _state_update(
        status="RUNNING",
        last_completed_iteration=iteration,
        last_atomic_unit=f"M{iteration - 1}→M{iteration}",
        current_atomic_unit=None,
        next_transition=f"M{iteration}→M{iteration + 1}" if iteration < 100 else None,
        next_transition_started=False,
        scientific_validity="PASS",
        stop_requested=_stop_requested(),
    )
    return row


def _selfplay_cap_winner(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def row_cap(row: Mapping[str, object]) -> int:
        execution = row.get("execution", {})
        if not isinstance(execution, Mapping):
            return -1
        return int(execution.get("cap", execution.get("batch_cap", -1)))

    observed = [row for row in rows if row.get("iteration") in {2, 3, 4, 5} and row.get("execution", {}).get("phase") == "cap_sweep"]  # type: ignore[union-attr]
    if len(observed) != 4 or any(int(row["games"]) != 64 or int(row["technical_games"]) != 0 for row in observed):
        raise RuntimeError("Self-play cap sweep is incomplete or technically invalid")
    ranked = sorted(observed, key=lambda row: (float(row["moves_per_sec"]), float(row["inference"]["inference_rows_per_sec"])), reverse=True)  # type: ignore[index]
    fastest = float(ranked[0]["moves_per_sec"])
    equivalent = [row for row in ranked if float(row["moves_per_sec"]) >= fastest * 0.95]
    winner_row = min(equivalent, key=lambda row: (row_cap(row), -float(row["inference"]["inference_rows_per_sec"])))  # type: ignore[index]
    return {
        "status": "CONFIRMED",
        "winner_cap": row_cap(winner_row),
        "winner_iteration": int(winner_row["iteration"]),
        "criterion": "same wait=6 ms; normalized by real moves/sec, rows/sec and batch telemetry; within 5% prefer smaller cap",
        "candidates": [
            {
                "iteration": int(row["iteration"]),
                "cap": row_cap(row),
                "wait_ms": float(row["execution"]["wait_ms"]),  # type: ignore[index]
                "moves_per_sec": float(row["moves_per_sec"]),
                "rows_per_sec": float(row["inference"]["inference_rows_per_sec"]),  # type: ignore[index]
                "mean_batch": float(row["inference"]["mean_inference_batch_rows"]),  # type: ignore[index]
                "p50_batch": float(row["inference"]["median_inference_batch_rows"]),  # type: ignore[index]
                "p95_batch": float(row["inference"]["p95_inference_batch_rows"]),  # type: ignore[index]
                "inference_calls": int(row["inference"]["inference_calls"]),  # type: ignore[index]
                "gpu_avg_pct": row["inference"].get("gpu_utilization_avg_pct"),  # type: ignore[union-attr]
                "gpu_peak_pct": row["inference"].get("gpu_utilization_peak_pct"),  # type: ignore[union-attr]
            }
            for row in sorted(observed, key=row_cap)
        ],
    }


def _wait_winner(rows: Sequence[Mapping[str, object]], cap: int) -> dict[str, object]:
    def row_cap(row: Mapping[str, object]) -> int:
        execution = row.get("execution", {})
        if not isinstance(execution, Mapping):
            return -1
        return int(execution.get("cap", execution.get("batch_cap", -1)))

    observed = [
        row for row in rows
        if row_cap(row) == int(cap)
        and float(row.get("execution", {}).get("wait_ms", -1)) in {1.0, 2.0, 4.0, 6.0}  # type: ignore[union-attr]
        and row.get("iteration") in {2, 3, 4, 5, 6, 7, 8}
    ]
    by_wait = {float(row["execution"]["wait_ms"]): row for row in observed}  # type: ignore[index]
    if set(by_wait) != {1.0, 2.0, 4.0, 6.0} or any(int(row["technical_games"]) != 0 for row in by_wait.values()):
        raise RuntimeError("Self-play wait sweep does not contain one valid observation for 1/2/4/6 ms")
    ranked = sorted(by_wait.values(), key=lambda row: (float(row["moves_per_sec"]), float(row["inference"]["inference_rows_per_sec"])), reverse=True)  # type: ignore[index]
    fastest = float(ranked[0]["moves_per_sec"])
    equivalent = [row for row in ranked if float(row["moves_per_sec"]) >= fastest * 0.95]
    winner = min(equivalent, key=lambda row: (float(row["execution"]["wait_ms"]), -float(row["inference"]["inference_rows_per_sec"])))  # type: ignore[index]
    monotonic = all(
        float(by_wait[left]["moves_per_sec"]) <= float(by_wait[right]["moves_per_sec"]) * 1.05
        for left, right in ((1.0, 2.0), (2.0, 4.0), (4.0, 6.0))
    )
    return {
        "status": "CONFIRMED",
        "winner_wait_ms": float(winner["execution"]["wait_ms"]),  # type: ignore[index]
        "winner_iteration": int(winner["iteration"]),
        "monotonic_or_meaningful_increase": monotonic,
        "criterion": "same selected cap; real rows/sec and moves/sec; within 5% prefer lower wait",
        "candidates": [
            {
                "iteration": int(row["iteration"]),
                "wait_ms": float(row["execution"]["wait_ms"]),  # type: ignore[index]
                "moves_per_sec": float(row["moves_per_sec"]),
                "rows_per_sec": float(row["inference"]["inference_rows_per_sec"]),  # type: ignore[index]
                "mean_batch": float(row["inference"]["mean_inference_batch_rows"]),  # type: ignore[index]
                "p50_batch": float(row["inference"]["median_inference_batch_rows"]),  # type: ignore[index]
                "p95_batch": float(row["inference"]["p95_inference_batch_rows"]),  # type: ignore[index]
                "inference_calls": int(row["inference"]["inference_calls"]),  # type: ignore[index]
            }
            for row in sorted(by_wait.values(), key=lambda item: float(item["execution"]["wait_ms"]))  # type: ignore[index]
        ],
    }


def _wilson(successes: float, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(max(0.0, p * (1.0 - p) / total + z * z / (4.0 * total * total))) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _arena_record_summary(summary: Mapping[str, object], *, wait_ms: float, wall: float, cpu: float, gpu: Mapping[str, object]) -> dict[str, object]:
    games = int(summary.get("games", 0))
    technical = int(summary.get("technical_games", 0))
    valid = int(summary.get("valid_games", 0))
    wins = int(summary.get("wins", 0))
    draws = int(summary.get("draws", 0))
    rows = int(summary.get("inference_rows", 0))
    rows_per_sec = rows / wall if wall else 0.0
    games_per_sec = games / wall if wall else 0.0
    enriched = dict(summary)
    enriched.update({
        "games": games,
        "technical_games": technical,
        "valid_games": valid,
        "wall_time_sec": wall,
        "games_per_sec": games_per_sec,
        "inference_rows_per_sec": rows_per_sec,
        "p50_inference_batch_rows": int(summary.get("p50_inference_batch_rows", 0)),
        "p95_inference_batch_rows": int(summary.get("p95_inference_batch_rows", 0)),
        "cpu_utilization_pct": 100.0 * cpu / (wall * max(1, os.cpu_count() or 1)) if wall else 0.0,
        "inference_batch_wait_ms": float(wait_ms),
        "arena_batch_size": 8,
        "workers": 16,
        "technical_outcomes_fail_closed": True,
        "wilson_95_percent_ci_score": _wilson(wins + 0.5 * draws, valid),
        "gpu_utilization_avg_pct": gpu.get("gpu_utilization_avg_pct"),
        "gpu_utilization_peak_pct": gpu.get("gpu_utilization_peak_pct"),
        "gpu_vram_peak_mb": gpu.get("gpu_vram_peak_mb"),
        "gpu_telemetry": dict(gpu),
        "performance_status": "PERFORMANCE DEGRADED" if float(summary.get("mean_inference_batch_rows", 0.0)) < 16.0 else "OK",
    })
    return enriched


def _arena(
    *,
    comparison: str,
    candidate_label: str,
    reference_label: str,
    wait_ms: float,
    starts: Sequence[Mapping[str, object]],
    startset_id: str,
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    output_dir = RUN_ROOT / "arena" / comparison
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("candidate") != candidate_label or summary.get("reference") != reference_label or float(summary.get("inference_batch_wait_ms", -1)) != float(wait_ms):
            raise RuntimeError(f"Existing Arena artifact has different contract: {comparison}")
        return summary
    if (output_dir / "games.jsonl").exists():
        raise RuntimeError(f"Arena has incomplete existing artifact; refusing to overwrite: {output_dir}")
    gpu = GpuSampler(RUN_ROOT / "telemetry" / f"arena-{comparison}-gpu.jsonl")
    cpu_before = _cpu_seconds()
    gpu.start()
    started = time.perf_counter()
    _state_update(status="RUNNING", current_atomic_unit=comparison, next_transition=None, scientific_validity="PENDING_ATOMIC_UNIT")
    try:
        summary = run_torus9_batched_arena(
            run_id=RUN_ID,
            comparison=comparison,
            candidate_path=candidate_path,
            reference_path=reference_path,
            candidate_label=candidate_label,
            reference_label=reference_label,
            starts=starts,
            master_seed=TORUS9_CURRENT_ARENA_MASTER_SEED,
            output_dir=output_dir,
            workers=16,
            arena_batch_size=8,
            inference_batch_wait_ms=float(wait_ms),
            device="cuda",
        )
    finally:
        wall = time.perf_counter() - started
        gpu_summary = gpu.stop()
    enriched = _arena_record_summary(summary, wait_ms=wait_ms, wall=wall, cpu=max(0.0, _cpu_seconds() - cpu_before), gpu=gpu_summary)
    enriched.update({
        "comparison": comparison,
        "candidate": candidate_label,
        "reference": reference_label,
        "startset_id": startset_id,
        "startset_hash": _fingerprint(starts),
        "games_expected": 64,
        "arena_contract": {
            "games": 64,
            "sims": 64,
            "cpuct": 1.25,
            "fpu": 0.0,
            "root_noise": False,
            "temperature": 0.0,
            "fast_search": False,
            "resign": False,
            "watchdog": TORUS9_ARENA_MOVE_LIMIT,
            "workers": 16,
            "batched": True,
            "arena_batch_size": 8,
            "paired_starts_color_swap": True,
            "komi": 0.5,
            "technical_outcomes": "fail-closed / excluded",
            "training": "PAUSED during Arena",
        },
    })
    if int(enriched["games"]) != 64:
        raise RuntimeError(f"Arena {comparison} did not complete 64 games")
    _atomic_write(summary_path, enriched)
    if float(enriched.get("mean_inference_batch_rows", 0.0)) < 16.0:
        enriched["performance_status"] = "PERFORMANCE DEGRADED"
    return enriched


def _ensure_startset() -> tuple[tuple[dict[str, object], ...], str]:
    path = RUN_ROOT / "arena" / "canonical-startset.json"
    if path.is_file():
        payload = _read_json(path)
        starts = tuple(payload["starts"])  # type: ignore[index]
        if len(starts) != 32 or payload.get("startset_hash") != _fingerprint(starts):
            raise RuntimeError("Persisted canonical Arena startset is invalid")
        return starts, str(payload["startset_id"])
    from gocube_golden.torus9 import generate_torus9_evaluation_starts

    starts = generate_torus9_evaluation_starts(master_seed=TORUS9_CURRENT_ARENA_MASTER_SEED, accepted_per_stratum=4)
    if len(starts) != 32:
        raise RuntimeError("Canonical Arena startset must contain 32 paired starts")
    startset_id = "torus9-golden-v3-canonical-32x2-v1"
    _atomic_write(path, {"startset_id": startset_id, "startset_hash": _fingerprint(starts), "starts": starts, "games": 64, "color_swap": True})
    return starts, startset_id


def _arena_performance_winner(arenas: Sequence[Mapping[str, object]]) -> tuple[float, dict[str, object]]:
    if not arenas:
        raise RuntimeError("No Arena telemetry available for wait selection")
    valid = [
        row for row in arenas
        if int(row.get("games", 0)) == 64
        and int(row.get("valid_games", 0)) > 0
        and float(row.get("inference_rows_per_sec", 0.0)) > 0.0
    ]
    if not valid:
        raise RuntimeError("Arena wait selection has no valid execution telemetry")
    ranked = sorted(valid, key=lambda row: (float(row.get("inference_rows_per_sec", 0.0)), float(row.get("games_per_sec", 0.0))), reverse=True)
    fastest = float(ranked[0].get("inference_rows_per_sec", 0.0))
    equivalent = [row for row in ranked if float(row.get("inference_rows_per_sec", 0.0)) >= fastest * 0.95]
    winner = min(equivalent, key=lambda row: float(row["inference_batch_wait_ms"]))
    return float(winner["inference_batch_wait_ms"]), {
        "winner_wait_ms": float(winner["inference_batch_wait_ms"]),
        "criterion": "normalized inference rows/sec primary; games/sec and batch telemetry secondary; within 5% prefer lower wait",
        "observations": [
            {
                "comparison": row["comparison"],
                "wait_ms": row["inference_batch_wait_ms"],
                "rows_per_sec": row["inference_rows_per_sec"],
                "games_per_sec": row["games_per_sec"],
                "mean_batch": row.get("mean_inference_batch_rows"),
                "p50_batch": row.get("p50_inference_batch_rows"),
                "p95_batch": row.get("p95_inference_batch_rows"),
                "inference_calls": row.get("inference_calls"),
                "gpu_avg_pct": row.get("gpu_utilization_avg_pct"),
                "cpu_pct": row.get("cpu_utilization_pct"),
            }
            for row in arenas
        ],
    }


def _write_execution_report(history: Sequence[Mapping[str, object]], arenas: Sequence[Mapping[str, object]], cap_selection: Mapping[str, object] | None, wait_selection: Mapping[str, object] | None, selfplay_frozen: Mapping[str, object] | None, arena_frozen: Mapping[str, object] | None, startset_id: str | None) -> None:
    report = {
        "report_schema": "torus9-golden-v3-continuation-v1",
        "run_id": RUN_ID,
        "base_commit": BASE_COMMIT,
        "current_profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775",
        "scientific_contract_unchanged": True,
        "device": "cuda",
        "workers": 16,
        "iterations": 100,
        "total_selfplay_games": len(history) * 64,
        "self_play": {
            "baseline_M0_M1": _read_json(RUN_ROOT / "iter-01-summary.json"),
            "cap_selection": cap_selection,
            "wait_selection": wait_selection,
            "freeze": selfplay_frozen,
            "rows": [
                {
                    "transition": f"M{int(row['iteration']) - 1}→M{row['iteration']}",
                    "cap": row["execution"].get("cap", row["execution"].get("batch_cap")),
                    "wait_ms": row["execution"].get("wait_ms"),
                    "time_sec": row["self_play_wall_time_sec"],
                    "moves_per_sec": row["moves_per_sec"],
                    "rows_per_sec": row["inference"].get("inference_rows_per_sec"),
                    "mean_batch": row["inference"].get("mean_inference_batch_rows"),
                    "p50": row["inference"].get("median_inference_batch_rows"),
                    "p95": row["inference"].get("p95_inference_batch_rows"),
                    "gpu_avg": row["inference"].get("gpu_utilization_avg_pct"),
                }
                for row in history
            ],
        },
        "arena": {
            "startset_id": startset_id,
            "rows": [
                {
                    "arena": row["comparison"],
                    "wait_ms": row["inference_batch_wait_ms"],
                    "games": row["games"],
                    "time_sec": row["wall_time_sec"],
                    "games_per_sec": row["games_per_sec"],
                    "rows_per_sec": row["inference_rows_per_sec"],
                    "mean_batch": row.get("mean_inference_batch_rows"),
                    "p95": row.get("p95_inference_batch_rows"),
                    "gpu_avg": row.get("gpu_utilization_avg_pct"),
                    "perf_status": row.get("performance_status"),
                    "W/L/D": row.get("W/L/D"),
                    "wilson_95_percent_ci_score": row.get("wilson_95_percent_ci_score"),
                }
                for row in arenas
            ],
            "wait_winner": wait_selection,
            "settings_freeze": arena_frozen,
        },
        "artifacts": {"run_root": str(RUN_ROOT)},
    }
    _atomic_write(RUN_ROOT / "continuation-report.json", report)


def _clean_stop(
    *,
    generation: int,
    last_atomic_unit: str,
    manifest: dict[str, object],
    history: Sequence[Mapping[str, object]],
    arenas: Sequence[Mapping[str, object]],
    cap_selection: Mapping[str, object] | None,
    wait_selection: Mapping[str, object] | None,
    selfplay_frozen: Mapping[str, object] | None,
    arena_frozen: Mapping[str, object] | None,
    startset_id: str,
) -> None:
    manifest.update({
        "status": "STOPPED_USER_REQUEST",
        "result": "STOPPED_USER_REQUEST_CLEAN_BOUNDARY",
        "stop_reason": "user-requested clean boundary stop",
        "stopped_after_atomic_unit": last_atomic_unit,
        "last_completed_iteration": generation,
        "next_transition_started": False,
        "resumable_from": f"M{generation}",
    })
    _atomic_write(RUN_ROOT / "manifest.json", manifest)
    _state_update(
        status="STOPPED_USER_REQUEST",
        stop_requested=True,
        last_atomic_unit=last_atomic_unit,
        reached_generation=generation,
        next_transition_started=False,
        run_resumable=True,
        scientific_validity="PASS",
    )
    _write_execution_report(history, arenas, cap_selection, wait_selection, selfplay_frozen, arena_frozen, startset_id)


def _backfill_missing_m5_arena() -> None:
    profile, profile_fp, _contract_value = _profile_and_contract()
    manifest = _validate_starting_run(profile_fp)
    complete = _complete_iterations()
    if not complete or complete[-1] != 100:
        raise RuntimeError("M5 Arena backfill is allowed only after M100 is complete")
    starts, startset_id = _ensure_startset()
    arenas: list[dict[str, object]] = []
    arena_root = RUN_ROOT / "arena"
    for summary_path in sorted(arena_root.glob("M*-vs-M*/summary.json")):
        arenas.append(_read_json(summary_path))
    backfilled: list[str] = []
    for comparison, candidate, reference, wait in (
        ("M5-vs-M1", "M5", "M1", 1.0),
        ("M5-vs-M0", "M5", "M0", 2.0),
    ):
        if any(row.get("comparison") == comparison for row in arenas):
            continue
        summary = _arena(
            comparison=comparison,
            candidate_label=candidate,
            reference_label=reference,
            wait_ms=wait,
            starts=starts,
            startset_id=startset_id,
            candidate_path=RUN_ROOT / "checkpoints" / f"{candidate}.pt",
            reference_path=RUN_ROOT / "checkpoints" / f"{reference}.pt",
        )
        arenas.append(summary)
        backfilled.append(comparison)
    cap_selection = _read_json(RUN_ROOT / "selfplay-cap-selection.json")
    wait_selection = _read_json(RUN_ROOT / "selfplay-wait-selection.json")
    selfplay_frozen = _read_json(RUN_ROOT / "selfplay-settings-freeze.json")
    arena_frozen = _read_json(RUN_ROOT / "arena-settings-freeze.json") if (RUN_ROOT / "arena-settings-freeze.json").is_file() else None
    _atomic_write(RUN_ROOT / "arena-backfill.json", {
        "status": "COMPLETE",
        "comparisons": backfilled,
        "reason": "Recovered after the continuation runner exited at the M5 cap-selection boundary before its scheduled M5 Arena controls.",
        "checkpoint_only": True,
        "scientific_contract_unchanged": True,
    })
    _write_execution_report(_load_history(complete), arenas, cap_selection, wait_selection, selfplay_frozen, arena_frozen, startset_id)
    manifest.update({"status": "COMPLETED", "result": "COMPLETED_M100", "last_completed_iteration": 100, "next_transition_started": False, "resumable_from": "M100"})
    _atomic_write(RUN_ROOT / "manifest.json", manifest)
    _state_update(status="COMPLETED", last_completed_iteration=100, last_atomic_unit="M99→M100", next_transition_started=False, next_transition=None, run_resumable=True, scientific_validity="PASS")


def run() -> None:
    global _STOP_REQUESTED
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    profile, profile_fp, contract = _profile_and_contract()
    manifest = _validate_starting_run(profile_fp)
    starts, startset_id = _ensure_startset()
    complete = _complete_iterations()
    if complete[-1] == 100:
        _state_update(status="COMPLETED", last_completed_iteration=100, next_transition_started=False, scientific_validity="PASS")
        return
    last = complete[-1]
    model = Torus9CurrentGraphNet().to("cuda")
    trainer = Torus9OwnershipScoreTrainer(
        model,
        score_loss_enabled=True,
        learning_rate=float(profile["training"]["learning_rate"]),  # type: ignore[index]
        weight_decay=float(profile["training"]["weight_decay"]),  # type: ignore[index]
        optimizer_steps_per_iteration=int(profile["training"]["optimizer_steps_per_iteration"]),  # type: ignore[index]
    )
    checkpoint_path = RUN_ROOT / "checkpoints" / f"M{last}.pt"
    metadata = torus9_load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=trainer.optimizer,
        expected={
            "run_id": RUN_ID,
            "checkpoint_label": f"M{last}",
            "profile_id": TORUS9_CURRENT_PROFILE_ID,
            "profile_fingerprint": profile_fp,
            "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
            "completed_games": last * 64,
            "device": "cuda",
            "device_locked": True,
            "ownership_loss_enabled": True,
            "score_loss_enabled": True,
        },
        device="cuda",
    )
    trainer.update_count = int(metadata["optimizer_updates"])
    trainer.samples_consumed = int(metadata["train_samples_consumed"])
    if trainer.assert_optimizer_continuity() != last * 80:
        raise RuntimeError("Optimizer continuation step does not match generation")
    replay = _rebuild_replay(last, profile)
    code = capture_code_identity(ROOT)
    history = _load_history(complete)
    arenas: list[dict[str, object]] = []
    arena_root = RUN_ROOT / "arena"
    for summary_path in sorted(arena_root.glob("M*-vs-M*/summary.json")):
        arenas.append(_read_json(summary_path))

    cap_selection: dict[str, object] | None = None
    wait_selection: dict[str, object] | None = None
    selfplay_frozen: dict[str, object] | None = None
    arena_frozen: dict[str, object] | None = None
    if (RUN_ROOT / "selfplay-cap-selection.json").is_file():
        cap_selection = _read_json(RUN_ROOT / "selfplay-cap-selection.json")
    if (RUN_ROOT / "selfplay-wait-selection.json").is_file():
        wait_selection = _read_json(RUN_ROOT / "selfplay-wait-selection.json")
    if (RUN_ROOT / "selfplay-settings-freeze.json").is_file():
        selfplay_frozen = _read_json(RUN_ROOT / "selfplay-settings-freeze.json")
    if (RUN_ROOT / "arena-settings-freeze.json").is_file():
        arena_frozen = _read_json(RUN_ROOT / "arena-settings-freeze.json")

    for iteration in range(last + 1, 101):
        if _stop_requested():
            _state_update(status="STOP_REQUESTED", stop_requested=True, next_transition_started=False, last_completed_iteration=iteration - 1, scientific_validity="PASS")
            return
        if iteration in {2, 3, 4, 5}:
            spec = CAP_SWEEP[iteration - 2]
            cap, wait_ms, phase = int(spec["cap"]), float(spec["wait_ms"]), str(spec["phase"])
        elif iteration in {6, 7, 8}:
            if cap_selection is None:
                cap_selection = _selfplay_cap_winner(history)
                _atomic_write(RUN_ROOT / "selfplay-cap-selection.json", cap_selection)
            cap, wait_ms, phase = int(cap_selection["winner_cap"]), float(WAIT_SWEEP[iteration - 6]["wait_ms"]), "wait_sweep"
        else:
            if cap_selection is None:
                cap_selection = _selfplay_cap_winner(history)
            if wait_selection is None and iteration == 9:
                wait_selection = _wait_winner(history, int(cap_selection["winner_cap"]))
                _atomic_write(RUN_ROOT / "selfplay-wait-selection.json", wait_selection)
            cap, wait_ms, phase = int(cap_selection["winner_cap"]), float(wait_selection["winner_wait_ms"]), "frozen"  # type: ignore[index]
        row = _run_iteration(
            iteration=iteration,
            cap=cap,
            wait_ms=wait_ms,
            phase=phase,
            profile=profile,
            profile_fp=profile_fp,
            contract=contract,
            model=model,
            trainer=trainer,
            replay=replay,
            code=code,
            manifest=manifest,
        )
        history.append(row)

        # A stop arriving during self-play/training is honored only after the
        # complete transition, before any separately scheduled Arena unit.
        if _stop_requested():
            _clean_stop(
                generation=iteration,
                last_atomic_unit=f"M{iteration - 1}→M{iteration}",
                manifest=manifest,
                history=history,
                arenas=arenas,
                cap_selection=cap_selection,
                wait_selection=wait_selection,
                selfplay_frozen=selfplay_frozen,
                arena_frozen=arena_frozen,
                startset_id=startset_id,
            )
            return

        if iteration == 5:
            cap_selection = _selfplay_cap_winner(history)
            _atomic_write(RUN_ROOT / "selfplay-cap-selection.json", cap_selection)
        if iteration == 8:
            cap_selection = _selfplay_cap_winner(history)
            wait_selection = _wait_winner(history, int(cap_selection["winner_cap"]))
            _atomic_write(RUN_ROOT / "selfplay-cap-selection.json", cap_selection)
            _atomic_write(RUN_ROOT / "selfplay-wait-selection.json", wait_selection)
        if iteration == 10 and wait_selection is not None:
            selfplay_frozen = {"generation": 10, "cap": int(cap_selection["winner_cap"]), "wait_ms": float(wait_selection["winner_wait_ms"]), "status": "FROZEN"}
            _atomic_write(RUN_ROOT / "selfplay-settings-freeze.json", selfplay_frozen)

        if iteration == 5:
            for comparison, candidate, reference, wait in (
                ("M5-vs-M1", "M5", "M1", 1.0),
                ("M5-vs-M0", "M5", "M0", 2.0),
            ):
                summary = _arena(
                    comparison=comparison,
                    candidate_label=candidate,
                    reference_label=reference,
                    wait_ms=wait,
                    starts=starts,
                    startset_id=startset_id,
                    candidate_path=RUN_ROOT / "checkpoints" / f"{candidate}.pt",
                    reference_path=RUN_ROOT / "checkpoints" / f"{reference}.pt",
                )
                arenas = [*arenas, summary]
                _state_update(last_atomic_unit=comparison, current_atomic_unit=None, next_transition=f"M5→M6", next_transition_started=False, scientific_validity="PASS")
                if _stop_requested():
                    _clean_stop(
                        generation=iteration,
                        last_atomic_unit=comparison,
                        manifest=manifest,
                        history=history,
                        arenas=arenas,
                        cap_selection=cap_selection,
                        wait_selection=wait_selection,
                        selfplay_frozen=selfplay_frozen,
                        arena_frozen=arena_frozen,
                        startset_id=startset_id,
                    )
                    return
        if iteration == 10:
            summary = _arena(
                comparison="M10-vs-M5",
                candidate_label="M10",
                reference_label="M5",
                wait_ms=4.0,
                starts=starts,
                startset_id=startset_id,
                candidate_path=RUN_ROOT / "checkpoints" / "M10.pt",
                reference_path=RUN_ROOT / "checkpoints" / "M5.pt",
            )
            arenas = [*arenas, summary]
            _write_execution_report(history, arenas, cap_selection, wait_selection, selfplay_frozen, arena_frozen, startset_id)

        if iteration == 10:
            arena_wait, arena_selection = _arena_performance_winner(arenas)
            _atomic_write(RUN_ROOT / "arena-wait-selection.json", arena_selection)
            if arena_wait in {1.0, 2.0}:
                next_arena = {"generation": 20, "wait_ms": arena_wait, "phase": "confirmation"}
            else:
                next_arena = {"generation": 20, "wait_ms": 6.0, "phase": "adaptive-probe-6"}
            _atomic_write(RUN_ROOT / "arena-next-schedule.json", {"provisional_winner_wait_ms": arena_wait, "next": next_arena})

        if iteration == 20:
            schedule = _read_json(RUN_ROOT / "arena-next-schedule.json")
            wait = float(schedule["next"]["wait_ms"])  # type: ignore[index]
            summary = _arena(
                comparison="M20-vs-M10",
                candidate_label="M20",
                reference_label="M10",
                wait_ms=wait,
                starts=starts,
                startset_id=startset_id,
                candidate_path=RUN_ROOT / "checkpoints" / "M20.pt",
                reference_path=RUN_ROOT / "checkpoints" / "M10.pt",
            )
            arenas = [*arenas, summary]
            provisional = float(schedule["provisional_winner_wait_ms"])
            if provisional in {1.0, 2.0}:
                arena_frozen = {"generation": 20, "wait_ms": provisional, "arena_batch_size": 8, "status": "FROZEN", "confirmation": "M20-vs-M10"}
                _atomic_write(RUN_ROOT / "arena-settings-freeze.json", arena_frozen)
            elif provisional == 4.0:
                if float(summary["inference_rows_per_sec"]) >= float(next(row for row in arenas if row["comparison"] == "M10-vs-M5")["inference_rows_per_sec"]) * 1.05:
                    _atomic_write(RUN_ROOT / "arena-next-schedule.json", {"provisional_winner_wait_ms": 4.0, "next": {"generation": 30, "wait_ms": 8.0, "phase": "adaptive-probe-8"}})
                else:
                    _atomic_write(RUN_ROOT / "arena-next-schedule.json", {"provisional_winner_wait_ms": 4.0, "next": {"generation": 30, "wait_ms": 4.0, "phase": "confirmation"}})
            _atomic_write(RUN_ROOT / "arena-wait-selection.json", {"provisional_winner_wait_ms": provisional, "after_M20": summary, "freeze": arena_frozen})

        if iteration == 30 and arena_frozen is None:
            schedule = _read_json(RUN_ROOT / "arena-next-schedule.json")
            wait = float(schedule["next"]["wait_ms"])  # type: ignore[index]
            summary = _arena(
                comparison="M30-vs-M20",
                candidate_label="M30",
                reference_label="M20",
                wait_ms=wait,
                starts=starts,
                startset_id=startset_id,
                candidate_path=RUN_ROOT / "checkpoints" / "M30.pt",
                reference_path=RUN_ROOT / "checkpoints" / "M20.pt",
            )
            arenas = [*arenas, summary]
            if wait == 4.0:
                arena_frozen = {"generation": 30, "wait_ms": 4.0, "arena_batch_size": 8, "status": "FROZEN", "confirmation": "M30-vs-M20"}
                _atomic_write(RUN_ROOT / "arena-settings-freeze.json", arena_frozen)
            elif wait == 8.0:
                m20 = next(row for row in arenas if row["comparison"] == "M20-vs-M10")
                if float(summary["inference_rows_per_sec"]) >= float(m20["inference_rows_per_sec"]) * 1.05:
                    _atomic_write(RUN_ROOT / "arena-next-schedule.json", {"provisional_winner_wait_ms": 6.0, "next": {"generation": 40, "wait_ms": 10.0, "phase": "adaptive-probe-10"}})
                else:
                    _atomic_write(RUN_ROOT / "arena-next-schedule.json", {"provisional_winner_wait_ms": 6.0, "next": {"generation": 40, "wait_ms": 6.0, "phase": "confirmation"}})

        if iteration == 40 and arena_frozen is None:
            schedule = _read_json(RUN_ROOT / "arena-next-schedule.json")
            wait = float(schedule["next"]["wait_ms"])  # type: ignore[index]
            summary = _arena(
                comparison="M40-vs-M30",
                candidate_label="M40",
                reference_label="M30",
                wait_ms=wait,
                starts=starts,
                startset_id=startset_id,
                candidate_path=RUN_ROOT / "checkpoints" / "M40.pt",
                reference_path=RUN_ROOT / "checkpoints" / "M30.pt",
            )
            arenas = [*arenas, summary]
            if wait == 6.0:
                arena_frozen = {"generation": 40, "wait_ms": 6.0, "arena_batch_size": 8, "status": "FROZEN", "confirmation": "M40-vs-M30"}
                _atomic_write(RUN_ROOT / "arena-settings-freeze.json", arena_frozen)
            elif wait == 10.0:
                m30 = next(row for row in arenas if row["comparison"] == "M30-vs-M20")
                winner = 10.0 if float(summary["inference_rows_per_sec"]) >= float(m30["inference_rows_per_sec"]) * 1.05 else 8.0
                _atomic_write(RUN_ROOT / "arena-next-schedule.json", {"provisional_winner_wait_ms": winner, "next": {"generation": 50, "wait_ms": winner, "phase": "confirmation"}})

        if iteration == 50 and arena_frozen is None:
            schedule = _read_json(RUN_ROOT / "arena-next-schedule.json")
            wait = float(schedule["next"]["wait_ms"])  # type: ignore[index]
            summary = _arena(
                comparison="M50-vs-M40",
                candidate_label="M50",
                reference_label="M40",
                wait_ms=wait,
                starts=starts,
                startset_id=startset_id,
                candidate_path=RUN_ROOT / "checkpoints" / "M50.pt",
                reference_path=RUN_ROOT / "checkpoints" / "M40.pt",
            )
            arenas = [*arenas, summary]
            arena_frozen = {"generation": 50, "wait_ms": wait, "arena_batch_size": 8, "status": "FROZEN", "confirmation": "M50-vs-M40"}
            _atomic_write(RUN_ROOT / "arena-settings-freeze.json", arena_frozen)

        if iteration >= 30 and arena_frozen is not None and iteration in {30, 40, 50, 60, 70, 80, 90, 100}:
            if any(row["comparison"] == f"M{iteration}-vs-M{iteration - 10}" for row in arenas):
                _write_execution_report(history, arenas, cap_selection, wait_selection, selfplay_frozen, arena_frozen, startset_id)
                continue
            candidate = f"M{iteration}"
            reference = f"M{iteration - 10}"
            summary = _arena(
                comparison=f"{candidate}-vs-{reference}",
                candidate_label=candidate,
                reference_label=reference,
                wait_ms=float(arena_frozen["wait_ms"]),
                starts=starts,
                startset_id=startset_id,
                candidate_path=RUN_ROOT / "checkpoints" / f"{candidate}.pt",
                reference_path=RUN_ROOT / "checkpoints" / f"{reference}.pt",
            )
            arenas = [*arenas, summary]

        _write_execution_report(history, arenas, cap_selection, wait_selection, selfplay_frozen, arena_frozen, startset_id)
        if iteration == 10 and arena_frozen is not None:
            _state_update(last_atomic_unit="M10-vs-M5", current_atomic_unit=None, next_transition="M10→M11", next_transition_started=False, scientific_validity="PASS")

    final = {
        "status": "COMPLETED",
        "reached_generation": 100,
        "next_transition_started": False,
        "run_resumable": True,
        "final_checkpoint": str(RUN_ROOT / "checkpoints" / "M100.pt"),
        "replay_state": str(RUN_ROOT / "replay" / "rolling-after-100.jsonl"),
        "scientific_contract_unchanged": True,
    }
    _atomic_write(RUN_ROOT / "continuation-final.json", final)
    manifest.update({"status": "COMPLETED", "result": "COMPLETED_M100", "resumable_from": "M100", "last_completed_iteration": 100, "next_transition_started": False, "scientific_contract_unchanged": True})
    _atomic_write(RUN_ROOT / "manifest.json", manifest)
    _state_update(status="COMPLETED", last_completed_iteration=100, last_atomic_unit="M99→M100", next_transition_started=False, next_transition=None, run_resumable=True, scientific_validity="PASS")
    _write_execution_report(history, arenas, cap_selection, wait_selection, selfplay_frozen, arena_frozen, startset_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-stop", action="store_true")
    parser.add_argument("--backfill-m5-arena", action="store_true")
    args = parser.parse_args()
    if args.request_stop:
        request_stop()
        print(json.dumps({"stop_request_written": str(STOP_PATH)}, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Scientific validity fail-closed: CUDA is unavailable")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.backfill_m5_arena:
        _backfill_missing_m5_arena()
        return
    if not STATE_PATH.exists():
        _atomic_write(STATE_PATH, {"status": "STARTING", "run_id": RUN_ID, "stop_requested": _stop_requested(), "last_completed_iteration": 1, "scientific_validity": "PENDING"})
    run()


if __name__ == "__main__":
    main()
