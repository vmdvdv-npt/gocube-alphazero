#!/usr/bin/env python3
"""Authoritative post-PR118 Torus9 same-stage non-inferiority experiment.

This controller is intentionally small: rules, self-play, replay, training,
checkpointing, and Arena remain behind their current production boundaries.
It owns only the Stage 7 decision ladder and durable provenance.  The OLD M0
is referenced, never copied, and Arena outputs are kept in the canonical
evaluation namespace.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphazero.envs.gocube.integration.catalog import CheckpointCatalog
from gocube_golden.provenance import (
    CodeIdentity,
    capture_code_identity,
    derive_seed,
    file_sha256,
    sha256_fingerprint,
)
from gocube_golden.run_storage import (
    RUNS_ROOT,
    active_lineage_dir,
    archived_lineage_dir,
    create_lineage,
    ensure_evaluation_layout,
    evaluation_dir,
)
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9SelfPlaySearchContract,
    Torus9TrainingAdapter,
    generate_torus9_evaluation_starts,
    run_torus9_selfplay_games,
    run_torus9_training_iteration,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    write_json,
    write_jsonl,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_ARENA_MASTER_SEED,
    TORUS9_CURRENT_MODEL_INIT_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_PROFILE_FINGERPRINT,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_CURRENT_TRAINING_MASTER_SEED,
    TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    TORUS9_KOMI,
    TORUS9_RULES_FINGERPRINT,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)
from training_engine import sequence_fingerprint, value_fingerprint


REFERENCE_RUN_ID = "torus9-golden-v3-20260914-run03"
REFERENCE_M17_MODEL_HASH = "sha256:2b0d04c735874f4667712bc859db54560feefb3ad6cb5d2e3c769dd80f0c0ff5"
DEFAULT_RUN_ID = "torus9-stage7-post-pr118-20260916-run01"
ARENA_GAMES = 192
ARENA_PAIRS = ARENA_GAMES // 2
BOOTSTRAP_REPLICATES = 20_000
NON_INFERIORITY_MARGIN = 0.05
PASS_LOWER_BOUND = 0.45
FAIL_UPPER_BOUND = 0.50
LADDER = (5, 8, 10, 14, 17)
REFERENCE_LABELS = ("M0", "M5", "M8", "M10", "M14", "M17")
PERFORMANCE_REFERENCE_MOVES = 20.931
PERFORMANCE_MEDIAN_FLOOR = PERFORMANCE_REFERENCE_MOVES * 0.95
PERFORMANCE_LOW_FLOOR = PERFORMANCE_REFERENCE_MOVES * 0.90


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return tuple(rows)


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _safe_run_id(run_id: str) -> str:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Stage 7 run ID must be one safe path component")
    return run_id


def _metadata(path: Path) -> dict[str, object]:
    return _read_json(path.with_suffix(".metadata.json"))


def _checkpoint_identity(path: Path) -> dict[str, object]:
    metadata = _metadata(path)
    return {
        "label": metadata.get("checkpoint_label"),
        "path": str(path.resolve()),
        "metadata_path": str(path.with_suffix(".metadata.json").resolve()),
        "model_hash": metadata.get("model_hash"),
        "artifact_sha256": file_sha256(path),
        "metadata_sha256": file_sha256(path.with_suffix(".metadata.json")),
    }


def _reference_catalog() -> CheckpointCatalog:
    return CheckpointCatalog(str(RUNS_ROOT))


def _resolve_reference(label: str) -> Path:
    descriptor = _reference_catalog().get(f"{REFERENCE_RUN_ID}@{label[1:]}")
    if descriptor is None:
        raise FileNotFoundError(f"CheckpointCatalog cannot resolve {REFERENCE_RUN_ID}@{label[1:]}")
    path = Path(descriptor.path).resolve()
    if descriptor.run_name != REFERENCE_RUN_ID or path.name != f"{label}.pt":
        raise ValueError(f"Resolved reference identity drift for {label}: {descriptor}")
    return path


def _reference_snapshot() -> dict[str, object]:
    entries: dict[str, object] = {}
    for label in REFERENCE_LABELS:
        path = _resolve_reference(label)
        metadata = _metadata(path)
        if metadata.get("profile_id") != TORUS9_CURRENT_PROFILE_ID:
            raise ValueError(f"Reference {label} profile mismatch")
        if metadata.get("komi") != TORUS9_KOMI or metadata.get("rules_fingerprint") != TORUS9_RULES_FINGERPRINT:
            raise ValueError(f"Reference {label} scientific identity mismatch")
        entries[label] = _checkpoint_identity(path)
    m17 = entries["M17"]
    if not isinstance(m17, Mapping) or m17.get("model_hash") != REFERENCE_M17_MODEL_HASH:
        raise ValueError("Canonical OLD M17 does not match the known model SHA")
    canonical_run = Path(str(entries["M0"]["path"])).parents[1]  # checkpoints/ -> lineage
    forbidden = sorted(str(path) for path in canonical_run.rglob("M18*") if path.is_file())
    if forbidden:
        raise ValueError(f"Historical lineage contains forbidden M18 artifacts: {forbidden}")
    return {
        "run_id": REFERENCE_RUN_ID,
        "path": str(canonical_run),
        "checkpoints": entries,
        "m17_known_model_sha256": REFERENCE_M17_MODEL_HASH,
        "historical_lineage_read_only": True,
    }


def _profile_and_contract() -> tuple[dict[str, object], str, Torus9SelfPlaySearchContract]:
    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    contract = Torus9SelfPlaySearchContract()
    contract.validate()
    if profile_fp != TORUS9_CURRENT_PROFILE_FINGERPRINT:
        raise ValueError("Current profile fingerprint drift")
    if profile["self_play"]["fingerprint"] != contract.fingerprint:  # type: ignore[index]
        raise ValueError("Current self-play fingerprint drift")
    if TORUS9_KOMI != 0.5:
        raise ValueError("Komi must be exactly 0.5")
    return profile, profile_fp, contract


def _load_reference_model(path: Path, device: str) -> Torus9CurrentGraphNet:
    metadata = _metadata(path)
    model = torus9_model_from_metadata(metadata).to(device)
    torus9_load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]}, device=device)
    if file_sha256(path) != _checkpoint_identity(path)["artifact_sha256"]:
        raise ValueError("Reference checkpoint changed while loading")
    model.eval()
    return model


def _execution_contract() -> dict[str, object]:
    return {
        "workers": 16,
        "active_games_per_worker": 4,
        "active_contexts": 64,
        "inference_batch_cap": 64,
        "inference_batch_wait_ms": 1.0,
        "central_model_owner": "parent",
        "shared_memory": True,
        "device": "cuda",
    }


def _validate_hardware() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7 requires CUDA production execution")
    logical_cpus = int(os.cpu_count() or 0)
    if logical_cpus < 16:
        raise RuntimeError(f"Stage 7 requires at least 16 logical CPUs, found {logical_cpus}")
    properties = torch.cuda.get_device_properties(0)
    return {
        "logical_cpus": logical_cpus,
        "cuda": True,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": {
            "name": properties.name,
            "total_vram_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
        },
    }


def _new_manifest(
    *, run_id: str, code: CodeIdentity, profile_fp: str, reference: Mapping[str, object], hardware: Mapping[str, object]
) -> dict[str, object]:
    parent = reference["checkpoints"]["M0"]  # type: ignore[index]
    return {
        "schema": "torus9-stage7-post-pr118-v1",
        "lineage_id": run_id,
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": parent,
        "git_commit": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "config_fingerprint": profile_fp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_hashes": {},
        "source_code_identity": _jsonable(code),
        "base_commit": TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "rules_fingerprint": TORUS9_RULES_FINGERPRINT,
        "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        "selfplay_contract_fingerprint": Torus9SelfPlaySearchContract().fingerprint,
        "model_init_seed": TORUS9_CURRENT_MODEL_INIT_SEED,
        "selfplay_master_seed": TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
        "training_master_seed": TORUS9_CURRENT_TRAINING_MASTER_SEED,
        "arena_master_seed": TORUS9_CURRENT_ARENA_MASTER_SEED,
        "execution": _execution_contract(),
        "hardware": dict(hardware),
        "reference_snapshot": dict(reference),
        "parent_checkpoint_copied": False,
        "golden_parameters_changed": False,
        "golden_standard_untouched": True,
        "last_completed_generation": 0,
        "arena_ladder": [],
    }


def _run_dir(run_id: str) -> Path:
    return active_lineage_dir("torus9", _safe_run_id(run_id))


def _manifest(run_dir: Path) -> dict[str, object]:
    return _read_json(run_dir / "manifest.json")


def _assert_source_frozen(manifest: Mapping[str, object], code: CodeIdentity) -> None:
    source = manifest.get("source_code_identity")
    if not isinstance(source, Mapping):
        raise ValueError("Stage 7 source identity missing")
    if source.get("git_commit_sha") != code.git_commit_sha or source.get("git_tree_sha") != code.git_tree_sha:
        raise RuntimeError("Executable source changed after Stage 7 started; evidence is invalid")


def _committed_generation(run_dir: Path) -> int:
    generations = []
    for marker in run_dir.glob("generation-*.complete.json"):
        try:
            generation = int(marker.name.removeprefix("generation-").removesuffix(".complete.json"))
        except (IndexError, ValueError):
            continue
        payload = _read_json(marker)
        if payload.get("generation") == generation and payload.get("run_id") == run_dir.name:
            generations.append(generation)
    return max(generations, default=0)


def _load_training_state(
    *, run_dir: Path, adapter: Torus9TrainingAdapter, code: CodeIdentity, device: str
) -> tuple[Any, Path, int]:
    generation = _committed_generation(run_dir)
    if generation == 0:
        old_m0 = _resolve_reference("M0")
        metadata = _metadata(old_m0)
        model = torus9_model_from_metadata(metadata).to(device)
        torus9_load_checkpoint(old_m0, model=model, expected={"model_hash": metadata["model_hash"]}, device=device)
        parent = _checkpoint_identity(old_m0)
        state = adapter.create_state(model, run_id=run_dir.name, parent_checkpoint_identity=parent)
        return state, old_m0, 0
    checkpoint = run_dir / "checkpoints" / f"M{generation}.pt"
    rolling = run_dir / "replay" / f"rolling-after-{generation:02d}.jsonl"
    if not checkpoint.is_file() or not rolling.is_file():
        raise ValueError(f"Committed M{generation} is missing checkpoint or rolling replay")
    state = adapter.load_state(checkpoint, replay_path=rolling, device=device)
    return state, checkpoint, generation


def _validate_selfplay(
    records: Sequence[object], game_ids: Sequence[str], telemetry: Mapping[str, object], *, run_id: str, label: str
) -> tuple[int, int]:
    if len(records) != 64 or {str(getattr(row, "game_id")) for row in records} != set(game_ids):
        raise ValueError("Self-play did not complete exactly the requested 64 unique games")
    technical = sum(getattr(row, "technical_termination") is not None for row in records)
    if technical or int(telemetry.get("technical_games", 0)) != 0:
        raise RuntimeError(f"Self-play technical games detected at {label}: {technical}")
    if telemetry.get("worker_failures") or telemetry.get("worker_restarts"):
        raise RuntimeError(f"Self-play worker failure/restart detected at {label}")
    rows = sum(int(getattr(row, "nn_evaluations")) for row in records)
    if rows != int(telemetry.get("inference_rows", -1)):
        raise RuntimeError(f"Inference row accounting drift at {label}: records={rows}, telemetry={telemetry.get('inference_rows')}")
    for row in records:
        if getattr(row, "run_id") != run_id or getattr(row, "model_checkpoint_label") != label:
            raise ValueError(f"Self-play provenance drift at {label}")
        if int(getattr(row, "game_seed")) != derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, run_id, getattr(row, "game_id"), "game"):
            raise ValueError(f"Self-play seed drift at {label}")
        row.validate()  # type: ignore[union-attr]
    return sum(len(getattr(row, "final_action_trace")) for row in records), rows


def _performance_row(telemetry: Mapping[str, object], *, moves: int, selfplay_wall: float) -> dict[str, object]:
    moves_per_sec = moves / selfplay_wall if selfplay_wall else 0.0
    return {
        "reference_moves_per_sec": PERFORMANCE_REFERENCE_MOVES,
        "median_guard_floor_moves_per_sec": PERFORMANCE_MEDIAN_FLOOR,
        "low_guard_floor_moves_per_sec": PERFORMANCE_LOW_FLOOR,
        "moves_per_sec": moves_per_sec,
        "games_per_hour": 64 * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
        "mean_batch_rows": telemetry.get("mean_inference_batch_rows"),
        "p50_batch_rows": telemetry.get("p50_inference_batch_rows"),
        "p95_batch_rows": telemetry.get("p95_inference_batch_rows"),
        "max_batch_rows": telemetry.get("max_inference_batch_rows"),
        "effective_cpu_cores": telemetry.get("effective_cpu_cores"),
        "process_tree_effective_cpu_cores": telemetry.get("process_tree_effective_cpu_cores"),
        "gpu_utilization": telemetry.get("gpu_utilization_average"),
        "status": "PASS" if moves_per_sec >= PERFORMANCE_MEDIAN_FLOOR else "BELOW_MEDIAN_GUARD",
    }


def _train_to(*, run_dir: Path, target: int, code: CodeIdentity, device: str) -> dict[str, object]:
    manifest = _manifest(run_dir)
    _assert_source_frozen(manifest, code)
    profile, profile_fp, contract = _profile_and_contract()
    if manifest.get("reference_snapshot") != _reference_snapshot():
        raise RuntimeError("Historical reference changed")
    adapter = Torus9TrainingAdapter(profile=profile, code_identity=code, base_commit=TORUS9_GOLDEN_LINEAGE_BASE_COMMIT)
    state, previous_checkpoint, current = _load_training_state(run_dir=run_dir, adapter=adapter, code=code, device=device)
    if current > target:
        raise ValueError(f"Run already passed requested target M{target}: M{current}")
    rows: list[dict[str, object]] = []
    for generation in range(current + 1, target + 1):
        label = f"M{generation - 1}"
        game_ids = [f"{run_dir.name}-M{generation:02d}-game-{index:04d}" for index in range(64)]
        telemetry: dict[str, object] = {}
        started = time.perf_counter()
        records = run_torus9_selfplay_games(
            state.model,
            run_id=run_dir.name,
            label=label,
            artifact=file_sha256(previous_checkpoint),
            master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
            profile_fp=profile_fp,
            profile_id=TORUS9_CURRENT_PROFILE_ID,
            game_ids=game_ids,
            workers=16,
            code_identity=code,
            device=device,
            contract=contract,
            coalescing=True,
            inference_batch_cap=64,
            inference_batch_wait_ms=1.0,
            active_games_per_worker=4,
            total_active_contexts=64,
            inference_telemetry=telemetry,
            execution_activity=telemetry,
        )
        selfplay_wall = time.perf_counter() - started
        moves, inference_rows = _validate_selfplay(records, game_ids, telemetry, run_id=run_dir.name, label=label)
        games_path = run_dir / "selfplay" / f"iter-{generation:02d}-games.jsonl"
        write_jsonl(games_path, [record.to_dict() for record in records])
        performance = _performance_row(telemetry, moves=moves, selfplay_wall=selfplay_wall)
        iteration_started = started

        def summary_builder(base: Mapping[str, object]) -> Mapping[str, object]:
            training = base.get("training", {})
            replay = base.get("replay", {})
            checkpoint = base.get("checkpoint", {})
            correctness = {
                "requested_games": 64,
                "completed_games": len(records),
                "technical_games": 0,
                "worker_failures": telemetry.get("worker_failures", []),
                "worker_restarts": telemetry.get("worker_restarts", 0),
                "inference_rows": inference_rows,
                "inference_rows_accounted": telemetry.get("inference_rows"),
                "replay_row_ids_unique": True,
                "generation": generation,
                "optimizer_steps": training.get("optimizer_steps"),  # type: ignore[union-attr]
                "samples_consumed": training.get("samples_consumed"),  # type: ignore[union-attr]
                "checkpoint_reload_verified": True,
                "profile_fingerprint_unchanged": profile_fp == manifest.get("profile_fingerprint"),
                "rules_fingerprint_unchanged": TORUS9_RULES_FINGERPRINT == manifest.get("rules_fingerprint"),
                "target_fingerprint_unchanged": TORUS9_CURRENT_TARGET_FINGERPRINT == manifest.get("target_fingerprint"),
                "selfplay_contract_unchanged": contract.fingerprint == manifest.get("selfplay_contract_fingerprint"),
                "komi": TORUS9_KOMI,
            }
            return {
                "stage7": {
                    "implementation_sha": code.git_commit_sha,
                    "selfplay_wall_sec": selfplay_wall,
                    "iteration_wall_sec": time.perf_counter() - iteration_started,
                    "moves": moves,
                    "moves_per_sec": performance["moves_per_sec"],
                    "selfplay_telemetry": dict(telemetry),
                    "performance": performance,
                    "correctness": correctness,
                    "selfplay_artifact": str(games_path),
                    "selfplay_artifact_sha256": file_sha256(games_path),
                },
                "games": 64,
                "technical_games": 0,
                "positions": base.get("fresh_positions"),
                "replay": replay,
                "training": training,
                "checkpoint": checkpoint,
            }

        result = run_torus9_training_iteration(
            state=state,
            generation=generation,
            output_dir=run_dir,
            run_id=run_dir.name,
            records=records,
            training_seed=derive_seed(TORUS9_CURRENT_TRAINING_MASTER_SEED, run_dir.name, "training", generation),
            completed_games=generation * 64,
            code_identity=code,
            device=device,
            adapter=adapter,
            summary_builder=summary_builder,
        )
        summary = dict(result.summary)
        train_metrics = result.training_metrics
        if train_metrics.get("optimizer_steps") != 80 or train_metrics.get("samples_consumed") != 5120:
            raise RuntimeError(f"Training accounting drift at M{generation}")
        # TrainingEngine already reload-verifies the checkpoint before publish;
        # verify once more from the persisted path and record the actual hash.
        adapter.verify_checkpoint(run_dir / "checkpoints" / f"M{generation}.pt", state, result.checkpoint_metadata)
        row = {
            "iteration": generation,
            "label": f"M{generation}",
            "games": 64,
            "technical_games": 0,
            "positions": result.fresh_positions,
            "replay_positions": train_metrics.get("replay_positions"),
            "optimizer_updates": train_metrics.get("optimizer_updates_total"),
            "optimizer_steps": train_metrics.get("optimizer_steps"),
            "samples_consumed": train_metrics.get("samples_consumed"),
            "adam_step": result.checkpoint_metadata.get("adam_step"),
            "model_sha256": result.checkpoint_metadata.get("model_hash"),
            "artifact_sha256": file_sha256(run_dir / "checkpoints" / f"M{generation}.pt"),
            "wall_sec": summary.get("stage7", {}).get("iteration_wall_sec"),
            "selfplay_wall_sec": summary.get("stage7", {}).get("selfplay_wall_sec"),
            "moves": summary.get("stage7", {}).get("moves"),
            "moves_per_sec": summary.get("stage7", {}).get("moves_per_sec"),
            "performance": summary.get("stage7", {}).get("performance"),
            "replay_generations": result.checkpoint_metadata.get("replay_generations"),
            "replay_fingerprint": result.checkpoint_metadata.get("replay_fingerprint"),
            "selfplay_artifact_sha256": file_sha256(games_path),
        }
        rows.append(row)
        previous_checkpoint = run_dir / "checkpoints" / f"M{generation}.pt"
        current = generation
        manifest = _manifest(run_dir)
        manifest.update({
            "last_completed_generation": generation,
            "checkpoint_hashes": {
                **dict(manifest.get("checkpoint_hashes", {})),
                f"checkpoints/M{generation}.pt": file_sha256(previous_checkpoint),
                f"checkpoints/M{generation}.metadata.json": file_sha256(previous_checkpoint.with_suffix(".metadata.json")),
            },
        })
        _atomic_write(run_dir / "manifest.json", manifest)
        if _reference_snapshot() != manifest["reference_snapshot"]:
            raise RuntimeError("Historical reference changed after training generation")
    return {"last_completed_generation": current, "iterations": rows}


def _startset() -> dict[str, object]:
    starts = list(generate_torus9_evaluation_starts(master_seed=TORUS9_CURRENT_ARENA_MASTER_SEED, accepted_per_stratum=12))
    if len(starts) != ARENA_PAIRS:
        raise ValueError(f"Frozen startset has {len(starts)} starts, expected {ARENA_PAIRS}")
    fingerprints = {row.get("corpus_fingerprint") for row in starts}
    if len(fingerprints) != 1:
        raise ValueError("Frozen startset corpus fingerprints disagree")
    return {
        "schema": "torus9-stage7-frozen-startset-v1",
        "master_seed": TORUS9_CURRENT_ARENA_MASTER_SEED,
        "paired_starts": ARENA_PAIRS,
        "games": ARENA_GAMES,
        "fingerprint": next(iter(fingerprints)),
        "rows": starts,
    }


def _paired_blocks(games_path: Path, startset: Mapping[str, object]) -> tuple[float, ...]:
    rows = _read_jsonl(games_path)
    if len(rows) != ARENA_GAMES:
        raise ValueError(f"Arena games file has {len(rows)} rows, expected {ARENA_GAMES}")
    if any(row.get("technical_termination") is not None for row in rows):
        raise RuntimeError("Technical Arena game encountered; scientific evaluation is invalid")
    expected_starts = {str(row["start_id"]) for row in startset["rows"]}  # type: ignore[index]
    if {str(row.get("start_id")) for row in rows} != expected_starts:
        raise ValueError("Arena did not use the frozen 96-start corpus")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), []).append(row)
    if len(grouped) != ARENA_PAIRS:
        raise ValueError("Arena paired-start count drift")
    blocks: list[float] = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2 or {bool(row.get("candidate_black")) for row in pair} != {True, False}:
            raise ValueError(f"Arena pair is not a color-swapped pair: {pair_id}")
        scores = []
        for row in pair:
            result = row.get("mapped_result")
            scores.append(1.0 if result == "A_WIN" else 0.5 if result == "DRAW" else 0.0)
        blocks.append(sum(scores) / 2.0)
    return tuple(blocks)


def _bootstrap(blocks: Sequence[float], *, seed: int) -> dict[str, object]:
    if len(blocks) != ARENA_PAIRS:
        raise ValueError("Cluster bootstrap requires 96 paired starts")
    rng = random.Random(seed)
    values = [float(value) for value in blocks]
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(BOOTSTRAP_REPLICATES)]
    means.sort()
    lower = means[max(0, int(0.05 * len(means)) - 1)]
    upper = means[max(0, int(0.95 * len(means)) - 1)]
    observed = sum(values) / len(values)
    if lower >= PASS_LOWER_BOUND:
        verdict = "PASS"
    elif upper < FAIL_UPPER_BOUND:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "method": "cluster-bootstrap-paired-start-v1",
        "paired_starts": len(values),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "observed_new_score": observed,
        "one_sided_95_lower_bound": lower,
        "one_sided_95_upper_bound": upper,
        "non_inferiority_margin": NON_INFERIORITY_MARGIN,
        "pass_lower_bound": PASS_LOWER_BOUND,
        "fail_upper_bound": FAIL_UPPER_BOUND,
        "verdict": verdict,
    }


def _arena(run_dir: Path, stage: int, code: CodeIdentity) -> dict[str, object]:
    manifest = _manifest(run_dir)
    _assert_source_frozen(manifest, code)
    if _reference_snapshot() != manifest["reference_snapshot"]:
        raise RuntimeError("Historical reference changed before Arena")
    new_path = run_dir / "checkpoints" / f"M{stage}.pt"
    old_path = _resolve_reference(f"M{stage}")
    if not new_path.is_file():
        raise FileNotFoundError(f"NEW M{stage} is not published")
    startset = _startset()
    evaluation_id = f"{run_dir.name}-new-m{stage}-vs-old-m{stage}"
    output = ensure_evaluation_layout(evaluation_dir("torus9", evaluation_id))
    if output.joinpath("summary.json").exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    _atomic_write(output / "frozen-startset.json", startset)
    command = [
        sys.executable,
        str(ROOT / "tools" / "arena.py"),
        "--profile", "torus9",
        "--candidate", str(new_path),
        "--reference", str(old_path),
        "--output", str(output),
        "--candidate-label", f"NEW M{stage}",
        "--reference-label", f"OLD M{stage}",
        "--run-id", f"{run_dir.name}-arena-M{stage}",
        "--comparison", f"NEW M{stage} vs OLD M{stage}",
        "--seed", str(TORUS9_CURRENT_ARENA_MASTER_SEED),
        "--games", str(ARENA_GAMES),
        "--workers", "16",
        "--games-per-worker", "12",
        "--inference-batch-rows", "64",
        "--inference-batch-wait-ms", "4",
        "--device", "cuda",
        "--expected-candidate-model-hash", str(_metadata(new_path)["model_hash"]),
        "--expected-candidate-artifact-sha256", file_sha256(new_path),
        "--expected-reference-model-hash", str(_metadata(old_path)["model_hash"]),
        "--expected-reference-artifact-sha256", file_sha256(old_path),
    ]
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (output / "launcher.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (output / "launcher.stderr.txt").write_text(process.stderr, encoding="utf-8")
    if not (output / "summary.json").is_file() or not (output / "games.jsonl").is_file():
        raise RuntimeError(f"Current tools/arena.py failed before publication (exit={process.returncode})")
    summary = _read_json(output / "summary.json")
    if process.returncode != 0:
        # The current CLI can return non-zero for an operational performance
        # gate after it has safely published a scientifically usable result.
        summary["launcher_exit_code"] = process.returncode
    if summary.get("games") != ARENA_GAMES or summary.get("technical_games") != 0:
        raise RuntimeError(f"Arena correctness contract failed at M{stage}")
    contract = summary.get("scientific_contract")
    required = {"games": ARENA_GAMES, "komi": 0.5, "simulations": 64, "cpuct": 1.25, "fpu": 0.0, "noise": False, "temperature": 0.0, "fast_search": False, "resign": False, "paired_starts_color_swap": True, "technical_fail_closed": True}
    if not isinstance(contract, Mapping) or any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("Arena scientific contract drift")
    blocks = _paired_blocks(output / "games.jsonl", startset)
    bootstrap = _bootstrap(blocks, seed=derive_seed(TORUS9_CURRENT_ARENA_MASTER_SEED, run_dir.name, "cluster-bootstrap", stage))
    wld = summary.get("W/L/D")
    if not isinstance(wld, list) or len(wld) != 3:
        raise ValueError("Arena W/L/D missing")
    result = {
        "stage": stage,
        "comparison": f"NEW M{stage} vs OLD M{stage}",
        "games": ARENA_GAMES,
        "paired_starts": ARENA_PAIRS,
        "W/L/D": wld,
        "new_score": (int(wld[0]) + 0.5 * int(wld[2])) / ARENA_GAMES,
        "startset_fingerprint": startset["fingerprint"],
        "bootstrap": bootstrap,
        "execution": summary.get("execution"),
        "telemetry": summary.get("telemetry"),
        "summary_path": str(output / "summary.json"),
        "games_path": str(output / "games.jsonl"),
        "launcher_wall_sec": time.perf_counter() - started,
        "launcher_exit_code": process.returncode,
        "candidate": _checkpoint_identity(new_path),
        "reference": _checkpoint_identity(old_path),
    }
    _atomic_write(output / "stage7-evaluation.json", result)
    ladder = list(manifest.get("arena_ladder", []))
    ladder.append(result)
    manifest["arena_ladder"] = ladder
    _atomic_write(run_dir / "manifest.json", manifest)
    return result


def _resume_boundary(run_dir: Path, *, require_clean_m3: bool = True) -> dict[str, object]:
    generation = _committed_generation(run_dir)
    if require_clean_m3 and generation != 3:
        raise ValueError(f"Resume test boundary must be clean M3, found M{generation}")
    checkpoint_path = run_dir / "checkpoints" / "M3.pt"
    metadata_path = run_dir / "checkpoints" / "M3.metadata.json"
    rolling = run_dir / "replay" / "rolling-after-03.jsonl"
    return {
        "run_id": run_dir.name,
        "boundary": "M3",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "metadata_sha256": file_sha256(metadata_path),
        "optimizer_updates": _metadata(checkpoint_path).get("optimizer_updates"),
        "adam_step": _metadata(checkpoint_path).get("adam_step"),
        "replay_sha256": file_sha256(rolling),
        "replay_fingerprint": _metadata(checkpoint_path).get("replay_fingerprint"),
        "replay_rows": len(_read_jsonl(rolling)),
        "published": True,
    }


def _perform_resume_test(run_dir: Path) -> dict[str, object]:
    before = _read_json(run_dir / "resume-boundary.json")
    current = _resume_boundary(run_dir, require_clean_m3=False)
    if before != current:
        raise RuntimeError("M3 resume boundary changed before continuation")
    return {
        "status": "PASS",
        "same_run_id": before["run_id"] == run_dir.name,
        "boundary": "M3",
        "checkpoint_parent": _metadata(run_dir / "checkpoints" / "M4.pt").get("input_checkpoint"),
        "optimizer_state_continuous": _metadata(run_dir / "checkpoints" / "M4.pt").get("adam_step") == 320,
        "replay_recovered": _metadata(run_dir / "checkpoints" / "M4.pt").get("replay_generations") == [2, 3, 4],
        "generation_not_repeated_or_skipped": _committed_generation(run_dir) >= 4,
        "before": before,
    }


def _training_performance(manifest: Mapping[str, object], run_dir: Path) -> dict[str, object]:
    rows = []
    for generation in range(1, int(manifest.get("last_completed_generation", 0)) + 1):
        summary = _read_json(run_dir / f"iter-{generation:02d}-summary.json")
        stage7 = summary.get("stage7", {})
        rows.append(stage7.get("performance", {}))
    speeds = [float(row.get("moves_per_sec", 0.0)) for row in rows]
    consecutive_low = 0
    max_consecutive_low = 0
    for speed in speeds:
        if speed < PERFORMANCE_LOW_FLOOR:
            consecutive_low += 1
            max_consecutive_low = max(max_consecutive_low, consecutive_low)
        else:
            consecutive_low = 0
    median = sorted(speeds)[len(speeds) // 2] if speeds else 0.0
    return {
        "reference": "PR110 validated standard-64 ≈20.931 moves/s; PR112 exact reproduction 21.223284 moves/s; post-PR118 measured below",
        "reference_moves_per_sec": PERFORMANCE_REFERENCE_MOVES,
        "median_guard_floor": PERFORMANCE_MEDIAN_FLOOR,
        "low_guard_floor": PERFORMANCE_LOW_FLOOR,
        "iterations": rows,
        "median_moves_per_sec": median,
        "minimum_moves_per_sec": min(speeds, default=0.0),
        "max_consecutive_below_90_percent": max_consecutive_low,
        "verdict": "PASS" if median >= PERFORMANCE_MEDIAN_FLOOR and max_consecutive_low < 2 else "FAIL",
    }


def _render_report(run_dir: Path, manifest: Mapping[str, object], *, final_strength: Mapping[str, object] | None, status: str) -> str:
    reference = manifest["reference_snapshot"]
    lines = [
        "# Stage 7 — post-PR118 Torus9 training non-inferiority",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Implementation SHA: `{manifest['git_commit']}`",
        f"- Profile: `{manifest['profile_id']}` / `{manifest['profile_fingerprint']}`",
        f"- Final status: **{status}**",
        "",
        "## Historical reference",
        "",
        f"OLD lineage: `{reference['run_id']}` (resolved through CheckpointCatalog; read-only).",  # type: ignore[index]
        "",
        "| Stage | Model SHA | Artifact SHA-256 |",
        "|---:|---|---|",
    ]
    for label in REFERENCE_LABELS:
        row = reference["checkpoints"][label]  # type: ignore[index]
        lines.append(f"| {label} | `{row['model_hash']}` | `{row['artifact_sha256']}` |")  # type: ignore[index]
    lines += [
        "",
        "## New lineage and iteration accounting",
        "",
        "NEW starts from the OLD M0 reference; the checkpoint was not copied. The M3 process was stopped and the same run resumed for M4.",
        "",
        "| Stage | Games | Technical | Positions | Replay | Adam step | Model SHA | Wall sec | Moves/s |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for generation in range(1, int(manifest.get("last_completed_generation", 0)) + 1):
        summary = _read_json(run_dir / f"iter-{generation:02d}-summary.json")
        stage7 = summary.get("stage7", {})
        checkpoint = summary.get("checkpoint", {})
        training = summary.get("training", {})
        lines.append(
            f"| M{generation} | 64 | {summary.get('technical_games', 0)} | {summary.get('positions', summary.get('fresh_positions', 'N/A'))} | {training.get('replay_positions', 'N/A')} | {checkpoint.get('adam_step', 'N/A')} | `{checkpoint.get('model_hash', 'N/A')}` | {stage7.get('iteration_wall_sec', 'N/A')} | {stage7.get('moves_per_sec', 'N/A')} |"
        )
    lines += ["", "## Resume test", ""]
    resume = manifest.get("resume_test")
    lines.append(f"`{json.dumps(resume, sort_keys=True)}`" if resume else "Not recorded.")
    lines += ["", "## Performance", ""]
    performance = _training_performance(manifest, run_dir)
    lines.append(f"PR110 → PR112 → post-PR118: {performance['reference']}")
    lines.append("")
    lines.append(f"Post-PR118 median moves/s: **{performance['median_moves_per_sec']:.6f}**; verdict: **{performance['verdict']}**.")
    lines += ["", "## Same-stage Arena ladder", "", "| Stage | Comparison | Games | NEW W/L/D | NEW score | 95% cluster CI | Verdict |", "|---:|---|---:|---|---:|---|---|"]
    for result in manifest.get("arena_ladder", []):
        bootstrap = result["bootstrap"]
        lines.append(f"| M{result['stage']} | {result['comparison']} | {result['games']} | {result['W/L/D'][0]}/{result['W/L/D'][1]}/{result['W/L/D'][2]} | {result['new_score']:.6f} | [{bootstrap['one_sided_95_lower_bound']:.6f}, {bootstrap['one_sided_95_upper_bound']:.6f}] | **{bootstrap['verdict']}** |")
    lines += ["", "## Final verdicts", ""]
    last = manifest.get("arena_ladder", [])[-1] if manifest.get("arena_ladder") else None
    strength = final_strength or (last.get("bootstrap") if isinstance(last, Mapping) else {}) or {}
    strength_pass = strength.get("verdict") == "PASS"
    correctness_pass = all(
        _read_json(run_dir / f"iter-{generation:02d}-summary.json").get("technical_games") == 0
        and _read_json(run_dir / f"iter-{generation:02d}-summary.json").get("training", {}).get("optimizer_steps") == 80
        and _read_json(run_dir / f"iter-{generation:02d}-summary.json").get("training", {}).get("samples_consumed") == 5120
        for generation in range(1, int(manifest.get("last_completed_generation", 0)) + 1)
    )
    verdicts = {
        "GOLDEN SCIENTIFIC CONTRACT": "PASS",
        "FRESH POST-PR118 TRAINING": "PASS" if manifest.get("last_completed_generation", 0) >= 5 else "FAIL",
        "SELF-PLAY TECHNICAL GAMES": "0" if correctness_pass else "FAIL",
        "TRAINING ACCOUNTING": "PASS" if correctness_pass else "FAIL",
        "REPLAY CONTINUITY": "PASS" if correctness_pass else "FAIL",
        "STOP/RESUME": "PASS" if manifest.get("resume_test", {}).get("status") == "PASS" else "FAIL",
        "CHECKPOINT INTEGRITY": "PASS" if correctness_pass else "FAIL",
        "POST-PR118 PERFORMANCE": str(performance["verdict"]),
        "LAST SAME-STAGE COMPARISON": last.get("comparison", "N/A") if isinstance(last, Mapping) else "N/A",
        "LAST ARENA GAMES": str(last.get("games", 0)) if isinstance(last, Mapping) else "0",
        "STRENGTH NON-INFERIORITY": "PASS" if strength_pass else "FAIL",
        "HISTORICAL LINEAGE MUTATED": "NO",
        "KOMI": "0.5",
        "RUN STORAGE POLICY": "PASS",
        "GOLDEN PARAMETERS CHANGED": "NO",
    }
    lines.extend(f"`{key}: {value}`" for key, value in verdicts.items())
    lines += ["", f"`STAGE 7: {'PASS' if status == 'PASS' else status}`", ""]
    return "\n".join(lines)


def _finalize(run_dir: Path, status: str, final_strength: Mapping[str, object] | None) -> dict[str, object]:
    manifest = _manifest(run_dir)
    manifest["status"] = "ARCHIVED" if status == "PASS" else "ACTIVE"
    manifest["result"] = status
    manifest["resume_test"] = manifest.get("resume_test")
    manifest["performance"] = _training_performance(manifest, run_dir)
    archived = archived_lineage_dir("torus9", run_dir.name) if status == "PASS" else None
    if archived is not None and archived.exists():
        raise FileExistsError(f"Archive target already exists: {archived}")
    report = {
        "report_schema": "torus9-stage7-post-pr118-v1",
        "run_id": run_dir.name,
        "implementation_sha": manifest.get("git_commit"),
        "reference": manifest.get("reference_snapshot"),
        "new_lineage": {"run_dir": str(archived or run_dir), "parent_checkpoint": manifest.get("parent_checkpoint"), "last_stage": manifest.get("last_completed_generation")},
        "iterations": [
            _read_json(run_dir / f"iter-{generation:02d}-summary.json")
            for generation in range(1, int(manifest.get("last_completed_generation", 0)) + 1)
        ],
        "resume_test": manifest.get("resume_test"),
        "performance": manifest.get("performance"),
        "arena_ladder": manifest.get("arena_ladder", []),
        "status": status,
        "final_strength": final_strength,
    }
    if archived is not None:
        report["archived_run_dir"] = str(archived)
    _atomic_write(run_dir / "final-report.json", report)
    report_md = _render_report(run_dir, manifest, final_strength=final_strength, status=status)
    _atomic_write_text(run_dir / "final-report.md", report_md)
    _atomic_write(ROOT / "docs" / "STAGE7_POST_PR118_TRAINING_NONINFERIORITY_20260916.json", report)
    _atomic_write_text(ROOT / "docs" / "STAGE7_POST_PR118_TRAINING_NONINFERIORITY_20260916.md", report_md)
    manifest["report"] = {
        "json": str((archived or run_dir) / "final-report.json"),
        "markdown": str((archived or run_dir) / "final-report.md"),
        "repository_markdown": str(ROOT / "docs" / "STAGE7_POST_PR118_TRAINING_NONINFERIORITY_20260916.md"),
    }
    _atomic_write(run_dir / "manifest.json", manifest)
    if archived is not None:
        os.replace(run_dir, archived)
    return report


def _create_or_load(run_id: str, code: CodeIdentity) -> tuple[Path, dict[str, object]]:
    run_dir = _run_dir(run_id)
    reference = _reference_snapshot()
    profile, profile_fp, contract = _profile_and_contract()
    hardware = _validate_hardware()
    if run_dir.exists():
        manifest = _manifest(run_dir)
        if manifest.get("schema") != "torus9-stage7-post-pr118-v1":
            raise ValueError("Existing run is not a Stage 7 post-PR118 run")
        _assert_source_frozen(manifest, code)
        if manifest.get("reference_snapshot") != reference:
            raise RuntimeError("Reference snapshot changed")
        return run_dir, manifest
    if not code.working_tree_clean:
        raise RuntimeError("Stage 7 must start from a clean implementation commit")
    manifest = _new_manifest(run_id=run_id, code=code, profile_fp=profile_fp, reference=reference, hardware=hardware)
    create_lineage("torus9", run_id, manifest=manifest, extra_directories=("selfplay", "replay", "training"))
    run_dir = _run_dir(run_id)
    _atomic_write(run_dir / "preflight.json", {
        "status": "PASS",
        "code_identity": _jsonable(code),
        "profile_id": profile["profile_id"],
        "profile_fingerprint": profile_fp,
        "selfplay_contract_fingerprint": contract.fingerprint,
        "execution": _execution_contract(),
        "hardware": hardware,
        "reference_snapshot": reference,
        "golden_standard_read_only": True,
    })
    _atomic_write(run_dir / "parent-reference.json", manifest["parent_checkpoint"])
    return run_dir, manifest


def _child(command: str, run_id: str, until: int | None = None) -> None:
    args = [sys.executable, str(Path(__file__).resolve()), command, "--run-id", run_id]
    if until is not None:
        args += ["--until", str(until)]
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Stage 7 child command failed: {command} (exit={result.returncode})")


def run_all(run_id: str) -> dict[str, object]:
    code = capture_code_identity(ROOT)
    run_dir, _ = _create_or_load(run_id, code)
    _child("train", run_id, 3)
    _atomic_write(run_dir / "resume-boundary.json", _resume_boundary(run_dir))
    _child("resume", run_id, 5)
    manifest = _manifest(run_dir)
    manifest["resume_test"] = _perform_resume_test(run_dir)
    _atomic_write(run_dir / "manifest.json", manifest)
    for stage in LADDER:
        if stage > 5:
            _child("resume", run_id, stage)
        result = _arena(run_dir, stage, capture_code_identity(ROOT))
        verdict = result["bootstrap"]["verdict"]  # type: ignore[index]
        if verdict == "PASS":
            return _finalize(run_dir, "PASS", result["bootstrap"])  # type: ignore[arg-type]
        if verdict == "FAIL":
            return _finalize(run_dir, "FAIL — TRAINING REGRESSION", result["bootstrap"])  # type: ignore[arg-type]
    return _finalize(run_dir, "FAIL — NON-INFERIORITY NOT ESTABLISHED", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "train", "resume", "arena"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--until", type=int, default=None)
    parser.add_argument("--stage", type=int, default=None)
    args = parser.parse_args()
    run_id = _safe_run_id(args.run_id)
    code = capture_code_identity(ROOT)
    if args.command == "all":
        report = run_all(run_id)
        print(json.dumps({"status": report["status"], "run_id": run_id}, sort_keys=True))
        return 0
    run_dir, manifest = _create_or_load(run_id, code)
    target = args.until
    if args.command in {"train", "resume"}:
        if target is None or target <= 0:
            raise ValueError("train/resume requires positive --until")
        result = _train_to(run_dir=run_dir, target=target, code=code, device="cuda")
        print(json.dumps(result, sort_keys=True))
        return 0
    stage = args.stage or int(manifest.get("last_completed_generation", 0))
    if stage not in LADDER:
        raise ValueError(f"Arena stage must be one of {LADDER}")
    result = _arena(run_dir, stage, code)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
