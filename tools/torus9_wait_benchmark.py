#!/usr/bin/env python3
"""Controlled Torus9 9x9 self-play benchmark for inference-batch wait.

This is an evaluation-only front door.  It resolves the canonical M17 through
the checkpoint catalog and calls the production Torus9 self-play adapter; it
does not create a training lineage, replay, or checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphazero.envs.gocube.integration.catalog import CheckpointCatalog
from gocube_golden.provenance import (
    CodeIdentity,
    canonical_json,
    capture_code_identity,
    file_sha256,
)
from gocube_golden.run_storage import RUNS_ROOT, ensure_evaluation_layout, evaluation_dir
from gocube_golden.torus9 import (
    Torus9SelfPlaySearchContract,
    run_torus9_selfplay_games,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_PROFILE_ID,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)
from tools.hardware_telemetry import HardwareTelemetry


SPEC_PATH = ROOT / "configs" / "gocube" / "torus9_wait_benchmark_v1.json"
DEFAULT_RUN_ID = "torus9-wait-benchmark-20260916"
BASE_VARIANT_IDS = ("wait-0.5ms", "wait-1.0ms", "wait-2.0ms")
ZERO_WAIT_ID = "wait-0.0ms"
REPEAT_VARIANT_IDS = ("wait-0.5ms-repeat", "wait-1.0ms-repeat")
BASELINE_VARIANT_ID = "wait-2.0ms"

# This is deliberately a tolerance-based numerical gate, not a tolerance for
# scientific configuration drift.  The values are aligned with the observed
# same-input CUDA batch-vs-single-row FP32 differences and are checked against
# raw model logits before any MCTS semantics are compared.
NUMERICAL_GATE_TOLERANCE = {
    "policy_logits_atol": 1.0e-5,
    "policy_logits_rtol": 1.0e-5,
    "wdl_logits_atol": 1.0e-5,
    "wdl_logits_rtol": 1.0e-5,
}


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _read_spec() -> dict[str, object]:
    value = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Torus9 wait benchmark run-spec must be an object")
    if value.get("schema") != "torus9-controlled-wait-benchmark-v1":
        raise ValueError("Torus9 wait benchmark run-spec schema drift")
    if value.get("topology") != "torus9":
        raise ValueError("Torus9 wait benchmark topology drift")
    variants = value.get("variants")
    if not isinstance(variants, list) or {item.get("id") for item in variants if isinstance(item, Mapping)} != set(BASE_VARIANT_IDS):
        raise ValueError("Torus9 wait benchmark variants drift")
    return value


def _safe_run_id(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("Benchmark run ID must be one safe path component")
    return value


def _variant_map(spec: Mapping[str, object]) -> dict[str, dict[str, object]]:
    variants = spec["variants"]
    assert isinstance(variants, list)
    return {str(item["id"]): dict(item) for item in variants if isinstance(item, Mapping)}


def _shuffled_base_order(spec: Mapping[str, object]) -> list[str]:
    order = list(BASE_VARIANT_IDS)
    random.Random(int(spec["variant_order_seed"])).shuffle(order)
    if order == list(BASE_VARIANT_IDS):
        raise AssertionError("wait benchmark variant order was not shuffled")
    return order


def _shuffled_repeat_order(spec: Mapping[str, object]) -> list[str]:
    order = list(REPEAT_VARIANT_IDS)
    random.Random(int(spec["variant_order_seed"]) + 1).shuffle(order)
    return order


def _catalog_checkpoint(spec: Mapping[str, object]) -> dict[str, object]:
    checkpoint = spec["checkpoint"]
    assert isinstance(checkpoint, Mapping)
    catalog_id = str(checkpoint["catalog_id"])
    expected_model_hash = str(checkpoint["expected_model_sha256"])
    descriptor = CheckpointCatalog(str(RUNS_ROOT)).get(catalog_id)
    if descriptor is None:
        raise FileNotFoundError(f"CheckpointCatalog cannot resolve {catalog_id}")
    path = Path(descriptor.path).resolve()
    if descriptor.profile_id != TORUS9_CURRENT_PROFILE_ID or path.name != "M17.pt":
        raise ValueError(f"Canonical M17 descriptor drift: {descriptor}")
    metadata_path = Path(descriptor.metadata_path or path.with_suffix(".metadata.json")).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("model_hash") != expected_model_hash:
        raise ValueError("Canonical OLD M17 model SHA-256 does not match run-spec")
    artifact_hash = file_sha256(path)
    return {
        "catalog_id": catalog_id,
        "path": str(path),
        "metadata_path": str(metadata_path),
        "model_hash": expected_model_hash,
        "artifact_sha256": artifact_hash,
        "descriptor": descriptor.to_api(),
        "copied": False,
    }


def _profile_contract() -> tuple[dict[str, object], str, Torus9SelfPlaySearchContract]:
    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    settings = profile["self_play"]
    assert isinstance(settings, Mapping)
    contract = Torus9SelfPlaySearchContract(
        contract_id=str(settings["contract_id"]),
        simulations=int(settings["mcts_simulations"]),
        cpuct=float(settings["cpuct"]),
        fpu=float(settings["fpu"]),
        temperature_until_ply=int(settings["temperature_plies"][1]),  # type: ignore[index]
        temperature_after=float(settings["temperature_after"]),
        dirichlet_epsilon=float(settings["dirichlet_epsilon"]),
        dirichlet_alpha=float(settings["dirichlet_alpha"]),
        watchdog=int(settings["watchdog"]),
    )
    contract.validate()
    return profile, profile_fp, contract


def _load_checkpoint(reference: Mapping[str, object], device: str):
    metadata = json.loads(Path(str(reference["metadata_path"])).read_text(encoding="utf-8"))
    model = torus9_model_from_metadata(metadata).to(device)
    torus9_load_checkpoint(
        Path(str(reference["path"])),
        model=model,
        expected={"model_hash": str(reference["model_hash"])},
        device=device,
    )
    model.eval()
    return model


def _game_ids(spec: Mapping[str, object]) -> tuple[str, ...]:
    workload = spec["workload"]
    assert isinstance(workload, Mapping)
    games = int(workload["games"])
    prefix = str(workload["game_id_prefix"])
    return tuple(f"{prefix}-{index:04d}" for index in range(games))


def _normalized_record(record: object) -> dict[str, object]:
    payload = record.to_dict()
    return {
        key: payload[key]
        for key in (
            "game_id",
            "game_seed",
            "start_state",
            "positions",
            "final_action_trace",
            "formal_result",
            "technical_termination",
            "error",
            "nn_evaluations",
        )
    }


def _normalized_digest(record: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(_normalized_record(record)).encode("utf-8")).hexdigest()


def _load_existing_baseline(
    evaluation_id: str,
    *,
    spec: Mapping[str, object],
    reference: Mapping[str, object],
    execution: Mapping[str, object],
) -> dict[str, object]:
    """Read and validate the existing wait=2 result without copying it."""
    baseline_root = evaluation_dir("torus9", _safe_run_id(evaluation_id))
    result_path = baseline_root / "results" / f"{BASELINE_VARIANT_ID}.json"
    manifest_path = baseline_root / "manifest.json"
    run_spec_path = baseline_root / "run-spec.json"
    if not result_path.is_file() or not manifest_path.is_file() or not run_spec_path.is_file():
        raise FileNotFoundError(f"Existing baseline evaluation is incomplete: {baseline_root}")
    baseline_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    if canonical_json(baseline_spec) != canonical_json(spec):
        raise ValueError("Existing wait=2 baseline run-spec does not match the current benchmark spec")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(result, Mapping):
        raise ValueError("Existing wait=2 baseline artifacts must be JSON objects")
    if result.get("id") != BASELINE_VARIANT_ID or float(result.get("wait_ms", -1.0)) != 2.0:
        raise ValueError("Existing wait=2 baseline artifact has the wrong variant")
    if int(result.get("games_requested", -1)) != 64 or int(result.get("completed_games", -1)) != 64 or int(result.get("technical_games", -1)) != 0:
        raise ValueError("Existing wait=2 baseline artifact failed the 64/64 technical=0 gate")
    digests = result.get("normalized_game_digests")
    if not isinstance(digests, Mapping) or len(digests) != 64:
        raise ValueError("Existing wait=2 baseline artifact has incomplete normalized game coverage")
    telemetry = result.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("Existing wait=2 baseline artifact has no inference telemetry")
    if (
        int(telemetry.get("batch_cap", -1)) != int(execution["inference_batch_cap"])
        or float(telemetry.get("inference_batch_wait_ms", -1.0)) != 2.0
        or int(telemetry.get("target_active_contexts", -1)) != int(execution["total_active_contexts"])
        or int(telemetry.get("active_games_per_worker", -1)) != int(execution["active_games_per_worker"])
        or not bool(telemetry.get("shared_memory_transport", False))
        or str(telemetry.get("inference_device")) != "cuda"
    ):
        raise ValueError("Existing wait=2 baseline artifact does not match the frozen execution contract")
    references = manifest.get("checkpoint_references")
    if not isinstance(references, list) or not references:
        raise ValueError("Existing wait=2 baseline manifest has no checkpoint reference")
    baseline_checkpoint = references[0]
    if not isinstance(baseline_checkpoint, Mapping) or (
        baseline_checkpoint.get("catalog_id") != reference.get("catalog_id")
        or baseline_checkpoint.get("model_hash") != reference.get("model_hash")
        or baseline_checkpoint.get("artifact_sha256") != reference.get("artifact_sha256")
        or bool(baseline_checkpoint.get("copied", True))
    ):
        raise ValueError("Existing wait=2 baseline checkpoint reference drift")
    return {
        "evaluation_id": evaluation_id,
        "result_path": str(result_path),
        "relative_result_path": str(result_path.relative_to(ROOT)),
        "manifest": dict(manifest),
        "result": dict(result),
    }


def _compact_result_reference(baseline: Mapping[str, object]) -> dict[str, object]:
    """Represent a reused result in a report without duplicating its artifact."""
    result = baseline["result"]
    assert isinstance(result, Mapping)
    telemetry = result.get("telemetry", {})
    assert isinstance(telemetry, Mapping)
    telemetry_keys = (
        "mean_inference_batch_rows",
        "p50_inference_batch_rows",
        "p95_inference_batch_rows",
        "max_inference_batch_rows",
        "inference_forwards",
        "inference_rows",
        "inference_rows_per_sec",
        "broker_queue_latency_ms",
        "worker_to_broker_latency_ms",
        "process_tree_effective_cpu_cores",
        "batch_cap",
        "inference_batch_wait_ms",
        "target_active_contexts",
        "shared_memory_transport",
        "inference_device",
    )
    return {
        key: result[key]
        for key in (
            "id",
            "wait_ms",
            "games_requested",
            "games_completed",
            "valid_games",
            "completed_games",
            "technical_games",
            "total_moves",
            "wall_time_sec",
            "moves_per_sec",
            "games_per_hour",
            "gpu_inference_duty_cycle_pct_estimate",
            "started_at",
        )
        if key in result
    } | {
        "telemetry": {key: telemetry[key] for key in telemetry_keys if key in telemetry},
        "reused_artifact": True,
        "artifact_reference": baseline["relative_result_path"],
    }


def _hardware_phase(variant_id: str) -> str:
    return "BENCHMARK_" + variant_id.upper().replace("-", "_").replace(".", "_")


def _tensor_difference_stats(
    batched: torch.Tensor,
    single_row: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    if tuple(batched.shape) != tuple(single_row.shape):
        raise ValueError("Numerical probe output shapes differ")
    difference = (batched.detach() - single_row.detach()).abs()
    reference = single_row.detach().abs()
    finite = bool(torch.isfinite(batched).all() and torch.isfinite(single_row).all())
    allowed = float(atol) + float(rtol) * reference
    violations = int((difference > allowed).sum().item()) if finite else -1
    relative = difference / reference.clamp_min(1.0e-12)
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "max_relative": float(relative.max().item()),
        "finite": finite,
        "tolerance_violations": violations,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _numerical_batch_probe(
    model: object,
    *,
    device: str,
    batch_rows: Sequence[object],
) -> dict[str, object]:
    """Measure same-input batched-vs-single-row FP32 variance for real batch sizes."""
    sizes = sorted({int(value) for value in batch_rows})
    if not sizes or sizes[0] <= 0 or sizes[-1] > 64:
        raise RuntimeError(f"Invalid inference batch sizes for numerical probe: {sizes}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(202609160002)
    observations = torch.randn((sizes[-1], 6, 81), generator=generator, dtype=torch.float32)
    target_device = torch.device(device)
    single_policies: list[torch.Tensor] = []
    single_wdls: list[torch.Tensor] = []
    with torch.inference_mode():
        for index in range(sizes[-1]):
            policy_logits, wdl_logits = model(observations[index:index + 1].to(target_device))  # type: ignore[operator]
            single_policies.append(policy_logits)
            single_wdls.append(wdl_logits)
        single_policy = torch.cat(single_policies, dim=0)
        single_wdl = torch.cat(single_wdls, dim=0)
        per_size: dict[str, object] = {}
        for size in sizes:
            policy_logits, wdl_logits = model(observations[:size].to(target_device))  # type: ignore[operator]
            policy_stats = _tensor_difference_stats(
                policy_logits,
                single_policy[:size],
                atol=NUMERICAL_GATE_TOLERANCE["policy_logits_atol"],
                rtol=NUMERICAL_GATE_TOLERANCE["policy_logits_rtol"],
            )
            wdl_stats = _tensor_difference_stats(
                wdl_logits,
                single_wdl[:size],
                atol=NUMERICAL_GATE_TOLERANCE["wdl_logits_atol"],
                rtol=NUMERICAL_GATE_TOLERANCE["wdl_logits_rtol"],
            )
            per_size[str(size)] = {"policy_logits": policy_stats, "wdl_logits": wdl_stats}
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    policy_stats_by_size = [entry["policy_logits"] for entry in per_size.values()]
    wdl_stats_by_size = [entry["wdl_logits"] for entry in per_size.values()]
    policy_failed = [entry for entry in policy_stats_by_size if not entry["finite"] or int(entry["tolerance_violations"]) > 0]
    wdl_failed = [entry for entry in wdl_stats_by_size if not entry["finite"] or int(entry["tolerance_violations"]) > 0]
    return {
        "status": "PASS" if not policy_failed and not wdl_failed else "FAIL",
        "probe": "same-input batched forward vs concatenated single-row forwards",
        "probe_seed": 202609160002,
        "batch_rows": sizes,
        "tolerance": dict(NUMERICAL_GATE_TOLERANCE),
        "policy_logits": {
            "max_abs": max(float(entry["max_abs"]) for entry in policy_stats_by_size),
            "max_relative": max(float(entry["max_relative"]) for entry in policy_stats_by_size),
            "mean_abs_max": max(float(entry["mean_abs"]) for entry in policy_stats_by_size),
            "tolerance_violations": sum(int(entry["tolerance_violations"]) for entry in policy_stats_by_size),
            "finite": not policy_failed,
        },
        "wdl_logits": {
            "max_abs": max(float(entry["max_abs"]) for entry in wdl_stats_by_size),
            "max_relative": max(float(entry["max_relative"]) for entry in wdl_stats_by_size),
            "mean_abs_max": max(float(entry["mean_abs"]) for entry in wdl_stats_by_size),
            "tolerance_violations": sum(int(entry["tolerance_violations"]) for entry in wdl_stats_by_size),
            "finite": not wdl_failed,
        },
        "by_batch_rows": per_size,
    }


def _run_variant(
    *,
    model: object,
    profile_fp: str,
    contract: Torus9SelfPlaySearchContract,
    reference: Mapping[str, object],
    spec: Mapping[str, object],
    run_id: str,
    selfplay_run_id: str,
    variant_id: str,
    wait_ms: float,
    code: CodeIdentity,
    hardware: HardwareTelemetry,
) -> dict[str, object]:
    execution = spec["execution"]
    workload = spec["workload"]
    assert isinstance(execution, Mapping) and isinstance(workload, Mapping)
    telemetry: dict[str, object] = {}
    started = time.perf_counter()
    hardware.set_phase(_hardware_phase(variant_id))
    records = run_torus9_selfplay_games(
        model,  # type: ignore[arg-type]
        run_id=selfplay_run_id,
        label="M17",
        artifact=str(reference["artifact_sha256"]),
        master_seed=profile_seed(spec),
        profile_fp=profile_fp,
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        game_ids=_game_ids(spec),
        workers=int(execution["workers"]),
        code_identity=code,
        device=str(execution["device"]),
        contract=contract,
        coalescing=bool(execution["coalescing"]),
        inference_batch_cap=int(execution["inference_batch_cap"]),
        inference_batch_wait_ms=float(wait_ms),
        active_games_per_worker=int(execution["active_games_per_worker"]),
        total_active_contexts=int(execution["total_active_contexts"]),
        inference_telemetry=telemetry,
        execution_activity=telemetry,
        execution_reference_interactive=False,
    )
    measured_wall = float(telemetry.get("wall_time_sec", time.perf_counter() - started))
    expected_ids = set(_game_ids(spec))
    if len(records) != len(expected_ids) or {record.game_id for record in records} != expected_ids:
        raise RuntimeError(f"{variant_id} returned an unexpected game set")
    master_seed = profile_seed(spec)
    for record in records:
        record.validate()
        expected_seed = _canonical_game_seed(master_seed, selfplay_run_id, record.game_id)
        if record.game_seed != expected_seed:
            raise RuntimeError(f"{variant_id} game seed drift for {record.game_id}")
    moves = sum(len(record.final_action_trace) for record in records)
    technical = sum(record.technical_termination is not None for record in records)
    if len(records) != 64 or technical != 0:
        raise RuntimeError(
            f"{variant_id} failed correctness gate: completed={len(records)}/64, technical={technical}"
        )
    if int(telemetry.get("games_completed", -1)) != 64 or int(telemetry.get("technical_games", -1)) != 0:
        raise RuntimeError(f"{variant_id} self-play telemetry correctness gate failed")
    if int(telemetry.get("target_active_contexts", -1)) != 64:
        raise RuntimeError(f"{variant_id} did not configure 64 active contexts")
    if int(telemetry.get("inference_batch_cap", -1)) != 64 or float(telemetry.get("inference_batch_wait_ms", -1.0)) != float(wait_ms):
        raise RuntimeError(f"{variant_id} inference wait/cap telemetry drift")
    if not bool(telemetry.get("shared_memory_transport", False)) or str(telemetry.get("inference_device")) != "cuda":
        raise RuntimeError(f"{variant_id} did not use shared-memory CUDA inference")
    if int(telemetry.get("inference_rows", -1)) != sum(record.nn_evaluations for record in records):
        raise RuntimeError(f"{variant_id} inference rows were lost or duplicated")
    batch_rows = telemetry.get("batch_rows")
    if not isinstance(batch_rows, list) or not batch_rows:
        raise RuntimeError(f"{variant_id} did not report inference batch rows")
    numerical_divergence = _numerical_batch_probe(
        model,
        device=str(execution["device"]),
        batch_rows=batch_rows,
    )
    if numerical_divergence["status"] != "PASS":
        raise RuntimeError(f"{variant_id} exceeded the FP32 batching numerical-variance gate")
    forward_count = int(telemetry.get("inference_forwards", 0))
    forward_summary = telemetry.get("model_forward_latency_ms")
    forward_ms = float(forward_summary.get("mean", 0.0)) if isinstance(forward_summary, Mapping) else 0.0
    hardware.set_phase(_hardware_phase(variant_id))
    return {
        "id": variant_id,
        "wait_ms": float(wait_ms),
        "selfplay_run_id": selfplay_run_id,
        "games_requested": len(expected_ids),
        "games_completed": len(records),
        "valid_games": len(records) - technical,
        "completed_games": len(records),
        "technical_games": int(technical),
        "total_moves": moves,
        "wall_time_sec": measured_wall,
        "moves_per_sec": moves / measured_wall if measured_wall else 0.0,
        "games_per_hour": len(records) * 3600.0 / measured_wall if measured_wall else 0.0,
        "normalized_game_digests": {record.game_id: _normalized_digest(record) for record in records},
        "numerical_divergence": numerical_divergence,
        "telemetry": telemetry,
        "gpu_inference_duty_cycle_pct_estimate": (
            forward_ms * forward_count / (1000.0 * measured_wall) * 100.0 if measured_wall else 0.0
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _canonical_game_seed(master_seed: int, run_id: str, game_id: str) -> int:
    from gocube_golden.provenance import derive_seed

    return derive_seed(master_seed, run_id, game_id, "game")


def profile_seed(spec: Mapping[str, object]) -> int:
    profile = load_torus9_current_profile()
    workload = spec["workload"]
    assert isinstance(workload, Mapping)
    source = str(workload["master_seed_source"])
    if source != "current_profile.seeds.selfplay_master_seed":
        raise ValueError("Unsupported benchmark master-seed source")
    seeds = profile["seeds"]
    assert isinstance(seeds, Mapping)
    return int(seeds["selfplay_master_seed"])


def _compare_digest_advisory(
    results: Mapping[str, Mapping[str, object]],
    *,
    baseline_id: str,
    baseline_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    baseline_source = baseline_result if baseline_result is not None else results[baseline_id]
    baseline = baseline_source["normalized_game_digests"]
    assert isinstance(baseline, Mapping)
    comparisons: dict[str, object] = {}
    any_difference = False
    for variant_id, result in results.items():
        digests = result["normalized_game_digests"]
        assert isinstance(digests, Mapping)
        differing = sorted(game_id for game_id in baseline if digests.get(game_id) != baseline[game_id])
        same = not differing and set(digests) == set(baseline)
        comparisons[variant_id] = {
            "baseline": baseline_id,
            "bit_exact_match": same,
            "differing_game_ids": differing,
        }
        any_difference = any_difference or not same
    return {
        "status": "BIT_EXACT" if not any_difference else "EXPECTED_VARIANCE_ALLOWED",
        "bit_exact_equality_required": False,
        "baseline": baseline_id,
        "comparisons": comparisons,
    }


def _correctness_gate(
    results: Mapping[str, Mapping[str, object]],
    *,
    baseline_id: str,
    baseline_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Gate scientific invariants while allowing bounded batch-shape FP32 variance."""
    all_rows: dict[str, Mapping[str, object]] = {}
    if baseline_result is not None:
        all_rows[baseline_id] = baseline_result
    all_rows.update(results)
    scientific_checks: dict[str, object] = {}
    scientific_pass = True
    numerical_checks: dict[str, object] = {}
    numerical_pass = True
    for variant_id, result in all_rows.items():
        completed = int(result.get("completed_games", -1))
        requested = int(result.get("games_requested", -1))
        technical = int(result.get("technical_games", -1))
        row_pass = completed == 64 and requested == 64 and technical == 0
        scientific_checks[variant_id] = {
            "completed_games": completed,
            "games_requested": requested,
            "technical_games": technical,
            "status": "PASS" if row_pass else "FAIL",
        }
        scientific_pass = scientific_pass and row_pass
        if variant_id == baseline_id and baseline_result is not None:
            continue
        numerical = result.get("numerical_divergence")
        numerical_row_pass = isinstance(numerical, Mapping) and numerical.get("status") == "PASS"
        numerical_checks[variant_id] = numerical if isinstance(numerical, Mapping) else {"status": "MISSING"}
        numerical_pass = numerical_pass and numerical_row_pass
    digest_advisory = _compare_digest_advisory(
        results,
        baseline_id=baseline_id,
        baseline_result=baseline_result,
    ) if results else {
        "status": "NO_COMPARISON",
        "bit_exact_equality_required": False,
        "baseline": baseline_id,
        "comparisons": {},
    }
    status = "PASS" if scientific_pass and numerical_pass else "FAIL"
    return {
        "status": status,
        "bit_exact_equality_required": False,
        "scientific_semantics": {"status": "PASS" if scientific_pass else "FAIL", "variants": scientific_checks},
        "numerical_gate": {
            "status": "PASS" if numerical_pass else "FAIL",
            "explanation": "bounded same-input CUDA batched-vs-single-row FP32 variance is allowed; larger/non-finite divergence fails",
            "tolerance": dict(NUMERICAL_GATE_TOLERANCE),
            "variants": numerical_checks,
        },
        "digest_comparison_advisory": digest_advisory,
    }


def _relative_improvement(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> float:
    base = float(baseline["moves_per_sec"])
    return (float(candidate["moves_per_sec"]) / base - 1.0) * 100.0 if base else 0.0


def _decision(results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    base = results["wait-1.0ms"]
    fast = results["wait-0.5ms"]
    first_improvement = _relative_improvement(fast, base)
    extra_wait_zero = "wait-0.0ms" in results
    if 3.0 <= first_improvement < 5.0 and "wait-0.5ms-repeat" in results and "wait-1.0ms-repeat" in results:
        fast_values = [float(results[key]["moves_per_sec"]) for key in ("wait-0.5ms", "wait-0.5ms-repeat")]
        base_values = [float(results[key]["moves_per_sec"]) for key in ("wait-1.0ms", "wait-1.0ms-repeat")]
        aggregate_improvement = (statistics.median(fast_values) / statistics.median(base_values) - 1.0) * 100.0
    else:
        aggregate_improvement = first_improvement
    main_ids = ["wait-0.5ms", "wait-1.0ms", "wait-2.0ms"]
    if extra_wait_zero:
        main_ids.append("wait-0.0ms")
    winner_id = max(main_ids, key=lambda key: (float(results[key]["moves_per_sec"]), -float(results[key]["wall_time_sec"])))
    winner_improvement = _relative_improvement(results[winner_id], base)
    if winner_id == "wait-1.0ms" or winner_improvement < 5.0:
        recommendation = "NO MATERIAL IMPROVEMENT — KEEP 1 ms"
    else:
        recommendation = f"RECOMMEND GOLDEN WAIT = {float(results[winner_id]['wait_ms']):g} ms"
    if first_improvement < 3.0:
        zero_reason = "not run: first 0.5 ms vs 1.0 ms improvement was below 3%"
    elif first_improvement < 5.0:
        zero_reason = "not run: 0.5 ms was in the 3–5% repeat band; zero wait is not eligible from first-run evidence"
    else:
        zero_reason = "run: first 0.5 ms vs 1.0 ms improvement was at least 5%"
    return {
        "first_0_5_vs_1_0_improvement_pct": first_improvement,
        "aggregate_0_5_vs_1_0_improvement_pct": aggregate_improvement,
        "winner_id": winner_id,
        "winner_improvement_vs_1_0_pct": winner_improvement,
        "recommendation": recommendation,
        "wait_0_ran": extra_wait_zero,
        "wait_0_reason": zero_reason,
    }


def _hardware_variant_summary(summary: Mapping[str, object], variant_id: str) -> dict[str, object]:
    phase = summary.get("phases", {})
    phase_data = phase.get(_hardware_phase(variant_id), {}) if isinstance(phase, Mapping) else {}
    return dict(phase_data) if isinstance(phase_data, Mapping) else {}


def _render_report(report: Mapping[str, object]) -> str:
    results = report["results"]
    assert isinstance(results, Mapping)
    hardware = report.get("hardware_telemetry", {})
    lines = [
        "# Torus9 9×9 controlled `inference_batch_wait_ms` benchmark",
        "",
        f"Run: `{report['run_id']}`; commit `{report['commit']}`.",
        "",
        "The benchmark used the production Torus9 self-play front door and stopped after self-play. No training, replay build, optimizer step, or checkpoint creation was performed.",
        "",
        "## Results",
        "",
        "| wait | wall | moves | moves/s | games/h | mean batch | p95 batch | CPU | GPU | technical |",
        "| ---: | ---: | ----: | ------: | ------: | ---------: | --------: | --: | --: | --------: |",
    ]
    for variant_id, row in results.items():
        telemetry = row["telemetry"]
        assert isinstance(telemetry, Mapping)
        hw = _hardware_variant_summary(hardware if isinstance(hardware, Mapping) else {}, variant_id)
        gpu = hw.get("gpu_util_percent", {})
        gpu_mean = gpu.get("mean") if isinstance(gpu, Mapping) else None
        cpu = telemetry.get("process_tree_effective_cpu_cores", "N/A")
        technical = f"{row['technical_games']} ({row['completed_games']}/{row['games_requested']})"
        lines.append(
            f"| {float(row['wait_ms']):g} ms | {float(row['wall_time_sec']):.3f}s | {row['total_moves']} | {float(row['moves_per_sec']):.3f} | {float(row['games_per_hour']):.1f} | {float(telemetry.get('mean_inference_batch_rows', 0.0)):.3f} | {float(telemetry.get('p95_inference_batch_rows', 0.0)):.3f} | {float(cpu):.3f} | {gpu_mean if gpu_mean is not None else 'N/A'} | {technical} |"
        )
    lines.extend([
        "",
        "Batch columns above are mean/p95; p50, max, inference calls, rows/s, and wait telemetry are in the JSON summaries.",
        "",
        "## Decision",
        "",
        f"{report['decision']['recommendation']}",
        "",
        f"Wait 0 ms: {report['decision']['wait_0_reason']}.",
        "",
        "## Correctness and provenance",
        "",
        f"Correctness gate: **{report['correctness_gate']['status']}**; bit-exact action/result equality is advisory only, while scientific invariants and bounded FP32 batch variance are gated.",
        f"Digest comparison advisory: **{report['correctness_gate']['digest_comparison_advisory']['status']}** against `{report['correctness_gate']['digest_comparison_advisory']['baseline']}`.",
        f"Checkpoint catalog reference: `{report['checkpoint']['catalog_id']}`.",
        f"Checkpoint path: `{report['checkpoint']['path']}` (reference only; copied: `{report['checkpoint']['copied']}`).",
        f"Model SHA-256: `{report['checkpoint']['model_hash']}`; artifact SHA-256: `{report['checkpoint']['artifact_sha256']}`.",
        f"Master seed: `{report['master_seed']}`; game seed: `derive_seed(master_seed, run_id, game_id, \"game\")`.",
        f"Game IDs: `{report['game_ids']['count']}` fixed IDs from `{report['game_ids']['first']}` through `{report['game_ids']['last']}`.",
        f"Actual new-run variant order: `{', '.join(report['variant_order'])}`.",
        "",
        "Frozen execution: 16 workers; 4 active games/worker; 64 active contexts; cap 64; central parent-owned CUDA inference; shared memory ON; coalescing ON; device CUDA. Only `inference_batch_wait_ms` varied.",
        "",
        "Frozen scientific contract was loaded from the current Torus9 profile: Torus 9×9, komi 0.5, 64 games, 64 simulations, cpuct 1.25, FPU 0, root noise ON, Dirichlet ε 0.25/α 0.11, temperature 1.0 on plies 1–8 then 0, fast search OFF, resign OFF, watchdog 500, current GoldenGraphNetV2-Torus9 80×8 M17.",
        "",
    ])
    return "\n".join(lines)


def _render_failure_report(report: Mapping[str, object]) -> str:
    results = report.get("results", {})
    lines = [
        "# Torus9 9×9 controlled `inference_batch_wait_ms` benchmark — STOPPED",
        "",
        f"Run: `{report['run_id']}`; commit `{report['commit']}`.",
        "",
        f"**STOP — {report['failure']}**",
        "",
        "No performance winner or Golden preset recommendation is valid because the correctness/parity gate failed.",
        "",
        "## Completed before STOP",
        "",
        "| wait | wall | moves | moves/s | games/h | mean batch | p95 batch | CPU | GPU | technical |",
        "| ---: | ---: | ----: | ------: | ------: | ---------: | --------: | --: | --: | --------: |",
    ]
    if isinstance(results, Mapping):
        hardware = report.get("hardware_telemetry", {})
        for variant_id, row in results.items():
            if not isinstance(row, Mapping):
                continue
            telemetry = row.get("telemetry", {})
            if not isinstance(telemetry, Mapping):
                continue
            hw = _hardware_variant_summary(hardware if isinstance(hardware, Mapping) else {}, str(variant_id))
            gpu = hw.get("gpu_util_percent", {})
            gpu_mean = gpu.get("mean") if isinstance(gpu, Mapping) else None
            lines.append(
                f"| {float(row['wait_ms']):g} ms | {float(row['wall_time_sec']):.3f}s | {row['total_moves']} | {float(row['moves_per_sec']):.3f} | {float(row['games_per_hour']):.1f} | {float(telemetry.get('mean_inference_batch_rows', 0.0)):.3f} | {float(telemetry.get('p95_inference_batch_rows', 0.0)):.3f} | {float(telemetry.get('process_tree_effective_cpu_cores', 0.0)):.3f} | {gpu_mean if gpu_mean is not None else 'N/A'} | {row['technical_games']} ({row['completed_games']}/{row['games_requested']}) |"
            )
    lines.extend([
        "",
        f"Variant order reached before STOP: `{', '.join(report.get('variant_order', []))}`.",
        f"Correctness-gate artifact: `{report['correctness_gate']}`.",
        "",
        "The remaining variants were intentionally not run after the first normalized-result divergence.",
        "",
    ])
    return "\n".join(lines)


def _combined_results(
    results: Mapping[str, Mapping[str, object]],
    baseline: Mapping[str, object] | None,
) -> dict[str, Mapping[str, object]]:
    combined: dict[str, Mapping[str, object]] = {}
    if baseline is not None:
        baseline_result = baseline["result"]
        assert isinstance(baseline_result, Mapping)
        combined[BASELINE_VARIANT_ID] = baseline_result
    combined.update(results)
    return combined


def _report_results(
    results: Mapping[str, Mapping[str, object]],
    baseline: Mapping[str, object] | None,
) -> dict[str, Mapping[str, object]]:
    report_results: dict[str, Mapping[str, object]] = {}
    if baseline is not None:
        report_results[BASELINE_VARIANT_ID] = _compact_result_reference(baseline)
    report_results.update(results)
    return report_results


def run_benchmark(
    run_id: str = DEFAULT_RUN_ID,
    *,
    seed_run_id: str | None = None,
    baseline_evaluation_id: str | None = None,
) -> dict[str, object]:
    spec = _read_spec()
    run_id = _safe_run_id(run_id)
    if baseline_evaluation_id is not None:
        baseline_evaluation_id = _safe_run_id(baseline_evaluation_id)
        seed_run_id = _safe_run_id(seed_run_id or baseline_evaluation_id)
        if seed_run_id != baseline_evaluation_id:
            raise ValueError("Continuation must reuse the existing baseline evaluation ID for game seeds")
    else:
        seed_run_id = _safe_run_id(seed_run_id or run_id)
    root = evaluation_dir("torus9", run_id)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing benchmark evaluation: {root}")
    ensure_evaluation_layout(root)
    code = capture_code_identity(ROOT)
    code.validate(require_canonical=True)
    profile, profile_fp, contract = _profile_contract()
    workload = spec["workload"]
    assert isinstance(workload, Mapping)
    if int(workload["games"]) != 64:
        raise ValueError("Torus9 wait benchmark workload must contain exactly 64 games")
    reference = _catalog_checkpoint(spec)
    execution = spec["execution"]
    assert isinstance(execution, Mapping)
    if not torch.cuda.is_available() or str(execution["device"]) != "cuda":
        raise RuntimeError("Torus9 wait benchmark requires CUDA")
    if (
        int(execution["workers"]) != 16
        or int(execution["active_games_per_worker"]) != 4
        or int(execution["total_active_contexts"]) != 64
        or int(execution["inference_batch_cap"]) != 64
        or not bool(execution["coalescing"])
        or execution["central_model_owner"] != "parent"
        or not bool(execution["shared_memory"])
        or execution["device"] != "cuda"
    ):
        raise ValueError("Torus9 wait benchmark execution contract drift")
    baseline = (
        _load_existing_baseline(
            baseline_evaluation_id,
            spec=spec,
            reference=reference,
            execution=execution,
        )
        if baseline_evaluation_id is not None
        else None
    )
    master_seed = profile_seed(spec)
    game_ids = _game_ids(spec)
    manifest = {
        "schema": "torus9-controlled-wait-benchmark-manifest-v1",
        "evaluation_id": run_id,
        "topology": "torus9",
        "status": "ACTIVE",
        "benchmark_status": "RUNNING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "config_fingerprint": "sha256:" + hashlib.sha256(canonical_json({"profile": profile_fp, "run_spec": spec}).encode("utf-8")).hexdigest(),
        "checkpoint_references": [reference],
        "seed_run_id": seed_run_id,
        "baseline_evaluation": (
            {
                "evaluation_id": baseline["evaluation_id"],
                "artifact_reference": baseline["relative_result_path"],
                "reused": True,
                "copied": False,
            }
            if baseline is not None
            else None
        ),
        "storage": {"canonical_path": str(root.relative_to(ROOT)), "contains_checkpoint_copies": False},
    }
    _atomic_write(root / "manifest.json", manifest)
    _atomic_write(root / "run-spec.json", spec)
    model = _load_checkpoint(reference, str(execution["device"]))
    telemetry_path = root / "logs" / "hardware-telemetry.jsonl"
    hardware = HardwareTelemetry(telemetry_path, interval_s=1.0)
    results: dict[str, dict[str, object]] = {}
    order = _shuffled_base_order(spec)
    if baseline is not None:
        order = [variant_id for variant_id in order if variant_id != BASELINE_VARIANT_ID]
        if set(order) != {"wait-0.5ms", "wait-1.0ms"}:
            raise ValueError("Continuation must run exactly wait=0.5 ms and wait=1.0 ms")
    executed_order: list[str] = []
    correctness_gate: dict[str, object] = {"status": "PENDING"}

    def update_correctness_gate() -> dict[str, object]:
        nonlocal correctness_gate
        comparison_baseline_id = BASELINE_VARIANT_ID if baseline is not None else executed_order[0]
        correctness_gate = _correctness_gate(
            results,
            baseline_id=comparison_baseline_id,
            baseline_result=(baseline["result"] if baseline is not None else None),
        )
        _atomic_write(root / "metrics" / "correctness-gate.json", correctness_gate)
        if correctness_gate["status"] != "PASS":
            raise RuntimeError("Torus9 wait benchmark correctness gate failed")
        return correctness_gate

    try:
        hardware.set_phase(_hardware_phase(order[0]))
        hardware.start()
        for variant_id in order:
            wait_ms = float(_variant_map(spec)[variant_id]["wait_ms"])
            result = _run_variant(model=model, profile_fp=profile_fp, contract=contract, reference=reference, spec=spec, run_id=run_id, selfplay_run_id=seed_run_id, variant_id=variant_id, wait_ms=wait_ms, code=code, hardware=hardware)
            results[variant_id] = result
            executed_order.append(variant_id)
            _atomic_write(root / "results" / f"{variant_id}.json", result)
            update_correctness_gate()

        decision_results = _combined_results(results, baseline)
        first_improvement = _relative_improvement(decision_results["wait-0.5ms"], decision_results["wait-1.0ms"])
        if 3.0 <= first_improvement < 5.0:
            for variant_id in _shuffled_repeat_order(spec):
                base_id = variant_id.removesuffix("-repeat")
                wait_ms = float(_variant_map(spec)[base_id]["wait_ms"])
                result = _run_variant(model=model, profile_fp=profile_fp, contract=contract, reference=reference, spec=spec, run_id=run_id, selfplay_run_id=seed_run_id, variant_id=variant_id, wait_ms=wait_ms, code=code, hardware=hardware)
                results[variant_id] = result
                executed_order.append(variant_id)
                _atomic_write(root / "results" / f"{variant_id}.json", result)
                update_correctness_gate()
        elif first_improvement >= 5.0:
            variant_id = ZERO_WAIT_ID
            result = _run_variant(model=model, profile_fp=profile_fp, contract=contract, reference=reference, spec=spec, run_id=run_id, selfplay_run_id=seed_run_id, variant_id=variant_id, wait_ms=0.0, code=code, hardware=hardware)
            results[variant_id] = result
            executed_order.append(variant_id)
            _atomic_write(root / "results" / f"{variant_id}.json", result)
            update_correctness_gate()
        hardware.stop()
        hardware_summary = hardware.summary()
        correctness_gate = update_correctness_gate()
        decision = _decision(_combined_results(results, baseline))
        report = {
            "schema": "torus9-controlled-wait-benchmark-report-v1",
            "run_id": run_id,
            "status": "COMPLETE",
            "benchmark_status": "COMPLETE",
            "commit": code.git_commit_sha,
            "git_tree": code.git_tree_sha,
            "profile_id": TORUS9_CURRENT_PROFILE_ID,
            "profile_fingerprint": profile_fp,
            "checkpoint": reference,
            "master_seed": master_seed,
            "seed_run_id": seed_run_id,
            "game_ids": {"count": len(game_ids), "first": game_ids[0], "last": game_ids[-1], "all": list(game_ids)},
            "variant_order": executed_order,
            "comparison_variant_order": ([BASELINE_VARIANT_ID] if baseline is not None else []) + executed_order,
            "baseline_artifacts": (
                [{"evaluation_id": baseline["evaluation_id"], "artifact_reference": baseline["relative_result_path"], "reused": True, "copied": False}]
                if baseline is not None
                else []
            ),
            "frozen_execution": dict(execution),
            "scientific_contract": dict(profile["self_play"]),
            "checkpoint_reference_only": True,
            "correctness_gate": correctness_gate,
            "decision": decision,
            "hardware_telemetry": hardware_summary,
            "results": _report_results(results, baseline),
            "storage": {"canonical_path": str(root.relative_to(ROOT)), "contains_checkpoint_copies": False},
        }
        _atomic_write(root / "report.json", report)
        _atomic_write_text(root / "report.md", _render_report(report))
        manifest.update({"status": "ARCHIVED", "benchmark_status": "COMPLETE", "completed_at": datetime.now(timezone.utc).isoformat(), "recommendation": decision["recommendation"], "semantic_parity": correctness_gate["status"], "correctness_gate_status": correctness_gate["status"]})
        _atomic_write(root / "manifest.json", manifest)
        return report
    except BaseException as exc:
        hardware.stop()
        failure_results = _report_results(results, baseline)
        failure_report = {
            "schema": "torus9-controlled-wait-benchmark-report-v1",
            "run_id": run_id,
            "status": "STOPPED",
            "benchmark_status": "FAILED",
            "failure": f"{type(exc).__name__}: {exc}",
            "commit": code.git_commit_sha,
            "git_tree": code.git_tree_sha,
            "profile_id": TORUS9_CURRENT_PROFILE_ID,
            "profile_fingerprint": profile_fp,
            "checkpoint": reference,
            "master_seed": master_seed,
            "seed_run_id": seed_run_id,
            "game_ids": {"count": len(game_ids), "first": game_ids[0], "last": game_ids[-1], "all": list(game_ids)},
            "variant_order": executed_order,
            "comparison_variant_order": ([BASELINE_VARIANT_ID] if baseline is not None else []) + executed_order,
            "baseline_artifacts": (
                [{"evaluation_id": baseline["evaluation_id"], "artifact_reference": baseline["relative_result_path"], "reused": True, "copied": False}]
                if baseline is not None
                else []
            ),
            "frozen_execution": dict(execution),
            "scientific_contract": dict(profile["self_play"]),
            "checkpoint_reference_only": True,
            "correctness_gate": correctness_gate,
            "hardware_telemetry": hardware.summary(),
            "results": failure_results,
            "storage": {"canonical_path": str(root.relative_to(ROOT)), "contains_checkpoint_copies": False},
        }
        _atomic_write(root / "report.json", failure_report)
        _atomic_write_text(root / "report.md", _render_failure_report(failure_report))
        manifest.update({"status": "ARCHIVED", "benchmark_status": "FAILED", "failure": f"{type(exc).__name__}: {exc}", "variant_order": executed_order, "semantic_parity": correctness_gate.get("status", "UNKNOWN"), "correctness_gate_status": correctness_gate.get("status", "UNKNOWN")})
        _atomic_write(root / "manifest.json", manifest)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--seed-run-id", default=None, help="reuse this self-play run ID for scheduling-independent game seeds")
    parser.add_argument("--baseline-evaluation-id", default=None, help="reuse an existing wait=2 evaluation artifact without copying or rerunning it")
    args = parser.parse_args(argv)
    report = run_benchmark(args.run_id, seed_run_id=args.seed_run_id, baseline_evaluation_id=args.baseline_evaluation_id)
    print(json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
