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


def _hardware_phase(variant_id: str) -> str:
    return "BENCHMARK_" + variant_id.upper().replace("-", "_").replace(".", "_")


def _run_variant(
    *,
    model: object,
    profile_fp: str,
    contract: Torus9SelfPlaySearchContract,
    reference: Mapping[str, object],
    spec: Mapping[str, object],
    run_id: str,
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
        run_id=run_id,
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
        expected_seed = _canonical_game_seed(master_seed, run_id, record.game_id)
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
    forward_count = int(telemetry.get("inference_forwards", 0))
    forward_summary = telemetry.get("model_forward_latency_ms")
    forward_ms = float(forward_summary.get("mean", 0.0)) if isinstance(forward_summary, Mapping) else 0.0
    hardware.set_phase(_hardware_phase(variant_id))
    return {
        "id": variant_id,
        "wait_ms": float(wait_ms),
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


def _compare_parity(
    results: Mapping[str, Mapping[str, object]],
    *,
    baseline_id: str,
) -> dict[str, object]:
    baseline = results[baseline_id]["normalized_game_digests"]
    assert isinstance(baseline, Mapping)
    comparisons: dict[str, object] = {}
    parity = True
    for variant_id, result in results.items():
        digests = result["normalized_game_digests"]
        assert isinstance(digests, Mapping)
        differing = sorted(game_id for game_id in baseline if digests.get(game_id) != baseline[game_id])
        same = not differing and set(digests) == set(baseline)
        comparisons[variant_id] = {
            "baseline": baseline_id,
            "same_normalized_results": same,
            "differing_game_ids": differing,
        }
        parity = parity and same
    return {"status": "PASS" if parity else "FAIL", "baseline": baseline_id, "comparisons": comparisons}


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
        f"Semantic parity: **{report['parity']['status']}** against `{report['parity']['baseline']}`; normalized per-game result digests were compared.",
        f"Checkpoint catalog reference: `{report['checkpoint']['catalog_id']}`.",
        f"Checkpoint path: `{report['checkpoint']['path']}` (reference only; copied: `{report['checkpoint']['copied']}`).",
        f"Model SHA-256: `{report['checkpoint']['model_hash']}`; artifact SHA-256: `{report['checkpoint']['artifact_sha256']}`.",
        f"Master seed: `{report['master_seed']}`; game seed: `derive_seed(master_seed, run_id, game_id, \"game\")`.",
        f"Game IDs: `{report['game_ids']['count']}` fixed IDs from `{report['game_ids']['first']}` through `{report['game_ids']['last']}`.",
        f"Actual variant order: `{', '.join(report['variant_order'])}`.",
        "",
        "Frozen execution: 16 workers; 4 active games/worker; 64 active contexts; cap 64; central parent-owned CUDA inference; shared memory ON; coalescing ON; device CUDA. Only `inference_batch_wait_ms` varied.",
        "",
        "Frozen scientific contract was loaded from the current Torus9 profile: Torus 9×9, komi 0.5, 64 games, 64 simulations, cpuct 1.25, FPU 0, root noise ON, Dirichlet ε 0.25/α 0.11, temperature 1.0 on plies 1–8 then 0, fast search OFF, resign OFF, watchdog 500, current GoldenGraphNetV2-Torus9 80×8 M17.",
        "",
    ])
    return "\n".join(lines)


def run_benchmark(run_id: str = DEFAULT_RUN_ID) -> dict[str, object]:
    spec = _read_spec()
    run_id = _safe_run_id(run_id)
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
        "storage": {"canonical_path": str(root.relative_to(ROOT)), "contains_checkpoint_copies": False},
    }
    _atomic_write(root / "manifest.json", manifest)
    _atomic_write(root / "run-spec.json", spec)
    model = _load_checkpoint(reference, str(execution["device"]))
    telemetry_path = root / "logs" / "hardware-telemetry.jsonl"
    hardware = HardwareTelemetry(telemetry_path, interval_s=1.0)
    results: dict[str, dict[str, object]] = {}
    order = _shuffled_base_order(spec)
    executed_order: list[str] = []
    parity = {"status": "PENDING"}
    try:
        hardware.set_phase(_hardware_phase(order[0]))
        hardware.start()
        for variant_id in order:
            wait_ms = float(_variant_map(spec)[variant_id]["wait_ms"])
            result = _run_variant(model=model, profile_fp=profile_fp, contract=contract, reference=reference, spec=spec, run_id=run_id, variant_id=variant_id, wait_ms=wait_ms, code=code, hardware=hardware)
            results[variant_id] = result
            executed_order.append(variant_id)
            _atomic_write(root / "results" / f"{variant_id}.json", result)
            if len(results) >= 2:
                parity = _compare_parity(results, baseline_id=executed_order[0])
                _atomic_write(root / "metrics" / "parity.json", parity)
                if parity["status"] != "PASS":
                    raise RuntimeError("Normalized self-play results diverged across wait variants; benchmark stopped")

        first_improvement = _relative_improvement(results["wait-0.5ms"], results["wait-1.0ms"])
        if 3.0 <= first_improvement < 5.0:
            for variant_id in _shuffled_repeat_order(spec):
                base_id = variant_id.removesuffix("-repeat")
                wait_ms = float(_variant_map(spec)[base_id]["wait_ms"])
                result = _run_variant(model=model, profile_fp=profile_fp, contract=contract, reference=reference, spec=spec, run_id=run_id, variant_id=variant_id, wait_ms=wait_ms, code=code, hardware=hardware)
                results[variant_id] = result
                executed_order.append(variant_id)
                _atomic_write(root / "results" / f"{variant_id}.json", result)
                parity = _compare_parity(results, baseline_id=executed_order[0])
                _atomic_write(root / "metrics" / "parity.json", parity)
                if parity["status"] != "PASS":
                    raise RuntimeError("Normalized self-play results diverged across wait variants; benchmark stopped")
        elif first_improvement >= 5.0:
            variant_id = ZERO_WAIT_ID
            result = _run_variant(model=model, profile_fp=profile_fp, contract=contract, reference=reference, spec=spec, run_id=run_id, variant_id=variant_id, wait_ms=0.0, code=code, hardware=hardware)
            results[variant_id] = result
            executed_order.append(variant_id)
            _atomic_write(root / "results" / f"{variant_id}.json", result)
            parity = _compare_parity(results, baseline_id=executed_order[0])
            _atomic_write(root / "metrics" / "parity.json", parity)
            if parity["status"] != "PASS":
                raise RuntimeError("Normalized self-play results diverged across wait variants; benchmark stopped")
        hardware.stop()
        hardware_summary = hardware.summary()
        parity = _compare_parity(results, baseline_id=executed_order[0])
        decision = _decision(results)
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
            "game_ids": {"count": len(game_ids), "first": game_ids[0], "last": game_ids[-1], "all": list(game_ids)},
            "variant_order": executed_order,
            "frozen_execution": dict(execution),
            "scientific_contract": dict(profile["self_play"]),
            "checkpoint_reference_only": True,
            "parity": parity,
            "decision": decision,
            "hardware_telemetry": hardware_summary,
            "results": results,
            "storage": {"canonical_path": str(root.relative_to(ROOT)), "contains_checkpoint_copies": False},
        }
        _atomic_write(root / "report.json", report)
        _atomic_write_text(root / "report.md", _render_report(report))
        manifest.update({"status": "ARCHIVED", "benchmark_status": "COMPLETE", "completed_at": datetime.now(timezone.utc).isoformat(), "recommendation": decision["recommendation"], "semantic_parity": parity["status"]})
        _atomic_write(root / "manifest.json", manifest)
        return report
    except BaseException as exc:
        hardware.stop()
        manifest.update({"status": "ARCHIVED", "benchmark_status": "FAILED", "failure": f"{type(exc).__name__}: {exc}", "variant_order": executed_order, "semantic_parity": parity.get("status", "UNKNOWN")})
        _atomic_write(root / "manifest.json", manifest)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    report = run_benchmark(args.run_id)
    print(json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
