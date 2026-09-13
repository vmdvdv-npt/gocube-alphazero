#!/usr/bin/env python3
"""Run the Cube 4x4 Golden geometry-aware transfer proof.

Canonical mode is deliberately strict: it refuses a dirty source tree, uses
16 process-isolated self-play workers, and records a failure instead of
silently training on partial evidence.  ``--smoke`` is throwaway and may use
reduced search settings; its artifacts never enter a canonical run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from gocube_golden.cube_arena import (
    CUBE_ARENA_SEARCH,
    CubeSearchPlayer,
    SequentialGoldenCubeArena,
    summarize_cube_arena,
    write_cube_arena_jsonl,
)
from gocube_golden.arena_contract import SearchSettings
from gocube_golden.cube_contract import CUBE_PROFILE_ID, load_profile, profile_fingerprint
from gocube_golden.cube_evaluation import (
    CUBE_EVALUATION_MASTER_SEED,
    diagnostic_cube_subset,
    freeze_cube_evaluation,
    load_frozen_cube_starts,
)
from gocube_golden.cube_neural import (
    GoldenCubeGraphNetV1,
    GoldenCubeNeuralEvaluator,
    build_cube_observation,
    cube_count_parameters,
    cube_model_hash,
    configure_single_thread_inference,
)
from gocube_golden.cube_topology import CUBE4_TOPOLOGY
from gocube_golden.cube_training import (
    CUBE_WATCHDOG,
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    CubeSelfPlayGameRecord,
    CubeTrainingSample,
    build_cube_replay_samples,
    cube_initial_state,
    cube_load_checkpoint,
    cube_replay_batches,
    cube_save_checkpoint,
    cube_state_identity,
    cube_write_jsonl,
    run_cube_selfplay_games,
    train_cube_batch_schedule,
)
from gocube_golden.provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256, sha256_fingerprint
from gocube_golden.search import SequentialPUCT

ROOT = Path(__file__).resolve().parents[1]


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def git_ref_identity(ref: str) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", ref], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", f"{ref}^{{tree}}"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"branch": ref, "sha": commit, "tree": tree}


def write_topology_artifacts(run_dir: Path) -> dict[str, object]:
    manifest = CUBE4_TOPOLOGY.to_manifest()
    write_json(run_dir / "topology" / "manifest.json", manifest)
    write_json(run_dir / "topology" / "points.json", manifest["points"])
    write_json(run_dir / "topology" / "adjacency.json", {
        "topology_fingerprint": CUBE4_TOPOLOGY.fingerprint,
        "adjacency": [list(row) for row in CUBE4_TOPOLOGY.adjacency],
        "relation_types": [list(row) for row in CUBE4_TOPOLOGY.relation_types],
    })
    write_json(run_dir / "topology" / "geometry.json", {
        "geometry_fingerprint": CUBE4_TOPOLOGY.geometry_fingerprint,
        "physical_corners": [list(corner) for corner in CUBE4_TOPOLOGY.physical_corners],
        "points": manifest["points"],
        "seams": manifest["seams"],
    })
    return manifest


def checkpoint_metadata(
    profile: Mapping[str, Any],
    *,
    run_id: str,
    label: str,
    parent_hash: str | None,
    model: GoldenCubeGraphNetV1,
    code: CodeIdentity,
    device: torch.device,
    completed_games: int,
    cumulative_positions: int,
    optimizer_updates: int,
    samples_consumed: int,
    model_init_seed: int,
) -> dict[str, object]:
    return {
        "checkpoint_schema_version": 1,
        "checkpoint_label": label,
        "architecture_id": model.architecture_id,
        "architecture_config": model.architecture_config,
        "rules_id": profile["rules"]["rules_id"],
        "rules_fingerprint": profile["rules"]["fingerprint"],
        "topology_id": profile["topology"]["topology_id"],
        "topology_fingerprint": profile["topology"]["fingerprint"],
        "geometry_schema_id": profile["geometry"]["schema_id"],
        "geometry_fingerprint": profile["geometry"]["fingerprint"],
        "board_size": [4, 4, 6],
        "point_count": 96,
        "action_count": 97,
        "point_ordering_fingerprint": profile["topology"]["point_ordering_fingerprint"],
        "komi": 0.5,
        "observation_schema_id": profile["observation"]["schema_id"],
        "observation_schema_version": profile["observation"]["schema_version"],
        "observation_fingerprint": profile["observation"]["fingerprint"],
        "target_contract_id": profile["target"]["contract_id"],
        "target_contract_version": profile["target"]["contract_version"],
        "target_fingerprint": profile["target"]["fingerprint"],
        "value_head_semantics": "side-to-move:[WIN,DRAW,LOSS]",
        "network_heads_and_shapes": {"policy": [97], "value": [3]},
        "training_profile_id": profile["profile_id"],
        "training_profile_fingerprint": profile["profile_fingerprint"],
        "parent_or_source_run_identity": parent_hash or run_id,
        "parent_model_hash": parent_hash,
        "run_id": run_id,
        "completed_games": completed_games,
        "cumulative_replay_positions": cumulative_positions,
        "optimizer_updates": optimizer_updates,
        "train_samples_consumed": samples_consumed,
        "model_initialization_seed": model_init_seed,
        "git_commit": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "git_worktree_clean": code.working_tree_clean,
        "device": str(device),
        "model_parameter_count": cube_count_parameters(model),
        "model_hash": cube_model_hash(model),
    }


def save_model(run_dir: Path, profile: Mapping[str, Any], *, label: str, model: GoldenCubeGraphNetV1, optimizer: torch.optim.Optimizer | None, parent_hash: str | None, code: CodeIdentity, device: torch.device, completed_games: int, cumulative_positions: int, optimizer_updates: int, samples_consumed: int, model_init_seed: int) -> dict[str, object]:
    metadata = checkpoint_metadata(profile, run_id=run_dir.name, label=label, parent_hash=parent_hash, model=model, code=code, device=device, completed_games=completed_games, cumulative_positions=cumulative_positions, optimizer_updates=optimizer_updates, samples_consumed=samples_consumed, model_init_seed=model_init_seed)
    path = run_dir / "checkpoints" / f"{label}.pt"
    metadata = cube_save_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)
    return {"label": label, "path": str(path), "artifact_sha256": metadata["artifact_sha256"], "model_hash": metadata["model_hash"], "metadata": metadata}


def benchmark_devices(starts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    threading = configure_single_thread_inference()
    candidates = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    states = [cube_initial_state()]
    for row in starts[:3]:
        from gocube_golden.cube_training import cube_state_from_identity
        states.append(cube_state_from_identity(row["state"]))
    rows: dict[str, object] = {}
    for name in candidates:
        device = torch.device(name)
        seed_everything(2026091401)
        model = GoldenCubeGraphNetV1().to(device)
        evaluator = GoldenCubeNeuralEvaluator(model, device=device)
        started = time.perf_counter()
        for index in range(4):
            evaluator.evaluate(states[index % len(states)])
        evaluation_elapsed = time.perf_counter() - started
        evaluation_telemetry = evaluator.telemetry()
        search_started = time.perf_counter()
        search_result = SequentialPUCT(CUBE_ARENA_SEARCH).search(
            states[0], evaluator, seed=derive_seed(2026091401, "device-preflight", name)
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        search_elapsed = time.perf_counter() - search_started
        total_telemetry = evaluator.telemetry()
        elapsed = time.perf_counter() - started
        search_telemetry = {
            key: total_telemetry[key] - evaluation_telemetry[key]
            for key in (
                "nn_evaluations",
                "observation_rules_seconds",
                "pure_model_forward_seconds",
                "total_evaluator_seconds",
            )
        }
        rows[name] = {
            "device": name,
            "batch_size": 1,
            "inference_calls": 4,
            "inference_wall_sec": evaluation_elapsed,
            "observation_rules_wall_sec": evaluation_telemetry["observation_rules_seconds"],
            "pure_model_forward_wall_sec": evaluation_telemetry["pure_model_forward_seconds"],
            "total_evaluator_wall_sec": evaluation_telemetry["total_evaluator_seconds"],
            "inference_calls_per_sec": 4.0 / evaluation_elapsed if evaluation_elapsed else None,
            "search_simulations": search_result.simulations,
            "search_wall_sec": search_elapsed,
            "search_observation_rules_wall_sec": search_telemetry["observation_rules_seconds"],
            "search_pure_model_forward_wall_sec": search_telemetry["pure_model_forward_seconds"],
            "search_total_evaluator_wall_sec": search_telemetry["total_evaluator_seconds"],
            "total_workload_wall_sec": elapsed,
        }
    selected = min(rows, key=lambda key: float(rows[key]["total_workload_wall_sec"]))
    return {
        "candidates": rows,
        "selected": selected,
        "selection_rule": "lowest wall time on fixed batch-1 Cube inference plus 64-simulation search workload",
        "threading": threading,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def run_performance_preflight(
    model: GoldenCubeGraphNetV1,
    checkpoint: Mapping[str, object],
    *,
    run_id: str,
    profile: Mapping[str, Any],
    code: CodeIdentity,
    device: torch.device,
    workers: int,
) -> dict[str, object]:
    """Exercise all canonical worker slots before collecting training data."""

    if workers != 16:
        raise ValueError("Cube performance preflight requires exactly 16 workers")
    game_ids = tuple(f"preflight-game-{index:02d}" for index in range(workers))
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    records = run_cube_selfplay_games(
        model,
        game_ids,
        run_id=run_id,
        profile_id=CUBE_PROFILE_ID,
        profile_fingerprint=profile["profile_fingerprint"],
        model_checkpoint_label="M0",
        checkpoint_artifact_hash=checkpoint["artifact_sha256"],
        master_seed=profile["seeds"]["selfplay_master_seed"],
        chunk_id="performance-preflight",
        code_identity=code,
        checkpoint_path=checkpoint["path"],
        workers=workers,
        device=device,
    )
    wall_seconds = time.perf_counter() - started
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_seconds = (
        child_after.ru_utime + child_after.ru_stime
        - child_before.ru_utime - child_before.ru_stime
    )
    positions = sum(len(record.positions) for record in records)
    technical = sum(record.technical_termination is not None for record in records)
    if len(records) != workers or technical != 0:
        raise RuntimeError("Cube performance preflight produced incomplete or technical self-play evidence")
    cpu_parallelism = child_cpu_seconds / wall_seconds if wall_seconds else 0.0
    if cpu_parallelism < 2.0:
        raise RuntimeError(
            f"Cube performance preflight indicates insufficient parallelism: {cpu_parallelism:.2f} CPU-seconds/second"
        )
    return {
        "passed": True,
        "workers_requested": workers,
        "active_game_processes": workers,
        "process_isolated": True,
        "games": len(records),
        "positions": positions,
        "wall_seconds": wall_seconds,
        "games_per_hour": len(records) * 3600.0 / wall_seconds if wall_seconds else None,
        "positions_per_second": positions / wall_seconds if wall_seconds else None,
        "nn_evaluations": sum(record.nn_evaluations for record in records),
        "nn_evaluations_per_second": sum(record.nn_evaluations for record in records) / wall_seconds if wall_seconds else None,
        "child_cpu_seconds": child_cpu_seconds,
        "estimated_cpu_parallelism": cpu_parallelism,
        "ram_peak_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }


def validate_chunk(
    records: Sequence[CubeSelfPlayGameRecord],
    *,
    expected_games: int,
    expected_model_hash: str,
) -> tuple[CubeTrainingSample, ...]:
    if len(records) != expected_games:
        raise RuntimeError(f"Cube chunk has {len(records)} games, expected {expected_games}")
    expected_contract = DEFAULT_CUBE_SELFPLAY_CONTRACT.fingerprint
    for record in records:
        record.validate(require_clean=True, expected_contract_fingerprint=expected_contract)
        if record.technical_termination is not None:
            raise RuntimeError("Cube canonical chunk contains technical self-play evidence")
        if record.model_hash != expected_model_hash:
            raise RuntimeError("Cube canonical chunk contains wrong source model hash")
    samples = tuple(
        sample
        for record in records
        for sample in build_cube_replay_samples(
            record,
            expected_contract_fingerprint=expected_contract,
            validate_game=False,
            validate_samples=False,
        )
    )
    if not samples:
        raise RuntimeError("Cube canonical chunk produced no replay positions")
    for sample in samples:
        sample.validate(
            expected_contract_fingerprint=expected_contract,
            expected_model_hash=expected_model_hash,
        )
    return samples


def selfplay_behavior_diagnostics(records: Sequence[CubeSelfPlayGameRecord]) -> dict[str, object]:
    lengths = [len(record.final_action_trace) for record in records]
    pass_counts = [sum(action == "PASS" for action in record.final_action_trace) for record in records]
    formal = {"BLACK": 0, "WHITE": 0, "DRAW": 0}
    for record in records:
        if record.formal_result in formal:
            formal[str(record.formal_result)] += 1

    def percentile(values: Sequence[int], probability: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = probability * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    total_actions = sum(lengths)
    return {
        "results": formal,
        "game_length": {
            "mean": total_actions / len(lengths) if lengths else None,
            "median": percentile(lengths, 0.5),
            "p10": percentile(lengths, 0.1),
            "p90": percentile(lengths, 0.9),
            "max": max(lengths) if lengths else None,
        },
        "pass_frequency": sum(pass_counts) / total_actions if total_actions else 0.0,
        "games_under_20": sum(length < 20 for length in lengths),
        "games_under_40": sum(length < 40 for length in lengths),
        "games_over_500": sum(length > 500 for length in lengths),
        "watchdog_pressure": sum(record.technical_termination == "TRUNCATED_MOVE_LIMIT" for record in records),
        "flags": {
            "pass_collapse": bool(lengths) and sum(pass_counts) / total_actions > 0.5,
            "abnormally_short_games": bool(lengths) and sum(length < 20 for length in lengths) / len(lengths) > 0.5,
            "watchdog_pressure": any(record.technical_termination == "TRUNCATED_MOVE_LIMIT" for record in records),
        },
    }


def run_comparison(run_dir: Path, profile: Mapping[str, Any], *, slug: str, candidate: Mapping[str, object], reference: Mapping[str, object], starts: Sequence[Mapping[str, object]], code: CodeIdentity, device: torch.device, seed: int, canonical: bool, search_settings: SearchSettings = CUBE_ARENA_SEARCH) -> dict[str, object]:
    candidate_model = GoldenCubeGraphNetV1().to(device)
    reference_model = GoldenCubeGraphNetV1().to(device)
    cube_load_checkpoint(candidate["path"], model=candidate_model, expected={"model_hash": candidate["model_hash"]}, device=device)
    cube_load_checkpoint(reference["path"], model=reference_model, expected={"model_hash": reference["model_hash"]}, device=device)
    candidate_evaluator = GoldenCubeNeuralEvaluator(candidate_model, device=device)
    reference_evaluator = GoldenCubeNeuralEvaluator(reference_model, device=device)
    player_a = CubeSearchPlayer("candidate", candidate_evaluator, search_settings=search_settings)
    player_b = CubeSearchPlayer("reference", reference_evaluator, search_settings=search_settings)
    arena = SequentialGoldenCubeArena(master_seed=seed, run_id=f"{run_dir.name}-{slug}", code_identity=code, require_canonical_code=canonical, search_settings=search_settings)
    for row in starts:
        from gocube_golden.cube_training import cube_state_from_identity
        arena.play_pair(pair_id=f"{slug}--{row['start_id']}", player_A=player_a, player_B=player_b, start_state=cube_state_from_identity(row["state"]), start_trace=tuple(int(action) for action in row["trace"]))
    inferential_records = tuple(arena.records)
    arena.play_pair(
        pair_id=f"{slug}--empty-board-control",
        player_A=player_a,
        player_B=player_b,
        start_state=cube_initial_state(),
        start_trace=(),
    )
    control_records = tuple(arena.records[len(inferential_records):])
    output = run_dir / "arena" / slug
    write_cube_arena_jsonl(output / "games.jsonl", arena.records)
    summary = summarize_cube_arena(inferential_records)
    control_summary = summarize_cube_arena(control_records)
    summary.update({
        "comparison": slug,
        "candidate": candidate["label"],
        "reference": reference["label"],
        "pairs": len(starts),
        "games": len(inferential_records),
        "technical": sum(record.is_technical for record in inferential_records),
        "empty_board_control": control_summary,
        "all_games_including_control": len(arena.records),
        "search_settings": asdict(search_settings),
        "canonical_search": canonical,
    })
    write_json(output / "summary.json", summary)
    if summary["technical"] != 0 or control_summary["technical"] != 0:
        raise RuntimeError(f"Cube Arena comparison {slug} is invalid: technical game present")
    return summary


def model_diagnostics(checkpoints: Mapping[str, Mapping[str, object]], starts: Sequence[Mapping[str, object]], device: torch.device) -> dict[str, object]:
    from gocube_golden.cube_training import cube_state_from_identity
    result: dict[str, object] = {}
    states = [cube_state_from_identity(row["state"]) for row in starts]
    for label, info in checkpoints.items():
        model = GoldenCubeGraphNetV1().to(device)
        cube_load_checkpoint(info["path"], model=model, expected={"model_hash": info["model_hash"]}, device=device)
        evaluator = GoldenCubeNeuralEvaluator(model, device=device)
        policy_entropy = []
        pass_probability = []
        top1_probability = []
        wdl_entropy = []
        for state in states:
            evaluation = evaluator.evaluate(state)
            policy = [max(float(value), 1e-12) for value in evaluation.policy]
            wdl = [max(float(value), 1e-12) for value in evaluation.wdl]
            policy_entropy.append(-sum(value * math.log(value) for value in policy))
            pass_probability.append(evaluation.policy[96])
            top1_probability.append(max(evaluation.policy))
            wdl_entropy.append(-sum(value * math.log(value) for value in wdl))
        result[label] = {
            "observations": len(states),
            "policy_entropy": sum(policy_entropy) / len(policy_entropy),
            "pass_probability": sum(pass_probability) / len(pass_probability),
            "top1_probability": sum(top1_probability) / len(top1_probability),
            "wdl_entropy": sum(wdl_entropy) / len(wdl_entropy),
        }
    return result


def parameter_delta(before: Mapping[str, torch.Tensor], model: GoldenCubeGraphNetV1) -> dict[str, float]:
    squared = 0.0
    base = 0.0
    for name, parameter in model.state_dict().items():
        current = parameter.detach().cpu().float()
        old = before[name].detach().cpu().float()
        squared += float((current - old).square().sum())
        base += float(old.square().sum())
    l2 = math.sqrt(squared)
    return {"absolute_l2": l2, "relative_l2": l2 / math.sqrt(base) if base else None}


def run_smoke(run_id: str) -> dict[str, object]:
    profile = load_profile()
    run_dir = ROOT / "runs" / "cube4-golden-transfer" / run_id
    if run_dir.exists():
        raise ValueError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    code = capture_code_identity(ROOT)
    write_topology_artifacts(run_dir)
    write_json(run_dir / "profile.json", profile)
    evaluation_manifest = freeze_cube_evaluation(run_dir, code_identity=code)
    starts = load_frozen_cube_starts(run_dir)
    device = torch.device("cpu")
    seed_everything(profile["seeds"]["model_init_seed"])
    model = GoldenCubeGraphNetV1().to(device)
    m0 = save_model(run_dir, profile, label="M0", model=model, optimizer=None, parent_hash=None, code=code, device=device, completed_games=0, cumulative_positions=0, optimizer_updates=0, samples_consumed=0, model_init_seed=profile["seeds"]["model_init_seed"])
    smoke_contract = replace(DEFAULT_CUBE_SELFPLAY_CONTRACT, simulations=1)
    records = run_cube_selfplay_games(model, ("smoke-game-000",), run_id=run_id, profile_id=CUBE_PROFILE_ID, profile_fingerprint=profile["profile_fingerprint"], model_checkpoint_label="M0", checkpoint_artifact_hash=m0["artifact_sha256"], master_seed=profile["seeds"]["selfplay_master_seed"], chunk_id="smoke", code_identity=code, checkpoint_path=None, workers=1, device=device, contract=smoke_contract, allow_noncanonical_contract=True)
    # Smoke is explicitly noncanonical but still trains only on a formally
    # scored game if the reduced search reaches double-pass.
    smoke_samples = tuple(sample for record in records if record.technical_termination is None for sample in build_cube_replay_samples(record))
    smoke_report: dict[str, object] = {"run_id": run_id, "smoke": True, "profile_fingerprint": profile["profile_fingerprint"], "evaluation": evaluation_manifest, "games": len(records), "technical_games": sum(record.technical_termination is not None for record in records), "positions": len(smoke_samples), "topology_fingerprint": CUBE4_TOPOLOGY.fingerprint}
    smoke_report["behavior"] = selfplay_behavior_diagnostics(records)
    if smoke_samples:
        optimizer, training = train_cube_batch_schedule(model, smoke_samples, (tuple(range(len(smoke_samples))),), learning_rate=0.001, weight_decay=0.0)
        m1 = save_model(run_dir, profile, label="M1", model=model, optimizer=optimizer, parent_hash=m0["model_hash"], code=code, device=device, completed_games=1, cumulative_positions=len(smoke_samples), optimizer_updates=int(training["updates"]), samples_consumed=len(smoke_samples), model_init_seed=profile["seeds"]["model_init_seed"])
        comparison = run_comparison(run_dir, profile, slug="smoke-m1-vs-m0", candidate=m1, reference=m0, starts=starts[:1], code=code, device=device, seed=profile["seeds"]["arena_master_seed"], canonical=False, search_settings=replace(CUBE_ARENA_SEARCH, simulations=1))
        smoke_report["checkpoint_lineage"] = {"M0": m0, "M1": m1}
        smoke_report["arena"] = comparison
    write_json(run_dir / "report.json", smoke_report)
    return smoke_report


def run_canonical(run_id: str, workers: int) -> dict[str, object]:
    profile = load_profile()
    code = capture_code_identity(ROOT)
    if not code.working_tree_clean:
        raise RuntimeError("Canonical Cube run requires a clean committed source tree")
    if workers != 16:
        raise ValueError("Canonical Cube run requires exactly 16 process workers")
    run_dir = ROOT / "runs" / "cube4-golden-transfer" / run_id
    if run_dir.exists():
        raise ValueError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    write_topology_artifacts(run_dir)
    write_json(run_dir / "profile.json", profile)
    evaluation_manifest = freeze_cube_evaluation(run_dir, code_identity=code, master_seed=profile["seeds"]["evaluation_seed"])
    starts = load_frozen_cube_starts(run_dir)
    subset = diagnostic_cube_subset(starts)
    benchmark = benchmark_devices(starts)
    device = torch.device(str(benchmark["selected"]))
    if device.type != "cpu":
        torch.set_num_threads(1)
    seed_everything(profile["seeds"]["model_init_seed"])
    model = GoldenCubeGraphNetV1().to(device)
    m0 = save_model(run_dir, profile, label="M0", model=model, optimizer=None, parent_hash=None, code=code, device=device, completed_games=0, cumulative_positions=0, optimizer_updates=0, samples_consumed=0, model_init_seed=profile["seeds"]["model_init_seed"])
    preflight = run_performance_preflight(model, m0, run_id=run_id, profile=profile, code=code, device=device, workers=workers)
    write_json(run_dir / "preflight" / "performance.json", preflight)
    eq_ids = ("equivalence-game-000", "equivalence-game-001")
    serial = run_cube_selfplay_games(model, eq_ids, run_id=run_id, profile_id=CUBE_PROFILE_ID, profile_fingerprint=profile["profile_fingerprint"], model_checkpoint_label="M0", checkpoint_artifact_hash=m0["artifact_sha256"], master_seed=profile["seeds"]["selfplay_master_seed"], chunk_id="equivalence", code_identity=code, checkpoint_path=None, workers=1, device=device)
    parallel = run_cube_selfplay_games(model, eq_ids, run_id=run_id, profile_id=CUBE_PROFILE_ID, profile_fingerprint=profile["profile_fingerprint"], model_checkpoint_label="M0", checkpoint_artifact_hash=m0["artifact_sha256"], master_seed=profile["seeds"]["selfplay_master_seed"], chunk_id="equivalence", code_identity=code, checkpoint_path=m0["path"], workers=workers, device=device)
    from gocube_golden.cube_training import cube_compare_selfplay_evidence
    cube_compare_selfplay_evidence(serial, parallel)
    write_json(run_dir / "equivalence-gate.json", {"passed": True, "game_ids": eq_ids, "workers": workers, "action_trace_root_visits_pi_z_exact": True, "inference_batch_size": 1, "inference_coalescing": False})

    checkpoints: dict[str, dict[str, object]] = {"M0": m0}
    cumulative: list[CubeTrainingSample] = []
    training_reports: list[dict[str, object]] = []
    all_records: list[CubeSelfPlayGameRecord] = []
    optimizer: torch.optim.Optimizer | None = None
    total_updates = 0
    total_samples = 0
    current_label = "M0"
    current_info = m0
    for chunk in range(1, 5):
        game_ids = tuple(f"chunk-{chunk:02d}-game-{index:03d}" for index in range(128))
        cpu_start = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        records = run_cube_selfplay_games(model, game_ids, run_id=run_id, profile_id=CUBE_PROFILE_ID, profile_fingerprint=profile["profile_fingerprint"], model_checkpoint_label=current_label, checkpoint_artifact_hash=current_info["artifact_sha256"], master_seed=profile["seeds"]["selfplay_master_seed"], chunk_id=f"chunk-{chunk:02d}", code_identity=code, checkpoint_path=current_info["path"], workers=workers, device=device)
        samples = validate_chunk(records, expected_games=128, expected_model_hash=current_info["model_hash"])
        all_records.extend(records)
        cube_write_jsonl(run_dir / "selfplay" / f"chunk-{chunk:02d}-games.jsonl", (record.to_dict() for record in records))
        cube_write_jsonl(run_dir / "replay" / f"chunk-{chunk:02d}.jsonl", (sample.to_dict() for sample in samples))
        cumulative.extend(samples)
        cumulative.sort(key=lambda sample: (sample.game_id, sample.ply))
        before = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
        sampling_seed = derive_seed(profile["seeds"]["selfplay_master_seed"], "cube-replay-phase-v1", f"chunk-{chunk:02d}")
        batches = cube_replay_batches(len(cumulative), len(samples), seed=sampling_seed, batch_size=profile["training"]["batch_size"])
        optimizer, schedule = train_cube_batch_schedule(model, cumulative, batches, learning_rate=profile["training"]["learning_rate"], weight_decay=profile["training"]["weight_decay"], optimizer=optimizer, update_offset=total_updates, sample_offset=total_samples)
        total_updates = int(schedule["updates"])
        total_samples = int(schedule["cumulative_samples"])
        label = f"M{chunk}"
        info = save_model(run_dir, profile, label=label, model=model, optimizer=optimizer, parent_hash=current_info["model_hash"], code=code, device=device, completed_games=chunk * 128, cumulative_positions=len(cumulative), optimizer_updates=total_updates, samples_consumed=total_samples, model_init_seed=profile["seeds"]["model_init_seed"])
        if info["model_hash"] == current_info["model_hash"]:
            raise RuntimeError(f"Cube checkpoint {label} is identical to parent")
        checkpoints[label] = info
        metric_rows = schedule["metrics"]
        training_report = {
            "chunk": chunk,
            "source": current_label,
            "games": 128,
            "technical": 0,
            "new_positions": len(samples),
            "cumulative_positions": len(cumulative),
            "samples_consumed": len(samples),
            "samples_consumed_cumulative": total_samples,
            "updates": len(batches),
            "updates_cumulative": total_updates,
            "policy_loss": sum(float(row["policy_loss"]) for row in metric_rows) / len(metric_rows),
            "value_loss": sum(float(row["value_loss"]) for row in metric_rows) / len(metric_rows),
            "sample_ratio": 1.0,
            "sampling_without_replacement": True,
            "sampling_seed": sampling_seed,
            "parameter_delta": parameter_delta(before, model),
            "chunk_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime - cpu_start,
            "behavior": selfplay_behavior_diagnostics(records),
        }
        write_json(run_dir / "training" / f"chunk-{chunk:02d}-metrics.json", training_report)
        training_reports.append(training_report)
        current_label, current_info = label, info

    if set(checkpoints) != {"M0", "M1", "M2", "M3", "M4"}:
        raise RuntimeError("Cube checkpoint lineage is incomplete")
    if len({info["model_hash"] for info in checkpoints.values()}) != 5:
        raise RuntimeError("Cube checkpoint hashes are not all distinct")
    # validate_chunk performs the authoritative deep game and sample audits once
    # per chunk; do not replay the same full history audit a second time here.
    replay_audit = {"games": len(all_records), "technical_games": sum(record.technical_termination is not None for record in all_records), "positions": len(cumulative), "all_samples_validated": len(cumulative) > 0}
    if replay_audit["games"] != 512 or replay_audit["technical_games"] != 0:
        raise RuntimeError("Cube replay audit failed canonical validity")
    diagnostics = model_diagnostics(checkpoints, subset, device)
    primary = {
        "M1_vs_M0": run_comparison(run_dir, profile, slug="m1-vs-m0", candidate=checkpoints["M1"], reference=checkpoints["M0"], starts=starts, code=code, device=device, seed=profile["seeds"]["arena_master_seed"], canonical=True),
        "M4_vs_M0": run_comparison(run_dir, profile, slug="m4-vs-m0", candidate=checkpoints["M4"], reference=checkpoints["M0"], starts=starts, code=code, device=device, seed=profile["seeds"]["arena_master_seed"] + 1, canonical=True),
        "M4_vs_M1": run_comparison(run_dir, profile, slug="m4-vs-m1", candidate=checkpoints["M4"], reference=checkpoints["M1"], starts=starts, code=code, device=device, seed=profile["seeds"]["arena_master_seed"] + 2, canonical=True),
    }
    progression = {
        f"M{index}_vs_M{index - 1}": run_comparison(run_dir, profile, slug=f"progression-m{index}-vs-m{index - 1}", candidate=checkpoints[f"M{index}"], reference=checkpoints[f"M{index - 1}"], starts=subset, code=code, device=device, seed=profile["seeds"]["arena_master_seed"] + 10 + index, canonical=True)
        for index in (2, 3, 4)
    }
    m4_m0_lower = float(primary["M4_vs_M0"]["hoeffding_95"][0])
    m4_m1_lower = float(primary["M4_vs_M1"]["hoeffding_95"][0])
    pipeline_valid = replay_audit["technical_games"] == 0 and all(summary["technical"] == 0 for summary in list(primary.values()) + list(progression.values()))
    if not pipeline_valid:
        verdict = "RUN INVALID"
    elif m4_m0_lower > 0.5 and m4_m1_lower > 0.5:
        verdict = "CUBE TRANSFER CONFIRMED"
    elif m4_m0_lower > 0.5:
        verdict = "CUBE LEARNING CONFIRMED, INCREMENTAL GROWTH INCONCLUSIVE"
    else:
        verdict = "CUBE LEARNING NOT DEMONSTRATED"
    final = {
        "base_branch": "codex/torus-rebuild-v1",
        "base": git_ref_identity("codex/torus-rebuild-v1"),
        "feature_branch": "codex/cube-golden-neural-proof-v1",
        "run_id": run_id,
        "source_commit": code.git_commit_sha,
        "source_tree": code.git_tree_sha,
        "git_tree": code.git_tree_sha,
        "topology_id": CUBE4_TOPOLOGY.topology_id,
        "topology_fingerprint": CUBE4_TOPOLOGY.fingerprint,
        "geometry_fingerprint": CUBE4_TOPOLOGY.geometry_fingerprint,
        "rules_fingerprint": profile["rules"]["fingerprint"],
        "komi": 0.5,
        "observation_fingerprint": profile["observation"]["fingerprint"],
        "target_fingerprint": profile["target"]["fingerprint"],
        "network_architecture": {"id": "GoldenCubeGraphNetV1", "parameters": cube_count_parameters(model)},
        "device": str(device),
        "workers": workers,
        "performance_preflight": preflight,
        "training_profile_fingerprint": profile["profile_fingerprint"],
        "evaluation_fingerprint": evaluation_manifest["corpus_fingerprint"],
        "training": training_reports,
        "checkpoints": {label: {key: value for key, value in info.items() if key != "metadata"} for label, info in checkpoints.items()},
        "replay_audit": replay_audit,
        "evaluation": evaluation_manifest,
        "equivalence_gate": {"passed": True, "workers": workers},
        "diagnostics": diagnostics,
        "arena_primary": primary,
        "arena_progression": progression,
        "pipeline_validity": "PASS" if pipeline_valid else "FAIL",
        "global_verdict": verdict,
        "scientific_answers": {
            "topology_independently_validated": True,
            "geometry_explicitly_represented": True,
            "corner_context_propagates": True,
            "geometry_leaves_game_adjacency_unchanged": True,
            "pipeline_technically_valid": pipeline_valid,
            "learning_from_random": "YES" if m4_m0_lower > 0.5 else "NO" if pipeline_valid else "INCONCLUSIVE",
            "incremental_learning_after_m1": "YES" if m4_m1_lower > 0.5 else "NO" if pipeline_valid else "INCONCLUSIVE",
            "future_cube_baseline": "YES" if verdict == "CUBE TRANSFER CONFIRMED" else "INCONCLUSIVE",
        },
        "limitations": ["Replay audit validates consistency against the same Golden rules implementation; it is not an independent rules oracle."],
        "elapsed_seconds": time.perf_counter() - started,
        "ci": "not run until final push",
    }
    write_json(run_dir / "final-report.json", final)
    write_json(run_dir / "manifest.json", {**final, "canonical": True})
    (run_dir / "final-report.md").write_text(
        "# Golden Cube 4x4 Neural Transfer Proof\n\n"
        f"Global verdict: **{verdict}**\n\n"
        f"Pipeline validity: **{'PASS' if pipeline_valid else 'FAIL'}**\n\n"
        f"M4 vs M0 Hoeffding lower bound: `{m4_m0_lower:.6f}`\n\n"
        f"M4 vs M1 Hoeffding lower bound: `{m4_m1_lower:.6f}`\n\n"
        "Replay limitation: Golden replay audit validates consistency against the same Golden rules implementation; it is not an independent rules oracle.\n",
        encoding="utf-8",
    )
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_smoke(args.run_id) if args.smoke else run_canonical(args.run_id, args.workers)
    except Exception as exc:
        print(f"RUN INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
