#!/usr/bin/env python3
"""Run the frozen three-arm Torus9 M137 5CH komi calibration.

This is an evaluation-only entrypoint.  It deliberately has no training or
lineage-continuation path: the one frozen checkpoint is evaluated at komi
0.5, 1.5, and 2.5 with the same 512 paired starts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.artifact_graph import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)
from gocube_golden.artifact_resolver import (
    ResolvedArtifact,
    ResolvedCheckpointNode,
    ResolvedEffectiveConfig,
)
from gocube_golden.orchestrator_v2._arena_runner_core import ArenaRunRequest
from gocube_golden.orchestrator_v2.arena_runner import ArenaRunnerV2
from gocube_golden.orchestrator_v2.contracts import StartsetRef
from gocube_golden.orchestrator_v2.version import mark_v2_process
from gocube_golden.provenance import derive_seed, file_sha256, sha256_fingerprint
from gocube_golden.run_storage import ResolvedCheckpoint
from gocube_golden.state import rules_fingerprint_for
from gocube_golden.torus9 import generate_torus9_evaluation_starts
from gocube_golden.torus9_contract import TORUS9_ARENA_MOVE_LIMIT, TORUS9_POINT_COUNT
from gocube_golden.torus9_m137_5ch import (
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    M137_FIVE_CHANNEL_CHANNELS,
)
from gocube_golden.topology import TORUS_9X9
from tools.arena import run_arena as production_arena
from tools.arena_engine import ArenaExecutionConfig


EVALUATION_ID = "torus9-m137-5ch-komi-calibration-1024x256-20260925-v1"
MASTER_SEED = 20260925
KOMI_ARMS = (0.5, 1.5, 2.5)
GAMES_PER_ARM = 1024
PAIRS = GAMES_PER_ARM // 2
SIMULATIONS = 256
BOOTSTRAP_RESAMPLES = 20_000
MAX_TECHNICAL_RETRIES = 3
CHECKPOINT_ID = "M137-5CH-bootstrap"
LINEAGE_ID = "new_komi"
CANONICAL_M137_SHA256 = "sha256:71cfc78dab3fe217b3c435a765790efe6f6c4fa42a7d21479f3fd909adf341fe"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Malformed JSONL row: {path}")
            rows.append(value)
    return rows


def _config_payload() -> dict[str, object]:
    return {
        "schema": "torus9-m137-5ch-komi-calibration-config-v1",
        "evaluation_id": EVALUATION_ID,
        "topology": "torus9",
        "rules": {
            "topology": "Torus 9x9",
            "points": 81,
            "scoring": "exact graph-area",
            "positional_superko": True,
            "suicide_forbidden": True,
            "termination": "two passes",
            "watchdog_plies": 1000,
            "komi_arms": list(KOMI_ARMS),
        },
        "search": {
            "simulations": SIMULATIONS,
            "cpuct": 1.25,
            "fpu": 0.0,
            "root_noise": False,
            "temperature": 0.0,
            "fast_search": False,
            "resign": False,
            "deterministic_tie_break": True,
        },
        "paired_design": {
            "games_per_arm": GAMES_PER_ARM,
            "paired_starts": PAIRS,
            "color_swap": True,
            "same_pair_ids_across_arms": True,
            "master_seed": MASTER_SEED,
        },
        "execution": {
            "device": "cuda",
            "workers": 16,
            "contexts_per_worker": 12,
            "simultaneous_context_capacity": 192,
            "central_cuda_inference_broker": True,
            "batching": True,
            "inference_batch_cap": 64,
            "inference_wait_ms": 4.0,
            "strict_production": True,
        },
        "statistics": {
            "cluster_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed_derivation": "sha256(master_seed, komi, statistic)",
            "conservative_interval": "paired-hoeffding-95-v1",
        },
        "training_started": False,
    }


def _config_fingerprint() -> str:
    return sha256_fingerprint(_config_payload())


def _startset_payload() -> dict[str, object]:
    generated = generate_torus9_evaluation_starts(
        master_seed=MASTER_SEED,
        accepted_per_stratum=PAIRS // 8,
        komi=0.5,
    )
    if len(generated) != PAIRS:
        raise RuntimeError(f"Expected {PAIRS} generated starts, got {len(generated)}")
    pairs: list[dict[str, object]] = []
    for index, row in enumerate(generated):
        pairs.append(
            {
                "pair_index": index,
                "pair_id": f"torus9-m137-5ch-calibration-pair-{index:04d}",
                "start_id": row["start_id"],
                "prefix_length": row["prefix_length"],
                "candidate_index": row["candidate_index"],
                "candidate_seed": row["candidate_seed"],
                "trace": row["trace"],
                "state": row["state"],
                "source_exact_identity_fingerprint": row["exact_identity_fingerprint"],
            }
        )
    return {
        "schema": "torus9-frozen-startset-v1",
        "generator": "gocube_golden.torus9.generate_torus9_evaluation_starts",
        "generator_version": "torus9-evaluation-starts-v2",
        "master_seed": MASTER_SEED,
        "pairs": pairs,
        "pair_count": PAIRS,
        "base_generation_komi": 0.5,
        "arm_rebinding": "rules-owned komi and rules fingerprint only; board/history/trace unchanged",
    }


def _ensure_frozen_startset(root: Path) -> tuple[Path, str, list[str]]:
    path = root / "frozen-startset.json"
    if path.is_file():
        payload = _read_json(path)
    else:
        payload = _startset_payload()
        payload["fingerprint"] = sha256_fingerprint(payload)
        _write_json(path, payload)
    fingerprint = str(payload.get("fingerprint", ""))
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if fingerprint != sha256_fingerprint(body):
        raise ValueError("Frozen startset fingerprint is not self-consistent")
    if int(payload.get("master_seed", -1)) != MASTER_SEED or int(payload.get("pair_count", -1)) != PAIRS:
        raise ValueError("Frozen startset master-seed/pair-count mismatch")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != PAIRS:
        raise ValueError("Frozen startset does not contain exactly 512 pairs")
    pair_ids = [str(pair["pair_id"]) for pair in pairs if isinstance(pair, Mapping) and "pair_id" in pair]
    if len(pair_ids) != PAIRS or len(set(pair_ids)) != PAIRS:
        raise ValueError("Frozen startset pair IDs are not unique")
    return path, fingerprint, pair_ids


def _ensure_schedule(root: Path, *, startset_fingerprint: str, pair_ids: Sequence[str]) -> tuple[Path, str]:
    path = root / "schedule.json"
    if path.is_file():
        payload = _read_json(path)
    else:
        games = []
        ordinal = 0
        for pair_index, pair_id in enumerate(pair_ids):
            for suffix, candidate_black in (("g1", True), ("g2", False)):
                games.append(
                    {
                        "ordinal": ordinal,
                        "pair_index": pair_index,
                        "pair_id": pair_id,
                        "game_id": f"{pair_id}--{suffix}",
                        "candidate_black": candidate_black,
                    }
                )
                ordinal += 1
        payload = {
            "schema": "torus9-frozen-calibration-schedule-v1",
            "master_seed": MASTER_SEED,
            "startset_fingerprint": startset_fingerprint,
            "games": games,
            "pair_ids": list(pair_ids),
        }
        payload["fingerprint"] = sha256_fingerprint(payload)
        _write_json(path, payload)
    fingerprint = str(payload.get("fingerprint", ""))
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if fingerprint != sha256_fingerprint(body):
        raise ValueError("Calibration schedule fingerprint is not self-consistent")
    if payload.get("startset_fingerprint") != startset_fingerprint or payload.get("pair_ids") != list(pair_ids):
        raise ValueError("Calibration schedule is not bound to the frozen startset")
    games = payload.get("games")
    if not isinstance(games, list) or len(games) != GAMES_PER_ARM:
        raise ValueError("Calibration schedule does not contain exactly 1024 games")
    return path, fingerprint


def _checkpoint_reference(checkpoint_path: Path, evaluation_root: Path) -> tuple[ResolvedCheckpointNode, dict[str, object]]:
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    metadata = _read_json(metadata_path)
    manifest_path = checkpoint_path.parent.parent / "manifest.json"
    manifest = _read_json(manifest_path)
    actual_sha = file_sha256(checkpoint_path)
    manifest_checkpoint = manifest.get("bootstrap_checkpoint")
    if not isinstance(manifest_checkpoint, Mapping) or str(manifest_checkpoint.get("sha256")) != actual_sha:
        raise ValueError("Bootstrap manifest does not match the selected checkpoint SHA-256")
    architecture = metadata.get("architecture_config")
    if (
        metadata.get("architecture_id") != M137_FIVE_CHANNEL_ARCHITECTURE_ID
        or not isinstance(architecture, Mapping)
        or architecture.get("input_channels") != 5
        or metadata.get("observation_shape") != [5, TORUS9_POINT_COUNT]
    ):
        raise ValueError("Selected checkpoint is not the canonical M137-derived 5CH artifact")
    if "komi" in metadata or "komi" in architecture.get("observation_channels", ()):
        raise ValueError("Komi is present in the selected neural observation contract")
    checkpoint_model_hash = str(metadata.get("converted_model_hash", ""))
    if not checkpoint_model_hash:
        raise ValueError("Selected checkpoint metadata has no converted model hash")
    if str(metadata.get("source_checkpoint_sha256")) != CANONICAL_M137_SHA256:
        raise ValueError("Selected checkpoint is not derived from the pinned canonical M137 source")
    ref = CheckpointRef(
        topology="torus9",
        lineage_id=LINEAGE_ID,
        checkpoint_id=CHECKPOINT_ID,
        generation=137,
        path="checkpoints/M137-5CH-bootstrap.pt",
        sha256=actual_sha,
    )
    config = EffectiveConfig(
        topology="torus9",
        compatibility={"topology": "torus9", "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID},
        arena={"simulations": SIMULATIONS, "komi_arms": list(KOMI_ARMS)},
    )
    config_ref = EffectiveConfigRef(
        ArtifactRef("metadata/calibration-effective-config.json", config.fingerprint),
        config.fingerprint,
    )
    config_artifact = ResolvedArtifact(
        ref=config_ref.artifact,
        path=evaluation_root / "config.json",
        owner_root=evaluation_root,
        owner_topology="torus9",
        owner_lineage_id=LINEAGE_ID,
        owner_status="ACTIVE",
    )
    resolved_config = ResolvedEffectiveConfig(config_ref, config_artifact, config)
    provenance_ref = ArtifactRef("metadata/calibration-provenance.json", sha256_fingerprint({"checkpoint": actual_sha}))
    provenance = ResolvedArtifact(
        ref=provenance_ref,
        path=evaluation_root / "checkpoint-reference.json",
        owner_root=evaluation_root,
        owner_topology="torus9",
        owner_lineage_id=LINEAGE_ID,
        owner_status="ACTIVE",
    )
    node = CheckpointNode(
        checkpoint=ref,
        genesis=True,
        parent=None,
        fresh_replay=None,
        effective_config=config_ref,
        provenance=provenance_ref,
    )
    physical = ResolvedCheckpoint(
        topology="torus9",
        lineage_id=LINEAGE_ID,
        checkpoint_id=CHECKPOINT_ID,
        generation=137,
        path=checkpoint_path,
        sha256=actual_sha,
        owner_status="ACTIVE",
        reference={"architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID, "model_hash": checkpoint_model_hash},
    )
    resolved = ResolvedCheckpointNode(
        node=node,
        checkpoint=physical,
        effective_config=resolved_config,
        provenance=provenance,
        owner_root=evaluation_root,
        owner_status="ACTIVE",
    )
    return resolved, {
        "checkpoint_id": CHECKPOINT_ID,
        "lineage_id": LINEAGE_ID,
        "path": str(checkpoint_path.resolve()),
        "sha256": actual_sha,
        "model_hash": checkpoint_model_hash,
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "input_channels": 5,
        "observation_channels": list(M137_FIVE_CHANNEL_CHANNELS),
        "komi_observation_channel": False,
        "source_lineage": metadata.get("source_lineage"),
        "source_checkpoint": metadata.get("source_checkpoint"),
        "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256"),
    }


def _execution_config(games: int) -> ArenaExecutionConfig:
    return ArenaExecutionConfig(
        games=games,
        workers=16,
        games_per_worker=12,
        inference_batch_rows=64,
        inference_batch_wait_ms=4.0,
        device="cuda",
        strict_production=True,
    )


def _scientific_contract(komi: float, games: int) -> dict[str, object]:
    return {
        "schema": "torus9-m137-5ch-komi-calibration-scientific-v1",
        "profile": "torus9",
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "input_channels": 5,
        "komi_observation_channel": False,
        "topology": "torus9x9",
        "points": 81,
        "scoring": "exact graph-area",
        "positional_superko": True,
        "suicide_forbidden": True,
        "termination": "two passes",
        "watchdog_plies": 1000,
        "komi": float(komi),
        "games": int(games),
        "mcts_simulations": SIMULATIONS,
        "cpuct": 1.25,
        "fpu": 0.0,
        "root_noise": False,
        "temperature": 0.0,
        "fast_search": False,
        "resign": False,
        "deterministic_tie_break": True,
        "paired_starts_color_swap": True,
        "technical_fail_closed": True,
    }


def _execution_contract(config: ArenaExecutionConfig) -> dict[str, object]:
    return {
        "engine": "process-central-inference-v1",
        "device": config.device,
        "workers": config.workers,
        "contexts_per_worker": config.games_per_worker,
        "configured_context_capacity": config.workers * config.games_per_worker,
        "inference_batch_cap": config.inference_batch_rows,
        "inference_wait_ms": config.inference_batch_wait_ms,
        "strict_production": config.strict_production,
    }


def _fake_arena_runner() -> ArenaRunnerV2:
    # ArenaRunnerV2 owns identity, paired-start references, durable output
    # naming, and reuse semantics.  The wrapper keeps this calibration on the
    # current clean checkout instead of silently selecting a training runtime.
    def execute(**kwargs: object) -> Mapping[str, object]:
        return production_arena(**kwargs)  # type: ignore[arg-type]

    return ArenaRunnerV2(engine=execute)


def _point(row: Mapping[str, object]) -> float:
    result = row.get("formal_result")
    if result == "BLACK":
        return 1.0
    if result == "WHITE":
        return 0.0
    if result == "DRAW":
        return 0.5
    raise ValueError("scientific point requested for non-terminal result")


def _pair_points(rows: Sequence[Mapping[str, object]]) -> list[float]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    values: list[float] = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2 or any(row.get("technical_termination") is not None for row in pair):
            raise ValueError(f"Cannot form a valid paired statistic for {pair_id}")
        values.append(sum(_point(row) for row in pair) / 2.0)
    return values


def _bootstrap_interval(values: Sequence[float], *, seed: int) -> list[float]:
    if not values:
        raise ValueError("Bootstrap requires non-empty pair values")
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES)]
    means.sort()
    return [means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]]


def _hoeffding_interval(values: Sequence[float]) -> list[float]:
    mean = statistics.fmean(values)
    radius = math.sqrt(math.log(2.0 / 0.05) / (2.0 * len(values)))
    return [max(0.0, mean - radius), min(1.0, mean + radius)]


def _arm_statistics(komi: float, rows: Sequence[Mapping[str, object]], telemetry: Sequence[Mapping[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if row.get("technical_termination") is None]
    technical = [row for row in rows if row.get("technical_termination") is not None]
    black = sum(row.get("formal_result") == "BLACK" for row in valid)
    white = sum(row.get("formal_result") == "WHITE" for row in valid)
    draws = sum(row.get("formal_result") == "DRAW" for row in valid)
    if black + white + draws != len(valid):
        raise ValueError(f"Komi {komi:g} has an unclassified valid result")
    pair_values = _pair_points(valid) if not technical else []
    black_rate = (black + 0.5 * draws) / len(valid) if valid else None
    white_rate = (white + 0.5 * draws) / len(valid) if valid else None
    bootstrap = (
        _bootstrap_interval(pair_values, seed=derive_seed(MASTER_SEED, komi, "black-point-rate"))
        if pair_values
        else None
    )
    wall = sum(float(item.get("wall_time_sec", 0.0)) for item in telemetry)
    executed = sum(int(item.get("games", 0)) for item in telemetry)
    moves = sum(int(item.get("moves", 0)) for item in telemetry)
    batch_calls = sum(int(item.get("inference_forward_calls", 0)) for item in telemetry)
    batch_weight = sum(float(item.get("mean_inference_batch_rows", 0.0)) * int(item.get("inference_forward_calls", 0)) for item in telemetry)
    peaks = [int(item.get("peak_contexts_global", 0)) for item in telemetry]
    steady = [item.get("steady_state_active_contexts", {}) for item in telemetry]
    technical_by_reason: dict[str, int] = {}
    for row in technical:
        reason = str(row.get("technical_termination"))
        technical_by_reason[reason] = technical_by_reason.get(reason, 0) + 1
    return {
        "komi": komi,
        "games": len(rows),
        "valid_games": len(valid),
        "black_wins": black,
        "white_wins": white,
        "draws": draws,
        "black_point_rate": black_rate,
        "white_point_rate": white_rate,
        "black_percent": None if black_rate is None else 100.0 * black_rate,
        "black_delta_from_50_percentage_points": None if black_rate is None else 100.0 * (black_rate - 0.5),
        "paired_cluster_bootstrap_95_ci": bootstrap,
        "conservative_paired_hoeffding_95_ci": _hoeffding_interval(pair_values) if pair_values else None,
        "technical_games": len(technical),
        "technical_by_reason": technical_by_reason,
        "execution_games": executed,
        "games_per_hour": executed * 3600.0 / wall if wall else 0.0,
        "moves_per_second": moves / wall if wall else 0.0,
        "mean_inference_batch": batch_weight / batch_calls if batch_calls else 0.0,
        "actual_peak_contexts": max(peaks, default=0),
        "actual_steady_contexts": [dict(value) for value in steady],
        "execution_wall_time_sec": wall,
        "pair_values": pair_values,
    }


def _comparison_statistics(arm_stats: Mapping[float, Mapping[str, object]], arm_rows: Mapping[float, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for left, right in ((0.5, 1.5), (1.5, 2.5), (0.5, 2.5)):
        left_pairs = {str(row["pair_id"]): row for row in arm_rows[left]}
        right_pairs = {str(row["pair_id"]): row for row in arm_rows[right]}
        pair_ids = sorted(set(left_pairs) & set(right_pairs))
        left_values = {pair: left_pairs[pair] for pair in pair_ids}
        right_values = {pair: right_pairs[pair] for pair in pair_ids}
        grouped_left: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        grouped_right: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for pair, row in left_values.items():
            grouped_left[pair].append(row)
        for pair, row in right_values.items():
            grouped_right[pair].append(row)
        deltas = []
        for pair in pair_ids:
            # arm_rows are full game rows; take both color-swapped games in
            # their persisted order and compare Black point rate per pair.
            lrows = [row for row in arm_rows[left] if str(row["pair_id"]) == pair]
            rrows = [row for row in arm_rows[right] if str(row["pair_id"]) == pair]
            if len(lrows) != 2 or len(rrows) != 2:
                raise ValueError(f"Paired comparison has incomplete pair {pair}")
            deltas.append(sum(_point(row) for row in rrows) / 2.0 - sum(_point(row) for row in lrows) / 2.0)
        mean = statistics.fmean(deltas)
        rng = random.Random(derive_seed(MASTER_SEED, left, right, "paired-difference"))
        n = len(deltas)
        boot = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
        output[f"{left:g}_vs_{right:g}"] = {
            "left_komi": left,
            "right_komi": right,
            "pairs": n,
            "black_point_rate_change_right_minus_left": mean,
            "paired_cluster_bootstrap_95_ci": [boot[int(0.025 * BOOTSTRAP_RESAMPLES)], boot[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]],
        }
    return output


def _intersection_estimate(arm_stats: Mapping[float, Mapping[str, object]]) -> dict[str, object]:
    points = [(komi, float(arm_stats[komi]["black_point_rate"])) for komi in KOMI_ARMS]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if (y0 - 0.5) * (y1 - 0.5) <= 0.0 and y1 != y0:
            return {"estimate": x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0), "method": "bracketed-linear-interpolation", "tested_points": points}
    x_mean = statistics.fmean(x for x, _ in points)
    y_mean = statistics.fmean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator if denominator else 0.0
    estimate = None if abs(slope) < 1e-12 else x_mean + (0.5 - y_mean) / slope
    return {"estimate": estimate, "method": "three-point-linear-regression", "tested_points": points, "warning": "auxiliary estimate; 1024 games per arm limit precision"}


def _render_report(summary: Mapping[str, object]) -> str:
    arms = summary["arms"]
    lines = [
        "# Torus9 M137 5CH komi calibration",
        "",
        "Arena-only evaluation. Training was not started.",
        "",
        "| Komi | B | W | D | Black % | Δ from 50% | 95% paired CI | Technical |",
        "|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for komi in KOMI_ARMS:
        row = arms[f"{komi:g}"]
        ci = row["paired_cluster_bootstrap_95_ci"]
        ci_text = "n/a" if ci is None else f"[{100*float(ci[0]):.2f}%, {100*float(ci[1]):.2f}%]"
        lines.append(
            f"| {komi:g} | {row['black_wins']} | {row['white_wins']} | {row['draws']} | "
            f"{float(row['black_percent']):.2f}% | {float(row['black_delta_from_50_percentage_points']):+.2f} pp | {ci_text} | {row['technical_games']} |"
        )
    selected = summary["most_balanced_tested_komi"]
    estimate = summary["estimated_50_50_intersection"].get("estimate")
    lines.extend(
        [
            "",
            f"Most balanced tested komi: **{float(selected):g}**.",
            f"Auxiliary estimated 50/50 intersection: **{('approximately ' + format(float(estimate), '.2f')) if estimate is not None else 'not estimable'}**, subject to statistical uncertainty.",
            "",
            f"Checkpoint: `{summary['checkpoint']['path']}`",
            f"Checkpoint SHA-256: `{summary['checkpoint']['sha256']}`",
            f"Architecture: `{M137_FIVE_CHANNEL_ARCHITECTURE_ID}`, input_channels=5, komi observation channel absent.",
            f"Search: {SIMULATIONS} MCTS simulations; master seed `{MASTER_SEED}`.",
            f"Frozen startset fingerprint: `{summary['startset_fingerprint']}`",
            f"Total valid: `{summary['total_valid_games']}/3072`; technical: `{summary['total_technical_games']}`.",
            f"Evaluation path: `{summary['evaluation_path']}`",
            f"Arena git commit: `{summary['git_commit']}`",
            f"Retries/fixes: `{summary['technical_events_count']}` technical events; code fixes were completed before Arena.",
            "",
            "The raw three-arm results are primary evidence; the intersection estimate is only auxiliary.",
        ]
    )
    return "\n".join(lines) + "\n"


def _arm_progress(arm: Mapping[str, object], *, root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows_path = root / str(arm.get("committed_games_path", "")) if arm.get("committed_games_path") else None
    if rows_path is None or not rows_path.is_file():
        return [], []
    rows = _read_jsonl(rows_path)
    telemetry: list[dict[str, object]] = []
    for attempt in arm.get("attempts", []):
        if isinstance(attempt, Mapping) and attempt.get("summary_path"):
            summary_path = Path(str(attempt["summary_path"]))
            if summary_path.is_file():
                payload = _read_json(summary_path)
                telemetry_payload = payload.get("telemetry")
                if isinstance(telemetry_payload, Mapping):
                    telemetry.append({**payload, **dict(telemetry_payload)})
    return rows, telemetry


def run(checkpoint: Path, *, runs_root: Path) -> dict[str, object]:
    mark_v2_process()
    evaluation_root = (runs_root / "torus9" / "evaluations" / EVALUATION_ID).resolve()
    evaluation_root.mkdir(parents=True, exist_ok=True)
    config_payload = _config_payload()
    config_fp = sha256_fingerprint(config_payload)
    startset_path, startset_fp, pair_ids = _ensure_frozen_startset(evaluation_root)
    schedule_path, schedule_fp = _ensure_schedule(evaluation_root, startset_fingerprint=startset_fp, pair_ids=pair_ids)
    checkpoint_node, checkpoint_ref = _checkpoint_reference(checkpoint.resolve(), evaluation_root)
    checkpoint_reference_path = evaluation_root / "checkpoint-reference.json"
    if checkpoint_reference_path.is_file():
        existing_checkpoint = _read_json(checkpoint_reference_path)
        if existing_checkpoint.get("sha256") != checkpoint_ref["sha256"]:
            raise RuntimeError("Existing calibration evaluation is bound to another checkpoint")
    else:
        _write_json(checkpoint_reference_path, checkpoint_ref)
    config_path = evaluation_root / "config.json"
    if config_path.is_file() and _read_json(config_path).get("config_fingerprint") != config_fp:
        raise RuntimeError("Existing calibration evaluation has a different config fingerprint")
    _write_json(config_path, {**config_payload, "config_fingerprint": config_fp})

    progress_path = evaluation_root / "progress.json"
    if progress_path.is_file():
        progress = _read_json(progress_path)
        if progress.get("startset_fingerprint") != startset_fp or progress.get("config_fingerprint") != config_fp or progress.get("checkpoint_sha256") != checkpoint_ref["sha256"]:
            raise RuntimeError("Calibration progress identity does not match frozen evaluation")
    else:
        progress = {
            "schema": "torus9-komi-calibration-progress-v1",
            "evaluation_id": EVALUATION_ID,
            "status": "PREPARED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "master_seed": MASTER_SEED,
            "startset_fingerprint": startset_fp,
            "schedule_fingerprint": schedule_fp,
            "config_fingerprint": config_fp,
            "checkpoint_sha256": checkpoint_ref["sha256"],
            "checkpoint_model_hash": checkpoint_ref["model_hash"],
            "arms": {f"{komi:g}": {"status": "PENDING", "valid_games": 0, "technical_games": 0, "attempts": []} for komi in KOMI_ARMS},
            "technical_events": [],
        }
        _write_json(progress_path, progress)

    runner = _fake_arena_runner()
    startset_ref = StartsetRef(
        id="torus9-m137-5ch-calibration-startset-v1",
        artifact=ArtifactRef("frozen-startset.json", startset_fp),
        fingerprint=startset_fp,
    )
    arm_rows: dict[float, list[dict[str, object]]] = {}
    arm_telemetry: dict[float, list[dict[str, object]]] = {}
    pair_index_by_id = {pair_id: index for index, pair_id in enumerate(pair_ids)}

    for komi in KOMI_ARMS:
        key = f"{komi:g}"
        arm_state = progress["arms"][key]
        if not isinstance(arm_state, dict):
            raise RuntimeError(f"Malformed progress arm {key}")
        committed_path = evaluation_root / "arms" / f"komi-{key}" / "games.jsonl"
        if committed_path.is_file():
            rows = _read_jsonl(committed_path)
            if len(rows) == GAMES_PER_ARM and all(row.get("technical_termination") is None for row in rows):
                arm_rows[komi], arm_telemetry[komi] = _arm_progress(arm_state, root=evaluation_root)
                if len(arm_rows[komi]) != GAMES_PER_ARM:
                    arm_rows[komi] = rows
                continue

        pending_pair_indices = list(range(PAIRS))
        if committed_path.is_file():
            existing_rows = _read_jsonl(committed_path)
            completed_pairs = {
                str(row["pair_id"])
                for row in existing_rows
                if row.get("technical_termination") is None
            }
            pending_pair_indices = [index for index, pair_id in enumerate(pair_ids) if pair_id not in completed_pairs]
        if len(pending_pair_indices) == 0:
            raise RuntimeError(f"Komi {key} has no committed valid games but no pending pairs")

        all_rows: list[dict[str, object]] = []
        telemetry: list[dict[str, object]] = []
        attempts = arm_state.get("attempts")
        if not isinstance(attempts, list):
            attempts = []
        retry_count_by_pair: dict[int, int] = defaultdict(int)
        while pending_pair_indices:
            attempt_no = len(attempts) + 1
            config = _execution_config(len(pending_pair_indices) * 2)
            profile = f"torus9-komi-calibration|{key}|simulations={SIMULATIONS}|5ch"
            workload = {
                "startset_path": str(startset_path),
                "startset_fingerprint": startset_fp,
                "pair_indices": pending_pair_indices,
                "attempt": attempt_no,
            }
            request = ArenaRunRequest(
                candidate=checkpoint_node,
                reference=checkpoint_node,
                master_seed=MASTER_SEED,
                startset=startset_ref,
                config=config,
                profile=profile,
                workload=workload,
                scientific_contract=_scientific_contract(komi, config.games),
                execution_contract=_execution_contract(config),
                candidate_label="M137-5CH",
                reference_label="M137-5CH",
                comparison=f"M137-5CH-self-komi-{key}",
                output_dir=evaluation_root / "arena" / f"komi-{key}" / f"attempt-{attempt_no:03d}",
            )
            progress["status"] = "RUNNING"
            progress["running_arm"] = key
            progress["running_attempt"] = attempt_no
            _write_json(progress_path, progress)
            result = runner.run(request)
            result_rows = _read_jsonl(result.output_dir / "games.jsonl")
            result_summary = _read_json(result.output_dir / "summary.json")
            attempt_record = {
                "attempt": attempt_no,
                "output_dir": str(result.output_dir),
                "summary_path": str(result.output_dir / "summary.json"),
                "requested_games": config.games,
                "returned_games": len(result_rows),
                "technical_games": sum(row.get("technical_termination") is not None for row in result_rows),
                "pair_indices": list(pending_pair_indices),
            }
            attempts.append(attempt_record)
            telemetry.append({**result_summary, **dict(result_summary.get("telemetry", {}))})
            technical_pairs = sorted({str(row["pair_id"]) for row in result_rows if row.get("technical_termination") is not None})
            technical_reasons = sorted({str(row.get("technical_termination")) for row in result_rows if row.get("technical_termination") is not None})
            if not technical_pairs:
                all_rows.extend(result_rows)
                pending_pair_indices = []
                break
            event = {
                "komi": komi,
                "attempt": attempt_no,
                "pair_ids": technical_pairs,
                "reasons": technical_reasons,
                "diagnostics": {pair: [row.get("error") for row in result_rows if str(row["pair_id"]) == pair and row.get("technical_termination") is not None] for pair in technical_pairs},
            }
            progress.setdefault("technical_events", []).append(event)
            for pair in technical_pairs:
                index = pair_index_by_id[pair]
                retry_count_by_pair[index] += 1
                if retry_count_by_pair[index] > MAX_TECHNICAL_RETRIES:
                    raise RuntimeError(f"Komi {key} technical pair {pair} remained invalid after bounded retries: {event}")
            all_rows.extend(row for row in result_rows if row.get("technical_termination") is None)
            pending_pair_indices = [pair_index_by_id[pair] for pair in technical_pairs]
            progress["arms"][key] = {"status": "RETRYING_TECHNICAL", "valid_games": len(all_rows), "technical_games": len(technical_pairs) * 2, "attempts": attempts}
            _write_json(progress_path, progress)

        # Replace a pair atomically after its rerun is valid, retaining only
        # one final row per scheduled game ID.
        by_game = {str(row["game_id"]): row for row in all_rows}
        final_rows = [by_game[ f"{pair_id}--{suffix}"] for pair_id in pair_ids for suffix in ("g1", "g2") if f"{pair_id}--{suffix}" in by_game]
        if len(final_rows) != GAMES_PER_ARM or any(row.get("technical_termination") is not None for row in final_rows):
            raise RuntimeError(f"Komi {key} did not commit exactly 1024 valid games")
        _write_jsonl(committed_path, final_rows)
        arm_state.update({"status": "VALID", "valid_games": GAMES_PER_ARM, "technical_games": 0, "committed_games_path": str(committed_path.relative_to(evaluation_root)), "attempts": attempts})
        progress["arms"][key] = arm_state
        progress["status"] = "RUNNING"
        progress.pop("running_arm", None)
        progress.pop("running_attempt", None)
        _write_json(progress_path, progress)
        arm_rows[komi] = final_rows
        arm_telemetry[komi] = telemetry

    stats = {komi: _arm_statistics(komi, arm_rows[komi], arm_telemetry[komi]) for komi in KOMI_ARMS}
    selected = min(KOMI_ARMS, key=lambda komi: abs(float(stats[komi]["black_point_rate"]) - 0.5))
    summary: dict[str, object] = {
        "schema": "torus9-m137-5ch-komi-calibration-summary-v1",
        "evaluation_id": EVALUATION_ID,
        "evaluation_path": str(evaluation_root),
        "git_commit": __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True).stdout.strip(),
        "checkpoint": checkpoint_ref,
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "input_channels": 5,
        "komi_observation_channel": False,
        "simulations": SIMULATIONS,
        "master_seed": MASTER_SEED,
        "startset_fingerprint": startset_fp,
        "schedule_fingerprint": schedule_fp,
        "config_fingerprint": config_fp,
        "total_valid_games": sum(int(stats[komi]["valid_games"]) for komi in KOMI_ARMS),
        "total_technical_games": sum(int(stats[komi]["technical_games"]) for komi in KOMI_ARMS),
        "technical_events_count": len(progress.get("technical_events", [])),
        "arms": {f"{komi:g}": stats[komi] for komi in KOMI_ARMS},
        "paired_differences": _comparison_statistics(stats, arm_rows),
        "most_balanced_tested_komi": selected,
        "estimated_50_50_intersection": _intersection_estimate(stats),
        "training_started": False,
    }
    _write_json(evaluation_root / "summary.json", summary)
    (evaluation_root / "report.md").write_text(_render_report(summary), encoding="utf-8")
    progress["status"] = "COMPLETE"
    progress["completed_at"] = datetime.now(timezone.utc).isoformat()
    progress["summary_path"] = str(evaluation_root / "summary.json")
    _write_json(progress_path, progress)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    summary = run(args.checkpoint, runs_root=args.runs_root)
    print(json.dumps({"evaluation_id": summary["evaluation_id"], "status": "COMPLETE", "valid": summary["total_valid_games"], "technical": summary["total_technical_games"], "path": summary["evaluation_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
