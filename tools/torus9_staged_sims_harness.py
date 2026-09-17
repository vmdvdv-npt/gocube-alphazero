#!/usr/bin/env python3
"""Staged equal-budget Torus9 cadence experiment on one shared parent state.

Stage 1:
    6 x 64 games/iteration vs 3 x 128 games/iteration.
    Both arms receive 384 self-play games, 480 Adam steps and 30,720 sample
    exposures, starting from the exact same checkpoint + optimizer + replay.

Stage 2:
    Only if the 128 cadence wins a 128-game paired Arena (W > L), run
    2 x 192 games/iteration with the same total budget and compare 192 vs 128
    in another 128-game Arena.

All training arms keep the current plateau-exit settings: 128 self-play sims,
LR 3e-4, replay 6 generations / 40k positions, 80x8 network and the existing
16x4 / 64-context / cap64 / wait1ms execution preset.  Arena search semantics
remain the current Torus9 Arena semantics (64 sims); only Arena game count is
raised to 128.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.artifact_catalog import ARTIFACT_VALIDATION_SCHEMA
from gocube_golden.operator_policy import install_operator_policy
from gocube_golden.provenance import file_sha256
from gocube_golden.run_spec import (
    StrictProductionTrainingOrchestrator,
    StrictRunSpec,
)
from gocube_golden.run_storage import (
    evaluation_dir,
    evaluation_id_for_comparison,
    resolve_checkpoint,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_BLOCKS,
    TORUS9_CURRENT_HIDDEN,
)
from tools.arena import run_arena
from tools.arena_engine import ArenaExecutionConfig


ARENA_GAMES = 128
ARENA_SEED = 202609131004
TOTAL_GAMES = 384
TOTAL_OPTIMIZER_STEPS = 480
TOTAL_SAMPLE_EXPOSURES = 30_720
BATCH_SIZE = 64


@dataclass(frozen=True)
class Arm:
    games: int
    iterations: int
    optimizer_steps: int
    spec_path: str

    @property
    def sample_exposures_per_iteration(self) -> int:
        return self.optimizer_steps * BATCH_SIZE

    @property
    def total_games(self) -> int:
        return self.games * self.iterations

    @property
    def total_optimizer_steps(self) -> int:
        return self.optimizer_steps * self.iterations

    @property
    def total_sample_exposures(self) -> int:
        return self.sample_exposures_per_iteration * self.iterations


ARM_64 = Arm(64, 6, 80, "configs/gocube/torus9_staged_cadence_64_v1.json")
ARM_128 = Arm(128, 3, 160, "configs/gocube/torus9_staged_cadence_128_v1.json")
ARM_192 = Arm(192, 2, 240, "configs/gocube/torus9_staged_cadence_192_v1.json")
ARMS = (ARM_64, ARM_128, ARM_192)


def validate_equal_budget() -> None:
    for arm in ARMS:
        if arm.total_games != TOTAL_GAMES:
            raise ValueError(f"Arm {arm.games} game budget drift: {arm.total_games}")
        if arm.total_optimizer_steps != TOTAL_OPTIMIZER_STEPS:
            raise ValueError(
                f"Arm {arm.games} optimizer budget drift: {arm.total_optimizer_steps}"
            )
        if arm.total_sample_exposures != TOTAL_SAMPLE_EXPOSURES:
            raise ValueError(
                f"Arm {arm.games} sample-exposure budget drift: {arm.total_sample_exposures}"
            )


def should_run_192(summary: Mapping[str, object]) -> bool:
    wld = summary.get("W/L/D")
    if not isinstance(wld, list) or len(wld) != 3:
        raise ValueError("Arena summary is missing candidate W/L/D")
    wins, losses, _draws = (int(value) for value in wld)
    return wins > losses


def arena_config() -> ArenaExecutionConfig:
    return ArenaExecutionConfig(
        games=ARENA_GAMES,
        workers=16,
        games_per_worker=4,
        inference_batch_rows=64,
        inference_batch_wait_ms=1.0,
        device="cuda",
        strict_production=True,
        min_mean_inference_batch_rows=0.0,
        min_effective_cpu_cores=0.0,
        early_gate_enabled=False,
        early_gate_min_forwards=128,
        early_gate_min_wall_sec=5.0,
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _source_parent_reference(lineage_id: str, generation: int) -> dict[str, object]:
    resolved = resolve_checkpoint(
        {
            "topology": "torus9",
            "lineage_id": lineage_id,
            "generation": int(generation),
        },
        topology="torus9",
    )
    checkpoint = resolved.path
    metadata_path = checkpoint.with_suffix(".metadata.json")
    replay_path = checkpoint.parents[1] / "replay" / f"rolling-after-{generation:02d}.jsonl"
    if not metadata_path.is_file() or not replay_path.is_file():
        raise FileNotFoundError(
            "Shared parent requires checkpoint metadata and rolling replay: "
            f"{metadata_path}, {replay_path}"
        )
    metadata = _read_json(metadata_path)
    if metadata.get("architecture_id") != TORUS9_CURRENT_ARCHITECTURE_ID:
        raise ValueError("Shared parent is not the current Torus9 80x8 architecture")
    architecture = metadata.get("architecture_config")
    if isinstance(architecture, Mapping):
        hidden = architecture.get("hidden")
        blocks = architecture.get("blocks")
        if hidden is not None and int(hidden) != TORUS9_CURRENT_HIDDEN:
            raise ValueError("Shared parent hidden width is not 80")
        if blocks is not None and int(blocks) != TORUS9_CURRENT_BLOCKS:
            raise ValueError("Shared parent residual block count is not 8")

    total_evictions = 0
    summary_path = checkpoint.parents[1] / f"iter-{generation:02d}-summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        replay = summary.get("replay")
        if isinstance(replay, Mapping):
            total_evictions = int(replay.get("total_evictions", 0))

    replay_rows = metadata.get(
        "replay_row_count", metadata.get("valid_replay_positions")
    )
    return {
        "topology": "torus9",
        "lineage_id": resolved.lineage_id,
        "checkpoint_id": resolved.checkpoint_id,
        "label": str(metadata.get("checkpoint_label") or f"M{generation}"),
        "generation": int(generation),
        "path": str(checkpoint),
        "sha256": resolved.sha256,
        "artifact_sha256": resolved.sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "model_hash": metadata.get("model_hash"),
        "replay_path": str(replay_path),
        "replay_sha256": file_sha256(replay_path),
        "replay_row_count": int(replay_rows) if replay_rows is not None else None,
        "replay_fingerprint": metadata.get("replay_fingerprint"),
        "replay_validation_schema": ARTIFACT_VALIDATION_SCHEMA,
        "total_evictions": total_evictions,
    }


def _lineage_id(experiment_id: str, arm: Arm) -> str:
    return f"{experiment_id}-g{arm.games}"


def _same_parent(actual: object, expected: Mapping[str, object]) -> bool:
    if not isinstance(actual, Mapping):
        return False
    return (
        str(actual.get("lineage_id")) == str(expected.get("lineage_id"))
        and int(actual.get("generation", -1)) == int(expected.get("generation", -2))
        and str(actual.get("artifact_sha256") or actual.get("sha256"))
        == str(expected.get("artifact_sha256") or expected.get("sha256"))
        and str(actual.get("metadata_sha256")) == str(expected.get("metadata_sha256"))
        and str(actual.get("replay_sha256")) == str(expected.get("replay_sha256"))
    )


def _run_arm(
    *, experiment_id: str, arm: Arm, parent: Mapping[str, object]
) -> dict[str, object]:
    lineage_id = _lineage_id(experiment_id, arm)
    spec = StrictRunSpec.load(ROOT / arm.spec_path, repo_root=ROOT)
    run = StrictProductionTrainingOrchestrator(
        repo_root=ROOT,
        run_spec=spec,
        lineage_id=lineage_id,
        terminal=True,
    )
    target_generation = int(parent["generation"]) + arm.iterations

    if run.paths.root.exists():
        manifest = _read_json(run.paths.manifest)
        if not _same_parent(manifest.get("parent_checkpoint"), parent):
            raise ValueError(
                f"Existing arm {lineage_id} does not reference the requested shared parent"
            )
        status = run.status()
        committed = int(status.get("last_committed_generation", 0))
        state = str(status.get("state"))
        if committed < target_generation:
            if state in {"SOFT_STOPPED", "RECOVERY_REQUIRED", "COMPLETED"}:
                run.prepare_resume()
            elif state not in {"CREATED"}:
                raise RuntimeError(
                    f"Arm {lineage_id} is not safely resumable from state {state}"
                )
            run.run(max_generations=target_generation)
    else:
        run.create(parent_checkpoint=parent)
        run.run(max_generations=target_generation)

    status = run.status()
    committed = int(status.get("last_committed_generation", 0))
    if committed != target_generation:
        raise RuntimeError(
            f"Arm {lineage_id} stopped at M{committed}, expected M{target_generation}"
        )
    manifest = _read_json(run.paths.manifest)
    relative = f"checkpoints/M{target_generation}.pt"
    checkpoint_hashes = manifest.get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, Mapping) or relative not in checkpoint_hashes:
        raise ValueError(f"Final checkpoint identity is missing for {lineage_id}")
    digest = str(checkpoint_hashes[relative])
    return {
        "arm": arm.games,
        "lineage_id": lineage_id,
        "generation": target_generation,
        "checkpoint": str(run.paths.root / relative),
        "sha256": digest,
        "budget": {
            "games": arm.total_games,
            "optimizer_steps": arm.total_optimizer_steps,
            "sample_exposures": arm.total_sample_exposures,
        },
    }


def _checkpoint_reference(arm_result: Mapping[str, object]) -> dict[str, object]:
    return {
        "topology": "torus9",
        "lineage_id": str(arm_result["lineage_id"]),
        "generation": int(arm_result["generation"]),
        "checkpoint_id": f"M{int(arm_result['generation'])}",
        "sha256": str(arm_result["sha256"]),
        "artifact_sha256": str(arm_result["sha256"]),
    }


def _existing_arena(
    output: Path,
    *,
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, object] | None:
    if not output.exists():
        return None
    summary_path = output / "summary.json"
    provenance_path = output / "provenance.json"
    if not summary_path.is_file() or not provenance_path.is_file():
        raise RuntimeError(
            f"Evaluation directory exists but is incomplete; refusing overwrite: {output}"
        )
    summary = _read_json(summary_path)
    provenance = _read_json(provenance_path)
    candidate_actual = provenance.get("candidate")
    reference_actual = provenance.get("reference")
    if not isinstance(candidate_actual, Mapping) or not isinstance(reference_actual, Mapping):
        raise ValueError(f"Existing Arena provenance is malformed: {output}")
    if (
        str(candidate_actual.get("artifact_sha256") or candidate_actual.get("sha256"))
        != str(candidate["artifact_sha256"])
        or str(reference_actual.get("artifact_sha256") or reference_actual.get("sha256"))
        != str(reference["artifact_sha256"])
        or int(summary.get("games", -1)) != ARENA_GAMES
    ):
        raise ValueError(f"Existing Arena identities/config do not match staged harness: {output}")
    telemetry = summary.get("telemetry")
    if not isinstance(telemetry, Mapping) or int(telemetry.get("technical_games", 0)) != 0:
        raise ValueError(f"Existing Arena has technical outcomes: {output}")
    return summary


def _compare(
    *,
    candidate_result: Mapping[str, object],
    reference_result: Mapping[str, object],
) -> dict[str, object]:
    candidate = _checkpoint_reference(candidate_result)
    reference = _checkpoint_reference(reference_result)
    evaluation_id = evaluation_id_for_comparison(
        candidate_lineage_id=str(candidate["lineage_id"]),
        candidate_generation=int(candidate["generation"]),
        reference_lineage_id=str(reference["lineage_id"]),
        reference_generation=int(reference["generation"]),
    )
    output = evaluation_dir("torus9", evaluation_id)
    existing = _existing_arena(
        output,
        candidate=candidate,
        reference=reference,
    )
    if existing is not None:
        return existing
    return run_arena(
        candidate_path=candidate,
        reference_path=reference,
        profile_name="torus9",
        output_dir=output,
        candidate_label=f"g{candidate_result['arm']}",
        reference_label=f"g{reference_result['arm']}",
        run_id=evaluation_id,
        comparison=(
            f"staged cadence {candidate_result['arm']} vs {reference_result['arm']}"
        ),
        master_seed=ARENA_SEED,
        config=arena_config(),
    )


def run_experiment(
    *, experiment_id: str, start_lineage: str, start_generation: int
) -> dict[str, object]:
    validate_equal_budget()
    install_operator_policy()
    parent = _source_parent_reference(start_lineage, start_generation)

    arm64 = _run_arm(experiment_id=experiment_id, arm=ARM_64, parent=parent)
    arm128 = _run_arm(experiment_id=experiment_id, arm=ARM_128, parent=parent)
    arena_128_vs_64 = _compare(
        candidate_result=arm128,
        reference_result=arm64,
    )
    run_192 = should_run_192(arena_128_vs_64)

    arm192: dict[str, object] | None = None
    arena_192_vs_128: dict[str, object] | None = None
    if run_192:
        arm192 = _run_arm(experiment_id=experiment_id, arm=ARM_192, parent=parent)
        arena_192_vs_128 = _compare(
            candidate_result=arm192,
            reference_result=arm128,
        )

    return {
        "schema": "torus9-staged-cadence-harness-v1",
        "experiment_id": experiment_id,
        "shared_parent": parent,
        "fixed_training": {
            "self_play_mcts_simulations": 128,
            "learning_rate": 0.0003,
            "replay_generations": 6,
            "replay_cap": 40000,
            "network": "80x8",
            "execution": "16x4 / 64 contexts / cap64 / wait1ms",
        },
        "equal_budget": {
            "games": TOTAL_GAMES,
            "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
            "sample_exposures": TOTAL_SAMPLE_EXPOSURES,
        },
        "arena": {
            "games": ARENA_GAMES,
            "search_simulations": 64,
            "first": arena_128_vs_64,
            "run_192": run_192,
            "second": arena_192_vs_128,
        },
        "arms": {
            "64": arm64,
            "128": arm128,
            "192": arm192,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--start-lineage", required=True)
    parser.add_argument("--start-generation", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_generation <= 0:
        raise SystemExit("--start-generation must be positive")
    result = run_experiment(
        experiment_id=args.experiment_id,
        start_lineage=args.start_lineage,
        start_generation=args.start_generation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
