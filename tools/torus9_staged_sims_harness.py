#!/usr/bin/env python3
"""Composition entrypoint for the config-driven staged Torus9 experiment harness.

The reusable experiment implementation lives in ``torus9_staged_sims_harness_impl``.
This entrypoint owns production-only dependency wiring, including the explicit
per-orchestrator code-update/provenance lifecycle policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.code_update_policy import CodeUpdateProvenancePolicy
from gocube_golden.run_spec import StrictProductionTrainingOrchestrator
from tools import torus9_staged_sims_harness_impl as _impl
from tools.torus9_staged_sims_harness_impl import *  # noqa: F401,F403


def _run_arm(
    experiment_id: str,
    spec: Mapping[str, object],
    arm: Arm,
    parent: Mapping[str, object],
) -> dict[str, object]:
    lineage_id = f"{experiment_id}-{arm.arm_id}"
    run = StrictProductionTrainingOrchestrator(
        repo_root=ROOT,
        run_spec=build_arm_run_spec(spec, arm),
        lineage_id=lineage_id,
        terminal=True,
        child_lifecycle_policy=CodeUpdateProvenancePolicy(),
    )
    target_generation = int(parent["generation"]) + arm.iterations
    if run.paths.root.exists():
        manifest = _impl._read_json(run.paths.manifest)
        if not _impl._same_parent(manifest.get("parent_checkpoint"), parent):
            raise ValueError(f"Existing arm {lineage_id} has a different parent")
        status = run.status()
        committed = int(status.get("last_committed_generation", 0))
        state = str(status.get("state"))
        if committed < target_generation:
            if state in {"SOFT_STOPPED", "RECOVERY_REQUIRED", "COMPLETED"}:
                run.prepare_resume()
            elif state != "CREATED":
                raise RuntimeError(f"Arm {lineage_id} is not safely resumable")
            run.run(max_generations=target_generation)
    else:
        run.create(parent_checkpoint=parent)
        run.run(max_generations=target_generation)
    status = run.status()
    if int(status.get("last_committed_generation", 0)) != target_generation:
        raise RuntimeError(f"Arm {lineage_id} did not reach M{target_generation}")
    manifest = _impl._read_json(run.paths.manifest)
    relative = f"checkpoints/M{target_generation}.pt"
    hashes = manifest.get("checkpoint_hashes")
    if not isinstance(hashes, Mapping) or relative not in hashes:
        raise ValueError(f"Final checkpoint identity is missing for {lineage_id}")
    return {
        "arm_id": arm.arm_id,
        "games_per_iteration": arm.games,
        "lineage_id": lineage_id,
        "generation": target_generation,
        "checkpoint": str(run.paths.root / relative),
        "sha256": str(hashes[relative]),
        "budget": {
            "games": arm.total_games,
            "optimizer_steps": arm.total_optimizer_steps,
            "sample_exposures": arm.total_sample_exposures,
        },
    }


def run_experiment(
    *,
    experiment_id: str,
    start_lineage: str,
    start_generation: int,
    experiment_spec: str | Path = DEFAULT_EXPERIMENT_SPEC,
) -> dict[str, object]:
    spec = load_experiment_spec(experiment_spec)
    arms = arms_from_spec(spec)
    evaluations = evaluations_from_spec(spec, set(arms))
    install_operator_policy()
    parent = _impl._source_parent_reference(start_lineage, start_generation)
    arm_results: dict[str, dict[str, object]] = {}
    evaluation_results: dict[str, dict[str, object]] = {}
    stage_results: list[dict[str, object]] = []

    for raw_stage in spec["stages"]:  # type: ignore[index]
        stage = _impl._mapping(raw_stage, "stage")
        run_if = stage.get("run_if")
        if isinstance(run_if, Mapping) and not condition_met(run_if, evaluation_results):
            stage_results.append({"id": stage["id"], "status": "SKIPPED"})
            continue
        for arm_id in stage["arms"]:  # type: ignore[index]
            arm_results[str(arm_id)] = _run_arm(
                experiment_id, spec, arms[str(arm_id)], parent
            )
        for evaluation_id in stage["evaluations"]:  # type: ignore[index]
            selected = evaluations[str(evaluation_id)]
            evaluation_results[str(evaluation_id)] = _impl._compare(
                spec, selected, arm_results
            )
        stage_results.append({"id": stage["id"], "status": "COMPLETED"})

    return {
        "schema": "gocube-torus9-experiment-harness-result-v1",
        "experiment_id": experiment_id,
        "experiment_kind": spec["kind"],
        "experiment_spec": {"path": spec["_path"], "sha256": spec["_sha256"]},
        "shared_parent": parent,
        "fixed_training": spec["fixed_training"],
        "budget": spec["budget"],
        "stages": stage_results,
        "arms": {arm_id: arm_results.get(arm_id) for arm_id in arms},
        "evaluations": {
            evaluation_id: evaluation_results.get(evaluation_id)
            for evaluation_id in evaluations
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-spec", default=DEFAULT_EXPERIMENT_SPEC)
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
        experiment_spec=args.experiment_spec,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def __getattr__(name: str) -> object:
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(main())
