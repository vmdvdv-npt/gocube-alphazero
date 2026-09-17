#!/usr/bin/env python3
"""Config-driven staged Torus9 experiment harness.

Experiment-specific arms, budgets, Arena sizes/seeds and stage conditions live
in a committed JSON spec. This module keeps the execution/storage/provenance
mechanics reusable across experiments.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.artifact_catalog import ARTIFACT_VALIDATION_SCHEMA
from gocube_golden.operator_policy import install_operator_policy
from gocube_golden.provenance import file_sha256
from gocube_golden.run_spec import StrictProductionTrainingOrchestrator, StrictRunSpec
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
from tools.arena_profiles.torus9 import PROFILE as TORUS9_ARENA_PROFILE


EXPERIMENT_SPEC_SCHEMA = "gocube-torus9-experiment-v1"
DEFAULT_EXPERIMENT_SPEC = "configs/gocube/torus9_staged_cadence_experiment_v1.json"
EVALUATION_IDENTITY_SCHEMA = "gocube-arena-evaluation-identity-v1"
EVALUATION_IDENTITY_RECORD_SCHEMA = "gocube-arena-evaluation-identity-record-v1"
EVALUATION_IDENTITY_FILENAME = "evaluation-identity.json"
EVALUATION_IDENTITY_HASH_PREFIX = 12


@dataclass(frozen=True)
class Arm:
    arm_id: str
    games: int
    iterations: int
    optimizer_steps: int
    batch_size: int

    @property
    def total_games(self) -> int:
        return self.games * self.iterations

    @property
    def total_optimizer_steps(self) -> int:
        return self.optimizer_steps * self.iterations

    @property
    def total_sample_exposures(self) -> int:
        return self.total_optimizer_steps * self.batch_size


@dataclass(frozen=True)
class Evaluation:
    evaluation_id: str
    candidate: str
    reference: str
    games: int
    master_seed: int


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0 or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be a positive integer")
    return result


def _safe_component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


def _repo_file(value: object, label: str) -> Path:
    relative = Path(str(value).strip())
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a repo-relative file")
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"{label} is not a repository file: {relative}")
    return path


def load_experiment_spec(
    path: str | Path = DEFAULT_EXPERIMENT_SPEC,
) -> dict[str, object]:
    spec_path = _repo_file(path, "experiment spec")
    spec = _read_json(spec_path)
    if spec.get("schema") != EXPERIMENT_SPEC_SCHEMA:
        raise ValueError(f"Unsupported experiment schema: {spec.get('schema')!r}")
    _safe_component(spec.get("kind"), "experiment kind")
    _repo_file(spec.get("profile_path"), "profile_path")
    _repo_file(spec.get("run_spec_template_path"), "run_spec_template_path")
    _positive_int(spec.get("batch_size"), "batch_size")
    for key in (
        "fixed_training",
        "budget",
        "arena_execution",
        "arena_scientific_contract",
    ):
        _mapping(spec.get(key), key)
    if not isinstance(spec.get("arms"), list) or not spec["arms"]:
        raise ValueError("arms must be a non-empty list")
    if not isinstance(spec.get("evaluations"), list) or not spec["evaluations"]:
        raise ValueError("evaluations must be a non-empty list")
    if not isinstance(spec.get("stages"), list) or not spec["stages"]:
        raise ValueError("stages must be a non-empty list")
    spec["_path"] = spec_path.relative_to(ROOT).as_posix()
    spec["_sha256"] = file_sha256(spec_path)
    arms = arms_from_spec(spec)
    evaluations_from_spec(spec, set(arms))
    validate_budget(spec)
    _validate_stages(spec, set(arms))
    return spec


def arms_from_spec(spec: Mapping[str, object]) -> dict[str, Arm]:
    batch_size = _positive_int(spec.get("batch_size"), "batch_size")
    result: dict[str, Arm] = {}
    raw_arms = spec.get("arms")
    if not isinstance(raw_arms, list):
        raise ValueError("arms must be a list")
    for index, value in enumerate(raw_arms):
        raw = _mapping(value, f"arms[{index}]")
        arm_id = _safe_component(raw.get("id"), f"arms[{index}].id")
        if arm_id in result:
            raise ValueError(f"Duplicate arm id: {arm_id}")
        result[arm_id] = Arm(
            arm_id=arm_id,
            games=_positive_int(raw.get("games_per_iteration"), f"{arm_id}.games"),
            iterations=_positive_int(raw.get("iterations"), f"{arm_id}.iterations"),
            optimizer_steps=_positive_int(
                raw.get("optimizer_steps_per_iteration"),
                f"{arm_id}.optimizer_steps_per_iteration",
            ),
            batch_size=batch_size,
        )
    return result


def evaluations_from_spec(
    spec: Mapping[str, object],
    arm_ids: set[str] | None = None,
) -> dict[str, Evaluation]:
    known = arm_ids if arm_ids is not None else set(arms_from_spec(spec))
    result: dict[str, Evaluation] = {}
    raw_evaluations = spec.get("evaluations")
    if not isinstance(raw_evaluations, list):
        raise ValueError("evaluations must be a list")
    for index, value in enumerate(raw_evaluations):
        raw = _mapping(value, f"evaluations[{index}]")
        evaluation_id = _safe_component(raw.get("id"), f"evaluations[{index}].id")
        if evaluation_id in result:
            raise ValueError(f"Duplicate evaluation id: {evaluation_id}")
        candidate = _safe_component(raw.get("candidate"), f"{evaluation_id}.candidate")
        reference = _safe_component(raw.get("reference"), f"{evaluation_id}.reference")
        if candidate not in known or reference not in known:
            raise ValueError(f"{evaluation_id} references an unknown arm")
        games = _positive_int(raw.get("games"), f"{evaluation_id}.games")
        if games % 2:
            raise ValueError(f"{evaluation_id}.games must be even")
        result[evaluation_id] = Evaluation(
            evaluation_id=evaluation_id,
            candidate=candidate,
            reference=reference,
            games=games,
            master_seed=_positive_int(raw.get("master_seed"), f"{evaluation_id}.master_seed"),
        )
    return result


def _validate_stages(spec: Mapping[str, object], arm_ids: set[str]) -> None:
    evaluations = evaluations_from_spec(spec, arm_ids)
    scheduled_arms: set[str] = set()
    scheduled_evaluations: set[str] = set()
    for index, value in enumerate(spec["stages"]):  # type: ignore[index]
        stage = _mapping(value, f"stages[{index}]")
        _safe_component(stage.get("id"), f"stages[{index}].id")
        stage_arms = stage.get("arms")
        stage_evaluations = stage.get("evaluations")
        if not isinstance(stage_arms, list) or any(item not in arm_ids for item in stage_arms):
            raise ValueError("Stage references an unknown arm")
        if not isinstance(stage_evaluations, list) or any(
            item not in evaluations for item in stage_evaluations
        ):
            raise ValueError("Stage references an unknown evaluation")
        if scheduled_arms.intersection(stage_arms):
            raise ValueError("An arm may be scheduled only once")
        if scheduled_evaluations.intersection(stage_evaluations):
            raise ValueError("An evaluation may be scheduled only once")
        run_if = stage.get("run_if")
        if run_if is not None:
            condition = _mapping(run_if, "stage.run_if")
            if condition.get("evaluation") not in scheduled_evaluations:
                raise ValueError("Stage run_if must reference an earlier evaluation")
            if condition.get("metric") not in {
                "wins",
                "losses",
                "draws",
                "wins_minus_losses",
                "win_rate",
            }:
                raise ValueError("Unsupported stage condition metric")
            if condition.get("operator") not in {">", ">=", "<", "<=", "==", "!="}:
                raise ValueError("Unsupported stage condition operator")
            float(condition.get("value"))
        available = scheduled_arms.union(stage_arms)
        for evaluation_id in stage_evaluations:
            evaluation = evaluations[evaluation_id]
            if evaluation.candidate not in available or evaluation.reference not in available:
                raise ValueError("Evaluation is scheduled before its arms")
        scheduled_arms.update(str(item) for item in stage_arms)
        scheduled_evaluations.update(str(item) for item in stage_evaluations)
    if scheduled_arms != arm_ids or scheduled_evaluations != set(evaluations):
        raise ValueError("Every declared arm/evaluation must be scheduled exactly once")


def validate_budget(spec: Mapping[str, object]) -> None:
    budget = _mapping(spec.get("budget"), "budget")
    mode = str(budget.get("mode", "")).strip()
    if mode == "none":
        return
    if mode != "equal":
        raise ValueError(f"Unsupported budget mode: {mode!r}")
    expected = _mapping(budget.get("expected"), "budget.expected")
    targets = (
        _positive_int(expected.get("self_play_games"), "expected self-play games"),
        _positive_int(expected.get("optimizer_steps"), "expected optimizer steps"),
        _positive_int(expected.get("sample_exposures"), "expected sample exposures"),
    )
    for arm in arms_from_spec(spec).values():
        actual = (
            arm.total_games,
            arm.total_optimizer_steps,
            arm.total_sample_exposures,
        )
        if actual != targets:
            raise ValueError(f"Arm {arm.arm_id} budget drift: {actual} != {targets}")


def validate_equal_budget(spec: Mapping[str, object] | None = None) -> None:
    selected = spec or load_experiment_spec()
    if _mapping(selected.get("budget"), "budget").get("mode") != "equal":
        raise ValueError("Experiment does not declare equal-budget validation")
    validate_budget(selected)


def _materialized_arm_payload(
    spec: Mapping[str, object],
    arm: Arm,
) -> dict[str, object]:
    template = deepcopy(
        _read_json(_repo_file(spec.get("run_spec_template_path"), "run_spec template"))
    )
    template["profile_path"] = str(spec["profile_path"])
    generation = _mapping(template.get("generation"), "template.generation")
    driver_config = _mapping(generation.get("driver_config"), "template.driver_config")
    driver_config["games"] = arm.games
    driver_config["optimizer_steps_per_iteration"] = arm.optimizer_steps
    generation["driver_config"] = driver_config
    template["generation"] = generation
    template["experiment"] = {
        "kind": spec["kind"],
        "experiment_spec_path": spec["_path"],
        "experiment_spec_sha256": spec["_sha256"],
        "arm_id": arm.arm_id,
        "games_per_iteration": arm.games,
        "iterations": arm.iterations,
        "optimizer_steps_per_iteration": arm.optimizer_steps,
        "total_self_play_games": arm.total_games,
        "total_optimizer_steps": arm.total_optimizer_steps,
        "total_sample_exposures": arm.total_sample_exposures,
    }
    return template


def build_arm_run_spec(
    spec: Mapping[str, object],
    arm: Arm,
) -> StrictRunSpec:
    payload = _materialized_arm_payload(spec, arm)
    with tempfile.TemporaryDirectory(prefix="gocube-experiment-spec-") as directory:
        path = Path(directory) / "run-spec.json"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        run_spec = StrictRunSpec.load(path, repo_root=ROOT)
    config = run_spec.payload["generation"]["driver_config"]  # type: ignore[index]
    if int(config["games"]) != arm.games or int(
        config["optimizer_steps_per_iteration"]
    ) != arm.optimizer_steps:
        raise ValueError(f"Materialized run spec drift for {arm.arm_id}")
    arena = run_spec.payload["arena"]
    if not isinstance(arena, Mapping) or arena.get("enabled") is not False:
        raise ValueError("Per-arm Arena must remain disabled")
    _validate_fixed_training(spec, run_spec)
    return run_spec


def _validate_fixed_training(
    spec: Mapping[str, object],
    run_spec: StrictRunSpec,
) -> None:
    profile = run_spec.orchestrator_spec.profile_payload
    expected = _mapping(spec.get("fixed_training"), "fixed_training")
    observed = {
        "self_play_mcts_simulations": int(profile["self_play"]["mcts_simulations"]),  # type: ignore[index]
        "learning_rate": float(profile["training"]["learning_rate"]),  # type: ignore[index]
        "replay_generations": int(profile["replay"]["generations"]),  # type: ignore[index]
        "replay_cap": int(profile["replay"]["cap"]),  # type: ignore[index]
        "network_hidden": int(profile["network"]["hidden"]),  # type: ignore[index]
        "network_blocks": int(profile["network"]["blocks"]),  # type: ignore[index]
        "batch_size": int(profile["training"]["batch_size"]),  # type: ignore[index]
    }
    if set(observed) - set(expected):
        raise ValueError("fixed_training does not pin every required field")
    for key, actual in observed.items():
        wanted = expected[key]
        if float(wanted) != float(actual):
            raise ValueError(f"fixed_training drift for {key}: {actual} != {wanted}")


def arena_config(
    spec: Mapping[str, object],
    evaluation: Evaluation,
) -> ArenaExecutionConfig:
    raw = _mapping(spec.get("arena_execution"), "arena_execution")
    keys = (
        "workers",
        "games_per_worker",
        "inference_batch_rows",
        "inference_batch_wait_ms",
        "device",
        "strict_production",
        "min_mean_inference_batch_rows",
        "min_effective_cpu_cores",
        "early_gate_enabled",
        "early_gate_min_forwards",
        "early_gate_min_wall_sec",
    )
    missing = [key for key in keys if key not in raw]
    if missing:
        raise ValueError("arena_execution missing: " + ", ".join(missing))
    config = ArenaExecutionConfig(
        games=evaluation.games,
        workers=int(raw["workers"]),
        games_per_worker=int(raw["games_per_worker"]),
        inference_batch_rows=int(raw["inference_batch_rows"]),
        inference_batch_wait_ms=float(raw["inference_batch_wait_ms"]),
        device=str(raw["device"]),
        strict_production=bool(raw["strict_production"]),
        min_mean_inference_batch_rows=float(raw["min_mean_inference_batch_rows"]),
        min_effective_cpu_cores=float(raw["min_effective_cpu_cores"]),
        early_gate_enabled=bool(raw["early_gate_enabled"]),
        early_gate_min_forwards=int(raw["early_gate_min_forwards"]),
        early_gate_min_wall_sec=float(raw["early_gate_min_wall_sec"]),
    )
    config.validate_base()
    expected = _mapping(
        spec.get("arena_scientific_contract"), "arena_scientific_contract"
    )
    contract = TORUS9_ARENA_PROFILE.scientific_contract(config)
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Arena scientific contract drift for {key}")
    return config


def _checkpoint_identity_payload(reference: Mapping[str, object]) -> dict[str, object]:
    sha256 = str(reference.get("artifact_sha256") or reference.get("sha256") or "")
    if not sha256:
        raise ValueError("Evaluation checkpoint identity is missing SHA-256")
    return {
        "lineage_id": str(reference["lineage_id"]),
        "generation": int(reference["generation"]),
        "checkpoint_sha256": sha256,
    }


def _canonical_evaluation_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _evaluation_fingerprint(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _canonical_evaluation_json(payload).encode("utf-8")
    ).hexdigest()


def _build_evaluation_identity(
    *,
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    profile: str,
    games: int,
    master_seed: int,
    scientific_contract: Mapping[str, object],
    execution: ArenaExecutionConfig,
) -> tuple[dict[str, object], str]:
    if int(execution.games) != int(games):
        raise ValueError("Evaluation identity games drift from Arena execution config")
    payload: dict[str, object] = {
        "schema": EVALUATION_IDENTITY_SCHEMA,
        "candidate": _checkpoint_identity_payload(candidate),
        "reference": _checkpoint_identity_payload(reference),
        "profile": str(profile),
        "games": int(games),
        "master_seed": int(master_seed),
        "scientific_contract": dict(scientific_contract),
        "execution": asdict(execution),
    }
    return payload, _evaluation_fingerprint(payload)


def _resolved_evaluation_identity(
    spec: Mapping[str, object],
    evaluation: Evaluation,
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    config: ArenaExecutionConfig,
) -> tuple[dict[str, object], str]:
    scientific_contract = dict(TORUS9_ARENA_PROFILE.scientific_contract(config))
    expected = _mapping(
        spec.get("arena_scientific_contract"), "arena_scientific_contract"
    )
    for key, value in expected.items():
        if scientific_contract.get(key) != value:
            raise ValueError(f"Arena scientific contract drift for {key}")
    return _build_evaluation_identity(
        candidate=candidate,
        reference=reference,
        profile=TORUS9_ARENA_PROFILE.profile_id,
        games=evaluation.games,
        master_seed=evaluation.master_seed,
        scientific_contract=scientific_contract,
        execution=config,
    )


def _evaluation_run_id(
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    fingerprint: str,
) -> str:
    base = evaluation_id_for_comparison(
        candidate_lineage_id=str(candidate["lineage_id"]),
        candidate_generation=int(candidate["generation"]),
        reference_lineage_id=str(reference["lineage_id"]),
        reference_generation=int(reference["generation"]),
    )
    return f"{base}-{fingerprint[:EVALUATION_IDENTITY_HASH_PREFIX]}"


def _write_evaluation_identity(
    output: Path,
    run_id: str,
    identity: Mapping[str, object],
    fingerprint: str,
) -> None:
    if _evaluation_fingerprint(identity) != fingerprint:
        raise ValueError("Evaluation identity fingerprint is internally inconsistent")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / EVALUATION_IDENTITY_FILENAME,
        {
            "schema": EVALUATION_IDENTITY_RECORD_SCHEMA,
            "evaluation_id": run_id,
            "fingerprint": fingerprint,
            "identity": dict(identity),
        },
    )


def _stamp_evaluation_identity_metadata(
    output: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    marker = {
        "schema": EVALUATION_IDENTITY_SCHEMA,
        "path": EVALUATION_IDENTITY_FILENAME,
        "fingerprint": fingerprint,
    }
    for filename in ("provenance.json", "manifest.json"):
        path = output / filename
        if not path.is_file():
            continue
        payload = _read_json(path)
        payload["evaluation_id"] = run_id
        payload["evaluation_identity"] = marker
        _write_json(path, payload)


def _identity_mismatch(output: Path, detail: str) -> RuntimeError:
    return RuntimeError(
        f"evaluation identity/contract mismatch: {detail}: {output}"
    )


def _provenance_checkpoint_matches(
    actual: object,
    expected: Mapping[str, object],
) -> bool:
    if not isinstance(actual, Mapping):
        return False
    return (
        str(actual.get("lineage_id")) == str(expected["lineage_id"])
        and int(actual.get("generation", -1)) == int(expected["generation"])
        and str(actual.get("artifact_sha256") or actual.get("sha256"))
        == str(expected["checkpoint_sha256"])
    )


def _summary_wld(summary: Mapping[str, object]) -> tuple[int, int, int]:
    wld = summary.get("W/L/D")
    if not isinstance(wld, list) or len(wld) != 3:
        raise ValueError("Arena summary is missing candidate W/L/D")
    return int(wld[0]), int(wld[1]), int(wld[2])


def condition_met(
    condition: Mapping[str, object],
    evaluation_results: Mapping[str, Mapping[str, object]],
) -> bool:
    evaluation_id = str(condition["evaluation"])
    if evaluation_id not in evaluation_results:
        raise ValueError(f"Missing evaluation result for condition: {evaluation_id}")
    wins, losses, draws = _summary_wld(evaluation_results[evaluation_id])
    metric = str(condition["metric"])
    values = {
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "wins_minus_losses": float(wins - losses),
        "win_rate": (wins + 0.5 * draws) / max(1, wins + losses + draws),
    }
    actual = values[metric]
    wanted = float(condition["value"])
    operator = str(condition["operator"])
    return {
        ">": actual > wanted,
        ">=": actual >= wanted,
        "<": actual < wanted,
        "<=": actual <= wanted,
        "==": actual == wanted,
        "!=": actual != wanted,
    }[operator]


def _source_parent_reference(lineage_id: str, generation: int) -> dict[str, object]:
    resolved = resolve_checkpoint(
        {"topology": "torus9", "lineage_id": lineage_id, "generation": generation},
        topology="torus9",
    )
    checkpoint = resolved.path
    metadata_path = checkpoint.with_suffix(".metadata.json")
    replay_path = checkpoint.parents[1] / "replay" / f"rolling-after-{generation:02d}.jsonl"
    if not metadata_path.is_file() or not replay_path.is_file():
        raise FileNotFoundError("Shared parent checkpoint metadata/replay is missing")
    metadata = _read_json(metadata_path)
    if metadata.get("architecture_id") != TORUS9_CURRENT_ARCHITECTURE_ID:
        raise ValueError("Shared parent is not the current Torus9 architecture")
    architecture = metadata.get("architecture_config")
    if isinstance(architecture, Mapping):
        if int(architecture.get("hidden", TORUS9_CURRENT_HIDDEN)) != TORUS9_CURRENT_HIDDEN:
            raise ValueError("Shared parent hidden width is not 80")
        if int(architecture.get("blocks", TORUS9_CURRENT_BLOCKS)) != TORUS9_CURRENT_BLOCKS:
            raise ValueError("Shared parent residual block count is not 8")
    total_evictions = 0
    summary_path = checkpoint.parents[1] / f"iter-{generation:02d}-summary.json"
    if summary_path.is_file():
        replay = _read_json(summary_path).get("replay")
        if isinstance(replay, Mapping):
            total_evictions = int(replay.get("total_evictions", 0))
    replay_rows = metadata.get("replay_row_count", metadata.get("valid_replay_positions"))
    return {
        "topology": "torus9",
        "lineage_id": resolved.lineage_id,
        "checkpoint_id": resolved.checkpoint_id,
        "label": str(metadata.get("checkpoint_label") or f"M{generation}"),
        "generation": generation,
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
    )
    target_generation = int(parent["generation"]) + arm.iterations
    if run.paths.root.exists():
        manifest = _read_json(run.paths.manifest)
        if not _same_parent(manifest.get("parent_checkpoint"), parent):
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
    manifest = _read_json(run.paths.manifest)
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


def _checkpoint_reference(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "topology": "torus9",
        "lineage_id": str(result["lineage_id"]),
        "generation": int(result["generation"]),
        "checkpoint_id": f"M{int(result['generation'])}",
        "sha256": str(result["sha256"]),
        "artifact_sha256": str(result["sha256"]),
    }


def _existing_arena(
    output: Path,
    expected_identity: Mapping[str, object],
    expected_fingerprint: str,
) -> dict[str, object] | None:
    if not output.exists():
        return None
    identity_path = output / EVALUATION_IDENTITY_FILENAME
    if not identity_path.is_file():
        raise _identity_mismatch(output, "missing persisted evaluation identity")
    try:
        record = _read_json(identity_path)
    except (OSError, ValueError) as exc:
        raise _identity_mismatch(output, "malformed persisted evaluation identity") from exc
    if record.get("schema") != EVALUATION_IDENTITY_RECORD_SCHEMA:
        raise _identity_mismatch(output, "unsupported persisted identity schema")
    saved_identity = record.get("identity")
    saved_fingerprint = record.get("fingerprint")
    if not isinstance(saved_identity, Mapping) or not isinstance(saved_fingerprint, str):
        raise _identity_mismatch(output, "malformed persisted identity payload")
    saved_payload = dict(saved_identity)
    try:
        recomputed = _evaluation_fingerprint(saved_payload)
    except (TypeError, ValueError) as exc:
        raise _identity_mismatch(output, "non-canonical persisted identity payload") from exc
    if recomputed != saved_fingerprint:
        raise _identity_mismatch(
            output, "persisted full fingerprint does not match canonical payload"
        )
    if _evaluation_fingerprint(expected_identity) != expected_fingerprint:
        raise ValueError("Expected evaluation identity fingerprint is internally inconsistent")
    if saved_payload != dict(expected_identity) or saved_fingerprint != expected_fingerprint:
        raise _identity_mismatch(output, "persisted payload does not match current contract")

    summary_path = output / "summary.json"
    provenance_path = output / "provenance.json"
    if not summary_path.is_file() or not provenance_path.is_file():
        raise RuntimeError(f"Incomplete evaluation directory: {output}")
    summary = _read_json(summary_path)
    provenance = _read_json(provenance_path)
    expected_candidate = _mapping(expected_identity.get("candidate"), "identity.candidate")
    expected_reference = _mapping(expected_identity.get("reference"), "identity.reference")
    if (
        not _provenance_checkpoint_matches(provenance.get("candidate"), expected_candidate)
        or not _provenance_checkpoint_matches(provenance.get("reference"), expected_reference)
        or int(summary.get("games", -1)) != int(expected_identity["games"])
        or str(provenance.get("profile")) != str(expected_identity["profile"])
        or int(provenance.get("master_seed", -1)) != int(expected_identity["master_seed"])
    ):
        raise _identity_mismatch(output, "Arena result metadata disagrees with persisted identity")

    telemetry = summary.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise RuntimeError(
            f"Existing evaluation failed production validity/performance gates: "
            f"malformed telemetry: {output}"
        )
    technical_games = telemetry.get("technical_games")
    if isinstance(technical_games, bool) or not isinstance(technical_games, int):
        raise RuntimeError(
            f"Existing evaluation failed production validity/performance gates: "
            f"malformed technical_games: {output}"
        )
    if technical_games != 0:
        raise ValueError(f"Existing Arena has technical outcomes: {output}")
    performance_status = telemetry.get("performance_status")
    performance_failures = telemetry.get("performance_failures")
    if not isinstance(performance_status, str) or not performance_status.strip():
        raise RuntimeError(
            f"Existing evaluation failed production validity/performance gates: "
            f"malformed performance_status: {output}"
        )
    if not isinstance(performance_failures, list):
        raise RuntimeError(
            f"Existing evaluation failed production validity/performance gates: "
            f"malformed performance_failures: {output}"
        )
    if performance_status.strip().upper() == "CRITICAL" or performance_failures:
        raise RuntimeError(
            "Existing evaluation failed production validity/performance gates: "
            f"performance_status={performance_status!r}, "
            f"performance_failures={performance_failures!r}: {output}"
        )
    return summary


def _compare(
    spec: Mapping[str, object],
    evaluation: Evaluation,
    arm_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    candidate = _checkpoint_reference(arm_results[evaluation.candidate])
    reference = _checkpoint_reference(arm_results[evaluation.reference])
    config = arena_config(spec, evaluation)
    identity, fingerprint = _resolved_evaluation_identity(
        spec, evaluation, candidate, reference, config
    )
    run_id = _evaluation_run_id(candidate, reference, fingerprint)
    output = evaluation_dir("torus9", run_id)
    existing = _existing_arena(output, identity, fingerprint)
    if existing is not None:
        return existing

    _write_evaluation_identity(output, run_id, identity, fingerprint)
    result = run_arena(
        candidate_path=candidate,
        reference_path=reference,
        profile_name="torus9",
        output_dir=output,
        candidate_label=evaluation.candidate,
        reference_label=evaluation.reference,
        run_id=run_id,
        comparison=f"{spec['kind']} {evaluation.candidate} vs {evaluation.reference}",
        master_seed=evaluation.master_seed,
        config=config,
    )
    _stamp_evaluation_identity_metadata(output, run_id, fingerprint)
    result["evaluation_id"] = run_id
    result["evaluation_fingerprint"] = fingerprint
    return result


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
    parent = _source_parent_reference(start_lineage, start_generation)
    arm_results: dict[str, dict[str, object]] = {}
    evaluation_results: dict[str, dict[str, object]] = {}
    stage_results: list[dict[str, object]] = []

    for raw_stage in spec["stages"]:  # type: ignore[index]
        stage = _mapping(raw_stage, "stage")
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
            evaluation_results[str(evaluation_id)] = _compare(
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


if __name__ == "__main__":
    raise SystemExit(main())
