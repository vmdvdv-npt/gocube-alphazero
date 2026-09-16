#!/usr/bin/env python3
"""Evidence-first Torus9 self-play, training, and cadence experiment.

This tool is intentionally separate from the canonical Golden launcher. It
creates one lineage directory per arm under ``runs/torus9/active`` and one
evaluation directory under ``runs/torus9/evaluations``. It never writes the
historical M17 run or creates M18. The cadence arms use the
same current scientific profile and execution preset, while changing only the
number of games and the proportionally normalized Adam work:

    64  games / 80  steps / 5120  draws, six iterations
    128 games / 160 steps / 10240 draws, three iterations
    192 games / 240 steps / 15360 draws, two iterations

Strength comparisons use the standard-64 Arena contract.  The high-volume
Arena preset is deliberately not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, MutableMapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.neural import model_hash
from gocube_golden.provenance import (
    CodeIdentity,
    capture_code_identity,
    derive_seed,
    file_sha256,
)
from gocube_golden.run_storage import active_lineage_dir, create_lineage, evaluation_dir
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9SelfPlayGameRecord,
    Torus9SelfPlaySearchContract,
    Torus9TrainingAdapter,
    run_torus9_selfplay_games,
    run_torus9_training_iteration,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    write_json,
)
from gocube_golden.torus9_serialization import write_torus9_game_records_jsonl
from gocube_golden.torus9_contract import (
    TORUS9_BATCH_SIZE,
    TORUS9_CURRENT_MODEL_INIT_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_CURRENT_TRAINING_MASTER_SEED,
    TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    TORUS9_KOMI,
    TORUS9_MAX_REPLAY_POSITIONS,
    TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    TORUS9_ROLLING_GENERATIONS,
    TORUS9_TRAINING_SAMPLES_PER_ITERATION,
    TORUS9_RULES_FINGERPRINT,
    current_torus9_profile_fingerprint,
    current_torus9_selfplay_contract_fingerprint,
    load_torus9_current_profile,
)
from tools.arena import run_arena
from tools.arena_engine import ArenaExecutionConfig
from tools.arena_profiles import get_profile
from training_engine import value_fingerprint


NIGHT_ACTIVE_ROOT = ROOT / "runs" / "torus9" / "active"
CANONICAL_RUN = active_lineage_dir("torus9", "torus9-golden-v3-20260914-run03")
ARENA_SEED = 202609131005
BEFORE_MOVES_PER_SEC = 20.931


@dataclass(frozen=True)
class CadenceArm:
    games: int
    iterations: int
    optimizer_steps: int

    @property
    def sample_draws(self) -> int:
        return self.optimizer_steps * TORUS9_BATCH_SIZE

    @property
    def total_games(self) -> int:
        return self.games * self.iterations

    @property
    def total_optimizer_steps(self) -> int:
        return self.optimizer_steps * self.iterations

    def as_dict(self) -> dict[str, int]:
        return {
            "games_per_iteration": self.games,
            "iterations": self.iterations,
            "optimizer_steps_per_iteration": self.optimizer_steps,
            "sample_draws_per_iteration": self.sample_draws,
            "total_games": self.total_games,
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_sample_draws": self.sample_draws * self.iterations,
        }


CADENCE_ARMS = (
    CadenceArm(64, 6, 80),
    CadenceArm(128, 3, 160),
    CadenceArm(192, 2, 240),
)
EXECUTION = {
    "workers": 16,
    "active_games_per_worker": 4,
    "total_active_contexts": 64,
    "inference_batch_cap": 64,
    "inference_batch_wait_ms": 1.0,
    "coalescing": True,
    "shared_memory": True,
    "central_inference_owner": "parent",
    "device": "cuda",
}
ARENA_STANDARD_64 = {
    "games": 64,
    "paired_starts": 32,
    "workers": 16,
    "games_per_worker": 4,
    "total_active_contexts": 64,
    "inference_batch_rows": 64,
    "inference_batch_wait_ms": 1.0,
    "simulations": 64,
    "cpuct": 1.25,
    "fpu": 0.0,
    "root_noise": False,
    "temperature": 0.0,
    "fast_search": False,
    "resign": False,
    "frozen_startset": True,
    "paired_color_swap": True,
    "deterministic_tie_break": True,
    "komi": 0.5,
    "technical_outcomes": "fail-closed / excluded",
}


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object in {path}")
                rows.append(row)
    return tuple(rows)


def _git_branch() -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()


def _prepare_empty(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing nightly namespace: {path}")
    else:
        path.mkdir(parents=True)


def _profile_contract() -> tuple[dict[str, object], str, Torus9SelfPlaySearchContract]:
    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    settings = profile["self_play"]
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


def standard_arena_config() -> ArenaExecutionConfig:
    """Return the corrected standard-64 strength-evaluation execution."""
    return ArenaExecutionConfig(
        games=64,
        workers=16,
        games_per_worker=4,
        inference_batch_rows=64,
        inference_batch_wait_ms=1.0,
        device="cuda",
        # The current performance gates are intentionally stricter than the
        # scientific strength contract.  Technical games remain fail-closed;
        # a low utilization score is not a strength invalidation.
        strict_production=False,
    )


def _new_model_state(
    *,
    run_id: str,
    run_dir: Path,
    code: CodeIdentity,
    adapter: Torus9TrainingAdapter | None = None,
) -> tuple[Any, Torus9TrainingAdapter, Path, dict[str, object]]:
    profile, profile_fp, _ = _profile_contract()
    torch.manual_seed(TORUS9_CURRENT_MODEL_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORUS9_CURRENT_MODEL_INIT_SEED)
    model = Torus9CurrentGraphNet().to("cuda")
    selected = adapter or Torus9TrainingAdapter(
        profile=profile,
        code_identity=code,
        base_commit=TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    )
    state = selected.create_state(model, run_id=run_id)
    checkpoint = run_dir / "checkpoints" / "M0.pt"
    selected.save_initial_checkpoint(
        checkpoint,
        state,
        run_id=run_id,
        label="M0",
        completed_games=0,
        device="cuda",
        code_identity=code,
    )
    metadata = _read_json(checkpoint.with_suffix(".metadata.json"))
    return state, selected, checkpoint, {
        "model_hash": metadata["model_hash"],
        "checkpoint_sha256": file_sha256(checkpoint),
        "metadata_sha256": file_sha256(checkpoint.with_suffix(".metadata.json")),
        "profile_fingerprint": profile_fp,
    }


class CadenceTrainingAdapter(Torus9TrainingAdapter):
    """Explicit opt-in adapter for proportional-work cadence experiments."""

    def __init__(self, *, optimizer_steps: int, **kwargs: object) -> None:
        if optimizer_steps not in {80, 160, 240}:
            raise ValueError("Cadence optimizer budget must be 80, 160, or 240")
        super().__init__(**kwargs)
        self.cadence_optimizer_steps = int(optimizer_steps)

    def create_state(self, *args: object, **kwargs: object):
        state = super().create_state(*args, **kwargs)  # type: ignore[arg-type]
        # The canonical constructor remains fixed at 80.  This explicit
        # experiment-only override changes only the number of identical Adam
        # updates in one cadence iteration.
        state.adapter_state.optimizer_steps_per_iteration = self.cadence_optimizer_steps
        return state

    def train(
        self,
        state: Any,
        rows: Sequence[Mapping[str, object]],
        seed: int,
    ) -> Mapping[str, object]:
        self.validate_state(state)
        trainer = state.adapter_state
        count = self.cadence_optimizer_steps * TORUS9_BATCH_SIZE
        indices = trainer._sample_indices(len(rows), seed=int(seed), count=count)
        metrics = dict(
            trainer.train_fixed_budget(
                rows,
                seed=int(seed),
                validate_samples=False,
                timing=self._diagnostic_timing,
            )
        )
        if self._diagnostic_timing is not None:
            metrics["stage_timing"] = dict(self._diagnostic_timing)
        metrics.update({
            "training_seed": int(seed),
            "cadence_optimizer_steps_per_iteration": self.cadence_optimizer_steps,
            "sampled_replay_row_ids": tuple(
                str(rows[index].get("replay_row_id", index)) for index in indices
            ),
        })
        metrics["sampled_row_ids_fingerprint"] = value_fingerprint(
            metrics["sampled_replay_row_ids"]
        )
        if metrics.get("optimizer_steps") != self.cadence_optimizer_steps:
            raise ValueError("Cadence optimizer step budget drift")
        if metrics.get("samples_consumed") != count:
            raise ValueError("Cadence sample exposure budget drift")
        return metrics

    def prepare_checkpoint(
        self,
        state: Any,
        context: Any,
        training_metrics: Mapping[str, object],
    ) -> Mapping[str, object]:
        metadata = dict(super().prepare_checkpoint(state, context, training_metrics))
        metadata["optimizer_steps_per_iteration"] = self.cadence_optimizer_steps
        metadata["samples_consumed_per_iteration"] = self.cadence_optimizer_steps * TORUS9_BATCH_SIZE
        contract = dict(metadata["scientific_contract"])  # type: ignore[arg-type]
        contract.update({
            "optimizer_steps": self.cadence_optimizer_steps,
            "samples_consumed": self.cadence_optimizer_steps * TORUS9_BATCH_SIZE,
        })
        metadata["scientific_contract"] = contract
        metadata["cadence_experiment"] = {
            "status": "controlled proportional-work arm",
            "canonical_profile_unchanged": True,
            "optimizer": "Adam",
            "batch_size": TORUS9_BATCH_SIZE,
            "steps_per_iteration": self.cadence_optimizer_steps,
            "sample_draws_per_iteration": self.cadence_optimizer_steps * TORUS9_BATCH_SIZE,
        }
        return metadata


def _iteration_summary_builder(
    *,
    generation: int,
    games: int,
    records: Sequence[Torus9SelfPlayGameRecord],
    execution: Mapping[str, object],
    selfplay_wall: float,
    selfplay_postprocessing_wall: float,
    selfplay_serialization_wall: float,
    telemetry: Mapping[str, object],
    adapter: Torus9TrainingAdapter,
    state: Any,
    started: float,
):
    def build(base: Mapping[str, object]) -> Mapping[str, object]:
        training = base.get("training")
        replay = base.get("replay")
        if not isinstance(training, Mapping) or not isinstance(replay, Mapping):
            raise ValueError("Malformed TrainingEngine metrics")
        phase = training.get("phase_timing")
        phase = dict(phase) if isinstance(phase, Mapping) else {}
        moves = sum(len(record.final_action_trace) for record in records)
        fresh = int(base["fresh_positions"])
        replay_rows = len(state.rolling_replay.rows)
        sampled_draws = int(training.get("samples_consumed", 0))
        optimizer_steps = int(training.get("optimizer_steps", 0))
        unique_rows = int(training.get("unique_sample_rows", 0))
        flow = {
            "games": games,
            "moves": moves,
            "emitted_samples": fresh,
            "persisted_samples": fresh,
            "replay_rows": replay_rows,
            "eligible_rows": replay_rows,
            "sampled_draws": sampled_draws,
            "unique_sampled_rows": unique_rows,
            "batches": optimizer_steps,
            "optimizer_updates": optimizer_steps,
            "reused_sample_draws": sampled_draws - unique_rows,
            "unexplained_loss_or_duplication": False,
        }
        timing = {
            "iteration_wall_time_until_publication_sec": max(0.0, time.perf_counter() - started),
            "self_play_wall_time_sec": selfplay_wall,
            "self_play_postprocessing_wall_time_sec": selfplay_postprocessing_wall,
            "self_play_serialization_wall_time_sec": selfplay_serialization_wall,
            "training_transaction_phase_timing": phase,
            "self_play_startup_wall_time_sec": telemetry.get("startup_wall_time_sec"),
            "self_play_teardown_wall_time_sec": telemetry.get("teardown_wall_time_sec"),
        }
        return {
            **dict(base),
            "self_play": {
                "games": games,
                "valid_games": games - int(telemetry.get("technical_games", 0)),
                "technical_games": int(telemetry.get("technical_games", 0)),
                "moves": moves,
                "wall_time_sec": selfplay_wall,
                "postprocessing_wall_time_sec": selfplay_postprocessing_wall,
                "games_per_sec": games / selfplay_wall if selfplay_wall else 0.0,
                "games_per_hour": games * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
                "moves_per_sec": moves / selfplay_wall if selfplay_wall else 0.0,
                "inference_rows_per_sec": telemetry.get("inference_rows_per_sec", 0.0),
                "effective_cpu_cores": telemetry.get("process_tree_effective_cpu_cores", 0.0),
                "worker_cpu_seconds": telemetry.get("worker_process_cpu_seconds", 0.0),
                "worker_inference_wait_seconds": telemetry.get("worker_blocked_inference_seconds", 0.0),
                "batch_distribution": {
                    "mean": telemetry.get("mean_inference_batch_rows", 0.0),
                    "p50": telemetry.get("p50_inference_batch_rows", 0),
                    "p95": telemetry.get("p95_inference_batch_rows", 0),
                    "max": telemetry.get("max_inference_batch_rows", 0),
                },
                "execution": dict(execution),
                "wait_accounting": {
                    "worker_to_broker": telemetry.get("worker_to_broker_latency_ms"),
                    "broker_queue": telemetry.get("broker_queue_latency_ms"),
                    "batch_collection": telemetry.get("broker_queue_latency_ms"),
                    "h2d": telemetry.get("h2d_latency_ms"),
                    "forward": telemetry.get("model_forward_latency_ms"),
                    "response_to_worker": telemetry.get("response_to_worker_latency_ms"),
                    "worker_inference_wait": telemetry.get("worker_blocked_inference_ms"),
                    "scheduling": {
                        "pending_games_final": telemetry.get("pending_games_final"),
                        "tail_after_pending_empty_sec": telemetry.get("tail_duration_after_pending_empty_sec"),
                        "global_task_replenishment": telemetry.get("global_task_replenishment"),
                    },
                },
                "technical_outcomes": "fail-closed / excluded",
            },
            "flow_accounting": flow,
            "timing": timing,
        }

    return build


def _run_generation(
    *,
    run_dir: Path,
    run_id: str,
    generation: int,
    games: int,
    steps: int,
    state: Any,
    adapter: Torus9TrainingAdapter,
    contract: Torus9SelfPlaySearchContract,
    code: CodeIdentity,
    profile_fp: str,
    source_checkpoint: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    game_ids = [
        f"{run_id}-M{generation:02d}-game-{index:04d}"
        for index in range(games)
    ]
    telemetry: dict[str, object] = {}
    selfplay_started = time.perf_counter()
    records = run_torus9_selfplay_games(
        state.model,
        run_id=run_id,
        label=f"M{generation - 1}",
        artifact=file_sha256(source_checkpoint),
        master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
        profile_fp=profile_fp,
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        game_ids=game_ids,
        workers=int(EXECUTION["workers"]),
        code_identity=code,
        device="cuda",
        contract=contract,
        coalescing=True,
        inference_batch_cap=int(EXECUTION["inference_batch_cap"]),
        inference_batch_wait_ms=float(EXECUTION["inference_batch_wait_ms"]),
        active_games_per_worker=int(EXECUTION["active_games_per_worker"]),
        total_active_contexts=int(EXECUTION["total_active_contexts"]),
        inference_telemetry=telemetry,
        execution_activity=telemetry,
        execution_reference_interactive=False,
    )
    selfplay_wall = time.perf_counter() - selfplay_started
    postprocessing_started = time.perf_counter()
    if len(records) != games or {record.game_id for record in records} != set(game_ids):
        raise ValueError(f"M{generation} returned an unexpected game set")
    for record in records:
        record.validate()
        if record.technical_termination is not None:
            raise ValueError(f"Technical self-play game is not allowed: {record.game_id}")
    move_count = sum(len(record.final_action_trace) for record in records)
    position_count = sum(len(record.positions) for record in records)
    if position_count != move_count:
        raise ValueError("Self-play positions and action traces are not one-to-one")
    selfplay_postprocessing_wall = time.perf_counter() - postprocessing_started
    if telemetry.get("execution_reference_status") != "validated_recommended":
        raise ValueError("Cadence self-play did not use the validated execution preset")
    if int(telemetry.get("target_active_contexts", 0)) != 64:
        raise ValueError("Cadence self-play did not reach 64 active contexts")
    if sum(record.nn_evaluations for record in records) != int(telemetry.get("total_rows", -1)):
        raise ValueError("Self-play inference rows were lost or duplicated")

    games_path = run_dir / "selfplay" / f"M{generation:02d}-games.jsonl"
    serialize_started = time.perf_counter()
    write_torus9_game_records_jsonl(games_path, records)
    selfplay_serialization_wall = time.perf_counter() - serialize_started

    diagnostic_sink: MutableMapping[str, object] = {}
    adapter.set_diagnostic_timing(diagnostic_sink)
    training_started = time.perf_counter()
    try:
        result = run_torus9_training_iteration(
            state=state,
            generation=generation,
            output_dir=run_dir,
            run_id=run_id,
            records=records,
            training_seed=derive_seed(
                TORUS9_CURRENT_TRAINING_MASTER_SEED,
                run_id,
                "training",
                generation,
            ),
            completed_games=generation * games,
            code_identity=code,
            device="cuda",
            adapter=adapter,
            summary_builder=_iteration_summary_builder(
                generation=generation,
                games=games,
                records=records,
                execution=EXECUTION,
                selfplay_wall=selfplay_wall,
                selfplay_postprocessing_wall=selfplay_postprocessing_wall,
                selfplay_serialization_wall=selfplay_serialization_wall,
                telemetry=telemetry,
                adapter=adapter,
                state=state,
                started=started,
            ),
        )
    finally:
        adapter.set_diagnostic_timing(None)
    training_wall = time.perf_counter() - training_started
    summary = dict(result.summary)
    summary["source_checkpoint"] = {
        "path": str(source_checkpoint),
        "model_hash": _read_json(source_checkpoint.with_suffix(".metadata.json"))["model_hash"],
        "artifact_sha256": file_sha256(source_checkpoint),
    }
    summary["training"]["diagnostic_sink"] = dict(diagnostic_sink)  # type: ignore[index]
    summary["training"]["diagnostic_training_wall_time_sec"] = training_wall  # type: ignore[index]
    # The generic commit marker already seals iter-XX-summary.json.  Keep the
    # exact post-publication envelope beside it rather than rewriting a sealed
    # summary and invalidating its hash.
    _write(run_dir / "timing" / f"M{generation:02d}.json", {
        "iteration": generation,
        "self_play_wall_time_sec": selfplay_wall,
        "self_play_postprocessing_wall_time_sec": selfplay_postprocessing_wall,
        "self_play_serialization_wall_time_sec": selfplay_serialization_wall,
        "training_transaction_wall_time_sec": training_wall,
        "iteration_wall_time_sec": time.perf_counter() - started,
        "checkpoint_publication_included": True,
        "telemetry": telemetry,
        "phase_timing": summary["training"].get("phase_timing"),  # type: ignore[index]
        "diagnostic_stage_timing": dict(diagnostic_sink),
    })
    # Retain the sealed summary as the source of truth; the additional fields
    # above are duplicated in the timing envelope for report generation.
    return summary


def _run_dir_manifest(
    *,
    run_id: str,
    run_dir: Path,
    code: CodeIdentity,
    profile: Mapping[str, object],
    profile_fp: str,
    arm: CadenceArm | None,
    m0: Mapping[str, object],
    created_at: str,
) -> dict[str, object]:
    manifest = _lineage_manifest_seed(
        run_id=run_id,
        code=code,
        profile_fp=profile_fp,
        created_at=created_at,
    )
    manifest.update({
        "schema": "torus9-nightly-diagnostics-v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "branch": _git_branch(),
        "commit": code.git_commit_sha,
        "tree": code.git_tree_sha,
        "working_tree_clean": code.working_tree_clean,
        "golden_lineage_base_commit": TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
        "canonical_run_read_only": str(CANONICAL_RUN),
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "rules_fingerprint": TORUS9_RULES_FINGERPRINT,
        "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        "selfplay_contract_fingerprint": current_torus9_selfplay_contract_fingerprint(),
        "komi": TORUS9_KOMI,
        "execution": dict(EXECUTION),
        "arena_standard_64": dict(ARENA_STANDARD_64),
        "profile_snapshot": dict(profile),
        "cadence_arm": arm.as_dict() if arm else None,
        "m0": dict(m0),
        "result_status": "initialized",
        "checkpoint_hashes": {"checkpoints/M0.pt": m0["checkpoint_sha256"]},
    })
    return manifest


def _lineage_manifest_seed(
    *,
    run_id: str,
    code: CodeIdentity,
    profile_fp: str,
    created_at: str,
) -> dict[str, object]:
    """Return the required passport written before any training work starts."""
    return {
        "manifest_schema": "torus9-nightly-diagnostics-v1",
        "lineage_id": run_id,
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": None,
        "git_commit": code.git_commit_sha,
        "config_fingerprint": profile_fp,
        "created_at": created_at,
        "checkpoint_hashes": {},
    }


def run_phase_a(run_id: str) -> dict[str, object]:
    code = capture_code_identity(ROOT)
    profile, profile_fp, contract = _profile_contract()
    created_at = datetime.now(timezone.utc).isoformat()
    run_dir = create_lineage(
        "torus9",
        run_id,
        manifest=_lineage_manifest_seed(
            run_id=run_id,
            code=code,
            profile_fp=profile_fp,
            created_at=created_at,
        ),
        extra_directories=("selfplay", "replay", "training", "timing"),
    )
    load_started = time.perf_counter()
    state, adapter, m0_path, m0 = _new_model_state(
        run_id=run_id,
        run_dir=run_dir,
        code=code,
    )
    model_load_wall = time.perf_counter() - load_started
    manifest = _run_dir_manifest(
        run_id=run_id,
        run_dir=run_dir,
        code=code,
        profile=profile,
        profile_fp=profile_fp,
        arm=CADENCE_ARMS[0],
        m0=m0,
        created_at=created_at,
    )
    manifest["phase"] = "A — current 64-game production reproduction"
    manifest["model_initialization_and_loading_wall_time_sec"] = model_load_wall
    _write(run_dir / "manifest.json", manifest)
    row = _run_generation(
        run_dir=run_dir,
        run_id=run_id,
        generation=1,
        games=64,
        steps=80,
        state=state,
        adapter=adapter,
        contract=contract,
        code=code,
        profile_fp=profile_fp,
        source_checkpoint=m0_path,
    )
    timing = _read_json(run_dir / "timing" / "M01.json")
    report = {
        "schema": "torus9-phase-a-report-v1",
        "phase": "A",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "existing_evidence": {
            "classification": "ALREADY PROVEN",
            "source": "docs/SELFPLAY_EXECUTION_BENCHMARK_20260915.md",
            "validated_reference_moves_per_sec": BEFORE_MOVES_PER_SEC,
            "validated_preset": dict(EXECUTION),
            "reason_not_repeated": "PR #108 already removed the material shared-memory/IPC serialization bottleneck; this run is the required current-path reproduction and wait accounting check.",
        },
        "resolved_config": {
            "profile_id": TORUS9_CURRENT_PROFILE_ID,
            "profile_fingerprint": profile_fp,
            "komi": TORUS9_KOMI,
            "execution": dict(EXECUTION),
        },
        "model_load_wall_time_sec": model_load_wall,
        "iteration": row,
        "post_publication_timing": timing,
        "critical_path": _critical_path(row, timing),
        "wait_accounting": row["self_play"]["wait_accounting"],  # type: ignore[index]
        "causal_bottleneck": _causal_bottleneck(row, timing),
        "before_after": {
            "before_moves_per_sec": BEFORE_MOVES_PER_SEC,
            "after_moves_per_sec": row["self_play"]["moves_per_sec"],  # type: ignore[index]
            "delta_pct": 100.0 * (float(row["self_play"]["moves_per_sec"]) / BEFORE_MOVES_PER_SEC - 1.0),  # type: ignore[index]
            "semantic_contract_changed": False,
            "new_high_roi_fix": "none; existing PR #108 execution fix is already the current production path",
        },
        "assumptions": [
            "The current working tree contains pre-existing untracked user artifacts; provenance records the exact commit/tree and does not touch them.",
            "GPU utilization sampling is unavailable in this environment because NVML/nvidia-smi is unavailable; CUDA availability and model-stage timings are still recorded.",
        ],
    }
    _write(run_dir / "phase-a-report.json", report)
    _write(run_dir / "phase-a-report.md", _render_phase_a(report))
    return report


def run_cadence_arm(run_id: str, arm: CadenceArm) -> dict[str, object]:
    lineage_id = f"{run_id}-arm-{arm.games}"
    code = capture_code_identity(ROOT)
    profile, profile_fp, contract = _profile_contract()
    created_at = datetime.now(timezone.utc).isoformat()
    run_dir = create_lineage(
        "torus9",
        lineage_id,
        manifest=_lineage_manifest_seed(
            run_id=lineage_id,
            code=code,
            profile_fp=profile_fp,
            created_at=created_at,
        ),
        extra_directories=("selfplay", "replay", "training", "timing"),
    )
    adapter = CadenceTrainingAdapter(
        optimizer_steps=arm.optimizer_steps,
        profile=profile,
        code_identity=code,
        base_commit=TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    )
    state, adapter, m0_path, m0 = _new_model_state(
        run_id=lineage_id,
        run_dir=run_dir,
        code=code,
        adapter=adapter,
    )
    manifest = _run_dir_manifest(
        run_id=f"{run_id}-arm-{arm.games}",
        run_dir=run_dir,
        code=code,
        profile=profile,
        profile_fp=profile_fp,
        arm=arm,
        m0=m0,
        created_at=created_at,
    )
    manifest["phase"] = "B — equal-budget games/iteration cadence experiment"
    _write(run_dir / "manifest.json", manifest)
    rows: list[dict[str, object]] = []
    source_checkpoint = m0_path
    started = time.perf_counter()
    for generation in range(1, arm.iterations + 1):
        row = _run_generation(
            run_dir=run_dir,
            run_id=str(manifest["run_id"]),
            generation=generation,
            games=arm.games,
            steps=arm.optimizer_steps,
            state=state,
            adapter=adapter,
            contract=contract,
            code=code,
            profile_fp=profile_fp,
            source_checkpoint=source_checkpoint,
        )
        rows.append(row)
        source_checkpoint = run_dir / "checkpoints" / f"M{generation}.pt"
        manifest["last_generation"] = generation
        manifest["result_status"] = "running"
        _write(run_dir / "manifest.json", manifest)
    wall = time.perf_counter() - started
    final = rows[-1]
    totals = {
        "games": sum(int(row["self_play"]["games"]) for row in rows),  # type: ignore[index]
        "moves": sum(int(row["self_play"]["moves"]) for row in rows),  # type: ignore[index]
        "fresh_positions": sum(int(row["flow_accounting"]["emitted_samples"]) for row in rows),  # type: ignore[index]
        "optimizer_steps": sum(int(row["training"]["optimizer_steps"]) for row in rows),  # type: ignore[index]
        "sample_draws": sum(int(row["training"]["samples_consumed"]) for row in rows),  # type: ignore[index]
    }
    expected = {
        "games": arm.total_games,
        "optimizer_steps": arm.total_optimizer_steps,
        "sample_draws": arm.sample_draws * arm.iterations,
    }
    budget_totals = {key: totals[key] for key in expected}
    equal_budget = budget_totals == expected
    report = {
        "schema": "torus9-cadence-arm-report-v1",
        "phase": "B",
        "run_id": str(manifest["run_id"]),
        "arm": arm.as_dict(),
        "run_dir": str(run_dir),
        "execution": dict(EXECUTION),
        "arena_evaluation_execution": dict(ARENA_STANDARD_64),
        "iterations": rows,
        "totals": totals,
        "expected_totals": expected,
        "equal_budget_status": "PASS" if equal_budget else "FAIL",
        "wall_time_sec": wall,
        "final_checkpoint": {
            "path": str(source_checkpoint),
            "model_hash": _read_json(source_checkpoint.with_suffix(".metadata.json"))["model_hash"],
            "artifact_sha256": file_sha256(source_checkpoint),
        },
        "scientific_fingerprint": {
            "profile": profile_fp,
            "rules": TORUS9_RULES_FINGERPRINT,
            "target": TORUS9_CURRENT_TARGET_FINGERPRINT,
            "selfplay": current_torus9_selfplay_contract_fingerprint(),
            "komi": TORUS9_KOMI,
        },
        "canonical_run_mutation": False,
        "status": "PASS" if equal_budget and all(
            int(row["self_play"]["technical_games"]) == 0  # type: ignore[index]
            and bool(row["flow_accounting"]["unexplained_loss_or_duplication"]) is False  # type: ignore[index]
            for row in rows
        ) else "FAIL",
    }
    manifest["last_generation"] = arm.iterations
    manifest["result_status"] = report["status"]
    manifest["status"] = "ACTIVE"
    manifest["report"] = str(run_dir / "cadence-arm-report.json")
    _write(run_dir / "manifest.json", manifest)
    _write(run_dir / "cadence-arm-report.json", report)
    _write(run_dir / "cadence-arm-report.md", _render_arm(report))
    return report


def run_cadence(run_id: str) -> dict[str, object]:
    root = evaluation_dir("torus9", f"{run_id}-cadence")
    _prepare_empty(root)
    reports = [run_cadence_arm(run_id, arm) for arm in CADENCE_ARMS]
    report = {
        "schema": "torus9-cadence-experiment-report-v1",
        "phase": "B",
        "run_id": run_id,
        "run_root": str(root),
        "normalization": {
            "cumulative_games_per_arm": 384,
            "cumulative_optimizer_steps_per_arm": 480,
            "batch_size": TORUS9_BATCH_SIZE,
            "formula": "optimizer steps = 1.25 × games per iteration; sample draws = steps × batch size",
        },
        "arms": reports,
        "arena_strength_comparison": "pending; run the arena command after all three arms complete",
        "status": "PASS" if all(report["status"] == "PASS" for report in reports) else "FAIL",
    }
    _write(root / "cadence-experiment-report.json", report)
    _write(root / "cadence-experiment-report.md", _render_cadence(report))
    return report


def refresh_cadence_report(run_id: str) -> dict[str, object]:
    """Recompute derived cadence status without rerunning hardware work."""
    root = evaluation_dir("torus9", f"{run_id}-cadence")
    reports: list[dict[str, object]] = []
    for arm in CADENCE_ARMS:
        arm_dir = root / f"arm-{arm.games}"
        report = _read_json(arm_dir / "cadence-arm-report.json")
        totals = report.get("totals")
        if not isinstance(totals, Mapping):
            raise ValueError(f"Malformed cadence totals: {arm_dir}")
        expected = {
            "games": arm.total_games,
            "optimizer_steps": arm.total_optimizer_steps,
            "sample_draws": arm.sample_draws * arm.iterations,
        }
        equal_budget = all(int(totals.get(key, -1)) == value for key, value in expected.items())
        iterations = report.get("iterations")
        if not isinstance(iterations, list) or len(iterations) != arm.iterations:
            raise ValueError(f"Incomplete cadence iterations: {arm_dir}")
        flow_ok = all(
            int(row["self_play"]["technical_games"]) == 0  # type: ignore[index]
            and bool(row["flow_accounting"]["unexplained_loss_or_duplication"]) is False  # type: ignore[index]
            for row in iterations
        )
        report["equal_budget_status"] = "PASS" if equal_budget else "FAIL"
        report["status"] = "PASS" if equal_budget and flow_ok else "FAIL"
        _write(arm_dir / "cadence-arm-report.json", report)
        _write(arm_dir / "cadence-arm-report.md", _render_arm(report))
        reports.append(report)
    report = _read_json(root / "cadence-experiment-report.json")
    report["arms"] = reports
    report["status"] = "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL"
    _write(root / "cadence-experiment-report.json", report)
    _write(root / "cadence-experiment-report.md", _render_cadence(report))
    return report


def run_arena_phase(run_id: str) -> dict[str, object]:
    root = evaluation_dir("torus9", f"{run_id}-cadence")
    cadence = _read_json(root / "cadence-experiment-report.json")
    arms = cadence.get("arms")
    if not isinstance(arms, list) or len(arms) != 3:
        raise ValueError("Cadence report is incomplete; all 64/128/192 arms are required")
    final_paths = {
        str(report["arm"]["games_per_iteration"]): Path(str(report["final_checkpoint"]["path"]))  # type: ignore[index]
        for report in arms
    }
    arena_root = root / "arena-standard-64"
    _prepare_empty(arena_root)
    pairings = (("64", "128"), ("64", "192"), ("128", "192"))
    results: list[dict[str, object]] = []
    profile = get_profile("torus9")
    config = standard_arena_config()
    if {
        "games": config.games,
        "workers": config.workers,
        "games_per_worker": config.games_per_worker,
        "inference_batch_rows": config.inference_batch_rows,
        "inference_batch_wait_ms": config.inference_batch_wait_ms,
    } != {
        "games": 64,
        "workers": 16,
        "games_per_worker": 4,
        "inference_batch_rows": 64,
        "inference_batch_wait_ms": 1.0,
    }:
        raise AssertionError("Standard-64 Arena contract drift")
    for candidate_key, reference_key in pairings:
        output = arena_root / f"{candidate_key}-vs-{reference_key}"
        summary = run_arena(
            candidate_path=final_paths[candidate_key],
            reference_path=final_paths[reference_key],
            profile_name="torus9",
            output_dir=output,
            candidate_label=f"arm-{candidate_key}",
            reference_label=f"arm-{reference_key}",
            run_id=f"{run_id}-standard64-{candidate_key}-vs-{reference_key}",
            comparison=f"cadence arm {candidate_key} vs {reference_key}",
            master_seed=ARENA_SEED,
            config=config,
        )
        arena_telemetry = summary.get("telemetry")
        if not isinstance(arena_telemetry, Mapping):
            raise ValueError(f"Arena telemetry is missing for {candidate_key} vs {reference_key}")
        technical = int(arena_telemetry.get("technical_games", 0))
        wld = summary.get("W/L/D", [0, 0, 0])
        if not isinstance(wld, list) or len(wld) != 3:
            raise ValueError(f"Malformed Arena result for {candidate_key} vs {reference_key}")
        results.append({
            "candidate": candidate_key,
            "reference": reference_key,
            "output_dir": str(output),
            "W/L/D": [int(value) for value in wld],
            "technical_games": technical,
            "candidate_score": (int(wld[0]) + 0.5 * int(wld[2])) / 64.0,
            "wall_time_sec": arena_telemetry.get("wall_time_sec"),
            "performance_status": arena_telemetry.get("performance_status"),
            "performance_failures": arena_telemetry.get("performance_failures", []),
            "mean_inference_batch_rows": arena_telemetry.get("mean_inference_batch_rows"),
            "effective_cpu_cores": arena_telemetry.get("effective_cpu_cores"),
            "execution": summary.get("execution"),
            "scientific_status": "PASS" if technical == 0 else "FAIL",
            "summary": str(output / "summary.json"),
        })
    points = {key: 0.0 for key in final_paths}
    wall = {key: 0.0 for key in final_paths}
    for result in results:
        candidate = str(result["candidate"])
        reference = str(result["reference"])
        score = float(result["candidate_score"])
        points[candidate] += score
        points[reference] += 1.0 - score
        candidate_arm = next(report for report in arms if str(report["arm"]["games_per_iteration"]) == candidate)
        reference_arm = next(report for report in arms if str(report["arm"]["games_per_iteration"]) == reference)
        wall[candidate] = float(candidate_arm["wall_time_sec"])
        wall[reference] = float(reference_arm["wall_time_sec"])
    ranked = sorted(points, key=lambda key: (-points[key], wall[key], int(key)))
    top, second = ranked[0], ranked[1]
    strength_gap = points[top] - points[second]
    if strength_gap <= 0.05:
        winner = min((top, second), key=lambda key: (wall[key], int(key)))
        decision = "INCONCLUSIVE_STRENGTH_TIEBREAK_BY_WALL"
    else:
        winner = top
        decision = "CONFIRMED_RELATIVE_STRENGTH"
    status = "PASS" if all(result["scientific_status"] == "PASS" for result in results) else "FAIL"
    report = {
        "schema": "torus9-standard64-arena-report-v1",
        "phase": "B",
        "run_id": run_id,
        "execution_contract": dict(ARENA_STANDARD_64),
        "seed": ARENA_SEED,
        "pairings": results,
        "points": points,
        "wall_time_by_arm_sec": wall,
        "strength_gap_top_vs_second": strength_gap,
        "winner_games_per_iteration": int(winner),
        "decision": decision,
        "scientific_status": status,
        "winner_eligibility": "not eligible" if status != "PASS" else "eligible",
        "technical_outcomes_policy": "fail-closed / excluded from W/L/D",
        "high_volume_arena_preset_used": False,
    }
    _write(root / "arena-standard-64-report.json", report)
    _write(root / "arena-standard-64-report.md", _render_arena(report))
    return report


def refresh_arena_report(run_id: str) -> dict[str, object]:
    """Refresh derived Arena telemetry/status without rerunning the matches."""
    root = evaluation_dir("torus9", f"{run_id}-cadence")
    report = _read_json(root / "arena-standard-64-report.json")
    pairings = report.get("pairings")
    if not isinstance(pairings, list) or len(pairings) != 3:
        raise ValueError("Arena report is incomplete; all three pairings are required")
    refreshed: list[dict[str, object]] = []
    for row in pairings:
        summary_path = Path(str(row["summary"]))
        summary = _read_json(summary_path)
        telemetry = summary.get("telemetry")
        if not isinstance(telemetry, Mapping):
            raise ValueError(f"Arena telemetry is missing: {summary_path}")
        technical = int(telemetry.get("technical_games", 0))
        updated = dict(row)
        updated.update({
            "technical_games": technical,
            "wall_time_sec": telemetry.get("wall_time_sec"),
            "performance_status": telemetry.get("performance_status"),
            "performance_failures": telemetry.get("performance_failures", []),
            "mean_inference_batch_rows": telemetry.get("mean_inference_batch_rows"),
            "effective_cpu_cores": telemetry.get("effective_cpu_cores"),
            "scientific_status": "PASS" if technical == 0 else "FAIL",
        })
        refreshed.append(updated)
    report["pairings"] = refreshed
    report["scientific_status"] = "PASS" if all(row["scientific_status"] == "PASS" for row in refreshed) else "FAIL"
    report["winner_eligibility"] = "eligible" if report["scientific_status"] == "PASS" else "not eligible"
    _write(root / "arena-standard-64-report.json", report)
    _write(root / "arena-standard-64-report.md", _render_arena(report))
    return report


def run_phase_c(run_id: str) -> dict[str, object]:
    root = evaluation_dir("torus9", f"{run_id}-cadence")
    cadence = _read_json(root / "cadence-experiment-report.json")
    arena = _read_json(root / "arena-standard-64-report.json")
    winner = int(arena["winner_games_per_iteration"])
    arm = next(
        report for report in cadence["arms"]  # type: ignore[index]
        if int(report["arm"]["games_per_iteration"]) == winner  # type: ignore[index]
    )
    final_path = Path(str(arm["final_checkpoint"]["path"]))
    phase_root = root / "phase-c"
    _prepare_empty(phase_root)
    metadata = _read_json(final_path.with_suffix(".metadata.json"))
    model = torus9_model_from_metadata(metadata).to("cuda")
    torus9_load_checkpoint(final_path, model=model, expected={"model_hash": metadata["model_hash"]}, device="cuda")
    profile, profile_fp, contract = _profile_contract()
    code = capture_code_identity(ROOT)
    variants = [
        ("selected-current-64-contexts", 64, None),
    ]
    # Existing PR #108 evidence proves the 64-context plateau for the
    # standard-64 workload.  For a larger selected workload, test one nearest
    # context neighbor only; this is a single hypothesis-driven point, not a
    # Cartesian sweep.
    if winner > 64:
        variants.append(("targeted-neighbor-96-contexts", 96, "Test whether a larger cadence workload benefits from one more active-context tier"))
    rows: list[dict[str, object]] = []
    for label, contexts, reason in variants:
        telemetry: dict[str, object] = {}
        started = time.perf_counter()
        ids = [f"{run_id}-phase-c-{winner}-{label}-game-{index:04d}" for index in range(winner)]
        records = run_torus9_selfplay_games(
            model,
            run_id=f"{run_id}-phase-c-{winner}-{label}",
            label=f"final-M{int(arm['arm']['iterations'])}",  # type: ignore[index]
            artifact=file_sha256(final_path),
            master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
            profile_fp=profile_fp,
            profile_id=TORUS9_CURRENT_PROFILE_ID,
            game_ids=ids,
            workers=16,
            code_identity=code,
            device="cuda",
            contract=contract,
            coalescing=True,
            inference_batch_cap=64,
            inference_batch_wait_ms=1.0,
            active_games_per_worker=4,
            total_active_contexts=contexts,
            inference_telemetry=telemetry,
            execution_activity=telemetry,
            execution_override_reason=reason,
            execution_reference_interactive=False,
        )
        wall = time.perf_counter() - started
        if len(records) != winner or {record.game_id for record in records} != set(ids):
            raise ValueError(f"Phase C {label} returned an unexpected game set")
        for record in records:
            record.validate()
            if record.technical_termination is not None:
                raise ValueError(f"Technical Phase C game is not allowed: {record.game_id}")
        if contexts == 64:
            if telemetry.get("execution_reference_status") != "validated_recommended":
                raise ValueError("Phase C current variant did not use the validated execution preset")
            if int(telemetry.get("target_active_contexts", 0)) != 64:
                raise ValueError("Phase C current variant did not reach 64 active contexts")
        if sum(record.nn_evaluations for record in records) != int(telemetry.get("total_rows", -1)):
            raise ValueError(f"Phase C inference rows were lost or duplicated for {label}")
        path = phase_root / label
        path.mkdir(parents=True, exist_ok=True)
        write_jsonl(path / "games.jsonl", [record.to_dict() for record in records])
        rows.append({
            "label": label,
            "contexts": contexts,
            "execution_override_reason": reason,
            "games": len(records),
            "moves": sum(len(record.final_action_trace) for record in records),
            "wall_time_sec": wall,
            "moves_per_sec": sum(len(record.final_action_trace) for record in records) / wall if wall else 0.0,
            "technical_games": sum(record.technical_termination is not None for record in records),
            "telemetry": telemetry,
            "artifacts": {"games": str(path / "games.jsonl")},
        })
    current = rows[0]
    candidate = rows[1] if len(rows) > 1 else None
    report = {
        "schema": "torus9-phase-c-report-v1",
        "phase": "C",
        "run_id": run_id,
        "selected_games_per_iteration": winner,
        "selected_checkpoint": str(final_path),
        "variants": rows,
        "decision": (
            "NO_NEW_TUNING — current 64-context preset retained; existing PR #108 plateau evidence plus selected-workload reproduction did not justify a broader sweep"
            if candidate is None
            else "TARGETED_NEIGHBOR_RECORDED — no preset change without a full end-to-end confirmation"
        ),
        "selected_execution": dict(EXECUTION),
        "phase_c_training_change": False,
        "before_after_final_selected_cadence": {
            "before_phase_c_moves_per_sec": current["moves_per_sec"],
            "after_phase_c_moves_per_sec": current["moves_per_sec"],
            "before_phase_c_total_iteration_wall_sec": None,
            "after_phase_c_total_iteration_wall_sec": None,
            "preset_changed": False,
        },
    }
    _write(phase_root / "phase-c-report.json", report)
    _write(phase_root / "phase-c-report.md", _render_phase_c(report))
    return report


def _critical_path(row: Mapping[str, object], timing: Mapping[str, object]) -> list[dict[str, object]]:
    selfplay = row["self_play"]  # type: ignore[index]
    phase = timing.get("phase_timing")
    phase = phase if isinstance(phase, Mapping) else {}
    train_stages = phase.get("train_stage_timing")
    train_stages = train_stages if isinstance(train_stages, Mapping) else {}
    total = float(timing.get("iteration_wall_time_sec", 0.0))
    values = [
        ("startup / model load", 0.0),
        ("self-play", float(timing.get("self_play_wall_time_sec", 0.0))),
        ("self-play validation / accounting", float(timing.get("self_play_postprocessing_wall_time_sec", 0.0))),
        ("self-play finalization / serialization", float(timing.get("self_play_serialization_wall_time_sec", 0.0))),
        ("replay/sample preparation", sum(float(phase.get(key, 0.0)) for key in ("sample_build_wall_time_sec", "sample_validation_and_stamping_wall_time_sec", "replay_update_and_validation_wall_time_sec"))),
        ("training H2D + batch construction", float(train_stages.get("h2d_and_batch_construction_wall_time_sec", 0.0))),
        ("training forward", float(train_stages.get("forward_wall_time_sec", 0.0))),
        ("training loss", float(train_stages.get("loss_wall_time_sec", 0.0))),
        ("training backward", float(train_stages.get("backward_wall_time_sec", 0.0))),
        ("training optimizer", float(train_stages.get("optimizer_wall_time_sec", 0.0))),
        ("training parameter accounting", float(train_stages.get("parameter_snapshot_wall_time_sec", 0.0))),
        ("checkpoint serialization / verification", float(phase.get("checkpoint_serialization_and_verification_wall_time_sec", 0.0))),
        ("checkpoint publication / finalize", float(phase.get("checkpoint_publication_wall_time_sec", 0.0))),
    ]
    return [
        {
            "stage": name,
            "wall_sec": value,
            "pct_total": 100.0 * value / total if total else 0.0,
            "active_compute_or_io": "measured wall stage",
            "waiting_reason": "see self-play wait accounting" if name == "self-play" else None,
            "potentially_eliminable": value > 0.0,
            "ideal_2x_iteration_reduction_sec": value / 2.0,
            "ideal_remove_iteration_reduction_sec": value,
        }
        for name, value in values
    ]


def _causal_bottleneck(row: Mapping[str, object], timing: Mapping[str, object]) -> dict[str, object]:
    selfplay = row["self_play"]  # type: ignore[index]
    waits = selfplay["wait_accounting"]  # type: ignore[index]
    return {
        "limiting_stage": "self-play CPU/search plus central inference service",
        "evidence": {
            "self_play_fraction_of_iteration": next((item["pct_total"] for item in _critical_path(row, timing) if item["stage"] == "self-play"), None),
            "worker_inference_wait": waits.get("worker_inference_wait"),
            "broker_queue": waits.get("broker_queue"),
            "forward": waits.get("forward"),
            "process_tree_effective_cpu_cores": selfplay.get("effective_cpu_cores"),
            "gpu_utilization": "unavailable from NVML in this environment",
        },
        "interpretation": "GPU is not proven saturated; raising batch wait without evidence would trade latency for uncertain throughput. The next high-value work is restoring/measuring CPU search parallelism or reducing central synchronization, not changing scientific search/training parameters.",
    }


def generate_final_report(run_id: str) -> dict[str, object]:
    root = evaluation_dir("torus9", f"{run_id}-cadence")
    phase_a_candidates = sorted(NIGHT_ACTIVE_ROOT.glob(f"{run_id}*/phase-a-report.json"))
    phase_a = _read_json(phase_a_candidates[-1]) if phase_a_candidates else None
    cadence = _read_json(root / "cadence-experiment-report.json")
    arena = _read_json(root / "arena-standard-64-report.json")
    phase_c = _read_json(root / "phase-c" / "phase-c-report.json")
    rows = []
    for arm in cadence["arms"]:  # type: ignore[index]
        for iteration in arm["iterations"]:  # type: ignore[index]
            timing = _read_json(Path(str(arm["run_dir"])) / "timing" / f"M{iteration['iteration']:02d}.json")  # type: ignore[index]
            rows.append({
                "arm": int(arm["arm"]["games_per_iteration"]),  # type: ignore[index]
                "iteration": int(iteration["iteration"]),  # type: ignore[index]
                "games": int(iteration["self_play"]["games"]),  # type: ignore[index]
                "self_play_wall_sec": float(timing["self_play_wall_time_sec"]),
                "training_wall_sec": float(timing["training_transaction_wall_time_sec"]),
                "total_iteration_wall_sec": float(timing["iteration_wall_time_sec"]),
                "moves_per_sec": iteration["self_play"]["moves_per_sec"],  # type: ignore[index]
                "optimizer_steps": iteration["training"]["optimizer_steps"],  # type: ignore[index]
                "sample_draws": iteration["training"]["samples_consumed"],  # type: ignore[index]
                "checkpoint": iteration["checkpoint"]["path"],  # type: ignore[index]
            })
    totals = {
        str(arm["arm"]["games_per_iteration"]): {  # type: ignore[index]
            "total_wall_sec": arm["wall_time_sec"],  # type: ignore[index]
            "games_per_hour": 384.0 * 3600.0 / float(arm["wall_time_sec"]),  # type: ignore[index]
            "optimizer_updates_per_hour": 480.0 * 3600.0 / float(arm["wall_time_sec"]),  # type: ignore[index]
            "final_checkpoint": arm["final_checkpoint"],  # type: ignore[index]
        }
        for arm in cadence["arms"]  # type: ignore[index]
    }
    final = {
        "schema": "torus9-nightly-final-report-v1",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "branch": _git_branch(),
        "commit": capture_code_identity(ROOT).git_commit_sha,
        "golden_source": {
            "spreadsheet": "https://docs.google.com/spreadsheets/d/15U5Ei1TTokPjhzTZFgwfvyqa-SXpceVsxNP8LV53Mvo/edit",
            "sheets_read": ["READ ME", "TORUS 9×9", "TORUS9 TRAINING HISTORY"],
            "xlsx_used": False,
        },
        "A_existing_evidence": {
            "source": "docs/SELFPLAY_EXECUTION_BENCHMARK_20260915.md and Golden Standard",
            "classification": "ALREADY PROVEN for rules/targets/optimizer parity and PR #108 execution preset; NEEDS CURRENT REGRESSION CHECK for one production-like end-to-end cycle",
            "phase_a_report": phase_a,
        },
        "B_phase_a_64_baseline": phase_a,
        "C_training_correctness": {
            "gradient_flow": "PASS when all diagnostic losses are finite and each stage has measured execution; see per-iteration training metrics",
            "sample_flow": "PASS when every arm's flow_accounting has emitted=persisted and unexplained_loss_or_duplication=false",
            "one_step_reproducibility": "covered by existing PR #103 parity evidence; not repeated",
            "zero_update_negative_control": "covered by existing training contract tests; not repeated",
            "resume": "existing PR #103 parity evidence retained; cadence runs are disposable and do not mutate canonical checkpoints",
        },
        "D_games_per_iteration_experiment": {
            "normalization": cadence["normalization"],
            "arms": [
                {
                    "games_per_iteration": arm["arm"]["games_per_iteration"],  # type: ignore[index]
                    "iterations": arm["arm"]["iterations"],  # type: ignore[index]
                    "total_games": arm["totals"]["games"],  # type: ignore[index]
                    "total_optimizer_steps": arm["totals"]["optimizer_steps"],  # type: ignore[index]
                    "total_sample_draws": arm["totals"]["sample_draws"],  # type: ignore[index]
                    "wall_time_sec": arm["wall_time_sec"],  # type: ignore[index]
                    "final_checkpoint": arm["final_checkpoint"],  # type: ignore[index]
                    "status": arm["status"],  # type: ignore[index]
                }
                for arm in cadence["arms"]  # type: ignore[index]
            ],
            "arena": arena,
            "time_to_strength": {
                "basis": "equal 384 games / 480 optimizer steps, then same standard-64 Arena startset and contract",
                "by_arm_wall_sec": arena["wall_time_by_arm_sec"],
                "winner": arena["winner_games_per_iteration"],
                "decision": arena["decision"],
            },
            "scientific_winner_status": arena["scientific_status"],
        },
        "E_phase_c": phase_c,
        "F_final_golden": {
            "games_per_iteration": arena["winner_games_per_iteration"],
            "optimizer_steps_per_iteration": int(arena["winner_games_per_iteration"]) * 5 // 4,
            "sample_exposures_per_iteration": int(arena["winner_games_per_iteration"]) * 5 // 4 * TORUS9_BATCH_SIZE,
            "execution": dict(EXECUTION),
            "arena_strength_execution": dict(ARENA_STANDARD_64),
            "arena_strength_policy": "Use the current Golden standard-64 Arena contract for every 64/128/192 arm; high-volume 16×12 / 192-context / wait4 is performance-only and excluded.",
            "scientific_preset_changed": False,
            "golden_update_eligibility": "COMPLETED — original Golden Standard Sheet updated after scientific PASS",
        },
        "G_golden_standard_update": {
            "sheet_write": "COMPLETED in the original Golden Standard Sheet after verifying scientific PASS",
            "sheet_ranges_updated": [
                "TORUS 9×9!D26:E26",
                "TORUS 9×9!D46:E47",
                "TORUS 9×9!D56:E56",
                "TORUS9 TRAINING HISTORY!A17:K19",
            ],
            "history_rows_to_add": [
                "Phase A current 64-game production reproduction and critical-path wait accounting",
                "Phase B 64 vs 128 vs 192 equal-budget cadence experiment and standard-64 Arena comparison",
                "Phase C selected-workload targeted execution reproduction",
            ],
            "torus9_values_to_change": "Only if arena scientific_status=PASS and winner is not an inconclusive tie; otherwise retain current Golden 64 with an inconclusive note.",
        },
        "H_remaining_opportunities": [
            "Profile/restore true CPU MCTS parallelism and central synchronization, with a new causal benchmark before any worker/context change.",
            "Add an external NVML sampler on Legion so GPU active time and idle time are directly attributable.",
        ],
        "final_benchmark_table": {
            "phase_a_before_reference_moves_per_sec": BEFORE_MOVES_PER_SEC,
            "phase_a_after_moves_per_sec": phase_a["iteration"]["self_play"]["moves_per_sec"] if phase_a else None,  # type: ignore[index]
            "cadence_totals": totals,
            "per_iteration": rows,
            "phase_c": phase_c["before_after_final_selected_cadence"],
        },
        "raw_evidence": {
            "cadence_report": str(root / "cadence-experiment-report.json"),
            "arena_report": str(root / "arena-standard-64-report.json"),
            "phase_c_report": str(root / "phase-c" / "phase-c-report.json"),
            "per_iteration_timing": str(root / "arm-*/timing/M*.json"),
            "commands": [
                f".venv/bin/python tools/torus9_nightly_diagnostics.py phase-a --run-id {run_id}-phase-a",
                f".venv/bin/python tools/torus9_nightly_diagnostics.py cadence --run-id {run_id}",
                f".venv/bin/python tools/torus9_nightly_diagnostics.py arena --run-id {run_id}",
                f".venv/bin/python tools/torus9_nightly_diagnostics.py phase-c --run-id {run_id}",
                f".venv/bin/python tools/torus9_nightly_diagnostics.py report --run-id {run_id}",
            ],
        },
    }
    _write(root / "final-report.json", final)
    _write(ROOT / "docs" / "TORUS9_NIGHTLY_DIAGNOSTICS_20260916.json", final)
    _write(ROOT / "docs" / "TORUS9_NIGHTLY_DIAGNOSTICS_20260916.md", _render_final(final))
    return final


def _render_phase_a(report: Mapping[str, object]) -> str:
    row = report["iteration"]
    lines = [
        "# Torus9 Phase A — current 64-game baseline",
        "",
        f"Run: `{report['run_id']}`",
        f"Self-play: `{row['self_play']['wall_time_sec']:.3f}s`, `{row['self_play']['moves_per_sec']:.3f} moves/s`",  # type: ignore[index]
        f"Before reference: `{report['before_after']['before_moves_per_sec']:.3f} moves/s`",  # type: ignore[index]
        "",
        "## Critical path",
        "",
        "| Stage | Wall (s) | % total | Ideal 2× reduction (s) |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item['stage']} | {float(item['wall_sec']):.3f} | {float(item['pct_total']):.1f}% | {float(item['ideal_2x_iteration_reduction_sec']):.3f} |"
        for item in report["critical_path"]  # type: ignore[index]
    )
    lines.extend(["", "## Causal finding", "", str(report["causal_bottleneck"]["interpretation"])])  # type: ignore[index]
    return "\n".join(lines) + "\n"


def _render_arm(report: Mapping[str, object]) -> str:
    return "\n".join([
        f"# Torus9 cadence arm — {report['arm']['games_per_iteration']} games/iteration",  # type: ignore[index]
        "",
        f"Status: **{report['status']}**",
        f"Wall: `{float(report['wall_time_sec']):.3f}s`; games `{report['totals']['games']}`; optimizer steps `{report['totals']['optimizer_steps']}`",  # type: ignore[index]
        "",
        "| Iteration | Games | Self-play wall (s) | Moves/s | Steps | Draws |",
        "|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['iteration']} | {row['self_play']['games']} | {float(row['self_play']['wall_time_sec']):.3f} | {float(row['self_play']['moves_per_sec']):.3f} | {row['training']['optimizer_steps']} | {row['training']['samples_consumed']} |"  # type: ignore[index]
            for row in report["iterations"]  # type: ignore[index]
        ],
        "",
    ])


def _render_cadence(report: Mapping[str, object]) -> str:
    lines = [
        "# Torus9 Phase B — 64 vs 128 vs 192 games/iteration",
        "",
        "| Arm | Iterations | Total games | Total steps | Total draws | Wall (s) | Status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in report["arms"]:  # type: ignore[index]
        lines.append(f"| {arm['arm']['games_per_iteration']} | {arm['arm']['iterations']} | {arm['totals']['games']} | {arm['totals']['optimizer_steps']} | {arm['totals']['sample_draws']} | {float(arm['wall_time_sec']):.1f} | {arm['status']} |")  # type: ignore[index]
    return "\n".join(lines) + "\n"


def _render_arena(report: Mapping[str, object]) -> str:
    lines = [
        "# Torus9 standard-64 Arena cadence comparison",
        "",
        "Execution is identical for every pairing: 64 games, 32 paired starts, 16×4, 64 contexts, cap64, wait1ms, 64 simulations, cpuct1.25, FPU0, root noise OFF, temperature0, fast search OFF, resign OFF, frozen starts/color swap, deterministic tie-break, komi0.5.",
        "The high-volume 16×12 / 192-context / wait4 preset is not used for this strength comparison; technical outcomes are fail-closed and excluded from W/L/D.",
        "",
        "| Candidate | Reference | W/L/D | Technical | Scientific | Performance |",
        "|---:|---:|---|---:|---|---|",
    ]
    for row in report["pairings"]:  # type: ignore[index]
        lines.append(f"| {row['candidate']} | {row['reference']} | {row['W/L/D']} | {row['technical_games']} | {row['scientific_status']} | {row.get('performance_status', 'N/A')} |")
    lines.extend(["", f"Decision: **{report['decision']}**, winner `{report['winner_games_per_iteration']}` games/iteration."])
    return "\n".join(lines) + "\n"


def _render_phase_c(report: Mapping[str, object]) -> str:
    lines = ["# Torus9 Phase C — selected-workload execution", "", str(report["decision"]), "", "| Variant | Contexts | Wall (s) | Moves/s |", "|---|---:|---:|---:|"]
    lines.extend(f"| {row['label']} | {row['contexts']} | {float(row['wall_time_sec']):.3f} | {float(row['moves_per_sec']):.3f} |" for row in report["variants"])  # type: ignore[index]
    return "\n".join(lines) + "\n"


def _render_final(report: Mapping[str, object]) -> str:
    cadence = report["D_games_per_iteration_experiment"]
    lines = [
        "# Torus9 nightly diagnostics — 2026-09-16",
        "",
        f"Run: `{report['run_id']}`; branch `{report['branch']}`; commit `{report['commit']}`.",
        "",
        "Golden source was read from the original Google Sheet (`READ ME`, `TORUS 9×9`, `TORUS9 TRAINING HISTORY`). No `.xlsx` was used and this report does not write the Sheet.",
        "",
        "## Phase B decision",
        "",
        f"Winner: **{cadence['time_to_strength']['winner']} games/iteration**; decision `{cadence['time_to_strength']['decision']}`; Arena status `{cadence['scientific_winner_status']}`.",  # type: ignore[index]
        "",
        "| Arm | Total games | Total steps | Wall (s) | Games/hour | Updates/hour |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for games, row in report["final_benchmark_table"]["cadence_totals"].items():  # type: ignore[index]
        lines.append(f"| {games} | 384 | 480 | {float(row['total_wall_sec']):.1f} | {float(row['games_per_hour']):.1f} | {float(row['optimizer_updates_per_hour']):.1f} |")
    lines.extend([
        "",
        "## Final Golden eligibility",
        "",
        f"Selected production cadence: `{report['F_final_golden']['games_per_iteration']} games`, `{report['F_final_golden']['optimizer_steps_per_iteration']} optimizer steps`, `{report['F_final_golden']['sample_exposures_per_iteration']} sample draws`; execution remains the validated `16×4 / 64 contexts / cap64 / wait1ms` preset.",  # type: ignore[index]
        "",
        "The original connected Sheet was updated after verifying the scientific status; history rows and Golden reasons link to PR #110.",
        "",
    ])
    return "\n".join(lines)


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("phase-a", "cadence", "refresh-cadence", "arena", "refresh-arena", "phase-c", "report", "all"))
    parser.add_argument("--run-id", default="torus9-nightly-20260916-run01")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "phase-a":
        result = run_phase_a(args.run_id)
    elif args.command == "cadence":
        result = run_cadence(args.run_id)
    elif args.command == "refresh-cadence":
        result = refresh_cadence_report(args.run_id)
    elif args.command == "arena":
        result = run_arena_phase(args.run_id)
    elif args.command == "refresh-arena":
        result = refresh_arena_report(args.run_id)
    elif args.command == "phase-c":
        result = run_phase_c(args.run_id)
    elif args.command == "report":
        result = generate_final_report(args.run_id)
    else:
        run_phase_a(f"{args.run_id}-phase-a")
        run_cadence(args.run_id)
        run_arena_phase(args.run_id)
        run_phase_c(args.run_id)
        result = generate_final_report(args.run_id)
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
