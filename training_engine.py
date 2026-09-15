"""Generic execution engine for one replay/training/checkpoint generation.

The engine deliberately knows nothing about a game, a model architecture, a
loss, or an optimizer.  Those semantics belong to an adapter.  The engine
owns the transaction boundary: conversion, replay update, training, checkpoint
verification, publication, and completion marking either finish together or
leave the previous state and previous committed artifacts usable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())  # type: ignore[no-any-return]
    return str(value)


def canonical_json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def value_fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sequence_fingerprint(values: Sequence[object]) -> str:
    """Hash an ordered JSON sequence without materializing one giant string."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(canonical_json(value).encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


@dataclass
class TrainingState:
    """Explicit resumable state shared by the engine and its adapter."""

    model: Any
    optimizer: Any
    optimizer_updates: int
    samples_consumed: int
    current_generation: int
    rolling_replay: Any
    profile_identity: Mapping[str, object]
    target_identity: Mapping[str, object]
    parent_checkpoint_identity: Mapping[str, object] | None = None
    completed_games: int = 0
    adapter_state: Any = None

    @property
    def replay_state(self) -> Any:
        """Named alias used by callers that treat replay as a state component."""
        return self.rolling_replay


@dataclass(frozen=True)
class CheckpointContext:
    run_id: str
    label: str
    parent_label: str | None
    generation: int
    training_seed: int
    fresh_positions: int
    replay_positions: int
    replay_generations: tuple[int, ...]
    replay_fingerprint: str
    sampled_row_ids_fingerprint: str
    completed_games: int
    parent_checkpoint_identity: Mapping[str, object] | None
    code_identity: Any
    device: str


@dataclass(frozen=True)
class TrainingIterationResult:
    generation: int
    label: str
    fresh_positions: int
    replay_metrics: Mapping[str, object]
    training_metrics: Mapping[str, object]
    checkpoint_metadata: Mapping[str, object]
    artifacts: Mapping[str, str]
    provenance: Mapping[str, object]
    summary: Mapping[str, object] = field(default_factory=dict)

    @property
    def metrics(self) -> Mapping[str, object]:
        return self.training_metrics

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "label": self.label,
            "fresh_positions": self.fresh_positions,
            "replay": _jsonable(self.replay_metrics),
            "training": _jsonable(self.training_metrics),
            "checkpoint": _jsonable(self.checkpoint_metadata),
            "artifacts": _jsonable(self.artifacts),
            "provenance": _jsonable(self.provenance),
        }


class TrainingAdapter(Protocol):
    """Protocol implemented by a profile-specific training adapter."""

    def validate_state(self, state: TrainingState) -> None: ...

    def build_samples(self, records: Sequence[object]) -> Sequence[Mapping[str, object]]: ...

    def validate_sample(self, sample: Mapping[str, object]) -> None: ...

    def stamp_samples(
        self, samples: Sequence[Mapping[str, object]], generation: int
    ) -> Sequence[Mapping[str, object]]: ...

    def update_replay(
        self,
        replay: Any,
        generation: int,
        samples: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]: ...

    def replay_rows(self, replay: Any) -> Sequence[Mapping[str, object]]: ...

    def validate_replay(self, rows: Sequence[Mapping[str, object]]) -> None: ...

    def train(
        self,
        state: TrainingState,
        rows: Sequence[Mapping[str, object]],
        seed: int,
    ) -> Mapping[str, object]: ...

    def prepare_checkpoint(
        self,
        state: TrainingState,
        context: CheckpointContext,
        training_metrics: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def save_checkpoint(
        self,
        path: Path,
        state: TrainingState,
        metadata: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    def verify_checkpoint(
        self,
        path: Path,
        state: TrainingState,
        metadata: Mapping[str, object],
    ) -> None: ...

    def artifact_hash(self, path: Path) -> str: ...

    def snapshot_state(self, state: TrainingState) -> Any: ...

    def restore_state(self, state: TrainingState, snapshot: Any) -> None: ...

    def sync_state(self, state: TrainingState) -> None: ...


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            # Adapter replay rows are already JSON-compatible mappings.  Avoid
            # recursively copying each large observation tensor here.
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class TrainingEngine:
    """Run one adapter-defined generation behind an atomic artifact boundary."""

    def __init__(self, adapter: TrainingAdapter | None = None) -> None:
        self.adapter = adapter

    def run_iteration(
        self,
        *,
        adapter: TrainingAdapter | None = None,
        state: TrainingState,
        generation: int,
        output_dir: str | Path,
        run_id: str,
        training_seed: int,
        records: Sequence[object] | None = None,
        samples: Sequence[Mapping[str, object]] | None = None,
        label: str | None = None,
        parent_checkpoint_identity: Mapping[str, object] | None = None,
        completed_games: int | None = None,
        code_identity: Any = None,
        device: str | None = None,
        summary_extra: Mapping[str, object] | None = None,
        summary_builder: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    ) -> TrainingIterationResult:
        selected_adapter = adapter or self.adapter
        if selected_adapter is None:
            raise ValueError("TrainingEngine requires an adapter")
        if (records is None) == (samples is None):
            raise ValueError("TrainingEngine requires exactly one of records or samples")
        generation = int(generation)
        if generation <= 0:
            raise ValueError("Training generation must be positive")
        if int(training_seed) <= 0:
            raise ValueError("Training seed must be a positive explicit value")
        selected_adapter.validate_state(state)

        root = Path(output_dir)
        replay_dir = root / "replay"
        checkpoint_dir = root / "checkpoints"
        training_dir = root / "training"
        fresh_final = replay_dir / f"iter-{generation:02d}-fresh.jsonl"
        rolling_final = replay_dir / f"rolling-after-{generation:02d}.jsonl"
        checkpoint_final = checkpoint_dir / f"M{generation}.pt"
        checkpoint_metadata_final = checkpoint_final.with_suffix(".metadata.json")
        training_final = training_dir / f"iter-{generation:02d}.json"
        summary_final = root / f"iter-{generation:02d}-summary.json"
        marker_final = root / f"generation-{generation:02d}.complete.json"
        final_paths = (
            fresh_final,
            rolling_final,
            checkpoint_final,
            checkpoint_metadata_final,
            training_final,
            summary_final,
            marker_final,
        )
        if any(path.exists() for path in final_paths):
            raise FileExistsError(
                f"Training generation {generation} already has published artifacts; refusing overwrite"
            )

        fresh_tmp = replay_dir / f".iter-{generation:02d}.tmp-fresh.jsonl"
        rolling_tmp = replay_dir / f".rolling-after-{generation:02d}.tmp.jsonl"
        checkpoint_tmp = checkpoint_dir / f".M{generation}.tmp.pt"
        training_tmp = training_dir / f".iter-{generation:02d}.tmp.json"
        summary_tmp = root / f".iter-{generation:02d}-summary.tmp.json"
        marker_tmp = root / f".generation-{generation:02d}.complete.tmp.json"
        checkpoint_metadata_tmp = checkpoint_tmp.with_suffix(".metadata.json")
        temporary_paths = (
            fresh_tmp,
            rolling_tmp,
            checkpoint_tmp,
            checkpoint_metadata_tmp,
            training_tmp,
            summary_tmp,
            marker_tmp,
        )

        snapshot = selected_adapter.snapshot_state(state)
        committed = False
        try:
            if records is not None:
                source_samples = tuple(selected_adapter.build_samples(tuple(records)))
            else:
                source_samples = tuple(dict(sample) for sample in samples or ())
            if not source_samples:
                raise ValueError("Training generation produced no replay samples")
            for sample in source_samples:
                selected_adapter.validate_sample(sample)
            stamped = tuple(selected_adapter.stamp_samples(source_samples, generation))
            if not stamped:
                raise ValueError("Training generation produced no stamped replay samples")
            for sample in stamped:
                selected_adapter.validate_sample(sample)

            replay_metrics = dict(selected_adapter.update_replay(state.rolling_replay, generation, stamped))
            replay_rows = tuple(selected_adapter.replay_rows(state.rolling_replay))
            selected_adapter.validate_replay(replay_rows)
            replay_fingerprint = sequence_fingerprint(replay_rows)

            train_started = time.perf_counter()
            training_metrics = dict(selected_adapter.train(state, replay_rows, int(training_seed)))
            training_metrics.setdefault("training_wall_time_sec", time.perf_counter() - train_started)
            selected_adapter.sync_state(state)

            sampled_ids = training_metrics.get("sampled_replay_row_ids", ())
            sampled_row_ids_fingerprint = value_fingerprint(sampled_ids)
            parent = parent_checkpoint_identity or state.parent_checkpoint_identity
            context = CheckpointContext(
                run_id=str(run_id),
                label=label or f"M{generation}",
                parent_label=(
                    str(parent.get("label"))
                    if isinstance(parent, Mapping) and parent.get("label") is not None
                    else (f"M{generation - 1}" if generation > 1 else None)
                ),
                generation=generation,
                training_seed=int(training_seed),
                fresh_positions=len(stamped),
                replay_positions=len(replay_rows),
                replay_generations=tuple(
                    sorted(
                        {
                            int(row["source_generation"])
                            for row in replay_rows
                            if str(row.get("source_generation", "")).lstrip("-").isdigit()
                        }
                    )
                ),
                replay_fingerprint=replay_fingerprint,
                sampled_row_ids_fingerprint=sampled_row_ids_fingerprint,
                completed_games=(
                    int(completed_games)
                    if completed_games is not None
                    else int(state.completed_games)
                ),
                parent_checkpoint_identity=parent,
                code_identity=code_identity,
                device=str(device if device is not None else getattr(state.model, "device", "cpu")),
            )
            metadata = dict(selected_adapter.prepare_checkpoint(state, context, training_metrics))

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(fresh_tmp, stamped)
            _write_jsonl(rolling_tmp, replay_rows)
            saved_metadata = dict(selected_adapter.save_checkpoint(checkpoint_tmp, state, metadata))
            if not checkpoint_metadata_tmp.is_file():
                raise RuntimeError("Training adapter did not publish checkpoint metadata sidecar")
            selected_adapter.verify_checkpoint(checkpoint_tmp, state, saved_metadata)
            _write_json(training_tmp, training_metrics)

            artifacts_tmp = {
                "fresh_replay": str(fresh_tmp),
                "rolling_replay": str(rolling_tmp),
                "checkpoint": str(checkpoint_tmp),
                "checkpoint_metadata": str(checkpoint_metadata_tmp),
                "training_metrics": str(training_tmp),
                "iteration_summary": str(summary_tmp),
                "completion_marker": str(marker_final),
            }
            provenance = {
                "run_id": str(run_id),
                "generation": generation,
                "label": context.label,
                "training_seed": int(training_seed),
                "input_checkpoint": _jsonable(parent),
                "replay_generations": list(context.replay_generations),
                "replay_positions": len(replay_rows),
                "replay_fingerprint": replay_fingerprint,
                "sampled_row_ids_fingerprint": sampled_row_ids_fingerprint,
                "optimizer_updates": int(state.optimizer_updates),
                "samples_consumed": int(state.samples_consumed),
                "output_model_hash": saved_metadata.get("model_hash"),
            }
            checkpoint_artifact_hash = selected_adapter.artifact_hash(checkpoint_tmp)
            checkpoint_summary = dict(saved_metadata)
            checkpoint_summary.update({
                "path": str(checkpoint_final),
                "metadata_path": str(checkpoint_metadata_final),
                "artifact_sha256": checkpoint_artifact_hash,
                "optimizer_state_present": True,
            })
            summary: dict[str, object] = {
                "iteration": generation,
                "label": context.label,
                "fresh_positions": len(stamped),
                "replay": replay_metrics,
                "training": training_metrics,
                "checkpoint": checkpoint_summary,
                "provenance": provenance,
            }
            if summary_extra:
                summary.update(dict(summary_extra))
            if summary_builder is not None:
                summary = dict(summary_builder(summary))
            _write_json(summary_tmp, summary)

            # Compute hashes before publication.  The marker is the commit
            # record; every preceding artifact can be discarded if a failure
            # occurs before the marker is atomically replaced.
            marker = {
                "schema": "training-generation-commit-v1",
                "run_id": str(run_id),
                "generation": generation,
                "label": context.label,
                "fresh_replay_sha256": selected_adapter.artifact_hash(fresh_tmp),
                "rolling_replay_sha256": selected_adapter.artifact_hash(rolling_tmp),
                "checkpoint_sha256": checkpoint_artifact_hash,
                "checkpoint_metadata_sha256": selected_adapter.artifact_hash(checkpoint_metadata_tmp),
                "training_metrics_sha256": selected_adapter.artifact_hash(training_tmp),
                "summary_sha256": selected_adapter.artifact_hash(summary_tmp),
                "model_hash": saved_metadata.get("model_hash"),
                "replay_fingerprint": replay_fingerprint,
            }
            _write_json(marker_tmp, marker)

            for source, target in (
                (fresh_tmp, fresh_final),
                (rolling_tmp, rolling_final),
                (checkpoint_tmp, checkpoint_final),
                (checkpoint_metadata_tmp, checkpoint_metadata_final),
                (training_tmp, training_final),
                (summary_tmp, summary_final),
                (marker_tmp, marker_final),
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
            committed = True
            state.current_generation = generation
            state.completed_games = context.completed_games
            state.parent_checkpoint_identity = {
                "label": context.label,
                "path": str(checkpoint_final),
                "metadata_path": str(checkpoint_metadata_final),
                "model_hash": saved_metadata.get("model_hash"),
                "artifact_sha256": selected_adapter.artifact_hash(checkpoint_final),
            }
            selected_adapter.sync_state(state)
            artifacts = {
                "fresh_replay": str(fresh_final),
                "rolling_replay": str(rolling_final),
                "checkpoint": str(checkpoint_final),
                "checkpoint_metadata": str(checkpoint_metadata_final),
                "training_metrics": str(training_final),
                "iteration_summary": str(summary_final),
                "completion_marker": str(marker_final),
            }
            return TrainingIterationResult(
                generation=generation,
                label=context.label,
                fresh_positions=len(stamped),
                replay_metrics=replay_metrics,
                training_metrics=training_metrics,
                checkpoint_metadata=saved_metadata,
                artifacts=artifacts,
                provenance=provenance,
                summary=summary,
            )
        except Exception:
            if not committed:
                selected_adapter.restore_state(state, snapshot)
                for path in temporary_paths + final_paths:
                    # A final path can only be ours here because all final
                    # paths were checked before the transaction began.
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
            raise
