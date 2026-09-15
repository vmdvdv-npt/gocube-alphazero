"""Cube training adapter for the universal :mod:`training_engine`.

This module supplies Cube's cumulative replay, deterministic sample budget,
loss/checkpoint semantics and resume state.  The numerical training primitive
continues to live in :mod:`gocube_golden.cube_training`; there is only one
Cube policy/value update implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import copy
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

torch = importlib.import_module("torch")

from training_engine import (
    CheckpointContext,
    TrainingEngine,
    TrainingIterationResult,
    TrainingState,
    sequence_fingerprint,
    value_fingerprint,
)

from .cube_contract import CUBE_PROFILE_ID, load_profile, profile_fingerprint
from .cube_neural import (
    CUBE_ACTION_COUNT,
    CUBE_OBSERVATION_FINGERPRINT,
    CUBE_OBSERVATION_SCHEMA_ID,
    CUBE_OBSERVATION_SCHEMA_VERSION,
    GoldenCubeGraphNetV1,
    cube_count_parameters,
    cube_model_hash,
)
from .cube_topology import (
    CUBE4_GEOMETRY_FINGERPRINT,
    CUBE4_TOPOLOGY,
    CUBE4_TOPOLOGY_ID,
    GEOMETRY_SCHEMA_ID,
)
from .cube_training import (
    CUBE_SELFPLAY_CONTRACT_ID,
    CUBE_TARGET_CONTRACT_ID,
    CUBE_TARGET_FINGERPRINT,
    CubeSelfPlayGameRecord,
    CubeTrainingSample,
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    build_cube_replay_samples,
    cube_load_checkpoint,
    cube_replay_batches,
    cube_save_checkpoint,
    train_cube_batch_schedule,
)
from .provenance import CodeIdentity, capture_code_identity, file_sha256


CUBE_BATCH_SIZE = 64
CUBE_GAMES_PER_CHUNK = 128
CUBE_CHUNKS = 4
CUBE_MODEL_INIT_SEED = 2026091401
CUBE_LEARNING_RATE = 0.001
CUBE_STAGE5_BASE_COMMIT = "1b775d15659286b8d6de6506a190dc23a1818b28"

_SAMPLE_FIELDS = frozenset(field.name for field in fields(CubeTrainingSample))


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


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Cube replay row {line_number} is not an object: {path}")
            rows.append(value)
    return tuple(rows)


def _as_sample(row: Mapping[str, object] | CubeTrainingSample) -> CubeTrainingSample:
    if isinstance(row, CubeTrainingSample):
        return row
    if not isinstance(row, Mapping):
        raise TypeError("Cube replay row must be a mapping or CubeTrainingSample")
    missing = sorted(field for field in _SAMPLE_FIELDS if field not in row)
    if missing:
        raise ValueError("Cube replay sample is missing: " + ", ".join(missing))
    return CubeTrainingSample(**{field: row[field] for field in _SAMPLE_FIELDS})  # type: ignore[arg-type]


def _adam_step(optimizer: torch.optim.Optimizer) -> int:
    steps: list[int] = []
    for value in optimizer.state.values():
        step = value.get("step") if isinstance(value, Mapping) else None
        if step is None:
            continue
        steps.append(int(step.item()) if hasattr(step, "item") else int(step))
    return max(steps, default=0)


@dataclass
class CubeCumulativeReplay:
    """Unbounded, generation-ordered Cube replay state."""

    _rows: list[dict[str, object]]
    _last_generation: int = 0

    def __init__(self, rows: Sequence[Mapping[str, object]] = (), last_generation: int = 0) -> None:
        self._rows = [dict(row) for row in rows]
        self._last_generation = int(last_generation)

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(row) for row in self._rows)

    @property
    def last_generation(self) -> int:
        return int(self._last_generation)

    @property
    def total_evictions(self) -> int:
        return 0

    @classmethod
    def from_persisted_rows(
        cls, rows: Sequence[Mapping[str, object]], *, last_generation: int | None = None
    ) -> "CubeCumulativeReplay":
        copied = [dict(row) for row in rows]
        ids = [str(row.get("replay_row_id", "")) for row in copied]
        if any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("Cube cumulative replay row IDs are missing or duplicated")
        generations = [int(row.get("source_generation", 0)) for row in copied]
        if any(generation <= 0 for generation in generations):
            raise ValueError("Cube cumulative replay generation is malformed")
        if generations != sorted(generations):
            raise ValueError("Cube cumulative replay ordering is not generation-stable")
        inferred = generations[-1] if generations else 0
        if last_generation is not None and int(last_generation) != inferred:
            raise ValueError("Cube cumulative replay last generation mismatch")
        return cls(copied, inferred)

    def append_generation(
        self, generation: int, samples: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        generation = int(generation)
        if generation <= self.last_generation:
            raise ValueError("Cube replay generations must increase strictly")
        rows = [dict(sample) for sample in samples]
        ids = [str(row.get("replay_row_id", "")) for row in rows]
        if any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("Cube replay rows require unique deterministic replay_row_id values")
        existing = {str(row.get("replay_row_id", "")) for row in self._rows}
        if existing.intersection(ids):
            raise ValueError("Cube cumulative replay contains a duplicate row ID")
        if any(int(row.get("source_generation", 0)) != generation for row in rows):
            raise ValueError("Cube replay row generation disagrees with append generation")
        self._rows.extend(rows)
        self._last_generation = generation
        return {
            "generation_added": generation,
            "new_positions": len(rows),
            "rolling_buffer_positions": len(self._rows),
            "replay_policy": "cumulative",
            "evicted_positions": 0,
            "eviction": False,
        }


class CubeTrainingAdapter:
    """Current Cube training science behind the generic ``TrainingEngine``."""

    def __init__(
        self,
        profile: Mapping[str, object] | None = None,
        *,
        code_identity: CodeIdentity | None = None,
        model_init_seed: int = CUBE_MODEL_INIT_SEED,
        selfplay_contract_fingerprint: str | None = None,
        base_commit: str = CUBE_STAGE5_BASE_COMMIT,
    ) -> None:
        self.profile = load_profile() if profile is None else dict(profile)
        from .cube_contract import validate_profile

        validate_profile(self.profile)
        self.profile_fingerprint = profile_fingerprint(self.profile)
        self.code_identity = code_identity
        self.model_init_seed = int(model_init_seed)
        self.base_commit = str(base_commit)
        self.selfplay_contract_fingerprint = (
            str(selfplay_contract_fingerprint)
            if selfplay_contract_fingerprint is not None
            else DEFAULT_CUBE_SELFPLAY_CONTRACT.fingerprint
        )
        self.training_profile = self.profile["training"]
        self._pending_new_positions = 0
        self.target_identity = {
            "contract_id": CUBE_TARGET_CONTRACT_ID,
            "fingerprint": CUBE_TARGET_FINGERPRINT,
            "perspective": "side-to-move",
        }

    @property
    def profile_identity(self) -> Mapping[str, object]:
        return {
            "profile_id": CUBE_PROFILE_ID,
            "profile_fingerprint": self.profile_fingerprint,
            "architecture_id": "GoldenCubeGraphNetV1",
            "komi": 0.5,
        }

    def create_state(
        self,
        model: GoldenCubeGraphNetV1,
        *,
        run_id: str,
        replay: CubeCumulativeReplay | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        parent_checkpoint_identity: Mapping[str, object] | None = None,
        completed_games: int = 0,
        current_generation: int = 0,
        optimizer_updates: int = 0,
        samples_consumed: int = 0,
    ) -> TrainingState:
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=CUBE_LEARNING_RATE, weight_decay=0.0)
        if replay is None:
            replay = CubeCumulativeReplay()
        state = TrainingState(
            model=model,
            optimizer=optimizer,
            optimizer_updates=int(optimizer_updates),
            samples_consumed=int(samples_consumed),
            current_generation=int(current_generation),
            rolling_replay=replay,
            profile_identity=self.profile_identity,
            target_identity=self.target_identity,
            parent_checkpoint_identity=(
                dict(parent_checkpoint_identity) if parent_checkpoint_identity is not None else None
            ),
            completed_games=int(completed_games),
        )
        self.validate_state(state)
        return state

    def save_initial_checkpoint(
        self,
        path: str | Path,
        state: TrainingState,
        *,
        run_id: str,
        label: str = "M0",
        completed_games: int = 0,
        device: str = "cpu",
        code_identity: CodeIdentity | None = None,
    ) -> Mapping[str, object]:
        self.validate_state(state)
        if state.optimizer_updates != 0 or state.samples_consumed != 0 or state.rolling_replay.rows:
            raise ValueError("Initial Cube checkpoint requires a zero training clock and empty replay")
        metadata = self._checkpoint_metadata(
            state,
            run_id=str(run_id),
            label=str(label),
            parent=None,
            completed_games=int(completed_games),
            replay_positions=0,
            optimizer_updates=0,
            samples_consumed=0,
            code_identity=code_identity or self.code_identity or capture_code_identity(),
            device=str(device),
            replay_fingerprint=value_fingerprint(()),
            sampled_row_ids_fingerprint=value_fingerprint(()),
            training_seed=0,
            replay_generations=(),
        )
        return cube_save_checkpoint(path, model=state.model, optimizer=None, metadata=metadata)

    def validate_state(self, state: TrainingState) -> None:
        if not isinstance(state.rolling_replay, CubeCumulativeReplay):
            raise TypeError("Cube training requires CubeCumulativeReplay")
        if not isinstance(state.model, GoldenCubeGraphNetV1):
            raise TypeError("Cube training requires GoldenCubeGraphNetV1")
        if state.model.architecture_id != "GoldenCubeGraphNetV1":
            raise ValueError("Cube training model architecture mismatch")
        if state.profile_identity.get("profile_id") != CUBE_PROFILE_ID or state.profile_identity.get("profile_fingerprint") != self.profile_fingerprint:
            raise ValueError("Cube training profile identity mismatch")
        if state.target_identity.get("fingerprint") != CUBE_TARGET_FINGERPRINT:
            raise ValueError("Cube training target identity mismatch")
        if not isinstance(state.optimizer, torch.optim.Adam):
            raise TypeError("Cube training requires Adam")
        if state.optimizer.param_groups[0]["lr"] != CUBE_LEARNING_RATE or state.optimizer.param_groups[0]["weight_decay"] != 0.0:
            raise ValueError("Cube Adam optimizer contract drift")
        if int(state.optimizer_updates) < 0 or int(state.samples_consumed) < 0:
            raise ValueError("Cube training clock cannot be negative")
        if _adam_step(state.optimizer) != int(state.optimizer_updates):
            raise ValueError("Cube Adam step does not match training clock")

    def build_samples(self, records: Sequence[object]) -> Sequence[Mapping[str, object]]:
        ordered = sorted(tuple(records), key=lambda record: str(getattr(record, "game_id", "")))
        samples: list[Mapping[str, object]] = []
        for record in ordered:
            if not isinstance(record, CubeSelfPlayGameRecord):
                raise TypeError("Cube training accepts CubeSelfPlayGameRecord values only")
            record.validate(expected_contract_fingerprint=self.selfplay_contract_fingerprint)
            if record.profile_id != CUBE_PROFILE_ID or record.profile_fingerprint != self.profile_fingerprint:
                raise ValueError("Cube training record profile identity mismatch")
            if record.selfplay_contract_id != CUBE_SELFPLAY_CONTRACT_ID:
                raise ValueError("Cube training record self-play contract mismatch")
            if record.technical_termination is not None:
                continue
            for sample in build_cube_replay_samples(
                record,
                expected_contract_fingerprint=self.selfplay_contract_fingerprint,
            ):
                row = sample.to_dict()
                self.validate_sample(row)
                samples.append(row)
        return tuple(samples)

    def validate_sample(self, sample: Mapping[str, object]) -> None:
        if not sample.get("run_id") or not sample.get("game_id"):
            raise ValueError("Cube replay sample provenance is incomplete")
        _as_sample(sample).validate(
            expected_contract_fingerprint=self.selfplay_contract_fingerprint
        )

    def stamp_samples(
        self, samples: Sequence[Mapping[str, object]], generation: int
    ) -> Sequence[Mapping[str, object]]:
        generation = int(generation)
        stamped: list[Mapping[str, object]] = []
        for position, sample in enumerate(samples):
            row = dict(sample)
            if row.get("source_generation") is not None and int(row["source_generation"]) != generation:
                raise ValueError("Cube replay sample source generation disagrees with iteration")
            expected = f"M{generation}:{row.get('game_id')}:{row.get('ply')}:{position}"
            if row.get("replay_row_id") is not None and str(row["replay_row_id"]) != expected:
                raise ValueError("Cube replay row ID is not deterministic")
            row["replay_row_id"] = expected
            row["source_generation"] = generation
            stamped.append(row)
        return tuple(stamped)

    def update_replay(
        self,
        replay: CubeCumulativeReplay,
        generation: int,
        samples: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        for sample in samples:
            self.validate_sample(sample)
        self._pending_new_positions = len(samples)
        return replay.append_generation(int(generation), samples)

    def replay_rows(self, replay: CubeCumulativeReplay) -> Sequence[Mapping[str, object]]:
        return replay.rows

    def validate_replay(self, rows: Sequence[Mapping[str, object]]) -> None:
        previous_generation = 0
        row_ids: set[str] = set()
        for row in rows:
            generation = int(row.get("source_generation", 0))
            if generation <= 0 or generation < previous_generation:
                raise ValueError("Cube replay generation ordering drift")
            row_id = str(row.get("replay_row_id", ""))
            if not row_id or row_id in row_ids:
                raise ValueError("Cube replay row IDs are missing or duplicated")
            self.validate_sample(row)
            previous_generation = generation
            row_ids.add(row_id)

    def train(
        self,
        state: TrainingState,
        rows: Sequence[Mapping[str, object]],
        seed: int,
    ) -> Mapping[str, object]:
        self.validate_state(state)
        if not rows:
            raise ValueError("Cube training requires a non-empty cumulative replay")
        new_positions = int(self._pending_new_positions)
        if new_positions <= 0:
            latest = max(int(row.get("source_generation", 0)) for row in rows)
            new_positions = sum(int(row.get("source_generation", 0)) == latest for row in rows)
        if not 0 < new_positions <= len(rows):
            raise ValueError("Cube training fresh-position budget is invalid")
        batches = cube_replay_batches(
            len(rows),
            new_positions,
            seed=int(seed),
            batch_size=CUBE_BATCH_SIZE,
        )
        samples = tuple(_as_sample(row) for row in rows)
        optimizer, core_metrics = train_cube_batch_schedule(
            state.model,
            samples,
            batches,
            learning_rate=CUBE_LEARNING_RATE,
            weight_decay=0.0,
            optimizer=state.optimizer,
            update_offset=int(state.optimizer_updates),
            sample_offset=int(state.samples_consumed),
        )
        state.optimizer = optimizer
        state.optimizer_updates = int(core_metrics["updates"])
        state.samples_consumed = int(core_metrics["cumulative_samples"])
        selected_indices = tuple(index for batch in batches for index in batch)
        sampled_ids = tuple(str(rows[index].get("replay_row_id", index)) for index in selected_indices)
        metrics = dict(core_metrics)
        metrics.update({
            "training_seed": int(seed),
            "new_positions": new_positions,
            "optimizer_steps": len(batches),
            "samples_consumed": new_positions,
            "sampled_indices": selected_indices,
            "sampled_replay_row_ids": sampled_ids,
            "sampled_row_ids_fingerprint": value_fingerprint(sampled_ids),
            "sampling_without_replacement": True,
            "replay_policy": "cumulative",
            "eviction": False,
        })
        if int(metrics["exact_samples_consumed"]) != new_positions:
            raise ValueError("Cube sample exposure budget drift")
        self._pending_new_positions = 0
        return metrics

    def prepare_checkpoint(
        self,
        state: TrainingState,
        context: CheckpointContext,
        training_metrics: Mapping[str, object],
    ) -> Mapping[str, object]:
        code = context.code_identity or self.code_identity or capture_code_identity()
        return self._checkpoint_metadata(
            state,
            run_id=context.run_id,
            label=context.label,
            parent=context.parent_label,
            completed_games=context.completed_games,
            replay_positions=context.replay_positions,
            optimizer_updates=state.optimizer_updates,
            samples_consumed=state.samples_consumed,
            code_identity=code,
            device=context.device,
            replay_fingerprint=context.replay_fingerprint,
            sampled_row_ids_fingerprint=context.sampled_row_ids_fingerprint,
            training_seed=context.training_seed,
            replay_generations=context.replay_generations,
            parent_identity=context.parent_checkpoint_identity,
            training_metrics=training_metrics,
        )

    def _checkpoint_metadata(
        self,
        state: TrainingState,
        *,
        run_id: str,
        label: str,
        parent: str | None,
        completed_games: int,
        replay_positions: int,
        optimizer_updates: int,
        samples_consumed: int,
        code_identity: CodeIdentity,
        device: str,
        replay_fingerprint: str,
        sampled_row_ids_fingerprint: str,
        training_seed: int,
        replay_generations: Sequence[int],
        parent_identity: Mapping[str, object] | None = None,
        training_metrics: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        profile = self.profile
        metadata: dict[str, object] = {
            "checkpoint_schema_version": 1,
            "checkpoint_label": str(label),
            "architecture_id": "GoldenCubeGraphNetV1",
            "architecture_config": state.model.architecture_config,
            "rules_id": "graph-area-v1",
            "rules_profile_id": "graph-area-v1",
            "rules_fingerprint": profile["rules"]["fingerprint"],  # type: ignore[index]
            "topology_id": CUBE4_TOPOLOGY_ID,
            "topology_fingerprint": CUBE4_TOPOLOGY.fingerprint,
            "geometry_schema_id": GEOMETRY_SCHEMA_ID,
            "geometry_fingerprint": CUBE4_GEOMETRY_FINGERPRINT,
            "board_size": [4, 4, 6],
            "point_count": 96,
            "action_count": CUBE_ACTION_COUNT,
            "point_ordering_fingerprint": profile["topology"]["point_ordering_fingerprint"],  # type: ignore[index]
            "komi": 0.5,
            "observation_schema_id": CUBE_OBSERVATION_SCHEMA_ID,
            "observation_schema_version": CUBE_OBSERVATION_SCHEMA_VERSION,
            "observation_shape": [15, 96],
            "observation_fingerprint": CUBE_OBSERVATION_FINGERPRINT,
            "target_contract_id": CUBE_TARGET_CONTRACT_ID,
            "target_contract_version": 1,
            "target_fingerprint": CUBE_TARGET_FINGERPRINT,
            "value_head_semantics": "side-to-move:[WIN,DRAW,LOSS]",
            "network_heads_and_shapes": {"policy": [CUBE_ACTION_COUNT], "value": [3]},
            "profile_id": CUBE_PROFILE_ID,
            "profile_fingerprint": self.profile_fingerprint,
            "training_profile_id": CUBE_PROFILE_ID,
            "training_profile_fingerprint": self.profile_fingerprint,
            "selfplay_contract_id": CUBE_SELFPLAY_CONTRACT_ID,
            "selfplay_contract_fingerprint": self.selfplay_contract_fingerprint,
            "parent_or_source_run_identity": parent or run_id,
            "parent_model_hash": (
                parent_identity.get("model_hash")
                if isinstance(parent_identity, Mapping)
                else None
            ),
            "parent_checkpoint_identity": dict(parent_identity) if isinstance(parent_identity, Mapping) else None,
            "run_id": str(run_id),
            "parent_label": parent,
            "completed_games": int(completed_games),
            "cumulative_replay_positions": int(replay_positions),
            "replay_row_count": int(replay_positions),
            "replay_generations": [int(value) for value in replay_generations],
            "replay_fingerprint": replay_fingerprint,
            "optimizer_updates": int(optimizer_updates),
            "train_samples_consumed": int(samples_consumed),
            "adam_step": int(optimizer_updates),
            "training_seed": int(training_seed),
            "fresh_replay_positions": int(replay_positions if training_metrics is None else training_metrics.get("new_positions", 0)),
            "sampled_row_ids_fingerprint": sampled_row_ids_fingerprint,
            "model_initialization_seed": self.model_init_seed,
            "model_init_seed": self.model_init_seed,
            "base_commit": self.base_commit,
            "git_commit": code_identity.git_commit_sha,
            "git_tree": code_identity.git_tree_sha,
            "git_worktree_clean": code_identity.working_tree_clean,
            "device": str(device),
            "model_parameter_count": cube_count_parameters(state.model),
            "scientific_contract": {
                "optimizer": "Adam",
                "learning_rate": CUBE_LEARNING_RATE,
                "weight_decay": 0.0,
                "batch_size": CUBE_BATCH_SIZE,
                "games_per_chunk": CUBE_GAMES_PER_CHUNK,
                "chunks": CUBE_CHUNKS,
                "samples_per_new_position": 1.0,
                "replay_policy": "cumulative",
                "eviction": False,
                "prioritized_replay": False,
                "reanalysis": False,
                "policy_loss": "cross_entropy",
                "value_loss": "cross_entropy",
                "komi": 0.5,
            },
            "execution_only_parameters": {
                "workers": "not checkpoint semantic",
                "active_games_per_worker": "not checkpoint semantic",
                "total_active_contexts": "not checkpoint semantic",
                "inference_batch_cap": "not checkpoint semantic",
                "inference_batch_wait_ms": "not checkpoint semantic",
            },
        }
        return metadata

    def save_checkpoint(
        self, path: Path, state: TrainingState, metadata: Mapping[str, object]
    ) -> Mapping[str, object]:
        return cube_save_checkpoint(path, model=state.model, optimizer=state.optimizer, metadata=metadata)

    def artifact_hash(self, path: Path) -> str:
        return file_sha256(path)

    def verify_checkpoint(
        self, path: Path, state: TrainingState, metadata: Mapping[str, object]
    ) -> None:
        if not path.is_file() or not path.with_suffix(".metadata.json").is_file():
            raise ValueError("Cube checkpoint or metadata sidecar is missing")
        sidecar = _read_json(path.with_suffix(".metadata.json"))
        for key in (
            "checkpoint_schema_version",
            "architecture_id",
            "topology_fingerprint",
            "geometry_fingerprint",
            "komi",
            "profile_id",
            "profile_fingerprint",
            "training_profile_id",
            "training_profile_fingerprint",
            "target_fingerprint",
            "model_hash",
            "optimizer_updates",
            "train_samples_consumed",
            "adam_step",
        ):
            if sidecar.get(key) != metadata.get(key):
                raise ValueError(f"Cube checkpoint metadata sidecar mismatch for {key}")
        model = GoldenCubeGraphNetV1(hidden=state.model.hidden, blocks=state.model.blocks_count)
        optimizer = torch.optim.Adam(model.parameters(), lr=CUBE_LEARNING_RATE, weight_decay=0.0)
        loaded = cube_load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected={
                "model_hash": metadata["model_hash"],
                "training_profile_id": CUBE_PROFILE_ID,
                "training_profile_fingerprint": self.profile_fingerprint,
            },
            device=next(state.model.parameters()).device,
        )
        if loaded.get("model_hash") != cube_model_hash(model):
            raise ValueError("Cube checkpoint model hash verification failed")
        if _adam_step(optimizer) != int(metadata["optimizer_updates"]):
            raise ValueError("Cube checkpoint Adam state does not match metadata")

    def _validate_checkpoint_metadata(self, metadata: Mapping[str, object], *, require_optimizer: bool) -> None:
        profile_id = metadata.get("profile_id", metadata.get("training_profile_id"))
        profile_fp = metadata.get("profile_fingerprint", metadata.get("training_profile_fingerprint"))
        required = (
            "checkpoint_schema_version",
            "checkpoint_label",
            "architecture_id",
            "topology_fingerprint",
            "geometry_fingerprint",
            "point_count",
            "action_count",
            "komi",
            "observation_fingerprint",
            "target_contract_id",
            "target_fingerprint",
            "network_heads_and_shapes",
            "model_hash",
            "optimizer_updates",
            "train_samples_consumed",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError("Cube checkpoint metadata is incomplete: " + ", ".join(missing))
        if metadata["checkpoint_schema_version"] != 1 or metadata["architecture_id"] != "GoldenCubeGraphNetV1":
            raise ValueError("Cube checkpoint schema/architecture mismatch")
        if profile_id != CUBE_PROFILE_ID or profile_fp != self.profile_fingerprint:
            raise ValueError("Cube checkpoint profile mismatch")
        if metadata["topology_fingerprint"] != CUBE4_TOPOLOGY.fingerprint or metadata["geometry_fingerprint"] != CUBE4_GEOMETRY_FINGERPRINT:
            raise ValueError("Cube checkpoint topology/geometry mismatch")
        if metadata["point_count"] != 96 or metadata["action_count"] != CUBE_ACTION_COUNT or metadata["komi"] != 0.5:
            raise ValueError("Cube checkpoint shape/komi mismatch")
        if metadata["observation_fingerprint"] != CUBE_OBSERVATION_FINGERPRINT or metadata["target_fingerprint"] != CUBE_TARGET_FINGERPRINT:
            raise ValueError("Cube checkpoint scientific fingerprint mismatch")
        if require_optimizer and int(metadata["optimizer_updates"]) <= 0:
            raise ValueError("Cube resume requires a positive optimizer update count")

    def load_state(
        self,
        checkpoint_path: str | Path,
        *,
        replay_path: str | Path | None = None,
        device: str = "cpu",
    ) -> TrainingState:
        checkpoint_path = Path(checkpoint_path)
        metadata_path = checkpoint_path.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            raise ValueError("Cube checkpoint metadata sidecar is missing")
        metadata = _read_json(metadata_path)
        updates = int(metadata.get("optimizer_updates", 0))
        self._validate_checkpoint_metadata(metadata, require_optimizer=updates > 0)
        architecture = metadata.get("architecture_config")
        hidden = int(architecture.get("hidden", 64)) if isinstance(architecture, Mapping) else 64
        blocks = int(architecture.get("blocks", 4)) if isinstance(architecture, Mapping) else 4
        model = GoldenCubeGraphNetV1(hidden=hidden, blocks=blocks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CUBE_LEARNING_RATE, weight_decay=0.0)
        cube_load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer if updates > 0 else None,
            expected={"model_hash": metadata["model_hash"]},
            device=device,
        )
        if replay_path is None:
            if int(metadata.get("cumulative_replay_positions", metadata.get("replay_row_count", 0))) != 0:
                raise ValueError("Cube resume replay path is required for a trained checkpoint")
            replay = CubeCumulativeReplay()
        else:
            rows = _read_jsonl(Path(replay_path))
            replay = CubeCumulativeReplay.from_persisted_rows(rows)
            self.validate_replay(rows)
            expected_count = int(metadata.get("cumulative_replay_positions", metadata.get("replay_row_count", len(rows))))
            if expected_count != len(rows):
                raise ValueError("Cube resume replay position count mismatch")
            if metadata.get("replay_fingerprint") is not None and sequence_fingerprint(rows) != metadata["replay_fingerprint"]:
                raise ValueError("Cube resume replay fingerprint mismatch")
        state = self.create_state(
            model,
            run_id=str(metadata.get("run_id", "cube-resume")),
            replay=replay,
            optimizer=optimizer,
            parent_checkpoint_identity={
                "label": metadata.get("checkpoint_label"),
                "path": str(checkpoint_path),
                "metadata_path": str(metadata_path),
                "model_hash": metadata["model_hash"],
                "artifact_sha256": file_sha256(checkpoint_path),
            },
            completed_games=int(metadata.get("completed_games", 0)),
            current_generation=self._generation_from_label(metadata.get("checkpoint_label")),
            optimizer_updates=updates,
            samples_consumed=int(metadata.get("train_samples_consumed", 0)),
        )
        self.validate_state(state)
        return state

    @staticmethod
    def _generation_from_label(label: object) -> int:
        value = str(label or "")
        return int(value[1:]) if value.startswith("M") and value[1:].isdigit() else 0

    def snapshot_state(self, state: TrainingState) -> object:
        return {
            "model_state": copy.deepcopy({name: value.detach().clone() for name, value in state.model.state_dict().items()}),
            "optimizer_state": copy.deepcopy(state.optimizer.state_dict()),
            "optimizer_updates": int(state.optimizer_updates),
            "samples_consumed": int(state.samples_consumed),
            "current_generation": int(state.current_generation),
            "completed_games": int(state.completed_games),
            "replay_rows": tuple(dict(row) for row in state.rolling_replay.rows),
            "replay_last_generation": state.rolling_replay.last_generation,
            "parent_checkpoint_identity": (
                dict(state.parent_checkpoint_identity) if state.parent_checkpoint_identity is not None else None
            ),
        }

    def restore_state(self, state: TrainingState, snapshot: object) -> None:
        if not isinstance(snapshot, Mapping):
            raise TypeError("Cube training snapshot is malformed")
        state.model.load_state_dict(snapshot["model_state"], strict=True)  # type: ignore[arg-type]
        state.optimizer.load_state_dict(snapshot["optimizer_state"])  # type: ignore[arg-type]
        state.optimizer_updates = int(snapshot["optimizer_updates"])
        state.samples_consumed = int(snapshot["samples_consumed"])
        state.current_generation = int(snapshot["current_generation"])
        state.completed_games = int(snapshot["completed_games"])
        state.rolling_replay = CubeCumulativeReplay(
            snapshot["replay_rows"],  # type: ignore[arg-type]
            int(snapshot["replay_last_generation"]),
        )
        parent = snapshot["parent_checkpoint_identity"]
        state.parent_checkpoint_identity = dict(parent) if isinstance(parent, Mapping) else None
        self.sync_state(state)

    def sync_state(self, state: TrainingState) -> None:
        state.optimizer_updates = int(state.optimizer_updates)
        state.samples_consumed = int(state.samples_consumed)


def run_cube_training_iteration(
    *,
    state: TrainingState,
    generation: int,
    output_dir: str | Path,
    run_id: str,
    training_seed: int,
    records: Sequence[object] | None = None,
    samples: Sequence[Mapping[str, object]] | None = None,
    adapter: CubeTrainingAdapter | None = None,
    **kwargs: object,
) -> TrainingIterationResult:
    selected = adapter or CubeTrainingAdapter()
    return TrainingEngine(selected).run_iteration(
        state=state,
        generation=generation,
        output_dir=output_dir,
        run_id=run_id,
        training_seed=training_seed,
        records=records,
        samples=samples,
        **kwargs,
    )


__all__ = [
    "CUBE_BATCH_SIZE",
    "CUBE_CHUNKS",
    "CUBE_GAMES_PER_CHUNK",
    "CUBE_LEARNING_RATE",
    "CUBE_MODEL_INIT_SEED",
    "CUBE_STAGE5_BASE_COMMIT",
    "CubeCumulativeReplay",
    "CubeTrainingAdapter",
    "TrainingState",
    "run_cube_training_iteration",
]
