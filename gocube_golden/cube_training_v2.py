"""Cube V2 Stage-6 adapter over the common replay/training/checkpoint transaction."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from training_engine import CheckpointContext, TrainingEngine, TrainingIterationResult, TrainingState, sequence_fingerprint

from .cube_checkpoint_v2 import (
    file_sha256,
    load_payload,
    restore_model_optimizer,
    save_checkpoint as save_checkpoint_file,
    sidecar_path,
    validate_checkpoint_metadata,
)
from .cube_family import cube_family_topology, initial_cube_state
from .cube_game_contract_v2 import concrete_game_fingerprint, concrete_game_identity, load_contract as load_cube_game_contract, validate_cube_size
from .cube_network_v2 import (
    ARCHITECTURE_FINGERPRINT,
    ARCHITECTURE_ID,
    CubeGraphNetV2,
    build_cube_model_from_metadata,
    cube_graphnet_v2_model_hash,
    cube_model_metadata,
    validate_cube_model_metadata,
)
from .cube_observation_v2 import concrete_observation_identity
from .cube_replay_v2 import CubeReplayCodecV2, collate_cube_rows, deterministic_sample_indices
from .cube_selfplay_contract import CUBE_SELFPLAY_SEMANTICS_FINGERPRINT, CUBE_TARGET_CONTRACT_ID, CUBE_TARGET_FINGERPRINT
from .cube_training_contract_v2 import (
    CHECKPOINT_SCHEMA,
    CubeTrainingConfig,
    OPTIMIZER_FAMILY,
    REPLAY_SCHEMA,
    TRAINING_CONTRACT_ID,
    load_cube_training_contract,
    validate_cube_training_contract,
)
from .cube_training_targets import build_cube_training_samples
from .rolling_replay import RollingGenerationReplay


@dataclass
class CubeTrainingRuntimeState:
    sampling_seed: int
    sampling_counter: int = 0

    def validate(self) -> None:
        if isinstance(self.sampling_seed, bool) or not isinstance(self.sampling_seed, int) or self.sampling_seed <= 0:
            raise ValueError("Cube replay sampling_seed must be a positive integer")
        if isinstance(self.sampling_counter, bool) or not isinstance(self.sampling_counter, int) or self.sampling_counter < 0:
            raise ValueError("Cube replay sampling_counter must be non-negative")


@dataclass(frozen=True)
class CubeTrainingGenerationResult:
    generation: int
    sample_count_new: int
    replay_generations: tuple[int, ...]
    replay_sample_count: int
    optimizer_steps: int
    batch_size: int
    effective_learning_rate: float
    policy_loss: float
    wdl_loss: float
    ownership_loss: float
    score_loss: float
    total_loss: float
    checkpoint_reference: Mapping[str, object]
    checkpoint_sha256: str
    timing_breakdown: Mapping[str, object]
    warnings: tuple[str, ...]
    engine_result: TrainingIterationResult


def _game_identity(size: int) -> tuple[dict[str, object], str]:
    topology = cube_family_topology(size)
    state = initial_cube_state(size=size, komi=0.5)
    identity = concrete_game_identity(
        load_cube_game_contract(),
        size,
        topology_id=topology.topology_id,
        topology_fingerprint=topology.fingerprint,
        rules_fingerprint=state.rules_fingerprint,
        komi=state.komi,
    )
    return identity, concrete_game_fingerprint(identity)


def _observation_fingerprint(size: int) -> str:
    return str(concrete_observation_identity(cube_family_topology(size))["concrete_observation_fingerprint"])


def _clone_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _clone_optimizer(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_optimizer(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer(item) for item in value)
    return deepcopy(value)


def _finite_model(model: torch.nn.Module) -> None:
    for name, value in model.state_dict().items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Cube model state contains NaN/Inf: {name}")


def _finite_gradients(model: torch.nn.Module) -> float:
    total = torch.tensor(0.0)
    seen = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        seen = True
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Cube training produced NaN/Inf gradient")
        total += torch.sum(gradient.float().cpu() ** 2)
    if not seen:
        raise FloatingPointError("Cube training produced no gradients")
    return float(torch.sqrt(total))


class CubeTrainingAdapter:
    """Only Cube scientific semantics; the common TrainingEngine owns the transaction."""

    def __init__(self, *, size: int, config: CubeTrainingConfig, training_contract: Mapping[str, object] | None = None) -> None:
        self.size = validate_cube_size(size)
        self.topology = cube_family_topology(self.size)
        self.config = config
        self.training_contract = dict(training_contract) if training_contract is not None else load_cube_training_contract()
        validate_cube_training_contract(self.training_contract)
        self.training_semantics_fingerprint = str(self.training_contract["contract_fingerprint"])
        self.concrete_training_config_fingerprint = config.fingerprint
        self.model_metadata = cube_model_metadata(self.size)
        self.game_identity, self.game_fingerprint = _game_identity(self.size)
        self.observation_fingerprint = _observation_fingerprint(self.size)
        self.codec = CubeReplayCodecV2(
            size=self.size,
            game_fingerprint=self.game_fingerprint,
            observation_fingerprint=self.observation_fingerprint,
        )

    @property
    def profile_identity(self) -> Mapping[str, object]:
        return {
            "training_contract_id": TRAINING_CONTRACT_ID,
            "training_semantics_fingerprint": self.training_semantics_fingerprint,
            "concrete_training_config_fingerprint": self.concrete_training_config_fingerprint,
            "size": self.size,
        }

    @property
    def target_identity(self) -> Mapping[str, object]:
        return {"id": CUBE_TARGET_CONTRACT_ID, "fingerprint": CUBE_TARGET_FINGERPRINT}

    def create_state(
        self,
        model: CubeGraphNetV2,
        *,
        sampling_seed: int,
        parent_checkpoint_identity: Mapping[str, object] | None = None,
    ) -> TrainingState:
        if not isinstance(model, CubeGraphNetV2):
            raise TypeError("Cube training requires CubeGraphNetV2")
        if model.model_metadata is None:
            model.model_metadata = deepcopy(self.model_metadata)
        validate_cube_model_metadata(model.model_metadata)
        if int(model.model_metadata["size"]) != self.size:
            raise ValueError("Cube model size does not match training adapter")
        runtime = CubeTrainingRuntimeState(int(sampling_seed))
        runtime.validate()
        return TrainingState(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=float(self.config.learning_rate), weight_decay=0.0),
            optimizer_updates=0,
            samples_consumed=0,
            current_generation=0,
            rolling_replay=RollingGenerationReplay(
                generations=self.config.replay_generations,
                maximum_positions=self.config.replay_cap,
            ),
            profile_identity=dict(self.profile_identity),
            target_identity=dict(self.target_identity),
            parent_checkpoint_identity=None if parent_checkpoint_identity is None else dict(parent_checkpoint_identity),
            adapter_state=runtime,
        )

    def validate_state(self, state: TrainingState) -> None:
        if not isinstance(state.model, CubeGraphNetV2) or state.model.model_metadata is None:
            raise TypeError("Cube training state must contain metadata-built CubeGraphNetV2")
        validate_cube_model_metadata(state.model.model_metadata)
        if int(state.model.model_metadata["size"]) != self.size:
            raise ValueError("Cube training state topology size mismatch")
        if not isinstance(state.optimizer, torch.optim.Adam) or len(state.optimizer.param_groups) != 1:
            raise TypeError("Cube training state optimizer must be Adam")
        lr = float(state.optimizer.param_groups[0]["lr"])
        if not math.isfinite(lr) or lr <= 0 or not math.isclose(lr, self.config.learning_rate, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("Cube training state learning rate/config mismatch")
        if state.profile_identity != self.profile_identity or state.target_identity != self.target_identity:
            raise ValueError("Cube training state contract identity mismatch")
        if not isinstance(state.adapter_state, CubeTrainingRuntimeState):
            raise TypeError("Cube training runtime state is missing")
        state.adapter_state.validate()
        if not isinstance(state.rolling_replay, RollingGenerationReplay):
            raise TypeError("Cube training must use the shared rolling replay primitive")
        if state.rolling_replay.generations != self.config.replay_generations or state.rolling_replay.maximum_positions != self.config.replay_cap:
            raise ValueError("Cube replay window/config mismatch")
        if state.optimizer_updates < 0 or state.samples_consumed < 0:
            raise ValueError("Cube training counters cannot be negative")
        _finite_model(state.model)

    def _rows_from_records(self, records: Sequence[object]) -> tuple[Mapping[str, object], ...]:
        # Stage 5 remains the only owner of policy/WDL/ownership/score target semantics.
        return tuple(self.codec.encode(sample) for sample in build_cube_training_samples(tuple(records)))

    def build_samples(self, records: Sequence[object]) -> Sequence[Mapping[str, object]]:
        rows = self._rows_from_records(records)
        for row in rows:
            self.codec.validate(row)
        return rows

    def build_samples_for_replay(self, records: Sequence[object]) -> Sequence[Mapping[str, object]]:
        return self._rows_from_records(records)

    def validate_sample(self, sample: Mapping[str, object]) -> None:
        self.codec.validate(sample)

    @staticmethod
    def _row_id(generation: int, sample: Mapping[str, object], position: int) -> str:
        return f"M{generation}:{sample.get('game_id')}:{sample.get('ply')}:{position}"

    def stamp_samples(self, samples: Sequence[Mapping[str, object]], generation: int) -> Sequence[Mapping[str, object]]:
        result: list[dict[str, object]] = []
        for position, sample in enumerate(samples):
            row = dict(sample)
            expected = self._row_id(int(generation), row, position)
            if row.get("source_generation") not in (None, int(generation)):
                raise ValueError("Cube replay source generation drift")
            if row.get("replay_row_id") not in (None, expected):
                raise ValueError("Cube replay row id drift")
            row["source_generation"] = int(generation)
            row["replay_row_id"] = expected
            result.append(row)
        return tuple(result)

    def update_replay(
        self,
        replay: RollingGenerationReplay,
        generation: int,
        samples: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        existing = tuple(replay.rows)
        generations = [int(row["source_generation"]) for row in existing if row.get("source_generation") is not None]
        if generations and int(generation) <= max(generations):
            raise ValueError("Cube replay generations must be strictly increasing")
        seen = {str(row.get("replay_row_id", "")) for row in existing}
        pending: set[str] = set()
        for position, row in enumerate(samples):
            self.codec.validate(row)
            if int(row.get("source_generation", 0)) != int(generation):
                raise ValueError("Cube replay source generation drift")
            expected = self._row_id(int(generation), row, position)
            row_id = str(row.get("replay_row_id", ""))
            if row_id != expected or row_id in seen or row_id in pending:
                raise ValueError("Cube replay row id is invalid or duplicated")
            pending.add(row_id)
        metrics = dict(replay.append_generation(int(generation), samples))
        metrics.update({"replay_schema": REPLAY_SCHEMA, "replay_generations_configured": self.config.replay_generations, "replay_cap": self.config.replay_cap})
        return metrics

    def replay_rows(self, replay: RollingGenerationReplay) -> Sequence[Mapping[str, object]]:
        return replay.rows

    def validate_replay(self, rows: Sequence[Mapping[str, object]]) -> None:
        previous_generation = 0
        ids: set[str] = set()
        for row in rows:
            generation = int(row.get("source_generation", 0))
            row_id = str(row.get("replay_row_id", ""))
            if generation <= 0 or generation < previous_generation:
                raise ValueError("Cube replay generation ordering drift")
            if not row_id or row_id in ids:
                raise ValueError("Cube replay row ids are missing or duplicated")
            self.codec.validate(row)
            previous_generation = generation
            ids.add(row_id)
        if self.config.replay_cap is not None and len(rows) > self.config.replay_cap:
            raise ValueError("Cube replay cap exceeded")

    def train(self, state: TrainingState, rows: Sequence[Mapping[str, object]], seed: int) -> Mapping[str, object]:
        self.validate_state(state)
        if not rows:
            raise ValueError("Cube training requires non-empty replay")
        runtime = state.adapter_state
        assert isinstance(runtime, CubeTrainingRuntimeState)
        total = self.config.batch_size * self.config.optimizer_steps
        indices = deterministic_sample_indices(
            len(rows),
            count=total,
            sampling_seed=runtime.sampling_seed,
            sampling_counter=runtime.sampling_counter,
            training_seed=int(seed),
        )
        device = next(state.model.parameters()).device
        state.model.train()
        losses: list[dict[str, float]] = []
        for step in range(self.config.optimizer_steps):
            start = step * self.config.batch_size
            selected = indices[start : start + self.config.batch_size]
            batch = collate_cube_rows(
                rows,
                selected,
                point_count=self.topology.point_count,
                action_count=self.topology.action_count,
                device=device,
            )
            output = state.model(batch["observation"])
            policy_loss = -(batch["policy"] * F.log_softmax(output.policy_logits, dim=1)).sum(dim=1).mean()
            wdl_loss = -(batch["wdl"] * F.log_softmax(output.wdl_logits, dim=1)).sum(dim=1).mean()
            ownership_loss = F.cross_entropy(output.ownership_logits.reshape(-1, 3), batch["ownership"].reshape(-1))
            score_loss = F.mse_loss(output.score, batch["score"])
            total_loss = policy_loss + wdl_loss + ownership_loss + score_loss
            named = {
                "policy_loss": policy_loss,
                "wdl_loss": wdl_loss,
                "ownership_loss": ownership_loss,
                "score_loss": score_loss,
                "total_loss": total_loss,
            }
            if any(not bool(torch.isfinite(value)) for value in named.values()):
                raise FloatingPointError("Cube training produced NaN/Inf loss")
            state.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = _finite_gradients(state.model)
            state.optimizer.step()
            _finite_model(state.model)
            state.optimizer_updates += 1
            state.samples_consumed += self.config.batch_size
            losses.append({**{key: float(value.detach().cpu()) for key, value in named.items()}, "gradient_norm": grad_norm})
        runtime.sampling_counter += 1

        def mean(key: str) -> float:
            return sum(item[key] for item in losses) / len(losses)

        return {
            "optimizer_family": OPTIMIZER_FAMILY,
            "optimizer_steps": self.config.optimizer_steps,
            "optimizer_updates_total": state.optimizer_updates,
            "batch_size": self.config.batch_size,
            "samples_consumed": total,
            "samples_consumed_total": state.samples_consumed,
            "effective_learning_rate": float(state.optimizer.param_groups[0]["lr"]),
            "policy_loss": mean("policy_loss"),
            "wdl_loss": mean("wdl_loss"),
            "ownership_loss": mean("ownership_loss"),
            "score_loss": mean("score_loss"),
            "total_loss": mean("total_loss"),
            "gradient_norm": mean("gradient_norm"),
            "sampled_replay_row_ids": tuple(str(rows[index]["replay_row_id"]) for index in indices),
            "sampling_state": asdict(runtime),
        }

    def prepare_checkpoint(self, state: TrainingState, context: CheckpointContext, training_metrics: Mapping[str, object]) -> Mapping[str, object]:
        self.validate_state(state)
        return {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_schema_version": 2,
            "generation": context.generation,
            "label": context.label,
            "size": self.size,
            "game_identity": dict(self.game_identity),
            "game_fingerprint": self.game_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
            "architecture_id": ARCHITECTURE_ID,
            "architecture_fingerprint": ARCHITECTURE_FINGERPRINT,
            "model_metadata": deepcopy(dict(state.model.model_metadata or {})),
            "model_hash": cube_graphnet_v2_model_hash(state.model),
            "optimizer_type": OPTIMIZER_FAMILY,
            "effective_learning_rate": float(state.optimizer.param_groups[0]["lr"]),
            "weight_decay": 0.0,
            "training_contract_id": TRAINING_CONTRACT_ID,
            "training_semantics_fingerprint": self.training_semantics_fingerprint,
            "concrete_training_config": self.config.identity_payload(),
            "concrete_training_config_fingerprint": self.concrete_training_config_fingerprint,
            "replay_schema": REPLAY_SCHEMA,
            "replay_contract_identity": {"schema": REPLAY_SCHEMA, "generations": self.config.replay_generations, "cap": self.config.replay_cap},
            "replay_fingerprint": context.replay_fingerprint,
            "replay_generations": list(context.replay_generations),
            "replay_positions": context.replay_positions,
            "target_contract_id": CUBE_TARGET_CONTRACT_ID,
            "target_contract_fingerprint": CUBE_TARGET_FINGERPRINT,
            "selfplay_semantics_fingerprint": CUBE_SELFPLAY_SEMANTICS_FINGERPRINT,
            "rng_state": {**asdict(state.adapter_state), "global_torch_rng_required": False},
            "optimizer_updates": state.optimizer_updates,
            "samples_consumed": state.samples_consumed,
            "training_seed": context.training_seed,
            "parent_checkpoint": None if context.parent_checkpoint_identity is None else dict(context.parent_checkpoint_identity),
            "git_commit": getattr(context.code_identity, "git_commit", None) if context.code_identity is not None else None,
        }

    def _metadata_kwargs(self) -> dict[str, object]:
        return {
            "expected_size": self.size,
            "game_fingerprint": self.game_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
            "training_semantics_fingerprint": self.training_semantics_fingerprint,
            "concrete_training_config_fingerprint": self.concrete_training_config_fingerprint,
            "effective_learning_rate": float(self.config.learning_rate),
        }

    def save_checkpoint(self, path: Path, state: TrainingState, metadata: Mapping[str, object]) -> Mapping[str, object]:
        validate_checkpoint_metadata(metadata, **self._metadata_kwargs())
        return save_checkpoint_file(
            path,
            model=state.model,
            optimizer=state.optimizer,
            metadata=metadata,
            runtime_state=asdict(state.adapter_state),
            optimizer_updates=state.optimizer_updates,
            samples_consumed=state.samples_consumed,
        )

    def verify_checkpoint(self, path: Path, state: TrainingState, metadata: Mapping[str, object]) -> None:
        payload, loaded = load_payload(path, map_location="cpu", **self._metadata_kwargs())
        if dict(loaded) != dict(metadata):
            raise ValueError("Cube checkpoint saved metadata drift")
        model, optimizer = restore_model_optimizer(payload, loaded, map_location="cpu")
        if cube_graphnet_v2_model_hash(model) != cube_graphnet_v2_model_hash(state.model):
            raise ValueError("Cube checkpoint does not match in-memory model")
        if int(payload["optimizer_updates"]) != state.optimizer_updates or int(payload["samples_consumed"]) != state.samples_consumed:
            raise ValueError("Cube checkpoint training counters mismatch")
        if len(optimizer.state_dict()["state"]) != len(state.optimizer.state_dict()["state"]):
            raise ValueError("Cube checkpoint optimizer state mismatch")

    def artifact_hash(self, path: Path) -> str:
        return file_sha256(path)

    def snapshot_state(self, state: TrainingState) -> Mapping[str, object]:
        replay = state.rolling_replay
        return {
            "model": _clone_model(state.model),
            "optimizer": _clone_optimizer(state.optimizer.state_dict()),
            "optimizer_updates": state.optimizer_updates,
            "samples_consumed": state.samples_consumed,
            "current_generation": state.current_generation,
            "completed_games": state.completed_games,
            "parent_checkpoint_identity": deepcopy(state.parent_checkpoint_identity),
            "runtime_state": deepcopy(state.adapter_state),
            "replay_rows": deepcopy(list(replay.rows)),
            "replay_last_generation": int(getattr(replay, "_last_generation", 0)),
            "replay_total_evictions": replay.total_evictions,
        }

    def restore_state(self, state: TrainingState, snapshot: Mapping[str, object]) -> None:
        state.model.load_state_dict(snapshot["model"], strict=True)
        state.optimizer.load_state_dict(snapshot["optimizer"])
        state.optimizer_updates = int(snapshot["optimizer_updates"])
        state.samples_consumed = int(snapshot["samples_consumed"])
        state.current_generation = int(snapshot["current_generation"])
        state.completed_games = int(snapshot["completed_games"])
        state.parent_checkpoint_identity = deepcopy(snapshot["parent_checkpoint_identity"])
        state.adapter_state = deepcopy(snapshot["runtime_state"])
        replay = RollingGenerationReplay(generations=self.config.replay_generations, maximum_positions=self.config.replay_cap)
        replay._rows = deepcopy(list(snapshot["replay_rows"]))
        replay._last_generation = int(snapshot["replay_last_generation"])
        replay.total_evictions = int(snapshot["replay_total_evictions"])
        state.rolling_replay = replay

    def sync_state(self, state: TrainingState) -> None:
        return None


def create_cube_m0_state(
    *,
    size: int,
    config: CubeTrainingConfig,
    seed: int,
    device: str | torch.device = "cpu",
) -> tuple[CubeTrainingAdapter, TrainingState]:
    size = validate_cube_size(size)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise ValueError("Cube M0 seed must be a positive explicit integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_cube_model_from_metadata(cube_model_metadata(size))
    model.to(device)
    adapter = CubeTrainingAdapter(size=size, config=config)
    return adapter, adapter.create_state(model, sampling_seed=seed)


def _read_replay(path: str | Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cube replay JSONL is invalid at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError("Cube replay row must be a mapping")
            rows.append(row)
    return tuple(rows)


def _restore_replay(adapter: CubeTrainingAdapter, rows: Sequence[Mapping[str, object]]) -> RollingGenerationReplay:
    adapter.validate_replay(rows)
    replay = RollingGenerationReplay(generations=adapter.config.replay_generations, maximum_positions=adapter.config.replay_cap)
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["source_generation"]), []).append(row)
    for generation in sorted(grouped):
        replay.append_generation(generation, grouped[generation])
    if tuple(replay.rows) != tuple(rows):
        raise ValueError("Cube replay reconstruction changed committed row order/content")
    return replay


def load_cube_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: CubeTrainingConfig,
    replay_path: str | Path,
    map_location: str | torch.device = "cpu",
    expected_size: int | None = None,
) -> tuple[CubeTrainingAdapter, TrainingState, Mapping[str, object]]:
    metadata_path = sidecar_path(checkpoint_path)
    if not metadata_path.is_file():
        raise ValueError("Cube checkpoint metadata sidecar is missing")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Cube checkpoint metadata sidecar is invalid")
    size = validate_cube_size(raw.get("size"))
    if expected_size is not None and size != validate_cube_size(expected_size):
        raise ValueError("Cube checkpoint topology size mismatch")
    adapter = CubeTrainingAdapter(size=size, config=config)
    payload, metadata = load_payload(checkpoint_path, map_location=map_location, **adapter._metadata_kwargs())
    model, optimizer = restore_model_optimizer(payload, metadata, map_location=map_location)
    runtime_raw = payload["runtime_state"]
    if not isinstance(runtime_raw, Mapping):
        raise ValueError("Cube checkpoint runtime state is invalid")
    runtime = CubeTrainingRuntimeState(int(runtime_raw["sampling_seed"]), int(runtime_raw["sampling_counter"]))
    runtime.validate()
    rows = _read_replay(replay_path)
    replay = _restore_replay(adapter, rows)
    if len(rows) != int(metadata["replay_positions"]) or sequence_fingerprint(rows) != metadata["replay_fingerprint"]:
        raise ValueError("Cube checkpoint replay identity mismatch")
    state = TrainingState(
        model=model,
        optimizer=optimizer,
        optimizer_updates=int(payload["optimizer_updates"]),
        samples_consumed=int(payload["samples_consumed"]),
        current_generation=int(metadata["generation"]),
        rolling_replay=replay,
        profile_identity=dict(adapter.profile_identity),
        target_identity=dict(adapter.target_identity),
        parent_checkpoint_identity={
            "label": metadata.get("label"),
            "path": str(checkpoint_path),
            "metadata_path": str(metadata_path),
            "model_hash": metadata.get("model_hash"),
            "artifact_sha256": metadata.get("checkpoint_sha256"),
        },
        adapter_state=runtime,
    )
    adapter.validate_state(state)
    return adapter, state, metadata


def run_cube_training_generation(
    *,
    adapter: CubeTrainingAdapter,
    state: TrainingState,
    generation: int,
    lineage_dir: str | Path,
    run_id: str,
    training_seed: int,
    records: Sequence[object],
    completed_games: int | None = None,
    parent_checkpoint_identity: Mapping[str, object] | None = None,
    code_identity: object = None,
    device: str | None = None,
) -> CubeTrainingGenerationResult:
    result = TrainingEngine(adapter).run_iteration(
        state=state,
        generation=generation,
        output_dir=Path(lineage_dir),
        run_id=run_id,
        records=tuple(records),
        training_seed=training_seed,
        completed_games=completed_games,
        parent_checkpoint_identity=parent_checkpoint_identity,
        code_identity=code_identity,
        device=device,
        summary_extra={
            "cube_stage": 6,
            "training_contract": {"id": TRAINING_CONTRACT_ID, "fingerprint": adapter.training_semantics_fingerprint},
            "concrete_training_config_fingerprint": adapter.concrete_training_config_fingerprint,
        },
    )
    metrics = result.training_metrics
    checkpoint_path = Path(result.artifacts["checkpoint"])
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != result.checkpoint_metadata.get("checkpoint_sha256"):
        raise ValueError("Committed Cube checkpoint SHA disagrees with sidecar identity")
    state.parent_checkpoint_identity = {
        "lineage_id": run_id,
        "checkpoint_id": result.label,
        "label": result.label,
        "path": str(checkpoint_path),
        "metadata_path": result.artifacts["checkpoint_metadata"],
        "model_hash": result.checkpoint_metadata.get("model_hash"),
        "artifact_sha256": checkpoint_sha,
    }
    return CubeTrainingGenerationResult(
        generation=result.generation,
        sample_count_new=result.fresh_positions,
        replay_generations=tuple(int(value) for value in result.checkpoint_metadata.get("replay_generations", ())),
        replay_sample_count=int(result.checkpoint_metadata["replay_positions"]),
        optimizer_steps=int(metrics["optimizer_steps"]),
        batch_size=int(metrics["batch_size"]),
        effective_learning_rate=float(metrics["effective_learning_rate"]),
        policy_loss=float(metrics["policy_loss"]),
        wdl_loss=float(metrics["wdl_loss"]),
        ownership_loss=float(metrics["ownership_loss"]),
        score_loss=float(metrics["score_loss"]),
        total_loss=float(metrics["total_loss"]),
        checkpoint_reference={
            "lineage_id": run_id,
            "checkpoint_id": result.label,
            "path": str(checkpoint_path),
            "metadata_path": result.artifacts["checkpoint_metadata"],
            "sha256": checkpoint_sha,
            "generation": result.generation,
            "size": adapter.size,
        },
        checkpoint_sha256=checkpoint_sha,
        timing_breakdown=dict(metrics.get("phase_timing", {})),
        warnings=(),
        engine_result=result,
    )


__all__ = [
    "CubeTrainingAdapter",
    "CubeTrainingConfig",
    "CubeTrainingGenerationResult",
    "CubeTrainingRuntimeState",
    "create_cube_m0_state",
    "load_cube_checkpoint",
    "run_cube_training_generation",
]
