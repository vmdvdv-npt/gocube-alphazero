"""Current Torus9 adapter for the generic :mod:`training_engine`.

The model, replay policy, loss implementation, and checkpoint wire format in
this module are intentionally delegated to the already proven Torus9
scientific core.  This module supplies the current-profile boundary,
resumable state, provenance, and the adapter methods consumed by the generic
execution engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping, Sequence

from training_engine import (
    CheckpointContext,
    TrainingEngine,
    TrainingIterationResult,
    TrainingState,
    sequence_fingerprint,
    value_fingerprint,
)

from . import torus9_monolith as _core
from .neural import model_hash
from .provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256
from .torus9_contract import (
    TORUS9_BATCH_SIZE,
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_BLOCKS,
    TORUS9_CURRENT_HIDDEN,
    TORUS9_CURRENT_MODEL_INIT_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_KOMI,
    TORUS9_MAX_REPLAY_POSITIONS,
    TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    TORUS9_POINT_COUNT,
    TORUS9_ROLLING_GENERATIONS,
    TORUS9_TARGET_CONTRACT_ID,
    TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    current_torus9_selfplay_contract_fingerprint,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


# Keep one scientific implementation.  These aliases expose numerical
# primitives consumed by the current adapter; orchestration lives below.
Torus9GraphNet = _core.Torus9GraphNet
Torus9CurrentGraphNet = _core.Torus9CurrentGraphNet
Torus9OwnershipGraphNet = _core.Torus9OwnershipGraphNet
Torus9OwnershipScoreGraphNet = _core.Torus9OwnershipScoreGraphNet
Torus9OwnershipTrainer = _core.Torus9OwnershipTrainer
Torus9OwnershipScoreTrainer = _core.Torus9OwnershipScoreTrainer
torus9_checkpoint_metadata = _core.torus9_checkpoint_metadata
torus9_save_checkpoint = _core.torus9_save_checkpoint
torus9_load_checkpoint = _core.torus9_load_checkpoint
torus9_model_from_metadata = _core.torus9_model_from_metadata
validate_torus9_replay_sample = _core.validate_torus9_replay_sample


def _checkpoint_label(metadata: Mapping[str, object]) -> str | None:
    value = metadata.get("checkpoint_label")
    return str(value) if value is not None else None


def _adam_step(optimizer: Any) -> int:
    return int(_core._adam_step(optimizer))


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
                raise ValueError(f"Replay row {line_number} is not an object: {path}")
            rows.append(value)
    return tuple(rows)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Replay row {line_number} is not an object: {path}")
            yield value


def _optimizer_parameter_order(model: Any) -> tuple[str, ...]:
    return tuple(name for name, _ in model.named_parameters())


class Torus9RollingReplay(_core.Torus9RollingReplay):
    """Current-path facade over the single proven rolling replay core."""

    @property
    def last_generation(self) -> int:
        return int(self._last_generation)

    @classmethod
    def from_persisted_rows(
        cls,
        rows: Sequence[Mapping[str, object]],
        *,
        generations: int = TORUS9_ROLLING_GENERATIONS,
        maximum_positions: int = TORUS9_MAX_REPLAY_POSITIONS,
        total_evictions: int = 0,
    ) -> "Torus9RollingReplay":
        replay = cls(generations=generations, maximum_positions=maximum_positions)
        copied = [dict(row) for row in rows]
        if len(copied) > int(maximum_positions):
            raise ValueError("Persisted Torus9 replay exceeds the configured cap")
        row_ids = [str(row.get("replay_row_id", "")) for row in copied]
        if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
            raise ValueError("Persisted Torus9 replay row IDs are missing or duplicated")
        generations_seen = [int(row.get("source_generation", 0)) for row in copied]
        if any(generation <= 0 for generation in generations_seen):
            raise ValueError("Persisted Torus9 replay generation is malformed")
        if generations_seen != sorted(generations_seen):
            raise ValueError("Persisted Torus9 replay ordering is not generation-stable")
        if generations_seen and generations_seen[-1] - generations_seen[0] >= int(generations):
            raise ValueError("Persisted Torus9 replay contains generations outside its window")
        replay._rows = copied
        replay._last_generation = generations_seen[-1] if generations_seen else 0
        replay.total_evictions = int(total_evictions)
        if replay.total_evictions < 0:
            raise ValueError("Persisted Torus9 replay eviction count is negative")
        return replay

    def append_generation(
        self, generation: int, samples: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        generation = int(generation)
        if generation <= self.last_generation:
            raise ValueError("Torus9 replay generations must increase strictly")
        incoming_ids = [str(sample.get("replay_row_id", "")) for sample in samples]
        if any(not row_id for row_id in incoming_ids):
            raise ValueError("Torus9 replay rows require deterministic replay_row_id values")
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("Torus9 replay generation contains duplicate row IDs")
        existing_ids = {str(row.get("replay_row_id")) for row in self.rows}
        if existing_ids.intersection(incoming_ids):
            raise ValueError("Torus9 replay contains a duplicate row ID")
        return super().append_generation(generation, samples)


@dataclass(frozen=True)
class _Torus9StateSnapshot:
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, object]
    optimizer_updates: int
    samples_consumed: int
    current_generation: int
    completed_games: int
    replay_rows: tuple[dict[str, object], ...]
    replay_last_generation: int
    replay_evictions: int
    parent_checkpoint_identity: Mapping[str, object] | None


class Torus9TrainingAdapter:
    """Current Golden Torus9 training semantics behind ``TrainingEngine``."""

    def __init__(
        self,
        profile: Mapping[str, object] | None = None,
        *,
        code_identity: CodeIdentity | None = None,
        base_commit: str = TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    ) -> None:
        self.profile = (
            load_torus9_current_profile()
            if profile is None
            else dict(profile)
        )
        self._validate_current_profile(self.profile)
        self.profile_fingerprint = current_torus9_profile_fingerprint(self.profile)
        self.code_identity = code_identity
        self.base_commit = str(base_commit)
        self.training_profile = self.profile["training"]
        self.replay_profile = self.profile["replay"]
        self._validated_sample_ids: set[str] = set()
        self._validated_sample_objects: set[int] = set()
        self._diagnostic_timing: MutableMapping[str, object] | None = None
        self.target_identity = {
            "contract_id": TORUS9_TARGET_CONTRACT_ID,
            "fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
            "perspective": "side-to-move",
        }

    def set_diagnostic_timing(
        self, timing: MutableMapping[str, object] | None
    ) -> None:
        """Attach an opt-in sink for stage timing on one controlled run.

        The default production path leaves this unset, so diagnostics cannot
        alter the canonical execution contract or add synchronization.
        """
        self._diagnostic_timing = timing

    @staticmethod
    def _validate_current_profile(profile: Mapping[str, object]) -> None:
        if profile.get("profile_id") != TORUS9_CURRENT_PROFILE_ID:
            raise ValueError("Current Torus9 training requires the current profile")
        if profile.get("profile_fingerprint") != current_torus9_profile_fingerprint(profile):
            raise ValueError("Current Torus9 training profile fingerprint drift")
        rules = profile.get("rules")
        if not isinstance(rules, Mapping) or rules.get("komi") != TORUS9_KOMI:
            raise ValueError("Current Torus9 training komi must be exactly 0.5")
        target = profile.get("target")
        if not isinstance(target, Mapping) or (
            target.get("contract_id") != TORUS9_TARGET_CONTRACT_ID
            or target.get("fingerprint") != TORUS9_CURRENT_TARGET_FINGERPRINT
            or target.get("perspective") != "side-to-move"
            or target.get("ownership_auxiliary") is not True
            or target.get("score_auxiliary") is not True
        ):
            raise ValueError("Current Torus9 target contract drift")
        network = profile.get("network")
        if not isinstance(network, Mapping) or network.get("architecture_id") != TORUS9_CURRENT_ARCHITECTURE_ID:
            raise ValueError("Current Torus9 training architecture drift")
        if network.get("hidden") != TORUS9_CURRENT_HIDDEN or network.get("blocks") != TORUS9_CURRENT_BLOCKS:
            raise ValueError("Current Torus9 training capacity drift")
        self_play = profile.get("self_play")
        if not isinstance(self_play, Mapping) or (
            self_play.get("contract_id") != TORUS9_CURRENT_SELFPLAY_CONTRACT_ID
            or self_play.get("fingerprint")
            != current_torus9_selfplay_contract_fingerprint()
            or self_play.get("komi") != TORUS9_KOMI
        ):
            raise ValueError("Current Torus9 self-play contract drift")
        training = profile.get("training")
        replay = profile.get("replay")
        if not isinstance(training, Mapping) or not isinstance(replay, Mapping):
            raise ValueError("Current Torus9 training/replay profile is incomplete")
        if (
            training.get("optimizer") != "Adam"
            or training.get("learning_rate") != 0.001
            or training.get("weight_decay") != 0.0
            or training.get("batch_size") != TORUS9_BATCH_SIZE
            or training.get("optimizer_steps_per_iteration") != TORUS9_OPTIMIZER_STEPS_PER_ITERATION
            or training.get("samples_consumed_per_iteration") != TORUS9_BATCH_SIZE * TORUS9_OPTIMIZER_STEPS_PER_ITERATION
            or training.get("lr_scheduler") is not None
            or training.get("model_gating") is not False
        ):
            raise ValueError("Current Torus9 training optimizer contract drift")
        if replay.get("generations") != TORUS9_ROLLING_GENERATIONS or replay.get("cap") != TORUS9_MAX_REPLAY_POSITIONS:
            raise ValueError("Current Torus9 replay contract drift")

    @property
    def profile_identity(self) -> Mapping[str, object]:
        network = self.profile["network"]
        return {
            "profile_id": TORUS9_CURRENT_PROFILE_ID,
            "profile_fingerprint": self.profile_fingerprint,
            "architecture_id": network["architecture_id"],  # type: ignore[index]
            "komi": TORUS9_KOMI,
        }

    def create_state(
        self,
        model: Torus9CurrentGraphNet,
        *,
        run_id: str,
        replay: Torus9RollingReplay | None = None,
        trainer: Torus9OwnershipScoreTrainer | None = None,
        parent_checkpoint_identity: Mapping[str, object] | None = None,
        completed_games: int = 0,
        current_generation: int = 0,
    ) -> TrainingState:
        if trainer is None:
            trainer = Torus9OwnershipScoreTrainer(
                model,
                score_loss_enabled=True,
                learning_rate=float(self.training_profile["learning_rate"]),  # type: ignore[index]
                weight_decay=float(self.training_profile["weight_decay"]),  # type: ignore[index]
                optimizer_steps_per_iteration=int(self.training_profile["optimizer_steps_per_iteration"]),  # type: ignore[index]
            )
        if replay is None:
            replay = Torus9RollingReplay(
                generations=int(self.replay_profile["generations"]),  # type: ignore[index]
                maximum_positions=int(self.replay_profile["cap"]),  # type: ignore[index]
            )
        state = TrainingState(
            model=model,
            optimizer=trainer.optimizer,
            optimizer_updates=int(trainer.update_count),
            samples_consumed=int(trainer.samples_consumed),
            current_generation=int(current_generation),
            rolling_replay=replay,
            profile_identity=self.profile_identity,
            target_identity=self.target_identity,
            parent_checkpoint_identity=(
                dict(parent_checkpoint_identity)
                if parent_checkpoint_identity is not None
                else None
            ),
            completed_games=int(completed_games),
            adapter_state=trainer,
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
        """Persist a fresh model before the first training generation."""
        self.validate_state(state)
        if state.optimizer_updates != 0 or state.samples_consumed != 0:
            raise ValueError("Initial Torus9 checkpoint requires a zero training clock")
        code = code_identity or self.code_identity or capture_code_identity()
        metadata = _core.torus9_checkpoint_metadata(
            model=state.model,
            run_id=str(run_id),
            label=str(label),
            parent=None,
            model_seed=TORUS9_CURRENT_MODEL_INIT_SEED,
            code=code,
            profile_fp=self.profile_fingerprint,
            completed_games=int(completed_games),
            replay_positions=0,
            optimizer_updates=0,
            samples_consumed=0,
            ownership_loss_enabled=True,
            score_loss_enabled=True,
            profile_id=TORUS9_CURRENT_PROFILE_ID,
            target_fingerprint=TORUS9_CURRENT_TARGET_FINGERPRINT,
            selfplay_contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
            selfplay_contract_fingerprint=str(self.profile["self_play"]["fingerprint"]),  # type: ignore[index]
            base_commit=self.base_commit,
        )
        metadata.update({
            "adam_step": 0,
            "model_init_seed": TORUS9_CURRENT_MODEL_INIT_SEED,
            "device": str(device),
            "device_locked": True,
            "replay_generations": [],
            "replay_fingerprint": value_fingerprint(()),
            "sampled_row_ids_fingerprint": value_fingerprint(()),
            "training_seed": 0,
            "fresh_replay_positions": 0,
            "replay_row_count": 0,
            "optimizer_parameter_order": list(_optimizer_parameter_order(state.model)),
            "optimizer_parameter_groups": [],
        })
        return _core.torus9_save_checkpoint(Path(path), model=state.model, optimizer=None, metadata=metadata)

    def validate_state(self, state: TrainingState) -> None:
        if not isinstance(state.rolling_replay, Torus9RollingReplay):
            raise TypeError("Current Torus9 training requires Torus9RollingReplay")
        if not isinstance(state.model, Torus9CurrentGraphNet):
            raise TypeError("Current Torus9 training requires GoldenGraphNetV2-Torus9")
        if state.model.architecture_config.get("architecture_id") != TORUS9_CURRENT_ARCHITECTURE_ID:
            raise ValueError("Current Torus9 training model architecture mismatch")
        if state.profile_identity.get("profile_fingerprint") != self.profile_fingerprint:
            raise ValueError("Current Torus9 training profile identity mismatch")
        if state.target_identity.get("fingerprint") != TORUS9_CURRENT_TARGET_FINGERPRINT:
            raise ValueError("Current Torus9 training target identity mismatch")
        trainer = state.adapter_state
        if not isinstance(trainer, Torus9OwnershipScoreTrainer) or trainer.model is not state.model:
            raise TypeError("Current Torus9 training state is not bound to its trainer")
        if trainer.optimizer is not state.optimizer:
            raise TypeError("Current Torus9 training state is not bound to its optimizer")
        if float(state.optimizer.param_groups[0]["lr"]) != 0.001:
            raise ValueError("Current Torus9 Adam learning rate drift")
        if float(state.optimizer.param_groups[0]["weight_decay"]) != 0.0:
            raise ValueError("Current Torus9 Adam weight decay drift")
        if int(state.optimizer_updates) != int(trainer.update_count):
            raise ValueError("Current Torus9 optimizer update counter drift")
        if int(state.samples_consumed) != int(trainer.samples_consumed):
            raise ValueError("Current Torus9 sample counter drift")
        if _adam_step(state.optimizer) != int(state.optimizer_updates):
            raise ValueError("Current Torus9 Adam step does not match training clock")

    def build_samples(self, records: Sequence[object]) -> Sequence[Mapping[str, object]]:
        ordered = sorted(tuple(records), key=lambda record: str(getattr(record, "game_id", "")))
        samples: list[Mapping[str, object]] = []
        for record in ordered:
            validate = getattr(record, "validate", None)
            if not callable(validate):
                raise ValueError("Current Torus9 training accepts validated Self-play game records only")
            validate()
            if getattr(record, "profile_id", None) != TORUS9_CURRENT_PROFILE_ID:
                raise ValueError("Current Torus9 training record profile mismatch")
            if getattr(record, "profile_fingerprint", None) != self.profile_fingerprint:
                raise ValueError("Current Torus9 training record profile fingerprint mismatch")
            if getattr(record, "selfplay_contract_id", None) != TORUS9_CURRENT_SELFPLAY_CONTRACT_ID:
                raise ValueError("Current Torus9 training record self-play contract mismatch")
            if getattr(record, "selfplay_contract_fingerprint", None) != current_torus9_selfplay_contract_fingerprint():
                raise ValueError("Current Torus9 training record self-play fingerprint mismatch")
            if getattr(record, "technical_termination", None) is not None:
                continue
            rows = _core.torus9_build_ownership_score_replay_samples(record)
            for row in rows:
                self.validate_sample(row)
            samples.extend(rows)
        return tuple(samples)

    def validate_sample(self, sample: Mapping[str, object]) -> None:
        if sample.get("run_id") is None or sample.get("game_id") is None:
            raise ValueError("Current Torus9 replay sample provenance is incomplete")
        _core.validate_torus9_replay_sample(
            sample,
            expected_target_fingerprint=TORUS9_CURRENT_TARGET_FINGERPRINT,
        )
        if sample.get("ownership_target") is None or sample.get("score_target") is None:
            raise ValueError("Current Torus9 training requires ownership and score targets")
        if sample.get("score_target_normalization") != 81.5:
            raise ValueError("Current Torus9 score target normalization drift")
        row_id = sample.get("replay_row_id")
        if row_id is not None:
            self._validated_sample_ids.add(str(row_id))
            self._validated_sample_objects.add(id(sample))

    def stamp_samples(
        self, samples: Sequence[Mapping[str, object]], generation: int
    ) -> Sequence[Mapping[str, object]]:
        generation = int(generation)
        stamped: list[Mapping[str, object]] = []
        for position, sample in enumerate(samples):
            row = dict(sample)
            if row.get("source_generation") is not None and int(row["source_generation"]) != generation:
                raise ValueError("Replay sample source generation disagrees with iteration")
            if row.get("replay_row_id") is not None:
                expected = f"M{generation}:{row.get('game_id')}:{row.get('ply')}:{position}"
                if str(row["replay_row_id"]) != expected:
                    raise ValueError("Current Torus9 replay row ID is not deterministic")
            else:
                row["replay_row_id"] = f"M{generation}:{row.get('game_id')}:{row.get('ply')}:{position}"
            row["source_generation"] = generation
            stamped.append(row)
        return tuple(stamped)

    def update_replay(
        self,
        replay: Torus9RollingReplay,
        generation: int,
        samples: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        for sample in samples:
            self.validate_sample(sample)
        return replay.append_generation(int(generation), samples)

    def replay_rows(self, replay: Torus9RollingReplay) -> Sequence[Mapping[str, object]]:
        return tuple(dict(row) for row in replay.rows)

    def validate_replay(self, rows: Sequence[Mapping[str, object]]) -> None:
        previous_generation = 0
        row_ids: set[str] = set()
        for row in rows:
            generation = int(row.get("source_generation", 0))
            if generation <= 0 or generation < previous_generation:
                raise ValueError("Current Torus9 replay generation ordering drift")
            row_id = str(row.get("replay_row_id", ""))
            if not row_id or row_id in row_ids:
                raise ValueError("Current Torus9 replay row IDs are missing or duplicated")
            observation = row.get("observation")
            if not isinstance(observation, (tuple, list)) or len(observation) != 6 or any(
                not isinstance(channel, (tuple, list)) or len(channel) != TORUS9_POINT_COUNT
                for channel in observation
            ):
                raise ValueError("Current Torus9 replay observation shape drift")
            if row.get("target_fingerprint") != TORUS9_CURRENT_TARGET_FINGERPRINT:
                raise ValueError("Current Torus9 replay target fingerprint drift")
            if row.get("ownership_target") is None or row.get("score_target") is None:
                raise ValueError("Current Torus9 replay auxiliary target is missing")
            if id(row) not in self._validated_sample_objects and row_id not in self._validated_sample_ids:
                self.validate_sample(row)
            previous_generation = generation
            row_ids.add(row_id)
        if len(rows) > TORUS9_MAX_REPLAY_POSITIONS:
            raise ValueError("Current Torus9 replay cap exceeded")

    def train(
        self,
        state: TrainingState,
        rows: Sequence[Mapping[str, object]],
        seed: int,
    ) -> Mapping[str, object]:
        self.validate_state(state)
        trainer = state.adapter_state
        count = TORUS9_BATCH_SIZE * TORUS9_OPTIMIZER_STEPS_PER_ITERATION
        indices = trainer._sample_indices(len(rows), seed=int(seed), count=count)
        metrics = dict(
            trainer.train_fixed_budget(
                rows,
                seed=int(seed),
                timing=self._diagnostic_timing,
                # New rows and persisted state are validated at the adapter
                # boundary; the core keeps its default validation for legacy
                # direct callers while avoiding a third O(replay) pass here.
                validate_samples=False,
            )
        )
        if self._diagnostic_timing is not None:
            metrics["stage_timing"] = dict(self._diagnostic_timing)
        metrics["training_seed"] = int(seed)
        metrics["sampled_replay_row_ids"] = tuple(
            str(rows[index].get("replay_row_id", index)) for index in indices
        )
        metrics["sampled_row_ids_fingerprint"] = value_fingerprint(metrics["sampled_replay_row_ids"])
        if metrics.get("optimizer_steps") != TORUS9_OPTIMIZER_STEPS_PER_ITERATION:
            raise ValueError("Current Torus9 optimizer step budget drift")
        if metrics.get("samples_consumed") != count:
            raise ValueError("Current Torus9 sample exposure budget drift")
        return metrics

    def prepare_checkpoint(
        self,
        state: TrainingState,
        context: CheckpointContext,
        training_metrics: Mapping[str, object],
    ) -> Mapping[str, object]:
        code = context.code_identity or self.code_identity or capture_code_identity()
        parent = context.parent_checkpoint_identity
        metadata = _core.torus9_checkpoint_metadata(
            model=state.model,
            run_id=context.run_id,
            label=context.label,
            parent=context.parent_label,
            model_seed=TORUS9_CURRENT_MODEL_INIT_SEED,
            code=code,
            profile_fp=self.profile_fingerprint,
            completed_games=context.completed_games,
            replay_positions=context.replay_positions,
            optimizer_updates=int(state.optimizer_updates),
            samples_consumed=int(state.samples_consumed),
            ownership_loss_enabled=True,
            score_loss_enabled=True,
            profile_id=TORUS9_CURRENT_PROFILE_ID,
            target_fingerprint=TORUS9_CURRENT_TARGET_FINGERPRINT,
            selfplay_contract_id=TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
            selfplay_contract_fingerprint=str(self.profile["self_play"]["fingerprint"]),  # type: ignore[index]
            base_commit=self.base_commit,
        )
        metadata.update({
            "adam_step": int(state.optimizer_updates),
            "model_init_seed": TORUS9_CURRENT_MODEL_INIT_SEED,
            "device": context.device,
            "device_locked": True,
            "input_checkpoint": dict(parent) if isinstance(parent, Mapping) else None,
            "input_model_hash": parent.get("model_hash") if isinstance(parent, Mapping) else None,
            "parent_checkpoint_identity": dict(parent) if isinstance(parent, Mapping) else None,
            "replay_generations": list(context.replay_generations),
            "replay_fingerprint": context.replay_fingerprint,
            "sampled_row_ids_fingerprint": context.sampled_row_ids_fingerprint,
            "training_seed": int(context.training_seed),
            "fresh_replay_positions": int(context.fresh_positions),
            "replay_row_count": int(context.replay_positions),
            "optimizer_parameter_order": list(_optimizer_parameter_order(state.model)),
            "optimizer_parameter_groups": [
                {
                    "lr": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                    "betas": list(group["betas"]),
                    "eps": float(group["eps"]),
                }
                for group in state.optimizer.param_groups
            ],
            "scientific_contract": {
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "batch_size": TORUS9_BATCH_SIZE,
                "optimizer_steps": TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
                "samples_consumed": TORUS9_BATCH_SIZE * TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
                "scheduler": None,
                "model_gating": False,
                "replay_generations": TORUS9_ROLLING_GENERATIONS,
                "replay_cap": TORUS9_MAX_REPLAY_POSITIONS,
                "komi": TORUS9_KOMI,
                "ownership_loss": True,
                "score_loss": True,
            },
            "execution_only_parameters": {
                "self_play_inference_batch_cap": "not checkpoint semantic",
                "self_play_inference_batch_wait_ms": "not checkpoint semantic",
            },
        })
        return metadata

    def save_checkpoint(
        self,
        path: Path,
        state: TrainingState,
        metadata: Mapping[str, object],
    ) -> Mapping[str, object]:
        return _core.torus9_save_checkpoint(
            path,
            model=state.model,
            optimizer=state.optimizer,
            metadata=metadata,
        )

    @staticmethod
    def artifact_hash(path: Path) -> str:
        return file_sha256(path)

    def verify_checkpoint(
        self,
        path: Path,
        state: TrainingState,
        metadata: Mapping[str, object],
    ) -> None:
        if not path.is_file() or not path.with_suffix(".metadata.json").is_file():
            raise ValueError("Current Torus9 checkpoint or metadata sidecar is missing")
        sidecar = _read_json(path.with_suffix(".metadata.json"))
        for key in (
            "checkpoint_schema_version",
            "profile_id",
            "profile_fingerprint",
            "target_fingerprint",
            "architecture_id",
            "model_hash",
            "optimizer_updates",
            "train_samples_consumed",
            "adam_step",
        ):
            if sidecar.get(key) != metadata.get(key):
                raise ValueError(f"Current Torus9 checkpoint metadata sidecar mismatch for {key}")
        self._validate_checkpoint_metadata(metadata, require_optimizer=True, require_stage3_fields=True)
        model = _core.torus9_model_from_metadata(metadata)
        optimizer = _core.torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        loaded = _core.torus9_load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected={
                "model_hash": metadata["model_hash"],
                "profile_id": TORUS9_CURRENT_PROFILE_ID,
                "profile_fingerprint": self.profile_fingerprint,
                "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
            },
            device=next(state.model.parameters()).device,
        )
        if _adam_step(optimizer) != int(loaded.get("optimizer_updates", -1)):
            raise ValueError("Current Torus9 checkpoint Adam state does not match metadata")
        if list(metadata.get("optimizer_parameter_order", ())) != list(_optimizer_parameter_order(model)):
            raise ValueError("Current Torus9 checkpoint optimizer parameter ordering drift")

    def _validate_checkpoint_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        require_optimizer: bool,
        require_stage3_fields: bool,
    ) -> None:
        required = (
            "checkpoint_schema_version",
            "checkpoint_label",
            "run_id",
            "profile_id",
            "profile_fingerprint",
            "target_contract_id",
            "target_fingerprint",
            "architecture_id",
            "architecture_config",
            "architecture_fingerprint",
            "model_hash",
            "topology_fingerprint",
            "board_size",
            "komi",
            "network_heads_and_shapes",
            "optimizer_updates",
            "train_samples_consumed",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError("Current Torus9 checkpoint metadata is incomplete: " + ", ".join(missing))
        if metadata.get("checkpoint_schema_version") != 1:
            raise ValueError("Current Torus9 checkpoint schema mismatch")
        if metadata.get("profile_id") != TORUS9_CURRENT_PROFILE_ID or metadata.get("profile_fingerprint") != self.profile_fingerprint:
            raise ValueError("Current Torus9 checkpoint profile mismatch")
        if metadata.get("target_contract_id") != TORUS9_TARGET_CONTRACT_ID or metadata.get("target_fingerprint") != TORUS9_CURRENT_TARGET_FINGERPRINT:
            raise ValueError("Current Torus9 checkpoint target fingerprint mismatch")
        if metadata.get("architecture_id") != TORUS9_CURRENT_ARCHITECTURE_ID:
            raise ValueError("Current Torus9 checkpoint architecture mismatch")
        if metadata.get("topology_fingerprint") != _core.TORUS9_TOPOLOGY_FINGERPRINT or metadata.get("board_size") != [9, 9] or metadata.get("komi") != TORUS9_KOMI:
            raise ValueError("Current Torus9 checkpoint topology/komi mismatch")
        if not _SHA256_RE.fullmatch(str(metadata.get("model_hash"))):
            raise ValueError("Current Torus9 checkpoint model hash is malformed")
        if require_optimizer:
            if int(metadata.get("optimizer_updates", -1)) <= 0:
                raise ValueError("Current Torus9 resume requires a positive optimizer update count")
            if int(metadata.get("train_samples_consumed", -1)) < 0:
                raise ValueError("Current Torus9 checkpoint sample clock is malformed")
        if require_stage3_fields:
            for key in (
                "adam_step",
                "replay_generations",
                "replay_fingerprint",
                "sampled_row_ids_fingerprint",
                "training_seed",
                "optimizer_parameter_order",
                "optimizer_parameter_groups",
            ):
                if key not in metadata:
                    raise ValueError(f"Current Torus9 Stage-3 checkpoint field is missing: {key}")
            if int(metadata["adam_step"]) != int(metadata["optimizer_updates"]):
                raise ValueError("Current Torus9 checkpoint Adam step mismatch")
            if int(metadata["training_seed"]) <= 0:
                raise ValueError("Current Torus9 checkpoint training seed is malformed")

    def load_state(
        self,
        checkpoint_path: str | Path,
        *,
        replay_path: str | Path,
        device: str = "cpu",
        allow_reference: bool = False,
        total_evictions: int = 0,
    ) -> TrainingState:
        checkpoint_path = Path(checkpoint_path)
        metadata_path = checkpoint_path.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            raise ValueError("Current Torus9 checkpoint metadata sidecar is missing")
        metadata = _read_json(metadata_path)
        self._validate_checkpoint_metadata(
            metadata,
            require_optimizer=True,
            require_stage3_fields=not allow_reference,
        )
        model = _core.torus9_model_from_metadata(metadata).to(device)
        trainer = Torus9OwnershipScoreTrainer(
            model,
            score_loss_enabled=True,
            learning_rate=0.001,
            weight_decay=0.0,
            optimizer_steps_per_iteration=TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
        )
        _core.torus9_load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=trainer.optimizer,
            expected={
                "model_hash": metadata["model_hash"],
                "profile_id": TORUS9_CURRENT_PROFILE_ID,
                "profile_fingerprint": self.profile_fingerprint,
                "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
            },
            device=device,
        )
        trainer.update_count = int(metadata["optimizer_updates"])
        trainer.samples_consumed = int(metadata["train_samples_consumed"])
        if _adam_step(trainer.optimizer) != trainer.update_count:
            raise ValueError("Current Torus9 resume optimizer step mismatch")
        rows = _read_jsonl(Path(replay_path))
        replay = Torus9RollingReplay.from_persisted_rows(
            rows,
            generations=int(self.replay_profile["generations"]),  # type: ignore[index]
            maximum_positions=int(self.replay_profile["cap"]),  # type: ignore[index]
            total_evictions=int(total_evictions),
        )
        self.validate_replay(rows)
        if len(rows) != int(metadata["valid_replay_positions"]):
            raise ValueError("Current Torus9 resume replay position count mismatch")
        if not allow_reference:
            if sequence_fingerprint(rows) != metadata.get("replay_fingerprint"):
                raise ValueError("Current Torus9 resume replay fingerprint mismatch")
            if int(metadata.get("replay_row_count", -1)) != len(rows):
                raise ValueError("Current Torus9 resume replay row count mismatch")
        label = _checkpoint_label(metadata)
        current_generation = int(label[1:]) if label and label.startswith("M") and label[1:].isdigit() else 0
        parent_identity = {
            "label": label,
            "path": str(checkpoint_path),
            "metadata_path": str(metadata_path),
            "model_hash": metadata["model_hash"],
            "artifact_sha256": file_sha256(checkpoint_path),
        }
        state = self.create_state(
            model,
            run_id=str(metadata["run_id"]),
            replay=replay,
            trainer=trainer,
            parent_checkpoint_identity=parent_identity,
            completed_games=int(metadata.get("completed_games", 0)),
            current_generation=current_generation,
        )
        return state

    def reconstruct_replay(
        self,
        *,
        fresh_paths: Sequence[str | Path],
        expected_rolling_path: str | Path,
        total_evictions: int = 0,
        validate_evicted_generations: bool = True,
    ) -> Torus9RollingReplay:
        replay = Torus9RollingReplay(
            generations=int(self.replay_profile["generations"]),  # type: ignore[index]
            maximum_positions=int(self.replay_profile["cap"]),  # type: ignore[index]
        )
        for generation, path_value in enumerate(fresh_paths, 1):
            is_in_rolling_window = generation > len(fresh_paths) - int(self.replay_profile["generations"])  # type: ignore[arg-type]
            if not validate_evicted_generations and not is_in_rolling_window:
                # Evicted generations cannot affect the training-visible
                # replay state.  Still parse and count every row so a missing
                # or malformed artifact fails closed without retaining its
                # large observations in memory.
                count = 0
                for row in _iter_jsonl(Path(path_value)):
                    if int(row.get("source_generation", 0)) != generation:
                        raise ValueError("Historical replay generation stamp drift")
                    count += 1
                if count <= 0:
                    raise ValueError("Historical replay generation is empty")
                continue
            rows = tuple(_iter_jsonl(Path(path_value)))
            if validate_evicted_generations or is_in_rolling_window:
                for row in rows:
                    self.validate_sample(row)
            replay.append_generation(generation, rows)
        if list(replay.rows) != list(_read_jsonl(Path(expected_rolling_path))):
            raise ValueError("Reconstructed Torus9 rolling replay differs from persisted state")
        replay.total_evictions = int(total_evictions)
        return replay

    def snapshot_state(self, state: TrainingState) -> _Torus9StateSnapshot:
        replay = state.rolling_replay
        return _Torus9StateSnapshot(
            model_state=copy.deepcopy({name: value.detach().clone() for name, value in state.model.state_dict().items()}),
            optimizer_state=copy.deepcopy(state.optimizer.state_dict()),
            optimizer_updates=int(state.optimizer_updates),
            samples_consumed=int(state.samples_consumed),
            current_generation=int(state.current_generation),
            completed_games=int(state.completed_games),
            replay_rows=tuple(dict(row) for row in replay.rows),
            replay_last_generation=int(replay.last_generation),
            replay_evictions=int(replay.total_evictions),
            parent_checkpoint_identity=(
                dict(state.parent_checkpoint_identity)
                if state.parent_checkpoint_identity is not None
                else None
            ),
        )

    def restore_state(self, state: TrainingState, snapshot: _Torus9StateSnapshot) -> None:
        state.model.load_state_dict(snapshot.model_state, strict=True)
        state.optimizer.load_state_dict(snapshot.optimizer_state)
        state.optimizer_updates = int(snapshot.optimizer_updates)
        state.samples_consumed = int(snapshot.samples_consumed)
        state.current_generation = int(snapshot.current_generation)
        state.completed_games = int(snapshot.completed_games)
        replay = state.rolling_replay
        replay._rows = [dict(row) for row in snapshot.replay_rows]
        replay._last_generation = int(snapshot.replay_last_generation)
        replay.total_evictions = int(snapshot.replay_evictions)
        state.parent_checkpoint_identity = (
            dict(snapshot.parent_checkpoint_identity)
            if snapshot.parent_checkpoint_identity is not None
            else None
        )
        self.sync_state(state)

    def sync_state(self, state: TrainingState) -> None:
        trainer = state.adapter_state
        state.optimizer = trainer.optimizer
        state.optimizer_updates = int(trainer.update_count)
        state.samples_consumed = int(trainer.samples_consumed)


def run_torus9_training_iteration(
    *,
    state: TrainingState,
    generation: int,
    output_dir: str | Path,
    run_id: str,
    training_seed: int,
    records: Sequence[object] | None = None,
    samples: Sequence[Mapping[str, object]] | None = None,
    adapter: Torus9TrainingAdapter | None = None,
    **kwargs: object,
) -> TrainingIterationResult:
    """The single current Torus9 production training front door."""
    selected = adapter or Torus9TrainingAdapter()
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
    "Torus9CurrentGraphNet",
    "Torus9GraphNet",
    "Torus9OwnershipGraphNet",
    "Torus9OwnershipScoreGraphNet",
    "Torus9OwnershipScoreTrainer",
    "Torus9OwnershipTrainer",
    "Torus9RollingReplay",
    "Torus9TrainingAdapter",
    "TrainingState",
    "run_torus9_training_iteration",
    "torus9_checkpoint_metadata",
    "torus9_load_checkpoint",
    "torus9_model_from_metadata",
    "torus9_save_checkpoint",
    "validate_torus9_replay_sample",
]
