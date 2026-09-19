"""Torus9 training adapter with run-owned LR and replay settings.

Adam moments/step are inherited from the parent checkpoint, while the selected
run learning rate is deliberately re-applied after ``optimizer.load_state_dict``.
Replay reconstruction applies the selected generation window and cap instead of
comparing either value with Golden defaults.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, MutableMapping, Sequence

from . import torus9_monolith as _core
from . import torus9_training as _base
from .provenance import file_sha256
from .torus9_contract import (
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_KOMI,
    TORUS9_TARGET_CONTRACT_ID,
)
from .torus9_run_owned import validate_run_owned_profile
from training_engine import TrainingState, sequence_fingerprint


class Torus9TrainingAdapter(_base.Torus9TrainingAdapter):
    """Current Torus9 adapter whose tuning knobs come from the resolved run."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _validate_current_profile(profile: Mapping[str, object]) -> None:
        validate_run_owned_profile(profile)

    def _effective_lr(self) -> float:
        return float(self.training_profile["learning_rate"])  # type: ignore[index]

    def _apply_effective_lr(self, optimizer: Any) -> None:
        learning_rate = self._effective_lr()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def create_state(
        self,
        model: Any,
        *,
        run_id: str,
        replay: _base.Torus9RollingReplay | None = None,
        trainer: _base.Torus9OwnershipScoreTrainer | None = None,
        parent_checkpoint_identity: Mapping[str, object] | None = None,
        completed_games: int = 0,
        current_generation: int = 0,
    ) -> TrainingState:
        if trainer is not None:
            self._apply_effective_lr(trainer.optimizer)
        return super().create_state(
            model,
            run_id=run_id,
            replay=replay,
            trainer=trainer,
            parent_checkpoint_identity=parent_checkpoint_identity,
            completed_games=completed_games,
            current_generation=current_generation,
        )

    def validate_state(self, state: TrainingState) -> None:
        if not isinstance(state.rolling_replay, _base.Torus9RollingReplay):
            raise TypeError("Current Torus9 training requires Torus9RollingReplay")
        if not isinstance(state.model, _base.Torus9CurrentGraphNet):
            raise TypeError("Current Torus9 training requires GoldenGraphNetV2-Torus9")
        if state.model.architecture_config.get("architecture_id") != TORUS9_CURRENT_ARCHITECTURE_ID:
            raise ValueError("Current Torus9 training model architecture mismatch")
        if state.profile_identity.get("profile_fingerprint") != self.profile_fingerprint:
            raise ValueError("Current Torus9 training profile identity mismatch")
        if state.target_identity.get("fingerprint") != TORUS9_CURRENT_TARGET_FINGERPRINT:
            raise ValueError("Current Torus9 training target identity mismatch")
        trainer = state.adapter_state
        if not isinstance(trainer, _base.Torus9OwnershipScoreTrainer) or trainer.model is not state.model:
            raise TypeError("Current Torus9 training state is not bound to its trainer")
        if trainer.optimizer is not state.optimizer:
            raise TypeError("Current Torus9 training state is not bound to its optimizer")
        if float(state.optimizer.param_groups[0]["lr"]) != self._effective_lr():
            raise ValueError("Torus9 Adam learning rate disagrees with the run-owned value")
        if float(state.optimizer.param_groups[0]["weight_decay"]) != 0.0:
            raise ValueError("Current Torus9 Adam weight decay drift")
        if int(state.rolling_replay.generations) != int(self.replay_profile["generations"]):  # type: ignore[index]
            raise ValueError("Torus9 replay generation window disagrees with the run-owned value")
        if int(state.rolling_replay.maximum_positions) != int(self.replay_profile["cap"]):  # type: ignore[index]
            raise ValueError("Torus9 replay cap disagrees with the run-owned value")
        if int(state.optimizer_updates) != int(trainer.update_count):
            raise ValueError("Current Torus9 optimizer update counter drift")
        if int(state.samples_consumed) != int(trainer.samples_consumed):
            raise ValueError("Current Torus9 sample counter drift")
        if _base._adam_step(state.optimizer) != int(state.optimizer_updates):
            raise ValueError("Current Torus9 Adam step does not match training clock")

    @staticmethod
    def _parent_replay_scope(metadata: Mapping[str, object]) -> tuple[int, int]:
        scientific = metadata.get("scientific_contract")
        contract = scientific if isinstance(scientific, Mapping) else {}
        generations = metadata.get("rolling_generations", contract.get("replay_generations", 0))
        cap = metadata.get("maximum_replay_positions", contract.get("replay_cap", 0))
        try:
            return int(generations or 0), int(cap or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Referenced Torus9 parent replay scope is malformed") from exc

    def _reference_sources(
        self,
        checkpoint_path: Path,
        fallback: Sequence[Path],
    ) -> tuple[Path, ...]:
        metadata = _base._read_json(checkpoint_path.with_suffix(".metadata.json"))
        label = str(metadata.get("checkpoint_label", ""))
        if not (label.startswith("M") and label[1:].isdigit()):
            return tuple(fallback)

        generation = int(label[1:])
        requested_generations = int(self.replay_profile["generations"])  # type: ignore[index]
        requested_cap = int(self.replay_profile["cap"])  # type: ignore[index]
        parent_generations, parent_cap = self._parent_replay_scope(metadata)

        # A parent rolling artifact is sufficient only if it was built with a
        # window/cap at least as large as the child asks for.  Expanding either
        # dimension requires reconstructing from the immutable fresh artifacts;
        # silently falling back to the smaller parent rolling replay would
        # change the declared experiment.
        needs_fresh_bootstrap = (
            requested_generations > parent_generations or requested_cap > parent_cap
        )
        if not needs_fresh_bootstrap:
            sources = tuple(fallback)
            if not sources or any(not path.is_file() for path in sources):
                raise FileNotFoundError("Referenced Torus9 parent rolling replay is missing")
            return sources

        # The production driver may provide an immutable fresh-replay window
        # spanning multiple lineage roots.  Keep those exact references; the
        # local-path reconstruction below is only the fallback for callers
        # that did not resolve an external parent window.
        expected_sources = min(requested_generations, generation)
        supplied = tuple(fallback)
        if len(supplied) == expected_sources and all(path.is_file() for path in supplied):
            return supplied

        first = max(1, generation - requested_generations + 1)
        parent_root = checkpoint_path.parents[1]
        candidates = tuple(
            parent_root / "replay" / f"iter-{value:02d}-fresh.jsonl"
            for value in range(first, generation + 1)
        )
        missing = tuple(path for path in candidates if not path.is_file())
        if missing:
            names = ", ".join(path.name for path in missing)
            raise FileNotFoundError(
                "Referenced Torus9 replay bootstrap is incomplete; "
                f"required fresh window M{first}..M{generation}, missing: {names}. "
                "Refusing fallback to the smaller parent rolling replay."
            )
        return candidates

    def load_state(
        self,
        checkpoint_path: str | Path,
        *,
        replay_path: str | Path,
        replay_paths: Sequence[str | Path] | None = None,
        device: str = "cpu",
        allow_reference: bool = False,
        replay_paths_are_authoritative: bool = False,
        total_evictions: int = 0,
        replay_artifact_identity: Mapping[str, object] | None = None,
        replay_artifact_identities: Sequence[Mapping[str, object]] | None = None,
        replay_identity: Mapping[str, object] | None = None,
        load_timing: MutableMapping[str, object] | None = None,
    ) -> TrainingState:
        checkpoint_path = Path(checkpoint_path)
        metadata_path = checkpoint_path.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            raise ValueError("Current Torus9 checkpoint metadata sidecar is missing")
        metadata = _base._read_json(metadata_path)
        stage3_fields = (
            "adam_step",
            "replay_generations",
            "replay_fingerprint",
            "sampled_row_ids_fingerprint",
            "training_seed",
            "optimizer_parameter_order",
            "optimizer_parameter_groups",
        )
        # ``allow_reference`` permits an intentionally different parent
        # profile fingerprint, not a blanket bypass of checkpoint integrity.
        # Only genuinely legacy references may omit Stage-3 fields.  Modern
        # parent checkpoints (including M47) must receive the same validation
        # as an in-lineage resume.
        legacy_reference = allow_reference and not all(
            key in metadata for key in stage3_fields
        )
        self._validate_checkpoint_metadata(
            metadata,
            require_optimizer=True,
            require_stage3_fields=not legacy_reference,
            allow_profile_reference=allow_reference,
        )
        model = _core.torus9_model_from_metadata(metadata).to(device)
        trainer = _base.Torus9OwnershipScoreTrainer(
            model,
            score_loss_enabled=True,
            learning_rate=self._effective_lr(),
            weight_decay=0.0,
            optimizer_steps_per_iteration=int(self.training_profile["optimizer_steps_per_iteration"]),  # type: ignore[index]
        )
        _core.torus9_load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=trainer.optimizer,
            expected={
                "model_hash": metadata["model_hash"],
                "profile_id": TORUS9_CURRENT_PROFILE_ID,
                "profile_fingerprint": self.profile_fingerprint if not allow_reference else metadata.get("profile_fingerprint"),
                "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
            },
            device=device,
        )
        # ``optimizer.load_state_dict`` restores the parent's param-group LR.
        # Preserve Adam moments/step, but make the run-selected LR effective.
        self._apply_effective_lr(trainer.optimizer)
        trainer.update_count = int(metadata["optimizer_updates"])
        trainer.samples_consumed = int(metadata["train_samples_consumed"])
        if _base._adam_step(trainer.optimizer) != trainer.update_count:
            raise ValueError("Current Torus9 resume optimizer step mismatch")

        replay_path = Path(replay_path)
        fallback_sources = tuple(Path(value) for value in (replay_paths or (replay_path,)))
        sources = (
            fallback_sources
            if replay_paths_are_authoritative
            else (
                self._reference_sources(checkpoint_path, fallback_sources)
                if allow_reference
                else fallback_sources
            )
        )
        replay_started = time.perf_counter()
        generation_identities = _base._replay_generation_identities_from_payload(
            replay_artifact_identity,
            metadata,
        )
        identity_schema = _base._replay_identity_schema_from_payload(
            replay_artifact_identity,
            metadata,
        )
        expected_identity_fingerprint = _base._replay_fingerprint_from_payload(
            replay_artifact_identity,
            metadata,
        )

        if allow_reference or len(sources) > 1:
            replay, source_digests = self._rolling_from_sources(
                sources,
                total_evictions=total_evictions,
                generation_identities=generation_identities,
                replay_artifact_identities=replay_artifact_identities,
                replay_identity=replay_identity,
            )
            rows = list(replay.rows)
            replay_digest = {
                "sha256": "multi-source-reference",
                "size_bytes": sum(int(item["size_bytes"]) for item in source_digests),
            }
        else:
            source_rows, replay_digest = _base._read_jsonl_with_identity(sources[0])
            source_digests = [replay_digest]
            rows = list(source_rows)
            replay = _base.Torus9RollingReplay.from_persisted_rows(
                rows,
                generations=int(self.replay_profile["generations"]),  # type: ignore[index]
                maximum_positions=int(self.replay_profile["cap"]),  # type: ignore[index]
                total_evictions=int(total_evictions),
                generation_identities=generation_identities,
            )

        if load_timing is not None:
            load_timing["replay_file_load_wall_time_sec"] = time.perf_counter() - replay_started
            load_timing["replay_files_parsed"] = len(source_digests)
            load_timing["replay_bytes_parsed"] = sum(
                int(item.get("size_bytes", 0)) for item in source_digests
            )
            load_timing["restore_source"] = (
                "rolling-replay"
                if len(sources) == 1 and sources[0].name.startswith("rolling-after-")
                else "fresh-window"
            )

        replay_fingerprint: str | None = None
        trusted_historical = False
        if replay_artifact_identity is not None and not (allow_reference or len(sources) > 1):
            replay_fingerprint, trusted_historical = self._verified_replay_evidence(
                replay_artifact_identity,
                replay_digest,
                row_count=len(rows),
            )

        validation_started = time.perf_counter()
        if trusted_historical:
            self._trust_historical_rows(rows)
        self.validate_replay(rows)
        if (
            identity_schema == _base.TORUS9_REPLAY_COMPOSITION_IDENTITY_SCHEMA
            and not allow_reference
            and len(sources) == 1
        ):
            if generation_identities is None or expected_identity_fingerprint is None:
                raise ValueError("Torus9 replay composition identity evidence is incomplete")
            actual_identity = replay.replay_identity_descriptor()
            if actual_identity["fingerprint"] != expected_identity_fingerprint:
                raise ValueError("Current Torus9 replay composition fingerprint mismatch")
            replay_fingerprint = expected_identity_fingerprint
        if load_timing is not None:
            load_timing["replay_validation_wall_time_sec"] = time.perf_counter() - validation_started

        if not allow_reference and len(rows) != int(metadata["valid_replay_positions"]):
            raise ValueError("Current Torus9 resume replay position count mismatch")
        if not allow_reference:
            if replay_fingerprint is None:
                replay_fingerprint = sequence_fingerprint(rows)
            if replay_fingerprint != metadata.get("replay_fingerprint"):
                raise ValueError("Current Torus9 resume replay fingerprint mismatch")
            if int(metadata.get("replay_row_count", -1)) != len(rows):
                raise ValueError("Current Torus9 resume replay row count mismatch")

        label = _base._checkpoint_label(metadata)
        current_generation = int(label[1:]) if label and label.startswith("M") and label[1:].isdigit() else 0
        parent_identity = {
            "label": label,
            "path": str(checkpoint_path),
            "metadata_path": str(metadata_path),
            "model_hash": metadata["model_hash"],
            "artifact_sha256": file_sha256(checkpoint_path),
        }
        return self.create_state(
            model,
            run_id=str(metadata["run_id"]),
            replay=replay,
            trainer=trainer,
            parent_checkpoint_identity=parent_identity,
            completed_games=int(metadata.get("completed_games", 0)),
            current_generation=current_generation,
        )


__all__ = ["Torus9TrainingAdapter"]
