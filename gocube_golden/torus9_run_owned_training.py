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
from .artifact_catalog import ARTIFACT_VALIDATION_SCHEMA
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
        # Row IDs loaded from an immutable replay artifact are trusted only
        # after that artifact's byte identity and durable validation evidence
        # were verified.  This lets later whole-replay structural checks avoid
        # re-running per-row semantic reconstruction for historical rows.
        self._trusted_historical_row_ids: set[str] = set()

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

    def _trust_historical_rows(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Mark rows whose immutable artifact already passed semantic validation."""
        row_ids = {str(row.get("replay_row_id", "")) for row in rows}
        if "" in row_ids or len(row_ids) != len(rows):
            raise ValueError("Trusted Torus9 replay row IDs are missing or duplicated")
        self._trusted_historical_row_ids.update(row_ids)

    def validate_replay(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Validate replay structure and deep-check only rows not already trusted."""
        validation_started = time.perf_counter()
        cache_lookup_elapsed = 0.0
        row_fingerprint_elapsed = 0.0
        semantic_fallback_elapsed = 0.0
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
                not isinstance(channel, (tuple, list)) or len(channel) != _base.TORUS9_POINT_COUNT
                for channel in observation
            ):
                raise ValueError("Current Torus9 replay observation shape drift")
            if row.get("target_fingerprint") != TORUS9_CURRENT_TARGET_FINGERPRINT:
                raise ValueError("Current Torus9 replay target fingerprint drift")
            if row.get("ownership_target") is None or row.get("score_target") is None:
                raise ValueError("Current Torus9 replay auxiliary target is missing")
            lookup_started = time.perf_counter()
            if row_id not in self._trusted_historical_row_ids:
                cached_fingerprint = self._validated_sample_fingerprints.get(row_id)
            else:
                cached_fingerprint = None
            cache_lookup_elapsed += time.perf_counter() - lookup_started
            if row_id not in self._trusted_historical_row_ids:
                fingerprint_started = time.perf_counter()
                current_fingerprint = _base.value_fingerprint(row)
                row_fingerprint_elapsed += time.perf_counter() - fingerprint_started
                if cached_fingerprint != current_fingerprint:
                    semantic_started = time.perf_counter()
                    self.validate_sample(row)
                    semantic_fallback_elapsed += time.perf_counter() - semantic_started
            previous_generation = generation
            row_ids.add(row_id)
        if len(rows) > int(self.replay_profile["cap"]):  # type: ignore[index]
            raise ValueError("Current Torus9 replay cap exceeded")
        if self._diagnostic_timing is not None:
            total = time.perf_counter() - validation_started
            self._diagnostic_timing.update({
                "replay_validation_cache_lookup_wall_time_sec": cache_lookup_elapsed,
                "replay_validation_per_row_fingerprint_wall_time_sec": row_fingerprint_elapsed,
                "replay_validation_semantic_fallback_wall_time_sec": semantic_fallback_elapsed,
                "replay_validation_semantic_fallback_and_cache_update_wall_time_sec": semantic_fallback_elapsed,
                "replay_validation_structural_and_other_wall_time_sec": max(
                    0.0,
                    total
                    - cache_lookup_elapsed
                    - row_fingerprint_elapsed
                    - semantic_fallback_elapsed,
                ),
                "replay_validation_accounted_sum_sec": total,
            })

    @staticmethod
    def _verified_replay_evidence(
        identity: Mapping[str, object] | None,
        digest: Mapping[str, object],
        *,
        row_count: int,
    ) -> tuple[str | None, bool]:
        """Verify persisted replay identity and decide whether deep validation may be skipped."""
        if identity is None:
            return None, False

        expected_sha = str(identity.get("sha256") or identity.get("artifact_sha256") or "")
        if expected_sha and expected_sha != str(digest.get("sha256", "")):
            raise ValueError("Current Torus9 replay artifact SHA-256 mismatch")

        expected_size = identity.get("size_bytes")
        if expected_size is not None and int(expected_size) != int(digest.get("size_bytes", -1)):
            raise ValueError("Current Torus9 replay artifact size mismatch")

        expected_rows = identity.get("row_count")
        if expected_rows is not None and int(expected_rows) != int(row_count):
            raise ValueError("Current Torus9 replay artifact row count mismatch")

        expected_schema = identity.get("validation_schema")
        if expected_schema not in (None, ARTIFACT_VALIDATION_SCHEMA):
            raise ValueError("Current Torus9 replay validation schema mismatch")

        expected_content = identity.get("canonical_replay_fingerprint")
        trusted = (
            bool(expected_sha)
            and expected_size is not None
            and expected_rows is not None
            and expected_schema == ARTIFACT_VALIDATION_SCHEMA
            and expected_content is not None
        )
        return (str(expected_content) if trusted else None), trusted

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

    def _rolling_from_sources(
        self,
        sources: Sequence[Path],
        *,
        total_evictions: int,
        generation_identities: Sequence[Mapping[str, object]] | None = None,
    ) -> tuple[_base.Torus9RollingReplay, list[dict[str, object]]]:
        replay = _base.Torus9RollingReplay(
            generations=int(self.replay_profile["generations"]),  # type: ignore[index]
            maximum_positions=int(self.replay_profile["cap"]),  # type: ignore[index]
        )
        replay.total_evictions = int(total_evictions)
        source_digests: list[dict[str, object]] = []
        seen_generations: set[int] = set()
        supplied_identities = {
            int(component["generation"]): dict(component)
            for component in _base._normalize_generation_identities(
                generation_identities,
                label="Torus9 referenced replay generation identities",
            )
        }

        # Process one source at a time and feed generations through the same
        # rolling replay object used during normal training.  This deliberately
        # avoids concatenating the whole bootstrap corpus and then rejecting it
        # merely because the source total exceeds the configured cap.
        for source in sources:
            source_rows, digest = _base._read_jsonl_with_identity(source)
            source_digests.append(digest)
            by_generation: dict[int, list[dict[str, object]]] = {}
            for row in source_rows:
                generation = int(row.get("source_generation", 0))
                if generation <= 0:
                    raise ValueError("Referenced Torus9 replay generation is malformed")
                by_generation.setdefault(generation, []).append(dict(row))
            if source.name.endswith("-fresh.jsonl") and len(by_generation) != 1:
                raise ValueError(
                    "Referenced Torus9 fresh replay artifact must contain exactly one generation"
                )
            for generation in sorted(by_generation):
                if generation in seen_generations:
                    raise ValueError(
                        f"Referenced Torus9 replay generation M{generation} appears in multiple sources"
                    )
                rows = by_generation[generation]
                for row in rows:
                    self.validate_sample(row)
                identity = supplied_identities.get(generation)
                if len(by_generation) == 1 and source.name.endswith("-fresh.jsonl"):
                    # A fresh source contains exactly one immutable generation;
                    # its already-computed physical SHA is the durable identity.
                    if identity is not None:
                        if str(identity.get("sha256", "")) != str(digest["sha256"]):
                            raise ValueError(
                                f"Referenced Torus9 fresh replay M{generation} SHA-256 mismatch"
                            )
                        declared_rows = identity.get("row_count")
                        if declared_rows is not None and int(declared_rows) != len(rows):
                            raise ValueError(
                                f"Referenced Torus9 fresh replay M{generation} row count mismatch"
                            )
                    identity = _base.replay_generation_identity_from_artifact(
                        generation,
                        digest["sha256"],
                        len(rows),
                    )
                replay.append_generation(
                    generation,
                    rows,
                    generation_identity=identity,
                )
                seen_generations.add(generation)

        if not seen_generations:
            raise ValueError("Referenced Torus9 replay is empty")
        # Every retained row above has already passed the full semantic
        # validator.  Mark only the final rolling window as trusted so the
        # subsequent whole-replay structural pass does not hash/deep-check it
        # again.
        self._trust_historical_rows(replay.rows)
        return replay, source_digests

    def load_state(
        self,
        checkpoint_path: str | Path,
        *,
        replay_path: str | Path,
        replay_paths: Sequence[str | Path] | None = None,
        device: str = "cpu",
        allow_reference: bool = False,
        total_evictions: int = 0,
        replay_artifact_identity: Mapping[str, object] | None = None,
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
            self._reference_sources(checkpoint_path, fallback_sources)
            if allow_reference
            else fallback_sources
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
            )
            rows = list(replay.rows)
            replay_digest = {
                "sha256": "multi-source-reference",
                "size_bytes": sum(int(item["size_bytes"]) for item in source_digests),
            }
        else:
            source_rows, replay_digest = _base._read_jsonl_with_identity(sources[0])
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
