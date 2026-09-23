"""Cube V2 composition adapter for the topology-neutral GenerationDriver."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from generation_driver import (
    GenerationCompletion,
    GenerationDriver,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
)
from training_engine import TrainingState

from .cube_arena_v2 import CubeArenaResult, run_cube_arena
from .cube_checkpoint_v2 import file_sha256
from .cube_generation_v2 import (
    CubeArenaRequest,
    CubeGenerationRequest,
    CubeSelfPlayPlan,
    CubeSelfPlayStageResult,
    build_cube_generation_profile_identity,
)
from .cube_network_v2 import cube_graphnet_v2_model_hash
from .cube_selfplay_v2 import (
    CubeSelfPlayExecutionConfig,
    CubeSelfPlayGameRecord,
    run_cube_selfplay_games,
)
from .cube_training_contract_v2 import CubeTrainingConfig
from .cube_training_v2 import (
    CubeTrainingAdapter,
    CubeTrainingGenerationResult,
    run_cube_training_generation,
)
from .run_storage import active_lineage_dir, resolve_checkpoint


def _same_sha(actual: str, expected: str) -> bool:
    return str(actual).removeprefix("sha256:") == str(expected).removeprefix("sha256:")


class CubeGenerationAdapter:
    """Bind Cube scientific contracts to existing self-play/training/Arena engines."""

    def __init__(
        self,
        *,
        training_adapter: CubeTrainingAdapter,
        training_state: TrainingState,
        temporary_root: str | Path | None = None,
    ) -> None:
        self.training_adapter = training_adapter
        self.training_state = training_state
        self.temporary_root = (
            None if temporary_root is None else Path(temporary_root).resolve()
        )

    @property
    def size(self) -> int:
        return self.training_adapter.size

    def _parts(
        self, request: GenerationRequest
    ) -> tuple[
        CubeSelfPlayPlan,
        CubeSelfPlayExecutionConfig,
        CubeTrainingConfig,
        CubeArenaRequest | None,
    ]:
        if request.topology != f"cube{self.size}":
            raise ValueError("Cube generation topology does not match bound adapter")
        if not isinstance(request.selfplay_scientific_config, CubeSelfPlayPlan):
            raise TypeError("Cube generation requires CubeSelfPlayPlan")
        if not isinstance(request.selfplay_execution_config, CubeSelfPlayExecutionConfig):
            raise TypeError("Cube generation requires CubeSelfPlayExecutionConfig")
        if not isinstance(request.training_config, CubeTrainingConfig):
            raise TypeError("Cube generation requires CubeTrainingConfig")
        if request.arena_request is not None and not isinstance(
            request.arena_request, CubeArenaRequest
        ):
            raise TypeError("Cube generation Arena request has the wrong type")
        return (
            request.selfplay_scientific_config,
            request.selfplay_execution_config,
            request.training_config,
            request.arena_request,
        )

    def _validate_lineage_dir(self, request: GenerationRequest) -> None:
        target = request.lineage_dir.resolve()
        if self.temporary_root is not None:
            try:
                target.relative_to(self.temporary_root)
            except ValueError as exc:
                raise ValueError(
                    "Temporary Cube generation lineage must stay under temporary_root"
                ) from exc
            return
        expected = active_lineage_dir(
            request.topology, request.lineage_id
        ).resolve()
        if target != expected:
            raise ValueError(
                "Cube generation lineage path must be the canonical active-lineage directory"
            )

    def _parent(self, request: GenerationRequest) -> dict[str, object]:
        if request.parent_checkpoint is None:
            raise ValueError("Cube generation requires an explicit parent checkpoint")
        if self.temporary_root is None:
            return resolve_checkpoint(
                request.parent_checkpoint,
                topology=request.topology,
                expected_lineage_id=request.lineage_id,
            ).as_reference()

        value = dict(request.parent_checkpoint)
        path_value = value.get("path")
        expected_sha = value.get("sha256") or value.get("artifact_sha256")
        if not path_value or not isinstance(expected_sha, str):
            raise ValueError("Temporary Cube parent checkpoint requires path and SHA-256")
        path = Path(str(path_value)).resolve()
        try:
            path.relative_to(self.temporary_root)
        except ValueError as exc:
            raise ValueError(
                "Temporary Cube parent checkpoint must stay under temporary_root"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(f"Cube parent checkpoint does not exist: {path}")
        actual_sha = file_sha256(path)
        if not _same_sha(actual_sha, expected_sha):
            raise ValueError("Cube parent checkpoint SHA-256 mismatch")
        value.update(
            path=str(path),
            sha256=actual_sha,
            artifact_sha256=actual_sha,
        )
        return value

    def committed_generation_exists(self, request: GenerationRequest) -> bool:
        return (
            request.lineage_dir
            / f"generation-{request.generation:02d}.complete.json"
        ).exists() or (
            request.lineage_dir
            / "checkpoints"
            / f"M{request.generation}.pt"
        ).exists()

    def validate_request(self, request: GenerationRequest) -> None:
        self._validate_lineage_dir(request)
        plan, execution, training, arena = self._parts(request)
        plan.validate()
        execution.validate()
        if training != self.training_adapter.config:
            raise ValueError(
                "Cube generation training config drifted from bound training adapter"
            )
        if request.generation != int(self.training_state.current_generation) + 1:
            raise ValueError(
                "Cube generation number must be exactly one greater than current training state"
            )
        parent = self._parent(request)
        parent_generation = parent.get("generation")
        if (
            parent_generation is not None
            and int(parent_generation) != int(self.training_state.current_generation)
        ):
            raise ValueError(
                "Cube generation parent generation does not match training state"
            )
        parent_model_hash = parent.get("model_hash")
        if (
            parent_model_hash is not None
            and str(parent_model_hash)
            != cube_graphnet_v2_model_hash(self.training_state.model)
        ):
            raise ValueError(
                "Cube generation parent model hash does not match in-memory state"
            )
        state_parent = self.training_state.parent_checkpoint_identity
        if isinstance(state_parent, Mapping):
            state_sha = state_parent.get("sha256") or state_parent.get(
                "artifact_sha256"
            )
            parent_sha = parent.get("sha256") or parent.get("artifact_sha256")
            if (
                state_sha is not None
                and parent_sha is not None
                and not _same_sha(str(state_sha), str(parent_sha))
            ):
                raise ValueError(
                    "Cube generation parent checkpoint disagrees with training state"
                )
        expected_profile = build_cube_generation_profile_identity(
            training_adapter=self.training_adapter,
            selfplay_plan=plan,
            selfplay_execution_config=execution,
            arena_request=arena,
        )
        if dict(request.profile_identity) != expected_profile:
            raise ValueError("Cube generation profile identity drift")
        if arena is not None:
            arena.validate()

    def run_selfplay(self, request: GenerationRequest) -> CubeSelfPlayStageResult:
        plan, execution, _, _ = self._parts(request)
        parent = self._parent(request)
        telemetry: dict[str, object] = {}
        activity: dict[str, object] = {}
        records = run_cube_selfplay_games(
            self.training_state.model,
            tuple(
                f"M{request.generation}-g{index:04d}"
                for index in range(plan.games)
            ),
            size=self.size,
            run_id=request.lineage_id,
            model_checkpoint_label=str(
                parent.get("checkpoint_id")
                or parent.get("label")
                or f"M{request.generation - 1}"
            ),
            checkpoint_artifact_hash=str(
                parent.get("sha256") or parent.get("artifact_sha256")
            ),
            master_seed=request.seed,
            profile_id=f"cube{self.size}-v2",
            profile_fingerprint=self.training_adapter.game_fingerprint,
            contract=plan.search,
            device=execution.device,
            execution_config=execution,
            inference_telemetry=telemetry,
            execution_activity=activity,
        )
        return CubeSelfPlayStageResult(records, telemetry, activity)

    def validate_selfplay_result(
        self, request: GenerationRequest, result: object
    ) -> None:
        if not isinstance(result, CubeSelfPlayStageResult):
            raise TypeError("Cube self-play stage returned the wrong result type")
        plan, _, _, _ = self._parts(request)
        if len(result.records) != plan.games:
            raise ValueError("Cube self-play game count does not match request")
        expected_hash = cube_graphnet_v2_model_hash(self.training_state.model)
        for record in result.records:
            if not isinstance(record, CubeSelfPlayGameRecord):
                raise TypeError("Cube self-play stage returned a non-Cube record")
            record.validate(deep=True)
            if record.size != self.size or record.model_hash != expected_hash:
                raise ValueError("Cube self-play result identity drift")

    def summarize_selfplay(
        self, request: GenerationRequest, result: object
    ) -> Mapping[str, object]:
        if not isinstance(result, CubeSelfPlayStageResult):
            raise TypeError("Cube self-play summary requires CubeSelfPlayStageResult")
        formal = sum(record.formal_result is not None for record in result.records)
        return {
            "games_requested": len(result.records),
            "formal_games": formal,
            "technical_games": len(result.records) - formal,
            "telemetry": dict(result.telemetry),
            "execution_activity": dict(result.execution_activity),
        }

    def run_training(
        self, request: GenerationRequest, selfplay_result: object
    ) -> CubeTrainingGenerationResult:
        if not isinstance(selfplay_result, CubeSelfPlayStageResult):
            raise TypeError("Cube training requires CubeSelfPlayStageResult")
        return run_cube_training_generation(
            adapter=self.training_adapter,
            state=self.training_state,
            generation=request.generation,
            lineage_dir=request.lineage_dir,
            run_id=request.lineage_id,
            training_seed=request.seed,
            records=selfplay_result.records,
            completed_games=len(selfplay_result.records),
            parent_checkpoint_identity=self._parent(request),
            device=str(next(self.training_state.model.parameters()).device),
        )

    def summarize_training(
        self, request: GenerationRequest, result: object
    ) -> Mapping[str, object]:
        if not isinstance(result, CubeTrainingGenerationResult):
            raise TypeError(
                "Cube training summary requires CubeTrainingGenerationResult"
            )
        checkpoint = dict(result.checkpoint_reference)
        checkpoint.update(
            sha256=result.checkpoint_sha256,
            artifact_sha256=result.checkpoint_sha256,
            model_hash=result.engine_result.checkpoint_metadata.get("model_hash"),
        )
        return {
            "samples_generated": result.sample_count_new,
            "replay_state": {
                "generations": list(result.replay_generations),
                "positions": result.replay_sample_count,
                "fingerprint": result.engine_result.provenance.get(
                    "replay_fingerprint"
                ),
            },
            "training_metrics": dict(result.engine_result.training_metrics),
            "checkpoint_reference": checkpoint,
        }

    def run_arena(
        self, request: GenerationRequest, training_result: object
    ) -> CubeArenaResult:
        if not isinstance(training_result, CubeTrainingGenerationResult):
            raise TypeError("Cube Arena requires CubeTrainingGenerationResult")
        _, _, _, arena = self._parts(request)
        if arena is None:
            raise RuntimeError("Cube Arena was not requested")
        candidate = dict(training_result.checkpoint_reference)
        candidate.update(
            sha256=training_result.checkpoint_sha256,
            artifact_sha256=training_result.checkpoint_sha256,
            model_hash=training_result.engine_result.checkpoint_metadata.get(
                "model_hash"
            ),
        )
        reference = dict(arena.reference_checkpoint)
        reference_id = str(
            reference.get("checkpoint_id")
            or reference.get("label")
            or Path(str(reference.get("path", "reference"))).stem
        )
        return run_cube_arena(
            size=self.size,
            candidate_checkpoint=candidate,
            reference_checkpoint=reference,
            output_dir=(
                request.lineage_dir
                / "arena"
                / f"M{request.generation}-vs-{reference_id}"
            ),
            search_config=arena.search_config,
            execution_config=arena.execution_config,
            seed=arena.seed,
            temporary_root=self.temporary_root,
        )

    def summarize_arena(
        self, request: GenerationRequest, result: object
    ) -> Mapping[str, object]:
        if not isinstance(result, CubeArenaResult):
            raise TypeError("Cube Arena summary requires CubeArenaResult")
        return result.to_dict()

    def validate_result(
        self, request: GenerationRequest, result: GenerationResult
    ) -> None:
        if result.generation != request.generation:
            raise ValueError("Cube GenerationResult generation drift")
        if result.completion_status == GenerationCompletion.FAILED.value:
            if GenerationStage.CHECKPOINT_COMMITTED.value in result.stages:
                raise ValueError(
                    "Failed Cube generation cannot claim a committed checkpoint"
                )
            return
        if result.checkpoint_reference is None:
            raise ValueError("Committed Cube generation lost checkpoint reference")
        path = Path(str(result.checkpoint_reference.get("path", "")))
        expected_sha = result.checkpoint_reference.get(
            "sha256"
        ) or result.checkpoint_reference.get("artifact_sha256")
        if not path.is_file() or not isinstance(expected_sha, str):
            raise ValueError(
                "Cube GenerationResult checkpoint reference is incomplete"
            )
        if not _same_sha(file_sha256(path), expected_sha):
            raise ValueError("Cube GenerationResult checkpoint SHA-256 mismatch")
        required = {
            GenerationStage.SELFPLAY_COMPLETE.value,
            GenerationStage.TRAINING_COMPLETE.value,
            GenerationStage.CHECKPOINT_COMMITTED.value,
        }
        if not required.issubset(set(result.stages)):
            raise ValueError("Cube GenerationResult lost committed stage markers")
        if result.completion_status == GenerationCompletion.COMPLETE.value:
            if request.arena_request is not None and result.arena_result is None:
                raise ValueError(
                    "Completed Cube generation lost requested Arena result"
                )
            if GenerationStage.GENERATION_COMPLETE.value not in result.stages:
                raise ValueError(
                    "Completed Cube generation lost completion marker"
                )
        elif result.completion_status == GenerationCompletion.ARENA_FAILED.value:
            if request.arena_request is None or result.error is None:
                raise ValueError("Malformed Cube arena_failed result")
        else:
            raise ValueError("Unknown Cube GenerationResult completion status")

    def persist_result(
        self, request: GenerationRequest, result: GenerationResult
    ) -> None:
        # No alternate manifest: TrainingEngine owns its commit marker; Arena
        # owns its result. Future Orchestrator remains manifest/lifecycle owner.
        return None


def run_cube_generation(
    request: CubeGenerationRequest,
    *,
    training_adapter: CubeTrainingAdapter,
    training_state: TrainingState,
    temporary_root: str | Path | None = None,
) -> GenerationResult:
    request.validate()
    return GenerationDriver(
        CubeGenerationAdapter(
            training_adapter=training_adapter,
            training_state=training_state,
            temporary_root=temporary_root,
        )
    ).run(request.to_common())


__all__ = ["CubeGenerationAdapter", "run_cube_generation"]
