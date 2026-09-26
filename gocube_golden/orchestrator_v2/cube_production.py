"""Thin Cube V2 production bridge for Orchestrator V2.

Scientific self-play/training remains in the existing Cube adapters and common
engines. This module only translates a resolved Orchestrator generation into
those contracts and publishes the common artifact graph at the existing
TrainingEngine commit boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path

from generation_driver import GenerationCompletion, GenerationDriver, GenerationRequest
from training_engine import CommitPreparation, TrainingEngine

from ..artifact_graph import ArtifactRef, CheckpointRef, publish_checkpoint_graph
from ..cube_checkpoint_v2 import file_sha256
from ..cube_generation_adapter_v2 import CubeGenerationAdapter
from ..cube_generation_v2 import CubeSelfPlayPlan, build_cube_generation_request
from ..cube_selfplay_contract import CubeSelfPlaySearchContract
from ..cube_selfplay_v2 import CubeSelfPlayExecutionConfig
from ..cube_training_contract_v2 import CubeTrainingConfig
from ..cube_training_v2 import (
    CubeTrainingGenerationResult,
    _checkpoint_source_metadata,
    load_cube_checkpoint,
    load_cube_checkpoint_for_config_transition,
)
from .generation_runner import GenerationExecutionResult, ResolvedGenerationInput


def _required(mapping: Mapping[str, object], names: tuple[str, ...], label: str) -> object:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise ValueError(f"{label} must be explicit in the resolved effective config")


def _training_config(resolved: ResolvedGenerationInput) -> CubeTrainingConfig:
    training = resolved.effective_config.config.training
    replay = resolved.effective_config.config.replay
    cap = _required(replay, ("cap",), "Cube replay cap")
    return CubeTrainingConfig(
        learning_rate=float(_required(training, ("learning_rate",), "Cube learning rate")),
        batch_size=int(_required(training, ("batch_size",), "Cube training batch size")),
        optimizer_steps=int(
            _required(
                training,
                ("optimizer_steps", "optimizer_steps_per_iteration"),
                "Cube optimizer steps",
            )
        ),
        replay_generations=int(
            _required(replay, ("generations", "window"), "Cube replay generations")
        ),
        replay_cap=None if cap is None else int(cap),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def _selfplay_plan(resolved: ResolvedGenerationInput) -> CubeSelfPlayPlan:
    raw = resolved.effective_config.config.self_play
    temperature = _required(raw, ("temperature_plies",), "Cube temperature plies")
    if not isinstance(temperature, (list, tuple)) or len(temperature) != 2:
        raise ValueError("Cube temperature_plies must contain exactly two plies")
    search = CubeSelfPlaySearchContract(
        simulations=int(
            _required(raw, ("mcts_simulations", "simulations"), "Cube self-play simulations")
        ),
        cpuct=float(_required(raw, ("cpuct",), "Cube self-play cpuct")),
        fpu=float(_required(raw, ("fpu",), "Cube self-play fpu")),
        root_noise=bool(_required(raw, ("root_noise",), "Cube self-play root_noise")),
        dirichlet_epsilon=float(
            _required(raw, ("dirichlet_epsilon", "epsilon"), "Cube Dirichlet epsilon")
        ),
        dirichlet_alpha=float(
            _required(raw, ("dirichlet_alpha", "alpha"), "Cube Dirichlet alpha")
        ),
        temperature_plies=(int(temperature[0]), int(temperature[1])),
        temperature_after=float(
            _required(raw, ("temperature_after",), "Cube post-temperature")
        ),
        resign=bool(_required(raw, ("resign",), "Cube resign policy")),
        technical_move_limit=int(
            _required(
                raw,
                ("technical_move_limit", "watchdog"),
                "Cube technical move limit",
            )
        ),
        komi=float(_required(raw, ("komi",), "Cube komi")),
    )
    return CubeSelfPlayPlan(
        games=int(
            _required(raw, ("games_per_iteration", "games"), "Cube self-play games")
        ),
        search=search,
    )


def _execution_config(resolved: ResolvedGenerationInput) -> CubeSelfPlayExecutionConfig:
    raw = resolved.effective_config.config.execution
    config = CubeSelfPlayExecutionConfig(
        workers=int(_required(raw, ("workers",), "Cube self-play workers")),
        active_games_per_worker=int(
            _required(raw, ("active_games_per_worker",), "Cube active games per worker")
        ),
        total_active_contexts=int(
            _required(raw, ("total_active_contexts", "active_contexts"), "Cube active contexts")
        ),
        inference_batch_cap=int(
            _required(raw, ("inference_batch_cap",), "Cube inference batch cap")
        ),
        inference_batch_wait_ms=float(
            _required(raw, ("inference_batch_wait_ms",), "Cube inference batch wait")
        ),
        device=str(_required(raw, ("device",), "Cube execution device")),
        process_start_method=(
            None
            if raw.get("process_start_method") is None
            else str(raw["process_start_method"])
        ),
        inference_request_timeout_s=float(raw.get("inference_request_timeout_s", 300.0)),
        worker_join_timeout_s=float(raw.get("worker_join_timeout_s", 15.0)),
    )
    overrides = resolved.execution_overrides or {}
    if overrides:
        config = replace(
            config,
            active_games_per_worker=int(
                overrides.get("active_games_per_worker", config.active_games_per_worker)
            ),
            total_active_contexts=int(
                overrides.get("total_active_contexts", config.total_active_contexts)
            ),
        )
    return config


def _seed(resolved: ResolvedGenerationInput) -> int:
    config = resolved.effective_config.config
    for mapping in (config.extensions, config.self_play):
        for key in ("master_seed", "seed"):
            if key in mapping:
                value = mapping[key]
                if type(value) is not int or value <= 0:
                    raise ValueError("Cube generation seed must be a positive integer")
                return int(value) + int(resolved.generation)
    raise ValueError("Cube generation master_seed must be explicit in effective config")


def _rolling_replay(parent) -> Path:
    provenance_path = Path(parent.provenance.path)
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cube parent provenance cannot be read") from exc
    identities = payload.get("artifact_identities") if isinstance(payload, Mapping) else None
    identity = identities.get("rolling_replay") if isinstance(identities, Mapping) else None
    if isinstance(identity, Mapping):
        relative = identity.get("path")
        expected_sha = identity.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise ValueError("Cube parent rolling replay identity is malformed")
        path = (Path(parent.owner_root) / relative).resolve()
        try:
            path.relative_to(Path(parent.owner_root).resolve())
        except ValueError as exc:
            raise ValueError("Cube parent rolling replay escapes its lineage") from exc
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError("Cube parent rolling replay identity is invalid")
        return path
    if int(parent.generation) == 0:
        path = Path(parent.owner_root) / "replay" / "rolling-after-00.jsonl"
        if not path.is_file():
            raise ValueError("Cube M0 must publish replay/rolling-after-00.jsonl")
        return path
    raise ValueError("Cube committed parent provenance lacks rolling replay identity")


class _OrchestratedCubeAdapter(CubeGenerationAdapter):
    def __init__(self, *, parent_reference, lineage_root: Path, prepare_commit, **kwargs) -> None:
        super().__init__(**kwargs)
        self._parent_reference = dict(parent_reference)
        self._lineage_root = Path(lineage_root).resolve()
        self._prepare_commit = prepare_commit

    def _validate_lineage_dir(self, request: GenerationRequest) -> None:
        if request.lineage_dir.resolve() != self._lineage_root:
            raise ValueError("Cube generation lineage path does not match Orchestrator owner")

    def _parent(self, request: GenerationRequest) -> dict[str, object]:
        if request.parent_checkpoint is None:
            raise ValueError("Cube generation requires an explicit parent checkpoint")
        return dict(self._parent_reference)

    def committed_generation_exists(self, request: GenerationRequest) -> bool:
        return (
            request.lineage_dir
            / f"generation-{request.generation:02d}.complete.json"
        ).is_file()

    def run_training(
        self, request: GenerationRequest, selfplay_result: object
    ) -> CubeTrainingGenerationResult:
        from ..cube_generation_v2 import CubeSelfPlayStageResult

        if not isinstance(selfplay_result, CubeSelfPlayStageResult):
            raise TypeError("Cube training requires CubeSelfPlayStageResult")
        engine_result = TrainingEngine(self.training_adapter).run_iteration(
            state=self.training_state,
            generation=request.generation,
            output_dir=request.lineage_dir,
            run_id=request.lineage_id,
            records=tuple(selfplay_result.records),
            training_seed=request.seed,
            completed_games=len(selfplay_result.records),
            parent_checkpoint_identity=self._parent(request),
            device=str(next(self.training_state.model.parameters()).device),
            summary_extra={
                "cube_stage": 8,
                "orchestrated": True,
                "embedded_arena": False,
            },
            prepare_commit=self._prepare_commit,
        )
        metrics = engine_result.training_metrics
        checkpoint_path = Path(engine_result.artifacts["checkpoint"])
        checkpoint_sha = file_sha256(checkpoint_path)
        if checkpoint_sha != engine_result.checkpoint_metadata.get("checkpoint_sha256"):
            raise ValueError("Committed Cube checkpoint SHA disagrees with sidecar identity")
        self.training_state.parent_checkpoint_identity = {
            "lineage_id": request.lineage_id,
            "checkpoint_id": engine_result.label,
            "label": engine_result.label,
            "path": str(checkpoint_path),
            "metadata_path": engine_result.artifacts["checkpoint_metadata"],
            "model_hash": engine_result.checkpoint_metadata.get("model_hash"),
            "artifact_sha256": checkpoint_sha,
            "sha256": checkpoint_sha,
            "generation": engine_result.generation,
            "size": self.size,
        }
        return CubeTrainingGenerationResult(
            generation=engine_result.generation,
            sample_count_new=engine_result.fresh_positions,
            replay_generations=tuple(
                int(value)
                for value in engine_result.checkpoint_metadata.get("replay_generations", ())
            ),
            replay_sample_count=int(engine_result.checkpoint_metadata["replay_positions"]),
            optimizer_steps=int(metrics["optimizer_steps"]),
            batch_size=int(metrics["batch_size"]),
            effective_learning_rate=float(metrics["effective_learning_rate"]),
            policy_loss=float(metrics["policy_loss"]),
            wdl_loss=float(metrics["wdl_loss"]),
            ownership_loss=float(metrics["ownership_loss"]),
            score_loss=float(metrics["score_loss"]),
            total_loss=float(metrics["total_loss"]),
            checkpoint_reference={
                "lineage_id": request.lineage_id,
                "checkpoint_id": engine_result.label,
                "path": str(checkpoint_path),
                "metadata_path": engine_result.artifacts["checkpoint_metadata"],
                "sha256": checkpoint_sha,
                "generation": engine_result.generation,
                "size": self.size,
            },
            checkpoint_sha256=checkpoint_sha,
            timing_breakdown=dict(metrics.get("phase_timing", {})),
            warnings=(),
            engine_result=engine_result,
        )


class CubeProductionGenerationPath:
    """Adapt one resolved Cube generation to the existing Stage-7 pipeline."""

    def __init__(self, *, size: int) -> None:
        if type(size) is not int or not 2 <= size <= 7:
            raise ValueError("Cube production bridge supports sizes 2..7")
        self.size = size

    def run_generation(self, resolved: ResolvedGenerationInput) -> GenerationExecutionResult:
        topology = f"cube{self.size}"
        if resolved.output_lineage.topology != topology:
            raise ValueError("Cube production bridge topology mismatch")
        if resolved.effective_config.config.topology != topology:
            raise ValueError("Cube effective config topology mismatch")

        training_config = _training_config(resolved)
        replay_path = _rolling_replay(resolved.parent_checkpoint)
        execution = _execution_config(resolved)
        parent_reference = resolved.parent_checkpoint.ref.to_dict()
        source_metadata = _checkpoint_source_metadata(resolved.parent_checkpoint.path)
        parent_reference.update(
            path=str(resolved.parent_checkpoint.path),
            metadata_path=str(resolved.parent_checkpoint.path.with_suffix(".metadata.json")),
            model_hash=source_metadata.get("model_hash"),
            artifact_sha256=resolved.parent_checkpoint.ref.sha256,
            sha256=resolved.parent_checkpoint.ref.sha256,
            size=self.size,
        )
        source_config = CubeTrainingConfig.from_identity_payload(
            source_metadata.get("concrete_training_config")  # type: ignore[arg-type]
        )
        transition = None
        if source_config.fingerprint == training_config.fingerprint:
            adapter, state, _ = load_cube_checkpoint(
                resolved.parent_checkpoint.path,
                config=training_config,
                replay_path=replay_path,
                expected_size=self.size,
                map_location=execution.device,
            )
        else:
            if resolved.output_lineage.lineage_id == resolved.parent_checkpoint.lineage_id:
                raise ValueError(
                    "Cube config transition requires a new child-lineage; "
                    "existing lineage config_fingerprint is immutable"
                )
            adapter, state, transition = load_cube_checkpoint_for_config_transition(
                resolved.parent_checkpoint.path,
                config=training_config,
                replay_path=replay_path,
                expected_size=self.size,
                map_location=execution.device,
                source_checkpoint_identity=parent_reference,
            )
        state.parent_checkpoint_identity = dict(parent_reference)

        committed: dict[str, object] = {}

        def prepare_commit(preparation: CommitPreparation) -> None:
            identities = preparation.artifact_identities
            checkpoint_identity = identities["checkpoint"]
            fresh_identity = identities["fresh_replay"]
            marker_identity = identities["completion_marker"]
            checkpoint = CheckpointRef(
                topology=topology,
                lineage_id=resolved.output_lineage.lineage_id,
                checkpoint_id=f"M{resolved.generation}",
                generation=resolved.generation,
                path=str(checkpoint_identity["path"]),
                sha256=str(checkpoint_identity["sha256"]),
            )
            fresh = ArtifactRef(
                str(fresh_identity["path"]), str(fresh_identity["sha256"])
            )
            marker = ArtifactRef(
                str(marker_identity["path"]), str(marker_identity["sha256"])
            )
            publish_checkpoint_graph(
                root=resolved.output_lineage.root,
                parent=resolved.parent_checkpoint.ref,
                checkpoint=checkpoint,
                fresh_replay=fresh,
                effective_config=resolved.effective_config.ref,
                generation_commit=marker,
                artifact_identities=identities,
                checkpoint_reload_verified=True,
            )
            committed.update(
                checkpoint=checkpoint,
                fresh_replay=fresh,
                commit_artifact=marker,
            )

        plan = _selfplay_plan(resolved)
        request = build_cube_generation_request(
            training_adapter=adapter,
            generation=resolved.generation,
            parent_checkpoint=parent_reference,
            selfplay_plan=plan,
            selfplay_execution_config=execution,
            training_config=training_config,
            arena_request=None,
            seed=_seed(resolved),
            lineage_id=resolved.output_lineage.lineage_id,
            lineage_dir=resolved.output_lineage.root,
        )
        scientific_adapter = _OrchestratedCubeAdapter(
            training_adapter=adapter,
            training_state=state,
            parent_reference=parent_reference,
            lineage_root=resolved.output_lineage.root,
            prepare_commit=prepare_commit,
        )
        result = GenerationDriver(scientific_adapter).run(request.to_common())
        if result.completion_status != GenerationCompletion.COMPLETE.value:
            return GenerationExecutionResult(
                generation=resolved.generation,
                committed=False,
            )
        if result.arena_result is not None or request.arena_request is not None:
            raise RuntimeError("Orchestrated Cube generation must not run embedded Arena")
        if set(committed) != {"checkpoint", "fresh_replay", "commit_artifact"}:
            raise RuntimeError(
                "Cube generation crossed completion without authoritative graph refs"
            )
        return GenerationExecutionResult(
            generation=resolved.generation,
            committed=True,
            checkpoint=committed["checkpoint"],  # type: ignore[arg-type]
            fresh_replay=committed["fresh_replay"],  # type: ignore[arg-type]
            commit_artifact=committed["commit_artifact"],  # type: ignore[arg-type]
        )


__all__ = ["CubeProductionGenerationPath"]
