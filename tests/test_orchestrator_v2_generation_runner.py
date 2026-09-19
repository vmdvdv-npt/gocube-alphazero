from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_catalog import sha256_file
from gocube_golden.orchestrator_v2 import (
    ArtifactRef,
    CheckpointRef,
    GenerationExecutionResult,
    GenerationNotCommitted,
    GenerationRunner,
    OutputLineage,
    ResolvedGenerationInput,
    Torus9ProductionGenerationPath,
)
from tools import torus9_run_driver


def _sha(path: Path) -> str:
    return sha256_file(path)


@dataclass
class FakeProductionGenerationPath:
    captured: ResolvedGenerationInput | None = None
    committed: bool = True

    def run_generation(
        self, resolved_input: ResolvedGenerationInput
    ) -> GenerationExecutionResult:
        self.captured = resolved_input
        if not self.committed:
            return GenerationExecutionResult(
                generation=resolved_input.generation,
                committed=False,
            )

        root = resolved_input.output_lineage.root
        checkpoint_path = root / "checkpoints" / f"M{resolved_input.generation}.pt"
        commit_path = root / "runtime" / "generations" / (
            f"generation-{resolved_input.generation:04d}.json"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"child-checkpoint")
        commit_path.write_bytes(b'{"status":"COMMITTED"}\n')
        return GenerationExecutionResult(
            generation=resolved_input.generation,
            committed=True,
            checkpoint=CheckpointRef(
                topology=resolved_input.output_lineage.topology,
                lineage_id=resolved_input.output_lineage.lineage_id,
                checkpoint_id=f"M{resolved_input.generation}",
                generation=resolved_input.generation,
                path=f"checkpoints/M{resolved_input.generation}.pt",
                sha256=_sha(checkpoint_path),
            ),
            commit_artifact=ArtifactRef(
                path=f"runtime/generations/generation-{resolved_input.generation:04d}.json",
                sha256=_sha(commit_path),
            ),
        )


def _resolved_input(tmp_path: Path) -> ResolvedGenerationInput:
    return ResolvedGenerationInput(
        parent_checkpoint=object(),  # type: ignore[arg-type]
        replay_artifacts=[object(), object(), object()],  # type: ignore[list-item]
        generation=94,
        effective_config=object(),  # type: ignore[arg-type]
        output_lineage=OutputLineage("torus9", "child-lineage", tmp_path),
    )


def test_runner_forwards_resolved_parent_replay_and_config_and_returns_commit(tmp_path: Path):
    resolved = _resolved_input(tmp_path)
    production_path = FakeProductionGenerationPath()

    result = GenerationRunner(production_path).run(resolved)

    assert production_path.captured is resolved
    assert production_path.captured.parent_checkpoint is resolved.parent_checkpoint
    assert production_path.captured.replay_artifacts is resolved.replay_artifacts
    assert production_path.captured.effective_config is resolved.effective_config
    assert result.generation == 94
    assert result.committed_checkpoint.lineage_id == "child-lineage"
    assert result.commit_marker.path.endswith("generation-0094.json")


def test_runner_does_not_return_result_when_generation_is_not_committed(tmp_path: Path):
    resolved = _resolved_input(tmp_path)
    production_path = FakeProductionGenerationPath(committed=False)

    with pytest.raises(GenerationNotCommitted, match="did not reach commit"):
        GenerationRunner(production_path).run(resolved)


def test_torus9_production_path_forwards_exact_input_and_maps_committed_result(
    tmp_path: Path,
):
    resolved = _resolved_input(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "M94.pt"
    marker = tmp_path / "generation-94.complete.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"production-checkpoint")
    marker.write_bytes(b'{"status":"COMPLETED"}\n')
    captured: list[ResolvedGenerationInput] = []

    def driver(value: ResolvedGenerationInput) -> dict[str, object]:
        captured.append(value)
        return {
            "status": "COMPLETED",
            "checkpoint_reload_verified": True,
            "checkpoint": {"path": "checkpoints/M94.pt"},
            "commit_artifact": {"path": "generation-94.complete.json"},
        }

    result = GenerationRunner(Torus9ProductionGenerationPath(driver)).run(resolved)

    assert captured == [resolved]
    assert result.checkpoint.path == "checkpoints/M94.pt"
    assert result.commit_artifact.path == "generation-94.complete.json"


def test_v2_driver_restore_uses_exact_parent_and_replay_without_legacy_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parent = tmp_path / "parent" / "checkpoints" / "M1.pt"
    replay_old = tmp_path / "replay" / "iter-01-fresh.jsonl"
    replay_new = tmp_path / "replay" / "iter-02-fresh.jsonl"
    parent.parent.mkdir(parents=True)
    replay_old.parent.mkdir(parents=True)
    parent.write_bytes(b"parent")
    replay_old.write_bytes(b"old")
    replay_new.write_bytes(b"new")
    captured: dict[str, object] = {}

    class Adapter:
        profile_fingerprint = "sha256:" + "a" * 64

        def load_state(self, checkpoint_path: Path, **kwargs: object) -> object:
            captured["checkpoint"] = checkpoint_path
            captured.update(kwargs)
            return object()

    bindings = torus9_run_driver.DriverBindings(
        training_adapter_factory=lambda **_kwargs: Adapter(),
        optimizer_steps_per_iteration=1,
        scientific_validator=lambda _profile, _config: None,
    )

    def fail_legacy_discovery(**_kwargs: object) -> tuple[Path, ...]:
        raise AssertionError("V2 restore must not invoke legacy replay discovery")

    monkeypatch.setattr(
        torus9_run_driver,
        "_resolve_parent_replay_reference_paths",
        fail_legacy_discovery,
    )
    torus9_run_driver._prepare_state_v2(
        root=tmp_path / "output",
        lineage_id="child",
        generation=3,
        profile={},
        config={},
        device="cpu",
        code_identity=object(),  # type: ignore[arg-type]
        parent_checkpoint=parent,
        replay_paths=(replay_old, replay_new),
        parent_sha256=sha256_file(parent),
        replay_sha256s=(sha256_file(replay_old), sha256_file(replay_new)),
        bindings=bindings,
    )

    assert captured["checkpoint"] == parent.resolve()
    assert captured["replay_paths"] == (replay_old.resolve(), replay_new.resolve())
    assert captured["replay_paths_are_authoritative"] is True


def test_v2_parent_identity_reads_checkpoint_ref_sha(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent" / "checkpoints" / "M93.pt"
    parent_ref = CheckpointRef(
        topology="torus9",
        lineage_id="parent",
        checkpoint_id="M93",
        generation=93,
        path="checkpoints/M93.pt",
        sha256="sha256:" + "a" * 64,
    )
    generation_input = SimpleNamespace(
        parent_checkpoint=SimpleNamespace(path=parent_path, ref=parent_ref)
    )

    assert torus9_run_driver._v2_parent_checkpoint_identity(generation_input) == (
        parent_path.resolve(),
        parent_ref.sha256,
    )


def test_v2_restore_forwards_resolver_replay_evidence_without_rehashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent" / "checkpoints" / "M93.pt"
    replay = tmp_path / "parent" / "replay" / "iter-93-fresh.jsonl"
    parent.parent.mkdir(parents=True)
    replay.parent.mkdir(parents=True)
    parent.write_bytes(b"parent")
    replay.write_bytes(b"replay")
    replay_sha = sha256_file(replay)
    captured: dict[str, object] = {}

    class Adapter:
        def load_state(self, checkpoint_path: Path, **kwargs: object) -> object:
            captured.update(kwargs)
            return object()

    bindings = torus9_run_driver.DriverBindings(
        training_adapter_factory=lambda **_kwargs: Adapter(),
        optimizer_steps_per_iteration=1,
        scientific_validator=lambda _profile, _config: None,
    )
    evidence = {
        "sha256": replay_sha,
        "size_bytes": replay.stat().st_size,
        "row_count": 1,
        "validation_schema": "torus9-replay-validation-v1",
        "immutable_verified": True,
    }
    monkeypatch.setattr(
        torus9_run_driver,
        "file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("V2 restore must reuse resolver identity, not rehash replay")
        ),
    )

    torus9_run_driver._prepare_state_v2(
        root=tmp_path / "output",
        lineage_id="child",
        generation=94,
        profile={},
        config={},
        device="cpu",
        code_identity=object(),  # type: ignore[arg-type]
        parent_checkpoint=parent,
        replay_paths=(replay,),
        parent_sha256="sha256:" + "a" * 64,
        replay_sha256s=(replay_sha,),
        replay_artifact_identities=(evidence,),
        bindings=bindings,
    )

    assert captured["replay_artifact_identities"] == (evidence,)


def test_v2_restore_installs_progress_callback_before_loading_replay(tmp_path: Path) -> None:
    parent = tmp_path / "parent" / "checkpoints" / "M93.pt"
    replay = tmp_path / "parent" / "replay" / "iter-93-fresh.jsonl"
    parent.parent.mkdir(parents=True)
    replay.parent.mkdir(parents=True)
    parent.write_bytes(b"parent")
    replay.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class Adapter:
        def set_progress_callback(self, callback: object) -> None:
            captured["progress_callback"] = callback

        def load_state(self, _checkpoint_path: Path, **_kwargs: object) -> object:
            return object()

    callback = lambda *_args, **_kwargs: None
    bindings = torus9_run_driver.DriverBindings(
        training_adapter_factory=lambda **_kwargs: Adapter(),
        optimizer_steps_per_iteration=1,
        scientific_validator=lambda _profile, _config: None,
    )

    torus9_run_driver._prepare_state_v2(
        root=tmp_path / "output",
        lineage_id="child",
        generation=94,
        profile={},
        config={},
        device="cpu",
        code_identity=object(),  # type: ignore[arg-type]
        parent_checkpoint=parent,
        replay_paths=(replay,),
        parent_sha256=sha256_file(parent),
        replay_sha256s=(sha256_file(replay),),
        bindings=bindings,
        progress_callback=callback,
    )

    assert captured["progress_callback"] is callback
