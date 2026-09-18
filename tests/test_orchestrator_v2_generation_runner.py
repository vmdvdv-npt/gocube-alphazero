from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
)


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
