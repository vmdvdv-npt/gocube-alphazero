"""Concrete V2 bridge to the existing Torus9 production driver."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from ..artifact_catalog import sha256_file
from .generation_runner import (
    GenerationExecutionResult,
    ResolvedGenerationInput,
)
from .contracts import ArtifactRef, CheckpointRef


def _default_driver(resolved_input: ResolvedGenerationInput) -> Mapping[str, object]:
    # Keep the scientific driver lazy: importing the V2 contracts remains
    # cheap for resolver/tests and does not import torch until execution.
    from tools.torus9_run_driver import run_generation_v2

    return run_generation_v2(resolved_input)


def _relative_artifact(root: Path, value: object, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Production driver result is missing {label} path")
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Production driver {label} escapes output lineage") from exc
    if not path.is_file():
        raise ValueError(f"Production driver {label} is missing: {relative}")
    return relative, path


def _artifact_identity(
    *,
    root: Path,
    payload: Mapping[str, object],
    key: str,
    fallback_path: str | None = None,
) -> ArtifactRef:
    raw = payload.get(key)
    declared: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
    raw_path = declared.get("path", fallback_path)
    relative, path = _relative_artifact(root, raw_path, key)
    declared_sha = declared.get("sha256") or declared.get("artifact_sha256")
    sha256 = str(declared_sha or sha256_file(path))
    if sha256 != sha256_file(path):
        raise ValueError(f"Production driver {key} SHA-256 mismatch")
    return ArtifactRef(path=relative, sha256=sha256)


class Torus9ProductionGenerationPath:
    """Use the existing Torus9 production generation driver as a V2 path.

    ``resolved_input`` is passed unchanged to the driver.  In particular this
    class never asks ``ArtifactResolver`` for a parent or replay window.
    ``driver`` is injectable only for cheap boundary tests; the default is the
    real production entrypoint in :mod:`tools.torus9_run_driver`.
    """

    def __init__(
        self,
        driver: Callable[[ResolvedGenerationInput], Mapping[str, object] | GenerationExecutionResult]
        | None = None,
    ) -> None:
        self._driver = driver or _default_driver

    def run_generation(
        self, resolved_input: ResolvedGenerationInput
    ) -> GenerationExecutionResult:
        if resolved_input.output_lineage.topology != "torus9":
            raise ValueError("Torus9 production path requires topology=torus9")
        produced = self._driver(resolved_input)
        if isinstance(produced, GenerationExecutionResult):
            return produced
        if not isinstance(produced, Mapping):
            raise TypeError("Torus9 production driver returned an invalid result")

        generation = int(resolved_input.generation)
        committed = (
            str(produced.get("status", "")).upper() == "COMPLETED"
            and bool(produced.get("checkpoint_reload_verified", True))
        )
        if not committed:
            return GenerationExecutionResult(generation=generation, committed=False)

        root = resolved_input.output_lineage.root
        checkpoint_artifact = _artifact_identity(
            root=root,
            payload=produced,
            key="checkpoint",
            fallback_path=f"checkpoints/M{generation}.pt",
        )
        commit_artifact = _artifact_identity(
            root=root,
            payload=produced,
            key="commit_artifact",
            fallback_path=f"generation-{generation:02d}.complete.json",
        )
        checkpoint_id = f"M{generation}"
        checkpoint_payload = produced.get("checkpoint")
        if isinstance(checkpoint_payload, Mapping) and checkpoint_payload.get("checkpoint_id"):
            checkpoint_id = str(checkpoint_payload["checkpoint_id"])
        return GenerationExecutionResult(
            generation=generation,
            committed=True,
            checkpoint=CheckpointRef(
                topology=resolved_input.output_lineage.topology,
                lineage_id=resolved_input.output_lineage.lineage_id,
                checkpoint_id=checkpoint_id,
                generation=generation,
                path=checkpoint_artifact.path,
                sha256=checkpoint_artifact.sha256,
            ),
            commit_artifact=commit_artifact,
        )


__all__ = ["Torus9ProductionGenerationPath"]
