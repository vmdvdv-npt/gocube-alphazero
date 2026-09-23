"""Generation-scoped recovery for work published before the commit fence.

The authoritative completion marker is the final generation commit fence. If
that marker is absent, a crashed attempt may leave generation-owned training
artifacts plus graph/manifest/catalog evidence behind. This module removes
only evidence owned by that incomplete generation so the same generation can
be retried safely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ..artifact_catalog import ArtifactCatalog
from ..process_supervision import atomic_write_text
from ..provenance import canonical_json


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object: {path}")
    return payload


def _generation_paths(root: Path, generation: int) -> tuple[Path, ...]:
    return (
        root / "replay" / f"iter-{generation:02d}-fresh.jsonl",
        root / "replay" / f"rolling-after-{generation:02d}.jsonl",
        root / "checkpoints" / f"M{generation}.pt",
        root / "checkpoints" / f"M{generation}.metadata.json",
        root / "training" / f"iter-{generation:02d}.json",
        root / f"iter-{generation:02d}-summary.json",
        root / "metadata" / "provenance-v2" / f"M{generation}.json",
        root / "metadata" / "checkpoints" / f"M{generation}.json",
    )


def _temporary_paths(root: Path, generation: int) -> tuple[Path, ...]:
    return (
        root / "replay" / f".iter-{generation:02d}.tmp-fresh.jsonl",
        root / "replay" / f".rolling-after-{generation:02d}.tmp.jsonl",
        root / "checkpoints" / f".M{generation}.tmp.pt",
        root / "checkpoints" / f".M{generation}.tmp.metadata.json",
        root / "training" / f".iter-{generation:02d}.tmp.json",
        root / f".iter-{generation:02d}-summary.tmp.json",
        root / f".generation-{generation:02d}.complete.tmp.json",
    )


def _validate_graph_owner(
    path: Path,
    *,
    lineage_id: str,
    generation: int,
    checkpoint_path: str,
    label: str,
) -> None:
    if not path.is_file():
        return
    payload = _read_object(path, label)
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"{label} does not identify its checkpoint")
    if (
        checkpoint.get("lineage_id") != lineage_id
        or checkpoint.get("generation") != generation
        or checkpoint.get("path") != checkpoint_path
    ):
        raise RuntimeError(
            f"refusing to remove {label} that is not owned by "
            f"{lineage_id}/M{generation}"
        )


def _later_generation_evidence_exists(
    root: Path,
    *,
    generation: int,
    checkpoint_hashes: Mapping[str, object],
    generation_commits: Mapping[str, object],
) -> bool:
    for marker in root.glob("generation-*.complete.json"):
        name = marker.name
        raw = name[len("generation-") : -len(".complete.json")]
        if raw.isdigit() and int(raw) > generation:
            return True
    for relative in checkpoint_hashes:
        prefix = "checkpoints/M"
        suffix = ".pt"
        if str(relative).startswith(prefix) and str(relative).endswith(suffix):
            raw = str(relative)[len(prefix) : -len(suffix)]
            if raw.isdigit() and int(raw) > generation:
                return True
    for raw_generation in generation_commits:
        if str(raw_generation).isdigit() and int(raw_generation) > generation:
            return True
    return False


def reconcile_uncommitted_generation(
    *,
    root: str | Path,
    lineage_id: str,
    generation: int,
) -> bool:
    """Remove one incomplete generation while preserving committed history.

    The function is idempotent and deliberately refuses to act if the
    authoritative completion marker exists. It also fails closed if later
    generation evidence exists or if generation-named graph files identify
    another owner.
    """
    if type(generation) is not int or generation <= 0:
        raise ValueError("generation must be a positive integer")
    lineage_root = Path(root).resolve()
    marker_path = lineage_root / f"generation-{generation:02d}.complete.json"
    if marker_path.is_file():
        return False

    manifest_path = lineage_root / "manifest.json"
    manifest = _read_object(manifest_path, "lineage manifest")
    if manifest.get("lineage_id") != lineage_id:
        raise RuntimeError("lineage manifest owner does not match recovery request")

    raw_hashes = manifest.get("checkpoint_hashes")
    if not isinstance(raw_hashes, Mapping):
        raise RuntimeError("lineage manifest checkpoint_hashes is malformed")
    raw_commits = manifest.get("generation_commits", {})
    if not isinstance(raw_commits, Mapping):
        raise RuntimeError("lineage manifest generation_commits is malformed")
    if _later_generation_evidence_exists(
        lineage_root,
        generation=generation,
        checkpoint_hashes=raw_hashes,
        generation_commits=raw_commits,
    ):
        raise RuntimeError(
            f"refusing to reconcile M{generation} while later generation evidence exists"
        )

    checkpoint_relative = f"checkpoints/M{generation}.pt"
    provenance_path = lineage_root / "metadata" / "provenance-v2" / f"M{generation}.json"
    node_path = lineage_root / "metadata" / "checkpoints" / f"M{generation}.json"
    _validate_graph_owner(
        provenance_path,
        lineage_id=lineage_id,
        generation=generation,
        checkpoint_path=checkpoint_relative,
        label="generation provenance",
    )
    _validate_graph_owner(
        node_path,
        lineage_id=lineage_id,
        generation=generation,
        checkpoint_path=checkpoint_relative,
        label="CheckpointNode",
    )

    commit_key = str(generation)
    stale_commit = raw_commits.get(commit_key)
    expected_marker_relative = marker_path.relative_to(lineage_root).as_posix()
    if stale_commit is not None:
        if not isinstance(stale_commit, Mapping):
            raise RuntimeError("generation commit manifest entry is malformed")
        if stale_commit.get("path") != expected_marker_relative:
            raise RuntimeError("generation commit manifest entry points outside the generation")

    permanent_paths = _generation_paths(lineage_root, generation)
    allowed_catalog_paths = {
        path.relative_to(lineage_root).as_posix() for path in permanent_paths
    }
    catalog_path = lineage_root / "runtime" / "artifact-catalog.json"
    catalog: ArtifactCatalog | None = None
    catalog_has_generation = False
    if catalog_path.is_file():
        catalog = ArtifactCatalog.load(catalog_path, root=lineage_root)
        if catalog.payload.get("lineage_id") != lineage_id:
            raise RuntimeError("artifact catalog owner does not match recovery request")
        generations = catalog.payload.get("generations")
        if not isinstance(generations, Mapping):
            raise RuntimeError("artifact catalog generations are malformed")
        record = generations.get(commit_key)
        if record is not None:
            if not isinstance(record, Mapping):
                raise RuntimeError("artifact catalog generation record is malformed")
            paths = record.get("artifact_paths")
            if not isinstance(paths, list):
                raise RuntimeError("artifact catalog generation paths are malformed")
            unexpected = {str(path) for path in paths} - allowed_catalog_paths
            if unexpected:
                raise RuntimeError(
                    "artifact catalog generation record contains unexpected paths: "
                    + ", ".join(sorted(unexpected))
                )
            catalog_has_generation = True
        elif set(catalog.entries).intersection(allowed_catalog_paths):
            raise RuntimeError(
                "artifact catalog contains unbound artifacts for the incomplete generation"
            )

    if marker_path.is_file():
        return False

    changed = False
    if catalog is not None and catalog_has_generation:
        catalog.discard_generation(generation)
        changed = True

    updated = dict(manifest)
    updated_hashes = dict(raw_hashes)
    if checkpoint_relative in updated_hashes:
        updated_hashes.pop(checkpoint_relative, None)
        updated["checkpoint_hashes"] = updated_hashes
        changed = True
    updated_commits = dict(raw_commits)
    if commit_key in updated_commits:
        updated_commits.pop(commit_key, None)
        updated["generation_commits"] = updated_commits
        changed = True
    if changed:
        atomic_write_text(manifest_path, canonical_json(updated) + "\n")

    for path in permanent_paths + _temporary_paths(lineage_root, generation):
        if path.exists() or path.is_symlink():
            path.unlink(missing_ok=True)
            changed = True
    return changed


__all__ = ["reconcile_uncommitted_generation"]
