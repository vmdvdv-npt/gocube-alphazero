"""Canonical paths and creation helpers for lineages and evaluations.

Runtime code should use these helpers instead of inventing a second artifact
root. Path lookup is side-effect free. Creating a lineage publishes its
manifest and owned subdirectories as one prepared directory, so a normal
creation path cannot leave a lineage without a manifest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"

ACTIVE = "active"
ARCHIVE = "archive"
EVALUATIONS = "evaluations"
ALLOWED_LINEAGE_STATES = frozenset({ACTIVE, ARCHIVE})
LINEAGE_STATUSES = frozenset({"ACTIVE", "ARCHIVED", "DISCARDED"})
REQUIRED_LINEAGE_MANIFEST_FIELDS = (
    "lineage_id",
    "topology",
    "status",
    "parent_checkpoint",
    "git_commit",
    "config_fingerprint",
    "created_at",
    "checkpoint_hashes",
)


class CheckpointResolutionError(ValueError):
    """Raised when a declared checkpoint reference cannot be trusted."""


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """A verified physical checkpoint selected from a logical reference."""

    topology: str
    lineage_id: str
    checkpoint_id: str
    generation: int | None
    path: Path
    sha256: str
    owner_status: str
    reference: dict[str, Any]

    def as_reference(self) -> dict[str, Any]:
        """Return a stable, machine-readable reference for reports/evaluations."""
        result = dict(self.reference)
        result.update(
            {
                "topology": self.topology,
                "lineage_id": self.lineage_id,
                "checkpoint_id": self.checkpoint_id,
                "generation": self.generation,
                "path": str(self.path),
                "sha256": self.sha256,
                "artifact_sha256": self.sha256,
                "owner_status": self.owner_status,
            }
        )
        return result


def _safe_component(value: str, *, label: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def topology_root(topology: str) -> Path:
    return RUNS_ROOT / _safe_component(topology, label="topology")


def lineage_dir(
    topology: str,
    lineage_id: str,
    *,
    state: str = ACTIVE,
) -> Path:
    if state not in ALLOWED_LINEAGE_STATES:
        raise ValueError(f"Invalid lineage state: {state!r}")
    return topology_root(topology) / state / _safe_component(lineage_id, label="lineage id")


def active_lineage_dir(topology: str, lineage_id: str) -> Path:
    return lineage_dir(topology, lineage_id, state=ACTIVE)


def archived_lineage_dir(topology: str, lineage_id: str) -> Path:
    return lineage_dir(topology, lineage_id, state=ARCHIVE)


def evaluation_dir(topology: str, evaluation_id: str) -> Path:
    return topology_root(topology) / EVALUATIONS / _safe_component(
        evaluation_id,
        label="evaluation id",
    )


def evaluations_root(topology: str) -> Path:
    """Return the only parent under which evaluation outputs may be stored."""
    return topology_root(topology) / EVALUATIONS


def evaluation_id_for_comparison(
    *,
    candidate_lineage_id: str,
    candidate_generation: int,
    reference_lineage_id: str,
    reference_generation: int | None,
) -> str:
    """Build a stable evaluation id without embedding filesystem separators."""
    candidate = _safe_component(candidate_lineage_id, label="candidate lineage id")
    reference = _safe_component(reference_lineage_id, label="reference lineage id")
    candidate_label = f"M{int(candidate_generation):04d}"
    reference_label = (
        f"M{int(reference_generation):04d}"
        if reference_generation is not None
        else "checkpoint"
    )
    return f"{candidate}-{candidate_label}-vs-{reference}-{reference_label}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _lineage_root_for_reference(
    topology: str,
    lineage_id: str,
    *,
    runs_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Find the unique active/archived owner and load its manifest."""
    matches: list[Path] = []
    for namespace in (ACTIVE, ARCHIVE):
        candidate = runs_root / _safe_component(topology, label="topology") / namespace / _safe_component(
            lineage_id,
            label="lineage id",
        )
        if candidate.is_dir():
            matches.append(candidate)
    if not matches:
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: lineage={lineage_id!r} "
            f"topology={topology!r}; owner lineage was not found"
        )
    if len(matches) > 1:
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: lineage={lineage_id!r}; "
            f"owner exists in multiple namespaces: {matches}"
        )
    root = matches[0]
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: lineage={lineage_id!r}; "
            f"cannot read owner manifest {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise CheckpointResolutionError(f"Owner manifest is not an object: {manifest_path}")
    if manifest.get("lineage_id") != lineage_id or manifest.get("topology") != topology:
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: owner manifest identity mismatch: {manifest_path}"
        )
    status = str(manifest.get("status", ""))
    if status not in {"ACTIVE", "ARCHIVED"}:
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: lineage={lineage_id!r}; "
            f"owner status={status!r} is not usable"
        )
    return root, manifest


def resolve_checkpoint(
    reference: Mapping[str, Any],
    *,
    topology: str | None = None,
    runs_root: str | Path | None = None,
) -> ResolvedCheckpoint:
    """Resolve and verify a canonical checkpoint reference.

    The declared owner lineage is authoritative for relative locators.  An
    absolute path is still required to be inside that owner, and the physical
    bytes must match the declared SHA-256 (or the owner manifest identity).
    """
    if not isinstance(reference, Mapping):
        raise CheckpointResolutionError("Checkpoint reference must be an object")
    raw_topology = topology or reference.get("topology")
    if not raw_topology:
        raise CheckpointResolutionError("Checkpoint reference is missing topology")
    resolved_topology = _safe_component(str(raw_topology), label="topology")
    raw_lineage = reference.get("lineage_id") or reference.get("parent_lineage")
    raw_path = reference.get("path") or reference.get("canonical_path")
    root_base = Path(runs_root).resolve() if runs_root is not None else RUNS_ROOT.resolve()

    inferred_root: Path | None = None
    if raw_path and Path(str(raw_path)).is_absolute():
        absolute_path = Path(str(raw_path)).resolve()
        topology_root_path = (root_base / resolved_topology).resolve()
        try:
            relative_to_topology = absolute_path.relative_to(topology_root_path)
        except ValueError as exc:
            raise CheckpointResolutionError(
                f"Checkpoint reference resolution failed: path={absolute_path}; "
                f"path is outside topology root {topology_root_path}"
            ) from exc
        parts = relative_to_topology.parts
        if len(parts) >= 3 and parts[0] in {ACTIVE, ARCHIVE}:
            inferred_root = topology_root_path / parts[0] / parts[1]
            if raw_lineage is None:
                raw_lineage = parts[1]

    if not raw_lineage:
        raise CheckpointResolutionError("Checkpoint reference is missing lineage_id")
    lineage_id = _safe_component(str(raw_lineage), label="lineage id")
    owner_root, manifest = _lineage_root_for_reference(
        resolved_topology,
        lineage_id,
        runs_root=root_base,
    )
    if inferred_root is not None and inferred_root != owner_root.resolve():
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: lineage={lineage_id!r}; "
            f"declared path owner {inferred_root} does not match manifest owner {owner_root}"
        )

    if raw_path:
        declared_path = Path(str(raw_path))
        candidate_path = (
            declared_path.resolve()
            if declared_path.is_absolute()
            else (owner_root / declared_path).resolve()
        )
    else:
        generation_value = reference.get("generation")
        if generation_value is None:
            raise CheckpointResolutionError(
                f"Checkpoint reference resolution failed: lineage={lineage_id!r}; "
                "path or generation is required"
            )
        candidate_path = (owner_root / "checkpoints" / f"M{int(generation_value)}.pt").resolve()
    try:
        candidate_path.relative_to(owner_root.resolve())
    except ValueError as exc:
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: lineage={lineage_id!r}; "
            f"resolved path escapes owner lineage: {candidate_path}"
        ) from exc

    relative = candidate_path.relative_to(owner_root.resolve()).as_posix()
    checkpoint_hashes = manifest.get("checkpoint_hashes")
    manifest_sha = None
    if isinstance(checkpoint_hashes, Mapping):
        value = checkpoint_hashes.get(relative)
        if value is not None:
            manifest_sha = str(value)
    declared_sha = reference.get("artifact_sha256") or reference.get("sha256")
    expected_sha = str(declared_sha or manifest_sha or "")
    if not expected_sha.startswith("sha256:"):
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: lineage={lineage_id!r} "
            f"checkpoint={reference.get('label') or reference.get('checkpoint_id') or relative}; "
            "expected SHA-256 is missing"
        )
    if manifest_sha is not None and manifest_sha != expected_sha:
        actual_sha = _sha256_file(candidate_path) if candidate_path.is_file() else None
        raise CheckpointResolutionError(
            f"Checkpoint reference resolution failed: lineage={lineage_id!r} checkpoint={relative}; "
            f"declared SHA={expected_sha}, owner manifest SHA={manifest_sha}, "
            f"actual sha256={actual_sha or '<missing>'}"
        )
    exists = candidate_path.is_file()
    actual_sha = _sha256_file(candidate_path) if exists else None
    if actual_sha != expected_sha:
        raise CheckpointResolutionError(
            "Checkpoint reference resolution failed\n"
            f"lineage: {lineage_id}\n"
            f"checkpoint: {reference.get('label') or reference.get('checkpoint_id') or relative}\n"
            f"expected path/reference: {raw_path or relative}\n"
            f"resolved path: {candidate_path}\n"
            f"exists: {str(exists).lower()}\n"
            f"expected sha256: {expected_sha}\n"
            f"actual sha256: {actual_sha or '<missing>'}"
        )

    generation: int | None
    raw_generation = reference.get("generation")
    if raw_generation is None:
        checkpoint_id = str(reference.get("checkpoint_id") or candidate_path.stem)
        generation = (
            int(checkpoint_id[1:])
            if checkpoint_id.startswith("M") and checkpoint_id[1:].isdigit()
            else None
        )
    else:
        generation = int(raw_generation)
        checkpoint_id = str(reference.get("checkpoint_id") or reference.get("label") or f"M{generation}")

    canonical = dict(reference)
    canonical.update(
        {
            "topology": resolved_topology,
            "lineage_id": lineage_id,
            "checkpoint_id": checkpoint_id,
            "generation": generation,
            "path": str(candidate_path),
            "sha256": expected_sha,
            "artifact_sha256": expected_sha,
        }
    )
    return ResolvedCheckpoint(
        topology=resolved_topology,
        lineage_id=lineage_id,
        checkpoint_id=checkpoint_id,
        generation=generation,
        path=candidate_path,
        sha256=expected_sha,
        owner_status=str(manifest["status"]),
        reference=canonical,
    )


def topology_for_profile(profile_id: str) -> str:
    """Map the current profile identifiers to the storage topology name."""
    normalized = str(profile_id).lower()
    if "torus9" in normalized:
        return "torus9"
    if "torus5" in normalized:
        return "torus5"
    if "cube" in normalized:
        return "cube4"
    raise ValueError(f"Cannot derive storage topology from profile: {profile_id!r}")


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    topology: str | None = None,
    lineage_id: str | None = None,
    active_only: bool = False,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("lineage manifest must be a mapping")
    missing = [field for field in REQUIRED_LINEAGE_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"lineage manifest is missing required fields: {', '.join(missing)}")
    result = dict(manifest)
    if result["status"] not in LINEAGE_STATUSES:
        raise ValueError(f"invalid lineage manifest status: {result['status']!r}")
    if active_only and result["status"] != "ACTIVE":
        raise ValueError("a newly-created lineage must have status 'ACTIVE'")
    if topology is not None and result["topology"] != topology:
        raise ValueError("lineage manifest topology does not match its storage path")
    if lineage_id is not None and result["lineage_id"] != lineage_id:
        raise ValueError("lineage manifest id does not match its storage path")
    if not isinstance(result["checkpoint_hashes"], Mapping):
        raise TypeError("lineage manifest checkpoint_hashes must be a mapping")
    return result


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError:
        # Directory fsync is a durability improvement on POSIX, not a reason
        # to make the storage API unusable on platforms without it.
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def ensure_lineage_layout(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    extra_directories: Iterable[str] = (),
) -> Path:
    """Ensure a lineage layout, requiring a valid manifest at creation time.

    A new lineage is prepared in a hidden sibling directory and published only
    after ``manifest.json`` and all owned directories exist. An existing
    directory must already have a valid manifest; this prevents callers from
    using this helper to normalize a manifest-less lineage in place.
    """
    path = Path(path)
    validated = _validate_manifest(manifest)
    directories = ["checkpoints", "data", "logs", "arena", "metrics"]
    for name in extra_directories:
        safe_name = _safe_component(name, label="lineage directory")
        if safe_name == "manifest.json":
            raise ValueError("manifest.json is reserved and cannot be an extra directory")
        if safe_name not in directories:
            directories.append(safe_name)

    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"lineage path is not a directory: {path}")
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(
                f"refusing to use an existing lineage without manifest.json: {path}"
            )
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid lineage manifest: {manifest_path}") from exc
        _validate_manifest(existing)
        for name in directories:
            (path / name).mkdir(exist_ok=True)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.creating-", dir=path.parent))
    try:
        for name in directories:
            (temporary / name).mkdir()
        _write_manifest(temporary / "manifest.json", validated)
        os.rename(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return path


def create_lineage(
    topology: str,
    lineage_id: str,
    *,
    manifest: Mapping[str, Any],
    extra_directories: Iterable[str] = (),
) -> Path:
    """Create a new active lineage in the canonical location."""
    validated = _validate_manifest(
        manifest,
        topology=topology,
        lineage_id=lineage_id,
        active_only=True,
    )
    return ensure_lineage_layout(
        active_lineage_dir(topology, lineage_id),
        manifest=validated,
        extra_directories=extra_directories,
    )


def ensure_evaluation_layout(path: Path) -> Path:
    """Create the standard subdirectories for a newly-created evaluation."""
    path.mkdir(parents=True, exist_ok=True)
    for name in ("results", "logs", "metrics"):
        (path / name).mkdir(exist_ok=True)
    return path
