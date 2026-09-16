"""Canonical paths and creation helpers for lineages and evaluations.

Runtime code should use these helpers instead of inventing a second artifact
root. Path lookup is side-effect free. Creating a lineage publishes its
manifest and owned subdirectories as one prepared directory, so a normal
creation path cannot leave a lineage without a manifest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
