#!/usr/bin/env python3
"""Replace byte-identical model checkpoint copies with verified references.

The operation is intentionally narrow: it considers only files below a
``checkpoint``/``checkpoints`` directory, verifies SHA-256 equality, keeps a
canonical physical file, and replaces duplicate files with relative symlinks.
The symlink is accompanied by a manifest reference containing the parent
lineage, canonical path, and hash.  No checkpoint bytes are copied.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-16"
REPORT = ROOT / "docs" / "experiments" / "checkpoint-deduplication-20260916.json"
MODEL_SUFFIXES = {".pt", ".pkl", ".ckpt", ".bin"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def checkpoint_files() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "runs").rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in MODEL_SUFFIXES
        and any(part in {"checkpoint", "checkpoints"} for part in path.relative_to(ROOT).parts)
    )


def choose_canonical(paths: list[Path]) -> Path:
    # Prefer an active copy, then the known Cube-4 parent lineage for the
    # legacy fork bundle, then stable lexical order.
    return min(
        paths,
        key=lambda path: (
            0 if "/active/" in f"/{path.relative_to(ROOT).as_posix()}" else 1,
            0 if "c4-t001-c4-c001" in path.name or "c4-t001-c4-c001" in path.as_posix() else 1,
            path.as_posix(),
        ),
    )


def load_manifest(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    manifest_path = path / "manifest.json"
    fd, temporary = tempfile.mkstemp(prefix=f".{manifest_path.name}.dedup-", dir=str(path))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)


def refresh_parent_checkpoint_references() -> int:
    refreshed = 0
    for manifest_path in sorted((ROOT / "runs").glob("*/archive/*/manifest.json")):
        lineage = manifest_path.parent
        provenance_path = lineage / "checkpoints" / "fork-provenance.json"
        if not provenance_path.is_file():
            continue
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(provenance, dict):
            continue
        parent_run = provenance.get("parent_run")
        parent_iteration = provenance.get("parent_iteration")
        parent_sha = provenance.get("parent_checkpoint_sha256")
        if not isinstance(parent_run, str) or not isinstance(parent_iteration, int) or not isinstance(parent_sha, str):
            continue
        parent_checkpoint = f"iteration-{parent_iteration:04d}.pkl"
        parent_lineage = ROOT / "runs" / "cube4" / "archive" / parent_run
        canonical = parent_lineage / "checkpoints" / parent_checkpoint
        if not canonical.is_file() or sha256(canonical) != f"sha256:{parent_sha}":
            raise ValueError(f"Parent checkpoint reference failed verification: {canonical}")
        manifest = load_manifest(lineage)
        manifest["parent_checkpoint"] = {
            "parent_lineage": parent_run,
            "checkpoint": parent_checkpoint,
            "path": str(canonical.relative_to(ROOT)),
            "sha256": f"sha256:{parent_sha}",
        }
        write_manifest(lineage, manifest)
        refreshed += 1
    return refreshed


def deduplicate(*, execute: bool) -> dict[str, object]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in checkpoint_files():
        groups[sha256(path)].append(path)
    duplicate_groups = {digest: paths for digest, paths in groups.items() if len(paths) > 1}
    entries: list[dict[str, object]] = []
    for digest, paths in sorted(duplicate_groups.items()):
        canonical = choose_canonical(paths)
        for duplicate in sorted(paths):
            if duplicate == canonical:
                continue
            entries.append(
                {
                    "canonical_path": str(canonical.relative_to(ROOT)),
                    "duplicate_path": str(duplicate.relative_to(ROOT)),
                    "sha256": digest,
                }
            )

    payload: dict[str, object] = {
        "migration": DATE,
        "mode": "execute" if execute else "dry-run",
        "move_only_at_storage_level": True,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": len(entries),
        "references": entries,
    }
    if not execute:
        return payload

    for entry in entries:
        canonical = ROOT / str(entry["canonical_path"])
        duplicate = ROOT / str(entry["duplicate_path"])
        if not canonical.is_file() or duplicate.is_symlink() or not duplicate.is_file():
            raise FileNotFoundError(f"Checkpoint dedup precondition failed: {duplicate}")
        if sha256(canonical) != str(entry["sha256"]) or sha256(duplicate) != str(entry["sha256"]):
            raise ValueError(f"Checkpoint changed during deduplication: {duplicate}")
        relative_target = os.path.relpath(canonical, duplicate.parent)
        temporary = duplicate.with_name(f".{duplicate.name}.dedup-pending")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(temporary)
        duplicate.rename(temporary)
        try:
            duplicate.symlink_to(relative_target)
        except BaseException:
            temporary.rename(duplicate)
            raise
        temporary.unlink()

    affected = {Path(str(entry["duplicate_path"])).parents[1] for entry in entries}
    for lineage in sorted(ROOT / p for p in affected):
        manifest = load_manifest(lineage)
        references = manifest.setdefault("checkpoint_references", [])
        if not isinstance(references, list):
            references = []
            manifest["checkpoint_references"] = references
        for entry in entries:
            duplicate = Path(str(entry["duplicate_path"]))
            if duplicate.parents[1] != lineage.relative_to(ROOT):
                continue
            canonical = Path(str(entry["canonical_path"]))
            references.append(
                {
                    "parent_lineage": canonical.parents[1].name,
                    "checkpoint": canonical.name,
                    "canonical_path": str(canonical),
                    "sha256": entry["sha256"],
                    "replaced_path": str(duplicate),
                }
            )
        manifest["checkpoint_references"] = sorted(
            {json.dumps(item, ensure_ascii=False, sort_keys=True): item for item in references}.values(),
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
        write_manifest(lineage, manifest)

    payload["parent_references_refreshed"] = refresh_parent_checkpoint_references()

    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(deduplicate(execute=args.execute), indent=2, ensure_ascii=False, sort_keys=True))
