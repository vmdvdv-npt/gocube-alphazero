"""Lifecycle operations for completed/recoverable training lineages.

These helpers implement the repository Run Storage and Archiving Policy:
archive moves a whole lineage without copying; discard requires an explicit
reason, conservatively checks references, writes the required short Markdown
record, and only then removes the heavy lineage directory.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from .orchestrator import atomic_write_json, atomic_write_text, read_json
from .run_spec import discover_active_lineage
from .run_storage import archived_lineage_dir


RUNNING_STATES = frozenset({"RUNNING", "SOFT_STOP_REQUESTED"})


def _git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _assert_not_running(lineage_root: Path) -> dict[str, object]:
    state_path = lineage_root / "runtime" / "state.json"
    if not state_path.is_file():
        raise ValueError("Lineage runtime state is missing")
    state = read_json(state_path)
    if str(state.get("state")) in RUNNING_STATES:
        raise RuntimeError("Cannot archive/discard a running lineage")
    lock = lineage_root / "runtime" / "orchestrator.lock"
    if lock.exists():
        raise RuntimeError("Cannot archive/discard while orchestrator lock exists")
    return state


def archive_lineage(*, repo_root: str | Path, lineage_id: str) -> Path:
    root = Path(repo_root).resolve()
    lineage_root = discover_active_lineage(root, lineage_id)
    _assert_not_running(lineage_root)
    manifest = read_json(lineage_root / "manifest.json")
    if manifest.get("status") != "ACTIVE":
        raise ValueError(f"Only ACTIVE lineage may be archived: {manifest.get('status')}")
    topology = str(manifest["topology"])
    target = archived_lineage_dir(topology, lineage_id)
    if target.exists():
        raise FileExistsError(f"Archive target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.rename(lineage_root, target)
    manifest["status"] = "ARCHIVED"
    manifest["archived_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(target / "manifest.json", manifest)
    return target


def _external_references(
    repo_root: Path,
    lineage_root: Path,
    lineage_id: str,
    checkpoint_hashes: Mapping[str, object],
) -> list[str]:
    needles = {str(lineage_id), *(str(value) for value in checkpoint_hashes.values())}
    findings: list[str] = []
    runs_root = repo_root / "runs"
    if not runs_root.exists():
        return findings
    for path in runs_root.rglob("*.json"):
        try:
            path.resolve().relative_to(lineage_root.resolve())
            continue
        except ValueError:
            pass
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(needle and needle in text for needle in needles):
            findings.append(str(path.relative_to(repo_root)))
    return sorted(set(findings))


def discard_lineage(
    *,
    repo_root: str | Path,
    lineage_id: str,
    reason: str,
    useful_result: str,
) -> Path:
    root = Path(repo_root).resolve()
    reason = str(reason).strip()
    useful_result = str(useful_result).strip()
    if not reason:
        raise ValueError("Discard requires an explicit non-empty reason")
    if not useful_result:
        raise ValueError("Discard requires an explicit useful-result statement (use 'none' if applicable)")
    lineage_root = discover_active_lineage(root, lineage_id)
    _assert_not_running(lineage_root)
    manifest = read_json(lineage_root / "manifest.json")
    checkpoint_hashes = manifest.get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, Mapping):
        raise ValueError("Lineage manifest checkpoint_hashes is malformed")
    references = _external_references(
        root, lineage_root, lineage_id, checkpoint_hashes
    )
    if references:
        raise RuntimeError(
            "Discard refused because retained artifacts reference this lineage/checkpoints: "
            + ", ".join(references[:20])
        )

    now = datetime.now(timezone.utc)
    record_dir = root / "docs" / "experiments" / "discarded"
    record = record_dir / f"{now.strftime('%Y%m%d')}-{lineage_id}.md"
    if record.exists():
        raise FileExistsError(f"Discard record already exists: {record}")
    manifest["status"] = "DISCARDED"
    manifest["discarded_at"] = now.isoformat()
    manifest["discard_reason"] = reason
    atomic_write_json(lineage_root / "manifest.json", manifest)
    text = "\n".join(
        [
            f"# Discarded training lineage `{lineage_id}`",
            "",
            f"Run: `{lineage_id}`",
            "Status: DISCARDED",
            f"Date: {now.date().isoformat()}",
            f"Reason: {reason}",
            f"Useful result: {useful_result}",
            f"PR / commit: `{_git_sha(root)}`",
            f"Topology: `{manifest.get('topology')}`",
            f"Config fingerprint: `{manifest.get('config_fingerprint')}`",
            "",
        ]
    )
    atomic_write_text(record, text)
    shutil.rmtree(lineage_root)
    return record


__all__ = ["archive_lineage", "discard_lineage"]
