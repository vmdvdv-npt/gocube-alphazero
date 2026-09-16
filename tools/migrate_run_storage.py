#!/usr/bin/env python3
"""Safely migrate the existing local run artifacts to the run-storage policy.

The migration is deliberately explicit and move-only.  It refuses destination
collisions, preserves all non-empty source data, records pre/post file counts
and sizes, and hashes checkpoint files.  Use ``--dry-run`` first and then
``--execute`` from the repository root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.run_storage import (
    active_lineage_dir,
    archived_lineage_dir,
    evaluation_dir,
)


MIGRATION_DATE = "2026-09-16"
REPORT_PATH = ROOT / "docs" / "experiments" / "run-storage-migration-20260916.json"


@dataclass(frozen=True)
class Move:
    source: Path
    destination: Path
    kind: str


def _p(value: str) -> Path:
    return ROOT / value


def _archive(topology: str, lineage_id: str) -> Path:
    return archived_lineage_dir(topology, lineage_id)


def _active(topology: str, lineage_id: str) -> Path:
    return active_lineage_dir(topology, lineage_id)


def build_moves() -> tuple[list[Move], list[Move]]:
    """Return whole-directory moves and cube4 artifact merges."""
    whole: list[Move] = [
        Move(
            _p("runs/torus9-golden-v3-active/torus9-golden-v3-20260914-run03"),
            _active("torus9", "torus9-golden-v3-20260914-run03"),
            "lineage",
        ),
        Move(
            _p("runs/torus9-stage7/torus9-golden-stage7-20260915-run01"),
            _archive("torus9", "torus9-golden-stage7-20260915-run01"),
            "lineage",
        ),
        Move(
            _p("artifacts/repro/pr112/exact3/phase-a/torus9-nightly-20260916-phase-a-final"),
            _archive("torus9", "torus9-pr112-repro-20260916-exact3"),
            "lineage",
        ),
        Move(
            _p("runs/torus5-aux-heads/torus5-aux-heads-night-ab-20260913"),
            _archive("torus5", "torus5-aux-heads-night-ab-20260913"),
            "lineage",
        ),
        Move(
            _p("runs/archive/torus9-invalid-20260914"),
            _archive("torus9", "torus9-invalid-20260914"),
            "lineage-bundle",
        ),
        Move(
            _p("runs/archive/torus9-legacy-20260913"),
            _archive("torus9", "torus9-legacy-20260913"),
            "lineage-bundle",
        ),
        Move(
            _p("runs/golden-arena-process-parallel"),
            evaluation_dir("torus9", "golden-arena-process-parallel"),
            "evaluation",
        ),
        Move(
            _p("runs/torus9-adaptive-arena-benchmark-20260914-v1"),
            evaluation_dir("torus9", "torus9-adaptive-arena-benchmark-20260914-v1"),
            "evaluation",
        ),
        Move(
            _p("runs/torus9-fast-arena-20260914"),
            evaluation_dir("torus9", "torus9-fast-arena-20260914"),
            "evaluation",
        ),
        Move(
            _p("runs/torus9-speed-benchmark-20260914"),
            evaluation_dir("torus9", "torus9-speed-benchmark-20260914"),
            "evaluation",
        ),
    ]

    for child in sorted((_p("runs/torus-golden-stage3")).iterdir()):
        if child.is_dir():
            kind = "evaluation" if child.name == "seed1-diagnosis-20260912" else "lineage"
            destination = (
                evaluation_dir("torus9", child.name)
                if kind == "evaluation"
                else _archive("torus9", child.name)
            )
            whole.append(Move(child, destination, kind))
    for child in sorted((_p("runs/torus-golden-stage4")).iterdir()):
        if child.is_dir():
            whole.append(Move(child, _archive("torus9", child.name), "lineage"))
    for child in sorted((_p("runs/cube4-golden-transfer")).iterdir()):
        if child.is_dir():
            whole.append(Move(child, _archive("cube4", child.name), "lineage"))

    nightly_phase_a = _p("runs/torus9-nightly-20260916/phase-a")
    for child in sorted(nightly_phase_a.iterdir()):
        if child.is_dir():
            whole.append(Move(child, _archive("torus9", child.name), "lineage"))

    nightly_cadence = _p("runs/torus9-nightly-20260916/cadence/torus9-nightly-20260916")
    for child in sorted(nightly_cadence.glob("arm-*")):
        if child.is_dir():
            whole.append(Move(child, _archive("torus9", child.name), "lineage"))
    whole.append(
        Move(
            nightly_cadence,
            evaluation_dir("torus9", "torus9-nightly-20260916-cadence"),
            "evaluation",
        )
    )

    for child in sorted((_p("arena-results")).iterdir()):
        if child.is_dir():
            whole.append(Move(child, evaluation_dir("torus9", child.name), "evaluation"))

    for name in (
        "ci-torus9-v3-budget-smoke",
        "codex-ci-torus9-v3-budget-smoke",
        "codex-ci-torus9-v3-budget-smoke-v2",
    ):
        whole.append(Move(_p(f"runs/{name}"), _archive("torus9", name), "lineage"))

    for name in (
        "c4-pr30-repro-k05-20260909-workers16-bootstrap-failed",
        "c4-pr30-repro-k05-20260909-workers2-incomplete",
    ):
        whole.append(Move(_p(f"experiment-backups/{name}"), _archive("cube4", name), "lineage-bundle"))
    whole.append(
        Move(
            _p("архив обучения/komi-7.5-cube4-20260906"),
            _archive("cube4", "komi-7.5-cube4-20260906"),
            "lineage-bundle",
        )
    )

    # Reports without a matching checkpoint/run directory are still retained
    # as archived lineage-owned evidence.
    for name in (
        "gocube-b-night-c4-k05-s0-20260909",
        "gocube-b05-legion-preflight-dryrun-corrected-r2",
        "gocube-b4-20260909",
    ):
        whole.append(Move(_p(f"training_reports/{name}"), _archive("cube4", name), "lineage"))

    return whole, build_cube4_merges()


def build_cube4_merges() -> list[Move]:
    names = {
        "c4-pr30-repro-k05-20260909",
        "ci-cube4-katago-hardened-smoke-f0",
        "ci-cube4-v3-budget-smoke",
        "ci-cube4-v3-budget-smoke-fix",
        "ci-cube4-v3-budget-smoke-fix2",
        "codex-ci-cube4-v3-budget-smoke",
        "codex-ci-cube4-v3-budget-smoke-v2",
        "codex-learning-sanity-pinned-cube4-20260909",
        "gocube-b0-c4-k05-s0-20260909",
        "gocube-b05-b0",
        "gocube-b05-b1",
        "gocube-b05-throughput-w16-r1",
        "gocube-b05-throughput-w16-r2",
        "gocube-b05-throughput-w16-r3",
        "gocube-b05-throughput-w8-r1",
        "gocube-b05-throughput-w8-r2",
        "gocube-b05-throughput-w8-r3",
        "gocube-b1-c4-k05-s0-20260909",
        "learning-sanity-20260909-131327",
    }
    sources = (
        ("runs", "logs"),
        ("checkpoint", "checkpoints"),
        ("data", "data"),
        ("training_reports", "reports"),
    )
    merges: list[Move] = []
    for name in sorted(names):
        destination = _archive("cube4", name)
        for source_root, target_subdir in sources:
            source = _p(f"{source_root}/{name}")
            if source.exists():
                merges.append(Move(source, destination / target_subdir, f"lineage-{target_subdir}"))
    return merges


def file_summary(path: Path) -> dict[str, Any]:
    files = [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()] if path.exists() else []
    checkpoint_hashes: dict[str, str] = {}
    for file in files:
        relative_parts = file.relative_to(path).parts
        # Legacy Cube runs also store replay tensors as .pkl files below
        # ``data/``.  They are not model checkpoints and must not pollute the
        # checkpoint identity index or trigger false duplicate-model reports.
        is_checkpoint_path = any(part in {"checkpoint", "checkpoints"} for part in relative_parts)
        if is_checkpoint_path and file.suffix.lower() in {".pt", ".pkl", ".ckpt", ".bin"}:
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checkpoint_hashes[str(file.relative_to(path))] = f"sha256:{digest.hexdigest()}"
    return {
        "exists": path.exists(),
        "file_count": len(files),
        "bytes": sum(file.stat().st_size for file in files),
        "checkpoint_hashes": checkpoint_hashes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _existing_manifest(path: Path) -> tuple[dict[str, Any], str | None]:
    manifest = path / "manifest.json"
    if manifest.exists():
        return _load_json(manifest), "manifest.json"
    legacy = path / "run-manifest.json"
    if legacy.exists():
        return _load_json(legacy), "run-manifest.json"
    return {}, None


def _checkpoint_parent(manifest: dict[str, Any]) -> Any:
    existing = manifest.get("parent_checkpoint")
    if existing is not None:
        return existing
    resume = manifest.get("resume_contract")
    if isinstance(resume, dict) and resume.get("from_checkpoint"):
        return {
            "legacy_reference": resume.get("from_checkpoint"),
            "parent_lineage": None,
            "checkpoint": Path(str(resume["from_checkpoint"])).name,
        }
    return None


def _created_at(path: Path, manifest: dict[str, Any]) -> str:
    value = _first(manifest, "created_at", "started_at", "started")
    if value is not None:
        return str(value)
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def write_lineage_manifest(
    path: Path,
    *,
    topology: str,
    status: str,
    source_paths: list[str],
) -> None:
    manifest, legacy_name = _existing_manifest(path)
    old_status = manifest.get("status")
    if old_status not in (None, status):
        manifest["legacy_status"] = old_status
    manifest.update(
        {
            "lineage_id": path.name,
            "topology": topology,
            "status": status,
            "parent_checkpoint": _checkpoint_parent(manifest),
            "git_commit": _first(manifest, "git_commit", "source_commit", "commit", "base_sha", "base_commit") or "unknown",
            "config_fingerprint": _first(
                manifest,
                "config_fingerprint",
                "profile_fingerprint",
                "experiment_fingerprint",
            ) or "unknown",
            "created_at": _created_at(path, manifest),
            "checkpoint_hashes": file_summary(path).get("checkpoint_hashes", {}),
            "storage": {
                "canonical_path": str(path.relative_to(ROOT)),
                "migrated_on": MIGRATION_DATE,
                "migrated_from": source_paths,
                "legacy_manifest": legacy_name,
            },
        }
    )
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extract_checkpoint_references(value: Any, output: list[dict[str, Any]], role: str | None = None) -> None:
    if isinstance(value, dict):
        local_role = role
        if "candidate" in value:
            local_role = "candidate"
        if "reference" in value:
            local_role = "reference"
        path = value.get("path")
        if isinstance(path, str) and (".pt" in path or ".pkl" in path):
            output.append(
                {
                    "role": local_role,
                    "path": path,
                    "artifact_sha256": value.get("artifact_sha256") or value.get("checkpoint_sha256"),
                    "model_hash": value.get("model_hash"),
                }
            )
        for key, child in value.items():
            _extract_checkpoint_references(child, output, key if key in {"candidate", "reference"} else local_role)
    elif isinstance(value, list):
        for child in value:
            _extract_checkpoint_references(child, output, role)


def _checkpoint_index() -> dict[str, Path]:
    """Index retained physical checkpoints by the hashes in lineage manifests."""
    index: dict[str, Path] = {}
    for manifest_path in sorted((ROOT / "runs").glob("*/active/*/manifest.json")) + sorted(
        (ROOT / "runs").glob("*/archive/*/manifest.json")
    ):
        manifest = _load_json(manifest_path)
        hashes = manifest.get("checkpoint_hashes", {})
        if not isinstance(hashes, dict):
            continue
        for relative, digest in hashes.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                continue
            candidate = manifest_path.parent / relative
            if candidate.is_file() and not candidate.is_symlink():
                index.setdefault(digest, candidate.resolve())
    return index


def _normalize_checkpoint_reference_path(raw: str, *, topology: str) -> str:
    value = raw
    root_prefix = str(ROOT) + "/"
    if value.startswith(root_prefix):
        value = value[len(root_prefix) :]
    legacy_roots = (
        "checkpoint/",
        "runs/cube4/archive/komi-7.5-cube4-20260906/checkpoint/",
    )
    for prefix in legacy_roots:
        if value.startswith(prefix):
            remainder = value[len(prefix) :]
            name, separator, suffix = remainder.partition("/")
            if name:
                value = f"runs/cube4/archive/{name}/checkpoints/{suffix}" if separator else f"runs/cube4/archive/{name}/checkpoints"
            break
    if value.startswith("runs/torus9/archive/torus9-legacy-20260913/"):
        value = value.replace(
            "runs/torus9/archive/torus9-legacy-20260913/",
            "runs/torus9/archive/",
            1,
        )
    candidate = ROOT / value if not value.startswith("/") else Path(value)
    return str(candidate.resolve()) if candidate.exists() else raw


def write_evaluation_manifest(path: Path, *, topology: str, source_paths: list[str]) -> None:
    manifest, legacy_name = _existing_manifest(path)
    references: list[dict[str, Any]] = []
    _extract_checkpoint_references(manifest, references)
    for json_path in sorted(path.rglob("*.json")):
        if json_path.name == "manifest.json":
            continue
        _extract_checkpoint_references(_load_json(json_path), references)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    checkpoint_index = _checkpoint_index()
    for ref in references:
        ref = dict(ref)
        digest = ref.get("artifact_sha256") or ref.get("checkpoint_sha256")
        canonical = checkpoint_index.get(digest) if isinstance(digest, str) else None
        if canonical is not None:
            ref["path"] = str(canonical)
        elif isinstance(ref.get("path"), str):
            ref["path"] = _normalize_checkpoint_reference_path(ref["path"], topology=topology)
        reference_path = ref.get("path")
        if not ref.get("artifact_sha256") and isinstance(reference_path, str):
            target = Path(reference_path)
            if target.is_file():
                ref["artifact_sha256"] = _sha256_file(target)
        key = (ref.get("role"), ref.get("path"), ref.get("artifact_sha256"), ref.get("model_hash"))
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    manifest.update(
        {
            "evaluation_id": path.name,
            "topology": topology,
            "status": "ARCHIVED",
            "checkpoint_references": unique,
            "git_commit": _first(manifest, "git_commit", "source_commit", "commit") or "unknown",
            "config_fingerprint": _first(manifest, "config_fingerprint", "profile_fingerprint") or "unknown",
            "created_at": _created_at(path, manifest),
            "storage": {
                "canonical_path": str(path.relative_to(ROOT)),
                "migrated_on": MIGRATION_DATE,
                "migrated_from": source_paths,
                "legacy_manifest": legacy_name,
                "contains_checkpoint_copies": False,
            },
        }
    )
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _check_sources(moves: Iterable[Move]) -> list[Move]:
    moves = list(moves)
    missing = [str(move.source.relative_to(ROOT)) for move in moves if not move.source.exists()]
    collisions = [
        str(move.destination.relative_to(ROOT))
        for move in moves
        if move.destination.exists() and move.source != move.destination
    ]
    if missing:
        raise FileNotFoundError(f"Migration source missing: {missing}")
    if collisions:
        raise FileExistsError(f"Migration destination already exists: {collisions}")
    return moves


def _perform_move(move: Move) -> None:
    move.destination.parent.mkdir(parents=True, exist_ok=True)
    if move.destination.exists():
        raise FileExistsError(move.destination)
    shutil.move(str(move.source), str(move.destination))


def _perform_merge(move: Move) -> None:
    move.destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(move.source.iterdir()):
        destination = move.destination / child.name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(child), str(destination))
    move.source.rmdir()


def _prune_empty(paths: Iterable[Path]) -> None:
    roots = {ROOT / name for name in ("runs", "checkpoint", "arena-results", "training_reports", "experiment-backups", "artifacts")}
    roots.add(ROOT / "архив обучения")
    for path in sorted(set(paths), key=lambda item: len(item.parts), reverse=True):
        current = path
        while current in roots or any(current.is_relative_to(root) for root in roots):
            if current in {ROOT / "runs", ROOT / "data"}:
                break
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def migrate(*, execute: bool) -> dict[str, Any]:
    whole, merges = build_moves()
    all_moves = _check_sources([*whole, *merges])
    if not execute:
        return {
            "migration": MIGRATION_DATE,
            "mode": "dry-run",
            "whole_moves": [
                {"source": str(item.source.relative_to(ROOT)), "destination": str(item.destination.relative_to(ROOT)), "kind": item.kind}
                for item in whole
            ],
            "merges": [
                {"source": str(item.source.relative_to(ROOT)), "destination": str(item.destination.relative_to(ROOT)), "kind": item.kind}
                for item in merges
            ],
        }

    before = {
        str(item.source.relative_to(ROOT)): file_summary(item.source)
        for item in all_moves
    }
    source_paths_by_destination: dict[Path, list[str]] = {}
    for item in all_moves:
        source_paths_by_destination.setdefault(item.destination if item.kind in {"lineage", "lineage-bundle", "evaluation"} else item.destination.parent, []).append(
            str(item.source.relative_to(ROOT))
        )

    for item in whole:
        _perform_move(item)
    for item in merges:
        _perform_merge(item)

    lineage_status: dict[Path, str] = {}
    for item in whole:
        if item.kind in {"lineage", "lineage-bundle"}:
            lineage_status[item.destination] = "ACTIVE" if "/active/" in str(item.destination) else "ARCHIVED"
    for item in merges:
        lineage_status.setdefault(item.destination.parent, "ARCHIVED")

    for path, status in sorted(lineage_status.items()):
        write_lineage_manifest(
            path,
            topology=path.relative_to(ROOT / "runs").parts[0],
            status=status,
            source_paths=source_paths_by_destination.get(path, []),
        )

    evaluations = {
        item.destination
        for item in whole
        if item.kind == "evaluation"
    }
    for path in sorted(evaluations):
        write_evaluation_manifest(
            path,
            topology=path.relative_to(ROOT / "runs").parts[0],
            source_paths=source_paths_by_destination.get(path, []),
        )

    after = {
        str(item.destination.relative_to(ROOT)): file_summary(item.destination)
        for item in whole
    }
    after.update(
        {
            str(item.destination.parent.relative_to(ROOT)): file_summary(item.destination.parent)
            for item in merges
        }
    )
    _prune_empty([item.source for item in all_moves])
    report = {
        "migration": MIGRATION_DATE,
        "mode": "execute",
        "move_only": True,
        "source_data_deleted": False,
        "whole_move_count": len(whole),
        "merge_count": len(merges),
        "before": before,
        "after": after,
        "lineages": sorted(str(path.relative_to(ROOT)) for path in lineage_status),
        "evaluations": sorted(str(path.relative_to(ROOT)) for path in evaluations),
        "global_exceptions_kept": ["data/.gocube-game-ids", "docs/"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def refresh_manifests() -> dict[str, int]:
    """Repair/refresh policy manifests after a path-only migration step."""
    lineages = 0
    evaluations = 0
    for topology_root in sorted((ROOT / "runs").iterdir()):
        if not topology_root.is_dir() or topology_root.name.startswith("."):
            continue
        topology = topology_root.name
        for state in ("active", "archive"):
            state_root = topology_root / state
            if not state_root.is_dir():
                continue
            for path in sorted(item for item in state_root.iterdir() if item.is_dir()):
                existing, _ = _existing_manifest(path)
                storage = existing.get("storage")
                source_paths = storage.get("migrated_from", []) if isinstance(storage, dict) else []
                write_lineage_manifest(
                    path,
                    topology=topology,
                    status="ACTIVE" if state == "active" else "ARCHIVED",
                    source_paths=[str(item) for item in source_paths],
                )
                lineages += 1
        evaluation_root = topology_root / "evaluations"
        if evaluation_root.is_dir():
            for path in sorted(item for item in evaluation_root.iterdir() if item.is_dir()):
                existing, _ = _existing_manifest(path)
                storage = existing.get("storage")
                source_paths = storage.get("migrated_from", []) if isinstance(storage, dict) else []
                write_evaluation_manifest(
                    path,
                    topology=topology,
                    source_paths=[str(item) for item in source_paths],
                )
                evaluations += 1
    return {"lineages": lineages, "evaluations": evaluations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Perform the move-only migration")
    parser.add_argument("--refresh-manifests", action="store_true", help="Refresh policy manifests after migration")
    args = parser.parse_args()
    if args.refresh_manifests:
        print(json.dumps(refresh_manifests(), indent=2, sort_keys=True))
        return 0
    report = migrate(execute=args.execute)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
