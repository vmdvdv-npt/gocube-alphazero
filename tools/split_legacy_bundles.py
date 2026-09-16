#!/usr/bin/env python3
"""Split multi-lineage legacy bundles without deleting their contents.

This is a one-time, move-only follow-up to ``migrate_run_storage.py``.  It
refuses missing sources and destination collisions, then records where the
bundle-level metadata was moved.  Run from the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.run_storage import archived_lineage_dir, evaluation_dir


MIGRATION_DATE = "2026-09-16"
REPORT_PATH = ROOT / "docs" / "experiments" / "run-storage-split-legacy-bundles-20260916.json"


@dataclass(frozen=True)
class Move:
    source: Path
    destination: Path
    kind: str


def _p(value: str) -> Path:
    return ROOT / value


def _archive(topology: str, lineage_id: str) -> Path:
    return archived_lineage_dir(topology, lineage_id)


def build_moves() -> list[Move]:
    moves: list[Move] = []

    # The bundle roots contain their own aggregate metadata.  Move those small
    # records to tracked experiment evidence after splitting the heavy data.
    for bundle in ("torus9-legacy-20260913", "torus9-invalid-20260914"):
        source_root = _p(f"runs/torus9/archive/{bundle}")
        evidence_root = ROOT / "docs" / "experiments" / "legacy-bundles" / bundle
        for name in ("manifest.json", "archive-manifest.json"):
            source = source_root / name
            if source.exists():
                moves.append(Move(source, evidence_root / name, "bundle-metadata"))

    invalid_root = _p("runs/torus9/archive/torus9-invalid-20260914")
    for child in sorted(invalid_root.iterdir()):
        if child.is_dir():
            moves.append(Move(child, _archive("torus9", child.name), "lineage"))

    legacy_root = _p("runs/torus9/archive/torus9-legacy-20260913")
    moves.append(
        Move(
            legacy_root / "torus9-alpha-score-ab",
            _archive("torus9", "torus9-alpha-score-ab-20260913"),
            "lineage",
        )
    )
    for version in ("v1", "v2", "v3"):
        moves.append(
            Move(
                legacy_root / "torus9-golden-learning-proof" / f"torus9-golden-learning-proof-20260913-{version}",
                _archive("torus9", f"torus9-golden-learning-proof-20260913-{version}"),
                "lineage",
            )
        )
    moves.append(
        Move(
            legacy_root / "torus9-golden-learning-proof" / "torus9-preflight-20260913",
            _archive("torus9", "torus9-preflight-20260913"),
            "lineage",
        )
    )
    moves.append(
        Move(
            legacy_root / "torus9-komi-calibration-20260913",
            evaluation_dir("torus9", "torus9-komi-calibration-20260913"),
            "evaluation",
        )
    )
    moves.append(
        Move(
            legacy_root / "torus9-ownership-ab" / "torus9-wdl-ownership-ab-20260913-v1",
            _archive("torus9", "torus9-wdl-ownership-ab-20260913-v1"),
            "lineage",
        )
    )
    moves.append(
        Move(
            legacy_root / "torus9-stable-learning-v2" / "torus9-stable-learning-20260913-v1",
            _archive("torus9", "torus9-stable-learning-20260913-v1"),
            "lineage",
        )
    )

    cube_bundle = _p("runs/cube4/archive/komi-7.5-cube4-20260906")
    arm_names = (
        "c4-hparam-night-20260906-005747-a",
        "c4-hparam-night-20260906-005747-b",
        "c4-hparam-night-20260906-005747-c",
        "c4-hparam-night-20260906-005747-d",
        "c4-hparam-night-20260906-005747-e",
        "c4-hparam-night-20260906-005747-f",
        "c4-hparam-night-20260906-005747-g",
        "c4-t001-c4-c001",
    )
    for name in arm_names:
        destination = _archive("cube4", name)
        for source_root, target_name in (
            ("checkpoint", "checkpoints"),
            ("data", "data"),
            ("runs", "logs"),
        ):
            source = cube_bundle / source_root / name
            if source.exists():
                moves.append(Move(source, destination / target_name, f"lineage-{target_name}"))
    report = cube_bundle / "training_reports" / "c4-hparam-night-20260906-005747"
    if report.exists():
        moves.append(
            Move(
                report,
                evaluation_dir("cube4", "komi-7.5-cube4-20260906"),
                "evaluation",
            )
        )

    return moves


def _preflight(moves: list[Move]) -> None:
    missing = [str(move.source.relative_to(ROOT)) for move in moves if not move.source.exists()]
    collisions = [
        str(move.destination.relative_to(ROOT))
        for move in moves
        if move.destination.exists() and move.source != move.destination
    ]
    if missing:
        raise FileNotFoundError(f"Missing split source(s): {missing}")
    if collisions:
        raise FileExistsError(f"Split destination collision(s): {collisions}")


def split(*, execute: bool) -> dict[str, object]:
    moves = build_moves()
    _preflight(moves)
    payload = {
        "migration": MIGRATION_DATE,
        "mode": "execute" if execute else "dry-run",
        "move_only": True,
        "moves": [
            {
                "source": str(move.source.relative_to(ROOT)),
                "destination": str(move.destination.relative_to(ROOT)),
                "kind": move.kind,
            }
            for move in moves
        ],
    }
    if not execute:
        return payload

    for move in moves:
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        if move.destination.exists():
            raise FileExistsError(move.destination)
        shutil.move(str(move.source), str(move.destination))

    # Remove only now-empty bundle directories; no non-empty directory is
    # touched.  Keep the operation local to the three explicitly split roots.
    for root in (
        _p("runs/torus9/archive/torus9-legacy-20260913"),
        _p("runs/torus9/archive/torus9-invalid-20260914"),
        _p("runs/cube4/archive/komi-7.5-cube4-20260906"),
    ):
        for path in sorted([root, *root.parents], key=lambda item: len(item.parts), reverse=True):
            if path in {ROOT, ROOT / "runs"} or not path.is_dir():
                continue
            try:
                path.rmdir()
            except OSError:
                pass

    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Perform the move-only split")
    args = parser.parse_args()
    print(json.dumps(split(execute=args.execute), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
