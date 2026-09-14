#!/usr/bin/env python3
"""Move pre-existing Torus 9×9 run namespaces into a non-discoverable archive.

This is intentionally a move of complete run directories, never a delete.  The
current launcher uses ``runs/torus9-golden-v3-active/<run-id>`` only, so the
archive namespace cannot participate in automatic resume/discovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "runs"
ARCHIVE_NAME = "archive/torus9-legacy-20260913"
ACTIVE_NAME = "torus9-golden-v3-active"


def archive_torus9_runs(runs_root: Path = DEFAULT_RUNS_ROOT) -> dict[str, Any]:
    runs_root = runs_root.resolve()
    archive_root = runs_root / ARCHIVE_NAME
    active_root = runs_root / ACTIVE_NAME
    if active_root.exists():
        raise RuntimeError(f"Refusing to archive while the new active namespace exists: {active_root}")

    sources = sorted(
        path for path in runs_root.glob("torus9-*")
        if path.is_dir() and path.name != ACTIVE_NAME
    )
    destinations = [archive_root / source.name for source in sources]
    collisions = [str(path) for path in destinations if path.exists()]
    if collisions:
        raise FileExistsError(f"Archive destination already exists; refusing a partial move: {collisions}")

    archive_root.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for source, destination in zip(sources, destinations):
        shutil.move(str(source), str(destination))
        moved.append({
            "source": str(source.relative_to(runs_root)),
            "archive": str(destination.relative_to(runs_root)),
        })

    manifest = {
        "manifest_schema": "torus9-legacy-archive-v1",
        "runs_root": str(runs_root),
        "archive_namespace": str(archive_root.relative_to(runs_root)),
        "automatic_discovery": False,
        "deletion_performed": False,
        "moved_runs": moved,
    }
    (archive_root / "archive-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    args = parser.parse_args()
    manifest = archive_torus9_runs(args.runs_root)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
