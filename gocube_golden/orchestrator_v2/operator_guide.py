"""Versioned launch instructions and their reviewed interface contract."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
import subprocess

GUIDE_PATH = "docs/orchestrator-v2/ORCHESTRATOR_V2_LAUNCH_GUIDE.md"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Changes to these launch/mode interfaces require reviewing the guide and
# updating its contract marker. This is checked by CI and before job launch.
INTERFACE_PATHS = (
    "gocube_golden/orchestrator_v2/operator_guide.py",
    "gocube_golden/orchestrator_v2/operator_job.py",
    "gocube_golden/orchestrator_v2/production_entrypoint.py",
    "gocube_golden/orchestrator_v2/experiment_plan.py",
    "gocube_golden/orchestrator_v2/offline_replay.py",
    "gocube_golden/orchestrator_v2/run_spec.py",
    "gocube_golden/orchestrator_v2/workflow.py",
    "gocube_golden/scenarios/experiment/runner.py",
    "gocube_golden/torus9_five_channel_training.py",
    "gocube_golden/torus9_pcr.py",
)


def interface_fingerprint(root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    for relative in INTERFACE_PATHS:
        digest.update(relative.encode() + b"\0")
        digest.update((root / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def guide_metadata(root: Path = REPO_ROOT) -> dict[str, str]:
    path = root / GUIDE_PATH
    raw = path.read_bytes()
    marker = re.search(r"<!-- reviewed-interface-sha256: ([0-9a-f]{64}) -->", raw.decode())
    fingerprint = interface_fingerprint(root)
    if marker is None or marker.group(1) != fingerprint:
        raise ValueError(f"V2 launch guide is stale: review {path}, update its reviewed-interface-sha256 marker, and run the guide contract tests before launching.")
    identity = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=False)
    identity = identity.stdout.strip() if identity.returncode == 0 else "main"
    return {"code_commit": identity, "path": str(path), "repository_path": GUIDE_PATH,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "reviewed_interface_sha256": fingerprint,
            "github": "https://github.com/vmdvdv-npt/gocube-alphazero/blob/" + identity + "/" + GUIDE_PATH}


def announce_guide() -> dict[str, str]:
    metadata = guide_metadata()
    print("Orchestrator V2: before selecting a mode or launching/resuming a job, "
          "read the launch guide for this code revision:\n" + metadata["path"] +
          "\nGitHub: " + metadata["github"], file=sys.stderr)
    return metadata


def read_guide() -> str:
    metadata = guide_metadata()
    return Path(metadata["path"]).read_text(encoding="utf-8")
