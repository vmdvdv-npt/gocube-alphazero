"""Supervised child entrypoint for one production V2 generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from .generation_runner import GenerationRunner
from .production_arm import _deserialize_resolved_input, _read_json
from .torus9_production import Torus9ProductionGenerationPath, V2CheckpointPublisher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    resolved = _deserialize_resolved_input(_read_json(request_path))
    result = GenerationRunner(
        Torus9ProductionGenerationPath(publisher=V2CheckpointPublisher())
    ).run(resolved)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        result_path,
        canonical_json(
            {
                "schema": "gocube-orchestrator-v2-generation-result-v1",
                "generation": result.generation,
                "checkpoint": result.checkpoint.to_dict(),
                "commit_artifact": result.commit_artifact.to_dict(),
            }
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

