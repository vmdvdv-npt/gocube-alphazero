#!/usr/bin/env python3
"""Single production Arena CLI for every supported board/game profile.

Examples:
    .venv/bin/python tools/arena.py --candidate path/to/M14.pt
    .venv/bin/python tools/arena.py --profile torus9 --candidate A.pt --reference B.pt

The default production workload is the Legion high-volume preset (192 games,
16 workers, 12 contexts per worker).  For the standard-64 comparison use
explicit workload overrides, including ``--games 64 --games-per-worker 4``.

With no --reference, the candidate plays itself. With --profile auto (default),
the game profile is resolved from checkpoint metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.arena_engine import (
    ArenaExecutionConfig,
    DEFAULT_GAMES,
    DEFAULT_GAMES_PER_WORKER,
    DEFAULT_INFERENCE_BATCH_ROWS,
    DEFAULT_INFERENCE_BATCH_WAIT_MS,
    DEFAULT_MASTER_SEED,
    DEFAULT_WORKERS,
    run_arena as run_engine,
)
from tools.arena_profiles import available_profiles, detect_profile, get_profile
from gocube_golden.run_storage import (
    ResolvedCheckpoint,
    evaluation_dir,
    evaluations_root,
    resolve_checkpoint,
    topology_for_profile,
)


def _resolve_profile(profile_name: str, candidate: Path):
    if profile_name == "auto":
        return detect_profile(candidate)
    return get_profile(profile_name)


def _default_output(profile_id: str, candidate: Path, reference: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return evaluation_dir(
        topology_for_profile(profile_id),
        f"{profile_id}-{candidate.stem}-vs-{reference.stem}-{stamp}",
    )


def _canonical_evaluation_output(profile_id: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    root = evaluations_root(topology_for_profile(profile_id)).resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Arena output must be inside the canonical runs/<topology>/evaluations tree: "
            f"{output_dir}"
        ) from exc
    return output_dir


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_provenance(output_dir: Path, payload: Mapping[str, object]) -> None:
    _write_json(output_dir / "provenance.json", payload)


def run_arena(
    *,
    candidate_path: Path | Mapping[str, Any],
    reference_path: Path | Mapping[str, Any] | None = None,
    profile_name: str = "auto",
    output_dir: Path | None = None,
    candidate_label: str | None = None,
    reference_label: str | None = None,
    run_id: str | None = None,
    comparison: str | None = None,
    master_seed: int = DEFAULT_MASTER_SEED,
    config: ArenaExecutionConfig = ArenaExecutionConfig(),
    expected_candidate_model_hash: str | None = None,
    expected_candidate_artifact_sha256: str | None = None,
    expected_reference_model_hash: str | None = None,
    expected_reference_artifact_sha256: str | None = None,
) -> dict[str, object]:
    """Resolve checkpoint paths/references and invoke the universal Arena."""
    candidate_identity: ResolvedCheckpoint | None = None
    if isinstance(candidate_path, Mapping) and not candidate_path.get("path"):
        candidate_identity = resolve_checkpoint(
            candidate_path,
            topology=str(candidate_path.get("topology") or "") or None,
        )
        candidate_probe = candidate_identity.path
    else:
        candidate_probe = (
            Path(str(candidate_path.get("path")))
            if isinstance(candidate_path, Mapping)
            else Path(candidate_path)
        )
    profile = _resolve_profile(profile_name, candidate_probe)
    topology = topology_for_profile(profile.profile_id)
    reference_identity: ResolvedCheckpoint | None = None
    if isinstance(candidate_path, Mapping):
        if candidate_identity is None:
            candidate_identity = resolve_checkpoint(candidate_path, topology=topology)
        candidate_path = candidate_identity.path
    else:
        candidate_path = Path(candidate_path).resolve()
    if reference_path is None:
        reference_path = candidate_path
    elif isinstance(reference_path, Mapping):
        reference_identity = resolve_checkpoint(reference_path, topology=topology)
        reference_path = reference_identity.path
    else:
        reference_path = Path(reference_path).resolve()
    output_dir = _canonical_evaluation_output(
        profile.profile_id,
        output_dir or _default_output(profile.profile_id, candidate_path, reference_path),
    )
    result = run_engine(
        profile=profile,
        candidate_path=candidate_path,
        reference_path=reference_path,
        output_dir=output_dir,
        candidate_label=candidate_label,
        reference_label=reference_label,
        run_id=run_id,
        comparison=comparison,
        master_seed=master_seed,
        config=config,
        expected_candidate_model_hash=(
            expected_candidate_model_hash
            or (candidate_identity.reference.get("model_hash") if candidate_identity else None)
        ),
        expected_candidate_artifact_sha256=(
            expected_candidate_artifact_sha256
            or (candidate_identity.sha256 if candidate_identity else None)
        ),
        expected_reference_model_hash=(
            expected_reference_model_hash
            or (reference_identity.reference.get("model_hash") if reference_identity else None)
        ),
        expected_reference_artifact_sha256=(
            expected_reference_artifact_sha256
            or (reference_identity.sha256 if reference_identity else None)
        ),
    )
    if candidate_identity is not None or reference_identity is not None:
        candidate_ref = candidate_identity.as_reference() if candidate_identity else {
            "path": str(candidate_path),
            "sha256": expected_candidate_artifact_sha256,
        }
        reference_ref = reference_identity.as_reference() if reference_identity else {
            "path": str(reference_path),
            "sha256": expected_reference_artifact_sha256,
        }
        cross_lineage = (
            candidate_identity is not None
            and reference_identity is not None
            and candidate_identity.lineage_id != reference_identity.lineage_id
        )
        evaluation_id = (
            f"{candidate_identity.lineage_id}-M{candidate_identity.generation:04d}-vs-"
            f"{reference_identity.lineage_id}-M{reference_identity.generation:04d}"
            if cross_lineage
            else None
        )
        _write_provenance(
            output_dir,
            {
                "schema": "gocube-checkpoint-evaluation-provenance-v1",
                "evaluation_id": evaluation_id,
                "candidate": candidate_ref,
                "reference": reference_ref,
                "profile": profile.profile_id,
                "master_seed": master_seed,
                "training_mutated": False,
            },
        )
        manifest_path = output_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["evaluation_id"] = evaluation_id
            manifest["checkpoint_references"] = {
                "candidate": candidate_ref,
                "reference": reference_ref,
            }
            manifest["provenance"] = str(output_dir / "provenance.json")
            _write_json(manifest_path, manifest)
            _write_provenance(
                output_dir,
                {
                    "schema": "gocube-checkpoint-evaluation-provenance-v1",
                    "evaluation_id": evaluation_id,
                    "candidate": candidate_ref,
                    "reference": reference_ref,
                    "profile": profile.profile_id,
                    "master_seed": master_seed,
                    "training_mutated": False,
                },
            )
        result["evaluation_id"] = evaluation_id
        result["candidate_reference"] = candidate_ref
        result["reference_reference"] = reference_ref
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="auto",
        choices=("auto", *available_profiles()),
        help="Game/board semantics only. Execution engine is always the same.",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Defaults to --candidate for self A/B",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--reference-label", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--comparison", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--games-per-worker", type=int, default=DEFAULT_GAMES_PER_WORKER)
    parser.add_argument(
        "--inference-batch-rows",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_ROWS,
    )
    parser.add_argument(
        "--inference-batch-wait-ms",
        type=float,
        default=DEFAULT_INFERENCE_BATCH_WAIT_MS,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-candidate-model-hash", default=None)
    parser.add_argument("--expected-candidate-artifact-sha256", default=None)
    parser.add_argument("--expected-reference-model-hash", default=None)
    parser.add_argument("--expected-reference-artifact-sha256", default=None)
    parser.add_argument(
        "--debug-non-production",
        action="store_true",
        help="Allow reduced games/workers or CPU; result is explicitly non-production.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = args.reference or args.candidate
    profile = _resolve_profile(args.profile, args.candidate)
    output = args.output or _default_output(
        profile.profile_id,
        args.candidate,
        reference,
    )
    output = _canonical_evaluation_output(profile.profile_id, output)
    config = ArenaExecutionConfig(
        games=args.games,
        workers=args.workers,
        games_per_worker=args.games_per_worker,
        inference_batch_rows=args.inference_batch_rows,
        inference_batch_wait_ms=args.inference_batch_wait_ms,
        device=args.device,
        strict_production=not args.debug_non_production,
    )
    summary = run_engine(
        profile=profile,
        candidate_path=args.candidate,
        reference_path=reference,
        output_dir=output,
        candidate_label=args.candidate_label,
        reference_label=args.reference_label,
        run_id=args.run_id,
        comparison=args.comparison,
        master_seed=args.seed,
        config=config,
        expected_candidate_model_hash=args.expected_candidate_model_hash,
        expected_candidate_artifact_sha256=args.expected_candidate_artifact_sha256,
        expected_reference_model_hash=args.expected_reference_model_hash,
        expected_reference_artifact_sha256=args.expected_reference_artifact_sha256,
    )
    print(
        json.dumps(
            {
                "arena_engine": summary["arena_engine"],
                "arena_profile": summary["arena_profile"],
                "run_id": summary["run_id"],
                "comparison": summary["comparison"],
                "games": summary["games"],
                "W/L/D": summary["W/L/D"],
                "performance_status": summary["telemetry"]["performance_status"],
                "games_per_hour": summary["telemetry"]["games_per_hour"],
                "mean_inference_batch_rows": summary["telemetry"][
                    "mean_inference_batch_rows"
                ],
                "effective_cpu_cores": summary["telemetry"][
                    "effective_cpu_cores"
                ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
