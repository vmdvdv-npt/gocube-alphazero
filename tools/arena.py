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
from typing import Callable

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
from gocube_golden.arena_identity import stamp_evaluation_identity_metadata
from gocube_golden.run_storage import (
    ResolvedCheckpoint,
    evaluation_dir,
    evaluations_root,
    resolve_checkpoint,
    topology_for_profile,
)


ARENA_RESULT_PROVENANCE_SCHEMA = "gocube-arena-evaluation-provenance-v2"


def _require_v2_process(entrypoint: str) -> None:
    """Load the V2 guard only after this module has finished importing.

    ``gocube_golden.orchestrator_v2`` exports ``ArenaRunner`` from its package
    initializer, and ``ArenaRunner`` imports this module.  Importing the V2
    package at module scope here would therefore create a circular import.
    """
    from gocube_golden.orchestrator_v2.version import require_v2_process

    require_v2_process(entrypoint)


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


def _canonical_evaluation_output(
    profile_id: str,
    output_dir: Path,
    *,
    allowed_lineage_arena_root: Path | None = None,
    allowed_evaluation_root: Path | None = None,
) -> Path:
    output_dir = Path(output_dir).resolve()
    root = evaluations_root(topology_for_profile(profile_id)).resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        if allowed_lineage_arena_root is not None:
            lineage_arena_root = Path(allowed_lineage_arena_root).resolve()
            try:
                output_dir.relative_to(lineage_arena_root)
            except ValueError:
                pass
            else:
                return output_dir
        if allowed_evaluation_root is not None:
            evaluation_root = Path(allowed_evaluation_root).resolve()
            try:
                output_dir.relative_to(evaluation_root)
            except ValueError:
                pass
            else:
                return output_dir
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


def _normalized_validity(summary: Mapping[str, object]) -> str:
    """Classify the completed engine result at the production boundary.

    The engine owns all execution semantics and emits the telemetry used here.
    This function only turns that already-owned result into the stable boundary
    status consumed by orchestration.
    """
    telemetry = summary.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return "INVALID"
    technical_games = telemetry.get("technical_games")
    if isinstance(technical_games, bool) or not isinstance(technical_games, int):
        return "INVALID"
    if technical_games != 0:
        return "TECHNICAL"
    performance_status = telemetry.get("performance_status")
    performance_failures = telemetry.get("performance_failures")
    if not isinstance(performance_status, str) or not performance_status.strip():
        return "INVALID"
    if not isinstance(performance_failures, list):
        return "INVALID"
    if performance_status.upper() == "CRITICAL" or performance_failures:
        return "CRITICAL"
    return "VALID"


def normalize_arena_validity(summary: Mapping[str, object]) -> str:
    """Expose the production boundary classifier to legacy test seams."""
    return _normalized_validity(summary)


def _reference_payload(
    explicit: Mapping[str, object] | None,
    resolved: ResolvedCheckpoint | None,
    path: Path,
    expected_sha256: str | None,
) -> dict[str, object]:
    if explicit is not None:
        return dict(explicit)
    if resolved is not None:
        return resolved.as_reference()
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "artifact_sha256": expected_sha256,
    }


def _publish_evaluation_metadata(
    *,
    output_dir: Path,
    summary: dict[str, object],
    profile_id: str,
    master_seed: int,
    run_id: str | None,
    candidate_ref: Mapping[str, object],
    reference_ref: Mapping[str, object],
    evaluation_identity: Mapping[str, object] | None,
    evaluation_fingerprint: str | None,
) -> str | None:
    """Commit the complete Arena result and its single provenance record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validity = _normalized_validity(summary)
    summary["validity"] = validity
    _write_json(output_dir / "summary.json", summary)

    published_evaluation_id = run_id
    if evaluation_identity is not None:
        scientific = evaluation_identity.get("scientific_contract")
        if not isinstance(scientific, Mapping):
            scientific = {}
        provenance: dict[str, object] = {
            "schema": ARENA_RESULT_PROVENANCE_SCHEMA,
            "evaluation_id": run_id,
            "candidate": dict(evaluation_identity.get("candidate", candidate_ref)),
            "reference": dict(evaluation_identity.get("reference", reference_ref)),
            "profile": str(scientific.get("profile", profile_id)),
            "master_seed": int(evaluation_identity.get("master_seed", master_seed)),
            "startset": evaluation_identity.get("startset"),
            "arena_contract": {
                "scientific": dict(scientific),
                "execution": dict(evaluation_identity.get("execution_contract", {})),
                "workload": dict(evaluation_identity.get("workload", {})),
            },
            "training_mutated": False,
        }
    else:
        provenance = {
            "schema": "gocube-checkpoint-evaluation-provenance-v1",
            "evaluation_id": run_id,
            "candidate": dict(candidate_ref),
            "reference": dict(reference_ref),
            "profile": profile_id,
            "master_seed": master_seed,
            "training_mutated": False,
        }

    _write_provenance(output_dir, provenance)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise RuntimeError(f"Arena manifest is not an object: {manifest_path}")
        updated_manifest = dict(manifest)
        updated_manifest["evaluation_id"] = published_evaluation_id
        updated_manifest["checkpoint_references"] = {
            "candidate": dict(candidate_ref),
            "reference": dict(reference_ref),
        }
        updated_manifest["provenance"] = str(output_dir / "provenance.json")
        updated_manifest["validity"] = validity
        _write_json(manifest_path, updated_manifest)

    if evaluation_identity is not None and evaluation_fingerprint is not None and run_id is not None:
        identity_schema = str(evaluation_identity.get("schema", ""))
        if not identity_schema:
            raise ValueError("evaluation identity schema is required for publication")
        stamp_evaluation_identity_metadata(
            output_dir,
            run_id,
            evaluation_fingerprint,
            identity_schema=identity_schema,
        )
    return published_evaluation_id


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
    evaluation_identity: Mapping[str, object] | None = None,
    evaluation_fingerprint: str | None = None,
    allowed_lineage_arena_root: Path | None = None,
    allowed_evaluation_root: Path | None = None,
    workload: Mapping[str, object] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Resolve checkpoint paths/references and invoke the universal Arena."""
    _require_v2_process("tools.arena.run_arena")
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
        allowed_lineage_arena_root=allowed_lineage_arena_root,
        allowed_evaluation_root=allowed_evaluation_root,
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
        workload=workload,
        progress_callback=progress_callback,
    )
    if candidate_identity is not None or reference_identity is not None or evaluation_identity is not None:
        candidate_ref = _reference_payload(
            (
                evaluation_identity.get("candidate")
                if isinstance(evaluation_identity, Mapping)
                and isinstance(evaluation_identity.get("candidate"), Mapping)
                else None
            ),
            candidate_identity,
            candidate_path,
            expected_candidate_artifact_sha256,
        )
        reference_ref = _reference_payload(
            (
                evaluation_identity.get("reference")
                if isinstance(evaluation_identity, Mapping)
                and isinstance(evaluation_identity.get("reference"), Mapping)
                else None
            ),
            reference_identity,
            reference_path,
            expected_reference_artifact_sha256,
        )
        cross_lineage = (
            isinstance(candidate_ref.get("lineage_id"), str)
            and isinstance(reference_ref.get("lineage_id"), str)
            and candidate_ref.get("lineage_id") != reference_ref.get("lineage_id")
        )
        publication_run_id = (
            run_id
            if evaluation_identity is not None
            else (
                f"{candidate_ref['lineage_id']}-M{int(candidate_ref['generation']):04d}-vs-"
                f"{reference_ref['lineage_id']}-M{int(reference_ref['generation']):04d}"
                if cross_lineage
                else None
            )
        )
        published_evaluation_id = _publish_evaluation_metadata(
            output_dir=output_dir,
            summary=result,
            profile_id=profile.profile_id,
            master_seed=master_seed,
            run_id=publication_run_id,
            candidate_ref=candidate_ref,
            reference_ref=reference_ref,
            evaluation_identity=evaluation_identity,
            evaluation_fingerprint=evaluation_fingerprint,
        )
        result["evaluation_id"] = published_evaluation_id
        result["candidate_reference"] = candidate_ref
        result["reference_reference"] = reference_ref
    else:
        # A path-only CLI invocation has no canonical checkpoint references to
        # publish, but it still returns the normalized production status.
        output_dir.mkdir(parents=True, exist_ok=True)
        result["validity"] = _normalized_validity(result)
        _write_json(output_dir / "summary.json", result)
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
    _require_v2_process("tools/arena.py")
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
                "validity": summary["validity"],
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
