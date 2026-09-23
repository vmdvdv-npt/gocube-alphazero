"""Cube V2 wrapper around the single board-agnostic Arena engine."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

from tools.arena_engine import ArenaExecutionConfig, run_arena
from tools.arena_profiles.cube_v2 import CubeV2ArenaProfile

from .cube_arena_contract_v2 import (
    CUBE_ARENA_RESULT_SCHEMA,
    CUBE_ARENA_RESULT_SCHEMA_VERSION,
    CubeArenaSearchConfig,
)
from .cube_game_contract_v2 import validate_cube_size
from .run_storage import active_lineage_dir, evaluations_root, resolve_checkpoint


@dataclass(frozen=True)
class CubeArenaResult:
    arena_schema: str
    schema_version: int
    topology: str
    cube_size: int
    checkpoint_a: Mapping[str, object]
    checkpoint_b: Mapping[str, object]
    games_requested: int
    games_valid: int
    wins_a: int
    wins_b: int
    draws: int
    technical: int
    invalid: int
    search_config: Mapping[str, object]
    search_config_fingerprint: str
    seed: int
    completion_status: str
    timing: Mapping[str, object]
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_schema": self.arena_schema,
            "schema_version": self.schema_version,
            "topology": self.topology,
            "cube_size": self.cube_size,
            "checkpoint_a": dict(self.checkpoint_a),
            "checkpoint_b": dict(self.checkpoint_b),
            "games_requested": self.games_requested,
            "games_valid": self.games_valid,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "draws": self.draws,
            "technical": self.technical,
            "invalid": self.invalid,
            "search_config": dict(self.search_config),
            "search_config_fingerprint": self.search_config_fingerprint,
            "seed": self.seed,
            "completion_status": self.completion_status,
            "timing": dict(self.timing),
            "output_dir": self.output_dir,
        }


def _reference(
    value: Mapping[str, object],
    *,
    label: str,
    topology: str,
    temporary_root: Path | None,
) -> tuple[Path, str, dict[str, object]]:
    if temporary_root is None:
        resolved = resolve_checkpoint(value, topology=topology)
        result = resolved.as_reference()
        return resolved.path, resolved.sha256, result

    path_value = value.get("path")
    sha = value.get("sha256") or value.get("artifact_sha256")
    if not path_value or not isinstance(sha, str) or not sha.startswith("sha256:"):
        raise ValueError(f"Cube Arena {label} reference requires path and SHA-256")
    path = Path(str(path_value)).resolve()
    try:
        path.relative_to(temporary_root)
    except ValueError as exc:
        raise ValueError(f"Temporary Cube Arena {label} checkpoint must stay under temporary_root") from exc
    result = dict(value)
    result["path"] = str(path)
    result["sha256"] = sha
    result["artifact_sha256"] = sha
    return path, sha, result


def _write_result(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_cube_arena(
    *,
    size: int,
    candidate_checkpoint: Mapping[str, object],
    reference_checkpoint: Mapping[str, object],
    output_dir: str | Path,
    search_config: CubeArenaSearchConfig,
    execution_config: ArenaExecutionConfig,
    seed: int,
    temporary_root: str | Path | None = None,
) -> CubeArenaResult:
    """Run one diagnostic Cube Arena comparison without gating training."""

    size = validate_cube_size(size)
    search_config.validate()
    execution_config.validate_base()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise ValueError("Cube Arena seed must be a positive explicit integer")

    topology = f"cube{size}"
    temporary = None if temporary_root is None else Path(temporary_root).resolve()
    candidate_path, candidate_sha, candidate_ref = _reference(
        candidate_checkpoint,
        label="candidate",
        topology=topology,
        temporary_root=temporary,
    )
    reference_path, reference_sha, reference_ref = _reference(
        reference_checkpoint,
        label="reference",
        topology=topology,
        temporary_root=temporary,
    )
    profile = CubeV2ArenaProfile(size=size, search_config=search_config)
    target = Path(output_dir).resolve()
    if temporary is not None:
        try:
            target.relative_to(temporary)
        except ValueError as exc:
            raise ValueError("Temporary Cube Arena output must stay under temporary_root") from exc
    else:
        candidate_lineage = str(candidate_ref.get("lineage_id") or "")
        reference_lineage = str(reference_ref.get("lineage_id") or "")
        if candidate_lineage and candidate_lineage == reference_lineage:
            owner = active_lineage_dir(topology, candidate_lineage).resolve() / "arena"
        else:
            owner = evaluations_root(topology).resolve()
        try:
            target.relative_to(owner)
        except ValueError as exc:
            raise ValueError(
                "Cube Arena output path violates canonical run-storage ownership"
            ) from exc
    summary = run_arena(
        profile=profile,
        candidate_path=candidate_path,
        reference_path=reference_path,
        output_dir=target,
        candidate_label=str(
            candidate_ref.get("checkpoint_id")
            or candidate_ref.get("label")
            or candidate_path.stem
        ),
        reference_label=str(
            reference_ref.get("checkpoint_id")
            or reference_ref.get("label")
            or reference_path.stem
        ),
        comparison=(
            f"{candidate_ref.get('lineage_id', 'candidate')}--"
            f"{candidate_ref.get('checkpoint_id') or candidate_path.stem}-vs-"
            f"{reference_ref.get('lineage_id', 'reference')}--"
            f"{reference_ref.get('checkpoint_id') or reference_path.stem}"
        ),
        master_seed=seed,
        config=execution_config,
        expected_candidate_model_hash=(
            str(candidate_ref["model_hash"])
            if candidate_ref.get("model_hash") is not None
            else None
        ),
        expected_candidate_artifact_sha256=candidate_sha,
        expected_reference_model_hash=(
            str(reference_ref["model_hash"])
            if reference_ref.get("model_hash") is not None
            else None
        ),
        expected_reference_artifact_sha256=reference_sha,
    )
    telemetry = summary.get("telemetry")
    timing: dict[str, object] = {}
    if isinstance(telemetry, Mapping):
        for key in ("wall_time_sec", "startup_wall_time_sec", "process_wall_time_sec"):
            if key in telemetry:
                timing[key] = telemetry[key]

    result = CubeArenaResult(
        arena_schema=CUBE_ARENA_RESULT_SCHEMA,
        schema_version=CUBE_ARENA_RESULT_SCHEMA_VERSION,
        topology=topology,
        cube_size=size,
        checkpoint_a=candidate_ref,
        checkpoint_b=reference_ref,
        games_requested=int(summary.get("games_requested", execution_config.games)),
        games_valid=int(summary.get("games_valid", 0)),
        wins_a=int(summary.get("candidate_wins", 0)),
        wins_b=int(summary.get("reference_wins", 0)),
        draws=int(summary.get("draws", 0)),
        technical=int(
            summary.get("technical_only_games", summary.get("technical_games", 0))
        ),
        invalid=int(summary.get("invalid_games", 0)),
        search_config=search_config.identity_payload(),
        search_config_fingerprint=search_config.fingerprint,
        seed=seed,
        completion_status=str(summary.get("completion_status", "COMPLETE")),
        timing=timing,
        output_dir=str(target),
    )
    _write_result(target / "arena-result.json", result.to_dict())
    return result


__all__ = ["CubeArenaResult", "CubeArenaSearchConfig", "run_cube_arena"]
