"""Shared Arena evaluation identity and reuse primitives.

The staged V1 harness and Orchestrator V2 use different identity payload
schemas, but they must share the same canonical serialization, persistence,
and fail-closed reuse rules.  This module deliberately knows nothing about
checkpoint discovery or Arena execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EVALUATION_IDENTITY_FILENAME = "evaluation-identity.json"
EVALUATION_IDENTITY_RECORD_SCHEMA = "gocube-arena-evaluation-identity-record-v1"
EVALUATION_IDENTITY_HASH_PREFIX = 12


def canonical_evaluation_json(payload: Mapping[str, object]) -> str:
    """Return the stable JSON representation used by Arena identity hashes."""
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def evaluation_fingerprint(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        canonical_evaluation_json(payload).encode("utf-8")
    ).hexdigest()


def evaluation_id(
    *,
    candidate_lineage_id: str,
    candidate_generation: int,
    reference_lineage_id: str,
    reference_generation: int,
    fingerprint: str,
) -> str:
    """Build a stable, path-safe ID from checkpoint identities and fingerprint."""
    return (
        f"{candidate_lineage_id}-M{int(candidate_generation):04d}-vs-"
        f"{reference_lineage_id}-M{int(reference_generation):04d}-"
        f"{fingerprint[:EVALUATION_IDENTITY_HASH_PREFIX]}"
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_evaluation_identity(
    output: Path,
    run_id: str,
    identity: Mapping[str, object],
    fingerprint: str,
) -> None:
    """Create the durable identity marker before Arena execution starts."""
    if evaluation_fingerprint(identity) != fingerprint:
        raise ValueError("Evaluation identity fingerprint is internally inconsistent")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / EVALUATION_IDENTITY_FILENAME,
        {
            "schema": EVALUATION_IDENTITY_RECORD_SCHEMA,
            "evaluation_id": run_id,
            "fingerprint": fingerprint,
            "identity": dict(identity),
        },
    )


def stamp_evaluation_identity_metadata(
    output: Path,
    run_id: str,
    fingerprint: str,
    *,
    identity_schema: str,
) -> None:
    marker = {
        "schema": identity_schema,
        "path": EVALUATION_IDENTITY_FILENAME,
        "fingerprint": fingerprint,
    }
    for filename in ("provenance.json", "manifest.json"):
        path = output / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Arena metadata is not an object: {path}")
        updated = dict(payload)
        updated["evaluation_id"] = run_id
        updated["evaluation_identity"] = marker
        _write_json(path, updated)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _checkpoint_sha(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    return str(
        value.get("checkpoint_sha256")
        or value.get("artifact_sha256")
        or value.get("sha256")
        or ""
    )


def _checkpoint_matches(actual: object, expected: object) -> bool:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    actual_lineage = str(actual.get("lineage_id", ""))
    expected_lineage = str(expected.get("lineage_id", ""))
    actual_generation = int(actual.get("generation", -1))
    expected_generation = int(expected.get("generation", -1))
    return (
        actual_lineage == expected_lineage
        and actual_generation == expected_generation
        and _checkpoint_sha(actual) == _checkpoint_sha(expected)
    )


def _identity_profile(identity: Mapping[str, object]) -> str:
    profile = identity.get("profile")
    if profile is not None:
        return str(profile)
    scientific = identity.get("scientific_contract")
    if isinstance(scientific, Mapping):
        return str(scientific.get("profile", ""))
    return ""


def _identity_games(identity: Mapping[str, object]) -> int:
    return int(identity.get("games", -1))


def load_reusable_evaluation(
    output: Path,
    expected_identity: Mapping[str, object],
    expected_fingerprint: str,
    *,
    allow_legacy_synthetic: bool = False,
) -> dict[str, object] | None:
    """Validate a complete prior result, or return ``None`` for an interrupted one."""
    identity_path = output / EVALUATION_IDENTITY_FILENAME
    if not identity_path.is_file():
        raise RuntimeError(
            f"evaluation identity/contract mismatch: missing persisted evaluation identity: {output}"
        )
    try:
        record = _read_json(identity_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"evaluation identity/contract mismatch: malformed persisted evaluation identity: {output}"
        ) from exc
    if record.get("schema") != EVALUATION_IDENTITY_RECORD_SCHEMA:
        raise RuntimeError(
            f"evaluation identity/contract mismatch: unsupported persisted identity schema: {output}"
        )
    saved_identity = record.get("identity")
    saved_fingerprint = record.get("fingerprint")
    if not isinstance(saved_identity, Mapping) or not isinstance(saved_fingerprint, str):
        raise RuntimeError(
            f"evaluation identity/contract mismatch: malformed persisted identity payload: {output}"
        )
    try:
        recomputed = evaluation_fingerprint(saved_identity)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"evaluation identity/contract mismatch: non-canonical persisted identity payload: {output}"
        ) from exc
    if recomputed != saved_fingerprint:
        raise RuntimeError(
            f"evaluation identity/contract mismatch: persisted full fingerprint does not match canonical payload: {output}"
        )
    if evaluation_fingerprint(expected_identity) != expected_fingerprint:
        raise ValueError("Expected evaluation identity fingerprint is internally inconsistent")
    if dict(saved_identity) != dict(expected_identity) or saved_fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"evaluation identity/contract mismatch: persisted payload does not match current contract: {output}"
        )
    if str(record.get("evaluation_id")) != output.name:
        raise RuntimeError(
            f"evaluation identity/contract mismatch: persisted evaluation ID disagrees with directory: {output}"
        )

    summary_path = output / "summary.json"
    provenance_path = output / "provenance.json"
    manifest_path = output / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        return None
    try:
        summary = _read_json(summary_path)
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"evaluation identity/contract mismatch: malformed committed evaluation artifact: {output}"
        ) from exc

    if not provenance_path.is_file():
        # Older injected synthetic Arena seams predate the production boundary
        # publication contract.  They may still be reused when the durable
        # identity and manifest prove the same run; real production results
        # always include provenance and are checked below.
        if (
            allow_legacy_synthetic
            and "validity" not in summary
            and str(manifest.get("run_id")) == output.name
        ):
            return summary
        return None
    try:
        provenance = _read_json(provenance_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"evaluation identity/contract mismatch: malformed committed evaluation artifact: {output}"
        ) from exc

    expected_candidate = expected_identity.get("candidate")
    expected_reference = expected_identity.get("reference")
    if (
        str(manifest.get("run_id")) != output.name
        or not _checkpoint_matches(provenance.get("candidate"), expected_candidate)
        or not _checkpoint_matches(provenance.get("reference"), expected_reference)
        or int(summary.get("games", -1)) != _identity_games(expected_identity)
        or str(provenance.get("profile", "")) != _identity_profile(expected_identity)
        or int(provenance.get("master_seed", -1)) != int(expected_identity.get("master_seed", -2))
    ):
        raise RuntimeError(
            f"evaluation identity/contract mismatch: Arena result metadata disagrees with persisted identity: {output}"
        )

    persisted_validity = summary.get("validity")
    if persisted_validity is not None and str(persisted_validity).upper() != "VALID":
        raise RuntimeError(
            "Existing evaluation failed production validity/performance gates: "
            f"validity={persisted_validity!r}: {output}"
        )

    telemetry = summary.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise RuntimeError(
            f"Existing evaluation failed production validity/performance gates: malformed telemetry: {output}"
        )
    technical_games = telemetry.get("technical_games")
    if isinstance(technical_games, bool) or not isinstance(technical_games, int):
        raise RuntimeError(
            f"Existing evaluation failed production validity/performance gates: malformed technical_games: {output}"
        )
    if technical_games != 0:
        raise ValueError(f"Existing Arena has technical outcomes: {output}")
    performance_status = telemetry.get("performance_status")
    performance_failures = telemetry.get("performance_failures")
    if not isinstance(performance_status, str) or not performance_status.strip():
        raise RuntimeError(
            f"Existing evaluation failed production validity/performance gates: malformed performance_status: {output}"
        )
    if not isinstance(performance_failures, list):
        raise RuntimeError(
            f"Existing evaluation failed production validity/performance gates: malformed performance_failures: {output}"
        )
    if performance_status.strip().upper() == "CRITICAL" or performance_failures:
        raise RuntimeError(
            "Existing evaluation failed production validity/performance gates: "
            f"performance_status={performance_status!r}, performance_failures={performance_failures!r}: {output}"
        )
    return summary


__all__ = [
    "EVALUATION_IDENTITY_FILENAME",
    "EVALUATION_IDENTITY_HASH_PREFIX",
    "EVALUATION_IDENTITY_RECORD_SCHEMA",
    "canonical_evaluation_json",
    "evaluation_fingerprint",
    "evaluation_id",
    "load_reusable_evaluation",
    "stamp_evaluation_identity_metadata",
    "write_evaluation_identity",
]
