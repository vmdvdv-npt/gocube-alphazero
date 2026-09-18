from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import gocube_golden.torus9_run_owned_training as run_owned_training
from gocube_golden.artifact_catalog import ARTIFACT_VALIDATION_SCHEMA, sha256_file
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    current_torus9_content_fingerprint,
    load_torus9_current_profile,
    profile_fingerprint,
)
from gocube_golden.torus9_training import TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA


def _profile():
    profile = deepcopy(load_torus9_current_profile())
    profile["experiment"] = {"kind": "trusted-historical-replay-test"}
    profile["training"]["learning_rate"] = 0.0003
    profile["replay"]["window"] = "rolling last 6 generations"
    profile["replay"]["generations"] = 6
    profile["replay"]["cap"] = 40000
    profile["self_play"]["mcts_simulations"] = 128
    profile["content_fingerprint"] = current_torus9_content_fingerprint(profile)
    profile["profile_fingerprint"] = profile_fingerprint(profile)
    return profile


def _row(row_id: str = "M50:game:0:0") -> dict[str, object]:
    return {
        "source_generation": 50,
        "replay_row_id": row_id,
        "observation": [[0.0] * 81 for _ in range(6)],
        "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        "ownership_target": [0] * 81,
        "score_target": 0.0,
    }


def test_complete_catalog_evidence_is_trusted() -> None:
    fingerprint = "sha256:" + "4" * 64
    digest = {"sha256": "sha256:" + "1" * 64, "size_bytes": 626_000_000}
    identity = {
        **digest,
        "row_count": 35_455,
        "canonical_replay_fingerprint": fingerprint,
        "validation_schema": ARTIFACT_VALIDATION_SCHEMA,
    }

    resolved, trusted = run_owned_training.Torus9TrainingAdapter._verified_replay_evidence(
        identity,
        digest,
        row_count=35_455,
    )

    assert trusted is True
    assert resolved == fingerprint


def test_incomplete_catalog_evidence_falls_back_to_full_validation() -> None:
    digest = {"sha256": "sha256:" + "1" * 64, "size_bytes": 1234}
    identity = {
        **digest,
        "canonical_replay_fingerprint": "sha256:" + "4" * 64,
        "validation_schema": ARTIFACT_VALIDATION_SCHEMA,
        # row_count is intentionally absent: the artifact is not trusted.
    }

    resolved, trusted = run_owned_training.Torus9TrainingAdapter._verified_replay_evidence(
        identity,
        digest,
        row_count=12,
    )

    assert trusted is False
    assert resolved is None


def test_catalog_identity_mismatch_still_fails_closed() -> None:
    digest = {"sha256": "sha256:" + "1" * 64, "size_bytes": 1234}
    identity = {
        "sha256": "sha256:" + "2" * 64,
        "size_bytes": 1234,
        "row_count": 12,
        "canonical_replay_fingerprint": "sha256:" + "4" * 64,
        "validation_schema": ARTIFACT_VALIDATION_SCHEMA,
    }

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_owned_training.Torus9TrainingAdapter._verified_replay_evidence(
            identity,
            digest,
            row_count=12,
        )


def test_trusted_historical_rows_skip_deep_validation_and_row_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = run_owned_training.Torus9TrainingAdapter(profile=_profile())
    row = _row()
    adapter._trust_historical_rows((row,))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("trusted historical replay must not be deep-validated or re-fingerprinted")

    monkeypatch.setattr(adapter, "validate_sample", forbidden)
    monkeypatch.setattr(run_owned_training._base, "value_fingerprint", forbidden)

    adapter.validate_replay((row,))


def test_untrusted_row_keeps_existing_semantic_validation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = run_owned_training.Torus9TrainingAdapter(profile=_profile())
    row = _row("M50:game:0:1")
    calls: list[str] = []

    monkeypatch.setattr(
        run_owned_training._base,
        "value_fingerprint",
        lambda _row: "sha256:" + "9" * 64,
    )
    monkeypatch.setattr(
        adapter,
        "validate_sample",
        lambda sample: calls.append(str(sample["replay_row_id"])),
    )

    adapter.validate_replay((row,))

    assert calls == ["M50:game:0:1"]


def test_resolved_fresh_sources_skip_historical_semantics_and_preserve_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = run_owned_training.Torus9TrainingAdapter(profile=_profile())
    paths: list[Path] = []
    evidence: list[dict[str, object]] = []
    for generation in (88, 89, 90):
        row = _row(f"M{generation}:game:0:0")
        row["source_generation"] = generation
        path = tmp_path / f"iter-{generation:02d}-fresh.jsonl"
        path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        digest = sha256_file(path)
        paths.append(path)
        evidence.append(
            {
                "path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "row_count": 1,
                "validation_schema": ARTIFACT_VALIDATION_SCHEMA,
                "immutable_verified": True,
                "generation_identity": {
                    "schema": TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA,
                    "generation": generation,
                    "sha256": digest,
                    "row_count": 1,
                },
            }
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resolved historical sources must not be semantically revalidated")

    monkeypatch.setattr(adapter, "validate_sample", forbidden)
    monkeypatch.setattr(run_owned_training._base, "value_fingerprint", forbidden)

    replay, _digests = adapter._rolling_from_sources(
        paths,
        total_evictions=0,
        replay_artifact_identities=evidence,
    )

    assert [row["source_generation"] for row in replay.rows] == [88, 89, 90]


def test_incomplete_resolved_source_evidence_uses_full_validation_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = run_owned_training.Torus9TrainingAdapter(profile=_profile())
    row = _row("M88:game:0:0")
    path = tmp_path / "iter-88-fresh.jsonl"
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256_file(path)
    calls: list[str] = []
    monkeypatch.setattr(
        adapter,
        "validate_sample",
        lambda sample: calls.append(str(sample["replay_row_id"])),
    )

    adapter._rolling_from_sources(
        [path],
        total_evictions=0,
        replay_artifact_identities=[
            {
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "validation_schema": ARTIFACT_VALIDATION_SCHEMA,
                "immutable_verified": True,
                # Missing row_count and generation identity is intentionally
                # incomplete and must not authorize the trusted path.
            }
        ],
    )

    assert calls == ["M88:game:0:0"]
