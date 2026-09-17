from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from tools.arena_engine import ArenaExecutionConfig
from tools.torus9_staged_sims_harness_impl import (
    EVALUATION_IDENTITY_FILENAME,
    _build_evaluation_identity,
    _evaluation_run_id,
    _existing_arena,
    _write_evaluation_identity,
)


CANDIDATE = {
    "lineage_id": "experiment-g128",
    "generation": 50,
    "artifact_sha256": "a" * 64,
}
REFERENCE = {
    "lineage_id": "experiment-g64",
    "generation": 53,
    "artifact_sha256": "b" * 64,
}
GAMES = 128
MASTER_SEED = 202609131004
WLD = [80, 48, 0]
SCIENTIFIC_CONTRACT = {
    "simulations": 64,
    "noise": False,
    "temperature": 0.0,
    "fast_search": False,
    "resign": False,
    "komi": 0.5,
    "paired_starts_color_swap": True,
    "deterministic_tie_break": True,
    "technical_fail_closed": True,
}
EXECUTION = ArenaExecutionConfig(
    games=GAMES,
    workers=16,
    games_per_worker=4,
    inference_batch_rows=64,
    inference_batch_wait_ms=1.0,
    device="cuda",
    strict_production=True,
    min_mean_inference_batch_rows=0.0,
    min_effective_cpu_cores=0.0,
    early_gate_enabled=False,
    early_gate_min_forwards=128,
    early_gate_min_wall_sec=5.0,
)


def _identity(
    *,
    candidate: dict[str, object] = CANDIDATE,
    reference: dict[str, object] = REFERENCE,
    master_seed: int = MASTER_SEED,
    scientific_contract: dict[str, object] = SCIENTIFIC_CONTRACT,
    execution: ArenaExecutionConfig = EXECUTION,
) -> tuple[dict[str, object], str, str]:
    payload, fingerprint = _build_evaluation_identity(
        candidate=candidate,
        reference=reference,
        profile="torus9",
        games=GAMES,
        master_seed=master_seed,
        scientific_contract=scientific_contract,
        execution=execution,
    )
    return payload, fingerprint, _evaluation_run_id(candidate, reference, fingerprint)


def _write_existing_arena(
    output: Path,
    identity: dict[str, object],
    fingerprint: str,
    run_id: str,
    telemetry: object,
) -> None:
    _write_evaluation_identity(output, run_id, identity, fingerprint)
    candidate = identity["candidate"]
    reference = identity["reference"]
    assert isinstance(candidate, dict)
    assert isinstance(reference, dict)
    (output / "summary.json").write_text(
        json.dumps({"games": GAMES, "W/L/D": WLD, "telemetry": telemetry}) + "\n",
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "candidate": {
                    "lineage_id": candidate["lineage_id"],
                    "generation": candidate["generation"],
                    "artifact_sha256": candidate["checkpoint_sha256"],
                },
                "reference": {
                    "lineage_id": reference["lineage_id"],
                    "generation": reference["generation"],
                    "artifact_sha256": reference["checkpoint_sha256"],
                },
                "profile": identity["profile"],
                "master_seed": identity["master_seed"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _healthy_telemetry(status: str = "HEALTHY") -> dict[str, object]:
    return {
        "technical_games": 0,
        "performance_status": status,
        "performance_failures": [],
    }


def test_exact_identity_is_deterministic_and_reusable(tmp_path: Path) -> None:
    identity, fingerprint, run_id = _identity()
    second_identity, second_fingerprint, second_run_id = _identity()

    assert identity == second_identity
    assert fingerprint == second_fingerprint
    assert len(fingerprint) == 64
    assert run_id == second_run_id
    assert run_id.endswith(fingerprint[:12])
    assert identity["execution"] == asdict(EXECUTION)
    assert identity["scientific_contract"] == SCIENTIFIC_CONTRACT

    output = tmp_path / run_id
    _write_existing_arena(output, identity, fingerprint, run_id, _healthy_telemetry())

    result = _existing_arena(output, identity, fingerprint)

    assert result is not None
    assert result["W/L/D"] == WLD


def test_seed_drift_changes_identity_and_evaluation_id() -> None:
    identity, fingerprint, run_id = _identity()
    changed_identity, changed_fingerprint, changed_run_id = _identity(
        master_seed=MASTER_SEED + 1
    )

    assert changed_identity != identity
    assert changed_fingerprint != fingerprint
    assert changed_run_id != run_id


def test_scientific_contract_drift_changes_identity_and_evaluation_id() -> None:
    changed_contract = dict(SCIENTIFIC_CONTRACT)
    changed_contract["simulations"] = 128

    _, fingerprint, run_id = _identity()
    _, changed_fingerprint, changed_run_id = _identity(
        scientific_contract=changed_contract
    )

    assert changed_fingerprint != fingerprint
    assert changed_run_id != run_id


def test_execution_contract_drift_changes_identity_and_evaluation_id() -> None:
    changed_execution = replace(EXECUTION, inference_batch_wait_ms=2.0)

    _, fingerprint, run_id = _identity()
    changed_identity, changed_fingerprint, changed_run_id = _identity(
        execution=changed_execution
    )

    assert changed_identity["execution"] == asdict(changed_execution)
    assert changed_fingerprint != fingerprint
    assert changed_run_id != run_id


def test_checkpoint_sha_drift_changes_identity_and_evaluation_id() -> None:
    changed_candidate = dict(CANDIDATE)
    changed_candidate["artifact_sha256"] = "c" * 64

    _, fingerprint, run_id = _identity()
    _, changed_fingerprint, changed_run_id = _identity(candidate=changed_candidate)

    assert changed_fingerprint != fingerprint
    assert changed_run_id != run_id


def test_tampered_identity_fingerprint_fails_closed(tmp_path: Path) -> None:
    identity, fingerprint, run_id = _identity()
    output = tmp_path / run_id
    _write_existing_arena(output, identity, fingerprint, run_id, _healthy_telemetry())

    identity_path = output / EVALUATION_IDENTITY_FILENAME
    record = json.loads(identity_path.read_text(encoding="utf-8"))
    record["fingerprint"] = "0" * 64
    identity_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="evaluation identity/contract mismatch"):
        _existing_arena(output, identity, fingerprint)


def test_tampered_identity_payload_fails_closed(tmp_path: Path) -> None:
    identity, fingerprint, run_id = _identity()
    output = tmp_path / run_id
    _write_existing_arena(output, identity, fingerprint, run_id, _healthy_telemetry())

    identity_path = output / EVALUATION_IDENTITY_FILENAME
    record = json.loads(identity_path.read_text(encoding="utf-8"))
    record["identity"]["master_seed"] = MASTER_SEED + 1
    identity_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="evaluation identity/contract mismatch"):
        _existing_arena(output, identity, fingerprint)


def test_legacy_evaluation_without_identity_is_not_reused_or_modified(
    tmp_path: Path,
) -> None:
    identity, fingerprint, run_id = _identity()
    output = tmp_path / run_id
    output.mkdir()
    summary = json.dumps(
        {"games": GAMES, "W/L/D": WLD, "telemetry": _healthy_telemetry()}
    ) + "\n"
    provenance = json.dumps(
        {
            "candidate": {"artifact_sha256": CANDIDATE["artifact_sha256"]},
            "reference": {"artifact_sha256": REFERENCE["artifact_sha256"]},
            "profile": "torus9",
            "master_seed": MASTER_SEED,
        }
    ) + "\n"
    (output / "summary.json").write_text(summary, encoding="utf-8")
    (output / "provenance.json").write_text(provenance, encoding="utf-8")

    with pytest.raises(RuntimeError, match="evaluation identity/contract mismatch"):
        _existing_arena(output, identity, fingerprint)

    assert (output / "summary.json").read_text(encoding="utf-8") == summary
    assert (output / "provenance.json").read_text(encoding="utf-8") == provenance
    assert not (output / EVALUATION_IDENTITY_FILENAME).exists()


def test_existing_arena_rejects_critical_restart_result_before_wld_reuse(
    tmp_path: Path,
) -> None:
    identity, fingerprint, run_id = _identity()
    output = tmp_path / run_id
    _write_existing_arena(
        output,
        identity,
        fingerprint,
        run_id,
        {
            "technical_games": 0,
            "performance_status": "CRITICAL",
            "performance_failures": ["lane_occupancy"],
        },
    )

    with pytest.raises(RuntimeError, match="production validity/performance gates"):
        _existing_arena(output, identity, fingerprint)

    persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert persisted["W/L/D"] == WLD
    assert persisted["telemetry"]["performance_status"] == "CRITICAL"


@pytest.mark.parametrize("status", ["WARNING", "SEVERE_WARNING"])
def test_existing_arena_reuses_warning_without_hard_failures(
    tmp_path: Path,
    status: str,
) -> None:
    identity, fingerprint, run_id = _identity()
    output = tmp_path / run_id
    _write_existing_arena(
        output,
        identity,
        fingerprint,
        run_id,
        _healthy_telemetry(status),
    )

    result = _existing_arena(output, identity, fingerprint)

    assert result is not None
    assert result["W/L/D"] == WLD


@pytest.mark.parametrize(
    "telemetry",
    [
        {"technical_games": 0, "performance_failures": []},
        {
            "technical_games": 0,
            "performance_status": None,
            "performance_failures": [],
        },
        {"technical_games": 0, "performance_status": "HEALTHY"},
        {
            "technical_games": 0,
            "performance_status": "HEALTHY",
            "performance_failures": {},
        },
    ],
)
def test_existing_arena_fails_closed_on_missing_or_malformed_performance_telemetry(
    tmp_path: Path,
    telemetry: object,
) -> None:
    identity, fingerprint, run_id = _identity()
    output = tmp_path / run_id
    _write_existing_arena(output, identity, fingerprint, run_id, telemetry)

    with pytest.raises(RuntimeError, match="production validity/performance gates"):
        _existing_arena(output, identity, fingerprint)
