from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.hardware_telemetry as hardware_telemetry
import tools.torus9_run_driver as run_driver
from gocube_golden.artifact_catalog import ArtifactCatalog, sha256_file
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_PROFILE_FINGERPRINT,
    load_torus9_current_profile,
)


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROFILE = ROOT / "configs/gocube/torus9_golden_current_v3.json"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("rules", "ko", "simple-ko"),
        ("self_play", "root_noise", False),
        ("target", "ownership_auxiliary", False),
        ("rules", "komi", 1.0),
        ("replay", "cap", 19_999),
        ("training", "optimizer_steps_per_iteration", 79),
    ),
)
def test_driver_rejects_tampered_golden_payload_with_stale_fingerprint(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    payload = json.loads(CURRENT_PROFILE.read_text(encoding="utf-8"))
    payload[section][field] = value
    candidate = tmp_path / "tampered-profile.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        run_driver._load_profile(candidate, TORUS9_CURRENT_PROFILE_FINGERPRINT)


def test_driver_accepts_only_the_canonical_golden_payload() -> None:
    profile = run_driver._load_profile(CURRENT_PROFILE, TORUS9_CURRENT_PROFILE_FINGERPRINT)
    assert profile == load_torus9_current_profile()


def test_semantic_heartbeat_publishes_incremental_counts() -> None:
    path = Path("/tmp") / f"torus9-heartbeat-test-{id(object())}.json"
    try:
        with run_driver._Heartbeat(path, generation=3, interval=60.0) as heartbeat:
            heartbeat.advance(
                "self-play",
                completed=7,
                total=64,
                unit="games",
                subphase="games",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["progress"] == {
                "completed": 7,
                "total": 64,
                "unit": "games",
                "subphase": "games",
            }
            assert payload["completed"] == 7
            assert payload["total"] == 64
            assert payload["progress_token"] == "self-play:7/64 games"
    finally:
        path.unlink(missing_ok=True)


def test_nvidia_smi_env_override_and_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / "nvidia-smi-test"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("GOCUBE_NVIDIA_SMI", str(binary))
    monkeypatch.setattr(hardware_telemetry.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        hardware_telemetry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="0, NVIDIA Test, 73, 100, 900, 1000, 61, 120\n",
            stderr="",
        ),
    )

    rows, status = hardware_telemetry._query_nvidia_smi()
    assert status["status"] == "ok"
    assert status["resolver"] == "env_override"
    assert rows == [
        {
            "index": 0,
            "name": "NVIDIA Test",
            "gpu_util_percent": 73.0,
            "gpu_memory_used_mib": 100.0,
            "gpu_memory_free_mib": 900.0,
            "gpu_memory_total_mib": 1000.0,
            "gpu_temperature_c": 61.0,
            "gpu_power_w": 120.0,
        }
    ]


def test_nvidia_smi_wsl_fallback_and_explicit_statuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "wsl-nvidia-smi"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.delenv("GOCUBE_NVIDIA_SMI", raising=False)
    monkeypatch.setattr(hardware_telemetry.shutil, "which", lambda _name: None)
    monkeypatch.setattr(hardware_telemetry, "WSL_NVIDIA_SMI_PATH", str(binary))
    monkeypatch.setattr(
        hardware_telemetry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    rows, status = hardware_telemetry._query_nvidia_smi()
    assert rows == []
    assert status["status"] == "gpu_not_found"
    assert status["resolver"] == "wsl_fallback"

    monkeypatch.setattr(hardware_telemetry, "WSL_NVIDIA_SMI_PATH", str(tmp_path / "missing"))
    _rows, missing = hardware_telemetry._query_nvidia_smi()
    assert missing["status"] == "nvidia_smi_missing"


def test_nvidia_smi_malformed_row_does_not_break_sampler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "nvidia-smi-test"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("GOCUBE_NVIDIA_SMI", str(binary))
    monkeypatch.setattr(
        hardware_telemetry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="bad row\n0, GPU, 1, 2, 3, 4, 5, 6\n",
            stderr="",
        ),
    )
    rows, status = hardware_telemetry._query_nvidia_smi()
    assert len(rows) == 1
    assert status["status"] == "parse_failed"


def test_artifact_catalog_verifies_selected_files_without_history_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "lineage"
    (root / "runtime").mkdir(parents=True)
    candidate = root / "checkpoints" / "M8.pt"
    reference = root / "checkpoints" / "M7.pt"
    ancient = root / "replay" / "rolling-after-01.jsonl"
    for path, content in (
        (candidate, b"candidate"),
        (reference, b"reference"),
        (ancient, b"ancient"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    catalog = ArtifactCatalog.initialize(
        root / "runtime" / "artifact-catalog.json",
        lineage_id="test-lineage",
        root=root,
    )
    catalog.register_generation(
        8,
        [
            {"path": "checkpoints/M8.pt", "sha256": sha256_file(candidate), "size_bytes": candidate.stat().st_size},
            {"path": "checkpoints/M7.pt", "sha256": sha256_file(reference), "size_bytes": reference.stat().st_size},
        ],
    )

    def fail_if_walk(*_args, **_kwargs):
        raise AssertionError("historical recursive scan is forbidden")

    monkeypatch.setattr(Path, "rglob", fail_if_walk)
    snapshot = run_driver._training_snapshot(
        root,
        generation=8,
        tracked_paths=(candidate, reference),
    )
    assert "checkpoints/M8.pt" in snapshot
    assert "replay/rolling-after-01.jsonl" not in snapshot
    assert catalog.verify(("checkpoints/M8.pt", "checkpoints/M7.pt"))

    candidate.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="(?:size|hash) mismatch"):
        catalog.verify(("checkpoints/M8.pt",))
