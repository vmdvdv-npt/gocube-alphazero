from __future__ import annotations

import json
from pathlib import Path

import pytest

from gocube_golden.run_spec import RUN_SPEC_SCHEMA, StrictRunSpec, run_spec_fingerprint
from tools.torus9_run_driver import _arena_config, _generation_config


def _write_profile(root: Path) -> tuple[str, str]:
    path = root / "configs" / "torus.json"
    path.parent.mkdir(parents=True)
    fingerprint = "sha256:test-profile"
    path.write_text(
        json.dumps(
            {
                "profile_id": "torus9-golden-current-v3",
                "profile_fingerprint": fingerprint,
                "self_play": {"games_per_iteration": 37},
            }
        ),
        encoding="utf-8",
    )
    return "configs/torus.json", fingerprint


def _payload(root: Path) -> dict[str, object]:
    profile_path, fingerprint = _write_profile(root)
    return {
        "schema": RUN_SPEC_SCHEMA,
        "topology": "torus9",
        "board_size": 9,
        "adapter": {"id": "torus9-test-adapter", "transport": "process"},
        "profile_path": profile_path,
        "expected_profile_fingerprint": fingerprint,
        "generation": {
            "command": ["python", "tools/torus9_run_driver.py", "generation", "--generation", "{generation}"],
            "resume_command": ["python", "tools/torus9_run_driver.py", "generation", "--generation", "{generation}", "--resume"],
            "driver_config": {
                "games": 37,
                "device": "cpu",
                "workers": 3,
                "active_games_per_worker": 7,
                "total_active_contexts": 17,
                "inference_batch_cap": 11,
                "inference_batch_wait_ms": 2.75,
                "coalescing": False,
                "heartbeat_interval_seconds": 23,
                "model_init_seed": 101,
                "selfplay_master_seed": 202,
                "training_master_seed": 303,
            },
        },
        "arena": {
            "enabled": True,
            "required": True,
            "every_generations": 3,
            "command": ["python", "tools/torus9_run_driver.py", "arena", "--generation", "{generation}"],
            "driver_config": {
                "comparison_mode": "candidate-vs-prior-generation",
                "reference_gap": 2,
                "games": 18,
                "master_seed": 404,
                "heartbeat_interval_seconds": 29,
                "execution": {
                    "workers": 3,
                    "games_per_worker": 5,
                    "inference_batch_rows": 13,
                    "inference_batch_wait_ms": 6.5,
                    "device": "cpu",
                    "strict_production": False,
                    "min_mean_inference_batch_rows": 0.0,
                    "min_effective_cpu_cores": 0.0,
                    "early_gate_enabled": False,
                    "early_gate_min_forwards": 8,
                    "early_gate_min_wall_sec": 0.0,
                },
            },
            "startset": {
                "schema": "test-startset-v1",
                "generator": "test",
                "master_seed": 404,
                "pairs": 9,
            },
        },
        "health": {
            "poll_seconds": 7,
            "heartbeat_warning_seconds": 41,
            "heartbeat_critical_seconds": 83,
            "min_disk_free_gb_warning": 9,
            "min_disk_free_gb_critical": 4,
            "min_ram_free_gb_warning": 1.5,
            "min_ram_free_gb_critical": 0.4,
        },
        "supervision": {
            "startup_ack_timeout_seconds": 11,
            "progress_warning_seconds": 47,
            "progress_critical_seconds": 97,
            "critical_child_grace_seconds": 3,
            "restart_backoff_seconds": 2,
            "max_generation_restarts": 1,
        },
        "soft_stop": {
            "default_minutes": 47,
            "minimum_minutes": 13,
            "maximum_minutes": 91,
        },
        "performance": {
            "checks": [
                {
                    "metric": "moves_per_sec",
                    "baseline": 17.25,
                    "warning_ratio": 0.81,
                    "fail_ratio": 0.62,
                    "policy": "warning",
                }
            ]
        },
        "learning": {
            "metrics": ["loss.total"],
            "stall_checks": [
                {
                    "kind": "generation",
                    "metric": "learning.samples_consumed_total",
                    "window": 4,
                    "minimum_delta": 123,
                    "direction": "increase",
                    "policy": "warning",
                }
            ],
        },
        "required_generation_metrics": ["moves_per_sec"],
        "required_arena_metrics": ["games"],
    }


def _load(tmp_path: Path, payload: dict[str, object]) -> StrictRunSpec:
    spec_path = tmp_path / "one-shot.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    return StrictRunSpec.load(spec_path, repo_root=tmp_path)


def test_run_spec_accepts_arbitrary_explicit_cadence_and_execution(tmp_path: Path):
    payload = _payload(tmp_path)
    spec = _load(tmp_path, payload)
    assert spec.orchestrator_spec.arena_every_generations == 3
    assert spec.board_size == 9
    assert spec.adapter_id == "torus9-test-adapter"
    assert spec.orchestrator_spec.health.poll_seconds == 7
    assert spec.supervision.progress_critical_seconds == 97
    assert spec.orchestrator_spec.soft_stop.default_minutes == 47
    assert spec.orchestrator_spec.performance["checks"][0]["baseline"] == 17.25
    assert spec.fingerprint == run_spec_fingerprint(payload)
    generation = _generation_config(spec)
    assert generation["workers"] == 3
    assert generation["inference_batch_cap"] == 11
    assert generation["inference_batch_wait_ms"] == 2.75
    assert generation["coalescing"] is False
    arena, startset = _arena_config(spec)
    assert arena["reference_gap"] == 2
    assert arena["games"] == 18
    assert arena["execution"]["workers"] == 3
    assert arena["execution"]["inference_batch_wait_ms"] == 6.5
    assert arena["execution"]["early_gate_enabled"] is False
    assert startset["pairs"] == 9


def test_run_spec_has_no_health_defaults(tmp_path: Path):
    payload = _payload(tmp_path)
    del payload["health"]["poll_seconds"]  # type: ignore[index]
    with pytest.raises(ValueError, match="health is missing explicit fields: poll_seconds"):
        _load(tmp_path, payload)


def test_run_spec_has_no_supervision_defaults(tmp_path: Path):
    payload = _payload(tmp_path)
    del payload["supervision"]["progress_critical_seconds"]  # type: ignore[index]
    with pytest.raises(ValueError, match="supervision is missing explicit fields"):
        _load(tmp_path, payload)


def test_run_spec_has_no_soft_stop_defaults(tmp_path: Path):
    payload = _payload(tmp_path)
    del payload["soft_stop"]["default_minutes"]  # type: ignore[index]
    with pytest.raises(ValueError, match="soft_stop is missing explicit fields"):
        _load(tmp_path, payload)


def test_run_spec_has_no_performance_threshold_defaults(tmp_path: Path):
    payload = _payload(tmp_path)
    del payload["performance"]["checks"][0]["fail_ratio"]  # type: ignore[index]
    with pytest.raises(ValueError, match="fail_ratio"):
        _load(tmp_path, payload)


def test_arena_cadence_is_run_owned_not_hard_limited(tmp_path: Path):
    payload = _payload(tmp_path)
    payload["arena"]["every_generations"] = 1  # type: ignore[index]
    assert _load(tmp_path, payload).orchestrator_spec.arena_every_generations == 1
    payload["arena"]["every_generations"] = 27  # type: ignore[index]
    assert _load(tmp_path, payload).orchestrator_spec.arena_every_generations == 27


def test_arena_startset_must_match_explicit_workload(tmp_path: Path):
    payload = _payload(tmp_path)
    payload["arena"]["startset"]["pairs"] = 8  # type: ignore[index]
    spec = _load(tmp_path, payload)
    with pytest.raises(ValueError, match="pairs must equal"):
        _arena_config(spec)


def test_generation_contexts_must_fit_explicit_lane_capacity(tmp_path: Path):
    payload = _payload(tmp_path)
    payload["generation"]["driver_config"]["total_active_contexts"] = 22  # type: ignore[index]
    spec = _load(tmp_path, payload)
    with pytest.raises(ValueError, match="lane capacity"):
        _generation_config(spec)


def test_arena_critical_execution_gates_are_explicit(tmp_path: Path):
    payload = _payload(tmp_path)
    del payload["arena"]["driver_config"]["execution"]["early_gate_min_forwards"]  # type: ignore[index]
    spec = _load(tmp_path, payload)
    with pytest.raises(ValueError, match="early_gate_min_forwards"):
        _arena_config(spec)
