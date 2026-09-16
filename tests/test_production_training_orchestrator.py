from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import gocube_golden.orchestrator as legacy_orchestrator
import gocube_golden.run_storage as run_storage
from gocube_golden.production_orchestrator import CriticalHealthError
from gocube_golden.run_spec import (
    RUN_SPEC_SCHEMA,
    StrictProductionTrainingOrchestrator,
    StrictRunSpec,
)


FAKE_DRIVER = r'''
import datetime, hashlib, json, os, pathlib, sys, time
root = pathlib.Path(os.environ["AZ_RUN_ROOT"])
g = int(os.environ["AZ_GENERATION"])
kind = sys.argv[1]
behavior = sys.argv[2] if len(sys.argv) > 2 else "normal"

def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
def fp(v):
    return "sha256:" + hashlib.sha256(canon(v).encode()).hexdigest()
def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
def heartbeat(progress_at=None, token=None):
    path = pathlib.Path(os.environ["AZ_DRIVER_HEARTBEAT_PATH"])
    payload = {
        "schema":"gocube-training-driver-heartbeat-v2",
        "liveness_at":time.time(),
        "progress_at":time.time() if progress_at is None else progress_at,
        "progress_token":token or f"{kind}-{g}",
        "pid":os.getpid(),
        "generation":g,
        "phase":kind,
    }
    write(path, json.dumps(payload))

heartbeat()
if behavior == "stall":
    stale = time.time() - 10.0
    while True:
        heartbeat(progress_at=stale, token="stuck")
        time.sleep(0.02)
if behavior == "crash-once" and os.environ.get("AZ_RESUME") != "1":
    raise SystemExit(9)

if kind == "arena":
    spec = json.loads(pathlib.Path(os.environ["AZ_RUN_SPEC_PATH"]).read_text())
    arena = spec["arena"]
    path = pathlib.Path(os.environ["AZ_ARENA_RESULT_PATH"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema":"gocube-arena-driver-result-v1",
        "generation":g,
        "status":"COMPLETED",
        "profile_fingerprint":os.environ["AZ_PROFILE_FINGERPRINT"],
        "technical_games":0,
        "invalid_games":0,
        "training_mutated":False,
        "preset_fingerprint":fp(arena["driver_config"]),
        "startset_fingerprint":fp(arena["startset"]),
        "metrics":{"games":8,"win_rate":0.5,"games_per_hour":800.0},
    }
    write(path, json.dumps(payload))
    heartbeat(token=f"arena-{g}-complete")
    raise SystemExit(0)

checkpoint = root / "checkpoints" / f"M{g}.pt"
replay = root / "data" / f"replay-{g:04d}.jsonl"
resume = root / "runtime" / "resume" / f"generation-{g:04d}.json"
write(checkpoint, f"checkpoint-{g}-{os.environ.get('AZ_RESUME')}\n")
write(replay, json.dumps({"generation":g}) + "\n")
write(resume, json.dumps({"generation":g,"optimizer":g,"rng":"explicit"}))
arts=[]
for path in (checkpoint,replay,resume):
    arts.append({"path":str(path.relative_to(root)),"sha256":digest(path),"size_bytes":path.stat().st_size})
result={
    "schema":"gocube-generation-driver-result-v1",
    "generation":g,
    "status":"COMPLETED",
    "profile_fingerprint":os.environ["AZ_PROFILE_FINGERPRINT"],
    "checkpoint_reload_verified":True,
    "checkpoint":{"path":str(checkpoint.relative_to(root))},
    "replay":{"path":str(replay.relative_to(root))},
    "resume_state":{"path":str(resume.relative_to(root)),"components":["model","optimizer","replay","generation","rng"]},
    "artifacts":arts,
    "technical_games":0,
    "invalid_games":0,
    "metrics":{
        "games_per_hour":100.0 + g,
        "moves_per_sec":20.0 + g,
        "optimizer_updates_per_sec":3.0,
        "learning":{"samples_consumed_total":100*g},
        "loss":{"total":1.0/g},
    },
}
write(pathlib.Path(os.environ["AZ_GENERATION_RESULT_PATH"]), json.dumps(result))
heartbeat(token=f"generation-{g}-complete")
if behavior == "request-stop" and g == 2:
    now = datetime.datetime.now(datetime.timezone.utc)
    stop = {
        "schema":"gocube-production-training-orchestrator-v1",
        "requested_at":now.isoformat(),
        "requested_by":"fake-driver-test",
        "target_deadline_at":(now + datetime.timedelta(minutes=30)).isoformat(),
        "window_minutes":30,
        "mode":"finish-current-safe-boundary-no-hard-kill",
    }
    write(pathlib.Path(os.environ["AZ_SOFT_STOP_REQUEST_PATH"]), json.dumps(stop))
'''


def _payload(tmp_path: Path, *, behavior: str = "normal", arena_every: int = 2, restarts: int = 1) -> dict[str, object]:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({"profile_id":"fake-cube4","profile_fingerprint":"sha256:fake-cube-profile"}),
        encoding="utf-8",
    )
    driver = tmp_path / "fake_driver.py"
    driver.write_text(FAKE_DRIVER, encoding="utf-8")
    return {
        "schema": RUN_SPEC_SCHEMA,
        "topology": "cube4",
        "board_size": 4,
        "adapter": {"id":"fake-cube-adapter","transport":"process"},
        "profile_path": "profile.json",
        "expected_profile_fingerprint": "sha256:fake-cube-profile",
        "generation": {
            "command": [sys.executable,"fake_driver.py","generation",behavior],
            "resume_command": [sys.executable,"fake_driver.py","generation","normal"],
            "driver_config": {"games":4},
        },
        "arena": {
            "enabled": True,
            "required": True,
            "every_generations": arena_every,
            "command": [sys.executable,"fake_driver.py","arena","normal"],
            "driver_config": {"mode":"fake","games":8},
            "startset": {"seed":123,"pairs":4},
        },
        "health": {
            "poll_seconds":0.01,
            "heartbeat_warning_seconds":5.0,
            "heartbeat_critical_seconds":10.0,
            "min_disk_free_gb_warning":0.0,
            "min_disk_free_gb_critical":0.0,
            "min_ram_free_gb_warning":0.0,
            "min_ram_free_gb_critical":0.0,
        },
        "supervision": {
            "startup_ack_timeout_seconds":1.0,
            "progress_warning_seconds":1.0,
            "progress_critical_seconds":5.0,
            "critical_child_grace_seconds":0.05,
            "restart_backoff_seconds":0.0,
            "max_generation_restarts":restarts,
        },
        "soft_stop": {"default_minutes":30,"minimum_minutes":30,"maximum_minutes":100},
        "performance": {"checks":[]},
        "learning": {"metrics":["loss.total"],"stall_checks":[]},
        "required_generation_metrics":["moves_per_sec","learning.samples_consumed_total"],
        "required_arena_metrics":["games"],
    }


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, behavior: str = "normal", arena_every: int = 2, restarts: int = 1):
    monkeypatch.setattr(run_storage, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(legacy_orchestrator, "_git_sha", lambda _root: "deadbeef")
    payload = _payload(tmp_path, behavior=behavior, arena_every=arena_every, restarts=restarts)
    spec_path = tmp_path / "run-spec.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    spec = StrictRunSpec.load(spec_path, repo_root=tmp_path)
    run = StrictProductionTrainingOrchestrator(
        repo_root=tmp_path,
        run_spec=spec,
        lineage_id="fake-cube-lineage",
        terminal=False,
    )
    return run


def test_fake_cube_runs_through_same_generic_supervisor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = _run(tmp_path, monkeypatch)
    run.create()
    run.run(max_generations=3)
    status = run.status()
    assert status["state"] == "COMPLETED"
    assert status["topology"] == "cube4"
    assert status["board_size"] == 4
    assert status["adapter_id"] == "fake-cube-adapter"
    assert status["last_committed_generation"] == 3
    assert status["arena_generations"] == [2]
    assert (run.paths.root / "arena" / "generation-0002" / "result.json").is_file()


def test_soft_stop_after_generation_does_not_start_new_arena(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = _run(tmp_path, monkeypatch, behavior="request-stop", arena_every=2)
    run.create()
    run.run(max_generations=4)
    status = run.status()
    assert status["state"] == "SOFT_STOPPED"
    assert status["last_committed_generation"] == 2
    assert status["arena_generations"] == []
    assert not (run.paths.root / "arena" / "generation-0002" / "result.json").exists()


def test_nonzero_child_gets_bounded_automatic_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = _run(tmp_path, monkeypatch, behavior="crash-once", arena_every=9, restarts=1)
    run.create()
    run.run(max_generations=1)
    assert run.status()["state"] == "COMPLETED"
    tx = json.loads(run._generation_tx_path(1).read_text())
    assert tx["restart_attempts"] == 1
    events = run.paths.events.read_text()
    assert "bounded automatic resume scheduled" in events


def test_resume_finishes_commit_after_transaction_before_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run = _run(tmp_path, monkeypatch, arena_every=9)
    run.create()
    run._write_state(state="RUNNING", pid=12345, active_phase="startup")
    original_record = run._record_generation_artifacts

    def stop_before_catalog(*_args, **_kwargs):
        raise RuntimeError("injected stop before catalog write")

    monkeypatch.setattr(run, "_record_generation_artifacts", stop_before_catalog)
    with pytest.raises(RuntimeError, match="injected stop before catalog write"):
        run._run_generation(1)
    monkeypatch.setattr(run, "_record_generation_artifacts", original_record)

    transaction = json.loads(run._generation_tx_path(1).read_text(encoding="utf-8"))
    assert transaction["status"] == "COMMITTED"
    assert json.loads(run.paths.manifest.read_text(encoding="utf-8"))["orchestrator"][
        "last_committed_generation"
    ] == 0

    run.prepare_resume()
    status = run.status()
    assert status["state"] == "CREATED"
    assert status["last_committed_generation"] == 1
    assert json.loads(run.paths.artifact_catalog.read_text(encoding="utf-8"))["generations"]["1"]
    run.run(max_generations=2)
    assert run.status()["last_committed_generation"] == 2


def test_resume_finishes_commit_after_catalog_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run = _run(tmp_path, monkeypatch, arena_every=9)
    run.create()
    run._write_state(state="RUNNING", pid=12345, active_phase="startup")
    original_update = run._update_manifest

    def stop_before_manifest(*_args, **_kwargs):
        raise RuntimeError("injected stop before manifest write")

    monkeypatch.setattr(run, "_update_manifest", stop_before_manifest)
    with pytest.raises(RuntimeError, match="injected stop before manifest write"):
        run._run_generation(1)
    monkeypatch.setattr(run, "_update_manifest", original_update)

    transaction = json.loads(run._generation_tx_path(1).read_text(encoding="utf-8"))
    catalog = json.loads(run.paths.artifact_catalog.read_text(encoding="utf-8"))
    manifest = json.loads(run.paths.manifest.read_text(encoding="utf-8"))
    assert transaction["status"] == "COMMITTED"
    assert "1" in catalog["generations"]
    assert manifest["orchestrator"]["last_committed_generation"] == 0

    run.prepare_resume()
    status = run.status()
    assert status["state"] == "CREATED"
    assert status["last_committed_generation"] == 1
    recovered_manifest = json.loads(run.paths.manifest.read_text(encoding="utf-8"))
    assert (
        recovered_manifest["orchestrator"]["artifact_catalog"]["fingerprint"]
        == catalog["catalog_fingerprint"]
    )


def test_resume_refuses_live_supervisor_before_commit_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run = _run(tmp_path, monkeypatch, arena_every=9)
    run.create()
    run._write_state(state="RUNNING", pid=12345, active_phase="startup")
    original_update = run._update_manifest

    def stop_before_manifest(*_args, **_kwargs):
        raise RuntimeError("injected stop before manifest write")

    monkeypatch.setattr(run, "_update_manifest", stop_before_manifest)
    with pytest.raises(RuntimeError, match="injected stop before manifest write"):
        run._run_generation(1)
    monkeypatch.setattr(run, "_update_manifest", original_update)

    before_state = json.loads(run.paths.runtime_state.read_text(encoding="utf-8"))
    before_manifest = json.loads(run.paths.manifest.read_text(encoding="utf-8"))
    assert before_state["state"] == "RUNNING"
    assert before_manifest["orchestrator"]["last_committed_generation"] == 0

    with legacy_orchestrator.RunLock(run.paths.lock):
        with pytest.raises(RuntimeError, match="active orchestrator lock"):
            run.prepare_resume()

    after_state = json.loads(run.paths.runtime_state.read_text(encoding="utf-8"))
    after_manifest = json.loads(run.paths.manifest.read_text(encoding="utf-8"))
    assert after_state["state"] == "RUNNING"
    assert after_manifest["orchestrator"]["last_committed_generation"] == 0


def test_stale_progress_is_fail_closed_and_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = _run(tmp_path, monkeypatch, behavior="stall", arena_every=9, restarts=0)
    run.create()
    with pytest.raises(CriticalHealthError, match="no observable progress"):
        run.run(max_generations=1)
    status = run.status()
    assert status["state"] == "RECOVERY_REQUIRED"
    assert status["health"] == "CRITICAL"
    assert not run.active_child_path.exists()


def test_generic_supervisor_source_contains_no_game_specific_imports():
    source = (Path(__file__).resolve().parents[1] / "gocube_golden" / "production_orchestrator.py").read_text(encoding="utf-8").lower()
    assert "torus9" not in source
    assert "cube4" not in source
    assert "arena_profiles" not in source


def test_active_torus_driver_does_not_import_or_monkeypatch_legacy_driver():
    source = (Path(__file__).resolve().parents[1] / "tools" / "torus9_run_driver.py").read_text(encoding="utf-8")
    assert "torus9_orchestrator_driver" not in source
    assert "monkeypatch" not in source.lower()
    assert "PERIODIC_ARENA_PRESET" not in source
    assert "LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE" not in source
    assert 'root / "arena" / f"generation-{args.generation:04d}"' in source
