from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

import gocube_golden.run_storage as run_storage
import gocube_golden.orchestrator as orch
from gocube_golden.orchestrator import OrchestratorSpec, ProductionTrainingOrchestrator


FAKE_DRIVER = r'''
import hashlib, json, os, pathlib, sys, time
root = pathlib.Path(os.environ["AZ_RUN_ROOT"])
g = int(os.environ["AZ_GENERATION"])
mode = sys.argv[1] if len(sys.argv) > 1 else "generation"

def digest(path):
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:"+h

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")

hb=pathlib.Path(os.environ["AZ_DRIVER_HEARTBEAT_PATH"])
hb.parent.mkdir(parents=True, exist_ok=True)
hb.write_text(json.dumps({"at": time.time(), "generation": g}), encoding="utf-8")

if mode == "exit-fail":
    raise SystemExit(9)
if mode == "arena":
    path=pathlib.Path(os.environ["AZ_ARENA_RESULT_PATH"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload={
      "schema":"gocube-arena-driver-result-v1","generation":g,"status":"COMPLETED",
      "profile_fingerprint":os.environ["AZ_PROFILE_FINGERPRINT"],"technical_games":0,"invalid_games":0,
      "training_mutated":False,"preset_fingerprint":"arena-preset-test","startset_fingerprint":"arena-startset-test",
      "metrics":{"win_rate":0.6,"games":64}
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    raise SystemExit(0)

checkpoint=root/"checkpoints"/f"M{g}.pt"
replay=root/"data"/f"replay-{g:04d}.jsonl"
resume=root/"runtime"/"resume"/f"generation-{g:04d}.json"
write(checkpoint, f"checkpoint-{g}-{os.environ.get('AZ_RESUME')}\n")
write(replay, json.dumps({"generation":g})+"\n")
write(resume, json.dumps({"generation":g,"rng":"deterministic-test","optimizer":g,"replay":str(replay)}))
arts=[]
for p in (checkpoint,replay,resume):
    arts.append({"path":str(p.relative_to(root)),"sha256":digest(p),"size_bytes":p.stat().st_size})
result={
 "schema":"gocube-generation-driver-result-v1","generation":g,"status":"COMPLETED",
 "profile_fingerprint":os.environ["AZ_PROFILE_FINGERPRINT"],
 "checkpoint_reload_verified":True,
 "checkpoint":{"path":str(checkpoint.relative_to(root))},
 "replay":{"path":str(replay.relative_to(root))},
 "resume_state":{"path":str(resume.relative_to(root)),"components":["model","optimizer","replay","generation","rng"]},
 "artifacts":arts,"technical_games":0,"invalid_games":0,
 "metrics":{"games_per_hour":100.0+g,"moves_per_sec":20.0+g,"loss":{"policy":1.0/g,"value":0.5/g}}
}
if mode == "technical": result["technical_games"]=1
if mode == "bad-hash": result["artifacts"][0]["sha256"]="sha256:"+"0"*64
path=pathlib.Path(os.environ["AZ_GENERATION_RESULT_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result), encoding="utf-8")
'''


def _write_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, generation_mode: str = "generation", arena_every: int = 5, fail_closed_perf: bool = False):
    monkeypatch.setattr(run_storage, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(orch, "_git_sha", lambda _root: "deadbeef")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"profile_id":"test-profile","profile_fingerprint":"sha256:test-profile"}), encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(FAKE_DRIVER, encoding="utf-8")
    performance = {
        "checks":[{"metric":"games_per_hour","baseline":100.0,"warning_ratio":0.8,"fail_ratio":0.5,"policy":"fail-closed" if fail_closed_perf else "warning"}]
    }
    spec_payload = {
      "schema":"gocube-production-training-orchestrator-v1",
      "topology":"test-topology",
      "profile_path":"profile.json",
      "expected_profile_fingerprint":"sha256:test-profile",
      "execution":{
        "generation_command":[sys.executable,"driver.py",generation_mode],
        "generation_resume_command":[sys.executable,"driver.py",generation_mode]
      },
      "arena":{
        "every_generations":arena_every,"required":True,
        "command":[sys.executable,"driver.py","arena"],
        "preset_fingerprint":"arena-preset-test","startset_fingerprint":"arena-startset-test"
      },
      "health":{
        "poll_seconds":0.01,"heartbeat_warning_seconds":5,"heartbeat_critical_seconds":10,
        "min_disk_free_gb_warning":0,"min_disk_free_gb_critical":0,
        "min_ram_free_gb_warning":0,"min_ram_free_gb_critical":0
      },
      "soft_stop":{"default_minutes":60,"minimum_minutes":30,"maximum_minutes":100},
      "performance":performance,
      "learning":{"metrics":["loss.policy","loss.value","moves_per_sec"]}
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_payload), encoding="utf-8")
    spec=OrchestratorSpec.load(spec_path, repo_root=tmp_path)
    run=ProductionTrainingOrchestrator(repo_root=tmp_path,spec=spec,lineage_id="lineage-test",terminal=False)
    return run, spec_path


def test_five_generations_commit_and_arena(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch)
    run.create()
    run.run(max_generations=5)
    status=run.status()
    assert status["state"] == "COMPLETED"
    assert status["last_committed_generation"] == 5
    manifest=json.loads(run.paths.manifest.read_text())
    assert manifest["orchestrator"]["arena_generations"] == [5]
    assert "checkpoints/M5.pt" in manifest["checkpoint_hashes"]
    assert run.paths.final_json.is_file()
    report=run.paths.report_md.read_text()
    assert "Learning velocity" in report
    assert "committed generation: **5**" in report


def test_low_arena_batch_does_not_override_technical_fail_closed(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch)
    run.create()
    result_path = run._arena_result_path(5)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "schema": "gocube-arena-driver-result-v1",
                "generation": 5,
                "status": "COMPLETED",
                "profile_fingerprint": run.spec.profile_fingerprint,
                "technical_games": 1,
                "invalid_games": 0,
                "training_mutated": False,
                "preset_fingerprint": run.spec.arena_preset_fingerprint,
                "startset_fingerprint": run.spec.arena_startset_fingerprint,
                "metrics": {
                    "inference_mean_batch_rows": 7.0,
                    "performance_status": "SEVERE_WARNING",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="technical/invalid"):
        run._validate_arena_result(5)


def test_soft_stop_is_durable_and_bounded(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch)
    run.create()
    request=run.request_soft_stop(30)
    assert request["window_minutes"] == 30
    with pytest.raises(ValueError):
        run.request_soft_stop(29)
    with pytest.raises(ValueError):
        run.request_soft_stop(101)
    run.run(max_generations=5)
    assert run.status()["state"] == "SOFT_STOPPED"
    assert run.status()["last_committed_generation"] == 0


def test_interrupted_generation_uses_resume_command(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch)
    run.create()
    tx=run._generation_tx_path(1)
    tx.parent.mkdir(parents=True, exist_ok=True)
    tx.write_text(json.dumps({"generation":1,"status":"RUNNING"}), encoding="utf-8")
    run.run(max_generations=1)
    result=json.loads(run._generation_result_path(1).read_text())
    checkpoint=run.paths.root/result["checkpoint"]["path"]
    assert checkpoint.read_text().strip().endswith("-1")
    events=run.paths.events.read_text()
    assert "Resuming interrupted generation" in events


def test_technical_generation_fails_closed_without_commit(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch, generation_mode="technical")
    run.create()
    with pytest.raises(ValueError, match="technical/invalid"):
        run.run(max_generations=1)
    status=run.status()
    assert status["state"] == "RECOVERY_REQUIRED"
    assert status["last_committed_generation"] == 0


def test_corrupt_artifact_hash_fails_closed(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch, generation_mode="bad-hash")
    run.create()
    with pytest.raises(ValueError, match="hash mismatch"):
        run.run(max_generations=1)
    assert run.status()["state"] == "RECOVERY_REQUIRED"


def test_killed_driver_fails_closed(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch, generation_mode="exit-fail")
    run.create()
    with pytest.raises(RuntimeError, match="exited with code 9"):
        run.run(max_generations=1)
    assert run.status()["state"] == "RECOVERY_REQUIRED"


def test_critical_health_requests_soft_stop(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch)
    run.create()
    run._emit_health_warnings({"disk_free_gb":0.0,"ram_free_gb":10.0,"driver_heartbeat_age_sec":0.0})
    assert run.paths.stop_request.is_file()


def test_spec_rejects_arena_cadence_outside_five_to_ten(tmp_path, monkeypatch):
    _, spec_path = _write_fixture(tmp_path, monkeypatch)
    payload=json.loads(spec_path.read_text())
    payload["arena"]["every_generations"]=4
    spec_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="between 5 and 10"):
        OrchestratorSpec.load(spec_path, repo_root=tmp_path)


def test_resume_rejects_config_drift(tmp_path, monkeypatch):
    run, spec_path = _write_fixture(tmp_path, monkeypatch)
    run.create()
    payload=json.loads(spec_path.read_text())
    payload["health"]["poll_seconds"]=0.02
    spec_path.write_text(json.dumps(payload))
    drift=OrchestratorSpec.load(spec_path, repo_root=tmp_path)
    run2=ProductionTrainingOrchestrator(repo_root=tmp_path,spec=drift,lineage_id="lineage-test",terminal=False)
    with pytest.raises(ValueError, match="fingerprint drift"):
        run2.status()


def test_soft_stopped_lineage_can_explicitly_resume(tmp_path, monkeypatch):
    run, _ = _write_fixture(tmp_path, monkeypatch)
    run.create()
    run.request_soft_stop(30)
    run.run(max_generations=2)
    assert run.status()["state"] == "SOFT_STOPPED"
    run.prepare_resume()
    run.run(max_generations=2)
    assert run.status()["state"] == "COMPLETED"
    assert run.status()["last_committed_generation"] == 2
