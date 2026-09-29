import copy
import json
import os
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig
from gocube_golden.orchestrator_v2 import operator_job as job
from gocube_golden.orchestrator_v2 import production_entrypoint as entry
from gocube_golden.orchestrator_v2.execution_permit import PERMIT_ENV
from gocube_golden.orchestrator_v2.workflow import WorkflowRunner, WorkflowSpec


def parameters():
    return {"schema": job.SCHEMA, "run_id": "five-iterations", "parent": "source/update-2400",
            "training": {"iterations": 5, "learning_rate": 5e-5},
            "arena": {"every_iterations": 5, "games": 192, "mcts_simulations": 128},
            "ab_tests": [{"id": "lr", "iterations": 2, "A": {"learning_rate": 5e-5},
                          "B": {"learning_rate": 3e-5}}]}


@pytest.fixture
def parent(monkeypatch):
    cfg = EffectiveConfig(topology="torus9", compatibility={"input_channels": 5},
        self_play={"komi": 1.5}, training={"optimizer": "Adam", "batch_size": 64,
        "weight_decay": 0., "l2_sp": False, "gradient_clip": 1.},
        replay={"cap": None}, execution={"device": "cuda", "workers": 16},
        extensions={"training_driver": job.DRIVER})
    node = SimpleNamespace(ref=CheckpointRef("torus9", "source", "update-2400", 198,
                           "source.pt", "sha256:" + "a" * 64),
                           effective_config=SimpleNamespace(config=cfg))
    monkeypatch.setattr(job, "resolve_parent", lambda *a, **kw: node)
    return node


def test_strict_parameters_and_normalization_are_stable():
    normalized = job.parse_job(parameters())
    assert job.parse_job(normalized) == normalized
    assert normalized["training"]["games_per_iteration"] == 384
    assert normalized["ab_tests"][0]["arena"] == {"games": 192, "mcts_simulations": 128}


def test_gradient_clip_is_normalized_and_compiled_into_effective_config(parent):
    raw = parameters()
    raw["training"]["gradient_clip"] = 8.0
    normalized = job.parse_job(raw)
    assert normalized["training"]["gradient_clip"] == 8.0
    compiled = job.compile_job(raw, runs_root=".")
    config = compiled["workflow"]["steps"][0]["config"]["effective_config"]
    assert config["training"]["gradient_clip"] == 8.0
    assert config["training"]["optimizer"] == "Adam"


@pytest.mark.parametrize("patch", [
    {"notifications": False}, {"command": "arbitrary executable"},
    {"parent": "../../checkpoint"}, {"training": {"learning_rate": float("nan")}},
    {"training": {"iterations": True}}, {"training": {"batch_size": 32}},
    {"training": {"gradient_clip": 0}}, {"training": {"gradient_clip": float("nan")}},
    {"arena": {"games": 193}}, {"execution": {"workers": 1}},
    {"ab_tests": [{"id": "test", "iterations": 1, "A": {}, "B": {}, "arena": {"every_iterations": 2}}]},
    {"ab_tests": [{"id": "test", "iterations": 1, "A": {}, "B": {"typo": 1}}]},
])
def test_bad_parameters_rejected_before_parent_resolution(monkeypatch, patch):
    monkeypatch.setattr(job, "resolve_parent", lambda *a, **kw: pytest.fail("resolved before validation"))
    with pytest.raises(ValueError):
        job.compile_job({**parameters(), **patch})


def test_compiled_workflow_runs_ab_from_main_result_and_resume_does_not_repeat(parent, tmp_path):
    spec = WorkflowSpec.from_dict(job.compile_job(parameters(), runs_root=tmp_path)["workflow"])
    calls = []
    child = {**parent.ref.to_dict(), "checkpoint_id": "generation-203", "generation": 203}
    def training(*, config, **_):
        calls.append("training")
        assert config["generations"] == config["arena_cadence"] == config["arena_reference_gap"] == 5
        assert config["arena_config"]["games"] == 192
        assert "simulations=128" in config["arena_profile"]
        assert config["effective_config"]["training"]["learning_rate"] == 5e-5
        return {"checkpoint": child}
    def experiment(*, config, **_):
        calls.append("experiment")
        assert config["parent"] == child
        assert [a["generations"] for a in config["arms"]] == [2, 2]
        assert [a["config"]["training"]["learning_rate"] for a in config["arms"]] == [5e-5, 3e-5]
        return {"state": "COMPLETED"}
    for _ in range(2):
        result = WorkflowRunner(spec, root=tmp_path / "workflow", handlers={
            "continuous_training": training, "experiment": experiment}).run()
        assert result["state"] == "COMPLETED"
    assert calls == ["training", "experiment"]


def test_ab_only_uses_initial_parent(parent, tmp_path):
    raw = parameters()
    raw["training"]["iterations"] = 0
    steps = job.compile_job(raw, runs_root=tmp_path)["workflow"]["steps"]
    assert len(steps) == 1
    assert steps[0]["config"]["parent"] == parent.ref.to_dict()


def test_check_is_read_only_and_missing_telegram_prevents_launch(parent, tmp_path, monkeypatch):
    from gocube_golden.notifications import telegram
    monkeypatch.setattr(telegram, "load_config", lambda: None)
    monkeypatch.setattr(entry, "_launch_durable_workflow_controller", lambda *a, **kw: pytest.fail("launched"))
    result = entry.launch_operator_job(parameters(), runs_root=tmp_path, check_only=True)
    assert result["state"] == "VALIDATED" and not result["telegram_configured"]
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="Telegram is not configured"):
        entry.launch_operator_job(parameters(), runs_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_launch_pins_runtime_and_rejects_parameter_drift(parent, tmp_path, monkeypatch):
    from gocube_golden.notifications import telegram
    from gocube_golden.orchestrator_v2.immutable_runtime import ImmutableRuntimeManager
    monkeypatch.setattr(telegram, "load_config", lambda: ("fake-token", "fake-chat"))
    monkeypatch.setattr(entry, "_require_file_backed_entrypoint", lambda: None)
    monkeypatch.setattr(entry, "_require_committed_job_code", lambda: None)
    monkeypatch.setattr(entry, "_entrypoint_code_identity", lambda: "commit-one")
    pins = []
    monkeypatch.setattr(ImmutableRuntimeManager, "ensure", lambda self, commit: pins.append(commit) or commit)
    launches = []
    monkeypatch.setattr(entry, "_launch_durable_workflow_controller",
        lambda path, **kw: launches.append((path, kw, json.loads(os.environ[PERMIT_ENV]))) or {"state": "STARTED"})
    entry.launch_operator_job(parameters(), runs_root=tmp_path)
    monkeypatch.setattr(entry, "_entrypoint_code_identity", lambda: "commit-two")
    entry.launch_operator_job(parameters(), runs_root=tmp_path)
    assert pins == ["commit-one", "commit-one"]
    assert launches[1][1]["runtime"] == "commit-one"
    assert launches[1][2]["action_type"] == "workflow-controller"
    assert launches[1][2]["run_id"] == "five-iterations"
    altered = copy.deepcopy(parameters())
    altered["training"]["learning_rate"] = 1e-5
    with pytest.raises(ValueError, match="different parameters"):
        entry.launch_operator_job(altered, runs_root=tmp_path)
    assert len(launches) == 2


def test_start_settings_survive_real_outbox_and_are_delivered_once(tmp_path):
    from test_orchestrator_v2_continuous_training import _runner
    from gocube_golden.notifications import NotificationStore, NotificationDispatcher, format_event
    cfg = EffectiveConfig(topology="torus9", compatibility={"input_channels": 5},
        self_play={"komi": 1.5, "games_per_iteration": 384, "mcts_simulations": 200},
        training={"learning_rate": 5e-5, "optimizer_steps_per_iteration": 160, "batch_size": 64,
                  "gradient_clip": 8.0},
        replay={"generations": 6, "cap": None}, arena={"simulations": 128},
        execution={"workers": 16, "active_games_per_worker": 4})
    store = NotificationStore(tmp_path / "outbox")
    messages = []
    class Transport:
        def send(self, text):
            messages.append(text)
            return {"ok": True, "message_id": len(messages)}
    dispatcher = NotificationDispatcher(store, transport=Transport(), background=False)
    runner, _, _, _, parent = _runner(tmp_path, generations=5, effective_config=cfg, notifier=dispatcher)
    runner._launch_id = "test-launch"
    runner._report_start(parent, SimpleNamespace(config=cfg))
    events = list(store.iter_events())
    assert len(events) == 1
    event = events[0]
    assert event.payload["stop_after_iterations"] == 5
    assert event.payload["training"]["optimizer_steps_per_iteration"] == 160
    text = format_event(event)
    for expected in ("LR=5e-05", "Steps: 160", "Batch: 64", "Gradient clip: 8.0", "Contexts: 64", "no position cap",
                     "every 5 generations", "MCTS: 128 sims", "Stop: after 5 iterations"):
        assert expected in text
    dispatcher.flush(1)
    NotificationDispatcher(store, transport=Transport(), background=False).flush(1)
    assert messages == [text]


def test_whitespace_environment_does_not_disable_standard_telegram_file(tmp_path):
    from gocube_golden.notifications.telegram import load_config, TOKEN_ENV, CHAT_ID_ENV
    path = tmp_path / "telegram.env"
    path.write_text(f"{TOKEN_ENV}=fake-token\n{CHAT_ID_ENV}=fake-chat\n")
    assert load_config(environ={TOKEN_ENV: " ", CHAT_ID_ENV: "\t"}, env_file=path) == ("fake-token", "fake-chat")


def test_uncommitted_implementation_cannot_be_launched(monkeypatch):
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=" M gocube_golden/orchestrator_v2/operator_job.py\n"))
    with pytest.raises(ValueError, match="committed implementation"):
        entry._require_committed_job_code()


def test_controller_uses_pinned_working_directory_and_environment(tmp_path, monkeypatch):
    import json
    from gocube_golden.orchestrator_v2.immutable_runtime import ImmutableRuntime
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"workflow_id": "pin-test", "topology": "torus9",
        "steps": [{"step_id": "train", "action": "continuous_training", "config": {}}]}))
    runtime = ImmutableRuntime(repo_root=tmp_path / "source", commit="commit-one", tree="tree-one", path=tmp_path / "pinned")
    calls = []
    monkeypatch.setattr(entry.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(pid=1234))
    monkeypatch.setattr(entry.os, "getpgid", lambda pid: pid)
    result = entry._launch_durable_workflow_controller(plan, runs_root=tmp_path / "runs", runtime=runtime)
    assert result["state"] == "STARTED"
    assert calls[0][1]["cwd"] == runtime.path
    assert calls[0][1]["env"]["AZ_ORCHESTRATOR_RUNTIME_COMMIT"] == "commit-one"
    assert calls[0][1]["env"]["PYTHONPATH"].split(":")[0] == str(runtime.path)
