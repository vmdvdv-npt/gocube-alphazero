import copy
import json
import os
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig
from gocube_golden.orchestrator_v2 import operator_job as job
from gocube_golden.orchestrator_v2 import production_entrypoint as entry
from gocube_golden.orchestrator_v2.execution_permit import (
    PERMIT_ENV,
    PERMIT_KEY_ENV,
    _child_execution_permit,
    _production_authority,
)
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


@pytest.mark.parametrize("seed", [0, 2026100501])
@pytest.mark.parametrize("mode", ["fixed", "pcr"])
def test_selfplay_seed_reaches_every_training_arm_without_changing_learner_seed(parent, seed, mode):
    inherited = parent.effective_config.config.to_dict()
    inherited['execution'].update(selfplay_master_seed=2026092901, training_master_seed=2026092701)
    parent.effective_config.config = EffectiveConfig.from_dict(inherited)
    raw = parameters()
    raw['self_play'] = {'master_seed': seed, 'search_mode': mode}
    if mode == 'pcr':
        raw['self_play']['pcr'] = {'cheap_simulations': 100, 'full_simulations': 500,
                                 'full_probability': 0.25}
    normalized = job.parse_job(raw)
    assert job.parse_job(normalized) == normalized
    steps = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps']
    configs = [steps[0]['config']['effective_config']]
    configs.extend(arm['config'] for arm in steps[1]['config']['arms'])
    for config in configs:
        assert config['execution']['selfplay_master_seed'] == seed
        assert config['execution']['training_master_seed'] == 2026092701
        assert 'master_seed' not in config['self_play']
    assert raw['self_play']['master_seed'] == seed
    assert parent.effective_config.config.execution['selfplay_master_seed'] == 2026092901


def test_omitted_selfplay_seed_preserves_parent_seed_and_explicit_seed_changes_fingerprint(parent):
    inherited = parent.effective_config.config.to_dict()
    inherited['execution']['selfplay_master_seed'] = 123
    parent.effective_config.config = EffectiveConfig.from_dict(inherited)
    raw = parameters()
    old = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps'][0]['config']['effective_config']
    raw['self_play'] = {'master_seed': 456}
    new = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps'][0]['config']['effective_config']
    assert old['execution']['selfplay_master_seed'] == 123
    assert new['execution']['selfplay_master_seed'] == 456
    assert EffectiveConfig.from_dict(old).fingerprint != EffectiveConfig.from_dict(new).fingerprint


@pytest.mark.parametrize('seed', [True, False, -1, 1.5, '2026100501', None, float('nan')])
def test_invalid_selfplay_seed_rejected_before_parent_resolution(monkeypatch, seed):
    monkeypatch.setattr(job, 'resolve_parent', lambda *a, **kw: pytest.fail('resolved invalid seed'))
    raw = parameters()
    raw['self_play'] = {'master_seed': seed}
    with pytest.raises(ValueError, match='self_play.master_seed'):
        job.compile_job(raw)


def test_null_iterations_compiles_as_unbounded_continuation(parent, tmp_path):
    raw = parameters()
    raw["training"] = {"iterations": None, "games_per_iteration": 768,
                       "updates_per_iteration": 1280, "batch_size": 64,
                       "learning_rate": 1e-4, "replay_generations": 3}
    raw["ab_tests"] = []
    raw["arena"] = {"every_iterations": 1, "games": 192,
                    "mcts_simulations": 200}

    normalized = job.parse_job(raw)
    assert normalized["training"]["iterations"] is None

    compiled = job.compile_job(raw, runs_root=tmp_path)
    step = compiled["workflow"]["steps"][0]
    assert step["action"] == "continuous_training"
    assert step["config"]["generations"] is None
    assert step["config"]["arena_cadence"] == 1


def test_null_iterations_still_requires_training_parent():
    raw = {"schema": job.SCHEMA, "run_id": "unbounded-no-parent",
           "training": {"iterations": None}, "ab_tests": []}
    with pytest.raises(ValueError, match="parent"):
        job.parse_job(raw)


def test_unbounded_training_cannot_be_followed_by_ab_tests():
    raw = parameters()
    raw["training"]["iterations"] = None
    with pytest.raises(ValueError, match="cannot be followed by A/B"):
        job.parse_job(raw)


@pytest.mark.parametrize("patch", [
    {"notifications": False}, {"command": "arbitrary executable"},
    {"parent": "../../checkpoint"}, {"training": {"learning_rate": float("nan")}},
    {"training": {"iterations": True}}, {"training": {"batch_size": 0}},
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
    assert result["operator_guide"]["repository_path"].endswith("ORCHESTRATOR_V2_LAUNCH_GUIDE.md")
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
    guide_path = tmp_path / "torus9/orchestration/jobs/five-iterations/operator-guide.json"
    initial_guide = guide_path.read_bytes()
    assert json.loads(initial_guide)["sha256"]
    monkeypatch.setattr(entry, "_entrypoint_code_identity", lambda: "commit-two")
    entry.launch_operator_job(parameters(), runs_root=tmp_path)
    assert pins == ["commit-one", "commit-one"]
    assert guide_path.read_bytes() == initial_guide
    assert launches[1][1]["runtime"] == "commit-one"
    assert launches[1][2]["action_type"] == "workflow-controller"
    assert launches[1][2]["run_id"] == "five-iterations"
    altered = copy.deepcopy(parameters())
    altered["training"]["learning_rate"] = 1e-5
    with pytest.raises(ValueError, match="different parameters"):
        entry.launch_operator_job(altered, runs_root=tmp_path)
    reseeded = copy.deepcopy(parameters())
    reseeded['self_play'] = {'master_seed': 2026100501}
    with pytest.raises(ValueError, match="different parameters"):
        entry.launch_operator_job(reseeded, runs_root=tmp_path)
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
    monkeypatch.setattr(entry, "_wait_for_controller_ready", lambda *a, **kw: None)
    with _production_authority(mode="workflow", topology="torus9", run_id="pin-test", code_identity="commit-one"):
        with _child_execution_permit(
            action_type="workflow-controller",
            topology="torus9",
            run_id="pin-test",
            code_identity="commit-one",
        ):
            result = entry._launch_durable_workflow_controller(
                plan, runs_root=tmp_path / "runs", runtime=runtime
            )
    assert result["state"] == "STARTED"
    assert calls[0][1]["cwd"] == runtime.path
    assert calls[0][1]["env"]["AZ_ORCHESTRATOR_RUNTIME_COMMIT"] == "commit-one"
    assert calls[0][1]["env"]["PYTHONPATH"].split(":")[0] == str(runtime.path)
    assert PERMIT_ENV in calls[0][1]["env"]
    assert PERMIT_KEY_ENV in calls[0][1]["env"]
    assert len(calls[0][1]["pass_fds"]) == 1
    assert "--startup-ready-fd" in calls[0][0]


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_batch_size_reaches_effective_config(parent, batch_size):
    raw = parameters()
    raw["training"]["batch_size"] = batch_size
    compiled = job.compile_job(raw, runs_root=".")
    config = compiled["workflow"]["steps"][0]["config"]["effective_config"]
    assert config["training"]["batch_size"] == batch_size


def test_offline_named_arms_are_stable_and_experiment_only(parent, monkeypatch):
    from gocube_golden.orchestrator_v2 import offline_replay
    raw = parameters()
    raw['training']['iterations'] = 0
    raw['ab_tests'] = [{'id': 'batch', 'iterations': 2,
        'offline_replay': ['source/M199', 'source/M200'],
        'arms': {f'B{b}': {'batch_size': b, 'updates_per_iteration': 256 // b}
                 for b in (64, 128, 256)}}]
    normalized = job.parse_job(raw)
    assert job.parse_job(normalized) == normalized
    replay = [{'generation': g, 'buckets': [{}] * 6} for g in (199, 200)]
    monkeypatch.setattr(offline_replay, 'resolve_offline_replay', lambda *a, **kw: replay)
    steps = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps']
    assert len(steps) == 1 and steps[0]['action'] == 'experiment'
    arms = steps[0]['config']['arms']
    assert [a['arm_id'] for a in arms] == ['B64', 'B128', 'B256']
    assert all(a['config']['extensions']['offline_ab_replay'] == replay for a in arms)
    raw['training']['iterations'] = 1
    with pytest.raises(ValueError, match='iterations=0'):
        job.parse_job(raw)
    raw['training'] = {'offline_replay': ['source/M199']}
    with pytest.raises(ValueError, match='unknown fields'):
        job.parse_job(raw)


def _offline_sampling_parameters():
    raw = parameters()
    raw['training']['iterations'] = 0
    raw['ab_tests'] = [{
        'id': 'sampling-seed2',
        'iterations': 2,
        'offline_replay': ['source/M199', 'source/M200'],
        'arms': {
            'S50': {'replay_sampling': {'mode': 'policy_surprise', 'weight': .5}},
            'U64': {},
        },
        'arena': {
            'games': 192,
            'mcts_simulations': 64,
            'tree_reuse': True,
            'master_seed': 2026100702,
        },
    }]
    return raw


def _set_parent_training_seed(parent, seed):
    config = parent.effective_config.config.to_dict()
    config['execution']['training_master_seed'] = seed
    parent.effective_config.config = EffectiveConfig.from_dict(config)


def test_offline_ab_training_seed_is_shared_by_all_arms_and_arena_seed_reaches_experiment(
    parent, monkeypatch,
):
    from gocube_golden import policy_surprise
    from gocube_golden.orchestrator_v2 import offline_replay
    from gocube_golden.orchestrator_v2.experiment_plan import ExperimentConfig

    _set_parent_training_seed(parent, 2026092701)
    raw = _offline_sampling_parameters()
    raw['ab_tests'][0]['training_seed'] = 2026092702
    replay = [{'generation': generation, 'buckets': [{}] * 6} for generation in (199, 200)]
    monkeypatch.setattr(offline_replay, 'resolve_offline_replay', lambda *a, **kw: replay)
    monkeypatch.setattr(policy_surprise, 'resolve_spec', lambda *a, **kw: {'fingerprint': 'synthetic'})

    normalized = job.parse_job(raw)
    assert normalized['ab_tests'][0]['training_seed'] == 2026092702
    compiled = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps'][0]['config']
    arms = {arm['arm_id']: arm['config'] for arm in compiled['arms']}

    assert compiled['parent'] == parent.ref.to_dict()
    assert arms['S50']['execution']['training_master_seed'] == 2026092702
    assert arms['U64']['execution']['training_master_seed'] == 2026092702
    assert arms['S50']['extensions']['offline_training_seed_override'] is True
    assert arms['U64']['extensions']['offline_training_seed_override'] is True
    assert arms['S50']['execution']['training_master_seed'] == arms['U64']['execution']['training_master_seed']
    assert arms['S50']['replay']['sampling'] == {'mode': 'policy_surprise', 'weight': .5}
    assert 'sampling' not in arms['U64']['replay']
    assert arms['S50']['training']['optimizer'] == arms['U64']['training']['optimizer'] == 'Adam'
    assert parent.effective_config.config.execution['training_master_seed'] == 2026092701
    assert compiled['arena']['master_seed'] == 2026100702
    assert ExperimentConfig.from_dict(compiled).arena_master_seed == 2026100702


def test_offline_ab_without_new_seeds_keeps_parent_seed_and_legacy_arena_default(
    parent, monkeypatch,
):
    from gocube_golden import policy_surprise
    from gocube_golden.orchestrator_v2 import offline_replay

    _set_parent_training_seed(parent, 2026092701)
    raw = _offline_sampling_parameters()
    raw['ab_tests'][0]['arena'].pop('master_seed')
    replay = [{'generation': generation, 'buckets': [{}] * 6} for generation in (199, 200)]
    monkeypatch.setattr(offline_replay, 'resolve_offline_replay', lambda *a, **kw: replay)
    monkeypatch.setattr(policy_surprise, 'resolve_spec', lambda *a, **kw: {'fingerprint': 'synthetic'})

    compiled = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps'][0]['config']
    assert [arm['config']['execution']['training_master_seed'] for arm in compiled['arms']] == [
        2026092701,
        2026092701,
    ]
    assert all('offline_training_seed_override' not in arm['config']['extensions'] for arm in compiled['arms'])
    assert compiled['arena']['master_seed'] == job.ARENA_RUN_MASTER_SEED
    assert 'training_seed' not in job.parse_job(raw)['ab_tests'][0]


def test_zero_offline_training_and_arena_seeds_are_valid(parent, monkeypatch):
    from gocube_golden import policy_surprise
    from gocube_golden.orchestrator_v2 import offline_replay

    raw = _offline_sampling_parameters()
    raw['ab_tests'][0]['training_seed'] = 0
    raw['ab_tests'][0]['arena']['master_seed'] = 0
    replay = [{'generation': generation, 'buckets': [{}] * 6} for generation in (199, 200)]
    monkeypatch.setattr(offline_replay, 'resolve_offline_replay', lambda *a, **kw: replay)
    monkeypatch.setattr(policy_surprise, 'resolve_spec', lambda *a, **kw: {'fingerprint': 'synthetic'})

    compiled = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps'][0]['config']
    assert [arm['config']['execution']['training_master_seed'] for arm in compiled['arms']] == [0, 0]
    assert compiled['arena']['master_seed'] == 0


def test_training_seed_is_rejected_for_non_offline_ab_and_per_arm():
    ordinary = parameters()
    ordinary['ab_tests'][0]['training_seed'] = 2026092702
    with pytest.raises(ValueError, match='only for offline A/B'):
        job.parse_job(ordinary)

    offline = _offline_sampling_parameters()
    offline['ab_tests'][0]['arms']['S50']['training_seed'] = 2026092702
    with pytest.raises(ValueError, match='unknown fields: training_seed'):
        job.parse_job(offline)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('training_seed', -1),
        ('training_seed', True),
        ('training_seed', 1.5),
        ('training_seed', None),
        ('arena.master_seed', -1),
        ('arena.master_seed', True),
        ('arena.master_seed', 1.5),
        ('arena.master_seed', None),
    ],
)
def test_invalid_offline_training_and_arena_seeds_are_rejected(field, value):
    raw = _offline_sampling_parameters()
    if field == 'training_seed':
        raw['ab_tests'][0]['training_seed'] = value
    else:
        raw['ab_tests'][0]['arena']['master_seed'] = value
    with pytest.raises(ValueError):
        job.parse_job(raw)


def test_ordinary_job_from_offline_checkpoint_does_not_inherit_offline_mode(parent):
    config = parent.effective_config.config.to_dict()
    config['extensions']['offline_ab_replay'] = [{'generation': 199, 'buckets': []}]
    parent.effective_config.config = EffectiveConfig.from_dict(config)
    raw = parameters()
    steps = job.compile_job(raw, resolver=SimpleNamespace())['workflow']['steps']
    assert 'offline_ab_replay' not in steps[0]['config']['effective_config']['extensions']
    assert all('offline_ab_replay' not in arm['config']['extensions']
               for arm in steps[1]['config']['arms'])


def test_policy_surprise_compile_only_sampling_differs(parent, monkeypatch):
    from gocube_golden.orchestrator_v2 import offline_replay
    from gocube_golden import policy_surprise
    raw = parameters(); raw['training']['iterations'] = 0
    raw['ab_tests'] = [{'id':'sampling', 'iterations':2,
        'offline_replay':['source/M199','source/M200'],
        'arms':{'S50':{'replay_sampling':{'mode':'policy_surprise','weight':.5}},'U64':{}}}]
    replay = [{'generation':g,'buckets':[{}]*6} for g in (199,200)]
    monkeypatch.setattr(offline_replay,'resolve_offline_replay',lambda *a, **kw:replay)
    monkeypatch.setattr(policy_surprise,'resolve_spec',lambda *a, **kw:{'fingerprint':'synthetic'})
    compiled = job.compile_job(raw,resolver=SimpleNamespace())['workflow']['steps'][0]['config']
    arms = compiled['arms']
    assert [a['arm_id'] for a in arms] == ['S50','U64']
    a,b = [arm['config'] for arm in arms]
    assert a['replay'].pop('sampling') == {'mode':'policy_surprise','weight':.5}
    assert a == b
    assert compiled['arena']['master_seed'] == job.ARENA_RUN_MASTER_SEED
    assert job.parse_job(job.parse_job(raw)) == job.parse_job(raw)


@pytest.mark.parametrize('setting', [{}, {'mode':'unknown'}, {'mode':'uniform','weight':.5},
                                       {'mode':'policy_surprise','weight':float('nan')}])
def test_invalid_replay_sampling_rejected(setting):
    raw = parameters(); raw['training']['iterations']=0
    raw['ab_tests']=[{'id':'sampling','iterations':1,'offline_replay':['source/M199'],
                     'arms':{'S50':{'replay_sampling':setting},'U64':{}}}]
    with pytest.raises(ValueError):
        job.parse_job(raw)


def test_new_ordinary_job_drops_inherited_sampling_and_cache(parent):
    cfg=parent.effective_config.config.to_dict()
    cfg['replay']['sampling']={'mode':'policy_surprise','weight':.5}
    cfg['extensions'].update(offline_ab_replay=[{}],policy_surprise_spec={},policy_surprise_cache={})
    parent.effective_config.config=EffectiveConfig.from_dict(cfg)
    steps=job.compile_job(parameters(),resolver=SimpleNamespace())['workflow']['steps']
    result=steps[0]['config']['effective_config']
    assert 'sampling' not in result['replay']
    assert not set(('offline_ab_replay','policy_surprise_spec','policy_surprise_cache')) & result['extensions'].keys()
