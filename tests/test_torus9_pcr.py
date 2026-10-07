"""PCR regressions use synthetic two-pass games and temporary lineage storage."""
from dataclasses import asdict, replace
import copy
import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from gocube_golden import torus9_monolith as core
from gocube_golden import torus9_selfplay as sp
from gocube_golden import torus9_five_channel_training as ordinary
from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig, validate_generation_commit
from gocube_golden.artifact_resolver import ArtifactResolver
from gocube_golden.orchestrator_v2 import operator_job as job
from gocube_golden.orchestrator_v2.execution_permit import _test_authority
from gocube_golden.orchestrator_v2.generation_runner import OutputLineage, ResolvedGenerationInput
from gocube_golden.orchestrator_v2.operator_messages import format_training_started
from gocube_golden.orchestrator_v2.torus9_production import Torus9ProductionLineage
from gocube_golden.provenance import CodeIdentity, file_sha256
from gocube_golden.search import Evaluation
from gocube_golden.selfplay_engine import GameFinished, InferenceNeed
from gocube_golden.torus9_adaptation import (
    AdaptationSearchContract, FINGERPRINT, PCRSearchContract, game_targets, save_torch, validate_game,
)
from gocube_golden.torus9_pcr import PlayoutCapRandomization, position_telemetry, resolve_search_mode
from test_torus9_five_channel_training import parent, game as legacy_game

CAPS = PlayoutCapRandomization(3, 7, .5)


def context(contract, master_seed=1):
    return sp.Torus9SelfPlayWorkerContext(
        run_id='synthetic-pcr', model_checkpoint_label='M0', checkpoint_artifact_hash='sha256:synthetic',
        master_seed=master_seed, profile_fingerprint=FINGERPRINT,
        code_identity=CodeIdentity('a'*40, 'b'*40, True), contract=contract,
        profile_id=core.TORUS9_CURRENT_PROFILE_ID, expected_model_hash='synthetic-actor')


def two_pass_game(monkeypatch, contract, identity='two-pass', master_seed=1):
    temperatures = []
    def play_pass(result, *, temperature, **kwargs):
        temperatures.append(temperature)
        return core.PASS
    monkeypatch.setattr(sp, 'sample_action_from_search_result', play_pass)
    worker = sp._Torus9CooperativeGame(context(contract, master_seed), identity, None)
    policy = (0.,)*81 + (1.,)
    for _ in range(100):
        step = worker.advance()
        if isinstance(step, GameFinished):
            assert step.record.technical_termination is None, step.record.error
            assert temperatures == [1., 1.]
            return step.record
        assert isinstance(step, InferenceNeed)
        worker.resume(Evaluation(policy=policy, wdl=(0., 1., 0.)))
    pytest.fail('synthetic two-pass search did not finish')


def test_selection_is_per_game_ply_and_independent_of_scheduling():
    caps = PlayoutCapRandomization(100, 500, .25)
    keys = [(sp.torus9_game_seed(31, 'run', f'g{i}'), ply) for i in range(20) for ply in range(1,101)]
    expected = {key: caps.mode(*key) for key in keys}
    assert {key: caps.mode(*key) for key in reversed(keys)} == expected
    assert {key: caps.mode(*key) for key in keys[::2]+keys[1::2]} == expected
    assert .20 < list(expected.values()).count('full') / len(keys) < .30
    assert caps.nominal_mean_simulations == 200


@pytest.mark.parametrize('mode', ['cheap', 'full', 'fixed'])
def test_search_cap_noise_and_unchanged_temperature(monkeypatch, mode):
    contract = (AdaptationSearchContract(simulations=7, komi=1.5) if mode == 'fixed'
                else PCRSearchContract(simulations=200, komi=1.5, pcr=CAPS))
    worker = sp._Torus9CooperativeGame(context(contract), 'test', None)
    if mode != 'fixed':
        worker.game_seed = next(seed for seed in range(100) if CAPS.mode(seed,1) == mode)
    worker._start_search()
    settings = worker._session.settings
    assert settings.simulations == (3 if mode == 'cheap' else 7)
    assert settings.cpuct == 1.25 and settings.fpu == 0.
    transform = worker._session._transform
    if mode == 'cheap':
        assert transform is None
    else:
        state = worker.state
        legal = core.prepare_legal_actions(state)
        base = Evaluation(policy=(0.,)*81+(1.,), wdl=(0.,1.,0.))
        transformed = transform(base, state, legal)
        assert transformed != base and transformed.wdl == base.wdl
    record = two_pass_game(monkeypatch, contract)
    for p in record.positions:
        expected = CAPS.mode(record.game_seed,p.ply) if mode != 'fixed' else 'fixed'
        assert p.search_mode == expected
        assert p.search_simulations == (3 if expected == 'cheap' else 7)
        assert sum(p.root_visits) == p.search_simulations
        assert p.training_eligible == (expected != 'cheap')


def mixed_record(monkeypatch):
    contract = PCRSearchContract(simulations=200, komi=1.5, pcr=CAPS)
    for seed in range(20):
        game_seed = sp.torus9_game_seed(seed, 'synthetic-pcr', 'two-pass')
        if [CAPS.mode(game_seed, p) for p in (1,2)] == ['cheap','full']:
            return two_pass_game(monkeypatch, contract, master_seed=seed)
    pytest.fail('No mixed test seed')


def test_raw_trajectory_retained_and_all_learner_targets_exclude_cheap(monkeypatch, tmp_path):
    record = mixed_record(monkeypatch)
    assert len(record.positions) == len(record.final_action_trace) == 2
    assert record.to_dict()['positions'][0]['search_mode'] == 'cheap'
    targets = game_targets(record)
    validate_game(targets)
    assert all(len(targets[k]) == 1 for k in ('observation','pi','z','ownership','score','legal','visits'))
    assert torch.equal(targets['pi'][0], torch.tensor(record.positions[1].pi))
    final = core.initial_state(topology=core.TORUS_9X9, komi=1.5)
    for action in record.final_action_trace:
        final = core.apply_action(final,action).after
    assert targets['score'][0].item() == pytest.approx(core.torus9_score_target(final,'WHITE')/81.5)
    telemetry = position_telemetry([record], CAPS)
    assert telemetry['raw_positions'] == 2 and telemetry['training_positions'] == 1
    assert telemetry['pcr_full_fraction'] == telemetry['pcr_cheap_fraction'] == .5
    targets['split'] = 'train'
    path = tmp_path/'replay.pt'
    save_torch(path, {'contract': FINGERPRINT, 'actor_hash': record.model_hash, 'games': [targets]})
    loaded = ordinary.load_replay([{'shards':[{'path':str(path),'sha':file_sha256(path)}]}], split='train')
    trainer = object.__new__(ordinary.OrdinaryTrainer)
    trainer.seed = 91
    trainer.batch_size = 64
    trainer.model = torch.nn.Linear(1,1)
    batch = trainer.batch(loaded,1)
    assert batch['score'].shape == (64,)
    assert torch.equal(batch['pi'], targets['pi'].repeat(64,1))
    all_cheap = replace(record, positions=tuple(replace(p,search_mode='cheap',search_simulations=3,
                                                       training_eligible=False) for p in record.positions))
    assert game_targets(all_cheap) is None
    assert len(all_cheap.to_dict()['positions']) == 2


def test_fixed_and_historical_records_keep_all_targets(monkeypatch):
    contract = AdaptationSearchContract(simulations=7, komi=1.5)
    old_fingerprint = contract.fingerprint
    record = two_pass_game(monkeypatch,contract)
    assert all(p.search_mode == 'fixed' and p.training_eligible for p in record.positions)
    historical = replace(record, positions=tuple(replace(p,search_simulations=None) for p in record.positions))
    for key, tensor in game_targets(record).items():
        old = game_targets(historical)[key]
        assert torch.equal(tensor,old) if isinstance(tensor,torch.Tensor) else tensor == old
    assert len(game_targets(record)['score']) == 2
    assert AdaptationSearchContract(simulations=7,komi=1.5).fingerprint == old_fingerprint
    assert resolve_search_mode({}) is None


@pytest.mark.parametrize('value', [
    {'search_mode':'unknown'}, {'search_mode':'pcr'}, {'pcr':asdict(CAPS)},
    *[{'search_mode':'pcr','pcr':{**asdict(CAPS),**patch}} for patch in (
        {'cheap_simulations':0}, {'cheap_simulations':True}, {'full_simulations':3},
        {'full_simulations':7.5}, {'full_probability':0}, {'full_probability':1},
        {'full_probability':float('nan')}, {'full_probability':True}, {'typo':1})],
])
def test_invalid_settings_rejected(value):
    with pytest.raises(ValueError):
        resolve_search_mode(value)


def test_operator_config_identity_and_workflow_resume(tmp_path, monkeypatch):
    base = EffectiveConfig(topology='torus9',compatibility={'input_channels':5},
        self_play={'komi':1.5},training={'optimizer':'Adam','weight_decay':0.,'l2_sp':False},
        replay={'cap':None},execution={},extensions={'training_driver':ordinary.SCHEMA})
    node = SimpleNamespace(ref=CheckpointRef('torus9','source','M0',0,'M0.pt','sha256:'+'a'*64),
                           effective_config=SimpleNamespace(config=base))
    monkeypatch.setattr(job,'resolve_parent',lambda *a,**kw:node)
    raw = {'schema':job.SCHEMA,'run_id':'pcr-test','parent':'source/M0',
           'self_play':{'search_mode':'pcr','pcr':{'cheap_simulations':100,'full_simulations':500,'full_probability':.25}}}
    normalized = job.parse_job(raw)
    assert job.parse_job(normalized) == normalized
    compiled = job.compile_job(raw,runs_root=tmp_path)
    cfg = EffectiveConfig.from_dict(compiled['workflow']['steps'][0]['config']['effective_config'])
    assert cfg.self_play['pcr']['full_simulations'] == 500
    changed = copy.deepcopy(cfg.to_dict())
    changed['self_play']['pcr']['full_probability'] = .3
    assert EffectiveConfig.from_dict(changed).fingerprint != cfg.fingerprint
    assert EffectiveConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).fingerprint == cfg.fingerprint
    assert PCRSearchContract(komi=1.5,pcr=CAPS).fingerprint != PCRSearchContract(
        komi=1.5,pcr=replace(CAPS,full_probability=.25)).fingerprint
    text = format_training_started(topology='torus9',lineage_id='test',parent_label='M0',network=None,
                                   effective_config=cfg,arena_cadence=5,arena_config=SimpleNamespace(games=192))
    for line in ('Self-play MCTS: PCR','Cheap: 100 sims / 75%','Full: 500 sims / 25%','Nominal mean: 200 sims'):
        assert line in text
    from gocube_golden.orchestrator_v2.workflow import WorkflowRunner, WorkflowSpec
    spec = WorkflowSpec.from_dict(compiled['workflow'])
    calls = []
    def training(*,config,**kwargs):
        calls.append(config['effective_config']['self_play'])
        return {'checkpoint':node.ref.to_dict()}
    for _ in range(2):
        result = WorkflowRunner(spec,root=tmp_path/'workflow',handlers={'continuous_training':training}).run()
        assert result['state'] == 'COMPLETED'
    assert len(calls) == 1 and calls[0]['pcr']['full_probability'] == .25
    # Omitting the mode in a new job must use fixed even after a PCR parent.
    fixed = job._effective(cfg.to_dict(),normalized['training'],normalized['arena'])
    assert resolve_search_mode(fixed['self_play']) is None


def test_generation_resume_reuses_raw_shard_and_filtered_replay(parent, tmp_path, monkeypatch):
    raw = torch.load(parent,weights_only=False)
    actor = raw['metadata']['model_hash']
    inherited = tmp_path/'old-fixed.pt'
    save_torch(inherited,{'contract':FINGERPRINT,'actor_hash':actor,
        'games':[legacy_game('old','train',actor),legacy_game('heldout','validation',actor)]})
    buckets = [{'generation':0,'shards':[{'path':str(inherited),'sha':file_sha256(inherited)}]}]
    raw['metadata']['replay_ids'] = [file_sha256(inherited)]
    save_torch(parent,raw)
    before = file_sha256(inherited)
    ref = CheckpointRef('torus9','source','adapted',0,parent.name,file_sha256(parent))
    source = SimpleNamespace(ref=ref,path=parent,generation=0)
    cfg = EffectiveConfig(topology='torus9',compatibility={'input_channels':5},
        self_play={'komi':1.5,'games_per_iteration':2,'mcts_simulations':200,
                   'search_mode':'pcr','pcr':asdict(CAPS)},
        training={'batch_size':64,'optimizer':'Adam','weight_decay':0.,'l2_sp':False,
                  'learning_rate':5e-5,'optimizer_steps_per_iteration':1},
        replay={'cap':None,'generations':6},execution={'device':'cpu','workers':1,
                  'training_master_seed':91,'selfplay_master_seed':92},
        extensions={'training_driver':ordinary.SCHEMA,'adaptation_parent':ref.to_dict(),
                    'initial_replay_buckets':buckets,'validation_buckets':buckets})
    root, resolved_cfg = Torus9ProductionLineage(tmp_path/'runs').prepare(
        topology='torus9',lineage_id='pcr-isolated',parent=source,effective_config=cfg,
        experiment_id='pcr-test',arm_id='pcr-test')
    record = mixed_record(monkeypatch)
    calls = []
    def fake_selfplay(model,**kwargs):
        assert kwargs['search_mode'] == 'pcr' and kwargs['pcr'] == asdict(CAPS)
        calls.append(kwargs['ids'])
        records = [replace(record,game_id=identity,model_hash=actor,
                   positions=tuple(replace(p,model_hash=actor) for p in record.positions))
                   for identity in kwargs['ids']]
        return SimpleNamespace(records=records,telemetry={})
    monkeypatch.setattr(ordinary,'selfplay',fake_selfplay)
    monkeypatch.setattr(ordinary.OrdinaryTrainer,'evaluate',lambda *a,**kw:{'synthetic':0.})
    original_step = ordinary.OrdinaryTrainer.step
    def crash_before_learner(*args):
        raise RuntimeError('test interruption after shard commit')
    monkeypatch.setattr(ordinary.OrdinaryTrainer,'step',crash_before_learner)
    request = ResolvedGenerationInput(source,1,resolved_cfg,OutputLineage(topology='torus9',lineage_id=root.name,root=root))
    with _test_authority(), pytest.raises(RuntimeError,match='test interruption'):
        ordinary.run_generation(request)
    committed_identity_path = root/'replay/g0001-0000.identity.json'
    committed_shard_path = root/'replay/g0001-0000.pt'
    committed_identity_before = json.loads(committed_identity_path.read_text())
    committed_identity_sha_before = file_sha256(committed_identity_path)
    committed_shard_sha_before = file_sha256(committed_shard_path)
    monkeypatch.setattr(ordinary.OrdinaryTrainer,'step',original_step)
    with _test_authority():
        result = ordinary.run_generation(request)
    assert len(calls) == 1
    assert file_sha256(committed_identity_path) == committed_identity_sha_before
    assert file_sha256(committed_shard_path) == committed_shard_sha_before
    validate_generation_commit(root=root,lineage_id=root.name,generation=1,
                               reuse_committed_rolling_replay_identity=True)
    summary = json.loads((root/'iter-01-summary.json').read_text())
    for key,value in {'raw_positions':4,'training_positions':2,'pcr_full_positions':2,
                      'pcr_cheap_positions':2,'pcr_full_fraction':.5,'nominal_mean_simulations':5}.items():
        assert summary[key] == value
    with gzip.open(root/'replay/g0001-0000.games.jsonl.gz','rt') as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 2 and all(len(g['positions']) == 2 for g in records)
    assert records[0]['positions'][0]['training_eligible'] is False
    shard = torch.load(root/'replay/g0001-0000.pt',weights_only=False)
    assert all(len(g['score']) == 1 for g in shard['games'])
    assert shard['pcr'] == asdict(CAPS)
    telemetry = json.loads((root/'replay/g0001-0000.telemetry.json').read_text())
    assert telemetry['training_positions'] == 2 and telemetry['raw_positions'] == 4
    assert file_sha256(inherited) == before
    checkpoint = ArtifactResolver(tmp_path/'runs').checkpoint(result.checkpoint)
    assert checkpoint.effective_config.config.fingerprint == cfg.fingerprint
    assert checkpoint.effective_config.config.self_play['pcr']['cheap_simulations'] == 3

    uninterrupted_root, uninterrupted_cfg = Torus9ProductionLineage(
        tmp_path/'uninterrupted-runs').prepare(
            topology='torus9',lineage_id=root.name,parent=source,effective_config=cfg,
            experiment_id='pcr-test',arm_id='pcr-test')
    uninterrupted_request = ResolvedGenerationInput(
        source,1,uninterrupted_cfg,
        OutputLineage(topology='torus9',lineage_id=uninterrupted_root.name,root=uninterrupted_root))
    with _test_authority():
        ordinary.run_generation(uninterrupted_request)
    assert len(calls) == 2 and calls[0] == calls[1]
    with gzip.open(uninterrupted_root/'replay/g0001-0000.games.jsonl.gz','rt') as stream:
        uninterrupted_records = [json.loads(line) for line in stream]
    assert records == uninterrupted_records
    uninterrupted_identity = json.loads(
        (uninterrupted_root/'replay/g0001-0000.identity.json').read_text())
    assert committed_identity_before['request'] == uninterrupted_identity['request']
    resumed_shard = torch.load(committed_shard_path,weights_only=False)
    uninterrupted_shard = torch.load(
        uninterrupted_root/'replay/g0001-0000.pt',weights_only=False)
    for key in ('contract','actor_hash','generation','selfplay_simulations','search_mode','pcr'):
        assert resumed_shard[key] == uninterrupted_shard[key]
    assert len(resumed_shard['games']) == len(uninterrupted_shard['games'])
    for resumed_game, uninterrupted_game in zip(resumed_shard['games'],uninterrupted_shard['games']):
        assert resumed_game.keys() == uninterrupted_game.keys()
        for key in resumed_game:
            left, right = resumed_game[key], uninterrupted_game[key]
            if torch.is_tensor(left):
                assert torch.equal(left,right), key
            else:
                assert left == right, key


def test_start_outbox_and_manifest_show_resolved_pcr_without_sending(tmp_path):
    from test_orchestrator_v2_continuous_training import _runner
    from gocube_golden.notifications import NotificationStore, NotificationDispatcher, format_event
    cfg = EffectiveConfig(topology='torus9',compatibility={'input_channels':5},
        self_play={'komi':1.5,'games_per_iteration':384,'mcts_simulations':200,
                   'search_mode':'pcr','pcr':{'cheap_simulations':100,'full_simulations':500,'full_probability':.25}},
        training={'learning_rate':5e-5},replay={'generations':6,'cap':None},arena={'simulations':128})
    store = NotificationStore(tmp_path/'outbox')
    messages = []
    class Transport:
        def send(self,text):
            messages.append(text)
            return {'ok':True,'message_id':len(messages)}
    dispatcher = NotificationDispatcher(store,transport=Transport(),background=False)
    runner, _, _, _, source = _runner(tmp_path,generations=5,effective_config=cfg,notifier=dispatcher)
    runner._launch_id = 'pcr-test'
    runner._report_start(source,SimpleNamespace(config=cfg))
    events = list(store.iter_events())
    assert len(events) == 1
    assert events[0].payload['self_play']['pcr']['full_probability'] == .25
    text = format_event(events[0])
    for line in ('Self-play MCTS: PCR','Cheap: 100 sims / 75%','Full: 500 sims / 25%','Nominal mean: 200 sims'):
        assert line in text
    dispatcher.flush(1)
    assert messages == [text]
    root = runner.lineage_root
    root.mkdir(parents=True,exist_ok=True)
    (root/'manifest.json').write_text('{}')
    runner._persist_operator_metadata(source)
    tunables = json.loads((root/'manifest.json').read_text())['operator_tunables']
    assert tunables['self_play_search_mode'] == 'pcr'
    assert tunables['self_play_pcr'] == dict(cfg.self_play['pcr'])
    assert 'Nominal mean: 200 sims' in tunables['self_play_search_description']
