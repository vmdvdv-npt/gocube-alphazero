import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gocube_golden import torus9_five_channel_training as ordinary
from gocube_golden.artifact_graph import (
    CheckpointRef, EffectiveConfig, validate_generation_commit,
)
from gocube_golden.artifact_resolver import ArtifactResolver
from gocube_golden.neural import model_hash
from gocube_golden.orchestrator_v2.execution_permit import _test_authority
from gocube_golden.orchestrator_v2.generation_runner import OutputLineage, ResolvedGenerationInput
from gocube_golden.orchestrator_v2.torus9_production import Torus9ProductionLineage, _default_driver
from gocube_golden.process_supervision import atomic_write_json
from gocube_golden.provenance import file_sha256
from gocube_golden.torus9_adaptation import AdaptationModel, FINGERPRINT, activation, save_torch


def game(identity, split, actor):
    from gocube_golden import torus9_monolith as core
    from gocube_golden.torus9_m137_5ch import build_m137_five_channel_observation
    obs = build_m137_five_channel_observation(core.initial_state(topology=core.TORUS_9X9,komi=1.5))
    return {'game_id':identity,'split':split,'actor_hash':actor,'contract':FINGERPRINT,
        'observation':obs.unsqueeze(0).repeat(3,1,1),'pi':torch.full((3,82),1/82),
        'z':torch.tensor([[1.,0.,0.]]*3),'ownership':torch.zeros(3,81,dtype=torch.long),
        'score':torch.zeros(3),'legal':torch.ones(3,82,dtype=torch.bool),
        'visits':torch.ones(3,82,dtype=torch.long)}


@pytest.fixture
def parent(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = AdaptationModel()
    opt = torch.optim.Adam([{'params':[p],'name':n,'lr':1e-5} for n,p in model.named_parameters()])
    for n,p in model.named_parameters():
        opt.state[p] = {'step':torch.tensor(float((0 if n=='input_projection.bias' else 15520)+2400-activation(n))),
                        'exp_avg':torch.full_like(p,.0001),'exp_avg_sq':torch.full_like(p,.0002)}
    path = tmp_path/'source.pt'
    meta = {'architecture_id':model.architecture_id,'architecture_config':model.architecture_config,
            'observation_shape':[5,81],'model_hash':model_hash(model),'target_fingerprint':FINGERPRINT,'komi':1.5}
    save_torch(path,{'metadata':meta,'update':2400,'model_state_dict':model.state_dict(),
                     'optimizer_state_dict':opt.state_dict()})
    return path


def test_transition_preserves_moments_and_all_clocks_with_deterministic_resume(parent,tmp_path):
    raw = torch.load(parent,weights_only=False)
    trainer = ordinary.OrdinaryTrainer(parent,learning_rate=5e-5,seed=91)
    for group, source in zip(trainer.optimizer.param_groups,raw['optimizer_state_dict']['param_groups']):
        assert group['lr']==5e-5
        for key in ('step','exp_avg','exp_avg_sq'):
            assert torch.equal(trainer.optimizer.state[group['params'][0]][key],
                               raw['optimizer_state_dict']['state'][source['params'][0]][key])
    original = {n:p.clone() for n,p in trainer.model.named_parameters()}
    games=[game('train','train',model_hash(trainer.model))]
    with _test_authority():
        metrics=trainer.step(games)
    assert metrics['l2_sp_coefficient']==0
    for n,p in trainer.model.named_parameters():
        assert p.requires_grad
        assert not torch.equal(p,original[n]),n
        assert trainer.optimizer.state[p]['step']==trainer.clock_origin[n]+1
    saved=tmp_path/'ordinary.pt'
    trainer.save(saved,config_hash='cfg',parent={},replay_buckets=[])
    resumed=ordinary.OrdinaryTrainer(saved,learning_rate=5e-5,seed=91)
    with _test_authority():
        trainer.step(games)
        resumed.step(games)
    assert model_hash(trainer.model)==model_hash(resumed.model)
    with pytest.raises(ValueError,match='seed'):
        ordinary.OrdinaryTrainer(saved,learning_rate=5e-5,seed=92)


def test_gradient_clip_is_forwarded_to_torch(parent, monkeypatch):
    trainer = ordinary.OrdinaryTrainer(parent, learning_rate=5e-5, seed=91, gradient_clip=8.0)
    games = [game('train', 'train', model_hash(trainer.model))]
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def capture(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', capture)
    with _test_authority():
        metrics = trainer.step(games)
    assert calls == [8.0]
    assert metrics['gradient_clip'] == 8.0


def test_legacy_rejected_before_driver_or_workers():
    config=EffectiveConfig(topology='torus9',compatibility={'input_channels':6})
    resolved=SimpleNamespace(effective_config=SimpleNamespace(config=config))
    with pytest.raises(ValueError,match='retired'):
        _default_driver(resolved)


@pytest.mark.parametrize('tree_reuse', [False, True])
@pytest.mark.parametrize('offline', [False, True])
def test_two_generations_commit_replay_rollover_and_restart(parent,tmp_path,monkeypatch,tree_reuse,offline):
    raw=torch.load(parent,weights_only=False)
    actor=raw['metadata']['model_hash']
    buckets=[]
    for i in range(6):
        p=tmp_path/f'old-{i}.pt'
        games=[game(f'old-{i}','train',actor)]
        if i==0:
            games.append(game('heldout','validation',actor))
        save_torch(p,{'contract':FINGERPRINT,'actor_hash':actor,'games':games})
        buckets.append({'generation':i,'shards':[{'path':str(p),'sha':file_sha256(p)}]})
    raw['metadata']['replay_ids']=[s['sha'] for b in buckets for s in b['shards']]
    save_torch(parent,raw)
    ref=CheckpointRef('torus9','source','update-2400',198,'source.pt',file_sha256(parent))
    source=SimpleNamespace(ref=ref,path=parent,generation=198)
    cfg=EffectiveConfig(topology='torus9',compatibility={'input_channels':5},
        self_play={'komi':1.5,'games_per_iteration':2,'mcts_simulations':1,'tree_reuse':tree_reuse},
        training={'batch_size':128,'optimizer':'Adam','weight_decay':0.,'l2_sp':False,
                  'gradient_clip':1.,'learning_rate':5e-5,'optimizer_steps_per_iteration':2},
        replay={'cap':None,'generations':6},execution={'device':'cpu','workers':1,
                  'training_master_seed':91,'selfplay_master_seed':92},
        extensions={'training_driver':ordinary.SCHEMA,'adaptation_parent':ref.to_dict(),
                    'initial_replay_buckets':buckets,'validation_buckets':[buckets[0]]})
    if offline:
        rows = []
        for g in (199, 200):
            fresh = {'generation': g, 'shards': buckets[-1]['shards']}
            rolling = buckets[:5] + [fresh]
            fp, rp = tmp_path/f'fresh-{g}.json', tmp_path/f'rolling-{g}.jsonl'
            fp.write_text(json.dumps(fresh))
            rp.write_text(''.join(json.dumps(b)+'\n' for b in rolling))
            rows.append({'generation': g, 'fresh_bucket': fresh, 'buckets': rolling,
                         'fresh_replay': {'path': str(fp), 'sha256': file_sha256(fp)},
                         'rolling_replay': {'path': str(rp), 'sha256': file_sha256(rp)}})
        payload = cfg.to_dict()
        payload['extensions']['offline_ab_replay'] = rows
        cfg = EffectiveConfig.from_dict(payload)
    runs=tmp_path/'runs'
    root,resolved_cfg=Torus9ProductionLineage(runs).prepare(topology='torus9',lineage_id='test-ordinary',
        parent=source,effective_config=cfg,experiment_id='test',arm_id='test')
    calls=[]
    def fake_selfplay(model,**kwargs):
        assert not offline, 'offline must never call selfplay'
        assert kwargs.get('tree_reuse', False) is tree_reuse
        calls.append(kwargs['ids'])
        records=[SimpleNamespace(to_dict=lambda:{}, payload=game(i,'train',model_hash(model))) for i in kwargs['ids']]
        return SimpleNamespace(records=records,telemetry={})
    monkeypatch.setattr(ordinary,'selfplay',fake_selfplay)
    monkeypatch.setattr(ordinary,'game_targets',lambda g:copy.deepcopy(g.payload))
    output=OutputLineage('torus9',root.name,root)
    # Fail after durable collection, then retry: committed shards must be reused.
    original_step=ordinary.OrdinaryTrainer.step
    monkeypatch.setattr(ordinary.OrdinaryTrainer,'step',lambda *a:(_ for _ in ()).throw(RuntimeError('interruption')))
    request=ResolvedGenerationInput(source,199,resolved_cfg,output)
    with _test_authority(),pytest.raises(RuntimeError,match='interruption'):
        ordinary.run_generation(request)
    assert not (root/'generation-199.complete.json').exists()
    monkeypatch.setattr(ordinary.OrdinaryTrainer,'step',original_step)
    with _test_authority():
        first=ordinary.run_generation(request)
    assert len(calls)==(0 if offline else 1)
    validate_generation_commit(root=root,lineage_id=root.name,generation=199,reuse_committed_rolling_replay_identity=True)
    resolver=ArtifactResolver(runs)
    child=resolver.checkpoint(first.checkpoint)
    with _test_authority():
        second=ordinary.run_generation(ResolvedGenerationInput(child,200,resolved_cfg,output))
    validate_generation_commit(root=root,lineage_id=root.name,generation=200,reuse_committed_rolling_replay_identity=True)
    saved=torch.load(root/second.checkpoint.path,weights_only=False)
    assert saved['ordinary_update']==4
    assert saved['metadata']['batch_size'] == 128
    assert [b['generation'] for b in saved['replay_buckets']]==([0,1,2,3,4,200] if offline else [2,3,4,5,199,200])
    telemetry=json.loads((root/'training/iter-200.json').read_text())['performance']
    assert telemetry['processed_samples']==256
    assert telemetry['training_minutes'] > 0
    train=ordinary.load_replay(saved['replay_buckets'],split='train')
    assert 'heldout' not in {g['game_id'] for g in train}
    assert len(calls)==(0 if offline else 2)
    from tools.arena_profiles import get_profile
    profile=get_profile('torus9|komi=1.5|simulations=128|cpuct=1.25|fpu=0|watchdog=1000|5ch')
    identity=profile.load_identity(root/second.checkpoint.path)
    assert identity.model_hash==saved['metadata']['model_hash']
    loaded_model=profile.load_parent_model(identity,torch.device('cpu'))
    assert model_hash(loaded_model)==saved['metadata']['model_hash']
    ordinary.write_block_report(SimpleNamespace(state='COMPLETED',lineage_root=root,
        original_parent=source,final_checkpoint=resolver.checkpoint(second.checkpoint),arenas=[]))
    report=json.loads((root/'reports/block-report.json').read_text())
    assert len(report['iterations'])==2
    assert report['checkpoint']==second.checkpoint.to_dict()
    # A tampered replay shard is rejected before it can enter another iteration.
    shard=Path(saved['replay_buckets'][0]['shards'][0]['path'])
    shard.write_bytes(b'corruption')
    with pytest.raises(ValueError,match='SHA'):
        ordinary.load_replay(saved['replay_buckets'],split='train')


def test_batch128_trains_and_resumes(parent, tmp_path):
    trainer = ordinary.OrdinaryTrainer(parent, learning_rate=2.5e-5, seed=91, batch_size=128)
    games = [game('train', 'train', model_hash(trainer.model))]
    assert all(value.shape[0] == 128 for value in trainer.batch(games, 1).values())
    with _test_authority():
        assert trainer.step(games)['batch_size'] == 128
    saved = tmp_path / 'batch128.pt'
    trainer.save(saved, config_hash='cfg', parent={}, replay_buckets=[])
    resumed = ordinary.OrdinaryTrainer(saved, learning_rate=2.5e-5, seed=91)
    assert resumed.batch_size == 128
    with _test_authority():
        trainer.step(games)
        resumed.step(games)
    assert model_hash(trainer.model) == model_hash(resumed.model)


def test_profile_spans_preserve_exact_b64_trajectory(parent):
    from gocube_golden.training_profile import Collector
    from gocube_golden.b64_perf_audit import exact_equal
    trainer = ordinary.OrdinaryTrainer(parent, learning_rate=5e-5, seed=91, batch_size=64)
    profiled = copy.deepcopy(trainer)
    games = [game('train-a', 'train', model_hash(trainer.model)),
             game('train-b', 'train', model_hash(trainer.model))]
    observer = Collector(timings=False, capture_positions=True)
    collector = Collector(capture_positions=True)
    with _test_authority(), observer.activate():
        plain_metrics = [trainer.step(games) for _ in range(3)]
    with _test_authority(), collector.activate():
        profile_metrics = [profiled.step(games) for _ in range(3)]
    assert observer.rows == []
    assert observer.positions == collector.positions
    assert all(len(positions) == 64 for positions in observer.positions)
    assert plain_metrics == profile_metrics
    assert trainer.update == profiled.update == 3
    assert exact_equal(trainer.model.state_dict(), profiled.model.state_dict())
    assert exact_equal(trainer.optimizer.state_dict(), profiled.optimizer.state_dict())
    assert {'validate.pre', 'validate.post', 'forward', 'adam', 'scalar.telemetry'} <= collector.finish().keys()


@pytest.mark.parametrize('corruption', ['clock', 'average_nan', 'variance_inf', 'negative', 'shape',
                                        'missing_step', 'missing_average', 'multiple_errors'])
def test_packed_adam_validation_preserves_first_error(parent, corruption):
    trainer = ordinary.OrdinaryTrainer(parent, learning_rate=5e-5, seed=91)
    params = list(trainer.model.named_parameters())
    first = trainer.optimizer.state[params[0][1]]
    second = trainer.optimizer.state[params[1][1]]
    if corruption == 'clock':
        first['step'].add_(1)
    elif corruption == 'average_nan':
        first['exp_avg'].fill_(float('nan'))
    elif corruption == 'variance_inf':
        first['exp_avg_sq'].fill_(float('inf'))
    elif corruption == 'negative':
        first['exp_avg_sq'].fill_(-1)
    elif corruption == 'shape':
        first['exp_avg'] = first['exp_avg'].reshape(-1)[:1]
    elif corruption == 'missing_step':
        del first['step']
    elif corruption == 'missing_average':
        del first['exp_avg']
    else:
        first['exp_avg'].fill_(float('nan'))
        second['step'].add_(1)
    with pytest.raises(Exception) as original:
        trainer._validate_clocks_detailed()
    with pytest.raises(type(original.value)) as packed:
        trainer.validate_clocks()
    assert str(packed.value) == str(original.value)


def test_cached_replay_preserves_sampling_and_mutable_callers(parent):
    from gocube_golden.training_profile import Collector
    trainer = ordinary.OrdinaryTrainer(parent, learning_rate=5e-5, seed=91)
    a = game('a', 'train', model_hash(trainer.model))
    b = game('b', 'train', model_hash(trainer.model))
    a['score'] = torch.arange(3, dtype=torch.float32)
    b['score'] = torch.arange(3, dtype=torch.float32) + 100
    mutable = [a, b]
    window = ordinary.OrdinaryReplayWindow.from_games(mutable)
    assert window.games[0] is a  # Targets are referenced, never copied/prepacked.
    assert copy.deepcopy(window) is window
    for update in (1,2,31,128,255,1024):
        left, right = Collector(timings=False,capture_positions=True), Collector(timings=False,capture_positions=True)
        with left.activate():
            expected = trainer.batch(mutable,update)
        with right.activate():
            actual = trainer.batch(window,update)
        assert left.positions == right.positions
        assert all(torch.equal(expected[k],actual[k]) for k in expected)
    mutable.append(game('c','train',model_hash(trainer.model)))
    refreshed = ordinary.OrdinaryReplayWindow.from_games(mutable)
    assert len(window) == 2 and len(refreshed) == 3
    assert all(torch.equal(v,trainer.batch(refreshed,2)[k]) for k,v in trainer.batch(mutable,2).items())


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_packed_weight_check_retains_fail_closed_behavior(parent, monkeypatch, invalid):
    trainer=ordinary.OrdinaryTrainer(parent,learning_rate=5e-5,seed=91)
    games=[game('train','train',model_hash(trainer.model))]
    real_step=trainer.optimizer.step
    def corrupt_after_update():
        real_step()
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(invalid)
    monkeypatch.setattr(trainer.optimizer,'step',corrupt_after_update)
    with _test_authority(), pytest.raises(FloatingPointError,match='Nonfinite ordinary weights'):
        trainer.step(games)
    assert trainer.update == 1
