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


def test_two_generations_commit_replay_rollover_and_restart(parent,tmp_path,monkeypatch):
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
        self_play={'komi':1.5,'games_per_iteration':2,'mcts_simulations':1},
        training={'batch_size':64,'optimizer':'Adam','weight_decay':0.,'l2_sp':False,
                  'gradient_clip':1.,'learning_rate':5e-5,'optimizer_steps_per_iteration':2},
        replay={'cap':None,'generations':6},execution={'device':'cpu','workers':1,
                  'training_master_seed':91,'selfplay_master_seed':92},
        extensions={'training_driver':ordinary.SCHEMA,'adaptation_parent':ref.to_dict(),
                    'initial_replay_buckets':buckets,'validation_buckets':[buckets[0]]})
    runs=tmp_path/'runs'
    root,resolved_cfg=Torus9ProductionLineage(runs).prepare(topology='torus9',lineage_id='test-ordinary',
        parent=source,effective_config=cfg,experiment_id='test',arm_id='test')
    calls=[]
    def fake_selfplay(model,**kwargs):
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
    assert len(calls)==1
    validate_generation_commit(root=root,lineage_id=root.name,generation=199,reuse_committed_rolling_replay_identity=True)
    resolver=ArtifactResolver(runs)
    child=resolver.checkpoint(first.checkpoint)
    with _test_authority():
        second=ordinary.run_generation(ResolvedGenerationInput(child,200,resolved_cfg,output))
    validate_generation_commit(root=root,lineage_id=root.name,generation=200,reuse_committed_rolling_replay_identity=True)
    saved=torch.load(root/second.checkpoint.path,weights_only=False)
    assert saved['ordinary_update']==4
    assert [b['generation'] for b in saved['replay_buckets']]==[2,3,4,5,199,200]
    train=ordinary.load_replay(saved['replay_buckets'],split='train')
    assert 'heldout' not in {g['game_id'] for g in train}
    assert len(calls)==2
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
