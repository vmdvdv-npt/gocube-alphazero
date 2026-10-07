from types import SimpleNamespace as NS
import pytest
from gocube_golden.scenarios.experiment.runner import ExperimentRunnerV2


def test_offline_round_robin_has_three_independent_pairs_and_common_settings(tmp_path):
    runner = ExperimentRunnerV2.__new__(ExperimentRunnerV2)
    runner.experiment_root = tmp_path
    runner.config = NS(parent={}, fingerprint='stable', experiment_id='offline',
        arms=[NS(arm_id=a, generations=1, effective_config=NS(training={"batch_size": int(a[1:])})) for a in ('B64', 'B128', 'B256')],
        arena_config=NS(games=192), arena_master_seed=17, arena_startset=None,
        arena_profile='torus9', arena_scientific_contract=None,
        arena_execution_contract=None, arena_workload={})
    runner.resolver = NS(checkpoint=lambda _: NS(generation=0))
    runner._notify_operator = lambda *a, **kw: None
    import json
    (tmp_path / "training").mkdir()
    (tmp_path / "training/iter-01.json").write_text(json.dumps({
        "performance": {"training_minutes": 1.0}, "validation": {}}))
    runner._run_arms = lambda *a, **kw: {name: NS(owner_root=tmp_path, ref=NS(to_dict=lambda: {})) for name in kw['arms']}
    calls = []
    def arena(**kw):
        calls.append(kw)
        return NS(validity='VALID', evaluation_id=kw['comparison'], summary={}, output_dir=tmp_path)
    runner._run_arena = arena
    result = runner._run_offline()
    assert result['state'] == 'STOPPED'
    assert [(x['candidate_label'], x['reference_label']) for x in calls] == [
        ('B64', 'B128'), ('B64', 'B256'), ('B128', 'B256')]
    assert all(x['arena_master_seed'] == 17 and x['arena_config'].games == 192 for x in calls)
    assert all(x['arena_workload']['paired_starts'] and x['arena_workload']['color_swap'] for x in calls)
    runner.config.fingerprint = 'changed'
    with pytest.raises(RuntimeError, match='configuration changed'):
        runner._run_offline()


def test_surprise_cache_sealed_before_arms_reused_and_checked_on_resume(tmp_path, monkeypatch):
    import json
    from gocube_golden import policy_surprise as ps
    from gocube_golden.artifact_graph import EffectiveConfig
    from gocube_golden.orchestrator_v2.experiment_plan import ExperimentArmConfig
    runner = ExperimentRunnerV2.__new__(ExperimentRunnerV2)
    runner.experiment_root = tmp_path
    cfg = EffectiveConfig(topology='torus9', compatibility={'input_channels':5}, training={'batch_size':64},
        replay={'sampling':{'mode':'policy_surprise','weight':0.5}},
        execution={'device':'cpu'}, extensions={'policy_surprise_spec':{'fingerprint':'spec'}})
    runner.config = NS(parent={},fingerprint='stable',experiment_id='offline',
        arms=[ExperimentArmConfig(a,1,cfg) for a in ('S50','U64')],
        arena_config=NS(games=192),arena_master_seed=17,arena_startset=None,
        arena_profile='torus9',arena_scientific_contract=None,
        arena_execution_contract=None,arena_workload={})
    runner.resolver=NS(checkpoint=lambda _:NS(generation=0))
    runner._notify_operator=lambda *a, **kw:None
    cache={'path':'synthetic','sha256':'sha256:cache','fingerprint':'spec'}
    builds=[]; reads=[]
    monkeypatch.setattr(ps,'build_cache',lambda *a, **kw:builds.append(kw) or cache)
    monkeypatch.setattr(ps,'load_cache',lambda ref,spec:reads.append(ref) or {})
    def interrupted(*a, **kw):
        assert all(arm.effective_config.extensions['policy_surprise_cache']==cache
                   for arm in kw['arms'].values())
        raise RuntimeError('interrupted')
    runner._run_arms=interrupted
    with pytest.raises(RuntimeError,match='interrupted'):
        runner._run_offline()
    assert json.loads(runner.state_path.read_text())['policy_surprise_cache']==cache
    (tmp_path/'training').mkdir()
    (tmp_path/'training/iter-01.json').write_text(json.dumps({'performance':{'training_minutes':1},'validation':{}}))
    runner._run_arms=lambda *a, **kw:{name:NS(owner_root=tmp_path,ref=NS(to_dict=lambda:{})) for name in kw['arms']}
    runner._run_arena=lambda **kw:NS(validity='VALID',evaluation_id='arena',summary={},output_dir=tmp_path)
    assert runner._run_offline()['state']=='STOPPED'
    assert len(builds)==1 and len(reads)==2
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['arms']['S50']['sampling']=={'mode':'policy_surprise','weight':0.5}
    def tampered(*a):
        raise ValueError('cache SHA mismatch')
    monkeypatch.setattr(ps,'load_cache',tampered)
    with pytest.raises(ValueError,match='SHA'):
        runner._run_offline()
