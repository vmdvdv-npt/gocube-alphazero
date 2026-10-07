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
