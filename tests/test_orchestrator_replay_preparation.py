"""Isolated preparation tests; no production replay or network transports."""
import json
import os
from types import SimpleNamespace

import pytest

from gocube_golden.orchestrator_v2 import replay_preparation as prep
from gocube_golden.orchestrator_v2.execution_permit import _production_authority
from gocube_golden import policy_surprise


def test_replay_preparation_requires_authority_before_writing(tmp_path):
    with pytest.raises(RuntimeError, match='Orchestrator V2'):
        prep.prepare_replay_cache(None, spec={}, experiment_root=tmp_path / 'experiment',
                                  experiment_id='abcd', device='cpu')
    assert not list(tmp_path.iterdir())


def test_supervised_preparation_signs_child_and_reuses_verified_result(tmp_path, monkeypatch):
    root = tmp_path / 'experiment'
    runtime = SimpleNamespace(path=tmp_path / 'runtime', environment=lambda env: dict(env))
    adapter = SimpleNamespace(repo_root=tmp_path, python_executable='python',
        supervisor_policy=None, runtime_manager=SimpleNamespace(ensure=lambda commit: runtime))
    monkeypatch.setattr(prep, 'resolve_execution_commit', lambda *a: 'test-commit')
    calls = []
    def spawn(command, **kwargs):
        permit = json.loads(kwargs['env']['AZ_V2_EXECUTION_PERMIT'])
        assert permit['code_identity'] == 'test-commit' and permit['action_type'] == 'generation'
        assert permit['run_id'] == 'abcd' and permit['supervisor_pid'] == os.getpid()
        assert 'AZ_V2_EXECUTION_PERMIT_KEY' in kwargs['env']
        assert command[2] == 'gocube_golden.orchestrator_v2.replay_preparation'
        calls.append('spawn')
        return object()
    monkeypatch.setattr(prep, 'start_owned_child', spawn)
    class Supervisor:
        def __init__(self, root, **kwargs):
            self.kwargs = kwargs
        def run_once(self):
            self.kwargs['launcher'](SimpleNamespace(attempt=1))
            request = json.loads((root / 'runtime/requests/replay-preparation.json').read_text())
            assert request['spec'] == {'fingerprint': 'test-spec'}
            prep._write_json(root / 'runtime/results/replay-preparation.json',
                {'schema': prep.REQUEST_SCHEMA, 'execution_code_commit': 'test-commit',
                 'cache': {'fingerprint': 'test-spec', 'sha256': 'test-sha'}})
            return SimpleNamespace(success=True)
        def reconcile_completed_execution(self):
            calls.append('reconcile')
            return SimpleNamespace(success=True)
    monkeypatch.setattr(prep, 'SupervisorV2', Supervisor)
    def verify(cache, spec):
        assert cache['sha256'] == 'test-sha' and spec['fingerprint'] == cache['fingerprint']
        calls.append('verify')
    monkeypatch.setattr(policy_surprise, 'load_cache', verify)
    monkeypatch.setattr(policy_surprise, 'build_cache', lambda *a, **kw: pytest.fail('controller inference'))
    with _production_authority(mode='experiment', topology='torus9', run_id='abcd', code_identity='test-commit'):
        kwargs = dict(spec={'fingerprint': 'test-spec'}, experiment_root=root, experiment_id='abcd', device='cpu')
        first = prep.prepare_replay_cache(adapter, **kwargs)
        assert prep.prepare_replay_cache(adapter, **kwargs) == first
        assert calls == ['spawn', 'verify', 'verify', 'reconcile']
        path = root / 'runtime/results/replay-preparation.json'
        payload = json.loads(path.read_text()); payload['execution_code_commit'] = 'foreign'
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match='identity mismatch'):
            prep.prepare_replay_cache(adapter, **kwargs)
        assert calls == ['spawn', 'verify', 'verify', 'reconcile']


def test_preparation_worker_requires_child_permit_before_reading(tmp_path):
    with pytest.raises(RuntimeError, match='Orchestrator V2'):
        prep.run_worker(tmp_path / 'missing.json', tmp_path / 'result.json')
    assert not list(tmp_path.iterdir())
