import json
from pathlib import Path
import pytest
from gocube_golden.orchestrator_v2.execution_permit import _test_authority, require_engine_execution
from gocube_golden.orchestrator_v2 import adaptation
from gocube_golden.orchestrator_v2.adaptation import write_decision, Workflow
from gocube_golden.provenance import file_sha256, sha256_fingerprint


@pytest.mark.parametrize('gate,allow,reason,accepted', [
    ('INCONCLUSIVE', True, 'operator accepts uncertainty', True),
    ('INCONCLUSIVE', False, 'operator accepts uncertainty', False),
    ('INCONCLUSIVE', True, None, False),
    ('FAIL', True, 'operator accepts uncertainty', False),
    ('PASS', False, None, True),
])
def test_review_exception_never_changes_gate_or_admits_fail(tmp_path, gate, allow, reason, accepted):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'gate': gate}))
    digest = file_sha256(report)
    state = {'status': 'NEEDS_REVIEW', 'report': str(report), 'report_sha': digest,
             'stage': 1, 'arena_attempt': 1, 'checkpoint': 'cp', 'checkpoint_sha': 'sha'}
    (tmp_path / 'state.json').write_text(json.dumps(state))
    with _test_authority():
        if not accepted:
            with pytest.raises(ValueError):
                write_decision(tmp_path, 'continue', digest, allow_inconclusive=allow, reason=reason)
            assert not (tmp_path / 'decision.json').exists()
        else:
            write_decision(tmp_path, 'continue', digest, allow_inconclusive=allow, reason=reason)
            workflow = object.__new__(Workflow)
            workflow.root, workflow.state = tmp_path, state
            assert workflow.consume_decision()
            assert state['stage'] == 2 and state['status'] == 'READY'
            assert state['last_decision']['allow_inconclusive'] == allow
    assert file_sha256(report) == digest


def test_standard_workflow_owns_adaptation_authority(tmp_path, monkeypatch):
    from gocube_golden.orchestrator_v2 import adaptation, production_entrypoint
    calls = []
    def phase(config, *, runs_root):
        require_engine_execution('phase-test', action='training', topology='torus9')
        calls.append(config)
        return {'status': 'NEEDS_REVIEW', 'update': 480}
    monkeypatch.setattr(adaptation, 'run_workflow_phase', phase)
    spec = {'workflow_id': 'adaptation-test', 'topology': 'torus9', 'steps': [
        {'step_id': 'partial', 'action': 'adaptation_phase', 'config': {'target_stage': 2}}]}
    production_entrypoint.run_workflow_from_config(spec, runs_root=tmp_path)
    production_entrypoint.run_workflow_from_config(spec, runs_root=tmp_path)
    assert calls == [{'target_stage': 2}]


def test_code_rollover_recovery_preserves_checkpoint_and_replay(tmp_path, monkeypatch):
    root = tmp_path / 'torus9' / 'active' / 'adaptation'
    (root / 'runtime').mkdir(parents=True)
    (root / 'checkpoints').mkdir()
    (root / 'replay').mkdir()
    checkpoint = root / 'checkpoints' / 'update-0160.pt'
    replay = root / 'replay' / 'g02-000640.pt'
    checkpoint.write_bytes(b'checkpoint')
    replay.write_bytes(b'replay')
    config = {
        'schema': 'test',
        'code': {'git_commit_sha': 'source-commit'},
        'fingerprint': '',
    }
    config['fingerprint'] = sha256_fingerprint({k: v for k, v in config.items() if k != 'fingerprint'})
    state = {
        'status': 'FAILED', 'stage': 2, 'update': 160,
        'error': 'ValueError: Code changed before spawning self-play workers',
        'checkpoint': str(checkpoint), 'checkpoint_sha': file_sha256(checkpoint),
        'shards': [{'path': str(replay), 'sha': file_sha256(replay), 'generation': 2, 'games': 128}],
        'stage_start': {'update': 160},
    }
    pin = {
        'config_hash': config['fingerprint'], 'source_commit': 'source-commit',
        'execution_commit': 'old-commit', 'checkpoint_sha': state['checkpoint_sha'],
    }
    (root / 'config.json').write_text(json.dumps(config))
    (root / 'state.json').write_text(json.dumps(state))
    (root / 'runtime' / 'execution-pin.json').write_text(json.dumps(pin))
    monkeypatch.setattr(adaptation, 'REPO', tmp_path)
    monkeypatch.setattr(adaptation, 'capture_code_identity', lambda: type('Identity', (), {'git_commit_sha': 'new-commit'})())
    monkeypatch.setattr(adaptation.subprocess, 'run', lambda *args, **kwargs: type(
        'Result', (), {'returncode': 0,
                       'stdout': 'gocube_golden/orchestrator_v2/adaptation.py\n'
                                  'tests/test_adaptation_v2_migration.py\n'})())

    recovered = adaptation.recover_code_rollover(
        root,
        expected_error='ValueError: Code changed before spawning self-play workers',
    )

    assert recovered['status'] == 'READY'
    assert recovered['update'] == 160
    assert recovered['checkpoint_sha'] == state['checkpoint_sha']
    assert json.loads((root / 'runtime' / 'execution-pin.json').read_text())['execution_commit'] == 'new-commit'
    assert json.loads((root / 'state.json').read_text())['recovery']['replay_preserved'] is True


def test_code_rollover_recovery_rejects_changed_tree(tmp_path, monkeypatch):
    root = tmp_path / 'adaptation'
    (root / 'runtime').mkdir(parents=True)
    checkpoint = root / 'checkpoint.pt'
    replay = root / 'replay.pt'
    checkpoint.write_bytes(b'checkpoint')
    replay.write_bytes(b'replay')
    config = {'code': {'git_commit_sha': 'source'}, 'fingerprint': 'config'}
    state = {
        'status': 'FAILED', 'error': 'ValueError: Code changed before spawning self-play workers',
        'checkpoint': str(checkpoint), 'checkpoint_sha': file_sha256(checkpoint),
        'shards': [{'path': str(replay), 'sha': file_sha256(replay)}],
    }
    (root / 'config.json').write_text(json.dumps(config))
    (root / 'state.json').write_text(json.dumps(state))
    (root / 'runtime' / 'execution-pin.json').write_text(json.dumps({
        'config_hash': 'config', 'source_commit': 'source', 'execution_commit': 'old',
        'checkpoint_sha': state['checkpoint_sha'],
    }))
    monkeypatch.setattr(adaptation, 'REPO', tmp_path)
    monkeypatch.setattr(adaptation, 'capture_code_identity', lambda: type('Identity', (), {'git_commit_sha': 'new'})())
    monkeypatch.setattr(adaptation.subprocess, 'run', lambda *args, **kwargs: type(
        'Result', (), {'returncode': 0, 'stdout': 'gocube_golden/model.py\n'})())
    with pytest.raises(ValueError, match='unapproved files'):
        adaptation.recover_code_rollover(root, expected_error=state['error'])
    assert json.loads((root / 'state.json').read_text())['status'] == 'FAILED'
