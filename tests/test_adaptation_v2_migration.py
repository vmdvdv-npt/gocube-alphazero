import json
from pathlib import Path
import pytest
from gocube_golden.orchestrator_v2.execution_permit import _test_authority, require_engine_execution
from gocube_golden.orchestrator_v2.adaptation import write_decision, Workflow
from gocube_golden.provenance import file_sha256


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
