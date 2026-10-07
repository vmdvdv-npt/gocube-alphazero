"""Contract checks for the launch documentation; no engines or transports."""
import json
from pathlib import Path
import re
import shutil

import pytest

from gocube_golden.orchestrator_v2 import operator_guide as guide
from gocube_golden.orchestrator_v2 import operator_job
from gocube_golden.orchestrator_v2 import production_entrypoint as entry


def test_reviewed_guide_matches_supported_interfaces():
    metadata = guide.guide_metadata()
    assert metadata['reviewed_interface_sha256'] == guide.interface_fingerprint()
    assert Path(metadata['path']).name == 'ORCHESTRATOR_V2_LAUNCH_GUIDE.md'
    assert metadata['sha256'] and metadata['github'].endswith(guide.GUIDE_PATH)


def test_interface_drift_blocks_launch_instructions(tmp_path):
    for relative in (*guide.INTERFACE_PATHS, guide.GUIDE_PATH):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(guide.REPO_ROOT / relative, path)
    guide.guide_metadata(tmp_path)
    interface = tmp_path / guide.INTERFACE_PATHS[1]
    interface.write_text(interface.read_text() + '\n# changed supported interface\n')
    with pytest.raises(ValueError, match='launch guide is stale'):
        guide.guide_metadata(tmp_path)


def test_guide_command_is_read_only_and_displays_offline_mode(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(entry, 'launch_operator_job', lambda *a, **kw: pytest.fail('launch'))
    monkeypatch.setattr(entry, 'telegram_test', lambda: pytest.fail('transport'))
    monkeypatch.setattr(entry, 'drain_notifications', lambda *a, **kw: pytest.fail('transport'))
    assert entry.main(['guide']) == 0
    output = capsys.readouterr().out
    assert 'offline_replay' in output and 'B256' in output
    assert 'performance_tuning' in output and 'winner_selection' in output
    assert not list(tmp_path.iterdir())


def test_job_check_announces_guide_before_parameter_validation(capsys):
    with pytest.raises(ValueError):
        entry.launch_operator_job({'schema': 'unsupported'}, check_only=True)
    assert guide.GUIDE_PATH in capsys.readouterr().err


def test_complete_json_examples_use_supported_operator_parameters():
    examples = re.findall(r'```json\n(.*?)\n```', guide.read_guide(), re.S)
    parsed_jobs = []
    for example in examples:
        payload = json.loads(example)
        if payload.get('schema') == operator_job.SCHEMA:
            parsed_jobs.append(operator_job.parse_job(payload))
    assert len(parsed_jobs) == 6
    offline = next(job for job in parsed_jobs if job['run_id'] == 'offline-batch-example')
    assert offline['training']['iterations'] == 0
    assert set(offline['ab_tests'][0]['arms']) == {'B64', 'B128', 'B256'}
    assert {arm['batch_size'] * arm['updates_per_iteration']
            for arm in offline['ab_tests'][0]['arms'].values()} == {163840}
    single = next(job for job in parsed_jobs if job['run_id'] == 'offline-single-arm-example')
    assert single['training']['iterations'] == 5
    assert len(single['offline_replay']) == 5
    assert single['training']['replay_sampling'] == {'mode': 'policy_surprise', 'weight': 0.5}


def test_uncommitted_guide_is_included_in_launch_cleanliness_check(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def dirty_status(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=' M ' + guide.GUIDE_PATH)
    monkeypatch.setattr(entry.subprocess, 'run', dirty_status)
    with pytest.raises(ValueError, match='launch guide'):
        entry._require_committed_job_code()
    assert guide.GUIDE_PATH in calls[0]
