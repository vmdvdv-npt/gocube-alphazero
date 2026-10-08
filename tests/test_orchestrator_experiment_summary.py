"""Isolated summary publication: real durable store, mocked transport only."""
import hashlib
import json

import pytest

from gocube_golden.notifications import NotificationDispatcher, NotificationStore
from gocube_golden.notifications.formatter import format_event
from gocube_golden.orchestrator_v2 import experiment_summary as summary


def ref(path):
    return {'path': str(path), 'sha256': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def saved(tmp_path):
    evidence = tmp_path / 'arena.json'
    evidence.write_text('{"valid_games": 256, "technical_games": 0}')
    report = {'schema': 'gocube-experiment-conclusion-v1', 'status': 'COMPLETED',
        'combinations': [{'id': name, 'learning_rate': lr, 'updates_per_generation': updates,
                          'seed_results': {'2026092701': 'screening 50%', '2026092702': 'screening 51%'}}
                         for name, lr, updates in [('C1280', .0001, 1280), ('C2560', .0001, 2560),
                                                   ('D1280', .0002, 1280), ('D2560', .0002, 2560)]],
        'confirmation_results': {'2026092701': '512/512/0, 1024 games × 200 sims',
                                 '2026092702': '510/514/0, 1024 games × 200 sims'},
        'decision': 'INCONCLUSIVE', 'rationale': 'Both independent intervals include 50%.',
        'evidence': [ref(evidence)]}
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    config = {'schema': 'gocube-experiment-summary-v1', 'summary_id': 'az20', 'report': ref(path)}
    return config, path, evidence


def test_check_has_no_writes_or_transport(tmp_path, saved, monkeypatch):
    config, _, _ = saved
    monkeypatch.setattr(summary, 'create_telegram_dispatcher', lambda *a: pytest.fail('transport'))
    root = tmp_path / 'runs'
    assert summary.publish_experiment_summary(config, runs_root=root, check_only=True)['state'] == 'VALIDATED'
    assert not root.exists()


def test_public_cli_check(tmp_path, saved, monkeypatch, capsys):
    from gocube_golden.orchestrator_v2 import production_entrypoint as entry
    config, _, _ = saved
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(summary, 'create_telegram_dispatcher', lambda *a: pytest.fail('transport'))
    assert entry.main(['experiment-summary', str(path), '--runs-root', str(tmp_path / 'runs'), '--check']) == 0
    assert json.loads(capsys.readouterr().out)['state'] == 'VALIDATED'
    assert not (tmp_path / 'runs').exists()


def test_one_durable_event_and_one_send_across_repeated_publication(tmp_path, saved, monkeypatch):
    config, _, _ = saved
    messages = []
    class Transport:
        def send(self, text):
            messages.append(text)
            return {'message_id': 1}
    monkeypatch.setattr(summary, 'create_telegram_dispatcher', lambda root:
        NotificationDispatcher(NotificationStore(root), transport=Transport(), background=False))
    root = tmp_path / 'runs'
    first = summary.publish_experiment_summary(config, runs_root=root)
    second = summary.publish_experiment_summary(config, runs_root=root)
    assert first['state'] == second['state'] == 'DELIVERED'
    assert first['event_id'] == second['event_id']
    store = NotificationStore(first['owner_root'])
    events = list(store.iter_events())
    assert len(events) == len(messages) == 1
    text = format_event(events[0])
    assert all(name in text for name in ['C1280', 'C2560', 'D1280', 'D2560'])
    assert '2026092701' in text and '2026092702' in text and 'INCONCLUSIVE' in text
    assert 'Both independent intervals' in text and '200 sims' in text
    assert len(text) < 3900


def test_changed_evidence_fails_before_publication(tmp_path, saved, monkeypatch):
    config, _, evidence = saved
    evidence.write_text('{}')
    monkeypatch.setattr(summary, 'create_telegram_dispatcher', lambda *a: pytest.fail('transport'))
    with pytest.raises(ValueError, match='SHA mismatch'):
        summary.publish_experiment_summary(config, runs_root=tmp_path / 'runs')
    assert not (tmp_path / 'runs').exists()


def test_incomplete_report_and_repurposed_id_rejected(tmp_path, saved, monkeypatch):
    config, path, _ = saved
    monkeypatch.setattr(summary, 'create_telegram_dispatcher', lambda root:
        NotificationDispatcher(NotificationStore(root), enabled=False, background=False))
    summary.publish_experiment_summary(config, runs_root=tmp_path / 'runs')
    report = json.loads(path.read_text())
    report['rationale'] = 'Different conclusion.'
    path.write_text(json.dumps(report))
    config['report'] = ref(path)
    with pytest.raises(ValueError, match='different immutable report'):
        summary.publish_experiment_summary(config, runs_root=tmp_path / 'runs', check_only=True)
    report['status'] = 'RUNNING'
    path.write_text(json.dumps(report))
    config['report'] = ref(path)
    with pytest.raises(ValueError, match='completed report'):
        summary.publish_experiment_summary(config, runs_root=tmp_path / 'other')


@pytest.mark.parametrize('mutation, message', [
    (lambda r: r.update(decision='UNKNOWN'), 'decision'),
    (lambda r: r['combinations'][0]['seed_results'].pop('2026092701'), 'two learner seeds'),
    (lambda r: r['confirmation_results'].pop('2026092701'), 'confirmation'),
])
def test_rejects_inconsistent_conclusion(tmp_path, saved, monkeypatch, mutation, message):
    config, path, _ = saved
    report = json.loads(path.read_text())
    mutation(report)
    path.write_text(json.dumps(report))
    config['report'] = ref(path)
    monkeypatch.setattr(summary, 'create_telegram_dispatcher', lambda *a: pytest.fail('transport'))
    with pytest.raises(ValueError, match=message):
        summary.publish_experiment_summary(config, runs_root=tmp_path / 'runs')
