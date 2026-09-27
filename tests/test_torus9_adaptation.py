from pathlib import Path
import copy
import json

import pytest
import torch

from gocube_golden import torus9_monolith as core
from gocube_golden.torus9_adaptation import (
    AdaptationTrainer, AdaptationSearchContract, FINGERPRINT, activation,
    game_targets, validate_game, lr_for,
)
from gocube_golden.orchestrator_v2.adaptation import (
    BOOTSTRAP, Workflow, arena_statistics, write_decision,
)
from gocube_golden.process_supervision import atomic_write_json
from gocube_golden.provenance import file_sha256


def data():
    state = core.initial_state(topology=core.TORUS_9X9, komi=1.5)
    from gocube_golden.torus9_m137_5ch import build_m137_five_channel_observation
    obs = build_m137_five_channel_observation(state)
    return [{'game_id': 'synthetic-test', 'contract': FINGERPRINT,
        'observation': obs.unsqueeze(0).repeat(3, 1, 1),
        'pi': torch.full((3, 82), 1/82), 'z': torch.tensor([[1., 0., 0.]] * 3),
        'ownership': torch.zeros(3, 81, dtype=torch.long), 'score': torch.zeros(3),
        'legal': torch.ones(3, 82, dtype=torch.bool), 'visits': torch.ones(3, 82, dtype=torch.long)}]


@pytest.fixture
def trainer():
    if not BOOTSTRAP.exists():
        pytest.skip('Local canonical checkpoint audit fixture unavailable')
    torch.set_num_threads(1)
    return AdaptationTrainer(BOOTSTRAP)


def test_migration_exact_except_bias(trainer):
    raw = torch.load(BOOTSTRAP, map_location='cpu', weights_only=False)['optimizer_state_dict']
    for (n, p), sid in zip(trainer.model.named_parameters(), raw['param_groups'][0]['params']):
        state = trainer.optimizer.state[p]
        for k, old in raw['state'][sid].items():
            expected = torch.zeros_like(old) if n == 'input_projection.bias' else old
            assert torch.equal(state[k], expected)


def test_freeze_and_deterministic_resume(trainer, tmp_path):
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    states = {n: copy.deepcopy(trainer.optimizer.state[p]) for n, p in trainer.model.named_parameters()}
    trainer.step(data())
    for n, p in trainer.model.named_parameters():
        if activation(n) > 0:
            assert torch.equal(p, before[n])
            for k in states[n]:
                assert torch.equal(trainer.optimizer.state[p][k], states[n][k])
    path = tmp_path / 'resume.pt'
    trainer.save(path, replay_ids=['abc'], config_hash='cfg')
    trainer.step(data())
    resumed = AdaptationTrainer(BOOTSTRAP)
    resumed.restore(path, config_hash='cfg', replay_ids=['abc'])
    resumed.step(data())
    for (n, p), (_, q) in zip(trainer.model.named_parameters(), resumed.model.named_parameters()):
        assert torch.equal(p, q), n
    with pytest.raises(ValueError, match='identity'):
        resumed.restore(path, config_hash='cfg', replay_ids=['different'])


def test_phase_clocks_and_lr_on_load(trainer, tmp_path):
    # Use clock offsets to exercise both boundaries without 480 redundant updates.
    for boundary in (160, 480):
        trainer.update = boundary
        for n, p in trainer.model.named_parameters():
            trainer.optimizer.state[p]['step'].fill_((0 if n == 'input_projection.bias' else 15520)
                                                   + max(0, boundary - activation(n)))
        path = tmp_path / f'{boundary}.pt'
        trainer.save(path, replay_ids=[], config_hash='test')
        resumed = AdaptationTrainer(BOOTSTRAP)
        resumed.restore(path, replay_ids=[], config_hash='test')
        resumed.step(data())
        for group in resumed.optimizer.param_groups:
            assert group['lr'] == lr_for(group['name'], boundary + 1)
        resumed.validate_clocks()
        if boundary == 480:
            assert int(resumed.optimizer.state[resumed.model.input_projection.bias]['step']) == 1


def test_replay_rejects_old_and_invalid():
    g = data()[0]
    validate_game(g)
    old = {**g, 'contract': 'old-0.5'}
    with pytest.raises(ValueError, match='parent history'):
        validate_game(old)
    g['z'][0, 0] = .5
    with pytest.raises(ValueError, match='WDL'):
        validate_game(g)
    with pytest.raises(ValueError, match='komi'):
        AdaptationSearchContract(komi=.5).validate()


def test_score_komi_side_and_observation():
    from gocube_golden.torus9_m137_5ch import build_m137_five_channel_observation
    starts = [core.initial_state(topology=core.TORUS_9X9, komi=k) for k in (.5, 1.5)]
    assert torch.equal(*[build_m137_five_channel_observation(s) for s in starts])
    ends = [core.apply_action(core.apply_action(s, core.PASS).after, core.PASS).after for s in starts]
    for side, delta in [('BLACK', -1.), ('WHITE', 1.)]:
        assert core.torus9_score_target(ends[1], side) - core.torus9_score_target(ends[0], side) == delta
        assert core.torus9_ownership_target(ends[0], side) == core.torus9_ownership_target(ends[1], side)


def test_decision_bound_to_report_and_no_duplicate(tmp_path):
    report = tmp_path / 'report.json'
    atomic_write_json(report, {'gate': 'PASS'})
    digest = file_sha256(report)
    atomic_write_json(tmp_path / 'state.json', {'status': 'NEEDS_REVIEW', 'stage': 0,
        'report_sha': digest, 'report': str(report), 'arena_attempt': 0})
    with pytest.raises(ValueError):
        write_decision(tmp_path, 'continue', 'stale')
    write_decision(tmp_path, 'continue', digest)
    with pytest.raises(ValueError, match='already pending'):
        write_decision(tmp_path, 'continue', digest)


def test_arena_pairs_and_gate(tmp_path):
    rows = [{'pair_id': str(i), 'candidate_black': color, 'mapped_result': 'A_WIN'}
            for i in range(8) for color in (True, False)]
    (tmp_path / 'games.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    assert arena_statistics(tmp_path)['gate'] == 'PASS'
    rows.pop()
    (tmp_path / 'games.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match='pairs'):
        arena_statistics(tmp_path)


def test_stage_report_pause_and_notification_retry(tmp_path, monkeypatch):
    from gocube_golden.notifications import NotificationDispatcher, NotificationStore
    from gocube_golden.notifications.telegram import TelegramTransportError
    from gocube_golden.orchestrator_v2.adaptation import init_run
    if not BOOTSTRAP.exists():
        pytest.skip('Canonical local fixture missing')
    init_run(tmp_path)
    class Transport:
        def __init__(self):
            self.calls = 0
        def send(self, text):
            self.calls += 1
            if self.calls == 1:
                raise TelegramTransportError('temporary-test', retryable=True, retry_after=0)
            return {'message_id': 7}
    store = NotificationStore(tmp_path)
    # Use the dispatcher with a fake network: no actual test messages.
    from gocube_golden.notifications.dispatcher import DeliveryPolicy
    sender = Transport()
    dispatcher = NotificationDispatcher(store, transport=sender, background=False,
        policy=DeliveryPolicy(base_delay_seconds=0))
    monkeypatch.setattr('gocube_golden.orchestrator_v2.adaptation.create_telegram_dispatcher', lambda _: dispatcher)
    workflow = Workflow(tmp_path)
    workflow.finish({'gate': 'PASS', 'validation': {'wdl': 1.}, 'arena': None}, 'Review first')
    assert workflow.state['status'] == 'NEEDS_REVIEW'
    stage = workflow.state['stage']
    assert workflow.consume_decision() is False
    assert workflow.state['stage'] == stage
    workflow.publish_review()
    dispatcher.flush(timeout=2)
    events = list((tmp_path / 'notifications' / 'events').glob('*.json'))
    assert len(events) == 1
    event = json.loads(events[0].read_text())
    assert store.read_delivery(event['event_id']).status == 'DELIVERED'
    assert sender.calls == 2
    write_decision(tmp_path, 'continue', workflow.state['report_sha'])
    assert workflow.consume_decision()
    assert workflow.state['stage'] == 1
    assert workflow.state['status'] == 'READY'
    # Simulate crash after state commit but before archiving the decision.
    atomic_write_json(tmp_path / 'decision.json', workflow.state['last_decision'])
    assert workflow.consume_decision() is False
    assert not (tmp_path / 'decision.json').exists()
    dispatcher.close()


def test_arena_error_is_fail_closed(tmp_path):
    (tmp_path / 'games.jsonl').write_text(json.dumps({'technical_termination': 'ERROR_SEARCH'}))
    with pytest.raises(ValueError, match='Technical'):
        arena_statistics(tmp_path)


def test_real_trained_artifact_arena_load(trainer, tmp_path):
    from tools.arena_profiles import get_profile
    from gocube_golden.neural import model_hash
    trainer.step(data())
    path = tmp_path / 'candidate.pt'
    trainer.save(path, replay_ids=[], config_hash='test')
    profile = get_profile('torus9|komi=1.5|simulations=256|watchdog=1000|5ch')
    identity = profile.load_identity(path)
    loaded = profile.load_parent_model(identity, torch.device('cpu'))
    assert model_hash(loaded) == model_hash(trainer.model)
    assert profile.observation_shape == (5, 81)


def test_retry_rolls_back_entire_phase(trainer, tmp_path):
    from gocube_golden.orchestrator_v2.adaptation import init_run, retry_phase
    root, target = tmp_path / 'original', tmp_path / 'retry'
    init_run(root)
    state = json.loads((root / 'state.json').read_text())
    start = copy.deepcopy(state)
    state.update(status='FAILED', stage_start=start)
    atomic_write_json(root / 'state.json', state)
    retry_phase(root, target, .5)
    new = json.loads((target / 'state.json').read_text())
    cfg = json.loads((target / 'config.json').read_text())
    trainer.restore(Path(new['checkpoint']), config_hash=cfg['fingerprint'], replay_ids=[])
    assert trainer.update == 0
    assert trainer.lr_scale == .5
    trainer.step(data())
    for group in trainer.optimizer.param_groups:
        assert group['lr'] == .5 * lr_for(group['name'], 1)
    assert json.loads((root / 'state.json').read_text())['status'] == 'FAILED'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke')
def test_cuda_full_unfreeze_bias(trainer):
    t = AdaptationTrainer(BOOTSTRAP, device='cuda')
    t.update = 480
    for n, p in t.model.named_parameters():
        t.optimizer.state[p]['step'].fill_((0 if n == 'input_projection.bias' else 15520)
                                          + max(0, 480 - activation(n)))
    row = t.step(data())
    assert row['groups']['input_projection.bias']['step'] == 1
    assert all(torch.isfinite(p).all() for p in t.model.parameters())


def test_report_commit_survives_notification_crash(tmp_path, monkeypatch):
    from gocube_golden.orchestrator_v2.adaptation import init_run
    from gocube_golden.notifications import RecordingEventSink
    if not BOOTSTRAP.exists():
        pytest.skip('Canonical local fixture missing')
    init_run(tmp_path)
    sink = RecordingEventSink()
    monkeypatch.setattr('gocube_golden.orchestrator_v2.adaptation.create_telegram_dispatcher', lambda _: sink)
    w = Workflow(tmp_path)
    def crash(*args, **kwargs):
        raise RuntimeError('injected crash before enqueue')
    monkeypatch.setattr(w, 'publish_review', crash)
    with pytest.raises(RuntimeError, match='injected'):
        w.finish({'gate': 'PASS', 'validation': {}, 'arena': None}, 'review')
    persisted = json.loads((tmp_path / 'state.json').read_text())
    assert persisted['status'] == 'NEEDS_REVIEW'
    assert Path(persisted['report']).exists()
    w2 = Workflow(tmp_path)
    assert w2.state['stage'] == 0
    assert w2.state['status'] == 'NEEDS_REVIEW'
