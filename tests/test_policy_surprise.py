import copy
import random
from types import SimpleNamespace as NS

import pytest
import torch

from gocube_golden.policy_surprise import (
    ALGORITHM, SamplingTelemetry, build_cache, frequency_weights, load_cache,
    policy_kl, resolve_spec, sampling_setting,
)
from gocube_golden.provenance import file_sha256, sha256_fingerprint
from gocube_golden.torus9_five_channel_training import OrdinaryReplayWindow, OrdinaryTrainer
from test_torus9_five_channel_training import parent, game


@pytest.mark.parametrize('surprises,weight,expected', [
    ([0, 1, 3], .5, [.5, .875, 1.625]),
    ([0, 1, 3], 0, [1, 1, 1]),
    ([2, 2, 2], .5, [1, 1, 1]),
    ([0, 0, 0], .5, [1, 1, 1]),
])
def test_frequency_formula_and_per_game_mass(surprises, weight, expected):
    f = frequency_weights(surprises, weight)
    assert f.tolist() == expected
    assert float(f.sum()) == len(surprises)
    assert (f >= 1-weight).all()


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_surprise_fails(value):
    with pytest.raises(ValueError):
        frequency_weights([value], .5)


def test_legal_kl_extreme_logits_and_illegal_actions():
    logits = torch.tensor([[1000., -1000., 1e20]])
    pi = torch.tensor([[0., 1., 0.]])
    mask = torch.tensor([[True, True, False]])
    assert policy_kl(logits, pi, mask).tolist() == [2000.]
    assert policy_kl(torch.zeros(1, 3), torch.tensor([[.5, .5, 0.]]), mask).tolist() == [0.]
    with pytest.raises(ValueError):
        policy_kl(logits, torch.tensor([[0., 0., 1.]]), mask)


def tiny_games():
    games = []
    for index, n in enumerate([3, 5, 2]):
        g = {'game_id': str(index), 'actor_hash': 'actor'}
        for key in ('observation', 'pi', 'z', 'ownership', 'score'):
            g[key] = torch.arange(sum([3,5,2][:index]),sum([3,5,2][:index])+n)
        games.append(g)
    return OrdinaryReplayWindow.from_games(games)


def bare_trainer(setting=None):
    trainer = OrdinaryTrainer.__new__(OrdinaryTrainer)
    trainer.model = torch.nn.Linear(1,1)
    trainer.seed = 91; trainer.batch_size = 64
    trainer.replay_sampling = sampling_setting(setting)
    trainer.sampling_telemetry = None
    return trainer


@pytest.mark.parametrize('setting', [None, {'mode': 'uniform'}, {'mode':'policy_surprise','weight':0}])
def test_uniform_positions_match_frozen_original_sampler(setting):
    trainer = bare_trainer(setting)
    games = tiny_games()
    for update in [0, 1, 197, 2560]:
        rng = random.Random(91+update)
        expected = [rng.randrange(10) for _ in range(64)]
        assert trainer.batch(games, update)['score'].tolist() == expected


def test_weighted_draws_deterministic_full_loss_weight_and_budget():
    games = tiny_games()
    evidence = {g['game_id']: {'surprise': torch.arange(len(g['score']),dtype=torch.float64),
        'frequency': frequency_weights(torch.arange(len(g['score'])), .5),
        'source_model_hash':'actor', 'source_generation':256} for g in games}
    results = []
    for _ in range(2):
        trainer = bare_trainer({'mode':'policy_surprise','weight':.5})
        trainer.sampling_telemetry = SamplingTelemetry(games,evidence,{},.5)
        draws = [trainer.batch(games,u)['score'].tolist() for u in range(100)]
        results.append(draws)
        assert trainer.sampling_telemetry.report()['sample_draws'] == 64*100
        # Ordinary batch keys and tensors are unchanged: no importance multipliers.
        assert set(trainer.batch(games,101)) == {'observation','pi','z','ownership','score'}
    assert results[0] == results[1]
    flat = sum(results[0], [])
    assert flat.count(7) > 3 * flat.count(3) # same game: high KL vs zero KL


def make_spec(parent, tmp_path):
    raw = torch.load(parent, weights_only=False)
    actor = raw['metadata']['model_hash']; sha = file_sha256(parent)
    g = game('eligible-only', 'train', actor); g['actor_artifact']=sha
    shard = tmp_path/'replay.pt'
    torch.save({'contract':g['contract'], 'actor_hash':actor, 'games':[g]},shard)
    ref = {'path':str(shard),'sha':file_sha256(shard),'generation':256,'actor_hashes':[actor]}
    import gocube_golden.policy_surprise as module
    body = {'algorithm':ALGORITHM, 'implementation_sha256':file_sha256(module.__file__),
        'weight':.5,'shards':[ref], 'sources':{actor:{'path':str(parent),'checkpoint':{'sha256':sha}}}}
    return {**body,'fingerprint':sha256_fingerprint(body)}, shard


def test_cache_historical_inference_reuse_tamper_provenance_and_source_integrity(parent,tmp_path):
    spec, shard = make_spec(parent,tmp_path)
    before = (file_sha256(parent),file_sha256(shard))
    ref = build_cache(spec,tmp_path/'experiment',device='cpu')
    rows = load_cache(ref,spec)
    assert list(rows) == ['eligible-only']
    assert len(rows['eligible-only']['surprise']) == 3
    assert float(rows['eligible-only']['frequency'].sum()) == pytest.approx(3)
    assert before == (file_sha256(parent),file_sha256(shard))
    wrong = copy.deepcopy(spec); wrong['weight']=.1
    with pytest.raises(ValueError,match='provenance'):
        load_cache(ref,wrong)
    from pathlib import Path
    with Path(ref['path']).open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(ValueError,match='SHA'):
        load_cache(ref,spec)


def test_unresolved_historical_actor_fail_closed(parent,tmp_path):
    spec, shard = make_spec(parent,tmp_path)
    ref = spec['shards'][0]
    node = NS(ref=NS(sha256='different'),path=parent)
    resolver = NS(checkpoint=lambda _:node,parent=lambda _:None)
    schedule = [{'source_checkpoint':{},'buckets':[{'generation':256,'shards':[ref]}]}]
    with pytest.raises(ValueError,match='Unresolved historical source model hash'):
        resolve_spec(schedule,parent=node,resolver=resolver,weight=.5)


def test_validation_does_not_use_weighted_batch(parent,monkeypatch):
    trainer = OrdinaryTrainer(parent, learning_rate=2.5e-5,seed=91,
                              replay_sampling={'mode':'policy_surprise','weight':.5})
    trainer.batch = lambda *args: pytest.fail('training sampler used for validation')
    raw = torch.load(parent, weights_only=False)
    trainer.evaluate([game('validation','validation',raw['metadata']['model_hash'])], batches=1)


def test_training_adapter_prepares_cache_with_durable_progress(parent, tmp_path):
    from gocube_golden.orchestrator_v2.execution_permit import _production_authority
    from gocube_golden.torus9_five_channel_training import prepare_policy_surprise_cache
    import json
    spec, shard = make_spec(parent, tmp_path)
    before = (file_sha256(parent), file_sha256(shard))
    request = {'spec': spec, 'device': 'cpu', 'cache_root': str(tmp_path / 'cache'),
               'heartbeat': str(tmp_path / 'heartbeat.json')}
    with _production_authority(mode='experiment', topology='torus9', run_id='test', code_identity='test'):
        cache = prepare_policy_surprise_cache(request)
    assert load_cache(cache, spec)
    heartbeat = json.loads((tmp_path / 'heartbeat.json').read_text())
    assert heartbeat['phase'] == 'complete' and heartbeat['done'] == heartbeat['total'] == 1
    assert before == (file_sha256(parent), file_sha256(shard))
