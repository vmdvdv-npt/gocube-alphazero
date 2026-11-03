"""Independent, append-only 64-pass experiment. Uses canonical referee and PUCT.

Inference is exclusively model(observation); the referee supplies only the
unchanged five input planes, legality, search transitions and final outcomes.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[3]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO))
from weights import restore_checkpoint, sha256, write_json
from diagnostics import replay, load_reference, positions
from gocube_golden.rules import apply_action, prepare_legal_actions
from gocube_golden.state import BLACK, PASS
from gocube_golden.result import result_from_terminal
from gocube_golden.search import Evaluation, SearchTree, SequentialPUCTSession
from gocube_golden.arena_contract import SearchSettings
from gocube_golden.torus9_m137_5ch import build_m137_five_channel_observation


def load(path):
    path = Path(path).resolve()
    if path.suffix == '.json':
        model, meta = load_reference(path)
        return model.eval().requires_grad_(False), meta['artifact_sha256']
    model, meta = restore_checkpoint(path)
    zero = json.loads((path.parent/'zero-pass.json').read_text())
    if zero['status'] != 'PASS' or zero['metadata']['artifact_sha256'] != meta['artifact_sha256']:
        raise ValueError('Missing matching zero-pass')
    return model, meta['artifact_sha256']


def sources():
    paths = [*HERE.glob('*.py'), *BASE.glob('*.py'),
             REPO/'gocube_golden/search.py', REPO/'gocube_golden/rules.py',
             REPO/'gocube_golden/torus9_m137_5ch.py']
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def openings(seed, count):
    rng = random.Random(seed)
    result, seen = [], set()
    while len(result) < count:
        length = (4, 8, 12, 16, 20, 24, 32, 40)[len(result) % 8]
        state, moves = replay([], 1.5), []
        for _ in range(length):
            legal = [a for a in prepare_legal_actions(state).actions if a != PASS]
            action = rng.choice(legal)
            state = apply_action(state, action).after
            moves.append(action)
        # Exclude duplicate final boards under translations and D4, including
        # collisions between development and holdout checked by init().
        key = orbit_key(state.stones)
        if key not in seen:
            seen.add(key)
            result.append(moves)
    return result


def orbit_key(stones):
    a = np.array([int(x) for x in stones], dtype=np.int8).reshape(9,9)
    return min(np.roll(np.roll(v,r,0),c,1).tobytes()
               for v in [np.rot90(a,k) for k in range(4)] +
                        [np.rot90(a.T,k) for k in range(4)]
               for r in range(9) for c in range(9))


def init(path):
    path.mkdir(parents=True, exist_ok=False)
    (path/'rounds').mkdir()
    dev = openings(2026110302, 32)
    holdout = openings(2026110399, 96)
    assert not ({orbit_key(replay(x,1.5).stones) for x in dev} &
                {orbit_key(replay(x,1.5).stones) for x in holdout})
    write_json(path/'development.json', dev)
    write_json(path/'holdout.json', holdout)
    write_json(path/'session.json', {
        'schema': 'weightspace-session-v002', 'budget': 64,
        'allocation': {'diagnostic': 8, 'development': 8, 'holdout': 48},
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'komi': 1.5, 'simulations': [0,64], 'policy': 'greedy legal argmax',
        'cpuct': 1.25, 'fpu': 0, 'tree_reuse': True, 'watchdog': 1000,
        'dev_seed': 2026110302, 'holdout_seed': 2026110399,
        'development_sha256': sha256(path/'development.json'),
        'holdout_sha256': sha256(path/'holdout.json'),
        'selection': 'at most 8 development batches; one frozen candidate; no holdout tuning',
        'holdout_comparisons': ['v002-v001-policy','v002-M269-policy','v001-M269-policy',
                               'v002-v001-s64','v002-M269-s64','v001-M269-s64'],
        'games_per_comparison': 192, 'pairs_per_round': 12,
        'success': 'v002-v001 paired 95% CI lower bound > 0.5; report M269 and multiplicity',
        'sources': sources()})


@contextmanager
def reserve(session, phase, request):
    with (session/'session.json').open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = json.load(lock)
        if config['schema'] != 'weightspace-session-v002':
            raise ValueError('Not a v002 session')
        previous = sorted((session/'rounds').iterdir())
        requests = [json.loads((p/'request.json').read_text()) for p in previous]
        if len(previous) >= config['budget'] or sum(r['phase']==phase for r in requests) >= config['allocation'][phase]:
            raise ValueError('Budget exhausted')
        if phase != 'holdout' and (session/'freeze.json').exists():
            raise ValueError('Development is closed after freeze')
        if phase == 'holdout':
            frozen = json.loads((session/'freeze.json').read_text())
            if sha256(session/'holdout.json') != config['holdout_sha256']:
                raise ValueError('Holdout changed')
            if frozen['sources'] != sources():
                raise ValueError('Execution sources changed after freeze')
            for label in ('candidate','opponent'):
                if request[label+'_sha256'] not in frozen['models'].values():
                    raise ValueError('Unfrozen model')
            if request['comparison'] not in config['holdout_comparisons'] or request['chunk'] not in range(8):
                raise ValueError('Unplanned holdout comparison/chunk')
            left, right, mode = request['comparison'].split('-')
            if (request['candidate_sha256'] != frozen['models'][left] or
                request['opponent_sha256'] != frozen['models'][right] or
                request['simulations'] != (64 if mode=='s64' else 0)):
                raise ValueError('Holdout model pairing or search mismatch')
            if any(r.get('comparison') == request.get('comparison') and
                   r.get('chunk') == request.get('chunk') for r in requests):
                raise ValueError('Holdout chunk already reserved; do not rerun selectively')
        folder = session/'rounds'/('%03d' % (len(previous)+1))
        folder.mkdir()
        write_json(folder/'request.json', dict(request, phase=phase, sources=sources(),
                   revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                   command=sys.argv, round=len(previous)+1))
        archive = session.parent/'source-snapshots'
        archive.mkdir(exist_ok=True)
        for path,digest in sources().items():
            destination = archive/(digest+'.py')
            if not destination.exists():
                destination.write_bytes((REPO/path).read_bytes())
        # The reservation is durable before releasing the lock. Different
        # rounds may now run concurrently without sharing game or model state.
        fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            yield folder
        except BaseException as error:
            write_json(folder/'error.json', {'error': str(error), 'round_consumed': True})
            raise


def batch_evaluate(model, requests, device):
    obs = torch.stack([build_m137_five_channel_observation(s,legal_context=c) for s,c in requests]).to(device)
    with torch.inference_mode():
        p,w = model(obs)
    if not bool(torch.isfinite(p).all() and torch.isfinite(w).all()):
        raise ValueError('Nonfinite inference')
    return p.cpu(), w.softmax(-1).cpu()


def arena(candidate, opponent, starts, *, simulations, seed=20261103, device='cpu', watchdog=1000):
    models = [candidate.to(device), opponent.to(device)]
    settings = SearchSettings(simulations=simulations,cpuct=1.25,fpu=0.,tree_reuse=True) if simulations else None
    games = []
    for pair_id, moves in enumerate(starts):
        for black in (True,False):
            games.append({'state': replay(moves,1.5), 'pair_id': pair_id, 'candidate_black': black,
                          'opening': moves, 'trace': [], 'trees': [SearchTree(True),SearchTree(True)],
                          'session': None, 'done': False})
    start_time, calls = time.perf_counter(), [0,0]
    while not all(g['done'] for g in games):
        queues = [[],[]]
        for g in games:
            if g['done']:
                continue
            state = g['state']
            idx = 0 if (state.side_to_move==BLACK)==g['candidate_black'] else 1
            g['player'] = idx
            if settings and g['session'] is None:
                g['session'] = SequentialPUCTSession(state,settings,seed=seed+g['pair_id'],tree=g['trees'][idx])
            if settings:
                pending = g['session'].pending
                if pending:
                    queues[idx].append((g,pending.state,pending.legal_context))
            else:
                queues[idx].append((g,state,prepare_legal_actions(state)))
        for idx, queue in enumerate(queues):
            if not queue:
                continue
            logits, wdl = batch_evaluate(models[idx],[(s,c) for _,s,c in queue],device)
            calls[idx] += len(queue)
            for (g,s,context), p, w in zip(queue,logits,wdl):
                if settings:
                    if s.state_key == g['state'].state_key:
                        g['root_wdl'] = w.tolist()
                    g['session'].resume(Evaluation(tuple(p.softmax(0).tolist()),tuple(w.tolist())))
                else:
                    legal = p.masked_fill(~torch.tensor(context.action_mask,dtype=torch.bool),-torch.inf)
                    a = int(legal.argmax())
                    g['action'] = PASS if a==81 else a
                    g['root_wdl'] = w.tolist()
        for g in games:
            if g['done'] or settings and g['session'].result is None:
                continue
            action = g['session'].result.action if settings else g.pop('action')
            state = g['state']
            transition = apply_action(state,action)
            if settings and 'root_wdl' not in g:
                g['root_wdl'] = list(g['trees'][g['player']].root.evaluation.wdl)
            g['trace'].append({'action': action, 'color': int(state.side_to_move),
                               'player': g['player'], 'captured': list(transition.captured),
                               'wdl': g.pop('root_wdl',None)})
            g['state'] = transition.after
            g['session'] = None
            for tree in g['trees']:
                tree.advance(action,g['state'])
            g['done'] = g['state'].is_terminal or len(g['trace']) >= watchdog
    rows = []
    for g in games:
        row = {k:g[k] for k in ('pair_id','candidate_black','opening','trace')}
        if g['state'].is_terminal:
            score = result_from_terminal(g['state'])
            row['score'] = asdict(score)
            row['result'] = 'DRAW' if score.winner.value=='DRAW' else (
                'WIN' if (score.winner.value=='BLACK')==g['candidate_black'] else 'LOSS')
        else:
            row.update(result=None,technical_termination='WATCHDOG')
        rows.append(row)
    return {'games':rows,'status':'VALID' if all(r['result'] for r in rows) else 'INVALID',
            'wins':sum(r['result']=='WIN' for r in rows),'losses':sum(r['result']=='LOSS' for r in rows),
            'draws':sum(r['result']=='DRAW' for r in rows),
            'technical_terminations':sum(r['result'] is None for r in rows),
            'seconds':time.perf_counter()-start_time,'model_calls':calls,
            'settings':asdict(settings) if settings else {'simulations':0,'selection':'legal argmax'},
            'runtime':{'device':device,'torch':str(torch.__version__),'dtype':'float32','tf32':False}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('init','positions','arena','freeze'))
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--candidate',type=Path)
    p.add_argument('--opponent',type=Path)
    p.add_argument('--positions',type=Path)
    p.add_argument('--phase',choices=('diagnostic','development','holdout'),default='development')
    p.add_argument('--simulations',type=int,choices=(0,64),default=0)
    p.add_argument('--chunk',type=int,default=0)
    p.add_argument('--pairs',type=int,default=4)
    p.add_argument('--comparison',default='development')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode=='init':
        init(args.session)
        return
    if args.mode=='freeze':
        models = {'v002':load(args.candidate)[1], 'v001':load(BASE/'versions/v001/checkpoint.pt')[1],
                  'M269':load(BASE/'reference-lock.json')[1]}
        write_json(args.session/'freeze.json',{'models':models,'sources':sources(),
                   'candidate':str(args.candidate.resolve()),'selection_completed':True})
        return
    candidate, candidate_sha = load(args.candidate)
    request = dict(candidate=str(args.candidate.resolve()),candidate_sha256=candidate_sha,
                   simulations=args.simulations,chunk=args.chunk,comparison=args.comparison,device=args.device)
    if args.mode=='positions':
        requests = json.loads(args.positions.read_text())
        with reserve(args.session,'diagnostic',request) as folder:
            write_json(folder/'positions.json',requests)
            report = positions(candidate,requests,komi=1.5,top_k=8,activations=False)
            write_json(folder/'report.json',report)
    else:
        opponent, opponent_sha = load(args.opponent)
        request.update(opponent=str(args.opponent.resolve()),opponent_sha256=opponent_sha)
        bank = json.loads((args.session/('holdout.json' if args.phase=='holdout' else 'development.json')).read_text())
        count = 12 if args.phase=='holdout' else args.pairs
        starts = bank[args.chunk*count:(args.chunk+1)*count]
        if len(starts)!=count:
            raise ValueError('Incomplete opening chunk')
        with reserve(args.session,args.phase,request) as folder:
            write_json(folder/'openings.json',starts)
            report = arena(candidate,opponent,starts,simulations=args.simulations,device=args.device)
            write_json(folder/'report.json',report)
    print(json.dumps({'round':folder.name,'report':str(folder/'report.json'),
                      **{k:v for k,v in report.items() if k in ('status','wins','losses','seconds','technical_terminations')}}),flush=True)


if __name__=='__main__':
    main()
