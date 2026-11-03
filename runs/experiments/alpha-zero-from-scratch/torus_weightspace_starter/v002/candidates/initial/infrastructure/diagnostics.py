"""Explicit opt-in inference and paired games, using the existing rules and PUCT."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from gocube_golden.rules import apply_action, prepare_legal_actions
from gocube_golden.state import BLACK, PASS, initial_state, rules_fingerprint_for
from gocube_golden.topology import TORUS_9X9
from gocube_golden.result import result_from_terminal
from gocube_golden.search import Evaluation, SequentialPUCT
from gocube_golden.arena_contract import SearchSettings
from gocube_golden.torus9_m137_5ch import build_m137_five_channel_observation


def replay(moves, komi):
    state = initial_state(topology=TORUS_9X9, komi=komi)
    for action in moves:
        state = apply_action(state, action).after
    return state


class Evaluator:
    def __init__(self, model):
        self.model = model.eval()
        self.seconds = []

    def logits(self, state, context=None):
        context = context or prepare_legal_actions(state)
        observation = build_m137_five_channel_observation(state, legal_context=context)
        start = time.perf_counter()
        with torch.inference_mode():
            policy, wdl = self.model(observation.unsqueeze(0))
        self.seconds.append(time.perf_counter() - start)
        if policy.shape != (1,82) or wdl.shape != (1,3):
            raise ValueError('Invalid output shapes')
        if not bool(torch.isfinite(policy).all() and torch.isfinite(wdl).all()):
            raise ValueError('Nonfinite logits')
        return observation, policy[0], wdl[0], context

    def evaluate_prepared(self, state, legal_context):
        _, policy, wdl, _ = self.logits(state, legal_context)
        return Evaluation(tuple(policy.softmax(0).tolist()), tuple(wdl.softmax(0).tolist()))

    def evaluate(self, state):
        return self.evaluate_prepared(state, prepare_legal_actions(state))

    def timing(self):
        return {'device': 'cpu', 'dtype': 'float32', 'batch_rows': 1,
                'calls': len(self.seconds), 'forward_seconds': sum(self.seconds),
                'mean_forward_seconds': sum(self.seconds)/len(self.seconds) if self.seconds else None,
                'scope': 'forward only; excludes observation, search, model loading; no warmup'}


def positions(model, requests, *, komi, top_k=5, activations=False):
    if not isinstance(requests, list) or not 1 <= len(requests) <= 256:
        raise ValueError('Positions pass requires 1..256 move-history objects')
    if not 1 <= top_k <= 82:
        raise ValueError('top_k must be 1..82')
    evaluator = Evaluator(model)
    results = []
    for request in requests:
        state = replay(request['moves'], komi)
        captures = {}
        handles = []
        if activations:
            def hook(name):
                def capture(module, inputs, output):
                    captures[name] = output.detach().cpu().tolist()
                return capture
            for name, module in model.named_modules():
                if name in ['input_projection', 'output_norm'] or name.startswith('blocks.') and name.count('.') == 1:
                    handles.append(module.register_forward_hook(hook(name)))
        try:
            obs, logits, wdl, context = evaluator.logits(state)
        finally:
            for handle in handles:
                handle.remove()
        mask = torch.tensor(context.action_mask, dtype=torch.bool)
        probs = logits.masked_fill(~mask, -torch.inf).softmax(0)
        indices = sorted((i for i, legal in enumerate(context.action_mask) if legal),
                         key=lambda i: (-float(probs[i]), i))[:top_k]
        results.append({'id': request.get('id'), 'moves': request['moves'],
                        'input': obs.tolist(), 'stones': list(map(int, state.stones)),
                        'side_to_move': int(state.side_to_move), 'legal_actions': list(context.actions),
                        'legal_mask': list(context.action_mask), 'logits': logits.tolist(),
                        'probabilities': probs.tolist(), 'wdl_logits': wdl.tolist(),
                        'wdl_probabilities': wdl.softmax(0).tolist(),
                        'top_k': [{'action': PASS if i == 81 else i, 'probability': float(probs[i]),
                                   'logit': float(logits[i])} for i in indices],
                        'activations': captures if activations else None})
    return {'positions': results, 'inference': evaluator.timing(),
            'rules_fingerprint': rules_fingerprint_for(TORUS_9X9, komi), 'komi': komi}


def paired_games(candidate, *, komi, openings, opponent='random', reference=None,
                 simulations=0, cpuct=1.25, fpu=0.0, tree_reuse=False,
                 seed=0, watchdog=1000):
    if not isinstance(openings, list) or not 1 <= len(openings) <= 64:
        raise ValueError('Games pass requires 1..64 paired opening histories')
    if not 1 <= watchdog <= 1000 or simulations < 0:
        raise ValueError('Invalid watchdog/simulations')
    if opponent not in ('random', 'pass', 'reference') or (opponent == 'reference') != (reference is not None):
        raise ValueError('Invalid opponent/reference')
    if reference is not None and simulations <= 0:
        raise ValueError('Arena requires a positive matched MCTS budget')
    if reference is None and simulations:
        raise ValueError('Simple control games are policy-only')
    evaluators = [Evaluator(candidate), Evaluator(reference) if reference is not None else None]
    settings = SearchSettings(simulations=simulations, cpuct=cpuct, fpu=fpu, tree_reuse=tree_reuse) if simulations else None
    rows = []
    for pair_id, moves in enumerate(openings):
        start = replay(moves, komi)
        if start.is_terminal:
            raise ValueError('Terminal opening')
        for candidate_black in (True, False):
            state = start
            rng = random.Random(seed + pair_id)  # common random seed in color pair
            searches = [SequentialPUCT(settings=settings), SequentialPUCT(settings=settings)] if settings else None
            trace = []
            for ply in range(watchdog):
                candidate_turn = (state.side_to_move == BLACK) == candidate_black
                index = 0 if candidate_turn else 1
                if searches:
                    result = searches[index].search(state, evaluators[index], seed=seed + pair_id)
                    action = result.action
                elif candidate_turn:
                    _, logits, _, context = evaluators[0].logits(state)
                    mask = torch.tensor(context.action_mask, dtype=torch.bool)
                    # Policy-only diagnostics: temperature 1, seeded sampling.
                    action_index = rng.choices(range(82), weights=logits.masked_fill(~mask, -torch.inf).softmax(0).tolist())[0]
                    action = PASS if action_index == 81 else action_index
                else:
                    legal = prepare_legal_actions(state).actions
                    action = PASS if opponent == 'pass' else rng.choice(legal)
                transition = apply_action(state, action)
                trace.append({'ply': ply, 'color': int(state.side_to_move), 'action': action,
                              'captured': list(transition.captured)})
                state = transition.after
                if searches:
                    for search in searches:
                        search.advance_root(action, state)
                if state.is_terminal:
                    break
            row = {'pair_id': pair_id, 'candidate_black': candidate_black,
                   'opening_moves': moves, 'trace': trace, 'final_stones': list(map(int,state.stones)),
                   'result': None, 'technical_termination': None}
            if state.is_terminal:
                result = result_from_terminal(state)
                row['score'] = asdict(result)
                row['result'] = 'DRAW' if result.winner.value == 'DRAW' else (
                    'WIN' if (result.winner.value == 'BLACK') == candidate_black else 'LOSS')
            else:
                row['technical_termination'] = 'WATCHDOG'
            rows.append(row)
    return {'games': rows, 'wins': sum(r['result']=='WIN' for r in rows),
            'losses': sum(r['result']=='LOSS' for r in rows),
            'draws': sum(r['result']=='DRAW' for r in rows),
            'technical_terminations': sum(r['technical_termination'] is not None for r in rows),
            'status': 'VALID' if all(r['result'] is not None for r in rows) else 'INVALID',
            'komi': komi, 'rules_fingerprint': rules_fingerprint_for(TORUS_9X9, komi),
            'search': asdict(settings) if settings else {'simulations': 0, 'temperature': 1},
            'opponent': opponent, 'seed': seed, 'watchdog': watchdog,
            'architectures': {'candidate': candidate.architecture,
                              'reference': reference.architecture_config if reference is not None else None},
            'inference': {'candidate': evaluators[0].timing(),
                          'reference': evaluators[1].timing() if evaluators[1] else None}}


def load_reference(lock_path):
    from weights import sha256
    from tools.arena_profiles.torus9 import Torus9ArenaProfile
    lock = json.loads(Path(lock_path).read_text())
    if not lock.get('arena_champion_verified'):
        raise ValueError('Reference champion has not been verified')
    checkpoint = Path(lock['checkpoint'])
    if sha256(checkpoint) != lock['artifact_sha256'] or sha256(checkpoint.with_suffix('.metadata.json')) != lock['sidecar_sha256']:
        raise ValueError('Frozen reference changed')
    for evidence in lock['arena_evidence']:
        if sha256(evidence['path']) != evidence['sha256']:
            raise ValueError('Arena evidence changed')
    profile = Torus9ArenaProfile(komi=lock['komi'], five_channel=True)
    identity = profile.load_identity(checkpoint)
    if identity.model_hash != lock['model_hash']:
        raise ValueError('Reference model hash drift')
    model = profile.load_parent_model(identity, torch.device('cpu'))
    return model, lock
