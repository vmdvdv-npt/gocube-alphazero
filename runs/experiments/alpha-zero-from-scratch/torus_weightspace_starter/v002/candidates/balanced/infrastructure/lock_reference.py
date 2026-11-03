"""Fingerprint an evidence-selected champion; never select by generation number."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from weights import sha256, write_json
from gocube_golden.state import rules_fingerprint_for
from gocube_golden.topology import TORUS_9X9


def read_manifest(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    sidecar = checkpoint.with_suffix('.metadata.json')
    meta = json.loads(sidecar.read_text())
    arch = meta.get('architecture_config', {})
    if arch.get('input_channels') != 5 or meta.get('observation_shape') != [5,81]:
        raise ValueError('Experiment requires the verified 5CH reference contract')
    komi = meta.get('komi')
    if type(komi) not in (int,float):
        raise ValueError('Missing referee komi')
    rules = rules_fingerprint_for(TORUS_9X9, komi)
    if meta.get('rules_fingerprint', rules) != rules:
        raise ValueError('Reference rules fingerprint mismatch')
    if arch.get('topology_fingerprint') != TORUS_9X9.fingerprint:
        raise ValueError('Reference topology mismatch')
    return {'checkpoint': str(checkpoint), 'sidecar': str(sidecar),
            'artifact_sha256': sha256(checkpoint), 'sidecar_sha256': sha256(sidecar),
            'model_hash': meta.get('model_hash') or meta.get('converted_model_hash'),
            'architecture_id': meta.get('architecture_id'), 'architecture': arch,
            'input_channels': 5, 'komi': komi, 'rules_fingerprint': rules,
            'rules_fingerprint_source': 'engine-derived from canonical topology and metadata komi',
            'topology_fingerprint': TORUS_9X9.fingerprint,
            'arena_champion_verified': False, 'arena_evidence': []}


def freeze(checkpoint, evidence_paths, rationale):
    lock = read_manifest(checkpoint)
    positive = False
    search = None
    for path in evidence_paths:
        path = Path(path).resolve()
        report = json.loads(path.read_text())
        contract = report.get('scientific_contract', {})
        # Completed valid pairs and no technical outcomes are mandatory.
        if (report.get('validity') != 'VALID' or report.get('technical_games') != 0
                or report.get('valid_games') != report.get('games')
                or report.get('games',0) < 192 or report.get('pairs_valid',0)*2 != report['games']
                or report.get('wins',0)+report.get('losses',0)+report.get('draws',0) != report['games']
                or contract.get('komi') != lock['komi']
                or contract.get('input_channels') != 5
                or contract.get('paired_starts_color_swap') is not True):
            raise ValueError('Incomplete or incompatible Arena evidence')
        role = 'candidate' if report.get('candidate_model_hash') == lock['model_hash'] else 'reference'
        if report.get(role+'_model_hash') != lock['model_hash'] or report.get(role+'_artifact_sha256','').removeprefix('sha256:') != lock['artifact_sha256']:
            raise ValueError('Evidence does not identify this checkpoint artifact')
        w,l = report['wins'], report['losses']
        if role == 'reference':
            w,l = l,w
        positive |= w > l
        selected = {k:contract[k] for k in ('simulations','cpuct','fpu','tree_reuse','watchdog')}
        if search is not None and search != selected:
            raise ValueError('Arena search contracts differ')
        search = selected
        lock['arena_evidence'].append({'path':str(path), 'sha256':sha256(path),
                                     'role':role, 'wins':w, 'losses':l, 'draws':report['draws'],
                                     'games':report['games']})
    if not positive or not rationale.strip():
        raise ValueError('Winning completed Arena evidence and selection rationale required')
    lock.update(arena_champion_verified=True, search=search, selection_rationale=rationale,
                selection_scope='latest verified reference in inspected local Arena records; no global-strength claim')
    return lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--evidence', type=Path, action='append', required=True)
    p.add_argument('--rationale', required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    lock=freeze(args.checkpoint,args.evidence,args.rationale)
    write_json(args.output,lock)
    print(json.dumps(lock))


if __name__=='__main__':
    main()
