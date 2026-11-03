"""Secondary, post-evaluation statistics; never used to choose a checkpoint.

Predeclared primary bounds remain those in summarize.py. These paired bootstrap
intervals and exact sign tests are labelled supplementary rather than substituted
for the conservative original acceptance rule.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def analyze(session):
    output = {}
    for name in ('v002-v001-policy','v002-v001-s64'):
        pairs = {}
        for request_path in (session/'rounds').glob('*/request.json'):
            req = json.loads(request_path.read_text())
            if req.get('comparison')!=name or req['phase']!='holdout':
                continue
            report = json.loads(request_path.with_name('report.json').read_text())
            assert report['status']=='VALID'
            for g in report['games']:
                pair = req['chunk']*12+g['pair_id']
                row = pairs.setdefault(pair,{'scores':[],'brier_diff':[]})
                row['scores'].append({'WIN':1,'DRAW':.5,'LOSS':0}[g['result']])
                black_win = (g['result']=='WIN')==g['candidate_black']
                differences = []
                for step in g['trace']:
                    if not step.get('comparison_wdl'):
                        continue
                    label = 1 if g['result']=='DRAW' else (0 if (step['color']==1)==black_win else 2)
                    target = np.eye(3)[label]
                    errors = np.square(np.array(step['comparison_wdl'])-target).sum(1)
                    differences.append(float(errors[0]-errors[1]))
                row['brier_diff'].append(float(np.mean(differences)))
        assert len(pairs)==96 and all(len(p['scores'])==2 for p in pairs.values())
        scores = np.array([np.mean(pairs[i]['scores']) for i in range(96)])
        brier = np.array([np.mean(pairs[i]['brier_diff']) for i in range(96)])
        won,lost,tied = int((scores>.5).sum()),int((scores<.5).sum()),int((scores==.5).sum())
        decisive=won+lost
        pvalue=min(1.,2*sum(math.comb(decisive,k) for k in range(min(won,lost)+1))/2**decisive)
        rng=np.random.default_rng(2026110366)
        samples=rng.integers(0,96,size=(100000,96))
        output[name]={'paired_won_split_lost':[won,tied,lost],
                      'exact_two_sided_sign_p':pvalue,
                      'paired_bootstrap_score_95':np.quantile(scores[samples].mean(1),[.025,.975]).tolist(),
                      'same_state_brier_difference':float(brier.mean()),
                      'paired_bootstrap_brier_difference_95':np.quantile(brier[samples].mean(1),[.025,.975]).tolist(),
                      'resamples':100000,'seed':2026110366,
                      'status':'SUPPLEMENTARY_POST_EVALUATION_NOT_PRIMARY_ACCEPTANCE'}
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.session)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(result,indent=2))
