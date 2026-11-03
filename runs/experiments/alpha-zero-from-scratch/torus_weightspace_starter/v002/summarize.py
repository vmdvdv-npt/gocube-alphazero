"""Paired analysis of immutable reports. No inference or weight generation."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def summarize(session):
    rows = {}
    incomplete = []
    for folder in sorted((session/'rounds').iterdir()):
        req = json.loads((folder/'request.json').read_text())
        if req['phase']!='holdout':
            continue
        if not (folder/'report.json').exists():
            incomplete.append(str(folder))
            continue
        report = json.loads((folder/'report.json').read_text())
        rows.setdefault(req['comparison'],[]).append((req,report))
    output = {'comparisons':{},'missing_reports':incomplete}
    for name,batches in rows.items():
        games = []
        for request,report in sorted(batches,key=lambda item:item[0]['chunk']):
            for g in report['games']:
                games.append(dict(g,pair_id=g['pair_id']+12*request['chunk']))
        valid = len(batches)==8 and all(g['result'] is not None for g in games)
        pairs = {}
        wdl_errors = [[],[]]
        pass_stats = [[],[]]
        for game in games:
            if game['result'] is None:
                continue
            points = {'WIN':1,'DRAW':.5,'LOSS':0}[game['result']]
            pairs.setdefault(game['pair_id'],[]).append(points)
            black_won = (game['result']=='WIN')==game['candidate_black']
            game_brier = [[],[]]
            for ply,step in enumerate(game['trace']):
                predictions = step.get('comparison_wdl')
                if predictions:
                    label = 1 if game['result']=='DRAW' else (0 if (step['color']==1)==black_won else 2)
                    target = np.eye(3)[label]
                    for i in (0,1):
                        game_brier[i].append(float(np.square(np.array(predictions[i])-target).sum()))
                if step['action']=='PASS':
                    pass_stats[step['player']].append(len(game['trace'])-ply-1)
            for i in (0,1):
                if game_brier[i]:
                    wdl_errors[i].append(float(np.mean(game_brier[i])))
        pair_scores = [np.mean(v) for _,v in sorted(pairs.items()) if len(v)==2]
        mean = float(np.mean(pair_scores)) if pair_scores else None
        # Each opening/color pair is one bounded [0,1] observation. Hoeffding
        # does not pretend the two games are independent; it also handles 0 wins.
        radius = math.sqrt(math.log(40)/(2*len(pair_scores))) if pair_scores else 1
        family_radius = math.sqrt(math.log(80)/(2*len(pair_scores))) if pair_scores else 1
        result = {'status':'VALID' if valid else 'INCOMPLETE_OR_INVALID',
                  'games':len(games),'pairs':len(pair_scores),
                  'wins':sum(g['result']=='WIN' for g in games),
                  'losses':sum(g['result']=='LOSS' for g in games),
                  'draws':sum(g['result']=='DRAW' for g in games),
                  'technical_terminations':sum(g['result'] is None for g in games),
                  'score_rate':mean,'paired_hoeffding_95':[max(0,mean-radius),min(1,mean+radius)] if mean is not None else None,
                  'two_primary_comparisons_familywise_95':[max(0,mean-family_radius),min(1,mean+family_radius)] if mean is not None else None,
                  'paired_scores':pair_scores,
                  'same_states_wdl_brier_game_mean':[float(np.mean(x)) if x else None for x in wdl_errors],
                  'same_states_wdl_brier_games':[len(x) for x in wdl_errors],
                  'moves_remaining_after_pass_mean':[float(np.mean(x)) if x else None for x in pass_stats],
                  'pass_counts':[len(x) for x in pass_stats],
                  'worker_seconds':sum(r['seconds'] for _,r in batches)}
        output['comparisons'][name] = result
    output['criterion_met'] = all(
        output['comparisons'].get(name,{}).get('status')=='VALID' and
        output['comparisons'][name]['two_primary_comparisons_familywise_95'][0]>.5
        for name in ('v002-v001-policy','v002-v001-s64'))
    return output


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    result = summarize(args.session)
    if args.output:
        with args.output.open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False)
            f.write('\n')
    print(json.dumps({name:{k:v for k,v in r.items() if k!='paired_scores'} for name,r in result['comparisons'].items()},indent=2))
