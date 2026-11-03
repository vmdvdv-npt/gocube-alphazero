import json,statistics,math
from pathlib import Path
out=Path('/tmp/torus9-v000-diagnostic-2'); sess=Path('/tmp/torus9-v000-diagnostic-20261103/session/rounds')
p=json.load(open(sess/'004/report.json')); checks={x['id']:x for x in json.load(open(out/'referee_checks.json'))}
regs=json.load(open('versions/v000/writer.stdout.txt'))['registers']
for x in p['positions']:
 c=checks[x['id']]; t=c['target']; a=x['activations']['blocks.7'][0];
 def reg(point,name):
  j=regs[name]; return round((a[point][j]-a[point][j+1])/2,4)
 print(x['id'],'target',t,'prob',x['probabilities'][t],'logit',x['logits'][t],'rank',next((i+1 for i,y in enumerate(x['top_k']) if y['action']==t),None),'top',[(v['action'],round(v['probability'],5)) for v in x['top_k'][:5]],'pass',x['probabilities'][81], 'registers', {n:reg(t,n) for n in ['capture','rescue','self_atari','eye','safe_rescue','weak2_own','weak2_opp']})
 if x['id']=='long_distributed':print('ring weak2 own/opp',[(i,reg(i,'weak2_opp'),reg(i,'weak3_opp')) for i in range(9)])
if (sess/'005/report.json').exists():
 d=json.load(open(sess/'005/report.json'));g=d['games']; print('GAMES',d['wins'],d['losses'],d['draws'],d['technical_terminations'],d['status'])
 for black in (True,False):
  rows=[x for x in g if x['candidate_black']==black]; print('COLOR',black,'outcomes', {k:sum(x['result']==k for x in rows) for k in ['WIN','LOSS','DRAW',None]},'length',min(map(lambda x:len(x['trace']),rows)),statistics.mean(len(x['trace']) for x in rows),max(len(x['trace']) for x in rows))
 lens=[len(x['trace']) for x in g];print('LENGTH',min(lens),statistics.median(lens),round(statistics.mean(lens),3),max(lens),'individual',lens)
 for who in ['candidate','random']:
  plies=[(x,m) for x in g for m in x['trace'] if ((m['color']==1)==x['candidate_black'])==(who=='candidate')];passes=[(x,m) for x,m in plies if m['action']=='PASS'];print('PASS',who,'count',len(passes),'moves',len(plies),'rate',len(passes)/len(plies),'games',sum(any(m['action']=='PASS' and (((m['color']==1)==x['candidate_black'])==(who=='candidate')) for m in x['trace']) for x in g))
 print('pairs',[(x['pair_id'],x['candidate_black'],x['result'],x['technical_termination'],len(x['trace'])) for x in g])
