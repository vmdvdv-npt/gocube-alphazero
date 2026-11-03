import json,math
from pathlib import Path
root=Path('/tmp/torus9-v000-diagnostic-20261103/session/rounds')
def flat(x):
 if isinstance(x,list):
  for y in x: yield from flat(y)
 elif isinstance(x,(int,float)): yield x
for n in ['001','002']:
 d=json.loads((root/n/'report.json').read_text())
 for p in d['positions']:
  print(n,p['id'],'top',[(x['action'],round(x['probability'],5),round(x['logit'],3)) for x in p['top_k'][:5]],'PASS',round(p['probabilities'][81],6),round(p['logits'][81],3),'wdl',list(map(lambda x:round(x,5),p['wdl_probabilities'])))
  print(' target0',p['legal_mask'][0],round(p['logits'][0],3),round(p['probabilities'][0],6),'target1',p['legal_mask'][1],round(p['logits'][1],3),round(p['probabilities'][1],6),'finite',all(math.isfinite(v) for k in ['input','logits','probabilities','wdl_logits','wdl_probabilities','activations'] for v in flat(p[k])),'mask', all((p['probabilities'][i]==0 if not x else p['probabilities'][i]>0) for i,x in enumerate(p['legal_mask'])), 'sum',sum(p['probabilities']))
  print(' activation ranges',[(k,round(min(flat(v)),3),round(max(flat(v)),3)) for k,v in p['activations'].items()])
  if p['id']=='empty': print('empty spread',max(p['logits'][:81])-min(p['logits'][:81]))
 if n=='002':
  by={p['id']:p for p in d['positions']}; a=by['sym_base']
  for label,fn in [('sym_translate',lambda p:p//9*9+(p%9+3)%9),('sym_rotate',lambda p:(p%9)*9+8-p//9)]:
   b=by[label]; print('SYM',label,'logit maxdiff',max(abs(a['logits'][i]-b['logits'][fn(i)]) for i in range(81)),'prob maxdiff',max(abs(a['probabilities'][i]-b['probabilities'][fn(i)]) for i in range(81)),'passdiff',abs(a['logits'][81]-b['logits'][81]),'wdldiff',max(abs(x-y) for x,y in zip(a['wdl_logits'],b['wdl_logits'])))
