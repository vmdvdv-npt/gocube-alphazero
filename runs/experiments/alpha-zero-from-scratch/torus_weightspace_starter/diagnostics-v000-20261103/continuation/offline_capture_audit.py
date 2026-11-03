import sys,json
from pathlib import Path
sys.path.insert(0,'.')
from diagnostics import replay
from gocube_golden.rules import apply_action
p=Path('diagnostics-v000-20261103/session/rounds/004/report.json')
d=json.loads(p.read_text()); totals=[0,0,0,0]
for x in d['positions']:
 s=replay(x['moves'],1.5); a=x['activations']['blocks.7'][0]; tp=[];fp=[];fn=[];tn=[]
 for action in x['legal_actions']:
  if action=='PASS':continue
  signal=(a[action][92]-a[action][93])/2
  captured=list(apply_action(s,action).captured)
  if signal>0.5 and captured:tp.append((action,round(signal,5),captured))
  elif signal>0.5:fp.append((action,round(signal,5)))
  elif captured:fn.append((action,round(signal,5),captured))
  else:tn.append(action)
 for i,v in enumerate((tp,fp,fn,tn)):totals[i]+=len(v)
 print(x['id'],'TP',tp,'FP',fp,'FN',fn,'TN',len(tn))
print('TOTAL TP FP FN TN',totals)
