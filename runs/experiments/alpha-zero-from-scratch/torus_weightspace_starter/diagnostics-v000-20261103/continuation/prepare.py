import json,sys
from pathlib import Path
sys.path.insert(0,'.')
from diagnostics import replay
from gocube_golden.rules import apply_action,prepare_legal_actions
from gocube_golden.state import BLACK,WHITE,EMPTY
out=Path('/tmp/torus9-v000-diagnostic-2')
def n(p):
 r,c=divmod(p,9);return [((r-1)%9)*9+c,((r+1)%9)*9+c,r*9+(c-1)%9,r*9+(c+1)%9]
def group(s,p):
 seen={p}; q=[p]
 for x in q:
  for y in n(x):
   if s.stones[y]==s.stones[p] and y not in seen: seen.add(y);q.append(y)
 libs={y for x in seen for y in n(x) if s.stones[y]==EMPTY}
 return sorted(seen),sorted(libs)
def inter(a,b):
 v=[]
 for i in range(max(len(a),len(b))):
  if i<len(a):v.append(a[i])
  if i<len(b):v.append(b[i])
 return v
def trans(x):return ((x//9+4)%9)*9+(x%9+4)%9
wrap=[40,0,72,42,8,44,9,46]
cases=[('wrap_capture',wrap,1),('ordinary_capture',[trans(x) for x in wrap],trans(1)),
 ('futile_atari',inter([0,40,42,44,46,48],[72,8,9,2,10,73]),1),
 ('connect_weak',inter([39,41,0,2,4,6],[30,38,48,32,42,50]),40),
 ('false_eye',inter([39,41,31,49],[30,32,0,2]),40),
 ('profitable_capture',inter([39,41,31,48,58,0],[40,49,60,62,64,66]),50)]
# Ring of nine black stones; two distant liberties, each bordered by the ring.
black=list(range(9))+[36,37,38,39,40,41,42,43]
white=[x for x in list(range(9,18))+list(range(72,81)) if x not in (9,76)]
cases.append(('long_distributed',inter(black,white),9))
checks=[];requests=[]
for name,moves,target in cases:
 s=replay(moves,1.5); assert not s.is_terminal
 ctx=prepare_legal_actions(s); legal=target in ctx.actions
 info={'id':name,'moves':moves,'side_to_move':int(s.side_to_move),'target':target,'target_legal':legal,'legal_alternatives':len(ctx.actions)-1,'groups_before':{},'target_capture':None,'target_group_after':None}
 for p in range(81):
  if s.stones[p]!=EMPTY:
   stones,libs=group(s,p)
   if p==stones[0]:info['groups_before'][str(p)]={'stones':stones,'liberties':libs,'color':int(s.stones[p])}
 if legal:
  t=apply_action(s,target); info['target_capture']=list(t.captured); g,l=group(t.after,target);info['target_group_after']={'stones':g,'liberties':l}
 assert (not legal)==(name=='futile_atari'),(name,legal)
 if name=='wrap_capture':assert info['target_capture']==[0]
 if name=='ordinary_capture':assert info['target_capture']==[trans(0)]
 if name=='profitable_capture':assert info['target_capture']==[40,49]
 if name=='connect_weak':assert group(s,39)[1]==[40] and group(s,41)[1]==[40] and {39,41,40}.issubset(info['target_group_after']['stones']) and len(info['target_group_after']['liberties'])>=2
 if name=='futile_atari':assert group(s,0)[1]==[1]
 if name=='false_eye':assert all(s.stones[x]==BLACK for x in n(40)) and s.stones[30]==WHITE and s.stones[32]==WHITE
 if name=='long_distributed':assert group(s,0)[1]==[9,76] and info['target_group_after'] and group(apply_action(s,target).after,0)[1]==[76]
 checks.append(info);requests.append({'id':name,'moves':moves})
 print(name,'side',int(s.side_to_move),'target',target,'legal',legal,'captured',info['target_capture'],'alternatives',info['legal_alternatives'])
# Sixteen distinct, short, explicitly replayed openings using translations of existing starts.
base=json.load(open('openings.json')); openings=[[]]
for shift in range(8):
 for source in base[1:]:
  op=[(p//9)*9+(p%9+shift)%9 for p in source]
  if op not in openings and len(openings)<16:openings.append(op)
assert len(openings)==16
for op in openings: assert not replay(op,1.5).is_terminal
for name,obj in [('positions.json',requests),('referee_checks.json',checks),('openings.json',openings)]:
 with (out/name).open('x') as f:json.dump(obj,f,indent=2)
