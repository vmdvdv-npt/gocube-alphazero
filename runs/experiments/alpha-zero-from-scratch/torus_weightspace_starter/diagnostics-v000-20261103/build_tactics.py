import json,sys
sys.path.insert(0,'.')
from diagnostics import replay
from gocube_golden.rules import prepare_legal_actions
cases={
 'atari_capture':[40,0,72,42,8,44,9,46,1],
 'wrap_capture':[40,0,72,42,8,44,9,46,1],
 'rescue':[0,72,40,8,42,9],
 'own_eye':[72,40,8,42,9,44,1,46],
 'self_atari':[40,72,42,8,44,9],
 'long_distributed':[0,20,1,22,2,24,3,26,4,28,5,30,6,32,7,34],
}
# Translate and rotate a control history to exercise forward equivariance.
base=[0,40,8,41,72]
def tr(p): return (p//9*9+(p%9+3)%9)
def rot(p): return (p%9)*9+(8-p//9)
cases['sym_base']=base
cases['sym_translate']=[tr(p) for p in base]
cases['sym_rotate']=[rot(p) for p in base]
for name,moves in cases.items():
 s=replay(moves,1.5)
 assert not s.is_terminal
 print(name, 'side',s.side_to_move,'legal',len(prepare_legal_actions(s).actions))
with open('/tmp/torus9-v000-diagnostic-20261103/tactical_positions.json','x') as f: json.dump([{'id':k,'moves':v} for k,v in cases.items()],f,indent=2)
