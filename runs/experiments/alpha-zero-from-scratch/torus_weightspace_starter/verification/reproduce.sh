#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WSP_PY:=/home/codex/projects/gocube-alphazero/.venv/bin/python}"
: "${WSP_VERIFY_OUT:=$(mktemp -d /tmp/torus-weightspace-verify.XXXXXX)}"
"$WSP_PY" -m unittest discover -s tests -v
"$WSP_PY" experiment_cli.py schema > "$WSP_VERIFY_OUT/tensor-schema.json"
"$WSP_PY" experiment_cli.py zero-pass --writer design_weights.py --output "$WSP_VERIFY_OUT/v000" > "$WSP_VERIFY_OUT/zero.json"
"$WSP_PY" experiment_cli.py verify --checkpoint "$WSP_VERIFY_OUT/v000/checkpoint.pt" > "$WSP_VERIFY_OUT/verify.json"
"$WSP_PY" experiment_cli.py init --session "$WSP_VERIFY_OUT/session10" --budget 10
"$WSP_PY" experiment_cli.py positions --session "$WSP_VERIFY_OUT/session10" --checkpoint "$WSP_VERIFY_OUT/v000/checkpoint.pt" --komi 1.5 --positions positions.json --activations
# Explicit technical-path smoke only: two games are stopped after one ply.
"$WSP_PY" experiment_cli.py games --session "$WSP_VERIFY_OUT/session10" --checkpoint "$WSP_VERIFY_OUT/v000/checkpoint.pt" --komi 1.5 --opponent pass --watchdog 1
"$WSP_PY" - "$WSP_VERIFY_OUT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
zero=json.loads((root/'zero.json').read_text())
assert zero['forward_calls']==0 and zero['reproducible']
positions=json.loads((root/'session10/rounds/001/report.json').read_text())
assert len(positions['positions'])==4
assert all(len(p['logits'])==82 for p in positions['positions'])
games=json.loads((root/'session10/rounds/002/report.json').read_text())
assert games['status']=='INVALID' and games['technical_terminations']==2
assert games['wins']==games['losses']==games['draws']==0
print(json.dumps({'status':'PASS','scope':'infrastructure fixtures only','output':str(root)}))
PY
