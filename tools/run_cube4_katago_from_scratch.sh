#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-gocube-cube-4-katago-final-s50-20260907}"

for path in "checkpoint/$RUN_NAME" "data/$RUN_NAME" "runs/$RUN_NAME"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to reuse existing path: $path" >&2
    echo "Choose a new RUN_NAME for a guaranteed from-scratch test." >&2
    exit 2
  fi
done

python -m alphazero.envs.gocube.production_hardening \
  --topology cube \
  --size 4 \
  --workers 16 \
  --sims 50 \
  --arena-sims 50 \
  --games-per-iteration 256 \
  --iterations 8 \
  --train-batch-size 256 \
  --fast-game-prob 0.25 \
  --endgame-sample-weight 1 \
  --run-name "$RUN_NAME"
