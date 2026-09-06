#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-gocube-cube-4-katago-hardened-s50-20260907}"
RESUME="${RESUME:-0}"

existing=0
for path in "checkpoint/$RUN_NAME" "data/$RUN_NAME" "runs/$RUN_NAME"; do
  if [[ -e "$path" ]]; then
    existing=1
  fi
done

extra_args=()
if [[ "$existing" == "1" ]]; then
  if [[ "$RESUME" != "1" ]]; then
    echo "Run namespace already exists: $RUN_NAME" >&2
    echo "Set RESUME=1 only when intentionally resuming this hardened run." >&2
    exit 2
  fi
  extra_args+=(--allow-existing-run)
elif [[ "$RESUME" == "1" ]]; then
  echo "RESUME=1 requires an existing run namespace: $RUN_NAME" >&2
  exit 2
fi

python -m alphazero.envs.gocube.hardened_train \
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
  --run-name "$RUN_NAME" \
  "${extra_args[@]}"
