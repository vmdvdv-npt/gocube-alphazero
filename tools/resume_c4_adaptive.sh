#!/usr/bin/env bash
set -Eeuo pipefail

EXP_ID=${1:?usage: resume_c4_adaptive.sh EXPERIMENT_ID}
SOURCE_RUN=${2:-c4-t001-c4-c001}
FROZEN_COMMIT=85c87a7cfd467a4d3f4b2844253fb63d746d672a

fail() {
  echo "ADAPTIVE RESUME FAIL: $*" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || fail "run inside gocube-alphazero"
cd "$REPO_ROOT"
[[ $(git rev-parse HEAD) == "$FROZEN_COMMIT" ]] || fail "training checkout must remain at frozen commit $FROZEN_COMMIT"
git diff --quiet || fail "tracked working tree has unstaged changes"
git diff --cached --quiet || fail "tracked working tree has staged changes"

mkdir -p .git/info
if ! grep -qxF 'training_reports/' .git/info/exclude 2>/dev/null; then
  printf '\ntraining_reports/\n' >> .git/info/exclude
fi

STATE="training_reports/$EXP_ID/state-private.json"
[[ -f "$STATE" ]] || fail "missing local experiment state: $STATE"

python3 - "$STATE" "$EXP_ID" "$SOURCE_RUN" "$FROZEN_COMMIT" <<'PY'
from pathlib import Path
import json, sys
path = Path(sys.argv[1])
experiment_id, source_run, frozen = sys.argv[2:5]
state = json.loads(path.read_text(encoding="utf-8"))
if state.get("experiment_id") != experiment_id:
    raise SystemExit("ADAPTIVE RESUME FAIL: experiment ID mismatch in state")
if state.get("source_run") != source_run:
    raise SystemExit("ADAPTIVE RESUME FAIL: source run mismatch in state")
if state.get("frozen_training_commit") != frozen:
    raise SystemExit("ADAPTIVE RESUME FAIL: frozen commit mismatch in state")
if state.get("status") == "DONE":
    raise SystemExit("ADAPTIVE RESUME FAIL: experiment is already DONE")
print(f"ADAPTIVE STATE VALID: {experiment_id}, status={state.get('status')}")
PY

if systemctl --user list-units --type=service --state=running,activating --no-legend 2>/dev/null \
  | grep -q 'gocube-c4-'; then
  systemctl --user list-units --type=service --state=running,activating --no-legend \
    | grep 'gocube-c4-' >&2 || true
  fail "another Cube 4 service is active or restarting; stop the old unit first"
fi

if pgrep -af 'alphazero/envs/gocube/train.py|c4_overnight_(experiment|hardened)\.py|c4_adaptive_finish\.py|evaluate_gocube_checkpoints\.py' \
  >/tmp/gocube-adaptive-running.$$ 2>/dev/null; then
  cat /tmp/gocube-adaptive-running.$$ >&2 || true
  rm -f /tmp/gocube-adaptive-running.$$
  fail "another GoCube training/evaluation process is already running"
fi
rm -f /tmp/gocube-adaptive-running.$$ 2>/dev/null || true

git fetch origin main training-reports
TOOLING_COMMIT=$(git rev-parse origin/main)
TMP_DIR="$HOME/.cache/gocube-night-tools-$TOOLING_COMMIT"
mkdir -p "$TMP_DIR"
for tool in \
  run_with_github_reports.sh \
  c4_overnight_experiment.py \
  c4_overnight_hardened.py \
  c4_adaptive_finish.py \
  evaluate_gocube_checkpoints.py; do
  git show "origin/main:tools/$tool" > "$TMP_DIR/$tool"
done
chmod +x "$TMP_DIR"/*.sh "$TMP_DIR"/*.py

UNIT="gocube-c4-adaptive-${EXP_ID##c4-hparam-night-}-$(date +%H%M%S)"
SYSTEMD_COMMAND=$(printf '%q ' \
  bash "$TMP_DIR/run_with_github_reports.sh" "$EXP_ID" -- \
  "$REPO_ROOT/.venv/bin/python" "$TMP_DIR/c4_adaptive_finish.py" \
  --experiment-id "$EXP_ID" \
  --source-run "$SOURCE_RUN" \
  --frozen-commit "$FROZEN_COMMIT" \
  --max-hours 999 \
  --adaptive-block-games 24 \
  --adaptive-max-games 72 \
  --adaptive-final-block-games 32 \
  --adaptive-final-max-games 128)

systemd-run --user \
  --unit="$UNIT" \
  --collect \
  --property="WorkingDirectory=$REPO_ROOT" \
  --property="Restart=on-failure" \
  --property="RestartSec=30s" \
  --property="StartLimitIntervalSec=600" \
  --property="StartLimitBurst=3" \
  --setenv="PYTHONUNBUFFERED=1" \
  --setenv="PYTHONPATH=$REPO_ROOT" \
  --setenv="GOCUBE_NIGHT_TOOLING_COMMIT=$TOOLING_COMMIT" \
  /bin/bash -lc "exec $SYSTEMD_COMMAND"

sleep 2
systemctl --user is-active --quiet "$UNIT.service" || {
  systemctl --user status "$UNIT.service" --no-pager >&2 || true
  fail "adaptive service did not stay active after launch"
}

cat <<EOF
ADAPTIVE RESUME PASS
Experiment: $EXP_ID
Unit: $UNIT.service
Frozen training commit: $FROZEN_COMMIT
Tooling commit: $TOOLING_COMMIT
Runner: adaptive leader-first sequential finisher

Status:
  systemctl --user status $UNIT.service

Live log:
  journalctl --user -u $UNIT.service -f

Local report:
  training_reports/$EXP_ID/publish/summary.md
EOF
