#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_RUN=${1:-c4-t001-c4-c001}
MAX_HOURS=${2:-8}
FROZEN_COMMIT=85c87a7cfd467a4d3f4b2844253fb63d746d672a

fail() {
  echo "LAUNCH FAIL: $*" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || fail "run inside gocube-alphazero"
cd "$REPO_ROOT"
[[ $(git rev-parse HEAD) == "$FROZEN_COMMIT" ]] || fail "training checkout must remain at frozen commit $FROZEN_COMMIT"
git diff --quiet || fail "tracked working tree has unstaged changes"
git diff --cached --quiet || fail "tracked working tree has staged changes"

# The frozen commit predates the training_reports/ ignore entry. Keep generated
# reports out of `git status` without modifying any tracked file.
mkdir -p .git/info
if ! grep -qxF 'training_reports/' .git/info/exclude 2>/dev/null; then
  printf '\ntraining_reports/\n' >> .git/info/exclude
fi

if systemctl --user list-units --type=service --state=running --no-legend 2>/dev/null \
  | grep -q 'gocube-c4-night-'; then
  systemctl --user list-units --type=service --state=running --no-legend | grep 'gocube-c4-night-' >&2 || true
  fail "another Cube 4 night experiment service is already running"
fi

if pgrep -af 'python[^ ]* .*alphazero/envs/gocube/train.py' >/tmp/gocube-night-running.$$ 2>/dev/null; then
  cat /tmp/gocube-night-running.$$ >&2 || true
  rm -f /tmp/gocube-night-running.$$
  fail "another GoCube training process is already running"
fi
rm -f /tmp/gocube-night-running.$$ 2>/dev/null || true

git fetch origin main training-reports
TOOLING_COMMIT=$(git rev-parse origin/main)
TMP_DIR="$HOME/.cache/gocube-night-tools-$TOOLING_COMMIT"
mkdir -p "$TMP_DIR"
for tool in \
  run_with_github_reports.sh \
  preflight_c4_overnight.sh \
  c4_overnight_experiment.py \
  evaluate_gocube_checkpoints.py; do
  git show "origin/main:tools/$tool" > "$TMP_DIR/$tool"
done
chmod +x "$TMP_DIR"/*.sh "$TMP_DIR"/*.py

# Host sleep is explicitly outside the Linux-side preflight scope. The current
# preflight on older tooling revisions contained a WSL/powercfg block; remove
# only that block from the disposable copy, never from the frozen checkout.
python3 - "$TMP_DIR/preflight_c4_overnight.sh" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
start_marker = "# Under WSL, Linux inhibitors cannot prevent the Windows host from sleeping."
end_marker = 'CHECKPOINT_DIR="checkpoint/$SOURCE_RUN"'
if start_marker in s:
    start = s.index(start_marker)
    end = s.index(end_marker, start)
    s = s[:start] + "# Host power/sleep policy intentionally outside Linux preflight scope.\n\n" + s[end:]
p.write_text(s, encoding="utf-8")
PY
bash -n "$TMP_DIR/preflight_c4_overnight.sh"
if grep -q 'powercfg\|STANDBYIDLE\|HIBERNATEIDLE' "$TMP_DIR/preflight_c4_overnight.sh"; then
  fail "temporary preflight still contains Windows power-policy checks"
fi

EXP_ID="c4-hparam-night-$(date +%Y%m%d-%H%M%S)"
UNIT="gocube-c4-night-${EXP_ID##c4-hparam-night-}"
mkdir -p "training_reports/$EXP_ID/publish/evaluations"

PREFLIGHT_ID="${EXP_ID}-preflight"
echo "===== NIGHT PREFLIGHT ====="
echo "Experiment: $EXP_ID"
echo "Tooling commit: $TOOLING_COMMIT"
echo "Frozen training commit: $FROZEN_COMMIT"
echo
bash "$TMP_DIR/run_with_github_reports.sh" "$PREFLIGHT_ID" -- \
  bash "$TMP_DIR/preflight_c4_overnight.sh" "$SOURCE_RUN"

echo
echo "===== STARTING DETACHED NIGHT SERVICE ====="
SYSTEMD_COMMAND=$(printf '%q ' \
  bash "$TMP_DIR/run_with_github_reports.sh" "$EXP_ID" -- \
  "$REPO_ROOT/.venv/bin/python" "$TMP_DIR/c4_overnight_experiment.py" \
  --experiment-id "$EXP_ID" \
  --source-run "$SOURCE_RUN" \
  --frozen-commit "$FROZEN_COMMIT" \
  --max-hours "$MAX_HOURS")

systemd-run --user \
  --unit="$UNIT" \
  --collect \
  --property="WorkingDirectory=$REPO_ROOT" \
  --property="Restart=on-failure" \
  --property="RestartSec=30s" \
  --property="StartLimitIntervalSec=600" \
  --property="StartLimitBurst=3" \
  --setenv="PYTHONUNBUFFERED=1" \
  /bin/bash -lc "exec $SYSTEMD_COMMAND"

sleep 2
systemctl --user is-active --quiet "$UNIT.service" || {
  systemctl --user status "$UNIT.service" --no-pager >&2 || true
  fail "night service did not stay active after launch"
}

cat <<EOF
NIGHT LAUNCH PASS
Experiment: $EXP_ID
Unit: $UNIT.service
Max hours: $MAX_HOURS
Frozen training commit: $FROZEN_COMMIT
Tooling commit: $TOOLING_COMMIT
Auto-restart: on-failure, 30s delay, max 3 starts per 10 minutes

Status:
  systemctl --user status $UNIT.service

Live log:
  journalctl --user -u $UNIT.service -f

Local report:
  training_reports/$EXP_ID/publish/summary.md

GitHub report path:
  training-reports / reports/$EXP_ID/
EOF
