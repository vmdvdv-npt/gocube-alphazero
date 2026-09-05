#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_with_cloud_log.sh RUN_NAME -- COMMAND [ARGS...]

Environment variables:
  GOCUBE_REPORT_REMOTE       rclone destination base
                             default: gdrive:GoCube AlphaZero/Training Reports
  GOCUBE_REPORT_DIR          local report root
                             default: training_reports
  GOCUBE_REPORT_SYNC_SECONDS sync interval
                             default: 60
USAGE
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

RUN_NAME=$1
shift
if [[ ${1:-} == "--" ]]; then
  shift
fi
if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

REMOTE_BASE=${GOCUBE_REPORT_REMOTE:-gdrive:GoCube AlphaZero/Training Reports}
REPORT_ROOT=${GOCUBE_REPORT_DIR:-training_reports}
SYNC_SECONDS=${GOCUBE_REPORT_SYNC_SECONDS:-60}
RUN_DIR="$REPORT_ROOT/$RUN_NAME"
LOG_FILE="$RUN_DIR/console.log"
MD_FILE="$RUN_DIR/run.md"
REMOTE_DIR="$REMOTE_BASE/$RUN_NAME"

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone is not installed." >&2
  exit 3
fi

REMOTE_NAME="${REMOTE_BASE%%:*}:"
if ! rclone listremotes | grep -Fxq "$REMOTE_NAME"; then
  echo "ERROR: rclone remote $REMOTE_NAME is not configured." >&2
  exit 4
fi

mkdir -p "$RUN_DIR"
touch "$LOG_FILE"

STARTED_AT=$(date -Iseconds)
ENDED_AT=""
FINAL_STATUS="RUNNING"
EXIT_CODE=""
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || printf 'unknown')
WORKTREE=$(pwd)
printf -v COMMAND_Q '%q ' "$@"
COMMAND_Q=${COMMAND_Q% }

write_md() {
  cat > "$MD_FILE" <<EOF_MD
# Training run

- Run: $RUN_NAME
- Status: $FINAL_STATUS
- Started: $STARTED_AT
- Ended: ${ENDED_AT:-—}
- Exit code: ${EXIT_CODE:-—}
- Git commit: $GIT_COMMIT
- Working directory: $WORKTREE
- Cloud path: $REMOTE_DIR

## Command

\`\`\`bash
$COMMAND_Q
\`\`\`

## Files

- console.log — full terminal output
- run.md — this short run summary
EOF_MD
}

sync_once() {
  rclone mkdir "$REMOTE_DIR" >/dev/null 2>&1 || return 1
  rclone copyto "$MD_FILE" "$REMOTE_DIR/run.md" --retries 2 --low-level-retries 3 >/dev/null 2>&1 || return 1
  rclone copyto "$LOG_FILE" "$REMOTE_DIR/console.log" --retries 2 --low-level-retries 3 >/dev/null 2>&1 || return 1
}

write_md
if ! sync_once; then
  echo "ERROR: initial Google Drive sync failed; training was not started." >&2
  exit 5
fi

echo "Cloud logging: $REMOTE_DIR"
echo "Local logging: $RUN_DIR"

sync_loop() {
  while true; do
    sleep "$SYNC_SECONDS"
    if ! sync_once; then
      echo "WARNING: periodic cloud sync failed; will retry." >&2
    fi
  done
}

sync_loop &
SYNC_PID=$!

finalize() {
  trap - EXIT
  if kill -0 "$SYNC_PID" >/dev/null 2>&1; then
    kill "$SYNC_PID" >/dev/null 2>&1 || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  ENDED_AT=$(date -Iseconds)
  if [[ -z ${EXIT_CODE:-} ]]; then
    EXIT_CODE=130
    FINAL_STATUS="INTERRUPTED"
  elif [[ $EXIT_CODE -eq 0 ]]; then
    FINAL_STATUS="DONE"
  else
    FINAL_STATUS="FAILED"
  fi
  write_md
  if ! sync_once; then
    echo "WARNING: final cloud sync failed. Local files are in $RUN_DIR" >&2
  fi
}
trap finalize EXIT

set +e
"$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e
exit "$EXIT_CODE"
