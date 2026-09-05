#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_with_github_reports.sh RUN_NAME -- COMMAND [ARGS...]

Environment variables:
  GOCUBE_REPORT_BRANCH       remote report branch (default: training-reports)
  GOCUBE_REPORT_WORKTREE     dedicated report worktree path
                             (default: ~/.cache/gocube-alphazero-training-reports)
  GOCUBE_REPORT_DIR          local full-log root (default: training_reports)
  GOCUBE_REPORT_POLL_SECONDS manifest poll interval (default: 15)
  GOCUBE_REPORT_TAIL_LINES   console lines published to GitHub (default: 2000)
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

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed." >&2
  exit 3
fi

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "ERROR: run this command inside the gocube-alphazero repository." >&2
  exit 4
}
cd "$REPO_ROOT"

REPORT_BRANCH=${GOCUBE_REPORT_BRANCH:-training-reports}
REPORT_WORKTREE=${GOCUBE_REPORT_WORKTREE:-$HOME/.cache/gocube-alphazero-training-reports}
REPORT_ROOT=${GOCUBE_REPORT_DIR:-training_reports}
POLL_SECONDS=${GOCUBE_REPORT_POLL_SECONDS:-15}
TAIL_LINES=${GOCUBE_REPORT_TAIL_LINES:-2000}
RUN_DIR="$REPORT_ROOT/$RUN_NAME"
LOG_FILE="$RUN_DIR/console.log"
MD_FILE="$RUN_DIR/run.md"
MANIFEST_ROOT="data/$RUN_NAME/records"
DEST_REL="reports/$RUN_NAME"

mkdir -p "$RUN_DIR"
touch "$LOG_FILE"

STARTED_AT=$(date -Iseconds)
ENDED_AT=""
FINAL_STATUS="RUNNING"
EXIT_CODE=""
SOURCE_COMMIT=$(git rev-parse HEAD 2>/dev/null || printf 'unknown')
printf -v COMMAND_Q '%q ' "$@"
COMMAND_Q=${COMMAND_Q% }
REPORTING_READY=0
WATCH_PID=""

manifest_count() {
  if [[ ! -d "$MANIFEST_ROOT" ]]; then
    printf '0\n'
    return
  fi
  find "$MANIFEST_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'iteration-manifest.json' -print0 2>/dev/null \
    | tr -cd '\0' | wc -c
}

manifest_signature() {
  if [[ ! -d "$MANIFEST_ROOT" ]]; then
    printf 'none\n'
    return
  fi
  local listing
  listing=$(find "$MANIFEST_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'iteration-manifest.json' \
    -printf '%p\t%s\t%T@\n' 2>/dev/null | LC_ALL=C sort)
  if [[ -z "$listing" ]]; then
    printf 'none\n'
  else
    printf '%s\n' "$listing" | sha256sum | awk '{print $1}'
  fi
}

write_md() {
  local completed
  completed=$(manifest_count)
  cat > "$MD_FILE" <<EOF_MD
# Training run

- Run: $RUN_NAME
- Status: $FINAL_STATUS
- Started: $STARTED_AT
- Ended: ${ENDED_AT:-—}
- Exit code: ${EXIT_CODE:-—}
- Source commit: $SOURCE_COMMIT
- Completed iteration manifests: $completed
- Reports branch: $REPORT_BRANCH

## Command

\`\`\`bash
$COMMAND_Q
\`\`\`

## Published to GitHub

- run.md — this status summary
- console-tail.log — last $TAIL_LINES console lines
- manifests/ — compact completed iteration manifests

Full console output remains local at \`$LOG_FILE\`.
EOF_MD
}

ensure_report_worktree() {
  if ! git fetch --quiet origin "$REPORT_BRANCH"; then
    echo "WARNING: cannot fetch origin/$REPORT_BRANCH; GitHub reporting is disabled for this run." >&2
    return 1
  fi

  if [[ -e "$REPORT_WORKTREE/.git" ]]; then
    if ! git -C "$REPORT_WORKTREE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "WARNING: $REPORT_WORKTREE exists but is not a git worktree; GitHub reporting is disabled." >&2
      return 1
    fi
    return 0
  fi

  if [[ -e "$REPORT_WORKTREE" ]] && [[ -n $(ls -A "$REPORT_WORKTREE" 2>/dev/null) ]]; then
    echo "WARNING: $REPORT_WORKTREE already exists and is not empty; GitHub reporting is disabled." >&2
    return 1
  fi

  mkdir -p "$(dirname "$REPORT_WORKTREE")"
  git worktree prune >/dev/null 2>&1 || true
  if ! git worktree add --quiet --detach "$REPORT_WORKTREE" "origin/$REPORT_BRANCH"; then
    echo "WARNING: could not create report worktree at $REPORT_WORKTREE; GitHub reporting is disabled." >&2
    return 1
  fi
}

copy_manifests() {
  local dest="$REPORT_WORKTREE/$DEST_REL/manifests"
  mkdir -p "$dest"
  [[ -d "$MANIFEST_ROOT" ]] || return 0

  local manifest iteration_name
  while IFS= read -r -d '' manifest; do
    iteration_name=$(basename "$(dirname "$manifest")")
    cp "$manifest" "$dest/${iteration_name}.json"
  done < <(find "$MANIFEST_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'iteration-manifest.json' -print0 2>/dev/null | sort -z)
}

sync_once() {
  local reason=${1:-update}
  [[ $REPORTING_READY -eq 1 ]] || return 1

  local common_git_dir lock_file
  common_git_dir=$(git rev-parse --git-common-dir)
  if [[ "$common_git_dir" != /* ]]; then
    common_git_dir="$REPO_ROOT/$common_git_dir"
  fi
  lock_file="$common_git_dir/gocube-training-reports.lock"

  (
    flock -x 9

    mkdir -p "$REPORT_WORKTREE/$DEST_REL"
    cp "$MD_FILE" "$REPORT_WORKTREE/$DEST_REL/run.md"
    tail -n "$TAIL_LINES" "$LOG_FILE" > "$REPORT_WORKTREE/$DEST_REL/console-tail.log"
    copy_manifests

    git -C "$REPORT_WORKTREE" add -- "$DEST_REL"
    if git -C "$REPORT_WORKTREE" diff --cached --quiet; then
      exit 0
    fi

    if ! git -C "$REPORT_WORKTREE" -c user.name='GoCube Training Reporter' \
      -c user.email='training-reports@localhost' \
      commit --quiet -m "reports: $RUN_NAME ($reason)"; then
      echo "WARNING: could not commit GitHub training report." >&2
      exit 1
    fi

    if git -C "$REPORT_WORKTREE" push --quiet origin "HEAD:refs/heads/$REPORT_BRANCH"; then
      exit 0
    fi

    echo "WARNING: report push was rejected; fetching and retrying once." >&2
    if ! git -C "$REPORT_WORKTREE" fetch --quiet origin "$REPORT_BRANCH"; then
      exit 1
    fi
    if ! git -C "$REPORT_WORKTREE" rebase --quiet "origin/$REPORT_BRANCH"; then
      git -C "$REPORT_WORKTREE" rebase --abort >/dev/null 2>&1 || true
      exit 1
    fi
    git -C "$REPORT_WORKTREE" push --quiet origin "HEAD:refs/heads/$REPORT_BRANCH"
  ) 9>"$lock_file"
}

write_md
if ensure_report_worktree; then
  REPORTING_READY=1
  if sync_once started; then
    echo "GitHub reporting: origin/$REPORT_BRANCH -> $DEST_REL"
  else
    echo "WARNING: initial GitHub report sync failed; training will continue with local logs." >&2
  fi
else
  echo "Local logging only: $RUN_DIR"
fi

echo "Local full log: $LOG_FILE"

watch_manifests() {
  local last current
  last=$(manifest_signature)
  while true; do
    sleep "$POLL_SECONDS"
    current=$(manifest_signature)
    if [[ "$current" != "$last" ]]; then
      write_md
      if ! sync_once iteration; then
        echo "WARNING: GitHub iteration report sync failed; will retry on the next completed iteration or at exit." >&2
      fi
      last=$current
    fi
  done
}

if [[ $REPORTING_READY -eq 1 ]]; then
  watch_manifests &
  WATCH_PID=$!
fi

finalize() {
  trap - EXIT
  if [[ -n ${WATCH_PID:-} ]] && kill -0 "$WATCH_PID" >/dev/null 2>&1; then
    kill "$WATCH_PID" >/dev/null 2>&1 || true
    wait "$WATCH_PID" 2>/dev/null || true
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

  if [[ $REPORTING_READY -eq 1 ]] && ! sync_once final; then
    echo "WARNING: final GitHub report sync failed. Local files are in $RUN_DIR" >&2
  fi
}
trap finalize EXIT

set +e
"$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e
exit "$EXIT_CODE"
