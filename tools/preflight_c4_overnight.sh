#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_RUN=${1:-c4-t001-c4-c001}
EXPECTED_HEAD=${GOCUBE_PREFLIGHT_EXPECTED_HEAD:-85c87a7cfd467a4d3f4b2844253fb63d746d672a}
EXPECTED_CHECKPOINT_SHA=${GOCUBE_PREFLIGHT_EXPECTED_CHECKPOINT_SHA:-64cf800460f6090880c7818cbeff80123257dcd14c79689f108cc5523fb58722}
EXPECTED_MANIFEST_SHA=${GOCUBE_PREFLIGHT_EXPECTED_MANIFEST_SHA:-909252d5d793c446b163837019098cb60036a5fa6c18b390ee154bcb9ff3414a}
EXPECTED_RECORDS=${GOCUBE_PREFLIGHT_EXPECTED_RECORDS:-256}
WORKERS=${GOCUBE_PREFLIGHT_WORKERS:-16}
MIN_FREE_GIB=${GOCUBE_PREFLIGHT_MIN_FREE_GIB:-10}

fail() {
  echo "PREFLIGHT FAIL: $*" >&2
  exit 1
}

ok() {
  echo "PREFLIGHT OK: $*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || fail "run inside the gocube-alphazero repository"
cd "$REPO_ROOT"
PYTHON="$REPO_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || fail "missing executable virtualenv Python: $PYTHON"

for cmd in git sha256sum find sort awk cmp pgrep df; do
  need_cmd "$cmd"
done

if pgrep -af 'python[^ ]* .*alphazero/envs/gocube/train.py' >/tmp/gocube-preflight-running.$$ 2>/dev/null; then
  cat /tmp/gocube-preflight-running.$$ >&2 || true
  rm -f /tmp/gocube-preflight-running.$$
  fail "another GoCube training process is already running"
fi
rm -f /tmp/gocube-preflight-running.$$ 2>/dev/null || true

CURRENT_HEAD=$(git rev-parse HEAD)
[[ "$CURRENT_HEAD" == "$EXPECTED_HEAD" ]] || fail "training checkout HEAD is $CURRENT_HEAD, expected frozen source $EXPECTED_HEAD"
git diff --quiet || fail "tracked working tree has unstaged changes"
git diff --cached --quiet || fail "tracked working tree has staged changes"
ok "training checkout is the frozen source commit and tracked files are clean"

# The reporting infrastructure may be newer than the frozen training checkout,
# but the actual training code must remain pinned to EXPECTED_HEAD.
git fetch --quiet origin training-reports || fail "cannot fetch origin/training-reports"
git push --dry-run --quiet origin refs/remotes/origin/training-reports:refs/heads/training-reports \
  || fail "terminal cannot authenticate/push to origin/training-reports"
ok "GitHub training-reports push permission is available"

CHECKPOINT_DIR="checkpoint/$SOURCE_RUN"
DATA_DIR="data/$SOURCE_RUN"
ITER1_RECORD_DIR="$DATA_DIR/records/iteration-0001"
CHECKPOINT1="$CHECKPOINT_DIR/iteration-0001.pkl"
MANIFEST1="$ITER1_RECORD_DIR/iteration-manifest.json"

[[ -f "$CHECKPOINT_DIR/iteration-0000.pkl" ]] || fail "missing iteration-0000 checkpoint"
[[ -f "$CHECKPOINT1" ]] || fail "missing iteration-0001 checkpoint"
[[ -f "$CHECKPOINT_DIR/gocube-run.json" ]] || fail "missing source run manifest"

DATA_SUFFIXES=(data policy value score ownership ownership-mask)
for suffix in "${DATA_SUFFIXES[@]}"; do
  [[ -f "$DATA_DIR/iteration-0001-$suffix.pkl" ]] || fail "missing iteration-0001-$suffix.pkl"
done
[[ -f "$MANIFEST1" ]] || fail "missing iteration-0001 record manifest"

ACTUAL_CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT1" | awk '{print $1}')
[[ "$ACTUAL_CHECKPOINT_SHA" == "$EXPECTED_CHECKPOINT_SHA" ]] \
  || fail "iteration-0001 checkpoint SHA256 mismatch: $ACTUAL_CHECKPOINT_SHA"
ACTUAL_MANIFEST_SHA=$(sha256sum "$MANIFEST1" | awk '{print $1}')
[[ "$ACTUAL_MANIFEST_SHA" == "$EXPECTED_MANIFEST_SHA" ]] \
  || fail "iteration-0001 manifest SHA256 mismatch: $ACTUAL_MANIFEST_SHA"

RECORD_COUNT=$(find "$ITER1_RECORD_DIR" -maxdepth 1 -type f -name '*.json' ! -name 'iteration-manifest.json' | wc -l)
[[ "$RECORD_COUNT" -eq "$EXPECTED_RECORDS" ]] \
  || fail "iteration-0001 record count is $RECORD_COUNT, expected $EXPECTED_RECORDS"

# Refuse to continue over any partial/unknown iteration 2 state.
[[ ! -e "$CHECKPOINT_DIR/iteration-0002.pkl" ]] || fail "source run already has iteration-0002 checkpoint"
if find "$DATA_DIR" -maxdepth 1 -type f -name 'iteration-0002-*.pkl' | grep -q .; then
  fail "source run has partial iteration-0002 tensor data"
fi
[[ ! -d "$DATA_DIR/records/iteration-0002" ]] || fail "source run has partial iteration-0002 records"
ok "published iteration-0001 artifacts match expected hashes/counts and iteration 2 is untouched"

AVAILABLE_KB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
REQUIRED_KB=$((MIN_FREE_GIB * 1024 * 1024))
[[ "$AVAILABLE_KB" -ge "$REQUIRED_KB" ]] \
  || fail "only $((AVAILABLE_KB / 1024 / 1024)) GiB free; require at least $MIN_FREE_GIB GiB"
ok "disk free: $((AVAILABLE_KB / 1024 / 1024)) GiB (minimum $MIN_FREE_GIB GiB)"

NOFILE=$(ulimit -n)
if [[ "$NOFILE" =~ ^[0-9]+$ ]] && (( NOFILE < 1024 )); then
  fail "open-file limit is only $NOFILE; require at least 1024"
fi
ok "open-file limit: $NOFILE"

"$PYTHON" - "$SOURCE_RUN" "$WORKERS" "$EXPECTED_RECORDS" <<'PY'
import json
import os
import sys

import torch

run_name = sys.argv[1]
workers = int(sys.argv[2])
expected_records = int(sys.argv[3])

cpu_count = os.cpu_count() or 0
if cpu_count < workers:
    raise SystemExit(f"PREFLIGHT FAIL: os.cpu_count()={cpu_count}, need at least {workers} workers")

checkpoint_path = os.path.join("checkpoint", run_name, "iteration-0001.pkl")
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
required = {"state_dict", "opt_state", "sch_state", "args"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"PREFLIGHT FAIL: checkpoint missing keys: {sorted(missing)}")
args = checkpoint["args"]
def arg(name):
    return args.get(name) if hasattr(args, "get") else getattr(args, name)
if arg("gocube_topology") != "cube" or int(arg("gocube_size")) != 4:
    raise SystemExit("PREFLIGHT FAIL: checkpoint is not Cube 4")
if float(arg("gocube_komi")) != 7.5:
    raise SystemExit("PREFLIGHT FAIL: checkpoint komi is not 7.5")

suffixes = ("data", "policy", "value", "score", "ownership", "ownership-mask")
counts = []
for suffix in suffixes:
    path = os.path.join("data", run_name, f"iteration-0001-{suffix}.pkl")
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    counts.append(int(tensor.size(0)))
if len(set(counts)) != 1 or counts[0] <= 0:
    raise SystemExit(f"PREFLIGHT FAIL: iteration-0001 tensor row mismatch: {counts}")

manifest_path = os.path.join("data", run_name, "records", "iteration-0001", "iteration-manifest.json")
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("run_name") != run_name or int(manifest.get("iteration", -1)) != 1:
    raise SystemExit("PREFLIGHT FAIL: iteration manifest identity mismatch")
if len(manifest.get("records", [])) != expected_records:
    raise SystemExit("PREFLIGHT FAIL: iteration manifest record count mismatch")

print(f"PREFLIGHT OK: Python dependencies/checkpoint/tensors valid; cpu_count={cpu_count}; tensor_rows={counts[0]}")
PY

SOURCE_SNAPSHOT_BEFORE=$(mktemp)
SOURCE_SNAPSHOT_AFTER=$(mktemp)
SANDBOX=""
cleanup() {
  if [[ -n ${SANDBOX:-} && -d "$SANDBOX" ]]; then
    rm -rf "$SANDBOX"
  fi
  rm -f "$SOURCE_SNAPSHOT_BEFORE" "$SOURCE_SNAPSHOT_AFTER" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

snapshot_source() {
  {
    find "$CHECKPOINT_DIR" -type f -print0
    find "$DATA_DIR" -type f -print0
    if [[ -d data/.gocube-game-ids/C4 ]]; then
      find data/.gocube-game-ids/C4 -type f -print0
    fi
  } | sort -z | xargs -0 sha256sum | LC_ALL=C sort
}

snapshot_source > "$SOURCE_SNAPSHOT_BEFORE"

SANDBOX=$(mktemp -d "${TMPDIR:-/tmp}/gocube-c4-preflight.XXXXXX")
CLONE_RUN="c4-preflight-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$SANDBOX/checkpoint/$CLONE_RUN" "$SANDBOX/data/$CLONE_RUN"
cp "$CHECKPOINT_DIR/iteration-0000.pkl" "$SANDBOX/checkpoint/$CLONE_RUN/"
cp "$CHECKPOINT_DIR/iteration-0001.pkl" "$SANDBOX/checkpoint/$CLONE_RUN/"
for suffix in "${DATA_SUFFIXES[@]}"; do
  cp "$DATA_DIR/iteration-0001-$suffix.pkl" "$SANDBOX/data/$CLONE_RUN/"
done

MICRO_LOG="$SANDBOX/micro-training.log"
echo "PREFLIGHT: disposable resume test run=$CLONE_RUN, workers=$WORKERS"
(
  cd "$SANDBOX"
  PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/alphazero/envs/gocube/train.py" \
    --topology cube \
    --size 4 \
    --workers "$WORKERS" \
    --sims 2 \
    --arena-sims 2 \
    --games-per-iteration "$WORKERS" \
    --iterations 2 \
    --train-batch-size "$WORKERS" \
    --train-steps-per-iteration 1 \
    --fast-game-prob 0 \
    --endgame-sample-weight 1 \
    --no-arena \
    --run-name "$CLONE_RUN"
) 2>&1 | tee "$MICRO_LOG"

[[ -f "$SANDBOX/checkpoint/$CLONE_RUN/iteration-0002.pkl" ]] \
  || fail "disposable resume did not create iteration-0002 checkpoint"
for suffix in "${DATA_SUFFIXES[@]}"; do
  [[ -f "$SANDBOX/data/$CLONE_RUN/iteration-0002-$suffix.pkl" ]] \
    || fail "disposable resume missing iteration-0002-$suffix.pkl"
done
MICRO_MANIFEST="$SANDBOX/data/$CLONE_RUN/records/iteration-0002/iteration-manifest.json"
[[ -f "$MICRO_MANIFEST" ]] || fail "disposable resume missing iteration-0002 manifest"

grep -Eq 'Optimizer steps actual:[[:space:]]+1' "$MICRO_LOG" \
  || fail "disposable resume did not complete exactly one optimizer step"
grep -Fq 'Arena: OFF' "$MICRO_LOG" \
  || fail "disposable resume unexpectedly enabled Arena"

(
  cd "$SANDBOX"
  PYTHONPATH="$REPO_ROOT" "$PYTHON" - "$CLONE_RUN" "$WORKERS" <<'PY'
import json
import os
import sys
import torch

run_name = sys.argv[1]
workers = int(sys.argv[2])
checkpoint_path = os.path.join("checkpoint", run_name, "iteration-0002.pkl")
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
for key in ("state_dict", "opt_state", "sch_state", "args"):
    if key not in checkpoint:
        raise SystemExit(f"PREFLIGHT FAIL: generated checkpoint missing {key}")

suffixes = ("data", "policy", "value", "score", "ownership", "ownership-mask")
counts = []
for suffix in suffixes:
    path = os.path.join("data", run_name, f"iteration-0002-{suffix}.pkl")
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    counts.append(int(tensor.size(0)))
if len(set(counts)) != 1 or counts[0] <= 0:
    raise SystemExit(f"PREFLIGHT FAIL: generated tensor row mismatch: {counts}")

manifest_path = os.path.join("data", run_name, "records", "iteration-0002", "iteration-manifest.json")
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("run_name") != run_name or int(manifest.get("iteration", -1)) != 2:
    raise SystemExit("PREFLIGHT FAIL: generated manifest identity mismatch")
if len(manifest.get("records", [])) != workers:
    raise SystemExit(f"PREFLIGHT FAIL: generated manifest has {len(manifest.get('records', []))} records, expected {workers}")
params = manifest.get("effective_iteration_parameters", {})
checks = {
    "workers": workers,
    "gamesPerIteration": workers,
    "numMCTSSims": 2,
    "probFastSim": 0.0,
}
for key, expected in checks.items():
    if params.get(key) != expected:
        raise SystemExit(f"PREFLIGHT FAIL: generated manifest {key}={params.get(key)!r}, expected {expected!r}")
print(f"PREFLIGHT OK: disposable iteration 2 is complete/readable with {workers} workers and {counts[0]} tensor rows")
PY
)

# A second invocation at the same target must load iteration 2 cleanly and do no new work.
(
  cd "$SANDBOX"
  PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/alphazero/envs/gocube/train.py" \
    --topology cube \
    --size 4 \
    --workers "$WORKERS" \
    --sims 2 \
    --arena-sims 2 \
    --games-per-iteration "$WORKERS" \
    --iterations 2 \
    --train-batch-size "$WORKERS" \
    --train-steps-per-iteration 1 \
    --fast-game-prob 0 \
    --endgame-sample-weight 1 \
    --no-arena \
    --run-name "$CLONE_RUN" >/dev/null
)
[[ ! -e "$SANDBOX/checkpoint/$CLONE_RUN/iteration-0003.pkl" ]] \
  || fail "resume idempotence check unexpectedly created iteration 3"
ok "generated checkpoint can be loaded again and resume indexing is correct"

snapshot_source > "$SOURCE_SNAPSHOT_AFTER"
cmp -s "$SOURCE_SNAPSHOT_BEFORE" "$SOURCE_SNAPSHOT_AFTER" \
  || fail "SOURCE RUN CHANGED during disposable preflight"
[[ ! -e "$CHECKPOINT_DIR/iteration-0002.pkl" ]] || fail "source iteration 2 appeared during preflight"
ok "source run and global C4 game-ID registry are byte-for-byte unchanged"

echo "PREFLIGHT PASS"
echo "Source run: $SOURCE_RUN"
echo "Frozen source commit: $EXPECTED_HEAD"
echo "Workers validated: $WORKERS"
echo "Disposable resume: iteration 1 -> 2 completed"
echo "Arena during smoke: OFF"
echo "Original training state modified: NO"
