#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_RUN=${1:-c4-t001-c4-c001}
EXPECTED_HEAD=${GOCUBE_PREFLIGHT_EXPECTED_HEAD:-85c87a7cfd467a4d3f4b2844253fb63d746d672a}
EXPECTED_CHECKPOINT_SHA=${GOCUBE_PREFLIGHT_EXPECTED_CHECKPOINT_SHA:-64cf800460f6090880c7818cbeff80123257dcd14c79689f108cc5523fb58722}
EXPECTED_MANIFEST_SHA=${GOCUBE_PREFLIGHT_EXPECTED_MANIFEST_SHA:-909252d5d793c446b163837019098cb60036a5fa6c18b390ee154bcb9ff3414a}
EXPECTED_RECORDS=${GOCUBE_PREFLIGHT_EXPECTED_RECORDS:-256}
EXPECTED_NEXT_GAME_ID=${GOCUBE_PREFLIGHT_EXPECTED_NEXT_GAME_ID:-519}
WORKERS=${GOCUBE_PREFLIGHT_WORKERS:-16}
MIN_FREE_GIB=${GOCUBE_PREFLIGHT_MIN_FREE_GIB:-10}
SMOKE_GAMES=256
SMOKE_SIMS=2

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

for cmd in git sha256sum find sort awk cmp pgrep df xargs grep tee wc cp mktemp systemd-run systemctl; do
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

# The real overnight launcher will use the user systemd manager so an SSH or
# terminal disconnect cannot kill the experiment. Verify that path now.
systemctl --user show-environment >/dev/null 2>&1 \
  || fail "systemd user manager is unavailable"
systemd-run --user --wait --collect --quiet /bin/true >/dev/null 2>&1 \
  || fail "systemd user transient services cannot be started"
ok "systemd user service path is available for detached overnight execution"

# Under WSL, Linux inhibitors cannot prevent the Windows host from sleeping.
# Fail closed unless AC sleep and AC hibernation are both disabled.
if grep -qi microsoft /proc/sys/kernel/osrelease /proc/version 2>/dev/null; then
  POWERCFG=$(command -v powercfg.exe || true)
  [[ -n "$POWERCFG" ]] || fail "WSL detected but powercfg.exe is unavailable; cannot verify Windows host sleep policy"

  read_ac_timeout() {
    local setting=$1
    local output hex idx
    local -a values
    output=$("$POWERCFG" /query SCHEME_CURRENT SUB_SLEEP "$setting" 2>/dev/null | tr -d '\r') \
      || return 1
    mapfile -t values < <(printf '%s\n' "$output" | grep -oE '0x[0-9A-Fa-f]{8}')
    ((${#values[@]} >= 2)) || return 1
    idx=$((${#values[@]} - 2))
    hex=${values[$idx]#0x}
    printf '%d\n' "$((16#$hex))"
  }

  AC_SLEEP=$(read_ac_timeout STANDBYIDLE) \
    || fail "cannot determine Windows AC sleep timeout"
  AC_HIBERNATE=$(read_ac_timeout HIBERNATEIDLE) \
    || fail "cannot determine Windows AC hibernate timeout"
  (( AC_SLEEP == 0 )) \
    || fail "Windows AC sleep timeout is ${AC_SLEEP}s; set AC sleep to Never before overnight compute"
  (( AC_HIBERNATE == 0 )) \
    || fail "Windows AC hibernate timeout is ${AC_HIBERNATE}s; set AC hibernation to Never before overnight compute"
  ok "WSL host power policy: AC sleep=Never, AC hibernate=Never"
fi

CHECKPOINT_DIR="checkpoint/$SOURCE_RUN"
DATA_DIR="data/$SOURCE_RUN"
ITER1_RECORD_DIR="$DATA_DIR/records/iteration-0001"
CHECKPOINT1="$CHECKPOINT_DIR/iteration-0001.pkl"
MANIFEST1="$ITER1_RECORD_DIR/iteration-manifest.json"
GAME_ID_COUNTER="data/.gocube-game-ids/C4/game-id-counter.json"

[[ -f "$CHECKPOINT_DIR/iteration-0000.pkl" ]] || fail "missing iteration-0000 checkpoint"
[[ -f "$CHECKPOINT1" ]] || fail "missing iteration-0001 checkpoint"
[[ -f "$CHECKPOINT_DIR/gocube-run.json" ]] || fail "missing source run manifest"
[[ -f "$GAME_ID_COUNTER" ]] || fail "missing global C4 game-ID counter"

CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'iteration-*.pkl' | wc -l)
[[ "$CHECKPOINT_COUNT" -eq 2 ]] \
  || fail "source checkpoint directory contains $CHECKPOINT_COUNT iteration checkpoints; expected exactly 2 (0 and 1)"

DATA_SUFFIXES=(data policy value score ownership ownership-mask)
for suffix in "${DATA_SUFFIXES[@]}"; do
  [[ -f "$DATA_DIR/iteration-0001-$suffix.pkl" ]] || fail "missing iteration-0001-$suffix.pkl"
done
DATA_ITER_FILES=$(find "$DATA_DIR" -maxdepth 1 -type f -name 'iteration-*.pkl' | wc -l)
[[ "$DATA_ITER_FILES" -eq 6 ]] \
  || fail "source data directory contains $DATA_ITER_FILES iteration tensor files; expected exactly 6"
[[ -f "$MANIFEST1" ]] || fail "missing iteration-0001 record manifest"
RECORD_ITER_DIRS=$(find "$DATA_DIR/records" -mindepth 1 -maxdepth 1 -type d -name 'iteration-*' | wc -l)
[[ "$RECORD_ITER_DIRS" -eq 1 ]] \
  || fail "source records contain $RECORD_ITER_DIRS iteration directories; expected exactly iteration-0001"

ACTUAL_CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT1" | awk '{print $1}')
[[ "$ACTUAL_CHECKPOINT_SHA" == "$EXPECTED_CHECKPOINT_SHA" ]] \
  || fail "iteration-0001 checkpoint SHA256 mismatch: $ACTUAL_CHECKPOINT_SHA"
ACTUAL_MANIFEST_SHA=$(sha256sum "$MANIFEST1" | awk '{print $1}')
[[ "$ACTUAL_MANIFEST_SHA" == "$EXPECTED_MANIFEST_SHA" ]] \
  || fail "iteration-0001 manifest SHA256 mismatch: $ACTUAL_MANIFEST_SHA"

RECORD_COUNT=$(find "$ITER1_RECORD_DIR" -maxdepth 1 -type f -name '*.json' ! -name 'iteration-manifest.json' | wc -l)
[[ "$RECORD_COUNT" -eq "$EXPECTED_RECORDS" ]] \
  || fail "iteration-0001 record count is $RECORD_COUNT, expected $EXPECTED_RECORDS"

# Refuse to continue over any partial/unknown later state.
if find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'iteration-*.pkl' ! -name 'iteration-0000.pkl' ! -name 'iteration-0001.pkl' | grep -q .; then
  fail "source run has an unexpected checkpoint beyond iteration 1"
fi
if find "$DATA_DIR" -maxdepth 1 -type f -name 'iteration-*.pkl' ! -name 'iteration-0001-*.pkl' | grep -q .; then
  fail "source run has unexpected iteration tensor data"
fi
if find "$DATA_DIR/records" -mindepth 1 -maxdepth 1 -type d -name 'iteration-*' ! -name 'iteration-0001' | grep -q .; then
  fail "source run has unexpected later record directories"
fi
ok "published iteration-0001 artifacts match expected hashes/counts and no later state exists"

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

"$PYTHON" - "$SOURCE_RUN" "$WORKERS" "$EXPECTED_RECORDS" "$EXPECTED_NEXT_GAME_ID" <<'PY'
import hashlib
import json
import os
import sys

import torch

run_name = sys.argv[1]
workers = int(sys.argv[2])
expected_records = int(sys.argv[3])
expected_next_game_id = int(sys.argv[4])

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

expected_args = {
    "gocube_topology": "cube",
    "gocube_size": 4,
    "gocube_komi": 7.5,
    "workers": 2,
    "numMCTSSims": 100,
    "numFastSims": 20,
    "probFastSim": 0.25,
    "gamesPerIteration": 256,
    "train_batch_size": 256,
    "gocube_endgame_sample_weight": 1,
    "arenaMCTSSims": 100,
    "arenaTemp": 0.0,
    "arenaBatched": False,
    "model_gating": False,
}
for name, expected in expected_args.items():
    actual = arg(name)
    if actual != expected:
        raise SystemExit(f"PREFLIGHT FAIL: checkpoint arg {name}={actual!r}, expected {expected!r}")

run_manifest_path = os.path.join("checkpoint", run_name, "gocube-run.json")
with open(run_manifest_path, "r", encoding="utf-8") as handle:
    run_manifest = json.load(handle)
expected_manifest_fields = {
    "version": 3,
    "runName": run_name,
    "topology": "cube",
    "size": 4,
    "ruleSet": "japanese",
    "komi": 7.5,
    "terminalAdjudicator": "gocube-katago-japanese-v3",
}
for name, expected in expected_manifest_fields.items():
    if run_manifest.get(name) != expected:
        raise SystemExit(
            f"PREFLIGHT FAIL: run manifest {name}={run_manifest.get(name)!r}, expected {expected!r}"
        )

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
records = manifest.get("records", [])
if len(records) != expected_records:
    raise SystemExit("PREFLIGHT FAIL: iteration manifest record count mismatch")
if int(manifest.get("aggregate_metrics", {}).get("games", -1)) != expected_records:
    raise SystemExit("PREFLIGHT FAIL: aggregate game count mismatch")

ids = [entry.get("game_id") for entry in records]
if len(set(ids)) != expected_records:
    raise SystemExit("PREFLIGHT FAIL: duplicate game IDs in iteration manifest")
if ids[0] != "C4-000263" or ids[-1] != "C4-000518":
    raise SystemExit(f"PREFLIGHT FAIL: unexpected game-ID range: {ids[0]}..{ids[-1]}")

for entry in records:
    record_path = entry.get("record_path")
    if not record_path or not os.path.isfile(record_path):
        raise SystemExit(f"PREFLIGHT FAIL: missing game record: {record_path!r}")
    with open(record_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != entry.get("sha256"):
        raise SystemExit(f"PREFLIGHT FAIL: game record SHA mismatch: {record_path}")

counter_path = os.path.join("data", ".gocube-game-ids", "C4", "game-id-counter.json")
with open(counter_path, "r", encoding="utf-8") as handle:
    counter = json.load(handle)
if counter.get("prefix") != "C4" or int(counter.get("next_number", -1)) != expected_next_game_id:
    raise SystemExit(f"PREFLIGHT FAIL: unexpected C4 game-ID counter: {counter!r}")

print(
    "PREFLIGHT OK: Python/checkpoint/tensors/records valid; "
    f"cpu_count={cpu_count}; tensor_rows={counts[0]}; baseline args and all record hashes verified"
)
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

# Keep the disposable test on the same filesystem as the real project so the
# disk-space check and I/O path are representative. training_reports is ignored.
mkdir -p "$REPO_ROOT/training_reports"
SANDBOX=$(mktemp -d "$REPO_ROOT/training_reports/.c4-preflight.XXXXXX")
CLONE_RUN="c4-preflight-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$SANDBOX/checkpoint/$CLONE_RUN" "$SANDBOX/data/$CLONE_RUN"
cp "$CHECKPOINT_DIR/iteration-0000.pkl" "$SANDBOX/checkpoint/$CLONE_RUN/"
cp "$CHECKPOINT_DIR/iteration-0001.pkl" "$SANDBOX/checkpoint/$CLONE_RUN/"
for suffix in "${DATA_SUFFIXES[@]}"; do
  cp "$DATA_DIR/iteration-0001-$suffix.pkl" "$SANDBOX/data/$CLONE_RUN/"
done

MICRO_LOG="$SANDBOX/micro-training.log"
echo "PREFLIGHT: disposable resume test run=$CLONE_RUN, workers=$WORKERS, games=$SMOKE_GAMES, sims=$SMOKE_SIMS"
(
  cd "$SANDBOX"
  PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/alphazero/envs/gocube/train.py" \
    --topology cube \
    --size 4 \
    --workers "$WORKERS" \
    --sims "$SMOKE_SIMS" \
    --arena-sims 100 \
    --games-per-iteration "$SMOKE_GAMES" \
    --iterations 2 \
    --train-batch-size 256 \
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
  PYTHONPATH="$REPO_ROOT" "$PYTHON" - "$CLONE_RUN" "$WORKERS" "$SMOKE_GAMES" "$SMOKE_SIMS" <<'PY'
import json
import os
import sys
import torch

run_name = sys.argv[1]
workers = int(sys.argv[2])
smoke_games = int(sys.argv[3])
smoke_sims = int(sys.argv[4])
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
if len(manifest.get("records", [])) != smoke_games:
    raise SystemExit(
        f"PREFLIGHT FAIL: generated manifest has {len(manifest.get('records', []))} records, expected {smoke_games}"
    )
params = manifest.get("effective_iteration_parameters", {})
checks = {
    "workers": workers,
    "gamesPerIteration": smoke_games,
    "process_batch_size": smoke_games // workers,
    "train_batch_size": 256,
    "numMCTSSims": smoke_sims,
    "arenaMCTSSims": 100,
    "probFastSim": 0.0,
    "compareWithBaseline": False,
    "compareWithPast": False,
}
for key, expected in checks.items():
    if params.get(key) != expected:
        raise SystemExit(f"PREFLIGHT FAIL: generated manifest {key}={params.get(key)!r}, expected {expected!r}")
print(
    f"PREFLIGHT OK: disposable iteration 2 complete/readable; workers={workers}; "
    f"games={smoke_games}; process_batch_size={smoke_games // workers}; tensor_rows={counts[0]}"
)
PY
)

# A second invocation at the same target must load iteration 2 cleanly and do no new work.
(
  cd "$SANDBOX"
  PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/alphazero/envs/gocube/train.py" \
    --topology cube \
    --size 4 \
    --workers "$WORKERS" \
    --sims "$SMOKE_SIMS" \
    --arena-sims 100 \
    --games-per-iteration "$SMOKE_GAMES" \
    --iterations 2 \
    --train-batch-size 256 \
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
git diff --quiet || fail "tracked working tree changed during preflight"
git diff --cached --quiet || fail "git index changed during preflight"
ok "source run, global C4 game-ID registry, and tracked checkout are unchanged"

echo "PREFLIGHT PASS"
echo "Source run: $SOURCE_RUN"
echo "Frozen source commit: $EXPECTED_HEAD"
echo "Workers validated: $WORKERS"
echo "Production batching geometry validated: $SMOKE_GAMES games / $WORKERS workers = $((SMOKE_GAMES / WORKERS)) process batch"
echo "Disposable resume: iteration 1 -> 2 completed"
echo "Arena during smoke: OFF"
echo "Original training state modified: NO"
