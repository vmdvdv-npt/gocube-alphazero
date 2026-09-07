#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SHA="f6bc4b19a1686caa2d088b56251e8c11c8be6d51"
SOURCE="$("$ROOT/tools/katago_reference/ensure_pinned_source.sh")"
BUILD="$ROOT/.cache/katago-reference/build-$SHA"
ORACLE="$BUILD/katago-rule-oracle"

if [[ -x "$ORACLE" && "$ORACLE" -nt "$ROOT/tools/katago_reference/oracle.cpp" && "$ORACLE" -nt "$0" ]]; then
  ACTUAL="$(git -C "$SOURCE" rev-parse HEAD)"
  [[ "$ACTUAL" == "$SHA" ]] || { echo "Oracle source is not pinned to $SHA" >&2; exit 3; }
  printf '%s\n' "$ORACLE"
  exit 0
fi

mkdir -p "$BUILD"

# Use only build tools and libraries already installed on the machine.  This
# intentionally does not invoke sudo, apt, or any system package manager.
if command -v make >/dev/null 2>&1 && [[ -f "$SOURCE/Makefile" ]]; then
  make -C "$SOURCE" -j2 kata 2>/dev/null || true
fi

# The rule engine does not depend on KataGo's file/logger helpers.  Keep the
# source list explicit so the oracle remains buildable without the optional
# ghc::filesystem header used by those helpers.
CORE_SOURCES=(
  "$SOURCE"/cpp/core/base64.cpp
  "$SOURCE"/cpp/core/bsearch.cpp
  "$SOURCE"/cpp/core/datetime.cpp
  "$SOURCE"/cpp/core/elo.cpp
  "$SOURCE"/cpp/core/global.cpp
  "$SOURCE"/cpp/core/hash.cpp
  "$SOURCE"/cpp/core/md5.cpp
  "$SOURCE"/cpp/core/multithread.cpp
  "$SOURCE"/cpp/core/rand.cpp
  "$SOURCE"/cpp/core/rand_helpers.cpp
  "$SOURCE"/cpp/core/sha2.cpp
  "$SOURCE"/cpp/core/test.cpp
  "$SOURCE"/cpp/core/threadsafecounter.cpp
  "$SOURCE"/cpp/core/threadsafequeue.cpp
  "$SOURCE"/cpp/core/timer.cpp
)

g++ -std=c++17 -O2 -pthread -DNO_GIT_REVISION=1 \
  -I"$SOURCE/cpp" -I"$SOURCE/cpp/external" \
  "$ROOT/tools/katago_reference/oracle.cpp" \
  "${CORE_SOURCES[@]}" \
  "$SOURCE"/cpp/game/board.cpp \
  "$SOURCE"/cpp/game/rules.cpp \
  "$SOURCE"/cpp/game/boardhistory.cpp \
  -o "$ORACLE" \
  2>"$BUILD/build.log" || {
    echo "Unable to build the pinned KataGo rule-only oracle with installed tools." >&2
    echo "See $BUILD/build.log; no system dependency was installed." >&2
    exit 4
  }
printf '%s\n' "$ORACLE"
