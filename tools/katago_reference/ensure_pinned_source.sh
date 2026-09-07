#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SHA="f6bc4b19a1686caa2d088b56251e8c11c8be6d51"
CACHE="$ROOT/.cache/katago-reference/$SHA"
REPO_URL="https://github.com/lightvector/KataGo.git"

mkdir -p "$(dirname "$CACHE")"
if [[ ! -d "$CACHE/.git" ]]; then
  rm -rf "$CACHE"
  git clone "$REPO_URL" "$CACHE"
fi

git -C "$CACHE" fetch --all --tags --prune
git -C "$CACHE" checkout --detach "$SHA"
ACTUAL="$(git -C "$CACHE" rev-parse HEAD)"
if [[ "$ACTUAL" != "$SHA" ]]; then
  echo "Pinned KataGo commit mismatch: expected $SHA, got $ACTUAL" >&2
  exit 3
fi
printf '%s\n' "$CACHE"
