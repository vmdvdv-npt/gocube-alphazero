from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RETIRED_DIRECTORIES = (
    "AlphaZeroGUI",
    "alphazero/pytorch_classification",
    "alphazero/envs/brandubh",
    "alphazero/envs/chess",
    "alphazero/envs/connect4",
    "alphazero/envs/gobang",
    "alphazero/envs/hnefatafl",
    "alphazero/envs/othello",
    "alphazero/envs/stratego",
    "alphazero/envs/tictactoe",
    "boardgame",
    "fastafl",
    "hnefatafl",
)


def _tracked_paths() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(item.decode("utf-8") for item in result.stdout.split(b"\0") if item)


def test_no_tracked_python_bytecode_or_cache_directories():
    tracked = _tracked_paths()
    forbidden = tuple(
        path
        for path in tracked
        if path.endswith((".pyc", ".pyo")) or "__pycache__" in Path(path).parts
    )
    assert forbidden == ()


def test_retired_directories_have_no_tracked_paths():
    tracked = _tracked_paths()
    for relative in RETIRED_DIRECTORIES:
        prefix = relative.rstrip("/") + "/"
        assert not any(
            path == relative or path.startswith(prefix) for path in tracked
        ), relative
