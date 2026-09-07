"""Differential runner for the pinned KataGo JSONL rule oracle.

The oracle is the source of legal moves and expected values.  This module only
adapts coordinates and normalizes snapshots; it never computes expected rules
outcomes from GoCube and then compares GoCube with itself.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from alphazero.envs.gocube.katago_v3 import (
    CLEANUP_1,
    CLEANUP_2,
    MAIN,
    NO_RESULT,
    SCORED,
    V3IllegalMove,
    _pseudolegal_candidate,
    apply_v3_action,
    initial_v3_state,
    terminal_from_state,
    v3_state_from_board,
    v3_valid_moves,
)
from alphazero.envs.gocube.core import Topology

from gocube_reference_topology import rectangular_test_topology


KATAGO_REFERENCE_COMMIT = "f6bc4b19a1686caa2d088b56251e8c11c8be6d51"
ROOT = Path(__file__).resolve().parents[1]
ORACLE_BUILD = ROOT / "tools" / "katago_reference" / "build_oracle.sh"


def oracle_path() -> str:
    configured = os.environ.get("GOCUBE_KATAGO_ORACLE")
    if configured:
        path = Path(configured)
    else:
        result = subprocess.run(
            [os.fspath(ORACLE_BUILD)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        path = Path(result.stdout.strip().splitlines()[-1])
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"KataGo reference oracle is not executable: {path}")
    return os.fspath(path)


class KatagoOracleProcess:
    def __init__(self, *, x_size: int, y_size: int, komi: float = 0.5):
        if float(komi) != 0.5:
            raise ValueError("KataGo reference adapter only permits komi=0.5")
        self.process = subprocess.Popen(
            [oracle_path()],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        response = self.request({"op": "new", "x_size": int(x_size), "y_size": int(y_size), "komi": 0.5})
        self._check_commit(response)

    def _check_commit(self, response: dict[str, Any]) -> None:
        actual = response.get("katago_commit")
        if actual is None and isinstance(response.get("snapshot"), dict):
            actual = response["snapshot"].get("katago_commit")
        if actual != KATAGO_REFERENCE_COMMIT:
            raise AssertionError(
                f"KataGo reference commit mismatch: expected {KATAGO_REFERENCE_COMMIT}, got {actual!r}"
            )

    def request(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("KataGo oracle pipes are unavailable")
        self.process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError(f"KataGo oracle exited without a response: {stderr}")
        response = json.loads(line)
        self._check_commit(response)
        return response

    def setup(self, setup: dict[str, Any]) -> dict[str, Any]:
        payload = {"op": "setup", **setup}
        return self.request(payload)

    def play(self, move: Any) -> dict[str, Any]:
        return self.request({"op": "play", "move": move})

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=5)

    def __enter__(self) -> "KatagoOracleProcess":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _local_state_from_setup(topology: Topology, setup: dict[str, Any]):
    def point(item):
        x, y = (int(item[0]), int(item[1]))
        if not (0 <= x < topology.size and 0 <= y < len(topology.point_ids) // topology.size):
            raise ValueError(f"point outside test board: {item}")
        return y * topology.size + x

    return v3_state_from_board(
        topology,
        black=(point(item) for item in setup.get("black", [])),
        white=(point(item) for item in setup.get("white", [])),
        current_player=0 if setup.get("next_player", "B") in ("B", "black") else 1,
    )


def _local_simple_ko(state, topology: Topology):
    if state.phase != MAIN or state.previous_board is None:
        return None
    old = np.asarray(state.previous_board)
    current = np.asarray(state.board)
    emptied = np.flatnonzero((old != 0) & (current == 0))
    added = np.flatnonzero((old == 0) & (current != 0))
    if len(emptied) != 1 or len(added) != 1:
        return None
    try:
        candidate, _ = _pseudolegal_candidate(current, state.current_player, int(emptied[0]), topology)
    except V3IllegalMove:
        return None
    if np.array_equal(candidate, old):
        point = int(emptied[0])
        return [point % topology.size, point // topology.size]
    return None


def local_snapshot(state, topology: Topology) -> dict[str, Any]:
    terminal = terminal_from_state(state, topology, 0.5)
    if state.phase == MAIN:
        phase = "MAIN"
    elif state.phase == CLEANUP_1:
        phase = "CLEANUP_1"
    elif state.phase == CLEANUP_2:
        phase = "CLEANUP_2"
    elif state.phase == SCORED:
        phase = "SCORED"
    elif state.phase == NO_RESULT:
        phase = "NO_RESULT"
    else:
        raise AssertionError(f"unknown local phase: {state.phase}")
    score = None
    winner = None
    if terminal is not None:
        winner = terminal.winner
        if terminal.score is not None:
            score = float(terminal.score.white - terminal.score.black)
    return {
        "board": [int(value) for value in np.asarray(state.board).reshape(-1)],
        "next_player": "B" if state.current_player == 0 else "W",
        "legal_mask": [int(value) for value in v3_valid_moves(state, topology)],
        "phase": phase,
        "simple_ko": _local_simple_ko(state, topology),
        "ko_recap_blocked": [
            [int(point) % topology.size, int(point) // topology.size]
            for point in state.ko_recap_blocked
        ],
        "is_game_finished": state.terminal_kind == SCORED,
        "is_no_result": state.terminal_kind == NO_RESULT,
        "winner": winner,
        "final_score": score,
        # KataGo exposes prisoner counters by captured colour, while GoCube's
        # state stores captures by capturing player.
        "captures": {"black": int(state.captures[1]), "white": int(state.captures[0])},
        "encore_phase": {MAIN: 0, CLEANUP_1: 1, CLEANUP_2: 2, SCORED: 2, NO_RESULT: 0}[state.phase],
    }


SNAPSHOT_FIELDS = (
    "board",
    "next_player",
    "legal_mask",
    "phase",
    "simple_ko",
    "ko_recap_blocked",
    "is_game_finished",
    "is_no_result",
    "winner",
    "final_score",
    "captures",
)


def assert_snapshot_equal(reference: dict[str, Any], local: dict[str, Any], *, context: str = "") -> None:
    for field in SNAPSHOT_FIELDS:
        expected = reference.get(field)
        actual = local.get(field)
        if field == "final_score" and expected is not None and actual is not None:
            if not np.isclose(float(expected), float(actual), rtol=0.0, atol=1e-6):
                raise AssertionError(f"{context} field={field}: KataGo={expected!r}, GoCube={actual!r}")
        elif expected != actual:
            raise AssertionError(f"{context} field={field}: KataGo={expected!r}, GoCube={actual!r}")


def run_fixture(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    if fixture.get("katago_commit") != KATAGO_REFERENCE_COMMIT:
        raise AssertionError(f"Fixture {fixture.get('id')} is not pinned to the required KataGo commit")
    width, height = (int(value) for value in fixture["board_size"])
    topology = rectangular_test_topology(width, height)
    setup = fixture.get("setup", {})
    local_state = _local_state_from_setup(topology, setup)
    snapshots = []
    with KatagoOracleProcess(x_size=width, y_size=height, komi=0.5) as oracle:
        reference = oracle.setup(setup) if setup else oracle.request({"op": "snapshot"})
        assert_snapshot_equal(reference, local_snapshot(local_state, topology), context=f"{fixture['id']} before")
        snapshots.append(reference)
        for move_number, move in enumerate(fixture.get("moves", []), start=1):
            reference = oracle.play(move)
            action = width * height if move == "pass" else int(move[1]) * width + int(move[0])
            legal = int(v3_valid_moves(local_state, topology)[action])
            if bool(reference.get("ok", True)) != bool(legal):
                raise AssertionError(
                    f"{fixture['id']} move={move_number} move={move!r}: "
                    f"oracle_ok={reference.get('ok')} local_legal={legal}"
                )
            if legal:
                local_state = apply_v3_action(local_state, action, topology)
            local = local_snapshot(local_state, topology)
            response_snapshot = reference.get("snapshot", reference)
            assert_snapshot_equal(
                response_snapshot,
                local,
                context=f"{fixture['id']} move={move_number} move={move!r}",
            )
            snapshots.append(response_snapshot)
    return snapshots
