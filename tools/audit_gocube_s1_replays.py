#!/usr/bin/env python3
"""Read-only audit of persisted GoCube records against the S1 scorer.

The audit never rewrites replay/checkpoint data.  It uses the final position
and the recorded terminal result to report whether a pre-S1 record can be
reconstructed, then recomputes the corrected score/winner/ownership where the
record contains enough state.  Ordinary production games start from an empty
board, so their MAIN move counters reconstruct the pre-S1 offset.  Synthetic
cleanup and fork records are only reconstructed when they carry an explicit
start-type marker; otherwise they are reported as not reconstructible rather
than guessed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphazero.envs.gocube import (
    CLEANUP_2,
    MAIN,
    cube_topology,
    final_v3_score,
    independent_life_analysis,
    v3_state_from_board,
)
from alphazero.envs.gocube.contract_versions import (
    OWNERSHIP_TARGET_SEMANTICS,
    REPLAY_FORMAT_VERSION,
    SCORE_INITIALIZATION_CONTRACT,
)
from alphazero.envs.gocube.katago_v3 import (
    BLACK,
    EMPTY,
    WHITE,
    _legacy_board_only_life_analysis,
)


def _record_files(roots: list[Path]):
    for root in roots:
        if root.is_file() and root.suffix == ".json":
            yield root
            continue
        if root.is_dir():
            yield from sorted(root.rglob("*.json"))


def _is_record(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("final_position"), dict) and isinstance(payload.get("rules"), dict)


def _board_and_ids(position: dict[str, Any]) -> tuple[list[int], list[str]]:
    entries = position.get("board")
    if not isinstance(entries, list) or not entries:
        raise ValueError("final_position.board is missing")
    values = []
    point_ids = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("final_position.board entry is not an object")
        values.append(int(entry["value"]))
        point_ids.append(str(entry["point"]))
    if any(value not in (EMPTY, BLACK, WHITE) for value in values):
        raise ValueError("final_position.board contains an invalid occupancy")
    return values, point_ids


def _contract_version(record: dict[str, Any]) -> int | None:
    rules = record.get("rules", {})
    params = record.get("effective_parameters", {})
    value = rules.get("replay_format_version", params.get("gocube_replay_format_version"))
    return None if value is None else int(value)


def _start_type(record: dict[str, Any], position: dict[str, Any]) -> str:
    explicit = position.get("start_type") or record.get("start_type")
    if explicit:
        text = str(explicit).lower()
        if "fork" in text:
            return "fork"
        if "cleanup" in text or "rebase" in text:
            return "synthetic_cleanup"
        return "ordinary"
    for source in (record.get("effective_parameters", {}), record.get("checkpoint", {})):
        if isinstance(source, dict):
            for key, value in source.items():
                if "fork" in str(key).lower() and bool(value):
                    return "fork"
    diagnostics = record.get("cleanup_endgame_diagnostics", {})
    cleanup_moves = position.get("cleanup1_moves", [0, 0])
    cleanup2_moves = position.get("cleanup2_moves", [0, 0])
    if (
        bool(position.get("entered_cleanup1"))
        or bool(position.get("entered_cleanup2"))
        or any(int(value) for value in cleanup_moves or ())
        or any(int(value) for value in cleanup2_moves or ())
        or bool(diagnostics.get("entered_cleanup1"))
        or bool(diagnostics.get("entered_cleanup2"))
    ) and not any(int(value) for value in (position.get("main_moves") or (0, 0))):
        return "synthetic_cleanup"
    if any(int(value) for value in (position.get("main_moves") or (0, 0))) or bool(position.get("pass_alive_early_end")):
        return "ordinary"
    return "unknown"


def _labels_from_life(board: np.ndarray, life) -> np.ndarray:
    labels = np.full(board.shape[0], 2, dtype=np.int64)
    for point in life.black_area:
        if int(board[point]) == BLACK or point in life.black_territory:
            labels[point] = 0
    for point in life.white_area:
        if int(board[point]) == WHITE or point in life.white_territory:
            labels[point] = 1
    return labels


def _labels_from_ownership(ownership: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(ownership), axis=1).astype(np.int64)


def _old_labels(board: np.ndarray, topology, start_type: str) -> np.ndarray:
    life = (
        _legacy_board_only_life_analysis(board, topology)
        if start_type == "synthetic_cleanup"
        else independent_life_analysis(board, topology)
    )
    return _labels_from_life(board, life)


def _recompute(record: dict[str, Any], start_type: str) -> dict[str, Any]:
    position = record["final_position"]
    values, point_ids = _board_and_ids(position)
    topology_data = record.get("topology", {})
    kind = str(topology_data.get("kind", "cube"))
    size = int(topology_data.get("size", round(len(values) ** (1 / 3))))
    if kind != "cube":
        raise ValueError(f"audit currently supports persisted cube records, got {kind!r}")
    topology = cube_topology(size)
    if tuple(topology.point_ids) != tuple(point_ids):
        raise ValueError("record point ordering does not match the current topology")
    board = np.asarray(values, dtype=np.uint8)
    captures = tuple(int(value) for value in (position.get("captures") or (0, 0)))
    second_start_raw = position.get("second_cleanup_start_colors")
    second_start = None if second_start_raw is None else bytes(int(value) for value in second_start_raw)
    phase = CLEANUP_2 if second_start is not None else MAIN
    state = v3_state_from_board(
        topology,
        black=np.flatnonzero(board == BLACK),
        white=np.flatnonzero(board == WHITE),
        current_player=int(position.get("current_player", 0)),
        captures=captures,
        phase=phase,
        second_cleanup_start_colors=second_start,
    )

    rules = record.get("rules", {})
    if "white_bonus_score" in position:
        state = replace(state, white_bonus_score=float(position["white_bonus_score"]))
    elif start_type == "ordinary":
        main = tuple(int(value) for value in (position.get("main_moves") or (0, 0)))
        cleanup1 = tuple(int(value) for value in (position.get("cleanup1_moves") or (0, 0)))
        state = replace(
            state,
            white_bonus_score=float(main[0] - main[1] + cleanup1[0] - cleanup1[1]),
        )
    else:
        raise ValueError("record lacks explicit S1 score offset and reconstructible start metadata")

    score, ownership, _ = final_v3_score(state, topology, float(rules.get("komi", 0.5)))
    old_labels = _old_labels(board, topology, start_type)
    new_labels = _labels_from_ownership(ownership)
    old_score = record.get("final_score")
    if not isinstance(old_score, dict):
        raise ValueError("record lacks a scored final_score object")
    old_wb = float(old_score["white"]) - float(old_score["black"])
    return {
        "old_w_minus_b": old_wb,
        "new_w_minus_b": float(score.white - score.black),
        "score_delta": float(score.white - score.black - old_wb),
        "old_winner": record.get("winner"),
        "new_winner": score.winner,
        "winner_changed": record.get("winner") != score.winner,
        "ownership_changed_points": int(np.count_nonzero(old_labels != new_labels)),
        "ownership_changed": bool(np.any(old_labels != new_labels)),
        "scored": True,
        "phase": position.get("phase"),
        "captures": list(captures),
        "main_moves": list(position.get("main_moves") or (0, 0)),
        "cleanup1_moves": list(position.get("cleanup1_moves") or (0, 0)),
        "cleanup2_moves": list(position.get("cleanup2_moves") or (0, 0)),
        "second_cleanup_start_present": second_start is not None,
    }


def _empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "reconstructible": 0,
        "not_reconstructible": 0,
        "scored": 0,
        "no_result": 0,
        "score_changed": 0,
        "winner_changed": 0,
        "ownership_changed": 0,
        "score_delta_distribution": {},
    }


def _add_stats(stats: dict[str, Any], result: dict[str, Any] | None, *, no_result: bool = False) -> None:
    stats["total"] += 1
    if no_result:
        stats["no_result"] += 1
    if result is None:
        stats["not_reconstructible"] += 1
        return
    stats["reconstructible"] += 1
    if result.get("scored"):
        stats["scored"] += 1
    if result["old_w_minus_b"] != result["new_w_minus_b"]:
        stats["score_changed"] += 1
    if result["winner_changed"]:
        stats["winner_changed"] += 1
    if result["ownership_changed"]:
        stats["ownership_changed"] += 1
    key = f"{result['score_delta']:+.6f}"
    distribution = Counter(stats["score_delta_distribution"])
    distribution[key] += 1
    stats["score_delta_distribution"] = dict(sorted(distribution.items()))


def audit(roots: list[Path]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "audit": "gocube-s1-replay-audit-v1",
        "read_only": True,
        "s1_contract": {
            "replay_format_version": REPLAY_FORMAT_VERSION,
            "score_initialization_contract": SCORE_INITIALIZATION_CONTRACT,
            "ownership_target_semantics": OWNERSHIP_TARGET_SEMANTICS,
        },
        "roots": [str(root) for root in roots],
        "records": [],
        "overall": _empty_stats(),
        "by_start_type": {
            name: _empty_stats() for name in ("ordinary", "synthetic_cleanup", "fork")
        },
        "contract_versions": {},
    }
    for path in _record_files(roots):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _is_record(payload):
            continue
        version = _contract_version(payload)
        version_key = "unknown" if version is None else str(version)
        report["contract_versions"][version_key] = report["contract_versions"].get(version_key, 0) + 1
        position = payload["final_position"]
        start_type = _start_type(payload, position)
        stats = report["by_start_type"].setdefault(start_type, _empty_stats())
        result = None
        error = None
        try:
            result = _recompute(payload, start_type)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            error = str(exc)
        no_result = bool(payload.get("terminal_kind") == "no_result" or position.get("terminal_kind") == "no_result")
        _add_stats(report["overall"], result, no_result=no_result)
        _add_stats(stats, result, no_result=no_result)
        entry = {
            "path": str(path),
            "contract_version": version,
            "start_type": start_type,
            "reconstructible": result is not None,
            "phase": position.get("phase"),
            "entered_cleanup1": bool(position.get("entered_cleanup1")),
            "entered_cleanup2": bool(position.get("entered_cleanup2")),
            "main_moves": position.get("main_moves"),
            "cleanup1_moves": position.get("cleanup1_moves"),
            "cleanup2_moves": position.get("cleanup2_moves"),
            "captures": position.get("captures"),
            "second_cleanup_start_present": position.get("second_cleanup_start_colors") is not None,
        }
        if result is not None:
            entry["comparison"] = result
        else:
            entry["not_reconstructible_reason"] = error or "record is not scored or lacks required state"
        report["records"].append(entry)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("data")],
        help="record files or directories to scan (default: data)",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.roots), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
