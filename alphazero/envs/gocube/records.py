"""Persistent, machine-readable GoCube self-play records.

The record writer deliberately lives outside the learning code.  It receives a
completed game and records the already adjudicated state; it does not alter
move legality, scoring, targets, or optimizer configuration.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .core import BLACK, EMPTY, WHITE

GAME_RECORD_SCHEMA_VERSION = 1
ITERATION_MANIFEST_SCHEMA_VERSION = 1
ITERATION_MANIFEST_FILENAME = "iteration-manifest.json"
_COUNTER_FILENAME = "game-id-counter.json"
_LOCK_FILENAME = "game-id-counter.lock"
_ID_RE = re.compile(r"^(?P<prefix>[A-Z]\d+)-(?P<number>\d+)$")
_PREFIX_RE = re.compile(r"^[A-Z]\d+$")


def game_id_prefix(game_cls) -> str:
    """Return the stable short ID prefix for a configured game class."""

    kind = game_cls.topology_kind()
    letter = {"cube": "C", "torus": "T"}[kind]
    return f"{letter}{int(game_cls.board_size())}"


def reserve_game_id(registry_dir: str | os.PathLike[str], prefix: str) -> str:
    """Reserve the next ID using an inter-process lock and atomic counter.

    ``registry_dir`` is the shared registry root, independent of any run's
    output directory.  Each prefix gets its own counter namespace below that
    root, so Cube and torus IDs cannot reset or overwrite one another.
    """

    if not _PREFIX_RE.fullmatch(str(prefix)):
        raise ValueError(f"Invalid game ID prefix: {prefix!r}")
    registry = Path(registry_dir) / str(prefix)
    registry.mkdir(parents=True, exist_ok=True)
    lock_path = registry / _LOCK_FILENAME
    counter_path = registry / _COUNTER_FILENAME
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            next_number = 1
            if counter_path.exists():
                with counter_path.open("r", encoding="utf-8") as handle:
                    counter = json.load(handle)
                if counter.get("prefix") == prefix:
                    next_number = max(1, int(counter.get("next_number", 1)))
            game_id = f"{prefix}-{next_number:06d}"
            payload = {
                "schema_version": 1,
                "prefix": prefix,
                "next_number": next_number + 1,
            }
            fd, temporary = tempfile.mkstemp(prefix=f"{_COUNTER_FILENAME}.", dir=registry)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, counter_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return game_id
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def effective_parameter_snapshot(args: Any) -> dict[str, Any]:
    """Convert the complete effective args object into JSON-safe data."""

    values = dict(args) if isinstance(args, Mapping) else vars(args)
    return {str(key): _json_safe(value) for key, value in sorted(values.items())}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _timestamp(timestamp: float) -> str:
    return _datetime.datetime.fromtimestamp(
        float(timestamp), tz=_datetime.timezone.utc
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _occupancy(value: int) -> str:
    return {EMPTY: "empty", BLACK: "black", WHITE: "white"}.get(int(value), f"unknown:{value}")


def _board_values(board: np.ndarray | None, point_ids: Iterable[str]) -> list[dict[str, Any]] | None:
    if board is None:
        return None
    return [
        {"point": point_id, "value": int(value), "occupancy": _occupancy(int(value))}
        for point_id, value in zip(point_ids, np.asarray(board).reshape(-1))
    ]


def _score_to_dict(score: Any) -> dict[str, Any] | None:
    if score is None:
        return None
    return _json_safe(dataclasses.asdict(score)) if dataclasses.is_dataclass(score) else _json_safe(score)


def _terminal_to_dict(terminal: Any) -> dict[str, Any] | None:
    if terminal is None:
        return None
    score = getattr(terminal, "score", None)
    return {
        "kind": getattr(terminal, "terminal_kind", None),
        "winner": getattr(terminal, "winner", None),
        "no_result": bool(getattr(terminal, "no_result", False)),
        "no_result_reason": getattr(terminal, "reason", None),
        "score": _score_to_dict(score),
        "adjudicator": getattr(terminal, "adjudicator_id", None),
    }


def _final_position(game: Any) -> dict[str, Any]:
    topology = game.logical_topology()
    state = getattr(game, "semantic_state", None)
    board = getattr(state, "board", getattr(game, "_board", None))
    position: dict[str, Any] = {
        "point_ids": list(topology.point_ids),
        "board": _board_values(board, topology.point_ids),
        "turns": int(getattr(game, "turns", getattr(state, "turns", 0))),
        "current_player": getattr(state, "current_player", getattr(game, "player", None)),
    }
    if state is not None:
        for field in (
            "consecutive_passes", "captures", "phase", "cleanup_stage",
            "ko_recap_blocked", "cleanup2_moves", "main_moves", "cleanup1_moves",
            "terminal_kind", "no_result_reason", "pass_alive_early_end",
            "entered_cleanup1", "entered_cleanup2", "cleanup_captures",
            "ko_unblock_actions",
        ):
            if hasattr(state, field):
                value = getattr(state, field)
                position[field] = _json_safe(value)
        previous_board = getattr(state, "previous_board", None)
        position["previous_board"] = _board_values(previous_board, topology.point_ids)
        second_start = getattr(state, "second_cleanup_start_colors", None)
        position["second_cleanup_start_colors"] = (
            list(second_start) if second_start is not None else None
        )
    return position


def _cleanup_diagnostics(game: Any) -> dict[str, Any]:
    state = getattr(game, "semantic_state", None)
    diagnostics: dict[str, Any] = {}
    if hasattr(game, "diagnostic_counters"):
        diagnostics.update(_json_safe(game.diagnostic_counters()))
    if state is not None:
        for field in (
            "main_moves", "cleanup1_moves", "cleanup2_moves", "cleanup_captures",
            "ko_unblock_actions", "entered_cleanup1", "entered_cleanup2",
            "pass_alive_early_end",
        ):
            if hasattr(state, field):
                diagnostics[field] = _json_safe(getattr(state, field))
    return diagnostics


def build_game_record(
    *,
    game: Any,
    game_id: str,
    run_name: str,
    iteration: int,
    game_number: int,
    checkpoint: Mapping[str, Any],
    parameters: Mapping[str, Any],
    moves: list[Mapping[str, Any]],
    start_time: float,
    end_time: float,
    winstate: Any,
    record_path: str,
) -> dict[str, Any]:
    topology = game.logical_topology()
    terminal = getattr(game, "terminal_adjudication", None)
    terminal_kind = getattr(game, "terminal_kind", None)
    terminal_data = _terminal_to_dict(terminal)
    score = getattr(terminal, "score", None) if terminal is not None else None
    winner = getattr(terminal, "winner", None) if terminal is not None else None
    if winner is None:
        winner = "draw" if bool(np.asarray(winstate)[-1]) else None
    result = "draw" if winner == "draw" else f"{winner}_win" if winner else None
    no_result_reason = (
        getattr(terminal, "reason", None)
        or getattr(getattr(game, "semantic_state", None), "no_result_reason", None)
    )
    field_notes: dict[str, str] = {}
    if score is None:
        field_notes["final_score"] = "No score exists for this terminal result."
        field_notes["final_score_margin"] = "No score exists for this terminal result."
    if no_result_reason is None:
        field_notes["no_result_reason"] = "Not applicable because the terminal result is not no-result."
    record = {
        "schema_version": GAME_RECORD_SCHEMA_VERSION,
        "game_id": game_id,
        "run_name": run_name,
        "iteration": int(iteration),
        "game_number_inside_iteration": int(game_number),
        "checkpoint": _json_safe(dict(checkpoint)),
        "start_time": _timestamp(start_time),
        "end_time": _timestamp(end_time),
        "duration_seconds": round(float(end_time) - float(start_time), 6),
        "topology": {
            "kind": topology.kind,
            "size": int(topology.size),
            "point_count": int(topology.point_count),
        },
        "size": int(topology.size),
        "rules": {
            "rule_set": getattr(game, "RULESET", None),
            "komi": float(getattr(game, "KOMI", 0.0)),
            "terminal_adjudicator": getattr(game, "TERMINAL_ADJUDICATOR_ID", None),
            "observation_schema": getattr(game, "OBSERVATION_SCHEMA", None),
            "rules_fingerprint": game.rules_fingerprint() if hasattr(game, "rules_fingerprint") else None,
            "katago_rules_version": getattr(game, "KATAGO_RULES_VERSION", None),
            "katago_reference_commit": getattr(game, "KATAGO_REFERENCE_COMMIT", None),
        },
        "effective_parameters": _json_safe(dict(parameters)),
        "moves": [_json_safe(dict(move)) for move in moves],
        "number_of_moves": len(moves),
        "final_position": _final_position(game),
        "winner": winner,
        "result": result,
        "final_score": _score_to_dict(score),
        "final_score_margin": getattr(score, "margin", None),
        "terminal_kind": terminal_kind,
        "no_result_reason": no_result_reason,
        "terminal": terminal_data,
        "cleanup_endgame_diagnostics": _cleanup_diagnostics(game),
        "record_path": record_path,
        "field_notes": field_notes,
    }
    return record


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_game_record(record_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> dict[str, str]:
    """Atomically write one JSON record and return its manifest index entry."""

    record_dir_path = Path(record_dir)
    record_dir_path.mkdir(parents=True, exist_ok=True)
    game_id = str(record["game_id"])
    path = record_dir_path / f"{game_id}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing game record: {path}")
    data = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f"{game_id}.", suffix=".tmp", dir=record_dir_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "game_id": game_id,
        "record_path": os.path.relpath(path.resolve(), Path.cwd().resolve()),
        "sha256": _sha256_bytes(data),
    }


def iteration_manifest_path(record_dir: str | os.PathLike[str]) -> str:
    return str(Path(record_dir) / ITERATION_MANIFEST_FILENAME)


def write_iteration_manifest(
    record_dir: str | os.PathLike[str],
    *,
    run_name: str,
    iteration: int,
    checkpoint: Mapping[str, Any],
    parameters: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    aggregate_metrics: Mapping[str, Any],
) -> str:
    """Write the immutable iteration index after all accepted games are recorded."""

    record_dir_path = Path(record_dir)
    record_dir_path.mkdir(parents=True, exist_ok=True)
    normalized_records = []
    for entry in records:
        record_path = Path(str(entry["record_path"]))
        if not record_path.is_absolute():
            record_path = Path.cwd() / record_path
        if not record_path.exists():
            raise FileNotFoundError(f"Iteration manifest record does not exist: {entry['record_path']}")
        normalized_entry = {
            "game_id": str(entry["game_id"]),
            "record_path": str(entry["record_path"]),
            "sha256": str(entry["sha256"]),
        }
        if "game_number_inside_iteration" in entry:
            normalized_entry["game_number_inside_iteration"] = int(
                entry["game_number_inside_iteration"]
            )
        normalized_records.append(normalized_entry)
    manifest = {
        "schema_version": ITERATION_MANIFEST_SCHEMA_VERSION,
        "run_name": run_name,
        "iteration": int(iteration),
        "checkpoint": _json_safe(dict(checkpoint)),
        "effective_iteration_parameters": _json_safe(dict(parameters)),
        "records": normalized_records,
        "aggregate_metrics": _json_safe(dict(aggregate_metrics)),
    }
    path = record_dir_path / ITERATION_MANIFEST_FILENAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return str(path)
