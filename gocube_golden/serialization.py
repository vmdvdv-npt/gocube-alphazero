from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Mapping

from .arena import (
    ActionEvidence,
    GameRecord,
    MappedResult,
    TerminationReason,
    recompute_summary,
    validate_game_record,
)
from .arena_contract import DEFAULT_ARENA_CONTRACT
from .provenance import PROVENANCE_SCHEMA_VERSION
from .result import Winner
from .state import PASS
from .topology import TORUS_5X5


class GoldenSerializationError(ValueError):
    pass


def _error(path: str, message: str) -> GoldenSerializationError:
    return GoldenSerializationError(f"{path}: {message}")


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(path, "expected JSON object")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise _error(path, "object keys must be strings")
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise _error(path, "missing required fields: " + ", ".join(missing))
    if extra:
        raise _error(path, "unexpected fields: " + ", ".join(extra))


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise _error(path, "expected string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise _error(path, "expected boolean")
    return value


def _int(value: object, path: str) -> int:
    if type(value) is not int:
        raise _error(path, "expected integer")
    return value


def _optional_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _int(value, path)


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "expected finite JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise _error(path, "expected finite JSON number") from exc
    if not math.isfinite(number):
        raise _error(path, "expected finite JSON number")
    return number


def _optional_number(value: object, path: str) -> float | None:
    if value is None:
        return None
    return _number(value, path)


def _sequence(value: object, path: str) -> list[object] | tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise _error(path, "expected JSON array")
    return value


def _board(value: object, path: str) -> tuple[int, ...]:
    seq = _sequence(value, path)
    if len(seq) != TORUS_5X5.point_count:
        raise _error(path, f"expected {TORUS_5X5.point_count} board points")
    board: list[int] = []
    for index, item in enumerate(seq):
        stone = _int(item, f"{path}[{index}]")
        if stone not in (0, 1, 2):
            raise _error(f"{path}[{index}]", "stone must be 0, 1, or 2")
        board.append(stone)
    return tuple(board)


def _history(value: object, path: str) -> tuple[tuple[int, ...], ...]:
    seq = _sequence(value, path)
    if not seq:
        raise _error(path, "superko history must not be empty")
    return tuple(_board(item, f"{path}[{index}]") for index, item in enumerate(seq))


def _start_action(value: object, path: str) -> int | str:
    if value == PASS:
        return PASS
    action = _int(value, path)
    if not 0 <= action < TORUS_5X5.point_count:
        raise _error(path, "start_trace action is outside Golden action space")
    return action


def _evidence_action(value: object, path: str) -> int | str:
    if isinstance(value, str):
        if not value:
            raise _error(path, "string action must be non-empty")
        return value
    return _int(value, path)


def _start_state_key(value: object, path: str) -> tuple[object, ...]:
    seq = _sequence(value, path)
    if len(seq) != 10:
        raise _error(path, "Golden state key must contain exactly 10 fields")
    side = _int(seq[1], f"{path}[1]")
    if side not in (1, 2):
        raise _error(f"{path}[1]", "side_to_move must be BLACK(1) or WHITE(2)")
    passes = _int(seq[3], f"{path}[3]")
    if passes not in (0, 1, 2):
        raise _error(f"{path}[3]", "consecutive_passes must be 0, 1, or 2")
    return (
        _board(seq[0], f"{path}[0]"),
        side,
        _history(seq[2], f"{path}[2]"),
        passes,
        _string(seq[4], f"{path}[4]"),
        _string(seq[5], f"{path}[5]"),
        _string(seq[6], f"{path}[6]"),
        _string(seq[7], f"{path}[7]"),
        _number(seq[8], f"{path}[8]"),
        _string(seq[9], f"{path}[9]"),
    )


def _enum(enum_type, value: object, path: str):
    raw = _string(value, path)
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise _error(path, f"unknown {enum_type.__name__} value {raw!r}") from exc


def _optional_enum(enum_type, value: object, path: str):
    if value is None:
        return None
    return _enum(enum_type, value, path)


def _search_settings(value: object, path: str) -> tuple[tuple[str, object], ...]:
    seq = _sequence(value, path)
    expected = dict(DEFAULT_ARENA_CONTRACT.search.evidence())
    if len(seq) != len(expected):
        raise _error(path, f"expected exactly {len(expected)} search settings")
    result: list[tuple[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(seq):
        pair = _sequence(item, f"{path}[{index}]")
        if len(pair) != 2:
            raise _error(f"{path}[{index}]", "search setting must be [name, value]")
        key = _string(pair[0], f"{path}[{index}][0]")
        if key in seen:
            raise _error(path, f"duplicate search setting {key!r}")
        if key not in expected:
            raise _error(path, f"unknown search setting {key!r}")
        seen.add(key)
        expected_value = expected[key]
        raw_value = pair[1]
        if type(expected_value) is bool:
            parsed_value: object = _bool(raw_value, f"{path}[{index}][1]")
        elif type(expected_value) is int:
            parsed_value = _int(raw_value, f"{path}[{index}][1]")
        elif isinstance(expected_value, float):
            parsed_value = _number(raw_value, f"{path}[{index}][1]")
        else:
            raise RuntimeError(f"Unsupported Golden search setting type for {key!r}")
        result.append((key, parsed_value))
    if seen != set(expected):
        raise _error(path, "search settings are incomplete")
    return tuple(result)


def _action_evidence(value: object, path: str) -> ActionEvidence:
    mapping = _require_mapping(value, path)
    expected = {field.name for field in fields(ActionEvidence)}
    _require_exact_keys(mapping, expected, path)
    side_to_move = _string(mapping["side_to_move"], f"{path}.side_to_move")
    if side_to_move not in {"BLACK", "WHITE"}:
        raise _error(f"{path}.side_to_move", "expected BLACK or WHITE")
    player_slot = _string(mapping["player_slot"], f"{path}.player_slot")
    if player_slot not in {"A", "B"}:
        raise _error(f"{path}.player_slot", "expected A or B")
    return ActionEvidence(
        ply=_int(mapping["ply"], f"{path}.ply"),
        side_to_move=side_to_move,
        player_slot=player_slot,
        player_id=_string(mapping["player_id"], f"{path}.player_id"),
        action=_evidence_action(mapping["action"], f"{path}.action"),
        legal=_bool(mapping["legal"], f"{path}.legal"),
        error=_optional_string(mapping["error"], f"{path}.error"),
    )


def game_record_from_dict(payload: Mapping[str, object]) -> GameRecord:
    mapping = _require_mapping(payload, "GameRecord")
    expected = {field.name for field in fields(GameRecord)}
    _require_exact_keys(mapping, expected, "GameRecord")

    schema_version = _int(mapping["schema_version"], "GameRecord.schema_version")
    if schema_version != PROVENANCE_SCHEMA_VERSION:
        raise _error(
            "GameRecord.schema_version",
            f"unsupported schema version {schema_version}; expected {PROVENANCE_SCHEMA_VERSION}",
        )

    start_trace_raw = _sequence(mapping["start_trace"], "GameRecord.start_trace")
    action_trace_raw = _sequence(mapping["action_trace"], "GameRecord.action_trace")

    return GameRecord(
        schema_version=schema_version,
        run_id=_string(mapping["run_id"], "GameRecord.run_id"),
        run_identity_fingerprint=_string(mapping["run_identity_fingerprint"], "GameRecord.run_identity_fingerprint"),
        experiment_profile_id=_string(mapping["experiment_profile_id"], "GameRecord.experiment_profile_id"),
        experiment_fingerprint=_string(mapping["experiment_fingerprint"], "GameRecord.experiment_fingerprint"),
        git_commit_sha=_string(mapping["git_commit_sha"], "GameRecord.git_commit_sha"),
        git_tree_sha=_string(mapping["git_tree_sha"], "GameRecord.git_tree_sha"),
        git_worktree_clean=_bool(mapping["git_worktree_clean"], "GameRecord.git_worktree_clean"),
        seed_derivation_id=_string(mapping["seed_derivation_id"], "GameRecord.seed_derivation_id"),
        master_seed=_int(mapping["master_seed"], "GameRecord.master_seed"),
        game_id=_string(mapping["game_id"], "GameRecord.game_id"),
        pair_id=_string(mapping["pair_id"], "GameRecord.pair_id"),
        rules_fingerprint=_string(mapping["rules_fingerprint"], "GameRecord.rules_fingerprint"),
        topology_fingerprint=_string(mapping["topology_fingerprint"], "GameRecord.topology_fingerprint"),
        komi=_number(mapping["komi"], "GameRecord.komi"),
        start_state_key=_start_state_key(mapping["start_state_key"], "GameRecord.start_state_key"),
        start_history=_history(mapping["start_history"], "GameRecord.start_history"),
        start_trace=tuple(
            _start_action(item, f"GameRecord.start_trace[{index}]")
            for index, item in enumerate(start_trace_raw)
        ),
        player_A_id=_string(mapping["player_A_id"], "GameRecord.player_A_id"),
        player_B_id=_string(mapping["player_B_id"], "GameRecord.player_B_id"),
        player_A_identity_fingerprint=_string(mapping["player_A_identity_fingerprint"], "GameRecord.player_A_identity_fingerprint"),
        player_B_identity_fingerprint=_string(mapping["player_B_identity_fingerprint"], "GameRecord.player_B_identity_fingerprint"),
        black_player=_string(mapping["black_player"], "GameRecord.black_player"),
        white_player=_string(mapping["white_player"], "GameRecord.white_player"),
        seed_game=_int(mapping["seed_game"], "GameRecord.seed_game"),
        seed_A=_int(mapping["seed_A"], "GameRecord.seed_A"),
        seed_B=_int(mapping["seed_B"], "GameRecord.seed_B"),
        search_contract_id=_string(mapping["search_contract_id"], "GameRecord.search_contract_id"),
        search_contract_fingerprint=_string(mapping["search_contract_fingerprint"], "GameRecord.search_contract_fingerprint"),
        search_implementation_id=_string(mapping["search_implementation_id"], "GameRecord.search_implementation_id"),
        search_implementation_fingerprint=_string(mapping["search_implementation_fingerprint"], "GameRecord.search_implementation_fingerprint"),
        search_settings=_search_settings(mapping["search_settings"], "GameRecord.search_settings"),
        action_trace=tuple(
            _action_evidence(item, f"GameRecord.action_trace[{index}]")
            for index, item in enumerate(action_trace_raw)
        ),
        final_board=_board(mapping["final_board"], "GameRecord.final_board"),
        absolute_rule_result=_optional_enum(Winner, mapping["absolute_rule_result"], "GameRecord.absolute_rule_result"),
        mapped_result=_optional_enum(MappedResult, mapping["mapped_result"], "GameRecord.mapped_result"),
        black_area=_optional_int(mapping["black_area"], "GameRecord.black_area"),
        white_area=_optional_int(mapping["white_area"], "GameRecord.white_area"),
        margin_black=_optional_number(mapping["margin_black"], "GameRecord.margin_black"),
        termination_reason=_enum(TerminationReason, mapping["termination_reason"], "GameRecord.termination_reason"),
        error_details=_optional_string(mapping["error_details"], "GameRecord.error_details"),
    )


def _reject_json_constant(value: str):
    raise GoldenSerializationError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_object_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GoldenSerializationError(f"duplicate JSON object key {key!r} is forbidden")
        result[key] = value
    return result


def game_record_from_json(payload: str) -> GameRecord:
    if not isinstance(payload, str):
        raise GoldenSerializationError("GameRecord JSON payload must be a string")
    try:
        decoded = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except GoldenSerializationError:
        raise
    except json.JSONDecodeError as exc:
        raise GoldenSerializationError(
            f"invalid GameRecord JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    return game_record_from_dict(_require_mapping(decoded, "GameRecord"))


def _jsonable(value: object):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def game_record_to_dict(record: GameRecord) -> dict[str, object]:
    if not isinstance(record, GameRecord):
        raise TypeError("game_record_to_dict expects GameRecord")
    validate_game_record(record)
    return _jsonable(record)


def game_record_to_json(record: GameRecord) -> str:
    return json.dumps(
        game_record_to_dict(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def load_and_validate_game_record(payload: str) -> GameRecord:
    record = game_record_from_json(payload)
    validate_game_record(record)
    return record


def read_records_jsonl(path: str | Path, *, validate: bool = True) -> tuple[GameRecord, ...]:
    source = Path(path)
    records: list[GameRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise GoldenSerializationError(f"{source}: line {line_number}: blank lines are forbidden")
        try:
            record = game_record_from_json(line)
            if validate:
                validate_game_record(record)
        except (GoldenSerializationError, ValueError) as exc:
            raise GoldenSerializationError(f"{source}: line {line_number}: {exc}") from exc
        records.append(record)
    result = tuple(records)
    if validate and result:
        try:
            recompute_summary(result)
        except ValueError as exc:
            raise GoldenSerializationError(f"{source}: record-set validation failed: {exc}") from exc
    return result
