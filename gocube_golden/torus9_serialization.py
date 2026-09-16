from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Iterable


class _Torus9SelfPlayJSONEncoder(json.JSONEncoder):
    """Encode frozen self-play dataclasses without ``asdict`` deep copies."""

    def default(self, value: object) -> object:
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: getattr(value, field.name) for field in fields(value)}
        return super().default(value)


_ENCODER = _Torus9SelfPlayJSONEncoder(sort_keys=True)


def write_torus9_game_records_jsonl(path: str | Path, records: Iterable[object]) -> None:
    """Stream Torus9 self-play records using the legacy JSONL wire format.

    The previous hot path eagerly converted every record with ``asdict`` and
    ``_jsonable``, converted the resulting dictionaries a second time inside
    ``write_jsonl``, then joined the whole generation into one giant string.
    This writer lets the JSON encoder walk each dataclass directly and writes
    one encoded game at a time while preserving ``json.dumps(...,
    sort_keys=True)`` output exactly.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_ENCODER.encode(record))
            handle.write("\n")


__all__ = ["write_torus9_game_records_jsonl"]
