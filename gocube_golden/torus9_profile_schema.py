"""Canonical field-name migration rules for Torus 9x9 Golden profiles.

Historical profiles remain immutable evidence.  Their old field names are accepted
only as migration input; current profiles must use canonical names exclusively.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping


class LegacyFieldConflictError(ValueError):
    """Raised when legacy and canonical fields disagree during migration."""


@dataclass(frozen=True)
class LegacyFieldRule:
    legacy: str
    canonical: str | None
    status: str
    transform: Callable[[Any], Any] = lambda value: value


def _normalize_none(value: Any) -> Any:
    if value is None or (isinstance(value, str) and value.lower() == "none"):
        return None
    return value


LEGACY_FIELD_RULES: tuple[LegacyFieldRule, ...] = (
    LegacyFieldRule("self_play.simulations", "self_play.mcts_simulations", "legacy-alias"),
    LegacyFieldRule("self_play.batch_size", "self_play.existing_batch_size", "legacy-alias"),
    LegacyFieldRule("arena.simulations", "arena.mcts_simulations", "legacy-alias"),
    LegacyFieldRule("training.scheduler", "training.lr_scheduler", "legacy-alias", _normalize_none),
    LegacyFieldRule("training.gating", "training.model_gating", "legacy-alias"),
    LegacyFieldRule("replay.maximum_positions", "replay.cap", "legacy-alias"),
    LegacyFieldRule("replay.policy", "replay_policy", "legacy-alias"),
    LegacyFieldRule("training.replay", "replay_policy", "legacy-alias"),
    LegacyFieldRule("self_play.fast_sims", None, "legacy-retired"),
    LegacyFieldRule("arena.one_game_per_process", None, "legacy-retired"),
    LegacyFieldRule("arena.technical_fail_closed", None, "legacy-retired"),
    LegacyFieldRule("rules.legacy_komi_7_5_rejected", None, "legacy-retired"),
)

LEGACY_FIELD_PATHS = frozenset(rule.legacy for rule in LEGACY_FIELD_RULES)
CANONICAL_ALIAS_MAP = {
    rule.legacy: rule.canonical
    for rule in LEGACY_FIELD_RULES
    if rule.canonical is not None
}


def _parts(path: str) -> tuple[str, ...]:
    return tuple(path.split("."))


def _lookup(root: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    node: Any = root
    for part in _parts(path):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _set(root: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = _parts(path)
    node: MutableMapping[str, Any] = root
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, MutableMapping):
            raise LegacyFieldConflictError(f"Cannot create canonical field {path}: {part} is not a mapping")
        node = child
    node[parts[-1]] = value


def _delete(root: MutableMapping[str, Any], path: str) -> None:
    parts = _parts(path)
    node: MutableMapping[str, Any] = root
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            return
        node = child
    node.pop(parts[-1], None)


def legacy_fields_present(profile: Mapping[str, Any]) -> tuple[str, ...]:
    """Return legacy field paths present in a profile, sorted for stable diagnostics."""
    return tuple(sorted(path for path in LEGACY_FIELD_PATHS if _lookup(profile, path)[0]))


def validate_no_legacy_fields(profile: Mapping[str, Any]) -> None:
    """Fail closed when a current profile contains any legacy field name."""
    present = legacy_fields_present(profile)
    if present:
        raise ValueError("Current Torus9 profile contains legacy fields: " + ", ".join(present))


def migrate_legacy_field_names(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with legacy aliases promoted to canonical names.

    Retired fields are removed.  If both names are present, their normalized values
    must agree; otherwise migration fails instead of guessing which value wins.
    This normalizes names only and does not turn an old scientific profile into the
    current Golden profile.
    """
    migrated = copy.deepcopy(dict(profile))
    for rule in LEGACY_FIELD_RULES:
        legacy_present, legacy_value = _lookup(migrated, rule.legacy)
        if not legacy_present:
            continue
        if rule.canonical is None:
            _delete(migrated, rule.legacy)
            continue

        normalized = rule.transform(legacy_value)
        canonical_present, canonical_value = _lookup(migrated, rule.canonical)
        if canonical_present and canonical_value != normalized:
            raise LegacyFieldConflictError(
                f"Legacy field {rule.legacy} conflicts with canonical field {rule.canonical}: "
                f"{legacy_value!r} != {canonical_value!r}"
            )
        if not canonical_present:
            _set(migrated, rule.canonical, normalized)
        _delete(migrated, rule.legacy)
    return migrated
