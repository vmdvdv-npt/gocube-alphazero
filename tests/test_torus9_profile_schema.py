from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gocube_golden.torus9_contract import load_torus9_current_profile, load_torus9_profile
from gocube_golden.torus9_profile_schema import (
    CANONICAL_ALIAS_MAP,
    LEGACY_FIELD_PATHS,
    LegacyFieldConflictError,
    legacy_fields_present,
    migrate_legacy_field_names,
    validate_no_legacy_fields,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs/gocube/torus9_field_schema_v1.json"


def test_current_torus9_profile_contains_no_legacy_field_names():
    profile = load_torus9_current_profile()
    assert legacy_fields_present(profile) == ()
    validate_no_legacy_fields(profile)


def test_historical_profile_keeps_legacy_names_as_immutable_evidence():
    profile = load_torus9_profile()
    present = set(legacy_fields_present(profile))
    assert {
        "self_play.simulations",
        "self_play.batch_size",
        "arena.simulations",
        "training.scheduler",
        "training.gating",
        "replay.maximum_positions",
        "replay.policy",
        "training.replay",
    } <= present


def test_legacy_migration_normalizes_names_without_mutating_source():
    legacy = load_torus9_profile()
    original = copy.deepcopy(legacy)
    migrated = migrate_legacy_field_names(legacy)

    assert legacy == original
    assert legacy_fields_present(migrated) == ()
    assert migrated["self_play"]["mcts_simulations"] == 64
    assert migrated["self_play"]["existing_batch_size"] == 64
    assert migrated["arena"]["mcts_simulations"] == 64
    assert migrated["training"]["lr_scheduler"] is None
    assert migrated["training"]["model_gating"] is False
    assert migrated["replay"]["cap"] == 20_000
    assert migrated["replay_policy"] == "rolling-recent-generations"


def test_legacy_migration_fails_closed_on_old_new_conflict():
    profile = load_torus9_current_profile()
    conflicted = copy.deepcopy(profile)
    conflicted["self_play"]["simulations"] = 32

    with pytest.raises(LegacyFieldConflictError, match="self_play.simulations"):
        migrate_legacy_field_names(conflicted)


def test_field_registry_marks_every_old_name_legacy_and_matches_code():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = registry["legacy_fields"]
    assert all(row["status"].startswith("legacy-") for row in rows)
    assert {row["path"] for row in rows} == LEGACY_FIELD_PATHS
    assert {
        row["path"]: row["canonical"]
        for row in rows
        if row["canonical"] is not None
    } == CANONICAL_ALIAS_MAP


def test_current_profile_rejector_reports_legacy_names():
    profile = load_torus9_current_profile()
    bad = copy.deepcopy(profile)
    bad["training"]["scheduler"] = "none"
    bad["replay"]["maximum_positions"] = 20_000

    with pytest.raises(ValueError) as excinfo:
        validate_no_legacy_fields(bad)
    message = str(excinfo.value)
    assert "training.scheduler" in message
    assert "replay.maximum_positions" in message
