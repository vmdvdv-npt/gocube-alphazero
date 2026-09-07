from __future__ import annotations

import json
from pathlib import Path

import pytest

from katago_reference_runner import run_fixture


pytestmark = pytest.mark.katago_reference
FIXTURES = Path(__file__).parent / "reference" / "katago" / "rules_fixtures.json"


def _fixtures():
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def test_static_fixture_inventory_is_pinned_and_complete():
    fixtures = _fixtures()
    required = {
        "ordinary-legal-move", "occupied-point-illegal", "single-stone-capture",
        "multi-stone-capture", "suicide-illegal", "capture-not-suicide", "simple-ko",
        "legal-after-ko-threat", "snapback", "pass", "two-ending-passes",
        "main-to-cleanup-1", "cleanup-1-to-cleanup-2", "cleanup-capture", "cleanup-ko",
        "ko-recap-block", "pass-for-ko-form-1", "pass-for-ko-form-2",
        "repeated-ko-prevention", "pass-alive-terminal", "territory-scoring", "seki-tax",
        "prisoner-capture-contribution", "cleanup-2-compensation", "no-result-cycle-repetition",
    }
    assert len(fixtures) == 25
    assert {fixture["id"] for fixture in fixtures} >= required
    for fixture in fixtures:
        assert fixture["katago_commit"] == "f6bc4b19a1686caa2d088b56251e8c11c8be6d51"
        assert set(("id", "katago_commit", "source_file", "source_test", "board_size", "setup", "moves")) <= set(fixture)
        assert fixture.get("postconditions"), f"{fixture['id']} has no semantic postcondition"


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda fixture: fixture["id"])
def test_static_fixture_matches_pinned_katago(fixture):
    run_fixture(fixture)
