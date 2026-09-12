from __future__ import annotations

import pytest

import gocube_golden as g


def test_search_budget_is_fixed_by_contract_and_official_search_player():
    mismatched = g.SearchSettings(simulations=7)
    with pytest.raises(ValueError, match="fixed by the Arena contract"):
        g.GoldenArenaContract(search=mismatched)
    with pytest.raises(ValueError, match="exactly match"):
        g.SearchPlayer("MISMATCH", g.SequentialPUCT(mismatched), object())
