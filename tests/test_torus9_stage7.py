from __future__ import annotations

from tools.torus9_stage7 import (
    ARENA_PAIRS,
    BOOTSTRAP_REPLICATES,
    _bootstrap,
    _resolve_reference,
    _safe_run_id,
    _startset,
)


def test_stage7_bootstrap_is_clustered_and_uses_fixed_decision_rules():
    result = _bootstrap((1.0,) * ARENA_PAIRS, seed=123)

    assert result["method"] == "cluster-bootstrap-paired-start-v1"
    assert result["paired_starts"] == 96
    assert result["replicates"] == BOOTSTRAP_REPLICATES
    assert result["observed_new_score"] == 1.0
    assert result["verdict"] == "PASS"


def test_stage7_frozen_startset_is_96_pairs_and_reproducible():
    first = _startset()
    second = _startset()

    assert first == second
    assert first["paired_starts"] == 96
    assert len(first["rows"]) == 96
    assert len({row["start_id"] for row in first["rows"]}) == 96
    assert len({row["corpus_fingerprint"] for row in first["rows"]}) == 1


def test_stage7_reference_is_resolved_by_catalog_not_a_stage7_copy():
    path = _resolve_reference("M17")

    assert path.name == "M17.pt"
    assert path.parent.parent.name == "torus9-golden-v3-20260914-run03"
    assert "stage7" not in str(path)


def test_stage7_run_id_rejects_path_traversal():
    assert _safe_run_id("run-01") == "run-01"
    for value in ("", ".", "..", "nested/run"):
        try:
            _safe_run_id(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe run id accepted: {value!r}")
