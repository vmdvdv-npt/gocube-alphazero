from __future__ import annotations

from types import SimpleNamespace

import tools.torus9_stage7 as stage7
from tools.torus9_stage7 import (
    ARENA_PAIRS,
    BOOTSTRAP_REPLICATES,
    REFERENCE_RUN_ID,
    _bootstrap,
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


def test_stage7_reference_is_resolved_by_catalog_not_a_stage7_copy(tmp_path, monkeypatch):
    checkpoint = tmp_path / "runs" / "torus9" / "archive" / REFERENCE_RUN_ID / "checkpoints" / "M17.pt"

    class FakeCatalog:
        def get(self, key: str):
            assert key == f"{REFERENCE_RUN_ID}@17"
            return SimpleNamespace(path=str(checkpoint), run_name=REFERENCE_RUN_ID)

    monkeypatch.setattr(stage7, "_reference_catalog", lambda: FakeCatalog())
    path = stage7._resolve_reference("M17")

    assert path == checkpoint.resolve()
    assert path.name == "M17.pt"
    assert path.parent.parent.name == REFERENCE_RUN_ID


def test_stage7_run_id_rejects_path_traversal():
    assert _safe_run_id("run-01") == "run-01"
    for value in ("", ".", "..", "nested/run"):
        try:
            _safe_run_id(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe run id accepted: {value!r}")
