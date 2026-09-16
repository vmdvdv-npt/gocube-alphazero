from __future__ import annotations

from pathlib import Path

import pytest

from gocube_golden.run_safety import assert_isolated, safe_run_id


def test_safe_run_id_accepts_one_path_component() -> None:
    assert safe_run_id("run-01") == "run-01"
    with pytest.raises(ValueError):
        safe_run_id("../escape")


def test_assert_isolated_rejects_protected_namespace(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    with pytest.raises(ValueError):
        assert_isolated(protected, protected)
    with pytest.raises(ValueError):
        assert_isolated(protected / "nested", protected)
    assert assert_isolated(tmp_path / "sibling", protected) == (tmp_path / "sibling").resolve()
