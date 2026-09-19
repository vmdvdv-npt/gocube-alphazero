from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_torus9_driver_uses_low_level_graph_modules() -> None:
    source_path = ROOT / "tools" / "torus9_run_driver.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "gocube_golden.orchestrator_v2.artifact_resolver" not in imported_modules
    assert "gocube_golden.orchestrator_v2.contracts" not in imported_modules
    assert "gocube_golden.artifact_resolver" in imported_modules
    assert "gocube_golden.artifact_graph" in imported_modules
