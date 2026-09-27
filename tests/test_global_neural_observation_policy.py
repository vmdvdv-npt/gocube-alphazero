from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gocube_golden.cube_observation_v2 import CHANNELS as CUBE_CHANNELS
from gocube_golden.neural_observation_policy import (
    NO_KOMI_NEURAL_OBSERVATION_POLICY_ID,
    assert_no_komi_neural_observation_channels,
    assert_no_komi_neural_observation_schema,
)
from gocube_golden.torus9_m137_5ch import M137_FIVE_CHANNEL_CHANNELS


ROOT = Path(__file__).resolve().parents[1]
LEGACY_READ_ONLY_CHANNEL_DECLARATIONS = {
    ("gocube_golden/torus9_monolith.py", "TORUS9_OBSERVATION_CHANNELS"),
}
LEGACY_READ_ONLY_OBSERVATION_BUILDERS = {
    "gocube_golden/torus9_monolith.py",
}


def _assignment_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if isinstance(target, ast.Name):
            return target.id
    return None


def _literal_strings(node: ast.AST | None) -> tuple[str, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return ()
    values: list[str] = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            values.append(element.value)
        else:
            return ()
    return tuple(values)


def test_global_policy_rejects_direct_and_derived_komi_channel_names():
    assert NO_KOMI_NEURAL_OBSERVATION_POLICY_ID == "no-komi-neural-observation-channel-v1"
    for name in ("komi", "komi_stm_normalized", "rules-komi", "current komi value"):
        with pytest.raises(ValueError, match="project-wide policy"):
            assert_no_komi_neural_observation_channels(("own_stones", name))


def test_current_cube_and_torus_trainable_channels_obey_global_policy():
    assert_no_komi_neural_observation_channels(CUBE_CHANNELS, context="Cube current observation")
    assert_no_komi_neural_observation_channels(
        M137_FIVE_CHANNEL_CHANNELS,
        context="Torus9 current trainable observation",
    )
    assert_no_komi_neural_observation_schema(
        {"channel_order": list(CUBE_CHANNELS)},
        context="Cube current schema",
    )


def test_repository_channel_declarations_cannot_reintroduce_komi_without_explicit_legacy_exception():
    violations: list[str] = []
    for path in sorted((ROOT / "gocube_golden").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            name = _assignment_name(node)
            if name is None or "CHANNEL" not in name.upper():
                continue
            strings = _literal_strings(node.value)
            if not strings:
                continue
            offenders = [value for value in strings if "komi" in value.lower()]
            if offenders and (rel, name) not in LEGACY_READ_ONLY_CHANNEL_DECLARATIONS:
                violations.append(f"{rel}:{name} -> {offenders}")
    assert not violations, "Forbidden komi neural-channel declarations:\n" + "\n".join(violations)


def test_observation_builders_cannot_read_state_komi_outside_legacy_read_only_source():
    violations: list[str] = []
    for path in sorted((ROOT / "gocube_golden").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in LEGACY_READ_ONLY_OBSERVATION_BUILDERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "observation" not in node.name.lower():
                continue
            if any(isinstance(child, ast.Attribute) and child.attr == "komi" for child in ast.walk(node)):
                violations.append(f"{rel}:{node.name}")
    assert not violations, "Observation builders read rules komi:\n" + "\n".join(violations)
