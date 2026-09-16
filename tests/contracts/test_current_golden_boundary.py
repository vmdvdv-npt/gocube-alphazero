from __future__ import annotations

import json
from pathlib import Path

import pytest

from gocube_golden.torus9_contract import (
    current_torus9_content_fingerprint,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)
from tools.arena_engine import ArenaExecutionConfig
from tools.arena_profiles.torus9 import Torus9ArenaProfile


ROOT = Path(__file__).resolve().parents[2]
CURRENT_PROFILE = ROOT / "configs/gocube/torus9_golden_current_v3.json"
REFERENCE_PROFILE_FINGERPRINT = "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775"
CURRENT_CONTENT_FINGERPRINT = "sha256:7e97c50e1697641fb8f5b9a3566144f0a58c105e3b688940f42e7b6154fb0831"


def test_exactly_one_torus9_profile_is_current():
    current = []
    for path in sorted((ROOT / "configs/gocube").glob("torus9*.json")):
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "current":
            current.append(path.relative_to(ROOT).as_posix())
    assert current == ["configs/gocube/torus9_golden_current_v3.json"]


def test_current_torus9_scientific_contract_is_reference_locked():
    profile = load_torus9_current_profile()
    assert current_torus9_profile_fingerprint(profile) == REFERENCE_PROFILE_FINGERPRINT
    assert current_torus9_content_fingerprint(profile) == CURRENT_CONTENT_FINGERPRINT
    assert profile["rules"]["komi"] == 0.5
    assert profile["observation"]["shape"] == [6, 81]
    assert profile["observation"]["action_count"] == 82
    assert profile["target"]["ownership_auxiliary"] is True
    assert profile["target"]["score_auxiliary"] is True
    assert profile["replay"] == {
        "window": "rolling last 3 generations",
        "generations": 3,
        "cap": 20_000,
        "sampling": "deterministic / reproducible",
    }


def test_current_profile_rejects_noncanonical_komi(tmp_path: Path):
    profile = json.loads(CURRENT_PROFILE.read_text(encoding="utf-8"))
    profile["rules"]["komi"] = 15 / 2
    candidate = tmp_path / "invalid-torus9.json"
    candidate.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="komi"):
        load_torus9_current_profile(candidate, verify_fingerprint=False)


def test_current_profile_content_fingerprint_fails_closed(tmp_path: Path):
    profile = json.loads(CURRENT_PROFILE.read_text(encoding="utf-8"))
    profile["content_fingerprint"] = "sha256:" + "0" * 64
    candidate = tmp_path / "tampered-torus9.json"
    candidate.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="content fingerprint"):
        load_torus9_current_profile(candidate)


def test_current_arena_contract_matches_current_profile():
    profile = load_torus9_current_profile()
    arena = profile["arena"]
    contract = Torus9ArenaProfile().scientific_contract(
        ArenaExecutionConfig(device="cpu", strict_production=False)
    )
    assert contract["komi"] == arena["komi"] == 0.5
    assert contract["simulations"] == arena["mcts_simulations"]
    assert contract["cpuct"] == arena["cpuct"]
    assert contract["fpu"] == arena["fpu"]
    assert contract["watchdog"] == arena["watchdog"]
    assert contract["technical_fail_closed"] is True


def test_retired_runtime_entrypoints_are_physically_absent():
    for relative in (
        "alphazero/Coach.py",
        "alphazero/SelfPlayAgent.pyx",
        "alphazero/NNetWrapper.py",
        "alphazero/GenericPlayers.py",
        "alphazero/MCTS.pyx",
        "gocube_golden/training.py",
        "gocube_golden/arena.py",
        "gocube_golden/arena_process.py",
        "tools/continue_torus9_golden_m1_m100.py",
        "tools/torus9_alpha_score_ab.py",
    ):
        assert not (ROOT / relative).exists(), relative


def test_current_production_tree_has_no_legacy_execution_imports():
    roots = (
        ROOT / "gocube_golden",
        ROOT / "alphazero/envs/gocube/integration",
        ROOT / "selfplay_engine.py",
        ROOT / "training_engine.py",
        ROOT / "tools/arena.py",
        ROOT / "tools/arena_engine.py",
        ROOT / "tools/arena_profiles",
    )
    forbidden = (
        "alphazero.Coach",
        "alphazero.SelfPlayAgent",
        "alphazero.NNetWrapper",
        "alphazero.GenericPlayers",
        "from .MCTS",
        "from alphazero.MCTS",
        "iteration-0000.pkl",
        "legacy_mode",
        "use_old_mcts",
        "use_nnet_wrapper",
        "old_checkpoint_format",
    )
    for root in roots:
        files = root.rglob("*.py") if root.is_dir() else (root,)
        for path in files:
            source = path.read_text(encoding="utf-8")
            assert not any(token in source for token in forbidden), path


def test_current_runtime_contains_no_forbidden_komi_literal():
    paths = (
        "configs/gocube/cube4_golden_training_v1.json",
        "configs/gocube/torus9_golden_current_v3.json",
        "gocube_golden",
        "alphazero/envs/gocube/integration",
        "tools/arena.py",
        "tools/arena_engine.py",
    )
    for relative in paths:
        path = ROOT / relative
        files = path.rglob("*") if path.is_dir() else (path,)
        for candidate in files:
            if candidate.is_file() and candidate.suffix in {".py", ".json"}:
                assert str(15 / 2) not in candidate.read_text(encoding="utf-8"), candidate
