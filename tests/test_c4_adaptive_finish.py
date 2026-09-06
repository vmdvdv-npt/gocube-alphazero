from __future__ import annotations

import importlib.util
import py_compile
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE = ROOT / "tools" / "c4_adaptive_finish.py"
ADAPTIVE_RESUME = ROOT / "tools" / "resume_c4_adaptive.sh"


def load_adaptive_module():
    spec = importlib.util.spec_from_file_location("c4_adaptive_finish", ADAPTIVE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_adaptive_tools_have_valid_syntax():
    py_compile.compile(str(ADAPTIVE), doraise=True)
    subprocess.run(["bash", "-n", str(ADAPTIVE_RESUME)], check=True)


def test_adaptive_runner_uses_screening_not_fixed_24_game_decision():
    source = ADAPTIVE.read_text(encoding="utf-8")
    required = (
        "leader-first-sequential",
        "adaptive_match",
        "EARLY_Z",
        "FINAL_Z",
        "99% sequential interval",
        "95% interval",
        "adaptive_final_max_games",
        "adaptive_shortlist_selected",
        "adaptive_fallback",
        "fresh-depth confirmation",
        "stable_seed",
        "adaptive_reference_branch",
    )
    for token in required:
        assert token in source


def test_interim_stopping_is_conservative_and_final_cap_is_larger():
    module = load_adaptive_module()
    # 18/24 was the observed D common-parent screen. It is strong at 95%, but
    # is deliberately not enough for the 99% repeated-look early-stop rule.
    low99, high99 = module.wilson(18 / 24, 24, module.EARLY_Z)
    assert low99 < 0.5 < high99
    # A very large direct lead can stop after a single 24-game block.
    low99, _ = module.wilson(20 / 24, 24, module.EARLY_Z)
    assert low99 > 0.5
    assert module.DEFAULT_FINAL_MAX > module.DEFAULT_SCREEN_MAX


def test_shortlist_keeps_baseline_axis_winners_and_close_axis_pair():
    module = load_adaptive_module()
    exp = module.AdaptiveExperiment.__new__(module.AdaptiveExperiment)
    scores = {
        "A": 0.458,
        "B": 0.542,
        "C": 0.500,
        "D": 0.750,
        "E": 0.458,
        "F": 0.250,
        "G": 0.600,
    }
    shortlist = exp._shortlist(scores, "D")
    assert "A" in shortlist
    assert "D" in shortlist
    assert "B" in shortlist and "C" in shortlist  # pFast screen is effectively tied.
    assert "G" in shortlist
    assert "E" not in shortlist
    assert "F" not in shortlist


def test_block_seeds_are_deterministic_but_independent():
    module = load_adaptive_module()
    seed1 = module.stable_seed(20260906, "adaptive_screen_D_i6_vs_A_i6_b01")
    seed2 = module.stable_seed(20260906, "adaptive_screen_D_i6_vs_A_i6_b02")
    assert seed1 == module.stable_seed(20260906, "adaptive_screen_D_i6_vs_A_i6_b01")
    assert seed1 != seed2


def test_adaptive_resume_is_transactional_and_uses_unique_unit():
    source = ADAPTIVE_RESUME.read_text(encoding="utf-8")
    assert 'STATE="training_reports/$EXP_ID/state-private.json"' in source
    assert '"$TMP_DIR/c4_adaptive_finish.py"' in source
    assert 'c4_adaptive_finish.py' in source
    assert '--adaptive-max-games 72' in source
    assert '--adaptive-final-max-games 128' in source
    assert '$(date +%H%M%S)' in source
    assert '--state=running,activating' in source
    assert 'setenv="GOCUBE_NIGHT_TOOLING_COMMIT=$TOOLING_COMMIT"' in source
    assert "ADAPTIVE RESUME PASS" in source
