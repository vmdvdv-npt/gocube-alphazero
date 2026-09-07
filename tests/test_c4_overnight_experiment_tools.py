from __future__ import annotations

import py_compile
import subprocess
from pathlib import Path

from alphazero.envs.gocube.contract import CUBE4_PRODUCTION
from tools import c4_overnight_complete as implementation
from tools import c4_overnight_experiment as canonical
from tools.c4_preflight import validate_runtime_constants


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "tools" / "c4_overnight_experiment.py"
IMPLEMENTATION = ROOT / "tools" / "c4_overnight_complete.py"
PREFLIGHT = ROOT / "tools" / "c4_preflight.py"
SAFETY = ROOT / "tools" / "gocube_overnight_safety.py"
REPORTER = ROOT / "tools" / "run_with_github_reports.sh"

LEGACY_TOOLS = (
    ROOT / "tools" / "c4_overnight_hardened.py",
    ROOT / "tools" / "c4_adaptive_finish.py",
    ROOT / "tools" / "launch_c4_overnight.sh",
    ROOT / "tools" / "preflight_c4_overnight.sh",
    ROOT / "tools" / "resume_c4_overnight.sh",
    ROOT / "tools" / "resume_c4_adaptive.sh",
)


def test_active_overnight_components_have_valid_syntax():
    for path in (CANONICAL, IMPLEMENTATION, PREFLIGHT, SAFETY):
        py_compile.compile(str(path), doraise=True)
    subprocess.run(["bash", "-n", str(REPORTER)], check=True)


def test_obsolete_a_g_runtime_tools_are_removed():
    missing = [str(path.relative_to(ROOT)) for path in LEGACY_TOOLS if path.exists()]
    assert missing == []


def test_canonical_entrypoint_has_no_historical_a_g_surface():
    source = CANONICAL.read_text(encoding="utf-8")
    for token in (
        "FROZEN_TRAINING_COMMIT",
        "PARENT_ITERATION",
        "BRANCHES =",
        "AXIS_PAIRS",
        "LEGACY_ARENA_SIMS",
        "--max-hours",
    ):
        assert token not in source
    assert "c4_overnight_complete" in source
    assert "build_fresh_heldout_suite" in source
    assert "preflight(cli, _impl)" in source


def test_runtime_mirrors_match_the_single_contract_component():
    observed = validate_runtime_constants(implementation)
    contract = CUBE4_PRODUCTION
    assert observed == {
        "workers": contract.workers,
        "regular_sims": contract.regular_sims,
        "fast_sims": contract.fast_sims,
        "games_per_iteration": contract.games_per_iteration,
        "train_batch_size": contract.train_batch_size,
        "arena_sims": contract.arena_sims,
        "komi": contract.komi,
    }
    assert canonical.Experiment is not implementation.Experiment


def test_deprecated_adaptive_filename_is_only_a_thin_alias():
    alias = (ROOT / "tools" / "c4_adaptive_parameter_experiment.py").read_text(encoding="utf-8")
    assert "from tools.c4_overnight_experiment import *" in alias
    assert "class Experiment" not in alias
    assert "PARAMETER_SPECS =" not in alias


def test_reporter_still_mirrors_structured_publish_directory():
    source = REPORTER.read_text(encoding="utf-8")
    for token in (
        'PUBLISH_DIR="$RUN_DIR/publish"',
        "publish_signature",
        "report_signature",
        "copy_structured_publish",
        'cp -a "$PUBLISH_DIR/." "$dest/"',
        "watch_reports",
    ):
        assert token in source
