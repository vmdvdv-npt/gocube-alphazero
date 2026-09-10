from pathlib import Path

import pytest

from alphazero.envs.gocube.core import cube_topology
from alphazero.envs.gocube.game import (
    Cube4ChineseGame,
    Cube4JapaneseGame,
    Cube4JapaneseV2Game,
    DEFAULT_KOMI,
    V3_DEFAULT_KOMI,
)
from alphazero.envs.gocube.integration.dev_launcher import KnownRun
from alphazero.envs.gocube.integration.register_run import parse_args as parse_register_args
from alphazero.envs.gocube.katago_v3 import rules_fingerprint
from alphazero.envs.gocube.komi_policy import (
    GOCUBE_DEFAULT_KOMI,
    LEGACY_FORBIDDEN_KOMI,
    LegacyKomiError,
    validate_gocube_komi,
)
from alphazero.envs.gocube.production_contract import CUBE4_PRODUCTION, GOCUBE_KOMI
from tools import c4_overnight_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_point_five_remains_the_default_baseline_not_the_only_valid_komi():
    assert GOCUBE_DEFAULT_KOMI == 0.5
    assert GOCUBE_KOMI == GOCUBE_DEFAULT_KOMI
    assert DEFAULT_KOMI == GOCUBE_DEFAULT_KOMI
    assert V3_DEFAULT_KOMI == GOCUBE_DEFAULT_KOMI
    assert Cube4JapaneseGame.KOMI == GOCUBE_DEFAULT_KOMI
    assert Cube4JapaneseV2Game.KOMI == GOCUBE_DEFAULT_KOMI
    assert Cube4ChineseGame.KOMI == GOCUBE_DEFAULT_KOMI
    assert KnownRun("legacy", "cube", 4).komi == GOCUBE_DEFAULT_KOMI
    assert c4_overnight_experiment.EXPECTED_KOMI == GOCUBE_DEFAULT_KOMI


@pytest.mark.parametrize("komi", [-0.5, 0.0, 0.5, 1.5, 2.5, 6.5, 8.5])
def test_project_wide_komi_policy_accepts_explicit_finite_nonlegacy_values(komi):
    assert validate_gocube_komi(komi, context="test") == float(komi)


def test_legacy_seven_point_five_is_hard_banned_with_owner_escalation():
    assert LEGACY_FORBIDDEN_KOMI == 7.5
    with pytest.raises(LegacyKomiError, match="contact the project owner"):
        validate_gocube_komi(7.5, context="test")


@pytest.mark.parametrize("komi", [float("nan"), float("inf"), float("-inf"), True, None])
def test_invalid_nonfinite_or_nonnumeric_komi_fails_closed(komi):
    with pytest.raises(ValueError, match="finite numeric komi"):
        validate_gocube_komi(komi, context="test")


def test_cube4_historical_baseline_remains_locally_pinned_for_reproducibility():
    assert CUBE4_PRODUCTION.komi == GOCUBE_DEFAULT_KOMI
    assert CUBE4_PRODUCTION.workers == 16
    assert CUBE4_PRODUCTION.regular_sims == 50
    assert CUBE4_PRODUCTION.fast_sims == 20
    assert CUBE4_PRODUCTION.games_per_iteration == 256
    assert CUBE4_PRODUCTION.train_batch_size == 1024
    assert CUBE4_PRODUCTION.arena_sims == 50

    args = {
        "gocube_komi": 1.5,
        "gocube_topology": "cube",
        "gocube_size": 4,
        "numMCTSSims": 50,
        "numFastSims": 20,
        "arenaMCTSSims": 50,
        "train_batch_size": 1024,
        "workers": 16,
        "probFastSim": 0.25,
        "gocube_train_samples_per_new_sample": 1.0,
    }
    with pytest.raises(ValueError, match="experiment-specific reproducibility pin"):
        CUBE4_PRODUCTION.validate_checkpoint_args(args)


def test_legacy_registration_cli_default_is_point_five():
    args = parse_register_args(["--run-name", "legacy", "--topology", "cube", "--size", "4"])
    assert args.komi == GOCUBE_DEFAULT_KOMI


def test_v3_rules_fingerprint_default_is_point_five_and_explicit_komi_changes_identity():
    topology = cube_topology(4)
    assert rules_fingerprint(topology) == rules_fingerprint(topology, GOCUBE_DEFAULT_KOMI)
    assert rules_fingerprint(topology, 1.5) != rules_fingerprint(topology, GOCUBE_DEFAULT_KOMI)


def test_obsolete_runtime_entrypoints_are_deleted():
    obsolete = (
        "tools/c4_adaptive_finish.py",
        "tools/c4_overnight_hardened.py",
        "tools/launch_c4_overnight.sh",
        "tools/preflight_c4_overnight.sh",
        "tools/resume_c4_adaptive.sh",
        "tools/resume_c4_overnight.sh",
        "tools/c4_adaptive_parameter_experiment.py",
        "tools/c4_overnight_complete.py",
    )
    for relative in obsolete:
        assert not (ROOT / relative).exists(), relative
    assert (ROOT / "tools/c4_overnight_experiment.py").exists()
    assert (ROOT / "tools/_c4_overnight_runtime.py").exists()


def test_active_runtime_contains_no_unintentional_legacy_komi_literal():
    # production_contract.py and komi_policy.py intentionally contain 7.5 as
    # the centrally documented forbidden sentinel. It must not leak back into
    # ordinary active defaults/configuration code.
    active_paths = (
        "tools/c4_overnight_experiment.py",
        "tools/gocube_experiment_runner.py",
        "tools/gocube_experiment_resume.py",
        "tools/_c4_overnight_runtime.py",
        "alphazero/envs/gocube/game.py",
        "alphazero/envs/gocube/katago_train.py",
        "alphazero/envs/gocube/hardened_train.py",
        "alphazero/envs/gocube/train.py",
    )
    for relative in active_paths:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "7.5" not in text, relative

    entrypoint = (ROOT / "tools/c4_overnight_experiment.py").read_text(encoding="utf-8")
    for obsolete_token in (
        "FROZEN_TRAINING_COMMIT",
        "SOURCE_RUN_DEFAULT",
        "PARENT_ITERATION",
        "LEGACY_ARENA_SIMS",
        "BRANCHES =",
        "AXIS_PAIRS",
        "summary_markdown",
    ):
        assert obsolete_token not in entrypoint
