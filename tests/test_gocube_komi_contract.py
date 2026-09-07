from pathlib import Path

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
from alphazero.envs.gocube.production_contract import CUBE4_PRODUCTION, GOCUBE_KOMI
from tools import c4_overnight_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_gocube_komi_has_one_authoritative_value():
    assert GOCUBE_KOMI == 0.5
    assert DEFAULT_KOMI == GOCUBE_KOMI
    assert V3_DEFAULT_KOMI == GOCUBE_KOMI
    assert Cube4JapaneseGame.KOMI == GOCUBE_KOMI
    assert Cube4JapaneseV2Game.KOMI == GOCUBE_KOMI
    assert Cube4ChineseGame.KOMI == GOCUBE_KOMI
    assert KnownRun("legacy", "cube", 4).komi == GOCUBE_KOMI
    assert c4_overnight_experiment.EXPECTED_KOMI == GOCUBE_KOMI


def test_cube4_fixed_runtime_contract_is_centralized():
    assert CUBE4_PRODUCTION.komi == GOCUBE_KOMI
    assert CUBE4_PRODUCTION.workers == 16
    assert CUBE4_PRODUCTION.regular_sims == 50
    assert CUBE4_PRODUCTION.fast_sims == 20
    assert CUBE4_PRODUCTION.games_per_iteration == 256
    assert CUBE4_PRODUCTION.train_batch_size == 1024
    assert CUBE4_PRODUCTION.arena_sims == 50


def test_legacy_registration_cli_default_is_point_five():
    args = parse_register_args(["--run-name", "legacy", "--topology", "cube", "--size", "4"])
    assert args.komi == GOCUBE_KOMI


def test_v3_rules_fingerprint_default_is_point_five():
    topology = cube_topology(4)
    assert rules_fingerprint(topology) == rules_fingerprint(topology, GOCUBE_KOMI)


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


def test_active_cube4_runtime_contains_no_obsolete_komi_literal_or_a_to_g_surface():
    active_paths = (
        "tools/c4_overnight_experiment.py",
        "tools/_c4_overnight_runtime.py",
        "alphazero/envs/gocube/production_contract.py",
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
