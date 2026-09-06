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


def test_gocube_komi_contract_is_always_point_five():
    assert DEFAULT_KOMI == 0.5
    assert V3_DEFAULT_KOMI == 0.5
    assert Cube4JapaneseGame.KOMI == 0.5
    assert Cube4JapaneseV2Game.KOMI == 0.5
    assert Cube4ChineseGame.KOMI == 0.5
    assert KnownRun("legacy", "cube", 4).komi == 0.5


def test_legacy_registration_cli_default_is_point_five():
    args = parse_register_args(["--run-name", "legacy", "--topology", "cube", "--size", "4"])
    assert args.komi == 0.5


def test_v3_rules_fingerprint_default_is_point_five():
    topology = cube_topology(4)
    assert rules_fingerprint(topology) == rules_fingerprint(topology, 0.5)
