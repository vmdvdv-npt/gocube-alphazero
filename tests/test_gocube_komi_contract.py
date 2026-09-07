from pathlib import Path

from alphazero.envs.gocube.contract import CUBE4_PRODUCTION, GOCUBE_KOMI, require_gocube_komi
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


ROOT = Path(__file__).resolve().parents[1]


def test_gocube_komi_contract_is_always_point_five():
    assert GOCUBE_KOMI == 0.5
    assert CUBE4_PRODUCTION.komi == GOCUBE_KOMI
    assert DEFAULT_KOMI == GOCUBE_KOMI
    assert V3_DEFAULT_KOMI == GOCUBE_KOMI
    assert Cube4JapaneseGame.KOMI == GOCUBE_KOMI
    assert Cube4JapaneseV2Game.KOMI == GOCUBE_KOMI
    assert Cube4ChineseGame.KOMI == GOCUBE_KOMI
    assert KnownRun("legacy", "cube", 4).komi == GOCUBE_KOMI
    assert require_gocube_komi(GOCUBE_KOMI) == GOCUBE_KOMI


def test_legacy_registration_cli_default_is_point_five():
    args = parse_register_args(["--run-name", "legacy", "--topology", "cube", "--size", "4"])
    assert args.komi == GOCUBE_KOMI


def test_v3_rules_fingerprint_default_is_point_five():
    topology = cube_topology(4)
    assert rules_fingerprint(topology) == rules_fingerprint(topology, GOCUBE_KOMI)


def test_active_gocube_runtime_contains_no_old_komi_literal():
    forbidden = "7" + ".5"
    offenders = []
    for relative_root in ("alphazero/envs/gocube", "tools"):
        root = ROOT / relative_root
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if forbidden in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
