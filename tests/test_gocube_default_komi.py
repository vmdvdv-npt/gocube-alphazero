import pytest

from alphazero.envs.gocube.game import Cube4JapaneseGame, Torus9JapaneseGame, DEFAULT_KOMI
from alphazero.envs.gocube.integration.manifest import RunManifest
from alphazero.envs.gocube.integration.register_run import parse_args as parse_register_args


def test_global_gocube_default_komi_is_half_point():
    assert DEFAULT_KOMI == pytest.approx(0.5)


def test_cube_and_torus_share_half_point_default_komi():
    assert Cube4JapaneseGame.KOMI == pytest.approx(0.5)
    assert Torus9JapaneseGame.KOMI == pytest.approx(0.5)


def test_new_manifest_defaults_to_half_point_komi():
    manifest = RunManifest.create(
        run_name="komi05-default-test",
        topology="cube",
        size=4,
    )
    assert manifest.komi == pytest.approx(0.5)


def test_register_run_cli_defaults_to_half_point_komi():
    args = parse_register_args([
        "--run-name", "legacy-test",
        "--topology", "cube",
        "--size", "4",
    ])
    assert args.komi == pytest.approx(0.5)
