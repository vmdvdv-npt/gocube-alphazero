from pathlib import Path

import pytest

from alphazero.envs.gocube.game import Cube4JapaneseGame, Torus9JapaneseGame, DEFAULT_KOMI
from alphazero.envs.gocube.integration.manifest import RunManifest
from alphazero.envs.gocube.integration.register_run import parse_args as parse_register_args


ROOT = Path(__file__).resolve().parents[1]


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


def _current_gocube_text_files():
    yield ROOT / "README.md"
    for path in (ROOT / "alphazero" / "envs" / "gocube").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".pyx", ".md", ".json", ".sh"}:
            yield path
    for path in (ROOT / "tests").glob("test_gocube*"):
        if path.is_file():
            yield path
    for path in (ROOT / "tests").glob("test_c4_*"):
        if path.is_file():
            yield path
    for path in (ROOT / "tools").glob("*"):
        if path.is_file() and ("gocube" in path.name or path.name.startswith("c4_")):
            yield path
    for path in (ROOT / "docs").glob("*gocube*"):
        if path.is_file():
            yield path


def test_current_gocube_sources_do_not_reintroduce_legacy_komi_75():
    banned = ("7" + ".5", "japanese" + "75", "komi" + "75")
    offenders = []
    for path in sorted(set(_current_gocube_text_files())):
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
    assert not offenders, "legacy GoCube komi tokens found:\n" + "\n".join(offenders)
