import socket

import pytest

from alphazero.envs.gocube.integration.dev_launcher import (
    KnownRun,
    assert_port_available,
    ensure_known_runs,
)
from alphazero.envs.gocube.integration.manifest import (
    ManifestExistsError,
    RunManifest,
    load_run_manifest,
    write_run_manifest,
)


def make_checkpoint(run_dir, iteration=0):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"iteration-{iteration:04d}.pkl"
    path.write_bytes(b"checkpoint")
    return path


def test_ensure_known_runs_registers_present_run_and_skips_missing(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    cube_run = checkpoint_dir / "gocube-cube4-stage4-v1"
    make_checkpoint(cube_run, 25)
    messages = []

    ready = ensure_known_runs(
        str(checkpoint_dir),
        runs=(
            KnownRun("gocube-cube4-stage4-v1", "cube", 4),
            KnownRun("torus-9x9-30iter", "torus", 9),
        ),
        emit=messages.append,
    )

    assert len(ready) == 1
    manifest = load_run_manifest(str(cube_run))
    assert manifest.run_name == "gocube-cube4-stage4-v1"
    assert manifest.topology == "cube"
    assert manifest.size == 4
    assert manifest.komi == 0.5
    assert any("skip missing legacy run torus-9x9-30iter" in message for message in messages)


def test_ensure_known_runs_is_idempotent_for_compatible_manifest(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    run_dir = checkpoint_dir / "gocube-cube4-stage4-v1"
    make_checkpoint(run_dir, 25)
    spec = (KnownRun("gocube-cube4-stage4-v1", "cube", 4),)

    first = ensure_known_runs(str(checkpoint_dir), runs=spec, emit=lambda _message: None)
    second = ensure_known_runs(str(checkpoint_dir), runs=spec, emit=lambda _message: None)

    assert first == second
    assert load_run_manifest(str(run_dir)).topology == "cube"


def test_ensure_known_runs_refuses_incompatible_existing_manifest(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    run_dir = checkpoint_dir / "gocube-cube4-stage4-v1"
    make_checkpoint(run_dir, 25)
    incompatible = RunManifest.create(
        run_name=run_dir.name,
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
    )
    write_run_manifest(str(run_dir), incompatible)

    with pytest.raises(ManifestExistsError, match="Refusing to overwrite incompatible"):
        ensure_known_runs(
            str(checkpoint_dir),
            runs=(KnownRun("gocube-cube4-stage4-v1", "cube", 4),),
            emit=lambda _message: None,
        )


def test_ensure_known_runs_requires_checkpoint_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint directory does not exist"):
        ensure_known_runs(str(tmp_path / "missing"), emit=lambda _message: None)


def test_assert_port_available_rejects_occupied_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        with pytest.raises(RuntimeError, match="already in use"):
            assert_port_available("127.0.0.1", port)


def test_assert_port_available_accepts_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    assert_port_available("127.0.0.1", port)
