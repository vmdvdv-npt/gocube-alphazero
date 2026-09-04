import json

import pytest

from alphazero.envs.gocube.integration.catalog import CheckpointCatalog
from alphazero.envs.gocube.integration.manifest import (
    MANIFEST_FILENAME,
    ManifestError,
    ManifestExistsError,
    RunManifest,
    load_run_manifest,
    write_run_manifest,
)
from alphazero.envs.gocube.integration.register_run import register_run


def make_checkpoint(run_dir, iteration, content=b"checkpoint"):
    path = run_dir / f"iteration-{iteration:04d}.pkl"
    path.write_bytes(content)
    return path


def make_manifest(run_dir, *, topology="cube", size=4, run_name=None):
    manifest = RunManifest.create(
        run_name=run_name or run_dir.name,
        topology=topology,
        size=size,
        rule_set="chinese",
        komi=7.5,
    )
    write_run_manifest(str(run_dir), manifest)
    return manifest


def test_manifest_round_trip_and_directory_match(tmp_path):
    run_dir = tmp_path / "cube-run"
    run_dir.mkdir()
    expected = make_manifest(run_dir)
    assert load_run_manifest(str(run_dir)) == expected


def test_manifest_rejects_missing_fields_and_unsupported_version(tmp_path):
    run_dir = tmp_path / "cube-run"
    run_dir.mkdir()
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ManifestError, match="missing fields"):
        load_run_manifest(str(run_dir))

    payload = {
        "version": 2,
        "runName": run_dir.name,
        "topology": "cube",
        "size": 4,
        "ruleSet": "chinese",
        "komi": 7.5,
        "terminalAdjudicator": "gocube-conservative-area-v1",
    }
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="Unsupported run manifest version"):
        load_run_manifest(str(run_dir))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("topology", "plane", "Unsupported topology"),
        ("size", 1, "No enabled Chinese self-play game"),
        ("komi", float("nan"), "finite"),
        ("rule_set", "japanese", "Unsupported ruleSet"),
    ],
)
def test_manifest_validation(field, value, match):
    kwargs = dict(
        run_name="bad-run",
        topology="cube",
        size=4,
        rule_set="chinese",
        komi=7.5,
    )
    kwargs[field] = value
    with pytest.raises(ManifestError, match=match):
        RunManifest.create(**kwargs)


def test_manifest_rejects_run_directory_mismatch(tmp_path):
    run_dir = tmp_path / "actual"
    run_dir.mkdir()
    payload = RunManifest.create(run_name="other", topology="cube", size=4).to_dict()
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="does not match directory"):
        load_run_manifest(str(run_dir))


def test_catalog_exposes_only_manifested_well_named_nonempty_checkpoints(tmp_path):
    run_b = tmp_path / "b-run"
    run_a = tmp_path / "a-run"
    hidden = tmp_path / "no-manifest"
    for path in (run_b, run_a, hidden):
        path.mkdir()
    make_manifest(run_b)
    make_manifest(run_a)
    make_checkpoint(run_b, 5)
    make_checkpoint(run_b, 0)
    make_checkpoint(run_a, 2)
    make_checkpoint(hidden, 1)
    (run_a / "checkpoint-latest.pkl").write_bytes(b"x")
    (run_a / "iteration-0003.pkl").write_bytes(b"")
    (run_a / "iteration-abc.pkl").write_bytes(b"x")

    catalog = CheckpointCatalog(str(tmp_path))
    items = catalog.list()

    assert [item.checkpoint_id for item in items] == ["a-run@2", "b-run@0", "b-run@5"]
    assert [item.iteration for item in items if item.run_name == "b-run"] == [0, 5]
    assert all("path" not in item.to_api() for item in items)
    assert catalog.get("b-run@5").checkpoint_id == "b-run@5"
    assert catalog.get("../../secret") is None


def test_legacy_registration_and_no_silent_incompatible_overwrite(tmp_path):
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    make_checkpoint(run_dir, 5)

    path = register_run(
        checkpoint_dir=str(tmp_path),
        run_name="legacy",
        topology="cube",
        size=4,
        rule_set="chinese",
        komi=7.5,
    )
    assert path == str(run_dir / MANIFEST_FILENAME)
    assert CheckpointCatalog(str(tmp_path)).get("legacy@5") is not None

    incompatible = RunManifest.create(run_name="legacy", topology="cube", size=3)
    with pytest.raises(ManifestExistsError, match="Refusing to overwrite"):
        write_run_manifest(str(run_dir), incompatible)

    write_run_manifest(str(run_dir), incompatible, force=True)
    assert load_run_manifest(str(run_dir)).size == 3


def test_registration_requires_existing_checkpoint(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No supported iteration checkpoints"):
        register_run(
            checkpoint_dir=str(tmp_path),
            run_name="empty",
            topology="cube",
            size=4,
            rule_set="chinese",
            komi=7.5,
        )
