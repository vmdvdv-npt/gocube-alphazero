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
from alphazero.envs.gocube.katago_v3 import KATAGO_REFERENCE_COMMIT, KATAGO_RULES_VERSION
from alphazero.envs.gocube.terminal import JAPANESE_CLEANUP_ADJUDICATOR_V2


def make_checkpoint(run_dir, iteration, content=b"checkpoint"):
    path = run_dir / f"iteration-{iteration:04d}.pkl"
    path.write_bytes(content)
    return path


def make_manifest(run_dir, *, topology="cube", size=4, run_name=None):
    manifest = RunManifest.create(
        run_name=run_name or run_dir.name, topology=topology, size=size,
        rule_set="chinese", komi=7.5,
    )
    write_run_manifest(str(run_dir), manifest)
    return manifest


def test_manifest_round_trip_and_directory_match(tmp_path):
    run_dir = tmp_path / "cube-run"
    run_dir.mkdir()
    expected = make_manifest(run_dir)
    assert expected.version == 1
    assert load_run_manifest(str(run_dir)) == expected


def test_new_japanese_manifest_uses_v3_contract(tmp_path):
    run_dir = tmp_path / "japanese-run"
    run_dir.mkdir()
    manifest = RunManifest.create(run_name=run_dir.name, topology="cube", size=4)
    write_run_manifest(str(run_dir), manifest)
    loaded = load_run_manifest(str(run_dir))
    assert loaded.version == 3
    assert loaded.rule_set == "japanese"
    assert loaded.terminal_adjudicator == "gocube-katago-japanese-v3"
    assert loaded.observation_schema == "gocube-observation-v3"
    assert loaded.rules_fingerprint
    assert loaded.katago_rules_version == KATAGO_RULES_VERSION
    assert loaded.katago_reference_commit == KATAGO_REFERENCE_COMMIT


def test_legacy_japanese_v2_manifest_remains_loadable(tmp_path):
    run_dir = tmp_path / "legacy-v2"
    run_dir.mkdir()
    manifest = RunManifest.create(
        run_name=run_dir.name, topology="cube", size=4,
        terminal_adjudicator=JAPANESE_CLEANUP_ADJUDICATOR_V2,
    )
    write_run_manifest(str(run_dir), manifest)
    loaded = load_run_manifest(str(run_dir))
    assert loaded.version == 2
    assert loaded.terminal_adjudicator == JAPANESE_CLEANUP_ADJUDICATOR_V2
    assert loaded.observation_schema is None


def test_manifest_rejects_missing_fields_and_unsupported_version(tmp_path):
    run_dir = tmp_path / "cube-run"
    run_dir.mkdir()
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ManifestError, match="missing fields"):
        load_run_manifest(str(run_dir))
    payload = {
        "version": 99, "runName": run_dir.name, "topology": "cube", "size": 4,
        "ruleSet": "japanese", "komi": 7.5,
        "terminalAdjudicator": "gocube-katago-japanese-v3",
    }
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="Unsupported run manifest version"):
        load_run_manifest(str(run_dir))


def test_manifest_rejects_wrong_json_field_types(tmp_path):
    run_dir = tmp_path / "cube-run"
    run_dir.mkdir()
    payload = RunManifest.create(run_name=run_dir.name, topology="cube", size=4).to_dict()
    payload["topology"] = []
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="topology must be a string"):
        load_run_manifest(str(run_dir))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("topology", "plane", "Unsupported topology"),
        ("size", 1, "No game"),
        ("komi", float("nan"), "finite"),
        ("rule_set", "korean", "ruleSet"),
    ],
)
def test_manifest_validation(field, value, match):
    kwargs = dict(run_name="bad-run", topology="cube", size=4, rule_set="chinese", komi=7.5)
    kwargs[field] = value
    with pytest.raises((ManifestError, ValueError), match=match):
        RunManifest.create(**kwargs)


def test_manifest_rejects_cross_version_rule_contract():
    payload = RunManifest.create(run_name="japanese", topology="cube", size=4).to_dict()
    payload["version"] = 1
    with pytest.raises(ManifestError, match="version 1"):
        RunManifest.from_dict(payload)


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
    for path in (run_b, run_a, hidden): path.mkdir()
    make_manifest(run_b); make_manifest(run_a)
    make_checkpoint(run_b, 5); make_checkpoint(run_b, 0); make_checkpoint(run_a, 2); make_checkpoint(hidden, 1)
    (run_a / "checkpoint-latest.pkl").write_bytes(b"x")
    (run_a / "iteration-0003.pkl").write_bytes(b"")
    (run_a / "iteration-abc.pkl").write_bytes(b"x")
    items = CheckpointCatalog(str(tmp_path)).list()
    assert [item.checkpoint_id for item in items] == ["a-run@2", "b-run@0", "b-run@5"]


def test_legacy_registration_and_no_silent_incompatible_overwrite(tmp_path):
    run_dir = tmp_path / "legacy"
    run_dir.mkdir(); make_checkpoint(run_dir, 5)
    path = register_run(checkpoint_dir=str(tmp_path), run_name="legacy", topology="cube", size=4,
                        rule_set="chinese", komi=7.5)
    assert path == str(run_dir / MANIFEST_FILENAME)
    incompatible = RunManifest.create(run_name="legacy", topology="cube", size=3)
    with pytest.raises(ManifestExistsError, match="Refusing to overwrite"):
        write_run_manifest(str(run_dir), incompatible)


def test_registration_requires_existing_checkpoint(tmp_path):
    run_dir = tmp_path / "empty"; run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No supported iteration checkpoints"):
        register_run(checkpoint_dir=str(tmp_path), run_name="empty", topology="cube", size=4,
                     rule_set="chinese", komi=7.5)
