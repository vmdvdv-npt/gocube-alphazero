from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import gocube_golden as g
import gocube_golden.experiment_profile as ep


CLEAN_CODE = g.CodeIdentity("a" * 40, "b" * 40, True)
DIRTY_CODE = g.CodeIdentity("a" * 40, "b" * 40, False)


def make_arena(*, code_identity=CLEAN_CODE, run_id="golden-proof", seed=20260912):
    return g.SequentialGoldenArena(
        master_seed=seed,
        run_id=run_id,
        code_identity=code_identity,
    )


def play_proof_pair(*, code_identity=CLEAN_CODE):
    arena = make_arena(code_identity=code_identity)
    pair = arena.play_pair(
        pair_id="pair-0001",
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
    )
    return arena, pair


def checkpoint_metadata(model_hash: str) -> dict[str, object]:
    return {
        "rules_profile_id": ep.RULES_PROFILE_ID,
        "rules_fingerprint": ep.RULES_FINGERPRINT,
        "topology_fingerprint": ep.TOPOLOGY_FINGERPRINT,
        "board_size": [5, 5],
        "point_id_order_identity": ep.POINT_ID_ORDER_IDENTITY,
        "komi": ep.BASELINE_KOMI,
        "observation_schema_id": ep.OBSERVATION_SCHEMA_ID,
        "observation_schema_version": ep.OBSERVATION_SCHEMA_VERSION,
        "observation_fingerprint": ep.OBSERVATION_FINGERPRINT,
        "target_contract_id": ep.TARGET_CONTRACT_ID,
        "target_contract_version": ep.TARGET_CONTRACT_VERSION,
        "target_fingerprint": ep.TARGET_FINGERPRINT,
        "value_head_semantics": ep.VALUE_HEAD_SEMANTICS,
        "network_heads_and_shapes": ep.NETWORK_HEADS_AND_SHAPES,
        "parent_or_source_run_identity": "source-run-0001",
        "model_hash": model_hash,
    }


class IdentityPassPlayer:
    is_search_player = False

    def __init__(self, identity: g.PlayerIdentity):
        self.identity = identity
        self.player_id = identity.logical_player_id

    def select_action(self, state, context):
        return g.PASS


def test_v2_machine_passport_loads_and_recomputes_pinned_fingerprints():
    profile = g.load_experiment_profile()
    assert profile["profile_id"] == "gocube-torus-golden-unified-v2"
    assert profile["experiment_fingerprint"] == g.EXPERIMENT_FINGERPRINT
    assert ep.compute_experiment_fingerprint(profile) == g.EXPERIMENT_FINGERPRINT
    assert ep.compute_search_contract_fingerprint(profile) == g.SEARCH_CONTRACT_FINGERPRINT
    assert profile["rules"]["komi"] == 0.5
    assert profile["topology"]["production_torus_factory"] is False


def test_v2_passport_contains_no_legacy_komi_literal():
    path = Path(__file__).resolve().parents[1] / "configs/gocube/torus_golden_v2.json"
    assert "7" + ".5" not in path.read_text(encoding="utf-8")


def test_seed_tampering_is_rejected_by_independent_derivation():
    _, (record, _) = play_proof_pair()
    corrupted = replace(record, seed_A=record.seed_A + 1)
    with pytest.raises(ValueError, match="derived seed evidence"):
        g.validate_game_record(corrupted)


def test_run_identity_tampering_is_rejected_by_recomputation():
    _, (record, _) = play_proof_pair()
    corrupted = replace(record, run_identity_fingerprint="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="does not recompute"):
        g.validate_game_record(corrupted)


def test_incomplete_pair_is_rejected_fail_closed():
    _, (first, _) = play_proof_pair()
    with pytest.raises(ValueError, match="exactly two games per pair"):
        g.recompute_summary((first,))


def test_three_games_for_same_pair_are_rejected_fail_closed():
    arena, _ = play_proof_pair()
    arena.play_game(
        game_id="pair-0001-g3",
        pair_id="pair-0001",
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )
    with pytest.raises(ValueError, match="exactly two games per pair"):
        g.recompute_summary(arena.records)


def test_pair_requires_one_A_black_and_one_B_black_game():
    arena = make_arena()
    first = arena.play_game(
        game_id="same-color-g1",
        pair_id="same-color",
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )
    second = arena.play_game(
        game_id="same-color-g2",
        pair_id="same-color",
        player_A=g.GoodPlayer("A"),
        player_B=g.BadPlayer("B"),
        black_player="A",
    )
    with pytest.raises(ValueError, match="one A-black and one B-black"):
        g.validate_pair_records((first, second))


def test_manifest_persists_exact_pair_schedule_and_validator_reconstructs_it():
    arena, _ = play_proof_pair()
    manifest = arena.manifest()
    assert manifest.pair_schedule == (("pair-0001", "pair-0001-g1", "pair-0001-g2"),)
    corrupted = replace(
        manifest,
        pair_schedule=(("pair-0001", "pair-0001-g2", "pair-0001-g1"),),
    )
    with pytest.raises(ValueError, match="pair_schedule"):
        g.validate_run_evidence(corrupted, arena.records)


def test_run_cannot_change_structured_player_identity_midstream():
    arena, _ = play_proof_pair()
    with pytest.raises(ValueError, match="cannot change A/B structured player identities"):
        arena.play_pair(
            pair_id="pair-0002",
            player_A=g.GoodPlayer("DIFFERENT-A"),
            player_B=g.BadPlayer("B"),
        )


def test_manifest_player_identity_tamper_is_rejected_against_raw_records():
    arena, _ = play_proof_pair()
    manifest = arena.manifest()
    tampered_A = replace(manifest.player_A, source_identity="tampered-player-source")
    corrupted = replace(manifest, player_A=tampered_A)
    with pytest.raises(ValueError, match="run identity|player identity"):
        g.validate_run_evidence(corrupted, arena.records)


def test_dirty_code_identity_cannot_be_canonical_evidence():
    arena, _ = play_proof_pair(code_identity=DIRTY_CODE)
    manifest = arena.manifest()
    with pytest.raises(ValueError, match="dirty working tree|non-canonical"):
        g.validate_run_evidence(manifest, arena.records, require_canonical=True)


def test_scripted_players_cannot_be_promoted_to_canonical_checkpoint_evidence():
    arena, _ = play_proof_pair()
    manifest = arena.manifest()
    with pytest.raises(ValueError, match="checkpoint identities"):
        g.validate_run_evidence(manifest, arena.records, require_canonical=True)


def test_checkpoint_identity_requires_full_metadata_and_matching_artifact_hash(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"golden-model-A")
    model_hash = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    metadata = checkpoint_metadata(model_hash)
    identity = g.checkpoint_player_identity(
        logical_player_id="checkpoint-A",
        checkpoint_path=artifact,
        metadata=metadata,
    )
    assert identity.player_kind == "checkpoint"
    assert identity.model_file_sha256 == model_hash
    assert identity.observation_fingerprint == ep.OBSERVATION_FINGERPRINT
    assert identity.target_fingerprint == ep.TARGET_FINGERPRINT

    incomplete = dict(metadata)
    incomplete.pop("network_heads_and_shapes")
    with pytest.raises(ValueError, match="metadata is incomplete"):
        g.checkpoint_player_identity(
            logical_player_id="checkpoint-A",
            checkpoint_path=artifact,
            metadata=incomplete,
        )

    wrong_hash = dict(metadata)
    wrong_hash["model_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="does not match artifact SHA256"):
        g.checkpoint_player_identity(
            logical_player_id="checkpoint-A",
            checkpoint_path=artifact,
            metadata=wrong_hash,
        )


def test_canonical_evidence_writes_manifest_and_raw_records_for_exact_checkpoints(tmp_path):
    artifacts = []
    identities = []
    for slot in ("A", "B"):
        artifact = tmp_path / f"model-{slot}.bin"
        artifact.write_bytes(f"golden-model-{slot}".encode("ascii"))
        model_hash = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifacts.append(artifact)
        identities.append(
            g.checkpoint_player_identity(
                logical_player_id=f"checkpoint-{slot}",
                checkpoint_path=artifact,
                metadata=checkpoint_metadata(model_hash),
            )
        )

    arena = make_arena(run_id="canonical-run")
    arena.play_pair(
        pair_id="canonical-pair",
        player_A=IdentityPassPlayer(identities[0]),
        player_B=IdentityPassPlayer(identities[1]),
    )
    manifest = arena.manifest(require_canonical=True)
    manifest_path, records_path = g.write_run_evidence(
        tmp_path / "evidence",
        manifest,
        arena.records,
        require_canonical=True,
    )
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["run_identity_fingerprint"] == manifest.run_identity_fingerprint
    assert persisted["pair_schedule"] == [["canonical-pair", "canonical-pair-g1", "canonical-pair-g2"]]
    assert len(records_path.read_text(encoding="utf-8").splitlines()) == 2


def test_game_seeds_preserve_existing_derivation_contract():
    expected_game = int.from_bytes(
        hashlib.sha256(b"99:pair:same:game").digest()[:8], "big"
    )
    expected_A = int.from_bytes(
        hashlib.sha256(f"{expected_game}:A".encode("utf-8")).digest()[:8], "big"
    )
    expected_B = int.from_bytes(
        hashlib.sha256(f"{expected_game}:B".encode("utf-8")).digest()[:8], "big"
    )
    assert g.derive_game_seeds(99, "pair", "same") == (expected_game, expected_A, expected_B)
