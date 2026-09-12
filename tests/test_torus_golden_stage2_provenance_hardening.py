from __future__ import annotations

import hashlib

import pytest

import gocube_golden as g
import gocube_golden.experiment_profile as ep


def _metadata(model_hash: str) -> dict[str, object]:
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
        "parent_or_source_run_identity": "source-run",
        "model_hash": model_hash,
    }


def test_checkpoint_default_source_identity_is_independent_of_local_path(tmp_path):
    payload = b"same-exact-checkpoint"
    model_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    left = tmp_path / "left" / "model.bin"
    right = tmp_path / "right" / "renamed.bin"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(payload)
    right.write_bytes(payload)

    identity_left = g.checkpoint_player_identity(
        logical_player_id="same-model",
        checkpoint_path=left,
        metadata=_metadata(model_hash),
    )
    identity_right = g.checkpoint_player_identity(
        logical_player_id="same-model",
        checkpoint_path=right,
        metadata=_metadata(model_hash),
    )
    assert identity_left == identity_right
    assert identity_left.source_identity == f"checkpoint:{model_hash}"


def test_seed_derivation_rejects_colon_ambiguous_pair_or_game_ids():
    with pytest.raises(ValueError, match="must not contain"):
        g.derive_game_seeds(1, "pair:ambiguous", "game")
    with pytest.raises(ValueError, match="must not contain"):
        g.derive_game_seeds(1, "pair", "game:ambiguous")
