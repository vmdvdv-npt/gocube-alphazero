from __future__ import annotations

import gocube_golden as g


def test_forensic_wdl_perspective_and_pass_contract_are_unchanged():
    assert g.torus9_z_target("BLACK", g.BLACK) == (1.0, 0.0, 0.0)
    assert g.torus9_z_target("BLACK", g.WHITE) == (0.0, 0.0, 1.0)
    assert g.torus9_z_target("DRAW", g.BLACK) == (0.0, 1.0, 0.0)
    state = g.initial_state(topology=g.TORUS_9X9, komi=0.5)
    assert g.PASS in g.legal_actions(state)
    after_pass = g.apply_action(state, g.PASS).after
    assert g.PASS in g.legal_actions(after_pass)
    assert g.apply_action(after_pass, g.PASS).after.is_terminal
    assert g.TORUS9_PASS_INDEX == 81


def test_forensic_profile_rejects_legacy_fresh_only_semantics():
    profile = g.load_torus9_profile()
    assert profile["profile_id"] == "gocube-torus9-stable-learning-v2"
    assert profile["network"]["blocks"] == 8
    assert profile["replay"]["policy"] == "rolling-recent-generations"
    assert profile["replay"]["generations"] == 3
    assert profile["replay"]["maximum_positions"] == 20000
    assert profile["training"]["optimizer_steps_per_iteration"] == 80
    assert profile["training"]["samples_consumed_per_iteration"] == 5120


def test_forensic_legacy_four_block_model_is_not_canonical():
    legacy = g.Torus9GraphNet(blocks=4, architecture_id="GoldenGraphNetV1-Torus9")
    canonical = g.Torus9GraphNet()
    assert legacy.architecture_config["blocks"] == 4
    assert canonical.architecture_config["blocks"] == 8
    assert legacy.architecture_config["architecture_id"] != canonical.architecture_config["architecture_id"]
