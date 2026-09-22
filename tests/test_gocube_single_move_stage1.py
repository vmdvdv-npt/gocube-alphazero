from __future__ import annotations

from dataclasses import replace

import pytest

from alphazero.envs.gocube.integration.catalog import CheckpointDescriptor
from alphazero.envs.gocube.integration.errors import (
    CheckpointIncompatible,
    InvalidMoveHistory,
    TerminalPosition,
)
from alphazero.envs.gocube.integration.golden_generation import GOLDEN_PROTOCOL_RULESET
from alphazero.envs.gocube.integration.golden_mapping import mapping_for
from alphazero.envs.gocube.integration.golden_models import GoldenPlayableModel
from alphazero.envs.gocube.integration.golden_move import (
    GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID,
    GoldenMoveSelector,
    GoldenPositionContract,
    initial_state_for_position,
    replay_action_history,
)
from gocube_golden.arena_contract import SEARCH_IMPLEMENTATION_ID
from gocube_golden.search import Evaluation
from gocube_golden.state import BLACK, PASS, WHITE


class PreferredEvaluator:
    def __init__(self, preferred_action):
        self.preferred_action = preferred_action

    def evaluate(self, state):
        policy = [0.0] * (state.topology.point_count + 1)
        index = state.topology.point_count if self.preferred_action == PASS else int(self.preferred_action)
        policy[index] = 1.0
        return Evaluation(policy=policy, wdl=(0.5, 0.0, 0.5))


def _descriptor() -> CheckpointDescriptor:
    position = GoldenPositionContract(
        topology="torus",
        size=9,
        rule_set=GOLDEN_PROTOCOL_RULESET,
        komi=0.5,
    )
    state = initial_state_for_position(position)
    mapping = mapping_for("torus", 9)
    return CheckpointDescriptor(
        checkpoint_id="torus-test@1",
        run_name="torus-test",
        iteration=1,
        topology="torus",
        size=9,
        rule_set=GOLDEN_PROTOCOL_RULESET,
        komi=0.5,
        terminal_adjudicator="golden-graph-area-v1",
        path="/tmp/torus-test-M1.pt",
        profile_id="test-torus-profile",
        architecture_id="test-torus-architecture",
        rules_fingerprint=state.rules_fingerprint,
        observation_fingerprint="sha256:" + "3" * 64,
        target_fingerprint="sha256:" + "4" * 64,
        serving_contract_data={
            "checkpoint_format": "golden_pt",
            "topology": "torus",
            "size": 9,
            "rule_set": GOLDEN_PROTOCOL_RULESET,
            "terminal_adjudicator": "golden-graph-area-v1",
            "architecture_id": "test-torus-architecture",
            "topology_fingerprint": mapping.golden_topology_fingerprint,
            "rules_fingerprint": state.rules_fingerprint,
            "observation_fingerprint": "sha256:" + "3" * 64,
            "target_fingerprint": "sha256:" + "4" * 64,
            "komi": 0.5,
        },
        published=True,
    )


def _model(descriptor: CheckpointDescriptor, preferred_action=0) -> GoldenPlayableModel:
    mapping = mapping_for("torus", 9)
    return GoldenPlayableModel(
        descriptor=descriptor,
        network=object(),
        evaluator=PreferredEvaluator(preferred_action),
        metadata={
            "profile_id": descriptor.profile_id,
            "architecture_id": descriptor.architecture_id,
            "rules_fingerprint": descriptor.rules_fingerprint,
            "observation_fingerprint": descriptor.observation_fingerprint,
            "target_fingerprint": descriptor.target_fingerprint,
            "topology_fingerprint": mapping.golden_topology_fingerprint,
            "komi": descriptor.komi,
        },
        device="cpu",
    )


def test_empty_history_restores_initial_torus_state():
    state = replay_action_history(
        topology="torus", size=9, rule_set="chinese", komi=0.5, moves=[]
    )
    expected = initial_state_for_position(
        GoldenPositionContract(topology="torus", size=9, rule_set="chinese", komi=0.5)
    )
    assert state == expected
    assert state.side_to_move == BLACK


def test_legal_history_restores_expected_state_and_next_color():
    action = mapping_for("torus", 9).golden_action_to_protocol(0)
    state = replay_action_history(
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        moves=[{"moveNumber": 1, "color": "black", "action": action}],
    )
    assert int(state.stones[0]) == int(BLACK)
    assert state.side_to_move == WHITE
    assert not state.is_terminal


def test_selector_returns_place_action_through_real_golden_search():
    descriptor = _descriptor()
    state = initial_state_for_position(
        GoldenPositionContract(topology="torus", size=9, rule_set="chinese", komi=0.5)
    )
    selection = GoldenMoveSelector().select_move(
        state=state,
        descriptor=descriptor,
        model=_model(descriptor, preferred_action=0),
        mcts_sims=2,
    )
    assert selection.action == 0
    assert selection.action in selection.legal_actions
    assert selection.simulations == 2
    assert selection.implementation_id == SEARCH_IMPLEMENTATION_ID
    assert selection.search_profile_id == GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID


def test_selector_can_return_pass_action():
    descriptor = _descriptor()
    state = initial_state_for_position(
        GoldenPositionContract(topology="torus", size=9, rule_set="chinese", komi=0.5)
    )
    selection = GoldenMoveSelector().select_move(
        state=state,
        descriptor=descriptor,
        model=_model(descriptor, preferred_action=PASS),
        mcts_sims=2,
    )
    assert selection.action == PASS
    assert PASS in selection.legal_actions


def test_double_pass_terminal_never_starts_search():
    state = replay_action_history(
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        moves=[
            {"moveNumber": 1, "color": "black", "action": {"type": "pass"}},
            {"moveNumber": 2, "color": "white", "action": {"type": "pass"}},
        ],
    )
    assert state.is_terminal
    descriptor = _descriptor()
    with pytest.raises(TerminalPosition):
        GoldenMoveSelector().select_move(
            state=state,
            descriptor=descriptor,
            model=_model(descriptor),
            mcts_sims=2,
        )


def test_illegal_history_is_rejected():
    action = mapping_for("torus", 9).golden_action_to_protocol(0)
    with pytest.raises(InvalidMoveHistory):
        replay_action_history(
            topology="torus",
            size=9,
            rule_set="chinese",
            komi=0.5,
            moves=[
                {"moveNumber": 1, "color": "black", "action": action},
                {"moveNumber": 2, "color": "white", "action": action},
            ],
        )


def test_retired_cube_position_has_no_serving_initial_state():
    with pytest.raises(Exception):
        initial_state_for_position(
            GoldenPositionContract(topology="cube", size=4, rule_set="chinese", komi=0.5)
        )


@pytest.mark.parametrize(
    "descriptor",
    [
        replace(_descriptor(), topology="cube", size=4),
        replace(_descriptor(), size=8),
        replace(_descriptor(), rule_set="japanese"),
        replace(_descriptor(), komi=6.5),
    ],
    ids=["topology", "size", "rules", "komi"],
)
def test_incompatible_descriptor_fails_closed(descriptor):
    requested = GoldenPositionContract(
        topology="torus", size=9, rule_set="chinese", komi=0.5
    )
    state = initial_state_for_position(requested)
    with pytest.raises((CheckpointIncompatible, Exception)):
        GoldenMoveSelector().select_move(
            state=state,
            descriptor=descriptor,
            model=_model(_descriptor()),
            mcts_sims=1,
        )
