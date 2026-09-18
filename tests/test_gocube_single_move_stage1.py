from __future__ import annotations

from dataclasses import replace

import pytest

from alphazero.envs.gocube.integration.catalog import CheckpointDescriptor
from alphazero.envs.gocube.integration.errors import (
    CheckpointIncompatible,
    InvalidMoveHistory,
    TerminalPosition,
)
from alphazero.envs.gocube.integration.golden_generation import (
    GOLDEN_PROTOCOL_RULESET,
    GoldenGameGenerator,
    replay_protocol_moves,
)
from alphazero.envs.gocube.integration.golden_mapping import mapping_for
from alphazero.envs.gocube.integration.golden_models import GoldenPlayableModel
from alphazero.envs.gocube.integration.golden_move import (
    GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID,
    GoldenMoveSelector,
    GoldenPositionContract,
    MoveSelection,
    initial_state_for_position,
    replay_action_history,
)
from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService
from gocube_golden.arena_contract import SEARCH_IMPLEMENTATION_ID
from gocube_golden.result import DOUBLE_PASS
from gocube_golden.search import Evaluation, SearchResult
from gocube_golden.state import BLACK, PASS, WHITE


class PreferredEvaluator:
    def __init__(self, preferred_action):
        self.preferred_action = preferred_action

    def evaluate(self, state):
        policy = [0.0] * (state.topology.point_count + 1)
        index = state.topology.point_count if self.preferred_action == PASS else int(self.preferred_action)
        policy[index] = 1.0
        return Evaluation(policy=policy, wdl=(0.5, 0.0, 0.5))


def _descriptor(topology: str) -> CheckpointDescriptor:
    size = 4 if topology == "cube" else 9
    position = GoldenPositionContract(
        topology=topology,
        size=size,
        rule_set=GOLDEN_PROTOCOL_RULESET,
        komi=0.5,
    )
    state = initial_state_for_position(position)
    mapping = mapping_for(topology, size)
    architecture_id = f"test-{topology}-architecture"
    observation_fingerprint = "sha256:" + "3" * 64
    target_fingerprint = "sha256:" + "4" * 64
    return CheckpointDescriptor(
        checkpoint_id=f"{topology}-test@1",
        run_name=f"{topology}-test",
        iteration=1,
        topology=topology,
        size=size,
        rule_set=GOLDEN_PROTOCOL_RULESET,
        komi=0.5,
        terminal_adjudicator="golden-graph-area-v1",
        path=f"/tmp/{topology}-test-M1.pt",
        profile_id=f"test-{topology}-profile",
        architecture_id=architecture_id,
        rules_fingerprint=state.rules_fingerprint,
        observation_fingerprint=observation_fingerprint,
        target_fingerprint=target_fingerprint,
        serving_contract_data={
            "checkpoint_format": "golden_pt",
            "topology": topology,
            "size": size,
            "rule_set": GOLDEN_PROTOCOL_RULESET,
            "terminal_adjudicator": "golden-graph-area-v1",
            "architecture_id": architecture_id,
            "topology_fingerprint": mapping.golden_topology_fingerprint,
            "rules_fingerprint": state.rules_fingerprint,
            "observation_fingerprint": observation_fingerprint,
            "target_fingerprint": target_fingerprint,
            "komi": 0.5,
        },
        published=True,
    )


def _model(descriptor: CheckpointDescriptor, preferred_action=0) -> GoldenPlayableModel:
    mapping = mapping_for(descriptor.topology, 4 if descriptor.topology == "cube" else 9)
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


class FakeCatalog:
    checkpoint_dir = "/tmp"
    publication_manifest = "/tmp/publication.json"

    def __init__(self, descriptor: CheckpointDescriptor):
        self.descriptor = descriptor

    def get(self, checkpoint_id: str):
        return self.descriptor if checkpoint_id == self.descriptor.checkpoint_id else None

    def list(self):
        return [self.descriptor]


class FakeLoader:
    device = "cpu"

    def __init__(self, descriptor: CheckpointDescriptor, model: GoldenPlayableModel):
        self.descriptor = descriptor
        self.model = model
        self.calls = 0

    def load(self, checkpoint_id: str):
        self.calls += 1
        assert checkpoint_id == self.descriptor.checkpoint_id
        return self.descriptor, self.model


class NeverSelector:
    def __init__(self):
        self.calls = 0

    def select_move(self, **_kwargs):
        self.calls += 1
        raise AssertionError("selector must not run")


def _service(
    descriptor: CheckpointDescriptor,
    *,
    model: GoldenPlayableModel | None = None,
    selector=None,
):
    canonical = _descriptor(descriptor.topology)
    loader = FakeLoader(descriptor, model or _model(canonical))
    service = GoCubeAlphaZeroService(
        "/tmp",
        device="cpu",
        catalog=FakeCatalog(descriptor),
        loader=loader,
        move_selector=selector or GoldenMoveSelector(),
    )
    return service, loader


def _place_history(topology: str, point: int = 0):
    size = 4 if topology == "cube" else 9
    action = mapping_for(topology, size).golden_action_to_protocol(point)
    return [{"moveNumber": 1, "color": "black", "action": action}]


@pytest.mark.parametrize(("topology", "size"), [("torus", 9), ("cube", 4)])
def test_empty_history_restores_initial_golden_state(topology, size):
    state = replay_action_history(
        topology=topology,
        size=size,
        rule_set="chinese",
        komi=0.5,
        moves=[],
    )
    expected = initial_state_for_position(
        GoldenPositionContract(topology=topology, size=size, rule_set="chinese", komi=0.5)
    )
    assert state == expected
    assert state.side_to_move == BLACK


def test_empty_history_selects_first_black_move():
    descriptor = _descriptor("torus")
    service, _loader = _service(descriptor, model=_model(descriptor, preferred_action=0))
    result = service.select_move(
        checkpoint_id=descriptor.checkpoint_id,
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        history=[],
        mcts_sims=2,
    )
    assert result["color"] == "black"
    assert result["action"]["type"] == "place"


def test_legal_history_restores_expected_state_and_next_color():
    state = replay_action_history(
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        moves=_place_history("torus"),
    )
    assert int(state.stones[0]) == int(BLACK)
    assert state.side_to_move == WHITE
    assert not state.is_terminal


def test_selector_returns_place_action_through_real_golden_search():
    descriptor = _descriptor("torus")
    model = _model(descriptor, preferred_action=0)
    state = initial_state_for_position(
        GoldenPositionContract(topology="torus", size=9, rule_set="chinese", komi=0.5)
    )
    selection = GoldenMoveSelector().select_move(
        state=state,
        descriptor=descriptor,
        model=model,
        mcts_sims=2,
    )
    assert selection.action == 0
    assert selection.action in selection.legal_actions
    assert selection.simulations == 2
    assert selection.implementation_id == SEARCH_IMPLEMENTATION_ID


def test_selector_can_return_pass_action():
    descriptor = _descriptor("torus")
    model = _model(descriptor, preferred_action=PASS)
    state = initial_state_for_position(
        GoldenPositionContract(topology="torus", size=9, rule_set="chinese", komi=0.5)
    )
    selection = GoldenMoveSelector().select_move(
        state=state,
        descriptor=descriptor,
        model=model,
        mcts_sims=2,
    )
    assert selection.action == PASS
    assert PASS in selection.legal_actions


def test_one_pass_position_remains_playable():
    history = [{"moveNumber": 1, "color": "black", "action": {"type": "pass"}}]
    state = replay_action_history(
        topology="torus", size=9, rule_set="chinese", komi=0.5, moves=history
    )
    assert not state.is_terminal
    assert state.side_to_move == WHITE
    descriptor = _descriptor("torus")
    selection = GoldenMoveSelector().select_move(
        state=state,
        descriptor=descriptor,
        model=_model(descriptor, preferred_action=0),
        mcts_sims=2,
    )
    assert selection.action in selection.legal_actions


class BombSearch:
    calls = 0

    def __init__(self, *_args, **_kwargs):
        BombSearch.calls += 1
        raise AssertionError("MCTS must not be constructed")


def test_double_pass_terminal_never_starts_mcts():
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
    BombSearch.calls = 0
    descriptor = _descriptor("torus")
    with pytest.raises(TerminalPosition):
        GoldenMoveSelector(search_factory=BombSearch).select_move(
            state=state,
            descriptor=descriptor,
            model=_model(descriptor),
            mcts_sims=2,
        )
    assert BombSearch.calls == 0


def test_illegal_action_in_history_is_rejected():
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


def test_broken_move_numbering_is_rejected():
    with pytest.raises(InvalidMoveHistory):
        replay_action_history(
            topology="torus",
            size=9,
            rule_set="chinese",
            komi=0.5,
            moves=[{"moveNumber": 2, "color": "black", "action": {"type": "pass"}}],
        )


def test_wrong_color_sequence_is_rejected():
    with pytest.raises(InvalidMoveHistory):
        replay_action_history(
            topology="torus",
            size=9,
            rule_set="chinese",
            komi=0.5,
            moves=[{"moveNumber": 1, "color": "white", "action": {"type": "pass"}}],
        )


def test_action_after_terminal_is_rejected():
    with pytest.raises(InvalidMoveHistory):
        replay_action_history(
            topology="torus",
            size=9,
            rule_set="chinese",
            komi=0.5,
            moves=[
                {"moveNumber": 1, "color": "black", "action": {"type": "pass"}},
                {"moveNumber": 2, "color": "white", "action": {"type": "pass"}},
                {"moveNumber": 3, "color": "black", "action": {"type": "pass"}},
            ],
        )


def test_strict_generated_game_replay_still_requires_captured():
    with pytest.raises(Exception):
        replay_protocol_moves(
            topology="torus",
            size=9,
            moves=[{"moveNumber": 1, "color": "black", "action": {"type": "pass"}}],
        )


class RecordingNeverSelector:
    def __init__(self):
        self.calls = 0

    def select_move(self, **_kwargs):
        self.calls += 1
        raise AssertionError("selector must not run before compatibility validation")


@pytest.mark.parametrize(
    "descriptor",
    [
        _descriptor("cube"),
        replace(_descriptor("torus"), size=8),
        replace(_descriptor("torus"), rule_set="japanese"),
        replace(_descriptor("torus"), komi=6.5),
    ],
    ids=["topology", "size", "rules", "komi"],
)
def test_incompatible_position_is_rejected_before_model_load_or_search(descriptor):
    selector = RecordingNeverSelector()
    service, loader = _service(descriptor, selector=selector)
    with pytest.raises(CheckpointIncompatible):
        service.select_move(
            checkpoint_id=descriptor.checkpoint_id,
            topology="torus",
            size=9,
            rule_set="chinese",
            komi=0.5,
            history=[],
            mcts_sims=3,
        )
    assert loader.calls == 0
    assert selector.calls == 0


class RecordingSearch:
    settings = None

    def __init__(self, settings, *, adapter):
        RecordingSearch.settings = settings
        self.adapter = adapter

    def search(self, state, evaluator, *, seed):
        legal = self.adapter.legal_actions(state)
        return SearchResult(
            action=legal[0],
            legal_actions=legal,
            root_visits=(),
            pi=(),
            simulations=RecordingSearch.settings.simulations,
            evaluator_calls=0,
        )


def test_requested_mcts_sims_reaches_fixed_interactive_search_contract():
    descriptor = _descriptor("torus")
    state = initial_state_for_position(
        GoldenPositionContract(topology="torus", size=9, rule_set="chinese", komi=0.5)
    )
    selection = GoldenMoveSelector(search_factory=RecordingSearch).select_move(
        state=state,
        descriptor=descriptor,
        model=_model(descriptor),
        mcts_sims=7,
    )
    settings = RecordingSearch.settings
    assert settings.simulations == 7
    assert settings.cpuct == 1.25
    assert settings.fpu == 0.0
    assert settings.root_noise is False
    assert settings.fast_search is False
    assert settings.resign is False
    assert settings.root_policy_temperature is False
    assert settings.move_temperature == 0.0
    assert settings.deterministic_tie_break is True
    assert selection.search_profile_id == GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID
    assert selection.action in selection.legal_actions


@pytest.mark.parametrize(("topology", "size"), [("torus", 9), ("cube", 4)])
def test_supported_topologies_pass_real_golden_search_path(topology, size):
    descriptor = _descriptor(topology)
    state = initial_state_for_position(
        GoldenPositionContract(topology=topology, size=size, rule_set="chinese", komi=0.5)
    )
    selection = GoldenMoveSelector().select_move(
        state=state,
        descriptor=descriptor,
        model=_model(descriptor, preferred_action=0),
        mcts_sims=1,
    )
    assert selection.action in selection.legal_actions


class PassSelector:
    def __init__(self):
        self.calls = []

    def select_move(self, *, state, descriptor, model, mcts_sims):
        self.calls.append((state.side_to_move, descriptor.checkpoint_id, mcts_sims))
        return MoveSelection(
            action=PASS,
            legal_actions=(PASS,),
            simulations=mcts_sims,
            implementation_id=SEARCH_IMPLEMENTATION_ID,
            search_profile_id=GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID,
        )


def test_golden_game_generator_uses_shared_move_selector():
    descriptor = _descriptor("torus")
    model = _model(descriptor)
    selector = PassSelector()
    game = GoldenGameGenerator(move_selector=selector).generate(
        black=descriptor,
        white=descriptor,
        black_model=model,
        white_model=model,
        mcts_sims=5,
    )
    assert [color for color, _checkpoint, _sims in selector.calls] == [BLACK, WHITE]
    assert all(sims == 5 for _color, _checkpoint, sims in selector.calls)
    assert [move["action"] for move in game["moves"]] == [
        {"type": "pass"},
        {"type": "pass"},
    ]
    assert game["result"]["terminationReason"] == DOUBLE_PASS


def test_service_calls_are_history_stateless_and_have_no_game_session_state():
    descriptor = _descriptor("torus")
    service, _loader = _service(descriptor, model=_model(descriptor, preferred_action=0))

    first = service.select_move(
        checkpoint_id=descriptor.checkpoint_id,
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        history=[],
        mcts_sims=1,
    )
    second = service.select_move(
        checkpoint_id=descriptor.checkpoint_id,
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        history=[{"moveNumber": 1, "color": "black", "action": {"type": "pass"}}],
        mcts_sims=1,
    )

    assert first["color"] == "black"
    assert second["color"] == "white"
    forbidden = {"games", "game_id", "current_game", "current_state", "history", "sessions"}
    assert forbidden.isdisjoint(service.__dict__)
