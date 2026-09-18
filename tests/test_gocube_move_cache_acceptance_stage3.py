from __future__ import annotations

from alphazero.envs.gocube.integration.catalog import CheckpointDescriptor
from alphazero.envs.gocube.integration.golden_mapping import mapping_for
from alphazero.envs.gocube.integration.golden_models import GoldenPlayableModel
from alphazero.envs.gocube.integration.golden_move import (
    GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID,
    GoldenMoveSelector,
    GoldenPositionContract,
    MoveSelection,
    initial_state_for_position,
)
from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService
from gocube_golden.arena_contract import SEARCH_IMPLEMENTATION_ID
from gocube_golden.search import Evaluation
from gocube_golden.state import PASS


def _descriptor() -> CheckpointDescriptor:
    position = GoldenPositionContract(
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
    )
    state = initial_state_for_position(position)
    mapping = mapping_for("torus", 9)
    architecture_id = "test-torus-architecture"
    observation_fingerprint = "sha256:" + "3" * 64
    target_fingerprint = "sha256:" + "4" * 64
    return CheckpointDescriptor(
        checkpoint_id="torus-cache-acceptance@1",
        run_name="torus-cache-acceptance",
        iteration=1,
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        terminal_adjudicator="golden-graph-area-v1",
        path="/tmp/torus-cache-acceptance-M1.pt",
        profile_id="test-torus-profile",
        architecture_id=architecture_id,
        rules_fingerprint=state.rules_fingerprint,
        observation_fingerprint=observation_fingerprint,
        target_fingerprint=target_fingerprint,
        serving_contract_data={
            "checkpoint_format": "golden_pt",
            "topology": "torus",
            "size": 9,
            "rule_set": "chinese",
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


class FakeCatalog:
    checkpoint_dir = "/tmp"
    publication_manifest = "/tmp/nonexistent-publication.json"

    def __init__(self, descriptor: CheckpointDescriptor):
        self.descriptor = descriptor

    def get(self, checkpoint_id: str):
        return self.descriptor if checkpoint_id == self.descriptor.checkpoint_id else None

    def list(self):
        return [self.descriptor]


class PassEvaluator:
    def evaluate(self, state):
        policy = [0.0] * (state.topology.point_count + 1)
        policy[-1] = 1.0
        return Evaluation(policy=policy, wdl=(0.5, 0.0, 0.5))


def _playable_model(descriptor: CheckpointDescriptor) -> GoldenPlayableModel:
    mapping = mapping_for(descriptor.topology, descriptor.size)
    return GoldenPlayableModel(
        descriptor=descriptor,
        network=object(),
        evaluator=PassEvaluator(),
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


def _select_kwargs(descriptor: CheckpointDescriptor, *, history):
    return {
        "checkpoint_id": descriptor.checkpoint_id,
        "topology": "torus",
        "size": 9,
        "rule_set": "chinese",
        "komi": 0.5,
        "history": history,
        "mcts_sims": 1,
    }


def test_select_move_real_golden_search_reuses_cached_model_across_replayed_positions():
    descriptor = _descriptor()
    service = GoCubeAlphaZeroService(
        "/tmp",
        device="cpu",
        catalog=FakeCatalog(descriptor),
        move_selector=GoldenMoveSelector(),
        model_cache_size=2,
    )
    model = _playable_model(descriptor)
    uncached_loads = 0

    def fake_uncached(actual_descriptor):
        nonlocal uncached_loads
        assert actual_descriptor == descriptor
        uncached_loads += 1
        return model

    service.loader._load_uncached = fake_uncached

    first = service.select_move(**_select_kwargs(descriptor, history=[]))
    first_place = mapping_for("torus", 9).golden_action_to_protocol(0)
    second = service.select_move(
        **_select_kwargs(
            descriptor,
            history=[
                {
                    "moveNumber": 1,
                    "color": "black",
                    "action": first_place,
                }
            ],
        )
    )

    assert first["color"] == "black"
    assert second["color"] == "white"
    assert first["action"] == {"type": "pass"}
    assert second["action"] == {"type": "pass"}
    assert first["searchProfileId"] == GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID
    assert second["searchProfileId"] == GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID
    assert uncached_loads == 1
    assert len(service.model_cache) == 1


class CheapPassSelector:
    def __init__(self):
        self.calls = 0

    def select_move(self, *, state, descriptor, model, mcts_sims):
        self.calls += 1
        return MoveSelection(
            action=PASS,
            legal_actions=(PASS,),
            simulations=mcts_sims,
            implementation_id=SEARCH_IMPLEMENTATION_ID,
            search_profile_id=GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID,
        )


def test_50_select_move_requests_reuse_one_uncached_checkpoint_load():
    descriptor = _descriptor()
    selector = CheapPassSelector()
    service = GoCubeAlphaZeroService(
        "/tmp",
        device="cpu",
        catalog=FakeCatalog(descriptor),
        move_selector=selector,
        model_cache_size=2,
    )
    uncached_loads = 0
    loaded_model = object()

    def fake_uncached(actual_descriptor):
        nonlocal uncached_loads
        assert actual_descriptor == descriptor
        uncached_loads += 1
        return loaded_model

    service.loader._load_uncached = fake_uncached

    results = [
        service.select_move(**_select_kwargs(descriptor, history=[]))
        for _ in range(50)
    ]

    assert selector.calls == 50
    assert uncached_loads == 1
    assert len(service.model_cache) == 1
    assert all(result["action"] == {"type": "pass"} for result in results)
