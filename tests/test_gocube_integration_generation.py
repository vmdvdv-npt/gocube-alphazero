import numpy as np
import pytest

from alphazero.envs.gocube.game import Cube2ChineseGame, Cube4ChineseGame, Torus9ChineseGame
from alphazero.envs.gocube.integration.catalog import CheckpointDescriptor
from alphazero.envs.gocube.integration.errors import CheckpointIncompatible, GenerationBusy
from alphazero.envs.gocube.integration.generation import (
    GameGenerator,
    captured_point_ids,
    serialize_action,
)
from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService
from alphazero.utils import dotdict


def descriptor(checkpoint_id="run@5", *, topology="cube", size=2, rule_set="chinese"):
    run, iteration = checkpoint_id.rsplit("@", 1)
    return CheckpointDescriptor(
        checkpoint_id=checkpoint_id,
        run_name=run,
        iteration=int(iteration),
        topology=topology,
        size=size,
        rule_set=rule_set,
        komi=0.5,
        terminal_adjudicator="gocube-conservative-area-v1",
        path=f"/tmp/{checkpoint_id}.pkl",
    )


class PassPlayer:
    def __init__(self, game_cls, updates, args):
        self.game_cls = game_cls
        self.updates = updates
        self.args = args
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, state):
        return self.game_cls.pass_action()

    def update(self, state, action):
        self.updates.append((state.turns, state.player, action))


class FakeModel:
    def __init__(self):
        self.args = dotdict({"startTemp": 1.0, "arenaTemp": 0.25, "add_root_noise": True, "add_root_temp": True})


def test_action_serialization_preserves_canonical_cube_and_torus_order():
    assert serialize_action(Cube4ChineseGame, 0) == {"type": "place", "pointId": "front:0:0"}
    assert serialize_action(Cube4ChineseGame, Cube4ChineseGame.action_size() - 2) == {
        "type": "place",
        "pointId": "bottom:3:3",
    }
    assert serialize_action(Cube4ChineseGame, Cube4ChineseGame.pass_action()) == {"type": "pass"}
    assert serialize_action(Torus9ChineseGame, 0) == {"type": "place", "pointId": "0,0"}
    assert serialize_action(Torus9ChineseGame, 80) == {"type": "place", "pointId": "8,8"}


def test_capture_serialization_uses_opponent_board_diff_in_canonical_order():
    point_count = Cube2ChineseGame.logical_topology().point_count
    pre = np.zeros(point_count, dtype=np.uint8)
    post = np.zeros(point_count, dtype=np.uint8)
    pre[5] = 2
    pre[1] = 2
    pre[9] = 1
    post[9] = 1

    assert captured_point_ids(Cube2ChineseGame, pre, post, current_player=0) == [
        Cube2ChineseGame.point_id_for_action(1),
        Cube2ChineseGame.point_id_for_action(5),
    ]


def test_game_generator_uses_real_game_terminal_flow_and_evaluation_settings():
    updates = []
    seen_args = []

    def player_factory(model, game_cls, args):
        seen_args.append(args)
        return PassPlayer(game_cls, updates, args)

    generator = GameGenerator(player_factory=player_factory)
    black = descriptor("run@5")
    white = descriptor("run@0")
    game = generator.generate(
        black=black,
        white=white,
        black_model=FakeModel(),
        white_model=FakeModel(),
        mcts_sims=20,
    )

    assert game["moves"] == [
        {"moveNumber": 1, "color": "black", "action": {"type": "pass"}, "captured": []},
        {"moveNumber": 2, "color": "white", "action": {"type": "pass"}, "captured": []},
    ]
    assert game["result"]["winner"] == "white"
    assert game["result"]["adjudicatorId"] == "gocube-conservative-area-v1"
    assert game["result"]["score"]["komi"] == 0.5
    assert len(updates) == 4
    assert updates[0][:2] == (0, 0)
    assert updates[2][:2] == (1, 1)
    assert all(args.numMCTSSims == 20 for args in seen_args)
    assert all(args.add_root_noise is False for args in seen_args)
    assert all(args.add_root_temp is False for args in seen_args)
    assert all(args.startTemp == 0.0 for args in seen_args)
    assert all(args.arenaTemp == 0.0 for args in seen_args)


class FakeCatalog:
    def __init__(self, items):
        self.items = {item.checkpoint_id: item for item in items}

    def get(self, checkpoint_id):
        return self.items.get(checkpoint_id)

    def list(self):
        return list(self.items.values())


class FakeLoader:
    device = "cpu"

    def __init__(self, catalog):
        self.catalog = catalog
        self.calls = []
        self.models = {}

    def load(self, checkpoint_id):
        self.calls.append(checkpoint_id)
        item = self.catalog.get(checkpoint_id)
        model = self.models.setdefault(checkpoint_id, FakeModel())
        return item, model


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"moves": [], "result": {"winner": "draw"}}


def request(black="run@5", white="run@5", sims=20):
    return {
        "protocolVersion": 1,
        "blackCheckpointId": black,
        "whiteCheckpointId": white,
        "mctsSims": sims,
    }


def make_service(items):
    catalog = FakeCatalog(items)
    loader = FakeLoader(catalog)
    generator = RecordingGenerator()
    service = GoCubeAlphaZeroService(
        "/unused",
        catalog=catalog,
        loader=loader,
        generator=generator,
    )
    return service, loader, generator


def test_service_reuses_one_model_when_black_equals_white_and_propagates_sims():
    item = descriptor("run@5")
    service, loader, generator = make_service([item])
    response = service.generate_game(request(sims=33))

    assert loader.calls == ["run@5"]
    assert generator.calls[0]["black_model"] is generator.calls[0]["white_model"]
    assert generator.calls[0]["mcts_sims"] == 33
    assert response["protocolVersion"] == 1


def test_service_loads_two_models_for_different_compatible_checkpoints():
    items = [descriptor("run@0"), descriptor("run@5")]
    service, loader, _ = make_service(items)
    service.generate_game(request(black="run@5", white="run@0"))
    assert loader.calls == ["run@5", "run@0"]


def test_service_rejects_incompatible_models():
    service, _, _ = make_service([
        descriptor("cube@5", topology="cube", size=2),
        descriptor("torus@5", topology="torus", size=9),
    ])
    with pytest.raises(CheckpointIncompatible):
        service.generate_game(request(black="cube@5", white="torus@5"))


def test_service_rejects_parallel_generation_as_busy():
    item = descriptor("run@5")
    service, _, _ = make_service([item])
    assert service._generation_lock.acquire(blocking=False)
    try:
        with pytest.raises(GenerationBusy):
            service.generate_game(request())
    finally:
        service._generation_lock.release()
