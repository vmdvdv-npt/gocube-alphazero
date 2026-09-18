import http.client
import json
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from alphazero.envs.gocube.integration.errors import (
    CheckpointIncompatible,
    CheckpointNotFound,
    GenerationBusy,
    GenerationFailed,
)
from alphazero.envs.gocube.integration.golden_mapping import mapping_for
from alphazero.envs.gocube.integration.golden_move import replay_action_history
from alphazero.envs.gocube.integration.server import make_handler
from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService
from gocube_golden.rules import apply_action
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, PASS


class ProtocolMoveService:
    device = "cpu"

    def __init__(self):
        self.mode = "place"
        self.calls = []
        self.game_requests = []

    def health(self):
        return {
            "protocolVersion": 1,
            "status": "ok",
            "service": "gocube-alphazero",
            "device": "cpu",
            "runtimeIdentity": {"schema": "test"},
            "capabilities": {"generateGame": True, "selectMove": True},
        }

    def checkpoints(self):
        return {"protocolVersion": 1, "checkpoints": [{"id": "torus-test@1"}]}

    def generate_game(self, request):
        self.game_requests.append(request)
        return {"protocolVersion": 1, "game": {"moves": [], "result": {"winner": "draw"}}}

    def select_move(
        self,
        *,
        checkpoint_id,
        topology,
        size,
        rule_set,
        komi,
        history,
        mcts_sims,
    ):
        self.calls.append(
            {
                "checkpoint_id": checkpoint_id,
                "topology": topology,
                "size": size,
                "rule_set": rule_set,
                "komi": komi,
                "history": list(history),
                "mcts_sims": mcts_sims,
            }
        )
        if checkpoint_id == "missing@1":
            raise CheckpointNotFound("Unknown checkpoint: missing@1")
        if checkpoint_id == "incompatible@1":
            raise CheckpointIncompatible("Checkpoint is incompatible")
        if self.mode == "busy":
            raise GenerationBusy("capacity exhausted")
        if self.mode == "search-failed":
            raise GenerationFailed("private evaluator failure")
        if self.mode == "crash":
            raise RuntimeError("secret traceback detail")

        state = replay_action_history(
            topology=topology,
            size=size,
            rule_set=rule_set,
            komi=komi,
            moves=history,
        )
        if state.is_terminal:
            from alphazero.envs.gocube.integration.errors import TerminalPosition

            raise TerminalPosition("terminal")

        legal = tuple(GoldenSearchAdapter().legal_actions(state))
        if self.mode == "pass":
            action = PASS
        else:
            action = next((candidate for candidate in legal if candidate != PASS), PASS)
        protocol_action = mapping_for(topology, size).golden_action_to_protocol(action)
        return {
            "checkpointId": checkpoint_id,
            "topology": topology,
            "size": size,
            "ruleSet": rule_set,
            "komi": komi,
            "color": "black" if state.side_to_move == BLACK else "white",
            "action": protocol_action,
            "mctsSims": mcts_sims,
            "searchProfileId": "golden-interactive-deterministic-v1",
        }


class EmptyCatalog:
    checkpoint_dir = "/tmp"
    publication_manifest = "/tmp/nonexistent-publication.json"

    def get(self, _checkpoint_id):
        return None

    def list(self):
        return []


class EmptyLoader:
    device = "cpu"


@contextmanager
def running_server(service):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, ()))
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(port, method, path, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if body is not None:
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    result = response.status, json.loads(raw.decode("utf-8")) if raw else None
    connection.close()
    return result


def move_payload(*, moves=None, request_id="req-1", checkpoint_id="torus-test@1", sims=128):
    return {
        "protocolVersion": 1,
        "requestId": request_id,
        "checkpointId": checkpoint_id,
        "mctsSims": sims,
        "position": {
            "topology": "torus",
            "size": 9,
            "ruleSet": "chinese",
            "komi": 0.5,
            "moves": [] if moves is None else moves,
        },
    }


def post_move(port, payload):
    return request(port, "POST", "/v1/move", payload)


def place(point):
    return mapping_for("torus", 9).golden_action_to_protocol(point)


def test_move_empty_history_returns_black_place_and_exact_search_contract():
    service = ProtocolMoveService()
    with running_server(service) as port:
        status, body = post_move(port, move_payload(sims=7))
    assert status == 200
    assert body == {
        "protocolVersion": 1,
        "requestId": "req-1",
        "checkpointId": "torus-test@1",
        "mctsSims": 7,
        "moveNumber": 1,
        "color": "black",
        "action": place(0),
        "search": {
            "simulations": 7,
            "implementationId": "golden-sequential-puct-v1",
        },
    }
    assert service.calls[0]["mcts_sims"] == 7
    assert service.calls[0]["history"] == []


def test_legal_history_returns_next_legal_white_action():
    history = [{"moveNumber": 1, "color": "black", "action": place(0)}]
    service = ProtocolMoveService()
    with running_server(service) as port:
        status, body = post_move(port, move_payload(moves=history))
    assert status == 200
    assert body["moveNumber"] == 2
    assert body["color"] == "white"
    state = replay_action_history(
        topology="torus", size=9, rule_set="chinese", komi=0.5, moves=history
    )
    golden_action = mapping_for("torus", 9).protocol_action_to_golden(body["action"])
    transition = apply_action(state, golden_action)
    assert transition.after.side_to_move == BLACK


def test_request_id_is_opaque_and_returned_without_change():
    opaque = "  req/Ä/日本/\\u0000 literal  "
    service = ProtocolMoveService()
    with running_server(service) as port:
        status, body = post_move(port, move_payload(request_id=opaque))
    assert status == 200
    assert body["requestId"] == opaque


def test_pass_has_only_protocol_v1_pass_shape():
    service = ProtocolMoveService()
    service.mode = "pass"
    with running_server(service) as port:
        status, body = post_move(port, move_payload())
    assert status == 200
    assert body["action"] == {"type": "pass"}
    assert "pointId" not in body["action"]


def test_double_pass_history_maps_to_position_terminal():
    history = [
        {"moveNumber": 1, "color": "black", "action": {"type": "pass"}},
        {"moveNumber": 2, "color": "white", "action": {"type": "pass"}},
    ]
    service = ProtocolMoveService()
    with running_server(service) as port:
        status, body = post_move(port, move_payload(moves=history))
    assert status == 409
    assert body["error"]["code"] == "position_terminal"


def test_illegal_history_maps_to_position_invalid():
    history = [
        {"moveNumber": 1, "color": "black", "action": place(0)},
        {"moveNumber": 2, "color": "white", "action": place(0)},
    ]
    service = ProtocolMoveService()
    with running_server(service) as port:
        status, body = post_move(port, move_payload(moves=history))
    assert status == 422
    assert body["error"]["code"] == "position_invalid"


@pytest.mark.parametrize(
    "history",
    [
        [{"moveNumber": 2, "color": "black", "action": {"type": "pass"}}],
        [{"moveNumber": 1, "color": "white", "action": {"type": "pass"}}],
    ],
    ids=["bad-numbering", "wrong-color"],
)
def test_invalid_history_metadata_is_controlled_4xx(history):
    service = ProtocolMoveService()
    with running_server(service) as port:
        status, body = post_move(port, move_payload(moves=history))
    assert 400 <= status < 500
    assert body["error"]["code"] == "position_invalid"


@pytest.mark.parametrize(
    "checkpoint_id,status,code",
    [
        ("missing@1", 404, "checkpoint_not_found"),
        ("incompatible@1", 422, "checkpoint_incompatible"),
    ],
)
def test_checkpoint_errors_remain_distinct(checkpoint_id, status, code):
    service = ProtocolMoveService()
    with running_server(service) as port:
        actual, body = post_move(port, move_payload(checkpoint_id=checkpoint_id))
    assert actual == status
    assert body["error"]["code"] == code


def test_unsupported_protocol_is_controlled_and_never_calls_service():
    service = ProtocolMoveService()
    payload = move_payload()
    payload["protocolVersion"] = 99
    with running_server(service) as port:
        status, body = post_move(port, payload)
    assert status == 400
    assert body["error"]["code"] == "unsupported_protocol"
    assert service.calls == []


@pytest.mark.parametrize(
    "mode,status,code",
    [
        ("search-failed", 500, "search_failed"),
        ("busy", 503, "service_busy"),
        ("crash", 500, "search_failed"),
    ],
)
def test_move_failures_have_endpoint_specific_machine_readable_errors(mode, status, code):
    service = ProtocolMoveService()
    service.mode = mode
    with running_server(service) as port:
        actual, body = post_move(port, move_payload())
    assert actual == status
    assert body["error"]["code"] == code
    assert "secret traceback detail" not in json.dumps(body)
    assert "private evaluator failure" not in json.dumps(body)


def test_real_service_health_adds_capability_without_removing_old_fields():
    service = GoCubeAlphaZeroService(
        "/tmp",
        device="cpu",
        catalog=EmptyCatalog(),
        loader=EmptyLoader(),
    )
    with running_server(service) as port:
        status, body = request(port, "GET", "/v1/health")
    assert status == 200
    assert body["protocolVersion"] == 1
    assert body["status"] == "ok"
    assert body["service"] == "gocube-alphazero"
    assert body["device"] == "cpu"
    assert "runtimeIdentity" in body
    assert body["capabilities"] == {"generateGame": True, "selectMove": True}


def test_checkpoints_endpoint_semantics_are_unchanged():
    service = ProtocolMoveService()
    with running_server(service) as port:
        status, body = request(port, "GET", "/v1/checkpoints")
    assert status == 200
    assert body == {"protocolVersion": 1, "checkpoints": [{"id": "torus-test@1"}]}


def test_games_endpoint_semantics_are_unchanged():
    service = ProtocolMoveService()
    payload = {
        "protocolVersion": 1,
        "blackCheckpointId": "torus-test@1",
        "whiteCheckpointId": "torus-test@1",
        "mctsSims": 2,
    }
    with running_server(service) as port:
        status, body = request(port, "POST", "/v1/games", payload)
    assert status == 200
    assert body == {"protocolVersion": 1, "game": {"moves": [], "result": {"winner": "draw"}}}
    assert service.game_requests == [payload]


def test_two_positions_do_not_share_game_state():
    service = ProtocolMoveService()
    history = [{"moveNumber": 1, "color": "black", "action": place(0)}]
    with running_server(service) as port:
        status1, body1 = post_move(port, move_payload(moves=history, request_id="with-history"))
        status2, body2 = post_move(port, move_payload(moves=[], request_id="fresh"))
    assert status1 == status2 == 200
    assert body1["moveNumber"] == 2
    assert body1["color"] == "white"
    assert body2["moveNumber"] == 1
    assert body2["color"] == "black"
    assert service.calls[0]["history"] == history
    assert service.calls[1]["history"] == []


def test_service_restart_needs_no_bot_session_restore():
    payload = move_payload(
        moves=[{"moveNumber": 1, "color": "black", "action": place(0)}],
        request_id="restart-proof",
    )
    with running_server(ProtocolMoveService()) as port:
        status1, body1 = post_move(port, payload)
    with running_server(ProtocolMoveService()) as port:
        status2, body2 = post_move(port, payload)
    assert status1 == status2 == 200
    assert body1 == body2


def test_side_to_move_is_not_accepted_as_an_authoritative_position_field():
    service = ProtocolMoveService()
    payload = move_payload()
    payload["position"]["sideToMove"] = "white"
    with running_server(service) as port:
        status, body = post_move(port, payload)
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert service.calls == []
