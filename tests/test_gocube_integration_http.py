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
    UnsupportedProtocol,
)
from alphazero.envs.gocube.integration.server import make_handler


class FakeService:
    device = "cpu"

    def __init__(self):
        self.mode = "ok"

    def health(self):
        return {"protocolVersion": 1, "status": "ok", "service": "gocube-alphazero", "device": "cpu"}

    def checkpoints(self):
        return {"protocolVersion": 1, "checkpoints": [{"id": "run@5"}]}

    def generate_game(self, request):
        if request.get("protocolVersion") != 1:
            raise UnsupportedProtocol("Unsupported protocolVersion")
        if self.mode == "not-found":
            raise CheckpointNotFound("Unknown checkpoint: missing@5")
        if self.mode == "incompatible":
            raise CheckpointIncompatible("Models are incompatible")
        if self.mode == "busy":
            raise GenerationBusy("Another game generation is already running")
        if self.mode == "crash":
            raise RuntimeError("secret traceback detail")
        return {"protocolVersion": 1, "game": {"moves": [], "result": {"winner": "draw"}}}


@contextmanager
def running_server(service, allowed_origins=("http://localhost:5173",)):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, allowed_origins))
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(port, method, path, *, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    raw = response.read()
    result = (response.status, dict(response.getheaders()), raw)
    connection.close()
    return result


def json_body(raw):
    return json.loads(raw.decode("utf-8"))


def test_health_checkpoints_and_options():
    service = FakeService()
    with running_server(service) as port:
        status, _, raw = request(port, "GET", "/v1/health")
        assert status == 200
        assert json_body(raw)["status"] == "ok"

        status, _, raw = request(port, "GET", "/v1/checkpoints")
        assert status == 200
        assert json_body(raw)["checkpoints"][0]["id"] == "run@5"

        status, headers, raw = request(
            port,
            "OPTIONS",
            "/v1/games",
            headers={"Origin": "http://localhost:5173"},
        )
        assert status == 204
        assert raw == b""
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        assert "POST" in headers["Access-Control-Allow-Methods"]


def test_post_game_and_cors_origin_echo():
    service = FakeService()
    payload = json.dumps({"protocolVersion": 1}).encode()
    with running_server(service) as port:
        status, headers, raw = request(
            port,
            "POST",
            "/v1/games",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
                "Origin": "http://localhost:5173",
            },
        )
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert json_body(raw)["game"]["result"]["winner"] == "draw"


def test_invalid_json_and_content_type_are_machine_readable():
    service = FakeService()
    with running_server(service) as port:
        status, _, raw = request(
            port,
            "POST",
            "/v1/games",
            body=b"not-json",
            headers={"Content-Type": "application/json", "Content-Length": "8"},
        )
        assert status == 400
        assert json_body(raw)["error"]["code"] == "invalid_request"

        status, _, raw = request(
            port,
            "POST",
            "/v1/games",
            body=b"{}",
            headers={"Content-Type": "text/plain", "Content-Length": "2"},
        )
        assert status == 400
        assert json_body(raw)["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "mode,status,code",
    [
        ("not-found", 404, "checkpoint_not_found"),
        ("incompatible", 422, "checkpoint_incompatible"),
        ("busy", 409, "generation_busy"),
    ],
)
def test_service_errors_keep_stable_http_contract(mode, status, code):
    service = FakeService()
    service.mode = mode
    payload = json.dumps({"protocolVersion": 1}).encode()
    with running_server(service) as port:
        actual, _, raw = request(
            port,
            "POST",
            "/v1/games",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
    assert actual == status
    assert json_body(raw)["error"]["code"] == code


def test_unsupported_protocol_and_internal_failure_do_not_leak_traceback():
    service = FakeService()
    payload = json.dumps({"protocolVersion": 99}).encode()
    with running_server(service) as port:
        status, _, raw = request(
            port,
            "POST",
            "/v1/games",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        assert status == 400
        assert json_body(raw)["error"]["code"] == "unsupported_protocol"

        service.mode = "crash"
        payload = json.dumps({"protocolVersion": 1}).encode()
        status, _, raw = request(
            port,
            "POST",
            "/v1/games",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        text = raw.decode("utf-8")
        assert status == 500
        assert json.loads(text)["error"]["code"] == "generation_failed"
        assert "Traceback" not in text
        assert "secret traceback detail" not in text


def test_disallowed_browser_origin_is_rejected_without_wildcard_cors():
    service = FakeService()
    with running_server(service) as port:
        status, headers, raw = request(
            port,
            "OPTIONS",
            "/v1/games",
            headers={"Origin": "https://example.com"},
        )
    assert status == 400
    assert "Access-Control-Allow-Origin" not in headers
    assert json_body(raw)["error"]["code"] == "invalid_request"
