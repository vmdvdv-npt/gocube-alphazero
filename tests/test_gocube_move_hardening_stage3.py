from __future__ import annotations

import http.client
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from threading import Barrier, Event, Lock, Thread

import pytest

from alphazero.envs.gocube.integration.catalog import CheckpointDescriptor
from alphazero.envs.gocube.integration.errors import ServiceBusy
from alphazero.envs.gocube.integration.golden_mapping import mapping_for
from alphazero.envs.gocube.integration.golden_models import GOLDEN_BACKEND_KIND
from alphazero.envs.gocube.integration.golden_move import (
    GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID,
    GoldenPositionContract,
    MoveSelection,
    initial_state_for_position,
)
from alphazero.envs.gocube.integration.model_cache import BoundedModelCache
from alphazero.envs.gocube.integration.server import (
    MAX_MOVE_REQUEST_BYTES,
    MAX_REQUEST_BYTES,
    make_handler,
    parse_args,
)
from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService
from gocube_golden.arena_contract import SEARCH_IMPLEMENTATION_ID
from gocube_golden.state import PASS


def _descriptor(checkpoint_id: str = "torus-cache@1", *, path: str | None = None) -> CheckpointDescriptor:
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
        checkpoint_id=checkpoint_id,
        run_name=checkpoint_id.split("@", 1)[0],
        iteration=1,
        topology="torus",
        size=9,
        rule_set="chinese",
        komi=0.5,
        terminal_adjudicator="golden-graph-area-v1",
        path=path or f"/tmp/{checkpoint_id.replace('@', '-')}.pt",
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

    def __init__(self, *descriptors: CheckpointDescriptor):
        self.descriptors = {descriptor.checkpoint_id: descriptor for descriptor in descriptors}

    def get(self, checkpoint_id: str):
        return self.descriptors.get(checkpoint_id)

    def list(self):
        return list(self.descriptors.values())


def test_service_owned_cache_reuses_one_loaded_model_across_50_requests():
    descriptor = _descriptor()
    service = GoCubeAlphaZeroService(
        "/tmp",
        device="cpu",
        catalog=FakeCatalog(descriptor),
        model_cache_size=2,
    )
    uncached_calls = 0
    loaded_model = object()

    def fake_uncached(actual_descriptor):
        nonlocal uncached_calls
        assert actual_descriptor == descriptor
        uncached_calls += 1
        return loaded_model

    service.loader._load_uncached = fake_uncached

    models = [service.loader.load(descriptor.checkpoint_id)[1] for _ in range(50)]

    assert service.loader.cache is service.model_cache
    assert len(service.model_cache) == 1
    assert uncached_calls == 1
    assert all(model is loaded_model for model in models)


def test_service_cache_does_not_confuse_different_checkpoints():
    first = _descriptor("torus-cache@1")
    second = _descriptor("torus-cache@2")
    service = GoCubeAlphaZeroService(
        "/tmp",
        device="cpu",
        catalog=FakeCatalog(first, second),
        model_cache_size=2,
    )
    calls: list[str] = []
    models: dict[str, object] = {}

    def fake_uncached(descriptor):
        calls.append(descriptor.checkpoint_id)
        return models.setdefault(descriptor.checkpoint_id, object())

    service.loader._load_uncached = fake_uncached

    first_model = service.loader.load(first.checkpoint_id)[1]
    second_model = service.loader.load(second.checkpoint_id)[1]
    repeated_first_model = service.loader.load(first.checkpoint_id)[1]

    assert first_model is repeated_first_model
    assert first_model is not second_model
    assert calls == [first.checkpoint_id, second.checkpoint_id]


def test_model_cache_is_bounded_and_evicts_lru_entry():
    cache = BoundedModelCache(max_entries=2)
    calls: list[str] = []

    def load(label):
        calls.append(label)
        return object()

    key_a = ("a", "cpu", GOLDEN_BACKEND_KIND)
    key_b = ("b", "cpu", GOLDEN_BACKEND_KIND)
    key_c = ("c", "cpu", GOLDEN_BACKEND_KIND)

    a1 = cache.get_or_load(key_a, lambda: load("a"))
    cache.get_or_load(key_b, lambda: load("b"))
    assert cache.get_or_load(key_a, lambda: load("unexpected-a")) is a1
    cache.get_or_load(key_c, lambda: load("c"))

    assert len(cache) == 2
    assert cache.keys() == (key_a, key_c)

    cache.get_or_load(key_b, lambda: load("b-reload"))
    assert len(cache) == 2
    assert calls == ["a", "b", "c", "b-reload"]


def test_model_cache_distinguishes_device_and_backend_identity():
    cache = BoundedModelCache(max_entries=4)
    calls: list[str] = []

    def load(label):
        calls.append(label)
        return object()

    keys = (
        ("checkpoint@1", "cpu", GOLDEN_BACKEND_KIND),
        ("checkpoint@1", "cuda", GOLDEN_BACKEND_KIND),
        ("checkpoint@1", "cpu", "alternate-backend"),
    )
    values = [cache.get_or_load(key, lambda key=key: load(str(key))) for key in keys]

    assert len({id(value) for value in values}) == 3
    assert len(calls) == 3
    for key, value in zip(keys, values):
        assert cache.get_or_load(key, lambda: pytest.fail("cache identity was not reused")) is value


def test_model_cache_single_flight_coalesces_parallel_first_load():
    cache = BoundedModelCache(max_entries=2)
    barrier = Barrier(8)
    load_started = Event()
    release_load = Event()
    count_lock = Lock()
    load_count = 0
    loaded_model = object()
    key = ("checkpoint@1", "cpu", GOLDEN_BACKEND_KIND)

    def loader():
        nonlocal load_count
        with count_lock:
            load_count += 1
        load_started.set()
        assert release_load.wait(timeout=2)
        return loaded_model

    def worker():
        barrier.wait(timeout=2)
        return cache.get_or_load(key, loader)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker) for _ in range(8)]
        try:
            assert load_started.wait(timeout=2)
        finally:
            release_load.set()
        results = [future.result(timeout=2) for future in futures]

    assert load_count == 1
    assert all(result is loaded_model for result in results)


def test_model_cache_rejects_non_positive_bound():
    with pytest.raises(ValueError):
        BoundedModelCache(max_entries=0)


class HttpProbeService:
    device = "cpu"

    def __init__(self):
        self.move_calls = 0
        self.game_calls = 0

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
        return {"protocolVersion": 1, "checkpoints": []}

    def generate_game(self, _request):
        self.game_calls += 1
        return {"protocolVersion": 1, "game": {"moves": [], "result": {"winner": "draw"}}}

    def select_move(self, **kwargs):
        self.move_calls += 1
        return {
            "checkpointId": kwargs["checkpoint_id"],
            "topology": kwargs["topology"],
            "size": kwargs["size"],
            "ruleSet": kwargs["rule_set"],
            "komi": kwargs["komi"],
            "color": "black",
            "action": {"type": "pass"},
            "mctsSims": kwargs["mcts_sims"],
            "searchProfileId": GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID,
        }


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


def raw_request(port: int, path: str, body: bytes):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    response = connection.getresponse()
    raw = response.read()
    result = response.status, json.loads(raw.decode("utf-8")) if raw else None
    connection.close()
    return result


def move_body(request_id: str) -> bytes:
    return json.dumps(
        {
            "protocolVersion": 1,
            "requestId": request_id,
            "checkpointId": "torus-cache@1",
            "mctsSims": 1,
            "position": {
                "topology": "torus",
                "size": 9,
                "ruleSet": "chinese",
                "komi": 0.5,
                "moves": [],
            },
        }
    ).encode("utf-8")


def test_move_request_can_exceed_legacy_64k_limit_within_endpoint_bound():
    service = HttpProbeService()
    body = move_body("x" * (MAX_REQUEST_BYTES + 1024))
    assert MAX_REQUEST_BYTES < len(body) < MAX_MOVE_REQUEST_BYTES

    with running_server(service) as port:
        status, response = raw_request(port, "/v1/move", body)

    assert status == 200
    assert response["requestId"].startswith("x")
    assert service.move_calls == 1


def test_oversized_move_request_is_controlled_4xx():
    service = HttpProbeService()
    body = move_body("x" * MAX_MOVE_REQUEST_BYTES)
    assert len(body) > MAX_MOVE_REQUEST_BYTES

    with running_server(service) as port:
        status, response = raw_request(port, "/v1/move", body)

    assert 400 <= status < 500
    assert response["error"]["code"] == "invalid_request"
    assert service.move_calls == 0


def test_games_endpoint_keeps_legacy_64k_request_limit():
    service = HttpProbeService()
    body = json.dumps(
        {
            "protocolVersion": 1,
            "blackCheckpointId": "a",
            "whiteCheckpointId": "a",
            "mctsSims": 1,
            "padding": "x" * MAX_REQUEST_BYTES,
        }
    ).encode("utf-8")
    assert len(body) > MAX_REQUEST_BYTES

    with running_server(service) as port:
        status, response = raw_request(port, "/v1/games", body)

    assert 400 <= status < 500
    assert response["error"]["code"] == "invalid_request"
    assert service.game_calls == 0


class FakeLoader:
    device = "cpu"

    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.calls = 0

    def load(self, checkpoint_id):
        self.calls += 1
        assert checkpoint_id == self.descriptor.checkpoint_id
        return self.descriptor, object()


class BlockingSelector:
    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.calls = 0

    def select_move(self, *, state, descriptor, model, mcts_sims):
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=2)
        return MoveSelection(
            action=PASS,
            legal_actions=(PASS,),
            simulations=mcts_sims,
            implementation_id=SEARCH_IMPLEMENTATION_ID,
            search_profile_id=GOLDEN_INTERACTIVE_SEARCH_PROFILE_ID,
        )


def test_move_capacity_exhaustion_is_runtime_policy_not_game_state():
    descriptor = _descriptor()
    selector = BlockingSelector()
    loader = FakeLoader(descriptor)
    service = GoCubeAlphaZeroService(
        "/tmp",
        device="cpu",
        catalog=FakeCatalog(descriptor),
        loader=loader,
        move_selector=selector,
        move_concurrency=1,
    )
    kwargs = {
        "checkpoint_id": descriptor.checkpoint_id,
        "topology": "torus",
        "size": 9,
        "rule_set": "chinese",
        "komi": 0.5,
        "history": [],
        "mcts_sims": 1,
    }

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(service.select_move, **kwargs)
        try:
            assert selector.started.wait(timeout=2)
            with pytest.raises(ServiceBusy) as exc_info:
                service.select_move(**kwargs)
            assert exc_info.value.code == "service_busy"
        finally:
            selector.release.set()
        result = first.result(timeout=2)

    assert result["action"] == {"type": "pass"}
    assert selector.calls == 1
    assert loader.calls == 1


def test_cli_exposes_cache_and_move_capacity_controls():
    defaults = parse_args([])
    assert defaults.model_cache_size == 2
    assert defaults.move_concurrency == 1

    configured = parse_args(["--model-cache-size", "5", "--move-concurrency", "3"])
    assert configured.model_cache_size == 5
    assert configured.move_concurrency == 3
