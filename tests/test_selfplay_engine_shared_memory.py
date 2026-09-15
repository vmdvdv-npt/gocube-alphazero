from __future__ import annotations

import multiprocessing as mp

import pytest
import torch

from gocube_golden.rules import apply_action, prepare_legal_actions
from gocube_golden.selfplay_engine import (
    GameFinished,
    InferenceNeed,
    SharedMemorySpec,
    SelfPlayEngine,
    SelfPlayEngineConfig,
)
from gocube_golden.state import PASS, initial_state
from gocube_golden.topology import TORUS_9X9
from gocube_golden.torus9 import (
    build_torus9_observation,
    build_torus9_observation_into,
)


class _CooperativeGame:
    def __init__(self, game_id: str, steps: int) -> None:
        self.game_id = game_id
        self.value = int(game_id.rsplit("-", 1)[-1])
        self.steps = steps
        self.done = 0

    def advance(self):
        if self.done == self.steps:
            return GameFinished({"game_id": self.game_id, "value": self.value, "moves": self.steps})
        return InferenceNeed(self.value * 10 + self.done)

    def resume(self, evaluation) -> None:
        assert evaluation == self.value * 10 + self.done
        self.done += 1


def _write_int(payload, destination) -> None:
    destination.fill_(float(payload))


def _decode_int(policy, wdl):
    return int(round(float(policy[0])))


def _make_game(context, game_id, _client):
    return _CooperativeGame(game_id, int(context["steps"]))


def _infer_shared(batch):
    values = batch[:, 0].to(dtype=torch.float32)
    policy = torch.stack((values, torch.zeros_like(values)), dim=1)
    wdl = torch.ones((batch.shape[0], 1), dtype=torch.float32)
    return policy, wdl


def _metrics(record):
    return {"moves": record["moves"], "technical": False}


def _shared_config(**overrides):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("shared-memory process test requires fork")
    values = {
        "workers": 2,
        "inference_batch_cap": 4,
        "inference_batch_wait_ms": 2.0,
        "device": "cpu",
        "process_start_method": "fork",
        "inference_request_timeout_s": 5.0,
        "active_games_per_worker": 2,
    }
    values.update(overrides)
    return SelfPlayEngineConfig(**values)


def test_shared_memory_cooperative_scheduler_batches_and_replenishes():
    spec = SharedMemorySpec(
        observation_shape=(2,),
        policy_size=2,
        wdl_size=1,
        write_input=_write_int,
        decode_output=_decode_int,
    )
    telemetry = {}
    records = SelfPlayEngine(_shared_config()).run(
        [f"game-{index}" for index in range(8)],
        worker_play=lambda *_args: None,
        worker_context={"steps": 3},
        infer_batch=None,
        infer_shared_batch=_infer_shared,
        worker_game_factory=_make_game,
        shared_memory=spec,
        active_games_per_worker=2,
        record_metrics=_metrics,
        telemetry=telemetry,
    )
    assert [record["game_id"] for record in records] == [f"game-{index}" for index in range(8)]
    assert telemetry["shared_memory_transport"] is True
    assert telemetry["target_active_contexts"] == 4
    assert telemetry["peak_concurrent_search_contexts"] == 4
    assert telemetry["global_task_replenishment"] is True
    assert telemetry["pending_queue_exhausted"] is True
    assert telemetry["minimum_active_contexts"] > 0
    assert telemetry["tail_duration_after_pending_empty_sec"] >= 0.0
    assert telemetry["inference_rows"] == 24
    assert telemetry["max_inference_batch_rows"] >= 2
    assert telemetry["worker_blocked_inference_calls"] >= telemetry["inference_forwards"]
    assert telemetry["worker_to_broker_latency_ms"]["count"] == telemetry["worker_blocked_inference_calls"]


def test_total_active_contexts_caps_contexts_without_reducing_worker_pool():
    spec = SharedMemorySpec(
        observation_shape=(2,),
        policy_size=2,
        wdl_size=1,
        write_input=_write_int,
        decode_output=_decode_int,
    )
    telemetry = {}
    records = SelfPlayEngine(_shared_config(workers=4)).run(
        [f"game-{index}" for index in range(6)],
        worker_play=lambda *_args: None,
        worker_context={"steps": 2},
        infer_batch=None,
        infer_shared_batch=_infer_shared,
        worker_game_factory=_make_game,
        shared_memory=spec,
        active_games_per_worker=2,
        total_active_contexts=3,
        record_metrics=_metrics,
        telemetry=telemetry,
    )
    assert len(records) == 6
    assert telemetry["target_active_contexts"] == 3
    assert telemetry["peak_concurrent_search_contexts"] == 3
    assert telemetry["real_worker_pid_count"] == 4


def test_preallocated_torus9_observation_is_byte_identical_to_canonical_builder():
    state = initial_state(topology=TORUS_9X9, komi=0.5)
    for action in (0, 10, 20, PASS):
        state = apply_action(state, action).after
    context = prepare_legal_actions(state)
    expected = build_torus9_observation(state, legal_context=context)
    actual = torch.empty_like(expected)
    build_torus9_observation_into(state, actual, legal_context=context)
    assert torch.equal(expected, actual)
    assert expected.numpy().tobytes() == actual.numpy().tobytes()
