from __future__ import annotations

import multiprocessing as mp
import os
import time

import pytest

from gocube_golden.selfplay_engine import SelfPlayEngine, SelfPlayEngineConfig, SelfPlayEngineError


def _worker(context, game_id, client):
    value = int(game_id.rsplit("-", 1)[-1])
    result = client.request(value)
    time.sleep(float(context.get("sleep", 0.0)))
    return {"game_id": game_id, "value": result, "moves": value + 1, "technical": False}


def _worker_failure(context, game_id, client):
    del context, client
    if game_id == "game-1":
        raise RuntimeError("intentional worker failure")
    return {"game_id": game_id}


def _worker_death(context, game_id, client):
    del context, client
    if game_id == "game-1":
        os._exit(17)
    return {"game_id": game_id}


def _infer(payloads):
    time.sleep(0.01)
    return tuple(int(value) * 2 for value in payloads)


def _broken_infer(payloads):
    return tuple(payloads[:-1])


def _metrics(record):
    return {"moves": record.get("moves", 0), "technical": record.get("technical", False)}


def _config(**overrides):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("process execution test requires fork")
    values = {
        "workers": 4,
        "inference_batch_cap": 4,
        "inference_batch_wait_ms": 20.0,
        "device": "cpu",
        "process_start_method": "fork",
        "inference_request_timeout_s": 5.0,
    }
    values.update(overrides)
    return SelfPlayEngineConfig(**values)


def test_process_workers_central_batching_and_ordering():
    telemetry = {}
    engine = SelfPlayEngine(_config())
    requested = [f"game-{index}" for index in reversed(range(8))]
    records = engine.run(
        requested,
        worker_play=_worker,
        worker_context={"sleep": 0.01},
        infer_batch=_infer,
        record_metrics=_metrics,
        telemetry=telemetry,
    )
    assert [record["game_id"] for record in records] == [f"game-{index}" for index in range(8)]
    assert [record["value"] for record in records] == [index * 2 for index in range(8)]
    assert telemetry["configured_workers"] == 4
    assert telemetry["real_worker_pid_count"] == 4
    assert len(set(telemetry["worker_pids"])) == 4
    assert telemetry["peak_concurrent_search_workers"] >= 2
    assert telemetry["inference_rows"] == 8
    assert telemetry["inference_forwards"] < telemetry["inference_rows"]
    assert telemetry["max_inference_batch_rows"] >= 2
    assert telemetry["games_completed"] == 8
    assert telemetry["games_failed"] == 0
    assert telemetry["result_order"] == [f"game-{index}" for index in range(8)]


def test_multiple_search_lanes_per_process_route_responses_without_pid_duplication():
    telemetry = {}
    engine = SelfPlayEngine(_config(workers=2, inference_batch_cap=4, lanes_per_worker=2))
    records = engine.run(
        [f"game-{index}" for index in range(8)],
        worker_play=_worker,
        worker_context={"sleep": 0.01},
        infer_batch=_infer,
        record_metrics=_metrics,
        telemetry=telemetry,
    )
    assert len(records) == 8
    assert telemetry["worker_processes_started"] == 2
    assert telemetry["real_worker_pid_count"] == 2
    assert telemetry["configured_search_lanes"] == 4
    assert telemetry["peak_concurrent_search_workers"] == 4
    assert telemetry["inference_rows"] == 8
    assert telemetry["games_failed"] == 0


def test_worker_exception_fails_closed_instead_of_returning_partial_games():
    engine = SelfPlayEngine(_config(workers=2, inference_batch_cap=2, inference_batch_wait_ms=0.0))
    with pytest.raises(SelfPlayEngineError, match="intentional worker failure"):
        engine.run(
            ["game-0", "game-1", "game-2"],
            worker_play=_worker_failure,
            worker_context={},
            infer_batch=_infer,
        )


def test_worker_process_death_is_detected_fail_closed():
    engine = SelfPlayEngine(_config(workers=2, inference_batch_cap=2, inference_batch_wait_ms=0.0))
    with pytest.raises(SelfPlayEngineError, match="worker process died|workers exited"):
        engine.run(
            ["game-0", "game-1", "game-2"],
            worker_play=_worker_death,
            worker_context={},
            infer_batch=_infer,
        )


def test_malformed_inference_batch_fails_entire_engine():
    engine = SelfPlayEngine(_config(workers=2, inference_batch_cap=2, inference_batch_wait_ms=20.0))
    with pytest.raises(SelfPlayEngineError, match="inference owner returned"):
        engine.run(
            ["game-0", "game-1"],
            worker_play=_worker,
            worker_context={},
            infer_batch=_broken_infer,
        )
