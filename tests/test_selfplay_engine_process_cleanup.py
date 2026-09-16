from __future__ import annotations

import multiprocessing as mp
import os
import time

import pytest

from gocube_golden.selfplay_engine import (
    SelfPlayEngine,
    SelfPlayEngineConfig,
    SelfPlayEngineError,
)


def _worker_failure(_context, game_id, _client):
    if game_id == "game-1":
        raise RuntimeError("intentional cleanup regression failure")
    time.sleep(0.05)
    return {"game_id": game_id}


def _infer(payloads):
    return tuple(payloads)


def _selfplay_children():
    return [
        process
        for process in mp.active_children()
        if process.name.startswith("selfplay-search-")
    ]


def test_failure_path_leaves_no_selfplay_worker_processes():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("process cleanup regression requires fork")

    before = {process.pid for process in _selfplay_children()}
    engine = SelfPlayEngine(
        SelfPlayEngineConfig(
            workers=2,
            inference_batch_cap=2,
            inference_batch_wait_ms=0.0,
            device="cpu",
            process_start_method="fork",
            inference_request_timeout_s=5.0,
            worker_join_timeout_s=1.0,
        )
    )

    with pytest.raises(SelfPlayEngineError, match="intentional cleanup regression failure"):
        engine.run(
            ["game-0", "game-1", "game-2"],
            worker_play=_worker_failure,
            worker_context={},
            infer_batch=_infer,
        )

    leaked = [
        process
        for process in _selfplay_children()
        if process.pid not in before
    ]
    try:
        assert leaked == []
    finally:
        # Keep a failed regression test from poisoning the pytest interpreter.
        # This is scoped only to workers created by this test, never broad pkill.
        for process in leaked:
            if process.is_alive():
                process.kill()
            process.join(timeout=2.0)
