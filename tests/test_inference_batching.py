from queue import Queue
from threading import Event

import pytest
import torch

from alphazero.inference_batching import (
    collect_routed_rows,
    collect_ready_worker_ids,
    process_coalesced_inference,
)
from alphazero.arena_bookkeeping import ArenaRoutingKey
from alphazero.search_contract import SearchOutput


class RecordingNet:
    def __init__(self):
        self.calls = []

    def process(self, batch):
        self.calls.append(batch.clone())
        rows = batch.size(0)
        policy = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
        value = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) + 100
        return policy, value


def test_collect_ready_workers_drains_already_queued_requests():
    ready_queue = Queue()
    ready_queue.put(2)
    ready_queue.put(0)
    ready_queue.put(3)

    worker_ids = collect_ready_worker_ids(ready_queue, worker_count=4, wait_ms=0)

    assert worker_ids == [2, 0, 3]


def test_collect_ready_workers_rejects_duplicate_request():
    ready_queue = Queue()
    ready_queue.put(1)
    ready_queue.put(1)

    with pytest.raises(RuntimeError, match="queued inference twice"):
        collect_ready_worker_ids(ready_queue, worker_count=2, wait_ms=0)


def test_process_coalesced_inference_uses_one_network_call_and_splits_outputs():
    nnet = RecordingNet()
    input_tensors = [
        torch.tensor([[10.0], [11.0]]),
        torch.tensor([[20.0], [21.0]]),
        torch.tensor([[30.0], [31.0]]),
    ]
    policy_tensors = [torch.zeros(2, 3) for _ in range(3)]
    value_tensors = [torch.zeros(2, 2) for _ in range(3)]
    batch_ready = [Event() for _ in range(3)]

    rows = process_coalesced_inference(
        nnet,
        worker_ids=[2, 0],
        input_tensors=input_tensors,
        policy_tensors=policy_tensors,
        value_tensors=value_tensors,
        batch_ready=batch_ready,
    )

    assert rows == 4
    assert len(nnet.calls) == 1
    assert torch.equal(
        nnet.calls[0],
        torch.tensor([[30.0], [31.0], [10.0], [11.0]]),
    )

    assert torch.equal(
        policy_tensors[2],
        torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
    )
    assert torch.equal(
        policy_tensors[0],
        torch.tensor([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]),
    )
    assert torch.equal(
        value_tensors[2],
        torch.tensor([[100.0, 101.0], [102.0, 103.0]]),
    )
    assert torch.equal(
        value_tensors[0],
        torch.tensor([[104.0, 105.0], [106.0, 107.0]]),
    )
    assert batch_ready[2].is_set()
    assert batch_ready[0].is_set()
    assert not batch_ready[1].is_set()


def test_process_coalesced_inference_rejects_wrong_network_batch_size():
    class BadNet:
        def process(self, batch):
            return torch.zeros(1, 3), torch.zeros(1, 2)

    input_tensors = [torch.zeros(2, 1)]
    policy_tensors = [torch.zeros(2, 3)]
    value_tensors = [torch.zeros(2, 2)]
    batch_ready = [Event()]

    with pytest.raises(RuntimeError, match="different batch size"):
        process_coalesced_inference(
            BadNet(),
            worker_ids=[0],
            input_tensors=input_tensors,
            policy_tensors=policy_tensors,
            value_tensors=value_tensors,
            batch_ready=batch_ready,
        )


def test_process_coalesced_inference_routes_all_four_heads_by_explicit_key():
    class FourHeadNet:
        def __init__(self):
            self.calls = []

        def process_for_search(self, batch):
            self.calls.append(batch.clone())
            rows = int(batch.size(0))
            values = torch.arange(rows, dtype=torch.float32).view(rows, 1)
            return SearchOutput(
                policy=values + 10,
                value=values + 20,
                score=values + 30,
                ownership=values.view(rows, 1, 1) + 40,
            )

    nnet = FourHeadNet()
    inputs = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0]])]
    policies = [torch.zeros(2, 1), torch.zeros(1, 1)]
    values = [torch.zeros(2, 1), torch.zeros(1, 1)]
    scores = [torch.zeros(2, 1), torch.zeros(1, 1)]
    ownership = [torch.zeros(2, 1, 1), torch.zeros(1, 1, 1)]
    ready = [Event(), Event()]
    responses = [Queue(), Queue()]
    keys = {
        0: [ArenaRoutingKey(0, 0, 0, 10), ArenaRoutingKey(0, 1, 0, 11)],
        1: [ArenaRoutingKey(1, 0, 0, 12)],
    }

    assert collect_routed_rows([1, 0], inputs, {1: keys[1], 0: keys[0]})[0][0] == keys[1][0]
    rows = process_coalesced_inference(
        nnet,
        worker_ids=[1, 0],
        input_tensors=inputs,
        policy_tensors=policies,
        value_tensors=values,
        batch_ready=ready,
        score_tensors=scores,
        ownership_tensors=ownership,
        routing_keys=keys,
        result_queues=responses,
    )

    assert rows == 3
    assert len(nnet.calls) == 1
    assert torch.equal(policies[1], torch.tensor([[10.0]]))
    assert torch.equal(policies[0], torch.tensor([[11.0], [12.0]]))
    assert torch.equal(values[1], torch.tensor([[20.0]]))
    assert torch.equal(scores[0], torch.tensor([[31.0], [32.0]]))
    assert torch.equal(ownership[1], torch.tensor([[[40.0]]]))
    assert responses[1].get_nowait()["routing_keys"] == tuple(keys[1])
    assert responses[0].get_nowait()["routing_keys"] == tuple(keys[0])
