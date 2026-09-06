from queue import Queue
from threading import Event

import pytest
import torch

from alphazero.inference_batching import (
    collect_ready_worker_ids,
    process_coalesced_inference,
)
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


class RecordingSearchNet:
    def __init__(self):
        self.calls = []

    def process_for_search(self, batch):
        self.calls.append(batch.clone())
        rows = batch.size(0)
        policy = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
        value = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) + 100
        score = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) + 200
        ownership = torch.arange(rows * 4 * 3, dtype=torch.float32).reshape(rows, 4, 3) + 300
        return SearchOutput(policy=policy, value=value, score=score, ownership=ownership)


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


def test_score_aware_inference_splits_policy_value_score_and_ownership_in_one_call():
    nnet = RecordingSearchNet()
    input_tensors = [torch.tensor([[10.0], [11.0]]), torch.tensor([[20.0], [21.0]])]
    policy_tensors = [torch.zeros(2, 3) for _ in range(2)]
    value_tensors = [torch.zeros(2, 2) for _ in range(2)]
    score_tensors = [torch.zeros(2, 1) for _ in range(2)]
    ownership_tensors = [torch.zeros(2, 4, 3) for _ in range(2)]
    batch_ready = [Event() for _ in range(2)]

    rows = process_coalesced_inference(
        nnet,
        worker_ids=[1, 0],
        input_tensors=input_tensors,
        policy_tensors=policy_tensors,
        value_tensors=value_tensors,
        batch_ready=batch_ready,
        score_tensors=score_tensors,
        ownership_tensors=ownership_tensors,
    )

    assert rows == 4
    assert len(nnet.calls) == 1
    assert torch.equal(score_tensors[1], torch.tensor([[200.0], [201.0]]))
    assert torch.equal(score_tensors[0], torch.tensor([[202.0], [203.0]]))
    assert torch.equal(ownership_tensors[1], nnet.process_for_search(nnet.calls[0]).ownership[:2])
    assert batch_ready[0].is_set() and batch_ready[1].is_set()


def test_score_aware_inference_works_for_single_worker_without_extra_forward():
    nnet = RecordingSearchNet()
    input_tensors = [torch.tensor([[10.0], [11.0]])]
    policy_tensors = [torch.zeros(2, 3)]
    value_tensors = [torch.zeros(2, 2)]
    score_tensors = [torch.zeros(2, 1)]
    ownership_tensors = [torch.zeros(2, 4, 3)]
    batch_ready = [Event()]

    rows = process_coalesced_inference(
        nnet, [0], input_tensors, policy_tensors, value_tensors, batch_ready,
        score_tensors=score_tensors, ownership_tensors=ownership_tensors,
    )
    assert rows == 2
    assert len(nnet.calls) == 1
    assert score_tensors[0].shape == (2, 1)
    assert ownership_tensors[0].shape == (2, 4, 3)


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


def test_score_aware_inference_fails_fast_when_any_head_has_wrong_batch_size():
    class BadSearchNet:
        def process_for_search(self, batch):
            rows = batch.size(0)
            return SearchOutput(
                policy=torch.zeros(rows, 3),
                value=torch.zeros(rows, 2),
                score=torch.zeros(rows, 1),
                ownership=torch.zeros(rows - 1, 4, 3),
            )

    inputs = [torch.zeros(2, 1)]
    policy = [torch.zeros(2, 3)]
    value = [torch.zeros(2, 2)]
    score = [torch.zeros(2, 1)]
    ownership = [torch.zeros(2, 4, 3)]
    ready = [Event()]
    with pytest.raises(RuntimeError, match="different batch size"):
        process_coalesced_inference(
            BadSearchNet(), [0], inputs, policy, value, ready,
            score_tensors=score, ownership_tensors=ownership,
        )
