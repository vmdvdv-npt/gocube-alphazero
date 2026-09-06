from queue import Empty
from time import monotonic

import torch

from alphazero.search_contract import SearchOutput


def collect_ready_worker_ids(ready_queue, worker_count, wait_ms):
    """Collect one or more workers that are waiting for neural inference.

    The first worker is allowed to wait for up to one second, matching the
    existing Coach polling behavior. After that first request arrives, a short
    coalescing window lets other workers join the same neural-network call.
    A zero wait still drains workers that are already queued.
    """
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if wait_ms < 0:
        raise ValueError("wait_ms must be non-negative")

    try:
        first_id = ready_queue.get(timeout=1)
    except Empty:
        return []

    if first_id < 0 or first_id >= worker_count:
        raise RuntimeError(f"invalid inference worker id {first_id}")

    worker_ids = [first_id]
    seen = {first_id}
    deadline = monotonic() + wait_ms / 1000.0

    while len(worker_ids) < worker_count:
        remaining = deadline - monotonic()
        try:
            if remaining > 0:
                worker_id = ready_queue.get(timeout=remaining)
            else:
                worker_id = ready_queue.get_nowait()
        except Empty:
            break

        if worker_id in seen:
            raise RuntimeError(f"worker {worker_id} queued inference twice before completion")
        if worker_id < 0 or worker_id >= worker_count:
            raise RuntimeError(f"invalid inference worker id {worker_id}")

        worker_ids.append(worker_id)
        seen.add(worker_id)

    return worker_ids


def process_coalesced_inference(
    nnet,
    worker_ids,
    input_tensors,
    policy_tensors,
    value_tensors,
    batch_ready,
    score_tensors=None,
    ownership_tensors=None,
):
    """Run one NN call for several ready workers and split all search heads.

    Legacy games keep the historical policy/value contract. Score-aware GoCube
    search supplies both ``score_tensors`` and ``ownership_tensors``; all four
    heads come from the same network forward pass and are copied back to the
    worker-owned shared-memory slices before the worker is released.

    Returns the number of positions evaluated in the combined neural batch.
    """
    if not worker_ids:
        return 0

    score_aware = score_tensors is not None or ownership_tensors is not None
    if score_aware and (score_tensors is None or ownership_tensors is None):
        raise ValueError("score-aware inference requires both score and ownership tensors")

    batch_sizes = [int(input_tensors[worker_id].size(0)) for worker_id in worker_ids]
    total_rows = sum(batch_sizes)
    if total_rows <= 0:
        raise RuntimeError("coalesced inference batch must contain at least one position")

    if len(worker_ids) == 1:
        combined = input_tensors[worker_ids[0]]
    else:
        combined = torch.cat([input_tensors[worker_id] for worker_id in worker_ids], dim=0)

    if score_aware:
        if not hasattr(nnet, "process_for_search"):
            raise RuntimeError("score-aware search requires process_for_search()")
        output = nnet.process_for_search(combined)
        if not isinstance(output, SearchOutput):
            raise RuntimeError("process_for_search() must return SearchOutput")
        policy = output.policy.detach().cpu()
        value = output.value.detach().cpu()
        score = output.score.detach().cpu() if output.score is not None else None
        ownership = output.ownership.detach().cpu() if output.ownership is not None else None
        if score is None or ownership is None:
            raise RuntimeError("score-aware network omitted score or ownership search head")
    else:
        policy, value = nnet.process(combined)
        policy = policy.detach().cpu()
        value = value.detach().cpu()
        score = None
        ownership = None

    returned = {
        "policy": int(policy.size(0)),
        "value": int(value.size(0)),
    }
    if score_aware:
        returned["score"] = int(score.size(0))
        returned["ownership"] = int(ownership.size(0))
    mismatched = {name: rows for name, rows in returned.items() if rows != total_rows}
    if mismatched:
        details = ", ".join(f"{name}={rows}" for name, rows in returned.items())
        raise RuntimeError(
            "network returned a different batch size than requested: "
            f"requested={total_rows}, {details}"
        )

    offset = 0
    for worker_id, batch_size in zip(worker_ids, batch_sizes):
        end = offset + batch_size
        policy_tensors[worker_id].copy_(policy[offset:end])
        value_tensors[worker_id].copy_(value[offset:end])
        if score_aware:
            score_tensors[worker_id].copy_(score[offset:end])
            ownership_tensors[worker_id].copy_(ownership[offset:end])
        batch_ready[worker_id].set()
        offset = end

    return total_rows
