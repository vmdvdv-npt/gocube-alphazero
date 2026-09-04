from queue import Empty
from time import monotonic

import torch


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
):
    """Run one NN call for several ready self-play workers and split results.

    Each worker owns a fixed shared-memory input/output tensor. Concatenating
    ready inputs turns several tiny GPU launches into one larger launch. The
    outputs are copied back to CPU once, split by worker, and each worker is
    released only after its slices are ready.

    Returns the number of positions evaluated in the combined neural batch.
    """
    if not worker_ids:
        return 0

    batch_sizes = [int(input_tensors[worker_id].size(0)) for worker_id in worker_ids]
    total_rows = sum(batch_sizes)
    if total_rows <= 0:
        raise RuntimeError("coalesced inference batch must contain at least one position")

    if len(worker_ids) == 1:
        combined = input_tensors[worker_ids[0]]
    else:
        combined = torch.cat([input_tensors[worker_id] for worker_id in worker_ids], dim=0)

    policy, value = nnet.process(combined)
    policy = policy.detach().cpu()
    value = value.detach().cpu()

    if policy.size(0) != total_rows or value.size(0) != total_rows:
        raise RuntimeError(
            "network returned a different batch size than requested: "
            f"requested={total_rows}, policy={policy.size(0)}, value={value.size(0)}"
        )

    offset = 0
    for worker_id, batch_size in zip(worker_ids, batch_sizes):
        end = offset + batch_size
        policy_tensors[worker_id].copy_(policy[offset:end])
        value_tensors[worker_id].copy_(value[offset:end])
        batch_ready[worker_id].set()
        offset = end

    return total_rows
