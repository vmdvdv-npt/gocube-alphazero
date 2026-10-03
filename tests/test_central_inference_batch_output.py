"""Batch materialization, shared-slot mapping and fail-closed regressions."""
from __future__ import annotations

from queue import Queue
import time

import pytest
import torch

from gocube_golden.inference import BatchedPolicyWDLInferenceOwner
from selfplay_engine import SharedMemorySpec, _CentralInference, _Request


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


def _logits(batch):
    # Row identity affects both heads, making cross-request swaps observable.
    values = batch[:, :1]
    return values * torch.tensor([1., -1., .25, .5], device=batch.device), values * torch.tensor([.5, -1., 1.], device=batch.device)


def _owner(device, forward=_logits):
    return BatchedPolicyWDLInferenceOwner(
        object(), device=device, expected_observation_shape=(1,),
        expected_policy_size=4, forward_policy_wdl_logits=forward)


def _broker(device, callback):
    inputs = [torch.arange(w * 4, w * 4 + 4, dtype=torch.float32).reshape(4, 1) / 32 for w in range(16)]
    policies = [torch.full((4, 4), -99.).share_memory_() for _ in inputs]
    wdls = [torch.full((4, 3), -99.).share_memory_() for _ in inputs]
    responses = [[Queue()] for _ in inputs]
    broker = _CentralInference(Queue(), responses, None, batch_cap=64, wait_ms=1., device=device,
        shared_spec=SharedMemorySpec((1,), 4, 3, lambda *_: None, lambda *_: None),
        infer_shared_batch=callback, shared_inputs=inputs, shared_policy=policies, shared_wdl=wdls)
    return broker


def _requests(rows):
    if rows == 1:
        return [_Request(7, 0, 123, slot_ids=(2,), broker_received_at=time.perf_counter())]
    # Deliberately shuffled workers and non-contiguous slots within envelopes.
    return [_Request(w, 0, 100 + w, slot_ids=(3, 1, 0, 2), broker_received_at=time.perf_counter())
            for w in reversed(range(16))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rows', [1, 64])
def test_batch_values_match_previous_softmax_outputs_and_share_cpu_materialization(device, rows):
    observations = torch.linspace(-1, 1, rows).reshape(rows, 1)
    raw_policy, raw_wdl = _logits(observations.to(device))
    # Previous owner returned the GPU softmax heads, subsequently copied to CPU.
    expected_policy = raw_policy.softmax(1).cpu()
    expected_wdl = raw_wdl.softmax(1).cpu()
    owner = _owner(device)
    result = owner.evaluate_shared_batch(observations)
    torch.testing.assert_close(result.policy, expected_policy, rtol=0, atol=0)
    torch.testing.assert_close(result.wdl, expected_wdl, rtol=0, atol=0)
    assert result.policy.device.type == result.wdl.device.type == 'cpu'
    assert result.policy.untyped_storage().data_ptr() == result.wdl.untyped_storage().data_ptr()
    assert result.h2d_started_at <= result.h2d_finished_at <= result.forward_started_at <= result.forward_finished_at
    # The next forward must not overwrite a batch still being consumed.
    owner.evaluate_shared_batch(observations + 1)
    torch.testing.assert_close(result.policy, expected_policy, rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rows', [1, 64])
def test_dispatch_routes_batch_rows_to_exact_worker_slots_and_request_ids(device, rows):
    broker = _broker(device, _owner(device).evaluate_shared_batch)
    requests = _requests(rows)
    broker._dispatch(requests)
    for request in requests:
        response = broker.responses[request.worker_id][0].get_nowait()
        assert response.request_id == request.request_id
        assert response.slot_ids == request.slot_ids
        assert response.error is None
        for slot in request.slot_ids:
            policy, wdl = _logits(broker.shared_inputs[request.worker_id][slot:slot + 1].to(device))
            torch.testing.assert_close(broker.shared_policy[request.worker_id][slot], policy.softmax(1).cpu()[0], rtol=0, atol=0)
            torch.testing.assert_close(broker.shared_wdl[request.worker_id][slot], wdl.softmax(1).cpu()[0], rtol=0, atol=0)
    assert sum(broker._rows) == rows
    if rows == 1:
        assert broker.shared_policy[0].eq(-99).all()
        assert broker.shared_policy[7][0].eq(-99).all()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('head', [0, 1])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_head_fails_before_shared_slots_or_success_response(device, head, bad):
    def forward(batch):
        logits = list(_logits(batch))
        logits[head][-1, :] = bad
        return tuple(logits)
    broker = _broker(device, _owner(device, forward).evaluate_shared_batch)
    with pytest.raises(ValueError, match='invalid probabilities'):
        broker._dispatch(_requests(64))
    assert all(p.eq(-99).all() for p in broker.shared_policy)
    assert all(w.eq(-99).all() for w in broker.shared_wdl)
    assert all(q[0].empty() for q in broker.responses)


@pytest.mark.parametrize('head', [0, 1])
def test_invalid_probability_fails_closed(monkeypatch, head):
    real_softmax = torch.softmax
    calls = 0
    def invalid_softmax(*args, **kwargs):
        nonlocal calls
        output = real_softmax(*args, **kwargs)
        if calls == head:
            output[-1, 0] = -.01
        calls += 1
        return output
    monkeypatch.setattr(torch, 'softmax', invalid_softmax)
    with pytest.raises(ValueError, match='invalid probabilities'):
        _owner('cpu').evaluate_shared_batch(torch.ones(64, 1))


@pytest.mark.parametrize('head', [0, 1])
def test_shape_drift_still_fails_closed(head):
    def forward(batch):
        logits = list(_logits(batch))
        logits[head] = logits[head][:, :-1]
        return tuple(logits)
    with pytest.raises(ValueError, match='head shape drift'):
        _owner('cpu', forward).evaluate_shared_batch(torch.ones(1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('rows', [1, 64])
def test_cuda_dispatch_materializes_once_without_per_row_d2h(rows):
    from torch.utils._python_dispatch import TorchDispatchMode

    class TransportCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.d2h = 0
            self.cuda_scalars = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten._to_copy.default:
                if args[0].device.type == 'cuda' and result.device.type == 'cpu':
                    self.d2h += 1
            elif func == torch.ops.aten.copy_.default:
                if args[0].device.type == 'cpu' and args[1].device.type == 'cuda':
                    self.d2h += 1
            elif func == torch.ops.aten._local_scalar_dense.default:
                self.cuda_scalars += int(args[0].device.type == 'cuda')
            return result

    broker = _broker('cuda', _owner('cuda').evaluate_shared_batch)
    with TransportCounter() as counter:
        broker._dispatch(_requests(rows))
    assert counter.d2h == 1
    # Preserve the original two scalar GPU validation checks per batch.
    assert counter.cuda_scalars == 2


def test_dispatcher_reports_invalid_batch_as_technical_failure_to_every_envelope():
    def invalid(batch):
        policy, wdl = _logits(batch)
        wdl[-1].fill_(float('nan'))
        return policy, wdl
    broker = _broker('cpu', _owner('cpu', invalid).evaluate_shared_batch)
    requests = _requests(64)
    broker._pending.extend(requests)
    broker._dispatcher_stop = True
    broker._dispatch_loop()
    assert 'invalid probabilities' in broker.fatal
    for request in requests:
        response = broker.responses[request.worker_id][0].get_nowait()
        assert response.request_id == request.request_id
        assert response.slot_ids == request.slot_ids
        assert 'invalid probabilities' in response.error
    assert all(p.eq(-99).all() for p in broker.shared_policy)
    assert all(w.eq(-99).all() for w in broker.shared_wdl)
