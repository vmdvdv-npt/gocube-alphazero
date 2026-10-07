import copy

import pytest
import torch

from gocube_golden import ordinary_adam
from gocube_golden.b64_speedup_comparison import state_hash
from gocube_golden.ordinary_adam import OrdinaryAdam


def test_batched_runtime_guard_requires_verified_version_and_signature(monkeypatch):
    def compatible(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs,
                   max_exp_avg_sqs, state_steps):
        pass

    monkeypatch.setattr(torch, '__version__', ordinary_adam._VERIFIED_TORCH_VERSION)
    monkeypatch.setattr(torch.version, 'cuda', ordinary_adam._VERIFIED_CUDA_VERSION)
    monkeypatch.setattr(torch.optim.Adam, '_init_group', compatible)
    supported, reason = ordinary_adam._batched_adam_runtime_status()
    assert supported
    assert reason is None

    monkeypatch.setattr(torch, '__version__', '9.9.9+cu124')
    supported, reason = ordinary_adam._batched_adam_runtime_status()
    assert not supported
    assert 'PyTorch 9.9.9+cu124' in reason

    monkeypatch.setattr(torch, '__version__', ordinary_adam._VERIFIED_TORCH_VERSION)

    def incompatible(self, group):
        pass

    monkeypatch.setattr(torch.optim.Adam, '_init_group', incompatible)
    supported, reason = ordinary_adam._batched_adam_runtime_status()
    assert not supported
    assert 'signature differs' in reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_unverified_cuda_runtime_fails_before_training(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0], device='cuda'))
    monkeypatch.setattr(ordinary_adam, '_BATCHED_ADAM_RUNTIME_SUPPORTED', False)
    monkeypatch.setattr(
        ordinary_adam, '_BATCHED_ADAM_RUNTIME_DISABLED_REASON', 'synthetic runtime mismatch'
    )
    with pytest.raises(
        RuntimeError,
        match='B64 Adam batching cannot run.*synthetic runtime mismatch.*Training is stopped',
    ):
        OrdinaryAdam([{'params': [parameter], 'name': 'p', 'lr': 2.5e-5}])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_batched_adam_exact_bytes_and_legacy_roundtrip():
    generator = torch.Generator(device='cuda').manual_seed(919)
    params = [torch.nn.Parameter(torch.randn(shape, device='cuda', generator=generator))
              for shape in ((32, 17), (17,), (1,), (4, 81))]
    other = [torch.nn.Parameter(p.detach().clone()) for p in params]

    def groups(values):
        return [{'params': [p], 'name': str(i), 'lr': 2.5e-5} for i, p in enumerate(values)]

    reference = torch.optim.Adam(groups(params))
    candidate = OrdinaryAdam(groups(other))
    for i, p in enumerate(params):
        reference.state[p] = {'step': torch.tensor(float(100 + i * 123)),
                              'exp_avg': torch.randn(p.shape, device='cuda', generator=generator),
                              'exp_avg_sq': torch.rand(p.shape, device='cuda', generator=generator)}
    candidate.load_state_dict(copy.deepcopy(reference.state_dict()))
    for update in range(160):
        for p, q in zip(params, other):
            grad = torch.randn(p.shape, device='cuda', generator=generator)
            p.grad = grad
            q.grad = grad.clone()
        reference.step()
        candidate.step()
        assert state_hash(params) == state_hash(other), update
        assert state_hash(reference.state_dict()) == state_hash(candidate.state_dict()), update
    legacy = torch.optim.Adam(groups(other))
    legacy.load_state_dict(candidate.state_dict())
    assert state_hash(legacy.state_dict()) == state_hash(reference.state_dict())


@pytest.mark.parametrize('options', [{}, {'foreach': False}, {'amsgrad': True}, {'maximize': True}])
def test_cpu_fallback_exact(options):
    left = torch.nn.Parameter(torch.tensor([1., -2.]))
    right = torch.nn.Parameter(left.detach().clone())
    reference = torch.optim.Adam([left], **options)
    candidate = OrdinaryAdam([right], **options)
    for _ in range(8):
        left.grad = torch.tensor([.3, -.4])
        right.grad = left.grad.clone()
        reference.step()
        candidate.step()
        assert state_hash([left, reference.state_dict()]) == state_hash([right, candidate.state_dict()])
