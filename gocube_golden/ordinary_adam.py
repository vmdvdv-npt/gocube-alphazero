"""Execution batching for ordinary CUDA Adam; serialized groups stay untouched."""
import inspect

import torch
from torch.optim.adam import adam


_VERIFIED_TORCH_VERSION = '2.4.1+cu124'
_VERIFIED_CUDA_VERSION = '12.4'
_VERIFIED_INIT_GROUP_PARAMETERS = (
    'self', 'group', 'params_with_grad', 'grads', 'exp_avgs', 'exp_avg_sqs',
    'max_exp_avg_sqs', 'state_steps',
)


def _batched_adam_runtime_status():
    """Return whether the private Adam fast path is verified, plus a reason."""
    actual_torch = str(torch.__version__)
    actual_cuda = torch.version.cuda
    if actual_torch != _VERIFIED_TORCH_VERSION:
        return False, (
            f'PyTorch {actual_torch} is not the byte-exact-verified '
            f'{_VERIFIED_TORCH_VERSION}'
        )
    if actual_cuda != _VERIFIED_CUDA_VERSION:
        return False, (
            f'CUDA runtime {actual_cuda} is not the byte-exact-verified '
            f'{_VERIFIED_CUDA_VERSION}'
        )
    try:
        parameters = tuple(inspect.signature(torch.optim.Adam._init_group).parameters)
    except (AttributeError, TypeError, ValueError) as error:
        return False, f'Adam._init_group contract is unavailable: {type(error).__name__}'
    if parameters != _VERIFIED_INIT_GROUP_PARAMETERS:
        return False, 'Adam._init_group signature differs from the byte-exact-verified contract'
    return True, None


def _batched_adam_runtime_supported():
    return _batched_adam_runtime_status()[0]


_BATCHED_ADAM_RUNTIME_SUPPORTED, _BATCHED_ADAM_RUNTIME_DISABLED_REASON = (
    _batched_adam_runtime_status()
)


class OrdinaryAdam(torch.optim.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (not _BATCHED_ADAM_RUNTIME_SUPPORTED
                and any(p.device.type == 'cuda'
                        for group in self.param_groups for p in group['params'])):
            raise RuntimeError(
                'B64 Adam batching cannot run on this CUDA runtime: '
                f'{_BATCHED_ADAM_RUNTIME_DISABLED_REASON}. '
                'Training is stopped rather than silently falling back to slower Adam. '
                'Re-run the exact parity gate and explicitly verify the new runtime '
                'before enabling production training.'
            )

    @torch.no_grad()
    def step(self, closure=None):
        groups = self.param_groups
        keys = ('betas', 'lr', 'weight_decay', 'eps', 'amsgrad', 'maximize',
                'foreach', 'capturable', 'differentiable', 'fused')
        eligible = (_BATCHED_ADAM_RUNTIME_SUPPORTED and bool(groups) and closure is None
                    and not hasattr(self, 'grad_scale') and not hasattr(self, 'found_inf'))
        if eligible:
            first = groups[0]
            parameters = [p for g in groups for p in g['params']]
            eligible = (first['foreach'] is not False and not first['fused'] and
                        not first['capturable'] and not first['differentiable'] and
                        len({id(p) for p in parameters}) == len(parameters) and
                        all(all(g[k] == first[k] for k in keys) and
                            all(p.device.type == 'cuda' and p.dtype == torch.float32 and
                                (p.grad is None or not p.grad.is_sparse) for p in g['params'])
                            for g in groups))
        if not eligible:
            return super().step(closure)
        self._cuda_graph_capture_health_check()
        params, grads, averages, variances, maxima, clocks = [], [], [], [], [], []
        for group in groups:
            self._init_group(group, params, grads, averages, variances, maxima, clocks)
        adam(params, grads, averages, variances, maxima, clocks,
             amsgrad=first['amsgrad'], has_complex=False,
             beta1=first['betas'][0], beta2=first['betas'][1], lr=first['lr'],
             weight_decay=first['weight_decay'], eps=first['eps'],
             maximize=first['maximize'], foreach=first['foreach'],
             capturable=False, differentiable=False, fused=first['fused'],
             grad_scale=None, found_inf=None)
