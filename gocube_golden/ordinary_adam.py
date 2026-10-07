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


def _batched_adam_runtime_supported():
    """Fail closed if the private PyTorch Adam contract is not the verified one."""
    if str(torch.__version__) != _VERIFIED_TORCH_VERSION or torch.version.cuda != _VERIFIED_CUDA_VERSION:
        return False
    try:
        parameters = tuple(inspect.signature(torch.optim.Adam._init_group).parameters)
    except (AttributeError, TypeError, ValueError):
        return False
    return parameters == _VERIFIED_INIT_GROUP_PARAMETERS


_BATCHED_ADAM_RUNTIME_SUPPORTED = _batched_adam_runtime_supported()


class OrdinaryAdam(torch.optim.Adam):
    @torch.no_grad()
    def step(self, closure=None):
        groups = self.param_groups
        keys = ('betas', 'lr', 'weight_decay', 'eps', 'amsgrad', 'maximize',
                'foreach', 'capturable', 'differentiable', 'fused')
        # Only the byte-exact verified PyTorch runtime and established FP32 CUDA
        # foreach path are batched. Everything else uses PyTorch's own Adam step.
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
