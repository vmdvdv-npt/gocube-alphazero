"""Five-channel komi adaptation: explicit targets, frozen Adam clocks and schedules.

The immutable inference bootstrap is never modified. Production process execution
is provided by the existing cooperative self-play and arena engines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import copy
import os
from pathlib import Path
import random
from typing import Any

import torch
from torch.nn import functional as F

from . import torus9_monolith as core
from .inference import BatchedPolicyWDLInferenceOwner
from .neural import model_hash
from .provenance import capture_code_identity, file_sha256, sha256_fingerprint
from .process_supervision import atomic_write_json
from .selfplay_engine import SharedMemorySpec, SelfPlayEngineConfig, run_cooperative_selfplay
from .torus9_m137_5ch import Torus9M137FiveChannelGraphNet, build_m137_five_channel_observation
from .torus9_new_komi_guard import assert_new_komi_training_model
from .torus9_pcr import PlayoutCapRandomization, resolve_search_mode
from .search import search_implementation_fingerprint, search_semantics
from .torus9_run_owned import RunOwnedTorus9SelfPlaySearchContract
from .torus9_selfplay import (Torus9SelfPlayAdapter, Torus9SelfPlayWorkerContext,
    _decode_torus9_shared_output)

CONTRACT = {
    'schema': 'torus9-komi15-adaptation-targets-v1', 'komi': 1.5,
    'observation_shape': [5, 81], 'score_normalization': 81.5,
    'policy': 'fresh-mcts-visits', 'wdl': 'terminal-side-to-move',
    'ownership': 'terminal-area-side-to-move', 'replay': 'fresh-only',
}
FINGERPRINT = sha256_fingerprint(CONTRACT)
BOOTSTRAP_SHA = 'sha256:2face09ec5326f39f53faaae78cd4435518849d0a70ce29b3538189843f19b55'
STAGES = ('collect', 'bridge', 'partial', 'full', 'stabilize')
ENDS = (0, 160, 480, 1120, 2400)


def save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class AdaptationModel(Torus9M137FiveChannelGraphNet):
    @property
    def architecture_config(self):
        return {**super().architecture_config, 'training_ready': True,
                'training_contract': FINGERPRINT}


class AdaptationSearchContract(RunOwnedTorus9SelfPlaySearchContract):
    def validate(self):
        super().validate()
        if self.komi != 1.5:
            raise ValueError('Adaptation self-play requires komi=1.5')

    @property
    def fingerprint(self):
        search = asdict(self)
        if not self.tree_reuse:
            search.pop('tree_reuse')
        return sha256_fingerprint({'target': FINGERPRINT, 'search': search})


@dataclass(frozen=True)
class PCRSearchContract(AdaptationSearchContract):
    pcr: PlayoutCapRandomization | None = None

    def validate(self):
        AdaptationSearchContract(**{
            f.name: getattr(self, f.name) for f in fields(AdaptationSearchContract)
        }).validate()
        if not isinstance(self.pcr, PlayoutCapRandomization):
            raise ValueError('PCR search requires resolved playout caps')


def write_observation(payload, destination):
    state, context = payload
    if state.komi != 1.5:
        raise ValueError('Self-play referee komi drift')
    destination.copy_(build_m137_five_channel_observation(state, legal_context=context))


class AdaptationSelfPlayAdapter(Torus9SelfPlayAdapter):
    def __init__(self, model, *, run_id, checkpoint, seed, device, simulations=200, search_mode='fixed', pcr=None, tree_reuse=False):
        caps = resolve_search_mode({'search_mode': search_mode, 'pcr': pcr})
        contract = (AdaptationSearchContract(simulations=simulations, komi=1.5, tree_reuse=tree_reuse) if caps is None
                    else PCRSearchContract(simulations=simulations, komi=1.5, pcr=caps, tree_reuse=tree_reuse))
        contract.validate()
        assert_new_komi_training_model(model)
        self.model = model
        self.device = torch.device(device)
        model.to(device).eval()
        self.owner = BatchedPolicyWDLInferenceOwner(
            model, device=self.device, expected_observation_shape=(5, 81),
            expected_policy_size=82, wdl_size=3, forward_policy_wdl_logits=model)
        self.worker_context = Torus9SelfPlayWorkerContext(
            run_id=run_id, model_checkpoint_label=Path(checkpoint).stem,
            checkpoint_artifact_hash=file_sha256(checkpoint), master_seed=seed,
            profile_fingerprint=FINGERPRINT, code_identity=capture_code_identity(),
            contract=contract,
            profile_id=core.TORUS9_CURRENT_PROFILE_ID, expected_model_hash=model_hash(model))
        self.shared_memory = SharedMemorySpec(
            observation_shape=(5, 81), policy_size=82, wdl_size=3,
            write_input=write_observation, decode_output=_decode_torus9_shared_output)


def selfplay(model, *, checkpoint, run_id, ids, seed, device='cuda', workers=16,
             simulations=200, progress=None, search_mode='fixed', pcr=None, tree_reuse=False):
    from gocube_golden.orchestrator_v2.execution_permit import require_engine_execution
    require_engine_execution('gocube_golden/torus9_adaptation.py:selfplay', action='selfplay', topology='torus9')
    adapter = AdaptationSelfPlayAdapter(model, run_id=run_id, checkpoint=checkpoint,
                                        seed=seed, device=device, simulations=simulations,
                                        search_mode=search_mode, pcr=pcr, tree_reuse=tree_reuse)
    result = run_cooperative_selfplay(
        ids, adapter=adapter,
        engine_config=SelfPlayEngineConfig(workers=workers, inference_batch_cap=64,
            inference_batch_wait_ms=1.0, device=device, process_start_method='spawn',
            lanes_per_worker=1, active_games_per_worker=4),
        active_games_per_worker=4, progress_callback=progress)
    result.telemetry.update(
        tree_reuse=tree_reuse,
        search_semantics=search_semantics(tree_reuse),
        search_implementation_fingerprint=search_implementation_fingerprint(tree_reuse),
    )
    return result


def game_targets(game) -> dict | None:
    """Reconstruct formal trajectory; never accept inherited replay targets."""
    game.validate()
    if game.technical_termination or not game.formal_result:
        raise ValueError('Technical game cannot enter adaptation replay')
    final = core.torus9_state_from_identity(game.start_state, expected_komi=1.5)
    for action in game.final_action_trace:
        final = core.apply_action(final, action).after
    if not final.is_terminal or core.result_from_terminal(final).winner.value != game.formal_result:
        raise ValueError('Terminal result/trace mismatch')
    fields = {k: [] for k in ('observation', 'pi', 'z', 'ownership', 'score', 'legal', 'visits')}
    for p in game.positions:
        if not p.training_eligible:
            continue
        state = core.torus9_state_from_identity(p.state, expected_komi=1.5)
        context = core.prepare_legal_actions(state)
        fields['observation'].append(build_m137_five_channel_observation(state, legal_context=context))
        fields['pi'].append(p.pi)
        fields['z'].append(core.torus9_z_target(game.formal_result, state.side_to_move))
        fields['ownership'].append(core.torus9_ownership_target(final, state.side_to_move))
        fields['score'].append(core.torus9_score_target(final, state.side_to_move) / 81.5)
        fields['legal'].append(context.action_mask)
        fields['visits'].append(p.root_visits)
    if not fields['observation']:
        return None  # Entirely cheap games remain in raw storage only.
    return {
        'game_id': game.game_id, 'actor_hash': game.model_hash,
        'actor_artifact': game.checkpoint_artifact_hash, 'contract': FINGERPRINT,
        'observation': torch.stack(fields['observation']),
        **{k: torch.tensor(v, dtype=(torch.long if k in ('ownership', 'visits') else
                                    torch.bool if k == 'legal' else torch.float32))
           for k, v in fields.items() if k != 'observation'},
    }


def validate_game(g):
    if g.get('contract') != FINGERPRINT:
        raise ValueError('Replay contract mismatch; parent history is forbidden')
    n = len(g['observation'])
    shapes = {'observation': (n, 5, 81), 'pi': (n, 82), 'z': (n, 3),
              'ownership': (n, 81), 'score': (n,), 'legal': (n, 82), 'visits': (n, 82)}
    for k, shape in shapes.items():
        if tuple(g[k].shape) != shape or not torch.isfinite(g[k]).all():
            raise ValueError('Replay tensor shape/nonfinite: ' + k)
    if n == 0 or (g['pi'] < 0).any() or (g['pi'][~g['legal']] != 0).any():
        raise ValueError('Invalid legal policy')
    if not torch.allclose(g['pi'].sum(1), torch.ones(n), atol=1e-5):
        raise ValueError('Policy normalization')
    if (g['visits'] < 0).any() or (g['visits'][~g['legal']] != 0).any():
        raise ValueError('Invalid root visits')
    visits = g['visits'].float()
    if not torch.allclose(g['pi'], visits / visits.sum(1, keepdim=True), atol=1e-5):
        raise ValueError('Policy does not match visits')
    if not ((g['z'] == 0) | (g['z'] == 1)).all() or not (g['z'].sum(1) == 1).all():
        raise ValueError('WDL target invalid')
    if ((g['ownership'] < 0) | (g['ownership'] > 2)).any():
        raise ValueError('Ownership target invalid')


def activation(name):
    if name.startswith(('value_head.', 'score_head.')):
        return 0
    if name.startswith(('blocks.6.', 'blocks.7.', 'output_norm.', 'point_policy.',
                        'pass_policy.', 'ownership_head.')):
        return 160
    return 480


def lr_for(name, update):
    if update <= activation(name):
        return 0.0
    if name.startswith(('value_head.', 'score_head.')):
        return 2e-6 + 8e-6 * min(1., update / 80.) if update <= 160 else 1e-5
    if update <= 480:
        return 1e-6 + 1e-6 * min(1., (update - 160) / 80.)
    if name == 'input_projection.bias':
        return 1e-6 + 4e-6 * min(1., (update - 480) / 160.)
    # Hold at 5e-6 after full warmup; later LR increases require a new reviewed config.
    return 2e-6 + 3e-6 * min(1., (update - 480) / 160.)


def shared(name):
    return name.startswith(('input_projection.', 'blocks.', 'output_norm.'))


class AdaptationTrainer:
    def __init__(self, bootstrap: Path, *, device='cpu', seed=2026092701):
        if file_sha256(bootstrap) != BOOTSTRAP_SHA:
            raise ValueError('Canonical bootstrap SHA mismatch')
        raw = torch.load(bootstrap, map_location='cpu', weights_only=False)
        self.model = AdaptationModel().to(device)
        self.model.load_state_dict(raw['model_state_dict'], strict=True)
        assert_new_komi_training_model(self.model)
        names = list(dict(self.model.named_parameters()))
        src = raw['optimizer_state_dict']
        if len(src['param_groups']) != 1 or len(src['param_groups'][0]['params']) != len(names):
            raise ValueError('Source Adam order/group mismatch')
        opts = {k: src['param_groups'][0][k] for k in ('betas', 'eps', 'weight_decay', 'amsgrad')}
        self.optimizer = torch.optim.Adam([
            {'params': [p], 'name': n, 'lr': 0.0} for n, p in self.model.named_parameters()], **opts)
        for (n, p), sid in zip(self.model.named_parameters(), src['param_groups'][0]['params']):
            old = src['state'][sid]
            self.optimizer.state[p] = {k: (v.clone().to(device if k != 'step' else 'cpu')
                                                   if torch.is_tensor(v) else copy.deepcopy(v))
                                       for k, v in old.items()}
            if n == 'input_projection.bias':
                for k in ('exp_avg', 'exp_avg_sq', 'step'):
                    self.optimizer.state[p][k].zero_()
        self.anchor = {n: p.detach().clone() for n, p in self.model.named_parameters() if shared(n)}
        self.update = 0
        self.seed = seed
        self.sp_coefficient = None
        self.lr_scale = 1.0
        self.validate_clocks()

    def validate_clocks(self):
        for n, p in self.model.named_parameters():
            s = self.optimizer.state[p]
            expected = (0 if n == 'input_projection.bias' else 15520) + max(0, self.update - activation(n))
            if int(s['step']) != expected:
                raise ValueError('Adam clock mismatch: ' + n)
            for k in ('exp_avg', 'exp_avg_sq'):
                if s[k].shape != p.shape or not torch.isfinite(s[k]).all():
                    raise ValueError('Adam state invalid: ' + n)
            if (s['exp_avg_sq'] < 0).any():
                raise ValueError('Negative second moment')

    def batch(self, games, update):
        rng = random.Random(self.seed + update)
        chosen = [games[rng.randrange(len(games))] for _ in range(64)]
        indices = []
        for i, g in enumerate(chosen):
            n = len(g['score'])
            phase = i % 3
            lo, hi = n * phase // 3, n * (phase + 1) // 3
            indices.append(rng.randrange(lo, max(lo + 1, hi)))
        device = next(self.model.parameters()).device
        return {k: torch.stack([g[k][j] for g, j in zip(chosen, indices)]).to(device)
                for k in ('observation', 'pi', 'z', 'ownership', 'score')}

    def losses(self, batch):
        from .training_profile import measured
        policy, value, ownership, score = measured('forward', lambda: self.model.forward_auxiliary(batch['observation']))
        losses = {
            'policy': measured('loss.policy', lambda: -(batch['pi'] * F.log_softmax(policy, 1)).sum(1).mean()),
            'wdl': measured('loss.wdl', lambda: -(batch['z'] * F.log_softmax(value, 1)).sum(1).mean()),
            'ownership': measured('loss.ownership', lambda: F.cross_entropy(ownership.reshape(-1, 3), batch['ownership'].reshape(-1))),
            'score': measured('loss.score', lambda: F.mse_loss(score, batch['score'])),
        }
        metrics = {'brier': measured('metric.brier', lambda: ((value.softmax(1) - batch['z']) ** 2).sum(1).mean()),
                   'score_mae_points': measured('metric.score_mae_points', lambda: ((score - batch['score']).abs() * 81.5).mean()),
                   'policy_entropy': measured('metric.policy_entropy', lambda: -(policy.softmax(1) * policy.log_softmax(1)).sum(1).mean())}
        return losses, metrics

    def step(self, games):
        from gocube_golden.orchestrator_v2.execution_permit import require_engine_execution
        require_engine_execution('gocube_golden/torus9_adaptation.py:step', action='training', topology='torus9')
        self.validate_clocks()
        u = self.update + 1
        self.model.train()
        for group in self.optimizer.param_groups:
            group['lr'] = lr_for(group['name'], u) * self.lr_scale
            group['params'][0].requires_grad_(group['lr'] > 0)
        self.optimizer.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        losses, _ = self.losses(self.batch(games, u))
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite training loss')
        total.backward()
        params = dict(self.model.named_parameters())
        size = sum(p.numel() for n, p in params.items() if shared(n))
        # Calibrate after shared weights have actually moved; never at theta=theta0.
        sp_grads = {n: 2 * (p.detach() - self.anchor[n]) / size for n, p in params.items()
                    if shared(n) and p.grad is not None}
        if self.sp_coefficient is None and u >= 200:
            a = sum(float((params[n].grad ** 2).sum()) for n in sp_grads) ** .5
            b = sum(float((g ** 2).sum()) for g in sp_grads.values()) ** .5
            if b > 1e-15:
                self.sp_coefficient = .05 * a / b
        coef = (self.sp_coefficient or 0.) * (0.5 if u > 1120 else 1.)
        for n, g in sp_grads.items():
            params[n].grad.add_(g, alpha=coef)
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.update = u
        self.validate_clocks()
        groups = {}
        for group in self.optimizer.param_groups:
            n, p = group['name'], group['params'][0]
            if not torch.isfinite(p).all():
                raise FloatingPointError('Nonfinite weights: ' + n)
            delta = float((p.detach() - before[n]).norm())
            if group['lr'] == 0 and delta != 0:
                raise ValueError('Frozen parameter moved')
            groups[n] = {'lr': group['lr'], 'step': int(self.optimizer.state[p]['step']),
                'grad_norm': 0. if p.grad is None else float(p.grad.norm()),
                'delta': delta, 'relative_delta': delta / max(float(before[n].norm()), 1e-8),
                'drift': float((p.detach() - self.anchor[n]).norm()) if n in self.anchor else None,
                'm_norm': float(self.optimizer.state[p]['exp_avg'].norm()),
                'v_max': float(self.optimizer.state[p]['exp_avg_sq'].max())}
        return {'update': u, 'losses': {k: float(v.detach()) for k, v in losses.items()},
                'grad_norm_before_clip': float(grad), 'clipped': float(grad) > 1.,
                'l2_sp_coefficient': coef, 'groups': groups}

    @torch.no_grad()
    def evaluate(self, games, batches=32):
        self.model.eval()
        totals = {}
        for i in range(batches):
            losses, metrics = self.losses(self.batch(games, -10000 - i))
            for k, v in {**losses, **metrics}.items():
                totals[k] = totals.get(k, 0.) + float(v) / batches
        return totals

    def save(self, path, *, replay_ids, config_hash):
        self.validate_clocks()
        meta = {'architecture_id': self.model.architecture_id,
                'architecture_config': self.model.architecture_config, 'observation_shape': [5, 81],
                'model_hash': model_hash(self.model), 'komi': 1.5, 'training_ready': True,
                'target_fingerprint': FINGERPRINT, 'bootstrap_sha': BOOTSTRAP_SHA,
                'bias_reset': 'moments-and-local-step-zero', 'adaptation_update': self.update,
                'config_hash': config_hash, 'replay_ids': replay_ids}
        save_torch(path, {'metadata': meta, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 'update': self.update,
            'seed': self.seed, 'sp_coefficient': self.sp_coefficient,
            'lr_scale': self.lr_scale,
            'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
        atomic_write_json(path.with_suffix('.metadata.json'), {**meta, 'artifact_sha256': file_sha256(path)})

    def restore(self, path, *, config_hash, replay_ids):
        raw = torch.load(path, map_location='cpu', weights_only=False)
        meta = raw['metadata']
        if (meta['config_hash'] != config_hash or meta['target_fingerprint'] != FINGERPRINT
                or meta['replay_ids'] != replay_ids or meta['bootstrap_sha'] != BOOTSTRAP_SHA):
            raise ValueError('Checkpoint/config/replay identity mismatch')
        self.model.load_state_dict(raw['model_state_dict'])
        self.optimizer.load_state_dict(raw['optimizer_state_dict'])
        if [g['name'] for g in self.optimizer.param_groups] != list(dict(self.model.named_parameters())):
            raise ValueError('Optimizer named group order mismatch')
        if model_hash(self.model) != meta['model_hash']:
            raise ValueError('Checkpoint weights hash mismatch')
        self.update, self.seed = raw['update'], raw['seed']
        self.sp_coefficient = raw['sp_coefficient']
        self.lr_scale = float(raw.get('lr_scale', 1.0))
        torch.set_rng_state(raw['torch_rng'])
        random.setstate(raw['python_rng'])
        if torch.cuda.is_available() and raw['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(raw['cuda_rng'])
        self.validate_clocks()
