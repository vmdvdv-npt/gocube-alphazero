"""Immutable historical-actor KL evidence and frequency-only replay sampling."""
from __future__ import annotations

import bisect
import math
from pathlib import Path

import torch

from .neural import model_hash
from .provenance import file_sha256, sha256_fingerprint
from .torus9_adaptation import AdaptationModel, FINGERPRINT, validate_game

ALGORITHM = 'historical-actor-legal-kl-frequency-v1'


def sampling_setting(value=None):
    value = {'mode': 'uniform'} if value is None else dict(value)
    if set(value) - {'mode', 'weight'} or value.get('mode') not in ('uniform', 'policy_surprise'):
        raise ValueError('Invalid replay sampling setting')
    if value['mode'] == 'uniform':
        if 'weight' in value:
            raise ValueError('uniform sampling has no weight')
    else:
        w = value.get('weight')
        if type(w) not in (int, float) or not math.isfinite(w) or not 0 <= w <= 1:
            raise ValueError('policy_surprise weight must be finite in [0,1]')
    return value


def frequency_weights(surprises, weight):
    sampling_setting({'mode': 'policy_surprise', 'weight': weight})
    s = torch.as_tensor(surprises, dtype=torch.float64)
    if s.ndim != 1 or not len(s) or not torch.isfinite(s).all() or (s < 0).any():
        raise ValueError('Invalid policy surprises')
    total = float(s.sum())
    f = torch.ones_like(s) if total == 0 else (1 - weight) + weight * len(s) * s / total
    if not math.isclose(float(f.sum()), len(s), rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError('Per-game frequency mass drift')
    return f


def policy_kl(logits, pi, legal):
    logits, pi = logits.double(), pi.double()
    if (logits.shape != pi.shape or legal.shape != pi.shape or
            not torch.isfinite(logits).all() or not torch.isfinite(pi).all() or
            (pi < 0).any() or (pi[~legal] != 0).any() or not legal.any(1).all() or
            not torch.allclose(pi.sum(1), torch.ones(len(pi), device=pi.device, dtype=pi.dtype), atol=1e-5)):
        raise ValueError('Invalid historical policy input')
    # Renormalize the float32 target's rounding error for a true nonnegative KL.
    pi = pi / pi.sum(1, keepdim=True)
    log_prior = logits.masked_fill(~legal, -torch.inf).log_softmax(1)
    positive = pi > 0
    terms = torch.zeros_like(pi)
    terms[positive] = pi[positive] * (pi[positive].log() - log_prior[positive])
    result = terms.sum(1)
    if not torch.isfinite(result).all() or (result < -1e-10).any():
        raise ValueError('Nonfinite or negative historical policy KL')
    return result.clamp_min(0).cpu()


def unique_shards(schedule):
    found = {}
    for row in schedule:
        for bucket in row['buckets']:
            for ref in bucket['shards']:
                key = ref['sha']
                entry = {'path': ref['path'], 'sha': key, 'generation': bucket['generation']}
                if key in found and found[key] != entry:
                    raise ValueError('Ambiguous replay shard identity')
                found[key] = entry
    return list(found.values())


def read_shard(ref):
    if file_sha256(ref['path']) != ref['sha']:
        raise ValueError('Policy surprise replay SHA mismatch')
    raw = torch.load(ref['path'], map_location='cpu', weights_only=False)
    if raw['contract'] != FINGERPRINT:
        raise ValueError('Policy surprise replay target mismatch')
    games = [g for g in raw['games'] if g.get('split') == 'train']
    for g in games:
        validate_game(g)
        if g['actor_hash'] != raw['actor_hash']:
            raise ValueError('Policy surprise actor mismatch')
    return games


def resolve_spec(schedule, *, parent, resolver, weight):
    """Resolve exact actor artifact identities through the canonical parent graph."""
    shards = unique_shards(schedule)
    needed = {}
    for ref in shards:
        games = read_shard(ref)
        ref['actor_hashes'] = sorted({g['actor_hash'] for g in games})
        for game in games:
            actor, sha = game['actor_hash'], game.get('actor_artifact')
            if not sha or (actor in needed and needed[actor] != sha):
                raise ValueError('Ambiguous historical source model hash: ' + actor)
            needed[actor] = sha
    sources, visited = {}, set()
    node = resolver.checkpoint(schedule[-1]['source_checkpoint'])
    while node is not None and len(sources) < len(needed):
        if node.ref.sha256 in visited:
            raise ValueError('Historical actor checkpoint cycle')
        visited.add(node.ref.sha256)
        matching = [actor for actor, sha in needed.items() if sha == node.ref.sha256]
        if matching:
            raw = torch.load(node.path, map_location='cpu', weights_only=False)
            for actor in matching:
                if raw['metadata']['model_hash'] != actor:
                    raise ValueError('Historical actor model hash mismatch')
                sources[actor] = {'checkpoint': node.ref.to_dict(), 'path': str(node.path)}
        node = resolver.parent(node) if len(sources) < len(needed) else None
    if set(sources) != set(needed):
        raise ValueError('Unresolved historical source model hash: ' + ', '.join(sorted(set(needed)-set(sources))))
    body = {'algorithm': ALGORITHM, 'implementation_sha256': file_sha256(__file__),
            'weight': weight, 'shards': shards, 'sources': sources,
            'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda,
            'inference_device': 'cuda', 'inference_batch_size': 256,
            'prior': 'historical-actor-raw-logits-legal-log-softmax-float64',
            'row_identity': 'shard-sha/game-id/learner-row-index'}
    return {**body, 'fingerprint': sha256_fingerprint(body)}


def validate_spec(spec):
    body = {k: v for k, v in spec.items() if k != 'fingerprint'}
    if (spec['algorithm'] != ALGORITHM or spec['implementation_sha256'] != file_sha256(__file__) or
            sha256_fingerprint(body) != spec['fingerprint']):
        raise ValueError('Policy surprise cache provenance mismatch')


def build_cache(spec, root, *, device):
    validate_spec(spec)
    if (spec.get('torch_version', torch.__version__) != torch.__version__ or
            spec.get('torch_cuda_version', torch.version.cuda) != torch.version.cuda or
            spec.get('inference_device', device) != device):
        raise ValueError('Historical policy inference runtime provenance mismatch')
    rows, seen = {}, set()
    torch.set_num_threads(1)
    # One model at a time; shard grouping gives bounded memory and batched inference.
    for actor, source in spec['sources'].items():
        if file_sha256(source['path']) != source['checkpoint']['sha256']:
            raise ValueError('Historical actor checkpoint SHA mismatch')
        raw = torch.load(source['path'], map_location='cpu', weights_only=False)
        model = AdaptationModel().to(device)
        model.load_state_dict(raw['model_state_dict'], strict=True)
        if model_hash(model) != actor or raw['metadata']['model_hash'] != actor:
            raise ValueError('Historical actor network hash mismatch')
        model.eval()
        with torch.inference_mode():
            for ref in spec['shards']:
                if actor not in ref['actor_hashes']:
                    continue
                games = read_shard(ref)
                selected = [g for g in games if g['actor_hash'] == actor]
                if not selected:
                    continue
                observation = torch.cat([g['observation'] for g in selected])
                targets = torch.cat([g['pi'] for g in selected])
                masks = torch.cat([g['legal'] for g in selected])
                parts = []
                for offset in range(0, len(observation), 256):
                    logits, _ = model(observation[offset:offset+256].to(device))
                    parts.append(policy_kl(logits, targets[offset:offset+256].to(device),
                                           masks[offset:offset+256].to(device)))
                all_surprises = torch.cat(parts)
                offset = 0
                for game in selected:
                    if game['actor_artifact'] != source['checkpoint']['sha256']:
                        raise ValueError('Historical actor artifact identity mismatch')
                    identity = game['game_id']
                    if identity in seen:
                        raise ValueError('Duplicate policy surprise replay game')
                    seen.add(identity)
                    count = len(game['score'])
                    surprises = all_surprises[offset:offset+count].clone(); offset += count
                    rows[identity] = {'shard_sha': ref['sha'], 'game_id': identity,
                        'learner_row_indices': torch.arange(count),
                        'source_generation': ref['generation'], 'source_model_hash': actor,
                        'checkpoint': source['checkpoint'], 'surprise': surprises,
                        'frequency': frequency_weights(surprises, spec['weight'])}
        del model, raw
    payload = {'spec': spec, 'games': rows}
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    # Content-addressed publication: never replace existing immutable evidence.
    import os, tempfile
    fd, name = tempfile.mkstemp(prefix='.surprise-', suffix='.pt', dir=root)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        sha = file_sha256(temporary)
        path = root / (sha.split(':')[1] + '.pt')
        if path.exists():
            if file_sha256(path) != sha:
                raise ValueError('Existing policy surprise cache SHA mismatch')
        else:
            os.link(temporary, path)
        return {'path': str(path), 'sha256': sha, 'fingerprint': spec['fingerprint']}
    finally:
        temporary.unlink(missing_ok=True)


def load_cache(ref, spec):
    validate_spec(spec)
    if ref['fingerprint'] != spec['fingerprint'] or file_sha256(ref['path']) != ref['sha256']:
        raise ValueError('Policy surprise cache SHA/provenance mismatch')
    raw = torch.load(ref['path'], map_location='cpu', weights_only=False)
    if raw['spec'] != spec:
        raise ValueError('Policy surprise cache specification mismatch')
    return raw['games']


class SamplingTelemetry:
    def __init__(self, games, evidence, cache_ref, weight, mode="policy_surprise"):
        sampling_setting({"mode": mode, **({"weight": weight} if mode == "policy_surprise" else {})})
        self.evidence, self.cache_ref = evidence, cache_ref
        self.cumulative, self.offsets, self.keys = [], [], []
        all_s, all_f, total, count = [], [], 0., 0
        for game in games:
            key = game['game_id']; row = evidence[key]
            s, f = row['surprise'], row['frequency']
            if len(s) != len(game['score']) or row['source_model_hash'] != game['actor_hash']:
                raise ValueError('Replay/cache row mismatch')
            if not torch.equal(frequency_weights(s, weight), f):
                raise ValueError('Replay/cache frequency mismatch')
            self.offsets.append(count); count += len(s); self.keys.append(key)
            for value in f.tolist():
                total += value; self.cumulative.append(total)
            all_s.append(s); all_f.append(f if mode == 'policy_surprise' else torch.ones_like(f))
        self.s = torch.cat(all_s); self.f = torch.cat(all_f)
        # Exactly ceil(10% * N) rows; stable replay order breaks ties.
        self.top = set(torch.argsort(self.s, descending=True, stable=True)[:math.ceil(len(self.s)*.1)].tolist())
        self.total = total
        self.draws = self.top_draws = 0; self.surprise_sum = 0.
        self.unique = set(); self.generations = {}

    def draw(self, rng, games):
        flat = bisect.bisect_right(self.cumulative, rng.random() * self.total)
        game_index = bisect.bisect_right(games.cumulative, flat)
        return games[game_index], flat - (games.cumulative[game_index-1] if game_index else 0)

    def record(self, indices):
        for game, index in indices:
            key = game['game_id']; row = self.evidence[key]
            flat = self.offsets[self.game_indices[key]] + index
            self.draws += 1; self.top_draws += int(flat in self.top)
            self.surprise_sum += float(row['surprise'][index]); self.unique.add((key, index))
            generation = str(row['source_generation'])
            self.generations[generation] = self.generations.get(generation, 0) + 1

    @property
    def game_indices(self):
        if not hasattr(self, '_game_indices'):
            self._game_indices = dict(zip(self.keys, range(len(self.keys))))
        return self._game_indices

    def report(self):
        def stats(values):
            return dict(zip(('mean', 'median', 'p95', 'max'),
                [float(values.mean()), float(values.quantile(.5)), float(values.quantile(.95)), float(values.max())]))
        return {'policy_surprise': stats(self.s), 'frequency_weight': stats(self.f),
                'sampled_mean_surprise': self.surprise_sum / self.draws,
                'replay_mean_surprise': float(self.s.mean()), 'top_10_percent_sampling_share': self.top_draws / self.draws,
                'unique_sampled_rows': len(self.unique), 'sample_draws': self.draws,
                'source_generation_samples': self.generations, 'cache': self.cache_ref}
