"""Optional one-generation replay-budget experiment; standard training/Arena workers."""
from __future__ import annotations
import json
import math
import random
import os
import shutil
import tempfile
from pathlib import Path
from ..provenance import file_sha256
from ..process_supervision import atomic_write_json
from ..artifact_graph import ArtifactRef, CheckpointRef, CheckpointNode


def parameters(raw, updates):
    if not isinstance(raw, dict) or set(raw) != {'snapshots', 'holdout_fraction', 'split_seed', 'arena_seed'}:
        raise ValueError('replay_budget requires snapshots, holdout_fraction, split_seed, arena_seed')
    snaps = raw['snapshots']
    if (not isinstance(snaps, list) or len(snaps) != 2
            or any(type(n) is not int or n <= 0 for n in snaps)
            or snaps != sorted(set(snaps)) or snaps[-1] != updates):
        raise ValueError('replay_budget requires two increasing snapshots ending at training updates')
    fraction = raw['holdout_fraction']
    if type(fraction) not in (float, int) or not math.isfinite(fraction) or not 0 < fraction < .5:
        raise ValueError('holdout_fraction must be between zero and .5')
    for key in ('split_seed', 'arena_seed'):
        if type(raw[key]) is not int or raw[key] < 0:
            raise ValueError(key + ' must be a nonnegative integer')
    return dict(raw)


def frozen_split(root, generation, count, config):
    ids = [f'{root.name}-g{generation:04d}-game-{i:04d}' for i in range(count)]
    held = sorted(random.Random(config['split_seed']).sample(ids, math.ceil(count*config['holdout_fraction'])))
    value = {'generation': generation, 'games': count, 'heldout_game_ids': held,
             'split_seed': config['split_seed'], 'holdout_fraction': config['holdout_fraction']}
    path = root / 'training' / 'replay-budget-split.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen holdout split changed')
    else:
        atomic_write_json(path, value)
    return set(held)


def read_snapshot(path):
    manifest = json.loads((path/'manifest.json').read_text())
    if set(manifest) != {'checkpoint.pt', 'checkpoint.metadata.json', 'evidence.json'}:
        raise ValueError('Unexpected snapshot manifest')
    for name, digest in manifest.items():
        if file_sha256(path/name) != digest:
            raise ValueError('Replay budget snapshot hash mismatch: ' + name)
    return json.loads((path/'evidence.json').read_text())


def save_snapshot(trainer, root, update, cfg, parent, buckets, metrics, evaluations):
    directory = root/'snapshots'/f'u{update}'
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise ValueError('Refusing to overwrite a replay-budget snapshot')
    temporary = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=directory.parent))
    try:
        trainer.save(temporary/'checkpoint.pt', config_hash=cfg.fingerprint,
                     parent=parent.ref.to_dict(), replay_buckets=buckets)
        evidence = {'update': update, 'ordinary_update': trainer.update,
                    'parent': parent.ref.to_dict(), 'config': cfg.fingerprint,
                    'metrics': metrics, 'evaluations': evaluations}
        atomic_write_json(temporary/'evidence.json', evidence)
        atomic_write_json(temporary/'manifest.json', {name: file_sha256(temporary/name)
            for name in ('checkpoint.pt', 'checkpoint.metadata.json', 'evidence.json')})
        for path in temporary.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        os.rename(temporary, directory)
        descriptor = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return directory


def publish_snapshots(root, generation, parent, effective, fresh_ref, config):
    """Register actual sibling checkpoints; source parent and generation are real."""
    refs = {}
    for update in config['snapshots']:
        directory = root/'snapshots'/f'u{update}'
        read_snapshot(directory)
        ref = CheckpointRef('torus9', root.name, f'M{generation}-u{update}', generation,
                            (directory/'checkpoint.pt').relative_to(root).as_posix(),
                            file_sha256(directory/'checkpoint.pt'))
        provenance = directory/'evidence.json'
        node = CheckpointNode(checkpoint=ref, genesis=False, parent=parent.ref,
             fresh_replay=fresh_ref, effective_config=effective,
             provenance=ArtifactRef(provenance.relative_to(root).as_posix(), file_sha256(provenance)))
        path = root/'metadata'/'checkpoints'/(ref.checkpoint_id+'.json')
        value = node.to_dict()
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Snapshot graph identity changed')
        if not path.exists(): atomic_write_json(path, value)
        refs['u'+str(update)] = ref.to_dict()
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for ref in refs.values():
        existing = manifest['checkpoint_hashes'].get(ref['path'])
        if existing is not None and existing != ref['sha256']:
            raise ValueError('Snapshot manifest identity changed')
        manifest['checkpoint_hashes'][ref['path']] = ref['sha256']
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(root/'training'/'replay-budget-snapshots.json', refs)
    return refs
