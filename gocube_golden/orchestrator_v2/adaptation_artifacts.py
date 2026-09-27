"""Register adaptation checkpoints in the existing V2 artifact graph.

This is artifact publication only, not another execution coordinator. Adaptation
replay ledgers retain their own schema and are not ordinary-generation replay.
"""
from pathlib import Path
import json
import torch

from gocube_golden.artifact_graph import ArtifactRef, CheckpointRef, CheckpointNode, EffectiveConfig, EffectiveConfigRef
from gocube_golden.artifact_resolver import ArtifactResolver, checkpoint_node_path
from gocube_golden.process_supervision import atomic_write_json
from gocube_golden.provenance import file_sha256, canonical_json
from gocube_golden.torus9_adaptation import BOOTSTRAP_SHA, FINGERPRINT
from gocube_golden.torus9_new_komi import publish_new_komi_bootstrap_graph


def _read(path):
    return json.loads(path.read_text())


def _immutable(path, value):
    text = canonical_json(value) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Immutable adaptation artifact drift: {path}')
    else:
        from gocube_golden.process_supervision import atomic_write_text
        atomic_write_text(path, text)


def publish_checkpoints(root):
    root = Path(root).resolve()
    cfg, state = _read(root / 'config.json'), _read(root / 'state.json')
    if cfg['target'] != FINGERPRINT or cfg['bootstrap_sha'] != BOOTSTRAP_SHA:
        raise ValueError('Unexpected adaptation scientific identity')
    runs_root = root.parents[2]
    baseline = publish_new_komi_bootstrap_graph(root.parent / 'new_komi')
    reference = CheckpointRef.from_dict(baseline['checkpoint'])
    if reference.sha256 != BOOTSTRAP_SHA:
        raise ValueError('Bootstrap identity drift')
    manifest_path = root / 'manifest.json'
    manifest = _read(manifest_path) if manifest_path.exists() else {
        'schema': 'gocube-adaptation-lineage-v1', 'topology': 'torus9',
        'lineage_id': root.name, 'status': 'ACTIVE',
        'parent_checkpoint': reference.to_dict(), 'checkpoint_hashes': {},
        'scientific_config_hash': cfg['fingerprint'], 'kind': 'staged-adaptation',
    }
    if (manifest.get('lineage_id') != root.name or manifest.get('topology') != 'torus9'
            or manifest.get('scientific_config_hash') != cfg['fingerprint']):
        raise ValueError('Adaptation lineage identity mismatch')
    shards = {item['sha']: item for item in state['shards']}
    for item in shards.values():
        path = Path(item['path']).resolve()
        path.relative_to(root)
        if file_sha256(path) != item['sha']:
            raise ValueError('Replay SHA mismatch')
    parent = reference
    previous_replay = set()
    refs = {}
    for path in sorted((root / 'checkpoints').glob('update-*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        meta = payload['metadata']
        update = int(payload['update'])
        if update > state['update']:
            continue
        if meta['config_hash'] != cfg['fingerprint'] or meta['target_fingerprint'] != FINGERPRINT:
            raise ValueError('Checkpoint scientific configuration mismatch')
        if path.stem != f'update-{update:04d}':
            raise ValueError('Checkpoint update/path mismatch')
        replay_ids = meta['replay_ids']
        if set(replay_ids) - shards.keys():
            raise ValueError('Checkpoint references missing replay')
        ref = CheckpointRef('torus9', root.name, path.stem, parent.generation + 1,
                            path.relative_to(root).as_posix(), file_sha256(path))
        effective = EffectiveConfig(topology='torus9',
            compatibility={'architecture_id': meta['architecture_id'], 'input_channels': 5,
                           'observation_shape': [5, 81], 'target_fingerprint': FINGERPRINT},
            self_play={'komi': 1.5, 'mcts_simulations': cfg['selfplay_simulations']},
            training={'adaptation_update': update, 'lr_policy': cfg['lr_policy']},
            replay={'schema': 'adaptation-replay-ledger-v1'},
            extensions={'scientific_config_hash': cfg['fingerprint'], 'kind': 'adaptation'})
        config_path = root / 'metadata' / 'effective-config-v2' / f'{effective.fingerprint}.json'
        _immutable(config_path, effective.to_dict())
        config_ref = EffectiveConfigRef(ArtifactRef(config_path.relative_to(root).as_posix(), file_sha256(config_path)), effective.fingerprint)
        replay_path = root / 'metadata' / 'adaptation-replay' / f'{path.stem}.json'
        _immutable(replay_path, {'schema': 'adaptation-replay-ledger-v1',
            'fresh_shards': [shards[x] for x in replay_ids if x not in previous_replay],
            'training_replay_ids': replay_ids, 'adaptation_update': update})
        replay_ref = ArtifactRef(replay_path.relative_to(root).as_posix(), file_sha256(replay_path))
        provenance_path = root / 'metadata' / 'provenance-v2' / f'{path.stem}.json'
        _immutable(provenance_path, {'schema': 'adaptation-checkpoint-provenance-v1',
            'checkpoint': ref.to_dict(), 'immediate_parent': parent.to_dict(),
            'fresh_replay': replay_ref.to_dict(), 'effective_config': config_ref.to_dict(),
            'scientific_config_hash': cfg['fingerprint'], 'model_hash': meta['model_hash'],
            'optimizer_preserved': True, 'adaptation_update': update,
            'training_replay_ids': replay_ids})
        node = CheckpointNode(ref, False, parent, replay_ref, config_ref,
            ArtifactRef(provenance_path.relative_to(root).as_posix(), file_sha256(provenance_path)))
        _immutable(checkpoint_node_path(root, ref), node.to_dict())
        manifest['checkpoint_hashes'][ref.path] = ref.sha256
        refs[ref.sha256] = ref.to_dict()
        parent, previous_replay = ref, set(replay_ids)
    if state['checkpoint_sha'] not in refs:
        raise ValueError('Current checkpoint absent from graph')
    atomic_write_json(manifest_path, manifest)
    resolver = ArtifactResolver(runs_root)
    resolver.checkpoint(reference)
    for ref in refs.values():
        resolver.checkpoint(ref)
    result = {'reference': reference.to_dict(), 'candidates': refs}
    atomic_write_json(root / 'metadata' / 'arena-checkpoint-refs.json', result)
    return result
