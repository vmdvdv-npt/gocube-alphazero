"""Register an immutable experimental checkpoint as an Arena-only genesis.

This is an evaluation input, not a training parent. The source is never modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from gocube_golden.artifact_graph import ArtifactRef, CheckpointNode, CheckpointRef, EffectiveConfig, EffectiveConfigRef
from gocube_golden.provenance import file_sha256
from gocube_golden.run_storage import ensure_lineage_layout
from tools.arena_profiles.weightspace import load_weightspace


def register(source: Path, runs_root: Path, lineage: str, revision: str):
    source = source.resolve()
    meta_path = source.with_suffix('.metadata.json')
    meta = json.loads(meta_path.read_text())
    load_weightspace(source, meta)
    root = runs_root.resolve() / 'torus9' / 'active' / lineage
    if root.exists() or (runs_root.resolve() / 'torus9' / 'archive' / lineage).exists():
        raise FileExistsError('Lineage already exists')
    if not lineage or lineage in ('.', '..') or '/' in lineage or '\\' in lineage:
        raise ValueError('Invalid lineage')
    config = EffectiveConfig(topology='torus9', compatibility={
        'topology': 'torus9', 'input_channels': 5,
        'architecture_id': meta['architecture']['architecture_id'], 'evaluation_only': True,
    }, extensions={'weightspace_source_state_sha256': meta['state_sha256']})
    sha = file_sha256(source)
    manifest = {'lineage_id': lineage, 'topology': 'torus9', 'status': 'ACTIVE',
                'parent_checkpoint': None, 'git_commit': revision,
                'config_fingerprint': config.fingerprint,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'checkpoint_hashes': {'checkpoints/v001.pt': sha},
                'purpose': 'evaluation-only experimental weight-space model'}
    ensure_lineage_layout(root, manifest=manifest, include_data=False)
    checkpoint = root / 'checkpoints/v001.pt'
    shutil.copyfile(source, checkpoint)
    shutil.copyfile(meta_path, checkpoint.with_suffix('.metadata.json'))
    if file_sha256(checkpoint) != sha:
        raise ValueError('Copied checkpoint SHA mismatch')
    config_path = root / 'metadata/effective-config-v2/config.json'
    provenance_path = root / 'metadata/provenance-v2/v001.json'
    node_path = root / 'metadata/checkpoints/v001.json'
    for path in (config_path, provenance_path, node_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config.canonical_json + '\n')
    provenance_path.write_text(json.dumps({'schema': 'weightspace-arena-import-v1',
        'source': str(source), 'source_sha256': sha, 'source_metadata_sha256': file_sha256(meta_path),
        'state_sha256': meta['state_sha256'], 'revision': revision}, sort_keys=True) + '\n')
    ref = CheckpointRef('torus9', lineage, 'v001', 0, 'checkpoints/v001.pt', sha)
    effective = EffectiveConfigRef(ArtifactRef(str(config_path.relative_to(root)), file_sha256(config_path)), config.fingerprint)
    node = CheckpointNode(ref, True, None, None, effective,
                          ArtifactRef(str(provenance_path.relative_to(root)), file_sha256(provenance_path)))
    node_path.write_text(json.dumps(node.to_dict(), sort_keys=True) + '\n')
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--runs-root', required=True, type=Path)
    parser.add_argument('--lineage', required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    print(json.dumps(register(args.checkpoint, args.runs_root, args.lineage, revision).to_dict(), indent=2))


if __name__ == '__main__':
    main()
