"""Verified historical replay inputs, exclusively for operator A/B experiments."""
from __future__ import annotations
import json
from pathlib import Path
from ..provenance import file_sha256


def resolve_offline_replay(selectors, *, parent, resolver):
    from .operator_job import resolve_checkpoint
    result = []
    previous = parent.ref
    for offset, selector in enumerate(selectors, 1):
        node = resolve_checkpoint(selector, resolver=resolver)
        if node.generation != parent.generation + offset:
            raise ValueError('offline replay generations must be consecutive after the parent')
        if node.effective_config.config.compatibility != parent.effective_config.config.compatibility:
            raise ValueError('offline replay network/target compatibility mismatch')
        if node.node.parent != previous:
            raise ValueError('offline source checkpoints do not form the parent lineage')
        previous = node.ref
        root = node.owner_root
        provenance = json.loads((root / 'metadata/provenance-v2' / (node.checkpoint_id + '.json')).read_text())
        identities = provenance['artifact_identities']
        artifacts = {}
        for key in ('fresh_replay', 'rolling_replay'):
            identity = identities[key]
            path = (root / identity['path']).resolve()
            if not path.is_relative_to(root.resolve()) or file_sha256(path) != identity['sha256']:
                raise ValueError('offline replay manifest SHA mismatch')
            artifacts[key] = {'path': str(path), 'sha256': identity['sha256']}
        fresh = json.loads(Path(artifacts['fresh_replay']['path']).read_text())
        buckets = [json.loads(l) for l in Path(artifacts['rolling_replay']['path']).read_text().splitlines()]
        if not buckets or buckets[-1] != fresh or fresh['generation'] != node.generation:
            raise ValueError('offline fresh/rolling replay disagree')
        for bucket in buckets:
            for shard in bucket['shards']:
                if file_sha256(Path(shard['path'])) != shard['sha']:
                    raise ValueError('offline replay shard SHA mismatch')
        result.append({'generation': node.generation, 'source_checkpoint': node.ref.to_dict(),
                       **artifacts, 'buckets': buckets, 'fresh_bucket': fresh})
    return result


def iteration_input(config, generation):
    rows = config.to_dict()['extensions'].get('offline_ab_replay')
    if rows is None:
        return None
    matches = [row for row in rows if row['generation'] == generation]
    if len(matches) != 1:
        raise ValueError('offline generation has no unique replay input')
    row = matches[0]
    for key in ('fresh_replay', 'rolling_replay'):
        if file_sha256(Path(row[key]['path'])) != row[key]['sha256']:
            raise ValueError('offline replay manifest changed after preflight')
    if json.loads(Path(row['fresh_replay']['path']).read_text()) != row['fresh_bucket']:
        raise ValueError('offline fresh manifest changed')
    if [json.loads(l) for l in Path(row['rolling_replay']['path']).read_text().splitlines()] != row['buckets']:
        raise ValueError('offline rolling manifest changed')
    return row
