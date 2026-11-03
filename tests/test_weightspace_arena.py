import hashlib
import json

import pytest
import torch

from gocube_golden.artifact_resolver import ArtifactResolver
from gocube_golden.orchestrator_v2.operator_job import _base_config, resolve_checkpoint
from gocube_golden.provenance import file_sha256
from tools.arena_profiles.torus9 import Torus9ArenaProfile
from tools.arena_profiles.weightspace import load_weightspace
from tools.arena_profiles.weightspace_model import TorusNet
from tools.import_weightspace_arena import register


def _source(tmp_path):
    model = TorusNet()
    state = model.state_dict()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)], separators=(',', ':')).encode())
        digest.update(array.astype(array.dtype.newbyteorder('<'), copy=False).tobytes())
    metadata = {'schema': 'weightspace-torus9-experiment-v1', 'architecture': model.architecture,
                'state_sha256': digest.hexdigest()}
    path = tmp_path / 'input.pt'
    torch.save({'metadata': metadata, 'model_state_dict': state}, path)
    metadata['artifact_sha256'] = file_sha256(path).removeprefix('sha256:')
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    return path, metadata


def test_registration_and_profile(tmp_path):
    source, meta = _source(tmp_path)
    runs = tmp_path / 'runs'
    register(source, runs, 'experimental', 'test-revision')
    node = resolve_checkpoint('experimental/v001', resolver=ArtifactResolver(runs))
    assert node.effective_config.config.compatibility['evaluation_only'] is True
    with pytest.raises(ValueError, match='cannot be training parents'):
        _base_config(node)
    profile = Torus9ArenaProfile(five_channel=True, komi=1.5)
    identity = profile.load_identity(node.path)
    model = profile.load_parent_model(identity, torch.device('cpu'))
    policy, wdl = model(torch.zeros(2, 5, 81))
    assert policy.shape == (2, 82) and wdl.shape == (2, 3)
    with pytest.raises(FileExistsError):
        register(source, runs, 'experimental', 'test-revision')
    assert file_sha256(source) == 'sha256:' + meta['artifact_sha256']


def test_corrupt_source_rejected(tmp_path):
    source, meta = _source(tmp_path)
    meta['state_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        load_weightspace(source, meta)
