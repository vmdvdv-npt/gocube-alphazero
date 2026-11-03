"""Read the model-only experimental Torus9 weight-space format for Arena."""
import hashlib
import json
from pathlib import Path

import torch

from gocube_golden.provenance import file_sha256
from tools.arena_profiles.weightspace_model import TorusNet


def load_weightspace(path: Path, metadata: dict) -> TorusNet:
    if file_sha256(path) != 'sha256:' + metadata.get('artifact_sha256', ''):
        raise ValueError('Weight-space checkpoint SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if set(payload) != {'metadata', 'model_state_dict'} or payload['metadata'] != {
        k: v for k, v in metadata.items() if k != 'artifact_sha256'
    }:
        raise ValueError('Weight-space payload and sidecar disagree')
    model = TorusNet()
    if metadata.get('architecture') != model.architecture:
        raise ValueError('Unsupported experimental architecture')
    state = payload['model_state_dict']
    template = model.state_dict()
    if set(state) != set(template):
        raise ValueError('Weight-space tensor names differ')
    digest = hashlib.sha256()
    parameters = dict(model.named_parameters())
    for name, value in sorted(state.items()):
        expected = template[name]
        if value.shape != expected.shape or value.dtype != expected.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Invalid weight-space tensor: {name}')
        if name not in parameters and not torch.equal(value, expected):
            raise ValueError(f'Weight-space topology mismatch: {name}')
        array = value.contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)], separators=(',', ':')).encode())
        digest.update(array.astype(array.dtype.newbyteorder('<'), copy=False).tobytes())
    if digest.hexdigest() != metadata.get('state_sha256'):
        raise ValueError('Weight-space state hash mismatch')
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
