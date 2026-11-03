"""Structural-only weight I/O. This module never evaluates a network."""
from __future__ import annotations
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import torch
from model import TorusNet

PARAMETERS = 396009
SCHEMA = 'weightspace-torus9-experiment-v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def tensor_schema(model=None):
    model = model if model is not None else TorusNet()
    learned = dict(model.named_parameters())
    return [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
             'count': value.numel(), 'writable': name in learned}
            for name, value in model.state_dict().items()]


def load_arrays(model):
    """Independent NumPy copies of learned parameters; topology is read-only."""
    return {name: value.detach().cpu().numpy().copy() for name, value in model.named_parameters()}


def store_arrays(model, arrays):
    expected = dict(model.named_parameters())
    if set(arrays) != set(expected):
        raise ValueError('Writer must return exactly the learned tensor names')
    with torch.no_grad():
        for name, target in expected.items():
            value = np.asarray(arrays[name])
            if value.shape != tuple(target.shape) or value.dtype != np.float32:
                raise ValueError(f'{name}: expected FP32 {tuple(target.shape)}')
            if not np.isfinite(value).all():
                raise ValueError(f'{name}: nonfinite weights')
            target.copy_(torch.from_numpy(value.copy()))


def validate(model):
    template = TorusNet()
    if model.architecture != template.architecture:
        raise ValueError('Expected unchanged Torus9 5CH GraphNet 8x120')
    state, expected = model.state_dict(), template.state_dict()
    if set(state) != set(expected) or sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('Tensor names/parameter count drift')
    learned = dict(template.named_parameters())
    for name, value in state.items():
        ref = expected[name]
        if value.shape != ref.shape or value.dtype != ref.dtype:
            raise ValueError(f'{name}: tensor shape/dtype drift')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name}: nonfinite tensor')
        if name not in learned and not torch.equal(value.cpu(), ref):
            raise ValueError(f'{name}: immutable topology drift')
    return {'parameter_count': PARAMETERS, 'tensor_count': len(state), 'finite': True,
            'architecture': model.architecture}


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name, str(array.dtype), list(array.shape)], separators=(',', ':')).encode())
        h.update(array.astype(array.dtype.newbyteorder('<'), copy=False).tobytes())
    return h.hexdigest()


def save_checkpoint(model, path, *, design='zeros', writer_sha256=None):
    info = validate(model)
    meta = dict(info, schema=SCHEMA, design=design, training='none',
                state_sha256=state_digest(model), writer_sha256=writer_sha256,
                game_evaluation_status='NOT_EVALUATED', wdl_semantics='side-to-move:[WIN,DRAW,LOSS]',
                runtime={'python': platform.python_version(), 'torch': str(torch.__version__),
                         'numpy': np.__version__})
    payload = {'model_state_dict': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
               'metadata': meta}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path = Path(path)
    sidecar = path.with_suffix('.metadata.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError('Checkpoint version already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(buffer.getvalue())
    meta = dict(meta, artifact_sha256=sha256(path))
    write_json(sidecar, meta)
    return meta


def restore_checkpoint(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.metadata.json').read_text())
    if meta.get('schema') != SCHEMA or meta.get('artifact_sha256') != sha256(path):
        raise ValueError('Checkpoint schema/SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['metadata'] != {k:v for k,v in meta.items() if k != 'artifact_sha256'}:
        raise ValueError('Embedded metadata differs from sidecar')
    if set(payload) != {'model_state_dict', 'metadata'}:
        raise ValueError('Model-only checkpoint required')
    model = TorusNet()
    raw = payload['model_state_dict']
    expected = model.state_dict()
    if set(raw) != set(expected) or any(raw[k].shape != v.shape or raw[k].dtype != v.dtype for k,v in expected.items()):
        raise ValueError('Stored tensors have wrong names/shapes/dtypes')
    model.load_state_dict(raw, strict=True)
    info = validate(model)
    if any(meta.get(k) != v for k,v in info.items()) or state_digest(model) != meta['state_sha256']:
        raise ValueError('Architecture/state hash mismatch')
    model.eval().requires_grad_(False)
    return model, meta
