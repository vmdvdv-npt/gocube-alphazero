import pytest
import torch
from gocube_golden.b64_speedup_comparison import load_reference, state_hash
from gocube_golden.provenance import file_sha256


def test_reference_is_verified_before_execution(tmp_path):
    source=tmp_path/'reference.py'
    source.write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError,match='Reference trainer SHA mismatch'):
        load_reference(source,'sha256:'+'0'*64)
    source.write_text('class OrdinaryTrainer:\n    pass\n')
    cls=load_reference(source,file_sha256(source))
    assert cls.__name__=='OrdinaryTrainer'
    assert cls.__module__.startswith('gocube_golden._b64_reference_')


def test_state_hash_enforces_tensor_bytes_and_structure():
    assert state_hash(torch.tensor(0.)) != state_hash(torch.tensor(-0.))
    assert state_hash(torch.tensor([1.])) != state_hash(torch.tensor([1.],dtype=torch.float64))
    assert state_hash({'x':[torch.tensor([1.]),2]}) == state_hash({'x':[torch.tensor([1.]),2]})
    assert state_hash({'a':{'b':1},'c':2}) != state_hash({'a':{'b':1,'c':2}})
