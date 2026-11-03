"""Structural audit of the frozen v000 artifact; never builds observations.

Run from torus_weightspace_starter with the protocol's Python interpreter.
All nn.Module calls are disabled during the audit. No writer/network execution.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from weights import restore_checkpoint, sha256, validate


def audit():
    torch.set_num_threads(1)
    version = ROOT / 'versions/v000'
    with patch.object(torch.nn.Module, '_call_impl',
                      side_effect=AssertionError('All module execution forbidden')):
        model, meta = restore_checkpoint(version / 'checkpoint.pt')
        genesis, _ = restore_checkpoint(version / 'genesis.pt')
        assert all(not torch.count_nonzero(p) for p in genesis.parameters())
        learned = {k: p.detach().numpy() for k, p in model.named_parameters()}
        state = model.state_dict()
        layout = json.loads((version / 'writer.stdout.txt').read_text())
        report = validate(model)
        nonzero = sum(int(np.count_nonzero(p)) for p in learned.values())
        assert nonzero == layout['nonzero_parameters'] == 1921
        assert len(learned) == 84
        assert learned['input_projection.bias'][118:].tolist() == [16384.0, -16384.0]

        # Exact weight algebra, not activation comparisons.
        for key in ('input_projection.weight', 'input_projection.bias'):
            assert np.array_equal(learned[key][0:118:2], -learned[key][1:118:2])
        written = set(range(0, 12, 2))
        block_details = []
        for block in range(8):
            prefix = f'blocks.{block}.'
            out = learned[prefix + 'output.weight']
            assert np.array_equal(out[0:118:2], -out[1:118:2])
            assert not np.any(out[118:])
            assert not np.any(learned[prefix + 'output.bias'])
            destinations = set(np.flatnonzero(np.any(out[:118:2], axis=1)) * 2)
            assert not destinations.intersection(written)
            inputs = set()
            for component in ('self', 'neighbor'):
                mat = learned[prefix + f'message.{component}_linear.weight']
                assert np.array_equal(mat[:, 0:118:2], -mat[:, 1:118:2])
                assert not np.any(mat[:, 118:])
                inputs.update(np.flatnonzero(np.any(mat[:, :118:2], axis=0)) * 2)
            assert inputs <= written  # No same-block or future-register reads.
            norm = learned[prefix + 'norm.weight']
            assert np.array_equal(norm[0:118:2], norm[1:118:2])
            assert np.all(norm[:118] == np.float32(16384.0 / np.sqrt(60.0)))
            assert not np.any(learned[prefix + 'norm.bias'])
            written.update(destinations)
            block_details.append({'block': block, 'new_registers': len(destinations),
                                  'relu_units': int(np.count_nonzero(np.any(out, axis=0)))})
        assert len(written) == len(layout['registers']) == 55
        for key in ('point_policy.weight', 'pass_policy.weight', 'value_head.0.weight',
                    'score_head.0.weight', 'ownership_head.0.weight'):
            assert np.array_equal(learned[key][:, 0:118:2], -learned[key][:, 1:118:2])
            assert not np.any(learned[key][:, 118:])
        assert np.all(learned['output_norm.bias'][:110] == 32.0)

        adjacency = [set(row) for row in state['blocks.0.message.neighbors'].tolist()]
        assert all(len(row) == 4 for row in adjacency)
        # EMPTY x cannot be the occupied star center p. The entire star is a
        # superset of its same-color part, so this proves the multiplicity bound.
        multiplicity = max(len(({p} | adjacency[p]) & adjacency[x])
                           for p in range(81) for x in range(81) if x != p)
        assert multiplicity == 2
        symmetries = 0
        for swap in (False, True):
            for sr in (-1, 1):
                for sc in (-1, 1):
                    for dr in range(9):
                        for dc in range(9):
                            permutation = []
                            for p in range(81):
                                r, c = divmod(p, 9)
                                if swap:
                                    r, c = c, r
                                permutation.append(((sr*r+dr) % 9)*9 + (sc*c+dc) % 9)
                            assert len(set(permutation)) == 81
                            assert all({permutation[q] for q in adjacency[p]} ==
                                       adjacency[permutation[p]] for p in range(81))
                            symmetries += 1

        source = (version / 'writer.py').read_bytes()
        assert source == (ROOT / 'design_weights.py').read_bytes()
        assert source == (version / 'infrastructure/design_weights.py').read_bytes()
        assert sha256(version / 'writer.py') == meta['writer_sha256']
        tree = ast.parse(source)
        imports = sorted(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                         for alias in node.names)
        assert imports == ['json', 'math', 'numpy']
        assert not any(isinstance(node, ast.ImportFrom) for node in ast.walk(tree))
        forbidden = {'open', 'eval', 'exec', '__import__', 'forward', 'backward',
                     'load', 'save', 'random', 'optim', 'fit', 'lstsq'}
        symbols = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        symbols.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        assert not symbols.intersection(forbidden)
        session = json.loads((ROOT / 'versions/session0/session.json').read_text())
        assert session['budget'] == 0
        # Git omits this empty directory; a fresh checkout may not contain it.
        assert not list((ROOT / 'versions/session0/rounds').glob('*'))
        frozen = json.loads((version / 'zero-pass.json').read_text())
        assert frozen['status'] == 'PASS' and frozen['forward_calls'] == 0
        assert frozen['reproducible'] is True
        report.update(
            status='PASS', mode='structural-only', forward_calls=0,
            observations_constructed=0, diagnostic_rounds=0, diagnostic_budget=0,
            learned_tensor_count=len(learned), nonzero_parameters=nonzero,
            nonzero_per_tensor={k: int(np.count_nonzero(v)) for k, v in learned.items()},
            logical_registers=len(written), block_details=block_details,
            antisymmetric_register_writes=True, antisymmetric_feature_reads=True,
            append_only_residual_registers=True, immutable_anchor_writes=True,
            maximum_star_liberty_multiplicity=multiplicity,
            adjacency_symmetries_checked=symmetries,
            symmetry_check_scope='topology and shared-weight structure, not forward outputs',
            writer_imports=imports, writer_snapshot_matches=True,
            artifact_sha256=meta['artifact_sha256'], state_sha256=meta['state_sha256'],
            writer_sha256=meta['writer_sha256'], runtime=meta['runtime'],
            source_revision_before_v000_commit=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            ideal_register_energy_bound=layout['ideal_register_energy_bound'],
            ideal_layernorm_contraction_bound=layout['ideal_layernorm_contraction_bound'],
            numeric_bound_scope='ideal real-arithmetic circuit; FP32 behavior unmeasured',
            game_evaluation_status='NOT_EVALUATED', training='none',
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    audit()
