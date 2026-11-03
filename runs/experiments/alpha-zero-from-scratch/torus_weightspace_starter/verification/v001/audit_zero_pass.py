"""Audit frozen v001 tensors and dependencies without executing any circuit.

No observations, game replay, local motif evaluation, or candidate forward calls.
The saved v000 and diagnostic files are read only for integrity hashes.
"""
import ast
import hashlib
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
    version = ROOT / 'versions/v001'
    with patch.object(torch.nn.Module, '_call_impl',
                      side_effect=AssertionError('Module execution forbidden')):
        model, meta = restore_checkpoint(version / 'checkpoint.pt')
        genesis, _ = restore_checkpoint(version / 'genesis.pt')
        assert all(not torch.count_nonzero(p) for p in genesis.parameters())
        learned = {k: p.detach().numpy() for k, p in model.named_parameters()}
        state = model.state_dict()
        layout = json.loads((version / 'writer.stdout.txt').read_text())
        regs = layout['registers']
        report = validate(model)
        nonzero = sum(int(np.count_nonzero(p)) for p in learned.values())
        assert nonzero == layout['nonzero_parameters'] == 2092
        assert len(learned) == 84 and len(regs) == 59
        assert learned['input_projection.bias'][118:].tolist() == [16384.0, -16384.0]

        for key in ('input_projection.weight', 'input_projection.bias'):
            assert np.array_equal(learned[key][0:118:2], -learned[key][1:118:2])
        written = set(range(0, 12, 2))
        details = []
        for block in range(8):
            prefix = f'blocks.{block}.'
            out = learned[prefix + 'output.weight']
            assert np.array_equal(out[0:118:2], -out[1:118:2])
            assert not np.any(out[118:])
            assert not np.any(learned[prefix + 'output.bias'])
            dest = set(np.flatnonzero(np.any(out[:118:2], axis=1)) * 2)
            assert not dest.intersection(written)
            inputs = set()
            for component in ('self', 'neighbor'):
                mat = learned[prefix + f'message.{component}_linear.weight']
                assert np.array_equal(mat[:, 0:118:2], -mat[:, 1:118:2])
                assert not np.any(mat[:, 118:])
                inputs.update(np.flatnonzero(np.any(mat[:, :118:2], axis=0)) * 2)
            assert inputs <= written
            norm = learned[prefix + 'norm.weight']
            assert np.all(norm[:118] == np.float32(16384.0 / np.sqrt(60.0)))
            assert not np.any(norm[118:])
            assert not np.any(learned[prefix + 'norm.bias'])
            written.update(dest)
            details.append({'block': block, 'new_registers': len(dest),
                            'relu_units': int(np.count_nonzero(np.any(out, axis=0)))})
        assert len(written) == 59
        for key in ('point_policy.weight', 'pass_policy.weight', 'value_head.0.weight',
                    'score_head.0.weight', 'ownership_head.0.weight'):
            assert np.array_equal(learned[key][:, 0:118:2], -learned[key][:, 1:118:2])
            assert not np.any(learned[key][:, 118:])
        assert np.all(learned['output_norm.bias'][:118] == 32.0)
        assert not np.any(learned['output_norm.bias'][118:])
        assert np.all(learned['output_norm.weight'][:118] == norm[0])

        def read_coefficients(row, neighbor=False):
            # Algebraically decode weight coefficients; never multiply by inputs.
            factor = 0.5 if neighbor else 2.0
            return {name: float(row[col]) * factor for name, col in regs.items() if row[col]}

        def units(block, name):
            prefix = f'blocks.{block}.'
            out = learned[prefix + 'output.weight'][regs[name]]
            return [{'local': read_coefficients(learned[prefix + 'message.self_linear.weight'][u]),
                     'neighbors': read_coefficients(learned[prefix + 'message.neighbor_linear.weight'][u], True),
                     'bias': float(learned[prefix + 'message.self_linear.bias'][u] +
                                   learned[prefix + 'message.neighbor_linear.bias'][u]),
                     'gain': float(out[u])} for u in np.flatnonzero(out)]

        # Verify actual compiled positive-proof routes, not a duplicate strategy.
        for name, color in (('capture', 'opp'), ('rescue', 'own')):
            expected = [{'local': {'legal': 8.0}, 'neighbors': {'closed_atari_' + color: 2.0},
                         'bias': b, 'gain': g} for b, g in ((-8.5, 1.0), (-9.5, -1.0))]
            assert units(6, name) == expected
        for color in ('own', 'opp'):
            assert units(0, f'peeled_{color}_0') == [
                {'local': {color: 8.0}, 'neighbors': {'empty': -5.0, color: -1.0},
                 'bias': b, 'gain': g} for b, g in ((-6.25, 2.0), (-6.75, -2.0))]
            for block in (1, 2):
                assert units(block, f'peeled_{color}_{block}') == [
                    {'local': {color: 8.0, 'lib_' + color: -5.0, 'n_' + color: -1.0},
                     'neighbors': {f'peeled_{color}_{block-1}': 1.0},
                     'bias': b, 'gain': g} for b, g in ((-6.25, 2.0), (-6.75, -2.0))]
            assert units(3, 'closed_atari_' + color) == [
                {'local': {color: 8.0, 'lib_' + color: -1.0, 'n_' + color: -4.0},
                 'neighbors': {f'peeled_{color}_2': 4.0},
                 'bias': b, 'gain': g} for b, g in ((-6.25, 2.0), (-6.75, -2.0))]
        danger = units(6, 'self_atari')[0]
        assert danger['neighbors'] == {'weak3_own': 2.0, 'closed_atari_opp': -2.0}
        assert danger['local'] == {'n_own': -1.0, 'n_opp': 1.0}
        assert danger['bias'] == -2.5
        margin = read_coefficients(learned['value_head.0.weight'][0])
        assert margin == {'own': 81.0, 'opp': -81.0, 'territory_own': 20.25,
                          'territory_opp': -20.25, 'closed_atari_own': -20.25,
                          'closed_atari_opp': 20.25, 'side': -1.5}
        value = learned['value_head.0.weight']
        assert np.array_equal(value[0], -value[1])
        assert np.array_equal(value[2], value[0]) and np.array_equal(value[3], value[1])
        assert learned['value_head.0.bias'][:4].tolist() == [0.0, 0.0, -36.0, -36.0]
        final = learned['value_head.2.weight']
        assert np.array_equal(final[0], -final[2]) and not np.any(final[1])
        assert np.array_equal(final[0, :4], np.array([1, -1, -1, 1], np.float32)/24.0)
        assert not np.any(final[:, 4:])

        adjacency = [set(row) for row in state['blocks.0.message.neighbors'].tolist()]
        multiplicity = max(2*int(p in adjacency[x]) + len(adjacency[p] & adjacency[x])
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
                            assert all({permutation[q] for q in adjacency[p]} ==
                                       adjacency[permutation[p]] for p in range(81))
                            symmetries += 1
        source = (version / 'writer.py').read_bytes()
        assert source == (ROOT / 'design_weights.py').read_bytes()
        assert source == (version / 'infrastructure/design_weights.py').read_bytes()
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
        before = json.loads((ROOT / 'verification/v001/preservation-before.json').read_text())
        for name, digest in before.items():
            assert sha256(ROOT / name) == digest, name
        sessions = [ROOT / 'diagnostics-v000-20261103/session',
                    Path('/tmp/torus9-v000-diagnostic-20261103/session')]
        budgets = []
        for session in sessions:
            config = json.loads((session / 'session.json').read_text())
            rounds = sorted(p.name for p in (session / 'rounds').iterdir())
            assert config['budget'] == 10 and rounds == ['001', '002', '003', '004', '005', '006']
            budgets.append({'path': str(session), 'used': 6, 'budget': 10, 'remaining': 4})
        failure = ROOT / 'V001_FAILURE_CASES.md'
        blob = hashlib.sha1(b'blob ' + str(failure.stat().st_size).encode() + b'\0' + failure.read_bytes()).hexdigest()
        assert blob == '5029f8f4fdf79413f36b9937eeb9aae03758c96a'
        zero = json.loads((version / 'zero-pass.json').read_text())
        assert zero['status'] == 'PASS' and zero['forward_calls'] == 0 and zero['reproducible']
        report.update(status='PASS', mode='structural-only', forward_calls=0,
            observations_constructed=0, diagnostic_rounds_consumed=0, sessions=budgets,
            nonzero_parameters=nonzero, learned_tensor_count=84, logical_registers=59,
            nonzero_per_tensor={k: int(np.count_nonzero(v)) for k, v in learned.items()},
            block_details=details, antisymmetric_registers=True, append_only_registers=True,
            positive_capture_weight_routes=True, self_atari_uses_positive_capture_proof=True,
            wdl_coefficient_symmetry=True, wdl_logit_cap_ideal=1.5,
            topology_symmetries_checked=symmetries, weighted_star_multiplicity=multiplicity,
            symmetry_check_scope='structure and topology, not measured forward outputs',
            regression_execution_status='NOT_RUN', game_evaluation_status='NOT_EVALUATED',
            preservation_files_checked=len(before), writer_imports=imports,
            failure_cases_pr=266, failure_cases_commit='ae5dcced64fdc783adcfda924de688920b375bfd',
            failure_cases_git_blob=blob, artifact_sha256=meta['artifact_sha256'],
            state_sha256=meta['state_sha256'], writer_sha256=meta['writer_sha256'],
            runtime=meta['runtime'], training='none',
            source_revision_before_v001_commit=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            ideal_register_energy_bound=layout['ideal_register_energy_bound'],
            ideal_layernorm_contraction_bound=layout['ideal_layernorm_contraction_bound'])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    audit()
