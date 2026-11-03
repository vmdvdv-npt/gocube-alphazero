"""Torus9 WSP v000: compile a bounded graph circuit into zero FP32 weights.

No observations, inference, external data, random numbers, fitting or training.
See CIRCUITS_V000.md for equations, assumptions and diagnostic limitations.
All linear matrices use PyTorch's [output, input] convention.
"""
import json
import math

import numpy as np


ANCHOR = 16384.0
LN_SCALE = ANCHOR / math.sqrt(60.0)
OUTPUT_OFFSET = 32.0
KOMI = 1.5  # The observation has no komi plane; this is the experiment's komi.


class _Circuit:
    """Append-only residual registers, stored as adjacent (+x, -x) pairs.

    ReLU units are temporary within a block; output matrices write fresh pairs.
    A neighbor coefficient denotes a SUM: multiply it by four to compensate
    for GraphMessageLayer's mean over the four torus neighbors.
    """

    def __init__(self, weights):
        self.w = weights
        self.registers = {}
        self.birth = {}
        self.bounds = {}
        self.block = -1
        self.used_units = [0] * 8

    def new(self, name, bound=1.0):
        assert name not in self.registers and len(self.registers) < 59
        self.registers[name] = 2 * len(self.registers)
        self.birth[name] = self.block
        self.bounds[name] = bound
        return name

    def read(self, matrix, row, terms, factor=1.0):
        for name, coefficient in terms.items():
            assert self.birth[name] < self.block, (self.block, name)
            col = self.registers[name]
            matrix[row, col] += factor * coefficient * 0.5
            matrix[row, col + 1] -= factor * coefficient * 0.5

    def input(self, name, terms, bias=0.0):
        self.new(name)
        row = self.registers[name]
        for column, coefficient in terms.items():
            self.w['input_projection.weight'][row, column] = coefficient
            self.w['input_projection.weight'][row + 1, column] = -coefficient
        self.w['input_projection.bias'][row:row + 2] = (bias, -bias)

    def begin(self, index):
        assert index == self.block + 1
        self.block = index

    def relu(self, destination, local=None, neighbors=None, bias=0.0, gain=1.0):
        assert self.birth[destination] == self.block
        unit = self.used_units[self.block]
        assert unit < 120
        self.used_units[self.block] += 1
        prefix = f'blocks.{self.block}.'
        self.read(self.w[prefix + 'message.self_linear.weight'], unit, local or {})
        self.read(self.w[prefix + 'message.neighbor_linear.weight'], unit,
                  neighbors or {}, factor=4.0)
        self.w[prefix + 'message.self_linear.bias'][unit] = bias
        row = self.registers[destination]
        self.w[prefix + 'output.weight'][row, unit] = gain
        self.w[prefix + 'output.weight'][row + 1, unit] = -gain

    def ramp(self, destination, local=None, neighbors=None, bias=0.0,
             low=0.0, high=1.0):
        """clip((affine-low)/(high-low), 0, 1) using two ReLUs."""
        assert high > low
        self.relu(destination, local, neighbors, bias - low, 1.0 / (high - low))
        self.relu(destination, local, neighbors, bias - high, -1.0 / (high - low))


def program(weights):
    """Write all designed tensors in place, starting from supplied genesis."""
    assert sum(value.size for value in weights.values()) == 396009
    assert len(weights) == 84
    assert all(value.dtype == np.float32 and not np.any(value)
               for value in weights.values()), 'A zero FP32 genesis is required'
    c = _Circuit(weights)
    c.input('own', {0: 1.0})
    c.input('opp', {1: 1.0})
    c.input('empty', {0: -1.0, 1: -1.0}, bias=1.0)
    c.input('side', {2: 1.0})  # +1 black, -1 white; stones are mover-relative.
    c.input('previous_pass', {3: 1.0})
    c.input('legal', {4: 1.0})
    weights['input_projection.bias'][118:120] = (ANCHOR, -ANCHOR)

    # Anchors stay in the residual stream but are never read as features.
    # Equal paired scales let readers cancel LayerNorm's mean.
    for block in range(8):
        weights[f'blocks.{block}.norm.weight'][:118] = LN_SCALE

    # Block 0: counts, color-gated direct liberties, integer-margin flags.
    c.begin(0)
    for color in ('own', 'opp', 'empty'):
        c.relu(c.new('n_' + color, 4.0), neighbors={color: 1.0})
    for color in ('own', 'opp'):
        c.relu(c.new('lib_' + color, 4.0), {color: 4.0}, {'empty': 1.0}, -4.0)
        for liberties in (2, 3):
            c.ramp(c.new(f'direct{liberties}_{color}'), {color: 8.0}, {'empty': 1.0},
                   low=8.0 + liberties - 0.75, high=8.0 + liberties - 0.25)

    # Block 1: certificates over the same-color star (center + adjacent stones).
    # Any EMPTY point touches at most two vertices of this star on the 9x9 torus.
    # Thus S >= 3 certifies >=2 distinct liberties; S >= 5 certifies >=3.
    c.begin(1)
    for color in ('own', 'opp'):
        for liberties, threshold, boost in ((2, 3, 4.0), (3, 5, 8.0)):
            c.ramp(c.new(f'cert{liberties}_{color}_1'),
                   {color: 24.0, 'lib_' + color: 1.0,
                    f'direct{liberties}_{color}': boost},
                   {'lib_' + color: 1.0},
                   low=24.0 + threshold - 0.75, high=24.0 + threshold - 0.25)
    c.relu(c.new('two_step_opp', 16.0), neighbors={'n_opp': 1.0})
    for color in ('own', 'opp'):
        c.relu(c.new(f'influence_{color}_1'), {'n_' + color: 0.125},
               {'n_' + color: 0.03125})

    # Blocks 2..4: bounded OR through same-color stones and isotropic influence.
    # Fresh registers prevent residual double-counting.
    for block in range(2, 5):
        c.begin(block)
        for color in ('own', 'opp'):
            for liberties in (2, 3):
                source = f'cert{liberties}_{color}_{block - 1}'
                c.ramp(c.new(f'cert{liberties}_{color}_{block}'),
                       {color: 8.0, source: 1.0}, {source: 1.0},
                       low=8.25, high=8.75)
            source = f'influence_{color}_{block - 1}'
            c.relu(c.new(f'influence_{color}_{block}'), {source: 0.5}, {source: 0.125})

    # Block 5: final color-gated certificate reach. Absence means suspicion,
    # not an exact liberty count. Soft territory is NOT exact area scoring.
    c.begin(5)
    for color, other in (('own', 'opp'), ('opp', 'own')):
        for liberties in (2, 3):
            source = f'cert{liberties}_{color}_4'
            c.relu(c.new(f'weak{liberties}_{color}'),
                   {color: 1.0, source: -2.0}, {source: -2.0})
        c.ramp(c.new('territory_' + color),
               {f'influence_{color}_4': 2.0, f'influence_{other}_4': -2.0,
                'empty': 2.0}, bias=-2.25)

    # Block 6: move-level tactics. Legal gating makes pooled threats useful to
    # PASS as well. Occupancy alone is not treated as a legal mask.
    c.begin(6)
    for name, color in (('capture', 'opp'), ('rescue', 'own')):
        c.ramp(c.new(name), {'legal': 8.0}, {f'weak2_{color}': 2.0}, bias=-8.5)
    c.relu(c.new('self_atari', 1.5), {'n_empty': -1.0, 'n_own': -2.0},
           {'weak3_own': 2.0, 'weak2_opp': -2.0}, bias=1.5)
    c.relu(c.new('eye'), {'n_own': 1.0, 'two_step_opp': -0.25},
           {'weak2_own': -1.0}, bias=-3.0)
    c.relu(c.new('unsupported_contact', 4.0), {'n_opp': 1.0, 'n_own': -2.0},
           {'weak3_own': 2.0})
    c.ramp(c.new('attack_two'), {'legal': 8.0},
           {'weak3_opp': 2.0, 'weak2_opp': -2.0}, bias=-8.5)

    # Block 7: remove futile rescue/attack bonuses and summarize useful moves.
    # This does not simulate the board after a move.
    c.begin(7)
    c.relu(c.new('safe_rescue'), {'rescue': 1.0, 'self_atari': -2.0})
    c.relu(c.new('safe_attack'), {'attack_two': 1.0, 'self_atari': -2.0})
    c.ramp(c.new('useful_move'), {'legal': 1.0, 'eye': -1.0, 'self_atari': -1.0,
                                'territory_own': -1.0, 'territory_opp': -1.0})

    # A positive offset keeps BOTH rails above the model's final ReLU.
    # Paired readers cancel the offset and LayerNorm mean in real arithmetic,
    # preserving the signed side-to-move feature.
    lanes = 2 * len(c.registers)
    weights['output_norm.weight'][:lanes] = LN_SCALE
    weights['output_norm.bias'][:lanes] = OUTPUT_OFFSET
    c.block = 8
    c.read(weights['point_policy.weight'], 0, {
        'legal': 32.0, 'capture': 8.0, 'safe_rescue': 5.0, 'safe_attack': 1.25,
        'self_atari': -7.0, 'eye': -12.0, 'unsupported_contact': -0.8,
        'n_empty': 0.15, 'n_own': 0.15, 'n_opp': 0.35,
        'influence_own_4': 0.8, 'influence_opp_4': -0.2,
        'territory_own': -2.0, 'territory_opp': -1.3,
    })
    weights['point_policy.bias'][0] = -32.0
    c.read(weights['pass_policy.weight'], 0, {
        'own': 2.0, 'opp': 2.0, 'previous_pass': 0.75, 'useful_move': -4.0,
        # One legal capture reduces PASS by 2 logit units after mean pooling.
        'capture': -162.0, 'safe_rescue': -121.5, 'safe_attack': -40.5,
        'territory_own': 0.5, 'territory_opp': -0.5,
    })
    weights['pass_policy.bias'][0] = -2.0

    # Mover-relative approximate area margin. Undo mean pooling with 81.
    # Weak stones are discounted by half; territory is only a local estimate.
    margin = {'own': 81.0, 'opp': -81.0,
              'territory_own': 81.0, 'territory_opp': -81.0,
              'weak2_own': -40.5, 'weak2_opp': 40.5, 'side': -KOMI}
    for head in ('value_head', 'score_head'):
        c.read(weights[head + '.0.weight'], 0, margin)
        c.read(weights[head + '.0.weight'], 1, margin, factor=-1.0)
    weights['value_head.2.weight'][0, :2] = (1.0 / 12.0, -1.0 / 12.0)
    weights['value_head.2.weight'][2, :2] = (-1.0 / 12.0, 1.0 / 12.0)
    weights['value_head.2.bias'][1] = -6.0
    weights['score_head.2.weight'][0, :2] = (1.0, -1.0)

    # Auxiliary convention of this artifact: [own, neutral, opponent].
    for row, color in ((0, 'own'), (1, 'opp')):
        c.read(weights['ownership_head.0.weight'], row,
               {color: 1.0, 'territory_' + color: 1.0})
    weights['ownership_head.2.weight'][0, 0] = 4.0
    weights['ownership_head.2.weight'][2, 1] = 4.0
    weights['ownership_head.2.bias'][1] = 1.0

    # Ideal real-arithmetic bounds, NOT measured activations.
    energy_bound = sum(bound * bound for bound in c.bounds.values())
    norm_contraction_bound = 0.5 * (energy_bound + 60.0e-5) / ANCHOR**2
    assert energy_bound < 512.0 and max(c.bounds.values()) < OUTPUT_OFFSET
    assert all(np.isfinite(value).all() for value in weights.values())
    print(json.dumps({
        'design': 'torus9-paired-register-graph-circuit-v000',
        'parameters': sum(value.size for value in weights.values()),
        'nonzero_parameters': sum(int(np.count_nonzero(value)) for value in weights.values()),
        'registers': c.registers, 'register_ideal_abs_bounds': c.bounds,
        'relu_units_per_block': c.used_units, 'anchor': ANCHOR,
        'ideal_register_energy_bound': energy_bound,
        'ideal_layernorm_contraction_bound': norm_contraction_bound,
        'komi_assumption': KOMI, 'forward_calls': 0, 'training': 'none',
    }, sort_keys=True))
