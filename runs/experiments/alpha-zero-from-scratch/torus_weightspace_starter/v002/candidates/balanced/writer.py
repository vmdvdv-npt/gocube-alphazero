"""Analytical WSP v002: reusable single-rail fields and nonlinear feature banks.

No observations, trained weights, randomness, fitting, or network execution.
The only inputs are zero FP32 parameter arrays of the unchanged GraphNet.
"""
import json
import math
import numpy as np

ANCHOR = 1048576.0
SCALE = ANCHOR / math.sqrt(60.0)
BALANCE = 116
ZERO = 117


class Circuit:
    """Single-rail signals; one shared balancing lane makes their sum zero.

    Each block reads the previous state. replace() cancels the old nonnegative
    field through an identity ReLU before accumulating its new expression.
    A signed field requires two cancellation units. No extra inference code.
    """
    def __init__(self, w):
        self.w, self.reg, self.birth = w, {}, {}
        self.block = -1
        self.units = [0] * 8
        self.replacements = []

    def new(self, name):
        if name not in self.reg:
            assert len(self.reg) < BALANCE
            self.reg[name] = len(self.reg)
            self.birth[name] = self.block
        return name

    def input(self, name, terms, bias=0):
        row = self.reg[self.new(name)]
        for col, value in terms.items():
            self.w['input_projection.weight'][row, col] = value
            self.w['input_projection.weight'][BALANCE, col] -= value
        self.w['input_projection.bias'][row] = bias
        self.w['input_projection.bias'][BALANCE] -= bias

    def read(self, matrix, row, terms, factor=1):
        for name, value in (terms or {}).items():
            assert self.birth[name] < self.block, (self.block, name)
            matrix[row, self.reg[name]] += factor * value
            matrix[row, ZERO] -= factor * value

    def relu(self, name, local=None, neighbors=None, bias=0, gain=1):
        self.new(name)
        unit = self.units[self.block]
        assert unit < 120, (self.block, unit)
        self.units[self.block] += 1
        p = 'blocks.%d.' % self.block
        self.read(self.w[p+'message.self_linear.weight'], unit, local)
        self.read(self.w[p+'message.neighbor_linear.weight'], unit, neighbors, 4)
        self.w[p+'message.self_linear.bias'][unit] = bias
        self.w[p+'output.weight'][self.reg[name], unit] += gain
        self.w[p+'output.weight'][BALANCE, unit] -= gain

    def clip(self, name, local=None, neighbors=None, bias=0, ceiling=1):
        self.relu(name, local, neighbors, bias)
        self.relu(name, local, neighbors, bias-ceiling, -1)

    def replace(self, name):
        self.relu(name, {name: 1}, gain=-1)
        self.replacements.append([self.block, name])

    def begin(self, block):
        assert block == self.block + 1
        self.block = block


def program(w):
    assert len(w) == 84 and sum(a.size for a in w.values()) == 396009
    assert all(a.dtype == np.float32 and not np.any(a) for a in w.values())
    c = Circuit(w)
    c.input('own', {0: 1})
    c.input('opp', {1: 1})
    c.input('empty', {0: -1, 1: -1}, 1)
    c.input('side', {2: 1})
    c.input('pass', {3: 1})
    c.input('legal', {4: 1})
    w['input_projection.bias'][118:120] = (ANCHOR, -ANCHOR)
    for i in range(8):
        w['blocks.%d.norm.weight' % i][:118] = SCALE

    c.begin(0)
    for color in ('own', 'opp', 'empty'):
        c.relu('n_'+color, neighbors={color: 1})
    for color in ('own', 'opp'):
        c.relu('lib_'+color, {color: 4}, {'empty': 1}, -4)
        c.clip('peel_'+color, {color: 16}, {'empty': -10, color: -2}, -12.5)
        for degree in range(5):
            for shift, gain in ((1,1), (0,-2), (-1,1)):
                c.relu('degree_%s_%d' % (color,degree), {color:16},
                       {color:1}, -16-degree+shift, gain)
        # Three decay scales: short liberty access, medium reach, broad reach.
        for k in range(3):
            c.relu('access_%s_%d' % (color, k), {color: 4}, {'empty': 1}, -4, .25)
            c.relu('field_%s_%d' % (color, k), {color: 1})
        # Connected local mass, bounded later; used to weight endangered stones.
        c.relu('mass_'+color, {color: 1})

    # Reuse the same planes on every iteration. Liberty certificates remain
    # lower-bound witnesses; continuous access and mass are explicitly heuristic.
    for block in range(1, 6):
        c.begin(block)
        for color, other in (('own', 'opp'), ('opp', 'own')):
            if block == 1:
                c.relu('crowd_'+color, {'n_'+color: .125}, {'n_'+color: .03125})
                c.relu('spread_'+color, {'n_'+color: .125}, {'n_'+color: .03125})
            elif block < 5:
                c.replace('spread_'+color)
                c.relu('spread_'+color, {'spread_'+color: .5}, {'spread_'+color: .125})
            for count, threshold in ((2, 3), (3, 5)):
                name = 'cert%d_%s' % (count, color)
                if block == 1:
                    c.clip(name, {color: 48, 'lib_'+color: 4},
                           {'lib_'+color: 2}, -48-2*threshold+1.5)
                else:
                    c.replace(name)
                    c.clip(name, {color: 16, name: 2}, {name: 2}, -16.5)
            c.replace('peel_'+color)
            c.clip('peel_'+color,
                   {color: 16, 'lib_'+color: -10, 'n_'+color: -2},
                   {'peel_'+color: 2}, -12.5)
            for k, decay in enumerate((.35, .65, .9)):
                access = 'access_%s_%d' % (color, k)
                field = 'field_%s_%d' % (color, k)
                c.replace(access)
                # Degree-conditioned averaging along the actual stone graph.
                # Five disjoint gates implement division by degree analytically.
                for degree in range(5):
                    c.relu(access, {'degree_%s_%d' % (color,degree): 4,
                                    'lib_'+color: (1-decay)/4},
                           {access: decay/max(1,degree)}, -4)
                c.replace(field)
                # Diffusion with a persistent source and opponent obstruction.
                c.relu(field, {color: 1-decay, other: -1}, {field: decay/4})
            c.replace('mass_'+color)
            c.clip('mass_'+color, {color: 8}, {'mass_'+color: .5}, -7, ceiling=8)
        if block == 5:
            for color, other in (('own', 'opp'), ('opp', 'own')):
                c.clip('atari_'+color, {color: 16, 'lib_'+color: -2, 'n_'+color: -8},
                       {'peel_'+color: 8}, -12.5)
                c.relu('weak2_'+color, {color: 1, 'cert2_'+color: -2}, {'cert2_'+color: -2})
                c.relu('weak3_'+color, {color: 1, 'cert3_'+color: -2}, {'cert3_'+color: -2})
                c.clip('area_'+color, {'empty':2,'spread_'+color:2,'spread_'+other:-2},bias=-2.25)
                c.relu('risk_degree_'+color, {'n_'+color:1,color:4,'cert3_'+color:-4},
                       {'cert3_'+color:-4},-4)
                # Empty-point ownership from three scales of competing fields.
                for k in range(3):
                    c.clip('territory_%s_%d' % (color, k),
                           {'empty': 4, 'field_%s_%d' % (color,k): 8,
                            'field_%s_%d' % (other,k): -8}, bias=-4.2)

    c.begin(6)
    for name, color in (('capture', 'opp'), ('rescue', 'own')):
        c.clip(name, {'legal': 16}, {'atari_'+color: 4}, -16.5)
    c.relu('self_atari', {'n_empty': -1, 'n_own': -2},
           {'weak3_own': 2, 'atari_opp': -4}, 1.5)
    c.clip('eye', {'n_own': 2}, {'n_opp': -.5, 'weak2_own': -2}, -6)
    # Bank of continuous risk responses; distinct slopes/thresholds reflect
    # uncertainty at several diffusion scales instead of one absent certificate.
    for color in ('own', 'opp'):
        for k in range(3):
            c.relu('risk_%s_%d' % (color,k),
                   {color: 1, 'access_%s_%d' % (color,k): -4,
                    'cert3_'+color: -.5})
        if color == 'own':
            c.clip('pressure_'+color, {'legal':4},
                   {'weak3_'+color:.5,'risk_degree_'+color:.125},-4)
        else:
            c.clip('pressure_'+color, {'legal': 8},
                   {'weak3_'+color: 2, 'weak2_'+color: -1.5, 'atari_'+color: -2}, -8)
        c.relu('exposure_'+color, {'mass_'+color: 1, 'cert3_'+color: -8})
    # Piecewise-linear geometric basis, gated to legal moves. Many hinges
    # contribute jointly; no independent persistent register for every hinge.
    for color, other in (('own','opp'), ('opp','own')):
        for k in range(3):
            for threshold, slope in ((0,1), (.125,-.5), (.25,-.25), (.5,-.25)):
                c.relu('potential_'+color,
                       {'field_%s_%d' % (color,k): 1,
                        'field_%s_%d' % (other,k): -.6, 'legal': 2},
                       bias=-2-threshold, gain=slope/(k+1))
    c.clip('connection', {'n_own': 1, 'n_opp': .5}, bias=-1)
    c.clip('cut', {'n_opp': 1, 'n_own': .5}, bias=-1)
    c.relu('contact', {'n_opp': 1, 'n_own': -1.5}, {'weak3_own': 1.5})

    c.begin(7)
    c.relu('safe_rescue', {'rescue': 1, 'self_atari': -2})
    c.relu('safe_defense', {'pressure_own': 1, 'self_atari': -2})
    c.clip('useful', {'legal': 8, 'eye': -8, 'self_atari': -8}, bias=-4)
    # Danger is spread back to candidate liberties. Nonlinear aggregation caps
    # duplicate neighboring stones; degree is not a count of distinct groups.
    for color in ('own','opp'):
        for k in range(3):
            c.clip('urgency_%s_%d' % (color,k), {'legal': 8, 'self_atari': -4},
                   {'risk_%s_%d' % (color,k): 1}, -8, ceiling=2)
        c.clip('stakes_'+color, {'legal': 16, 'self_atari': -8},
               {'exposure_'+color: .25}, -16, ceiling=2)
    # Candidate safety surface: a hinge lattice over extension liberties and
    # allied/opposing access. The shared output is an approximate utility.
    for libs, reward, direction in ((1.5, -4, -1), (1.5, .25, 1), (3.5, -.25, 1)):
        for support in (0, .25, .5, .75):
            c.relu('shape', {'n_empty': direction, 'self_atari': -2*direction, 'legal': 16},
                   {'access_own_1': 2*direction, 'access_opp_1': -.5*direction},
                   -16-direction*(libs+support), reward/4)

    w['output_norm.weight'][:118] = SCALE
    # ZERO and feature lanes share a positive offset; read() cancels it. The
    # balancing lane is never consumed, so clipping it is irrelevant.
    w['output_norm.bias'][:118] = 32
    c.block = 8
    policy = {'legal': 32, 'capture': 8, 'safe_rescue': 5, 'safe_defense': 3,
              'self_atari': -7, 'eye': -12, 'contact': -.8,
              'n_own': -.35, 'n_opp': -.1, 'crowd_own': -2,
              'spread_own': 1, 'spread_opp': -1, 'area_own': -1, 'area_opp': -.75,
              'connection': .1, 'cut': .1, 'shape': .05,
              'stakes_own': .1, 'stakes_opp': .05, 'pressure_opp': .75}
    for k in range(3):
        policy['urgency_own_%d' % k] = .1/(k+1)
        policy['urgency_opp_%d' % k] = .05/(k+1)
    c.read(w['point_policy.weight'], 0, policy)
    w['point_policy.bias'][0] = -31.2
    c.read(w['pass_policy.weight'], 0,
           {'own': 2, 'opp': 2, 'pass': .25, 'useful': -162,
            'capture': -162, 'safe_rescue': -121.5, 'safe_defense': -81})
    w['pass_policy.bias'][0] = -2.5
    margin = {'own': 81, 'opp': -81, 'side': -1.5,
              'atari_own': -20.25, 'atari_opp': 20.25}
    for k in range(3):
        margin['territory_own_%d' % k] = 20.25/3
        margin['territory_opp_%d' % k] = -20.25/3
    for head in ('value_head', 'score_head'):
        for row, sign in ((0,1), (1,-1)):
            c.read(w[head+'.0.weight'], row, margin, sign)
    for row, sign in ((2,1), (3,-1)):
        c.read(w['value_head.0.weight'], row, margin, sign)
        w['value_head.0.bias'][row] = -36
    w['value_head.2.weight'][0,:4] = np.array([1,-1,-1,1])/24
    w['value_head.2.weight'][2,:4] = -w['value_head.2.weight'][0,:4]
    w['value_head.2.bias'][1] = -6
    w['score_head.2.weight'][0,:2] = (1,-1)
    for row, color in ((0,'own'), (1,'opp')):
        c.read(w['ownership_head.0.weight'], row, {color: 1, 'territory_'+color+'_1': .5})
        w['ownership_head.2.weight'][row*2,row] = 4
    w['ownership_head.2.bias'][1] = 1
    assert all(np.isfinite(a).all() for a in w.values())
    print(json.dumps({'design': 'single-rail-recurrent-fields-v002',
                      'registers': c.reg, 'relu_units_per_block': c.units,
                      'replacements': c.replacements, 'anchor': ANCHOR,
                      'parameters': 396009,
                      'nonzero_parameters': sum(int(np.count_nonzero(a)) for a in w.values()),
                      'training': 'none', 'forward_calls': 0}, sort_keys=True))
