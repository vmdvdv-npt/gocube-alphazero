# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from libc.math cimport sqrt, log, erf, pow

import numpy as np
cimport numpy as np

from alphazero.utils import dotdict
from alphazero.mcts_policy import normalize_masked_policy
from alphazero.envs.gocube.exploration_contract import (
    KATAGO_PINNED_EXPLORATION_DEFAULTS,
    apply_chosen_move_pruning,
    apply_lcb_play_selection,
    retrospectively_reduce_root_visits,
    retrospectively_reduce_root_weights,
    root_policy_temperature,
    shaped_dirichlet_alpha_distribution,
)
from alphazero.search_contract import (
    KATAGO_PINNED_SEARCH_UTILITY_MODE,
    SearchOutput,
    combined_white_utility,
    conservative_root_observation,
    normalized_black_minus_white_to_white_score,
    recent_score_center,
    root_ending_white_score_bonuses,
    score_utility_diff,
    white_owner_map,
)


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

NOISE_ALPHA_RATIO = 10.83
_DRAW_VALUE = 0.5
_TOTAL_CHILD_WEIGHT_PUCT_OFFSET = 0.01
_SQRT_TWO = 1.4142135623730951

np.seterr(all='raise')


def _optional_arg(args, name, default):
    if hasattr(args, 'get'):
        return args.get(name, default)
    try:
        return getattr(args, name)
    except (AttributeError, KeyError):
        return default


cdef class Node:
    cdef public list _children
    cdef public int a
    cdef public np.ndarray e
    cdef public float q
    cdef public float v
    cdef public float score_q
    cdef public float score_v
    cdef public float utility_sq
    cdef public float weight_sum
    cdef public float weight_sq_sum
    cdef public bint has_score
    cdef public int n
    cdef public float p
    cdef public int player

    def __init__(self, int action, int num_players):
        self._children = []
        self.a = action
        self.e = np.zeros(num_players, dtype=np.uint8)
        self.q = 0
        self.v = 0
        self.score_q = 0
        self.score_v = 0
        self.utility_sq = 0
        self.weight_sum = 0
        self.weight_sq_sum = 0
        self.has_score = False
        self.n = 0
        self.p = 0
        self.player = 0

    def __repr__(self):
        return (
            'Node(a={}, q={}, v={}, score_q={}, utility_sq={}, weight_sum={}, n={}, p={}, player={})'
            .format(
                self.a, self.q, self.v, self.score_q, self.utility_sq,
                self.weight_sum, self.n, self.p, self.player,
            )
        )

    cdef void add_children(self, np.ndarray v, int num_players):
        self._children.extend([Node(a, num_players) for a, valid in enumerate(v) if valid])
        np.random.shuffle(self._children)

    cdef void update_policy(self, float[:] pi):
        cdef Node c
        for c in self._children:
            c.p = pi[c.a]

    cdef float uct(self, float sqrt_parent_n, float fpu_value, float cpuct):
        return (fpu_value if self.n == 0 else self.q) + cpuct * self.p * sqrt_parent_n / (1 + self.n)

    cdef Node best_child(self, float fpu_reduction, float cpuct):
        cdef Node c
        cdef float seen_policy = sum([c.p for c in self._children if c.n > 0])
        cdef float fpu_value = self.v - fpu_reduction * sqrt(seen_policy)
        cdef float cur_best = -float('inf')
        cdef float sqrt_n = sqrt(self.n)
        cdef float uct_value
        cdef Node child = None

        for c in self._children:
            uct_value = c.uct(sqrt_n, fpu_value, cpuct)
            if uct_value > cur_best:
                cur_best = uct_value
                child = c
        return child


cdef class MCTS:
    cdef public float root_noise_frac
    cdef public float root_temp
    cdef public float min_discount
    cdef public float fpu_reduction
    cdef public float cpuct
    cdef public int _num_players
    cdef public Node _root
    cdef public Node _curnode
    cdef public list _path
    cdef public int depth
    cdef public int max_depth
    cdef public int _discount_max_depth

    cdef public object search_utility_mode
    cdef public bint _katago_search
    cdef public bint _force_legacy_search
    cdef public float win_loss_utility_factor
    cdef public float static_score_utility_factor
    cdef public float dynamic_score_utility_factor
    cdef public float dynamic_score_center_zero_weight
    cdef public float dynamic_score_center_scale
    cdef public float cpuct_exploration
    cdef public float cpuct_exploration_log
    cdef public float cpuct_exploration_base
    cdef public float root_fpu_reduction
    cdef public bint fpu_parent_weight_by_visited_policy
    cdef public float fpu_parent_weight_by_visited_policy_pow
    cdef public float root_ending_bonus_points
    cdef public bint fill_dame_before_pass
    cdef public bint conservative_pass

    cdef public float root_dirichlet_noise_total_concentration
    cdef public float root_policy_temperature_early
    cdef public float root_policy_temperature_normal
    cdef public float root_policy_temperature_halflife
    cdef public float root_desired_per_child_visits_coeff
    cdef public float value_weight_exponent
    cdef public bint use_lcb_for_selection
    cdef public float lcb_stdevs
    cdef public float min_visit_prop_for_lcb
    cdef public float chosen_move_subtract
    cdef public float chosen_move_prune

    cdef public int _point_count
    cdef public float _recent_score_center_white
    cdef public float _root_score_white
    cdef public object _root_ownership
    cdef public object _root_ending_bonus_by_action
    cdef public object _root_nn_policy
    cdef public object _root_exploration_policy
    cdef public bint _root_context_ready

    def __init__(self, args: dotdict):
        self.root_noise_frac = args.root_noise_frac
        self.root_temp = args.root_policy_temp
        self.min_discount = args.min_discount
        self.fpu_reduction = args.fpu_reduction
        self.cpuct = args.cpuct
        self._num_players = args._num_players
        self.search_utility_mode = _optional_arg(args, 'search_utility_mode', 'legacy')
        self._katago_search = self.search_utility_mode == KATAGO_PINNED_SEARCH_UTILITY_MODE
        self._force_legacy_search = False

        self.win_loss_utility_factor = float(_optional_arg(args, 'gocube_win_loss_utility_factor', 1.0))
        self.static_score_utility_factor = float(_optional_arg(args, 'gocube_static_score_utility_factor', 0.0))
        self.dynamic_score_utility_factor = float(_optional_arg(args, 'gocube_dynamic_score_utility_factor', 0.30))
        self.dynamic_score_center_zero_weight = float(
            _optional_arg(args, 'gocube_dynamic_score_center_zero_weight', 0.25)
        )
        self.dynamic_score_center_scale = float(_optional_arg(args, 'gocube_dynamic_score_center_scale', 0.50))
        self.cpuct_exploration = float(_optional_arg(args, 'gocube_cpuct_exploration', self.cpuct))
        self.cpuct_exploration_log = float(_optional_arg(args, 'gocube_cpuct_exploration_log', 0.0))
        self.cpuct_exploration_base = float(_optional_arg(args, 'gocube_cpuct_exploration_base', 500.0))
        self.root_fpu_reduction = float(_optional_arg(args, 'gocube_root_fpu_reduction', 0.0))
        self.fpu_parent_weight_by_visited_policy = bool(
            _optional_arg(args, 'gocube_fpu_parent_weight_by_visited_policy', True)
        )
        self.fpu_parent_weight_by_visited_policy_pow = float(
            _optional_arg(args, 'gocube_fpu_parent_weight_by_visited_policy_pow', 2.0)
        )
        self.root_ending_bonus_points = float(_optional_arg(args, 'gocube_root_ending_bonus_points', 0.5))
        self.fill_dame_before_pass = bool(_optional_arg(args, 'gocube_fill_dame_before_pass', True))
        self.conservative_pass = bool(_optional_arg(args, 'gocube_conservative_pass', True))

        exploration = KATAGO_PINNED_EXPLORATION_DEFAULTS
        self.root_dirichlet_noise_total_concentration = float(_optional_arg(
            args, 'gocube_root_dirichlet_noise_total_concentration',
            exploration['root_dirichlet_noise_total_concentration'],
        ))
        self.root_policy_temperature_early = float(_optional_arg(
            args, 'gocube_root_policy_temperature_early', exploration['root_policy_temperature_early']
        ))
        self.root_policy_temperature_normal = float(_optional_arg(
            args, 'gocube_root_policy_temperature', exploration['root_policy_temperature']
        ))
        self.root_policy_temperature_halflife = float(_optional_arg(
            args, 'gocube_root_policy_temperature_halflife', exploration['root_policy_temperature_halflife']
        ))
        self.root_desired_per_child_visits_coeff = float(_optional_arg(
            args, 'gocube_root_desired_per_child_visits_coeff', exploration['root_desired_per_child_visits_coeff']
        ))
        self.value_weight_exponent = float(_optional_arg(
            args, 'gocube_value_weight_exponent', exploration['value_weight_exponent']
        ))
        self.use_lcb_for_selection = bool(_optional_arg(
            args, 'gocube_use_lcb_for_selection', exploration['use_lcb_for_selection']
        ))
        self.lcb_stdevs = float(_optional_arg(args, 'gocube_lcb_stdevs', exploration['lcb_stdevs']))
        self.min_visit_prop_for_lcb = float(_optional_arg(
            args, 'gocube_min_visit_prop_for_lcb', exploration['min_visit_prop_for_lcb']
        ))
        self.chosen_move_subtract = float(_optional_arg(
            args, 'gocube_chosen_move_subtract', exploration['chosen_move_subtract']
        ))
        self.chosen_move_prune = float(_optional_arg(
            args, 'gocube_chosen_move_prune', exploration['chosen_move_prune']
        ))

        if self.value_weight_exponent < 0:
            raise ValueError('gocube_value_weight_exponent must be non-negative')
        if self.lcb_stdevs < 0:
            raise ValueError('gocube_lcb_stdevs must be non-negative')
        if not 0 <= self.min_visit_prop_for_lcb <= 1:
            raise ValueError('gocube_min_visit_prop_for_lcb must be within [0,1]')

        self._root = Node(-1, self._num_players)
        self._curnode = self._root
        self._path = []
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0
        self._point_count = 0
        self._recent_score_center_white = 0
        self._root_score_white = 0
        self._root_ownership = None
        self._root_ending_bonus_by_action = None
        self._root_nn_policy = None
        self._root_exploration_policy = None
        self._root_context_ready = False

    def __repr__(self):
        return (
            'MCTS(root_noise_frac={}, root_temp={}, min_discount={}, fpu_reduction={}, cpuct={}, '
            'search_utility_mode={}, value_weight_exponent={}, use_lcb={}, depth={}, max_depth={})'
        ).format(
            self.root_noise_frac, self.root_temp, self.min_discount, self.fpu_reduction, self.cpuct,
            self.search_utility_mode, self.value_weight_exponent, self.use_lcb_for_selection,
            self.depth, self.max_depth,
        )

    cpdef void reset(self):
        self._root = Node(-1, self._num_players)
        self._curnode = self._root
        self._path = []
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0
        self._point_count = 0
        self._recent_score_center_white = 0
        self._root_score_white = 0
        self._root_ownership = None
        self._root_ending_bonus_by_action = None
        self._root_nn_policy = None
        self._root_exploration_policy = None
        self._root_context_ready = False

    cpdef object search_observation(self, object gs):
        cdef object observation = gs.observation()
        if (
            self._katago_search and not self._force_legacy_search
            and self.conservative_pass and self.depth == 0
        ):
            return conservative_root_observation(gs, observation)
        return observation

    cpdef void search(self, object gs, object nn, int sims, bint add_root_noise, bint add_root_temp):
        cdef float[:] v
        cdef float[:] p
        cdef object out
        self.max_depth = 0
        for _ in range(sims):
            leaf = self.find_leaf(gs)
            if self._katago_search:
                if not hasattr(nn, 'predict_for_search'):
                    raise RuntimeError('KataGo-derived search requires predict_for_search()')
                out = nn.predict_for_search(self.search_observation(leaf))
                if not isinstance(out, SearchOutput):
                    raise RuntimeError('predict_for_search() must return SearchOutput')
                if out.score is None or out.ownership is None:
                    raise RuntimeError('KataGo-derived GoCube search requires score and ownership heads')
                self.process_search_results(
                    leaf, out.value, out.policy, out.score, out.ownership,
                    add_root_noise, add_root_temp,
                )
            else:
                p, v = nn(leaf.observation())
                self.process_results(leaf, v, p, add_root_noise, add_root_temp)

    cpdef void raw_search(self, object gs, int sims, bint add_root_noise, bint add_root_temp):
        cdef Py_ssize_t policy_size = gs.action_size()
        cdef float[:] v = np.zeros(gs.num_players() + 1, dtype=np.float32)
        cdef float[:] p = np.full(policy_size, 1, dtype=np.float32)
        self.max_depth = 0
        self._force_legacy_search = True
        try:
            for _ in range(sims):
                leaf = self.find_leaf(gs)
                self.process_results(leaf, v, p, add_root_noise, add_root_temp)
        finally:
            self._force_legacy_search = False

    cpdef void update_root(self, object gs, int a):
        if not self._root._children:
            self._root.add_children(gs.valid_moves(), self._num_players)
        cdef Node c
        for c in self._root._children:
            if c.a == a:
                if self._katago_search:
                    # Dynamic score utility is root-centered. Reset rather than
                    # retaining statistics centered on the old root.
                    self.reset()
                else:
                    self._root = c
                return
        raise ValueError(f'Invalid action encountered while updating root: {a}')

    cdef np.ndarray _root_policy_array(self, int action_size):
        cdef np.ndarray policy = np.zeros(action_size, dtype=np.float64)
        cdef Node c
        for c in self._root._children:
            policy[c.a] = c.p
        return policy

    cpdef void _add_root_noise(self):
        cdef int num_valid_moves = len(self._root._children)
        cdef Node c
        cdef double n
        cdef object alpha_props
        cdef object alpha
        cdef object policy
        cdef object noise
        if num_valid_moves <= 0:
            return
        if self._katago_search and not self._force_legacy_search:
            policy = np.asarray([c.p for c in self._root._children], dtype=np.float64)
            alpha_props = shaped_dirichlet_alpha_distribution(policy)
            alpha = np.asarray(alpha_props, dtype=np.float64) * self.root_dirichlet_noise_total_concentration
            noise = np.random.dirichlet(alpha)
        else:
            noise = np.random.dirichlet([NOISE_ALPHA_RATIO / num_valid_moves] * num_valid_moves)
        for n, c in zip(noise, self._root._children):
            c.p = c.p * (1 - self.root_noise_frac) + self.root_noise_frac * n

    cdef float _katago_root_policy_temp(self, object gs):
        return float(root_policy_temperature(
            int(gs.turns), int(gs.logical_topology().point_count),
            early_temperature=self.root_policy_temperature_early,
            temperature=self.root_policy_temperature_normal,
            halflife=self.root_policy_temperature_halflife,
        ))

    cdef float _katago_fpu_value(self, Node parent, bint is_root, float visited_policy):
        cdef float parent_utility = parent.q
        cdef float parent_nn_utility = parent.v
        cdef float parent_for_fpu = parent_utility
        cdef float avg_weight
        cdef float reduction
        if self.fpu_parent_weight_by_visited_policy:
            avg_weight = visited_policy ** self.fpu_parent_weight_by_visited_policy_pow
            if avg_weight > 1.0:
                avg_weight = 1.0
            parent_for_fpu = avg_weight * parent_utility + (1.0 - avg_weight) * parent_nn_utility
        reduction = (self.root_fpu_reduction if is_root else self.fpu_reduction) * sqrt(visited_policy)
        if parent.player == 1:
            return parent_for_fpu - reduction
        return parent_for_fpu + reduction

    cdef float _katago_explore_scaling(self, float total_child_weight):
        cdef float cpuct_value = self.cpuct_exploration
        if self.cpuct_exploration_log != 0:
            cpuct_value += self.cpuct_exploration_log * log(
                (total_child_weight + self.cpuct_exploration_base) / self.cpuct_exploration_base
            )
        return cpuct_value * sqrt(total_child_weight + _TOTAL_CHILD_WEIGHT_PUCT_OFFSET)

    cdef float _katago_root_child_utility(self, Node c):
        cdef float child_utility = c.q
        cdef float ending_bonus
        if self._root_context_ready and self._root_ending_bonus_by_action is not None and c.has_score:
            ending_bonus = float(self._root_ending_bonus_by_action[c.a])
            if ending_bonus != 0:
                child_utility += float(score_utility_diff(
                    c.score_q, ending_bonus,
                    recent_center=self._recent_score_center_white,
                    point_count=self._point_count,
                    static_factor=self.static_score_utility_factor,
                    dynamic_factor=self.dynamic_score_utility_factor,
                    dynamic_scale=self.dynamic_score_center_scale,
                ))
        return child_utility

    cdef Node _best_child_katago(self, Node parent, object gs, bint is_root):
        cdef Node c
        cdef Node child = None
        cdef float visited_policy = 0.0
        cdef float total_child_weight = 0.0
        cdef float child_weight
        cdef float fpu
        cdef float explore_scaling
        cdef float child_utility
        cdef float value_component
        cdef float selection_value
        cdef float cur_best = -float('inf')
        cdef float ending_bonus
        cdef float score_for_bonus

        for c in parent._children:
            if c.n > 0:
                visited_policy += c.p
                total_child_weight += c.weight_sum

        fpu = self._katago_fpu_value(parent, is_root, visited_policy)
        explore_scaling = self._katago_explore_scaling(total_child_weight)
        for c in parent._children:
            child_weight = c.weight_sum if c.n > 0 else 0.0
            child_utility = fpu if c.n == 0 else c.q
            if is_root and self._root_context_ready and self._root_ending_bonus_by_action is not None:
                ending_bonus = float(self._root_ending_bonus_by_action[c.a])
                if ending_bonus != 0:
                    score_for_bonus = self._root_score_white if c.n == 0 else c.score_q
                    child_utility += float(score_utility_diff(
                        score_for_bonus, ending_bonus,
                        recent_center=self._recent_score_center_white,
                        point_count=self._point_count,
                        static_factor=self.static_score_utility_factor,
                        dynamic_factor=self.dynamic_score_utility_factor,
                        dynamic_scale=self.dynamic_score_center_scale,
                    ))
            value_component = child_utility if parent.player == 1 else -child_utility
            if (
                is_root and self.root_desired_per_child_visits_coeff > 0.0
                and c.p > 0.0
                and child_weight < sqrt(c.p * total_child_weight * self.root_desired_per_child_visits_coeff)
            ):
                selection_value = 1e20
            else:
                selection_value = value_component + explore_scaling * c.p / (1.0 + child_weight)
            if selection_value > cur_best:
                cur_best = selection_value
                child = c
        return child

    cpdef object find_leaf(self, object gs):
        self.depth = 0
        self._curnode = self._root
        self._path = []
        cdef object leaf = gs.clone()
        cdef Node next_child
        cdef bint use_katago = self._katago_search and not self._force_legacy_search
        while self._curnode.n > 0 and not self._curnode.e.any():
            self._path.append(self._curnode)
            if use_katago:
                next_child = self._best_child_katago(self._curnode, leaf, self._curnode == self._root)
            else:
                next_child = self._curnode.best_child(self.fpu_reduction, self.cpuct)
            if next_child is None:
                raise RuntimeError('MCTS could not select a child')
            self._curnode = next_child
            leaf.play_action(self._curnode.a)
            self.depth += 1
        if self.depth > self.max_depth:
            self.max_depth = self.depth
            self._discount_max_depth = self.depth
        if self._curnode.n == 0:
            self._curnode.player = leaf.player
            self._curnode.e = leaf.win_state()
            self._curnode.add_children(leaf.valid_moves(), self._num_players)
        return leaf

    cpdef void process_results(self, object gs, float[:] value, float[:] pi, bint add_root_noise, bint add_root_temp):
        cdef float[:] valids
        cdef Node c
        if self._curnode.e.any():
            value = np.array(self._curnode.e, dtype=np.float32)
        else:
            valids = np.zeros(gs.action_size(), dtype=np.float32)
            for c in self._curnode._children:
                valids[c.a] = 1
            pi = normalize_masked_policy(np.asarray(pi), np.asarray(valids))
            if self._curnode == self._root:
                if add_root_temp:
                    pi = np.asarray(pi) ** (1.0 / self.root_temp)
                    pi /= np.sum(pi)
                self._curnode.update_policy(pi)
                if add_root_noise:
                    self._add_root_noise()
            else:
                self._curnode.update_policy(pi)

        cdef Py_ssize_t num_players = gs.num_players()
        cdef Node parent
        cdef float v
        cdef float discount
        cdef int i = 0
        if self._curnode.n == 0:
            self._curnode.v = self._get_value(value, self._curnode.player, num_players)
        while self._path:
            parent = self._path.pop()
            v = self._get_value(value, parent.player, num_players)
            discount = (self.min_discount ** (i / self._discount_max_depth))
            if v < _DRAW_VALUE:
                discount = 2 - discount
            elif v == _DRAW_VALUE:
                discount = 1
            self._curnode.q = (self._curnode.q * self._curnode.n + v * discount) / (self._curnode.n + 1)
            self._curnode.n += 1
            self._curnode = parent
            i += 1
        self._root.n += 1

    cdef float _normal_cdf(self, float z):
        return 0.5 * (1.0 + erf(z / _SQRT_TWO))

    cdef float _katago_child_value_factor(self, float child_weight, float child_self_utility, float simple_value):
        cdef float precision
        cdef float stdev
        cdef float z
        cdef float probability
        if self.value_weight_exponent == 0.0 or child_weight <= 0.0:
            return 1.0
        precision = 1.5 * sqrt(child_weight)
        stdev = sqrt(0.00000001 + 1.0 / precision)
        z = (child_self_utility - simple_value) / stdev
        probability = self._normal_cdf(z) + 0.0001
        return pow(probability, self.value_weight_exponent)

    cdef void _init_katago_node(self, Node node, float utility, float white_score, bint score_available):
        node.v = utility
        node.q = utility
        node.utility_sq = utility * utility
        node.weight_sum = 1.0
        node.weight_sq_sum = 1.0
        node.has_score = score_available
        if score_available:
            node.score_v = white_score
            node.score_q = white_score
        node.n = 1

    cdef void _append_terminal_katago_node(self, Node node, float utility, float white_score, bint score_available):
        cdef float old_weight = node.weight_sum
        cdef float new_weight = old_weight + 1.0
        if node.n <= 0 or old_weight <= 0:
            self._init_katago_node(node, utility, white_score, score_available)
            return
        node.q = (node.q * old_weight + utility) / new_weight
        node.utility_sq = (node.utility_sq * old_weight + utility * utility) / new_weight
        if score_available:
            if node.has_score:
                node.score_q = (node.score_q * old_weight + white_score) / new_weight
            else:
                node.score_q = white_score
                node.score_v = white_score
                node.has_score = True
        node.weight_sum = new_weight
        node.weight_sq_sum += 1.0
        node.n += 1

    cdef void _recompute_katago_node(self, Node node):
        """Pinned KataGo child-value aggregation specialized to a single-thread tree."""
        cdef Node c
        cdef float raw_child_weight
        cdef float total_child_weight = 0.0
        cdef float simple_value_sum = 0.0
        cdef float child_self_utility
        cdef float simple_value
        cdef float factor
        cdef float adjusted_sum = 0.0
        cdef float normalize = 1.0
        cdef float desired_weight
        cdef float child_scale
        cdef float utility_sum = node.v
        cdef float utility_sq_sum = node.v * node.v
        cdef float weight_sum = 1.0
        cdef float weight_sq_sum = 1.0
        cdef float score_sum = node.score_v if node.has_score else 0.0
        cdef float score_weight = 1.0 if node.has_score else 0.0

        for c in node._children:
            if c.n <= 0 or c.weight_sum <= 0.0:
                continue
            raw_child_weight = c.weight_sum
            child_self_utility = c.q if node.player == 1 else -c.q
            total_child_weight += raw_child_weight
            simple_value_sum += child_self_utility * raw_child_weight

        if total_child_weight <= 0.0:
            node.q = node.v
            node.utility_sq = node.v * node.v
            node.weight_sum = 1.0
            node.weight_sq_sum = 1.0
            if node.has_score:
                node.score_q = node.score_v
            return

        simple_value = simple_value_sum / total_child_weight
        for c in node._children:
            if c.n <= 0 or c.weight_sum <= 0.0:
                continue
            raw_child_weight = c.weight_sum
            child_self_utility = c.q if node.player == 1 else -c.q
            factor = self._katago_child_value_factor(raw_child_weight, child_self_utility, simple_value)
            adjusted_sum += raw_child_weight * factor
        if adjusted_sum > 0.0:
            normalize = total_child_weight / adjusted_sum

        for c in node._children:
            if c.n <= 0 or c.weight_sum <= 0.0:
                continue
            raw_child_weight = c.weight_sum
            child_self_utility = c.q if node.player == 1 else -c.q
            factor = self._katago_child_value_factor(raw_child_weight, child_self_utility, simple_value)
            desired_weight = raw_child_weight * factor * normalize
            utility_sum += desired_weight * c.q
            utility_sq_sum += desired_weight * c.utility_sq
            weight_sum += desired_weight
            child_scale = desired_weight / raw_child_weight
            weight_sq_sum += child_scale * child_scale * c.weight_sq_sum
            if node.has_score and c.has_score:
                score_sum += desired_weight * c.score_q
                score_weight += desired_weight

        node.q = utility_sum / weight_sum
        node.utility_sq = utility_sq_sum / weight_sum
        node.weight_sum = weight_sum
        node.weight_sq_sum = weight_sq_sum
        if node.has_score and score_weight > 0.0:
            node.score_q = score_sum / score_weight

    cpdef void process_search_results(
        self, object gs, object value, object pi, object score, object ownership,
        bint add_root_noise, bint add_root_temp,
    ):
        cdef np.ndarray valids
        cdef Node c
        cdef Node parent
        cdef object terminal
        cdef object exact_score
        cdef np.ndarray score_arr
        cdef np.ndarray bonus_arr
        cdef float white_score = 0.0
        cdef float utility
        cdef float root_temp
        cdef bint score_available = True
        cdef bint was_terminal = self._curnode.e.any()

        if not self._katago_search:
            raise RuntimeError('process_search_results called outside KataGo-derived search mode')
        self._point_count = int(gs.logical_topology().point_count)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        pi = np.asarray(pi, dtype=np.float32).reshape(-1)

        if was_terminal:
            value = np.asarray(self._curnode.e, dtype=np.float32)
            terminal = getattr(gs, 'terminal_adjudication', None)
            exact_score = getattr(terminal, 'score', None) if terminal is not None else None
            if exact_score is None:
                score_available = False
            else:
                white_score = float(exact_score.white) - float(exact_score.black)
        else:
            score_arr = np.asarray(score, dtype=np.float32).reshape(-1)
            if score_arr.size != 1:
                raise ValueError(f'GoCube score head must contain one value, got size={score_arr.size}')
            white_score = float(normalized_black_minus_white_to_white_score(float(score_arr[0]), self._point_count))
            valids = np.zeros(gs.action_size(), dtype=np.float32)
            for c in self._curnode._children:
                valids[c.a] = 1
            pi = normalize_masked_policy(pi, valids)
            if self._curnode == self._root:
                self._root_nn_policy = np.array(pi, dtype=np.float64, copy=True)
                if add_root_temp:
                    root_temp = self._katago_root_policy_temp(gs)
                    pi = np.asarray(pi) ** (1.0 / root_temp)
                    pi /= np.sum(pi)
                self._curnode.update_policy(pi)
                self._root_ownership = np.asarray(ownership, dtype=np.float32)
                if self._root_ownership.shape != (self._point_count, 3):
                    raise ValueError(
                        f'GoCube ownership head shape {self._root_ownership.shape} != ({self._point_count}, 3)'
                    )
                white_owner_map(self._root_ownership)
                self._root_score_white = white_score
                self._recent_score_center_white = float(recent_score_center(
                    white_score,
                    zero_weight=self.dynamic_score_center_zero_weight,
                    center_scale=self.dynamic_score_center_scale,
                    point_count=self._point_count,
                ))
                bonus_arr = np.asarray(root_ending_white_score_bonuses(
                    gs, self._root_ownership, self.root_ending_bonus_points
                ), dtype=np.float32).reshape(-1)
                if bonus_arr.size != gs.action_size():
                    raise ValueError(f'root-ending bonus size {bonus_arr.size} != {gs.action_size()}')
                self._root_ending_bonus_by_action = bonus_arr
                self._root_context_ready = True
                if add_root_noise:
                    self._add_root_noise()
                self._root_exploration_policy = self._root_policy_array(gs.action_size())
            else:
                self._curnode.update_policy(pi)

        utility = float(combined_white_utility(
            value, white_score if score_available else None,
            recent_center=self._recent_score_center_white,
            point_count=self._point_count,
            win_loss_factor=self.win_loss_utility_factor,
            static_score_factor=self.static_score_utility_factor,
            dynamic_score_factor=self.dynamic_score_utility_factor,
            dynamic_score_scale=self.dynamic_score_center_scale,
        ))

        if was_terminal and self._curnode.n > 0:
            self._append_terminal_katago_node(self._curnode, utility, white_score, score_available)
        else:
            self._init_katago_node(self._curnode, utility, white_score, score_available)

        while self._path:
            parent = self._path.pop()
            parent.n += 1
            self._recompute_katago_node(parent)
            self._curnode = parent

    cpdef float _get_value(self, float[:] value, Py_ssize_t player, Py_ssize_t num_players):
        if value.size > num_players:
            return value[player] + value[num_players] / num_players
        return value[player]

    cpdef int[:] raw_counts(self, object gs):
        cdef int[:] counts = np.zeros(gs.action_size(), dtype=np.int32)
        cdef Node c
        for c in self._root._children:
            counts[c.a] = c.n
        return np.asarray(counts)

    cdef bint _should_suppress_pass(self, object gs):
        cdef Node c
        cdef Node pass_node = None
        cdef int pass_action
        cdef float pass_weight
        cdef float pass_utility
        cdef float pass_score
        cdef float child_weight
        cdef float pla_ownership
        cdef bint opp_owned
        cdef bint adj_to_pla_owned
        cdef object owner
        cdef object topology
        cdef object state
        cdef int player
        cdef int neighbor
        if not self._katago_search or not self.fill_dame_before_pass or not self._root_context_ready:
            return False
        state = getattr(gs, 'semantic_state', None)
        if state is None or getattr(state, 'phase', None) != 'main':
            return False
        pass_action = int(gs.pass_action())
        for c in self._root._children:
            if c.a == pass_action:
                pass_node = c
                break
        if pass_node is None or pass_node.n <= 0:
            return False
        pass_weight = float(pass_node.weight_sum)
        pass_utility = pass_node.q
        pass_score = pass_node.score_q
        owner = white_owner_map(self._root_ownership)
        topology = gs.logical_topology()
        player = int(gs.player)
        for c in self._root._children:
            if c.a == pass_action or c.n <= 0:
                continue
            pla_ownership = float(owner[c.a]) if player == 1 else -float(owner[c.a])
            opp_owned = pla_ownership < -0.95
            adj_to_pla_owned = False
            for neighbor in topology.neighbor_indices(c.a):
                if player == 1:
                    if float(owner[int(neighbor)]) > 0.95:
                        adj_to_pla_owned = True
                        break
                elif -float(owner[int(neighbor)]) > 0.95:
                    adj_to_pla_owned = True
                    break
            if opp_owned and not adj_to_pla_owned:
                continue
            child_weight = float(c.weight_sum)
            if ((c.n <= 500 and child_weight <= 2.0 * sqrt(pass_weight)) or child_weight <= 1e-10):
                continue
            if player == 1 and c.q > pass_utility - 0.1 and c.score_q > pass_score - 0.5:
                return True
            if player == 0 and c.q < pass_utility + 0.1 and c.score_q < pass_score + 0.5:
                return True
        return False

    cdef np.ndarray _katago_corrected_counts(self, object gs, np.ndarray search_counts):
        cdef np.ndarray policy = self._root_policy_array(gs.action_size())
        cdef np.ndarray utilities = np.zeros(gs.action_size(), dtype=np.float64)
        cdef np.ndarray legal = np.zeros(gs.action_size(), dtype=np.bool_)
        cdef float total_child_weight = 0.0
        cdef Node c
        for c in self._root._children:
            legal[c.a] = True
            total_child_weight += c.n
            if c.n > 0:
                utilities[c.a] = self._katago_root_child_utility(c)
        return np.asarray(retrospectively_reduce_root_visits(
            search_counts, policy, utilities,
            root_player=int(self._root.player),
            explore_scaling=float(self._katago_explore_scaling(total_child_weight)),
            legal_mask=legal,
        ), dtype=np.int32)

    cpdef int[:] counts(self, object gs):
        cdef np.ndarray counts = np.asarray(self.raw_counts(gs), dtype=np.int32).copy()
        if self._should_suppress_pass(gs):
            counts[int(gs.pass_action())] = 0
        if self._katago_search and not self._force_legacy_search:
            counts = self._katago_corrected_counts(gs, counts)
        return counts

    cdef tuple _katago_play_selection_detail(self, object gs, bint apply_lcb):
        cdef int action_size = gs.action_size()
        cdef np.ndarray raw_weights = np.zeros(action_size, dtype=np.float64)
        cdef np.ndarray edge_visits = np.zeros(action_size, dtype=np.float64)
        cdef np.ndarray policy = self._root_policy_array(action_size)
        cdef np.ndarray utilities = np.zeros(action_size, dtype=np.float64)
        cdef np.ndarray utility_sq = np.zeros(action_size, dtype=np.float64)
        cdef np.ndarray weight_sum = np.zeros(action_size, dtype=np.float64)
        cdef np.ndarray weight_sq_sum = np.zeros(action_size, dtype=np.float64)
        cdef np.ndarray ending_diffs = np.zeros(action_size, dtype=np.float64)
        cdef np.ndarray legal = np.zeros(action_size, dtype=np.bool_)
        cdef float total_child_weight = 0.0
        cdef float ending_bonus
        cdef float utility_radius
        cdef Node c
        cdef object lcb_result
        cdef np.ndarray reduced
        cdef np.ndarray final_values
        cdef object lcbs = None
        cdef object radii = None
        cdef object best_lcb_idx = None

        for c in self._root._children:
            legal[c.a] = True
            edge_visits[c.a] = c.n
            if c.n <= 0:
                continue
            raw_weights[c.a] = c.weight_sum
            weight_sum[c.a] = c.weight_sum
            weight_sq_sum[c.a] = c.weight_sq_sum
            utilities[c.a] = c.q
            utility_sq[c.a] = c.utility_sq
            total_child_weight += c.weight_sum
            if self._root_context_ready and self._root_ending_bonus_by_action is not None and c.has_score:
                ending_bonus = float(self._root_ending_bonus_by_action[c.a])
                if ending_bonus != 0:
                    ending_diffs[c.a] = float(score_utility_diff(
                        c.score_q, ending_bonus,
                        recent_center=self._recent_score_center_white,
                        point_count=self._point_count,
                        static_factor=self.static_score_utility_factor,
                        dynamic_factor=self.dynamic_score_utility_factor,
                        dynamic_scale=self.dynamic_score_center_scale,
                    ))

        if self._should_suppress_pass(gs):
            raw_weights[int(gs.pass_action())] = 0.0

        reduced = np.asarray(retrospectively_reduce_root_weights(
            raw_weights, policy, utilities + ending_diffs,
            root_player=int(self._root.player),
            explore_scaling=float(self._katago_explore_scaling(total_child_weight)),
            legal_mask=legal,
            edge_visits=edge_visits,
        ), dtype=np.float64)
        final_values = reduced.copy()

        if apply_lcb and self.use_lcb_for_selection:
            utility_radius = (
                self.win_loss_utility_factor + self.static_score_utility_factor + self.dynamic_score_utility_factor
            )
            lcb_result = apply_lcb_play_selection(
                final_values, edge_visits, policy, utilities, utility_sq,
                weight_sum, weight_sq_sum, ending_diffs,
                root_player=int(self._root.player),
                utility_range_radius=float(utility_radius),
                lcb_stdevs=float(self.lcb_stdevs),
                min_visit_prop_for_lcb=float(self.min_visit_prop_for_lcb),
            )
            final_values = np.asarray(lcb_result[0], dtype=np.float64)
            lcbs, radii, best_lcb_idx = lcb_result[1], lcb_result[2], lcb_result[3]

        final_values = np.asarray(apply_chosen_move_pruning(
            final_values,
            subtract=float(self.chosen_move_subtract),
            prune=float(self.chosen_move_prune),
        ), dtype=np.float64)
        if float(np.sum(final_values)) <= 0.0:
            final_values = np.where(legal, np.maximum(policy, 0.0), 0.0)
        return final_values, reduced, lcbs, radii, best_lcb_idx

    cpdef object root_search_telemetry(self, object gs):
        cdef np.ndarray raw = np.asarray(self.raw_counts(gs), dtype=np.int32).copy()
        cdef np.ndarray search_counts = raw.copy()
        cdef bint pass_suppressed = self._should_suppress_pass(gs)
        cdef int pass_action = int(gs.pass_action())
        cdef int pass_suppressed_visits = 0
        cdef np.ndarray corrected
        cdef np.ndarray forced
        cdef tuple detail
        cdef np.ndarray target_values
        cdef np.ndarray target
        cdef float total
        if pass_suppressed:
            pass_suppressed_visits = int(search_counts[pass_action])
            search_counts[pass_action] = 0
        corrected = self._katago_corrected_counts(gs, search_counts) if self._katago_search else search_counts
        forced = np.maximum(search_counts - corrected, 0).astype(np.int32)

        if self._katago_search and not self._force_legacy_search:
            detail = self._katago_play_selection_detail(gs, True)
            target_values = np.asarray(detail[0], dtype=np.float64)
        else:
            detail = (corrected.astype(np.float64), corrected.astype(np.float64), None, None, None)
            target_values = np.asarray(detail[0], dtype=np.float64)
        total = float(np.sum(target_values))
        target = target_values / total if total > 0.0 else target_values
        return {
            'nn_root_policy': (
                np.asarray(self._root_nn_policy, dtype=np.float64).tolist()
                if self._root_nn_policy is not None else None
            ),
            'exploration_policy': (
                np.asarray(self._root_exploration_policy, dtype=np.float64).tolist()
                if self._root_exploration_policy is not None else self._root_policy_array(gs.action_size()).tolist()
            ),
            'root_visit_counts': raw.tolist(),
            'policy_training_target': target.tolist(),
            'forced_exploration_visits': forced.tolist(),
            'forced_exploration_visit_total': int(np.sum(forced)),
            'pass_suppressed_visits': pass_suppressed_visits,
            'play_selection_pre_lcb': np.asarray(detail[1], dtype=np.float64).tolist(),
            'play_selection_post_lcb': target_values.tolist(),
            'lcb_values': None if detail[2] is None else np.asarray(detail[2], dtype=np.float64).tolist(),
            'lcb_radii': None if detail[3] is None else np.asarray(detail[3], dtype=np.float64).tolist(),
            'lcb_best_action': None if detail[4] is None else int(detail[4]),
        }

    cpdef int best_action(self, object gs):
        if self._katago_search and not self._force_legacy_search:
            return int(np.argmax(self._katago_play_selection_detail(gs, True)[0]))
        return int(np.argmax(self.counts(gs)))

    cpdef np.ndarray probs(self, object gs, float temp=1.0, object apply_lcb=None):
        cdef np.ndarray weights
        cdef np.ndarray probs
        cdef Py_ssize_t best_action
        cdef bint use_lcb
        if self._katago_search and not self._force_legacy_search:
            if apply_lcb is None:
                # Pinned self-play move temperatures are < 1, while policy
                # supervision uses temp=1 and deterministic Arena uses temp=0.
                # This reproduces KataGo's self-play-only LCB disable for the
                # played move while keeping LCB in the training target/Arena.
                use_lcb = temp <= 1.0e-8 or abs(temp - 1.0) <= 1.0e-8
            else:
                use_lcb = bool(apply_lcb)
            weights = np.asarray(self._katago_play_selection_detail(gs, use_lcb)[0], dtype=np.float64)
        else:
            weights = np.asarray(self.counts(gs), dtype=np.float64)

        if temp == 0:
            best_action = int(np.argmax(weights))
            probs = np.zeros_like(weights, dtype=np.float32)
            probs[best_action] = 1.0
            return probs
        if float(np.sum(weights)) <= 0.0:
            probs = np.asarray(gs.valid_moves(), dtype=np.float64)
            probs /= np.sum(probs)
            return probs.astype(np.float32)
        try:
            probs = (weights / np.sum(weights)) ** (1.0 / temp)
            probs /= np.sum(probs)
            return probs.astype(np.float32)
        except (OverflowError, FloatingPointError):
            best_action = int(np.argmax(weights))
            probs = np.zeros_like(weights, dtype=np.float32)
            probs[best_action] = 1.0
            return probs

    cpdef float value(self, bint average=False):
        cdef float value = 0
        cdef Node c
        cdef float utility_radius
        cdef float perspective
        if self._katago_search:
            if self._root.n <= 0:
                return 0.5
            utility_radius = self.win_loss_utility_factor + self.static_score_utility_factor + self.dynamic_score_utility_factor
            if utility_radius < 1e-6:
                utility_radius = 1e-6
            perspective = self._root.q if self._root.player == 1 else -self._root.q
            value = 0.5 + 0.5 * perspective / utility_radius
            if value < 0:
                return 0
            if value > 1:
                return 1
            return value
        if average:
            value = sum([c.q for c in self._root._children if c.n > 0]) / len(self._root._children)
        else:
            for c in self._root._children:
                if c.q > value and c.n > 0:
                    value = c.q
        return value
