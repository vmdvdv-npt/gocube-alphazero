# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from libc.math cimport sqrt, log

import numpy as np
cimport numpy as np
from alphazero.utils import dotdict
from alphazero.mcts_policy import normalize_masked_policy
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

np.seterr(all='raise')


cdef class Node:
    cdef public list _children
    cdef public int a
    cdef public np.ndarray e
    cdef public float q
    cdef public float v
    cdef public float score_q
    cdef public float score_v
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
        self.n = 0
        self.p = 0
        self.player = 0

    def __repr__(self):
        return 'Node(a={}, e={}, q={}, v={}, score_q={}, n={}, p={}, player={})' \
            .format(self.a, self.e, self.q, self.v, self.score_q, self.n, self.p, self.player)

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
        cdef float uct
        child = None

        for c in self._children:
            uct = c.uct(sqrt_n, fpu_value, cpuct)
            if uct > cur_best:
                cur_best = uct
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
    cdef public int _point_count
    cdef public float _recent_score_center_white
    cdef public float _root_score_white
    cdef public object _root_ownership
    cdef public object _root_ending_bonus_by_action
    cdef public bint _root_context_ready

    def __init__(self, args: dotdict):
        self.root_noise_frac = args.root_noise_frac
        self.root_temp = args.root_policy_temp
        self.min_discount = args.min_discount
        self.fpu_reduction = args.fpu_reduction
        self.cpuct = args.cpuct
        self._num_players = args._num_players
        self.search_utility_mode = getattr(args, 'search_utility_mode', 'legacy')
        self._katago_search = self.search_utility_mode == KATAGO_PINNED_SEARCH_UTILITY_MODE
        self._force_legacy_search = False

        self.win_loss_utility_factor = float(getattr(args, 'gocube_win_loss_utility_factor', 1.0))
        self.static_score_utility_factor = float(getattr(args, 'gocube_static_score_utility_factor', 0.0))
        self.dynamic_score_utility_factor = float(getattr(args, 'gocube_dynamic_score_utility_factor', 0.30))
        self.dynamic_score_center_zero_weight = float(
            getattr(args, 'gocube_dynamic_score_center_zero_weight', 0.25)
        )
        self.dynamic_score_center_scale = float(getattr(args, 'gocube_dynamic_score_center_scale', 0.50))
        self.cpuct_exploration = float(getattr(args, 'gocube_cpuct_exploration', self.cpuct))
        self.cpuct_exploration_log = float(getattr(args, 'gocube_cpuct_exploration_log', 0.0))
        self.cpuct_exploration_base = float(getattr(args, 'gocube_cpuct_exploration_base', 500.0))
        self.root_fpu_reduction = float(getattr(args, 'gocube_root_fpu_reduction', 0.0))
        self.fpu_parent_weight_by_visited_policy = bool(
            getattr(args, 'gocube_fpu_parent_weight_by_visited_policy', True)
        )
        self.fpu_parent_weight_by_visited_policy_pow = float(
            getattr(args, 'gocube_fpu_parent_weight_by_visited_policy_pow', 2.0)
        )
        self.root_ending_bonus_points = float(getattr(args, 'gocube_root_ending_bonus_points', 0.5))
        self.fill_dame_before_pass = bool(getattr(args, 'gocube_fill_dame_before_pass', True))
        self.conservative_pass = bool(getattr(args, 'gocube_conservative_pass', True))

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
        self._root_context_ready = False

    def __repr__(self):
        return (
            'MCTS(root_noise_frac={}, root_temp={}, min_discount={}, fpu_reduction={}, cpuct={}, '
            'search_utility_mode={}, _num_players={}, _root={}, depth={}, max_depth={})'
        ).format(
            self.root_noise_frac, self.root_temp, self.min_discount,
            self.fpu_reduction, self.cpuct, self.search_utility_mode,
            self._num_players, self._root, self.depth, self.max_depth,
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
        self._root_context_ready = False

    cpdef object search_observation(self, object gs):
        cdef object observation = gs.observation()
        if (
            self._katago_search
            and not self._force_legacy_search
            and self.conservative_pass
            and self.depth == 0
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
                    # Dynamic score utility is centered on the current root. A
                    # faithful tree-reuse port would need to re-center retained
                    # subtree statistics. Reset rather than mixing utilities
                    # computed under two different score centers.
                    self.reset()
                else:
                    self._root = c
                return

        raise ValueError(f'Invalid action encountered while updating root: {a}')

    cpdef void _add_root_noise(self):
        cdef int num_valid_moves = len(self._root._children)
        cdef double[:] noise = np.random.dirichlet(
            [NOISE_ALPHA_RATIO / num_valid_moves] * num_valid_moves
        )
        cdef Node c
        cdef double n

        for n, c in zip(noise, self._root._children):
            c.p = c.p * (1 - self.root_noise_frac) + self.root_noise_frac * n

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

    cdef Node _best_child_katago(self, Node parent, object gs, bint is_root):
        cdef Node c
        cdef Node child = None
        cdef float visited_policy = 0.0
        cdef float total_child_weight = 0.0
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
            total_child_weight += c.n

        fpu = self._katago_fpu_value(parent, is_root, visited_policy)
        explore_scaling = self._katago_explore_scaling(total_child_weight)

        for c in parent._children:
            child_utility = fpu if c.n == 0 else c.q
            if is_root and self._root_context_ready and self._root_ending_bonus_by_action is not None:
                ending_bonus = float(self._root_ending_bonus_by_action[c.a])
                if ending_bonus != 0:
                    score_for_bonus = self._root_score_white if c.n == 0 else c.score_q
                    child_utility += float(score_utility_diff(
                        score_for_bonus,
                        ending_bonus,
                        recent_center=self._recent_score_center_white,
                        point_count=self._point_count,
                        static_factor=self.static_score_utility_factor,
                        dynamic_factor=self.dynamic_score_utility_factor,
                        dynamic_scale=self.dynamic_score_center_scale,
                    ))

            value_component = child_utility if parent.player == 1 else -child_utility
            selection_value = value_component + explore_scaling * c.p / (1.0 + c.n)
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
                next_child = self._best_child_katago(
                    self._curnode, leaf, self._curnode == self._root
                )
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

    cdef void _update_katago_node(self, Node node, float utility, float white_score, bint score_available):
        node.q = (node.q * node.n + utility) / (node.n + 1)
        if score_available:
            node.score_q = (node.score_q * node.n + white_score) / (node.n + 1)
        node.n += 1

    cpdef void process_search_results(
        self,
        object gs,
        object value,
        object pi,
        object score,
        object ownership,
        bint add_root_noise,
        bint add_root_temp,
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
        cdef bint score_available = True

        if not self._katago_search:
            raise RuntimeError('process_search_results called outside KataGo-derived search mode')

        self._point_count = int(gs.logical_topology().point_count)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        pi = np.asarray(pi, dtype=np.float32).reshape(-1)

        if self._curnode.e.any():
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
                raise ValueError(f'GoCube score head must contain one value, got {score_arr.shape}')
            white_score = float(normalized_black_minus_white_to_white_score(
                float(score_arr[0]), self._point_count
            ))

            valids = np.zeros(gs.action_size(), dtype=np.float32)
            for c in self._curnode._children:
                valids[c.a] = 1
            pi = normalize_masked_policy(pi, valids)

            if self._curnode == self._root:
                if add_root_temp:
                    pi = np.asarray(pi) ** (1.0 / self.root_temp)
                    pi /= np.sum(pi)
                self._curnode.update_policy(pi)
                self._root_ownership = np.asarray(ownership, dtype=np.float32)
                if self._root_ownership.shape != (self._point_count, 3):
                    raise ValueError(
                        f'GoCube ownership head shape {self._root_ownership.shape} '
                        f'!= ({self._point_count}, 3)'
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
                if bonus_arr.shape != (gs.action_size(),):
                    raise ValueError(
                        f'root-ending bonus shape {bonus_arr.shape} != ({gs.action_size()},)'
                    )
                self._root_ending_bonus_by_action = bonus_arr
                self._root_context_ready = True
                if add_root_noise:
                    self._add_root_noise()
            else:
                self._curnode.update_policy(pi)

        utility = float(combined_white_utility(
            value,
            white_score if score_available else None,
            recent_center=self._recent_score_center_white,
            point_count=self._point_count,
            win_loss_factor=self.win_loss_utility_factor,
            static_score_factor=self.static_score_utility_factor,
            dynamic_score_factor=self.dynamic_score_utility_factor,
            dynamic_score_scale=self.dynamic_score_center_scale,
        ))

        if self._curnode.n == 0:
            self._curnode.v = utility
            if score_available:
                self._curnode.score_v = white_score

        self._update_katago_node(self._curnode, utility, white_score, score_available)
        while self._path:
            parent = self._path.pop()
            self._update_katago_node(parent, utility, white_score, score_available)
            self._curnode = parent

    cpdef float _get_value(self, float[:] value, Py_ssize_t player, Py_ssize_t num_players):
        if value.size > num_players:
            return value[player] + value[num_players] / num_players
        else:
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

        pass_weight = float(pass_node.n)
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
                else:
                    if -float(owner[int(neighbor)]) > 0.95:
                        adj_to_pla_owned = True
                        break
            if opp_owned and not adj_to_pla_owned:
                continue

            child_weight = float(c.n)
            if ((c.n <= 500 and child_weight <= 2.0 * sqrt(pass_weight)) or child_weight <= 1e-10):
                continue

            # KataGo additionally compares a distinct lead head. GoCube has one
            # scalar score head and no separate lead head, so do not invent a
            # second signal: retain KataGo's utility and score-mean conditions.
            if (
                player == 1
                and c.q > pass_utility - 0.1
                and c.score_q > pass_score - 0.5
            ):
                return True
            if (
                player == 0
                and c.q < pass_utility + 0.1
                and c.score_q < pass_score + 0.5
            ):
                return True
        return False

    cpdef int[:] counts(self, object gs):
        cdef np.ndarray counts = np.asarray(self.raw_counts(gs), dtype=np.int32).copy()
        if self._should_suppress_pass(gs):
            counts[int(gs.pass_action())] = 0
        return counts

    cpdef int best_action(self, object gs):
        return np.argmax(self.counts(gs))

    cpdef np.ndarray probs(self, object gs, float temp=1.0):
        cdef float[:] counts = np.array(self.counts(gs), dtype=np.float32)
        cdef np.ndarray[dtype=np.float32_t, ndim=1] probs
        cdef Py_ssize_t best_action

        if temp == 0:
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
            return probs

        try:
            probs = (counts / np.sum(counts)) ** (1.0 / temp)
            probs /= np.sum(probs)
            return probs
        except (OverflowError, FloatingPointError):
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
            return probs

    cpdef float value(self, bint average=False):
        cdef float value = 0
        cdef Node c
        cdef float utility_radius
        cdef float perspective

        if self._katago_search:
            if self._root.n <= 0:
                return 0.5
            utility_radius = (
                self.win_loss_utility_factor
                + self.static_score_utility_factor
                + self.dynamic_score_utility_factor
            )
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
