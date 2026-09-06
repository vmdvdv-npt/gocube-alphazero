# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from libc.math cimport sqrt

import numpy as np
cimport numpy as np
from alphazero.utils import dotdict
from alphazero.mcts_policy import normalize_masked_policy
from alphazero.search_contract import (
    GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE,
    SearchOutput,
    equivalent_win_probability,
    ownership_root_ending_bonus_points,
    ownership_root_move_useful,
    player_score_points,
    recent_score_center,
    score_utility,
    score_utility_diff,
)


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

NOISE_ALPHA_RATIO = 10.83
_DRAW_VALUE = 0.5

np.seterr(all='raise')


cdef class Node:
    cdef public list _children
    cdef public int a
    cdef public np.ndarray e
    cdef public float q
    cdef public float v
    cdef public float win_q
    cdef public float win_v
    cdef public float score_q
    cdef public float score_v
    cdef public float score_utility_q
    cdef public float score_utility_v
    cdef public int n
    cdef public float p
    cdef public int player

    def __init__(self, int action, int num_players):
        self._children = []
        self.a = action
        self.e = np.zeros(num_players, dtype=np.uint8)
        self.q = 0
        self.v = 0
        self.win_q = 0.5
        self.win_v = 0.5
        self.score_q = 0
        self.score_v = 0
        self.score_utility_q = 0
        self.score_utility_v = 0
        self.n = 0
        self.p = 0
        self.player = 0

    def __repr__(self):
        return (
            'Node(a={}, e={}, q={}, v={}, win_q={}, score_q={}, score_utility_q={}, '
            'n={}, p={}, player={})'
        ).format(
            self.a, self.e, self.q, self.v, self.win_q, self.score_q,
            self.score_utility_q, self.n, self.p, self.player,
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
    cdef public bint _score_aware
    cdef public float win_loss_utility_factor
    cdef public float static_score_utility_factor
    cdef public float dynamic_score_utility_factor
    cdef public float dynamic_score_center_zero_weight
    cdef public float dynamic_score_center_scale
    cdef public float root_ending_bonus_points
    cdef public float score_improvement_threshold_points
    cdef public float win_probability_tolerance
    cdef public bint fill_dame_before_pass
    cdef public bint conservative_pass
    cdef public int _point_count
    cdef public float _root_score_points_bw
    cdef public float _root_recent_center_bw
    cdef public object _root_ownership
    cdef public bint _root_context_ready

    def __init__(self, args: dotdict):
        self.root_noise_frac = args.root_noise_frac
        self.root_temp = args.root_policy_temp
        self.min_discount = args.min_discount
        self.fpu_reduction = args.fpu_reduction
        self.cpuct = args.cpuct
        self._num_players = args._num_players
        self.search_utility_mode = getattr(args, 'search_utility_mode', 'legacy')
        self._score_aware = self.search_utility_mode == GOCUBE_KATAGO_V3_SEARCH_UTILITY_MODE
        self.win_loss_utility_factor = float(getattr(args, 'gocube_win_loss_utility_factor', 1.0))
        self.static_score_utility_factor = float(getattr(args, 'gocube_static_score_utility_factor', 0.0))
        self.dynamic_score_utility_factor = float(getattr(args, 'gocube_dynamic_score_utility_factor', 0.4))
        self.dynamic_score_center_zero_weight = float(getattr(args, 'gocube_dynamic_score_center_zero_weight', 0.25))
        self.dynamic_score_center_scale = float(getattr(args, 'gocube_dynamic_score_center_scale', 0.5))
        self.root_ending_bonus_points = float(getattr(args, 'gocube_root_ending_bonus_points', 0.5))
        self.score_improvement_threshold_points = float(
            getattr(args, 'gocube_score_improvement_threshold_points', 1.0)
        )
        self.win_probability_tolerance = float(getattr(args, 'gocube_win_probability_tolerance', 0.005))
        self.fill_dame_before_pass = bool(getattr(args, 'gocube_fill_dame_before_pass', True))
        self.conservative_pass = bool(getattr(args, 'gocube_conservative_pass', True))
        self._root = Node(-1, self._num_players)
        self._curnode = self._root
        self._path = []
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0
        self._point_count = 0
        self._root_score_points_bw = 0
        self._root_recent_center_bw = 0
        self._root_ownership = None
        self._root_context_ready = False

    def __repr__(self):
        return (
            'MCTS(root_noise_frac={}, root_temp={}, min_discount={}, fpu_reduction={}, cpuct={}, '
            'search_utility_mode={}, _num_players={}, _root={}, depth={}, max_depth={})'
        ).format(
            self.root_noise_frac, self.root_temp, self.min_discount, self.fpu_reduction,
            self.cpuct, self.search_utility_mode, self._num_players, self._root,
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
        self._root_score_points_bw = 0
        self._root_recent_center_bw = 0
        self._root_ownership = None
        self._root_context_ready = False

    cpdef void search(self, object gs, object nn, int sims, bint add_root_noise, bint add_root_temp):
        cdef float[:] v
        cdef float[:] p
        cdef object out
        self.max_depth = 0

        for _ in range(sims):
            leaf = self.find_leaf(gs)
            if self._score_aware:
                if not hasattr(nn, 'predict_for_search'):
                    raise RuntimeError('score-aware MCTS requires predict_for_search()')
                out = nn.predict_for_search(leaf.observation())
                if not isinstance(out, SearchOutput):
                    raise RuntimeError('predict_for_search() must return SearchOutput')
                if out.score is None or out.ownership is None:
                    raise RuntimeError('GoCube score-aware MCTS requires score and ownership heads')
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

        for _ in range(sims):
            leaf = self.find_leaf(gs)
            self.process_results(leaf, v, p, add_root_noise, add_root_temp)

    cpdef void update_root(self, object gs, int a):
        if not self._root._children:
            self._root.add_children(gs.valid_moves(), self._num_players)

        cdef Node c
        for c in self._root._children:
            if c.a == a:
                self._root = c
                self._curnode = self._root
                self._path = []
                self._root_ownership = None
                self._root_score_points_bw = 0
                self._root_recent_center_bw = 0
                self._root_context_ready = False
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

    cdef bint _is_second_pass_root(self, object gs):
        cdef object state
        if not self._score_aware:
            return False
        state = getattr(gs, 'semantic_state', None)
        if state is None:
            return False
        return (
            getattr(state, 'phase', None) == 'main'
            and int(getattr(state, 'consecutive_passes', 0)) == 1
            and getattr(state, 'terminal_kind', None) is None
        )

    cdef float _recent_center_for_player(self, int player):
        return self._root_recent_center_bw if player == 0 else -self._root_recent_center_bw

    cdef float _root_score_for_player(self, int player):
        return self._root_score_points_bw if player == 0 else -self._root_score_points_bw

    cdef float _ending_bonus_points(self, object gs, int action):
        if not self._score_aware or self.root_ending_bonus_points <= 0:
            return 0.0
        if hasattr(gs, 'search_root_ending_bonus_points'):
            return float(gs.search_root_ending_bonus_points(
                int(action), self._root_ownership, self.root_ending_bonus_points
            ))
        return float(ownership_root_ending_bonus_points(
            gs, int(action), self._root_ownership, self.root_ending_bonus_points
        ))

    cdef bint _root_move_useful(self, object gs, int action):
        if hasattr(gs, 'search_root_move_useful'):
            return bool(gs.search_root_move_useful(int(action), self._root_ownership))
        return bool(ownership_root_move_useful(gs, int(action), self._root_ownership))

    cdef Node _forced_nonpass_child(self, object gs):
        cdef Node c
        cdef Node best = None
        cdef Node fallback = None
        cdef float best_p = -1
        cdef float fallback_p = -1
        cdef int pass_action

        if not self._is_second_pass_root(gs) or not self.fill_dame_before_pass:
            return None
        pass_action = int(gs.pass_action())
        for c in self._root._children:
            if c.a != pass_action and c.n > 0:
                return None
        for c in self._root._children:
            if c.a == pass_action:
                continue
            if c.p > fallback_p:
                fallback = c
                fallback_p = c.p
            if self._root_move_useful(gs, c.a) and c.p > best_p:
                best = c
                best_p = c.p
        return best if best is not None else fallback

    cdef Node _best_root_child(self, object gs):
        cdef Node forced = self._forced_nonpass_child(gs)
        cdef Node c
        cdef Node child = None
        cdef float seen_policy
        cdef float fpu_value
        cdef float cur_best = -float('inf')
        cdef float sqrt_n = sqrt(self._root.n)
        cdef float base
        cdef float uct
        cdef float bonus_points
        cdef float points
        cdef float util_diff
        cdef int player = int(gs.player)

        if forced is not None:
            return forced

        seen_policy = sum([c.p for c in self._root._children if c.n > 0])
        fpu_value = self._root.v - self.fpu_reduction * sqrt(seen_policy)
        for c in self._root._children:
            base = fpu_value if c.n == 0 else c.q
            if self._score_aware and self._point_count > 0:
                bonus_points = self._ending_bonus_points(gs, c.a)
                if bonus_points != 0:
                    points = self._root_score_for_player(player) if c.n == 0 else c.score_q
                    util_diff = float(score_utility_diff(
                        points,
                        bonus_points,
                        recent_center=self._recent_center_for_player(player),
                        point_count=self._point_count,
                        static_factor=self.static_score_utility_factor,
                        dynamic_factor=self.dynamic_score_utility_factor,
                        dynamic_scale=self.dynamic_score_center_scale,
                    ))
                    base += util_diff
            uct = base + self.cpuct * c.p * sqrt_n / (1 + c.n)
            if uct > cur_best:
                cur_best = uct
                child = c
        return child

    cpdef object find_leaf(self, object gs):
        self.depth = 0
        self._curnode = self._root
        self._path = []
        cdef object leaf = gs.clone()
        cdef Node next_child

        # Tree reuse means the new root can already be visited. Score-aware
        # search deliberately re-evaluates that root once so ownership, score
        # center, and policy all describe the current root without a second NN
        # forward per leaf.
        if self._score_aware and self._root.n > 0 and not self._root_context_ready:
            return leaf

        while self._curnode.n > 0 and not self._curnode.e.any():
            self._path.append(self._curnode)
            if self._curnode == self._root and self._score_aware:
                next_child = self._best_root_child(gs)
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
        """Historical policy/value-only backup. Kept bit-for-bit for legacy games."""
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

    cdef tuple _score_aware_values(self, object value, float score_norm_bw, int player, bint score_available):
        cdef float win_prob = float(equivalent_win_probability(value, player, self._num_players))
        cdef float points
        cdef float score_util
        cdef float combined
        if not score_available:
            combined = (2.0 * win_prob - 1.0) * self.win_loss_utility_factor
            return combined, win_prob, 0.0, 0.0
        points = float(player_score_points(score_norm_bw, player, self._point_count))
        score_util = float(score_utility(
            points,
            recent_center=self._recent_center_for_player(player),
            point_count=self._point_count,
            static_factor=self.static_score_utility_factor,
            dynamic_factor=self.dynamic_score_utility_factor,
            dynamic_scale=self.dynamic_score_center_scale,
        ))
        combined = (2.0 * win_prob - 1.0) * self.win_loss_utility_factor + score_util
        return combined, win_prob, points, score_util

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
        """GoCube V3 policy/value/score/ownership search backup."""
        cdef np.ndarray valids
        cdef Node c
        cdef Node parent
        cdef object terminal
        cdef object exact_score
        cdef float score_norm_bw = 0.0
        cdef float exact_bw
        cdef bint score_available = True
        cdef tuple values
        cdef float combined
        cdef float win_prob
        cdef float points
        cdef float score_util

        if not self._score_aware:
            raise RuntimeError('process_search_results called for legacy MCTS')
        self._point_count = int(gs.logical_topology().point_count)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        pi = np.asarray(pi, dtype=np.float32).reshape(-1)

        if self._curnode.e.any():
            value = np.asarray(self._curnode.e, dtype=np.float32)
            terminal = getattr(gs, 'terminal_adjudication', None)
            exact_score = getattr(terminal, 'score', None) if terminal is not None else None
            if exact_score is None:
                score_available = False
                score_norm_bw = 0.0
            else:
                exact_bw = float(exact_score.black) - float(exact_score.white)
                score_norm_bw = exact_bw / float(self._point_count)
        else:
            score_arr = np.asarray(score, dtype=np.float32).reshape(-1)
            if score_arr.size != 1:
                raise ValueError(f'GoCube search score head must contain one value per position, got {score_arr.shape}')
            score_norm_bw = float(score_arr[0])
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
                        f'GoCube ownership search head shape {self._root_ownership.shape} '
                        f'!= ({self._point_count}, 3)'
                    )
                self._root_score_points_bw = score_norm_bw * float(self._point_count)
                self._root_recent_center_bw = float(recent_score_center(
                    self._root_score_points_bw,
                    zero_weight=self.dynamic_score_center_zero_weight,
                    center_scale=self.dynamic_score_center_scale,
                    point_count=self._point_count,
                ))
                self._root_context_ready = True
                if add_root_noise:
                    self._add_root_noise()
            else:
                self._curnode.update_policy(pi)

        if self._curnode.n == 0 or self._curnode == self._root:
            values = self._score_aware_values(value, score_norm_bw, self._curnode.player, score_available)
            self._curnode.v = values[0]
            self._curnode.win_v = values[1]
            self._curnode.score_v = values[2]
            self._curnode.score_utility_v = values[3]

        while self._path:
            parent = self._path.pop()
            values = self._score_aware_values(value, score_norm_bw, parent.player, score_available)
            combined = values[0]
            win_prob = values[1]
            points = values[2]
            score_util = values[3]
            self._curnode.q = (self._curnode.q * self._curnode.n + combined) / (self._curnode.n + 1)
            self._curnode.win_q = (self._curnode.win_q * self._curnode.n + win_prob) / (self._curnode.n + 1)
            self._curnode.score_q = (self._curnode.score_q * self._curnode.n + points) / (self._curnode.n + 1)
            self._curnode.score_utility_q = (
                self._curnode.score_utility_q * self._curnode.n + score_util
            ) / (self._curnode.n + 1)
            self._curnode.n += 1
            self._curnode = parent

        self._root.n += 1

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

    cdef bint _qualifying_nonpass_exists(self, object gs):
        cdef Node c
        cdef Node pass_node = None
        cdef int pass_action
        cdef float gain
        if not self._is_second_pass_root(gs) or not self.fill_dame_before_pass:
            return False
        pass_action = int(gs.pass_action())
        for c in self._root._children:
            if c.a == pass_action:
                pass_node = c
                break
        if pass_node is None or pass_node.n <= 0:
            return False
        for c in self._root._children:
            if c.a == pass_action or c.n <= 0:
                continue
            if not self._root_move_useful(gs, c.a):
                continue
            gain = c.score_q - pass_node.score_q
            if (
                gain >= self.score_improvement_threshold_points
                and c.win_q >= pass_node.win_q - self.win_probability_tolerance
            ):
                return True
        return False

    cpdef int[:] counts(self, object gs):
        cdef np.ndarray counts = np.asarray(self.raw_counts(gs), dtype=np.int32).copy()
        if self._score_aware and self._qualifying_nonpass_exists(gs):
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

    cpdef dict root_statistics(self, object gs):
        """Return debuggable raw root statistics without changing selection."""
        cdef Node c
        cdef dict stats = {}
        cdef int raw_total = sum([child.n for child in self._root._children])
        cdef float bonus
        for c in self._root._children:
            if c.n <= 0:
                continue
            bonus = self._ending_bonus_points(gs, c.a) if self._score_aware else 0.0
            stats[int(c.a)] = {
                'visits': int(c.n),
                'visit_fraction': (float(c.n) / raw_total) if raw_total else 0.0,
                'win_estimate': float(c.win_q),
                'score_estimate': float(c.score_q),
                'score_utility': float(c.score_utility_q),
                'combined_utility': float(c.q),
                'policy_prior': float(c.p),
                'ending_bonus_points': float(bonus),
            }
        return stats

    cpdef dict pass_diagnostic(self, object gs):
        cdef dict stats = self.root_statistics(gs)
        cdef int pass_action
        cdef dict pass_stats
        cdef dict best = None
        cdef int action
        cdef dict item
        cdef float gain
        cdef float best_gain = -float('inf')
        cdef float win_delta = 0.0
        cdef bint dominated = False
        if not self._score_aware or not hasattr(gs, 'pass_action'):
            return {}
        pass_action = int(gs.pass_action())
        pass_stats = stats.get(pass_action)
        if pass_stats is None:
            return {}
        for action, item in stats.items():
            if action == pass_action:
                continue
            gain = float(item['score_estimate']) - float(pass_stats['score_estimate'])
            if gain > best_gain:
                best_gain = gain
                best = item
        if best is not None:
            win_delta = float(best['win_estimate']) - float(pass_stats['win_estimate'])
            dominated = (
                best_gain >= self.score_improvement_threshold_points
                and win_delta >= -self.win_probability_tolerance
            )
        return {
            'pass_root_prior': float(pass_stats['policy_prior']),
            'pass_visit_fraction': float(pass_stats['visit_fraction']),
            'pass_win_utility': 2.0 * float(pass_stats['win_estimate']) - 1.0,
            'pass_score_utility': float(pass_stats['score_utility']),
            'pass_combined_utility': float(pass_stats['combined_utility']),
            'best_nonpass_score_gain': float(best_gain if best is not None else 0.0),
            'best_nonpass_win_delta': float(win_delta),
            'score_dominated_pass': bool(dominated),
            'pass_suppressed': bool(self._qualifying_nonpass_exists(gs)),
        }

    cpdef float value(self, bint average=False):
        cdef float value = 0
        cdef Node c

        if average:
            value = sum([c.q for c in self._root._children if c.n > 0]) / len(self._root._children)
        else:
            if self._score_aware:
                value = -float('inf')
            for c in self._root._children:
                if c.q > value and c.n > 0:
                    value = c.q
            if self._score_aware and value == -float('inf'):
                value = self._root.v
        return value
