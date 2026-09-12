from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import random
from typing import Mapping, Protocol, Sequence

from .arena_contract import SEARCH_IMPLEMENTATION_ID, SearchSettings
from .search_adapter import GoldenSearchAdapter
from .state import GoldenState

WDL_SEMANTICS = "side-to-move:[WIN,DRAW,LOSS]"
INTERNAL_Q_CONVENTION = "edge-Q-from-parent-side-to-move"
SEARCH_SEMANTICS = {
    "implementation": SEARCH_IMPLEMENTATION_ID,
    "value_input": WDL_SEMANTICS,
    "scalar_boundary": "u=P(WIN)-P(LOSS)",
    "q_convention": INTERNAL_Q_CONVENTION,
    "backup": "one-ply side-to-move sign flip",
    "terminal": "exact-golden-result",
    "tree_reuse": False,
    "transpositions": False,
    "virtual_loss": False,
    "batching": False,
}
SEARCH_IMPLEMENTATION_FINGERPRINT = "sha256:" + hashlib.sha256(
    json.dumps(SEARCH_SEMANTICS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

class SearchError(RuntimeError):
    pass

@dataclass(frozen=True)
class Evaluation:
    policy: Mapping[int | str, float] | Sequence[float]
    wdl: tuple[float, float, float]

class Evaluator(Protocol):
    def evaluate(self, state: GoldenState) -> Evaluation:
        ...

@dataclass
class _Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: "_Node | None" = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

@dataclass
class _Node:
    state: GoldenState
    expanded: bool = False
    edges: dict[int | str, _Edge] = field(default_factory=dict)

@dataclass(frozen=True)
class SearchResult:
    action: int | str
    legal_actions: tuple[int | str, ...]
    root_visits: tuple[int, ...]
    pi: tuple[float, ...]
    simulations: int
    evaluator_calls: int
    implementation_id: str = SEARCH_IMPLEMENTATION_ID
    implementation_fingerprint: str = SEARCH_IMPLEMENTATION_FINGERPRINT

def wdl_to_side_to_move_utility(wdl: Sequence[float]) -> float:
    """The one model-value semantic boundary used by Stage-2 search."""
    if len(wdl) != 3:
        raise SearchError("Golden evaluator WDL must be exactly [WIN, DRAW, LOSS]")
    try:
        win, draw, loss = (float(value) for value in wdl)
    except (TypeError, ValueError) as exc:
        raise SearchError("Golden evaluator WDL must contain numeric values") from exc
    values = (win, draw, loss)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise SearchError("Golden evaluator WDL must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise SearchError("Golden evaluator WDL must have positive total probability")
    return (win - loss) / total

def _child_to_parent_utility(child_utility: float) -> float:
    """The only turn-boundary sign conversion in Stage-2 search/solver."""
    return -float(child_utility)

def _policy_for_legal(
    evaluation: Evaluation,
    state: GoldenState,
    legal: tuple[int | str, ...],
    adapter: GoldenSearchAdapter,
) -> dict[int | str, float]:
    raw = evaluation.policy
    weights: dict[int | str, float] = {}
    if isinstance(raw, Mapping):
        for action in legal:
            value = raw.get(action, 0.0)
            try:
                weight = float(value)
            except (TypeError, ValueError) as exc:
                raise SearchError(f"Non-numeric policy weight for action {action!r}") from exc
            if not math.isfinite(weight) or weight < 0:
                raise SearchError(f"Invalid policy weight for action {action!r}")
            weights[action] = weight
    else:
        seq = tuple(raw)
        expected = state.topology.point_count + 1
        if len(seq) != expected:
            raise SearchError(f"Golden policy length {len(seq)} != action size {expected}")
        for action in legal:
            weight = float(seq[adapter.action_index(state, action)])
            if not math.isfinite(weight) or weight < 0:
                raise SearchError(f"Invalid policy weight for action {action!r}")
            weights[action] = weight
    total = sum(weights.values())
    if total <= 0.0:
        uniform = 1.0 / len(legal)
        return {action: uniform for action in legal}
    return {action: weight / total for action, weight in weights.items()}

class SequentialPUCT:
    """Small single-state Python PUCT used only because legacy pinned search failed qualification."""

    def __init__(
        self,
        settings: SearchSettings | None = None,
        *,
        adapter: GoldenSearchAdapter | None = None,
    ) -> None:
        self.settings = settings or SearchSettings()
        self.adapter = adapter or GoldenSearchAdapter()
        self._evaluator: Evaluator | None = None
        self._evaluator_calls = 0
        self._rng = random.Random(0)

    def _evaluate_and_expand(self, node: _Node) -> float:
        if self.adapter.is_terminal(node.state):
            return self.adapter.terminal_utility(node.state)
        if self._evaluator is None:
            raise SearchError("Search evaluator was not installed")
        evaluation = self._evaluator.evaluate(node.state)
        self._evaluator_calls += 1
        utility = wdl_to_side_to_move_utility(evaluation.wdl)
        legal = self.adapter.legal_actions(node.state)
        if not legal:
            raise SearchError("Nonterminal Golden state exposed no legal actions")
        priors = _policy_for_legal(evaluation, node.state, legal, self.adapter)
        node.edges = {action: _Edge(prior=priors[action]) for action in legal}
        node.expanded = True
        return utility

    def _tie_key(self, state: GoldenState, action: int | str) -> int:
        return self.adapter.action_index(state, action)

    def _select(self, node: _Node) -> tuple[int | str, _Edge]:
        total_visits = sum(edge.visits for edge in node.edges.values())
        scale = math.sqrt(total_visits + 1.0)
        best_value = -float("inf")
        candidates: list[tuple[int | str, _Edge]] = []
        for action, edge in node.edges.items():
            q = edge.q if edge.visits else float(self.settings.fpu)
            score = q + float(self.settings.cpuct) * edge.prior * scale / (1.0 + edge.visits)
            if score > best_value + 1e-15:
                best_value = score
                candidates = [(action, edge)]
            elif abs(score - best_value) <= 1e-15:
                candidates.append((action, edge))
        if not candidates:
            raise SearchError("PUCT could not select a legal edge")
        if self.settings.deterministic_tie_break:
            return min(candidates, key=lambda item: self._tie_key(node.state, item[0]))
        return self._rng.choice(candidates)

    def _simulate(self, node: _Node) -> float:
        if self.adapter.is_terminal(node.state):
            return self.adapter.terminal_utility(node.state)
        if not node.expanded:
            return self._evaluate_and_expand(node)
        action, edge = self._select(node)
        if edge.child is None:
            child_state = self.adapter.apply_action(node.state, action)
            edge.child = _Node(child_state)
        child_utility = self._simulate(edge.child)
        parent_utility = _child_to_parent_utility(child_utility)
        edge.visits += 1
        edge.value_sum += parent_utility
        return parent_utility

    def search(
        self,
        state: GoldenState,
        evaluator: Evaluator,
        *,
        seed: int = 0,
    ) -> SearchResult:
        if state.is_terminal:
            raise SearchError("Search cannot be started from a terminal Golden state")
        before = state.state_key
        self._evaluator = evaluator
        self._evaluator_calls = 0
        self._rng = random.Random(int(seed))
        root = _Node(state)
        self._evaluate_and_expand(root)
        for _ in range(self.settings.simulations):
            self._simulate(root)
        if state.state_key != before:
            raise SearchError("Golden search mutated its parent state")
        total = sum(edge.visits for edge in root.edges.values())
        if total <= 0:
            raise SearchError("Golden search produced zero root visits")

        legal = self.adapter.legal_actions(state)
        action_space = self.adapter.action_space(state)
        visit_map = {action: root.edges[action].visits for action in legal}
        root_visits = tuple(visit_map.get(action, 0) for action in action_space)
        pi = tuple(count / total for count in root_visits)

        max_visits = max(visit_map.values())
        candidates = [action for action, visits in visit_map.items() if visits == max_visits]
        if self.settings.deterministic_tie_break:
            selected = min(candidates, key=lambda action: self._tie_key(state, action))
        else:
            selected = self._rng.choice(candidates)
        if selected not in legal:
            raise SearchError("Search selected an illegal action")
        return SearchResult(
            action=selected,
            legal_actions=legal,
            root_visits=root_visits,
            pi=pi,
            simulations=self.settings.simulations,
            evaluator_calls=self._evaluator_calls,
        )

class SolveStatus(str, Enum):
    EXACT = "EXACT"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class ExactSolveResult:
    status: SolveStatus
    utility: float | None
    best_actions: tuple[int | str, ...]
    nodes: int

def solve_exact(
    state: GoldenState,
    *,
    adapter: GoldenSearchAdapter | None = None,
    node_limit: int = 100_000,
) -> ExactSolveResult:
    """Exhaust tiny Golden research trees or explicitly return UNKNOWN."""
    if node_limit <= 0:
        raise ValueError("node_limit must be positive")
    adapter = adapter or GoldenSearchAdapter()
    cache: dict[tuple[object, ...], tuple[float, tuple[int | str, ...]]] = {}
    nodes = 0
    exhausted = False

    def visit(current: GoldenState) -> tuple[float, tuple[int | str, ...]] | None:
        nonlocal nodes, exhausted
        key = current.state_key
        if key in cache:
            return cache[key]
        if nodes >= node_limit:
            exhausted = True
            return None
        nodes += 1
        if adapter.is_terminal(current):
            solved = (adapter.terminal_utility(current), ())
            cache[key] = solved
            return solved
        best = -float("inf")
        best_actions: list[int | str] = []
        for action in adapter.legal_actions(current):
            child = adapter.apply_action(current, action)
            child_solved = visit(child)
            if child_solved is None:
                return None
            value = _child_to_parent_utility(child_solved[0])
            if value > best + 1e-12:
                best = value
                best_actions = [action]
            elif abs(value - best) <= 1e-12:
                best_actions.append(action)
        solved = (best, tuple(best_actions))
        cache[key] = solved
        return solved

    solved = visit(state)
    if solved is None or exhausted:
        return ExactSolveResult(SolveStatus.UNKNOWN, None, (), nodes)
    return ExactSolveResult(SolveStatus.EXACT, solved[0], solved[1], nodes)
