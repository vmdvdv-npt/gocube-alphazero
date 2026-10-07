"""Tree reuse regressions use immutable synthetic games and fixed evaluators."""
from dataclasses import replace
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from gocube_golden.arena_contract import (
    SEARCH_CONTRACT_FINGERPRINT, SearchSettings, compute_search_contract_fingerprint,
)
from gocube_golden.rules import apply_action, prepare_legal_actions
from gocube_golden.search import (
    Evaluation, SEARCH_IMPLEMENTATION_FINGERPRINT, SEARCH_SEMANTICS,
    SearchEvaluationRequest, SearchTree, SequentialPUCT, SequentialPUCTSession,
    search_implementation_fingerprint, search_semantics,
)
from gocube_golden.state import PASS, initial_state


class FixedEvaluator:
    def __init__(self, pass_only=False, utility=0.):
        self.states = []
        self.pass_only = pass_only
        self.utility = utility

    def evaluate(self, state):
        self.states.append(state.state_key)
        legal = prepare_legal_actions(state).actions
        action = PASS if self.pass_only else legal[0]
        policy = tuple(float(point == action) for point in range(state.topology.point_count))
        return Evaluation(policy + (float(action == PASS),), (.5 + self.utility / 2, 0., .5 - self.utility / 2))


def complete(session, evaluator):
    step = session.advance()
    while isinstance(step, SearchEvaluationRequest):
        step = session.resume(evaluator.evaluate(step.state))
    return step


def run(frontend, state, settings, tree, evaluator, **kwargs):
    if frontend == 'sync':
        return SequentialPUCT(settings, tree=tree, **kwargs).search(state, evaluator)
    return complete(SequentialPUCTSession(state, settings, tree=tree, **kwargs), evaluator)


@pytest.mark.parametrize('frontend', ['sync', 'session'])
def test_off_is_identical_to_fresh_search_on_every_move(frontend):
    state = initial_state()
    settings = SearchSettings(simulations=8)
    tree = SearchTree()
    searcher = SequentialPUCT(settings, tree=tree)
    for _ in range(3):
        trace, fresh_trace = [], []
        evaluator, fresh_evaluator = FixedEvaluator(), FixedEvaluator()
        if frontend == 'sync':
            searcher.trace = trace
            actual = searcher.search(state, evaluator)
        else:
            actual = run(frontend, state, settings, tree, evaluator, trace=trace)
        expected = run(frontend, state, settings, SearchTree(), fresh_evaluator, trace=fresh_trace)
        assert actual == expected
        assert trace == fresh_trace and evaluator.states == fresh_evaluator.states
        assert sum(actual.root_visits) == settings.simulations
        state = apply_action(state, actual.action).after
        assert not tree.advance(actual.action, state)
        assert tree.root is None


@pytest.mark.parametrize('frontend', ['sync', 'session'])
def test_on_retains_exact_subtree_objects_and_statistics(frontend):
    state = initial_state()
    settings = SearchSettings(simulations=8, tree_reuse=True)
    tree = SearchTree(True)
    first = run(frontend, state, settings, tree, FixedEvaluator(utility=.4))
    old_root = tree.root
    child = old_root.edges[first.action].child
    assert child.expanded
    edges = child.edges
    before = {action: (edge, edge.visits, edge.value_sum, edge.prior, edge.child)
              for action, edge in edges.items()}
    inherited = sum(edge.visits for edge in edges.values())
    assert inherited > 0
    assert any(edge.value_sum != 0. for edge in edges.values())
    state = apply_action(state, first.action).after
    assert tree.advance(first.action, state)
    assert tree.root is child and tree.root is not old_root
    assert child.edges is edges
    for action, (edge, visits, value_sum, prior, descendant) in before.items():
        assert child.edges[action] is edge
        assert (edge.visits, edge.value_sum, edge.prior, edge.child) == (visits, value_sum, prior, descendant)
    evaluator = FixedEvaluator()
    second = run(frontend, state, settings, tree, evaluator)
    assert tree.root is child and child.edges is edges
    assert state.state_key not in evaluator.states  # No second root evaluation/expansion.
    assert sum(second.root_visits) == inherited + settings.simulations
    for action, (edge, visits, _, prior, descendant) in before.items():
        assert edge.visits >= visits and edge.prior == prior
        if descendant is not None:
            assert edge.child is descendant


@pytest.mark.parametrize('frontend', ['sync', 'session'])
@pytest.mark.parametrize('failure', ['missing_child', 'missing_edge', 'wrong_state', 'wrong_history', 'wrong_komi'])
def test_fallback_is_a_fresh_tree(frontend, failure):
    state = initial_state()
    settings = SearchSettings(simulations=8, tree_reuse=True)
    tree = SearchTree(True)
    first = run(frontend, state, settings, tree, FixedEvaluator())
    action = first.action
    after = apply_action(state, action).after
    if failure == 'missing_child':
        action = next(a for a, e in tree.root.edges.items() if e.child is None)
        after = apply_action(state, action).after
    elif failure == 'missing_edge':
        del tree.root.edges[action]
    elif failure == 'wrong_state':
        after = apply_action(state, 1).after
    elif failure == 'wrong_history':
        # Same board/turn, different superko history must never reuse.
        after = replace(after, superko_history=(after.board_key,))
    elif failure == 'wrong_komi':
        from gocube_golden.state import rules_fingerprint_for
        after = replace(after, komi=1.5, rules_fingerprint=rules_fingerprint_for(after.topology, 1.5))
    old_root = tree.root
    assert not tree.advance(action, after)
    assert tree.root is not old_root and not tree.root.expanded
    actual = run(frontend, after, settings, tree, FixedEvaluator())
    expected = run(frontend, after, settings, SearchTree(True), FixedEvaluator())
    assert actual == expected
    assert sum(actual.root_visits) == settings.simulations


def test_unannounced_state_change_resets_and_searchers_never_share_trees():
    state = initial_state()
    settings = SearchSettings(simulations=8, tree_reuse=True)
    a, b = SequentialPUCT(settings), SequentialPUCT(settings)
    first = a.search(state, FixedEvaluator())
    expected = b.search(state, FixedEvaluator())
    assert first == expected and a.tree.root is not b.tree.root
    child = a.tree.root.edges[first.action].child
    a.advance_root(first.action, apply_action(state, first.action).after)
    assert a.tree.root is child
    assert b.tree.root.state.state_key == state.state_key
    # A missed callback fails safely at search start.
    actual = a.search(state, FixedEvaluator())
    assert actual == expected and a.tree.root is not child
    a.reset_tree()
    assert a.tree.root is None and b.tree.root is not None
    assert a.search(state, FixedEvaluator()) == expected


def test_reused_session_can_finish_without_any_new_neural_request():
    state = initial_state()
    settings = SearchSettings(simulations=3, tree_reuse=True)
    tree = SearchTree(True)
    first = complete(SequentialPUCTSession(state, settings, tree=tree), FixedEvaluator(True))
    assert first.action == PASS
    after = apply_action(state, PASS).after
    child = tree.root.edges[PASS].child
    inherited = sum(e.visits for e in child.edges.values())
    assert tree.advance(PASS, after)
    session = SequentialPUCTSession(after, settings, tree=tree)
    assert session.pending is None and session.result is not None
    assert session.result.evaluator_calls == 0
    assert sum(session.result.root_visits) == inherited + settings.simulations
    assert session.result.action == PASS


def test_root_noise_refresh_preserves_statistics_and_does_not_accumulate():
    from gocube_golden.torus9_monolith import Torus9RootNoiseEvaluator
    from gocube_golden.topology import TORUS_9X9
    state = initial_state(topology=TORUS_9X9)
    tree = SearchTree(True)
    settings = SearchSettings(simulations=8, tree_reuse=True)
    first = complete(SequentialPUCTSession(state, settings, tree=tree), FixedEvaluator())
    after = apply_action(state, first.action).after
    tree.advance(first.action, after)
    root = tree.root
    original = root.evaluation
    visits = sum(e.visits for e in root.edges.values())
    transform = Torus9RootNoiseEvaluator(None, after, seed=17).transform
    expected = transform(original, after, root.legal_context)
    session = SequentialPUCTSession(after, settings, tree=tree,
        evaluation_transform=Torus9RootNoiseEvaluator(None, after, seed=17).transform)
    total = sum(expected.policy[a if a != PASS else 81] for a in root.edges)
    priors = {a: expected.policy[a if a != PASS else 81] / total for a in root.edges}
    assert {a: e.prior for a, e in root.edges.items()} == pytest.approx(priors)
    complete(session, FixedEvaluator())
    assert root.evaluation is original and sum(e.visits for e in root.edges.values()) == visits + 8
    repeat = SequentialPUCTSession(after, settings, tree=tree,
        evaluation_transform=Torus9RootNoiseEvaluator(None, after, seed=17).transform)
    assert {a: e.prior for a, e in root.edges.items()} == pytest.approx(priors)
    complete(repeat, FixedEvaluator())
    # A cheap PCR move restores clean priors; it keeps the retained visits.
    clean = SequentialPUCTSession(after, settings, tree=tree)
    assert root.edges[1].prior == 1.0
    complete(clean, FixedEvaluator())


def test_modes_have_distinct_fingerprints_and_default_identity_is_unchanged():
    assert SEARCH_SEMANTICS['tree_reuse'] is False
    assert search_semantics(True)['tree_reuse'] is True
    assert search_implementation_fingerprint() == SEARCH_IMPLEMENTATION_FINGERPRINT
    assert search_implementation_fingerprint(True) != SEARCH_IMPLEMENTATION_FINGERPRINT
    assert compute_search_contract_fingerprint() == SEARCH_CONTRACT_FINGERPRINT
    assert compute_search_contract_fingerprint(SearchSettings(tree_reuse=True)) != SEARCH_CONTRACT_FINGERPRINT
    for reuse in (False, True):
        result = SequentialPUCT(SearchSettings(simulations=1, tree_reuse=reuse)).search(initial_state(), FixedEvaluator())
        assert result.implementation_fingerprint == search_implementation_fingerprint(reuse)


@pytest.mark.parametrize('value', [0, 1, 'true', None, []])
def test_non_boolean_reuse_is_rejected(value):
    with pytest.raises(ValueError, match='boolean'):
        SearchSettings(tree_reuse=value)
    with pytest.raises(ValueError, match='boolean'):
        SearchTree(value)


@pytest.mark.parametrize('reuse', [False, True])
def test_arena_worker_advances_both_model_trees_and_discards_replenished_games(monkeypatch, reuse):
    import tools.arena_worker as worker
    snapshots, moves = [], []
    real_session = SequentialPUCTSession
    class TrackedSession(real_session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            root = self.tree.root
            snapshots.append((self.tree, root, sum(e.visits for e in root.edges.values()) if root else 0))
    monkeypatch.setattr(worker, 'SequentialPUCTSession', TrackedSession)
    class FakeRemote:
        blocked_inference_seconds = 0.
        blocked_inference_calls = 0
        def __init__(self, **kwargs):
            self.state = None
        def submit(self, state, *args, **kwargs):
            self.state = state
        def poll(self):
            return FixedEvaluator().evaluate(self.state)
    monkeypatch.setattr(worker, '_RemoteEvaluator', FakeRemote)
    real_advance = SearchTree.advance
    def track_advance(self, action, state):
        result = real_advance(self, action, state)
        moves.append((self, state.state_key, result))
        return result
    monkeypatch.setattr(SearchTree, 'advance', track_advance)
    tasks, requests = Queue(), Queue()
    for game_id in ('g1', 'g2'):
        tasks.put({'game_id': game_id})
    def apply(game, result):
        game.state = apply_action(game.state, result.action).after
        game.ply += 1
        return game.ply == 4
    def error(game, exc):
        pytest.fail(str(exc))
    callbacks = worker.CooperativeArenaCallbacks(
        search_settings=SearchSettings(simulations=8, tree_reuse=reuse), search_adapter=None,
        make_game=lambda task: SimpleNamespace(id=task['game_id'], state=initial_state(), ply=0),
        game_id=lambda g: g.id, search_state=lambda g: g.state, search_seed=lambda g: g.ply,
        model_role=lambda g: 'candidate' if g.ply % 2 == 0 else 'reference',
        build_observation=lambda *a: torch.zeros(1), apply_search_result=apply,
        mark_search_error=error, finish_record=lambda g: {'game_id': g.id, 'moves': g.ply})
    worker.run_cooperative_arena_worker(callbacks=callbacks, worker_id=0,
        task_queue=tasks, games_per_worker=1, worker_local_wait_ms=0.,
        candidate_hash='c', reference_hash='r', input_slot=torch.zeros((1, 1)),
        policy_slot=torch.zeros((1, 26)), wdl_slot=torch.zeros((1, 3)),
        request_queue=requests, response_queues=[Queue()], start_event=SimpleNamespace(wait=lambda: None))
    messages = list(requests.queue)
    assert not any(m['kind'] == 'error' for m in messages), messages
    done = next(m for m in messages if m['kind'] == 'done')
    assert done['records'] == [{'game_id': 'g1', 'moves': 4}, {'game_id': 'g2', 'moves': 4}]
    assert len(snapshots) == 8
    if reuse:
        for offset in (0, 4):
            a, b, c, d = snapshots[offset:offset+4]
            assert a[0] is c[0] and b[0] is d[0] and a[0] is not b[0]
            assert c[2] > 0 and d[2] > 0  # Retained across opponent moves.
        assert snapshots[0][0] is not snapshots[4][0]
        assert len(moves) == 14  # 1 tree on first ply, 2 on every later ply.
        assert any(retained for _, _, retained in moves)
    else:
        assert not moves and all(root is None for _, root, _ in snapshots)
