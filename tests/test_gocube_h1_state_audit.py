from __future__ import annotations

import json
from dataclasses import fields, replace

import numpy as np

from alphazero.envs.gocube import (
    CLEANUP_1,
    CLEANUP_2,
    Cube2JapaneseGame,
    Cube4JapaneseGame,
    cube_topology,
    initial_v3_state,
    v3_valid_moves,
)
from alphazero.envs.gocube.diversified_game import DiversifiedPinnedCube4JapaneseGame
from alphazero.envs.gocube.katago_v3 import V3State, apply_v3_action
from alphazero.envs.gocube.katago_v3 import (
    EMERGENCY_MOVE_CAP_BASE,
    EMERGENCY_MOVE_CAP_FACTOR,
)
from alphazero.envs.gocube.pinned_game import PinnedCube4JapaneseGame
from alphazero.envs.gocube.records import _final_position
from alphazero.envs.gocube.selfplay_semantics import rebase_cleanup_training_state

from tests.support.h1_probe import (
    DIVERSIFIED_WRAPPER_AUDIT_FIELDS,
    PINNED_WRAPPER_AUDIT_FIELDS,
    V3_STATE_AUDIT_FIELDS,
    ReachableSample,
    bounded_reachable_states,
    find_observation_collisions,
    v3_immediate_semantic_signature,
    v3_state_key,
)


SEED = 20260908


def _legal_subset(state, topology):
    actions = tuple(int(action) for action in np.flatnonzero(v3_valid_moves(state, topology)))
    # PASS is deliberately first so the bounded probe reaches phase boundaries
    # without enumerating the whole Cube action space.
    return tuple(sorted(actions, key=lambda action: (action != topology.pass_action, action))[:8])


def _ordinary_samples():
    topology = cube_topology(2)
    samples = bounded_reachable_states(
        initial_v3_state(topology),
        legal_actions=lambda state: _legal_subset(state, topology),
        transition=lambda state, action: apply_v3_action(state, action, topology),
        state_key=v3_state_key,
        max_depth=6,
        max_states=256,
        category="ordinary_start",
        seed=SEED,
    )
    # The bounded BFS is capped for CI. Add the canonical legal PASS path so
    # both ordinary phase boundaries are included even when the stone-move
    # frontier consumes the cap first.
    state = initial_v3_state(topology)
    boundary = []
    actions = []
    for pass_number in range(1, 7):
        state = apply_v3_action(state, topology.pass_action, topology)
        actions.append(topology.pass_action)
        boundary.append(
            ReachableSample(
                f"ordinary_phase_boundary:{pass_number:05d}",
                state,
                tuple(actions),
                "ordinary_phase_boundary",
                SEED,
            )
        )
    return topology, samples + tuple(boundary)


def _play_deterministic_points(game, count):
    for _ in range(count):
        valid = np.flatnonzero(game.valid_moves())
        valid = valid[valid != game.pass_action()]
        assert len(valid), "bounded fixture ran out of legal placement actions"
        game.play_action(int(valid[0]))


def _fork_samples():
    source = DiversifiedPinnedCube4JapaneseGame()
    source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    _play_deterministic_points(source, 6)
    candidate_state = source.semantic_state
    candidate_history = source._pinned_move_history

    samples = []
    plain_pool = DiversifiedPinnedCube4JapaneseGame._plain_fork_pool()
    plain_pool.clear()
    for kind in ("ordinary_fork", "early_fork"):
        plain_pool.clear()
        plain_pool.append((kind, candidate_state, candidate_history, len(candidate_history)))
        target = DiversifiedPinnedCube4JapaneseGame()
        assert target.maybe_start_plain_fork()["mode"] == kind
        samples.append(
            ReachableSample(
                f"{kind}:00000",
                target.semantic_state,
                tuple(action for _player, action in candidate_history),
                kind,
                SEED,
            )
        )

    seki_pool = DiversifiedPinnedCube4JapaneseGame._seki_pool()
    seki_pool.clear()
    seki_pool.append((candidate_state, candidate_history))
    target = DiversifiedPinnedCube4JapaneseGame()
    assert target.maybe_start_seki_fork(1.0)
    samples.append(
        ReachableSample(
            "seki_fork:00000",
            target.semantic_state,
            tuple(action for _player, action in candidate_history),
            "seki_fork",
            SEED,
        )
    )
    plain_pool.clear()
    seki_pool.clear()
    return samples


def _synthetic_cleanup_samples():
    source = DiversifiedPinnedCube4JapaneseGame()
    source.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=False,
        seki_fork_hack_prob=0.0,
    )
    _play_deterministic_points(source, 4)
    state = source.semantic_state
    return [
        ReachableSample(
            "synthetic_cleanup1:00000",
            rebase_cleanup_training_state(state, CLEANUP_1),
            tuple(),
            "synthetic_cleanup1",
            SEED,
        ),
        ReachableSample(
            "synthetic_cleanup2:00000",
            rebase_cleanup_training_state(state, CLEANUP_2),
            tuple(),
            "synthetic_cleanup2",
            SEED,
        ),
    ]


def _all_h1_samples():
    _topology, ordinary = _ordinary_samples()
    return ordinary + tuple(_fork_samples()) + tuple(_synthetic_cleanup_samples())


def _observation_for_state(state: V3State):
    if len(state.board) == 24:
        return Cube2JapaneseGame(state).observation()
    return PinnedCube4JapaneseGame(state).observation()


def test_v3_state_audit_covers_all_fields():
    assert set(V3_STATE_AUDIT_FIELDS) == {field.name for field in fields(V3State)}
    assert all(item["classification"] in {"A", "B", "C", "D", "E"} for item in V3_STATE_AUDIT_FIELDS.values())


def test_pinned_wrapper_audit_covers_initialized_state_fields():
    game = PinnedCube4JapaneseGame()
    assert set(PINNED_WRAPPER_AUDIT_FIELDS) == {
        name for name in vars(game) if name.startswith("_pinned_")
    }


def test_clone_preserves_rule_relevant_state_and_wrapper_accumulators():
    game = DiversifiedPinnedCube4JapaneseGame()
    game.configure_pinned_selfplay(
        auto_end_pass_alive=False,
        root_prune_useless_moves=True,
        seki_fork_hack_prob=0.25,
    )
    game.configure_diversification(
        early_fork_prob=0.1,
        ordinary_fork_prob=0.2,
        early_expected_move_prop=0.025,
    )
    _play_deterministic_points(game, 5)
    clone = game.clone()
    assert v3_state_key(clone.semantic_state) == v3_state_key(game.semantic_state)
    for name in PINNED_WRAPPER_AUDIT_FIELDS:
        left, right = getattr(clone, name), getattr(game, name)
        if name in ("_pinned_is_search_clone", "_pinned_at_search_root"):
            assert left is True and right is False, name
            continue
        if name == "_pinned_state_history":
            assert tuple(v3_state_key(item) for item in left) == tuple(v3_state_key(item) for item in right)
        else:
            assert left == right, name
    for name in DIVERSIFIED_WRAPPER_AUDIT_FIELDS:
        left, right = getattr(clone, name), getattr(game, name)
        if name == "_diverse_train_state_history":
            assert tuple(v3_state_key(item) for item in left) == tuple(v3_state_key(item) for item in right)
        else:
            assert left == right, name

    valid = np.flatnonzero(clone.valid_moves())
    action = int(valid[valid != clone.pass_action()][0])
    clone.play_action(action)
    assert game.semantic_state.turns + 1 == clone.semantic_state.turns
    assert game.semantic_state.turns == 5


def test_serialization_round_trip_marks_the_supported_record_boundary():
    topology = cube_topology(2)
    game = Cube2JapaneseGame()
    game.play_action(0)
    position = _final_position(game)
    restored_payload = json.loads(json.dumps(position))
    assert restored_payload["board"] == position["board"]
    assert restored_payload["previous_board"] == position["previous_board"]
    assert restored_payload["captures"] == list(game.semantic_state.captures)
    # Game records intentionally expose a replay/final-position boundary, not
    # a raw V3State snapshot. History-sensitive fields require replay or are
    # not reconstructible from this record alone.
    assert "phase_history" not in restored_payload
    assert "history_since_pass" not in restored_payload
    assert "ko_capture_history" not in restored_payload
    assert topology.point_count == len(restored_payload["board"])


def test_observation_collision_probe_uses_reachable_state_families():
    samples = _all_h1_samples()
    categories = {sample.category for sample in samples}
    assert {
        "ordinary_start",
        "ordinary_fork",
        "early_fork",
        "seki_fork",
        "synthetic_cleanup1",
        "synthetic_cleanup2",
    } <= categories
    report = find_observation_collisions(
        samples,
        _observation_for_state,
        lambda state: v3_immediate_semantic_signature(
            state,
            cube_topology(2) if len(state.board) == 24 else cube_topology(4),
        ),
    )
    assert report.samples_examined == len(samples)
    assert report.observation_groups <= report.samples_examined
    assert isinstance(report.semantic_collisions, tuple)


def test_collision_signature_compares_legality():
    topology = cube_topology(2)
    initial = initial_v3_state(topology)
    child = apply_v3_action(initial, 0, topology)
    samples = (
        ReachableSample("initial", initial),
        ReachableSample("after-legal-move", child),
    )
    report = find_observation_collisions(
        samples,
        lambda _state: np.zeros((1, 1), dtype=np.float32),
        lambda state: tuple(int(value) for value in v3_valid_moves(state, topology)),
    )
    assert report.semantic_collisions[0].semantic_collision


def test_collision_signature_compares_pass_transition():
    topology = cube_topology(2)
    initial = initial_v3_state(topology)
    after_pass = apply_v3_action(initial, topology.pass_action, topology)
    samples = (ReachableSample("initial", initial), ReachableSample("after-pass", after_pass))
    report = find_observation_collisions(
        samples,
        lambda _state: np.zeros((1, 1), dtype=np.float32),
        lambda state: v3_immediate_semantic_signature(state, topology),
    )
    assert report.semantic_collisions[0].semantic_collision
    assert "pass_after" in repr(report.semantic_collisions[0].semantic_signatures[0])


def test_collision_probe_is_deterministic():
    first = _all_h1_samples()
    second = _all_h1_samples()
    assert [(item.category, item.actions, v3_state_key(item.state)) for item in first] == [
        (item.category, item.actions, v3_state_key(item.state)) for item in second
    ]
    first_report = find_observation_collisions(
        first,
        _observation_for_state,
        lambda state: v3_immediate_semantic_signature(
            state,
            cube_topology(2) if len(state.board) == 24 else cube_topology(4),
        ),
    )
    second_report = find_observation_collisions(
        second,
        _observation_for_state,
        lambda state: v3_immediate_semantic_signature(
            state,
            cube_topology(2) if len(state.board) == 24 else cube_topology(4),
        ),
    )
    assert first_report == second_report


def test_synthetic_cleanup1_is_included():
    samples = _synthetic_cleanup_samples()
    cleanup1 = next(item for item in samples if item.category == "synthetic_cleanup1")
    assert cleanup1.state.phase == CLEANUP_1
    assert cleanup1.state.main_moves == (0, 0)
    assert cleanup1.state.previous_board is None


def test_synthetic_cleanup2_is_included():
    samples = _synthetic_cleanup_samples()
    cleanup2 = next(item for item in samples if item.category == "synthetic_cleanup2")
    assert cleanup2.state.phase == CLEANUP_2
    assert cleanup2.state.second_cleanup_start_colors == bytes(cleanup2.state.board.tolist())
    assert cleanup2.state.main_moves == (0, 0)


def test_fork_state_families_are_included():
    samples = _fork_samples()
    assert {sample.category for sample in samples} == {"ordinary_fork", "early_fork", "seki_fork"}
    assert all(sample.state.phase == "main" for sample in samples)


def test_technical_budget_candidate_is_diagnostic_only():
    topology = cube_topology(4)
    initial = initial_v3_state(topology)
    cap = EMERGENCY_MOVE_CAP_BASE + EMERGENCY_MOVE_CAP_FACTOR * topology.point_count
    candidate = replace(initial, turns=cap - 1)

    # The occupancy observation cannot see this manually injected counter,
    # while the runner-level episode budget is intentionally kept separate
    # from this formal transition. This is not part of _all_h1_samples:
    # candidate is not a legal replay, so it is evidence for S3 review rather
    # than an H1 finding.
    assert np.array_equal(
        Cube4JapaneseGame(initial).observation(),
        Cube4JapaneseGame(candidate).observation(),
    )
    initial_after_pass = apply_v3_action(initial, topology.pass_action, topology)
    candidate_after_pass = apply_v3_action(candidate, topology.pass_action, topology)
    assert initial_after_pass.terminal_kind is None
    assert candidate_after_pass.terminal_kind is None
