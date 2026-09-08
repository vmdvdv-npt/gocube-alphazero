from __future__ import annotations

import numpy as np

from alphazero.envs.gocube import Cube2JapaneseGame, cube_topology
from alphazero.envs.gocube.katago_v3 import apply_v3_action, initial_v3_state, v3_valid_moves

from tests.support.h1_probe import (
    ReachableSample,
    bounded_reachable_states,
    find_observation_collisions,
    stable_observation_bytes,
)


def test_observation_collision_probe_uses_stable_bytes_and_detects_semantic_difference():
    samples = (
        ReachableSample("a", {"state": "A"}, category="ordinary"),
        ReachableSample("b", {"state": "B"}, category="cleanup"),
    )
    report = find_observation_collisions(
        samples,
        observation_builder=lambda _state: np.zeros((2, 3), dtype=np.float32),
        semantic_signature=lambda state: state["state"],
    )
    assert len(report.collisions) == 1
    assert report.collisions[0].semantic_collision
    assert stable_observation_bytes(np.zeros((2, 3), dtype=np.float32)) == stable_observation_bytes(
        np.zeros((2, 3), dtype=np.float32)
    )


def test_h1_probe_can_enumerate_legal_reachable_v3_states_without_global_rng():
    topology = cube_topology(2)
    initial = initial_v3_state(topology)

    def legal_actions(state):
        return tuple(int(action) for action in np.flatnonzero(v3_valid_moves(state, topology)))

    samples = bounded_reachable_states(
        initial,
        legal_actions=legal_actions,
        transition=lambda state, action: apply_v3_action(state, action, topology),
        state_key=lambda state: (
            bytes(state.board),
            state.current_player,
            state.phase,
            state.consecutive_passes,
            state.ko_recap_blocked,
            state.captures,
            state.main_moves,
            state.cleanup1_moves,
            state.cleanup2_moves,
        ),
        max_depth=2,
        max_states=48,
        category="ordinary_start",
        seed=20260908,
    )
    assert samples
    assert all(sample.seed == 20260908 for sample in samples)
    assert all(sample.category == "ordinary_start" for sample in samples)

    def observation(state):
        return Cube2JapaneseGame(state).observation()

    def semantics(state):
        mask = bytes(v3_valid_moves(state, topology).tolist())
        pass_after = apply_v3_action(state, topology.pass_action, topology)
        return (
            mask,
            state.phase,
            pass_after.phase,
            pass_after.terminal_kind,
            getattr(state, "white_bonus_score", None),
            state.captures,
        )

    report = find_observation_collisions(samples, observation, semantics)
    assert report.samples_examined == len(samples)
    assert report.observation_groups <= report.samples_examined
    # A bounded result is evidence only for this search, never a global
    # Markov-property claim.  The report remains useful if future production
    # changes expose a reachable collision.
    assert isinstance(report.semantic_collisions, tuple)
