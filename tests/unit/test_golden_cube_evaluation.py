from __future__ import annotations

from gocube_golden.cube_evaluation import (
    CUBE_EVALUATION_PREFIX_LENGTHS,
    cube_evaluation_fingerprint,
    diagnostic_cube_subset,
    generate_cube_evaluation,
)
from gocube_golden.cube_training import cube_state_from_identity
from gocube_golden.cube_topology import CUBE4_TOPOLOGY
from gocube_golden.rules import apply_action


def test_cube_evaluation_v1_has_independent_strata_and_exact_semantic_identity():
    starts = generate_cube_evaluation()
    assert len(starts) == 64
    assert [sum(int(row["prefix_length"]) == prefix for row in starts) for prefix in CUBE_EVALUATION_PREFIX_LENGTHS] == [8] * 8
    identities = [str(row["state_fingerprint"]) for row in starts]
    assert len(identities) == len(set(identities))
    assert all(cube_state_from_identity(row["state"]).topology.fingerprint == CUBE4_TOPOLOGY.fingerprint for row in starts)
    assert all(not cube_state_from_identity(row["state"]).is_terminal for row in starts)


def test_cube_evaluation_starts_replay_exactly_and_are_nonpass_prefixes():
    for row in generate_cube_evaluation():
        state = cube_state_from_identity(row["state"])
        replayed = __import__("gocube_golden.cube_training", fromlist=["cube_initial_state"]).cube_initial_state()
        for action in row["trace"]:
            assert isinstance(action, int)
            replayed = apply_action(replayed, action).after
        assert replayed.state_key == state.state_key


def test_cube_diagnostic_subset_is_two_per_stratum_and_fingerprint_is_deterministic():
    starts_a = generate_cube_evaluation()
    starts_b = generate_cube_evaluation()
    subset = diagnostic_cube_subset(starts_a)
    assert len(subset) == 16
    assert [sum(int(row["prefix_length"]) == prefix for row in subset) for prefix in CUBE_EVALUATION_PREFIX_LENGTHS] == [2] * 8
    assert starts_a == starts_b
    assert cube_evaluation_fingerprint(starts_a) == cube_evaluation_fingerprint(starts_b)
