from __future__ import annotations

import pytest

from gocube_golden.stage4 import (
    EVALUATION_PREFIX_LENGTHS,
    diagnostic_subset,
    evaluation_start_fingerprint,
    generate_evaluation_v2,
    high_reuse_schedule,
    hoeffding_interval,
    low_reuse_schedule,
    state_from_start_row,
    symmetry_canonical_state_identity,
    symmetry_identity_fingerprint,
    torus_automorphism_permutations,
)


def test_evaluation_v2_is_deterministic_stratified_and_exactly_deduplicated():
    first = generate_evaluation_v2(master_seed=2026091301)
    second = generate_evaluation_v2(master_seed=2026091301)
    assert first == second
    assert len(first) == 64
    assert [sum(row["prefix_length"] == length for row in first) for length in EVALUATION_PREFIX_LENGTHS] == [8] * 8
    assert len({row["exact_identity_fingerprint"] for row in first}) == 64
    for row in first:
        state = state_from_start_row(row)
        assert not state.is_terminal
        assert len(row["trace"]) == row["prefix_length"]
    assert evaluation_start_fingerprint(first) == evaluation_start_fingerprint(second)
    assert evaluation_start_fingerprint(first) != evaluation_start_fingerprint(generate_evaluation_v2(master_seed=2026091302))


def test_full_history_symmetry_canonicalization_is_formally_defined():
    assert len(torus_automorphism_permutations()) == 200
    row = generate_evaluation_v2()[0]
    identity = row["state"]
    permutation = torus_automorphism_permutations()[37]
    from gocube_golden.stage4 import _transform_state_identity

    transformed = _transform_state_identity(identity, permutation)
    assert symmetry_canonical_state_identity(identity) == symmetry_canonical_state_identity(transformed)
    assert symmetry_identity_fingerprint(identity) == symmetry_identity_fingerprint(transformed)
    assert identity["superko_history"] != transformed["superko_history"] or identity["stones"] == transformed["stones"]


def test_diagnostic_subset_is_two_starts_per_stratum():
    subset = diagnostic_subset(generate_evaluation_v2())
    assert len(subset) == 16
    assert [sum(row["prefix_length"] == length for row in subset) for length in EVALUATION_PREFIX_LENGTHS] == [2] * 8


def test_hoeffding_interval_does_not_degenerate_at_zero_sample_variance():
    lo, hi = hoeffding_interval([0.5] * 8)
    assert lo < 0.5 < hi
    assert hoeffding_interval([0.5] * 64) == pytest.approx((0.330237, 0.669763), abs=1e-5)


def test_ablation_schedules_account_for_exact_samples_and_no_replacement():
    high = high_reuse_schedule(17, sample_budget=25600, batch_size=64, seed=7)
    assert len(high) == 400
    assert sum(map(len, high)) == 25600
    low = low_reuse_schedule(130, batch_size=64, seed=7)
    flattened = [index for batch in low for index in batch]
    assert [len(batch) for batch in low] == [64, 64, 2]
    assert len(flattened) == 130
    assert len(set(flattened)) == 130
