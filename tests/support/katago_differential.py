"""Deterministic native-KataGo differential execution for V1 CI."""

from __future__ import annotations

import random
from dataclasses import dataclass

from alphazero.envs.gocube.katago_v3 import apply_v3_action, v3_valid_moves
try:  # pytest puts ``tests`` on sys.path; regular imports do not.
    from gocube_reference_topology import rectangular_test_topology
    from katago_reference_runner import (
        KatagoOracleProcess,
        _local_state_from_setup,
        assert_snapshot_equal,
        local_snapshot,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by library-style imports
    from tests.gocube_reference_topology import rectangular_test_topology
    from tests.katago_reference_runner import (
    KatagoOracleProcess,
    _local_state_from_setup,
    assert_snapshot_equal,
    local_snapshot,
    )


V1_GENERATOR_VERSION = "gocube-v1-legal-intersection-v1"
V1_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905, 20260906, 20260907, 20260908)
V1_LENGTHS = (12, 24, 48, 12, 24, 48, 12, 24)


@dataclass(frozen=True)
class GeneratedSequenceResult:
    seed: int
    requested_steps: int
    steps_compared: int
    actions: tuple[object, ...]
    stopped: str


def run_generated_rectangular_differential(
    *,
    width: int = 5,
    height: int = 5,
    seeds: tuple[int, ...] = V1_SEEDS,
    lengths: tuple[int, ...] = V1_LENGTHS,
) -> tuple[GeneratedSequenceResult, ...]:
    """Run fixed legal sequences against native KataGo and GoCube V3.

    At every pre-action state the action is selected from the *intersection*
    of the two legal masks.  The masks are asserted equal first, so an
    accidental intersection cannot hide a disagreement.  A bound reached
    after matching all states is a successful bounded run, not a failure.
    """

    if len(seeds) != len(lengths):
        raise ValueError("seeds and lengths must have the same length")
    results = []
    topology = rectangular_test_topology(width, height)
    for seed, requested_steps in zip(seeds, lengths):
        rng = random.Random(seed)
        local_state = _local_state_from_setup(topology, {})
        actions: list[object] = []
        steps = 0
        stopped = "bound"
        with KatagoOracleProcess(x_size=width, y_size=height, komi=0.5) as oracle:
            reference = oracle.request({"op": "snapshot"})
            assert_snapshot_equal(reference, local_snapshot(local_state, topology), context=f"seed={seed} before")
            for step in range(requested_steps):
                if reference.get("is_game_finished") or reference.get("is_no_result"):
                    stopped = "terminal"
                    break
                oracle_mask = tuple(int(value) for value in reference["legal_mask"])
                local_mask = tuple(int(value) for value in v3_valid_moves(local_state, topology))
                if oracle_mask != local_mask:
                    raise AssertionError(
                        f"seed={seed} step={step} legal mask mismatch; "
                        f"actions={actions!r} KataGo={oracle_mask!r} GoCube={local_mask!r}"
                    )
                legal = tuple(index for index, value in enumerate(oracle_mask) if value and local_mask[index])
                if not legal:
                    raise AssertionError(f"seed={seed} step={step} no agreed legal action; actions={actions!r}")
                action = rng.choice(legal)
                move: object = "pass" if action == topology.pass_action else [action % width, action // width]
                actions.append(move)
                response = oracle.play(move)
                if not response.get("ok", True):
                    raise AssertionError(f"seed={seed} step={step} oracle rejected agreed action {move!r}")
                local_state = apply_v3_action(local_state, action, topology)
                response_snapshot = response.get("snapshot", response)
                assert_snapshot_equal(
                    response_snapshot,
                    local_snapshot(local_state, topology),
                    context=f"seed={seed} step={step} action={move!r} sequence={actions!r}",
                )
                reference = response_snapshot
                steps += 1
        results.append(GeneratedSequenceResult(seed, requested_steps, steps, tuple(actions), stopped))
    return tuple(results)
