from __future__ import annotations

import random

import pytest

from gocube_reference_topology import rectangular_test_topology
from katago_reference_runner import (
    KatagoOracleProcess,
    _local_state_from_setup,
    assert_snapshot_equal,
    local_snapshot,
)
from alphazero.envs.gocube.katago_v3 import apply_v3_action, v3_valid_moves


pytestmark = pytest.mark.katago_reference
SIZES = ((3, 3), (5, 3), (5, 5), (7, 4))
SEEDS = tuple(range(8))


@pytest.mark.parametrize("width,height", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_is_the_source_of_legal_moves_for_deterministic_games(width, height, seed):
    topology = rectangular_test_topology(width, height)
    local_state = _local_state_from_setup(topology, {})
    rng = random.Random(seed)
    prefix = []
    with KatagoOracleProcess(x_size=width, y_size=height, komi=0.5) as oracle:
        reference = oracle.request({"op": "snapshot"})
        assert_snapshot_equal(reference, local_snapshot(local_state, topology), context=f"seed={seed} size={width}x{height} before")
        for move_number in range(96):
            if reference.get("is_game_finished") or reference.get("is_no_result"):
                break
            legal = [index for index, value in enumerate(reference["legal_mask"]) if value]
            assert legal, f"seed={seed} size={width}x{height} move={move_number} has no oracle legal move"
            action = rng.choice(legal)
            move = "pass" if action == width * height else [action % width, action // width]
            prefix.append(move)
            response = oracle.play(move)
            assert response.get("ok", True), (
                f"seed={seed} size={width}x{height} move={move_number} "
                f"move_prefix={prefix!r} oracle rejected its own legal move"
            )
            local_legal = v3_valid_moves(local_state, topology)
            assert int(local_legal[action]) == 1, (
                f"seed={seed} size={width}x{height} move={move_number} "
                f"move_prefix={prefix!r} field=legal_mask KataGo=1 GoCube={local_legal[action]}"
            )
            local_state = apply_v3_action(local_state, action, topology)
            assert_snapshot_equal(
                response,
                local_snapshot(local_state, topology),
                context=(
                    f"seed={seed} size={width}x{height} move={move_number} "
                    f"move_prefix={prefix!r}"
                ),
            )
            reference = response
