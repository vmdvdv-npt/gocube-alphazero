from __future__ import annotations

import torch

import gocube_golden as g
import gocube_golden.torus9_monolith as torus9_monolith
from gocube_golden.torus9 import (
    build_torus9_observation,
    build_torus9_observation_into,
)


def test_torus9_network_komi_plane_stays_at_historical_half_point() -> None:
    state = g.initial_state(topology=g.TORUS_9X9, komi=4.5)
    context = g.prepare_legal_actions(state)

    observation = build_torus9_observation(state, legal_context=context)
    assert torch.all(observation[5] == 0.5)

    direct = torch.empty((6, 81), dtype=torch.float32)
    build_torus9_observation_into(state, direct, legal_context=context)
    assert torch.all(direct[5] == 0.5)

    monolith_direct = torch.empty((6, 81), dtype=torch.float32)
    torus9_monolith.build_torus9_observation_into(
        state,
        monolith_direct,
        legal_context=context,
    )
    assert torch.all(monolith_direct[5] == 0.5)

    # The feature plane is pinned only for the network.  The immutable game
    # state and referee still use the real calibrated komi.
    assert state.komi == 4.5
    first_pass = g.apply_action(state, g.PASS).after
    terminal = g.apply_action(first_pass, g.PASS).after
    score = g.score_terminal(terminal)
    assert score.komi == 4.5
    assert score.margin_black == -4.5
