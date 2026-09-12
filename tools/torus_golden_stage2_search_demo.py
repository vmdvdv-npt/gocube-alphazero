#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden import (
    Evaluation,
    PASS,
    SearchSettings,
    SequentialPUCT,
    initial_state,
    legal_actions,
    research_topology,
    solve_exact,
)


class UniformFakeEvaluator:
    def evaluate(self, state):
        legal = legal_actions(state)
        return Evaluation(
            policy={action: 1.0 for action in legal},
            wdl=(0.5, 0.0, 0.5),
        )


def main() -> int:
    topology = research_topology(
        ((1,), (0, 2), (1,)),
        topology_id="golden-stage2-demo-line3",
    )
    state = initial_state(topology=topology, komi=0.5)
    oracle = solve_exact(state, node_limit=10_000)
    search = SequentialPUCT(SearchSettings(simulations=128))
    result = search.search(state, UniformFakeEvaluator(), seed=20260912)

    print("Golden Search Qualification Demo")
    print("side_to_move:", state.side_to_move.name)
    print("legal actions:", list(result.legal_actions))
    print("fake policy/value: uniform / [0.5, 0.0, 0.5] side-to-move")
    print("root visits:", list(result.root_visits))
    print("selected action:", result.action)
    print("exact expected action:", list(oracle.best_actions))
    print("PASS:", PASS, "legal=", PASS in result.legal_actions)
    print("oracle:", oracle.status.value, "utility=", oracle.utility, "nodes=", oracle.nodes)
    print("search:", result.implementation_id, result.implementation_fingerprint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
