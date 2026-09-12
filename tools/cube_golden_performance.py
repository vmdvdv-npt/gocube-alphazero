#!/usr/bin/env python3
"""Reproducible Cube Golden rules/search cost and operation-count benchmark.

The reference mode intentionally preserves the pre-optimization search boundary:
each legality scan constructs fully validated hypothetical GoldenState children.
It is an oracle/performance baseline only; it is never used for self-play data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden import cube_neural
from gocube_golden.cube_neural import (
    GoldenCubeGraphNetV1,
    GoldenCubeNeuralEvaluator,
    build_cube_action_mask,
    build_cube_observation,
    configure_single_thread_inference,
)
from gocube_golden.cube_topology import CUBE4_TOPOLOGY
from gocube_golden.cube_training import (
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    cube_initial_state,
)
from gocube_golden.diagnostics import operation_stats
from gocube_golden.rules import (
    IllegalMoveError,
    LegalActionContext,
    reference_apply_action,
    reference_legal_actions,
)
from gocube_golden.search import SequentialPUCT
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, WHITE, research_state_from_stones


def _make_state(stones_count: int, history_length: int):
    history = []
    for index in range(history_length):
        board = [0] * CUBE4_TOPOLOGY.point_count
        for bit in range(12):
            if ((index + 1) >> bit) & 1:
                board[bit] = 1
        for point in range(min(stones_count, CUBE4_TOPOLOGY.point_count)):
            if board[point] == 0 and (point + index) % 7 < 3:
                board[point] = 2 if point % 2 else 1
        history.append(tuple(board))
    return research_state_from_stones(
        history[-1],
        side_to_move=BLACK if history_length % 2 else WHITE,
        topology=CUBE4_TOPOLOGY,
        superko_history=tuple(history),
    )


STATES = {
    "early": lambda: cube_initial_state(),
    "mid": lambda: _make_state(28, 25),
    "late": lambda: _make_state(62, 80),
    "long-history": lambda: _make_state(40, 200),
}


class _ReferenceAdapter:
    """Adapter that routes every speculative transition through the old oracle."""

    def prepare_legal_actions(self, state):
        actions = reference_legal_actions(state)
        mask = [False] * (state.topology.point_count + 1)
        for action in actions:
            mask[state.topology.point_count if action == "PASS" else int(action)] = True
        return LegalActionContext(state.state_key, actions, tuple(mask))

    def apply_action(self, state, action):
        return reference_apply_action(state, action).after

    def is_terminal(self, state):
        return state.is_terminal

    def terminal_utility(self, state):
        return GoldenSearchAdapter().terminal_utility(state)

    def action_index(self, state, action):
        return state.topology.point_count if action == "PASS" else int(action)

    def action_space(self, state):
        return tuple(range(state.topology.point_count)) + ("PASS",)


class _ReferenceEvaluator:
    """Do not expose evaluate_prepared: the old evaluator scans legality itself."""

    def __init__(self, evaluator: GoldenCubeNeuralEvaluator):
        self.evaluator = evaluator

    def evaluate(self, state):
        return self.evaluator.evaluate(state)


def _timed(function: Callable[[], object]) -> tuple[float, object]:
    started = time.perf_counter()
    result = function()
    return time.perf_counter() - started, result


def _reference_prepare(state):
    actions = reference_legal_actions(state)
    mask = [False] * (state.topology.point_count + 1)
    for action in actions:
        mask[state.topology.point_count if action == "PASS" else int(action)] = True
    return LegalActionContext(state.state_key, actions, tuple(mask))


def _run_mode(mode: str) -> dict[str, object]:
    is_reference = mode == "reference"
    old_prepare = cube_neural.prepare_legal_actions
    if is_reference:
        cube_neural.prepare_legal_actions = _reference_prepare
    try:
        torch.manual_seed(2026091401)
        configure_single_thread_inference()
        model = GoldenCubeGraphNetV1()
        adapter = _ReferenceAdapter() if is_reference else None
        evaluator = GoldenCubeNeuralEvaluator(model)
        output: dict[str, object] = {
            "mode": mode,
            "threading": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
            },
            "states": {},
        }
        for label, factory in STATES.items():
            state = factory()
            legal_time, legal = _timed(
                lambda state=state: reference_legal_actions(state)
                if is_reference
                else __import__("gocube_golden.rules", fromlist=["legal_actions"]).legal_actions(state)
            )
            legal_point = next((action for action in legal if action != "PASS"), None)
            apply_time = None
            if legal_point is not None:
                apply_time, _ = _timed(
                    lambda state=state, action=legal_point: reference_apply_action(state, action)
                    if is_reference
                    else __import__("gocube_golden.rules", fromlist=["apply_action"]).apply_action(state, action)
                )
            failed_point = next(
                (point for point, stone in enumerate(state.stones) if int(stone) != 0), None
            )
            failed_time = None
            if failed_point is not None:
                def failed_probe(state=state, point=failed_point):
                    try:
                        if is_reference:
                            reference_apply_action(state, point)
                        else:
                            __import__("gocube_golden.rules", fromlist=["apply_action"]).apply_action(state, point)
                    except IllegalMoveError:
                        pass
                failed_time, _ = _timed(failed_probe)
            mask_time, _ = _timed(lambda state=state: build_cube_action_mask(state))
            observation_time, observation = _timed(
                lambda state=state: build_cube_observation(state)
            )
            forward_time, _ = _timed(
                lambda observation=observation: model(observation.unsqueeze(0))
            )
            search_adapter = adapter or __import__(
                "gocube_golden.search_adapter", fromlist=["GoldenSearchAdapter"]
            ).GoldenSearchAdapter()
            with operation_stats() as stats:
                search_time, result = _timed(
                    lambda state=state: _search_with_reference_final_scan(
                        state, search_adapter, evaluator
                    )
                    if is_reference
                    else SequentialPUCT(
                        DEFAULT_CUBE_SELFPLAY_CONTRACT.puct_settings,
                        adapter=search_adapter,
                    ).search(state, evaluator, seed=123)
                )
            output["states"][label] = {
                "history_length": len(state.superko_history),
                "legal_action_count": len(legal),
                "legal_actions_sec": 1.0 / legal_time if legal_time else None,
                "legal_actions_wall_sec": legal_time,
                "apply_action_wall_sec": apply_time,
                "failed_hypothetical_wall_sec": failed_time,
                "action_mask_wall_sec": mask_time,
                "observation_wall_sec": observation_time,
                "pure_model_forward_wall_sec": forward_time,
                "search_wall_sec": search_time,
                "search_evaluator_calls": result.evaluator_calls,
                "search_root_visits": sum(result.root_visits),
                "operation_counts": stats.to_dict(),
                "evaluator_telemetry": evaluator.telemetry(),
            }
        return output
    finally:
        cube_neural.prepare_legal_actions = old_prepare


def _search_with_reference_final_scan(state, adapter, evaluator):
    result = SequentialPUCT(
        DEFAULT_CUBE_SELFPLAY_CONTRACT.puct_settings, adapter=adapter
    ).search(state, _ReferenceEvaluator(evaluator), seed=123)
    # The pre-optimization SequentialPUCT performed this final root scan to
    # materialize its result, even though root expansion had already scanned it.
    reference_legal_actions(state)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "optimized", "both"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    modes = ("reference", "optimized") if args.mode == "both" else (args.mode,)
    report = {mode: _run_mode(mode) for mode in modes}
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
