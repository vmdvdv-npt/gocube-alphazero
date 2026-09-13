#!/usr/bin/env python3
"""Reproducible Torus Golden semantic and legality-cost benchmark.

The reference mode models the pre-prepared neural/search boundary: legality is
computed by the slow validated oracle at observation, expansion, root-noise and
root-result boundaries.  It is an oracle and benchmark baseline only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.diagnostics import increment, operation_stats
from gocube_golden.neural import (
    GoldenGraphNetV1,
    GoldenNeuralEvaluator,
    SelfPlayRootNoiseEvaluator,
    build_action_mask,
    build_observation,
)
from gocube_golden.rules import (
    IllegalMoveError,
    LegalActionContext,
    apply_action,
    legal_actions,
    reference_apply_action,
    reference_legal_actions,
)
from gocube_golden.search import Evaluation, SequentialPUCT
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, EMPTY, PASS, WHITE, GoldenState, initial_state, research_state_from_stones
from gocube_golden.training import DEFAULT_SELFPLAY_CONTRACT


def _make_state(history_length: int) -> GoldenState:
    history: list[tuple[int, ...]] = []
    for index in range(history_length):
        board = [int(EMPTY)] * 25
        for bit in range(5):
            if ((index + 1) >> bit) & 1:
                board[bit] = int(BLACK)
        for point in range(25):
            if board[point] == int(EMPTY) and (point + index) % 7 < 2:
                board[point] = int(WHITE if point % 2 else BLACK)
        history.append(tuple(board))
    return research_state_from_stones(
        history[-1],
        side_to_move=BLACK if history_length % 2 else WHITE,
        superko_history=tuple(history),
    )


STATES = {
    "early": lambda: initial_state(),
    "mid": lambda: _make_state(8),
    "late": lambda: _make_state(24),
    "long-history": lambda: _make_state(120),
}


class _ReferenceAdapter:
    def prepare_legal_actions(self, state: GoldenState) -> LegalActionContext:
        actions = reference_legal_actions(state)
        mask = [False] * 26
        for action in actions:
            mask[25 if action == PASS else int(action)] = True
        return LegalActionContext(state.state_key, actions, tuple(mask))

    def apply_action(self, state: GoldenState, action: int | str) -> GoldenState:
        return reference_apply_action(state, action).after

    def is_terminal(self, state: GoldenState) -> bool:
        return state.is_terminal

    def terminal_utility(self, state: GoldenState) -> float:
        return GoldenSearchAdapter().terminal_utility(state)

    def action_index(self, state: GoldenState, action: int | str) -> int:
        return 25 if action == PASS else int(action)

    def action_space(self, state: GoldenState) -> tuple[int | str, ...]:
        return tuple(range(25)) + (PASS,)


class _ReferenceEvaluator:
    """Old evaluator boundary: scan reference legality before each NN call."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.nn_evaluations = 0

    def evaluate(self, state: GoldenState) -> Evaluation:
        actions = reference_legal_actions(state)
        observation = build_observation(state, legal_actions=actions)
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observation.unsqueeze(0))
            policy = torch.softmax(policy_logits[0], dim=0)
            wdl = torch.softmax(value_logits[0], dim=0)
        self.nn_evaluations += 1
        increment("nn_forwards")
        return Evaluation(
            policy=tuple(float(value) for value in policy),
            wdl=tuple(float(value) for value in wdl),
        )


class _ReferenceRootNoiseEvaluator:
    """Old root-noise wrapper with a second reference legality scan at root."""

    def __init__(self, evaluator: _ReferenceEvaluator, root_state: GoldenState, *, seed: int) -> None:
        self.evaluator = evaluator
        self.root_state_key = root_state.state_key
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))

    def evaluate(self, state: GoldenState) -> Evaluation:
        base = self.evaluator.evaluate(state)
        if state.state_key != self.root_state_key:
            return base
        increment("root_noise_legal_scans")
        legal = reference_legal_actions(state)
        if not legal:
            raise RuntimeError("Reference root has no legal actions")
        indices = [25 if action == PASS else int(action) for action in legal]
        prior = torch.tensor([float(base.policy[index]) for index in indices], dtype=torch.float64)
        prior = prior / prior.sum() if float(prior.sum()) > 0.0 else torch.full_like(prior, 1.0 / len(legal))
        noise = torch._standard_gamma(
            torch.full((len(legal),), 0.30, dtype=torch.float64),
            generator=self._generator,
        )
        noise = noise / noise.sum()
        mixed = 0.75 * prior + 0.25 * noise
        output = list(base.policy)
        for index, value in zip(indices, mixed.tolist()):
            output[index] = float(value)
        return Evaluation(policy=tuple(output), wdl=base.wdl)


def _timed(function: Callable[[], object]) -> tuple[float, object]:
    started = time.perf_counter()
    result = function()
    return time.perf_counter() - started, result


def _model(checkpoint: Path | None) -> tuple[torch.nn.Module, str]:
    torch.manual_seed(2026091201)
    model = GoldenGraphNetV1()
    if checkpoint is None:
        return model, "deterministic-M0"
    from gocube_golden.training import load_checkpoint

    metadata = load_checkpoint(checkpoint, model=model, device="cpu")
    return model, str(metadata.get("model_hash", "checkpoint"))


def _run_mode(mode: str, *, checkpoint: Path | None) -> dict[str, object]:
    reference = mode == "reference"
    torch.set_num_threads(1)
    model, model_id = _model(checkpoint)
    adapter = _ReferenceAdapter() if reference else GoldenSearchAdapter()
    evaluator = _ReferenceEvaluator(model) if reference else GoldenNeuralEvaluator(model)
    output: dict[str, object] = {
        "mode": mode,
        "model": model_id,
        "search_contract": DEFAULT_SELFPLAY_CONTRACT.fingerprint,
        "search_seed": 123,
        "states": {},
    }
    for label, factory in STATES.items():
        state = factory()
        legal_time, legal = _timed(
            lambda state=state: reference_legal_actions(state) if reference else legal_actions(state)
        )
        mask_time, _ = _timed(
            lambda state=state, legal=legal: build_action_mask(
                state, legal_actions=legal
            )
            if reference
            else build_action_mask(state)
        )
        observation_time, observation = _timed(
            lambda state=state, legal=legal: build_observation(
                state, legal_actions=legal
            )
            if reference
            else build_observation(state)
        )
        forward_time, _ = _timed(lambda observation=observation: model(observation.unsqueeze(0)))
        with operation_stats() as stats:
            search_time, result = _timed(
                lambda state=state: (
                    SequentialPUCT(DEFAULT_SELFPLAY_CONTRACT.puct_settings, adapter=adapter).search(
                        state,
                        _ReferenceRootNoiseEvaluator(evaluator, state, seed=456)
                        if reference
                        else SelfPlayRootNoiseEvaluator(evaluator, state, seed=456),
                        seed=123,
                    )
                )
            )
            if reference:
                # The old search materialized the root legal set again for its result.
                reference_legal_actions(state)
        counters = stats.to_dict()
        output["states"][label] = {
            "history_length": len(state.superko_history),
            "legal_action_count": len(legal),
            "legal_actions_wall_sec": legal_time,
            "action_mask_wall_sec": mask_time,
            "observation_wall_sec": observation_time,
            "pure_model_forward_wall_sec": forward_time,
            "search_wall_sec": search_time,
            "search_evaluator_calls": int(result.evaluator_calls),
            "search_root_visits": int(sum(result.root_visits)),
            "selected_action": result.action,
            "operation_counts": counters,
            "child_state_constructions": (
                counters["validated_state_constructions"]
                + counters["trusted_state_constructions"]
            ),
            "evaluator_calls": int(getattr(evaluator, "nn_evaluations", 0)),
        }
    return output


def _assert_equivalence(report: dict[str, object]) -> None:
    reference = report["reference"]["states"]
    optimized = report["optimized"]["states"]
    for label in STATES:
        left = reference[label]
        right = optimized[label]
        for key in ("legal_action_count", "search_evaluator_calls", "search_root_visits", "selected_action"):
            if left[key] != right[key]:
                raise RuntimeError(f"Reference/optimized parity drift for {label}: {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "optimized", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    modes = ("reference", "optimized") if args.mode == "both" else (args.mode,)
    report = {mode: _run_mode(mode, checkpoint=args.checkpoint) for mode in modes}
    if args.mode == "both":
        _assert_equivalence(report)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
