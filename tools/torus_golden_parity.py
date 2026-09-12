#!/usr/bin/env python3
"""Fail-closed correctness, determinism and Stage-4 parity gate for Golden Torus.

The script deliberately uses the existing slow Golden referee as an oracle and
the existing frozen Stage-4 schedule.  It does not tune a model or introduce a
new scientific workload.  A canonical run stops at the first mismatch and
leaves its partial evidence under ``runs/`` for investigation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.diagnostics import operation_stats
from gocube_golden.neural import (
    GoldenGraphNetV1,
    GoldenNeuralEvaluator,
    SelfPlayRootNoiseEvaluator,
    build_action_mask,
    build_observation_bundle,
    model_hash,
)
from gocube_golden.result import result_from_terminal
from gocube_golden.rules import (
    IllegalMoveError,
    LegalActionContext,
    apply_action,
    prepare_legal_actions,
    probe_action,
    reference_apply_action,
    reference_legal_actions,
)
from gocube_golden.search import (
    Evaluation,
    SEARCH_IMPLEMENTATION_FINGERPRINT,
    SequentialPUCT,
)
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.stage3_contract import load_profile as load_stage3_profile
from gocube_golden.state import (
    BLACK,
    EMPTY,
    PASS,
    WHITE,
    GoldenState,
    initial_state,
    research_state_from_stones,
)
from gocube_golden.topology import GoldenTopology, research_topology
from gocube_golden.training import (
    DEFAULT_SELFPLAY_CONTRACT,
    GoldenSelfPlayRunner,
    GoldenTrainingSample,
    SelfPlayGameRecord,
    build_replay_samples,
    load_checkpoint,
    sample_action_from_visits,
    save_checkpoint,
    state_identity,
    write_jsonl,
    z_target,
)
from gocube_golden.provenance import capture_code_identity, derive_seed, file_sha256, sha256_fingerprint

from tools.torus_golden_performance import (
    _ReferenceAdapter,
    _ReferenceEvaluator,
    _ReferenceRootNoiseEvaluator,
    _run_mode,
)
from tools.torus_golden_stage4 import (
    checkpoint_info,
    checkpoint_metadata,
    records_from_jsonl,
    run_comparison,
    run_games,
    stage4_profile,
    write_json,
)
from gocube_golden.stage4 import (
    EVALUATION_ACCEPTED_TOTAL,
    EVALUATION_MASTER_SEED,
    EVALUATION_PREFIX_LENGTHS,
    STAGE4_ARENA_MASTER_SEED,
    STAGE4_MODEL_INIT_SEED,
    STAGE4_PROFILE_ID,
    STAGE4_SELFPLAY_MASTER_SEED,
    audit_selfplay_records,
    evaluation_start_fingerprint,
    load_frozen_starts,
    state_from_start_row,
    summarize_pair_records,
    train_batch_schedule,
)


OLD_RUN_ID = "torus-golden-stage4-seed2-v4"
OLD_RUN = ROOT / "runs" / "torus-golden-stage4" / OLD_RUN_ID
OLD_SEED_NAMESPACE = OLD_RUN_ID
POINT_ACTIONS = tuple(range(25)) + (PASS,)
INVALID_ACTIONS = (-1, 25, 26, True, None, "0")
EXPECTED_ARENA = {
    "M1-M0": (91, 37, 0),
    "M4-M0": (124, 4, 0),
    "M4-M1": (113, 15, 0),
    "M2-M1": (25, 7, 0),
    "M3-M2": (19, 13, 0),
    "M4-M3": (18, 14, 0),
}


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    return str(value)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _synthetic_history_state(history_length: int, *, topology: GoldenTopology | None = None) -> GoldenState:
    topology = topology or initial_state().topology
    count = topology.point_count
    history: list[tuple[int, ...]] = []
    for index in range(history_length):
        board = [int(EMPTY)] * count
        for point in range(count):
            marker = (index * 17 + point * 7 + 3) % 19
            if marker < 3:
                board[point] = int(BLACK if (index + point) % 2 else WHITE)
        history.append(tuple(board))
    return research_state_from_stones(
        history[-1],
        side_to_move=BLACK if history_length % 2 else WHITE,
        topology=topology,
        superko_history=tuple(history),
    )


def _torus_topology(size: int) -> GoldenTopology:
    adjacency = []
    for y in range(size):
        for x in range(size):
            adjacency.append((
                ((y - 1) % size) * size + x,
                y * size + ((x + 1) % size),
                ((y + 1) % size) * size + x,
                y * size + ((x - 1) % size),
            ))
    return research_topology(adjacency, topology_id=f"parity-diagnostic-torus-{size}x{size}")


def _corpus() -> tuple[tuple[str, GoldenState], ...]:
    """Build deterministic reference-legal and explicit edge-case states."""
    rows: list[tuple[str, GoldenState]] = [("empty", initial_state())]
    seen = {rows[0][1].state_key}
    capture_state: GoldenState | None = None
    suicide_state: GoldenState | None = None
    superko_state: GoldenState | None = None
    released_state: GoldenState | None = None

    # Every state in this block is reached by applying a reference-legal move.
    trajectory_seeds = (17, 29, 41, 53, 67, 79, 97, 113, 127, 149)
    for seed in trajectory_seeds:
        rng = random.Random(seed)
        state = initial_state()
        for step in range(80):
            if state.state_key not in seen:
                rows.append((f"trajectory-{seed}-{step}", state))
                seen.add(state.state_key)
            legal = reference_legal_actions(state)
            if state.is_terminal:
                break
            placements = tuple(action for action in legal if action != PASS)
            if not placements:
                action = PASS
            elif state.consecutive_passes == 1 or step % 17 == 16:
                action = PASS
            else:
                action = rng.choice(placements)
            transition = reference_apply_action(state, action)
            if transition.captured and capture_state is None:
                capture_state = state
                released_state = transition.after
            state = transition.after

    # Ensure one-pass and formal terminal states are present explicitly.
    one_pass = reference_apply_action(initial_state(), PASS).after
    terminal = reference_apply_action(one_pass, PASS).after
    rows.extend((("one-pass", one_pass), ("terminal-double-pass", terminal)))

    # Find a real suicide candidate and turn one local legal move into an exact
    # synthetic superko candidate without changing its local board mechanics.
    for _, state in rows:
        if state.is_terminal:
            continue
        for action in range(25):
            try:
                reference_apply_action(state, action)
            except IllegalMoveError as exc:
                if exc.reason.value == "suicide" and suicide_state is None:
                    suicide_state = state
            else:
                if superko_state is None:
                    after = reference_apply_action(state, action).after
                    if after.board_key != state.board_key:
                        superko_state = research_state_from_stones(
                            state.stones,
                            side_to_move=state.side_to_move,
                            superko_history=(state.board_key, after.board_key),
                        )
            if suicide_state is not None and superko_state is not None:
                break
        if suicide_state is not None and superko_state is not None:
            break

    # Fallbacks keep the gate explicit even if a random corpus happens not to
    # hit a rare local shape on a future interpreter implementation.
    if capture_state is None:
        capture_state = research_state_from_stones(
            (0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            side_to_move=BLACK,
        )
    if released_state is None:
        released_state = capture_state
    if suicide_state is None:
        suicide_state = research_state_from_stones(
            (2, 1, 2, 0, 2, 1, 0, 0, 0, 0, 2, 1, 2, 0, 2, 1, 0, 0, 0, 0, 2, 1, 2, 0, 2),
            side_to_move=WHITE,
        )
    if superko_state is None:
        state = initial_state()
        after = reference_apply_action(state, 0).after
        superko_state = research_state_from_stones(
            state.stones, side_to_move=state.side_to_move,
            superko_history=(state.board_key, after.board_key),
        )
    special = (
        ("capture", capture_state),
        ("released-after-capture", released_state),
        ("suicide-candidate", suicide_state),
        ("superko-candidate", superko_state),
        ("long-history-120", _synthetic_history_state(120)),
        ("long-history-400", _synthetic_history_state(400)),
    )
    for label, state in special:
        if state.state_key not in seen:
            rows.append((label, state))
            seen.add(state.state_key)
    return tuple(rows)


def contract_snapshot(semantic: Mapping[str, object], stage4: Mapping[str, object]) -> dict[str, object]:
    selfplay = semantic["self_play"]
    arena = semantic["arena"]
    _assert(float(semantic["komi"]) == 0.5, "Golden contract komi is not 0.5")
    _assert(float(selfplay["komi"]) == 0.5, "Self-play contract komi is not 0.5")
    _assert(float(stage4.get("model_init_seed", STAGE4_MODEL_INIT_SEED)) == float(STAGE4_MODEL_INIT_SEED), "Stage4 model seed drift")
    puct = DEFAULT_SELFPLAY_CONTRACT.puct_settings
    return {
        "golden_torus": {
            "topology": "Torus 5x5",
            "points": 25,
            "actions": 26,
            "pass_action_index": 25,
            "komi": 0.5,
            "suicide": "forbidden",
            "superko": "positional",
            "pass_exempt_from_repetition": True,
            "two_passes_terminal": True,
            "scoring": "graph area",
            "resign": False,
            "cleanup": False,
            "benson": False,
            "watchdog": 500,
        },
        "search": {
            "implementation": "SequentialPUCT",
            "simulations": puct.simulations,
            "cpuct": puct.cpuct,
            "fpu": puct.fpu,
            "tree_reuse": False,
            "transpositions": False,
            "virtual_loss": False,
            "batching": False,
            "deterministic_tie_break": True,
            "fingerprint": SEARCH_IMPLEMENTATION_FINGERPRINT,
        },
        "arena": {
            "root_noise": arena["root_noise"],
            "temperature": arena["move_temperature"],
            "fast_search": arena["fast_search"],
            "resign": arena["resign"],
        },
        "self_play": {
            "profile": stage4["profile_id"],
            "simulations": selfplay["simulations"],
            "cpuct": selfplay["cpuct"],
            "fpu": selfplay["fpu"],
            "root_noise": selfplay["root_noise"],
            "dirichlet_epsilon": selfplay["dirichlet_epsilon"],
            "dirichlet_alpha": selfplay["dirichlet_alpha"],
            "temperature_plies": selfplay["temperature_plies"],
            "temperature_after": selfplay["temperature_after"],
            "watchdog": selfplay["watchdog"],
        },
        "seeds": {
            "model_initialization": STAGE4_MODEL_INIT_SEED,
            "self_play_master": STAGE4_SELFPLAY_MASTER_SEED,
            "arena_master": STAGE4_ARENA_MASTER_SEED,
            "evaluation_master": EVALUATION_MASTER_SEED,
            "seed_namespace_for_old_new_parity": OLD_SEED_NAMESPACE,
        },
        "training_profile": {
            "profile_id": stage4["profile_id"],
            "optimizer": stage4["training"]["optimizer"],
            "learning_rate": stage4["training"]["learning_rate"],
            "weight_decay": stage4["training"]["weight_decay"],
            "batch_size": stage4["training"]["batch_size"],
            "replay": stage4["training"]["replay"],
            "sampling": stage4["training"]["sampling"],
            "chunks": stage4["selfplay"]["chunks"],
            "games_per_chunk": stage4["selfplay"]["games_per_chunk"],
            "workers": stage4["selfplay"]["workers"],
        },
    }


def optimized_path_audit() -> dict[str, object]:
    files = {
        "state": ROOT / "gocube_golden" / "state.py",
        "rules": ROOT / "gocube_golden" / "rules.py",
        "search": ROOT / "gocube_golden" / "search.py",
        "search_adapter": ROOT / "gocube_golden" / "search_adapter.py",
        "neural": ROOT / "gocube_golden" / "neural.py",
        "training": ROOT / "gocube_golden" / "training.py",
    }
    source = {name: path.read_text(encoding="utf-8") for name, path in files.items()}
    required = {
        "state_exact_superko_membership": "superko_membership" in source["state"],
        "state_trusted_internal_transition": "_from_trusted_transition" in source["state"],
        "rules_probe_action": "def probe_action" in source["rules"],
        "rules_legal_context": "class LegalActionContext" in source["rules"],
        "search_prepares_context": "prepare_legal_actions" in source["search_adapter"] and "self.adapter.prepare_legal_actions" in source["search"],
        "search_evaluate_prepared": "evaluate_prepared" in source["search"],
        "neural_prepared_observation": "legal_context=legal_context" in source["neural"],
        "neural_evaluate_prepared": "def evaluate_prepared" in source["neural"],
        "training_prepared_replay": "build_observation(state, legal_context=legal_context)" in source["training"],
        "root_noise_reuses_context": "root_noise_legal_reuses" in source["neural"],
        "search_no_direct_legal_actions_scan": "self.adapter.legal_actions(" not in source["search"].split("class SequentialPUCT", 1)[1].split("class SolveStatus", 1)[0],
    }
    _assert(all(required.values()), "Optimized path audit failed: " + ", ".join(key for key, value in required.items() if not value))
    return {
        "files": {name: str(path) for name, path in files.items()},
        "required_symbols": required,
        "optimized_operations": [
            "exact superko_membership lookup",
            "trusted internal transitions after local rule checks",
            "probe_action for legality and selected child data",
            "LegalActionContext reused from search into evaluator/observation",
            "prepared observation and evaluate_prepared",
            "prepared legality reused by root Dirichlet noise",
            "one legality calculation per expanded node",
            "no full-history validation in optimized search hot path",
        ],
        "old_full_history_scan_in_optimized_path": False,
    }


def _mask(actions: Sequence[int | str]) -> tuple[bool, ...]:
    result = [False] * 26
    for action in actions:
        result[25 if action == PASS else int(action)] = True
    return tuple(result)


def _error_reason(function, state: GoldenState, action: object) -> str | None:
    try:
        function(state, action)
    except IllegalMoveError as exc:
        return exc.reason.value
    return None


def _transition_summary(transition) -> dict[str, object]:
    after = transition.after
    terminal_result = result_from_terminal(after) if after.is_terminal else None
    return {
        "action": transition.action,
        "captures": transition.captured,
        "stones": after.stones,
        "side_to_move": after.side_to_move,
        "consecutive_passes": after.consecutive_passes,
        "superko_history": after.superko_history,
        "superko_membership": after.superko_membership,
        "state_identity": after.state_key,
        "terminal": after.is_terminal,
        "score": None if terminal_result is None else result_from_terminal(after),
        "winner": None if terminal_result is None else terminal_result.winner,
    }


def rules_equivalence(corpus: Sequence[tuple[str, GoldenState]]) -> dict[str, object]:
    states = 0
    action_cases = 0
    legal_cases = 0
    reason_counts: dict[str, int] = {}
    category_hits = {"capture": 0, "suicide": 0, "superko": 0, "pass": 0, "terminal": 0}
    for label, state in corpus:
        states += 1
        reference_legal = reference_legal_actions(state)
        optimized_legal = prepare_legal_actions(state).actions
        _assert(optimized_legal == reference_legal, f"Rules legal-action mismatch at {label}")
        optimized_mask = prepare_legal_actions(state).action_mask if state.is_terminal else build_action_mask(state)
        _assert(optimized_mask == _mask(reference_legal), f"Rules mask mismatch at {label}")
        if state.is_terminal:
            category_hits["terminal"] += 1
        if PASS in reference_legal:
            category_hits["pass"] += 1
        for action in POINT_ACTIONS + INVALID_ACTIONS:
            action_cases += 1
            left_reason = _error_reason(apply_action, state, action)
            right_reason = _error_reason(reference_apply_action, state, action)
            _assert(left_reason == right_reason, f"IllegalMoveReason mismatch at {label}/{action!r}: {left_reason} != {right_reason}")
            if right_reason is not None:
                reason_counts[right_reason] = reason_counts.get(right_reason, 0) + 1
            else:
                legal_cases += 1
                optimized = apply_action(state, action)
                reference = reference_apply_action(state, action)
                _assert(_transition_summary(optimized) == _transition_summary(reference), f"Transition mismatch at {label}/{action!r}")
                if action != PASS:
                    prepared_probe = probe_action(state, action)
                    _assert(prepared_probe.captured == reference.captured, f"Probe capture mismatch at {label}/{action!r}")
                    _assert(prepared_probe.stones == reference.after.stones, f"Probe stones mismatch at {label}/{action!r}")
                    _assert(prepared_probe.board_key == reference.after.board_key, f"Probe board key mismatch at {label}/{action!r}")
            if right_reason == "suicide":
                category_hits["suicide"] += 1
            elif right_reason == "superko":
                category_hits["superko"] += 1
            elif right_reason is None and action != PASS and reference.captured:
                category_hits["capture"] += 1
    for category in ("capture", "suicide", "superko"):
        _assert(category_hits[category] > 0, f"Rules corpus did not exercise {category}")
    return {
        "status": "IDENTICAL",
        "states": states,
        "action_cases": action_cases,
        "legal_transition_cases": legal_cases,
        "illegal_move_reasons": reason_counts,
        "corpus_categories": category_hits,
        "semantic_fields": ["legal actions", "legal mask", "probe legality", "captures", "stones", "side_to_move", "consecutive_passes", "superko_history", "superko_membership", "state identity", "terminal", "score", "winner"],
        "mismatches": 0,
    }


def fuzz_equivalence() -> dict[str, object]:
    seeds = (2026091201, 2026091207, 2026091211, 2026091213, 2026091217, 2026091219)
    combinations = 0
    trajectories = 0
    for seed in seeds:
        rng = random.Random(seed)
        state = initial_state()
        trajectories += 1
        for step in range(90):
            if state.is_terminal:
                break
            reference_legal = reference_legal_actions(state)
            optimized_legal = prepare_legal_actions(state).actions
            _assert(reference_legal == optimized_legal, f"Fuzz legal mismatch seed={seed} step={step}")
            action = rng.choice(tuple(reference_legal))
            left = _error_reason(apply_action, state, action)
            right = _error_reason(reference_apply_action, state, action)
            _assert(left == right is None, f"Fuzz selected action mismatch seed={seed} step={step}")
            _assert(_transition_summary(apply_action(state, action)) == _transition_summary(reference_apply_action(state, action)), f"Fuzz transition mismatch seed={seed} step={step}")
            state = reference_apply_action(state, action).after
            combinations += 1
    return {"status": "IDENTICAL", "seeds": seeds, "trajectories": trajectories, "state_action_combinations": combinations, "mismatches": 0}


def observation_equivalence(corpus: Sequence[tuple[str, GoldenState]]) -> dict[str, object]:
    rows = []
    for label, state in corpus:
        if state.is_terminal:
            continue
        context = prepare_legal_actions(state)
        old = build_observation_bundle(state, legal_actions=reference_legal_actions(state))
        new = build_observation_bundle(state, legal_context=context)
        _assert(old.tensor.shape == new.tensor.shape and old.tensor.dtype == new.tensor.dtype, f"Observation shape/dtype mismatch at {label}")
        _assert(old.tensor.detach().cpu().numpy().tobytes() == new.tensor.detach().cpu().numpy().tobytes(), f"Observation tensor mismatch at {label}")
        _assert(old.action_mask == new.action_mask == context.action_mask, f"Observation mask mismatch at {label}")
        _assert(old.state_key == new.state_key == state.state_key, f"Observation state key mismatch at {label}")
        rows.append(label)
    return {"status": "IDENTICAL", "states": len(rows), "state_labels": rows, "tensor_bit_identical": True, "mask_identical": True, "state_key_identical": True, "mismatches": 0}


def _compare_eval(left: Evaluation, right: Evaluation, label: str) -> dict[str, float]:
    policy_diff = max(abs(float(a) - float(b)) for a, b in zip(left.policy, right.policy))
    wdl_diff = max(abs(float(a) - float(b)) for a, b in zip(left.wdl, right.wdl))
    _assert(tuple(left.policy) == tuple(right.policy), f"Evaluator policy mismatch at {label}")
    _assert(tuple(left.wdl) == tuple(right.wdl), f"Evaluator WDL mismatch at {label}")
    return {"max_abs_policy_diff": policy_diff, "max_abs_wdl_diff": wdl_diff}


def evaluator_equivalence(model: torch.nn.Module, corpus: Sequence[tuple[str, GoldenState]]) -> dict[str, object]:
    reference = _ReferenceEvaluator(model)
    optimized = GoldenNeuralEvaluator(model)
    diffs = []
    for label, state in corpus:
        if state.is_terminal:
            continue
        diffs.append(_compare_eval(reference.evaluate(state), optimized.evaluate_prepared(state, prepare_legal_actions(state)), label))
    return {"status": "IDENTICAL", "states": len(diffs), "policy_normalized": True, "wdl_normalized": True, "bit_exact": True, "max_abs_policy_diff": max(row["max_abs_policy_diff"] for row in diffs), "max_abs_wdl_diff": max(row["max_abs_wdl_diff"] for row in diffs), "mismatches": 0}


def root_noise_equivalence(model: torch.nn.Module, corpus: Sequence[tuple[str, GoldenState]]) -> dict[str, object]:
    rows = []
    for index, (label, state) in enumerate(corpus):
        if state.is_terminal:
            continue
        seed = 7000 + index
        context = prepare_legal_actions(state)
        old = _ReferenceRootNoiseEvaluator(_ReferenceEvaluator(model), state, seed=seed)
        new = SelfPlayRootNoiseEvaluator(GoldenNeuralEvaluator(model), state, seed=seed)
        first = _compare_eval(old.evaluate(state), new.evaluate_prepared(state, context), label)
        second = _compare_eval(old.evaluate(state), new.evaluate_prepared(state, context), label + "-second")
        _assert(old.evaluator is not None and context.actions == tuple(reference_legal_actions(state)), f"Root-noise legal ordering mismatch at {label}")
        rows.append({"label": label, "seed": seed, "legal_action_order_identical": True, "dirichlet_draw_and_mixed_policy_identical": True, "first": first, "second": second})
    return {"status": "IDENTICAL", "states": len(rows), "same_seed_same_mixed_policy": True, "rng_consumption_identical": True, "rows": rows, "mismatches": 0}


def search_equivalence(model: torch.nn.Module, corpus: Sequence[tuple[str, GoldenState]]) -> dict[str, object]:
    rows = []
    for index, (label, state) in enumerate(corpus):
        if state.is_terminal:
            continue
        seed = 9000 + index
        old_trace: list[dict[str, object]] = []
        new_trace: list[dict[str, object]] = []
        old = SequentialPUCT(DEFAULT_SELFPLAY_CONTRACT.puct_settings, adapter=_ReferenceAdapter(), trace=old_trace).search(
            state, _ReferenceRootNoiseEvaluator(_ReferenceEvaluator(model), state, seed=seed + 1), seed=seed
        )
        new = SequentialPUCT(DEFAULT_SELFPLAY_CONTRACT.puct_settings, adapter=GoldenSearchAdapter(), trace=new_trace).search(
            state, SelfPlayRootNoiseEvaluator(GoldenNeuralEvaluator(model), state, seed=seed + 1), seed=seed
        )
        for field in ("legal_actions", "legal_action_mask", "root_visits", "pi", "action", "root_q", "evaluator_calls", "simulations"):
            _assert(getattr(old, field) == getattr(new, field), f"Search {field} mismatch at {label}")
        _assert(old_trace == new_trace, f"Full-tree PUCT trace mismatch at {label}")
        rows.append({"label": label, "seed": seed, "root_visits": old.root_visits, "selected_action": old.action, "evaluator_calls": old.evaluator_calls, "trace_events": len(old_trace), "full_tree_trace_bit_identical": True})
    return {"status": "IDENTICAL", "states": len(rows), "simulations": 64, "cpuct": 1.25, "fpu": 0.0, "rows": rows, "full_tree_trace": "BIT-IDENTICAL", "mismatches": 0}


def _play_full_game(model: torch.nn.Module, *, seed_namespace: str, game_id: str, reference: bool) -> dict[str, object]:
    game_seed = derive_seed(STAGE4_SELFPLAY_MASTER_SEED, seed_namespace, game_id, "game")
    rng = random.Random(game_seed)
    state = initial_state()
    adapter = _ReferenceAdapter() if reference else GoldenSearchAdapter()
    evaluator = _ReferenceEvaluator(model) if reference else GoldenNeuralEvaluator(model)
    rows = []
    for ply in range(1, DEFAULT_SELFPLAY_CONTRACT.watchdog + 1):
        search_seed = derive_seed(game_seed, ply, "search")
        wrapper = _ReferenceRootNoiseEvaluator(evaluator, state, seed=derive_seed(search_seed, "dirichlet")) if reference else SelfPlayRootNoiseEvaluator(evaluator, state, seed=derive_seed(search_seed, "dirichlet"))
        result = SequentialPUCT(DEFAULT_SELFPLAY_CONTRACT.puct_settings, adapter=adapter).search(state, wrapper, seed=search_seed)
        action = sample_action_from_visits(result, temperature=1.0 if ply <= 8 else 0.0, rng=rng)
        transition = reference_apply_action(state, action) if reference else apply_action(state, action)
        rows.append({"ply": ply, "before": state.state_key, "legal_mask": result.legal_action_mask, "root_visits": result.root_visits, "pi": result.pi, "selected_action": action, "captures": transition.captured, "after": transition.after.state_key, "side_to_move_after": transition.after.side_to_move})
        state = transition.after
        if state.is_terminal:
            break
    _assert(state.is_terminal, f"Full-game trace did not terminate for {game_id}")
    result = result_from_terminal(state)
    targets = tuple(z_target(result.winner.value, row["before"][1]) for row in rows)
    return {"game_id": game_id, "game_seed": game_seed, "rows": tuple(rows), "game_length": len(rows), "winner": result.winner, "score": result, "z": targets}


def full_game_equivalence(model: torch.nn.Module) -> dict[str, object]:
    game_ids = ("parity-full-game-m0-00", "parity-full-game-m0-01")
    trained_ids = ("parity-full-game-m4-00", "parity-full-game-m4-01")
    rows = []
    for game_id in game_ids + trained_ids:
        old = _play_full_game(model, seed_namespace=OLD_SEED_NAMESPACE, game_id=game_id, reference=True)
        new = _play_full_game(model, seed_namespace=OLD_SEED_NAMESPACE, game_id=game_id, reference=False)
        _assert(old["rows"] == new["rows"], f"Full-game trace divergence at {game_id}")
        _assert(old["z"] == new["z"], f"Full-game z divergence at {game_id}")
        rows.append({"game_id": game_id, "game_length": old["game_length"], "winner": old["winner"], "score": old["score"], "z_identical": True, "trace": "BIT-IDENTICAL"})
    return {"status": "BIT-EXACT", "games": len(rows), "rows": rows, "mismatches": 0}


def _record_semantics(record: SelfPlayGameRecord) -> dict[str, object]:
    rows = []
    for position in record.positions:
        state = __import__("gocube_golden.training", fromlist=["state_from_identity"]).state_from_identity(position.state)
        transition = apply_action(state, position.selected_action)
        rows.append({"ply": position.ply, "state": position.state, "side_to_move": position.side_to_move, "root_visits": position.root_visits, "pi": position.pi, "selected_action": position.selected_action, "search_seed": position.search_seed, "captures": transition.captured, "after": state_identity(transition.after)})
    return {"game_id": record.game_id, "game_seed": record.game_seed, "positions": rows, "final_action_trace": record.final_action_trace, "formal_result": record.formal_result, "technical_termination": record.technical_termination, "model_hash": record.model_hash, "nn_evaluations": record.nn_evaluations}


def serial_process_equivalence(model: GoldenGraphNetV1, *, run_id: str, artifact: str, checkpoint: Path, semantic: Mapping[str, object], code, device: torch.device, workers: int) -> dict[str, object]:
    game_ids = ("serial-process-game-000", "serial-process-game-001", "serial-process-game-002", "serial-process-game-003")
    serial, serial_nn = run_games(model, run_id=run_id, label="M0", artifact=artifact, checkpoint_path=None, seed=STAGE4_SELFPLAY_MASTER_SEED, seed_namespace=OLD_SEED_NAMESPACE, semantic_profile=semantic, code=code, device=device, game_ids=game_ids, workers=1)
    process, process_nn = run_games(model, run_id=run_id, label="M0", artifact=artifact, checkpoint_path=str(checkpoint), seed=STAGE4_SELFPLAY_MASTER_SEED, seed_namespace=OLD_SEED_NAMESPACE, semantic_profile=semantic, code=code, device=device, game_ids=game_ids, workers=workers)
    _assert([_record_semantics(row) for row in serial] == [_record_semantics(row) for row in process], "Serial/process self-play trace mismatch")
    _assert(serial_nn == process_nn, "Serial/process NN evaluation count mismatch")
    return {"status": "IDENTICAL", "workers_serial": 1, "workers_process": workers, "game_ids": game_ids, "nn_evaluations_serial": serial_nn, "nn_evaluations_process": process_nn, "all_game_traces": "IDENTICAL", "replay_records": "IDENTICAL", "targets": "IDENTICAL", "mismatches": 0}


def _copy_frozen_evaluation(run_dir: Path) -> tuple[dict[str, object], ...]:
    starts = load_frozen_starts(OLD_RUN)
    destination = run_dir / "evaluation-v2"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OLD_RUN / "evaluation-v2" / "starts.jsonl", destination / "starts.jsonl")
    shutil.copy2(OLD_RUN / "evaluation-v2" / "diagnostic-subset.json", destination / "diagnostic-subset.json")
    old_manifest = json.loads((OLD_RUN / "evaluation-v2" / "manifest.json").read_text(encoding="utf-8"))
    _write(destination / "manifest.json", {**old_manifest, "copied_from_immutable_reference": OLD_RUN_ID, "new_run_id": run_dir.name})
    _assert(len(starts) == EVALUATION_ACCEPTED_TOTAL, "Frozen evaluation corpus is not 64 starts")
    _assert(evaluation_start_fingerprint(starts) == old_manifest["full_set_fingerprint"], "Frozen evaluation corpus fingerprint changed")
    return starts


def _checkpoint_info(path: Path, label: str, device: torch.device) -> dict[str, object]:
    return checkpoint_info(path, label=label, device=device)


def _load_replay_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _replay_semantics(path: Path) -> list[dict[str, object]]:
    ignored = {"run_id"}
    return [{key: value for key, value in row.items() if key not in ignored} for row in _load_replay_rows(path)]


def _compare_replay(old_path: Path, new_path: Path) -> dict[str, object]:
    old = _replay_semantics(old_path)
    new = _replay_semantics(new_path)
    _assert(old == new, f"Replay divergence: {old_path.name}")
    return {"old_rows": len(old), "new_rows": len(new), "semantic": "BIT-EXACT"}


def _train_one_chunk(model: GoldenGraphNetV1, optimizer, cumulative: list[GoldenTrainingSample], new_samples: Sequence[GoldenTrainingSample], *, chunk: int, update_offset: int, sample_offset: int, stage4: Mapping[str, object]) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    cumulative.extend(new_samples)
    cumulative.sort(key=lambda sample: (sample.game_id, sample.ply))
    sample_count = len(new_samples)
    sampling_seed = derive_seed(STAGE4_MODEL_INIT_SEED, "main-replay-sampling", chunk)
    selected = random.Random(sampling_seed).sample(range(len(cumulative)), sample_count)
    batches = tuple(tuple(selected[offset:offset + 64]) for offset in range(0, len(selected), 64))
    optimizer, schedule = train_batch_schedule(model, tuple(cumulative), batches, learning_rate=stage4["training"]["learning_rate"], weight_decay=stage4["training"]["weight_decay"], optimizer=optimizer, update_offset=update_offset, sample_offset=sample_offset)
    return optimizer, {"sampling_seed": sampling_seed, "new_positions": sample_count, "batches": batches, "schedule": schedule}


def run_stage4(*, run_dir: Path, run_id: str, semantic: Mapping[str, object], stage4: Mapping[str, object], code, device: torch.device, workers: int, m0_model: GoldenGraphNetV1, m0_info: Mapping[str, object]) -> dict[str, object]:
    checkpoints: dict[str, dict[str, object]] = {"M0": dict(m0_info)}
    chunk_records: list[SelfPlayGameRecord] = []
    all_samples: list[GoldenTrainingSample] = []
    training_rows = []
    old_models = {label: _checkpoint_info(OLD_RUN / "checkpoints" / f"{label}.pt", label, device) for label in ("M0", "M1", "M2", "M3", "M4")}
    _assert(m0_info["model_hash"] == old_models["M0"]["model_hash"], "M0 model hash mismatch; stop before self-play")
    optimizer = torch.optim.Adam(m0_model.parameters(), lr=stage4["training"]["learning_rate"], weight_decay=stage4["training"]["weight_decay"])
    update_offset = 0
    sample_offset = 0
    replay_comparisons = []
    for chunk in range(1, 5):
        label = f"M{chunk - 1}"
        seed = STAGE4_SELFPLAY_MASTER_SEED if chunk == 1 else derive_seed(STAGE4_SELFPLAY_MASTER_SEED, "chunk", chunk)
        ids = tuple(f"chunk-{chunk:02d}-game-{index:03d}" for index in range(128))
        records, nn_count = run_games(m0_model, run_id=run_id, label=label, artifact=str(checkpoints[label]["artifact_sha256"]), checkpoint_path=str(checkpoints[label]["path"]), seed=seed, seed_namespace=OLD_SEED_NAMESPACE, semantic_profile=semantic, code=code, device=device, game_ids=ids, workers=workers)
        _assert(len(records) == 128, f"Chunk {chunk} record count mismatch")
        _assert(all(record.technical_termination is None for record in records), f"Chunk {chunk} has a technical game")
        _assert(all(record.model_hash == checkpoints[label]["model_hash"] for record in records), f"Chunk {chunk} model identity mismatch")
        samples = tuple(sample for record in records for sample in build_replay_samples(record))
        chunk_records.extend(records)
        all_samples.extend(samples)
        write_jsonl(run_dir / "selfplay" / f"chunk-{chunk:02d}-games.jsonl", (record.to_dict() for record in records))
        write_jsonl(run_dir / "replay" / f"chunk-{chunk:02d}.jsonl", (sample.to_dict() for sample in samples))
        replay_comparisons.append({"chunk": chunk, **_compare_replay(OLD_RUN / "replay" / f"chunk-{chunk:02d}.jsonl", run_dir / "replay" / f"chunk-{chunk:02d}.jsonl")})
        before = {name: value.detach().clone() for name, value in m0_model.state_dict().items()}
        optimizer, training = _train_one_chunk(m0_model, optimizer, all_samples, samples, chunk=chunk, update_offset=update_offset, sample_offset=sample_offset, stage4=stage4)
        schedule = training["schedule"]
        update_offset = int(schedule["updates"])
        sample_offset = int(schedule["exact_samples_consumed"])
        checkpoint_label = f"M{chunk}"
        metadata = checkpoint_metadata(semantic, run_id=run_id, label=checkpoint_label, parent=str(checkpoints[label]["model_hash"]), code=code, model=m0_model, device=device, completed_games=chunk * 128, valid_replay_positions=len(all_samples), optimizer_updates=update_offset, train_samples_consumed=sample_offset, model_init_seed=STAGE4_MODEL_INIT_SEED, stage4_profile=stage4)
        path = run_dir / "checkpoints" / f"M{chunk}.pt"
        metadata = save_checkpoint(path, model=m0_model, optimizer=optimizer, metadata=metadata)
        info = {"label": checkpoint_label, "path": str(path), "metadata": metadata, "artifact_sha256": metadata["artifact_sha256"], "model_hash": metadata["model_hash"]}
        checkpoints[checkpoint_label] = info
        old_hash = old_models[checkpoint_label]["model_hash"]
        _assert(info["model_hash"] == old_hash, f"{checkpoint_label} checkpoint identity mismatch after M0 freeze")
        training_row = {
            "chunk": chunk,
            "games": 128,
            "valid_games": 128,
            "technical": 0,
            "new_positions": len(samples),
            "cumulative_replay_positions": len(all_samples),
            "nn_evaluations": nn_count,
            "sampling_seed": training["sampling_seed"],
            "optimizer_updates_this_phase": len(training["batches"]),
            "optimizer_updates_cumulative": update_offset,
            "exact_samples_consumed_cumulative": sample_offset,
            "actual_batch_sizes": schedule["batch_sizes"],
            "training_schedule": schedule,
            "model_hash": info["model_hash"],
            "old_model_hash": old_hash,
            "parameter_delta": {"l2": float(sum(torch.sum((m0_model.state_dict()[name] - before[name]) ** 2) for name in before).sqrt())},
        }
        _write(run_dir / "training" / f"chunk-{chunk:02d}-metrics.json", training_row)
        training_rows.append(training_row)
        old_chunk = tuple(records_from_jsonl(OLD_RUN / "selfplay" / f"chunk-{chunk:02d}-games.jsonl"))
        _assert([_record_semantics(row) for row in old_chunk] == [_record_semantics(row) for row in records], f"Old/new self-play divergence in chunk {chunk}")

    audit = audit_selfplay_records(chunk_records, expected_model_hash_by_label={label: info["model_hash"] for label, info in checkpoints.items()})
    _assert(audit["valid_games"] == 512 and audit["technical_games"] == 0, "Stage4 self-play audit failed")
    _assert(audit["positions"] == 15455, f"Stage4 position count drift: {audit['positions']} != 15455")
    return {"checkpoints": checkpoints, "training": training_rows, "replay": replay_comparisons, "selfplay_audit": audit, "total_positions": len(all_samples), "total_games": len(chunk_records), "technical_games": audit["technical_games"], "old_model_hashes": {label: info["model_hash"] for label, info in old_models.items()}}


def arena_matrix(*, run_dir: Path, run_id: str, stage4_result: Mapping[str, object], starts: Sequence[Mapping[str, object]], code, device: torch.device) -> dict[str, object]:
    checkpoints = stage4_result["checkpoints"]
    subset = tuple(starts[index] for index in range(0, len(starts), 4))
    _assert(len(subset) == 16, "Arena diagnostic subset must contain 16 starts")
    result: dict[str, object] = {}
    specs = (
        ("M1-M0", "M1", "M0", starts, 64),
        ("M4-M0", "M4", "M0", starts, 64),
        ("M4-M1", "M4", "M1", starts, 64),
        ("M2-M1", "M2", "M1", subset, 16),
        ("M3-M2", "M3", "M2", subset, 16),
        ("M4-M3", "M4", "M3", subset, 16),
    )
    for slug, candidate, reference, corpus, pair_count in specs:
        summary = run_comparison(run_dir=run_dir, run_id=run_id, slug=slug.lower(), candidate=checkpoints[candidate], reference=checkpoints[reference], candidate_label=candidate, reference_label=reference, starts=corpus, arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, slug), code=code, device=device, canonical=True, output_dir=run_dir / "arena" / slug.lower())
        expected = EXPECTED_ARENA[slug]
        actual = tuple(summary.get("W/L/D", ()))
        _assert(summary.get("technical") == 0, f"Arena {slug} has technical games")
        _assert(actual == expected, f"Arena {slug} changed from immutable Stage4 reference: {actual} != {expected}")
        _assert(summary.get("pairs") == pair_count and summary.get("games") == pair_count * 2, f"Arena {slug} pair count mismatch")
        result[slug] = summary
    return result


def scalability_diagnostic() -> dict[str, object]:
    rows = []
    for size in (5, 9, 13):
        topology = _torus_topology(size)
        for history_length in (1, 100, 400):
            state = _synthetic_history_state(history_length, topology=topology)
            with operation_stats() as stats:
                started = time.perf_counter()
                context = prepare_legal_actions(state)
                elapsed = time.perf_counter() - started
            counters = stats.to_dict()
            _assert(counters["full_history_validations"] == 0, "Synthetic optimized legality performed full-history validation")
            rows.append({"size": size, "history_length": history_length, "legal_actions": len(context.actions), "wall_sec": elapsed, "full_history_validations": counters["full_history_validations"], "action_probe_calls": counters["action_probe_calls"]})
    return {"status": "DIAGNOSTIC", "rows": rows, "purpose": "rules-only complexity counters; not a 9x9/13x13/19x19 network claim"}


def _old_new_checkpoint_comparison(stage4_result: Mapping[str, object]) -> dict[str, object]:
    rows = []
    for label, info in stage4_result["checkpoints"].items():
        old = json.loads((OLD_RUN / "checkpoints" / f"{label}.metadata.json").read_text(encoding="utf-8"))
        rows.append({"label": label, "old_model_hash": old["model_hash"], "new_model_hash": info["model_hash"], "model_hash_identical": old["model_hash"] == info["model_hash"], "metadata_difference_allowed": ["run_id", "git_commit", "git_tree", "artifact_sha256"]})
    _assert(all(row["model_hash_identical"] for row in rows), "Old/new checkpoint model identity mismatch")
    return {"status": "BIT-EXACT", "rows": rows}


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = ROOT / "runs" / "torus-golden-stage4" / args.run_id
    _assert(not run_dir.exists(), f"Refusing to overwrite existing run: {run_dir}")
    _assert(args.workers == 16, "Canonical parity requires the approved Stage4 worker count of 16")
    _assert(OLD_RUN.is_dir(), f"Immutable old Stage4 run is missing: {OLD_RUN}")
    code = capture_code_identity(ROOT)
    _assert(code.working_tree_clean, "Canonical parity requires a clean committed source tree")
    semantic = load_stage3_profile()
    stage4 = stage4_profile()
    _assert(stage4["profile_id"] == STAGE4_PROFILE_ID, "Stage4 profile identity drift")
    contract = contract_snapshot(semantic, stage4)
    run_dir.mkdir(parents=True)
    _write(run_dir / "contract.json", contract)
    _write(run_dir / "source.json", {"branch": "codex/torus-golden-performance-parity", "base_branch": "codex/torus-rebuild-v1", "base_commit": "d8ad1599424d5f3a706c3a1c406001ce2141e0cf", "feature_commit": code.git_commit_sha, "source_tree": code.git_tree_sha, "git_worktree_clean": code.working_tree_clean, "old_run": OLD_RUN_ID, "old_run_source_commit": json.loads((OLD_RUN / "final-report.json").read_text(encoding="utf-8")).get("source_commit")})
    audit = optimized_path_audit()
    _write(run_dir / "optimized-path-audit.json", audit)
    corpus = _corpus()
    with operation_stats() as stats:
        rules = rules_equivalence(corpus)
    rules["corpus_operation_counts"] = stats.to_dict()
    _write(run_dir / "correctness-rules.json", rules)
    fuzz = fuzz_equivalence()
    _write(run_dir / "correctness-fuzz.json", fuzz)
    observations = observation_equivalence(corpus)
    _write(run_dir / "correctness-observation.json", observations)
    torch.set_num_threads(1)
    torch.manual_seed(STAGE4_MODEL_INIT_SEED)
    model = GoldenGraphNetV1().to("cpu")
    evaluator = evaluator_equivalence(model, corpus)
    _write(run_dir / "correctness-evaluator.json", evaluator)
    root_noise = root_noise_equivalence(model, corpus[:24])
    _write(run_dir / "correctness-root-noise.json", root_noise)
    search = search_equivalence(model, corpus[:24])
    _write(run_dir / "correctness-search.json", search)
    _copy_frozen_evaluation(run_dir)
    device = torch.device("cpu")
    m0_meta = checkpoint_metadata(semantic, run_id=args.run_id, label="M0", parent=None, code=code, model=model, device=device, completed_games=0, valid_replay_positions=0, optimizer_updates=0, train_samples_consumed=0, model_init_seed=STAGE4_MODEL_INIT_SEED, stage4_profile=stage4)
    m0_path = run_dir / "checkpoints" / "M0.pt"
    m0_meta = save_checkpoint(m0_path, model=model, optimizer=None, metadata=m0_meta)
    m0_info = {"label": "M0", "path": str(m0_path), "metadata": m0_meta, "artifact_sha256": m0_meta["artifact_sha256"], "model_hash": m0_meta["model_hash"]}
    serial_process = serial_process_equivalence(model, run_id=args.run_id, artifact=str(m0_meta["artifact_sha256"]), checkpoint=m0_path, semantic=semantic, code=code, device=device, workers=args.workers)
    _write(run_dir / "serial-process-equivalence.json", serial_process)
    full_game = full_game_equivalence(model)
    _write(run_dir / "correctness-full-game.json", full_game)
    with operation_stats() as stats:
        performance = {"reference": _run_mode("reference", checkpoint=m0_path), "optimized": _run_mode("optimized", checkpoint=m0_path)}
    for label in performance["reference"]["states"]:
        old = performance["reference"]["states"][label]
        new = performance["optimized"]["states"][label]
        new["speedup_x"] = float(old["search_wall_sec"]) / float(new["search_wall_sec"])
        _assert(new["operation_counts"]["full_history_validations"] == 0, f"Optimized performance path validated history at {label}")
    performance["measurement_operation_counts"] = stats.to_dict()
    _write(run_dir / "performance.json", performance)
    starts = load_frozen_starts(run_dir)
    stage4_result = run_stage4(run_dir=run_dir, run_id=args.run_id, semantic=semantic, stage4=stage4, code=code, device=device, workers=args.workers, m0_model=model, m0_info=m0_info)
    _write(run_dir / "stage4-result.json", {key: value for key, value in stage4_result.items() if key != "checkpoints"})
    old_new_checkpoints = _old_new_checkpoint_comparison(stage4_result)
    arena = arena_matrix(run_dir=run_dir, run_id=args.run_id, stage4_result=stage4_result, starts=starts, code=code, device=device)
    _write(run_dir / "arena-summary.json", arena)
    scale = scalability_diagnostic()
    _write(run_dir / "scalability-diagnostic.json", scale)
    old_report = json.loads((OLD_RUN / "final-report.json").read_text(encoding="utf-8"))
    training = {"status": "BIT-EXACT", "games": stage4_result["total_games"], "positions": stage4_result["total_positions"], "technical_games": stage4_result["technical_games"], "checkpoint_hashes": old_new_checkpoints, "replay": stage4_result["replay"], "training_chunks": stage4_result["training"], "old_reference_positions": old_report["selfplay_audit_stage4"]["positions"]}
    final = {
        "run_id": args.run_id,
        "old_reference_run": OLD_RUN_ID,
        "branch": "codex/torus-golden-performance-parity",
        "base_branch": "codex/torus-rebuild-v1",
        "source": {"commit": code.git_commit_sha, "tree": code.git_tree_sha, "clean": code.working_tree_clean},
        "contract": contract,
        "optimized_path_audit": audit,
        "correctness": {"rules_semantics": rules, "fuzz_rules": fuzz, "observation_semantics": observations, "evaluator_semantics": evaluator, "root_noise_semantics": root_noise, "search_semantics": search, "full_game_trace": full_game, "serial_process_equivalence": serial_process},
        "performance": performance,
        "training": training,
        "arena": arena,
        "scalability_diagnostic": scale,
        "old_vs_new": {"checkpoint_identity": old_new_checkpoints, "old_reference_arena": old_report.get("arena_primary"), "new_arena_expected": EXPECTED_ARENA},
        "rules_semantics": "IDENTICAL",
        "search_semantics": "IDENTICAL",
        "full_game_trace": "BIT-EXACT",
        "serial_process_equivalence": "IDENTICAL",
        "training_reproduction": "BIT-EXACT",
        "self_play": {"games": 512, "valid_games": 512, "positions": stage4_result["total_positions"], "technical_games": stage4_result["technical_games"]},
        "technical_games": 0,
        "learning_system": "CONFIRMED",
        "verdict": "LEARNING SYSTEM CONFIRMED",
        "ci": "not run until final push",
        "artifact_root": str(run_dir),
    }
    _write(run_dir / "final-report.json", final)
    _write(run_dir / "manifest.json", {**final, "canonical_requirements": {"selfplay_games": 512, "technical_training_games": 0, "primary_pairs_each": 64, "progression_pairs_each": 16, "empty_board_control": True}})
    (run_dir / "final-report.md").write_text("# Golden Torus optimized-path parity\n\n**LEARNING SYSTEM CONFIRMED**\n\nRules, observation, evaluator, search, full-game and serial/process gates are exact; Stage4 reproduced 512 valid games and 15,455 positions with zero technical games.\n", encoding="utf-8")
    return final


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="torus-golden-stage4-seed2-parity-v1")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"RUN INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
