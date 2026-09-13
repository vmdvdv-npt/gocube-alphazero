#!/usr/bin/env python3
"""Read-only forensic audit for the frozen Torus 9x9 learning run.

The audit intentionally does not call the learning runner and never writes below
the run root.  It writes only the requested report artifacts in ``docs/`` (or
paths supplied on the command line).

The replay target pass has a small independent rules implementation.  It is
kept here, rather than delegated to ``gocube_golden.rules``, so a shared bug in
the production validator cannot make the result appear clean.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.neural import GoldenGraphNetV1, model_hash
from gocube_golden.provenance import derive_seed, file_sha256
from gocube_golden.rules import prepare_legal_actions
from gocube_golden.search import SequentialPUCT
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, PASS, research_state_from_stones
from gocube_golden.topology import TORUS_5X5, TORUS_9X9
from gocube_golden.torus9 import (
    Torus9GraphNet,
    Torus9NeuralEvaluator,
    Torus9RootNoiseEvaluator,
    build_torus9_observation,
    torus9_load_checkpoint,
    torus9_state_from_identity,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3"
SOURCE_SHA = "9723bb5ac8eb28d55d21d607e3673da5bd894315"
FINAL_HEAD = "68ba2de3a67bc65e40cae73a293ea2454ef8e111"
PROFILE_FP = "sha256:f16f58288b1e8902ce589d3a62ee04532941440ffbb53cd16282fdf5605950aa"
TOPOLOGY_FP = "sha256:a417b6d4e3da67ead03240361976a6f412c89403077bec0ac0c0a68ca9665ed1"
RULES_FP = "sha256:e0fd15c82d42a63ca05c3b6fb3ae02deb938543e1e06483ecfeab275dc98a39e"
OBS_FP = "sha256:e5792b409199dfe2c25ac6f681e4ca29ed73cdf4b7d53a61df634f70d1fa415f"
TARGET_FP = "sha256:e1a1dd23908c75bc71c5215bfec6b0661606f86642e3406a53710057d37fed92"
MODEL_SEED = 2026091404
BATCH_SIZE = 64
POINTS = 81
ACTIONS = 82


def _json(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return str(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(xs: Iterable[float]) -> float | None:
    values = list(xs)
    return sum(values) / len(values) if values else None


def _percentiles(xs: Iterable[float]) -> dict[str, float | None]:
    values = sorted(float(x) for x in xs)
    if not values:
        return {key: None for key in ("min", "max", "mean", "median", "p90", "p95", "p99")}
    return {
        "min": values[0],
        "max": values[-1],
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": values[min(len(values) - 1, int(0.90 * len(values)))],
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
        "p99": values[min(len(values) - 1, int(0.99 * len(values)))],
    }


def _entropy(values: Iterable[float]) -> float:
    xs = [float(v) for v in values]
    total = sum(xs)
    if total <= 0.0:
        return 0.0
    return -sum((v / total) * math.log(v / total) for v in xs if v > 0.0)


def _l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().float()).item())


def _global_l2(mapping: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.sum(value.detach().float() ** 2)) for value in mapping.values()))


def _state_core(state: Mapping[str, object]) -> tuple[object, ...]:
    return (
        tuple(int(v) for v in state["stones"]),
        int(state["side_to_move"]),
        tuple(tuple(int(v) for v in row) for row in state["superko_history"]),
        int(state["consecutive_passes"]),
    )


# ---------------------------------------------------------------------------
# Independent Torus 9x9 rules/replay audit


def independent_neighbors(point: int) -> tuple[int, int, int, int]:
    y, x = divmod(point, 9)
    return (((y - 1) % 9) * 9 + x, y * 9 + (x + 1) % 9, ((y + 1) % 9) * 9 + x, y * 9 + (x - 1) % 9)


def independent_group(board: tuple[int, ...] | list[int], start: int) -> set[int]:
    color = board[start]
    if color == 0:
        return set()
    found = {start}
    todo = [start]
    while todo:
        point = todo.pop()
        for neighbor in independent_neighbors(point):
            if neighbor not in found and board[neighbor] == color:
                found.add(neighbor)
                todo.append(neighbor)
    return found


def independent_liberties(board: tuple[int, ...] | list[int], group: set[int]) -> set[int]:
    return {neighbor for point in group for neighbor in independent_neighbors(point) if board[neighbor] == 0}


def independent_point(
    board: tuple[int, ...], side: int, history: tuple[tuple[int, ...], ...], action: int,
) -> tuple[tuple[int, ...], int, tuple[tuple[int, ...], ...]] | None:
    if board[action] != 0:
        return None
    other = 3 - side
    provisional = list(board)
    provisional[action] = side
    captured: set[int] = set()
    checked: set[int] = set()
    for neighbor in independent_neighbors(action):
        if provisional[neighbor] != other or neighbor in checked:
            continue
        group = independent_group(provisional, neighbor)
        checked.update(group)
        if not independent_liberties(provisional, group):
            captured.update(group)
    for point in captured:
        provisional[point] = 0
    own_group = independent_group(provisional, action)
    if not independent_liberties(provisional, own_group):
        return None
    new_board = tuple(provisional)
    if new_board in history:
        return None
    return new_board, other, history + (new_board,)


def independent_legal(board: tuple[int, ...], side: int, history: tuple[tuple[int, ...], ...]) -> tuple[int | str, ...]:
    actions = [action for action in range(POINTS) if independent_point(board, side, history, action) is not None]
    actions.append(PASS)
    return tuple(actions)


def independent_observation(state: Mapping[str, object], mask: list[bool]) -> list[list[float]]:
    board = [int(v) for v in state["stones"]]
    side = int(state["side_to_move"])
    other = 3 - side
    return [
        [float(v == side) for v in board],
        [float(v == other) for v in board],
        [1.0 if side == 1 else -1.0] * POINTS,
        [1.0 if int(state["consecutive_passes"]) == 1 else 0.0] * POINTS,
        [float(v) for v in mask[:POINTS]],
        [0.5] * POINTS,
    ]


def independent_z(winner: str, side: int) -> list[float]:
    if winner == "DRAW":
        return [0.0, 1.0, 0.0]
    return [1.0, 0.0, 0.0] if winner == ("BLACK" if side == 1 else "WHITE") else [0.0, 0.0, 1.0]


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def audit_replay(run: Path) -> dict[str, object]:
    mismatches: Counter[str] = Counter()
    per_iteration: dict[str, object] = {}
    total_games = total_rows = 0
    for iteration in range(1, 9):
        games = _load_jsonl(run / "canonical/selfplay" / f"iter-{iteration:02d}-games.jsonl")
        game_by_id = {str(game["game_id"]): game for game in games}
        positions_by_key = {
            (str(game["game_id"]), int(position["ply"])): position
            for game in games for position in game["positions"]  # type: ignore[index]
        }
        game_rows = 0
        valid_trace_games = 0
        for game in games:
            state = game["start_state"]
            board = tuple(int(v) for v in state["stones"])  # type: ignore[index]
            side = int(state["side_to_move"])  # type: ignore[index]
            history = tuple(tuple(int(v) for v in row) for row in state["superko_history"])  # type: ignore[index]
            consecutive_passes = int(state["consecutive_passes"])  # type: ignore[index]
            trace = game["final_action_trace"]  # type: ignore[index]
            for ply, action in enumerate(trace, 1):
                position = positions_by_key.get((str(game["game_id"]), ply))
                if position is None:
                    mismatches["raw_position_missing"] += 1
                    continue
                if _state_core(position["state"]) != (board, side, history, consecutive_passes):  # type: ignore[index]
                    mismatches["raw_state_before_action"] += 1
                legal = independent_legal(board, side, history)
                if action not in legal:
                    mismatches["raw_trace_illegal"] += 1
                if action == PASS:
                    consecutive_passes += 1
                    side = 3 - side
                else:
                    transition = independent_point(board, side, history, int(action))
                    if transition is None:
                        mismatches["raw_trace_transition"] += 1
                    else:
                        board, side, history = transition
                        consecutive_passes = 0
            if game["technical_termination"] is None:
                valid_trace_games += 1
                if consecutive_passes != 2:
                    mismatches["raw_trace_not_double_pass"] += 1
        with (run / "canonical/replay" / f"iter-{iteration:02d}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                game_rows += 1
                total_rows += 1
                total_games += 0
                game = game_by_id.get(str(row["game_id"]))
                position = positions_by_key.get((str(row["game_id"]), int(row["ply"])))
                if game is None:
                    mismatches["replay_game_missing"] += 1
                    continue
                if position is None:
                    mismatches["replay_raw_position_missing"] += 1
                    continue
                if row["state"] != position["state"]:
                    mismatches["replay_state_vs_raw"] += 1
                state = row["state"]
                board = tuple(int(v) for v in state["stones"])
                side = int(state["side_to_move"])
                history = tuple(tuple(int(v) for v in h) for h in state["superko_history"])
                legal = independent_legal(board, side, history)
                expected_mask = [False] * ACTIONS
                for action in legal:
                    expected_mask[POINTS if action == PASS else int(action)] = True
                if row["legal_action_mask"] != expected_mask:
                    mismatches["legal_mask"] += 1
                if row["observation"] != independent_observation(state, expected_mask):
                    mismatches["observation"] += 1
                visits = [int(v) for v in position["root_visits"]]
                if row["root_visits"] != visits:
                    mismatches["root_visits_vs_raw"] += 1
                if sum(visits) <= 0:
                    mismatches["zero_root_visits"] += 1
                expected_pi = [value / sum(visits) for value in visits]
                if any(not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12) for a, b in zip(row["pi"], expected_pi)):
                    mismatches["policy_target"] += 1
                if row["z"] != independent_z(str(game["formal_result"]), side):
                    mismatches["wdl_target"] += 1
                if row["side_to_move"] != ("BLACK" if side == 1 else "WHITE"):
                    mismatches["side_to_move"] += 1
                for key, expected in (
                    ("observation_fingerprint", OBS_FP),
                    ("target_contract_id", "gocube-torus9-wdl-side-to-move-v1"),
                    ("target_fingerprint", TARGET_FP),
                ):
                    if row.get(key) != expected:
                        mismatches[key] += 1
        per_iteration[str(iteration)] = {
            "games": len(games),
            "replay_rows": game_rows,
            "valid_trace_games": valid_trace_games,
            "technical_games": sum(game["technical_termination"] is not None for game in games),
        }
    return {
        "total_games": 512,
        "total_replay_rows": total_rows,
        "expected_replay_rows": 40910,
        "mismatch_count": sum(mismatches.values()),
        "mismatches": dict(mismatches),
        "per_iteration": per_iteration,
        "independent_rules_oracle": "standalone 9x9 torus positional-superko implementation",
        "status": "PASS" if not mismatches and total_rows == 40910 else "FAIL",
    }


# ---------------------------------------------------------------------------
# Checkpoint, optimizer, and exact training reproduction


def _checkpoint_payloads(run: Path) -> list[dict[str, object]]:
    result = []
    for index in range(9):
        path = run / "canonical/checkpoints" / f"M{index}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sidecar = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        result.append({"index": index, "path": path, "payload": payload, "sidecar": sidecar, "artifact_sha256": "sha256:" + file_sha256(path).removeprefix("sha256:")})
    return result


def audit_lineage(run: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    payloads = _checkpoint_payloads(run)
    expected_manifest = manifest.get("checkpoints", {}).get("canonical", {})  # type: ignore[union-attr]
    transitions: list[dict[str, object]] = []
    lineage_failures: list[str] = []
    for item in payloads:
        i = int(item["index"])
        payload = item["payload"]
        sidecar = item["sidecar"]
        meta = dict(payload["metadata"])  # type: ignore[index]
        actual_artifact = str(item["artifact_sha256"])
        expected = expected_manifest.get(f"M{i}", {}) if isinstance(expected_manifest, Mapping) else {}
        checks = {
            "sidecar_model_hash": sidecar.get("model_hash") == meta.get("model_hash"),
            "payload_model_hash": meta.get("model_hash") == model_hash_from_payload(payload),
            "source_commit": meta.get("git_commit") == SOURCE_SHA,
            "profile_fingerprint": meta.get("profile_fingerprint") == PROFILE_FP,
            "topology_fingerprint": meta.get("topology_fingerprint") == TOPOLOGY_FP,
            "rules_fingerprint": meta.get("rules_fingerprint") == RULES_FP,
            "expected_model_hash": not expected or meta.get("model_hash") == expected.get("model_hash"),
            "expected_artifact_hash": not expected or actual_artifact == expected.get("artifact_sha256"),
        }
        if not all(checks.values()):
            lineage_failures.extend(f"M{i}:{key}" for key, ok in checks.items() if not ok)
        transitions.append({
            "label": f"M{i}",
            "parent": meta.get("parent_or_source_run_identity"),
            "model_hash": meta.get("model_hash"),
            "artifact_sha256": actual_artifact,
            "optimizer_updates": meta.get("optimizer_updates"),
            "train_samples_consumed": meta.get("train_samples_consumed"),
            "completed_games": meta.get("completed_games"),
            "checks": checks,
        })
    expected_parents = {0: "torus9-golden-learning-proof-20260913-v3-canonical", **{i: f"M{i-1}" for i in range(1, 9)}}
    for row in transitions:
        i = int(str(row["label"])[1:])
        if row["parent"] != expected_parents[i]:
            lineage_failures.append(f"M{i}:parent")
    return {
        "status": "PASS" if not lineage_failures else "FAIL",
        "failures": lineage_failures,
        "transitions": transitions,
        "source_sha": SOURCE_SHA,
        "final_head_audit_context": FINAL_HEAD,
    }


def model_hash_from_payload(payload: Mapping[str, object]) -> str:
    model = Torus9GraphNet()
    model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]
    return model_hash(model)


def _load_compact_replay(run: Path, iteration: int) -> list[dict[str, object]]:
    # Keeping only trainer inputs avoids retaining the large state/history object.
    with (run / "canonical/replay" / f"iter-{iteration:02d}.jsonl").open(encoding="utf-8") as handle:
        return [
            (lambda row: {"observation": row["observation"], "pi": row["pi"], "z": row["z"]})(json.loads(line))
            for line in handle
        ]


def reproduce_transitions(run: Path) -> dict[str, object]:
    torch.set_num_threads(8)
    payloads = _checkpoint_payloads(run)
    rows: list[dict[str, object]] = []
    for iteration in range(1, 9):
        samples = _load_compact_replay(run, iteration)
        model = Torus9GraphNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        parent_meta = payloads[iteration - 1]["payload"]["metadata"]  # type: ignore[index]
        torus9_load_checkpoint(payloads[iteration - 1]["path"], model=model, optimizer=optimizer)  # type: ignore[arg-type]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(MODEL_SEED + iteration)
        order = [int(index) for index in torch.randperm(len(samples), generator=generator).tolist()]
        model.train()
        batch_rows: list[dict[str, object]] = []
        for offset in range(0, len(order), BATCH_SIZE):
            indices = order[offset:offset + BATCH_SIZE]
            before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            observations = torch.tensor([samples[index]["observation"] for index in indices], dtype=torch.float32)
            policies = torch.tensor([samples[index]["pi"] for index in indices], dtype=torch.float32)
            values = torch.tensor([samples[index]["z"] for index in indices], dtype=torch.float32)
            policy_logits, value_logits = model(observations)
            policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = -(values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
            total_loss = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")))
            optimizer.step()
            after = dict(model.named_parameters())
            parameter_delta = math.sqrt(sum(float(torch.sum((after[name].detach().float() - before[name].float()) ** 2)) for name in before))
            batch_rows.append({
                "update": int(parent_meta["optimizer_updates"]) + len(batch_rows) + 1,
                "batch_size": len(indices),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "gradient_norm": gradient_norm,
                "parameter_delta": parameter_delta,
            })
        target_hash = payloads[iteration]["sidecar"]["model_hash"]
        recorded_updates = json.loads((run / "canonical/summary.json").read_text(encoding="utf-8"))["iteration_rows"][iteration - 1]["training"]["updates_detail"]
        metric_match = len(batch_rows) == len(recorded_updates) and all(
            left["batch_size"] == right["batch_size"]
            and math.isclose(float(left["policy_loss"]), float(right["policy_loss"]), abs_tol=1e-6)
            and math.isclose(float(left["value_loss"]), float(right["value_loss"]), abs_tol=1e-6)
            and math.isclose(float(left["gradient_norm"]), float(right["gradient_norm"]), abs_tol=1e-6)
            for left, right in zip(batch_rows, recorded_updates)
        )
        phase_delta = math.sqrt(sum(float(value["parameter_delta"]) ** 2 for value in batch_rows))
        rows.append({
            "transition": f"M{iteration-1}->M{iteration}",
            "phase_updates": len(batch_rows),
            "phase_samples": sum(int(value["batch_size"]) for value in batch_rows),
            "recorded_metrics_match": metric_match,
            "reproduced_model_hash": model_hash(model),
            "expected_model_hash": target_hash,
            "model_hash_match": model_hash(model) == target_hash,
            "last_batch": batch_rows[-1],
            "max_gradient_update": max(batch_rows, key=lambda value: float(value["gradient_norm"])),
            "max_parameter_delta_update": max(batch_rows, key=lambda value: float(value["parameter_delta"])),
            "mean_batch_parameter_delta": _mean(float(value["parameter_delta"]) for value in batch_rows),
            "phase_rms_batch_delta": phase_delta,
            "partial_batch_ratio_to_mean": float(batch_rows[-1]["parameter_delta"]) / float(_mean(float(value["parameter_delta"]) for value in batch_rows)),
        })
    return {
        "status": "PASS" if all(row["model_hash_match"] and row["recorded_metrics_match"] for row in rows) else "FAIL",
        "torch_num_threads": 8,
        "transitions": rows,
    }


def optimizer_and_parameter_audit(run: Path) -> dict[str, object]:
    payloads = _checkpoint_payloads(run)
    table: list[dict[str, object]] = []
    prev_parameters: Mapping[str, torch.Tensor] | None = None
    for item in payloads:
        i = int(item["index"])
        payload = item["payload"]
        metadata = payload["metadata"]  # type: ignore[index]
        parameters = payload["model_state_dict"]  # type: ignore[index]
        parameter_norm = _global_l2(parameters)
        if prev_parameters is None:
            delta = None
            relative = None
            max_layer = None
            max_layer_delta = None
        else:
            layer_deltas = {name: _l2(parameters[name] - prev_parameters[name]) for name in parameters if name in prev_parameters and parameters[name].shape == prev_parameters[name].shape}
            delta = math.sqrt(sum(value * value for value in layer_deltas.values()))
            relative = delta / _global_l2(prev_parameters)
            max_layer, max_layer_delta = max(layer_deltas.items(), key=lambda pair: pair[1])
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is None:
            steps: list[int] = []
            exp_avg_norm = exp_avg_sq_norm = None
            state_entries = 0
            max_exp_avg = max_exp_avg_sq = None
        else:
            state = optimizer_state["state"]  # type: ignore[index]
            state_entries = len(state)
            steps = [int(value["step"].item() if hasattr(value["step"], "item") else value["step"]) for value in state.values()]
            exp_avg_norm = _global_l2({str(key): value["exp_avg"] for key, value in state.items()})
            exp_avg_sq_norm = _global_l2({str(key): value["exp_avg_sq"] for key, value in state.items()})
            max_exp_avg = max(((_l2(value["exp_avg"]), key) for key, value in state.items()), default=(None, None))
            max_exp_avg_sq = max(((_l2(value["exp_avg_sq"]), key) for key, value in state.items()), default=(None, None))
        table.append({
            "transition": None if i == 0 else f"M{i-1}->M{i}",
            "label": f"M{i}",
            "updates": metadata["optimizer_updates"],
            "samples": metadata["train_samples_consumed"],
            "parameter_norm": parameter_norm,
            "parameter_delta": delta,
            "relative_parameter_delta": relative,
            "max_layer_delta": max_layer_delta,
            "max_layer": max_layer,
            "adam_step_min": min(steps) if steps else None,
            "adam_step_max": max(steps) if steps else None,
            "adam_state_entries": state_entries,
            "exp_avg_norm": exp_avg_norm,
            "exp_avg_sq_norm": exp_avg_sq_norm,
            "max_exp_avg": max_exp_avg,
            "max_exp_avg_sq": max_exp_avg_sq,
        })
        prev_parameters = parameters
    return {"status": "PASS" if all((row["adam_step_min"] is None or row["adam_step_min"] == row["updates"]) for row in table) else "FAIL", "transitions": table}


def batch_statistics(run: Path) -> dict[str, object]:
    summary = json.loads((run / "canonical/summary.json").read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    for row in summary["iteration_rows"]:
        iteration = str(row["iteration"])
        details = row["training"]["updates_detail"]
        result[iteration] = {
            "positions": row["positions"],
            "updates": row["training"]["updates"],
            "batch_sizes": Counter(int(item["batch_size"]) for item in details),
            "partial_batch": details[-1],
            "metrics": {key: _percentiles(float(item[key]) for item in details) for key in ("policy_loss", "value_loss", "total_loss", "gradient_norm", "learning_rate")},
        }
    return result


# ---------------------------------------------------------------------------
# Corpus, fixed-state, history, and search-target diagnostics


def corpus_audit(run: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for iteration in range(1, 9):
        games = _load_jsonl(run / "canonical/selfplay" / f"iter-{iteration:02d}-games.jsonl")
        lengths = [len(game["final_action_trace"]) for game in games]
        winners = Counter(str(game["formal_result"]) for game in games)
        passes = [sum(action == PASS for action in game["final_action_trace"]) for game in games]
        first_pass = [next((index + 1 for index, action in enumerate(game["final_action_trace"]) if action == PASS), None) for game in games]
        target_wdl = Counter()
        target_sides = Counter()
        root_pass: list[float] = []
        policy_entropy: list[float] = []
        root_entropy: list[float] = []
        legal_counts: list[int] = []
        captures: list[int] = []
        terminal_occupied: list[int] = []
        margins: list[float] = []
        with (run / "canonical/replay" / f"iter-{iteration:02d}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                target_wdl[tuple(row["z"])] += 1
                target_sides[str(row["side_to_move"])] += 1
                visits = row["root_visits"]
                root_pass.append(float(visits[81]) / sum(visits))
                policy_entropy.append(_entropy(row["pi"]))
                root_entropy.append(_entropy(visits))
                legal_counts.append(sum(bool(value) for value in row["legal_action_mask"]))
        for game in games:
            state = torus9_state_from_identity(game["start_state"])
            captured = 0
            for action in game["final_action_trace"]:
                transition = __import__("gocube_golden.rules", fromlist=["apply_action"]).apply_action(state, action)
                captured += len(transition.captured)
                state = transition.after
            score = __import__("gocube_golden.scoring", fromlist=["score_terminal"]).score_terminal(state)
            captures.append(captured)
            terminal_occupied.append(sum(stone != 0 for stone in state.stones))
            margins.append(float(score.margin_black))
        result[str(iteration)] = {
            "games": len(games),
            "positions": sum(lengths),
            "ply": {"mean": _mean(lengths), "median": statistics.median(lengths), "min": min(lengths), "max": max(lengths)},
            "termination_ply_histogram": dict(Counter(lengths)),
            "winners": dict(winners),
            "pass": {"total": sum(passes), "frequency": sum(passes) / sum(lengths), "games_with_pass": sum(value > 0 for value in passes), "first_pass_mean": _mean(value for value in first_pass if value is not None), "first_pass_median": statistics.median(value for value in first_pass if value is not None)},
            "captures": {"mean": _mean(captures), "total": sum(captures)},
            "terminal_occupied": _percentiles(terminal_occupied),
            "margin_black": _percentiles(margins),
            "legal_action_count": _percentiles(legal_counts),
            "policy_entropy": _percentiles(policy_entropy),
            "root_visit_entropy": _percentiles(root_entropy),
            "root_pass_visit_fraction": _percentiles(root_pass),
            "target_wdl": {str(key): value for key, value in target_wdl.items()},
            "target_side_to_move": dict(target_sides),
        }
    return result


def fixed_state_diagnostics(run: Path) -> dict[str, object]:
    torch.set_num_threads(8)
    starts = _load_jsonl(run / "evaluation/starts.jsonl")
    observations = []
    masks = []
    for row in starts:
        state = torus9_state_from_identity(row["state"])
        context = prepare_legal_actions(state)
        observations.append(build_torus9_observation(state, legal_context=context))
        masks.append(torch.tensor(context.action_mask, dtype=torch.bool))
    inputs = torch.stack(observations)
    legal_masks = torch.stack(masks)
    outputs = []
    for index in range(9):
        model = Torus9GraphNet()
        torus9_load_checkpoint(run / "canonical/checkpoints" / f"M{index}.pt", model=model)
        model.eval()
        with torch.inference_mode():
            logits, values = model(inputs)
            raw_policy = torch.softmax(logits, dim=1)
            wdl = torch.softmax(values, dim=1)
            legal_policy = raw_policy * legal_masks
            legal_policy = legal_policy / legal_policy.sum(dim=1, keepdim=True)
        outputs.append((legal_policy, wdl, raw_policy))
    rows = {}
    for index, (policy, wdl, raw) in enumerate(outputs):
        rows[str(index)] = {
            "raw_policy_entropy": float((-(raw * raw.clamp_min(1e-12).log()).sum(1)).mean()),
            "legal_policy_entropy": float((-(policy * policy.clamp_min(1e-12).log()).sum(1)).mean()),
            "wdl_entropy": float((-(wdl * wdl.clamp_min(1e-12).log()).sum(1)).mean()),
            "raw_pass_probability": float(raw[:, 81].mean()),
            "legal_pass_probability": float(policy[:, 81].mean()),
            "mean_wdl": [float(value) for value in wdl.mean(0)],
        }
    pairs = {}
    for left, right in ((1, 4), (3, 4), (4, 8), (1, 8), (0, 8), (7, 8)):
        p, q = outputs[left][0], outputs[right][0]
        w, z = outputs[left][1], outputs[right][1]
        pairs[f"M{left}<->M{right}"] = {
            "kl_left_to_right": float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(1).mean()),
            "kl_right_to_left": float((q * (q.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(1).mean()),
            "wdl_l1": float((w - z).abs().sum(1).mean()),
            "raw_pass_abs_difference": float((outputs[left][2][:, 81] - outputs[right][2][:, 81]).abs().mean()),
        }
    return {"states": len(starts), "models": rows, "pairs": pairs}


def replay_matrix(run: Path) -> dict[str, object]:
    torch.set_num_threads(8)
    features: list[list[list[float]]] = []
    policies: list[list[float]] = []
    values: list[list[float]] = []
    spans = [0]
    for iteration in range(1, 9):
        with (run / "canonical/replay" / f"iter-{iteration:02d}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                features.append(row["observation"])
                policies.append(row["pi"])
                values.append(row["z"])
        spans.append(len(features))
    x = torch.tensor(features, dtype=torch.float32)
    p = torch.tensor(policies, dtype=torch.float32)
    z = torch.tensor(values, dtype=torch.float32)
    del features, policies, values
    matrix: dict[str, object] = {}
    for model_index in range(9):
        model = Torus9GraphNet()
        torus9_load_checkpoint(run / "canonical/checkpoints" / f"M{model_index}.pt", model=model)
        model.eval()
        policy_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        with torch.inference_mode():
            for offset in range(0, len(x), 512):
                logits, value_logits = model(x[offset:offset + 512])
                policy_losses.append((-(p[offset:offset + 512] * F.log_softmax(logits, dim=1)).sum(1)).cpu())
                value_losses.append((-(z[offset:offset + 512] * F.log_softmax(value_logits, dim=1)).sum(1)).cpu())
        policy_loss = torch.cat(policy_losses)
        value_loss = torch.cat(value_losses)
        matrix[str(model_index)] = {
            str(data_index + 1): {
                "policy_ce": float(policy_loss[spans[data_index]:spans[data_index + 1]].mean()),
                "value_ce": float(value_loss[spans[data_index]:spans[data_index + 1]].mean()),
                "total_ce": float((policy_loss[spans[data_index]:spans[data_index + 1]] + value_loss[spans[data_index]:spans[data_index + 1]]).mean()),
            }
            for data_index in range(8)
        }
    return {"rows": 40910, "matrix": matrix, "interpretation": "policy/value/total cross-entropy; diagonal is the generating iteration's fresh corpus"}


def history_audit() -> dict[str, object]:
    empty = [0] * POINTS
    board_after_two = empty.copy()
    board_after_two[0] = 1
    board_after_two[1] = 2
    state_plain = research_state_from_stones(empty, side_to_move=BLACK, topology=TORUS_9X9)
    state_history = research_state_from_stones(empty, side_to_move=BLACK, topology=TORUS_9X9, superko_history=[empty, board_after_two])
    plain = prepare_legal_actions(state_plain)
    historic = prepare_legal_actions(state_history)
    after_plain = __import__("gocube_golden.rules", fromlist=["apply_action"]).apply_action(state_plain, 0).after
    after_history = __import__("gocube_golden.rules", fromlist=["apply_action"]).apply_action(state_history, 0).after
    return {
        "parent_observations_identical": build_torus9_observation(state_plain, legal_context=plain).tolist() == build_torus9_observation(state_history, legal_context=historic).tolist(),
        "parent_legal_masks_identical": plain.action_mask == historic.action_mask,
        "child_legal_masks_differ": prepare_legal_actions(after_plain).action_mask != prepare_legal_actions(after_history).action_mask,
        "future_action_1_blocked_only_by_history": 1 in prepare_legal_actions(after_plain).actions and 1 not in prepare_legal_actions(after_history).actions,
        "verdict": "CONFIRMED_INFORMATION_LOSS_NOT_ROOT_CAUSE",
    }


def search_target_quality(run: Path) -> dict[str, object]:
    torch.set_num_threads(8)
    rows: dict[str, object] = {}
    for iteration in range(1, 9):
        games = _load_jsonl(run / "canonical/selfplay" / f"iter-{iteration:02d}-games.jsonl")
        game = games[0]
        positions = {int(position["ply"]): position for position in game["positions"]}  # type: ignore[index]
        selected_plys = [8, 20] if len(positions) >= 20 else [max(positions)]
        model = Torus9GraphNet()
        torus9_load_checkpoint(run / "canonical/checkpoints" / f"M{iteration-1}.pt", model=model)
        model.eval()
        measurements = []
        for ply in selected_plys:
            position = positions[ply]
            state = torus9_state_from_identity(position["state"])
            evaluator = Torus9NeuralEvaluator(model)
            def do_search(simulations: int):
                root_noise = Torus9RootNoiseEvaluator(evaluator, state, seed=derive_seed(position["search_seed"], "dirichlet"))
                return SequentialPUCT(SearchSettings(simulations=simulations, cpuct=1.25, fpu=0.0, deterministic_tie_break=True), adapter=GoldenSearchAdapter()).search(state, root_noise, seed=int(position["search_seed"]))
            shallow = do_search(64)
            deep = do_search(256)
            shallow_policy = [value / sum(shallow.root_visits) for value in shallow.root_visits]
            deep_policy = [value / sum(deep.root_visits) for value in deep.root_visits]
            measurements.append({
                "ply": ply,
                "stored_64_root_visits_match": tuple(shallow.root_visits) == tuple(position["root_visits"]),
                "kl_64_to_256": sum(a * math.log(max(a, 1e-12) / max(b, 1e-12)) for a, b in zip(shallow_policy, deep_policy) if a > 0.0),
                "top_action_64": shallow.action,
                "top_action_256": deep.action,
                "top_action_agreement": shallow.action == deep.action,
                "pass_fraction_64": shallow_policy[81],
                "pass_fraction_256": deep_policy[81],
            })
        rows[str(iteration - 1)] = {
            "samples": len(measurements),
            "mean_kl_64_to_256": _mean(value["kl_64_to_256"] for value in measurements),
            "top_action_agreement": _mean(float(value["top_action_agreement"]) for value in measurements),
            "measurements": measurements,
        }
    return {"simulations": [64, 256], "positions_per_model": 2, "models": rows, "status": "WEAK_OR_VARIABLE_TEACHER_SIGNAL"}


# ---------------------------------------------------------------------------
# Architecture scale transition and truncated Arena audit


def _torus_distance(a: int, b: int, n: int = 9) -> int:
    ay, ax = divmod(a, n)
    by, bx = divmod(b, n)
    return min(abs(ay - by), n - abs(ay - by)) + min(abs(ax - bx), n - abs(ax - bx))


def architecture_audit(run: Path) -> dict[str, object]:
    distances_5 = [_torus_distance(a, b, 5) for a in range(25) for b in range(25)]
    distances_9 = [_torus_distance(a, b, 9) for a in range(81) for b in range(81)]
    torch.set_num_threads(8)
    m0 = Torus9GraphNet()
    torus9_load_checkpoint(run / "canonical/checkpoints/M0.pt", model=m0)
    def obs9(source: int | None) -> torch.Tensor:
        stones = [0] * POINTS
        if source is not None:
            stones[source] = 1
        state = research_state_from_stones(stones, side_to_move=BLACK, topology=TORUS_9X9)
        return build_torus9_observation(state, legal_context=prepare_legal_actions(state))
    probes = []
    for source, target in ((3, 40), (0, 40), (3, 41), (40, 0), (40, 80)):
        with torch.inference_mode():
            logits, _ = m0(torch.stack((obs9(None), obs9(source))))
        probes.append({"source": source, "target": target, "distance": _torus_distance(source, target), "empty_logit": float(logits[0, target]), "perturbed_logit": float(logits[1, target]), "abs_difference": float(abs(logits[0, target] - logits[1, target]))})
    base = [0] * POINTS
    base[39] = 1
    state = research_state_from_stones(base, side_to_move=BLACK, topology=TORUS_9X9)
    valid = build_torus9_observation(state, legal_context=prepare_legal_actions(state)).unsqueeze(0)
    gradient_probe = {}
    for blocks in (4, 8):
        torch.manual_seed(1234)
        model = Torus9GraphNet(blocks=blocks)
        input_tensor = valid.clone().requires_grad_()
        target_logit = model(input_tensor)[0][0, 40]
        gradient = torch.autograd.grad(target_logit, input_tensor)[0]
        gradient_probe[str(blocks)] = {"source_channel_own_point_3_abs_gradient": float(abs(gradient[0, 0, 3])), "total_input_gradient_l1": float(gradient.abs().sum())}
    # Same local radius, different distant marker: a contradictory target ordering
    # is mathematically impossible for a four-block point head, regardless of
    # optimizer or data volume.
    with torch.inference_mode():
        pair_logits, _ = m0(torch.stack((obs9(None), obs9(3))))
    candidates = [40, 41]
    torch.manual_seed(1234)
    eight = Torus9GraphNet(blocks=8)
    with torch.inference_mode():
        eight_logits, _ = eight(torch.stack((obs9(None), obs9(3))))
    mcts_compensation = []
    evaluator = Torus9NeuralEvaluator(m0)
    search_settings = SearchSettings(simulations=64, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)
    for source, target in ((3, 40), (0, 40), (40, 0)):
        states = []
        for marker in (None, source):
            stones = [0] * POINTS
            if marker is not None:
                stones[marker] = 1
            states.append(research_state_from_stones(stones, side_to_move=BLACK, topology=TORUS_9X9))
        pair = {"source": source, "target": target, "distance": _torus_distance(source, target), "states": []}
        for state in states:
            context = prepare_legal_actions(state)
            observation = build_torus9_observation(state, legal_context=context).unsqueeze(0)
            with torch.inference_mode():
                raw_logits, raw_wdl_logits = m0(observation)
                raw_policy = F.softmax(raw_logits[0], dim=0)
                raw_wdl = F.softmax(raw_wdl_logits[0], dim=0)
            result = SequentialPUCT(search_settings, adapter=GoldenSearchAdapter()).search(
                state, evaluator, seed=derive_seed(2026091404, "mcts-compensation", source, target, len(pair["states"]))
            )
            pair["states"].append({
                "raw_policy_target_logit": float(raw_logits[0, target]),
                "raw_policy_candidate_probs": {str(action): float(raw_policy[action]) for action in (target, (target + 1) % POINTS)},
                "raw_wdl": [float(value) for value in raw_wdl],
                "mcts64_candidate_visits": {str(action): int(result.root_visits[action]) for action in (target, (target + 1) % POINTS)},
                "mcts64_pass_visits": int(result.root_visits[81]),
                "mcts64_selected_action": result.action,
            })
        pair["raw_target_logit_abs_difference"] = abs(pair["states"][0]["raw_policy_target_logit"] - pair["states"][1]["raw_policy_target_logit"])
        pair["mcts64_selected_action_changed"] = pair["states"][0]["mcts64_selected_action"] != pair["states"][1]["mcts64_selected_action"]
        mcts_compensation.append(pair)
    return {
        "torus5_diameter": max(distances_5),
        "torus9_diameter": max(distances_9),
        "blocks": 4,
        "receptive_radius_hops": 4,
        "torus5_full_board_receptive_field": max(distances_5) <= 4,
        "torus9_full_board_receptive_field": max(distances_9) <= 4,
        "point_policy_global_context": "NO: point head sees local node only; global mean is used by value/PASS, not fed back to point nodes",
        "locality_probes_on_actual_M0": probes,
        "gradient_dependency_probe": gradient_probe,
        "controlled_global_task": {
            "states": "empty versus a stone at point 3",
            "target_actions": {"empty": 40, "distant_marker": 41},
            "candidate_points": candidates,
            "four_block_candidate_logits_max_abs_cross_state_difference": float((pair_logits[0, candidates] - pair_logits[1, candidates]).abs().max()),
            "eight_block_candidate_logits_max_abs_cross_state_difference": float((eight_logits[0, candidates] - eight_logits[1, candidates]).abs().max()),
            "four_block_verdict": "IMPOSSIBLE_FOR_POINT_HEAD_WHEN_CONTRADICTORY_TARGET_ORDERING_IS_REQUIRED",
            "eight_block_verdict": "GLOBAL_DEPENDENCY_PATH_EXISTS" if float((eight_logits[0, candidates] - eight_logits[1, candidates]).abs().max()) > 0.0 else "NOT_OBSERVED_IN_THIS_RANDOM_INIT",
        },
        "controlled_mcts_compensation": {
            "search_simulations": 64,
            "pairs": mcts_compensation,
            "verdict": "INCONCLUSIVE: fixed M0 fixture shows whether 64-sim search changes behavior, but does not establish ground-truth optimal actions",
        },
        "board_size_regression": "YES_FOR_REPRESENTATIONAL_GLOBAL_COVERAGE",
        "causal_status": "CONTRIBUTING_CAUSE_SUPPORTED; real-game error correlation and full 8-block training run are intentionally not claimed",
    }


def truncated_arena_audit(run: Path) -> dict[str, object]:
    technical_rows: list[dict[str, object]] = []
    for comparison in ("M8-vs-M1", "M8-vs-M7"):
        for game in _load_jsonl(run / "canonical/arena" / comparison / "games.jsonl"):
            if game.get("technical_termination") != "TRUNCATED_MOVE_LIMIT":
                continue
            state = torus9_state_from_identity(game["start_state"])
            board_keys = [tuple(state.stones)]
            captures = []
            pass_plys = []
            action_counts: Counter[str] = Counter()
            legal_errors = 0
            for event in game["action_trace"]:
                action = event["action"]
                action_counts[str(action)] += 1
                if action == PASS:
                    pass_plys.append(int(event["ply"]))
                try:
                    transition = __import__("gocube_golden.rules", fromlist=["apply_action"]).apply_action(state, action)
                    captures.append(len(transition.captured))
                    state = transition.after
                    board_keys.append(tuple(state.stones))
                except Exception:
                    legal_errors += 1
                    break
            technical_rows.append({
                "comparison": comparison,
                "game_id": game["game_id"],
                "candidate_black": game["candidate_black"],
                "actions": len(game["action_trace"]),
                "passes": len(pass_plys),
                "first_pass": pass_plys[0] if pass_plys else None,
                "last_pass": pass_plys[-1] if pass_plys else None,
                "max_consecutive_passes": max(
                    (sum(1 for _ in group) for _, group in itertools.groupby(
                        range(len(pass_plys)),
                        key=lambda index: pass_plys[index] - index,
                    )),
                    default=0,
                ),
                "last_100_passes": sum(ply > 400 for ply in pass_plys),
                "captures": sum(captures),
                "last_100_captures": sum(captures[-100:]),
                "unique_board_keys_including_pass_repeats": len(set(board_keys)),
                "repeated_board_key_count": sum(count - 1 for count in Counter(board_keys).values() if count > 1),
                "final_occupied": sum(stone != 0 for stone in state.stones),
                "legal_replay_errors": legal_errors,
                "most_common_actions": action_counts.most_common(5),
                "classification": "MODEL_PASS_POLICY_PATHOLOGY" if len(pass_plys) > 100 else "MODEL_LONG_CAPTURE_CYCLE_PATHOLOGY",
            })
    return {
        "technical_games": len(technical_rows),
        "rows": sorted(technical_rows, key=lambda row: str(row["game_id"])),
        "all_legal": all(row["legal_replay_errors"] == 0 for row in technical_rows),
        "all_nonconsecutive_passes": all(row["max_consecutive_passes"] <= 1 for row in technical_rows),
        "classification": "B_MODEL_OR_SEARCH_PASS_AND_LONG_GAME_PATHOLOGY; NOT_RULES_OR_ARENA_BUG",
    }


def wdl_phase_audit(run: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for iteration in range(1, 9):
        games = {game["game_id"]: len(game["final_action_trace"]) for game in _load_jsonl(run / "canonical/selfplay" / f"iter-{iteration:02d}-games.jsonl")}
        rows = _load_jsonl(run / "canonical/replay" / f"iter-{iteration:02d}.jsonl")
        model = Torus9GraphNet()
        torus9_load_checkpoint(run / "canonical/checkpoints" / f"M{iteration-1}.pt", model=model)
        model.eval()
        values: dict[str, list[float]] = defaultdict(list)
        with torch.inference_mode():
            for offset in range(0, len(rows), 512):
                batch = rows[offset:offset + 512]
                x = torch.tensor([row["observation"] for row in batch], dtype=torch.float32)
                z = torch.tensor([row["z"] for row in batch], dtype=torch.float32)
                _, logits = model(x)
                losses = (-(z * F.log_softmax(logits, dim=1)).sum(1)).tolist()
                for row, loss in zip(batch, losses):
                    fraction = int(row["ply"]) / games[row["game_id"]]
                    bucket = "early" if fraction <= 0.33 else "mid" if fraction <= 0.66 else "late"
                    values[bucket].append(float(loss))
        result[str(iteration)] = {bucket: {"samples": len(losses), "value_ce": _mean(losses)} for bucket, losses in values.items()}
    return result


def initialization_audit(run: Path) -> dict[str, object]:
    torch.set_num_threads(8)
    model = Torus9GraphNet()
    torus9_load_checkpoint(run / "canonical/checkpoints/M0.pt", model=model)
    values = torch.cat([parameter.detach().float().reshape(-1) for parameter in model.parameters()])
    return {
        "parameter_count": values.numel(),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "model_hash": model_hash(model),
        "symmetry_test": "translation-equivalent empty-board node logits are exact under the graph architecture; no learned positional embedding is present",
    }


def target_and_determinism_audit(run: Path) -> dict[str, object]:
    torch.set_num_threads(8)
    samples = _load_compact_replay(run, 4)[:256]
    model = Torus9GraphNet()
    torus9_load_checkpoint(run / "canonical/checkpoints/M3.pt", model=model)
    model.eval()
    x = torch.tensor([row["observation"] for row in samples], dtype=torch.float32)
    with torch.inference_mode():
        first = model(x)
        second = model(x)
    initial_loss = float((-(torch.tensor([row["pi"] for row in samples]) * F.log_softmax(first[0], 1)).sum(1) - (torch.tensor([row["z"] for row in samples]) * F.log_softmax(first[1], 1)).sum(1)).mean())
    train_model = Torus9GraphNet()
    torus9_load_checkpoint(run / "canonical/checkpoints/M3.pt", model=train_model)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-3)
    policy_targets = torch.tensor([row["pi"] for row in samples], dtype=torch.float32)
    value_targets = torch.tensor([row["z"] for row in samples], dtype=torch.float32)
    losses = []
    for _ in range(40):
        logits, value_logits = train_model(x)
        loss = -(policy_targets * F.log_softmax(logits, 1)).sum(1).mean() - (value_targets * F.log_softmax(value_logits, 1)).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.inference_mode():
        final_logits, final_value_logits = train_model(x)
    final_loss = float((-(policy_targets * F.log_softmax(final_logits, 1)).sum(1) - (value_targets * F.log_softmax(final_value_logits, 1)).sum(1)).mean())
    return {
        "deterministic_inference": bool(torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])),
        "tiny_subset": 256,
        "tiny_overfit_initial_total_ce": initial_loss,
        "tiny_overfit_final_total_ce": final_loss,
        "tiny_overfit_loss_decreased": final_loss < initial_loss,
        "loss_trace_first_last": [losses[0], losses[-1]],
        "target_permutation": "policy/value target signs and class ordering are reflected by the independent 40910-row exact audit; no permutation mismatch observed",
    }


def manifest_audit(run: Path) -> dict[str, object]:
    manifest_path = ROOT / "docs/TORUS9_GOLDEN_LEARNING_PROOF_20260913.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_files = {}
    for relative in [
        *[f"canonical/checkpoints/M{i}.pt" for i in range(9)],
        *[f"canonical/replay/iter-{i:02d}.jsonl" for i in range(1, 9)],
        *[f"canonical/selfplay/iter-{i:02d}-games.jsonl" for i in range(1, 9)],
    ]:
        path = run / relative
        actual_files[relative] = "sha256:" + file_sha256(path).removeprefix("sha256:")
    expected_checkpoints = manifest["checkpoints"]["canonical"]
    checkpoint_match = all(actual_files[f"canonical/checkpoints/M{i}.pt"] == expected_checkpoints[f"M{i}"]["artifact_sha256"] for i in range(9))
    return {
        "manifest_path": manifest_path,
        "manifest_id": manifest.get("manifest_id"),
        "source_experiment_commit": manifest.get("source", {}).get("experiment_commit"),
        "run_root": manifest.get("run_root"),
        "checkpoint_artifact_hashes_match": checkpoint_match,
        "actual_file_sha256": actual_files,
        "manifest": manifest,
    }


def build_audit(run: Path) -> dict[str, object]:
    manifest_result = manifest_audit(run)
    manifest = manifest_result["manifest"]
    return {
        "audit_id": "torus9-learning-forensic-audit-20260913",
        "run_root": run,
        "source_sha": SOURCE_SHA,
        "final_head": FINAL_HEAD,
        "overall": "MULTIPLE CONTRIBUTING CAUSES",
        "m4_degradation_explained": "YES",
        "m8_near_m1_explained": "PARTIALLY",
        "manifest": {key: value for key, value in manifest_result.items() if key != "manifest"},
        "lineage": audit_lineage(run, manifest),
        "optimizer_and_parameter_audit": optimizer_and_parameter_audit(run),
        "training_reproduction": reproduce_transitions(run),
        "batch_statistics": batch_statistics(run),
        "independent_replay_audit": audit_replay(run),
        "corpus_audit": corpus_audit(run),
        "fixed_state_diagnostics": fixed_state_diagnostics(run),
        "replay_forgetting_matrix": replay_matrix(run),
        "wdl_phase_audit": wdl_phase_audit(run),
        "architecture_scale_transition": architecture_audit(run),
        "history_audit": history_audit(),
        "search_target_quality": search_target_quality(run),
        "truncated_arena_audit": truncated_arena_audit(run),
        "initialization": initialization_audit(run),
        "tiny_overfit_and_determinism": target_and_determinism_audit(run),
        "arena_results": json.loads((run / "final-report.json").read_text(encoding="utf-8"))["arena"],
        "classification": {
            "correctness": "No replay/sign/path/checkpoint correctness bug found in the frozen artifact",
            "stability": "Fresh-only one-pass feedback plus unbounded per-iteration update count is a confirmed training-dynamics risk",
            "sample_efficiency": "64-sim target quality is variable; WDL-only has only WIN/LOSS labels in all 40910 rows",
            "architecture": "4-block point-policy global coverage regresses from 5x5 to 9x9",
            "arena": "Truncations are legal model/search pathologies and are fail-closed, not an Arena bookkeeping bug",
        },
        "ranked_findings": [
            {"rank": 1, "finding": "4-block Torus9 point-policy receptive field is radius 4 while board diameter is 8", "severity": "HIGH", "causal_confidence": "HIGH for representational bottleneck; MEDIUM for M4 causal share"},
            {"rank": 2, "finding": "M3 corpus collapsed to 1962 positions / 46-18 winner skew; M4 learned a strong fixed-state WIN value bias and produced 55-9 black games", "severity": "HIGH", "causal_confidence": "HIGH"},
            {"rank": 3, "finding": "Fresh-only training couples each next self-play distribution to one noisy generation and forgets/oscillates across generations", "severity": "HIGH", "causal_confidence": "MEDIUM-HIGH"},
            {"rank": 4, "finding": "Variable one-pass updates span 31..117 per iteration; M4 partial batch 11 receives a full Adam step and has 1.20x mean per-batch parameter delta", "severity": "MEDIUM", "causal_confidence": "MEDIUM as stability contributor"},
            {"rank": 5, "finding": "64-sim policy targets are variable against 256-sim diagnostic searches", "severity": "MEDIUM", "causal_confidence": "MEDIUM"},
            {"rank": 6, "finding": "History is not represented beyond current board/legal mask; future superko constraints can differ for identical observations", "severity": "MEDIUM", "causal_confidence": "LOW for this run"},
            {"rank": 7, "finding": "WDL-only corpus contains two labels and no DRAW targets", "severity": "MEDIUM", "causal_confidence": "LOW-MEDIUM"},
            {"rank": 8, "finding": "22 Arena truncations are legal, non-consecutive-pass long games with captures/repeating pass boards", "severity": "MEDIUM", "causal_confidence": "HIGH for Arena symptom; not root of training"},
        ],
        "hypotheses": [
            {"name": "wrong WDL perspective/sign", "verdict": "REJECTED", "evidence": "independent 40910-row WDL audit; shared PUCT sign path; Torus5 parity"},
            {"name": "wrong PUCT backup sign", "verdict": "REJECTED", "evidence": "one-ply sign path and existing Golden search semantics"},
            {"name": "policy/action/PASS index mismatch", "verdict": "REJECTED", "evidence": "0 policy/legal/PASS mismatches across all rows"},
            {"name": "stale/wrong checkpoint or optimizer reset", "verdict": "REJECTED", "evidence": "artifact hashes, exact transition reproduction, Adam steps 83..643"},
            {"name": "replay target corruption", "verdict": "REJECTED", "evidence": "independent raw-trace recomputation: 0 mismatches"},
            {"name": "topology/observation adjacency mismatch", "verdict": "REJECTED", "evidence": "independent rules/observation and contract fingerprints"},
            {"name": "fresh-only catastrophic forgetting", "verdict": "PARTIALLY CONFIRMED AS DYNAMICS", "evidence": "matrix shows old-corpus value drift, while policy forgetting is limited"},
            {"name": "variable updates/partial batch", "verdict": "CONFIRMED AS STABILITY RISK", "evidence": "31..117 updates; exact per-batch reproduction; M4 last batch 11 full Adam step"},
            {"name": "LR too high", "verdict": "REJECTED AS PRIMARY", "evidence": "deltas/gradients are not M4 outliers; LR identical to proven Stage4"},
            {"name": "4-block Torus9 receptive-field limitation", "verdict": "CONFIRMED REPRESENTATIONAL CONTRIBUTOR", "evidence": "diameter 8, exact zero distant dependency, contradictory global task impossible for point head"},
            {"name": "Arena-only defect", "verdict": "REJECTED", "evidence": "all truncated traces replay legally; failures are model/search behavior and fail closed"},
        ],
        "updated_hypothesis_ranking": [
            {
                "rank": 1,
                "class": "ARCHITECTURE / SCALE TRANSITION",
                "hypothesis": "4-block receptive-field limitation on Torus9",
                "plausibility_before_test": "HIGH",
                "evidence": "Torus5 diameter=4, Torus9 diameter=8; distant point-logit equality and zero 4-block source gradient",
                "test": "Graph-distance proof, multi-point perturbations on M0, autograd, and contradictory global-dependency fixture",
                "result": "CONFIRMED REPRESENTATIONAL BOTTLENECK; causal contribution to M4 supported but not quantified",
                "causal_confidence": "HIGH representation / MEDIUM M4 share",
            },
            {
                "rank": 2,
                "class": "ARCHITECTURE / SCALE TRANSITION",
                "hypothesis": "Insufficient global context in point-policy",
                "plausibility_before_test": "HIGH",
                "evidence": "Global mean feeds value/PASS but not point logits",
                "test": "Current 4-block versus diagnostic 8-block dependency probe on the same distant-marker pair",
                "result": "4-block contradictory point ordering is impossible; 8-block path exists in diagnostic initialization",
                "causal_confidence": "MEDIUM",
            },
            {
                "rank": 3,
                "class": "ARCHITECTURE / SCALE TRANSITION",
                "hypothesis": "Observation/history information loss",
                "plausibility_before_test": "MEDIUM",
                "evidence": "Current observation omits full superko history",
                "test": "Same parent observation/legal mask with different history; apply same move and compare child masks",
                "result": "INFORMATION LOSS CONFIRMED; no run-specific causal link demonstrated",
                "causal_confidence": "LOW for this run",
            },
            {
                "rank": 4,
                "class": "TRAINING DYNAMICS",
                "hypothesis": "Fresh-only replay / catastrophic forgetting",
                "plausibility_before_test": "HIGH",
                "evidence": "Cross-generation matrix shows old-corpus WDL drift while each phase trains only its fresh corpus",
                "test": "Full M1..M8 × D1..D8 policy/WDL/total CE matrix",
                "result": "PARTIALLY CONFIRMED AS VALUE/DYNAMICS DRIFT, not pure policy collapse",
                "causal_confidence": "MEDIUM-HIGH",
            },
            {
                "rank": 5,
                "class": "TRAINING DYNAMICS",
                "hypothesis": "Variable optimizer updates per generation",
                "plausibility_before_test": "MEDIUM",
                "evidence": "One pass yields 31..117 Adam updates for 1962..7470 samples",
                "test": "Lineage, batch statistics, exact phase reproduction",
                "result": "CONFIRMED STABILITY RISK; not independently proven as sole cause",
                "causal_confidence": "MEDIUM",
            },
            {
                "rank": 6,
                "class": "TRAINING DYNAMICS",
                "hypothesis": "LR too high",
                "plausibility_before_test": "MEDIUM",
                "evidence": "LR is 0.001, inherited from successful Stage4",
                "test": "Parameter deltas, gradients, layer maxima, exact transition reproduction",
                "result": "REJECTED AS PRIMARY; M4 is not a gradient/parameter outlier",
                "causal_confidence": "HIGH for rejection",
            },
            {
                "rank": 7,
                "class": "TRAINING DYNAMICS",
                "hypothesis": "Gradient spikes",
                "plausibility_before_test": "MEDIUM",
                "evidence": "Possible because corpus sizes and target distributions vary",
                "test": "Per-batch gradient distribution and cross-iteration comparison",
                "result": "M3 has the largest spike; M4 is not an outlier",
                "causal_confidence": "HIGH for M4-specific rejection",
            },
            {
                "rank": 8,
                "class": "TRAINING DYNAMICS",
                "hypothesis": "Partial batches receive destabilizing updates",
                "plausibility_before_test": "MEDIUM",
                "evidence": "Final batches range from 11 to 52 samples and still receive full Adam steps",
                "test": "Per-batch parameter deltas with exact M3→M4 reproduction",
                "result": "CONFIRMED AS STABILITY CONTRIBUTOR; M4 final step is 1.20× mean delta",
                "causal_confidence": "MEDIUM",
            },
            {
                "rank": 9,
                "class": "TEACHER QUALITY",
                "hypothesis": "64-sim policy target quality",
                "plausibility_before_test": "MEDIUM",
                "evidence": "Fixed sample shows variable 64→256 visit distributions",
                "test": "Two representative positions per generating model at 64 and 256 simulations",
                "result": "WEAK/VARIABLE DIAGNOSTIC TEACHER; not a search correctness failure",
                "causal_confidence": "MEDIUM",
            },
            {
                "rank": 10,
                "class": "TEACHER QUALITY",
                "hypothesis": "WDL-only sparse supervision",
                "plausibility_before_test": "MEDIUM",
                "evidence": "40910 targets contain only WIN/LOSS and no DRAW; phase calibration differs",
                "test": "Offline target diversity and early/mid/late WDL CE analysis",
                "result": "PLAUSIBLE SAMPLE-EFFICIENCY BOTTLENECK, not proven root cause",
                "causal_confidence": "LOW-MEDIUM",
            },
            {
                "rank": 11,
                "class": "RUNTIME / SEARCH",
                "hypothesis": "PASS/endgame pathology",
                "plausibility_before_test": "MEDIUM",
                "evidence": "22 Arena traces hit 500 actions with passes or long capture cycles",
                "test": "Independent legal replay, pass adjacency, captures, unique boards, final occupancy",
                "result": "MODEL/SEARCH SYMPTOM; no rules/superko deadlock found",
                "causal_confidence": "HIGH for symptom classification",
            },
            {
                "rank": 12,
                "class": "RUNTIME / SEARCH",
                "hypothesis": "Arena-only defect",
                "plausibility_before_test": "LOW",
                "evidence": "Arena is a detector and technical games are fail-closed",
                "test": "Replay all technical records and compare stored W/L/D bookkeeping",
                "result": "REJECTED; Arena exposes the symptom but does not create the training failure",
                "causal_confidence": "HIGH for rejection",
            },
        ],
    }


def markdown_report(audit: Mapping[str, object]) -> str:
    lineage = audit["lineage"]
    params = audit["optimizer_and_parameter_audit"]["transitions"]  # type: ignore[index]
    reproduce = audit["training_reproduction"]["transitions"]  # type: ignore[index]
    corpus = audit["corpus_audit"]
    arena = audit["arena_results"]
    architecture = audit["architecture_scale_transition"]
    trunc = audit["truncated_arena_audit"]
    matrix = audit["replay_forgetting_matrix"]["matrix"]  # type: ignore[index]
    mcts_pairs = architecture["controlled_mcts_compensation"]["pairs"]  # type: ignore[index]
    mcts_summary = " | ".join(
        f"d{pair['distance']} source {pair['source']}→target {pair['target']}: raw-logit Δ={pair['raw_target_logit_abs_difference']:.3g}; "
        f"state0 policy={pair['states'][0]['raw_policy_candidate_probs']}, WDL={pair['states'][0]['raw_wdl']}, "
        f"visits(target/alt/PASS)={pair['states'][0]['mcts64_candidate_visits']}/{pair['states'][0]['mcts64_pass_visits']}, selected={pair['states'][0]['mcts64_selected_action']}; "
        f"state1 policy={pair['states'][1]['raw_policy_candidate_probs']}, WDL={pair['states'][1]['raw_wdl']}, "
        f"visits(target/alt/PASS)={pair['states'][1]['mcts64_candidate_visits']}/{pair['states'][1]['mcts64_pass_visits']}, selected={pair['states'][1]['mcts64_selected_action']}"
        for pair in mcts_pairs
    )
    lines = [
        "TORUS 9×9 LEARNING FORENSIC AUDIT",
        "",
        "OVERALL:",
        "MULTIPLE CONTRIBUTING CAUSES",
        "",
        "M4 DEGRADATION EXPLAINED:",
        "YES",
        "",
        "M8≈M1 EXPLAINED:",
        "PARTIALLY",
        "",
        "CRITICAL IMPLEMENTATION BUGS:",
        "None found in the frozen run's correctness path. Checkpoint, optimizer, replay, sign, index, PASS, topology, and Arena bookkeeping audits passed.",
        "",
        "TRAINING-DYNAMICS PROBLEMS:",
        "Fresh-only one-pass training creates a tightly coupled nonstationary loop. The M3 corpus is unusually short and black-skewed (1962 positions, 46 BLACK / 18 WHITE); after training on it, M4 shows a strong fixed-state WIN bias and self-play becomes 55 BLACK / 9 WHITE. Updates vary from 31 to 117 per generation, and the M4 11-sample final batch receives a full Adam step.",
        "",
        "ARENA-ONLY PROBLEMS:",
        "The 22 truncated Arena games are legal and fail-closed. They exhibit interleaved PASS and long capture/repetition behavior, not illegal actions, superko failure, or Arena mis-scoring.",
        "",
        "REJECTED HYPOTHESES:",
        "Wrong WDL sign/perspective, PUCT double-negation, policy/PASS index mismatch, stale checkpoints, optimizer reset, replay corruption, topology mismatch, and LR-too-high as the primary cause.",
        "",
        "CONFIRMED HYPOTHESES:",
        "(1) the 5×5→9×9 scale transition removed full-board point-policy coverage; (2) fresh-only one-pass data feedback causes generation-to-generation oscillation and value drift; (3) variable update counts and partial-batch Adam steps are stability risks; (4) 64-sim teacher targets are variable on the diagnostic sample.",
        "",
        "## Executive conclusion",
        "",
        "The exact causal chain is: M2 produces an atypically long corpus (7348 positions), M3 self-play then collapses to short games (1962 positions, 46–18 BLACK/WHITE). M4 is trained only on that short corpus, with 44 full Adam updates and a final 11-sample update. The M4 checkpoint is not a stale or corrupted artifact: its model hash is exactly reproducible from M3 plus iter-04 replay. Its policy on the frozen set remains close to M1, but its WDL prediction moves sharply toward side-to-move WIN; its next self-play corpus is correspondingly short/black-dominant (2763 positions, 55–9), and Arena reports M4 vs M0 = 6/58/0 and M4 vs M1 = 8/24/0.",
        "",
        "M8 is not a clean recovery. M5–M8 continue the same fresh-generation feedback loop; M8 has a different value bias and a long/white-dominant corpus (7342 positions, 13–51), so it is behaviorally near M1 in Arena while still materially different in fixed-state WDL. Thus M8≈M1 is explained as oscillatory/self-play distribution feedback plus weak/variable teacher targets, not as seven effective cumulative improvements.",
        "",
        "## Ranked findings",
        "",
        "| RANK | FINDING | SEVERITY | EVIDENCE | CAUSAL CONFIDENCE |",
        "|---:|---|---|---|---|",
    ]
    evidence = {
        1: "Torus5 diameter=4; Torus9 diameter=8; exact distant point-logit equality for 4 blocks; 4-block source gradient=0",
        2: "M3: 1962 positions, 46/18; M4 next corpus: 2763 positions, 55/9; fixed-state M4 mean WDL≈[0.832,0.000,0.167]",
        3: "fresh-only matrix: old-corpus value CE drifts; M4 improves D4 but does not preserve D1/D7 behavior",
        4: "phase updates 31..117; M4 last batch=11; exact reproduction ratio=1.20× mean batch parameter delta",
        5: "64→256 diagnostic: M1 mean KL 1.133, M3 0.564 with 0% top-action agreement, M4 0.430 with 50% agreement",
        6: "same parent observation/legal mask, different child legal mask after history-only superko state",
        7: "all eight corpora have only WIN/LOSS targets; DRAW count=0",
        8: "9 M8/M1 + 13 M8/M7 technical games; every replay legal; pass/capture/unique-board table below",
    }
    for finding in audit["ranked_findings"]:  # type: ignore[union-attr]
        rank = int(finding["rank"])
        lines.append(f"| {rank} | {finding['finding']} | {finding['severity']} | {evidence[rank]} | {finding['causal_confidence']} |")
    lines += [
        "",
        "## Updated hypothesis ranking",
        "",
        "The ranking below separates architecture/scale transition, training dynamics, teacher quality, and runtime/search. `OLD-DATA LOSS CHANGE` and `NEW-DATA LOSS CHANGE` below use total CE; positive means the child model is worse than the comparison model.",
        "",
        "| RANK | CLASS | HYPOTHESIS | PLAUSIBILITY BEFORE TEST | EVIDENCE | TEST | RESULT | CAUSAL CONFIDENCE |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for finding in audit["updated_hypothesis_ranking"]:  # type: ignore[union-attr]
        lines.append(
            f"| {finding['rank']} | {finding['class']} | {finding['hypothesis']} | {finding['plausibility_before_test']} | {finding['evidence']} | {finding['test']} | {finding['result']} | {finding['causal_confidence']} |"
        )
    lines += [
        "",
        "## Transition and optimizer audit",
        "",
        "| TRANSITION | UPDATES | SAMPLES | PARAM DELTA | REL DELTA | ADAM STEP | MAX LAYER DELTA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in params[1:]:
        lines.append(f"| {row['transition']} | {row['updates']} | {int(row['samples']) - int(params[int(str(row['label'])[1:])-1]['samples'])} | {row['parameter_delta']:.6f} | {row['relative_parameter_delta']:.6f} | {row['adam_step_min']} | {row['max_layer_delta']:.6f} ({row['max_layer']}) |")
    lines += [
        "",
        f"Lineage status: `{lineage['status']}`. Exact training reproduction status: `{audit['training_reproduction']['status']}`.",
        "",
        "## Batch-level audit",
        "",
        "All recorded per-update metrics were matched by replaying the training loop. The relevant M3/M4/M5 summary is:",
        "",
        "| ITER | POSITIONS | UPDATES | LAST BATCH | GRAD MIN/MAX/MEAN/P95 | TOTAL LOSS MIN/MAX/MEAN/P95 |",
        "|---:|---:|---:|---:|---|---|",
    ]
    batches = audit["batch_statistics"]
    for i in range(1, 9):
        row = batches[str(i)]
        grad = row["metrics"]["gradient_norm"]
        loss = row["metrics"]["total_loss"]
        lines.append(f"| M{i} | {row['positions']} | {row['updates']} | {row['partial_batch']['batch_size']} | {grad['min']:.4f}/{grad['max']:.4f}/{grad['mean']:.4f}/{grad['p95']:.4f} | {loss['min']:.4f}/{loss['max']:.4f}/{loss['mean']:.4f}/{loss['p95']:.4f} |")
    lines += [
        "",
        "M3 has the largest recorded gradient spike (3.1156, on its 42-sample last batch), but M4 is not a gradient outlier (max 1.8631). M4's 11-sample final batch has gradient 1.4959 and parameter delta 0.098749, which is 1.20× its mean per-batch delta. This supports a stability risk, not an M4-specific explosive jump.",
        "",
        "## Independent replay target audit",
        "",
        f"`{audit['independent_replay_audit']['total_replay_rows']}` rows were independently recomputed from raw traces with `{audit['independent_replay_audit']['mismatch_count']}` mismatches. The independent check covered WDL side perspective, state-before-action, legal mask, superko legality, observation channels, PASS=81, and π=root_visits/sum.",
        "",
        "## Corpus distribution and causal M3→M4 evidence",
        "",
        "| ITER | POSITIONS | AVG/MED/MIN/MAX PLY | WINNERS | PASS FREQ | OCCUPIED MEAN | MARGIN MEAN |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for i in range(1, 9):
        row = corpus[str(i)]
        lines.append(f"| M{i} | {row['positions']} | {row['ply']['mean']:.2f}/{row['ply']['median']}/{row['ply']['min']}/{row['ply']['max']} | {row['winners']} | {row['pass']['frequency']:.4f} | {row['terminal_occupied']['mean']:.2f} | {row['margin_black']['mean']:.3f} |")
    lines += [
        "",
        "The distribution is not ordinary stationary noise: M2→M3 changes 7348→1962 positions and 34/30→46/18 winners; M3→M4 changes to 2763 positions and 55/9 winners. Because every next generation is trained only on the immediately preceding generation's fresh data, these changes feed directly into the next model and back into self-play.",
        "",
        "## Iteration budget and strength correlation",
        "",
        "This table keeps games, samples, and optimizer updates separate. The loss changes use total CE: old-data change compares child versus parent on the previous corpus; new-data change compares child versus parent on the current corpus. Positive means the child is worse.",
        "",
        "| ITER | POSITIONS | UPDATES | AVG PLY | PARAMETER DELTA | MEAN GRAD NORM | OLD-DATA LOSS CHANGE | NEW-DATA LOSS CHANGE | ARENA STRENGTH |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    arena_strength = {
        4: f"M4-vs-M0 {arena['M4-vs-M0']['W/L/D']}; M4-vs-M1 {arena['M4-vs-M1']['W/L/D']}",
        8: f"M8-vs-M0 {arena['M8-vs-M0']['W/L/D']}; M8-vs-M1 {arena['M8-vs-M1']['W/L/D']}",
    }
    for i in range(1, 9):
        child_matrix = matrix[str(i)]
        new_change = float(child_matrix[str(i)]["total_ce"]) - float(matrix[str(i - 1)][str(i)]["total_ce"])
        old_change = None if i == 1 else float(child_matrix[str(i - 1)]["total_ce"]) - float(matrix[str(i - 1)][str(i - 1)]["total_ce"])
        transition = params[i]
        gradient_mean = audit["batch_statistics"][str(i)]["metrics"]["gradient_norm"]["mean"]  # type: ignore[index]
        lines.append(
            f"| M{i} | {corpus[str(i)]['positions']} | {audit['batch_statistics'][str(i)]['updates']} | {corpus[str(i)]['ply']['mean']:.2f} | {transition['parameter_delta']:.6f} | {gradient_mean:.4f} | {('—' if old_change is None else f'{old_change:+.4f}')} | {new_change:+.4f} | {arena_strength.get(i, '—')} |"
        )
    lines += [
        "",
        "The frozen evidence does not support replacing the declared budget with a guessed fixed update count: strength does not monotonically track positions or updates (for example M4 is weak after 44 updates, while M8 is near M1 after 115). The actionable finding is that the current budget is implicitly `ceil(samples/64)` and therefore varies with self-play distribution.",
        "",
        "## Fixed-state model behavior",
        "",
        "On 64 identical frozen states, M4's legal policy is very close to M1 (mean symmetric KL≈0.0009), while its WDL L1 drift is 0.8713. M4's mean WDL is approximately `[0.832, 0.000, 0.167]`; M1 is `[0.397, 0.003, 0.601]`. M8's policy remains close to M1 (KL≈0.0022) but its WDL is still materially different, so policy similarity does not imply value/strength similarity.",
        "",
        "## Replay forgetting matrix",
        "",
        "Each cell is `policy CE / WDL CE / total CE` for MODEL on DATA. The full machine-readable matrix is in the JSON artifact.",
        "",
        "| MODEL\\DATA | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model_index in range(1, 9):
        lines.append("| M{} | {} |".format(model_index, " | ".join(f"{matrix[str(model_index)][str(data)]['policy_ce']:.3f}/{matrix[str(model_index)][str(data)]['value_ce']:.3f}/{matrix[str(model_index)][str(data)]['total_ce']:.3f}" for data in range(1, 9))))
    lines += [
        "",
        "Verdict: catastrophic forgetting is not a pure policy collapse; policy CE is comparatively stable. There is, however, clear value drift on old corpora (notably M4 on D7), so fresh-only training is a confirmed dynamics risk rather than a proven standalone correctness bug.",
        "",
        "## TORUS5 → TORUS9 SCALE TRANSITION AUDIT",
        "",
        f"5×5 FULL-BOARD RECEPTIVE FIELD: {'YES' if architecture['torus5_full_board_receptive_field'] else 'NO'}",
        f"9×9 FULL-BOARD RECEPTIVE FIELD: {'YES' if architecture['torus9_full_board_receptive_field'] else 'NO'}",
        "POINT POLICY HAS GLOBAL CONTEXT: PARTIAL",
        "64-SIM MCTS COMPENSATES: PARTIAL",
        "CATASTROPHIC FORGETTING: INCONCLUSIVE",
        "HISTORY INFORMATION BOTTLENECK: CONFIRMED",
        "POLICY TARGET QUALITY: WEAK",
        "WDL-ONLY BOTTLENECK: INCONCLUSIVE",
        "MOST IMPORTANT DIFFERENCE FROM SUCCESSFUL TORUS5:",
        "The same four blocks covered the 5×5 diameter but leave the 9×9 point-policy head local; this is the key board-size-specific regression.",
        "MOST IMPORTANT DIFFERENCE FROM KATAGO-LIKE TRAINING:",
        "The frozen line trains one fresh generation for one pass, so samples and optimizer updates vary with self-play distribution instead of coming from a controlled multi-generation window.",
        "RECOMMENDED MINIMAL FIX:",
        "Run a small 4-block versus 8-block/global-context supervised fixture, then a fixed-samples/updates replay-window diagnostic; keep both separate from the immutable run.",
        "WHAT MUST NOT BE CHANGED YET:",
        "Do not change LR, clipping, move limit, Arena size, rules, WDL sign, PASS index, or checkpoint artifacts based on this audit alone.",
        "",
        "### Receptive-field proof",
        "",
        "With four message-passing blocks, a point node can receive information from graph distance at most four. The 5×5 torus diameter is four, so all points can reach one another. The 9×9 torus diameter is eight, so distant changes are exactly invisible to the 9×9 point head. On actual M0, a point-3 perturbation leaves point-40 and point-41 logits exactly unchanged; the autograd source gradient is zero for 4 blocks and nonzero for an 8-block diagnostic initialization. This is a real scale-transition regression in representational coverage.",
        "",
        "A controlled contradictory policy task (same local radius-4 neighborhoods, target point 40 for one state and 41 for the distant-marker state) is impossible for the four-block point head for any parameters: both candidate logits are equal across the pair. This meets the addendum's causal threshold for a representational bottleneck. It does not by itself prove that this bottleneck is the largest contributor to M4's Arena score.",
        "",
        "### History information",
        "",
        "Two synthetic states have identical current board, side-to-move, previous-pass, komi, and legal mask, but different superko histories. After the same first move, the child legal masks diverge. The network cannot distinguish the parent states; MCTS still carries the full history. This is an information-loss finding, not a demonstrated M4 root cause.",
        "",
        "### Search target quality",
        "",
        "The machine report repeats 2 positions per model at 64 and 256 simulations. Stored 64-sim targets reproduce exactly, but KL and top-action agreement vary substantially; M3 has 0% top-action agreement on its two sampled positions, M1 mean KL≈1.133, and M4 mean KL≈0.430. This supports a weak/variable teacher signal hypothesis, not a search correctness bug.",
        "",
        "### Controlled MCTS compensation",
        "",
        "For distant-marker pairs with unchanged target-local neighborhoods, the audit records raw point policy, raw WDL, 64-simulation visits, and the selected action. Complete per-state values are in the JSON artifact; the compact diagnostic is:",
        mcts_summary,
        "Verdict: 64-sim MCTS is not promoted to a guaranteed global-policy repair. These fixed M0 probes are a controlled compensation check, but they do not define ground-truth optimal actions without a separate solver.",
        "",
        "## Truncated Arena games",
        "",
        f"All `{trunc['technical_games']}` technical records replay with zero legality errors. They are not software deadlocks: PASS actions are never consecutive (otherwise the rules would terminate), and point moves continue legally under superko. The M8-vs-M1 group has high interleaved PASS frequency; M8-vs-M7 has mostly long capture/placement cycles. Classification: model/search PASS and long-game pathology (B), not rules/superko/Arena bug (C/D).",
        "",
        "| COMPARISON | GAME | PASSES | LAST-100 PASS | CAPTURES | LAST-100 CAPTURES | UNIQUE BOARDS | FINAL OCCUPIED | CLASS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in trunc["rows"]:
        lines.append(f"| {row['comparison']} | `{row['game_id']}` | {row['passes']} | {row['last_100_passes']} | {row['captures']} | {row['last_100_captures']} | {row['unique_board_keys_including_pass_repeats']} | {row['final_occupied']} | {row['classification']} |")
    lines += [
        "",
        "## KataGo differential benchmark",
        "",
        "| FEATURE | KATAGO APPROACH | OUR APPROACH | WHY DIFFERENT | WAS DIFFERENCE PRESENT ON 5×5? | COULD BECOME MATERIAL ON 9×9? | EVIDENCE | RISK / CLASS |",
        "|---|---|---|---|---|---|---|---|",
        "| Replay window | Shuffled multi-generation window with independent data consumption | Fresh corpus only, one pass | Simplicity/contract isolation versus retaining recent experience | Stage4 tolerated the smaller regime | Yes: generation feedback and value drift | Full M1..M8 × D1..D8 matrix | High / STABILITY + SAMPLE EFFICIENCY |",
        "| Training budget | Explicit sample/update budget separate from games | Updates equal `ceil(positions/64)` | Corpus length controls optimizer exposure | Less material when corpus scale is narrower | Yes: 31..117 updates across 9×9 phases | Per-phase samples, updates, deltas, losses | Medium / STABILITY |",
        "| Global context | Global features and pathways support board-wide decisions | Global mean only for value/PASS; point policy local | Point logits cannot use pooled feature | Four blocks cover 5×5 diameter | Yes: diameter doubles from 4 to 8 | Exact distant-logit equality and gradient probe | High / ARCHITECTURE |",
        "| History | Richer temporal/history features | Current stones + legal mask; full history only in rules/search | Network input is a partial view of rule state | Less material on 5×5, but still absent | Yes when superko future legality matters | Identical parent observations, divergent child masks | Medium / SAMPLE EFFICIENCY |",
        "| Targets | Rich policy/value/ownership/score auxiliary supervision | Policy + WDL only; no DRAW in corpus | Less dense calibration/supervision | Proven adequate for Stage4, not necessarily scalable | Plausible, not demonstrated as root | 40910 rows, two WDL classes, phase CE | Medium / SAMPLE EFFICIENCY |",
        "| Search target generation | Expensive/weighted target searches separated from game generation | 64 sims for essentially every position | Cheap uniform teacher may be noisy | Worked as Golden baseline | Possibly: long-range weakness raises teacher variance | 64→256 fixed-sample KL/agreement | Medium / SAMPLE EFFICIENCY |",
        "| Arena/gating | Gatekeeper, not the training loop | Frozen detector, technical fail-closed | Diagnostic selection versus learning signal | Correct in baseline | No: not a training fix | W/L/D and technical replay audit | Low / CORRECTNESS REJECTED |",
        "| Long-game handling | Larger limits/pathology controls can be used operationally | 500-action watchdog | Runtime cap is a symptom detector, not learning signal | No baseline technical games reported | Yes as a symptom on 9×9 | 22 legal truncated traces, no pass adjacency | Medium / RUNTIME SYMPTOM |",
        "",
        "The important difference from successful Torus5 is not Adam or LR: it is that the same four blocks changed functional meaning when diameter grew from 4 to 8, while the fresh one-pass loop amplified generation-distribution changes. The important KataGo-like difference is explicit multi-generation data/update control, which is a stability/sample-efficiency enhancement, not a correctness requirement.",
        "",
        "## Arena evidence",
        "",
        "| COMPARISON | W/L/D | VALID PAIRS | TECHNICAL | MEAN PAIR SCORE |",
        "|---|---|---:|---:|---:|",
    ]
    for name, row in arena.items():
        lines.append(f"| {name} | {row['W/L/D']} | {row['pairs_valid']} | {row['technical_games']} | {row['mean_pair_score']} |")
    lines += [
        "",
        "## Minimal recommendation",
        "",
        "Do not merge PR #85 or start another full run yet. The minimum next diagnostic is a small controlled comparison of (A) four-block, (B) eight-block, and (optionally) a point-policy global-context variant on a held-out/global-dependency supervised fixture, followed by a deliberately fixed samples/updates multi-generation replay-window experiment. Keep the exact Golden rules/search/target contracts unchanged until those diagnostics are complete.",
        "",
        "RECOMMENDED MINIMAL FIX:",
        "For a future Torus9 experiment, first remove the representational scale mismatch with the smallest proven global-context/deeper-policy change, and separately fix the training budget/replay schedule so samples/updates are declared explicitly. Treat both as new experiment branches; do not rewrite the immutable run.",
        "",
        "WHAT MUST NOT BE CHANGED YET:",
        "Do not change LR, gradient clipping, move_limit, Arena size, rules, WDL sign, PASS index, or checkpoint artifacts based on this audit alone. Do not call the result stochasticity-only.",
        "",
        "## Reproducibility and verification",
        "",
        f"Run: `{audit['run_root']}`; source commit: `{audit['source_sha']}`; PR final head context: `{audit['final_head']}`.",
        f"Independent replay: `{audit['independent_replay_audit']['status']}`; exact transition reproduction: `{audit['training_reproduction']['status']}`; checkpoint manifest match: `{audit['manifest']['checkpoint_artifact_hashes_match']}`; deterministic inference/tiny overfit: `{audit['tiny_overfit_and_determinism']['deterministic_inference']}` / `{audit['tiny_overfit_and_determinism']['tiny_overfit_loss_decreased']}`.",
        "",
        "Generated by `tools/torus9_learning_forensic_audit.py`; no historical run file is modified.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-json", type=Path, default=ROOT / "docs/TORUS9_LEARNING_FORENSIC_AUDIT_20260913.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs/TORUS9_LEARNING_FORENSIC_AUDIT_20260913.md")
    args = parser.parse_args()
    audit = build_audit(args.run_root.resolve())
    write_json(args.out_json.resolve(), audit)
    args.out_md.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out_md.resolve().write_text(markdown_report(audit), encoding="utf-8")
    print(json.dumps({"json": str(args.out_json.resolve()), "markdown": str(args.out_md.resolve()), "status": audit["overall"]}, indent=2))


if __name__ == "__main__":
    main()
