"""Strict, reproducible symmetry audit for the frozen Torus 9x9 M12-B line.

The audit deliberately stops at inference/training-step equivalence.  It does
not start self-play, training iterations, or Arena games.  It uses the Golden
Torus9 model and the immutable D12-B replay produced by the existing M12-B
experiment.

Run from the repository root with::

    python3 tools/torus9_symmetry_audit.py

The output directory contains a machine-readable JSON result, a Markdown
report, and the exact state/transform manifest used by the run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CHECKPOINT = ROOT / "runs/torus9-alpha-score-ab/torus9-alpha-score-ab-20260913-v1/arms/B/checkpoints/M12-B.pt"
DEFAULT_REPLAY = ROOT / "runs/torus9-alpha-score-ab/torus9-alpha-score-ab-20260913-v1/selfplay/D12-B/replay.jsonl"
DEFAULT_GAMES = ROOT / "runs/torus9-alpha-score-ab/torus9-alpha-score-ab-20260913-v1/selfplay/D12-B/games.jsonl"
DEFAULT_OUTPUT = ROOT / "docs/torus9-symmetry-audit-20260913"

POINT_COUNT = 81
ACTION_COUNT = 82
PASS_INDEX = 81
BOARD_SIZE = 9
INFERENCE_TOLERANCE = 1.0e-5
TRAINING_TOLERANCE = 1.0e-5
TRAINING_BATCH_SIZE = 64


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _point(x: int, y: int) -> int:
    return (y % BOARD_SIZE) * BOARD_SIZE + (x % BOARD_SIZE)


def _coord(point: int) -> tuple[int, int]:
    return point % BOARD_SIZE, point // BOARD_SIZE


def _d4_map(d4_index: int, x: int, y: int) -> tuple[int, int]:
    """Apply one of the eight linear D4 automorphisms of Z9 x Z9."""

    reflected = d4_index >= 4
    rotation = d4_index % 4
    if reflected:
        x = -x
    for _ in range(rotation):
        x, y = -y, x
    return x, y


def d4_name(index: int) -> str:
    if index < 4:
        return f"rotation_{index * 90}"
    return f"reflection_then_rotation_{(index - 4) * 90}"


def make_permutation(d4_index: int, dx: int, dy: int) -> tuple[int, ...]:
    mapped = []
    for source in range(POINT_COUNT):
        x, y = _coord(source)
        x, y = _d4_map(d4_index, x, y)
        mapped.append(_point(x + dx, y + dy))
    return tuple(mapped)


def inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[int(target)] = int(source)
    return tuple(inverse)


def build_transformations() -> list[dict[str, object]]:
    transformations: list[dict[str, object]] = []
    seen: set[tuple[int, ...]] = set()
    for d4_index in range(8):
        for dx in range(BOARD_SIZE):
            for dy in range(BOARD_SIZE):
                permutation = make_permutation(d4_index, dx, dy)
                if permutation in seen:
                    raise AssertionError("D4 x Z9 x Z9 generated duplicate transformations")
                seen.add(permutation)
                if d4_index == 0:
                    family = "translations"
                elif dx == 0 and dy == 0:
                    family = "d4"
                else:
                    family = "combined"
                transformations.append({
                    "index": len(transformations),
                    "d4_index": d4_index,
                    "d4": d4_name(d4_index),
                    "dx": dx,
                    "dy": dy,
                    "family": family,
                    "permutation": list(permutation),
                })
    if len(transformations) != 648 or len(seen) != 648:
        raise AssertionError(f"Expected 648 unique Torus9 transformations, got {len(seen)}")
    return transformations


def check_topology(transformations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    from gocube_golden.topology import TORUS_9X9

    failures: list[dict[str, object]] = []
    permutation_failures = 0
    for transform in transformations:
        permutation = tuple(int(value) for value in transform["permutation"])
        if sorted(permutation) != list(range(POINT_COUNT)):
            permutation_failures += 1
            if len(failures) < 10:
                failures.append({"transform": transform["index"], "reason": "not_a_point_permutation"})
            continue
        for source, neighbors in enumerate(TORUS_9X9.adjacency):
            expected = {permutation[neighbor] for neighbor in neighbors}
            actual = set(TORUS_9X9.adjacency[permutation[source]])
            if expected != actual:
                permutation_failures += 1
                if len(failures) < 10:
                    failures.append({
                        "transform": transform["index"],
                        "source": source,
                        "reason": "adjacency_not_preserved",
                        "expected": sorted(expected),
                        "actual": sorted(actual),
                    })
                break
    return {
        "expected_transformations": 648,
        "unique_transformations": len({tuple(item["permutation"]) for item in transformations}),
        "point_count": POINT_COUNT,
        "adjacency_edges_checked": len(transformations) * POINT_COUNT,
        "permutation_or_adjacency_failures": permutation_failures,
        "failures": failures,
        "passed": permutation_failures == 0,
    }


def transform_point_axis(
    values: Sequence[object] | np.ndarray | torch.Tensor,
    permutation: Sequence[int],
    *,
    axis: int = -1,
):
    """Apply a source->destination point map, returning destination order."""

    inverse = inverse_permutation(permutation)
    if isinstance(values, torch.Tensor):
        normalized_axis = axis if axis >= 0 else values.ndim + axis
        if normalized_axis < 0 or normalized_axis >= values.ndim or values.shape[normalized_axis] != POINT_COUNT:
            raise ValueError("Expected a point-indexed tensor with 81 entries")
        return values.index_select(normalized_axis, torch.as_tensor(inverse, dtype=torch.long, device=values.device))
    array = np.asarray(values)
    normalized_axis = axis if axis >= 0 else array.ndim + axis
    if normalized_axis < 0 or normalized_axis >= array.ndim or array.shape[normalized_axis] != POINT_COUNT:
        raise ValueError("Expected a point-indexed array with 81 entries")
    return np.take(array, inverse, axis=normalized_axis)


def transform_vector(values: Sequence[object] | np.ndarray | torch.Tensor, permutation: Sequence[int]):
    return transform_point_axis(values, permutation, axis=-1)


def transform_policy(values: Sequence[object] | np.ndarray | torch.Tensor, permutation: Sequence[int]):
    array = values if isinstance(values, torch.Tensor) else np.asarray(values)
    if array.shape[-1] != ACTION_COUNT:
        raise ValueError("Expected a policy with 82 actions")
    point_values = transform_vector(array[..., :POINT_COUNT], permutation)
    if isinstance(array, torch.Tensor):
        return torch.cat((point_values, array[..., PASS_INDEX:PASS_INDEX + 1]), dim=-1)
    return np.concatenate((point_values, array[..., PASS_INDEX:PASS_INDEX + 1]), axis=-1)


def transform_board(board: Sequence[int], permutation: Sequence[int]) -> tuple[int, ...]:
    transformed = [0] * POINT_COUNT
    for source, value in enumerate(board):
        transformed[int(permutation[source])] = int(value)
    return tuple(transformed)


def transform_state(state, permutation: Sequence[int]):
    from gocube_golden.state import GoldenState, Stone

    history = tuple(transform_board(board, permutation) for board in state.superko_history)
    return GoldenState(
        stones=tuple(Stone(value) for value in transform_board(state.stones, permutation)),
        side_to_move=state.side_to_move,
        superko_history=history,
        consecutive_passes=state.consecutive_passes,
        topology=state.topology,
        rules_id=state.rules_id,
        rules_fingerprint=state.rules_fingerprint,
        komi=state.komi,
        history_provenance=state.history_provenance,
    )


def transform_action(action: int | str, permutation: Sequence[int]) -> int | str:
    return action if action == "PASS" else int(permutation[int(action)])


def transformed_mask(mask: Sequence[bool], permutation: Sequence[int]) -> tuple[bool, ...]:
    point_mask = transform_vector(np.asarray(mask[:POINT_COUNT], dtype=np.bool_), permutation).tolist()
    return tuple(bool(value) for value in point_mask) + (bool(mask[PASS_INDEX]),)


def state_identity(state) -> dict[str, object]:
    return {
        "stones": [int(value) for value in state.stones],
        "side_to_move": int(state.side_to_move),
        "superko_history": [list(board) for board in state.superko_history],
        "consecutive_passes": int(state.consecutive_passes),
        "topology_id": state.topology.topology_id,
        "topology_fingerprint": state.topology.fingerprint,
        "rules_id": state.rules_id,
        "rules_fingerprint": state.rules_fingerprint,
        "komi": float(state.komi),
        "history_provenance": state.history_provenance,
    }


def board_stone_count(state) -> int:
    return sum(int(value) != 0 for value in state.stones)


def score_winner(score) -> str:
    if float(score.margin_black) > 0.0:
        return "BLACK"
    if float(score.margin_black) < 0.0:
        return "WHITE"
    return "DRAW"


def load_game_candidates(games_path: Path) -> list[dict[str, object]]:
    """Read immutable game records and tag stage/history/capture strata."""

    from gocube_golden.torus9 import torus9_state_from_identity

    candidates: list[dict[str, object]] = []
    with games_path.open(encoding="utf-8") as handle:
        for game_line in handle:
            game = json.loads(game_line)
            positions = game.get("positions", [])
            states = [torus9_state_from_identity(position["state"]) for position in positions]
            capture_seen = False
            for index, (position, state) in enumerate(zip(positions, states)):
                action = position["selected_action"]
                if index + 1 < len(states) and action != "PASS":
                    # A legal point move that does not increase occupancy by
                    # exactly one captured at least one opposing stone.
                    capture_seen = capture_seen or board_stone_count(states[index + 1]) < board_stone_count(state) + 1
                tags: list[str] = []
                ply = int(position["ply"])
                if ply <= 4:
                    tags.append("opening")
                elif ply <= 16:
                    tags.append("early_middle")
                elif ply <= 40:
                    tags.append("middle")
                else:
                    tags.append("late")
                if capture_seen:
                    tags.append("captures")
                if int(state.consecutive_passes) == 1:
                    tags.append("previous_pass")
                if len(state.superko_history) >= 10:
                    tags.append("nontrivial_superko_history")
                candidates.append({
                    "game_id": str(game["game_id"]),
                    "ply": ply,
                    "tags": tags,
                    "state": state,
                    "selected_action": action,
                })
    return candidates


def select_states(candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    desired = (
        "opening",
        "early_middle",
        "middle",
        "late",
        "captures",
        "previous_pass",
        "nontrivial_superko_history",
        "late_capture",
    )
    selected: list[dict[str, object]] = []
    used: set[tuple[str, int]] = set()
    used_games: set[str] = set()
    for wanted in desired:
        matching = []
        for candidate in candidates:
            key = (str(candidate["game_id"]), int(candidate["ply"]))
            tags = set(candidate["tags"])
            matches = wanted in tags if wanted != "late_capture" else {"late", "captures"}.issubset(tags)
            if matches and key not in used:
                matching.append(candidate)
        if matching:
            # Prefer a new game for every stratum. This avoids making the
            # representative set look diverse while drawing every state from
            # the first game in a long immutable corpus.
            candidate = next(
                (item for item in matching if str(item["game_id"]) not in used_games),
                matching[0],
            )
            selected.append(dict(candidate))
            used.add((str(candidate["game_id"]), int(candidate["ply"])))
            used_games.add(str(candidate["game_id"]))
    required = {"opening", "early_middle", "middle", "late", "captures", "previous_pass", "nontrivial_superko_history"}
    observed = {tag for candidate in selected for tag in candidate["tags"]}
    missing = sorted(required - observed)
    if missing:
        raise RuntimeError(f"Immutable corpus did not contain required state strata: {missing}")
    return selected


def load_replay_rows(replay_path: Path, wanted: set[tuple[str, int]]) -> dict[tuple[str, int], dict[str, object]]:
    rows: dict[tuple[str, int], dict[str, object]] = {}
    with replay_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["game_id"]), int(row["ply"]))
            if key in wanted:
                rows[key] = row
    missing = wanted - set(rows)
    if missing:
        raise RuntimeError(f"Replay is missing selected state rows: {sorted(missing)[:5]}")
    return rows


def load_uniform_training_batch(replay_path: Path, row_count: int = TRAINING_BATCH_SIZE) -> list[dict[str, object]]:
    """Select one deterministic real batch spread over immutable D12-B rows."""

    with replay_path.open(encoding="utf-8") as handle:
        total_rows = sum(1 for _ in handle)
    if total_rows < row_count:
        raise RuntimeError(f"Replay has only {total_rows} rows, need {row_count}")
    targets = set(np.linspace(0, total_rows - 1, row_count, dtype=np.int64).tolist())
    rows: list[dict[str, object]] = []
    with replay_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in targets:
                row = json.loads(line)
                row["_replay_row_index"] = index
                rows.append(row)
    if len(rows) != row_count:
        raise RuntimeError(f"Expected {row_count} selected replay rows, got {len(rows)}")
    return rows


def validate_replay_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Validate stored labels and ensure their observation is state-derived."""

    from gocube_golden.torus9 import (
        build_torus9_observation,
        torus9_state_from_identity,
        validate_torus9_replay_sample,
    )

    for row in rows:
        validate_torus9_replay_sample(row)
        state = torus9_state_from_identity(row["state"])
        stored = torch.tensor(row["observation"], dtype=torch.float32)
        expected = build_torus9_observation(state)
        if not torch.equal(stored, expected):
            raise RuntimeError(
                f"Replay observation is not state-derived for {row['game_id']} ply {row['ply']}"
            )


def _error_stats(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"max_absolute_error": 0.0, "mean_absolute_error": 0.0, "p99_absolute_error": 0.0}
    return {
        "max_absolute_error": float(np.max(array)),
        "mean_absolute_error": float(np.mean(array)),
        "p99_absolute_error": float(np.quantile(array, 0.99)),
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    return _error_stats(tensor.detach().cpu().numpy())


def run_state_audit(states: Sequence[Mapping[str, object]], transformations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    from gocube_golden.rules import apply_action, prepare_legal_actions
    from gocube_golden.scoring import score_terminal
    from gocube_golden.state import PASS

    failures: list[dict[str, object]] = []
    transformed_count = 0
    legal_masks_checked = 0
    children_checked = 0
    terminal_scores_checked = 0
    target_transforms_checked = 0
    transition_transform_indices = {
        int(item["index"])
        for item in transformations
        if (int(item["d4_index"]), int(item["dx"]), int(item["dy"]))
        in {(0, 0, 0), (1, 0, 0), (4, 0, 0), (0, 1, 2), (5, 3, 4)}
    }
    for state_index, entry in enumerate(states):
        state = entry["state"]
        original_context = prepare_legal_actions(state)
        original_terminal = apply_action(state, PASS).after if state.consecutive_passes == 1 else apply_action(apply_action(state, PASS).after, PASS).after
        original_score = score_terminal(original_terminal)
        for transform in transformations:
            permutation = tuple(int(value) for value in transform["permutation"])
            transformed = transform_state(state, permutation)
            transformed_count += 1
            target_transforms_checked += 1
            if transformed.side_to_move != state.side_to_move or transformed.consecutive_passes != state.consecutive_passes or transformed.komi != state.komi:
                failures.append({"state": state_index, "transform": transform["index"], "reason": "scalar_semantics_changed"})
            if transformed.board_key != tuple(transform_board(state.board_key, permutation)):
                failures.append({"state": state_index, "transform": transform["index"], "reason": "current_board_not_transformed"})
            if tuple(transformed.superko_history) != tuple(transform_board(board, permutation) for board in state.superko_history):
                failures.append({"state": state_index, "transform": transform["index"], "reason": "superko_history_not_transformed"})
            transformed_context = prepare_legal_actions(transformed)
            expected_mask = transformed_mask(original_context.action_mask, permutation)
            legal_masks_checked += 1
            if transformed_context.action_mask != expected_mask:
                failures.append({"state": state_index, "transform": transform["index"], "reason": "legal_action_mask_mismatch"})
            if int(transform["index"]) in transition_transform_indices:
                for action in original_context.actions:
                    expected_action = transform_action(action, permutation)
                    try:
                        expected_child = transform_state(apply_action(state, action).after, permutation)
                        actual_child = apply_action(transformed, expected_action).after
                    except Exception as exc:
                        failures.append({"state": state_index, "transform": transform["index"], "action": action, "reason": "mapped_transition_error", "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    children_checked += 1
                    if actual_child != expected_child:
                        failures.append({"state": state_index, "transform": transform["index"], "action": action, "reason": "mapped_transition_mismatch"})
            transformed_terminal = apply_action(transformed, PASS).after if transformed.consecutive_passes == 1 else apply_action(apply_action(transformed, PASS).after, PASS).after
            transformed_score = score_terminal(transformed_terminal)
            terminal_scores_checked += 1
            if transformed_terminal != transform_state(original_terminal, permutation):
                failures.append({"state": state_index, "transform": transform["index"], "reason": "terminal_state_mismatch"})
            if score_winner(transformed_score) != score_winner(original_score) or not math.isclose(float(transformed_score.margin_black), float(original_score.margin_black), abs_tol=0.0):
                failures.append({"state": state_index, "transform": transform["index"], "reason": "score_or_winner_mismatch"})
            expected_ownership = tuple(original_score.ownership[int(inverse_permutation(permutation)[point])] for point in range(POINT_COUNT))
            if tuple(transformed_score.ownership) != expected_ownership:
                failures.append({"state": state_index, "transform": transform["index"], "reason": "ownership_target_mapping_mismatch"})
            if len(failures) > 50:
                break
        if len(failures) > 50:
            break
    return {
        "states_checked": len(states),
        "transformations_checked_per_state": len(transformations),
        "transformed_states_checked": transformed_count,
        "target_transforms_checked": target_transforms_checked,
        "legal_masks_checked": legal_masks_checked,
        "mapped_legal_children_checked": children_checked,
        "mapped_child_transformations_checked_per_state": len(transition_transform_indices),
        "terminal_scores_checked": terminal_scores_checked,
        "failures": failures[:50],
        "passed": not failures,
    }


def _model_and_optimizer(checkpoint: Path):
    from gocube_golden.torus9 import torus9_load_checkpoint, torus9_model_from_metadata

    metadata = json.loads(checkpoint.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    model = torus9_model_from_metadata(metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    torus9_load_checkpoint(checkpoint, model=model, optimizer=optimizer, expected={"model_hash": metadata["model_hash"]})
    return model, optimizer, metadata


def _forward_auxiliary(model, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model.forward_auxiliary(observations)
    if len(outputs) != 3:
        raise RuntimeError("M12-B must expose policy, WDL, and ownership heads")
    return outputs


def run_inference_audit(model, states: Sequence[Mapping[str, object]], transformations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    from gocube_golden.torus9 import build_torus9_observation

    model.eval()
    base_observations = torch.stack([build_torus9_observation(entry["state"]) for entry in states])
    with torch.inference_mode():
        base_policy, base_wdl, base_ownership = _forward_auxiliary(model, base_observations)
    base_policy = base_policy.detach().cpu()
    base_wdl = base_wdl.detach().cpu()
    base_ownership = base_ownership.detach().cpu()
    errors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    chunks = 256
    for start in range(0, len(states) * len(transformations), chunks):
        batch_entries = []
        keys = []
        for flat in range(start, min(start + chunks, len(states) * len(transformations))):
            state_index = flat // len(transformations)
            transform_index = flat % len(transformations)
            transform = transformations[transform_index]
            permutation = tuple(int(value) for value in transform["permutation"])
            batch_entries.append(transform_vector(base_observations[state_index], permutation))
            keys.append((state_index, transform_index))
        observations = torch.stack(batch_entries)
        with torch.inference_mode():
            policy, wdl, ownership = _forward_auxiliary(model, observations)
        policy = policy.detach().cpu()
        wdl = wdl.detach().cpu()
        ownership = ownership.detach().cpu()
        for row, (state_index, transform_index) in enumerate(keys):
            transform = transformations[transform_index]
            family = str(transform["family"])
            permutation = tuple(int(value) for value in transform["permutation"])
            aligned_policy = transform_policy(policy[row], inverse_permutation(permutation))
            aligned_ownership = transform_point_axis(ownership[row], inverse_permutation(permutation), axis=0)
            base_policy_row = base_policy[state_index]
            base_wdl_row = base_wdl[state_index]
            base_ownership_row = base_ownership[state_index]
            policy_prob = torch.softmax(policy[row], dim=-1)
            aligned_policy_prob = transform_policy(policy_prob, inverse_permutation(permutation))
            base_policy_prob = torch.softmax(base_policy_row, dim=-1)
            wdl_prob = torch.softmax(wdl[row], dim=-1)
            base_wdl_prob = torch.softmax(base_wdl_row, dim=-1)
            ownership_prob = torch.softmax(ownership[row], dim=-1)
            aligned_ownership_prob = transform_point_axis(ownership_prob, inverse_permutation(permutation), axis=0)
            base_ownership_prob = torch.softmax(base_ownership_row, dim=-1)
            errors[family]["policy_logits"].append(torch.abs(aligned_policy - base_policy_row).numpy())
            errors[family]["policy_probabilities"].append(torch.abs(aligned_policy_prob - base_policy_prob).numpy())
            errors[family]["wdl_logits"].append(torch.abs(wdl[row] - base_wdl_row).numpy())
            errors[family]["wdl_probabilities"].append(torch.abs(wdl_prob - base_wdl_prob).numpy())
            errors[family]["ownership_logits"].append(torch.abs(aligned_ownership - base_ownership_row).numpy())
            errors[family]["ownership_probabilities"].append(torch.abs(aligned_ownership_prob - base_ownership_prob).numpy())
    summaries: dict[str, object] = {}
    passed = True
    for family in ("d4", "translations", "combined"):
        summaries[family] = {}
        for head in ("policy_logits", "policy_probabilities", "wdl_logits", "wdl_probabilities", "ownership_logits", "ownership_probabilities"):
            flat = np.concatenate(errors[family][head]) if errors[family][head] else np.zeros(0, dtype=np.float32)
            summary = _error_stats(flat)
            summary["tolerance"] = INFERENCE_TOLERANCE
            summary["passed"] = summary["max_absolute_error"] <= INFERENCE_TOLERANCE
            summaries[family][head] = summary
            passed = passed and bool(summary["passed"])
    return {
        "states_checked": len(states),
        "transformations_checked_per_state": len(transformations),
        "raw_logits_checked_before_softmax": True,
        "pass_action_invariant_checked": True,
        "metrics": summaries,
        "passed": passed,
    }


def _batch_tensors(rows: Sequence[Mapping[str, object]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.tensor([row["observation"] for row in rows], dtype=torch.float32)
    legal_masks = torch.tensor([row["legal_action_mask"] for row in rows], dtype=torch.bool)
    policies = torch.tensor([row["pi"] for row in rows], dtype=torch.float32)
    values = torch.tensor([row["z"] for row in rows], dtype=torch.float32)
    ownership = torch.tensor([row["ownership_target"] for row in rows], dtype=torch.long)
    return observations, legal_masks, policies, values, ownership


def _loss_and_step(model, optimizer, observations, policies, values, ownership) -> dict[str, object]:
    model.train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    policy_logits, wdl_logits, ownership_logits = _forward_auxiliary(model, observations)
    policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    wdl_loss = -(values * F.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()
    ownership_loss = F.cross_entropy(ownership_logits.reshape(-1, 3), ownership.reshape(-1))
    total_loss = policy_loss + wdl_loss + ownership_loss
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    optimizer.step()
    deltas = {name: parameter.detach() - before[name] for name, parameter in model.named_parameters()}
    return {
        "losses": {
            "total_loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "wdl_loss": float(wdl_loss.detach()),
            "ownership_loss": float(ownership_loss.detach()),
        },
        "gradients": gradients,
        "parameter_deltas": deltas,
        "parameter_delta_norms": {name: float(torch.linalg.vector_norm(value.float())) for name, value in deltas.items()},
    }


def _state_tensor_snapshot(optimizer) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for index, parameter in enumerate(optimizer.param_groups[0]["params"]):
        for name, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                snapshot[f"{index}:{name}"] = value.detach().clone().cpu()
    return snapshot


def run_training_step_audit(checkpoint: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    model_a, optimizer_a, metadata_a = _model_and_optimizer(checkpoint)
    model_b, optimizer_b, metadata_b = _model_and_optimizer(checkpoint)
    if metadata_a["model_hash"] != metadata_b["model_hash"]:
        raise RuntimeError("M12-B copies did not load the same model hash")
    observations, legal_masks, policies, values, ownership = _batch_tensors(rows)
    transformed_observations = []
    transformed_legal_masks = []
    transformed_policies = []
    transformed_ownership = []
    transform_schedule = []
    schedule_specs = ((1, 0, 0), (4, 0, 0), (0, 1, 2), (5, 3, 4))
    transformations = build_transformations()
    transform_by_spec = {(int(item["d4_index"]), int(item["dx"]), int(item["dy"])): item for item in transformations}
    for index in range(len(rows)):
        d4_index, dx, dy = schedule_specs[index % len(schedule_specs)]
        transform = transform_by_spec[(d4_index, dx, dy)]
        permutation = tuple(int(value) for value in transform["permutation"])
        transformed_observations.append(transform_vector(observations[index], permutation))
        transformed_legal_masks.append(torch.as_tensor(transformed_mask(legal_masks[index].tolist(), permutation), dtype=torch.bool))
        transformed_policies.append(transform_policy(policies[index], permutation))
        transformed_ownership.append(transform_point_axis(ownership[index], permutation, axis=0))
        transform_schedule.append({"batch_row": index, "transform_index": transform["index"], "d4": transform["d4"], "dx": dx, "dy": dy, "family": transform["family"]})
    observations_b = torch.stack(transformed_observations)
    legal_masks_b = torch.stack(transformed_legal_masks)
    policies_b = torch.stack(transformed_policies)
    ownership_b = torch.stack(transformed_ownership)
    values_b = values.clone()
    if not torch.equal(values, values_b):
        raise AssertionError("WDL target unexpectedly changed")
    if not torch.equal(observations_b[:, 4, :], legal_masks_b[:, :POINT_COUNT].to(observations_b.dtype)):
        raise AssertionError("Transformed legal action mask disagrees with transformed observation")
    result_a = _loss_and_step(model_a, optimizer_a, observations, policies, values, ownership)
    result_b = _loss_and_step(model_b, optimizer_b, observations_b, policies_b, values_b, ownership_b)
    loss_differences = {name: abs(result_a["losses"][name] - result_b["losses"][name]) for name in result_a["losses"]}

    def compare_named_tensors(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, object]:
        diffs = [torch.abs(left[name].cpu() - right[name].cpu()).numpy() for name in left]
        flat = np.concatenate([value.reshape(-1) for value in diffs]) if diffs else np.zeros(0)
        summary = _error_stats(flat)
        summary["tolerance"] = TRAINING_TOLERANCE
        summary["passed"] = summary["max_absolute_error"] <= TRAINING_TOLERANCE
        return summary

    gradient_error = compare_named_tensors(result_a["gradients"], result_b["gradients"])
    parameter_delta_error = compare_named_tensors(result_a["parameter_deltas"], result_b["parameter_deltas"])
    adam_a = _state_tensor_snapshot(optimizer_a)
    adam_b = _state_tensor_snapshot(optimizer_b)
    adam_error = compare_named_tensors(adam_a, adam_b)
    passed = all(value <= TRAINING_TOLERANCE for value in loss_differences.values()) and bool(gradient_error["passed"]) and bool(parameter_delta_error["passed"]) and bool(adam_error["passed"])
    return {
        "checkpoint_label": metadata_a["checkpoint_label"],
        "checkpoint_model_hash": metadata_a["model_hash"],
        "batch_size": len(rows),
        "batch_source": "immutable D12-B replay.jsonl",
        "batch_row_indices": [int(row["_replay_row_index"]) for row in rows],
        "batch_row_ids": [f"{row['game_id']}@ply-{row['ply']}" for row in rows],
        "losses_model_a": result_a["losses"],
        "losses_model_b": result_b["losses"],
        "loss_absolute_differences": loss_differences,
        "gradient_error": gradient_error,
        "parameter_delta_error": parameter_delta_error,
        "parameter_delta_norms_model_a": result_a["parameter_delta_norms"],
        "parameter_delta_norms_model_b": result_b["parameter_delta_norms"],
        "adam_state_error": adam_error,
        "adam_state_entries_compared": len(adam_a),
        "transform_schedule": transform_schedule,
        "legal_action_masks_transformed": True,
        "wdl_target_unchanged": True,
        "loss_contract": "policy CE + WDL CE + ownership CE; score head absent from M12-B",
        "passed": passed,
    }


def pipeline_audit() -> dict[str, object]:
    coach = (ROOT / "alphazero/Coach.py").read_text(encoding="utf-8")
    training_common = (ROOT / "alphazero/envs/gocube/training_common.py").read_text(encoding="utf-8")
    golden_torus = (ROOT / "gocube_golden/torus9.py").read_text(encoding="utf-8")
    return {
        "torus9_production_training": {
            "entrypoint": "gocube_golden.torus9 / tools/torus9_alpha_score_ab.py",
            "geometric_augmentation": "absent",
            "rotations": False,
            "reflections": False,
            "toroidal_x_y_translations": False,
            "random_permutations": False,
            "applied_at_replay_save": False,
            "applied_at_batch_load": False,
            "applied_before_forward": False,
            "applied_nowhere": True,
            "evidence": {
                "torus9_trainer_uses_raw_observation_rows": "Torus9Trainer/Torus9OwnershipTrainer build tensors directly from sample['observation']",
                "golden_train_fixed_budget_has_no_transform_call": "gocube_golden/torus9.py",
                "production_alpha_score_runner_uses_trainer_directly": "tools/torus9_alpha_score_ab.py",
            },
        },
        "legacy_or_unrelated_code": {
            "Coach_DEFAULT_ARGS_symmetricSamples": "True",
            "production_build_base_training_args_symmetricSamples": "False",
            "legacy_flag_references": coach.count("symmetricSamples"),
            "v3_flag_references": training_common.count("symmetricSamples"),
            "cube_rotation_utility_present": True,
            "cube_rotation_path_is_torus9_training": False,
            "golden_stage4_5x5_torus_automorphism_diagnostic_present": True,
            "golden_stage4_path_is_torus9_training": False,
            "same_dataset_appended_twice_in_legacy_v3_window": "tensor dataset duplication/reuse, not spatial augmentation",
        },
        "answer": "Current Torus9 training has no geometric data augmentation; the generic symmetricSamples name is not a Torus9 symmetry implementation.",
    }


def make_state_manifest(states: Sequence[Mapping[str, object]], transformations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "states": [
            {
                "state_index": index,
                "game_id": entry["game_id"],
                "ply": entry["ply"],
                "tags": entry["tags"],
                "selected_action": entry["selected_action"],
                "state": state_identity(entry["state"]),
            }
            for index, entry in enumerate(states)
        ],
        "transformations": list(transformations),
        "state_manifest_fingerprint": sha256_json([
            {"game_id": entry["game_id"], "ply": entry["ply"], "tags": entry["tags"]}
            for entry in states
        ]),
        "transformation_manifest_fingerprint": sha256_json(transformations),
    }


def verdict(topology: Mapping[str, object], state: Mapping[str, object], inference: Mapping[str, object], training: Mapping[str, object]) -> str:
    if bool(topology["passed"]) and bool(state["passed"]) and bool(inference["passed"]) and bool(training["passed"]):
        return "REDUNDANT"
    return "NEEDS A/B"


def render_markdown(result: Mapping[str, object]) -> str:
    v = str(result["verdict"])
    topology = result["topology"]
    state = result["state_level"]
    inference = result["inference"]
    training = result["training_step"]
    lines = [
        f"SYMMETRY AUGMENTATION: {v}",
        "",
        "# Torus9 symmetry-equivariance audit",
        "",
        "This report is diagnostic-only. No self-play, two-iteration A/B, or Arena games were started.",
        "",
        "## Scope and provenance",
        "",
        f"- Frozen checkpoint: `{result['checkpoint']['path']}` ({result['checkpoint']['label']}, `{result['checkpoint']['model_hash']}`).",
        f"- Immutable corpus: `{result['corpus']['replay_path']}` ({result['corpus']['positions']} replay rows).",
        f"- Selected real states: {state['states_checked']}; selected training batch: {training['batch_size']} rows.",
        "- Parent semantics: Torus 9x9, WDL + ownership, komi 0.5, alpha 0.11, score OFF, 8 blocks / hidden 64.",
        "",
        "## 1. Training pipeline",
        "",
        result["pipeline"]["answer"],
        "",
        "The generic `symmetricSamples` spelling is not geometric augmentation here: the production Golden Torus9 trainer reads raw observation/target rows. The unrelated legacy/Cube/5x5 diagnostic utilities are not on this Torus9 path.",
        "",
        "## 2. Group and topology",
        "",
        f"Generated and checked `{topology['unique_transformations']}` unique transformations = D4 x Z9 x Z9 = 8 x 9 x 9.",
        f"Checked `{topology['adjacency_edges_checked']}` point-to-neighborhood mappings; failures: `{topology['permutation_or_adjacency_failures']}`.",
        "",
        "## 3. State-level correctness",
        "",
        f"Checked `{state['transformed_states_checked']}` transformed states, `{state['legal_masks_checked']}` legal masks, `{state['mapped_legal_children_checked']}` mapped legal transitions, and `{state['terminal_scores_checked']}` terminal score/winner/ownership results.",
        "",
        "The state transform carries the current board and every positional-superko history board. Side to move, pass count, komi, rules identity, and PASS semantics remain scalar/invariant. The network observation still does not encode the full history; that information boundary is separate from symmetry correctness.",
        "",
        "## 4. Frozen M12-B inference",
        "",
        "Errors below are absolute errors after inverse point-action/ownership permutation. Raw logits are compared before softmax, and probabilities are compared after softmax.",
        "",
        "| Family | Head | max | mean | p99 |",
        "|---|---|---:|---:|---:|",
    ]
    for family in ("d4", "translations", "combined"):
        for head, metrics in inference["metrics"][family].items():
            lines.append(f"| {family} | {head} | {metrics['max_absolute_error']:.3e} | {metrics['mean_absolute_error']:.3e} | {metrics['p99_absolute_error']:.3e} |")
    lines += [
        "",
        "## 5. Training-step equivalence",
        "",
        "Model A and Model B were loaded independently from the same M12-B checkpoint and Adam state. A used the original batch; B used the same 64 rows with a deterministic mix of rotation, reflection, translation, and combined D4+translation transforms. WDL targets stayed unchanged.",
        "",
        f"Loss absolute differences: `{json.dumps(training['loss_absolute_differences'], sort_keys=True)}`.",
        f"Gradient error max/mean/p99: `{training['gradient_error']['max_absolute_error']:.3e}` / `{training['gradient_error']['mean_absolute_error']:.3e}` / `{training['gradient_error']['p99_absolute_error']:.3e}`.",
        f"Parameter-delta error max/mean/p99: `{training['parameter_delta_error']['max_absolute_error']:.3e}` / `{training['parameter_delta_error']['mean_absolute_error']:.3e}` / `{training['parameter_delta_error']['p99_absolute_error']:.3e}`.",
        f"Adam-state error max/mean/p99: `{training['adam_state_error']['max_absolute_error']:.3e}` / `{training['adam_state_error']['mean_absolute_error']:.3e}` / `{training['adam_state_error']['p99_absolute_error']:.3e}` across `{training['adam_state_entries_compared']}` tensor entries.",
        "",
        "## Decision",
        "",
    ]
    if v == "REDUNDANT":
        lines += [
            "Topology, full state transformation, frozen inference, and one-step Adam training equivalence all passed within the declared float tolerance.",
            "",
            "Explicit symmetry augmentation is redundant for this Torus9 GraphNet. The architecture is equivariant to the stronger spatial automorphism group D4 x Z9 x Z9 (648 transformations), so transformed rows reproduce the same optimizer signal up to floating-point noise.",
            "",
            "No conditional A/B was run, as required.",
        ]
    else:
        lines += [
            "At least one audit contract did not pass. This result is evidence for investigating the exact failing contract before any A/B.",
            "",
            "No A/B was run automatically.",
        ]
    lines += ["", "## Artifacts", "", "- `audit.json`: complete machine-readable result.", "- `state-transform-manifest.json`: selected states and all 648 permutations.", "- `audit.md`: this report.", ""]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(1)
    started = time.time()
    for path in (args.checkpoint, args.replay, args.games):
        if not path.exists():
            raise FileNotFoundError(path)
    print("[1/6] Loading immutable corpus and selecting representative states", flush=True)
    candidates = load_game_candidates(args.games)
    selected = select_states(candidates)
    wanted = {(str(entry["game_id"]), int(entry["ply"])) for entry in selected}
    replay_rows = load_replay_rows(args.replay, wanted)
    selected = [dict(entry, replay_row=replay_rows[(str(entry["game_id"]), int(entry["ply"]))]) for entry in selected]
    training_rows = load_uniform_training_batch(args.replay)
    validate_replay_rows([entry["replay_row"] for entry in selected] + training_rows)
    print("[2/6] Building and checking D4 x Z9 x Z9", flush=True)
    transformations = build_transformations()
    topology = check_topology(transformations)
    print("[3/6] Checking full state/history/rule semantics", flush=True)
    state_result = run_state_audit(selected, transformations)
    print("[4/6] Loading frozen M12-B and auditing raw inference heads", flush=True)
    model, _, metadata = _model_and_optimizer(args.checkpoint)
    inference = run_inference_audit(model, selected, transformations)
    print("[5/6] Running one controlled training step on a real immutable batch", flush=True)
    training = run_training_step_audit(args.checkpoint, training_rows)
    pipeline = pipeline_audit()
    result = {
        "verdict": verdict(topology, state_result, inference, training),
        "audit_id": "torus9-symmetry-equivariance-v1",
        "started_utc_epoch": started,
        "duration_seconds": time.time() - started,
        "a_b_started": False,
        "self_play_started": False,
        "arena_started": False,
        "pipeline": pipeline,
        "group": {
            "name": "D4 x Z9 x Z9",
            "d4_transformations": 8,
            "translation_transformations": 81,
            "total_transformations": 648,
            "transform_families": {"d4": 8, "translations": 81, "combined": 648},
        },
        "checkpoint": {
            "label": metadata["checkpoint_label"],
            "path": str(args.checkpoint.resolve()),
            "artifact_sha256": sha256_file(args.checkpoint),
            "model_hash": metadata["model_hash"],
            "metadata": metadata,
        },
        "corpus": {
            "replay_path": str(args.replay.resolve()),
            "games_path": str(args.games.resolve()),
            "replay_sha256": sha256_file(args.replay),
            "games_sha256": sha256_file(args.games),
            "positions": len(candidates),
            "immutable_manifest": str(args.replay.with_name("manifest.json").resolve()),
        },
        "selected_states": [
            {"game_id": e["game_id"], "ply": e["ply"], "tags": e["tags"], "selected_action": e["selected_action"]}
            for e in selected
        ],
        "training_batch": {
            "size": len(training_rows),
            "row_indices": [int(row["_replay_row_index"]) for row in training_rows],
            "row_ids": [f"{row['game_id']}@ply-{row['ply']}" for row in training_rows],
            "corpus": str(args.replay.resolve()),
        },
        "topology": topology,
        "state_level": state_result,
        "inference": inference,
        "training_step": training,
        "tolerances": {"inference_absolute_error": INFERENCE_TOLERANCE, "training_absolute_error": TRAINING_TOLERANCE},
    }
    manifest = make_state_manifest(selected, transformations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "audit.json", result)
    write_json(args.output_dir / "state-transform-manifest.json", manifest)
    (args.output_dir / "audit.md").write_text(render_markdown(result), encoding="utf-8")
    print("[6/6] Wrote audit artifacts", flush=True)
    print(f"VERDICT: SYMMETRY AUGMENTATION: {result['verdict']}", flush=True)
    print(f"Artifacts: {args.output_dir}", flush=True)
    return 0 if result["verdict"] == "REDUNDANT" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AUDIT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
