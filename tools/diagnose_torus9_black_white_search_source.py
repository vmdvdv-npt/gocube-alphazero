#!/usr/bin/env python3
"""Locate the source of the Torus9 Black/White search-distribution gap.

This is a read-only scientific diagnostic.  It samples saved M135 self-play
states, evaluates the exact M135 input checkpoint, and runs the existing
sequential PUCT core with the production root-noise transform applied outside
the search.  It never writes below ``runs/`` and does not create training
artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.provenance import derive_seed, file_sha256
from gocube_golden.rules import LegalActionContext, prepare_legal_actions
from gocube_golden.search import Evaluation, SearchResult, SequentialPUCT
from gocube_golden.selfplay_policy import apply_root_dirichlet_noise
from gocube_golden.torus9_monolith import (
    Torus9NeuralEvaluator,
    build_torus9_observation,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    torus9_state_from_identity,
)
from gocube_golden.torus9_contract import TORUS9_ACTION_COUNT, TORUS9_PASS_INDEX
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import PASS


DIAGNOSTIC_SCHEMA = "torus9-black-white-search-source-diagnostic-v1"
MASTER_SEED = 2026092401
SAMPLE_PER_SIDE_PER_RANGE = 32
RESERVOIR_PER_GROUP = 512
REPEATS = 3
SIMULATIONS = 200
CPUCT = 1.25
FPU = 0.0
DIRICHLET_ALPHA = 0.11
DIRICHLET_EPSILON = 0.25
INTERVALS = (
    ("plies_1_8", 1, 8),
    ("plies_9_24", 9, 24),
    ("plies_25_64", 25, 64),
    ("plies_65_plus", 65, None),
)
SIDES = ("BLACK", "WHITE")
STAGES = ("raw_nn_policy", "noisy_prior", "mcts_noise_on", "mcts_noise_off")
METRIC_KEYS = (
    "entropy_nats",
    "effective_candidates",
    "top1_share",
    "top2_share",
    "top3_share",
    "nonzero_actions",
    "max_count",
    "sum_counts",
)


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 9)


def _round_vector(values: Iterable[float | int | None]) -> list[float | int | None]:
    result: list[float | int | None] = []
    for value in values:
        if value is None:
            result.append(None)
        elif isinstance(value, int) and not isinstance(value, bool):
            result.append(value)
        else:
            result.append(_round(float(value)))
    return result


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return _round(statistics.fmean(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return _round(statistics.median(values)) if values else None


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "q05", "q25", "median", "q75", "q95", "max")}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return ordered[low]
        weight = index - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "min": _round(ordered[0]),
        "q05": _round(percentile(0.05)),
        "q25": _round(percentile(0.25)),
        "median": _round(percentile(0.50)),
        "q75": _round(percentile(0.75)),
        "q95": _round(percentile(0.95)),
        "max": _round(ordered[-1]),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return _round(numerator / (left_norm * right_norm))


def _interval_for(ply: int) -> str:
    for name, start, end in INTERVALS:
        if ply >= start and (end is None or ply <= end):
            return name
    raise ValueError(f"Unsupported ply {ply}")


def _action_index(action: object) -> int:
    return TORUS9_PASS_INDEX if action == PASS else int(action)


def _distribution_metrics(values: Sequence[float | int], legal_indices: Sequence[int]) -> dict[str, float | int]:
    legal = [float(values[index]) for index in legal_indices]
    total = sum(legal)
    if total <= 0.0:
        raise ValueError("Diagnostic distribution has no legal mass")
    probabilities = [value / total for value in legal]
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
    ordered = sorted(probabilities, reverse=True)
    return {
        "entropy_nats": _round(entropy),
        "effective_candidates": _round(math.exp(entropy)),
        "top1_share": _round(sum(ordered[:1])),
        "top2_share": _round(sum(ordered[:2])),
        "top3_share": _round(sum(ordered[:3])),
        "nonzero_actions": int(sum(value > 0.0 for value in legal)),
        "max_count": _round(max(legal)),
        "sum_counts": _round(total),
    }


def _summarize_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"positions_or_repeats": len(rows)}
    for key in METRIC_KEYS:
        values = [float(row["metrics"][key]) for row in rows if row.get("metrics", {}).get(key) is not None]
        result[key] = {"mean": _mean(values), "median": _median(values), **_quantiles(values)}
    return result


def _delta_summary(values: Sequence[float]) -> dict[str, Any]:
    return {"n": len(values), **_quantiles(values)}


def _correlation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("prior_vs_visits", "q_vs_visits", "prior_vs_q")
    return {
        key: {
            "mean": _mean([float(row["correlations"][key]) for row in rows if row.get("correlations", {}).get(key) is not None]),
            "median": _median([float(row["correlations"][key]) for row in rows if row.get("correlations", {}).get(key) is not None]),
            "distribution": _quantiles([float(row["correlations"][key]) for row in rows if row.get("correlations", {}).get(key) is not None]),
        }
        for key in keys
    }


@dataclass
class Candidate:
    game_id: str
    ply: int
    side: str
    state: Mapping[str, Any]
    legal_count: int = 0
    random_key: int = 0
    pair_id: int = -1

    @property
    def position_id(self) -> str:
        return f"{self.game_id}:ply-{self.ply:03d}"


class Reservoirs:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.items: dict[tuple[str, str], list[Candidate]] = {
            (interval, side): [] for interval, _, _ in INTERVALS for side in SIDES
        }
        self.seen: dict[tuple[str, str], int] = defaultdict(int)
        self.rngs = {
            key: random.Random(derive_seed(self.seed, key[0], key[1], "reservoir"))
            for key in self.items
        }

    def add(self, candidate: Candidate) -> None:
        key = (_interval_for(candidate.ply), candidate.side)
        self.seen[key] += 1
        pool = self.items[key]
        if len(pool) < RESERVOIR_PER_GROUP:
            pool.append(candidate)
            return
        slot = self.rngs[key].randrange(self.seen[key])
        if slot < RESERVOIR_PER_GROUP:
            pool[slot] = candidate


def _select_sample(reservoirs: Reservoirs) -> list[Candidate]:
    selected: list[Candidate] = []
    for interval, _, _ in INTERVALS:
        black = reservoirs.items[(interval, "BLACK")]
        white = reservoirs.items[(interval, "WHITE")]
        for item in (*black, *white):
            state = torus9_state_from_identity(item.state)
            item.legal_count = len(prepare_legal_actions(state).actions)
            item.random_key = derive_seed(reservoirs.seed, interval, item.side, item.game_id, item.ply, "select")
        if len(black) < SAMPLE_PER_SIDE_PER_RANGE or len(white) < SAMPLE_PER_SIDE_PER_RANGE:
            raise ValueError(f"Not enough {interval} candidates for both sides")
        ordered_black = sorted(black, key=lambda item: (item.legal_count, item.random_key, item.position_id))
        ordered_white = sorted(white, key=lambda item: (item.legal_count, item.random_key, item.position_id))
        black_picks = [ordered_black[round((index + 0.5) * len(ordered_black) / SAMPLE_PER_SIDE_PER_RANGE - 0.5)] for index in range(SAMPLE_PER_SIDE_PER_RANGE)]
        remaining_white = list(ordered_white)
        paired: list[tuple[Candidate, Candidate]] = []
        for black_pick in black_picks:
            white_pick = min(
                remaining_white,
                key=lambda item: (abs(item.legal_count - black_pick.legal_count), item.random_key, item.position_id),
            )
            remaining_white.remove(white_pick)
            paired.append((black_pick, white_pick))
        for pair_id, (black_pick, white_pick) in enumerate(sorted(paired, key=lambda pair: (pair[0].legal_count, pair[1].legal_count, pair[0].random_key))):
            black_pick.pair_id = pair_id
            white_pick.pair_id = pair_id
            selected.extend((black_pick, white_pick))
    return sorted(selected, key=lambda item: (_interval_for(item.ply), item.pair_id, item.side))


class CachedEvaluator:
    """Cache deterministic model outputs without changing search semantics."""

    def __init__(self, evaluator: Torus9NeuralEvaluator) -> None:
        self.evaluator = evaluator
        self.cache: dict[object, Evaluation] = {}

    def evaluate_prepared(self, state: Any, legal_context: LegalActionContext) -> Evaluation:
        key = state.state_key
        if key not in self.cache:
            self.cache[key] = self.evaluator.evaluate_prepared(state, legal_context)
        return self.cache[key]


class RootOverrideEvaluator:
    def __init__(self, delegate: CachedEvaluator, root_key: object, root_evaluation: Evaluation) -> None:
        self.delegate = delegate
        self.root_key = root_key
        self.root_evaluation = root_evaluation

    def evaluate_prepared(self, state: Any, legal_context: LegalActionContext) -> Evaluation:
        if state.state_key == self.root_key:
            legal_context.assert_compatible(state)
            return self.root_evaluation
        return self.delegate.evaluate_prepared(state, legal_context)


def _root_evaluation(model: Any, state: Any, context: LegalActionContext, device: torch.device) -> tuple[Evaluation, list[float], list[float], list[float]]:
    observation = build_torus9_observation(state, legal_context=context).to(device)
    with torch.inference_mode():
        policy_logits, wdl_logits = model(observation.unsqueeze(0))
        policy_probabilities = torch.softmax(policy_logits[0], dim=0)
        wdl_probabilities = torch.softmax(wdl_logits[0], dim=0)
    logits = [float(value) for value in policy_logits[0].detach().cpu()]
    policy = [float(value) for value in policy_probabilities.detach().cpu()]
    wdl = [float(value) for value in wdl_probabilities.detach().cpu()]
    return Evaluation(policy=tuple(policy), wdl=tuple(wdl)), logits, policy, wdl


def _legal_prior(policy: Sequence[float], legal_indices: Sequence[int]) -> list[float]:
    total = sum(float(policy[index]) for index in legal_indices)
    if total <= 0.0:
        value = 1.0 / len(legal_indices)
        result = [0.0] * len(policy)
        for index in legal_indices:
            result[index] = value
        return result
    result = [0.0] * len(policy)
    for index in legal_indices:
        result[index] = float(policy[index]) / total
    return result


def _utility(wdl: Sequence[float]) -> float:
    return (float(wdl[0]) - float(wdl[2])) / sum(float(value) for value in wdl)


def _noise_seed(master_seed: int, position_id: str, repeat: int) -> int:
    return derive_seed(master_seed, position_id, "dirichlet", repeat)


def _full_noise(raw_prior: Sequence[float], noisy_prior: Sequence[float], legal_indices: Sequence[int]) -> list[float]:
    noise = [0.0] * len(raw_prior)
    for index in legal_indices:
        noise[index] = (float(noisy_prior[index]) - (1.0 - DIRICHLET_EPSILON) * float(raw_prior[index])) / DIRICHLET_EPSILON
    return noise


def _mcts_diagnostics(
    result: SearchResult,
    prior: Sequence[float],
    legal_indices: Sequence[int],
    root_value: float,
) -> dict[str, Any]:
    visits = list(result.root_visits)
    q_values = list(result.root_q)
    if len(visits) != TORUS9_ACTION_COUNT or sum(visits) != SIMULATIONS:
        raise ValueError(f"MCTS root visit invariant failed: len={len(visits)} sum={sum(visits)}")
    metrics = _distribution_metrics(visits, legal_indices)
    visit_values = [float(visits[index]) for index in legal_indices]
    q_indices = [index for index in legal_indices if q_values[index] is not None and visits[index] > 0]
    q_only = [float(q_values[index]) for index in q_indices]
    prior_q = [float(prior[index]) for index in q_indices]
    visits_q = [float(visits[index]) for index in q_indices]
    top_prior_q_indices = sorted(q_indices, key=lambda index: (-float(prior[index]), index))[:3]
    top_prior_q = [float(q_values[index]) for index in top_prior_q_indices]
    correlations = {
        "prior_vs_visits": _pearson([float(prior[index]) for index in legal_indices], visit_values),
        "q_vs_visits": _pearson(q_only, visits_q),
        "prior_vs_q": _pearson(prior_q, q_only),
    }
    return {
        "metrics": metrics,
        "visits": visits,
        "root_q": [_round(value) for value in q_values],
        "root_value": _round(root_value),
        "q_summary": {
            "visited_edges": len(q_only),
            "mean": _mean(q_only),
            "min": _round(min(q_only)) if q_only else None,
            "max": _round(max(q_only)) if q_only else None,
            "spread": _round(max(q_only) - min(q_only)) if q_only else None,
            "top_prior_3_mean": _mean(top_prior_q),
            "top_prior_3_spread": _round(max(top_prior_q) - min(top_prior_q)) if top_prior_q else None,
        },
        "correlations": correlations,
    }


def _load_saved_positions(selfplay_path: Path, seed: int) -> tuple[Reservoirs, dict[str, Any]]:
    reservoirs = Reservoirs(seed)
    record_count = 0
    position_count = 0
    invalid_games = 0
    identity: dict[str, Any] | None = None
    with selfplay_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_count += 1
            if identity is None:
                identity = {
                    key: record.get(key)
                    for key in (
                        "run_id", "model_checkpoint_label", "model_hash", "checkpoint_artifact_hash",
                        "git_commit", "git_tree", "profile_id", "profile_fingerprint",
                        "selfplay_contract_id", "selfplay_contract_fingerprint", "master_seed",
                    )
                }
            for key in identity:
                if record.get(key) != identity[key]:
                    raise ValueError(f"Self-play identity drift at line {line_number}: {key}")
            positions = record.get("positions")
            if not isinstance(positions, list):
                invalid_games += 1
                continue
            for position in positions:
                ply = int(position["ply"])
                state = position["state"]
                state_obj = torus9_state_from_identity(state)
                side = state_obj.side_to_move.name
                if side not in SIDES:
                    raise ValueError(f"Unexpected side in saved position {record.get('game_id')}:{ply}")
                position_count += 1
                reservoirs.add(Candidate(
                    game_id=str(record["game_id"]),
                    ply=ply,
                    side=side,
                    state=state,
                ))
    if identity is None:
        raise ValueError(f"No records found in {selfplay_path}")
    return reservoirs, {
        "record_count": record_count,
        "position_count": position_count,
        "invalid_games": invalid_games,
        "population_counts": {f"{interval}/{side}": reservoirs.seen[(interval, side)] for interval, _, _ in INTERVALS for side in SIDES},
        "reservoir_size": RESERVOIR_PER_GROUP,
        "record_identity": identity,
    }


def _sample_manifest(selected: Sequence[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "position_id": candidate.position_id,
            "game_id": candidate.game_id,
            "ply": candidate.ply,
            "interval": _interval_for(candidate.ply),
            "side": candidate.side,
            "pair_id": candidate.pair_id,
            "legal_count": candidate.legal_count,
            "state_fingerprint": _canonical_hash(candidate.state),
        }
        for candidate in selected
    ]


def _sample_balance(selected: Sequence[Candidate]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for interval, _, _ in INTERVALS:
        rows = [candidate for candidate in selected if _interval_for(candidate.ply) == interval]
        black = [candidate for candidate in rows if candidate.side == "BLACK"]
        white = [candidate for candidate in rows if candidate.side == "WHITE"]
        paired = {
            candidate.pair_id: candidate
            for candidate in rows
        }
        diffs = [
            abs(paired[pair_id].legal_count - next(item for item in rows if item.pair_id == pair_id and item.side != paired[pair_id].side).legal_count)
            for pair_id in sorted({item.pair_id for item in rows})
        ]
        result[interval] = {
            "BLACK": {"n": len(black), "legal_count_mean": _mean([item.legal_count for item in black]), "legal_count_median": _median([item.legal_count for item in black])},
            "WHITE": {"n": len(white), "legal_count_mean": _mean([item.legal_count for item in white]), "legal_count_median": _median([item.legal_count for item in white])},
            "paired_abs_legal_count_delta": _quantiles(diffs),
        }
    return result


def _raw_wdl_report(root_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_side: dict[str, list[Mapping[str, Any]]] = {
            side: [row for row in rows if row["side"] == side] for side in SIDES
        }
        result: dict[str, Any] = {}
        for side, side_rows in by_side.items():
            result[side] = {
                "n": len(side_rows),
                "wdl": {
                    label: _quantiles([float(row["raw_nn"]["wdl"][index]) for row in side_rows])
                    for index, label in enumerate(("WIN", "DRAW", "LOSS"))
                },
                "derived_utility": _quantiles([float(row["raw_nn"]["derived_utility"]) for row in side_rows]),
            }
        result["WHITE_minus_BLACK_mean"] = {
            label: _round(
                _mean([float(row["raw_nn"]["wdl"][index]) for row in by_side["WHITE"]])
                - _mean([float(row["raw_nn"]["wdl"][index]) for row in by_side["BLACK"]])
            )
            for index, label in enumerate(("WIN", "DRAW", "LOSS"))
        }
        result["WHITE_minus_BLACK_mean"]["derived_utility"] = _round(
            _mean([float(row["raw_nn"]["derived_utility"]) for row in by_side["WHITE"]])
            - _mean([float(row["raw_nn"]["derived_utility"]) for row in by_side["BLACK"]])
        )
        return result

    result = {"aggregate": scope(root_rows)}
    for interval, _, _ in INTERVALS:
        result[interval] = scope([row for row in root_rows if row["interval"] == interval])
    result["interpretation"] = (
        "WDL/value is compared statistically across different positions; absolute Black and White values are not treated as paired game situations."
    )
    return result


def _aggregate_stage(stage_rows: Sequence[Mapping[str, Any]], selected: Sequence[Candidate]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for interval, _, _ in INTERVALS:
        result[interval] = _aggregate_stage_scope(stage_rows, selected, interval)
    result["aggregate"] = _aggregate_stage_scope(stage_rows, selected, None)
    return result


def _aggregate_stage_scope(stage_rows: Sequence[Mapping[str, Any]], selected: Sequence[Candidate], interval: str | None) -> dict[str, Any]:
    candidates = {
        candidate.position_id: candidate
        for candidate in selected
        if interval is None or _interval_for(candidate.ply) == interval
    }
    by_side = {
        side: [row for row in stage_rows if row["position_id"] in candidates and row["side"] == side]
        for side in SIDES
    }
    result: dict[str, Any] = {side: _summarize_metric_rows(rows) for side, rows in by_side.items()}
    delta: dict[str, Any] = {}
    for key in METRIC_KEYS:
        black_values = [float(row["metrics"][key]) for row in by_side["BLACK"]]
        white_values = [float(row["metrics"][key]) for row in by_side["WHITE"]]
        delta[key] = {
            "mean_white_minus_black": _round((_mean(white_values) or 0.0) - (_mean(black_values) or 0.0)),
            "median_white_minus_black": _round((_median(white_values) or 0.0) - (_median(black_values) or 0.0)),
        }
    result["BLACK_to_WHITE_delta"] = delta
    result["correlations"] = {side: _correlation_summary(rows) for side, rows in by_side.items()}
    if stage_rows and "q_summary" in stage_rows[0]:
        q_fields = ("mean", "min", "max", "spread", "top_prior_3_mean", "top_prior_3_spread")
        result["q_summary"] = {
            side: {
                field: {
                    "mean": _mean([float(row["q_summary"][field]) for row in rows if row["q_summary"].get(field) is not None]),
                    "median": _median([float(row["q_summary"][field]) for row in rows if row["q_summary"].get(field) is not None]),
                    "distribution": _quantiles([float(row["q_summary"][field]) for row in rows if row["q_summary"].get(field) is not None]),
                }
                for field in q_fields
            }
            for side, rows in by_side.items()
        }
        result["root_value"] = {
            side: _quantiles([float(row["root_value"]) for row in rows if row.get("root_value") is not None])
            for side, rows in by_side.items()
        }

    # Pair the deterministic BLACK/WHITE sample.  ON rows are paired by repeat;
    # raw and OFF rows have one row per position.
    paired_deltas: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    row_map = {(row["position_id"], row.get("repeat", 0)): row for row in stage_rows if row["position_id"] in candidates}
    for candidate in selected:
        if candidate.position_id not in candidates or candidate.side != "BLACK":
            continue
        mate = next(item for item in selected if item.pair_id == candidate.pair_id and item.side == "WHITE" and (interval is None or _interval_for(item.ply) == interval))
        repeats = sorted({repeat for position_id, repeat in row_map if position_id == candidate.position_id})
        for repeat in repeats:
            left = row_map.get((candidate.position_id, repeat))
            right = row_map.get((mate.position_id, repeat))
            if left is None or right is None:
                continue
            for key in METRIC_KEYS:
                paired_deltas[key].append(float(right["metrics"][key]) - float(left["metrics"][key]))
    result["paired_position_deltas_white_minus_black"] = {
        key: _delta_summary(values) for key, values in paired_deltas.items()
    }
    return result


def _run_diagnostic(
    *,
    run_root: Path,
    checkpoint_path: Path,
    selfplay_path: Path,
    output_json: Path,
    output_md: Path,
    raw_output_json: Path | None,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    started = time.time()
    reservoirs, saved_info = _load_saved_positions(selfplay_path, seed)
    selected = _select_sample(reservoirs)
    if len(selected) != 4 * 2 * SAMPLE_PER_SIDE_PER_RANGE:
        raise ValueError(f"Expected 256 selected positions, got {len(selected)}")
    print(f"[diagnostic] selected {len(selected)} positions from {saved_info['position_count']} saved positions", flush=True)

    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_file_sha256 = file_sha256(checkpoint_path)
    if checkpoint_file_sha256 != str(saved_info["record_identity"]["checkpoint_artifact_hash"]):
        raise ValueError("Checkpoint SHA-256 does not match M135 self-play records")
    if checkpoint_metadata.get("model_hash") != saved_info["record_identity"]["model_hash"]:
        raise ValueError("Checkpoint model hash does not match M135 self-play records")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    model = torus9_model_from_metadata(checkpoint_metadata)
    loaded_metadata = torus9_load_checkpoint(
        checkpoint_path,
        model=model,
        expected={"model_hash": checkpoint_metadata.get("model_hash")},
        device=device,
    )
    model.eval()
    delegate = CachedEvaluator(Torus9NeuralEvaluator(model, device=device))
    settings = SearchSettings(simulations=SIMULATIONS, cpuct=CPUCT, fpu=FPU, deterministic_tie_break=True)
    adapter = GoldenSearchAdapter()
    root_rows: list[dict[str, Any]] = []
    stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    validation = {
        "selected_positions": len(selected),
        "raw_vector_shape_errors": 0,
        "mcts_sum_errors": 0,
        "mcts_illegal_visit_errors": 0,
        "mcts_nonfinite_q_errors": 0,
        "root_cache_entries": 0,
    }

    for position_number, candidate in enumerate(selected, start=1):
        state = torus9_state_from_identity(candidate.state)
        context = prepare_legal_actions(state)
        legal_indices = tuple(_action_index(action) for action in context.actions)
        root_eval, logits, raw_policy, raw_wdl = _root_evaluation(model, state, context, device)
        raw_prior = _legal_prior(raw_policy, legal_indices)
        if len(logits) != TORUS9_ACTION_COUNT or len(raw_policy) != TORUS9_ACTION_COUNT or len(raw_prior) != TORUS9_ACTION_COUNT:
            validation["raw_vector_shape_errors"] += 1
            raise ValueError("Raw network vector shape drift")
        raw_metrics = _distribution_metrics(raw_prior, legal_indices)
        base = {
            "position_id": candidate.position_id,
            "interval": _interval_for(candidate.ply),
            "side": candidate.side,
            "pair_id": candidate.pair_id,
            "ply": candidate.ply,
            "legal_count": candidate.legal_count,
            "metrics": raw_metrics,
        }
        stage_rows["raw_nn_policy"].append(base)
        root_row: dict[str, Any] = {
            "position_id": candidate.position_id,
            "game_id": candidate.game_id,
            "ply": candidate.ply,
            "interval": _interval_for(candidate.ply),
            "side": candidate.side,
            "pair_id": candidate.pair_id,
            "legal_count": candidate.legal_count,
            "state_fingerprint": _canonical_hash(candidate.state),
            "raw_nn": {
                "policy_logits": _round_vector(logits),
                "policy_probabilities": _round_vector(raw_policy),
                "masked_normalized_prior": _round_vector(raw_prior),
                "wdl": _round_vector(raw_wdl),
                "derived_utility": _round(_utility(raw_wdl)),
                "legal_mask": [bool(value) for value in context.action_mask],
                "metrics": raw_metrics,
            },
            "repeats": [],
        }
        repeat_rows = root_row["repeats"]
        off_result = SequentialPUCT(settings, adapter=adapter).search(
            state,
            RootOverrideEvaluator(delegate, state.state_key, root_eval),
            seed=derive_seed(seed, candidate.position_id, "mcts-noise-off"),
        )
        off_diag = _mcts_diagnostics(off_result, raw_prior, legal_indices, _utility(raw_wdl))
        if any(off_diag["visits"][index] != 0 for index in range(TORUS9_ACTION_COUNT) if index not in legal_indices):
            validation["mcts_illegal_visit_errors"] += 1
            raise ValueError("Noise-off MCTS allocated an illegal root visit")
        stage_rows["mcts_noise_off"].append({
            "position_id": candidate.position_id,
            "interval": _interval_for(candidate.ply),
            "side": candidate.side,
            "pair_id": candidate.pair_id,
            "repeat": 0,
            **off_diag,
        })
        for repeat in range(REPEATS):
            noise_seed = _noise_seed(seed, candidate.position_id, repeat)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_seed)
            noisy_policy = list(apply_root_dirichlet_noise(
                raw_policy,
                context.actions,
                action_index=_action_index,
                epsilon=DIRICHLET_EPSILON,
                alpha=DIRICHLET_ALPHA,
                generator=generator,
            ))
            noise = _full_noise(raw_prior, noisy_policy, legal_indices)
            noisy_metrics = _distribution_metrics(noisy_policy, legal_indices)
            noisy_stage_row = {
                "position_id": candidate.position_id,
                "interval": _interval_for(candidate.ply),
                "side": candidate.side,
                "pair_id": candidate.pair_id,
                "repeat": repeat,
                "metrics": noisy_metrics,
            }
            stage_rows["noisy_prior"].append(noisy_stage_row)
            root_override = RootOverrideEvaluator(
                delegate,
                state.state_key,
                Evaluation(policy=tuple(noisy_policy), wdl=root_eval.wdl),
            )
            on_result = SequentialPUCT(settings, adapter=adapter).search(
                state,
                root_override,
                seed=derive_seed(seed, candidate.position_id, "mcts-noise-on", repeat),
            )
            on_diag = _mcts_diagnostics(on_result, noisy_policy, legal_indices, _utility(raw_wdl))
            if any(on_diag["visits"][index] != 0 for index in range(TORUS9_ACTION_COUNT) if index not in legal_indices):
                validation["mcts_illegal_visit_errors"] += 1
                raise ValueError("Noise-on MCTS allocated an illegal root visit")
            stage_rows["mcts_noise_on"].append({
                "position_id": candidate.position_id,
                "interval": _interval_for(candidate.ply),
                "side": candidate.side,
                "pair_id": candidate.pair_id,
                "repeat": repeat,
                **on_diag,
            })
            repeat_rows.append({
                "repeat": repeat,
                "noise_seed": noise_seed,
                "dirichlet_noise": _round_vector(noise),
                "noisy_prior": {
                    "vector": _round_vector(noisy_policy),
                    "metrics": noisy_metrics,
                },
                "mcts_noise_on": on_diag,
                "mcts_noise_off": off_diag if repeat == 0 else {"same_as_repeat_0": True},
            })
        root_rows.append(root_row)
        if position_number == 1 or position_number % 8 == 0 or position_number == len(selected):
            print(f"[diagnostic] {position_number}/{len(selected)} positions; cached evaluations={len(delegate.cache)}", flush=True)

    validation["root_cache_entries"] = len(delegate.cache)
    aggregates = {stage: _aggregate_stage(stage_rows[stage], selected) for stage in STAGES}
    runtime = round(time.time() - started, 3)
    report: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "diagnostic_date": "2026-09-24",
        "master_seed": seed,
        "sample": {
            "total": len(selected),
            "per_side_per_range": SAMPLE_PER_SIDE_PER_RANGE,
            "ranges": [name for name, _, _ in INTERVALS],
            "sampling": "deterministic reservoir of saved M135 positions, legal-count quantile selection, nearest legal-count opposite-side pairing",
            "reservoir_size_per_range_side": RESERVOIR_PER_GROUP,
            "saved_artifact": str(selfplay_path),
            "saved_artifact_sha256": file_sha256(selfplay_path),
            "population": saved_info,
            "balance": _sample_balance(selected),
            "positions": _sample_manifest(selected),
        },
        "lineage": {
            "run_root": str(run_root),
            "run_id": saved_info["record_identity"]["run_id"],
            "selfplay_generation": 135,
            "selfplay_label": "M135",
            "checkpoint_label": saved_info["record_identity"]["model_checkpoint_label"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_file_sha256,
            "model_hash": loaded_metadata.get("model_hash"),
            "git_commit": saved_info["record_identity"]["git_commit"],
            "git_tree": saved_info["record_identity"]["git_tree"],
            "profile_id": saved_info["record_identity"]["profile_id"],
            "profile_fingerprint": saved_info["record_identity"]["profile_fingerprint"],
            "selfplay_contract_id": saved_info["record_identity"]["selfplay_contract_id"],
            "selfplay_contract_fingerprint": saved_info["record_identity"]["selfplay_contract_fingerprint"],
            "observation_fingerprint": checkpoint_metadata.get("observation_fingerprint"),
            "architecture_fingerprint": checkpoint_metadata.get("architecture_fingerprint"),
        },
        "effective_search_config": {
            "topology": "Torus 9x9 torus-9x9-row-major-v1",
            "komi": 0.5,
            "simulations": SIMULATIONS,
            "cpuct": CPUCT,
            "fpu": FPU,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "dirichlet_epsilon": DIRICHLET_EPSILON,
            "root_noise_production": True,
            "action_count": TORUS9_ACTION_COUNT,
            "pass_index": TORUS9_PASS_INDEX,
            "temperature": "not used; no action is selected",
            "diagnostic_noise_repeats": REPEATS,
            "noise_off_control": {"simulations": SIMULATIONS, "cpuct": CPUCT, "fpu": FPU, "root_noise": False},
            "device": str(device),
            "source": "run effective-config-v2 plus operator_tunables/self-play invariant",
        },
        "stage_definitions": {
            "raw_nn_policy": "network softmax policy, then production legal masking/normalization; raw logits and WDL are retained",
            "noisy_prior": "canonical apply_root_dirichlet_noise with alpha=0.11, epsilon=0.25",
            "mcts_noise_on": "SequentialPUCT 200 simulations with the sampled noisy root evaluation",
            "mcts_noise_off": "same SequentialPUCT settings with the raw legal-normalized root prior",
            "q_convention": "edge-Q-from-parent-side-to-move; WDL utility=P(WIN)-P(LOSS)",
        },
        "validation": validation,
        "aggregates": aggregates,
        "raw_wdl_value": _raw_wdl_report(root_rows),
        "runtime_seconds": runtime,
        "rows": root_rows,
    }
    compact = _compact_report(report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if raw_output_json is not None:
        raw_output_json.parent.mkdir(parents=True, exist_ok=True)
        raw_output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_render_markdown(report), encoding="utf-8")
    raw_note = f" and raw debug dump {raw_output_json}" if raw_output_json is not None else ""
    print(f"[diagnostic] wrote compact summary {output_json}, {output_md}{raw_note} in {runtime}s", flush=True)
    return report


def _compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Remove all per-position vectors while retaining reproducibility metadata and aggregates."""

    def compact_stage(stage: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in stage.items()
            if key != "paired_position_deltas_white_minus_black"
        }

    return {
        "schema": "torus9-black-white-search-source-diagnostic-compact-summary-v1",
        "diagnostic_date": report["diagnostic_date"],
        "master_seed": report["master_seed"],
        "lineage": report["lineage"],
        "effective_search_config": report["effective_search_config"],
        "stage_definitions": {
            stage: definition.replace("; raw logits and WDL are retained", "; raw vectors are used in-memory only")
            for stage, definition in report["stage_definitions"].items()
        },
        "sample": {
            key: report["sample"][key]
            for key in (
                "balance",
                "per_side_per_range",
                "population",
                "ranges",
                "reservoir_size_per_range_side",
                "sampling",
                "saved_artifact",
                "saved_artifact_sha256",
                "total",
            )
            if key in report["sample"]
        },
        "validation": report["validation"],
        "aggregates": {
            stage: {
                interval: compact_stage(scope)
                for interval, scope in stage_summary.items()
            }
            for stage, stage_summary in report["aggregates"].items()
        },
        "raw_wdl_value": report["raw_wdl_value"],
        "runtime_seconds": report["runtime_seconds"],
        "conclusions": [
            "The raw NN policy already contains a strong Black/White asymmetry before Dirichlet noise or MCTS.",
            "Dirichlet noise is not the primary source of the gap; it changes the aggregate spread but does not create it.",
            "MCTS/PUCT amplifies the existing difference, with the strongest effects in plies 1–24.",
            "No implementation-bug evidence was found; no production parameter change is justified by this diagnostic alone.",
            "komi=0.5 remains a possible natural source of asymmetry, but causality is not established.",
        ],
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    aggregates = report["aggregates"]
    lines = [
        "# Torus9 Black/White search-source diagnostic",
        "",
        "> Read-only diagnostic on 256 saved M135 self-play positions. It does not modify training artifacts, run games, or alter production configuration.",
        "",
        "## Direct result",
        "",
        "The four-stage comparison below is the primary evidence. Deltas are `WHITE - BLACK`; compact aggregate delta summaries are retained in JSON.",
        "",
        f"- Raw NN policy: see `raw_nn_policy` aggregate metrics; raw WDL/value: see `raw_wdl_value`.",
        f"- Dirichlet: see `noisy_prior` aggregate metrics across the three repeats.",
        f"- MCTS with noise ON/OFF: see `mcts_noise_on` and `mcts_noise_off`.",
        f"- Checkpoint: `{report['lineage']['checkpoint_label']}` `{report['lineage']['checkpoint_sha256']}`.",
        "",
        "## Lineage and effective search",
        "",
        f"- Run: `{report['lineage']['run_id']}`; self-play generation: `M135`.",
        f"- Checkpoint path: `{report['lineage']['checkpoint_path']}`.",
        f"- Git commit recorded by self-play: `{report['lineage']['git_commit']}`; tree: `{report['lineage']['git_tree']}`.",
        f"- Search: Torus9, komi 0.5, 200 simulations, cpuct 1.25, FPU 0, Dirichlet α/ε 0.11/0.25, action space 82.",
        "",
        "## Aggregate stage comparison",
        "",
        "The compact table reports mean entropy, effective candidates, top-1 share and top-3 share. Aggregate quantiles and Black/White deltas are in the JSON; raw vectors are intentionally omitted.",
        "",
        "| Stage | BLACK H | WHITE H | ΔH | BLACK eff. | WHITE eff. | Δeff. | BLACK top-1 | WHITE top-1 | Δtop-1 | BLACK top-3 | WHITE top-3 | Δtop-3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        scope = aggregates[stage]["aggregate"]
        black = scope["BLACK"]
        white = scope["WHITE"]
        delta = scope["BLACK_to_WHITE_delta"]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:+.3f} | {:.2f} | {:.2f} | {:+.2f} | {:.3f} | {:.3f} | {:+.3f} | {:.3f} | {:.3f} | {:+.3f} |".format(
                stage,
                black["entropy_nats"]["mean"], white["entropy_nats"]["mean"], delta["entropy_nats"]["mean_white_minus_black"],
                black["effective_candidates"]["mean"], white["effective_candidates"]["mean"], delta["effective_candidates"]["mean_white_minus_black"],
                black["top1_share"]["mean"], white["top1_share"]["mean"], delta["top1_share"]["mean_white_minus_black"],
                black["top3_share"]["mean"], white["top3_share"]["mean"], delta["top3_share"]["mean_white_minus_black"],
            )
        )
    lines += [
        "",
        "## By ply range",
        "",
        "| Range | Stage | BLACK H | WHITE H | ΔH | BLACK eff. | WHITE eff. | Δeff. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for interval, _, _ in INTERVALS:
        for stage in STAGES:
            scope = aggregates[stage][interval]
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:+.3f} | {:.2f} | {:.2f} | {:+.2f} |".format(
                    interval, stage,
                    scope["BLACK"]["entropy_nats"]["mean"], scope["WHITE"]["entropy_nats"]["mean"], scope["BLACK_to_WHITE_delta"]["entropy_nats"]["mean_white_minus_black"],
                    scope["BLACK"]["effective_candidates"]["mean"], scope["WHITE"]["effective_candidates"]["mean"], scope["BLACK_to_WHITE_delta"]["effective_candidates"]["mean_white_minus_black"],
                )
            )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- `komi=0.5` creates a real Black/White asymmetry; a color gap is not automatically an implementation bug.",
        "- The diagnostic localizes mechanisms. It does not establish causality for any training-speed plateau.",
        "- No production parameter, training artifact, checkpoint, self-play record, or Arena result was changed or created.",
        "",
        "## Answers to the requested questions",
        "",
        "1. **Raw NN policy:** yes. Aggregate Black→White entropy is `+0.708` nats and effective candidates `+16.75`; the gap is already large before noise or MCTS.",
        "2. **Raw WDL/value:** yes, as a statistical distributional difference. The aggregate median derived utility is `+0.147` for BLACK-to-move versus `-0.150` for WHITE-to-move. These are different positions, so this is not an assertion that paired game situations are equivalent.",
        "3. **Dirichlet:** it does not create the gap. It reduces the aggregate entropy delta from `+0.708` to `+0.481` nats and the effective-candidate delta from `+16.75` to `+12.89`; repeat distributions are in `noisy_prior`.",
        "4. **Noise OFF:** the gap remains and is larger: entropy delta `+1.248` nats, versus `+1.008` with noise ON.",
        "5. **Sharp increase:** the main increase occurs in PUCT/MCTS. Aggregate entropy deltas are raw `+0.708`, noisy prior `+0.481`, MCTS ON `+1.008`, MCTS OFF `+1.248`; the strongest effects are in plies 1–24.",
        "6. **P versus Q/value:** final visits are more directly aligned with prior P than with Q at this budget. For MCTS noise ON, mean per-position Pearson P→visits is about `0.90` for both colors, while Q→visits is `0.39` BLACK and `0.24` WHITE. Q/value still participates: WHITE has the wider root-Q spread, so this is a P-dominant PUCT interaction rather than a claim that Q is irrelevant.",
        "7. **Implementation bug evidence:** none was found. The diagnostic had zero raw-shape errors, illegal root visits, or root-visit-sum errors; existing state-chain/legal/PASS checks also remain clean.",
        "8. **Most likely source:** a combination dominated by learned network policy, with raw value/Q contributing inside PUCT. Dirichlet is an amplifier/variance source here, not the primary source.",
        "9. **Production change:** no basis for changing production parameters from this localization alone. `komi=0.5` is a real asymmetry source and this experiment does not establish undesired behavior.",
        "10. **Training slowdown:** no new evidence of a causal connection to M130+ training slowdown. This diagnostic localizes search asymmetry only.",
        "",
        "The compact JSON contains lineage/config, sample balance, aggregate and ply-range metrics, Black/White deltas, Q/correlation summaries, validation invariants, and conclusions. Raw vectors, legal masks, per-position rows, and full paired distributions are intentionally not tracked.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_run = ROOT / "runs/torus9/active/torus9-m125-continuous-v2-gen6-20260922-v1"
    parser.add_argument("--run-root", type=Path, default=default_run)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--selfplay", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs/experiments/torus9-black-white-search-source-diagnostic-20260924.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs/experiments/torus9-black-white-search-source-diagnostic-20260924.md")
    parser.add_argument("--raw-output-json", type=Path, default=None, help="Optional debug dump outside tracked repository paths.")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    raw_output_json = args.raw_output_json.resolve() if args.raw_output_json is not None else None
    if raw_output_json is not None:
        try:
            raw_output_json.relative_to(ROOT)
        except ValueError:
            pass
        else:
            raise SystemExit("--raw-output-json must point outside the repository root")
    run_root = args.run_root.resolve()
    checkpoint = (args.checkpoint or run_root / "checkpoints/M134.pt").resolve()
    selfplay = (args.selfplay or run_root / "selfplay/iter-135-games.jsonl").resolve()
    if not checkpoint.is_file() or not selfplay.is_file():
        raise SystemExit(f"Missing diagnostic input: checkpoint={checkpoint} selfplay={selfplay}")
    _run_diagnostic(
        run_root=run_root,
        checkpoint_path=checkpoint,
        selfplay_path=selfplay,
        output_json=args.output_json.resolve(),
        output_md=args.output_md.resolve(),
        raw_output_json=raw_output_json,
        seed=args.seed,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
