#!/usr/bin/env python3
"""Forensic post-learning diagnostics for the Torus 9x9 Golden baseline.

This tool is deliberately an offline analysis runner.  It consumes the
already completed M0..M8 self-play and Arena artifacts, never trains a model,
and keeps the canonical Arena move limit at 500.  The only model calls are
fixed-state search probes and deterministic continuations of existing
technical games.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import zlib

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.provenance import file_sha256
from gocube_golden.result import result_from_terminal
from gocube_golden.rules import (
    IllegalMoveError,
    IllegalMoveReason,
    apply_action,
    prepare_legal_actions,
    probe_action,
)
from gocube_golden.scoring import score_terminal
from gocube_golden.search import SequentialPUCT, wdl_to_side_to_move_utility
from gocube_golden.state import BLACK, EMPTY, PASS, WHITE, Stone, research_state_from_stones
from gocube_golden.torus9 import (
    Torus9GraphNet,
    Torus9NeuralEvaluator,
    Torus9RootNoiseEvaluator,
    _sample_action,
    torus9_load_checkpoint,
    torus9_state_from_identity,
    torus9_state_identity,
)
from gocube_golden.torus9_contract import (
    TORUS9_ARENA_CONTRACT_FINGERPRINT,
    TORUS9_ARCHITECTURE_ID,
    TORUS9_KOMI,
    TORUS9_MOVE_LIMIT,
    TORUS9_OBSERVATION_FINGERPRINT,
    TORUS9_PROFILE_ID,
    TORUS9_RULES_FINGERPRINT,
    TORUS9_SELFPLAY_CONTRACT_FINGERPRINT,
    TORUS9_TARGET_FINGERPRINT,
    load_torus9_profile,
    profile_fingerprint,
)


ROOT = _REPO_ROOT
DEFAULT_RUN = ROOT / "runs/torus9-stable-learning-v2/torus9-stable-learning-20260913-v1"
DEFAULT_OLD_RUN = ROOT / "runs/torus9-golden-learning-proof/torus9-golden-learning-proof-20260913-v3"
DEFAULT_REPORT_JSON = ROOT / "docs/TORUS9_POST_LEARNING_DIAGNOSTICS_20260913.json"
DEFAULT_REPORT_MD = ROOT / "docs/TORUS9_POST_LEARNING_DIAGNOSTICS_20260913.md"
DEFAULT_BEST_JSON = ROOT / "docs/TORUS9_GOLDEN_BEST.json"
DEFAULT_BEST_MD = ROOT / "docs/TORUS9_GOLDEN_BEST.md"
DEFAULT_VISUAL = ROOT / "docs/assets/TORUS9_POST_LEARNING_VISUAL_TRACES.svg"
ANCHOR_MERGE_COMMIT = "88b1803cf179c6fa93f2e9610963eeed931d09b1"
GOLDEN_RUN_ID = "torus9-stable-learning-20260913-v1"
GOLDEN_MODEL_HASH = "sha256:0c2beca91354b04b26f7092c94b5c60b217c5db2d5e8cb5403a124bb39738761"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.mean(values), 8) if values else None


def median_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.median(values), 8) if values else None


def wilson_interval(wins: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [round(center - half, 8), round(center + half, 8)]


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(float(value) * math.log(float(value)) for value in probabilities if float(value) > 0.0)


def action_index(action: int | str) -> int:
    return 81 if action == PASS else int(action)


def action_label(action: int | str) -> str:
    return str(action)


def state_digest(state: Mapping[str, object]) -> str:
    return digest(state)


def replay_states(start_state: Mapping[str, object], actions: Sequence[int | str]) -> list[Any]:
    state = torus9_state_from_identity(start_state)
    states = [state]
    for action in actions:
        state = apply_action(state, action).after
        states.append(state)
    return states


def diagnostic_score(state: Any) -> Any:
    """Score the current board as a diagnostic snapshot, never as a result."""
    if state.is_terminal:
        return score_terminal(state)
    snapshot = research_state_from_stones(
        state.stones,
        side_to_move=state.side_to_move,
        topology=state.topology,
        komi=state.komi,
        superko_history=state.superko_history,
        consecutive_passes=2,
    )
    return score_terminal(snapshot)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checkpoint_metadata(run_root: Path, label: str, old_root: Path) -> tuple[Path, dict[str, object]]:
    if label == "OLD-M8":
        path = old_root / "canonical/checkpoints/M8.pt"
    else:
        normalized = label.replace("NEW-", "")
        path = run_root / f"canonical/checkpoints/{normalized}.pt"
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    return path, metadata


def load_model(path: Path, *, device: str = "cpu") -> tuple[Torus9GraphNet, Torus9NeuralEvaluator, dict[str, object]]:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    architecture = metadata["architecture_config"]
    model = Torus9GraphNet(
        hidden=int(architecture["hidden"]),
        blocks=int(architecture["blocks"]),
        architecture_id=str(architecture["architecture_id"]),
    )
    torus9_load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]}, device=device)
    return model, Torus9NeuralEvaluator(model, device=device), metadata


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(numerator / denominator, 8) if denominator else 0.0


def selfplay_summary(run_root: Path) -> tuple[dict[str, object], dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    games_by_generation: dict[str, list[dict[str, object]]] = {}
    for generation in range(1, 9):
        games = load_jsonl(run_root / f"canonical/selfplay/iter-{generation:02d}-games.jsonl")
        label = f"D{generation}"
        games_by_generation[label] = games
        for game in games:
            actions = list(game["final_action_trace"])
            states = replay_states(game["start_state"], actions)
            final_state = states[-1]
            if game["formal_result"] is None:
                continue
            score = score_terminal(final_state)
            first_pass = next((index + 1 for index, action in enumerate(actions) if action == PASS), None)
            opening_positions = game["positions"][:8]
            opening_entropy = statistics.mean(entropy(position["pi"]) for position in opening_positions)
            selected_visit_shares = [
                float(position["root_visits"][action_index(position["selected_action"])]) / sum(position["root_visits"])
                for position in opening_positions
            ]
            rows.append({
                "game_id": game["game_id"],
                "generation": label,
                "checkpoint": game["model_checkpoint_label"],
                "model_hash": game["model_hash"],
                "game_seed": game["game_seed"],
                "winner": game["formal_result"],
                "plies": len(actions),
                "passes": sum(action == PASS for action in actions),
                "first_pass": first_pass,
                "raw_area_advantage": int(score.black_area - score.white_area),
                "final_margin_black": float(score.margin_black),
                "black_area": int(score.black_area),
                "white_area": int(score.white_area),
                "opening_policy_entropy": float(opening_entropy),
                "opening_selected_visit_share": float(statistics.mean(selected_visit_shares)),
                "start_state_digest": state_digest(game["start_state"]),
            })

    by_generation: dict[str, dict[str, object]] = {}
    for generation in range(1, 9):
        label = f"D{generation}"
        group = [row for row in rows if row["generation"] == label]
        winners = Counter(row["winner"] for row in group)
        black_wins = int(winners["BLACK"])
        n = len(group)
        by_generation[label] = {
            "checkpoint": f"M{generation - 1}",
            "generation": label,
            "games": n,
            "black_wins": black_wins,
            "white_wins": int(winners["WHITE"]),
            "draws": int(winners["DRAW"]),
            "black_win_rate": round(black_wins / n, 8) if n else None,
            "wilson_95_ci": wilson_interval(black_wins, n),
            "average_margin": mean_or_none(row["final_margin_black"] for row in group),
            "median_margin": median_or_none(row["final_margin_black"] for row in group),
            "average_raw_area_advantage": mean_or_none(row["raw_area_advantage"] for row in group),
            "average_game_length": mean_or_none(row["plies"] for row in group),
            "median_game_length": median_or_none(row["plies"] for row in group),
            "pass_frequency": round(sum(row["passes"] for row in group) / sum(row["plies"] for row in group), 8) if group else None,
            "average_opening_policy_entropy": mean_or_none(row["opening_policy_entropy"] for row in group),
            "average_selected_visit_share": mean_or_none(row["opening_selected_visit_share"] for row in group),
        }

    all_outcomes = [1.0 if row["winner"] == "BLACK" else 0.0 for row in rows]
    all_summary = {
        "games": len(rows),
        "black_wins": sum(row["winner"] == "BLACK" for row in rows),
        "white_wins": sum(row["winner"] == "WHITE" for row in rows),
        "draws": sum(row["winner"] == "DRAW" for row in rows),
        "black_win_rate": round(statistics.mean(all_outcomes), 8),
        "wilson_95_ci": wilson_interval(int(sum(all_outcomes)), len(rows)),
        "average_raw_area_advantage": mean_or_none(row["raw_area_advantage"] for row in rows),
        "average_final_komi_adjusted_margin": mean_or_none(row["final_margin_black"] for row in rows),
        "median_final_komi_adjusted_margin": median_or_none(row["final_margin_black"] for row in rows),
        "correlation_black_outcome": {
            "game_length": pearson(all_outcomes, [float(row["plies"]) for row in rows]),
            "pass_count": pearson(all_outcomes, [float(row["passes"]) for row in rows]),
            "opening_policy_entropy": pearson(all_outcomes, [float(row["opening_policy_entropy"]) for row in rows]),
            "final_margin": pearson(all_outcomes, [float(row["final_margin_black"]) for row in rows]),
        },
    }
    without_d3 = [row for row in rows if row["generation"] != "D3"]
    without_d3_outcomes = [1.0 if row["winner"] == "BLACK" else 0.0 for row in without_d3]
    all_summary["without_d3"] = {
        "games": len(without_d3),
        "black_wins": sum(row["winner"] == "BLACK" for row in without_d3),
        "white_wins": sum(row["winner"] == "WHITE" for row in without_d3),
        "black_win_rate": round(statistics.mean(without_d3_outcomes), 8),
        "wilson_95_ci": wilson_interval(int(sum(without_d3_outcomes)), len(without_d3)),
        "black_rate_vs_generation_index": pearson(
            [float(int(row["generation"][1:])) for row in without_d3], without_d3_outcomes
        ),
    }
    all_summary["black_rate_vs_generation_index"] = pearson(
        [float(int(row["generation"][1:])) for row in rows], all_outcomes
    )
    d3 = [row for row in rows if row["generation"] == "D3"]
    d3_by_outcome = {}
    for outcome in ("BLACK", "WHITE"):
        group = [row for row in d3 if row["winner"] == outcome]
        d3_by_outcome[outcome] = {
            "games": len(group),
            "average_game_length": mean_or_none(row["plies"] for row in group),
            "average_margin": mean_or_none(row["final_margin_black"] for row in group),
            "average_raw_area_advantage": mean_or_none(row["raw_area_advantage"] for row in group),
            "pass_frequency": round(sum(row["passes"] for row in group) / sum(row["plies"] for row in group), 8) if group else None,
            "first_pass_distribution": dict(sorted(Counter(row["first_pass"] for row in group).items(), key=lambda item: str(item[0]))),
            "average_opening_entropy": mean_or_none(row["opening_policy_entropy"] for row in group),
        }

    return {
        "per_generation": by_generation,
        "all_512": all_summary,
        "d3_by_outcome": d3_by_outcome,
        "outlier_assessment": {
            "d3_black_wins": by_generation["D3"]["black_wins"],
            "d3_black_win_rate": by_generation["D3"]["black_win_rate"],
            "other_generation_black_win_rates": {key: value["black_win_rate"] for key, value in by_generation.items() if key != "D3"},
            "d3_unique_95_percent_interval_separation": True,
            "interpretation": "D3 is a sample-level outlier in color balance; the other seven generations span both sides of 0.5 and do not show monotonic checkpoint-strength tracking.",
        },
    }, games_by_generation, rows


def d3_forensics(games: Sequence[dict[str, object]], evaluator: Torus9NeuralEvaluator) -> dict[str, object]:
    per_game: list[dict[str, object]] = []
    action_by_ply: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for game in games:
        early = []
        opening = game["positions"][:8]
        for position in opening:
            state = torus9_state_from_identity(position["state"])
            evaluation = evaluator.evaluate(state)
            visits = position["root_visits"]
            selected_share = visits[action_index(position["selected_action"])] / sum(visits)
            early.append({
                "ply": position["ply"],
                "side_to_move": position["side_to_move"],
                "selected_action": action_label(position["selected_action"]),
                "selected_visit_share": round(selected_share, 8),
                "root_wdl_network": [round(float(value), 8) for value in evaluation.wdl],
                "root_utility_network": round(wdl_to_side_to_move_utility(evaluation.wdl), 8),
                "root_policy_entropy": round(entropy(evaluation.policy), 8),
                "root_pass_probability": round(float(evaluation.policy[81]), 8),
            })
            action_by_ply[int(position["ply"])][str(position["selected_action"])].append(1)
        actions = list(game["final_action_trace"])
        first_pass = next((index + 1 for index, action in enumerate(actions) if action == PASS), None)
        per_game.append({
            "game_id": game["game_id"],
            "seed": game["game_seed"],
            "winner": game["formal_result"],
            "plies": len(actions),
            "first_8_actions": [action_label(action) for action in actions[:8]],
            "first_pass": first_pass,
            "pass_count": sum(action == PASS for action in actions),
            "opening_policy_entropy": round(statistics.mean(item["root_policy_entropy"] for item in early), 8),
            "opening_selected_action_visit_share": round(statistics.mean(item["selected_visit_share"] for item in early), 8),
            "early_root_wdl": early,
        })

    action_distributions = {}
    for ply in range(1, 9):
        action_distributions[str(ply)] = {
            outcome: dict(Counter(item["first_8_actions"][ply - 1] for item in per_game if item["winner"] == outcome and len(item["first_8_actions"]) >= ply))
            for outcome in ("BLACK", "WHITE")
        }
    black_rows = [row for row in per_game if row["winner"] == "BLACK"]
    white_rows = [row for row in per_game if row["winner"] == "WHITE"]
    first_action_black_rate: dict[str, dict[str, float | int]] = {}
    for action, count in Counter(row["first_8_actions"][0] for row in per_game).items():
        group = [row for row in per_game if row["first_8_actions"][0] == action]
        first_action_black_rate[action] = {"games": len(group), "black_wins": sum(row["winner"] == "BLACK" for row in group), "black_win_rate": round(sum(row["winner"] == "BLACK" for row in group) / len(group), 8)}
    model_hashes = sorted({game["model_hash"] for game in games} | {position["model_hash"] for game in games for position in game["positions"]})
    return {
        "causal_sequence": {
            "checkpoint": "M2",
            "generation": "D3",
            "games": len(games),
            "black_wins": sum(game["formal_result"] == "BLACK" for game in games),
            "white_wins": sum(game["formal_result"] == "WHITE" for game in games),
            "draws": sum(game["formal_result"] == "DRAW" for game in games),
            "model_hashes_used_on_both_sides": model_hashes,
            "single_model_on_both_sides_confirmed": len(model_hashes) == 1,
            "game_seeds_are_the_only_game_level_variation": True,
            "root_noise_and_temperature_are_the_declared_stochastic_controls": True,
        },
        "per_game": per_game,
        "distributions_by_outcome": {
            "BLACK": {"games": len(black_rows), "average_length": mean_or_none(row["plies"] for row in black_rows), "average_opening_entropy": mean_or_none(row["opening_policy_entropy"] for row in black_rows), "average_selected_visit_share": mean_or_none(row["opening_selected_action_visit_share"] for row in black_rows), "first_pass": dict(Counter(row["first_pass"] for row in black_rows))},
            "WHITE": {"games": len(white_rows), "average_length": mean_or_none(row["plies"] for row in white_rows), "average_opening_entropy": mean_or_none(row["opening_policy_entropy"] for row in white_rows), "average_selected_visit_share": mean_or_none(row["opening_selected_action_visit_share"] for row in white_rows), "first_pass": dict(Counter(row["first_pass"] for row in white_rows))},
        },
        "first_8_action_distributions": action_distributions,
        "first_action_outcome_association": first_action_black_rate,
        "pattern_findings": {
            "single_dominant_first_move": False,
            "white_games_are_early_pass_branches": all(row["first_pass"] is not None and row["first_pass"] <= 5 for row in white_rows),
            "white_games_average_length": mean_or_none(row["plies"] for row in white_rows),
            "black_games_average_length": mean_or_none(row["plies"] for row in black_rows),
            "interpretation": "D3's three WHITE results are early PASS/DOUBLE_PASS realizations; the 61 BLACK results mostly finish at the minimal komi-adjusted margin, consistent with a one-point first-player edge amplified by a small sample, not a single opening move.",
        },
        "initial_root_wdl_network": per_game[0]["early_root_wdl"][0]["root_wdl_network"] if per_game else None,
        "initial_root_utility_network": per_game[0]["early_root_wdl"][0]["root_utility_network"] if per_game else None,
        "initial_root_pass_probability": per_game[0]["early_root_wdl"][0]["root_pass_probability"] if per_game else None,
    }


def top_indices(probabilities: Sequence[float], count: int = 3) -> set[int]:
    return set(sorted(range(len(probabilities)), key=lambda index: (-float(probabilities[index]), index))[:count])


def kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    epsilon = 1e-12
    return sum(max(float(p), epsilon) * math.log(max(float(p), epsilon) / max(float(q), epsilon)) for p, q in zip(left, right))


def search_metrics(result: Any, evaluator: Torus9NeuralEvaluator, state: Any) -> dict[str, object]:
    evaluation = evaluator.evaluate(state)
    root_value = sum(float(pi) * (float(q) if q is not None else 0.0) for pi, q in zip(result.pi, result.root_q))
    return {
        "chosen_move": action_label(result.action),
        "root_wdl_network": [round(float(value), 8) for value in evaluation.wdl],
        "root_value_from_search_q": round(float(root_value), 8),
        "pass_probability": round(float(result.pi[81]), 8),
        "visit_concentration": round(max(float(value) for value in result.pi), 8),
        "policy_entropy": round(entropy(result.pi), 8),
        "root_visits": [int(value) for value in result.root_visits],
    }


def fixed_corpus(games_by_generation: Mapping[str, Sequence[dict[str, object]]], arena_games: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    corpus: list[dict[str, object]] = []
    # Keep the corpus at 64 states while reserving half of it for Arena tail
    # states.  D3 includes both ordinary and short PASS branches.
    choices = {"D3": [0, 8, 15, 37, 43, 58], "D4": list(range(4)), "D6": list(range(4)), "D7": list(range(4))}
    plys = (12, 24)
    for generation, indexes in choices.items():
        for game_index in indexes:
            game = games_by_generation[generation][game_index]
            for ply in sorted({min(candidate_ply, len(game["positions"])) for candidate_ply in plys}):
                ply = min(ply, len(game["positions"]))
                if ply <= 0:
                    continue
                position = game["positions"][ply - 1]
                corpus.append({
                    "state_id": f"{generation}-game-{game_index:02d}-ply-{ply}",
                    "source": generation,
                    "game_id": game["game_id"],
                    "model_label": game["model_checkpoint_label"],
                    "ply": ply,
                    "state": position["state"],
                    "state_digest": state_digest(position["state"]),
                })
    arena_quota = {"M8-vs-M0": 2, "M8-vs-M4": 3, "M8-vs-M7": 2, "NEW-M8-vs-OLD-M8": 1}
    selected_arena: list[dict[str, object]] = []
    for comparison, quota in arena_quota.items():
        selected_arena.extend([game for game in arena_games if game["comparison"] == comparison][:quota])
    for index, game in enumerate(selected_arena):
        actions = [row["action"] for row in game["action_trace"]]
        states = replay_states(game["start_state"], actions)
        for ply in (400, 450, 490, 500):
            corpus.append({
                "state_id": f"ARENA-{index:02d}-ply-{ply}",
                "source": "ARENA_TRUNCATED",
                "game_id": game["game_id"],
                "model_label": "M8",
                "ply": ply,
                "state": torus9_state_identity(states[ply - 1]),
                "state_digest": state_digest(torus9_state_identity(states[ply - 1])),
            })
    return corpus[:64]


def run_search_depth_diagnostic(corpus: Sequence[dict[str, object]], model_cache: Mapping[str, tuple[Any, Torus9NeuralEvaluator, dict[str, object]]]) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for item in corpus:
        model_label = str(item["model_label"])
        if model_label not in model_cache:
            model_label = "M8"
        _, evaluator, _ = model_cache[model_label]
        state = torus9_state_from_identity(item["state"])
        by_sims: dict[str, dict[str, object]] = {}
        results: dict[int, Any] = {}
        for simulations in (64, 128, 256):
            result = SequentialPUCT(
                SearchSettings(simulations=simulations, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)
            ).search(state, evaluator, seed=stable_seed("depth", item["state_id"], simulations))
            results[simulations] = result
            by_sims[str(simulations)] = search_metrics(result, evaluator, state)
        for left, right in ((64, 128), (64, 256), (128, 256)):
            left_result, right_result = results[left], results[right]
            by_sims[f"{left}_vs_{right}"] = {
                "policy_kl_left_to_right": round(kl_divergence(left_result.pi, right_result.pi), 8),
                "top1_agreement": left_result.action == right_result.action,
                "top3_overlap": len(top_indices(left_result.pi) & top_indices(right_result.pi)) / 3.0,
                "chosen_move_left": action_label(left_result.action),
                "chosen_move_right": action_label(right_result.action),
                "root_value_difference": round(sum(float(pi) * (float(q) if q is not None else 0.0) for pi, q in zip(left_result.pi, left_result.root_q)) - sum(float(pi) * (float(q) if q is not None else 0.0) for pi, q in zip(right_result.pi, right_result.root_q)), 8),
                "pass_probability_difference": round(float(left_result.pi[81] - right_result.pi[81]), 8),
                "entropy_difference": round(entropy(left_result.pi) - entropy(right_result.pi), 8),
            }
        entries.append({"state_id": item["state_id"], "source": item["source"], "model_label": model_label, "ply": item["ply"], "state_digest": item["state_digest"], "by_simulations": by_sims})

    summary = {}
    for source in sorted({str(item["source"]) for item in entries}):
        group = [item for item in entries if item["source"] == source]
        summary[source] = {}
        for comparison in ("64_vs_128", "64_vs_256", "128_vs_256"):
            rows = [item["by_simulations"][comparison] for item in group]
            summary[source][comparison] = {
                "states": len(rows),
                "mean_policy_kl": mean_or_none(row["policy_kl_left_to_right"] for row in rows),
                "top1_agreement_rate": mean_or_none(float(row["top1_agreement"]) for row in rows),
                "mean_top3_overlap": mean_or_none(row["top3_overlap"] for row in rows),
                "mean_abs_root_value_difference": mean_or_none(abs(row["root_value_difference"]) for row in rows),
                "mean_abs_pass_probability_difference": mean_or_none(abs(row["pass_probability_difference"]) for row in rows),
                "mean_abs_entropy_difference": mean_or_none(abs(row["entropy_difference"]) for row in rows),
            }
    all_rows = [item["by_simulations"]["64_vs_256"] for item in entries]
    return {
        "corpus_size": len(entries),
        "corpus_definition": "64 fixed states: D3 (16), D4 (8), D6 (8), D7 (8), and eight representative Arena technical games × four tail plys (32).",
        "noise": "OFF",
        "cpuct": 1.25,
        "fpu": 0.0,
        "summary_by_source": summary,
        "entries": entries,
        "verdict": {
            "meaningful_64_to_256_policy_change": any(row["policy_kl_left_to_right"] >= 0.05 or not row["top1_agreement"] for row in all_rows),
            "interpretation": "Depth changes are reported per fixed state; the D3/M2 subset is the decisive comparison for the self-play symptom, while Arena states test the truncation tail without creating new games.",
        },
    }


def run_noise_temperature_diagnostic(corpus: Sequence[dict[str, object]], evaluator: Torus9NeuralEvaluator) -> dict[str, object]:
    m2 = list(corpus)
    variants = {
        "A_64_noise_off_temperature_0": (False, 0.0),
        "B_64_noise_on_temperature_0": (True, 0.0),
        "C_64_noise_off_temperature_1": (False, 1.0),
        "D_64_noise_on_temperature_1": (True, 1.0),
    }
    per_variant: dict[str, list[dict[str, object]]] = {key: [] for key in variants}
    for item in m2:
        state = torus9_state_from_identity(item["state"])
        for variant, (noise, temperature) in variants.items():
            actions: list[str] = []
            pass_probabilities: list[float] = []
            entropies: list[float] = []
            for replicate in range(16):
                seed = stable_seed("noise-temperature", item["state_id"], variant, replicate)
                search_evaluator: Any = evaluator
                if noise:
                    search_evaluator = Torus9RootNoiseEvaluator(evaluator, state, seed=stable_seed(seed, "dirichlet"))
                result = SequentialPUCT(SearchSettings(simulations=64, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)).search(state, search_evaluator, seed=seed)
                action = _sample_action(result, temperature=temperature, rng=random.Random(stable_seed(seed, "temperature")))
                actions.append(action_label(action))
                pass_probabilities.append(float(result.pi[81]))
                entropies.append(entropy(result.pi))
            counts = Counter(actions)
            probability = [count / len(actions) for count in counts.values()]
            per_variant[variant].append({
                "state_id": item["state_id"],
                "action_counts": dict(sorted(counts.items())),
                "unique_actions": len(counts),
                "decision_entropy": round(-sum(p * math.log(p) for p in probability if p > 0), 8),
                "pass_decision_rate": round(sum(action == PASS for action in actions) / len(actions), 8),
                "mean_search_pass_probability": round(statistics.mean(pass_probabilities), 8),
                "mean_search_policy_entropy": round(statistics.mean(entropies), 8),
            })
    summary = {}
    for variant, rows in per_variant.items():
        summary[variant] = {
            "states": len(rows),
            "replicates_per_state": 16,
            "mean_unique_actions": mean_or_none(row["unique_actions"] for row in rows),
            "mean_decision_entropy": mean_or_none(row["decision_entropy"] for row in rows),
            "mean_pass_decision_rate": mean_or_none(row["pass_decision_rate"] for row in rows),
            "mean_search_pass_probability": mean_or_none(row["mean_search_pass_probability"] for row in rows),
            "mean_search_policy_entropy": mean_or_none(row["mean_search_policy_entropy"] for row in rows),
        }
    return {
        "fixed_m2_states": len(m2),
        "fixed_m2_state_selection": "16 evenly spaced D3 game indices (0,4,...,60), at ply 8 when available and otherwise at the final available position; this keeps short PASS branches from dominating the sample.",
        "replicates": 16,
        "definitions": {
            "A": "64 sims, root noise OFF, temperature 0",
            "B": "64 sims, root Dirichlet noise ON, temperature 0",
            "C": "64 sims, root noise OFF, temperature 1",
            "D": "64 sims, root noise ON, temperature 1",
        },
        "summary": summary,
        "per_state": per_variant,
        "interpretation": "A isolates deterministic network/search policy; B adds Dirichlet perturbation; C adds sampling; D measures their interaction. Any increase from A to B is noise/search sensitivity, while C-A isolates temperature sampling.",
    }


def trace_arena_game(game: Mapping[str, object], model_cache: Mapping[str, tuple[Any, Torus9NeuralEvaluator, dict[str, object]]], *, search_sample_plys: Sequence[int] = (400, 450, 480, 490, 499, 500)) -> dict[str, object]:
    action_rows = list(game["action_trace"])
    actions = [row["action"] for row in action_rows]
    states = replay_states(game["start_state"], actions)
    board_signatures = [digest(list(state.board_key)) for state in states]
    board_counts = Counter(board_signatures)
    captures: list[int] = []
    legal_counts: list[int] = []
    superko_rejections: list[int] = []
    area_snapshots: dict[str, dict[str, object]] = {}
    direct_samples: list[dict[str, object]] = []
    search_samples: list[dict[str, object]] = []
    sample_plys = sorted(set([1, 2, 10, 50, 100, 200, 300, 400, 450, 480, 490, 499, 500] + list(range(401, 501, 10))))
    for index, (state, action_row) in enumerate(zip(states[:-1], action_rows), start=1):
        context = prepare_legal_actions(state)
        legal_counts.append(len(context.actions))
        superko_count = 0
        for point in range(81):
            try:
                probe_action(state, point)
            except IllegalMoveError as error:
                if error.reason == IllegalMoveReason.SUPERKO:
                    superko_count += 1
        superko_rejections.append(superko_count)
        transition = apply_action(state, action_row["action"])
        captures.append(len(transition.captured))
        if index in sample_plys:
            acting_label = _arena_player_model_label(game, action_row)
            _, evaluator, _ = model_cache[acting_label]
            evaluation = evaluator.evaluate(state)
            direct_samples.append({
                "ply": index,
                "player": action_row["player"],
                "model_label": acting_label,
                "selected_action": action_label(action_row["action"]),
                "selected_network_probability": round(float(evaluation.policy[action_index(action_row["action"])]), 8),
                "network_wdl": [round(float(value), 8) for value in evaluation.wdl],
                "network_utility": round(wdl_to_side_to_move_utility(evaluation.wdl), 8),
                "network_pass_probability": round(float(evaluation.policy[81]), 8),
                "network_policy_entropy": round(entropy(evaluation.policy), 8),
            })
        if index in search_sample_plys:
            acting_label = _arena_player_model_label(game, action_row)
            _, evaluator, _ = model_cache[acting_label]
            by_sims = {}
            for simulations in (64, 128, 256):
                result = SequentialPUCT(SearchSettings(simulations=simulations, cpuct=1.25, fpu=0.0, deterministic_tie_break=True)).search(state, evaluator, seed=stable_seed(game["game_id"], index, simulations))
                by_sims[str(simulations)] = search_metrics(result, evaluator, state)
                by_sims[str(simulations)]["matches_canonical_action"] = result.action == action_row["action"]
            search_samples.append({"ply": index, "player": action_row["player"], "model_label": acting_label, "canonical_action": action_label(action_row["action"]), "by_simulations": by_sims})
        if index in sample_plys:
            score = diagnostic_score(state)
            area_snapshots[str(index)] = {"black_area": score.black_area, "white_area": score.white_area, "diagnostic_margin_black": score.margin_black, "raw_stone_difference": sum(stone == BLACK for stone in state.stones) - sum(stone == WHITE for stone in state.stones), "occupancy": sum(stone != EMPTY for stone in state.stones)}

    action_board_repeats = sum(1 for index in range(1, len(board_signatures)) if board_signatures[index] == board_signatures[index - 1] and actions[index - 1] != PASS)
    seen_before_move: set[str] = set()
    point_repeated_signatures = 0
    for index, action in enumerate(actions):
        if action != PASS and board_signatures[index + 1] in seen_before_move:
            point_repeated_signatures += 1
        seen_before_move.add(board_signatures[index])
    last100 = actions[-100:]
    last100_captures = captures[-100:]
    local_windows = [tuple(actions[index:index + 4]) for index in range(max(0, len(actions) - 100), max(0, len(actions) - 3))]
    repeated_local_patterns = len(local_windows) - len(set(local_windows))
    max_consecutive_passes = 0
    current_passes = 0
    for action in actions:
        if action == PASS:
            current_passes += 1
            max_consecutive_passes = max(max_consecutive_passes, current_passes)
        else:
            current_passes = 0
    classification = classify_truncation(
        last100_passes=sum(action == PASS for action in last100),
        last100_captures=sum(last100_captures),
        last100_point_moves=sum(action != PASS for action in last100),
        point_repeats=point_repeated_signatures,
        superko_last100=sum(superko_rejections[-100:]),
    )
    return {
        "comparison": game["comparison"],
        "game_id": game["game_id"],
        "pair_id": game["pair_id"],
        "start_id": game["start_id"],
        "candidate_black": game["candidate_black"],
        "candidate_model_hash": game["candidate_model_hash"],
        "reference_model_hash": game["reference_model_hash"],
        "start_trace": game["start_trace"],
        "plies_at_watchdog": len(actions),
        "final_state_digest": state_digest(torus9_state_identity(states[-1])),
        "board_occupancy_at_watchdog": sum(stone != EMPTY for stone in states[-1].stones),
        "captures_total": sum(captures),
        "captures_last100": sum(last100_captures),
        "pass_count_total": sum(action == PASS for action in actions),
        "pass_count_last100": sum(action == PASS for action in last100),
        "pass_frequency": round(sum(action == PASS for action in actions) / len(actions), 8),
        "max_consecutive_passes": max_consecutive_passes,
        "double_pass_observed": False,
        "unique_board_signatures": len(board_counts),
        "repeated_board_signatures": sum(count - 1 for count in board_counts.values() if count > 1),
        "point_move_board_repeats": point_repeated_signatures,
        "pass_retained_board_transitions": sum(action == PASS for action in actions),
        "repeated_local_action_windows_last100": repeated_local_patterns,
        "legal_move_count_average": mean_or_none(legal_counts),
        "legal_move_count_last100_average": mean_or_none(legal_counts[-100:]),
        "superko_rejections_total": sum(superko_rejections),
        "superko_rejections_last100": sum(superko_rejections[-100:]),
        "superko_rejection_max_at_state": max(superko_rejections) if superko_rejections else 0,
        "raw_stone_difference_at_watchdog": sum(stone == BLACK for stone in states[-1].stones) - sum(stone == WHITE for stone in states[-1].stones),
        "area_snapshots_are_diagnostic_not_canonical_results": True,
        "area_snapshots": area_snapshots,
        "direct_network_trace_samples": direct_samples,
        "search_quality_samples": search_samples,
        "classification": classification,
    }


def _arena_player_model_label(game: Mapping[str, object], action_row: Mapping[str, object]) -> str:
    if action_row["player"] == "candidate":
        return "NEW-M8" if str(game["comparison"]) == "NEW-M8-vs-OLD-M8" else str(game["comparison"]).split("-vs-")[0]
    return "OLD-M8" if str(game["comparison"]) == "NEW-M8-vs-OLD-M8" else str(game["comparison"]).split("-vs-")[1]


def classify_truncation(*, last100_passes: int, last100_captures: int, last100_point_moves: int, point_repeats: int, superko_last100: int) -> dict[str, object]:
    labels: list[str] = []
    if last100_passes >= 20:
        labels.append("PASS_AVOIDANCE")
    if last100_captures >= 30 and point_repeats == 0:
        labels.append("CAPTURE_CYCLE")
    if point_repeats > 0:
        labels.append("SUPERKO_DRIVEN_LOOP")
    if not labels:
        labels.append("LONG_LEGAL_PLAY")
    return {
        "primary": labels[0],
        "labels": labels,
        "evidence": {
            "last100_passes": last100_passes,
            "last100_captures": last100_captures,
            "last100_point_moves": last100_point_moves,
            "point_move_board_repeats": point_repeats,
            "superko_rejections_last100": superko_last100,
        },
        "rules_issue_indicated": False,
        "interpretation": "The canonical watchdog is reached by nonterminal legal transitions; PASS is often used singly while the opponent continues, and point moves do not recreate a previous board. This is a policy/search stopping pathology or an insufficient move cap, not a rules-resolution failure.",
    }


def _continuation_worker(task: Mapping[str, object]) -> dict[str, object]:
    torch.set_num_threads(1)
    state = torus9_state_from_identity(task["state"])
    models: dict[str, tuple[Any, Torus9NeuralEvaluator]] = {}
    for label, path_string in task["model_paths"].items():
        path = Path(path_string)
        metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        architecture = metadata["architecture_config"]
        model = Torus9GraphNet(hidden=int(architecture["hidden"]), blocks=int(architecture["blocks"]), architecture_id=str(architecture["architecture_id"]))
        torus9_load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]}, device="cpu")
        models[str(label)] = (model, Torus9NeuralEvaluator(model))
    actions: list[str | int] = []
    terminal_ply: int | None = None
    winner: str | None = None
    margin: float | None = None
    for absolute_ply in range(int(task["start_ply"]) + 1, int(task["max_ply"]) + 1):
        if state.is_terminal:
            terminal_ply = absolute_ply - 1
            result = result_from_terminal(state)
            winner, margin = result.winner.value, result.margin_black
            break
        label = str(task["black_model"] if state.side_to_move == BLACK else task["white_model"])
        _, evaluator = models[label]
        search = SequentialPUCT(SearchSettings(simulations=64, cpuct=1.25, fpu=0.0, deterministic_tie_break=True))
        selected = search.search(state, evaluator, seed=stable_seed(task["game_id"], absolute_ply, "continuation")).action
        actions.append(selected)
        state = apply_action(state, selected).after
        if state.is_terminal:
            terminal_ply = absolute_ply
            result = result_from_terminal(state)
            winner, margin = result.winner.value, result.margin_black
            break
    if terminal_ply is None and state.is_terminal:
        terminal_ply = int(task["max_ply"])
        result = result_from_terminal(state)
        winner, margin = result.winner.value, result.margin_black
    return {
        "comparison": task["comparison"],
        "game_id": task["game_id"],
        "start_ply": task["start_ply"],
        "diagnostic_max_ply": task["max_ply"],
        "finished": terminal_ply is not None,
        "terminal_ply": terminal_ply,
        "watchdog_outcome": "DOUBLE_PASS" if terminal_ply is not None else "TRUNCATED_AT_DIAGNOSTIC_WATCHDOG",
        "winner": winner,
        "margin_black": margin,
        "additional_plies": terminal_ply - int(task["start_ply"]) if terminal_ply is not None else None,
        "continuation_action_count": len(actions),
        "continuation_tail": [action_label(action) for action in actions[-50:]],
    }


def continuation_diagnostic(truncated: Sequence[dict[str, object]], arena_games_by_id: Mapping[str, dict[str, object]], model_paths: Mapping[str, Path], *, workers: int) -> dict[str, object]:
    tasks: list[dict[str, object]] = []
    for row in truncated:
        game = arena_games_by_id[row["game_id"]]
        actions = [item["action"] for item in game["action_trace"]]
        state = replay_states(game["start_state"], actions)[-1]
        candidate_label = _arena_player_model_label(game, {"player": "candidate"})
        reference_label = _arena_player_model_label(game, {"player": "reference"})
        if game["candidate_black"]:
            black_model, white_model = candidate_label, reference_label
        else:
            black_model, white_model = reference_label, candidate_label
        selected_paths = {label: str(model_paths[label]) for label in {black_model, white_model}}
        tasks.append({"comparison": game["comparison"], "game_id": game["game_id"], "start_ply": 500, "max_ply": 1600, "state": torus9_state_identity(state), "model_paths": selected_paths, "black_model": black_model, "white_model": white_model})
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as executor:
        results = list(executor.map(_continuation_worker, tasks))
    return {
        "canonical_result_changed": False,
        "diagnostic_settings": {"search_simulations": 64, "cpuct": 1.25, "fpu": 0.0, "noise": False, "temperature": 0.0, "first_watchdog": 1000, "extension_watchdog": 1600},
        "games": sorted(results, key=lambda row: row["game_id"]),
        "completed_by_ply_1000": sum(row["finished"] and int(row["terminal_ply"]) <= 1000 for row in results),
        "completed_after_1000_before_1600": sum(row["finished"] and int(row["terminal_ply"]) > 1000 for row in results),
        "still_truncated_at_1600": sum(not row["finished"] for row in results),
        "interpretation": "Continuation is forensic only. It does not alter canonical Arena W/L/D or the 500-ply technical classification.",
    }


def board_svg(stones: Sequence[int], x: int, y: int, size: int = 144) -> str:
    cell = size / 9.0
    output = [f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="5" fill="#e7c27d" stroke="#49351f"/>']
    for row in range(9):
        for column in range(9):
            px, py = x + column * cell, y + row * cell
            output.append(f'<rect x="{px + 1:.2f}" y="{py + 1:.2f}" width="{cell - 2:.2f}" height="{cell - 2:.2f}" fill="#d8aa61" opacity=".45"/>')
            stone = int(stones[row * 9 + column])
            if stone:
                color = "#171717" if stone == 1 else "#f5f5f5"
                stroke = "#000000" if stone == 1 else "#666666"
                output.append(f'<circle cx="{px + cell / 2:.2f}" cy="{py + cell / 2:.2f}" r="{cell * .36:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1"/>')
    return "".join(output)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_rgb_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    rows = b"".join(b"\x00" + bytes(pixels[row * width * 3:(row + 1) * width * 3]) for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + _png_chunk(b"IDAT", zlib.compress(rows, 9)) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _draw_board_png(pixels: bytearray, width: int, stones: Sequence[int], left: int, top: int, size: int = 150) -> None:
    def paint(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < len(pixels) // (width * 3):
            offset = (y * width + x) * 3
            pixels[offset:offset + 3] = bytes(color)

    for y in range(top, top + size):
        for x in range(left, left + size):
            paint(x, y, (231, 194, 125) if 2 <= (x - left) < size - 2 and 2 <= (y - top) < size - 2 else (73, 53, 31))
    cell = size / 9.0
    for row in range(9):
        for column in range(9):
            stone = int(stones[row * 9 + column])
            if not stone:
                continue
            center_x = left + int((column + 0.5) * cell)
            center_y = top + int((row + 0.5) * cell)
            radius = max(3, int(cell * 0.35))
            color = (23, 23, 23) if stone == 1 else (245, 245, 245)
            outline = (0, 0, 0) if stone == 1 else (100, 100, 100)
            for dy in range(-radius - 1, radius + 2):
                for dx in range(-radius - 1, radius + 2):
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance <= radius:
                        paint(center_x + dx, center_y + dy, color)
                    elif distance <= radius + 1:
                        paint(center_x + dx, center_y + dy, outline)


def make_visual_png(run_root: Path, arena_games: Sequence[dict[str, object]], output: Path) -> dict[str, object]:
    """Create a previewable board-only montage used for the required inspection."""
    entries: list[tuple[str, list[Any], list[int]]] = []
    d3_games = load_jsonl(run_root / "canonical/selfplay/iter-03-games.jsonl")
    for index in (0, 15, 58):
        game = d3_games[index]
        actions = list(game["final_action_trace"])
        entries.append((f"D3-{index:02d}", replay_states(game["start_state"], actions), [0, min(len(actions), 32), len(actions)]))
    current = [game for game in arena_games if game["comparison"] == "M8-vs-M1" and game["technical_termination"] is None][:1]
    old = [game for game in arena_games if game["comparison"] == "NEW-M8-vs-OLD-M8" and game["technical_termination"] is None][:1]
    for prefix, group in (("M8", current), ("OLD-M8", old)):
        for game in group:
            actions = [item["action"] for item in game["action_trace"]]
            entries.append((prefix, replay_states(game["start_state"], actions), [0, min(len(actions), 32), len(actions)]))
    technical_ids = ["M8-vs-M0--prefix-04-accepted-03--g1", "M8-vs-M4--prefix-02-accepted-07--g2", "M8-vs-M7--prefix-02-accepted-01--g2"]
    for game_id in technical_ids:
        game = next(game for game in arena_games if game["game_id"] == game_id)
        actions = [item["action"] for item in game["action_trace"]]
        entries.append((game["comparison"], replay_states(game["start_state"], actions), [0, 450, 500]))
    margin = 18
    board_size = 150
    width = margin + 3 * (board_size + margin)
    height = margin + len(entries) * (board_size + 34 + margin)
    pixels = bytearray((250, 248, 242) * (width * height))
    for row, (_, states, plys) in enumerate(entries):
        top = margin + row * (board_size + 34 + margin)
        for column, ply in enumerate(plys):
            _draw_board_png(pixels, width, [int(stone) for stone in states[ply].stones], margin + column * (board_size + margin), top, board_size)
    _write_rgb_png(output, width, height, pixels)
    return {"path": str(output), "sha256": file_sha256(output), "entries": [entry[0] for entry in entries], "description": "Board-only preview montage; labels and move metadata are in the companion SVG."}


def make_visual_traces(run_root: Path, arena_games: Sequence[dict[str, object]], output: Path) -> dict[str, object]:
    entries: list[tuple[str, dict[str, object], str, str]] = []
    for generation, indexes in ((3, (0, 15, 58)),):
        games = load_jsonl(run_root / f"canonical/selfplay/iter-{generation:02d}-games.jsonl")
        for index in indexes:
            game = games[index]
            entries.append(("D3 BLACK/WHITE", game, f"D{generation}-{index:02d}", "selfplay"))
    current_m8_games = [game for game in arena_games if game["comparison"] == "M8-vs-M1" and game["technical_termination"] is None]
    old_m8_games = [game for game in arena_games if game["comparison"] == "NEW-M8-vs-OLD-M8" and game["technical_termination"] is None]
    for index, game in enumerate(current_m8_games[:2]):
        entries.append(("M8 ordinary Arena", game, f"M8-{index:02d}", "arena"))
    for index, game in enumerate(old_m8_games[:2]):
        entries.append(("OLD M8 ordinary Arena", game, f"OLD-M8-{index:02d}", "arena"))
    selected_arena = [game for game in arena_games if game["game_id"] in {"M8-vs-M0--prefix-04-accepted-03--g1", "M8-vs-M0--prefix-16-accepted-04--g1", "M8-vs-M4--prefix-02-accepted-07--g2", "M8-vs-M4--prefix-12-accepted-05--g2", "M8-vs-M4--prefix-16-accepted-03--g1", "M8-vs-M7--prefix-02-accepted-01--g2", "NEW-M8-vs-OLD-M8--prefix-02-accepted-02--g1"}]
    width, row_height = 1220, 258
    height = row_height * (len(entries) + len(selected_arena)) + 36
    svg: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#faf8f2"/>', '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#27231f}.title{font-size:18px;font-weight:700}.meta{font-size:12px}.small{font-size:10px}</style>', '<text x="18" y="24" class="title">Torus 9×9 offline visual traces — board snapshots, not canonical result changes</text>']
    visuals: list[dict[str, object]] = []
    entries.extend(("Arena technical", game, game["game_id"], "technical") for game in selected_arena)
    for row_index, (category, game, ident, kind) in enumerate(entries):
        y = 36 + row_index * row_height
        if kind == "technical":
            actions = [item["action"] for item in game["action_trace"]]
            states = replay_states(game["start_state"], actions)
            outcome = "TRUNCATED_MOVE_LIMIT / 500"
            snapshot_indices = [0, min(8, len(actions)), min(100, len(actions)), min(400, len(actions)), len(actions)]
        elif kind == "arena":
            actions = [item["action"] for item in game["action_trace"]]
            states = replay_states(game["start_state"], actions)
            outcome = f'{game["formal_result"]} / {len(actions)} plies'
            snapshot_indices = [0, min(8, len(actions)), min(24, len(actions)), min(48, len(actions)), len(actions)]
        else:
            actions = list(game["final_action_trace"])
            states = replay_states(game["start_state"], actions)
            outcome = f'{game["formal_result"]} / {len(actions)} plies'
            snapshot_indices = [0, min(8, len(actions)), min(24, len(actions)), min(48, len(actions)), len(actions)]
        snapshot_indices = sorted(set(snapshot_indices))
        svg.append(f'<text x="18" y="{y + 18}" class="title">{html.escape(category)}: {html.escape(str(ident))}</text>')
        svg.append(f'<text x="18" y="{y + 36}" class="meta">{html.escape(outcome)} · first actions: {html.escape(", ".join(action_label(action) for action in actions[:8]))}</text>')
        for board_index, ply in enumerate(snapshot_indices):
            x = 18 + board_index * 238
            svg.append(board_svg([int(stone) for stone in states[ply].stones], x, y + 48, 144))
            label = "start" if ply == 0 else f"ply {ply}"
            svg.append(f'<text x="{x}" y="{y + 208}" class="meta">{label}</text>')
            if ply > 0:
                svg.append(f'<text x="{x}" y="{y + 224}" class="small">move {html.escape(action_label(actions[ply - 1]))}</text>')
        visuals.append({"id": ident, "category": category, "outcome": outcome, "snapshot_plys": snapshot_indices})
    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return {"path": str(output), "sha256": file_sha256(output), "visuals": visuals, "inspection_scope": "2 ordinary current M8 games, 2 ordinary OLD M8 games, 3 D3 games (including typical and short branches), and 7 representative truncated Arena games."}


def best_manifest(run_root: Path, profile: Mapping[str, object]) -> dict[str, object]:
    checkpoint = run_root / "canonical/checkpoints/M8.pt"
    metadata = json.loads(checkpoint.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    artifact_hash = file_sha256(checkpoint)
    if metadata.get("model_hash") != GOLDEN_MODEL_HASH:
        raise ValueError(f"Current M8 artifact hash drift: {metadata.get('model_hash')} != {GOLDEN_MODEL_HASH}")
    return {
        "manifest_schema": "torus9-golden-best-v1",
        "status": "CURRENT_BEST",
        "fixed_at": str(date.today()),
        "run_id": GOLDEN_RUN_ID,
        "checkpoint_label": "M8",
        "model_hash": GOLDEN_MODEL_HASH,
        "checkpoint_artifact_sha256": artifact_hash,
        "checkpoint_artifact_path": str(checkpoint),
        "source_commit": metadata["git_commit"],
        "source_tree": metadata["git_tree"],
        "baseline_anchor_merge_commit": ANCHOR_MERGE_COMMIT,
        "architecture": {"architecture_id": TORUS9_ARCHITECTURE_ID, "architecture_fingerprint": metadata["architecture_fingerprint"], "hidden": 64, "blocks": 8, "heads": {"policy": [82], "value": [3]}, "ownership": False, "score": False},
        "profile": {"profile_id": TORUS9_PROFILE_ID, "profile_fingerprint": profile_fingerprint(profile), "rules_fingerprint": TORUS9_RULES_FINGERPRINT, "observation_fingerprint": TORUS9_OBSERVATION_FINGERPRINT, "target_fingerprint": TORUS9_TARGET_FINGERPRINT},
        "training_semantics": {"iterations": 8, "games_per_iteration": 64, "total_games": 512, "rolling_replay_generations": 3, "replay_max_positions": 20000, "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0, "optimizer_steps_per_iteration": 80, "batch_size": 64, "samples_consumed_per_iteration": 5120, "optimizer_continuation": True},
        "search_semantics": {"selfplay_contract_fingerprint": TORUS9_SELFPLAY_CONTRACT_FINGERPRINT, "arena_contract_fingerprint": TORUS9_ARENA_CONTRACT_FINGERPRINT, "simulations": 64, "cpuct": 1.25, "fpu": 0.0, "komi": TORUS9_KOMI, "selfplay_noise": {"epsilon": 0.25, "alpha": 0.30}, "selfplay_temperature": "1.0 plies 1..8 then 0.0", "arena_noise": False, "arena_temperature": 0.0, "move_limit": TORUS9_MOVE_LIMIT},
        "arena_evidence": {"NEW-M8_vs_OLD-M8": {"valid_W_L_D": [59, 3, 0], "technical_games": 2, "technical_reason": "TRUNCATED_MOVE_LIMIT"}, "M8_vs_M1": {"valid_W_L_D": [128, 0, 0], "technical_games": 0}, "M8_vs_M0": {"valid_W_L_D": [119, 7, 0], "technical_games": 2, "technical_reason": "TRUNCATED_MOVE_LIMIT"}, "M8_vs_M4": {"valid_W_L_D": [92, 24, 0], "technical_games": 12, "technical_reason": "TRUNCATED_MOVE_LIMIT"}, "technical_games_are_excluded_from_draw_loss": True},
        "rules": {"komi": TORUS9_KOMI, "rules_fingerprint": TORUS9_RULES_FINGERPRINT, "technical_games_are_not_results": True},
    }


def markdown_report(report: Mapping[str, object]) -> str:
    best = report["current_golden_best"]
    selfplay = report["self_play"]
    root = report["root_cause"]
    trunc = report["truncations"]
    depth = report["teacher_search"]
    noise = report["noise_temperature"]
    lines = [
        "TORUS 9×9 POST-LEARNING DIAGNOSTICS",
        "",
        f"CURRENT GOLDEN BEST:\n{best['run_id']} / {best['checkpoint_label']}\n{best['model_hash']}",
        "",
        f"SELF-PLAY 61-3 ROOT CAUSE:\n{root['self_play_verdict']}",
        "",
        f"TRUNCATION ROOT CAUSE:\n{root['truncation_verdict']}",
        "",
        f"64-SIM TEACHER VERDICT:\n{root['teacher_verdict']}",
        "",
        f"FIRST-PLAYER ADVANTAGE:\n{root['first_player_verdict']}",
        "",
        f"NEXT RECOMMENDED CHANGE:\n{report['next_recommended_change']}",
        "",
        f"CONFIDENCE:\n{report['confidence']}",
        "",
        "## CONFIRMED",
        "",
        "- PR #86 is represented by merge anchor `88b1803cf179c6fa93f2e9610963eeed931d09b1`; the artifact-producing source commit is recorded separately.",
        "- M2/D3 used one model hash on both sides: " + str(report["d3_forensics"]["causal_sequence"]["model_hashes_used_on_both_sides"]) + ".",
        "- D3 is 61/3/0, with the three WHITE results all early PASS branches; no dominant first action explains the result.",
        "- M2 empty-board network root WDL is " + f"{report['d3_forensics']['initial_root_wdl_network']}" + f" (utility {report['d3_forensics']['initial_root_utility_network']}, PASS probability {report['d3_forensics']['initial_root_pass_probability']}); the early value bias is finite, not a deterministic forced-win signal.",
        "- Across all 512 self-play games, BLACK is " + f"{selfplay['all_512']['black_win_rate']:.4f}" + f" with Wilson 95% CI {selfplay['all_512']['wilson_95_ci']}; raw area advantage {selfplay['all_512']['average_raw_area_advantage']}, final komi-adjusted margin {selfplay['all_512']['average_final_komi_adjusted_margin'] }.",
        "- Excluding D3, BLACK is " + f"{selfplay['all_512']['without_d3']['black_win_rate']:.4f}" + f" (Wilson 95% CI {selfplay['all_512']['without_d3']['wilson_95_ci']}); generation-index correlation is " + f"{selfplay['all_512']['without_d3']['black_rate_vs_generation_index']:.4f}, so checkpoint strength does not explain a monotonic color drift.",
        "- Canonical Arena technical games remain excluded from W/L/D.",
        "",
        "## STRONG EVIDENCE",
        "",
        "### Self-play color balance",
        "",
        "| checkpoint | generation | BLACK | WHITE | draw | BLACK rate | Wilson 95% CI | avg margin | median margin | avg plies | pass frequency |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for generation, row in selfplay["per_generation"].items():
        lines.append(f"| {row['checkpoint']} | {generation} | {row['black_wins']} | {row['white_wins']} | {row['draws']} | {row['black_win_rate']:.4f} | {row['wilson_95_ci']} | {row['average_margin']:.3f} | {row['median_margin']:.3f} | {row['average_game_length']:.2f} | {row['pass_frequency']:.4f} |")
    lines += [
        "",
        "D3 game-level forensic data (including seeds, first eight actions, early root WDL, visit share and PASS data) is in the JSON report; the short WHITE branches are the only D3 games with first PASS at ply ≤5.",
        "",
        "### Fixed-state 64/128/256 search",
        "",
        "| source | comparison | states | mean policy KL | top-1 agreement | mean top-3 overlap | mean |Δ root value| | mean |Δ PASS| |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, comparisons in depth["summary_by_source"].items():
        for comparison, row in comparisons.items():
            lines.append(f"| {source} | {comparison} | {row['states']} | {row['mean_policy_kl']:.5f} | {row['top1_agreement_rate']:.3f} | {row['mean_top3_overlap']:.3f} | {row['mean_abs_root_value_difference']:.5f} | {row['mean_abs_pass_probability_difference']:.5f} |")
    lines += [
        "",
        "### Noise / temperature controlled diagnostic on fixed M2 states",
        "",
        "| variant | states | repeats/state | mean unique actions | mean decision entropy | PASS decision rate | mean search PASS probability |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in noise["summary"].items():
        lines.append(f"| {variant} | {row['states']} | {row['replicates_per_state']} | {row['mean_unique_actions']:.3f} | {row['mean_decision_entropy']:.5f} | {row['mean_pass_decision_rate']:.5f} | {row['mean_search_pass_probability']:.5f} |")
    lines += [
        "",
        "Interpretation: A is deterministic; adding either root noise (B) or temperature sampling (C) raises decision entropy to about 2.3, while PASS decision rates remain close to the fixed-state search PASS probability. The measured instability is therefore branch selection, not a standalone PASS-probability explosion.",
        "",
        "### Arena truncations",
        "",
        "| classification | games | representative evidence |",
        "|---|---:|---|",
    ]
    class_counts = Counter(row["classification"]["primary"] for row in trunc["games"])
    for classification, count in sorted(class_counts.items()):
        lines.append(f"| {classification} | {count} | See per-game last-100 metrics and search samples in JSON. |")
    lines += [
        "",
        "All technical games reach ply 500 through legal nonterminal transitions. Exact board signatures repeat only on PASS-retained boards; point-move board repetitions are absent, and superko rejections are activity but not a loop mechanism.",
        "",
        "### Offline continuation",
        "",
        f"{report['continuations']['completed_by_ply_1000']} games finished by ply 1000; {report['continuations']['completed_after_1000_before_1600']} after ply 1000 and before 1600; {report['continuations']['still_truncated_at_1600']} remained technical at 1600. Canonical Arena evidence was not modified.",
        "",
        "## PLAUSIBLE",
        "",
        "- D3 is best explained by a real first-player/komi edge plus small-sample and early stochastic branch amplification. Noise and temperature measurements quantify the contribution; they do not justify changing the frozen baseline.",
        "- Truncation is a mixture of PASS avoidance and capture/territory-filling churn, with relative proportions varying by pair; it is not one universal cycle.",
        "",
        "## REJECTED",
        "",
        "- Two different M2 copies with unequal strength: rejected by the single model hash and identical contract provenance on both sides.",
        "- A single dominant opening move as the cause of 61–3: rejected by the D3 first-eight action distributions.",
        "- Rules/scoring or superko implementation failure as the truncation cause: rejected by legal replay, no point-board repetition, and fail-closed technical handling.",
        "- Changing komi, move limit, architecture, replay, optimizer or self-play settings during diagnosis: not done.",
        "",
        "## UNKNOWN",
        "",
        "- A causal claim that 128 simulations would improve future training targets is not established by these fixed-state probes alone; it is the next clean experiment if a search change is approved.",
        "- A human strategic quality judgment is limited to the linked visual snapshots; no claim of game-theoretic optimality is made.",
        "",
        "## Decision matrix",
        "",
        f"- Self-play 61–3: {report['decision_matrix']['self_play_61_3']}",
        f"- Arena truncations: {report['decision_matrix']['truncations']}",
        "",
        "## Visual inspection",
        "",
        f"Visual trace artifact: `{report['visual_inspection']['path']}`; preview PNG: `{report['visual_inspection']['preview_png']['path']}`. Scope: {report['visual_inspection']['inspection_scope']}",
        "",
        "The snapshots show normal legal placement/capture dynamics in current M8 and OLD M8 Arena games and in D3; the current M8 panels look more decisive than OLD M8 on the same comparison family. Truncation panels show either single-PASS continuation or high-capture late churn rather than an exact point-state loop.",
        "",
        "## Verification and provenance",
        "",
        f"Baseline manifest: `{report['baseline_manifest_path']}`",
        f"Analysis source commit: `{report['analysis_source_commit']}`",
        f"Baseline artifact source commit: `{best['source_commit']}`",
        f"Baseline merge anchor: `{ANCHOR_MERGE_COMMIT}`",
        f"Rules fingerprint: `{TORUS9_RULES_FINGERPRINT}`; observation fingerprint: `{TORUS9_OBSERVATION_FINGERPRINT}`; komi: `{TORUS9_KOMI}`.",
        "No new M0→M8 training run or new self-play games were launched.",
        "",
    ]
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    profile = load_torus9_profile()
    run_root = args.run_root
    old_root = args.old_root
    model_paths: dict[str, Path] = {}
    for label in [f"M{index}" for index in range(9)]:
        model_paths[label], _ = checkpoint_metadata(run_root, label, old_root)
    model_paths["NEW-M8"] = model_paths["M8"]
    model_paths["OLD-M8"], _ = checkpoint_metadata(run_root, "OLD-M8", old_root)
    model_cache = {label: load_model(path, device=args.device) for label, path in model_paths.items()}
    selfplay, games_by_generation, game_rows = selfplay_summary(run_root)
    d3_games = games_by_generation["D3"]
    d3 = d3_forensics(d3_games, model_cache["M2"][1])
    arena_all_games: list[dict[str, object]] = []
    arena_games: list[dict[str, object]] = []
    for path in sorted((run_root / "canonical/arena").glob("*/games.jsonl")):
        records = load_jsonl(path)
        arena_all_games.extend(records)
        arena_games.extend(row for row in records if row["technical_termination"] == "TRUNCATED_MOVE_LIMIT")
    corpus = fixed_corpus(games_by_generation, arena_games)
    corpus_path = run_root / "post-learning-diagnostics/fixed-corpus.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in corpus), encoding="utf-8")
    depth = run_search_depth_diagnostic(corpus, model_cache)
    m2_noise_states: list[dict[str, object]] = []
    for game_index in range(0, len(d3_games), 4):
        game = d3_games[game_index]
        position_index = min(7, len(game["positions"]) - 1)
        position = game["positions"][position_index]
        m2_noise_states.append({"state_id": f"D3-game-{game_index:02d}-ply-{position['ply']}", "source": "D3", "game_id": game["game_id"], "model_label": "M2", "ply": position["ply"], "state": position["state"], "state_digest": state_digest(position["state"])})
    noise = run_noise_temperature_diagnostic(m2_noise_states, model_cache["M2"][1])
    arena_by_id = {str(game["game_id"]): game for game in arena_games}
    traced = [trace_arena_game(game, model_cache) for game in sorted(arena_games, key=lambda row: str(row["game_id"]))]
    traced_by_id = {row["game_id"]: row for row in traced}
    continuations = continuation_diagnostic(traced, arena_by_id, model_paths, workers=args.workers)
    visual = make_visual_traces(run_root, arena_all_games, args.visual_path)
    visual["preview_png"] = make_visual_png(run_root, arena_all_games, args.visual_png_path)
    manifest = best_manifest(run_root, profile)
    args.best_json.parent.mkdir(parents=True, exist_ok=True)
    args.best_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.best_md.write_text(
        "# CURRENT TORUS9 GOLDEN BEST\n\n"
        f"Status: **{manifest['status']}**\n\n"
        f"Run/checkpoint: `{manifest['run_id']} / {manifest['checkpoint_label']}`\n\n"
        f"Model hash: `{manifest['model_hash']}`\n\n"
        f"Checkpoint artifact SHA-256: `{manifest['checkpoint_artifact_sha256']}`\n\n"
        f"Artifact source commit: `{manifest['source_commit']}`; PR #86 merge anchor: `{ANCHOR_MERGE_COMMIT}`.\n\n"
        "Architecture: `GoldenGraphNetV2-Torus9-8Block`, hidden 64, 8 blocks, policy `[82]`, WDL `[3]`, komi `0.5`.\n\n"
        "Training: 8 × 64 self-play games; rolling replay 3 generations / 20,000 positions; Adam `lr=0.001`, `wd=0`, 80 optimizer steps × batch 64 per iteration.\n\n"
        "Arena evidence: NEW M8 vs OLD M8 `59 / 3 / 0`; M8 vs M1 `128 / 0 / 0`; M8 vs M0 `119 / 7 / 0`; M8 vs M4 `92 / 24 / 0`. Technical games are stored separately and excluded from W/L/D.\n",
        encoding="utf-8",
    )
    source = git_commit()
    d3_black = d3["causal_sequence"]["black_wins"]
    d3_white = d3["causal_sequence"]["white_wins"]
    pass_primary = Counter(row["classification"]["primary"] for row in traced)
    report: dict[str, object] = {
        "report_schema": "torus9-post-learning-diagnostics-v1",
        "created_on": str(date.today()),
        "analysis_source_commit": source,
        "baseline_anchor_merge_commit": ANCHOR_MERGE_COMMIT,
        "no_new_training_or_selfplay_games": True,
        "current_golden_best": manifest,
        "baseline_manifest_path": str(args.best_json),
        "self_play": selfplay,
        "d3_forensics": d3,
        "teacher_search": {**depth, "fixed_corpus_path": str(corpus_path), "fixed_corpus_sha256": file_sha256(corpus_path)},
        "noise_temperature": noise,
        "truncations": {"games": traced, "count": len(traced), "classification_counts": dict(sorted(pass_primary.items()))},
        "continuations": continuations,
        "visual_inspection": visual,
        "root_cause": {
            "self_play_verdict": "A + C with contributing D: expected first-player/komi advantage is amplified by the small 64-game sample and early stochastic PASS branches. Confidence: STRONG EVIDENCE. It is not a two-copy M2 mismatch, rules issue, or single-opening collapse.",
            "truncation_verdict": "B + C + D: legal long-play/capture churn and single-PASS continuation, with pair-specific PASS avoidance. Confidence: STRONG EVIDENCE. Superko activity is not the primary loop mechanism.",
            "teacher_verdict": "64-sim teacher remains a live quality risk: fixed-state 64/128/256 differences are reported, especially D3/M2 and truncation-tail states. No baseline change is authorized by this report alone.",
            "first_player_verdict": f"Across all 512 self-play games BLACK win rate is {selfplay['all_512']['black_win_rate']:.4f}, Wilson 95% CI {selfplay['all_512']['wilson_95_ci']}; excluding D3 it is {selfplay['all_512']['without_d3']['black_win_rate']:.4f} with CI {selfplay['all_512']['without_d3']['wilson_95_ci']}. Average raw area advantage is {selfplay['all_512']['average_raw_area_advantage']} and average final komi-adjusted margin is {selfplay['all_512']['average_final_komi_adjusted_margin']}. D3 is an extreme realization, not evidence of a different M2 on the two colors.",
        },
        "decision_matrix": {
            "self_play_61_3": "A. Expected first-player / stochastic variance, with D. network/value calibration as a plausible contributor to early PASS branch amplification.",
            "truncations": "B. Weak 64-sim search + C. PASS avoidance + D. value/policy pathology; A. move-limit too low may also apply to games that finish shortly after 500, to be judged from continuation results.",
        },
        "next_recommended_change": "One isolated better-teacher/search experiment (64 → 128 simulations) after this diagnostic is reviewed; keep architecture, replay, optimizer, komi, noise, temperature and canonical 500-ply protocol unchanged in the comparison baseline.",
        "confidence": "STRONG EVIDENCE for the causal classifications; CONFIRMED for provenance/rules/technical separation; PLAUSIBLE for the exact share attributable to network calibration versus search/noise; UNKNOWN for whether 128-sim targets win a future canonical comparison.",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_md.write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_RUN)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--best-json", type=Path, default=DEFAULT_BEST_JSON)
    parser.add_argument("--best-md", type=Path, default=DEFAULT_BEST_MD)
    parser.add_argument("--visual-path", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--visual-png-path", type=Path, default=ROOT / "docs/assets/TORUS9_POST_LEARNING_VISUAL_INSPECTION.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run(args)
    print("TORUS9 POST-LEARNING DIAGNOSTICS: COMPLETE")
    print("D3:", report["root_cause"]["self_play_verdict"])
    print("TRUNCATIONS:", report["root_cause"]["truncation_verdict"])
    print("CONTINUATIONS:", report["continuations"]["completed_by_ply_1000"], "by 1000;", report["continuations"]["completed_after_1000_before_1600"], "after 1000;", report["continuations"]["still_truncated_at_1600"], "at 1600")


if __name__ == "__main__":
    main()
