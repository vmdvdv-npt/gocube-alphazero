#!/usr/bin/env python3
"""Offline Torus9 effective-komi calibration.

The tool only reads completed trajectories and, when necessary, runs the
explicitly separate frozen M8-vs-M8 diagnostic.  It never changes training
komi, replays, checkpoints, or historical Arena records.  Every non-0.5
komi in the output is counterfactual rescoring of an already played trajectory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.provenance import capture_code_identity, file_sha256
from gocube_golden.rules import apply_action
from gocube_golden.scoring import score_terminal
from gocube_golden.torus9 import (
    generate_torus9_evaluation_starts,
    run_torus9_arena,
    torus9_checkpoint_info,
    torus9_state_from_identity,
)
from gocube_golden.torus9_contract import (
    TORUS9_ARENA_CONTRACT_FINGERPRINT,
    TORUS9_ARENA_CONTRACT_ID,
    TORUS9_ARENA_MOVE_LIMIT,
    TORUS9_KOMI,
    TORUS9_PROFILE_ID,
    TORUS9_WORKERS,
    load_torus9_profile,
    profile_fingerprint,
)


DEFAULT_RUN_ROOT = ROOT / "runs/torus9-stable-learning-v2/torus9-stable-learning-20260913-v1"
DEFAULT_ARENA_ROOT = DEFAULT_RUN_ROOT / "canonical/arena"
DEFAULT_PAIRED_ROOT = ROOT / "runs/torus9-komi-calibration-20260913"
DEFAULT_REPORT_JSON = ROOT / "docs/TORUS9_KOMI_CALIBRATION_20260913.json"
DEFAULT_REPORT_MD = ROOT / "docs/TORUS9_KOMI_CALIBRATION_20260913.md"
DEFAULT_BOOTSTRAP_SEED = 20260913
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
KOMI_SWEEP = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
PAIR_START_SEED = 2026091311


@dataclass(frozen=True)
class Trajectory:
    dataset: str
    record_id: str
    generation: str | None
    source: str
    winner_at_training_komi: str
    raw_area_advantage: float
    black_area: int
    white_area: int
    plies: int
    pair_id: str | None = None


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson_interval(wins: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    rate = wins / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [round(max(0.0, centre - radius), 8), round(min(1.0, centre + radius), 8)]


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires non-empty values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def classify_margin(raw_area_advantage: float, komi: float) -> str:
    margin = float(raw_area_advantage) - float(komi)
    if margin > 0.0:
        return "BLACK"
    if margin < 0.0:
        return "WHITE"
    return "DRAW"


def crossing_point(sweep: Mapping[float, Mapping[str, object]]) -> float | None:
    points = sorted((float(komi), float(row["black_win_rate"])) for komi, row in sweep.items())
    for komi, rate in points:
        if math.isclose(rate, 0.5, abs_tol=1e-15):
            return komi
    for (left_komi, left_rate), (right_komi, right_rate) in zip(points, points[1:]):
        if left_rate > 0.5 and right_rate < 0.5:
            fraction = (left_rate - 0.5) / (left_rate - right_rate)
            return left_komi + fraction * (right_komi - left_komi)
        if left_rate >= 0.5 >= right_rate:
            if left_rate == right_rate:
                return (left_komi + right_komi) / 2.0
            fraction = (left_rate - 0.5) / (left_rate - right_rate)
            return left_komi + fraction * (right_komi - left_komi)
    return None


def crossing_interval(sweep: Mapping[float, Mapping[str, object]]) -> list[float] | None:
    points = sorted((float(komi), float(row["black_win_rate"])) for komi, row in sweep.items())
    for komi, rate in points:
        if math.isclose(rate, 0.5, abs_tol=1e-15):
            return [komi, komi]
    for (left_komi, left_rate), (right_komi, right_rate) in zip(points, points[1:]):
        if left_rate >= 0.5 >= right_rate:
            return [left_komi, right_komi]
    return None


def sweep_trajectories(trajectories: Sequence[Trajectory]) -> dict[str, object]:
    raw = [float(row.raw_area_advantage) for row in trajectories]
    by_komi: dict[str, object] = {}
    typed_sweep: dict[float, dict[str, object]] = {}
    for komi in KOMI_SWEEP:
        black = sum(classify_margin(value, komi) == "BLACK" for value in raw)
        white = sum(classify_margin(value, komi) == "WHITE" for value in raw)
        ties = len(raw) - black - white
        margins = [value - komi for value in raw]
        row = {
            "komi": komi,
            "black_wins": black,
            "white_wins": white,
            "ties": ties,
            "games": len(raw),
            "black_win_rate": round(black / len(raw), 8) if raw else None,
            "wilson_95_ci": wilson_interval(black, len(raw)),
            "mean_counterfactual_margin_black": round(statistics.mean(margins), 8) if margins else None,
            "median_counterfactual_margin_black": round(statistics.median(margins), 8) if margins else None,
        }
        by_komi[str(komi)] = row
        typed_sweep[komi] = row
    frequencies = Counter(raw)
    maximum_frequency = max(frequencies.values(), default=0)
    modes = sorted(value for value, count in frequencies.items() if count == maximum_frequency)
    distribution = {
        "count": len(raw),
        "mean": round(statistics.mean(raw), 8) if raw else None,
        "median": round(statistics.median(raw), 8) if raw else None,
        "mode": modes,
        "mode_frequency": maximum_frequency,
        "p25": round(quantile(raw, 0.25), 8) if raw else None,
        "p50": round(quantile(raw, 0.50), 8) if raw else None,
        "p75": round(quantile(raw, 0.75), 8) if raw else None,
        "min": min(raw) if raw else None,
        "max": max(raw) if raw else None,
        "histogram": {str(value): frequencies[value] for value in sorted(frequencies)},
    }
    point = crossing_point(typed_sweep)
    result = {
        "games": len(trajectories),
        "technical_games_excluded": 0,
        "komi_sweep": by_komi,
        "raw_area_advantage_distribution": distribution,
        "implied_fair_komi": {
            "mean_based": distribution["mean"],
            "median_based": distribution["median"],
            "win_rate_crossing_point": round(point, 8) if point is not None else None,
            "win_rate_crossing_interval": crossing_interval(typed_sweep),
        },
        "counterfactual_only": True,
        "training_komi": TORUS9_KOMI,
    }
    if trajectories and all(row.pair_id is not None for row in trajectories):
        grouped: dict[str, list[Trajectory]] = {}
        for row in trajectories:
            grouped.setdefault(str(row.pair_id), []).append(row)
        if all(len(rows) == 2 for rows in grouped.values()):
            pair_rows: dict[str, object] = {}
            pair_sweep: dict[float, dict[str, object]] = {}
            for komi in KOMI_SWEEP:
                pair_black_fractions = [
                    sum(classify_margin(row.raw_area_advantage, komi) == "BLACK" for row in pair) / 2.0
                    for _, pair in sorted(grouped.items())
                ]
                pair_sweep[komi] = {
                    "pair_count": len(pair_black_fractions),
                    "mean_black_win_fraction": round(statistics.mean(pair_black_fractions), 8),
                    "black_win_rate": statistics.mean(pair_black_fractions),
                    "black_wins": sum(classify_margin(row.raw_area_advantage, komi) == "BLACK" for row in trajectories),
                    "white_wins": sum(classify_margin(row.raw_area_advantage, komi) == "WHITE" for row in trajectories),
                    "ties": sum(classify_margin(row.raw_area_advantage, komi) == "DRAW" for row in trajectories),
                }
            pair_rows = {
                "unit": "independent_color-swapped start pair",
                "pairs": len(grouped),
                "komi_sweep": {str(komi): row for komi, row in pair_sweep.items()},
                "crossing_interval": crossing_interval(pair_sweep),
                "crossing_point": crossing_point(pair_sweep),
            }
            result["paired_analysis"] = pair_rows
    return result


def bootstrap_units(trajectories: Sequence[Trajectory], pairwise: bool) -> list[list[float]]:
    if not pairwise:
        return [[float(row.raw_area_advantage)] for row in trajectories]
    grouped: dict[str, list[float]] = {}
    for row in trajectories:
        if row.pair_id is None:
            raise ValueError("Paired bootstrap requires pair_id on every trajectory")
        grouped.setdefault(row.pair_id, []).append(float(row.raw_area_advantage))
    if any(len(values) != 2 for values in grouped.values()):
        raise ValueError("Paired bootstrap requires exactly two terminal games per pair")
    return [values for _, values in sorted(grouped.items())]


def bootstrap_estimates(
    trajectories: Sequence[Trajectory],
    *,
    seed: int,
    resamples: int,
    pairwise: bool = False,
) -> dict[str, object]:
    units = bootstrap_units(trajectories, pairwise)
    if not units:
        return {"resamples": resamples, "seed": seed, "pairwise": pairwise, "valid_crossing_resamples": 0}
    import random

    rng = random.Random(seed)
    means: list[float] = []
    medians: list[float] = []
    crossings: list[float] = []
    for _ in range(resamples):
        selected = [units[rng.randrange(len(units))] for _ in units]
        values = [value for unit in selected for value in unit]
        means.append(statistics.mean(values))
        medians.append(statistics.median(values))
        estimate = crossing_point({
            komi: {
                "black_win_rate": sum(classify_margin(value, komi) == "BLACK" for value in values) / len(values),
            }
            for komi in KOMI_SWEEP
        })
        if estimate is not None:
            crossings.append(estimate)

    def interval(values: Sequence[float]) -> list[float | None]:
        return [round(quantile(values, 0.025), 8), round(quantile(values, 0.975), 8)] if values else [None, None]

    return {
        "resamples": resamples,
        "seed": seed,
        "pairwise": pairwise,
        "sampling_unit_count": len(units),
        "valid_crossing_resamples": len(crossings),
        "mean_raw_advantage_95_percent_ci": interval(means),
        "median_raw_advantage_95_percent_ci": interval(medians),
        "crossing_point_95_percent_ci": interval(crossings),
    }


def _score_selfplay_game(game: Mapping[str, object]) -> tuple[int, int, float]:
    state = torus9_state_from_identity(game["start_state"])
    for action in game["final_action_trace"]:  # type: ignore[union-attr]
        state = apply_action(state, action).after
    if not state.is_terminal:
        raise ValueError(f"Self-play game is not terminal: {game.get('game_id')}")
    score = score_terminal(state)
    return int(score.black_area), int(score.white_area), float(score.black_area - score.white_area)


def load_selfplay_trajectories(run_root: Path) -> list[Trajectory]:
    rows: list[Trajectory] = []
    for generation in range(1, 9):
        path = run_root / f"canonical/selfplay/iter-{generation:02d}-games.jsonl"
        for game in load_jsonl(path):
            if game.get("technical_termination") is not None:
                continue
            black_area, white_area, raw = _score_selfplay_game(game)
            rows.append(Trajectory(
                dataset="selfplay",
                record_id=str(game["game_id"]),
                generation=f"D{generation}",
                source=str(path),
                winner_at_training_komi=str(game["formal_result"]),
                raw_area_advantage=raw,
                black_area=black_area,
                white_area=white_area,
                plies=len(game["final_action_trace"]),  # type: ignore[arg-type]
            ))
    if len(rows) != 512:
        raise ValueError(f"Expected 512 terminal self-play games, got {len(rows)}")
    return rows


def load_arena_trajectories(arena_root: Path) -> list[Trajectory]:
    rows: list[Trajectory] = []
    for path in sorted(arena_root.glob("*/games.jsonl")):
        comparison = path.parent.name
        for game in load_jsonl(path):
            if game.get("technical_termination") is not None or game.get("formal_result") is None:
                continue
            if game.get("black_area") is None or game.get("white_area") is None:
                raise ValueError(f"Terminal Arena row lacks area scores: {path}:{game.get('game_id')}")
            rows.append(Trajectory(
                dataset="arena",
                record_id=str(game["game_id"]),
                generation=None,
                source=comparison,
                winner_at_training_komi=str(game["formal_result"]),
                raw_area_advantage=float(game["black_area"]) - float(game["white_area"]),
                black_area=int(game["black_area"]),
                white_area=int(game["white_area"]),
                plies=len(game["action_trace"]),  # type: ignore[arg-type]
                pair_id=str(game["pair_id"]),
            ))
    return rows


def load_paired_trajectories(path: Path) -> list[Trajectory]:
    rows: list[Trajectory] = []
    for game in load_jsonl(path / "games.jsonl"):
        if game.get("technical_termination") is not None or game.get("formal_result") is None:
            continue
        if game.get("black_area") is None or game.get("white_area") is None:
            raise ValueError(f"Paired terminal row lacks area scores: {game.get('game_id')}")
        rows.append(Trajectory(
            dataset="paired_m8",
            record_id=str(game["game_id"]),
            generation=None,
            source="M8-vs-M8-paired",
            winner_at_training_komi=str(game["formal_result"]),
            raw_area_advantage=float(game["black_area"]) - float(game["white_area"]),
            black_area=int(game["black_area"]),
            white_area=int(game["white_area"]),
            plies=len(game["action_trace"]),  # type: ignore[arg-type]
            pair_id=str(game["pair_id"]),
        ))
    return rows


def ensure_paired_diagnostic(
    *,
    run_root: Path,
    output_root: Path,
    master_seed: int,
    workers: int,
    candidate_crossing_interval: Sequence[float],
    candidate_point_estimate: float | None,
) -> tuple[Path, dict[str, object]]:
    output = output_root / "m8-vs-m8-paired"
    games_path = output / "games.jsonl"
    starts_path = output / "starts.jsonl"
    model_path = run_root / "canonical/checkpoints/M8.pt"
    if not games_path.exists():
        starts = generate_torus9_evaluation_starts(master_seed=master_seed, accepted_per_stratum=8)
        starts_path.parent.mkdir(parents=True, exist_ok=True)
        starts_path.write_text("".join(json.dumps(start, sort_keys=True) + "\n" for start in starts), encoding="utf-8")
        plan = {
            "status": "FIXED_BEFORE_RUN",
            "candidate_crossing_interval": list(candidate_crossing_interval),
            "candidate_point_estimate": candidate_point_estimate,
            "independent_pairs": 64,
            "games": 128,
            "adaptive_extension": False,
            "canonical_komi": TORUS9_KOMI,
            "search_contract": {
                "simulations": 64,
                "noise": False,
                "temperature": 0.0,
                "fast": False,
                "workers": workers,
                "watchdog": TORUS9_ARENA_MOVE_LIMIT,
            },
        }
        (output / "diagnostic-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = run_torus9_arena(
            run_id="torus9-komi-calibration-20260913-m8-paired",
            comparison="M8-vs-M8-PAIRED",
            candidate_path=model_path,
            reference_path=model_path,
            candidate_label="M8",
            reference_label="M8",
            starts=starts,
            master_seed=master_seed,
            output_dir=output,
            workers=workers,
            device="cpu",
        )
    else:
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metadata = {
        "run_id": "torus9-komi-calibration-20260913-m8-paired",
        "model_label": "M8",
        "model_hash": torus9_checkpoint_info(model_path)["model_hash"],
        "model_artifact_sha256": file_sha256(model_path),
        "master_seed": master_seed,
        "independent_pairs": 64,
        "games_declared": 128,
        "fixed_plan": json.loads((output / "diagnostic-plan.json").read_text(encoding="utf-8")) if (output / "diagnostic-plan.json").exists() else None,
        "search_contract": {
            "simulations": 64,
            "cpuct": 1.25,
            "fpu": 0.0,
            "noise": False,
            "temperature": 0.0,
            "fast": False,
            "workers": workers,
            "watchdog": TORUS9_ARENA_MOVE_LIMIT,
            "technical_fail_closed": True,
            "adaptive_extension": False,
            "arena_contract_id": TORUS9_ARENA_CONTRACT_ID,
            "arena_contract_fingerprint": TORUS9_ARENA_CONTRACT_FINGERPRINT,
        },
        "starts_path": str(starts_path),
        "games_path": str(games_path),
        "starts_sha256": file_sha256(starts_path) if starts_path.exists() else None,
        "games_sha256": file_sha256(games_path),
        "start_prefix_length_histogram": dict(sorted(Counter(
            int(row["prefix_length"])
            for row in load_jsonl(starts_path)
        ).items())) if starts_path.exists() else {},
        "summary": summary,
    }
    (output / "diagnostic-manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, metadata


def winner_flips(trajectories: Sequence[Trajectory], candidate_komi: float) -> dict[str, object]:
    def changed(row: Trajectory) -> bool:
        return classify_margin(row.raw_area_advantage, TORUS9_KOMI) != classify_margin(row.raw_area_advantage, candidate_komi)

    by_generation: dict[str, dict[str, int]] = {}
    for generation in sorted({row.generation for row in trajectories if row.generation is not None}):
        group = [row for row in trajectories if row.generation == generation]
        by_generation[str(generation)] = {"games": len(group), "winner_flips": sum(changed(row) for row in group)}
    return {
        "from_komi": TORUS9_KOMI,
        "to_komi": candidate_komi,
        "games": len(trajectories),
        "winner_flips": sum(changed(row) for row in trajectories),
        "by_generation": by_generation,
        "counterfactual_only": True,
    }


def build_report(
    *,
    run_root: Path,
    arena_root: Path,
    paired_root: Path,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    paired_metadata: Mapping[str, object],
) -> dict[str, object]:
    selfplay = load_selfplay_trajectories(run_root)
    arena = load_arena_trajectories(arena_root)
    paired = load_paired_trajectories(paired_root)
    datasets: dict[str, list[Trajectory]] = calibration_datasets(selfplay, arena, paired)
    analyses: dict[str, object] = {}
    for index, (name, rows) in enumerate(datasets.items()):
        analysis = sweep_trajectories(rows)
        analysis["bootstrap"] = bootstrap_estimates(
            rows,
            seed=bootstrap_seed + index,
            resamples=bootstrap_resamples,
            pairwise=name == "M8_vs_M8_paired",
        )
        analyses[name] = analysis

    all_crossing = analyses["all_512_selfplay"]["implied_fair_komi"]["win_rate_crossing_point"]  # type: ignore[index]
    candidate = float(all_crossing) if all_crossing is not None else float(statistics.median([row.raw_area_advantage for row in selfplay]))
    generation_analysis: dict[str, object] = {}
    for generation in [f"D{i}" for i in range(1, 9)]:
        group = [row for row in selfplay if row.generation == generation]
        generation_analysis[generation] = sweep_trajectories(group)

    code = capture_code_identity()
    profile = load_torus9_profile()
    all_interval = analyses["all_512_selfplay"]["implied_fair_komi"]["win_rate_crossing_interval"]  # type: ignore[index]
    paired_interval = analyses["M8_vs_M8_paired"]["implied_fair_komi"]["win_rate_crossing_interval"]  # type: ignore[index]
    confirms = bool(all_interval and paired_interval and set(paired_interval) == set(all_interval))
    recommended_range = list(all_interval) if all_interval else None
    confidence = "HIGH" if confirms else ("MEDIUM" if all_interval else "LOW")
    paired_contradicts = bool(paired and all_interval and (not paired_interval or not confirms))
    if paired_contradicts:
        confidence = "INCONCLUSIVE"
    report = {
        "report_schema": "torus9-komi-calibration-v1",
        "created_on": str(date.today()),
        "source": {
            "git_commit": code.git_commit_sha,
            "git_tree": code.git_tree_sha,
            "worktree_clean": code.working_tree_clean,
            "calibration_tool": str(Path(__file__).relative_to(ROOT)),
            "calibration_tool_sha256": file_sha256(Path(__file__)),
            "run_root": str(run_root),
            "arena_root": str(arena_root),
        },
        "golden_historical_komi": TORUS9_KOMI,
        "golden_baseline": {
            "run_id": "torus9-stable-learning-20260913-v1",
            "checkpoint": "M8",
            "status": "CURRENT TORUS9 GOLDEN BEST",
            "training_komi_unchanged": True,
            "historical_artifacts_rewritten": False,
        },
        "arena_watchdog": {
            "legacy_universal_limit": 500,
            "new_board_size_scaled_limit": TORUS9_ARENA_MOVE_LIMIT,
            "board_size": [9, 9],
            "contract_id": TORUS9_ARENA_CONTRACT_ID,
            "contract_fingerprint": TORUS9_ARENA_CONTRACT_FINGERPRINT,
            "technical_fail_closed": True,
            "technical_outcomes_are_wdl": False,
            "normal_double_pass_before_watchdog_is_formal": True,
        },
        "profile": {
            "profile_id": TORUS9_PROFILE_ID,
            "profile_fingerprint": profile_fingerprint(profile),
            "arena_watchdog": profile["arena"]["watchdog"],
        },
        "datasets": analyses,
        "per_generation": generation_analysis,
        "winner_flips_at_all_512_candidate": winner_flips(selfplay, candidate),
        "winner_flips_at_all_512_upper_crossing": winner_flips(selfplay, float(all_interval[-1])) if all_interval else None,
        "paired_diagnostic": dict(paired_metadata),
        "interpretation": {
            "candidate_best_point_estimate": round(candidate, 8),
            "current_komi_black_win_rate": analyses["all_512_selfplay"]["komi_sweep"]["0.5"]["black_win_rate"],  # type: ignore[index]
            "current_komi_shift_description": "At K=0.5, BLACK wins more than half of terminal self-play trajectories; the amount is trajectory-conditioned and not a game-theoretic proof.",
            "late_generation_agreement": "See D6-D8 and paired datasets; disagreement lowers confidence rather than changing the Golden baseline.",
            "paired_discrepancy_analysis": {
                "512_selfplay_population": "Empty-board self-play across D1-D8 with root noise and early temperature sampling.",
                "128_paired_population": "Deterministic Arena from fixed non-empty starts stratified by prefix lengths 2,4,6,8,10,12,14,16, with the same frozen M8 on both sides and colors swapped.",
                "why_not_equivalent": "The paired run controls model identity and start state but samples a different state distribution and search regime from the 512 self-play corpus; its contradiction is evidence of population/semantics sensitivity, not proof that either historical corpus is invalid.",
                "next_action": "INCONCLUSIVE; do not automatically add another sample or change training komi.",
            },
            "fair_komi_semantics": "Effective/trajectory-conditioned counterfactual komi only; not a game-theoretic fair komi and not a training-semantic change.",
            "confidence_criteria_applied": ["mean", "median", "win-rate crossing", "without D3", "D6-D8", "frozen M8-vs-M8 paired", "10,000-resample bootstrap"],
        },
        "final_conclusion": {
            "recommended_fair_komi_range": recommended_range,
            "best_point_estimate": round(candidate, 8),
            "confidence": confidence,
            "paired_confirmation": confirms,
            "paired_result_contradicts_512": paired_contradicts,
            "do_not_extend_sample_automatically": paired_contradicts,
            "paired_interpretation": (
                "The 128-game paired result contradicts the 512 self-play crossing: BLACK is below 50% even at K=0.5, so no confirmation of the old range is claimed. This is INCONCLUSIVE and requires a separately designed follow-up before any more compute."
                if paired_contradicts else "The paired result is directionally consistent with the 512 self-play crossing."
            ),
            "future_training_komi_change_justified": "NO" if confidence == "HIGH" else "INCONCLUSIVE",
        },
    }
    return report


def calibration_datasets(
    selfplay: Sequence[Trajectory],
    arena: Sequence[Trajectory],
    paired: Sequence[Trajectory],
) -> dict[str, list[Trajectory]]:
    return {
        "all_512_selfplay": list(selfplay),
        "without_D3": [row for row in selfplay if row.generation != "D3"],
        "late_D6_D8": [row for row in selfplay if row.generation in {"D6", "D7", "D8"}],
        "M8_related_terminal_games": [row for row in arena if "M8" in row.source],
        "M8_vs_M8_paired": list(paired),
    }


def format_number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(report: Mapping[str, object]) -> str:
    datasets = report["datasets"]
    final = report["final_conclusion"]
    lines = [
        "# TORUS9 WATCHDOG + KOMI CALIBRATION",
        "",
        "## ARENA WATCHDOG",
        "",
        f"- legacy universal limit: **{report['arena_watchdog']['legacy_universal_limit']}** plies",
        f"- new board-size-scaled Torus9 limit: **{report['arena_watchdog']['new_board_size_scaled_limit']}** plies",
        "- verification: deterministic resolver; 5×5 = 500, Torus5 = 500, Torus9 = 1000; technical limit is fail-closed and never W/L/D",
        "",
        "## GOLDEN HISTORICAL KOMI",
        "",
        "**0.5**. All existing trajectories were played at K=0.5. Other values below are offline counterfactual rescoring; no model was retrained.",
        "",
        "## ESTIMATES",
        "",
        "| Dataset | Estimated fair komi | Crossing interval | Bootstrap crossing CI | Games |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["all_512_selfplay", "without_D3", "late_D6_D8", "M8_related_terminal_games", "M8_vs_M8_paired"]:
        row = datasets[name]
        implied = row["implied_fair_komi"]
        bootstrap = row["bootstrap"]
        interval = implied["win_rate_crossing_interval"]
        lines.append(
            f"| {name} | {format_number(implied['win_rate_crossing_point'])} | {interval or '—'} | {bootstrap['crossing_point_95_percent_ci']} | {row['games']} |"
        )
    lines.extend([
        "",
        "## REQUIRED READOUT",
        "",
        f"ALL SELF-PLAY ESTIMATE: mean-implied K={format_number(datasets['all_512_selfplay']['implied_fair_komi']['mean_based'])}, median-implied K={format_number(datasets['all_512_selfplay']['implied_fair_komi']['median_based'])}, crossing={format_number(datasets['all_512_selfplay']['implied_fair_komi']['win_rate_crossing_point'])}.",
        f"WITHOUT-D3 ESTIMATE: crossing={format_number(datasets['without_D3']['implied_fair_komi']['win_rate_crossing_point'])}.",
        f"LATE-GENERATION ESTIMATE: D6-D8 crossing={format_number(datasets['late_D6_D8']['implied_fair_komi']['win_rate_crossing_point'])}; late-model agreement is interpreted with its CI, not as a forced single number.",
        f"M8-ONLY ESTIMATE: M8-related terminal trajectories crossing={format_number(datasets['M8_related_terminal_games']['implied_fair_komi']['win_rate_crossing_point'])}.",
        f"M8-vs-M8 PAIRED ESTIMATE: crossing={format_number(datasets['M8_vs_M8_paired']['implied_fair_komi']['win_rate_crossing_point'])}; bootstrap unit is the independent start pair.",
        f"RECOMMENDED FAIR-KOMI RANGE: {final['recommended_fair_komi_range'] or 'see crossing intervals above; no narrow single range is promoted when datasets disagree'}.",
        f"BEST POINT ESTIMATE: {format_number(final['best_point_estimate'])}.",
        f"CONFIDENCE: {final['confidence']}.",
        f"PAIRED CONFIRMATION: {'YES' if final['paired_confirmation'] else 'NO'}; {final['paired_interpretation']}",
        "",
        "## RAW SCORE DISTRIBUTION",
        "",
        "| Dataset | mean | median | mode | P25 | P50 | P75 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ])
    for name in ["all_512_selfplay", "without_D3", "late_D6_D8", "M8_related_terminal_games", "M8_vs_M8_paired"]:
        dist = datasets[name]["raw_area_advantage_distribution"]
        lines.append(f"| {name} | {format_number(dist['mean'])} | {format_number(dist['median'])} | {dist['mode']} | {format_number(dist['p25'])} | {format_number(dist['p50'])} | {format_number(dist['p75'])} |")
    lines.extend([
        "",
        "### Exact all-512 histogram",
        "",
        "The JSON artifact contains the same histogram as machine-readable data; the exact counts are repeated here to make the primary crossing auditable:",
        "",
        "```json",
        json.dumps(datasets["all_512_selfplay"]["raw_area_advantage_distribution"]["histogram"], sort_keys=True),
        "```",
        "",
        "## GENERATIONS D1–D8",
        "",
        "| Generation | BLACK rate at K=0.5 | mean raw advantage | median raw advantage | crossing interval |",
        "|---|---:|---:|---:|---|",
    ])
    for generation in [f"D{i}" for i in range(1, 9)]:
        row = report["per_generation"][generation]
        lines.append(f"| {generation} | {format_number(row['komi_sweep']['0.5']['black_win_rate'])} | {format_number(row['raw_area_advantage_distribution']['mean'])} | {format_number(row['raw_area_advantage_distribution']['median'])} | {row['implied_fair_komi']['win_rate_crossing_interval'] or '—'} |")
    lines.extend([
        "",
        "## WINNER FLIPS",
        "",
        f"At the all-512 best point, **{report['winner_flips_at_all_512_candidate']['winner_flips']} / {report['winner_flips_at_all_512_candidate']['games']}** existing self-play trajectories change WDL under offline rescoring from K=0.5. This does not rewrite training targets.",
        f"At the upper observed crossing boundary K={report['winner_flips_at_all_512_upper_crossing']['to_komi'] if report['winner_flips_at_all_512_upper_crossing'] else '—'}, **{report['winner_flips_at_all_512_upper_crossing']['winner_flips'] if report['winner_flips_at_all_512_upper_crossing'] else '—'} / 512** self-play trajectories flip relative to K=0.5; this is why the median and histogram matter more than mean raw margin.",
        "",
        "## SCIENTIFIC LIMITS",
        "",
        "The komi estimate is trajectory-conditioned: a model trained and searched with another real komi could choose other moves. Existing technical Arena games are excluded from W/L/D; if their terminal continuation is used, it remains sensitivity analysis, not an Arena played under that komi.",
        "",
        "Would this justify changing training komi in the next separate scientific run? **INCONCLUSIVE**. The calibration supports a diagnostic range, but a future change needs a separately specified training experiment and must not reinterpret the frozen M8 Golden result.",
        "",
        f"Source commit: `{report['source']['git_commit']}`; source tree: `{report['source']['git_tree']}`; report schema: `{report['report_schema']}`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--arena-root", type=Path, default=DEFAULT_ARENA_ROOT)
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--paired-master-seed", type=int, default=PAIR_START_SEED)
    parser.add_argument("--workers", type=int, default=TORUS9_WORKERS)
    parser.add_argument("--skip-paired-run", action="store_true", help="Require existing paired games instead of launching them")
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0 or args.workers <= 0:
        raise SystemExit("bootstrap-resamples and workers must be positive")
    # Decide whether the independent confirmation is warranted from the old
    # trajectories first.  This is deliberately before any new Arena call.
    existing_selfplay = load_selfplay_trajectories(args.run_root)
    existing_arena = load_arena_trajectories(args.arena_root)
    existing_groups = calibration_datasets(existing_selfplay, existing_arena, ())
    existing_all = sweep_trajectories(existing_groups["all_512_selfplay"])
    candidate_interval = existing_all["implied_fair_komi"]["win_rate_crossing_interval"]
    candidate_point = existing_all["implied_fair_komi"]["win_rate_crossing_point"]
    print(json.dumps({
        "offline_preflight": "COMPLETE",
        "games": existing_all["games"],
        "raw_area_advantage_histogram": existing_all["raw_area_advantage_distribution"]["histogram"],
        "komi_0.5_black_win_rate": existing_all["komi_sweep"]["0.5"]["black_win_rate"],
        "komi_1.5_black_win_rate": existing_all["komi_sweep"]["1.5"]["black_win_rate"],
        "crossing_interval": candidate_interval,
        "crossing_point": candidate_point,
        "new_arena_runs_per_komi": 0,
    }, sort_keys=True))
    paired_path = args.paired_root / "m8-vs-m8-paired"
    if args.skip_paired_run:
        if not (paired_path / "games.jsonl").exists():
            raise SystemExit(f"Missing paired diagnostic: {paired_path / 'games.jsonl'}")
        paired_metadata = json.loads((paired_path / "diagnostic-manifest.json").read_text(encoding="utf-8")) if (paired_path / "diagnostic-manifest.json").exists() else {"games_path": str(paired_path / "games.jsonl")}
    else:
        if not candidate_interval:
            raise SystemExit("Existing trajectories do not produce a finite crossing; refusing an unanchored confirmation run")
        paired_path, paired_metadata = ensure_paired_diagnostic(
            run_root=args.run_root,
            output_root=args.paired_root,
            master_seed=args.paired_master_seed,
            workers=args.workers,
            candidate_crossing_interval=candidate_interval,
            candidate_point_estimate=candidate_point,
        )
    report = build_report(
        run_root=args.run_root,
        arena_root=args.arena_root,
        paired_root=paired_path,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        paired_metadata=paired_metadata,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"json": str(args.json_output), "markdown": str(args.markdown_output), "best_point": report["final_conclusion"]["best_point_estimate"], "paired_games": report["datasets"]["M8_vs_M8_paired"]["games"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
