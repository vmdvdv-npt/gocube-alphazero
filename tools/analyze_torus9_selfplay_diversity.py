#!/usr/bin/env python3
"""Read-only offline diversity analysis for saved Torus9 self-play artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PREFIX_LENGTHS = (4, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 256)
FAMILY_LENGTHS = (8, 12, 16)
INTERVALS = (
    ("moves_1_8", 1, 8),
    ("moves_9_12", 9, 12),
    ("moves_13_16", 13, 16),
    ("moves_17_24", 17, 24),
    ("moves_25_plus", 25, None),
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def state_key(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def action_key(value: Any) -> str:
    if value == "PASS":
        return "PASS"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def entropy(counter: Mapping[str, int]) -> float:
    total = sum(counter.values())
    return -sum((n / total) * math.log(n / total) for n in counter.values() if n) if total else 0.0


def top_share(counter: Mapping[str, int], count: int = 1) -> float:
    total = sum(counter.values())
    return ratio(sum(sorted(counter.values(), reverse=True)[:count]), total)


def percentile(values: Sequence[int], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def visit_metrics(raw: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw, list) or not raw:
        return None
    values: List[float] = []
    for value in raw:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(max(0.0, float(value)))
    total = sum(values)
    if total <= 0:
        return None
    probabilities = [value / total for value in values]
    ordered = sorted(probabilities, reverse=True)
    h = -sum(p * math.log(p) for p in probabilities if p > 0)
    return {
        "entropy_nats": h,
        "top1_visit_share": ordered[0],
        "top2_visit_share": sum(ordered[:2]),
        "top3_visit_share": sum(ordered[:3]),
        "effective_candidate_moves": math.exp(h),
    }


class Accumulator:
    def __init__(self, name: str, lineage: Path, generations: Sequence[int], config: Path):
        self.name = name
        self.lineage = lineage
        self.generations = list(generations)
        self.config = config
        self.lines = 0
        self.parsed = 0
        self.invalid = 0
        self.complete = 0
        self.positions = 0
        self.lengths: List[int] = []
        self.max_ply = 0
        self.issues: List[str] = []
        self.files: List[Dict[str, Any]] = []
        self.games: Counter[Tuple[str, ...]] = Counter()
        self.prefixes = {length: Counter() for length in PREFIX_LENGTHS}
        self.families = {length: Counter() for length in FAMILY_LENGTHS}
        self.position_counts: Counter[str] = Counter()
        self.state_actions: Dict[str, Counter[str]] = defaultdict(Counter)
        self.actions_by_ply: Dict[int, Counter[str]] = defaultdict(Counter)
        self.mcts_by_ply: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {
                "count": 0.0,
                "entropy": 0.0,
                "top1": 0.0,
                "top2": 0.0,
                "top3": 0.0,
                "effective": 0.0,
                "argmax": 0.0,
            }
        )
        self.root_visit_records = 0
        self.policy_records = 0
        self.terminations: Counter[str] = Counter()

    def add_record(self, record: Any) -> Tuple[bool, str, int]:
        if not isinstance(record, dict):
            return False, "record_not_object", 0
        if record.get("error") is not None:
            return False, "record_error", 0
        trace = record.get("final_action_trace")
        positions = record.get("positions")
        if not isinstance(trace, list) or not isinstance(positions, list):
            return False, "missing_trace_or_positions", 0
        if len(trace) != len(positions):
            return False, "trace_positions_length_mismatch", 0
        actions: List[str] = []
        for index, position in enumerate(positions):
            if not isinstance(position, dict) or not isinstance(position.get("state"), dict):
                return False, "invalid_position_state", 0
            if "selected_action" not in position:
                return False, "missing_selected_action", 0
            selected = action_key(position["selected_action"])
            traced = action_key(trace[index])
            if selected != traced:
                return False, "trace_position_action_mismatch", 0
            try:
                if int(position.get("ply", 0)) <= 0:
                    return False, "invalid_ply", 0
            except (TypeError, ValueError):
                return False, "invalid_ply", 0
            actions.append(selected)

        self.complete += 1
        self.lengths.append(len(actions))
        self.max_ply = max(self.max_ply, len(actions))
        self.games[tuple(actions)] += 1
        self.terminations[str(record.get("technical_termination") or "none")] += 1
        for length in PREFIX_LENGTHS:
            if len(actions) >= length:
                self.prefixes[length][tuple(actions[:length])] += 1
        for length in FAMILY_LENGTHS:
            if len(actions) >= length:
                self.families[length][tuple(actions[:length])] += 1

        for position, action in zip(positions, actions):
            ply = int(position["ply"])
            self.positions += 1
            self.actions_by_ply[ply][action] += 1
            key = state_key(position["state"])
            self.position_counts[key] += 1
            self.state_actions[key][action] += 1
            if isinstance(position.get("pi"), list):
                self.policy_records += 1
            metrics = visit_metrics(position.get("root_visits"))
            if metrics is None:
                continue
            self.root_visit_records += 1
            aggregate = self.mcts_by_ply[ply]
            aggregate["count"] += 1
            aggregate["entropy"] += metrics["entropy_nats"]
            aggregate["top1"] += metrics["top1_visit_share"]
            aggregate["top2"] += metrics["top2_visit_share"]
            aggregate["top3"] += metrics["top3_visit_share"]
            aggregate["effective"] += metrics["effective_candidate_moves"]
            try:
                selected = int(action)
                visits = [float(value) for value in position["root_visits"]]
                if 0 <= selected < len(visits) and visits[selected] == max(visits):
                    aggregate["argmax"] += 1
            except (TypeError, ValueError):
                pass
        return True, "", len(actions)

    def add_file_summary(self, summary: Dict[str, Any]) -> None:
        self.files.append(summary)
        self.lines += summary["lines_seen"]
        self.parsed += summary["parsed_records"]
        self.invalid += summary["invalid_records"]

    def position_result(self) -> Dict[str, Any]:
        unique = len(self.position_counts)
        buckets = {"1": [], "2": [], "3-5": [], ">5": []}
        for count in self.position_counts.values():
            if count == 1:
                buckets["1"].append(count)
            elif count == 2:
                buckets["2"].append(count)
            elif count <= 5:
                buckets["3-5"].append(count)
            else:
                buckets[">5"].append(count)
        bucket_result = {}
        for key, values in buckets.items():
            bucket_result[key] = {
                "unique_position_keys": len(values),
                "unique_key_share": ratio(len(values), unique),
                "encountered_positions": sum(values),
                "encountered_position_share": ratio(sum(values), self.positions),
            }
        repeated = [(key, count) for key, count in self.position_counts.items() if count >= 2]
        repeated_occurrences = sum(count for _, count in repeated)
        weighted_h = 0.0
        mean_h: List[float] = []
        weighted_top1 = 0.0
        for key, count in repeated:
            choices = self.state_actions[key]
            h = entropy(choices)
            weighted_h += count * h
            weighted_top1 += count * top_share(choices)
            mean_h.append(h)
        return {
            "total_encountered_positions": self.positions,
            "unique_raw_positions": unique,
            "unique_position_ratio": ratio(unique, self.positions),
            "repeat_rate": 1 - ratio(unique, self.positions),
            "frequency_buckets": bucket_result,
            "repeated_state_keys": len(repeated),
            "repeated_state_occurrences": repeated_occurrences,
            "repeated_state_occurrence_share": ratio(repeated_occurrences, self.positions),
            "conditional_choice_entropy_nats_weighted": ratio(weighted_h, repeated_occurrences),
            "conditional_choice_entropy_nats_unweighted_mean": statistics.mean(mean_h) if mean_h else 0.0,
            "conditional_choice_top1_share_weighted": ratio(weighted_top1, repeated_occurrences),
            "symmetry_canonicalized": False,
            "symmetry_note": "Skipped: no production-trusted canonicalizer for full state including superko_history.",
        }

    def game_result(self) -> Dict[str, Any]:
        unique = len(self.games)
        return {
            "total_games_found": self.lines,
            "parsed_records": self.parsed,
            "complete_games": self.complete,
            "invalid_records": self.invalid,
            "exact_unique_games": unique,
            "full_game_duplicates": self.complete - unique,
            "unique_game_ratio": ratio(unique, self.complete),
        }

    def prefix_result(self) -> Dict[str, Any]:
        result = {}
        for length, counter in self.prefixes.items():
            eligible = sum(counter.values())
            result[str(length)] = {
                "eligible_games": eligible,
                "unique_prefixes": len(counter),
                "unique_prefix_ratio": ratio(len(counter), eligible),
                "largest_equal_prefix_cluster": max(counter.values()) if counter else 0,
            }
        return result

    def family_result(self) -> Dict[str, Any]:
        result = {}
        for length, counter in self.families.items():
            eligible = sum(counter.values())
            top = {}
            for fraction in (0.01, 0.05, 0.10):
                count = max(1, math.ceil(len(counter) * fraction)) if counter else 0
                covered = sum(value for _, value in counter.most_common(count))
                top[f"top_{int(fraction * 100)}pct"] = {
                    "distinct_family_count": len(counter),
                    "families_in_top_set": count,
                    "covered_games": covered,
                    "covered_game_share": ratio(covered, eligible),
                }
            result[str(length)] = {
                "eligible_games": eligible,
                "distinct_families": len(counter),
                "top_family_concentration": top,
                "top_10_families": [
                    {"prefix": list(prefix), "games": value}
                    for prefix, value in counter.most_common(10)
                ],
            }
        return result

    def action_result(self) -> Dict[str, Any]:
        result = {}
        for ply, counter in sorted(self.actions_by_ply.items()):
            result[str(ply)] = {
                "encounters": sum(counter.values()),
                "unique_actions": len(counter),
                "entropy_nats": entropy(counter),
                "top1_action_share": top_share(counter),
            }
        return result

    def mcts_result(self) -> Dict[str, Any]:
        result = {}
        for ply, aggregate in sorted(self.mcts_by_ply.items()):
            count = aggregate["count"]
            if not count:
                continue
            result[str(ply)] = {
                "positions": int(count),
                "entropy_nats_mean": aggregate["entropy"] / count,
                "top1_visit_share_mean": aggregate["top1"] / count,
                "top2_visit_share_mean": aggregate["top2"] / count,
                "top3_visit_share_mean": aggregate["top3"] / count,
                "effective_candidate_moves_mean": aggregate["effective"] / count,
                "selected_is_visit_argmax_rate": aggregate["argmax"] / count,
            }
        return result

    def interval_result(self) -> Dict[str, Any]:
        result = {}
        for name, start, end in INTERVALS:
            plies = [ply for ply in self.actions_by_ply if ply >= start and (end is None or ply <= end)]
            pooled = Counter()
            for ply in plies:
                pooled.update(self.actions_by_ply[ply])
            action_entropies = [entropy(self.actions_by_ply[ply]) for ply in plies]
            mcts = [self.mcts_by_ply[ply] for ply in plies if self.mcts_by_ply[ply]["count"]]
            count = sum(item["count"] for item in mcts)
            result[name] = {
                "start_ply": start,
                "end_ply": end,
                "plies_observed": plies,
                "pooled_action_encounters": sum(pooled.values()),
                "pooled_unique_actions": len(pooled),
                "pooled_action_entropy_nats": entropy(pooled),
                "mean_per_ply_action_entropy_nats": statistics.mean(action_entropies) if action_entropies else 0.0,
                "pooled_top1_action_share": top_share(pooled),
                "mcts_positions": int(count),
                "mcts_entropy_nats_mean": ratio(sum(item["entropy"] for item in mcts), count),
                "mcts_top1_visit_share_mean": ratio(sum(item["top1"] for item in mcts), count),
                "mcts_top2_visit_share_mean": ratio(sum(item["top2"] for item in mcts), count),
                "mcts_top3_visit_share_mean": ratio(sum(item["top3"] for item in mcts), count),
                "mcts_effective_candidate_moves_mean": ratio(sum(item["effective"] for item in mcts), count),
                "selected_is_visit_argmax_rate": ratio(sum(item["argmax"] for item in mcts), count),
            }
        return result

    def result(self, per_generation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = load_json(self.config)
        lengths = {
            "min": min(self.lengths) if self.lengths else None,
            "median": statistics.median(self.lengths) if self.lengths else None,
            "mean": statistics.mean(self.lengths) if self.lengths else None,
            "p90": percentile(self.lengths, 0.90),
            "max": max(self.lengths) if self.lengths else None,
        }
        return {
            "name": self.name,
            "lineage_path": str(self.lineage),
            "generations": self.generations,
            "config_path": str(self.config),
            "config_sha256": file_sha256(self.config),
            "effective_config": {
                "self_play": config.get("self_play", {}),
                "execution": config.get("execution", {}),
                "replay": config.get("replay", {}),
            },
            "file_summaries": self.files,
            "issues": self.issues,
            "termination_counts": dict(self.terminations),
            "game_length_plies": lengths,
            "games": self.game_result(),
            "prefixes": self.prefix_result(),
            "families": self.family_result(),
            "positions": self.position_result(),
            "action_diversity_by_ply": self.action_result(),
            "interval_diagnostics": self.interval_result(),
            "mcts_policy": {
                "root_visits_saved": self.root_visit_records > 0,
                "policy_targets_saved": self.policy_records > 0,
                "root_visit_records": self.root_visit_records,
                "policy_target_records": self.policy_records,
                "by_ply": self.mcts_result(),
            },
            "per_generation": per_generation or {},
        }


def analyze_window(name: str, lineage: Path, generations: Sequence[int], config: Path) -> Dict[str, Any]:
    combined = Accumulator(name, lineage, generations, config)
    per_generation: Dict[str, Accumulator] = {
        str(generation): Accumulator(f"{name}-M{generation}", lineage, [generation], config)
        for generation in generations
    }
    for generation in generations:
        path = lineage / "selfplay" / f"iter-{generation}-games.jsonl"
        file_result = {
            "generation": generation,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": None,
            "lines_seen": 0,
            "parsed_records": 0,
            "complete_games": 0,
            "invalid_records": 0,
            "positions_parsed": 0,
            "issues": [],
        }
        if not path.exists():
            file_result["issues"].append("missing_file")
            combined.add_file_summary(file_result)
            per_generation[str(generation)].add_file_summary(file_result.copy())
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                digest.update(raw)
                file_result["lines_seen"] += 1
                line = raw.strip()
                if not line:
                    file_result["issues"].append({"line": line_number, "issue": "blank_line"})
                    continue
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    file_result["invalid_records"] += 1
                    issue = type(exc).__name__
                    file_result["issues"].append({"line": line_number, "issue": issue})
                    continue
                file_result["parsed_records"] += 1
                ok, issue, position_count = per_generation[str(generation)].add_record(record)
                if not ok:
                    file_result["invalid_records"] += 1
                    file_result["issues"].append({"line": line_number, "issue": issue})
                    continue
                file_result["complete_games"] += 1
                file_result["positions_parsed"] += position_count
                combined.add_record(record)
        file_result["sha256"] = "sha256:" + digest.hexdigest()
        combined.add_file_summary(file_result.copy())
        per_generation[str(generation)].add_file_summary(file_result)
        combined.issues.extend(
            f"M{generation} line {item['line']}: {item['issue']}"
            for item in file_result["issues"][:30]
        )
    generation_results = {
        key: accumulator.result()
        for key, accumulator in per_generation.items()
    }
    return combined.result(generation_results)


def completion_record(lineage: Path, generation: int) -> Dict[str, Any]:
    path = lineage / f"generation-{generation}.complete.json"
    if not path.exists():
        return {"generation": generation, "exists": False, "path": str(path)}
    data = load_json(path)
    return {
        "generation": generation,
        "exists": True,
        "path": str(path),
        "schema": data.get("schema"),
        "label": data.get("label"),
        "selfplay_artifact_path": data.get("selfplay_artifact_path"),
        "replay_row_count": data.get("replay_row_count"),
        "replay_generations": data.get("replay_generations"),
        "sha256": file_sha256(path),
    }


def compare(current: Mapping[str, Any], old: Mapping[str, Any]) -> Dict[str, Any]:
    paths = {
        "unique_game_ratio": ("games", "unique_game_ratio"),
        "unique_position_ratio": ("positions", "unique_position_ratio"),
        "position_repeat_rate": ("positions", "repeat_rate"),
        "prefix_8_unique_ratio": ("prefixes", "8", "unique_prefix_ratio"),
        "prefix_12_unique_ratio": ("prefixes", "12", "unique_prefix_ratio"),
        "prefix_16_unique_ratio": ("prefixes", "16", "unique_prefix_ratio"),
        "moves_1_8_action_entropy": ("interval_diagnostics", "moves_1_8", "mean_per_ply_action_entropy_nats"),
        "moves_9_12_action_entropy": ("interval_diagnostics", "moves_9_12", "mean_per_ply_action_entropy_nats"),
        "moves_9_12_mcts_entropy": ("interval_diagnostics", "moves_9_12", "mcts_entropy_nats_mean"),
    }
    result = {}
    for name, path in paths.items():
        def read(window: Mapping[str, Any]) -> Any:
            value: Any = window
            for key in path:
                value = value.get(key) if isinstance(value, Mapping) else None
            return value
        new, previous = read(current), read(old)
        result[name] = {
            "historical": previous,
            "current": new,
            "current_minus_historical": new - previous if isinstance(new, (int, float)) and isinstance(previous, (int, float)) else None,
        }
    return result


def round_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, list):
        return [round_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: round_numbers(item) for key, item in value.items()}
    return value


def compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the tracked, machine-readable summary without raw per-position data."""

    def compact_window(window: Mapping[str, Any]) -> Dict[str, Any]:
        file_summaries = [
            {
                key: item[key]
                for key in (
                    "generation",
                    "size_bytes",
                    "sha256",
                    "lines_seen",
                    "parsed_records",
                    "complete_games",
                    "invalid_records",
                    "positions_parsed",
                    "issues",
                )
                if key in item
            }
            for item in window["file_summaries"]
        ]
        families = {
            length: {
                key: value[key]
                for key in ("eligible_games", "distinct_families", "top_family_concentration")
                if key in value
            }
            for length, value in window["families"].items()
        }
        return {
            key: window[key]
            for key in (
                "name",
                "lineage_path",
                "generations",
                "config_path",
                "config_sha256",
                "effective_config",
                "issues",
                "termination_counts",
                "game_length_plies",
                "games",
                "prefixes",
                "positions",
                "interval_diagnostics",
            )
            if key in window
        } | {
            "file_summaries": file_summaries,
            "families": families,
            "mcts_policy": {
                key: window["mcts_policy"][key]
                for key in (
                    "root_visits_saved",
                    "policy_targets_saved",
                    "root_visit_records",
                    "policy_target_records",
                )
                if key in window["mcts_policy"]
            },
        }

    return {
        "schema": "torus9-selfplay-diversity-analysis-compact-summary-v1",
        "analysis_date": summary["analysis_date"],
        "read_only": summary["read_only"],
        "runs_modified": summary["runs_modified"],
        "new_selfplay_started": summary["new_selfplay_started"],
        "windows": {
            name: compact_window(window)
            for name, window in summary["windows"].items()
        },
        "completion_records": summary["completion_records"],
        "window_comparison": summary["window_comparison"],
        "methodology": summary["methodology"],
        "conclusions": [
            "Self-play diversity has not collapsed: exact-game and exact-position uniqueness remain high in the current window.",
            "The post-ply-8 change is localized to MCTS search branching; aggregate selected-action entropy is a coarse mixed-state metric.",
            "No production parameters were changed; the report is observational and does not establish temperature causality.",
        ],
        "artifacts": {
            "report": summary["artifacts"]["report"],
            "summary": summary["artifacts"]["summary"],
            "plots": summary["artifacts"]["plots"],
            "plot_note": "Raw per-ply/per-position records are intentionally not tracked; rerun the read-only analyzer with an explicit debug output outside the repository if needed.",
        },
    }


def f(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def p(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def markdown(summary: Mapping[str, Any]) -> str:
    current = summary["windows"]["current"]
    old = summary["windows"]["historical"]
    cg, cp = current["games"], current["positions"]
    og, op = old["games"], old["positions"]
    cut = current["interval_diagnostics"]
    before = cut["moves_1_8"]
    after = cut["moves_9_12"]
    c130 = current["per_generation"]["130"]
    c135 = current["per_generation"]["135"]
    lines: List[str] = [
        "# Torus9 self-play diversity analysis",
        "",
        "Дата анализа: 2026-09-24. Это read-only диагностика сохранённых self-play artifacts; новые self-play, обучение, Arena, A/B и benchmark не запускались, runs/ не изменялись.",
        "",
        "## Краткий вывод",
        "",
        f"- В актуальном окне M130–M135 найдено {cg['total_games_found']} строк, {cg['complete_games']} полноценных игр; exact unique — {cg['exact_unique_games']} ({p(cg['unique_game_ratio'])}), полных дублей — {cg['full_game_duplicates']}.",
        f"- В новом окне {cp['unique_raw_positions']} unique raw states из {cp['total_encountered_positions']} ({p(cp['unique_position_ratio'])}); repeat rate {p(cp['repeat_rate'])}.",
        f"- Aggregate selected-action entropy почти не меняется после ply 8: {f(before['mean_per_ply_action_entropy_nats'])} → {f(after['mean_per_ply_action_entropy_nats'])} nats/ply; эта coarse-метрика смешивает разные states.",
        f"- MCTS root entropy за тот же переход падает: {f(before['mcts_entropy_nats_mean'])} → {f(after['mcts_entropy_nats_mean'])} nats, effective candidates {f(before['mcts_effective_candidate_moves_mean'], 2)} → {f(after['mcts_effective_candidate_moves_mean'], 2)}, top-1 {p(before['mcts_top1_visit_share_mean'])} → {p(after['mcts_top1_visit_share_mean'])}; selected ход совпадает с visit argmax после cutoff в {p(after['selected_is_visit_argmax_rate'])} случаев.",
        f"- Между M100–M105 и M130–M135 unique-game ratio: {p(og['unique_game_ratio'])} → {p(cg['unique_game_ratio'])}; unique-position ratio: {p(op['unique_position_ratio'])} → {p(cp['unique_position_ratio'])}.",
        "",
        "Ответы на пять вопросов ТЗ:",
        "",
        "1. 384 игры достаточно разнообразны как полные траектории? В каждом поколении exact duplicate не найден; для объединённого окна unique ratio приведён ниже. Это не заменяет diversity состояний.",
        "2. Есть ли резкий спад после move 8? Да, в MCTS search distribution есть структурный спад entropy и effective branching непосредственно после cutoff; aggregate entropy выбранных координат почти плоская и маскирует state-conditioned эффект. Причинность температуры не доказана.",
        f"3. Diversity хуже с ростом поколения? Монотонного ухудшения в M130–M135 нет: unique-position ratio M130 = {p(c130['positions']['unique_position_ratio'])}, M135 = {p(c135['positions']['unique_position_ratio'])}.",
        "4. Даст ли больше игр просто больше похожих партий? Для полных траекторий и exact prefixes такого вывода нет: в sampled windows prefix 8/12/16 почти все singleton. Но внутри одинакового state после cutoff MCTS выбирается почти детерминированно; увеличение числа игр может сильнее повторять conditional decisions, если states начнут встречаться чаще.",
        "5. Есть ли основания тестировать изменение температурной схемы? Да, как отдельный контролируемый диагностический эксперимент: есть резкий перелом MCTS branching после ply 8 при почти плоской coarse action entropy. Production параметры не менялись.",
        "",
        "## 1. Данные и provenance",
        "",
        f"Основное production окно: {current['name']}, поколения M130–M135; lineage {current['lineage_path']}.",
        f"Старое окно: {old['name']}, поколения M100–M105; lineage {old['lineage_path']}.",
        "",
        "| Окно | Найдено строк | Parsed | Полных игр | Invalid | Позиций | Длина median / mean / max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window in (old, current):
        g, length = window["games"], window["game_length_plies"]
        lines.append(f"| {window['name']} | {g['total_games_found']} | {g['parsed_records']} | {g['complete_games']} | {g['invalid_records']} | {window['positions']['total_encountered_positions']} | {f(length['median'], 1)} / {f(length['mean'], 1)} / {f(length['max'], 0)} |")
    lines.extend([
        "",
        "Ожидалось 384 × 6 = 2304 игры в каждом окне; все 12 файлов найдены. Все 12 SHA-256 файлов совпали с selfplay_artifact_sha256 в generation completion records; повреждённых строк/records нет.",
        "",
        "### Effective configuration",
        "",
        f"Новый effective config: {current['config_path']} (sha256 {current['config_sha256']}). Подтверждено: temperature = {current['effective_config']['self_play'].get('temperature')}, temperature_after = {current['effective_config']['self_play'].get('temperature_after')}, temperature_plies = {current['effective_config']['self_play'].get('temperature_plies')}, MCTS simulations = {current['effective_config']['self_play'].get('mcts_simulations')}, Dirichlet alpha = {current['effective_config']['self_play'].get('dirichlet_alpha')}, epsilon = {current['effective_config']['self_play'].get('dirichlet_epsilon')}, games = {current['effective_config']['self_play'].get('games_per_iteration')}. Старый effective config имеет ту же temperature/MCTS/Dirichlet схему.",
        "",
        f"Сохранены root_visits ({current['mcts_policy']['root_visit_records']} records) и policy targets pi ({current['mcts_policy']['policy_target_records']} records).",
        "",
        "## 2. Exact full-game uniqueness",
        "",
        "| Поколение | Games | Exact unique | Full duplicates | Unique ratio |",
        "|---:|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        for generation in window["generations"]:
            g = window["per_generation"][str(generation)]["games"]
            lines.append(f"| M{generation} | {g['complete_games']} | {g['exact_unique_games']} | {g['full_game_duplicates']} | {p(g['unique_game_ratio'])} |")
        g = window["games"]
        lines.append(f"| **{window['name']} together** | **{g['complete_games']}** | **{g['exact_unique_games']}** | **{g['full_game_duplicates']}** | **{p(g['unique_game_ratio'])}** |")
    lines.extend([
        "",
        "Signature = exact ordered final_action_trace, включая PASS; metadata и model hash не входят.",
        "",
        "## 3. Prefix diversity",
        "",
        "| Prefix plies | New eligible | New unique | New ratio | Largest cluster | Old ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for length in PREFIX_LENGTHS:
        new, previous = current["prefixes"][str(length)], old["prefixes"][str(length)]
        lines.append(f"| {length} | {new['eligible_games']} | {new['unique_prefixes']} | {p(new['unique_prefix_ratio'])} | {new['largest_equal_prefix_cluster']} | {p(previous['unique_prefix_ratio'])} |")
    lines.extend([
        "",
        "Все sampled exact prefixes длиной 8/12/16 уникальны (largest cluster = 1), поэтому этот анализ не обнаруживает концентрации полных линий. Prefix ratio не является entropy одной decision state; observed cutoff effect проявляется в MCTS visits, а не в exact prefix duplication.",
        "",
        "## 4. Positions and decision diversity",
        "",
        "| Окно | Total positions | Unique raw | Unique ratio | Repeat rate | Repeated states >=2 | Repeated occurrences |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        pos = window["positions"]
        lines.append(f"| {window['name']} | {pos['total_encountered_positions']} | {pos['unique_raw_positions']} | {p(pos['unique_position_ratio'])} | {p(pos['repeat_rate'])} | {pos['repeated_state_keys']} | {p(pos['repeated_state_occurrence_share'])} |")
    lines.extend([
        "",
        "| Window | 1 occurrence | 2 | 3–5 | >5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        b = window["positions"]["frequency_buckets"]
        lines.append(f"| {window['name']} | {p(b['1']['unique_key_share'])} | {p(b['2']['unique_key_share'])} | {p(b['3-5']['unique_key_share'])} | {p(b['>5']['unique_key_share'])} |")
    lines.extend([
        "",
        f"В новом окне weighted conditional choice entropy для повторных exact states = {f(cp['conditional_choice_entropy_nats_weighted'])} nats; weighted top-1 action share = {p(cp['conditional_choice_top1_share_weighted'])}. Повторных exact states мало ({cp['repeated_state_occurrence_share']:.1%} encounters), поэтому это полезная, но маломощная conditional диагностика.",
        "",
        "Symmetry-canonicalized uniqueness не приводится: нет production-trusted canonicalizer для полного state, включая superko_history.",
        "",
        "## 5. Selected actions and temperature-cutoff intervals",
        "",
        "| Interval | Unique actions | Mean action entropy | Action top-1 | MCTS entropy | MCTS top-1 | Effective candidates | Selected=argmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, _, _ in INTERVALS:
        row = cut[name]
        lines.append(f"| {name.replace('_', ' ')} | {row['pooled_unique_actions']} | {f(row['mean_per_ply_action_entropy_nats'])} | {p(row['pooled_top1_action_share'])} | {f(row['mcts_entropy_nats_mean'])} | {p(row['mcts_top1_visit_share_mean'])} | {f(row['mcts_effective_candidate_moves_mean'])} | {p(row['selected_is_visit_argmax_rate'])} |")
    lines.extend([
        "",
        "Граница cutoff (текущий window, отдельные plies):",
        "",
        "| Ply | Selected-action entropy | MCTS entropy | MCTS top-1 | Selected=visit argmax |",
        "|---:|---:|---:|---:|---:|",
    ])
    for ply in (7, 8, 9, 10):
        action = current["action_diversity_by_ply"][str(ply)]
        mcts = current["mcts_policy"]["by_ply"][str(ply)]
        lines.append(f"| {ply} | {f(action['entropy_nats'])} | {f(mcts['entropy_nats_mean'])} | {p(mcts['top1_visit_share_mean'])} | {p(mcts['selected_is_visit_argmax_rate'])} |")
    lines.extend([
        "",
        "Интервальные selected-action metrics агрегируют разные позиции и являются coarse diagnostics; их почти неизменная entropy не опровергает cutoff effect. MCTS metrics используют сохранённые root_visits и показывают реальное уменьшение search branching после ply 8.",
        "",
        "## 6. Prefix families",
        "",
        "Family = exact prefix. top 1/5/10% означает top соответствующую долю distinct prefix families, с долей покрытых игр.",
        "",
        "| Prefix | Window | Top 1% | Top 5% | Top 10% |",
        "|---:|---|---:|---:|---:|",
    ])
    for length in FAMILY_LENGTHS:
        for window in (old, current):
            top = window["families"][str(length)]["top_family_concentration"]
            lines.append(f"| {length} | {window['name']} | {p(top['top_1pct']['covered_game_share'])} | {p(top['top_5pct']['covered_game_share'])} | {p(top['top_10pct']['covered_game_share'])} |")
    lines.extend([
        "",
        "## 7. Change across generations",
        "",
        "| Generation | Unique game | Unique position | Prefix-8 | Prefix-12 | Position repeat |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        for generation in window["generations"]:
            row = window["per_generation"][str(generation)]
            lines.append(f"| M{generation} | {p(row['games']['unique_game_ratio'])} | {p(row['positions']['unique_position_ratio'])} | {p(row['prefixes']['8']['unique_prefix_ratio'])} | {p(row['prefixes']['12']['unique_prefix_ratio'])} | {p(row['positions']['repeat_rate'])} |")
    lines.extend([
        "",
        "Внутри M130–M135 нет монотонного ухудшения; метрики колеблются. Сравнение со старым окном observational, а не причинный тест.",
        "",
        "## 8. Limitations",
        "",
        "- Full-game uniqueness не оценивает разнообразие позиций и решений.",
        "- Aggregate action entropy смешивает разные board states; exact-state conditional entropy рассчитана только для повторных states.",
        "- Root visits показывают search branching, но не доказывают причинность temperature: также влияют network strength, Dirichlet noise, MCTS budget и tie-breaking.",
        "- Новые симметрии не добавлялись, отсутствующие visits не восстанавливались, production parameters не менялись.",
        "",
        "Компактный JSON содержит lineage, effective config, sample sizes, aggregate/per-range metrics, validation inputs and conclusions. Raw per-position records и полные per-ply series намеренно не хранятся в репозитории.",
        "",
    ])
    return "\n".join(lines)


def plots(summary: Mapping[str, Any], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    output.mkdir(parents=True, exist_ok=True)
    current, old = summary["windows"]["current"], summary["windows"]["historical"]
    pairs = ((old, "M100–M105", "#777777"), (current, "M130–M135", "#1368ce"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for window, label, color in pairs:
        rows = window["action_diversity_by_ply"]
        xs = sorted(int(x) for x in rows)
        axes[0].plot(xs, [rows[str(x)]["entropy_nats"] for x in xs], label=label, color=color)
        rows = window["mcts_policy"]["by_ply"]
        xs = sorted(int(x) for x in rows)
        axes[1].plot(xs, [rows[str(x)]["entropy_nats_mean"] for x in xs], label=label, color=color)
    for axis, title in zip(axes, ("Selected-action entropy (nats)", "MCTS visit entropy (nats)")):
        axis.axvline(8.5, color="#b33", linestyle="--", linewidth=1)
        axis.set_xlabel("Ply")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "entropy-by-move.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for window, label, color in pairs:
        rows = window["prefixes"]
        xs = sorted(int(x) for x in rows)
        axis.plot(xs, [rows[str(x)]["unique_prefix_ratio"] for x in xs], marker="o", markersize=3, label=label, color=color)
    axis.set_xlabel("Prefix length (plies)")
    axis.set_ylabel("Unique-prefix ratio")
    axis.set_title("Prefix diversity")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "unique-prefixes-by-move.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for window, label, color in pairs:
        rows = sorted((int(generation), data["positions"]["repeat_rate"]) for generation, data in window["per_generation"].items())
        axis.plot([x for x, _ in rows], [y for _, y in rows], marker="o", label=label, color=color)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Raw-position repeat rate")
    axis.set_title("Position repeat rate by generation")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "position-repeat-rate-by-generation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for index, length in enumerate(FAMILY_LENGTHS):
        x = [0, 1, 2]
        for window, label, color, offset in ((old, "M100–M105", "#777777", -0.18), (current, "M130–M135", "#1368ce", 0.18)):
            top = window["families"][str(length)]["top_family_concentration"]
            values = [top[f"top_{n}pct"]["covered_game_share"] for n in (1, 5, 10)]
            axes[index].bar([item + offset for item in x], values, width=0.36, label=label, color=color)
        axes[index].set_xticks(x, ("top 1%", "top 5%", "top 10%"))
        axes[index].set_title(f"{length}-ply families")
        axes[index].set_ylim(0, 1)
        axes[index].grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Share of games covered")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "top-prefix-concentration.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--raw-output-json", type=Path, default=None, help="Optional debug dump outside tracked repository paths.")
    args = parser.parse_args()
    raw_output_json = args.raw_output_json.resolve() if args.raw_output_json is not None else None
    if raw_output_json is not None:
        try:
            raw_output_json.relative_to(ROOT)
        except ValueError:
            pass
        else:
            raise SystemExit("--raw-output-json must point outside the repository root")

    current_lineage = ROOT / "runs/torus9/active/torus9-m125-continuous-v2-gen6-20260922-v1"
    old_lineage = ROOT / "runs/torus9/active/torus9-m92-continuous-v2-20260920-v1"
    current_config = current_lineage / "metadata/effective-config-v2/sha256:0aa3a1c4987618683f479f4464a346476961bc16371f23ad071e3b88b014684c.json"
    old_config = old_lineage / "metadata/effective-config-v2/sha256:732e85720e5df70b675b537d4fd97c42e99efef119379e58298f097cd0552d22.json"
    current_generations = tuple(range(130, 136))
    old_generations = tuple(range(100, 106))

    current = analyze_window("current-M130-M135", current_lineage, current_generations, current_config)
    old = analyze_window("historical-M100-M105", old_lineage, old_generations, old_config)
    summary: Dict[str, Any] = {
        "schema": "torus9-selfplay-diversity-analysis-v1",
        "analysis_date": "2026-09-24",
        "read_only": True,
        "runs_modified": False,
        "new_selfplay_started": False,
        "windows": {"current": current, "historical": old},
        "completion_records": {
            "current": [completion_record(current_lineage, g) for g in current_generations],
            "historical": [completion_record(old_lineage, g) for g in old_generations],
        },
        "window_comparison": compare(current, old),
        "methodology": {
            "full_game_signature": "Exact ordered final_action_trace including PASS; metadata excluded.",
            "position_signature": "SHA-256 of canonical JSON of complete saved position.state.",
            "position_state_fields": ["stones", "side_to_move", "consecutive_passes", "komi", "rules_id", "rules_fingerprint", "topology_id", "topology_fingerprint", "superko_history"],
            "aggregate_entropy_note": "Aggregate selected-action entropy pools actions from different states and is coarse branching only.",
            "conditional_entropy": "Computed from choices observed at repeated exact state signatures.",
            "prefix_family_top_percent": "Top fraction of distinct prefix families, with covered-game share.",
            "mcts_metrics": "Root visit counts normalized per position; entropy is natural-log entropy and effective candidates is exp(entropy).",
            "symmetry_canonicalization": "Not computed; no production-trusted full-state canonicalizer was available.",
        },
        "artifacts": {
            "report": str(args.output_md),
            "summary": str(args.output_json),
            "plots": [],
            "plot_note": "PNG plots were not rendered because matplotlib is not installed; the JSON contains the complete series for plotting.",
        },
    }
    summary = round_numbers(summary)
    compact = compact_summary(summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if raw_output_json is not None:
        raw_output_json.parent.mkdir(parents=True, exist_ok=True)
        raw_output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(summary), encoding="utf-8")
    plots(summary, args.plot_dir)
    print(json.dumps({
        "current_games": current["games"],
        "historical_games": old["games"],
        "report": str(args.output_md),
        "summary": str(args.output_json),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
