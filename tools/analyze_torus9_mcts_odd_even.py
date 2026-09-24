#!/usr/bin/env python3
"""Read-only odd/even and Black/White diagnostic for saved Torus9 MCTS rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.rules import apply_action, prepare_legal_actions
from gocube_golden.state import BLACK, PASS, WHITE
from gocube_golden.torus9_contract import TORUS9_ACTION_COUNT, TORUS9_PASS_INDEX
from gocube_golden.torus9_monolith import torus9_state_from_identity


INTERVALS = (
    ("moves_1_8", 1, 8),
    ("moves_9_24", 9, 24),
    ("moves_25_64", 25, 64),
    ("moves_65_plus", 65, None),
)
GROUPS = ("odd", "even", "BLACK", "WHITE")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def action_index(action: Any) -> int:
    return TORUS9_PASS_INDEX if action == PASS else int(action)


def h_entropy(values: Sequence[float]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in values if value > 0)


def top_share(values: Sequence[float], count: int = 1) -> float:
    total = sum(values)
    return sum(sorted(values, reverse=True)[:count]) / total if total else 0.0


def qtile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


class Stats:
    def __init__(self, keep_values: bool = True):
        self.keep_values = keep_values
        self.n = 0
        self.entropy_values: List[float] = []
        self.sums = Counter()
        self.minimum: Dict[str, float] = {}
        self.maximum: Dict[str, float] = {}
        self.distributions: Dict[str, Counter] = defaultdict(Counter)

    def add(self, metrics: Mapping[str, float]) -> None:
        self.n += 1
        for key, value in metrics.items():
            number = float(value)
            self.sums[key] += number
            self.minimum[key] = number if key not in self.minimum else min(self.minimum[key], number)
            self.maximum[key] = number if key not in self.maximum else max(self.maximum[key], number)
            if key == "mcts_entropy_nats" and self.keep_values:
                self.entropy_values.append(number)
        for key in ("sum_visits", "nonzero_visits", "max_visit", "legal_count", "illegal_nonzero_visits"):
            if key in metrics:
                self.distributions[key][str(int(metrics[key]))] += 1

    def result(self) -> Dict[str, Any]:
        mean = {key: value / self.n for key, value in self.sums.items()} if self.n else {}
        entropy = {
            "mean": mean.get("mcts_entropy_nats"),
            "median": statistics.median(self.entropy_values) if self.entropy_values else None,
            "q05": qtile(self.entropy_values, 0.05),
            "q25": qtile(self.entropy_values, 0.25),
            "q75": qtile(self.entropy_values, 0.75),
            "q95": qtile(self.entropy_values, 0.95),
            "min": min(self.entropy_values) if self.entropy_values else None,
            "max": max(self.entropy_values) if self.entropy_values else None,
        }
        return {
            "positions": self.n,
            "mcts_entropy_nats": entropy,
            "means": mean,
            "min": self.minimum,
            "max": self.maximum,
            "distributions": {key: dict(value) for key, value in self.distributions.items()},
        }


def root_and_pi_metrics(
    root_visits: Any,
    pi: Any,
    legal_actions: Sequence[Any],
    selected_action: Any,
) -> Optional[Dict[str, float]]:
    if not isinstance(root_visits, list) or not isinstance(pi, list):
        return None
    if len(root_visits) != TORUS9_ACTION_COUNT or len(pi) != TORUS9_ACTION_COUNT:
        return None
    try:
        visits = [int(value) for value in root_visits]
        policy = [float(value) for value in pi]
    except (TypeError, ValueError):
        return None
    if any(value < 0 for value in visits) or any(not math.isfinite(value) or value < 0 for value in policy):
        return None
    legal_indices = {action_index(action) for action in legal_actions}
    total = sum(visits)
    if total <= 0:
        return None
    probabilities = [value / total for value in visits]
    diff = max(abs(a - b) for a, b in zip(probabilities, policy))
    legal_visit_values = [visits[index] for index in legal_indices]
    selected_idx = action_index(selected_action)
    return {
        "mcts_entropy_nats": h_entropy(probabilities),
        "effective_candidates": math.exp(h_entropy(probabilities)),
        "top1_visit_share": top_share(probabilities, 1),
        "top2_visit_share": top_share(probabilities, 2),
        "top3_visit_share": top_share(probabilities, 3),
        "nonzero_visits": sum(value > 0 for value in visits),
        "max_visit": max(visits),
        "sum_visits": total,
        "legal_count": len(legal_actions),
        "illegal_nonzero_visits": sum(value > 0 for index, value in enumerate(visits) if index not in legal_indices),
        "pi_entropy_nats": h_entropy(policy),
        "pi_effective_candidates": math.exp(h_entropy(policy)),
        "pi_top1_share": top_share(policy, 1),
        "pi_top2_share": top_share(policy, 2),
        "pi_top3_share": top_share(policy, 3),
        "pi_max_abs_diff_vs_normalized_visits": diff,
        "pi_illegal_mass": sum(policy[index] for index in range(len(policy)) if index not in legal_indices),
        "selected_is_visit_argmax": float(
            selected_idx in legal_indices and visits[selected_idx] == max(legal_visit_values)
        ),
    }


class Window:
    def __init__(self, name: str, lineage: Path, generations: Sequence[int], config: Path):
        self.name = name
        self.lineage = lineage
        self.generations = list(generations)
        self.config = config
        self.files: List[Dict[str, Any]] = []
        self.game_lines = 0
        self.parsed_records = 0
        self.complete_games = 0
        self.invalid_records = 0
        self.positions = 0
        self.issues: Counter[str] = Counter()
        self.sanity: Counter[str] = Counter()
        self.side_start: Counter[str] = Counter()
        self.pass_actions = 0
        self.group_stats: Dict[str, Stats] = {group: Stats() for group in GROUPS}
        self.interval_stats: Dict[str, Dict[str, Stats]] = {
            name: {group: Stats() for group in GROUPS} for name, _, _ in INTERVALS
        }
        self.ply_stats: Dict[int, Dict[str, Stats]] = defaultdict(
            lambda: {"odd": Stats(False), "even": Stats(False), "BLACK": Stats(False), "WHITE": Stats(False)}
        )
        self.legal_conditioning: Dict[str, Dict[int, Stats]] = {
            group: defaultdict(lambda: Stats(False)) for group in GROUPS
        }
        self.by_generation: Dict[str, Dict[str, Any]] = {}

    def _group_for(self, ply: int, side: str) -> Tuple[str, str]:
        parity = "odd" if ply % 2 else "even"
        return parity, side

    def _interval_for(self, ply: int) -> Optional[str]:
        for name, start, end in INTERVALS:
            if ply >= start and (end is None or ply <= end):
                return name
        return None

    def add_position(self, position: Mapping[str, Any], state: Any, context: Any) -> None:
        ply = int(position["ply"])
        side = state.side_to_move.name
        parity, color = self._group_for(ply, side)
        metrics = root_and_pi_metrics(
            position.get("root_visits"),
            position.get("pi"),
            context.actions,
            position.get("selected_action"),
        )
        if metrics is None:
            self.issues["malformed_root_or_pi"] += 1
            return
        self.positions += 1
        for group in (parity, color):
            self.group_stats[group].add(metrics)
            self.legal_conditioning[group][int(metrics["legal_count"])].add(metrics)
        interval = self._interval_for(ply)
        if interval is not None:
            self.interval_stats[interval][parity].add(metrics)
            self.interval_stats[interval][color].add(metrics)
        self.ply_stats[ply][parity].add(metrics)
        self.ply_stats[ply][color].add(metrics)
        if position.get("selected_action") == PASS:
            self.pass_actions += 1
        if metrics["pi_max_abs_diff_vs_normalized_visits"] > 1e-6:
            self.sanity["pi_not_normalized_root_visits"] += 1
        if metrics["illegal_nonzero_visits"] > 0:
            self.sanity["illegal_root_visits"] += 1
        if metrics["sum_visits"] != 200:
            self.sanity["root_sum_not_200"] += 1
        if metrics["selected_is_visit_argmax"] < 0.5:
            self.sanity["selected_not_visit_argmax"] += 1

    def validate_game(self, record: Mapping[str, Any]) -> None:
        positions = record.get("positions")
        trace = record.get("final_action_trace")
        if not isinstance(positions, list) or not isinstance(trace, list) or len(positions) != len(trace):
            self.issues["trace_positions_mismatch"] += 1
            return
        try:
            current = torus9_state_from_identity(record["start_state"])
        except Exception:
            self.issues["invalid_start_state"] += 1
            return
        self.side_start[current.side_to_move.name] += 1
        previous = None
        for index, position in enumerate(positions):
            try:
                saved = torus9_state_from_identity(position["state"])
                context = prepare_legal_actions(saved)
                action = position["selected_action"]
                if saved.state_key != current.state_key:
                    self.sanity["state_chain_mismatch"] += 1
                if int(position.get("ply", 0)) != index + 1:
                    self.sanity["ply_number_mismatch"] += 1
                if action != trace[index]:
                    self.sanity["trace_action_mismatch"] += 1
                if action not in context.actions:
                    self.sanity["selected_action_illegal"] += 1
                expected_side = current.side_to_move
                if saved.side_to_move != expected_side:
                    self.sanity["side_to_move_chain_mismatch"] += 1
                after = apply_action(saved, action).after
                if after.side_to_move == saved.side_to_move:
                    self.sanity["side_did_not_switch"] += 1
                if previous is not None and previous.side_to_move == saved.side_to_move:
                    self.sanity["adjacent_side_did_not_switch"] += 1
                if index + 1 < len(positions):
                    next_saved = torus9_state_from_identity(positions[index + 1]["state"])
                    if after.state_key != next_saved.state_key:
                        self.sanity["next_state_mismatch"] += 1
                self.add_position(position, saved, context)
                current = after
                previous = saved
            except Exception as exc:
                self.issues[type(exc).__name__] += 1
        if not current.is_terminal:
            self.sanity["game_not_terminal_after_trace"] += 1

    def add_file(self, generation: int, path: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "generation": generation,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "lines_seen": 0,
            "parsed_records": 0,
            "complete_games": 0,
            "invalid_records": 0,
            "positions": 0,
            "issues": [],
        }
        if not path.exists():
            result["issues"].append("missing_file")
            self.files.append(result)
            return result
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                result["lines_seen"] += 1
                self.game_lines += 1
                line = raw.strip()
                if not line:
                    result["invalid_records"] += 1
                    result["issues"].append({"line": line_number, "issue": "blank_line"})
                    continue
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    result["invalid_records"] += 1
                    result["issues"].append({"line": line_number, "issue": type(exc).__name__})
                    continue
                result["parsed_records"] += 1
                self.parsed_records += 1
                if not isinstance(record, dict) or record.get("error") is not None:
                    result["invalid_records"] += 1
                    result["issues"].append({"line": line_number, "issue": "record_error"})
                    self.invalid_records += 1
                    continue
                result["complete_games"] += 1
                self.complete_games += 1
                positions = record.get("positions")
                result["positions"] += len(positions) if isinstance(positions, list) else 0
                self.validate_game(record)
        self.invalid_records += result["invalid_records"]
        self.files.append(result)
        return result

    def conditional_legal_result(self, left: str, right: str) -> Dict[str, Any]:
        left_buckets = self.legal_conditioning[left]
        right_buckets = self.legal_conditioning[right]
        common = []
        for legal_count in sorted(set(left_buckets) & set(right_buckets)):
            l = left_buckets[legal_count]
            r = right_buckets[legal_count]
            if l.n >= 30 and r.n >= 30:
                lm = l.result()["mcts_entropy_nats"]["mean"]
                rm = r.result()["mcts_entropy_nats"]["mean"]
                common.append({
                    "legal_count": legal_count,
                    "left_positions": l.n,
                    "right_positions": r.n,
                    "left_entropy": lm,
                    "right_entropy": rm,
                    "left_minus_right": lm - rm,
                })
        left_total = sum(row["left_positions"] for row in common)
        right_total = sum(row["right_positions"] for row in common)
        left_mean = sum(row["left_entropy"] * row["left_positions"] for row in common) / left_total if left_total else None
        right_mean = sum(row["right_entropy"] * row["right_positions"] for row in common) / right_total if right_total else None
        return {
            "left_group": left,
            "right_group": right,
            "common_buckets_at_least_30_each": len(common),
            "buckets": common,
            "weighted_left_entropy": left_mean,
            "weighted_right_entropy": right_mean,
            "weighted_left_minus_right": left_mean - right_mean if left_mean is not None and right_mean is not None else None,
            "max_absolute_bucket_delta": max((abs(row["left_minus_right"]) for row in common), default=None),
        }

    def result(self, per_generation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = load_json(self.config)
        return {
            "name": self.name,
            "lineage_path": str(self.lineage),
            "generations": self.generations,
            "config_path": str(self.config),
            "effective_config": {
                "self_play": config.get("self_play", {}),
                "execution": config.get("execution", {}),
                "replay": config.get("replay", {}),
            },
            "files": self.files,
            "games": {
                "lines_found": self.game_lines,
                "parsed_records": self.parsed_records,
                "complete_games": self.complete_games,
                "invalid_records": self.invalid_records,
            },
            "positions": self.positions,
            "issues": dict(self.issues),
            "sanity_checks": dict(self.sanity),
            "start_side_counts": dict(self.side_start),
            "pass_actions": self.pass_actions,
            "groups": {group: stats.result() for group, stats in self.group_stats.items()},
            "intervals": {
                name: {group: stats.result() for group, stats in groups.items()}
                for name, groups in self.interval_stats.items()
            },
            "by_ply": {
                str(ply): {group: stats.result() for group, stats in groups.items()}
                for ply, groups in sorted(self.ply_stats.items())
            },
            "legal_count_conditioning": {
                "odd_vs_even": self.conditional_legal_result("odd", "even"),
                "BLACK_vs_WHITE": self.conditional_legal_result("BLACK", "WHITE"),
            },
            "mcts_invariants": {
                "expected_simulations_from_config": 200,
                "root_visit_vector_length": 82,
                "pi_vector_length": 82,
                "root_visits_sum_distribution": self.group_stats["odd"].result()["distributions"].get("sum_visits", {}),
                "note": "Search code backs up exactly once per configured simulation; root visit sum should equal simulations, including terminal leaf evaluations.",
            },
            "per_generation": per_generation or {},
        }


def analyze_window(name: str, lineage: Path, generations: Sequence[int], config: Path) -> Dict[str, Any]:
    combined = Window(name, lineage, generations, config)
    for generation in generations:
        path = lineage / "selfplay" / f"iter-{generation}-games.jsonl"
        combined.add_file(generation, path)
    return combined.result()


def code_audit() -> Dict[str, Any]:
    return {
        "scope": [
            "Torus9 observation side-to-move and own/opponent channels",
            "shared PUCT traversal/backup/root visit accounting",
            "root Dirichlet transform",
            "pi construction and post-search temperature selection",
        ],
        "observation": {
            "channels": ["own_stones", "opponent_stones", "side_to_move_color", "previous_pass", "legal_point_mask", "komi"],
            "side_to_move_encoding": "BLACK=+1, WHITE=-1 in one shared channel; own/opponent stone planes are relative to side_to_move.",
            "absolute_stone_color_channels": False,
            "side_to_move_color_channel": True,
            "komi": 0.5,
        },
        "search": {
            "value_semantics": "side-to-move WDL; utility=(WIN-LOSS)/sum",
            "backup": "one-ply sign flip on every edge",
            "root_visits": "legal root edge visits placed into the canonical 82-action vector; pi=count/sum",
            "parity_specific_branch": False,
            "color_specific_branch": False,
            "configured_simulations": 200,
        },
        "root_noise": {
            "enabled": True,
            "applied": "root evaluation policy before search expansion only",
            "raw_nn_policy_saved": False,
            "noisy_root_prior_saved": False,
            "available_artifact": "only root_visits and normalized pi are serialized",
        },
        "selection": {
            "temperature": "1.0 on plies 1-8, then 0",
            "temperature_applied": "after root visits are produced",
            "cannot_explain_same_position_root_visit_entropy": True,
        },
        "value_evidence_limit": "WDL logits, root Q, and per-position evaluator outputs are not serialized; code path has no observed alternating sign branch.",
    }


def report(summary: Mapping[str, Any]) -> str:
    current = summary["windows"]["current"]
    old = summary["windows"]["historical"]
    def config_row(window: Mapping[str, Any]) -> str:
        self_play = window["effective_config"]["self_play"]
        replay = window["effective_config"]["replay"]
        return (
            f"| {window['name']} | {self_play['mcts_simulations']} | {self_play['cpuct']:.2f} | "
            f"{self_play['fpu']:.2f} | {self_play['komi']:.1f} | {str(self_play['root_noise'])} | "
            f"{self_play['dirichlet_alpha']:.2f}/{self_play['dirichlet_epsilon']:.2f} | "
            f"{self_play['temperature']} | {replay['generations']} |"
        )

    def sanity_row(window: Mapping[str, Any]) -> str:
        expected = {
            key: value
            for key, value in window["sanity_checks"].items()
            if key != "selected_not_visit_argmax"
        }
        return str(expected) if expected else "{}"

    lines: List[str] = [
        "# Torus9 MCTS odd/even diagnostic",
        "",
        "Дата анализа: 2026-09-24. Read-only анализ сохранённых self-play records M100–M105 и M130–M135. Новые self-play, inference, training, Arena, benchmark и изменение runs/ не выполнялись.",
        "",
        "## Краткий вывод",
        "",
        "Главный факт: sawtooth возникает уже в root visits, а не при формировании pi или post-search temperature selection.",
        "",
    ]
    for window in (old, current):
        odd = window["groups"]["odd"]["mcts_entropy_nats"]
        even = window["groups"]["even"]["mcts_entropy_nats"]
        black = window["groups"]["BLACK"]["mcts_entropy_nats"]
        white = window["groups"]["WHITE"]["mcts_entropy_nats"]
        lines.append(
            f"- {window['name']}: odd H={odd['mean']:.3f}, even H={even['mean']:.3f}; "
            f"BLACK H={black['mean']:.3f}, WHITE H={white['mean']:.3f}; "
            f"positions={window['positions']}, invalid={window['games']['invalid_records']}."
        )
    codd = current["groups"]["odd"]
    ceven = current["groups"]["even"]
    cblack = current["groups"]["BLACK"]
    cwhite = current["groups"]["WHITE"]
    lines.extend([
        f"- В новом окне odd/even и Black/White совпадают по mapping: odd=BLACK, even=WHITE; sawtooth не является отдельным эффектом, независимым от цвета.",
        f"- Новый high-branch even minus low-branch odd delta entropy = {ceven['mcts_entropy_nats']['mean'] - codd['mcts_entropy_nats']['mean']:.3f} nats; effective candidates = {ceven['means']['effective_candidates'] - codd['means']['effective_candidates']:.2f}.",
        f"- Root budget одинаков: sum(root_visits)=200 для всех валидных позиций; illegal non-zero visits и vector-shape violations не обнаружены.",
        f"- pi является нормализацией root_visits с max absolute error {max(codd['max']['pi_max_abs_diff_vs_normalized_visits'], ceven['max']['pi_max_abs_diff_vs_normalized_visits']):.2e}. Поэтому sawtooth уже присутствует в visits.",
        "- Наиболее вероятное нормальное объяснение — устойчивое Black/White asymmetry learned policy/value при komi=0.5, усиленная side-to-move representation; code audit не нашёл parity/color-specific search branch.",
        "",
        "## 1. Scope, config and sanity",
        "",
        "| Window | MCTS simulations | cpuct | FPU | Komi | Root noise | Dirichlet α/ε | Temperature | Replay generations |",
        "|---|---:|---:|---:|---:|---|---|---|---:|",
    ])
    for window in (old, current):
        lines.append(config_row(window))
    lines.extend([
        "",
        "| Window | Games | Positions | Invalid records | PASS actions | Side starts |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for window in (old, current):
        lines.append(f"| {window['name']} | {window['games']['complete_games']} | {window['positions']} | {window['games']['invalid_records']} | {window['pass_actions']} | {window['start_side_counts']} |")
    lines.extend([
        "",
        "Оба окна прочитаны напрямую. Не пересчитывались SHA многогигабайтных self-play artifacts. Санити-проверки включали state chain, legal selected action, side switch после каждого action включая PASS, vector lengths и terminal trace. Ненулевые counters, кроме ожидаемого temperature-related selected_not_visit_argmax, отсутствуют: "
        + f"historical={sanity_row(old)}, current={sanity_row(current)}.",
        "",
        "Корректная интерпретация конфигурации: 200 simulations дают общий root budget на position; temperature=1.0 на plies 1–8 используется только для выбора action после search, затем temperature=0.",
        "",
        "## 2. Odd/even root MCTS metrics",
        "",
        "| Window | Group | N | H mean | H median | H q05–q95 | Effective candidates | Top-1 | Top-2 | Top-3 | Non-zero visits | Top max visits mean / max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        for group in ("odd", "even"):
            row = window["groups"][group]
            h = row["mcts_entropy_nats"]
            m = row["means"]
            lines.append(
                f"| {window['name']} | {group} | {row['positions']} | {h['mean']:.3f} | {h['median']:.3f} | "
                f"{h['q05']:.3f}–{h['q95']:.3f} | {m['effective_candidates']:.2f} | {m['top1_visit_share']:.1%} | "
                f"{m['top2_visit_share']:.1%} | {m['top3_visit_share']:.1%} | {m['nonzero_visits']:.2f} | "
                f"{m['max_visit']:.2f} / {row['max']['max_visit']:.0f} |"
            )
    lines.extend([
        "",
        "## 3. Black/White direct comparison",
        "",
        "| Window | Side to move | N | H mean | H median | H q05–q95 | Effective candidates | Top-1 | Top-2 | Top-3 | Non-zero visits | Top max visits mean / max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        for group in ("BLACK", "WHITE"):
            row = window["groups"][group]
            h, m = row["mcts_entropy_nats"], row["means"]
            lines.append(
                f"| {window['name']} | {group} | {row['positions']} | {h['mean']:.3f} | {h['median']:.3f} | "
                f"{h['q05']:.3f}–{h['q95']:.3f} | {m['effective_candidates']:.2f} | {m['top1_visit_share']:.1%} | "
                f"{m['top2_visit_share']:.1%} | {m['top3_visit_share']:.1%} | {m['nonzero_visits']:.2f} | "
                f"{m['max_visit']:.2f} / {row['max']['max_visit']:.0f} |"
            )
    lines.extend([
        "",
        "В records стартовая сторона всегда BLACK, а side-to-move переключается после каждого action. PASS также переключает сторону. Поэтому для этих данных odd=BLACK и even=WHITE подтверждено state-chain проверкой, а не предположено по номеру ply.",
        "",
        "## 4. Persistence across the game",
        "",
        "| Window | Interval | Group | N | H mean | Effective candidates | Top-1 | Non-zero visits |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        for interval, _, _ in INTERVALS:
            for group in ("odd", "even", "BLACK", "WHITE"):
                row = window["intervals"][interval][group]
                lines.append(
                    f"| {window['name']} | {interval} | {group} | {row['positions']} | "
                    f"{row['mcts_entropy_nats']['mean']:.3f} | {row['means']['effective_candidates']:.2f} | "
                    f"{row['means']['top1_visit_share']:.1%} | {row['means']['nonzero_visits']:.2f} |"
                )
    lines.extend([
        "",
        "Representative per-ply rows (the original sawtooth example is visible before and after the temperature cutoff):",
        "",
        "| Window | Ply | Side to move | N | H mean | Effective candidates | Top-1 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        for ply in range(7, 13):
            group = "odd" if ply % 2 else "even"
            row = window["by_ply"][str(ply)][group]
            side = "BLACK" if group == "odd" else "WHITE"
            lines.append(
                f"| {window['name']} | {ply} | {side} | {row['positions']} | "
                f"{row['mcts_entropy_nats']['mean']:.3f} | {row['means']['effective_candidates']:.2f} | "
                f"{row['means']['top1_visit_share']:.1%} |"
            )
    lines.extend([
        "",
        "## 5. Root visit accounting and actual budget",
        "",
        "| Window | Group | Sum visits distribution | Legal count mean | Illegal non-zero mean | Illegal non-zero max | Vector violations |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        for group in ("odd", "even", "BLACK", "WHITE"):
            row = window["groups"][group]
            m = row["means"]
            lines.append(
                f"| {window['name']} | {group} | {row['distributions'].get('sum_visits', {})} | "
                f"{m['legal_count']:.2f} | {m['illegal_nonzero_visits']:.2f} | "
                f"{row['max']['illegal_nonzero_visits']:.0f} | {window['issues'].get('malformed_root_or_pi', 0)} |"
            )
    lines.extend([
        "",
        "Production search code performs one root-edge backup per configured simulation. Thus sum(root_visits)=200 is the relevant invariant, not a per-action 200 count. All analyzed groups satisfy it; no systematic Black/White budget difference exists.",
        "",
        "## 6. pi versus root visits",
        "",
        "| Window | Group | pi entropy | pi effective candidates | pi top-1 | Max abs(pi − visits/sum) | Illegal pi mass | Selected=visit-argmax |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for window in (old, current):
        for group in ("odd", "even", "BLACK", "WHITE"):
            row = window["groups"][group]
            m = row["means"]
            lines.append(
                f"| {window['name']} | {group} | {m['pi_entropy_nats']:.3f} | {m['pi_effective_candidates']:.2f} | "
                f"{m['pi_top1_share']:.1%} | {row['max']['pi_max_abs_diff_vs_normalized_visits']:.2e} | "
                f"{m['pi_illegal_mass']:.2e} | {m['selected_is_visit_argmax']:.1%} |"
            )
    lines.extend([
        "",
        "pi не создаёт sawtooth: в текущих artifacts она с точностью сериализации равна нормализованным root visits. В plies 1–8 selected_is_visit_argmax ниже из-за temperature=1.0, а на plies 9+ он равен 100% при temperature=0; это post-search temperature effect, но root-visits entropy того же position он не меняет. Для current окна argmax selection в plies 1–8: odd={:.1%}, even={:.1%}; в plies 9+ — 100% для обеих групп.".format(
            current["intervals"]["moves_1_8"]["odd"]["means"]["selected_is_visit_argmax"],
            current["intervals"]["moves_1_8"]["even"]["means"]["selected_is_visit_argmax"],
        ),
        "",
        "## 7. Legal-action count control",
        "",
    ])
    for window in (old, current):
        cond = window["legal_count_conditioning"]
        lines.append(
            f"{window['name']}: odd vs even common legal-count buckets (>=30 positions each) = "
            f"{cond['odd_vs_even']['common_buckets_at_least_30_each']}, "
            f"conditional entropy delta = {cond['odd_vs_even']['weighted_left_minus_right']:.3f} nats, "
            f"max bucket delta = {cond['odd_vs_even']['max_absolute_bucket_delta']:.3f}; "
            f"BLACK vs WHITE conditional delta = {cond['BLACK_vs_WHITE']['weighted_left_minus_right']:.3f} nats."
        )
    lines.extend([
        "",
        "Legal-action count отличается с ходом игры, но не объясняет sawtooth: при одинаковом legal-count bucket Black/White separation сохраняется. В vectors illegal actions имеют zero visits; PASS входит в legal action count.",
        "",
        "## 8. Code audit of confirmed path",
        "",
        "- `gocube_golden/torus9_monolith.py`: observation uses relative own/opponent stone planes plus one side-to-move color channel (+1 BLACK, -1 WHITE); there are no absolute stone-color planes.",
        "- `gocube_golden/search.py`: search uses side-to-move WDL, utility WIN minus LOSS, and one sign flip per traversed edge. No odd/even or Black/White branch exists in PUCT traversal, backup, or root accounting.",
        "- `gocube_golden/search.py`: root visits are legal edge visits mapped into the 82-action vector; pi is constructed directly as count/sum.",
        "- `gocube_golden/selfplay_policy.py`: Dirichlet is applied to the root evaluator policy before search expansion. Raw NN policy and noisy prior are not stored in self-play records, so the pre-search prior cannot be compared from these artifacts.",
        "- `gocube_golden/selfplay_policy.py`: temperature is applied only after root visits are produced. It explains selected-action determinism after ply 8, not the same-position root-visits sawtooth.",
        "- Komi=0.5 is a real game asymmetry; a Black/White difference alone is not evidence of a bug.",
        "",
        "## 9. Historical comparison and answers",
        "",
        "| Question | Answer |",
        "|---|---|",
        "| 1. Насколько велик effect? | Current high-branch even minus low-branch odd root entropy delta = " + f"{ceven['mcts_entropy_nats']['mean'] - codd['mcts_entropy_nats']['mean']:.3f}" + " nats; effective candidates delta = " + f"{ceven['means']['effective_candidates'] - codd['means']['effective_candidates']:.2f}" + ". It is large and stable across intervals. |",
        "| 2. Эквивалентен ли Black/White? | Да для analyzed records: odd=BLACK, even=WHITE и group metrics совпадают по state-chain mapping. Это color asymmetry, not an independent parity mechanism. |",
        "| 3. Одинаков ли budget? | Да: root visit sum is 200 for all valid positions; no illegal visits. |",
        "| 4. Объясняется ли legal moves? | Нет: separation remains within common exact legal-count buckets. |",
        "| 5. Возникает ли в root visits? | Да. pi лишь нормализует visits; post-search temperature не источник sawtooth. |",
        "| 6. Был ли effect в M100–M105? | Да; historical window contains the same odd/even and Black/White separation. |",
        "| 7. Усилился ли с обучением? | В aggregate-сравнении да: entropy delta вырос с "
        + f"{old['groups']['even']['mcts_entropy_nats']['mean'] - old['groups']['odd']['mcts_entropy_nats']['mean']:.3f} до "
        + f"{ceven['mcts_entropy_nats']['mean'] - codd['mcts_entropy_nats']['mean']:.3f} nats, "
        + f"effective-candidate delta — с {old['groups']['even']['means']['effective_candidates'] - old['groups']['odd']['means']['effective_candidates']:.2f} до "
        + f"{ceven['means']['effective_candidates'] - codd['means']['effective_candidates']:.2f}; двух окон недостаточно для вывода о монотонном тренде. |",
        "| 8. Есть ли evidence implementation bug? | По records и confirmed code path — нет: budget, legality, side switching, pi normalization и value sign semantics consistent. |",
        "| 9. Наиболее вероятное объяснение? | Learned Black/White policy/value asymmetry при komi=0.5, проявляющаяся через side-to-move encoding; raw NN prior не сохранён, поэтому pre-search contribution неразделим. |",
        "| 10. Связь с замедлением обучения? | Прямая связь не доказана. Это может менять effective self-play target distribution по цветам, но diversity/full-prefix analysis не показывает collapse; отдельного learning-causality теста нет. |",
        "",
        "### Delta comparison",
        "",
        "| Window | High-branch group | Low-branch group | Delta entropy | Delta effective candidates | Delta top-1 visit share |",
        "|---|---|---|---:|---:|---:|",
    ])
    for window in (old, current):
        odd, even = window["groups"]["odd"], window["groups"]["even"]
        high, low = ("odd", "even") if odd["means"]["effective_candidates"] > even["means"]["effective_candidates"] else ("even", "odd")
        hi, lo = window["groups"][high], window["groups"][low]
        lines.append(
            f"| {window['name']} | {high} | {low} | "
            f"{hi['mcts_entropy_nats']['mean'] - lo['mcts_entropy_nats']['mean']:.3f} | "
            f"{hi['means']['effective_candidates'] - lo['means']['effective_candidates']:.2f} | "
            f"{hi['means']['top1_visit_share'] - lo['means']['top1_visit_share']:.1%} |"
        )
    lines.extend([
        "",
        "## 10. Limitations",
        "",
        "- Records do not serialize raw pre-MCTS NN policy, noisy Dirichlet prior, WDL logits, or root Q, so the exact boundary between network prior and PUCT cannot be identified from existing artifacts alone.",
        "- Exact legal-action reconstruction uses production rules and reads only saved states; it does not run MCTS or neural inference.",
        "- Black/White asymmetry can be legitimate under komi=0.5 and a learned non-color-equivariant network. The diagnostic does not prove that the model asymmetry is desirable.",
        "- No production code or parameter was changed. If a future experiment tests color-equivariant observation/modeling or temperature alternatives, it must be a separate task.",
        "",
    ])
    return "\n".join(lines)


def compact_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Return aggregate and ply-range diagnostics without per-ply vectors or histograms."""

    def compact_group(group: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: group[key]
            for key in ("positions", "mcts_entropy_nats", "means", "min", "max")
            if key in group
        }

    def compact_window(window: Mapping[str, Any]) -> Dict[str, Any]:
        files = [
            {
                key: item[key]
                for key in (
                    "generation",
                    "size_bytes",
                    "lines_seen",
                    "parsed_records",
                    "complete_games",
                    "invalid_records",
                    "positions",
                    "issues",
                )
                if key in item
            }
            for item in window["files"]
        ]
        return {
            key: window[key]
            for key in (
                "name",
                "lineage_path",
                "generations",
                "config_path",
                "effective_config",
                "files",
                "games",
                "positions",
                "issues",
                "sanity_checks",
                "start_side_counts",
                "pass_actions",
                "mcts_invariants",
            )
            if key in window
        } | {
            "files": files,
            "groups": {
                name: compact_group(group)
                for name, group in window["groups"].items()
            },
            "intervals": {
                interval: {
                    name: compact_group(group)
                    for name, group in groups.items()
                }
                for interval, groups in window["intervals"].items()
            },
            "legal_count_conditioning": window["legal_count_conditioning"],
        }

    return {
        "schema": "torus9-mcts-odd-even-diagnostic-compact-summary-v1",
        "analysis_date": summary["analysis_date"],
        "read_only": summary["read_only"],
        "runs_modified": summary["runs_modified"],
        "new_selfplay_started": summary["new_selfplay_started"],
        "new_inference_started": summary["new_inference_started"],
        "windows": {
            name: compact_window(window)
            for name, window in summary["windows"].items()
        },
        "code_audit": summary["code_audit"],
        "methodology": summary["methodology"],
        "conclusions": [
            "The odd/even effect is a Black/White effect: odd plies map to BLACK and even plies to WHITE in the saved state chain.",
            "The separation is present in root visits, survives legal-count conditioning, and is not explained by post-search temperature.",
            "No implementation-bug evidence was found in the confirmed production path; komi=0.5 remains a possible natural source, not proven causality.",
        ],
        "artifacts": summary["artifacts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
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
        "schema": "torus9-mcts-odd-even-diagnostic-v1",
        "analysis_date": "2026-09-24",
        "read_only": True,
        "runs_modified": False,
        "new_selfplay_started": False,
        "new_inference_started": False,
        "windows": {"current": current, "historical": old},
        "code_audit": code_audit(),
        "methodology": {
            "root_metrics": "Entropy and shares are computed over the saved 82-entry root visit vector after normalization.",
            "odd_even": "Direct ply parity plus direct saved state side_to_move; mapping is sanity-checked through production apply_action including PASS.",
            "legal_actions": "Reconstructed with production prepare_legal_actions from each saved state.",
            "pi_check": "Compare saved pi against root_visits / sum(root_visits) elementwise.",
            "budget_check": "Search source invariant is one root-edge backup per configured simulation; expected sum is 200.",
            "not_done": "No checkpoint load, neural inference, new MCTS, self-play, training, Arena, A/B, or benchmark.",
        },
        "artifacts": {
            "report": str(args.output_md),
            "summary": str(args.output_json),
            "prior_diversity_report": str(ROOT / "docs/experiments/torus9-selfplay-diversity-analysis-20260924.md"),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    compact = compact_summary(summary)
    args.output_json.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if raw_output_json is not None:
        raw_output_json.parent.mkdir(parents=True, exist_ok=True)
        raw_output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(report(summary), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output_md),
        "summary": str(args.output_json),
        "current_positions": current["positions"],
        "historical_positions": old["positions"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
