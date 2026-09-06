from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SelfPlayGuardResult:
    status: str
    warnings: tuple[str, ...]
    fatal_reasons: tuple[str, ...]
    metrics: dict[str, float | int]

    @property
    def training_allowed(self) -> bool:
        return self.status == "valid"


def _arg(args: Any, name: str, default):
    if hasattr(args, "get"):
        value = args.get(name, default)
    else:
        value = getattr(args, name, default)
    return value


def evaluate_selfplay_guard(
    telemetry: Mapping[str, float | int],
    *,
    games: int,
    args: Any,
) -> SelfPlayGuardResult:
    """Classify one finished self-play iteration before optimizer training.

    The gate is intentionally based only on already collected counters. It
    never changes game rules or move legality. A pathological iteration remains
    on disk for diagnosis but must not be admitted to the optimizer window.
    """

    games = int(games)
    total_decisions = int(
        telemetry.get("phase_main_decisions", 0)
        + telemetry.get("phase_cleanup1_decisions", 0)
        + telemetry.get("phase_cleanup2_decisions", 0)
    )
    early_within_2 = int(telemetry.get("main_double_pass_within_2", 0))
    early_within_4 = int(telemetry.get("main_double_pass_within_4", 0))
    early_within_8 = int(telemetry.get("main_double_pass_within_8", 0))
    # On both Cube4 and Torus9, ending MAIN inside the first eight plies is
    # unambiguously premature. Using only <=2 missed the fixed-20 recovery
    # pathology where 45% of games double-passed by ply 4.
    early_count = early_within_8
    early_rate = early_count / games if games else 0.0
    cleanup2_decisions = int(telemetry.get("phase_cleanup2_decisions", 0))
    cleanup2_fraction = cleanup2_decisions / total_decisions if total_decisions else 0.0
    audited = int(telemetry.get("search_audited_positions", 0))
    dominated = int(telemetry.get("search_score_dominated_pass", 0))
    dominated_rate = dominated / audited if audited else 0.0

    metrics: dict[str, float | int] = {
        "games": games,
        "main_double_pass_within_2": early_within_2,
        "main_double_pass_within_4": early_within_4,
        "main_double_pass_within_8": early_within_8,
        "main_early_double_pass_rate": early_rate,
        "cleanup2_decisions": cleanup2_decisions,
        "total_phase_decisions": total_decisions,
        "cleanup2_fraction": cleanup2_fraction,
        "search_audited_positions": audited,
        "search_score_dominated_pass": dominated,
        "search_score_dominated_pass_rate": dominated_rate,
    }

    warnings: list[str] = []
    fatal: list[str] = []
    min_games = int(_arg(args, "gocube_guard_min_games", 32))
    early_warning = float(_arg(args, "gocube_early_double_pass_warning_rate", 0.01))
    early_fatal = float(_arg(args, "gocube_early_double_pass_fatal_rate", 0.05))
    cleanup_warning = float(_arg(args, "gocube_cleanup2_warning_fraction", 0.50))
    cleanup_fatal = float(_arg(args, "gocube_cleanup2_fatal_fraction", 0.70))
    audit_min = int(_arg(args, "gocube_score_audit_min_positions", 16))
    dominated_fatal = float(_arg(args, "gocube_score_dominated_pass_fatal_rate", 0.25))

    if games >= min_games:
        if early_rate >= early_fatal:
            fatal.append(
                f"early MAIN double-pass by ply 8 rate {early_rate:.2%} >= fatal {early_fatal:.2%} "
                f"({early_count}/{games}; <=2:{early_within_2}, <=4:{early_within_4})"
            )
        elif early_rate >= early_warning:
            warnings.append(
                f"early MAIN double-pass by ply 8 rate {early_rate:.2%} >= warning {early_warning:.2%} "
                f"({early_count}/{games}; <=2:{early_within_2}, <=4:{early_within_4})"
            )

        if cleanup2_fraction > cleanup_fatal:
            fatal.append(
                f"CLEANUP_2 decision fraction {cleanup2_fraction:.2%} > fatal {cleanup_fatal:.2%}"
            )
        elif cleanup2_fraction > cleanup_warning:
            warnings.append(
                f"CLEANUP_2 decision fraction {cleanup2_fraction:.2%} > warning {cleanup_warning:.2%}"
            )

    if audited >= audit_min and dominated_rate >= dominated_fatal:
        fatal.append(
            f"escaped score-dominated second-PASS rate {dominated_rate:.2%} >= fatal {dominated_fatal:.2%} "
            f"({dominated}/{audited} audited positions)"
        )

    return SelfPlayGuardResult(
        status="invalid_selfplay" if fatal else "valid",
        warnings=tuple(warnings),
        fatal_reasons=tuple(fatal),
        metrics=metrics,
    )


def phase_fractions(telemetry: Mapping[str, float | int]) -> dict[str, float]:
    main = float(telemetry.get("phase_main_decisions", 0))
    cleanup1 = float(telemetry.get("phase_cleanup1_decisions", 0))
    cleanup2 = float(telemetry.get("phase_cleanup2_decisions", 0))
    total = main + cleanup1 + cleanup2
    if total <= 0:
        return {"main": 0.0, "cleanup1": 0.0, "cleanup2": 0.0}
    return {
        "main": main / total,
        "cleanup1": cleanup1 / total,
        "cleanup2": cleanup2 / total,
    }
