"""Pure statistics and komi selection policy.

The production 1.5/2.5 extension rule lives here so it cannot accidentally be
reimplemented by a coordinator or a notification handler.  Batch evidence is
validated before counters are combined; percentages are never averaged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math

from ...provenance import sha256_fingerprint


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    if type(successes) is not int or type(total) is not int or successes < 0 or successes > total:
        raise ValueError("Wilson interval counters are invalid")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _error(message: str) -> ValueError:
    return ValueError(message)


def aggregate_candidate_batches(
    batches: Sequence[Mapping[str, object]],
    *,
    expected_komi: float | None = None,
    expected_startset_fingerprint: str | None = None,
    initial_games: int | None = None,
    extension_games: int | None = None,
) -> dict[str, object]:
    """Aggregate ordered valid batch evidence with fail-closed validation."""

    if not batches:
        raise _error("cannot aggregate an empty komi calibration")
    ordered = sorted((deepcopy(dict(batch)) for batch in batches), key=lambda item: int(item.get("batch", 0)))
    expected_batch = 1
    total_games = total_valid = total_black = total_white = total_draws = 0
    margins: list[object] = []
    startset_fingerprint: str | None = expected_startset_fingerprint
    evaluation_ids: list[str] = []
    evaluation_fingerprints: list[str] = []
    output_dirs: list[str] = []
    observed_komi = expected_komi
    seen_batches: set[int] = set()

    for evidence in ordered:
        batch_number = int(evidence.get("batch", 0))
        if batch_number in seen_batches:
            raise _error("komi calibration ledger contains a duplicate batch")
        seen_batches.add(batch_number)
        if batch_number != expected_batch:
            raise _error("komi calibration batch evidence is not contiguous")
        expected_batch += 1
        if observed_komi is None:
            try:
                observed_komi = float(evidence["komi"])
            except (KeyError, TypeError, ValueError) as exc:
                raise _error("komi calibration batch komi is missing") from exc
        if float(evidence.get("komi", float("nan"))) != observed_komi:
            raise _error("komi calibration batches contain different candidates")
        if evidence.get("validity") != "VALID":
            raise _error("komi calibration batch is not VALID")

        stats = evidence.get("stats")
        identity = evidence.get("identity")
        if not isinstance(stats, Mapping) or not isinstance(identity, Mapping):
            raise _error("komi calibration batch evidence is malformed")
        technical = int(stats.get("technical_games", 0) or 0)
        invalid = int(stats.get("invalid_games", 0) or 0)
        valid = int(stats.get("valid_games", 0) or 0)
        games = int(stats.get("games", valid) or 0)
        black = int(stats.get("black_wins", 0) or 0)
        white = int(stats.get("white_wins", 0) or 0)
        draws = int(stats.get("draws", 0) or 0)
        if min(technical, invalid, valid, games, black, white, draws) < 0:
            raise _error("komi calibration batch counters are negative")
        if technical or invalid or valid <= 0 or black + white + draws != valid or games != valid:
            raise _error("komi calibration batch counters are inconsistent")
        expected_size = initial_games if batch_number == 1 else extension_games
        if expected_size is not None and valid != int(expected_size):
            raise _error(f"komi calibration batch {batch_number} has an unexpected size")

        startset = identity.get("startset")
        if not isinstance(startset, Mapping):
            raise _error("komi calibration batch lacks frozen startset identity")
        fingerprint = str(startset.get("fingerprint", ""))
        if not fingerprint:
            raise _error("komi calibration batch lacks startset fingerprint")
        if startset_fingerprint is None:
            startset_fingerprint = fingerprint
        elif startset_fingerprint != fingerprint:
            raise _error("komi calibration batches use different startsets")

        evaluation_id = str(evidence.get("evaluation_id", ""))
        evaluation_fingerprint = str(evidence.get("evaluation_fingerprint", ""))
        if not evaluation_id or not evaluation_fingerprint:
            raise _error("komi calibration batch lacks evaluation identity")
        if evaluation_id in evaluation_ids or evaluation_fingerprint in evaluation_fingerprints:
            raise _error("komi calibration ledger contains a duplicate evaluation")

        total_games += games
        total_valid += valid
        total_black += black
        total_white += white
        total_draws += draws
        raw_margins = stats.get("raw_score_margin_histogram")
        if isinstance(raw_margins, list):
            margins.extend(raw_margins)
        evaluation_ids.append(evaluation_id)
        evaluation_fingerprints.append(evaluation_fingerprint)
        output_dirs.append(str(evidence.get("output_dir", "")))

    black_rate = total_black / total_valid
    result = deepcopy(ordered[-1])
    result.update(
        {
            "aggregate": True,
            "batch": len(ordered),
            "evaluation_id": "aggregate:" + "+".join(evaluation_ids),
            "evaluation_fingerprint": sha256_fingerprint(
                {"kind": "komi-calibration-cumulative-v1", "batch_fingerprints": evaluation_fingerprints}
            ),
            "batch_evaluation_ids": evaluation_ids,
            "batch_output_dirs": output_dirs,
            "batches": ordered,
            "summary": {
                "games": total_games,
                "valid_games": total_valid,
                "technical_games": 0,
                "invalid_games": 0,
                "black_wins": total_black,
                "white_wins": total_white,
                "draws": total_draws,
            },
            "stats": {
                "games": total_games,
                "valid_games": total_valid,
                "technical_games": 0,
                "invalid_games": 0,
                "black_wins": total_black,
                "white_wins": total_white,
                "draws": total_draws,
                "black_win_rate": black_rate,
                "bias": abs(black_rate - 0.5),
                "confidence_interval": wilson_interval(total_black, total_valid),
                "confidence_interval_method": "wilson-95-percent",
                "raw_score_margin_histogram": margins or None,
                "batch_evaluation_ids": evaluation_ids,
            },
        }
    )
    return result


@dataclass(frozen=True)
class KomiDecision:
    selected_komi: float
    requires_extension: bool
    initial_biases: Mapping[float, float]
    cumulative_biases: Mapping[float, float] | None = None
    rule: str = "production-komi-selection-v1"


def select_komi(
    initial_biases: Mapping[float, float],
    *,
    candidates: Sequence[float] = (1.5, 2.5),
    ambiguity_threshold: float = 0.01,
    cumulative_biases: Mapping[float, float] | None = None,
) -> KomiDecision:
    """Apply the current selection/extension rule, including its strict ``<``."""

    configured = tuple(float(value) for value in candidates)
    if not configured or len(set(configured)) != len(configured):
        raise ValueError("komi candidates must be a non-empty unique sequence")
    if any(value not in initial_biases for value in configured):
        raise ValueError("initial bias is missing for a configured komi")
    initial = {value: float(initial_biases[value]) for value in configured}
    if any(not math.isfinite(value) or value < 0 for value in initial.values()):
        raise ValueError("komi biases must be finite non-negative numbers")
    if configured != (1.5, 2.5):
        minimum = min(initial.values())
        tied = [komi for komi in configured if initial[komi] == minimum]
        selected = 1.5 if 1.5 in tied else tied[0]
        return KomiDecision(selected, False, initial)

    ambiguous = abs(initial[1.5] - initial[2.5]) < float(ambiguity_threshold)
    if not ambiguous:
        return KomiDecision(1.5 if initial[1.5] <= initial[2.5] else 2.5, False, initial)
    if cumulative_biases is None:
        return KomiDecision(1.5, True, initial)
    cumulative = {value: float(cumulative_biases[value]) for value in configured}
    if any(not math.isfinite(value) or value < 0 for value in cumulative.values()):
        raise ValueError("cumulative komi biases must be finite non-negative numbers")
    selected = 1.5 if cumulative[1.5] <= cumulative[2.5] else 2.5
    return KomiDecision(selected, True, initial, cumulative)


__all__ = ["KomiDecision", "aggregate_candidate_batches", "select_komi", "wilson_interval"]
