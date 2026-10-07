"""Pure diagnostics and audit-only instrumentation for the continuation gate.

This module intentionally contains no production search or inference changes.
The tracing helpers subclass/wrap the immutable runtime objects only inside an
opt-in benchmark process; report functions operate on serialized evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EVALUATION_STREAM_SCHEMA = "torus9-fixed-evaluation-stream-v1"
ROOT_TRACE_SCHEMA = "torus9-sequential-puct-root-trace-v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def evaluation_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session_request_index": int(request["session_request_index"]),
        "state_sha256": str(request["state_sha256"]),
        "legal_mask_sha256": str(request["legal_mask_sha256"]),
        "observation_sha256": str(request["observation_sha256"]),
    }


def serialize_evaluation_stream(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Return only exact request identities and the Evaluation values consumed."""
    evaluations = []
    requests = list(trace.get("evaluation_requests", ()))
    for request in requests:
        if "policy" not in request or "wdl" not in request:
            raise ValueError("root trace contains an Evaluation request without values")
        evaluations.append({
            "request": evaluation_identity(request),
            "policy": [float(value) for value in request["policy"]],
            "wdl": [float(value) for value in request["wdl"]],
            "policy_sha256": str(request["policy_sha256"]),
            "wdl_sha256": str(request["wdl_sha256"]),
        })
    if not evaluations:
        raise ValueError("cannot serialize an empty Evaluation stream")
    return {
        "schema": EVALUATION_STREAM_SCHEMA,
        "root_identity": dict(trace["root_identity"]),
        "search_seed": int(trace["search_seed"]),
        "simulation_cap": int(trace["simulation_cap"]),
        "settings": dict(trace["settings"]),
        "noise": dict(trace["noise"]),
        "evaluations": evaluations,
    }


def first_divergent_simulation(left: Mapping[str, Any],
                               right: Mapping[str, Any]) -> dict[str, Any] | None:
    """Localize the earliest observable difference in two root traces."""
    left_requests = list(left.get("evaluation_requests", ()))
    right_requests = list(right.get("evaluation_requests", ()))
    left_sims = list(left.get("simulations", ()))
    right_sims = list(right.get("simulations", ()))
    if not left_requests or not right_requests:
        return {"simulation_index": 0, "category": "missing_root_evaluation"}
    left_root, right_root = left_requests[0], right_requests[0]
    if evaluation_identity(left_root) != evaluation_identity(right_root):
        return {"simulation_index": 0, "category": "root_neural_input",
                "left_request": evaluation_identity(left_root),
                "right_request": evaluation_identity(right_root)}
    if (left_root.get("policy_sha256") != right_root.get("policy_sha256")
            or left_root.get("wdl_sha256") != right_root.get("wdl_sha256")):
        return {"simulation_index": 0, "category": "root_neural_evaluation",
                "request": evaluation_identity(left_root),
                "left_policy_sha256": left_root.get("policy_sha256"),
                "right_policy_sha256": right_root.get("policy_sha256"),
                "left_wdl_sha256": left_root.get("wdl_sha256"),
                "right_wdl_sha256": right_root.get("wdl_sha256"),
                "left_policy": left_root.get("policy"),
                "right_policy": right_root.get("policy"),
                "left_wdl": left_root.get("wdl"), "right_wdl": right_root.get("wdl")}
    if left.get("root_priors") != right.get("root_priors"):
        return {"simulation_index": 0, "category": "root_evaluation_or_noise",
                "left_root_priors": left.get("root_priors"),
                "right_root_priors": right.get("root_priors")}
    n = max(len(left_sims), len(right_sims))
    for index in range(n):
        ls = left_sims[index] if index < len(left_sims) else None
        rs = right_sims[index] if index < len(right_sims) else None
        if ls is None or rs is None:
            return {"simulation_index": index,
                    "category": "simulation_count",
                    "left": ls, "right": rs}
        # PUCT decisions and traversal happen before this simulation's leaf Evaluation.
        if ls.get("puct_decisions") != rs.get("puct_decisions"):
            return {"simulation_index": index, "category": "puct_arithmetic_or_choice",
                    "left_decisions": ls.get("puct_decisions"),
                    "right_decisions": rs.get("puct_decisions")}
        if ls.get("traversal_path") != rs.get("traversal_path"):
            return {"simulation_index": index,
                    "category": "puct_selected_path",
                    "left_path": ls.get("traversal_path"),
                    "right_path": rs.get("traversal_path"),
                    "left_decisions": ls.get("puct_decisions"),
                    "right_decisions": rs.get("puct_decisions"),
                    "left_leaf": ls.get("leaf"), "right_leaf": rs.get("leaf")}
        if ls.get("leaf") != rs.get("leaf"):
            return {"simulation_index": index,
                    "category": "leaf_identity",
                    "left_leaf": ls.get("leaf"), "right_leaf": rs.get("leaf")}
        li = int(ls.get("evaluation_request_index", -1))
        ri = int(rs.get("evaluation_request_index", -1))
        if li != ri:
            return {"simulation_index": index, "category": "request_order",
                    "left_request_index": li, "right_request_index": ri}
        if li >= 0:
            if li >= len(left_requests) or ri >= len(right_requests):
                return {"simulation_index": index, "category": "missing_evaluation"}
            lreq, rreq = left_requests[li], right_requests[ri]
            if evaluation_identity(lreq) != evaluation_identity(rreq):
                return {"simulation_index": index, "category": "neural_input",
                        "left_request": evaluation_identity(lreq),
                        "right_request": evaluation_identity(rreq)}
            if (lreq.get("policy_sha256") != rreq.get("policy_sha256")
                    or lreq.get("wdl_sha256") != rreq.get("wdl_sha256")):
                return {"simulation_index": index, "category": "neural_evaluation",
                        "request": evaluation_identity(lreq),
                        "left_policy_sha256": lreq.get("policy_sha256"),
                        "right_policy_sha256": rreq.get("policy_sha256"),
                        "left_wdl_sha256": lreq.get("wdl_sha256"),
                        "right_wdl_sha256": rreq.get("wdl_sha256"),
                        "left_policy": lreq.get("policy"),
                        "right_policy": rreq.get("policy"),
                        "left_wdl": lreq.get("wdl"), "right_wdl": rreq.get("wdl")}
        if ls.get("root_visits_after") != rs.get("root_visits_after"):
            return {"simulation_index": index, "category": "backup_or_root_visits",
                    "left_root_visits": ls.get("root_visits_after"),
                    "right_root_visits": rs.get("root_visits_after")}
    return None


def compare_root_traces(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    same_root = left.get("root_identity") == right.get("root_identity")
    same_seed = left.get("search_seed") == right.get("search_seed")
    first = first_divergent_simulation(left, right)
    return {
        "same_root": same_root,
        "same_search_seed": same_seed,
        "same_evaluation_count": len(left.get("evaluation_requests", ()))
                                   == len(right.get("evaluation_requests", ())),
        "first_divergence": first,
        "identical": same_root and same_seed and first is None
                     and left.get("root_visits") == right.get("root_visits")
                     and left.get("selected_action") == right.get("selected_action"),
    }


def compare_controlled_geometry_traces(
        left: Mapping[str, Any], right: Mapping[str, Any], *,
        left_repeats: Sequence[Mapping[str, Any]] = (),
        right_repeats: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Assess a controlled root search where only nonroot batch geometry varies."""
    left_requests = list(left.get("evaluation_requests", ()))
    right_requests = list(right.get("evaluation_requests", ()))
    first_evaluation = None
    for index, (lreq, rreq) in enumerate(zip(left_requests, right_requests)):
        identity_left, identity_right = evaluation_identity(lreq), evaluation_identity(rreq)
        if identity_left != identity_right:
            first_evaluation = {"request_index": index, "same_request_identity": False,
                                "left": identity_left, "right": identity_right}
            break
        if (lreq.get("policy_sha256") != rreq.get("policy_sha256")
                or lreq.get("wdl_sha256") != rreq.get("wdl_sha256")):
            first_evaluation = {
                "request_index": index, "same_request_identity": True,
                "request": identity_left,
                "left_policy_sha256": lreq.get("policy_sha256"),
                "right_policy_sha256": rreq.get("policy_sha256"),
                "left_wdl_sha256": lreq.get("wdl_sha256"),
                "right_wdl_sha256": rreq.get("wdl_sha256"),
                "left_policy": lreq.get("policy"), "right_policy": rreq.get("policy"),
                "left_wdl": lreq.get("wdl"), "right_wdl": rreq.get("wdl"),
            }
            break

    first_puct = None
    first_path = None
    first_backup = None
    for index, (ls, rs) in enumerate(zip(left.get("simulations", ()),
                                         right.get("simulations", ()))):
        if first_puct is None and ls.get("puct_decisions") != rs.get("puct_decisions"):
            first_puct = {"simulation_index": index,
                          "left_decisions": ls.get("puct_decisions"),
                          "right_decisions": rs.get("puct_decisions")}
        if first_path is None and ls.get("traversal_path") != rs.get("traversal_path"):
            first_path = {"simulation_index": index,
                          "left_path": ls.get("traversal_path"),
                          "right_path": rs.get("traversal_path")}
        backup_fields = ("leaf_utility", "backup_utility", "root_visits_after")
        if first_backup is None and any(ls.get(field) != rs.get(field)
                                        for field in backup_fields):
            first_backup = {"simulation_index": index,
                            "left": {field: ls.get(field) for field in backup_fields},
                            "right": {field: rs.get(field) for field in backup_fields}}

    left_visits = [int(value) for value in left.get("root_visits", ())]
    right_visits = [int(value) for value in right.get("root_visits", ())]
    moved_left = sum(max(0, a - b) for a, b in zip(left_visits, right_visits))
    moved_right = sum(max(0, b - a) for a, b in zip(left_visits, right_visits))
    same_controls = (
        left.get("root_identity") == right.get("root_identity")
        and left.get("search_seed") == right.get("search_seed")
        and left.get("simulation_cap") == right.get("simulation_cap")
        and left.get("settings") == right.get("settings")
        and left.get("noise") == right.get("noise"))
    root_eval_equal = bool(left_requests and right_requests
                           and evaluation_identity(left_requests[0])
                           == evaluation_identity(right_requests[0])
                           and left_requests[0].get("policy_sha256")
                           == right_requests[0].get("policy_sha256")
                           and left_requests[0].get("wdl_sha256")
                           == right_requests[0].get("wdl_sha256"))
    root_priors_equal = left.get("root_priors") == right.get("root_priors")

    def stable(repeats: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any]) -> bool:
        def signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
            return (row.get("root_visits"), row.get("root_priors"),
                    row.get("selected_action"),
                    [(r.get("policy_sha256"), r.get("wdl_sha256"), evaluation_identity(r))
                     for r in row.get("evaluation_requests", ())])
        expected = signature(baseline)
        return all(signature(row) == expected for row in repeats)

    left_stable = stable(left_repeats, left)
    right_stable = stable(right_repeats, right)
    first_leaf_same_input_changed_output = bool(
        len(left_requests) > 1 and len(right_requests) > 1
        and evaluation_identity(left_requests[1]) == evaluation_identity(right_requests[1])
        and (left_requests[1].get("policy_sha256") != right_requests[1].get("policy_sha256")
             or left_requests[1].get("wdl_sha256") != right_requests[1].get("wdl_sha256")))
    downstream_after_eval = bool(first_puct and first_leaf_same_input_changed_output
                                 and first_puct["simulation_index"] >= 1)
    one_visit = moved_left == moved_right == 1 and sum(left_visits) == sum(right_visits)
    selected_equal = left.get("selected_action") == right.get("selected_action")
    supported = (same_controls and root_eval_equal and root_priors_equal
                 and first_leaf_same_input_changed_output and downstream_after_eval
                 and first_path is not None and one_visit and selected_equal
                 and left_stable and right_stable)
    return {
        "same_search_controls": same_controls,
        "root_evaluation_exact": root_eval_equal,
        "root_priors_exact": root_priors_equal,
        "first_evaluation_difference": first_evaluation,
        "first_leaf_same_input_changed_output": first_leaf_same_input_changed_output,
        "first_puct_difference": first_puct,
        "first_selected_path_difference": first_path,
        "first_backup_difference": first_backup,
        "left_root_visits": left_visits, "right_root_visits": right_visits,
        "visits_moved_left_to_right": moved_left,
        "visits_moved_right_to_left": moved_right,
        "selected_action_left": left.get("selected_action"),
        "selected_action_right": right.get("selected_action"),
        "left_geometry_repeat_stable": left_stable,
        "right_geometry_repeat_stable": right_stable,
        "status": "PROVEN_FOR_THIS_ROOT" if supported else "UNKNOWN",
    }


def _max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("tensor value lengths differ")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def _max_relative_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("tensor value lengths differ")
    values = [abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)
              for a, b in zip(left, right)]
    return max(values, default=0.0)


def aggregate_batch_geometry(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate batch/row sweeps against the matching batch-1 baseline."""
    indexed = {(str(row["scope"]), int(row["repeat"]), int(row["batch_size"]),
                int(row["target_row"])): row for row in rows}
    variants: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["batch_size"]), int(row["target_row"]))
        variants.setdefault(key, []).append(row)
    summaries = []
    for (batch_size, target_row), observations in sorted(variants.items()):
        deltas = []
        repeat_hashes: list[str] = []
        for observation in observations:
            repeat_key = (str(observation["scope"]), int(observation["repeat"]), 1, 0)
            baseline = indexed.get(repeat_key)
            if baseline is None:
                raise ValueError(f"missing batch-1 baseline for {repeat_key[:2]}")
            pdelta = _max_abs_delta(observation["policy"], baseline["policy"])
            wdelta = _max_abs_delta(observation["wdl"], baseline["wdl"])
            deltas.append({
                "scope": observation["scope"], "repeat": int(observation["repeat"]),
                "max_policy_abs_delta": pdelta,
                "max_policy_relative_delta": _max_relative_delta(
                    observation["policy"], baseline["policy"]),
                "max_policy_delta_action": max(
                    range(len(observation["policy"])),
                    key=lambda i: abs(float(observation["policy"][i])
                                      - float(baseline["policy"][i])), default=None),
                "max_wdl_abs_delta": wdelta,
                "wdl_delta": [float(a) - float(b) for a, b in
                              zip(observation["wdl"], baseline["wdl"])],
                "max_policy_logits_abs_delta": _max_abs_delta(
                    observation["policy_logits"], baseline["policy_logits"]),
                "max_wdl_logits_abs_delta": _max_abs_delta(
                    observation["wdl_logits"], baseline["wdl_logits"]),
            })
            repeat_hashes.append(str(observation["policy_sha256"])
                                 + ":" + str(observation["wdl_sha256"]))
        summaries.append({
            "batch_size": batch_size, "target_row": target_row,
            "repeats": len(observations),
            "policy_wdl_hashes": sorted(set(repeat_hashes)),
            "bitwise_identical_across_repeats": len(set(repeat_hashes)) == 1,
            "max_policy_abs_delta_vs_batch1": max(
                (item["max_policy_abs_delta"] for item in deltas), default=0.0),
            "max_policy_relative_delta_vs_batch1": max(
                (item["max_policy_relative_delta"] for item in deltas), default=0.0),
            "max_wdl_abs_delta_vs_batch1": max(
                (item["max_wdl_abs_delta"] for item in deltas), default=0.0),
            "max_policy_logits_abs_delta_vs_batch1": max(
                (item["max_policy_logits_abs_delta"] for item in deltas), default=0.0),
            "max_wdl_logits_abs_delta_vs_batch1": max(
                (item["max_wdl_logits_abs_delta"] for item in deltas), default=0.0),
            "repeat_deltas": deltas,
        })
    return {"variants": summaries,
            "shape_or_row_dependent": any(
                not item["bitwise_identical_across_repeats"]
                or item["max_policy_abs_delta_vs_batch1"] != 0.0
                or item["max_wdl_abs_delta_vs_batch1"] != 0.0
                or item["max_policy_logits_abs_delta_vs_batch1"] != 0.0
                or item["max_wdl_logits_abs_delta_vs_batch1"] != 0.0
                for item in summaries if item["batch_size"] != 1 or item["target_row"] != 0)}


def _root_pair_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_positions = {int(p["ply"]): p for p in left["raw_record"]["positions"]}
    right_positions = {int(p["ply"]): p for p in right["raw_record"]["positions"]}
    common = sorted(set(left_positions) & set(right_positions))
    mismatches = []
    moved = Counter()
    l1_values = []
    max_delta_values = []
    kl_values = []
    selected_mismatch = 0
    one_visit = 0
    for ply in common:
        lp, rp = left_positions[ply], right_positions[ply]
        lv = [int(value) for value in lp["root_visits"]]
        rv = [int(value) for value in rp["root_visits"]]
        if lv != rv:
            if len(lv) != len(rv):
                raise ValueError("root visit vector length differs")
            total_moved = sum(max(0, a - b) for a, b in zip(lv, rv))
            moved[total_moved] += 1
            one_visit += total_moved == 1 and sum(lv) == sum(rv)
            lpi = ([float(value) for value in lp["pi"]] if "pi" in lp else
                   [value / sum(lv) for value in lv])
            rpi = ([float(value) for value in rp["pi"]] if "pi" in rp else
                   [value / sum(rv) for value in rv])
            l1 = sum(abs(a - b) for a, b in zip(lpi, rpi))
            max_delta = max((abs(a - b) for a, b in zip(lpi, rpi)), default=0.0)
            l1_values.append(l1)
            max_delta_values.append(max_delta)
            finite_terms = []
            infinite = False
            for p, q in zip(lpi, rpi):
                if p > 0.0 and q <= 0.0:
                    infinite = True
                    break
                if p > 0.0:
                    finite_terms.append(p * math.log(p / q))
            kl_values.append(None if infinite else sum(finite_terms))
            mismatches.append({"ply": ply, "visits_moved": total_moved,
                               "fraction_target_probability_moved": total_moved / 200.0,
                               "root_l1": l1, "root_max_abs_delta": max_delta,
                               "selected_action_left": lp["selected_action"],
                               "selected_action_right": rp["selected_action"]})
        selected_mismatch += lp["selected_action"] != rp["selected_action"]
    lraw, rraw = left["raw_record"], right["raw_record"]
    ltrace = lraw.get("final_action_trace", ())
    rtrace = rraw.get("final_action_trace", ())
    return {
        "game_id": str(left["game_id"]),
        "common_roots": len(common), "root_visit_mismatch_count": len(mismatches),
        "root_visit_mismatch_fraction": len(mismatches) / len(common) if common else 0.0,
        "one_visit_redistribution_count": one_visit,
        "visits_moved_distribution": {str(key): int(value) for key, value in sorted(moved.items())},
        "max_visit_count_delta_in_root": max(
            (item["visits_moved"] for item in mismatches), default=0),
        "selected_action_mismatch_count": selected_mismatch,
        "action_trace_mismatch": list(ltrace) != list(rtrace),
        "result_mismatch": lraw.get("formal_result") != rraw.get("formal_result"),
        "ply_length_mismatch": len(ltrace) != len(rtrace),
        "mean_mismatched_root_l1": (sum(l1_values) / len(l1_values) if l1_values else 0.0),
        "max_mismatched_root_l1": max(l1_values, default=0.0),
        "max_per_action_pi_delta": max(max_delta_values, default=0.0),
        "max_kl_left_to_right": (None if any(v is None for v in kl_values)
                                  else max(kl_values, default=0.0)),
        "mismatched_roots": mismatches,
    }


def compare_game_runs(left_rows: Mapping[str, Mapping[str, Any]],
                      right_rows: Mapping[str, Mapping[str, Any]],
                      label: str) -> dict[str, Any]:
    common_games = sorted(set(left_rows) & set(right_rows))
    game_metrics = [_root_pair_metrics(left_rows[key], right_rows[key]) for key in common_games]
    total_roots = sum(item["common_roots"] for item in game_metrics)
    mismatch_roots = sum(item["root_visit_mismatch_count"] for item in game_metrics)
    return {
        "comparison": label,
        "same_game_id_set": set(left_rows) == set(right_rows),
        "left_games": len(left_rows), "right_games": len(right_rows),
        "common_games": len(common_games),
        "total_common_roots": total_roots,
        "root_visit_mismatch_count": mismatch_roots,
        "root_visit_mismatch_fraction": mismatch_roots / total_roots if total_roots else 0.0,
        "games_with_root_mismatch": sum(item["root_visit_mismatch_count"] > 0
                                         for item in game_metrics),
        "selected_action_mismatch_count": sum(
            item["selected_action_mismatch_count"] for item in game_metrics),
        "action_trace_mismatch_games": sum(item["action_trace_mismatch"] for item in game_metrics),
        "result_mismatch_games": sum(item["result_mismatch"] for item in game_metrics),
        "length_mismatch_games": sum(item["ply_length_mismatch"] for item in game_metrics),
        "one_visit_redistribution_count": sum(
            item["one_visit_redistribution_count"] for item in game_metrics),
        "max_visit_count_delta_in_root": max(
            (item["max_visit_count_delta_in_root"] for item in game_metrics), default=0),
        "mean_mismatched_root_l1": (
            sum(item["mean_mismatched_root_l1"] * item["root_visit_mismatch_count"]
                for item in game_metrics) / mismatch_roots if mismatch_roots else 0.0),
        "max_mismatched_root_l1": max(
            (item["max_mismatched_root_l1"] for item in game_metrics), default=0.0),
        "max_per_action_pi_delta": max(
            (item["max_per_action_pi_delta"] for item in game_metrics), default=0.0),
        "max_kl_left_to_right": max(
            (item["max_kl_left_to_right"] for item in game_metrics
             if item["max_kl_left_to_right"] is not None), default=0.0),
        "kl_has_infinite_support_mismatch": any(
            item["max_kl_left_to_right"] is None for item in game_metrics),
        "per_game": game_metrics,
    }


def aggregate_same_revision_baseline(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = ("root_visit_mismatch_fraction", "selected_action_mismatch_count",
                    "action_trace_mismatch_games", "result_mismatch_games",
                    "max_mismatched_root_l1", "one_visit_redistribution_count")
    ranges = {}
    for name in metric_names:
        values = [float(row[name]) for row in comparisons]
        ranges[name] = {"min": min(values, default=0.0), "max": max(values, default=0.0),
                        "mean": sum(values) / len(values) if values else 0.0}
    return {"pair_count": len(comparisons), "comparisons": list(comparisons),
            "metric_ranges": ranges,
            "all_pairs_bitwise_identical": all(
                float(row["root_visit_mismatch_fraction"]) == 0.0
                and int(row["selected_action_mismatch_count"]) == 0
                and int(row["action_trace_mismatch_games"]) == 0
                and int(row["result_mismatch_games"]) == 0
                for row in comparisons)}


def classify_against_baseline(comparison: Mapping[str, Any],
                              baseline: Mapping[str, Any]) -> dict[str, Any]:
    ranges = baseline["metric_ranges"]
    metrics = ("root_visit_mismatch_fraction", "selected_action_mismatch_count",
               "action_trace_mismatch_games", "result_mismatch_games",
               "max_mismatched_root_l1", "one_visit_redistribution_count")
    detail = {name: {
        "observed": comparison[name], "baseline_min": ranges[name]["min"],
        "baseline_max": ranges[name]["max"],
        "inside_observed_range": ranges[name]["min"] <= float(comparison[name]) <= ranges[name]["max"],
    } for name in metrics}
    return {"comparison": comparison["comparison"], "metrics": detail,
            "inside_all_observed_AA_ranges": all(
                value["inside_observed_range"] for value in detail.values()),
            "interpretation": "descriptive empirical range; not a statistical equivalence test"}


def final_verdicts(*, deterministic_core_pass: bool,
                   cross_revision_parity_pass: bool,
                   production_bitwise_reproducible: bool,
                   target_builder_pass: bool, optimizer_pass: bool, pcr_pass: bool,
                   inference_mapping_pass: bool, artifact_safety_pass: bool,
                   root_cause: str, production_baseline_consistent: bool) -> dict[str, str]:
    if root_cause not in {"PROVEN", "UNKNOWN", "REGRESSION"}:
        raise ValueError(f"invalid root-cause status: {root_cause}")
    core = "PASS" if deterministic_core_pass else "FAIL"
    cross = "PASS" if cross_revision_parity_pass else "FAIL"
    supported = all((deterministic_core_pass, cross_revision_parity_pass,
                     target_builder_pass, optimizer_pass, pcr_pass,
                     inference_mapping_pass, artifact_safety_pass,
                     root_cause == "PROVEN", production_baseline_consistent))
    return {
        "deterministic_scientific_core": core,
        "cross_revision_semantic_parity": cross,
        "production_concurrent_bitwise_reproducibility": (
            "PASS" if production_bitwise_reproducible else "FAIL"),
        "root_cause": root_cause,
        "continuation_safety": "SUPPORTED" if supported else "NOT SUPPORTED",
    }


def trace_batch_match(trace_request: Mapping[str, Any],
                      batch_row: Mapping[str, Any]) -> bool:
    return (trace_request.get("observation_sha256") == batch_row.get("observation_sha256")
            and trace_request.get("policy_sha256") == batch_row.get("policy_sha256")
            and trace_request.get("wdl_sha256") == batch_row.get("wdl_sha256"))

