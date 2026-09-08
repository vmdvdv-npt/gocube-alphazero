#!/usr/bin/env python3
"""Aggregate B4 per-seed evaluations with the registered bootstrap."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphazero.envs.gocube.b_evaluation import (
    B_BOOTSTRAP_REPLICATES,
    B_BOOTSTRAP_SEED,
    B_EXTENSION_CRITERION_ID,
    B_GAMES_PER_POSITION,
    B_HELDOUT_POSITION_IDS,
    B_HELDOUT_DEPTH_SCHEDULE,
    B_HELDOUT_GENERATOR_ID,
    B_HELDOUT_GENERATOR_MASTER_SEED,
    B_HELDOUT_SUITE_ID,
    B_HELDOUT_SUITE_POSITION_COUNT,
    B_HELDOUT_SUITE_SHA256,
    B_STATISTICAL_METHOD_IDENTIFIER,
    classify_delta_interval,
    extension_seed_decision,
    hierarchical_paired_bootstrap,
    outcome_score_b1,
    summarize_game_diagnostics,
    validate_pairing_invariants,
)
from alphazero.envs.gocube.b_experiment_contract import (
    B_EXPERIMENT_CONTRACT_ID,
    B_EXPERIMENT_CONTRACT_VERSION,
    B_SEED_LIST,
)


def _require_equal(artifacts: Sequence[Mapping[str, object]], key: str) -> object:
    values = {json.dumps(item.get(key), sort_keys=True) for item in artifacts}
    if len(values) != 1:
        raise ValueError(f"B evaluation artifacts disagree on {key}")
    return artifacts[0].get(key)


def _validate_game_record(game: Mapping[str, object]) -> None:
    raw = str(game.get("raw_outcome", ""))
    expected_score = outcome_score_b1(raw)
    normalized_raw = raw.strip().lower().replace("-", "_")
    normalized_raw = {
        "b1_win": "win", "b1win": "win", "b1_loss": "loss", "b1loss": "loss",
        "d": "draw", "noresult": "no_result", "nr": "no_result",
    }.get(normalized_raw, normalized_raw)
    actual_score = float(game.get("b1_game_score", float("nan")))
    if actual_score != expected_score:
        raise ValueError(
            f"Position {game.get('position_id')} game {game.get('pair_game_index')} has incorrect B1 score"
        )
    if bool(game.get("no_result")) != (normalized_raw == "no_result"):
        raise ValueError("B evaluation no_result flag disagrees with raw_outcome")
    if bool(game.get("draw")) != (normalized_raw == "draw"):
        raise ValueError("B evaluation draw flag disagrees with raw_outcome")
    expected_winner = {
        "win": "B1",
        "loss": "B0",
        "draw": "draw",
        "no_result": "NO_RESULT",
    }[normalized_raw]
    if str(game.get("winner")) != expected_winner:
        raise ValueError("B evaluation winner disagrees with raw_outcome")
    if str(game.get("b0_color")) not in {"black", "white"} or str(game.get("b1_color")) not in {"black", "white"}:
        raise ValueError("B evaluation colors must be black or white")
    if str(game.get("b0_color")) == str(game.get("b1_color")):
        raise ValueError("B0 and B1 cannot have the same color")
    required = (
        "training_seed", "sample_milestone", "position_id", "pair_game_index", "starting_player",
        "b0_color", "b1_color", "starting_semantic_state_fingerprint",
        "winner", "raw_outcome", "b1_game_score", "termination_reason",
        "no_result", "draw", "move_limit", "moves_played_from_start",
        "final_total_move_number", "elapsed_seconds",
    )
    missing = [key for key in required if key not in game]
    if missing:
        raise ValueError("B evaluation game is missing fields: " + ", ".join(missing))


def validate_seed_evaluation(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate one evaluator artifact before it enters the aggregate."""

    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported B seed evaluation schema")
    required = (
        "experiment_contract_id", "experiment_contract_sha256", "heldout_suite_id",
        "heldout_suite_sha256", "scientific_clock", "scientific_milestone",
        "rules_fingerprint", "topology", "size", "komi",
        "training_seed", "b0_checkpoint", "b1_checkpoint", "games",
        "position_results", "statistical_method_identifier",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("B seed evaluation is missing fields: " + ", ".join(missing))
    if payload.get("experiment_contract_id") != B_EXPERIMENT_CONTRACT_ID:
        raise ValueError("B seed evaluation has the wrong experiment contract")
    if payload.get("heldout_suite_id") != B_HELDOUT_SUITE_ID:
        raise ValueError("B seed evaluation has the wrong heldout suite")
    if payload.get("heldout_suite_sha256") != B_HELDOUT_SUITE_SHA256:
        raise ValueError("B seed evaluation heldout suite SHA mismatch")
    if payload.get("statistical_method_identifier") != B_STATISTICAL_METHOD_IDENTIFIER:
        raise ValueError("B seed evaluation has the wrong statistical method")
    if payload.get("topology") != "cube" or int(payload.get("size", -1)) != 4:
        raise ValueError("B seed evaluation must be Cube-4")
    if float(payload.get("komi", float("nan"))) != 0.5:
        raise ValueError("B seed evaluation komi must be exactly 0.5")
    if int(payload.get("position_count", -1)) != B_HELDOUT_SUITE_POSITION_COUNT:
        raise ValueError("B seed evaluation must contain 16 positions")
    if int(payload.get("games_per_position", -1)) != B_GAMES_PER_POSITION:
        raise ValueError("B seed evaluation must contain two games per position")
    seed = int(payload["training_seed"])
    if seed not in B_SEED_LIST:
        raise ValueError(f"B seed evaluation has unsupported training seed: {seed}")
    if int(payload["scientific_milestone"]) < 0:
        raise ValueError("B seed evaluation scientific milestone must be non-negative")
    games = payload.get("games")
    if not isinstance(games, list) or len(games) != B_HELDOUT_SUITE_POSITION_COUNT * B_GAMES_PER_POSITION:
        raise ValueError("B seed evaluation must contain exactly 32 games")
    for game in games:
        if not isinstance(game, Mapping):
            raise ValueError("B seed evaluation game must be an object")
        _validate_game_record(game)
        if int(game["training_seed"]) != seed:
            raise ValueError("B seed evaluation game has the wrong training seed")
        if int(game["sample_milestone"]) != int(payload["scientific_milestone"]):
            raise ValueError("B seed evaluation game has the wrong scientific milestone")
    grouped = validate_pairing_invariants(games, expected_position_ids=B_HELDOUT_POSITION_IDS)
    position_results = payload.get("position_results")
    if not isinstance(position_results, list) or len(position_results) != B_HELDOUT_SUITE_POSITION_COUNT:
        raise ValueError("B seed evaluation must contain 16 position results")
    if not all(isinstance(item, Mapping) for item in position_results):
        raise ValueError("B seed position result must be an object")
    if {str(item.get("position_id")) for item in position_results} != set(B_HELDOUT_POSITION_IDS):
        raise ValueError("B seed evaluation position result IDs are not canonical")
    canonical_positions = []
    for item in sorted(position_results, key=lambda value: B_HELDOUT_POSITION_IDS.index(str(value["position_id"]))):
        position_id = str(item["position_id"])
        expected = float(grouped[position_id]["pair_score_b1"])
        if float(item.get("pair_score_b1", float("nan"))) != expected:
            raise ValueError(f"B seed position {position_id} pair score mismatch")
        canonical_positions.append({
            "position_id": position_id,
            "pair_score_b1": expected,
        })
    expected_seed_score = float(np.mean([item["pair_score_b1"] for item in canonical_positions]))
    if float(payload.get("seed_score_b1", float("nan"))) != expected_seed_score:
        raise ValueError("B seed score does not equal the mean of its 16 pair scores")
    if float(payload.get("seed_delta", float("nan"))) != expected_seed_score - 0.5:
        raise ValueError("B seed delta does not equal seed score minus 0.5")
    for label in ("b0_checkpoint", "b1_checkpoint"):
        checkpoint = payload.get(label)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"B seed evaluation is missing {label} metadata")
        for key in ("iteration", "sha256", "milestone_target", "overshoot", "profile", "model_contract"):
            if key not in checkpoint:
                raise ValueError(f"B seed evaluation {label} is missing {key}")
    return {
        **dict(payload),
        "training_seed": seed,
        "games": [dict(game) for game in games],
        "position_results": canonical_positions,
    }


def _checkpoint_profile_identity(checkpoint: Mapping[str, object]) -> tuple[object, ...]:
    contract = checkpoint.get("model_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Checkpoint metadata must contain a model_contract")
    return (
        checkpoint.get("profile"),
        contract.get("contractId"),
        contract.get("contractVersion"),
        contract.get("gameClassId"),
        contract.get("observationSchema"),
        tuple(contract.get("observationShape", ())),
        contract.get("networkArchitectureId"),
        contract.get("rulesFingerprint"),
    )


def analyze_evaluations(
    evaluation_payloads: Sequence[Mapping[str, object]],
    *,
    extension_decision_payload: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return final JSON report and the separate extension decision payload."""

    if not evaluation_payloads:
        raise ValueError("At least one B seed evaluation is required")
    artifacts = [validate_seed_evaluation(payload) for payload in evaluation_payloads]
    seeds = sorted(int(item["training_seed"]) for item in artifacts)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate B training seed evaluation")
    if tuple(seeds) not in (tuple((0, 1, 2)), tuple((0, 1, 2, 3, 4))):
        raise ValueError("B aggregate requires exactly mandatory seeds 0,1,2 or all extension seeds 0..4")
    artifacts.sort(key=lambda item: int(item["training_seed"]))
    for key in (
        "experiment_contract_id", "experiment_contract_sha256", "heldout_suite_id",
        "heldout_suite_sha256", "scientific_clock", "scientific_milestone",
        "statistical_method_identifier", "rules_fingerprint", "topology", "size", "komi",
    ):
        _require_equal(artifacts, key)
    if any(_checkpoint_profile_identity(item["b0_checkpoint"]) != _checkpoint_profile_identity(artifacts[0]["b0_checkpoint"]) for item in artifacts):
        raise ValueError("B0 checkpoint profile identities differ across seed artifacts")
    if any(_checkpoint_profile_identity(item["b1_checkpoint"]) != _checkpoint_profile_identity(artifacts[0]["b1_checkpoint"]) for item in artifacts):
        raise ValueError("B1 checkpoint profile identities differ across seed artifacts")
    if _checkpoint_profile_identity(artifacts[0]["b0_checkpoint"])[0] != "baseline":
        raise ValueError("B0 checkpoint profile identity is not baseline")
    if _checkpoint_profile_identity(artifacts[0]["b1_checkpoint"])[0] != "g1":
        raise ValueError("B1 checkpoint profile identity is not g1")

    seed_scores = {
        int(item["training_seed"]): [float(position["pair_score_b1"]) for position in item["position_results"]]
        for item in artifacts
    }
    bootstrap = hierarchical_paired_bootstrap(
        seed_scores,
        replicates=B_BOOTSTRAP_REPLICATES,
        seed=B_BOOTSTRAP_SEED,
        confidence=0.95,
    )
    overall_score = float(np.mean([float(item["seed_score_b1"]) for item in artifacts]))
    overall_delta = overall_score - 0.5
    ci_low, ci_high = (float(value) for value in bootstrap["ci95_delta"])
    mandatory = {seed: seed_scores[seed] for seed in (0, 1, 2)}
    mandatory_bootstrap = hierarchical_paired_bootstrap(
        mandatory,
        replicates=B_BOOTSTRAP_REPLICATES,
        seed=B_BOOTSTRAP_SEED,
        confidence=0.95,
    )
    decision = extension_seed_decision(
        {int(item["training_seed"]): float(item["seed_delta"]) for item in artifacts if int(item["training_seed"]) in (0, 1, 2)},
        mandatory_bootstrap,
        experiment_contract_sha256=str(artifacts[0]["experiment_contract_sha256"]),
    )
    if extension_decision_payload is not None:
        if dict(extension_decision_payload) != decision:
            raise ValueError("Provided extension decision does not match the pre-registered criterion")
    if len(artifacts) == 5 and not bool(decision["approved"]):
        raise ValueError("Extension seeds were supplied although the registered criterion says stop at three")

    position_results = []
    for index, position_id in enumerate(B_HELDOUT_POSITION_IDS):
        scores = {str(item["training_seed"]): float(item["position_results"][index]["pair_score_b1"]) for item in artifacts}
        position_results.append({
            "position_id": position_id,
            "pair_scores_b1": scores,
            "mean_pair_score_b1": float(np.mean(list(scores.values()))),
        })
    all_games = [game for item in artifacts for game in item["games"]]
    diagnostics = summarize_game_diagnostics(all_games)
    seed_results = [
        {
            "training_seed": int(item["training_seed"]),
            "seed_score_b1": float(item["seed_score_b1"]),
            "seed_delta": float(item["seed_delta"]),
            "b0_checkpoint": item["b0_checkpoint"],
            "b1_checkpoint": item["b1_checkpoint"],
        }
        for item in artifacts
    ]
    report = {
        "schema_version": 1,
        "analysis_id": f"gocube-b-analysis-m{int(artifacts[0]['scientific_milestone'])}",
        "experiment_contract_id": artifacts[0]["experiment_contract_id"],
        "experiment_contract_version": B_EXPERIMENT_CONTRACT_VERSION,
        "experiment_contract_sha256": artifacts[0]["experiment_contract_sha256"],
        "heldout_suite_id": artifacts[0]["heldout_suite_id"],
        "heldout_suite_sha256": artifacts[0]["heldout_suite_sha256"],
        "heldout_suite_generator": {
            "generator_id": B_HELDOUT_GENERATOR_ID,
            "master_seed": B_HELDOUT_GENERATOR_MASTER_SEED,
            "depth_schedule": list(B_HELDOUT_DEPTH_SCHEDULE),
        },
        "scientific_clock": artifacts[0]["scientific_clock"],
        "scientific_milestone": int(artifacts[0]["scientific_milestone"]),
        "training_seeds": seeds,
        "seed_count": len(seeds),
        "training_seed_count": len(seeds),
        "position_count_per_seed": B_HELDOUT_SUITE_POSITION_COUNT,
        "positions_per_seed": B_HELDOUT_SUITE_POSITION_COUNT,
        "games_per_position": B_GAMES_PER_POSITION,
        "total_games": len(all_games),
        "result_semantics": {"win": 1.0, "draw": 0.5, "no_result": 0.5, "loss": 0.0},
        "statistical_method": {
            "identifier": B_STATISTICAL_METHOD_IDENTIFIER,
            "bootstrap": bootstrap,
        },
        "statistical_method_identifier": B_STATISTICAL_METHOD_IDENTIFIER,
        "bootstrap": bootstrap,
        "bootstrap_replicates": B_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": B_BOOTSTRAP_SEED,
        "overall_b1_paired_score": overall_score,
        "overall_delta_from_0_5": overall_delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "ci95": {"low": ci_low, "high": ci_high, "parameter": "overall_delta_from_0_5"},
        "classification": classify_delta_interval((ci_low, ci_high)),
        "seed_results": seed_results,
        "position_results": position_results,
        "diagnostics": diagnostics,
        "checkpoint_profile_identities": {
            "B0": list(_checkpoint_profile_identity(artifacts[0]["b0_checkpoint"])),
            "B1": list(_checkpoint_profile_identity(artifacts[0]["b1_checkpoint"])),
        },
        "extension_seed_decision": decision,
        "mandatory_analysis": {
            "seed_count": 3,
            "bootstrap": mandatory_bootstrap,
        },
    }
    return report, decision


def render_markdown_summary(report: Mapping[str, object]) -> str:
    ci = report["ci95"]
    lines = [
        f"# B0 vs B1 — {int(report['scientific_milestone']):,} {report['scientific_clock']}",
        "",
        f"Seeds: {','.join(str(seed) for seed in report['training_seeds'])}",
        f"Frozen positions: {report['position_count_per_seed']}/seed",
        f"Games: {report['total_games']}",
        "",
        f"B1 paired score: {float(report['overall_b1_paired_score']):.6f}",
        f"Delta vs 0.5: {float(report['overall_delta_from_0_5']):+.6f}",
        f"95% hierarchical bootstrap CI: [{float(ci['low']):+.6f}, {float(ci['high']):+.6f}]",
        "",
        f"Classification: {report['classification']}",
        "",
        "## Per seed",
        "",
    ]
    for item in report["seed_results"]:
        lines.append(
            f"- seed {item['training_seed']}: score={float(item['seed_score_b1']):.6f}, "
            f"delta={float(item['seed_delta']):+.6f}"
        )
    diagnostics = report["diagnostics"]
    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- B0 wins: {diagnostics['b0_wins']}",
        f"- B1 wins: {diagnostics['b1_wins']}",
        f"- draws: {diagnostics['draws']}",
        f"- NO_RESULT: {diagnostics['no_results']} ({float(diagnostics['no_result_rate']):.2%})",
        f"- move-limit: {diagnostics['move_limit_count']} ({float(diagnostics['move_limit_rate']):.2%})",
        f"- average game length: {float(diagnostics['average_game_length']):.2f}",
        f"- median game length: {float(diagnostics['median_game_length']):.2f}",
        f"- B1 black score: {diagnostics['b1_score_when_black']}",
        f"- B1 white score: {diagnostics['b1_score_when_white']}",
        "",
        f"Statistical method: {B_STATISTICAL_METHOD_IDENTIFIER}",
        f"Bootstrap: {B_BOOTSTRAP_REPLICATES} replicates, seed {B_BOOTSTRAP_SEED}",
        f"Extension criterion: {B_EXTENSION_CRITERION_ID}",
    ])
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", "--seed-evaluation", dest="evaluations", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--extension-decision-output", default=None)
    parser.add_argument("--extension-decision", default=None)
    args = parser.parse_args(argv)
    payloads = [_read_json(Path(path)) for path in args.evaluations]
    supplied_decision = _read_json(Path(args.extension_decision)) if args.extension_decision else None
    report, decision = analyze_evaluations(payloads, extension_decision_payload=supplied_decision)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    markdown = Path(args.markdown_output) if args.markdown_output else output.with_suffix(".md")
    markdown.write_text(render_markdown_summary(report), encoding="utf-8")
    if args.extension_decision_output:
        decision_path = Path(args.extension_decision_output)
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"B analysis: {report['classification']}; JSON: {output}; Markdown: {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
