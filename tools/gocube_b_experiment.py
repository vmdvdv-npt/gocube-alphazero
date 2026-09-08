#!/usr/bin/env python3
"""Hardened B0/B1 experiment entrypoint.

The operator selects only the treatment.  The launcher resolves the profile,
pins the production budgets, runs the separate-process preflight, and records
the complete machine-readable experiment contract before training starts.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Keep the standalone hardened entrypoint usable from the repository root
    # without requiring the operator to export PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphazero.envs.gocube.b_experiment_contract import (
    B0_MODEL_PROFILE,
    B0_TREATMENT,
    B1_MODEL_PROFILE,
    B1_TREATMENT,
    B_EXTENSION_SEEDS,
    B_SEED_LIST,
    DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET,
    preflight_b_experiment,
    validate_extension_seed_decision,
)
from alphazero.envs.gocube.contract_versions import B_EXPERIMENT_CONTRACT_ID
from alphazero.envs.gocube.production_contract import CUBE4_PRODUCTION
from alphazero.envs.gocube.production_training import (
    SampleBudgetTarget,
    build_sample_budget_target,
)


TREATMENT_TO_PROFILE = {
    B0_TREATMENT: B0_MODEL_PROFILE,
    B1_TREATMENT: B1_MODEL_PROFILE,
}


def _treatment(value: str) -> str:
    normalized = str(value).upper()
    if normalized not in TREATMENT_TO_PROFILE:
        raise argparse.ArgumentTypeError("treatment must be B0 or B1")
    return normalized


def _validate_extension_decision_target(path: str, scientific_target: SampleBudgetTarget) -> None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read extension-seed decision: {path}") from exc
    milestone = payload.get("scientific_milestone") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("scientific_clock") != scientific_target.kind
        or isinstance(milestone, bool)
        or not isinstance(milestone, int)
        or milestone != scientific_target.target
    ):
        raise ValueError(
            "extension-seed decision must target the selected B contract final milestone: "
            f"{scientific_target.kind}={scientific_target.target}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one hardened GoCube B experiment treatment")
    parser.add_argument("--treatment", type=_treatment, required=True, help="B0 or B1")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=4096,
        help="maximum iteration safety ceiling; sample target controls stopping",
    )
    budget_targets = parser.add_mutually_exclusive_group()
    budget_targets.add_argument(
        "--cumulative-new-samples-target",
        "--new-samples-target",
        "--sample-target",
        dest="cumulative_new_samples_target",
        type=int,
        default=None,
        help="stop at this cumulative new-sample milestone",
    )
    budget_targets.add_argument(
        "--cumulative-optimizer-examples-target",
        "--optimizer-examples-target",
        dest="cumulative_optimizer_examples_target",
        type=int,
        default=None,
        help="stop at this cumulative optimizer-examples milestone",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contract-path", default=None)
    parser.add_argument("--heldout-suite", default=None)
    parser.add_argument(
        "--extension-seed-decision",
        default=None,
        help="JSON approval required when running extension seed 3 or 4",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.seed not in B_SEED_LIST:
        parser.error(
            "B experiment seed must be one of the contract seed_list values: "
            + ", ".join(str(seed) for seed in B_SEED_LIST)
        )
    if args.seed in B_EXTENSION_SEEDS:
        if args.extension_seed_decision is None:
            parser.error(
                "seeds 3 and 4 require --extension-seed-decision approved by "
                "the pre-registered ambiguity/variance criterion"
            )
    elif args.extension_seed_decision is not None:
        parser.error("--extension-seed-decision is only valid for extension seeds 3 and 4")
    if args.run_name is None:
        args.run_name = f"gocube-b-{args.treatment.lower()}"
    if args.cumulative_new_samples_target is None and args.cumulative_optimizer_examples_target is None:
        args.cumulative_new_samples_target = DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET
    try:
        args.scientific_target = build_sample_budget_target(
            cumulative_new_samples_target=args.cumulative_new_samples_target,
            cumulative_optimizer_examples_target=args.cumulative_optimizer_examples_target,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.seed in B_EXTENSION_SEEDS:
        try:
            _validate_extension_decision_target(
                args.extension_seed_decision,
                args.scientific_target,
            )
        except ValueError as exc:
            parser.error(str(exc))
    return args


def training_command(
    args,
    *,
    python: str | None = None,
    contract_sha256: str | None = None,
) -> list[str]:
    """Return the fully pinned child command for the selected treatment."""

    interpreter = python or str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
    command = [
        interpreter,
        "-m",
        "alphazero.envs.gocube.hardened_train",
        "--model-profile",
        TREATMENT_TO_PROFILE[args.treatment],
        "--topology",
        "cube",
        "--size",
        "4",
        "--workers",
        str(CUBE4_PRODUCTION.workers),
        "--sims",
        str(CUBE4_PRODUCTION.regular_sims),
        "--arena-sims",
        str(CUBE4_PRODUCTION.arena_sims),
        "--games-per-iteration",
        str(CUBE4_PRODUCTION.games_per_iteration),
        "--iterations",
        str(args.iterations),
        "--train-batch-size",
        str(CUBE4_PRODUCTION.train_batch_size),
        "--fast-game-prob",
        str(CUBE4_PRODUCTION.fast_probability),
        "--train-samples-per-new-sample",
        str(CUBE4_PRODUCTION.train_samples_per_new_sample),
        "--endgame-sample-weight",
        "1",
        "--seed",
        str(args.seed),
        "--run-name",
        str(args.run_name),
        "--no-arena",
    ]
    scientific_target = getattr(args, "scientific_target", None)
    if scientific_target is None:
        scientific_target = build_sample_budget_target(
            cumulative_new_samples_target=getattr(args, "cumulative_new_samples_target", None),
            cumulative_optimizer_examples_target=getattr(
                args, "cumulative_optimizer_examples_target", None
            ),
        )
    if scientific_target is None:
        raise ValueError("B experiment requires a cumulative sample budget target")
    command.extend(
        [
            (
                "--cumulative-new-samples-target"
                if scientific_target.kind == SampleBudgetTarget.NEW_SAMPLES
                else "--cumulative-optimizer-examples-target"
            ),
            str(scientific_target.target),
        ]
    )
    if contract_sha256 is not None:
        command.extend(
            [
                "--experiment-contract-id",
                B_EXPERIMENT_CONTRACT_ID,
                "--experiment-contract-sha256",
                str(contract_sha256),
            ]
        )
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.heldout_suite is None and not args.dry_run:
        raise SystemExit(
            "--heldout-suite is required for a real B experiment; "
            "without it only --dry-run is permitted"
        )
    repo = Path.cwd().resolve()
    contract_path = (
        Path(args.contract_path).resolve()
        if args.contract_path
        else repo / "training_reports" / args.run_name / "gocube-b-experiment-contract.json"
    )
    preflight_payload = None
    if args.heldout_suite is not None:
        preflight_payload = preflight_b_experiment(
            repo=repo,
            contract_path=contract_path,
            heldout_suite_path=args.heldout_suite,
            scientific_target=args.scientific_target,
        )
        if args.seed in B_EXTENSION_SEEDS:
            validate_extension_seed_decision(
                args.extension_seed_decision,
                contract_sha256=str(preflight_payload["experiment_contract_sha256"]),
                experiment_contract=preflight_payload["experiment_contract"],
            )
    else:
        print("Dry-run only: no frozen heldout suite was supplied; no contract record was written.")
    contract_sha256 = (
        None
        if preflight_payload is None
        else str(preflight_payload["experiment_contract_sha256"])
    )
    command = training_command(
        args,
        python=str(repo / ".venv" / "bin" / "python"),
        contract_sha256=contract_sha256,
    )
    print(f"B experiment contract: {contract_path}")
    print("Launching: " + shlex.join(command))
    if args.dry_run:
        return 0
    completed = subprocess.run(command, cwd=repo, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
