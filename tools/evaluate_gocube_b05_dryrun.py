#!/usr/bin/env python3
"""Run the B4 paired evaluator in the explicitly non-scientific B05 mode.

The implementation delegates all game, loader, pairing, and search semantics
to ``evaluate_gocube_b_experiment``.  The only additional authority is the
dry-run marker, which is required here and rejected by the scientific B4
analyzer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphazero.envs.gocube.b_experiment_contract import load_b_experiment_contract
from tools.evaluate_gocube_b_experiment import evaluate_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-checkpoint", required=True)
    parser.add_argument("--b1-checkpoint", required=True)
    parser.add_argument("--suite", "--heldout-suite", dest="suite", required=True)
    parser.add_argument("--experiment-contract", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--sample-milestone", required=True, type=int)
    parser.add_argument("--sample-clock", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--move-limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    contract, _ = load_b_experiment_contract(args.experiment_contract)
    if not contract.non_scientific_dry_run:
        parser.error("--experiment-contract is not a B05 non-scientific dry-run contract")
    payload = evaluate_seed(
        b0_checkpoint=Path(args.b0_checkpoint),
        b1_checkpoint=Path(args.b1_checkpoint),
        suite_path=Path(args.suite),
        training_seed=args.training_seed,
        sample_milestone=args.sample_milestone,
        sample_clock=args.sample_clock,
        device=args.device,
        move_limit=args.move_limit,
        experiment_contract=contract,
        allow_non_scientific_dry_run=True,
    )
    payload["non_scientific_dry_run"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(
        f"B05 evaluation seed {args.training_seed}: integration-only; "
        f"paired_score={payload['seed_score_b1']:.6f}; JSON: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
