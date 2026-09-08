#!/usr/bin/env python3
"""Single public entrypoint for the hardened Cube-4 P1..P5 production sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.gocube_experiment_runner import *  # noqa: F401,F403
from tools import gocube_experiment_runner as _runner
from tools.gocube_experiment_storage import StorageEfficientExperiment
from tools import gocube_production_preflight as _preflight


_RESUME_OPTION_TO_ATTR = {
    "--bootstrap-run": "bootstrap_run",
    "--device": "device",
    "--arena-batch-wait-ms": "arena_batch_wait_ms",
    "--heldout-positions": "heldout_positions",
    "--health-gate-min-win-rate": "health_gate_min_win_rate",
    "--benchmark-games": "benchmark_games",
    "--skip-performance-benchmark": "skip_performance_benchmark",
    "--seed": "seed",
    "--candidate-new-samples-budget": "candidate_new_samples_budget",
    "--candidate-optimizer-examples-budget": "candidate_optimizer_examples_budget",
}


def _option_present(raw: list[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in raw)


def _restore_saved_resume_arguments(cli, raw: list[str]):
    state_path = Path.cwd() / "training_reports" / cli.experiment_id / "experiment-state.json"
    if not state_path.is_file():
        return cli
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    saved = payload.get("launch_config")
    if not isinstance(saved, dict):
        return cli
    for option, attribute in _RESUME_OPTION_TO_ATTR.items():
        if _option_present(raw, option):
            continue
        if attribute in saved:
            setattr(cli, attribute, saved[attribute])
    if (
        cli.candidate_optimizer_examples_budget is not None
        and not any(
            _option_present(raw, option)
            for option in ("--candidate-new-samples-budget", "--candidate-sample-budget")
        )
    ):
        cli.candidate_new_samples_budget = None
    return cli


def parse_args(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    cli = _runner.parse_args(raw)
    return _restore_saved_resume_arguments(cli, raw)


Experiment = StorageEfficientExperiment


def main(argv=None) -> int:
    cli = parse_args(argv)
    repo = Path.cwd().resolve()
    if not _preflight.is_supervised_invocation():
        return _preflight.launch_under_systemd(repo, cli)
    experiment = Experiment(cli)
    _preflight.apply_production_preflight(experiment)
    experiment.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
