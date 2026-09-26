"""Backward-compatible facade for the scenario experiment runner.

The implementation lives in :mod:`gocube_golden.scenarios.experiment.runner`.
This module intentionally preserves the historical import path used by CLI
callers, tests, and saved run specifications.
"""

from ..scenarios.experiment.runner import (
    EXPERIMENT_RUNNER_SCHEMA,
    EXPERIMENT_STATE_SCHEMA,
    EXPERIMENT_WINNER_RULE,
    ExperimentArmConfig,
    ExperimentConfig,
    ExperimentRunResult,
    ExperimentRunner,
    ExperimentRunnerError,
    ExperimentRunnerV2,
    ExperimentStage2Config,
    LineageFactory,
    Stage2Config,
    TrainOne,
    WinnerDecision,
    WinnerRule,
    WinnerRuleName,
    _read_json,
    _write_json,
)

__all__ = [
    "EXPERIMENT_RUNNER_SCHEMA",
    "EXPERIMENT_STATE_SCHEMA",
    "EXPERIMENT_WINNER_RULE",
    "ExperimentArmConfig",
    "ExperimentConfig",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentRunnerError",
    "ExperimentRunnerV2",
    "ExperimentStage2Config",
    "LineageFactory",
    "Stage2Config",
    "TrainOne",
    "WinnerDecision",
    "WinnerRule",
    "WinnerRuleName",
]
