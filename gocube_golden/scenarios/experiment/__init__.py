"""Experiment policy and coordinator facade."""

from .policy import EXPERIMENT_WINNER_RULE, WinnerDecision, WinnerRule, WinnerRuleName


def __getattr__(name: str):
    if name in {
        "ExperimentArmConfig",
        "ExperimentConfig",
        "ExperimentRunResult",
        "ExperimentRunner",
        "ExperimentRunnerError",
        "ExperimentRunnerV2",
        "ExperimentStage2Config",
        "Stage2Config",
        "WinnerDecision",
        "WinnerRule",
        "WinnerRuleName",
    }:
        from .runner import (
            ExperimentArmConfig,
            ExperimentConfig,
            ExperimentRunResult,
            ExperimentRunner,
            ExperimentRunnerError,
            ExperimentRunnerV2,
            ExperimentStage2Config,
            Stage2Config,
            WinnerDecision as RunnerWinnerDecision,
            WinnerRule as RunnerWinnerRule,
            WinnerRuleName as RunnerWinnerRuleName,
        )

        values = {
            "ExperimentArmConfig": ExperimentArmConfig,
            "ExperimentConfig": ExperimentConfig,
            "ExperimentRunResult": ExperimentRunResult,
            "ExperimentRunner": ExperimentRunner,
            "ExperimentRunnerError": ExperimentRunnerError,
            "ExperimentRunnerV2": ExperimentRunnerV2,
            "ExperimentStage2Config": ExperimentStage2Config,
            "Stage2Config": Stage2Config,
            "WinnerDecision": RunnerWinnerDecision,
            "WinnerRule": RunnerWinnerRule,
            "WinnerRuleName": RunnerWinnerRuleName,
        }
        return values[name]
    raise AttributeError(name)

__all__ = [
    "EXPERIMENT_WINNER_RULE",
    "ExperimentArmConfig",
    "ExperimentConfig",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentRunnerError",
    "ExperimentRunnerV2",
    "ExperimentStage2Config",
    "Stage2Config",
    "WinnerDecision",
    "WinnerRule",
    "WinnerRuleName",
]
