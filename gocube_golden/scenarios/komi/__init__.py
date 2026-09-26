"""Komi calibration policy and durable coordinator adapters."""

from .policy import (
    KomiDecision,
    aggregate_candidate_batches,
    select_komi,
    wilson_interval,
)


def __getattr__(name: str):
    if name in {
        "KomiCalibrationConfig",
        "KomiCalibrationError",
        "KomiCalibrationResult",
        "KomiCalibrationRunnerV2",
        "ProductionKomiCalibrationRunnerV2",
    }:
        from .runner import (
            KomiCalibrationConfig,
            KomiCalibrationError,
            KomiCalibrationResult,
            KomiCalibrationRunnerV2,
            ProductionKomiCalibrationRunnerV2,
        )
        return {
            "KomiCalibrationConfig": KomiCalibrationConfig,
            "KomiCalibrationError": KomiCalibrationError,
            "KomiCalibrationResult": KomiCalibrationResult,
            "KomiCalibrationRunnerV2": KomiCalibrationRunnerV2,
            "ProductionKomiCalibrationRunnerV2": ProductionKomiCalibrationRunnerV2,
        }[name]
    raise AttributeError(name)

__all__ = [
    "KomiDecision",
    "KomiCalibrationConfig",
    "KomiCalibrationError",
    "KomiCalibrationResult",
    "KomiCalibrationRunnerV2",
    "ProductionKomiCalibrationRunnerV2",
    "aggregate_candidate_batches",
    "select_komi",
    "wilson_interval",
]
