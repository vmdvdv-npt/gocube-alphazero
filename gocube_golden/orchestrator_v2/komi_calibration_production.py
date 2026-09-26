"""Backward-compatible facade for production komi calibration."""

from ..scenarios.komi.policy import aggregate_candidate_batches as _aggregate_candidate_batches
from ..scenarios.komi.policy import wilson_interval as _wilson_interval
from ..scenarios.komi.production_runner import ProductionKomiCalibrationRunnerV2
from .komi_calibration import KomiCalibrationError


def aggregate_candidate_batches(*args: object, **kwargs: object) -> dict[str, object]:
    """Preserve the old public helper and its historical exception type."""
    try:
        return _aggregate_candidate_batches(*args, **kwargs)  # type: ignore[arg-type]
    except ValueError as exc:
        raise KomiCalibrationError(str(exc)) from exc


__all__ = ["ProductionKomiCalibrationRunnerV2", "aggregate_candidate_batches"]
