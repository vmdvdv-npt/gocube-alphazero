"""Simple sequential calibration-arm scenario.

This is intentionally separate from the M137 komi state machine.  It accepts
an injected one-arm action and preserves the caller's order; no M137 checks or
komi statistics are applied here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationArm:
    arm_id: str
    request: Mapping[str, object]


class CalibrationRunner:
    """Run a supplied list of arms once, in the supplied order."""

    def __init__(self, arms: Sequence[CalibrationArm], execute: Callable[[CalibrationArm, int], object]) -> None:
        if not arms:
            raise ValueError("calibration requires at least one arm")
        self.arms = tuple(arms)
        self.execute = execute

    def run(self) -> list[object]:
        return [self.execute(arm, index) for index, arm in enumerate(self.arms)]


__all__ = ["CalibrationArm", "CalibrationRunner"]
