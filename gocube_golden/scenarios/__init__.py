"""Scenario policies and durable scenario coordinators.

The package contains domain sequencing and scientific decisions.  It does not
own process supervision, checkpoint discovery, or notification delivery.
"""

from .contracts import (
    ActionOutcome,
    ActionRequest,
    ExecutionStatus,
    ScientificValidity,
)

__all__ = [
    "ActionOutcome",
    "ActionRequest",
    "ExecutionStatus",
    "ScientificValidity",
]
