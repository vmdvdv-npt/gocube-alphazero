"""Single source of truth and production authorization facade for Orchestrator V2.

``AZ_ORCHESTRATOR_VERSION=V2`` remains a diagnostic compatibility marker only.
It is deliberately insufficient to authorize Arena or training execution.
"""
from __future__ import annotations

import os

from .execution_permit import require_orchestrator_execution
from .version_constants import ORCHESTRATOR_ENTRYPOINT, ORCHESTRATOR_VERSION, ORCHESTRATOR_VERSION_ENV

LEGACY_V1_DISABLED_MESSAGE = (
    "Legacy Orchestrator V1 production entrypoint is disabled. "
    f"Use {ORCHESTRATOR_ENTRYPOINT} (Orchestrator {ORCHESTRATOR_VERSION})."
)


def mark_v2_process() -> None:
    """Set the legacy diagnostic marker without granting production authority."""
    os.environ[ORCHESTRATOR_VERSION_ENV] = ORCHESTRATOR_VERSION


def require_v2_process(entrypoint: str) -> None:
    """Reject production work invoked without a V2 authority/child permit."""
    require_orchestrator_execution(entrypoint)


def reject_legacy_v1_entrypoint(entrypoint: str) -> None:
    raise SystemExit(f"{entrypoint}: {LEGACY_V1_DISABLED_MESSAGE}")


__all__ = [
    "LEGACY_V1_DISABLED_MESSAGE",
    "ORCHESTRATOR_ENTRYPOINT",
    "ORCHESTRATOR_VERSION",
    "ORCHESTRATOR_VERSION_ENV",
    "mark_v2_process",
    "require_v2_process",
    "reject_legacy_v1_entrypoint",
]
