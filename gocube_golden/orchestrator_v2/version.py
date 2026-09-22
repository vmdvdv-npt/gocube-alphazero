"""Single source of truth for the production orchestrator version.

The legacy V1 modules remain importable for bounded compatibility tests and
historical artifact tooling, but their executable production entrypoints are
disabled.  New production work must enter through the V2 production boundary.
"""

from __future__ import annotations

import os


ORCHESTRATOR_VERSION = "V2"
ORCHESTRATOR_ENTRYPOINT = "gocube_golden.orchestrator_v2.production_entrypoint"
ORCHESTRATOR_VERSION_ENV = "AZ_ORCHESTRATOR_VERSION"

LEGACY_V1_DISABLED_MESSAGE = (
    "Legacy Orchestrator V1 production entrypoint is disabled. "
    f"Use {ORCHESTRATOR_ENTRYPOINT} (Orchestrator {ORCHESTRATOR_VERSION})."
)


def mark_v2_process() -> None:
    """Mark the current production process as an Orchestrator V2 process."""

    os.environ[ORCHESTRATOR_VERSION_ENV] = ORCHESTRATOR_VERSION


def reject_legacy_v1_entrypoint(entrypoint: str) -> None:
    """Fail closed when an executable legacy V1 entrypoint is invoked."""

    raise SystemExit(f"{entrypoint}: {LEGACY_V1_DISABLED_MESSAGE}")


__all__ = [
    "LEGACY_V1_DISABLED_MESSAGE",
    "ORCHESTRATOR_ENTRYPOINT",
    "ORCHESTRATOR_VERSION",
    "ORCHESTRATOR_VERSION_ENV",
    "mark_v2_process",
    "reject_legacy_v1_entrypoint",
]
