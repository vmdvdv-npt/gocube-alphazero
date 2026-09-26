"""Standalone execution-profile tuning for Orchestrator V2."""

from .contracts import (  # noqa: F401
    Decision,
    DecisionKind,
    DecisionType,
    EXECUTION_ALIAS_NAMES,
    EXECUTION_OVERRIDE_ALLOWLIST,
    FailureCategory,
    MeasurementBudget,
    Mode,
    Observation,
    Plan,
    SelectedProfile,
    execution_overrides_for,
    scientific_contract_fingerprint,
    scientific_contract_payload,
    validate_execution_overrides,
)
from .policy import assess_observation, choose_next, classify_failure, select_mode, select_profile  # noqa: F401
from .report import build_report, write_report  # noqa: F401
from .runner import (  # noqa: F401
    GenerationExecutor,
    PerformanceTuningRunner,
    ProductionTrainOneExecutor,
    TuningExecutionError,
    TuningRunResult,
)
from .storage import (  # noqa: F401
    STATE_SCHEMA,
    TuningStateStore,
    selected_profile_path,
    tuning_report_path,
    tuning_root,
    tuning_state_path,
)
from .compatibility import CONCURRENCY_SWEEP_SCHEMA, LegacyConcurrencyAdapter, SelfPlayConcurrencySweep  # noqa: F401

__all__ = [
    "CONCURRENCY_SWEEP_SCHEMA", "Decision", "DecisionKind", "DecisionType", "EXECUTION_ALIAS_NAMES",
    "EXECUTION_OVERRIDE_ALLOWLIST", "FailureCategory", "GenerationExecutor",
    "LegacyConcurrencyAdapter", "MeasurementBudget", "Mode", "Observation", "SelfPlayConcurrencySweep",
    "PerformanceTuningRunner", "Plan", "ProductionTrainOneExecutor",
    "SelectedProfile", "STATE_SCHEMA", "TuningExecutionError", "TuningRunResult",
    "TuningStateStore", "assess_observation", "build_report", "choose_next",
    "classify_failure", "execution_overrides_for", "scientific_contract_fingerprint",
    "scientific_contract_payload", "selected_profile_path", "select_mode",
    "select_profile", "tuning_report_path", "tuning_root", "tuning_state_path",
    "validate_execution_overrides", "write_report",
]
