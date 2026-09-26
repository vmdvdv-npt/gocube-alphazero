"""Execution adapter and durable standalone tuning runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import fcntl
import inspect
from pathlib import Path
from typing import Any, Protocol

from ..artifact_graph import CheckpointRef
from ..notifications import operator_event
from .contracts import (
    Decision,
    DecisionType,
    FailureCategory,
    Mode,
    Observation,
    Plan,
    SelectedProfile,
    execution_overrides_for,
    scientific_contract_fingerprint,
)
from .policy import assess_observation, choose_next, classify_failure, select_profile
from .report import write_report
from .storage import TuningStateStore, selected_profile_path, tuning_report_path, tuning_root


class TuningExecutionError(RuntimeError):
    """A tuning action did not reach a committed generation."""


class GenerationExecutor(Protocol):
    def __call__(
        self,
        *,
        parent_checkpoint: Mapping[str, object],
        execution_profile: Mode,
        action_id: str,
    ) -> object: ...


@dataclass(frozen=True)
class TuningRunResult:
    state: str
    tuning_id: str
    selected_profile: SelectedProfile | None
    current_checkpoint: Mapping[str, object]
    actions_consumed: int
    dry_run: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "tuning_id": self.tuning_id,
            "selected_profile": None if self.selected_profile is None else self.selected_profile.to_dict(),
            "current_checkpoint": dict(self.current_checkpoint),
            "actions_consumed": self.actions_consumed,
            "dry_run": self.dry_run,
        }


class _OwnerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"performance tuning owner is already active: {self.path}") from exc
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _checkpoint_payload(value: object) -> dict[str, object]:
    if isinstance(value, CheckpointRef):
        return value.to_dict()
    if isinstance(value, Mapping):
        nested = value.get("checkpoint")
        if isinstance(nested, Mapping):
            return dict(nested)
        return dict(value)
    ref = getattr(value, "ref", None)
    if isinstance(ref, CheckpointRef):
        return ref.to_dict()
    if isinstance(ref, Mapping):
        return dict(ref)
    raise TuningExecutionError("generation executor returned no committed checkpoint reference")


def _generation(value: object) -> int:
    payload = _checkpoint_payload(value)
    raw = payload.get("generation")
    if type(raw) is not int:
        raise TuningExecutionError("generation executor returned a malformed checkpoint generation")
    return raw


def _metrics(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        for key in ("metrics", "raw_metrics", "orchestrator_selfplay"):
            candidate = value.get(key)
            if isinstance(candidate, Mapping):
                return candidate if key != "metrics" else candidate
    for key in ("metrics", "raw_metrics"):
        candidate = getattr(value, key, None)
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _evidence(value: object) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    owner_root = getattr(value, "owner_root", None)
    ref = getattr(value, "ref", None)
    if owner_root is not None and ref is not None:
        result.append({"ref": str(Path(owner_root) / f"generation-{getattr(ref, 'generation', 'unknown'):02d}.complete.json"), "kind": "generation-commit"})
        if hasattr(ref, "path"):
            result.append({"ref": str(Path(owner_root) / ref.path), "kind": "checkpoint", "sha256": getattr(ref, "sha256", None)})
    return tuple(item for item in result if item.get("sha256") is not None or item.get("kind") == "generation-commit")


class ProductionTrainOneExecutor:
    """Thin adapter from the tuning boundary to the existing train_one."""

    def __init__(self, *, train_one: object, config: object, output_lineage: object, workers: int, resolver: object | None = None) -> None:
        self.train_one = train_one
        self.config = config
        self.output_lineage = output_lineage
        self.workers = workers
        self.resolver = resolver

    def __call__(self, *, parent_checkpoint: Mapping[str, object], execution_profile: Mode, action_id: str) -> object:
        overrides = execution_overrides_for(execution_profile, workers=self.workers, scientific_config=self.config)
        parent = parent_checkpoint
        if self.resolver is not None and hasattr(self.resolver, "checkpoint"):
            parent = self.resolver.checkpoint(parent_checkpoint)
        kwargs = {
            "parent": parent,
            "config": self.config,
            "output_lineage": self.output_lineage,
            "execution_overrides": overrides,
            "action_id": action_id,
            "scientific_contract_fingerprint": scientific_contract_fingerprint(self.config),
        }
        callable_train = self.train_one
        try:
            signature = inspect.signature(callable_train)
            accepted = set(signature.parameters)
            kwargs = {key: value for key, value in kwargs.items() if key in accepted}
        except (TypeError, ValueError):
            pass
        return callable_train(**kwargs)  # type: ignore[operator]


class PerformanceTuningRunner:
    """Run exactly the actions represented by a :class:`Plan`."""

    def __init__(
        self,
        plan: Plan,
        *,
        execute_generation: GenerationExecutor | None = None,
        executor: GenerationExecutor | None = None,
        storage: TuningStateStore | None = None,
        owner_root: str | Path | None = None,
        metrics_reader: Callable[[object, int], Mapping[str, object]] | None = None,
        event_sink: object | None = None,
        report_path: str | Path | None = None,
    ) -> None:
        self.plan = plan
        self.execute_generation = execute_generation or executor
        if self.execute_generation is None:
            raise TypeError("PerformanceTuningRunner requires execute_generation")
        resolved_owner = owner_root or plan.owner_root
        self.owner_root = None if resolved_owner is None else Path(resolved_owner).resolve()
        self.storage = storage or (
            TuningStateStore.for_owner(self.owner_root, plan) if self.owner_root is not None else None
        )
        self.metrics_reader = metrics_reader
        self.event_sink = event_sink
        self.report_path = Path(report_path).resolve() if report_path is not None else (
            tuning_report_path(self.owner_root, plan.tuning_id) if self.owner_root is not None else None
        )

    def dry_run(self) -> TuningRunResult:
        actions = self.plan.action_profiles()
        return TuningRunResult(
            state="DRY_RUN",
            tuning_id=self.plan.tuning_id,
            selected_profile=None,
            current_checkpoint=self.plan.parent_checkpoint.to_dict(),  # type: ignore[union-attr]
            actions_consumed=0,
            dry_run=True,
        )

    def _emit(self, event_type: str, action_id: str, payload: Mapping[str, object]) -> None:
        sink = self.event_sink
        if sink is None or not hasattr(sink, "publish"):
            return
        try:
            sink.publish(
                operator_event(
                    event_type,
                    topology=self.plan.topology,
                    owner_type="lineage",
                    owner_id=self.plan.owner_id or self.plan.tuning_id,
                    action_id=action_id,
                    payload=dict(payload),
                    correlation_id=self.plan.tuning_id,
                    execution_code_commit=self.plan.execution_code_commit,
                )
            )
        except Exception:
            # Delivery is observability and is deliberately fail-open.
            return

    def _save_report(self, state: Mapping[str, object]) -> None:
        if self.report_path is not None:
            write_report(self.report_path, self.plan, state)

    def _save_profile(self, profile: SelectedProfile) -> None:
        if self.owner_root is None:
            return
        path = selected_profile_path(self.owner_root, self.plan.tuning_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        from ..process_supervision import atomic_write_text
        from ..provenance import canonical_json

        atomic_write_text(path, canonical_json(profile.to_dict()) + "\n")

    def _state(self) -> tuple[dict[str, object], TuningStateStore]:
        if self.storage is None:
            raise RuntimeError("standalone tuning requires an owner root or explicit storage")
        state = self.storage.load(self.plan) if self.storage.exists() else self.storage.create(self.plan)
        return state, self.storage

    def _invoke(self, *, parent: Mapping[str, object], mode: Mode, action_id: str) -> object:
        return self.execute_generation(  # type: ignore[misc]
            parent_checkpoint=dict(parent), execution_profile=mode, action_id=action_id
        )

    def _append_failure(self, state: dict[str, object], *, action_id: str, mode: Mode, error: BaseException, category: FailureCategory, generation: int) -> None:
        failures = state.setdefault("failed_modes", [])
        if not isinstance(failures, list):
            raise RuntimeError("tuning failed_modes is malformed")
        failures.append({
            "label": mode.label,
            "action_id": action_id,
            "generation": generation,
            "category": category.value,
            "exception": error.__class__.__name__,
            "reason": str(error)[:1000],
            "recorded_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        })

    def _append_observation(self, state: dict[str, object], observation: Observation) -> None:
        observations = state.setdefault("observations", [])
        if not isinstance(observations, list):
            raise RuntimeError("tuning observations are malformed")
        observations[:] = [item for item in observations if not isinstance(item, Mapping) or item.get("action_id") != observation.action_id]
        observations.append(observation.to_dict())

    @staticmethod
    def _mark_action(state: dict[str, object], action_id: str, *, status: str, **extra: object) -> None:
        actions = state.setdefault("planned_actions", [])
        if not isinstance(actions, list):
            raise RuntimeError("tuning planned_actions are malformed")
        for item in actions:
            if isinstance(item, Mapping) and item.get("action_id") == action_id:
                item.update({"status": status, **extra})
                return
        actions.append({"action_id": action_id, "status": status, **extra})

    def run(self, *, dry_run: bool = False) -> TuningRunResult:
        if dry_run:
            return self.dry_run()
        state, storage = self._state()
        if state.get("status") == "FAILED":
            raise TuningExecutionError(str(state.get("stop_reason") or "tuning state is terminally failed"))
        # The lock belongs to the canonical owner, not to a tuning id.  Two
        # commands targeting one lineage must not launch two generations.
        lock = _OwnerLock(self.owner_root / "runtime" / "tuning" / ".owner.lock") if self.owner_root is not None else None
        context = lock if lock is not None else _NullContext()
        with context:
            self._emit("TUNING_STARTED", f"{self.plan.tuning_id}:start", {"plan_fingerprint": self.plan.plan_fingerprint})
            while True:
                selected_raw = state.get("selected_profile")
                if selected_raw is not None:
                    selected = SelectedProfile.from_dict(selected_raw)  # type: ignore[arg-type]
                    state["status"] = "COMPLETED"
                    storage.save(state)
                    self._save_profile(selected)
                    self._save_report(state)
                    return TuningRunResult("COMPLETED", self.plan.tuning_id, selected, dict(state["current_checkpoint"]), int(state.get("actions_consumed", 0)), False)  # type: ignore[arg-type]

                decision = choose_next(self.plan, state)
                state["next_decision"] = decision.to_dict()
                storage.save(state)
                if decision.kind is DecisionType.FINISH:
                    observations = state.get("observations", [])
                    failures = state.get("failed_modes", [])
                    if not isinstance(observations, list) or not isinstance(failures, list):
                        raise RuntimeError("tuning evidence is malformed")
                    try:
                        selected = select_profile(self.plan, observations, failures)
                    except Exception as exc:
                        state["status"] = "FAILED"
                        state["stop_reason"] = str(exc)[:1000]
                        storage.save(state)
                        self._save_report(state)
                        self._emit("TUNING_FAILED", f"{self.plan.tuning_id}:failed", {"reason": state["stop_reason"]})
                        raise
                    state["selected_profile"] = selected.to_dict()
                    state["status"] = "COMPLETED"
                    state["stop_reason"] = None
                    storage.save(state)
                    self._save_profile(selected)
                    self._save_report(state)
                    self._emit("TUNING_SELECTED", f"{self.plan.tuning_id}:selected", {"profile_id": selected.profile_id, "source": selected.source})
                    return TuningRunResult("COMPLETED", self.plan.tuning_id, selected, dict(state["current_checkpoint"]), int(state.get("actions_consumed", 0)), False)  # type: ignore[arg-type]
                if decision.kind is DecisionType.STOP_WITH_ERROR:
                    state["status"] = "FAILED"
                    state["stop_reason"] = decision.reason
                    storage.save(state)
                    self._save_report(state)
                    self._emit("TUNING_FAILED", f"{self.plan.tuning_id}:failed", {"reason": decision.reason})
                    raise TuningExecutionError(decision.reason)

                if decision.mode is None or decision.action_id is None:
                    raise TuningExecutionError("tuning decision did not include an executable action")
                is_fallback = decision.kind is DecisionType.RETRY_BASELINE
                parent = state.get("current_checkpoint")
                if not isinstance(parent, Mapping):
                    raise TuningExecutionError("tuning state current checkpoint is malformed")
                action_id = decision.action_id
                index_before = int(state.get("next_index", 0))
                actions_consumed = int(state.get("actions_consumed", 0))
                state["active_action_id"] = action_id
                state["active_profile"] = decision.mode.to_dict()
                state["active_phase"] = "generation"
                self._mark_action(state, action_id, status="RUNNING", execution_profile=decision.mode.execution_overrides)
                storage.save(state)
                try:
                    committed = self._invoke(parent=parent, mode=decision.mode, action_id=action_id)
                    child = _checkpoint_payload(committed)
                    if child.get("topology") != self.plan.topology:
                        raise TuningExecutionError("committed generation has the wrong topology identity")
                    expected_generation = int(parent.get("generation", -1)) + 1
                    if child.get("generation") != expected_generation:
                        raise TuningExecutionError("committed generation is not the immediate child of the recorded checkpoint")
                    raw_metrics = self.metrics_reader(committed, expected_generation) if self.metrics_reader is not None else _metrics(committed)
                    contract = {
                        "mode": decision.mode,
                        "generation": expected_generation,
                        "action_id": action_id,
                        "role": "fallback" if is_fallback else ("baseline" if decision.mode.label == self.plan.baseline.label else "measurement"),
                        "workers": self.plan.workers,
                        "expected_games": self.plan.measurement_contract.get("expected_games"),
                    }
                    observation = assess_observation(raw_metrics, contract)
                    observation = Observation(
                        action_id=observation.action_id,
                        mode=observation.mode,
                        generation=observation.generation,
                        metrics=observation.metrics,
                        stable=observation.stable,
                        stability_reasons=observation.stability_reasons,
                        checkpoint_ref=child,
                        evidence_refs=_evidence(committed),
                        role=observation.role,
                        raw_metrics=observation.raw_metrics,
                    )
                    self._append_observation(state, observation)
                    self._mark_action(state, action_id, status="COMMITTED", checkpoint_ref=child)
                    state["current_checkpoint"] = child
                    state["actions_consumed"] = actions_consumed + 1
                    if is_fallback:
                        intent = state.get("fallback_intent")
                        state["next_index"] = int(intent.get("resume_index", index_before + 1)) if isinstance(intent, Mapping) else index_before + 1
                        state["fallback_intent"] = None
                    else:
                        state["next_index"] = index_before + 1
                    state["active_action_id"] = None
                    state["active_profile"] = None
                    state["active_phase"] = None
                    storage.save(state)
                    self._save_report(state)
                except Exception as error:
                    category = classify_failure(error)
                    state["actions_consumed"] = actions_consumed + 1
                    self._append_failure(state, action_id=action_id, mode=decision.mode, error=error, category=category, generation=int(parent.get("generation", -1)) + 1)
                    self._mark_action(state, action_id, status="FAILED", failure_category=category.value)
                    if is_fallback or decision.mode.label == self.plan.baseline.label:
                        state["status"] = "FAILED"
                        state["stop_reason"] = f"{category.value}: {str(error)[:1000]}"
                        state["active_action_id"] = None
                        state["active_profile"] = None
                        state["active_phase"] = None
                        storage.save(state)
                        self._save_report(state)
                        self._emit("TUNING_FAILED", action_id, {"category": category.value, "reason": str(error)[:1000]})
                        raise
                    budget = self.plan.measurement_budget
                    can_retry = category in {FailureCategory.TRANSIENT_EXECUTION_FAILURE, FailureCategory.RESOURCE_EXHAUSTED} and actions_consumed + 1 < budget.max_actions  # type: ignore[union-attr]
                    if can_retry and budget.baseline_recovery_actions > 0:  # type: ignore[union-attr]
                        fallback_action_id = f"{action_id}:baseline-recovery"
                        state["fallback_intent"] = {
                            "pending": True,
                            "action_id": fallback_action_id,
                            "failed_action_id": action_id,
                            "resume_index": index_before + 1,
                            "generation": int(parent.get("generation", -1)) + 1,
                            "category": category.value,
                        }
                        state["next_decision"] = Decision(DecisionType.RETRY_BASELINE, action_id=fallback_action_id, mode=self.plan.baseline, reason="technical failure requires same-generation baseline recovery").to_dict()
                        state["active_action_id"] = None
                        state["active_profile"] = None
                        state["active_phase"] = None
                        storage.save(state)
                        self._save_report(state)
                        self._emit("TUNING_MODE_REJECTED", action_id, {"category": category.value, "reason": str(error)[:1000], "recovery": "baseline"})
                        continue
                    state["status"] = "FAILED"
                    state["stop_reason"] = f"{category.value}: {str(error)[:1000]}"
                    state["active_action_id"] = None
                    state["active_profile"] = None
                    state["active_phase"] = None
                    storage.save(state)
                    self._save_report(state)
                    self._emit("TUNING_FAILED", action_id, {"category": category.value, "reason": str(error)[:1000]})
                    raise


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


__all__ = [
    "GenerationExecutor", "PerformanceTuningRunner", "ProductionTrainOneExecutor",
    "TuningExecutionError", "TuningRunResult",
]
