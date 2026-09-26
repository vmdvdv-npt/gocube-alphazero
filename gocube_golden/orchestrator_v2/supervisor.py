"""Generic durable supervision for one external process execution.

The supervisor owns process mechanics only: durable child identity, process
group ownership, liveness and progress freshness, bounded same-command retry,
reattachment, exit status, and TERM/KILL cleanup. Domain success criteria
belong to the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import subprocess
import time
import json
from typing import Any

from ..process_supervision import (
    ProcessOwnershipError,
    atomic_write_json,
    clear_active_child,
    heartbeat_timestamp,
    process_group_exists,
    process_group_for,
    process_group_owned_by,
    read_active_child,
    read_json,
    start_owned_child,
    terminate_process_group,
    timestamp_seconds,
    write_active_child,
)


SUPERVISOR_SCHEMA = "gocube-orchestrator-supervisor-v2"
ACTIVE_CHILD_SCHEMA = "gocube-orchestrator-v2-active-child-v2"
EXECUTION_INTENT_SCHEMA = "gocube-orchestrator-v2-execution-intent-v1"
STOP_SCHEMA = "gocube-orchestrator-v2-stop-v2"


class SupervisorAction(str, Enum):
    START = "START"
    REATTACH = "REATTACH"
    STOP = "STOP"


class SupervisorStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class SupervisorIntegrityError(RuntimeError):
    """Durable supervisor evidence is malformed or ownership is unsafe."""


class TechnicalFailure(RuntimeError):
    """A child attempt failed before the process supervisor could succeed."""

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


@dataclass(frozen=True)
class SupervisorPolicy:
    """Bounded process-health and restart policy."""

    heartbeat_grace_seconds: float = 5 * 60.0
    liveness_timeout_seconds: float | None = None
    progress_timeout_seconds: float | None = None
    max_retries: int = 1
    poll_interval_seconds: float = 1.0
    termination_grace_seconds: float = 5.0

    FIELD_NAMES = frozenset(
        {
            "max_retries",
            "liveness_timeout_seconds",
            "progress_timeout_seconds",
            "heartbeat_grace_seconds",
            "poll_interval_seconds",
            "termination_grace_seconds",
        }
    )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object] | None,
        *,
        base: "SupervisorPolicy | None" = None,
        label: str = "supervision",
    ) -> "SupervisorPolicy":
        """Parse a run-spec policy and fail before launching a child.

        ``None`` keeps the current V2 defaults.  A base policy is useful for
        per-action overrides: omitted fields inherit the common policy while
        explicitly supplied fields remain visible in the persisted run-spec.
        """
        if value is None:
            return base or cls()
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")
        unknown = set(value) - cls.FIELD_NAMES
        if unknown:
            raise ValueError(
                f"{label} contains unsupported fields: {', '.join(sorted(map(str, unknown)))}"
            )
        current = base or cls()
        payload = {
            name: getattr(current, name)
            for name in cls.FIELD_NAMES
        }
        payload.update(dict(value))
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "max_retries",
                "liveness_timeout_seconds",
                "progress_timeout_seconds",
                "heartbeat_grace_seconds",
                "poll_interval_seconds",
                "termination_grace_seconds",
            )
        }

    def __post_init__(self) -> None:
        if self.heartbeat_grace_seconds < 0:
            raise ValueError("heartbeat_grace_seconds must be non-negative")
        for name, value in (
            ("liveness_timeout_seconds", self.liveness_timeout_seconds),
            ("progress_timeout_seconds", self.progress_timeout_seconds),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if self.termination_grace_seconds < 0:
            raise ValueError("termination_grace_seconds must be non-negative")

    @property
    def max_attempts(self) -> int:
        return 1 + self.max_retries

    @property
    def liveness_grace_seconds(self) -> float:
        return (
            self.heartbeat_grace_seconds
            if self.liveness_timeout_seconds is None
            else self.liveness_timeout_seconds
        )

    @property
    def progress_grace_seconds(self) -> float | None:
        return self.progress_timeout_seconds


@dataclass(frozen=True)
class ActiveChild:
    """The minimum durable identity needed to reattach safely."""

    execution_id: str
    attempt: int
    pid: int
    process_group: int
    started_at: float
    liveness_path: Path
    progress_path: Path
    schema: str = ACTIVE_CHILD_SCHEMA

    @property
    def heartbeat_path(self) -> Path:
        """Compatibility alias for a single-file heartbeat layout."""
        return self.liveness_path

    def to_dict(self, root: Path) -> dict[str, object]:
        def relative(path: Path) -> str:
            return path.resolve().relative_to(root.resolve()).as_posix()

        return {
            "schema": self.schema,
            "supervisor_schema": SUPERVISOR_SCHEMA,
            "execution_id": self.execution_id,
            "attempt": self.attempt,
            "pid": self.pid,
            "process_group": self.process_group,
            "started_at": self.started_at,
            "liveness_path": relative(self.liveness_path),
            "progress_path": relative(self.progress_path),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        root: Path,
        owner_execution_id: str,
    ) -> "ActiveChild":
        if str(payload.get("schema", "")) != ACTIVE_CHILD_SCHEMA:
            raise SupervisorIntegrityError("active-child schema mismatch")
        execution_id = str(payload.get("execution_id", ""))
        if not execution_id or execution_id != owner_execution_id:
            raise SupervisorIntegrityError("active-child execution ownership mismatch")
        try:
            attempt = int(payload.get("attempt", 1))
            pid = int(payload["pid"])
            process_group = int(payload.get("process_group", pid))
            started_at = timestamp_seconds(payload.get("started_at"))
        except (KeyError, TypeError, ValueError) as exc:
            raise SupervisorIntegrityError("active-child identity is malformed") from exc
        if attempt < 1 or pid <= 1 or process_group <= 1:
            raise SupervisorIntegrityError("active-child identity contains unsafe values")

        raw_liveness = payload.get("liveness_path")
        raw_progress = payload.get("progress_path", raw_liveness)
        liveness_path = _confined_path(root, raw_liveness, "active-child liveness path")
        progress_path = _confined_path(root, raw_progress, "active-child progress path")
        return cls(
            execution_id=execution_id,
            attempt=attempt,
            pid=pid,
            process_group=process_group,
            started_at=started_at,
            liveness_path=liveness_path,
            progress_path=progress_path,
            schema=ACTIVE_CHILD_SCHEMA,
        )


@dataclass(frozen=True)
class LaunchRequest:
    execution_id: str
    attempt: int
    command: tuple[str, ...] | None
    cwd: Path | None
    env: Mapping[str, str] | None
    root: Path
    liveness_path: Path
    progress_path: Path


@dataclass(frozen=True)
class RecoveryPlan:
    action: SupervisorAction
    attempt: int
    reason: str
    active_child: ActiveChild | None = None

    @property
    def should_start(self) -> bool:
        return self.action is SupervisorAction.START

    @property
    def should_reattach(self) -> bool:
        return self.action is SupervisorAction.REATTACH


@dataclass(frozen=True)
class HeartbeatStatus:
    liveness_age_seconds: float | None
    progress_age_seconds: float | None
    liveness_stale: bool
    progress_stale: bool
    progress_token: object | None = None

    @property
    def stale(self) -> bool:
        return self.liveness_stale or self.progress_stale


@dataclass(frozen=True)
class ProcessResult:
    status: SupervisorStatus
    returncode: int | None
    attempts: int
    reattached: bool = False
    reason: str | None = None

    @property
    def success(self) -> bool:
        return self.status is SupervisorStatus.SUCCESS

    @property
    def exit_code(self) -> int | None:
        return self.returncode


SupervisionResult = ProcessResult


def _confined_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise SupervisorIntegrityError(f"{label} is missing")
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SupervisorIntegrityError(f"{label} escapes execution root") from exc
    return path


class SupervisorV2:
    """Supervise one opaque process execution."""

    def __init__(
        self,
        root: str | Path,
        *,
        execution_id: str,
        liveness_path: str | Path,
        progress_path: str | Path,
        launcher: Callable[[LaunchRequest], subprocess.Popen[bytes] | subprocess.Popen[str]] | None = None,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        policy: SupervisorPolicy | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = Path(root).resolve()
        self.execution_id = str(execution_id)
        if not self.execution_id:
            raise ValueError("execution_id must be non-empty")
        self.liveness_path = Path(liveness_path).resolve()
        self.progress_path = Path(progress_path).resolve()
        self.launcher = launcher
        self.command = tuple(str(value) for value in command) if command is not None else None
        if self.command is not None and not self.command:
            raise ValueError("command must not be empty")
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.env = dict(env) if env is not None else None
        self.policy = policy or SupervisorPolicy()
        self.clock = clock
        self.sleeper = sleeper

    @property
    def active_child_path(self) -> Path:
        return self.root / "runtime" / "active-child.json"

    @property
    def execution_intent_path(self) -> Path:
        return self.root / "runtime" / "execution-intent.json"

    @property
    def stop_path(self) -> Path:
        return self.root / "runtime" / "supervisor-stop.json"

    @property
    def result_path(self) -> Path:
        return self.root / "runtime" / "supervisor-result.json"

    @property
    def attempts_path(self) -> Path:
        return self.root / "runtime" / "supervisor-attempts.jsonl"

    def plan(self) -> RecoveryPlan:
        """Decide whether to start, reattach, or stop without launching."""
        try:
            intent = self._read_intent()
            active = self._read_active_child()
        except SupervisorIntegrityError as exc:
            return RecoveryPlan(
                action=SupervisorAction.STOP,
                attempt=1,
                reason=str(exc),
            )

        if self.stop_path.is_file():
            return RecoveryPlan(
                action=SupervisorAction.STOP,
                attempt=int(intent["attempt"]) if intent is not None else 1,
                reason="durable supervisor stop is present",
                active_child=active,
            )

        if active is not None:
            if process_group_exists(active.process_group):
                if self._child_group_matches(active):
                    return RecoveryPlan(
                        action=SupervisorAction.REATTACH,
                        attempt=active.attempt,
                        reason="matching live child owns the durable execution identity",
                        active_child=active,
                    )
                return RecoveryPlan(
                    action=SupervisorAction.STOP,
                    attempt=active.attempt,
                    reason="active child process group ownership does not match its durable identity",
                    active_child=active,
                )
            attempt = active.attempt + 1
        else:
            attempt = 1

        if intent is not None:
            attempt = max(attempt, int(intent["attempt"]))
        if attempt > self.policy.max_attempts:
            return RecoveryPlan(
                action=SupervisorAction.STOP,
                attempt=attempt,
                reason="execution exhausted its bounded technical retry budget",
                active_child=active,
            )
        return RecoveryPlan(
            action=SupervisorAction.START,
            attempt=attempt,
            reason=(
                "retry the same command after a technical failure"
                if attempt > 1 or intent is not None or active is not None
                else "no active child exists"
            ),
            active_child=active,
        )

    def supervise(self) -> ProcessResult:
        return self.run_once()

    def run_once(self) -> ProcessResult:
        result = self._run_once()
        atomic_write_json(
            self.result_path,
            {
                "schema": "gocube-orchestrator-supervisor-result-v1",
                "execution_id": self.execution_id,
                "status": result.status.value,
                "returncode": result.returncode,
                "attempts": result.attempts,
                "reattached": result.reattached,
                "reason": result.reason,
                "recorded_at": self.clock(),
            },
        )
        return result

    def _run_once(self) -> ProcessResult:
        """Supervise one command, including bounded same-command retries."""
        plan = self.plan()
        if plan.action is SupervisorAction.STOP:
            if plan.active_child is not None and process_group_exists(plan.active_child.process_group):
                try:
                    self._terminate_group(plan.active_child, reason=plan.reason)
                except SupervisorIntegrityError as exc:
                    return ProcessResult(
                        status=SupervisorStatus.FAILURE,
                        returncode=None,
                        attempts=0,
                        reason=str(exc),
                    )
            return ProcessResult(
                status=SupervisorStatus.FAILURE,
                returncode=None,
                attempts=0,
                reason=plan.reason,
            )

        attempt = plan.attempt
        reattached = plan.action is SupervisorAction.REATTACH
        ever_reattached = reattached
        active = plan.active_child if reattached else None
        process: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None
        while attempt <= self.policy.max_attempts:
            try:
                if not reattached:
                    process, active = self._start(attempt)
                if active is None:
                    raise TechnicalFailure("supervision has no active child identity")
                returncode = self._monitor(active, process)
                self._clear_runtime_identity()
                return ProcessResult(
                    status=SupervisorStatus.SUCCESS,
                    returncode=returncode,
                    attempts=attempt,
                    reattached=ever_reattached,
                )
            except TechnicalFailure as exc:
                self._record_attempt_event(attempt, active, str(exc))
                if active is not None:
                    try:
                        self._terminate_group(active, reason=str(exc), process=process)
                    except SupervisorIntegrityError as cleanup_exc:
                        self._write_stop(attempt, str(cleanup_exc), active=active)
                        clear_active_child(self.active_child_path)
                        return ProcessResult(
                            status=SupervisorStatus.FAILURE,
                            returncode=exc.returncode,
                            attempts=attempt,
                            reattached=ever_reattached,
                            reason=str(cleanup_exc),
                        )
                clear_active_child(self.active_child_path)
                if attempt >= self.policy.max_attempts:
                    reason = str(exc)
                    self._write_stop(attempt, reason, active=active)
                    return ProcessResult(
                        status=SupervisorStatus.FAILURE,
                        returncode=exc.returncode,
                        attempts=attempt,
                        reattached=ever_reattached,
                        reason=reason,
                    )
                attempt += 1
                self._write_intent(attempt)
                reattached = False
                active = None
                process = None
            except SupervisorIntegrityError as exc:
                if active is not None and self._can_terminate_group(active):
                    self._terminate_group(active, reason=str(exc), process=process)
                self._write_stop(attempt, str(exc), active=active)
                clear_active_child(self.active_child_path)
                return ProcessResult(
                    status=SupervisorStatus.FAILURE,
                    returncode=None,
                    attempts=attempt,
                    reattached=ever_reattached,
                    reason=str(exc),
                )

        raise AssertionError("bounded supervisor loop did not return")

    def acknowledge_stopped_execution(self) -> ProcessResult:
        """Explicitly retire one matching durable stop before a new attempt.

        This is intentionally opt-in.  A missing stop is a successful no-op:
        the caller may be recovering after a restart and an already-started
        child must remain available for :meth:`run_once` to reattach.  When a
        stop is present, every supervisor identity is revalidated before any
        cleanup is performed, and a live recorded process group is terminated
        through the normal ownership checks.
        """
        if not self.stop_path.exists() and not self.stop_path.is_symlink():
            return ProcessResult(
                status=SupervisorStatus.SUCCESS,
                returncode=0,
                attempts=0,
                reason="no durable supervisor stop is present",
            )

        stop = self._read_stop()
        if stop is None:
            # The existence check above can race with an external cleanup.  A
            # vanished stop has the same idempotent semantics as no stop.
            return ProcessResult(
                status=SupervisorStatus.SUCCESS,
                returncode=0,
                attempts=0,
                reason="no durable supervisor stop is present",
            )
        stop_attempt = int(stop["attempt"])
        stop_owner = self._stop_process_identity(stop)
        intent = self._read_intent()
        active = self._read_active_child()

        if intent is not None and int(intent["attempt"]) != stop_attempt:
            raise SupervisorIntegrityError(
                "execution intent does not match the stopped execution"
            )
        if active is not None:
            if active.attempt != stop_attempt:
                raise SupervisorIntegrityError(
                    "active-child identity does not match the stopped execution"
                )
            if stop_owner is not None and stop_owner != (active.pid, active.process_group):
                raise SupervisorIntegrityError(
                    "supervisor stop process ownership does not match active-child"
                )

        # Revalidate the stop before touching any identity record.  In
        # particular, do not turn a concurrent acknowledgement/restart into a
        # cleanup of the new execution intent.
        if self._read_stop() != stop:
            raise SupervisorIntegrityError("supervisor stop changed during acknowledgement")

        if active is not None:
            if process_group_exists(active.process_group):
                if not process_group_owned_by(active.pid, active.process_group):
                    raise SupervisorIntegrityError(
                        "active child process group ownership does not match its durable identity"
                    )
                self._terminate_group(
                    active,
                    reason="explicitly acknowledging the stopped execution",
                )
        elif stop_owner is not None:
            owner_pid, process_group = stop_owner
            if process_group_exists(process_group):
                if not process_group_owned_by(owner_pid, process_group):
                    raise SupervisorIntegrityError(
                        "stopped execution process group ownership does not match its durable identity"
                    )
                owner = ActiveChild(
                    execution_id=self.execution_id,
                    attempt=stop_attempt,
                    pid=owner_pid,
                    process_group=process_group,
                    started_at=0.0,
                    liveness_path=self.liveness_path,
                    progress_path=self.progress_path,
                )
                self._terminate_group(
                    owner,
                    reason="explicitly acknowledging the stopped execution",
                )

        # These helpers re-read and compare the records, so cleanup cannot
        # blindly unlink a foreign or concurrently replaced identity.
        self._clear_active_child_if_unchanged(active)
        self._clear_intent_if_unchanged(intent)
        self._clear_stop_if_unchanged(stop)
        return ProcessResult(
            status=SupervisorStatus.SUCCESS,
            returncode=0,
            attempts=stop_attempt,
            reason="matching durable supervisor stop acknowledged",
        )

    def reconcile_completed_execution(self) -> ProcessResult:
        """Safely finish an execution completed outside the supervisor.

        Callers may have authoritative evidence that the opaque child work is
        complete even though this supervisor instance did not reach its normal
        cleanup path.  The durable identity is still treated as ownership
        evidence: malformed or foreign records fail closed, a live matching
        process group is terminated through the normal ownership checks, and
        runtime files are removed only after their identity is revalidated.
        """
        intent = self._read_intent()
        active = self._read_active_child()
        stop = self._read_stop()
        attempt = (
            active.attempt
            if active is not None
            else int(intent["attempt"])
            if intent is not None
            else 0
        )

        if active is not None and process_group_exists(active.process_group):
            if not process_group_owned_by(active.pid, active.process_group):
                raise SupervisorIntegrityError(
                    "active child process group ownership does not match its durable identity"
                )
            self._terminate_group(
                active,
                reason="execution was externally confirmed complete",
            )

        self._clear_active_child_if_unchanged(active)
        self._clear_intent_if_unchanged(intent)
        self._clear_stop_if_unchanged(stop)
        return ProcessResult(
            status=SupervisorStatus.SUCCESS,
            returncode=0,
            attempts=attempt,
            reason="externally confirmed execution reconciled",
        )

    def reclaim_dead_child(self) -> ProcessResult:
        """Retire a dead child while preserving the next attempt number.

        This is the recovery path for a coordinator crash: the old process
        group must be proven gone, but the durable execution intent remains so
        the replacement supervisor starts attempt ``N + 1`` and mints a new
        execution permit for the same scientific evaluation.
        """
        active = self._read_active_child()
        if active is None:
            return ProcessResult(
                status=SupervisorStatus.SUCCESS,
                returncode=0,
                attempts=0,
                reason="no dead active child requires reclamation",
            )
        if process_group_exists(active.process_group):
            raise SupervisorIntegrityError(
                "cannot reclaim a live active child process group"
            )
        self._write_intent(active.attempt + 1)
        self._clear_active_child_if_unchanged(active)
        self.result_path.unlink(missing_ok=True)
        return ProcessResult(
            status=SupervisorStatus.SUCCESS,
            returncode=0,
            attempts=active.attempt + 1,
            reason="dead child reclaimed for a new bounded attempt",
        )

    def heartbeat_status(self, child: ActiveChild, *, now: float | None = None) -> HeartbeatStatus:
        """Return generic liveness/progress freshness for a child."""
        current = self.clock() if now is None else float(now)
        liveness_at, _ = self._heartbeat_value(child.liveness_path, "liveness_at")
        progress_at, progress_token = self._heartbeat_value(child.progress_path, "progress_at")
        started_age = max(0.0, current - child.started_at)
        liveness_age = started_age if liveness_at is None else max(0.0, current - liveness_at)
        progress_age = started_age if progress_at is None else max(0.0, current - progress_at)
        progress_grace = self.policy.progress_grace_seconds
        return HeartbeatStatus(
            liveness_age_seconds=liveness_age,
            progress_age_seconds=progress_age,
            liveness_stale=liveness_age >= self.policy.liveness_grace_seconds,
            progress_stale=(
                progress_grace is not None and progress_age >= progress_grace
            ),
            progress_token=progress_token,
        )

    def _read_intent(self) -> dict[str, object] | None:
        if not self.execution_intent_path.is_file():
            return None
        try:
            payload = read_json(self.execution_intent_path)
            if payload.get("schema") != EXECUTION_INTENT_SCHEMA:
                raise ValueError("schema mismatch")
            if str(payload.get("execution_id")) != self.execution_id:
                raise ValueError("execution ownership mismatch")
            attempt = int(payload.get("attempt", 1))
            if attempt < 1:
                raise ValueError("unsafe attempt")
            return payload
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SupervisorIntegrityError("execution intent is malformed") from exc

    def _read_active_child(self) -> ActiveChild | None:
        if not self.active_child_path.is_file():
            return None
        try:
            return ActiveChild.from_dict(
                read_active_child(self.active_child_path) or {},
                root=self.root,
                owner_execution_id=self.execution_id,
            )
        except (OSError, ValueError, SupervisorIntegrityError) as exc:
            if isinstance(exc, SupervisorIntegrityError):
                raise
            raise SupervisorIntegrityError("active-child record is unreadable") from exc

    def _read_stop(self) -> dict[str, object] | None:
        if not self.stop_path.exists():
            if self.stop_path.is_symlink():
                raise SupervisorIntegrityError("supervisor stop is malformed")
            return None
        try:
            payload = read_json(self.stop_path)
            if payload.get("schema") != STOP_SCHEMA:
                raise ValueError("schema mismatch")
            execution_id = payload.get("execution_id")
            if not isinstance(execution_id, str) or execution_id != self.execution_id:
                raise ValueError("execution ownership mismatch")
            attempt = int(payload.get("attempt", 1))
            if attempt < 1:
                raise ValueError("unsafe attempt")
            return payload
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise SupervisorIntegrityError("supervisor stop is malformed") from exc

    @staticmethod
    def _stop_process_identity(stop: Mapping[str, object]) -> tuple[int, int] | None:
        raw_pid = stop.get("pid")
        raw_process_group = stop.get("process_group")
        if raw_pid is None and raw_process_group is None:
            return None
        if (
            type(raw_pid) is not int
            or type(raw_process_group) is not int
            or raw_pid <= 1
            or raw_process_group <= 1
        ):
            raise SupervisorIntegrityError("supervisor stop process identity is unsafe")
        return raw_pid, raw_process_group

    def _write_intent(self, attempt: int) -> None:
        atomic_write_json(
            self.execution_intent_path,
            {
                "schema": EXECUTION_INTENT_SCHEMA,
                "supervisor_schema": SUPERVISOR_SCHEMA,
                "execution_id": self.execution_id,
                "attempt": attempt,
                "updated_at": self.clock(),
            },
        )

    def _record_attempt_event(
        self,
        attempt: int,
        active: ActiveChild | None,
        reason: str,
    ) -> None:
        """Persist every technical failure, including retryable timeouts."""
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "schema": "gocube-orchestrator-v2-supervisor-attempt-v1",
            "execution_id": self.execution_id,
            "attempt": int(attempt),
            "reason": reason,
            "recorded_at": self.clock(),
        }
        if active is not None:
            payload.update({"pid": active.pid, "process_group": active.process_group})
        with self.attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _start(
        self,
        attempt: int,
    ) -> tuple[subprocess.Popen[bytes] | subprocess.Popen[str], ActiveChild]:
        request = LaunchRequest(
            execution_id=self.execution_id,
            attempt=attempt,
            command=self.command,
            cwd=self.cwd,
            env=self.env,
            root=self.root,
            liveness_path=self.liveness_path,
            progress_path=self.progress_path,
        )
        self._write_intent(attempt)
        self.liveness_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.liveness_path.unlink(missing_ok=True)
        if self.progress_path != self.liveness_path:
            self.progress_path.unlink(missing_ok=True)
        if self.launcher is not None:
            process = self.launcher(request)
        else:
            if self.command is None:
                raise SupervisorIntegrityError("run_once requires launcher or command")
            process = start_owned_child(
                self.command,
                cwd=self.cwd,
                env=self.env,
                popen=subprocess.Popen,
            )
        try:
            pid = int(process.pid)
            process_group = process_group_for(process)
        except (AttributeError, OSError, ProcessOwnershipError, TypeError, ValueError) as exc:
            raise SupervisorIntegrityError("launcher did not return a usable child process") from exc
        active = ActiveChild(
            execution_id=self.execution_id,
            attempt=attempt,
            pid=pid,
            process_group=process_group,
            started_at=self.clock(),
            liveness_path=self.liveness_path,
            progress_path=self.progress_path,
        )
        try:
            write_active_child(self.active_child_path, active.to_dict(self.root))
        except (KeyboardInterrupt, SystemExit):
            self._terminate_group(active, reason="active-child publication interrupted", process=process)
            raise
        except Exception as exc:
            self._terminate_group(active, reason="active-child publication failed", process=process)
            raise SupervisorIntegrityError("could not publish active-child identity") from exc
        return process, active

    def _monitor(
        self,
        active: ActiveChild,
        process: subprocess.Popen[bytes] | subprocess.Popen[str] | None,
    ) -> int:
        previous = self._heartbeat_values(active)
        now = self.clock()
        liveness_changed_at = self._baseline_time(previous[0], active.started_at, now)
        progress_changed_at = self._baseline_time(previous[1], active.started_at, now)
        while True:
            if process is not None:
                returncode = process.poll()
                if returncode is not None:
                    if int(returncode) == 0:
                        return 0
                    raise TechnicalFailure(
                        f"child exited with code {int(returncode)}",
                        returncode=int(returncode),
                    )
            else:
                returncode = self._reattached_returncode(active.pid)
                if returncode is not None:
                    if returncode == 0:
                        if process_group_exists(active.process_group):
                            self._terminate_group(active, reason="reattached child left descendants")
                        return 0
                    raise TechnicalFailure(
                        "reattached child exited with a non-zero status",
                        returncode=returncode,
                    )
            if process is None and not process_group_exists(active.process_group):
                returncode = self._reattached_returncode(active.pid)
                if returncode == 0:
                    return 0
                raise TechnicalFailure(
                    "reattached child process group is gone without a successful exit status",
                    returncode=returncode,
                )

            now = self.clock()
            current = self._heartbeat_values(active)
            if current[0] != previous[0] and current[0] is not None:
                liveness_changed_at = now
            if (current[1] != previous[1] or current[2] != previous[2]) and (
                current[1] is not None or current[2] is not None
            ):
                progress_changed_at = now
            liveness_age = max(0.0, now - liveness_changed_at)
            progress_age = max(0.0, now - progress_changed_at)
            if liveness_age >= self.policy.liveness_grace_seconds:
                raise TechnicalFailure("liveness heartbeat is missing or stale")
            progress_grace = self.policy.progress_grace_seconds
            if progress_grace is not None and progress_age >= progress_grace:
                raise TechnicalFailure("progress heartbeat is missing or stale")
            previous = current
            self.sleeper(self.policy.poll_interval_seconds)

    def _heartbeat_values(self, child: ActiveChild) -> tuple[float | None, float | None, object | None]:
        liveness_at, _ = self._heartbeat_value(child.liveness_path, "liveness_at")
        progress_at, progress_token = self._heartbeat_value(child.progress_path, "progress_at")
        return liveness_at, progress_at, progress_token

    @staticmethod
    def _baseline_time(value: float | None, started_at: float, now: float) -> float:
        if value is None:
            return started_at
        return min(now, max(started_at, value))

    @staticmethod
    def _heartbeat_value(path: Path, field: str) -> tuple[float | None, object | None]:
        if not path.is_file():
            return None, None
        try:
            payload: Mapping[str, Any] = read_json(path)
        except (OSError, ValueError):
            payload = {}
        timestamp = heartbeat_timestamp(path, field)
        if timestamp is None:
            try:
                timestamp = path.stat().st_mtime
            except OSError:
                timestamp = None
        token = payload.get("progress_token") if field == "progress_at" else None
        return timestamp, token

    @staticmethod
    def _reattached_returncode(pid: int) -> int | None:
        try:
            waited_pid, status = os.waitpid(int(pid), os.WNOHANG)
        except (ChildProcessError, OSError):
            return None
        if waited_pid != int(pid):
            return None
        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        if os.WIFSIGNALED(status):
            return -os.WTERMSIG(status)
        return None

    def _child_group_matches(self, child: ActiveChild) -> bool:
        if not process_group_exists(child.process_group):
            return False
        return process_group_owned_by(child.pid, child.process_group)

    def _can_terminate_group(self, child: ActiveChild) -> bool:
        return process_group_owned_by(child.pid, child.process_group)

    def _terminate_group(
        self,
        child: ActiveChild,
        *,
        reason: str,
        process: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None,
    ) -> None:
        try:
            terminate_process_group(
                process,
                owner_pid=child.pid,
                process_group=child.process_group,
                term_timeout=self.policy.termination_grace_seconds,
                kill_timeout=self.policy.termination_grace_seconds,
                reason=reason,
                sleeper=self.sleeper,
            )
        except ProcessOwnershipError as exc:
            # A reattached supervisor has no Popen object to reap.  If the
            # verified leader was already terminated, reap it when it is our
            # child so a zombie cannot make an otherwise empty process group
            # look live.  Descendants are still checked below and remain a
            # hard failure if the group survives.
            if process is None:
                self._reattached_returncode(child.pid)
                if not process_group_exists(child.process_group):
                    return
            raise SupervisorIntegrityError(str(exc)) from exc

    def _write_stop(
        self,
        attempt: int,
        reason: str,
        *,
        active: ActiveChild | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "schema": STOP_SCHEMA,
            "supervisor_schema": SUPERVISOR_SCHEMA,
            "execution_id": self.execution_id,
            "attempt": attempt,
            "reason": reason,
            "stopped_at": self.clock(),
        }
        if active is not None:
            payload.update({"pid": active.pid, "process_group": active.process_group})
        atomic_write_json(
            self.stop_path,
            payload,
        )

    def _clear_runtime_identity(self) -> None:
        clear_active_child(self.active_child_path)
        self.execution_intent_path.unlink(missing_ok=True)

    def _clear_active_child_if_unchanged(self, expected: ActiveChild | None) -> None:
        current = self._read_active_child()
        if current is None:
            if expected is not None:
                return
            return
        if expected is None or current != expected:
            raise SupervisorIntegrityError(
                "active-child identity changed during completed-execution reconciliation"
            )
        clear_active_child(self.active_child_path)

    def _clear_intent_if_unchanged(self, expected: dict[str, object] | None) -> None:
        current = self._read_intent()
        if current is None:
            return
        if expected is None or current != expected:
            raise SupervisorIntegrityError(
                "execution intent changed during completed-execution reconciliation"
            )
        self.execution_intent_path.unlink(missing_ok=True)

    def _clear_stop_if_unchanged(self, expected: dict[str, object] | None) -> None:
        current = self._read_stop()
        if current is None:
            return
        if expected is None or current != expected:
            raise SupervisorIntegrityError(
                "supervisor stop changed during completed-execution reconciliation"
            )
        self.stop_path.unlink(missing_ok=True)


def supervise(
    command: Sequence[str],
    cwd: str | Path | None,
    env: Mapping[str, str] | None,
    execution_id: str,
    liveness_path: str | Path,
    progress_path: str | Path,
    policy: SupervisorPolicy | None = None,
    *,
    root: str | Path | None = None,
) -> ProcessResult:
    """Convenience API for supervising one opaque command execution."""
    state_root = Path(root).resolve() if root is not None else Path(liveness_path).resolve().parent
    return SupervisorV2(
        state_root,
        execution_id=execution_id,
        liveness_path=liveness_path,
        progress_path=progress_path,
        command=command,
        cwd=cwd,
        env=env,
        policy=policy,
    ).run_once()


__all__ = [
    "ACTIVE_CHILD_SCHEMA",
    "ActiveChild",
    "EXECUTION_INTENT_SCHEMA",
    "HeartbeatStatus",
    "LaunchRequest",
    "ProcessResult",
    "RecoveryPlan",
    "STOP_SCHEMA",
    "SupervisorAction",
    "SupervisorIntegrityError",
    "SupervisorPolicy",
    "SupervisorStatus",
    "SupervisorV2",
    "SupervisionResult",
    "TechnicalFailure",
    "supervise",
]
