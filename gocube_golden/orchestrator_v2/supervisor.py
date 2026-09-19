"""Small, topology-neutral Supervisor V2.

This module owns only the durable mechanics around one generation execution:
commit-marker discovery, child identity, process-group cleanup, heartbeat
health, and one bounded retry.  It deliberately does not inherit from the V1
orchestrator and does not contain a business-state machine, Arena handling, or
continuous scheduling loop.

The supervisor's restart decision is intentionally separate from execution.
``plan()`` is read-only; callers can use it to prove recovery semantics without
starting the next generation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

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
ACTIVE_CHILD_SCHEMA = "gocube-orchestrator-v2-active-child-v1"
LEGACY_ACTIVE_CHILD_SCHEMA = "gocube-training-active-child-v1"
GENERATION_INTENT_SCHEMA = "gocube-orchestrator-v2-generation-intent-v1"
STOP_SCHEMA = "gocube-orchestrator-v2-stop-v1"
_COMMIT_RE = re.compile(r"^generation-(\d+)\.complete\.json$")


class SupervisorAction(str, Enum):
    """The only decisions needed to recover one generation."""

    START = "START"
    REATTACH = "REATTACH"
    STOP = "STOP"


class SupervisorStatus(str, Enum):
    COMMITTED = "COMMITTED"
    STOPPED = "STOPPED"


class SupervisorIntegrityError(RuntimeError):
    """Durable supervisor evidence is malformed or ownership is unsafe."""


class TechnicalFailure(RuntimeError):
    """One child attempt failed before producing its commit marker."""


@dataclass(frozen=True)
class SupervisorPolicy:
    """Bounded technical-failure policy.

    The production defaults are deliberately fixed to five minutes of
    heartbeat grace, a bounded post-commit drain, and exactly one retry of
    the same generation.  A shorter grace is useful for unit tests; more than
    one retry is rejected so a caller cannot accidentally turn this into an
    unbounded loop.
    """

    heartbeat_grace_seconds: float = 5 * 60.0
    max_retries: int = 1
    poll_interval_seconds: float = 1.0
    termination_grace_seconds: float = 5.0
    committed_drain_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.heartbeat_grace_seconds < 0:
            raise ValueError("heartbeat_grace_seconds must be non-negative")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 1:
            raise ValueError("max_retries must be 0 or 1")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if self.termination_grace_seconds < 0:
            raise ValueError("termination_grace_seconds must be non-negative")
        if self.committed_drain_seconds < 0:
            raise ValueError("committed_drain_seconds must be non-negative")

    @property
    def max_attempts(self) -> int:
        return 1 + self.max_retries


@dataclass(frozen=True)
class CommitMarker:
    generation: int
    path: Path
    payload: Mapping[str, object]


@dataclass(frozen=True)
class ActiveChild:
    """The minimum durable identity needed to reattach safely."""

    lineage_id: str
    generation: int
    attempt: int
    execution_unit_id: str
    pid: int
    process_group: int
    started_at: float
    liveness_path: Path
    progress_path: Path
    schema: str = ACTIVE_CHILD_SCHEMA

    @property
    def heartbeat_path(self) -> Path:
        """Compatibility alias for the common single-file heartbeat layout."""
        return self.liveness_path

    def to_dict(self, root: Path) -> dict[str, object]:
        def relative(path: Path) -> str:
            return path.resolve().relative_to(root.resolve()).as_posix()

        return {
            "schema": self.schema,
            "supervisor_schema": SUPERVISOR_SCHEMA,
            "lineage_id": self.lineage_id,
            "generation": self.generation,
            "attempt": self.attempt,
            "execution_unit_id": self.execution_unit_id,
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
        owner_lineage_id: str,
    ) -> "ActiveChild":
        schema = str(payload.get("schema", ""))
        if schema not in {ACTIVE_CHILD_SCHEMA, LEGACY_ACTIVE_CHILD_SCHEMA}:
            raise SupervisorIntegrityError("active-child schema mismatch")
        raw_lineage = payload.get("lineage_id", owner_lineage_id)
        lineage_id = str(raw_lineage)
        try:
            generation = int(payload["generation"])
            attempt = int(payload.get("attempt", 1))
            pid = int(payload["pid"])
            process_group = int(payload.get("process_group", pid))
            started_at = timestamp_seconds(payload.get("started_at"))
        except (KeyError, TypeError, ValueError) as exc:
            raise SupervisorIntegrityError("active-child identity is malformed") from exc
        if lineage_id != owner_lineage_id:
            raise SupervisorIntegrityError("active-child lineage ownership mismatch")
        if generation < 0 or attempt < 1 or pid <= 1 or process_group <= 1:
            raise SupervisorIntegrityError("active-child identity contains unsafe values")

        default_heartbeat = root / "runtime" / "heartbeats" / f"generation-{generation:04d}.json"
        raw_liveness = payload.get("liveness_path", payload.get("heartbeat_path", str(default_heartbeat)))
        raw_progress = payload.get("progress_path", raw_liveness)
        liveness_path = _confined_path(root, raw_liveness, "active-child liveness path")
        progress_path = _confined_path(root, raw_progress, "active-child progress path")
        execution_unit_id = str(
            payload.get("execution_unit_id", f"legacy-generation-{generation:04d}-pid-{pid}")
        )
        if not execution_unit_id or execution_unit_id in {".", ".."}:
            raise SupervisorIntegrityError("active-child execution unit is malformed")
        return cls(
            lineage_id=lineage_id,
            generation=generation,
            attempt=attempt,
            execution_unit_id=execution_unit_id,
            pid=pid,
            process_group=process_group,
            started_at=started_at,
            liveness_path=liveness_path,
            progress_path=progress_path,
            schema=schema,
        )


@dataclass(frozen=True)
class LaunchRequest:
    generation: int
    attempt: int
    execution_unit_id: str
    root: Path
    liveness_path: Path
    progress_path: Path


@dataclass(frozen=True)
class RecoveryPlan:
    action: SupervisorAction
    generation: int
    last_committed_generation: int | None
    attempt: int
    reason: str
    active_child: ActiveChild | None = None

    @property
    def next_generation(self) -> int:
        return (self.last_committed_generation or 0) + 1

    @property
    def should_start(self) -> bool:
        return self.action is SupervisorAction.START

    @property
    def should_reattach(self) -> bool:
        return self.action is SupervisorAction.REATTACH


@dataclass(frozen=True)
class SupervisionResult:
    status: SupervisorStatus
    generation: int
    attempts: int
    last_committed_generation: int | None
    reattached: bool = False


@dataclass(frozen=True)
class HeartbeatStatus:
    liveness_age_seconds: float | None
    progress_age_seconds: float | None
    liveness_stale: bool
    progress_stale: bool

    @property
    def stale(self) -> bool:
        return self.liveness_stale or self.progress_stale


def _confined_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SupervisorIntegrityError(f"{label} is missing")
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SupervisorIntegrityError(f"{label} escapes lineage root") from exc
    return path


class SupervisorV2:
    """Supervise exactly one generation at a time.

    ``launcher`` receives a :class:`LaunchRequest` and must return a process
    started in its own session/process group.  If it is omitted, ``command``
    is launched with ``start_new_session=True``.  No topology-specific driver
    or V1 orchestrator is imported here.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        lineage_id: str,
        initial_committed_generation: int | None = None,
        target_generation: int | None = None,
        launcher: Callable[[LaunchRequest], subprocess.Popen[bytes] | subprocess.Popen[str]] | None = None,
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        policy: SupervisorPolicy | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = Path(root).resolve()
        self.lineage_id = str(lineage_id)
        if not self.lineage_id or "/" in self.lineage_id or "\\" in self.lineage_id:
            raise ValueError("lineage_id must be one safe path component")
        if initial_committed_generation is not None:
            if type(initial_committed_generation) is not int or initial_committed_generation < 0:
                raise ValueError("initial_committed_generation must be a non-negative integer")
        if target_generation is not None:
            if type(target_generation) is not int or target_generation < 0:
                raise ValueError("target_generation must be a non-negative integer")
            if (
                initial_committed_generation is not None
                and target_generation <= initial_committed_generation
            ):
                raise ValueError("target_generation must be after the initial committed generation")
        self.initial_committed_generation = initial_committed_generation
        self.target_generation = target_generation
        if launcher is not None and command is not None:
            raise ValueError("SupervisorV2 accepts launcher or command, not both")
        self.launcher = launcher
        self.command = tuple(str(value) for value in command) if command is not None else None
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.env = dict(env) if env is not None else None
        self.policy = policy or SupervisorPolicy()
        self.clock = clock
        self.sleeper = sleeper

    @property
    def active_child_path(self) -> Path:
        return self.root / "runtime" / "active-child.json"

    @property
    def generation_intent_path(self) -> Path:
        return self.root / "runtime" / "generation-intent.json"

    @property
    def stop_path(self) -> Path:
        return self.root / "runtime" / "supervisor-stop.json"

    def last_committed(self) -> CommitMarker | None:
        """Return the highest valid commit marker without changing the run."""
        markers: list[CommitMarker] = []
        for path in self.root.glob("generation-*.complete.json"):
            match = _COMMIT_RE.fullmatch(path.name)
            if match is None:
                continue
            generation = int(match.group(1))
            try:
                payload = read_json(path)
            except (OSError, ValueError) as exc:
                raise SupervisorIntegrityError(f"cannot read commit marker: {path}") from exc
            try:
                declared_generation = int(payload["generation"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SupervisorIntegrityError(f"commit marker generation is malformed: {path}") from exc
            if declared_generation != generation:
                raise SupervisorIntegrityError(f"commit marker filename/payload mismatch: {path}")
            for owner_key in ("lineage_id", "run_id"):
                owner = payload.get(owner_key)
                if owner is not None and str(owner) != self.lineage_id:
                    raise SupervisorIntegrityError(f"commit marker ownership mismatch: {path}")
            markers.append(CommitMarker(generation=generation, path=path, payload=payload))
        if not markers:
            return None
        markers.sort(key=lambda marker: marker.generation)
        return markers[-1]

    def plan(self) -> RecoveryPlan:
        """Read restart evidence and decide START, REATTACH, or STOP.

        This method is read-only.  In particular, a plan for generation M95
        does not launch M95.
        """
        committed = self.last_committed()
        committed_generation = committed.generation if committed is not None else None
        baseline = self.initial_committed_generation
        if committed_generation is None:
            last_generation = baseline
        elif baseline is None:
            last_generation = committed_generation
        else:
            if (
                self.target_generation is not None
                and committed_generation > self.target_generation
            ):
                return RecoveryPlan(
                    action=SupervisorAction.STOP,
                    generation=self.target_generation,
                    last_committed_generation=committed_generation,
                    attempt=1,
                    reason="committed marker is newer than the requested recovery generation",
                )
            if committed_generation < baseline:
                return RecoveryPlan(
                    action=SupervisorAction.STOP,
                    generation=baseline + 1,
                    last_committed_generation=baseline,
                    attempt=1,
                    reason="lineage commit marker is older than its declared initial generation",
                )
            last_generation = committed_generation
        default_generation = (
            self.target_generation
            if self.target_generation is not None
            else (last_generation or 0) + 1
        )
        try:
            intent = self._read_intent()
            active = self._read_active_child()
        except SupervisorIntegrityError as exc:
            return RecoveryPlan(
                action=SupervisorAction.STOP,
                generation=default_generation,
                last_committed_generation=last_generation,
                attempt=1,
                reason=str(exc),
            )

        generation = default_generation
        if intent is not None and int(intent["generation"]) > (last_generation or -1):
            generation = int(intent["generation"])
        if active is not None and active.generation > (last_generation or -1):
            if generation != active.generation:
                return RecoveryPlan(
                    action=SupervisorAction.STOP,
                    generation=default_generation,
                    last_committed_generation=last_generation,
                    attempt=active.attempt,
                    reason="active child and generation intent disagree",
                    active_child=active,
                )
            group_alive = process_group_exists(active.process_group)
            if group_alive and self._child_group_matches(active):
                return RecoveryPlan(
                    action=SupervisorAction.REATTACH,
                    generation=generation,
                    last_committed_generation=last_generation,
                    attempt=active.attempt,
                    reason="matching live child owns the uncommitted generation",
                    active_child=active,
                )
            if group_alive:
                return RecoveryPlan(
                    action=SupervisorAction.STOP,
                    generation=generation,
                    last_committed_generation=last_generation,
                    attempt=active.attempt,
                    reason="active child process group ownership does not match its durable identity",
                    active_child=active,
                )
            attempt = max(active.attempt + 1, int(intent["attempt"]) if intent else 1)
        else:
            attempt = int(intent["attempt"]) if intent is not None and int(intent["generation"]) == generation else 1

        if self.stop_path.is_file():
            return RecoveryPlan(
                action=SupervisorAction.STOP,
                generation=generation,
                last_committed_generation=last_generation,
                attempt=attempt,
                reason="durable supervisor stop is present",
                active_child=active,
            )
        if attempt > self.policy.max_attempts:
            return RecoveryPlan(
                action=SupervisorAction.STOP,
                generation=generation,
                last_committed_generation=last_generation,
                attempt=attempt,
                reason="generation exhausted its bounded technical retry budget",
                active_child=active,
            )
        return RecoveryPlan(
            action=SupervisorAction.START,
            generation=generation,
            last_committed_generation=last_generation,
            attempt=attempt,
            reason=(
                "uncommitted generation has no live matching child; rerun the same generation"
                if generation != default_generation or intent is not None or active is not None
                else "next generation after the last committed marker"
            ),
            active_child=active,
        )

    def run_once(self) -> SupervisionResult:
        """Supervise one generation, including at most one same-generation retry."""
        plan = self.plan()
        if plan.action is SupervisorAction.STOP:
            return SupervisionResult(
                status=SupervisorStatus.STOPPED,
                generation=plan.generation,
                attempts=plan.attempt,
                last_committed_generation=plan.last_committed_generation,
                reattached=False,
            )

        generation = plan.generation
        attempt = plan.attempt
        reattached = plan.action is SupervisorAction.REATTACH
        while attempt <= self.policy.max_attempts:
            process: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None
            active = plan.active_child if reattached else None
            try:
                if reattached:
                    if active is None:
                        raise TechnicalFailure("reattach plan has no active child")
                else:
                    self._discard_committed_stale_child(active, plan.last_committed_generation)
                    process, active = self._start(generation, attempt)
                self._monitor(active, process)
                committed = self.last_committed()
                if committed is None or committed.generation < generation:
                    raise TechnicalFailure(
                        f"generation {generation} exited without its commit marker"
                    )
                self._clear_runtime_identity()
                return SupervisionResult(
                    status=SupervisorStatus.COMMITTED,
                    generation=generation,
                    attempts=attempt,
                    last_committed_generation=committed.generation,
                    reattached=reattached,
                )
            except TechnicalFailure as exc:
                if active is not None:
                    self._terminate_group(active, reason=str(exc), process=process)
                clear_active_child(self.active_child_path)
                if attempt >= self.policy.max_attempts:
                    self._write_stop(generation, attempt, str(exc))
                    return SupervisionResult(
                        status=SupervisorStatus.STOPPED,
                        generation=generation,
                        attempts=attempt,
                        last_committed_generation=(
                            self.last_committed().generation if self.last_committed() is not None else None
                        ),
                        reattached=reattached,
                    )
                attempt += 1
                self._write_intent(generation, attempt)
                plan = RecoveryPlan(
                    action=SupervisorAction.START,
                    generation=generation,
                    last_committed_generation=plan.last_committed_generation,
                    attempt=attempt,
                    reason="retrying the same generation after one technical failure",
                )
                reattached = False
            except SupervisorIntegrityError as exc:
                if active is not None:
                    if self._can_terminate_group(active):
                        self._terminate_group(active, reason=str(exc), process=process)
                self._write_stop(generation, attempt, str(exc))
                return SupervisionResult(
                    status=SupervisorStatus.STOPPED,
                    generation=generation,
                    attempts=attempt,
                    last_committed_generation=plan.last_committed_generation,
                    reattached=reattached,
                )

        raise AssertionError("bounded supervisor loop did not return")

    def heartbeat_status(self, child: ActiveChild, *, now: float | None = None) -> HeartbeatStatus:
        """Return independent liveness/progress freshness for a child."""
        current = self.clock() if now is None else float(now)
        liveness_at = self._heartbeat_timestamp(child.liveness_path, "liveness_at", current)
        progress_at = self._heartbeat_timestamp(child.progress_path, "progress_at", current)
        grace = self.policy.heartbeat_grace_seconds
        # A newly started child gets the full grace window to publish its
        # first heartbeat.  Liveness and progress get that grace independently.
        started_age = max(0.0, current - child.started_at)
        liveness_age = started_age if liveness_at is None else max(0.0, current - liveness_at)
        progress_age = started_age if progress_at is None else max(0.0, current - progress_at)
        return HeartbeatStatus(
            liveness_age_seconds=liveness_age,
            progress_age_seconds=progress_age,
            liveness_stale=liveness_age >= grace,
            progress_stale=progress_age >= grace,
        )

    def _read_intent(self) -> dict[str, object] | None:
        if not self.generation_intent_path.is_file():
            return None
        try:
            payload = read_json(self.generation_intent_path)
            if payload.get("schema") != GENERATION_INTENT_SCHEMA:
                raise ValueError("schema mismatch")
            if str(payload.get("lineage_id")) != self.lineage_id:
                raise ValueError("lineage mismatch")
            generation = int(payload["generation"])
            attempt = int(payload.get("attempt", 1))
            if generation < 0 or attempt < 1:
                raise ValueError("unsafe generation intent")
            return payload
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SupervisorIntegrityError("generation intent is malformed") from exc

    def _read_active_child(self) -> ActiveChild | None:
        if not self.active_child_path.is_file():
            return None
        try:
            return ActiveChild.from_dict(
                read_active_child(self.active_child_path) or {},
                root=self.root,
                owner_lineage_id=self.lineage_id,
            )
        except (OSError, ValueError, SupervisorIntegrityError) as exc:
            if isinstance(exc, SupervisorIntegrityError):
                raise
            raise SupervisorIntegrityError("active-child record is unreadable") from exc

    def _write_intent(self, generation: int, attempt: int) -> None:
        atomic_write_json(
            self.generation_intent_path,
            {
                "schema": GENERATION_INTENT_SCHEMA,
                "supervisor_schema": SUPERVISOR_SCHEMA,
                "lineage_id": self.lineage_id,
                "generation": generation,
                "attempt": attempt,
                "updated_at": self.clock(),
            },
        )

    def _start(self, generation: int, attempt: int) -> tuple[subprocess.Popen[bytes] | subprocess.Popen[str], ActiveChild]:
        unit_id = f"generation-{generation:04d}-attempt-{attempt:02d}-{uuid.uuid4().hex[:12]}"
        liveness_path = self.root / "runtime" / "heartbeats" / f"generation-{generation:04d}.json"
        progress_path = liveness_path
        request = LaunchRequest(
            generation=generation,
            attempt=attempt,
            execution_unit_id=unit_id,
            root=self.root,
            liveness_path=liveness_path,
            progress_path=progress_path,
        )
        self._write_intent(generation, attempt)
        liveness_path.unlink(missing_ok=True)
        if progress_path != liveness_path:
            progress_path.unlink(missing_ok=True)
        if self.launcher is not None:
            process = self.launcher(request)
        else:
            if self.command is None:
                raise SupervisorIntegrityError("run_once requires launcher or command")
            child_env = dict(os.environ if self.env is None else self.env)
            child_env.update(
                {
                    "AZ_V2_LINEAGE_ID": self.lineage_id,
                    "AZ_V2_GENERATION": str(generation),
                    "AZ_V2_EXECUTION_UNIT_ID": unit_id,
                    "AZ_V2_LIVENESS_HEARTBEAT_PATH": str(liveness_path),
                    "AZ_V2_PROGRESS_HEARTBEAT_PATH": str(progress_path),
                    # The existing driver accepts this primitive directly.
                    "AZ_DRIVER_HEARTBEAT_PATH": str(liveness_path),
                }
            )
            process = start_owned_child(
                self.command,
                cwd=self.cwd,
                env=child_env,
                popen=subprocess.Popen,
            )
        try:
            pid = int(process.pid)
            process_group = process_group_for(process)
        except (AttributeError, OSError, ProcessOwnershipError, TypeError, ValueError) as exc:
            raise SupervisorIntegrityError("launcher did not return a usable child process") from exc
        active = ActiveChild(
            lineage_id=self.lineage_id,
            generation=generation,
            attempt=attempt,
            execution_unit_id=unit_id,
            pid=pid,
            process_group=process_group,
            started_at=self.clock(),
            liveness_path=liveness_path,
            progress_path=progress_path,
        )
        try:
            write_active_child(self.active_child_path, active.to_dict(self.root))
        except BaseException as exc:
            self._terminate_group(active, reason="active-child publication failed", process=process)
            raise SupervisorIntegrityError("could not publish active-child identity") from exc
        return process, active

    def _monitor(
        self,
        active: ActiveChild,
        process: subprocess.Popen[bytes] | subprocess.Popen[str] | None,
    ) -> None:
        while True:
            if self._has_commit(active.generation):
                self._drain_committed_child(active, process)
                clear_active_child(self.active_child_path)
                return

            if process is not None and process.poll() is not None:
                code = int(process.returncode)
                raise TechnicalFailure(f"child exited with code {code}")
            if not process_group_exists(active.process_group):
                raise TechnicalFailure("active child process group is gone")

            health = self.heartbeat_status(active)
            if health.stale:
                reasons: list[str] = []
                if health.liveness_stale:
                    reasons.append("liveness heartbeat missing or stale")
                if health.progress_stale:
                    reasons.append("progress heartbeat missing or stale")
                raise TechnicalFailure("; ".join(reasons))
            self.sleeper(self.policy.poll_interval_seconds)

    def _drain_committed_child(
        self,
        active: ActiveChild,
        process: subprocess.Popen[bytes] | subprocess.Popen[str] | None,
    ) -> None:
        """Let a committed child finish its post-marker publication work.

        The generation commit marker is written before the driver publishes
        its small result record.  Killing the process group as soon as the
        marker appears can therefore leave a committed generation without
        its result evidence.  Drain the leader (or the reattached group) for
        the bounded termination grace, then clean up any surviving descendants
        with the same scoped ownership checks.
        """
        timeout = max(0.0, float(self.policy.committed_drain_seconds))
        if process is not None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        else:
            deadline = time.monotonic() + timeout
            while process_group_exists(active.process_group):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self.sleeper(min(self.policy.poll_interval_seconds, remaining))
        if process_group_exists(active.process_group):
            self._terminate_group(active, reason="committed child did not drain", process=process)

    def _heartbeat_timestamp(self, path: Path, field: str, now: float) -> float | None:
        del now
        return heartbeat_timestamp(path, field)

    def _has_commit(self, generation: int) -> bool:
        marker = self.last_committed()
        return marker is not None and marker.generation >= generation

    def _child_group_matches(self, child: ActiveChild) -> bool:
        if not process_group_exists(child.process_group):
            return False
        return process_group_owned_by(child.pid, child.process_group)

    def _discard_committed_stale_child(
        self,
        child: ActiveChild | None,
        last_committed_generation: int | None,
    ) -> None:
        if child is None or last_committed_generation is None or child.generation > last_committed_generation:
            return
        if process_group_exists(child.process_group):
            if not self._can_terminate_group(child):
                raise SupervisorIntegrityError("committed stale child process group ownership mismatch")
            self._terminate_group(child, reason="child belongs to an already committed generation")
        clear_active_child(self.active_child_path)

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
            raise SupervisorIntegrityError(str(exc)) from exc

    def _write_stop(self, generation: int, attempt: int, reason: str) -> None:
        committed = self.last_committed()
        atomic_write_json(
            self.stop_path,
            {
                "schema": STOP_SCHEMA,
                "supervisor_schema": SUPERVISOR_SCHEMA,
                "lineage_id": self.lineage_id,
                "generation": generation,
                "attempt": attempt,
                "last_committed_generation": committed.generation if committed is not None else None,
                "reason": reason,
                "stopped_at": self.clock(),
            },
        )

    def _clear_runtime_identity(self) -> None:
        clear_active_child(self.active_child_path)
        self.generation_intent_path.unlink(missing_ok=True)


__all__ = [
    "ACTIVE_CHILD_SCHEMA",
    "ActiveChild",
    "CommitMarker",
    "GENERATION_INTENT_SCHEMA",
    "HeartbeatStatus",
    "LaunchRequest",
    "RecoveryPlan",
    "STOP_SCHEMA",
    "SupervisorAction",
    "SupervisorIntegrityError",
    "SupervisorPolicy",
    "SupervisorStatus",
    "SupervisorV2",
    "SupervisionResult",
    "TechnicalFailure",
]
