"""Universal production supervision for long-running AlphaZero training.

This module is intentionally game-agnostic.  Scientific rules, model/search
semantics and topology-specific persistence remain in profile adapters.  The
supervisor owns only lifecycle, durability, progress/health monitoring,
bounded recovery, soft-stop behaviour and operator-facing status/reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence

from .orchestrator import (
    ProductionTrainingOrchestrator,
    atomic_write_json,
    atomic_write_text,
    parse_utc,
    read_json,
    utc_now,
)


DRIVER_HEARTBEAT_SCHEMA = "gocube-training-driver-heartbeat-v2"
ACTIVE_CHILD_SCHEMA = "gocube-training-active-child-v1"


class CriticalHealthError(RuntimeError):
    """Fail-closed runtime health condition for the active child process."""


@dataclass(frozen=True)
class SupervisionPolicy:
    """Explicit run-owned runtime supervision policy.

    No values are selected implicitly in production.  The immutable run spec
    must provide every field.
    """

    startup_ack_timeout_seconds: float
    progress_warning_seconds: float
    progress_critical_seconds: float
    critical_child_grace_seconds: float
    restart_backoff_seconds: float
    max_generation_restarts: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SupervisionPolicy":
        required = tuple(cls.__dataclass_fields__.keys())
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(
                "supervision is missing explicit fields: " + ", ".join(missing)
            )
        result = cls(
            startup_ack_timeout_seconds=float(value["startup_ack_timeout_seconds"]),
            progress_warning_seconds=float(value["progress_warning_seconds"]),
            progress_critical_seconds=float(value["progress_critical_seconds"]),
            critical_child_grace_seconds=float(value["critical_child_grace_seconds"]),
            restart_backoff_seconds=float(value["restart_backoff_seconds"]),
            max_generation_restarts=int(value["max_generation_restarts"]),
        )
        if result.startup_ack_timeout_seconds <= 0:
            raise ValueError("supervision.startup_ack_timeout_seconds must be positive")
        if not 0 < result.progress_warning_seconds < result.progress_critical_seconds:
            raise ValueError("supervision progress warning must be below critical")
        if result.critical_child_grace_seconds < 0:
            raise ValueError("supervision.critical_child_grace_seconds must be non-negative")
        if result.restart_backoff_seconds < 0:
            raise ValueError("supervision.restart_backoff_seconds must be non-negative")
        if result.max_generation_restarts < 0:
            raise ValueError("supervision.max_generation_restarts must be non-negative")
        return result


def _timestamp_age_seconds(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, time.time() - float(value))
        parsed = parse_utc(str(value))
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _metric_number(payload: Mapping[str, object], dotted: str) -> float | None:
    current: object = payload
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return float(current)
    return None


class UniversalProductionTrainingOrchestrator(ProductionTrainingOrchestrator):
    """Game-independent production supervisor with autonomous safety gates."""

    def __init__(self, *args: object, supervision: SupervisionPolicy, **kwargs: object) -> None:
        self.supervision = supervision
        self._last_warning_key: tuple[str, str] | None = None
        self._last_warning_at = 0.0
        super().__init__(*args, **kwargs)

    @property
    def active_child_path(self) -> Path:
        return self.paths.runtime / "active-child.json"

    def _driver_env(self, generation: int, *, resume: bool, phase: str) -> dict[str, str]:
        env = super()._driver_env(generation, resume=resume, phase=phase)
        env.update(
            {
                "AZ_DRIVER_HEARTBEAT_SCHEMA": DRIVER_HEARTBEAT_SCHEMA,
                "AZ_LINEAGE_ARENA_ROOT": str(self.paths.root / "arena"),
                "AZ_STORAGE_SCOPE": "lineage-owned",
            }
        )
        return env

    def _health_snapshot(self, child: subprocess.Popen[bytes] | subprocess.Popen[str]) -> dict[str, object]:
        snapshot = super()._health_snapshot(child)
        driver = snapshot.get("driver_health")
        if isinstance(driver, Mapping):
            snapshot["driver_liveness_age_sec"] = _timestamp_age_seconds(
                driver.get("liveness_at", driver.get("at"))
            )
            snapshot["driver_progress_age_sec"] = _timestamp_age_seconds(
                driver.get("progress_at")
            )
            snapshot["driver_progress_token"] = driver.get("progress_token")
            snapshot["driver_progress"] = driver.get("progress")
        else:
            snapshot["driver_liveness_age_sec"] = None
            snapshot["driver_progress_age_sec"] = None
            snapshot["driver_progress_token"] = None
            snapshot["driver_progress"] = None
        return snapshot

    def _warning(self, level: str, message: str, **details: object) -> None:
        # Avoid printing the same unattended warning on every polling tick.
        key = (level, message)
        now = time.monotonic()
        if self._last_warning_key == key and now - self._last_warning_at < 60.0:
            return
        self._last_warning_key = key
        self._last_warning_at = now
        self.events.emit(level, message, **details)

    def _emit_health_warnings(self, snapshot: Mapping[str, object]) -> None:
        disk = float(snapshot["disk_free_gb"])
        ram = snapshot.get("ram_free_gb")
        liveness_age = snapshot.get("driver_liveness_age_sec")
        progress_age = snapshot.get("driver_progress_age_sec")
        driver = snapshot.get("driver_health")

        if disk <= self.spec.health.min_disk_free_gb_critical:
            raise CriticalHealthError(f"disk free critically low: {disk:.2f} GiB")
        if disk <= self.spec.health.min_disk_free_gb_warning:
            self._warning("WARNING", f"Disk free space low: {disk:.2f} GiB")

        if ram is not None:
            ram_value = float(ram)
            if ram_value <= self.spec.health.min_ram_free_gb_critical:
                raise CriticalHealthError(
                    f"RAM available critically low: {ram_value:.2f} GiB"
                )
            if ram_value <= self.spec.health.min_ram_free_gb_warning:
                self._warning("WARNING", f"RAM available low: {ram_value:.2f} GiB")

        if liveness_age is not None:
            age = float(liveness_age)
            if age >= self.spec.health.heartbeat_critical_seconds:
                raise CriticalHealthError(f"driver liveness heartbeat stale for {age:.0f}s")
            if age >= self.spec.health.heartbeat_warning_seconds:
                self._warning("WARNING", f"Driver liveness heartbeat stale for {age:.0f}s")

        if progress_age is not None:
            age = float(progress_age)
            if age >= self.supervision.progress_critical_seconds:
                raise CriticalHealthError(
                    f"driver made no observable progress for {age:.0f}s"
                )
            if age >= self.supervision.progress_warning_seconds:
                self._warning(
                    "WARNING",
                    f"Driver progress has not advanced for {age:.0f}s",
                    progress_token=snapshot.get("driver_progress_token"),
                )

        if isinstance(driver, Mapping):
            if driver.get("schema") not in (None, DRIVER_HEARTBEAT_SCHEMA):
                raise CriticalHealthError("driver heartbeat schema mismatch")
            expected = driver.get("workers_expected")
            alive = driver.get("workers_alive")
            if isinstance(expected, int) and isinstance(alive, int) and alive < expected:
                raise CriticalHealthError(f"worker health degraded: {alive}/{expected} alive")
            if driver.get("inference_alive") is False:
                raise CriticalHealthError("inference owner reports not alive")
            if driver.get("gpu_stuck") is True:
                raise CriticalHealthError("driver reports stuck GPU progress")
            errors = driver.get("errors")
            if isinstance(errors, list) and errors:
                raise CriticalHealthError(f"driver reported runtime error: {errors[-1]}")

    def _terminate_child_group(self, process: subprocess.Popen[bytes] | subprocess.Popen[str], *, reason: str) -> None:
        if process.poll() is not None:
            return
        self.events.emit(
            "CRITICAL",
            "Fail-closed termination of active run child process group",
            pid=process.pid,
            reason=reason,
        )
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=self.supervision.critical_child_grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"child process group {process.pid} survived scoped fail-closed termination"
            ) from exc

    def _run_child(self, command: Sequence[str], *, generation: int, resume: bool, phase: str) -> int:
        from .orchestrator import _render_command

        rendered = _render_command(
            command,
            generation=generation,
            generation04=f"{generation:04d}",
            lineage_id=self.lineage_id,
            run_root=str(self.paths.root),
            profile_path=str(self.spec.profile_path),
        )
        self.paths.driver_heartbeat.unlink(missing_ok=True)
        self.events.emit("INFO", f"Starting {phase}", generation=generation, argv=rendered)
        process = subprocess.Popen(
            rendered,
            cwd=self.repo_root,
            env=self._driver_env(generation, resume=resume, phase=phase),
            start_new_session=True,
        )
        atomic_write_json(
            self.active_child_path,
            {
                "schema": ACTIVE_CHILD_SCHEMA,
                "pid": process.pid,
                "process_group": process.pid,
                "generation": generation,
                "phase": phase,
                "started_at": utc_now(),
                "argv": rendered,
            },
        )
        started = time.monotonic()
        try:
            while True:
                code = process.poll()
                if code is not None:
                    return int(code)
                state = self._state()
                self._heartbeat(state)
                snapshot = self._health_snapshot(process)
                atomic_write_json(self.paths.metrics / "health-latest.json", snapshot)
                if not self.paths.driver_heartbeat.is_file():
                    if time.monotonic() - started >= self.supervision.startup_ack_timeout_seconds:
                        reason = "driver did not publish startup heartbeat before timeout"
                        self._terminate_child_group(process, reason=reason)
                        raise CriticalHealthError(reason)
                else:
                    try:
                        self._emit_health_warnings(snapshot)
                    except CriticalHealthError as exc:
                        self._terminate_child_group(process, reason=str(exc))
                        raise
                stop = self._stop_request()
                if stop is not None:
                    target = parse_utc(str(stop["target_deadline_at"]))
                    if datetime.now(timezone.utc) > target:
                        self._warning(
                            "WARNING",
                            "Soft-stop target window exceeded; active safe unit is still allowed to finish",
                            generation=generation,
                            phase=phase,
                        )
                time.sleep(self.spec.health.poll_seconds)
        finally:
            self.active_child_path.unlink(missing_ok=True)

    def _run_generation(self, generation: int) -> None:
        tx_path = self._generation_tx_path(generation)
        resume = False
        command = self.spec.generation_command
        previous_attempts = 0
        if tx_path.is_file():
            tx_existing = read_json(tx_path)
            status = tx_existing.get("status")
            previous_attempts = int(tx_existing.get("restart_attempts", 0))
            if status == "COMMITTED":
                raise RuntimeError(f"Generation {generation} is already committed")
            if status in {"RUNNING", "FAILED"}:
                if self.spec.generation_resume_command is None:
                    raise RuntimeError(
                        f"Generation {generation} was interrupted and no fail-closed resume command is configured"
                    )
                resume = True
                command = self.spec.generation_resume_command
                self.events.emit("WARNING", "Resuming interrupted generation", generation=generation)

        restarts = previous_attempts
        while True:
            tx: dict[str, object] = {
                "schema": "gocube-production-training-orchestrator-v1",
                "generation": generation,
                "status": "RUNNING",
                "started_at": utc_now(),
                "resume": resume,
                "restart_attempts": restarts,
                "profile_fingerprint": self.spec.profile_fingerprint,
            }
            atomic_write_json(tx_path, tx)
            self._write_state(active_generation=generation, active_phase="generation")
            code = self._run_child(command, generation=generation, resume=resume, phase="generation")
            if code == 0:
                break
            tx.update({"status": "FAILED", "finished_at": utc_now(), "exit_code": code})
            atomic_write_json(tx_path, tx)
            if (
                self.spec.generation_resume_command is not None
                and restarts < self.supervision.max_generation_restarts
            ):
                restarts += 1
                resume = True
                command = self.spec.generation_resume_command
                self.events.emit(
                    "WARNING",
                    "Generation child exited non-zero; bounded automatic resume scheduled",
                    generation=generation,
                    exit_code=code,
                    restart_attempt=restarts,
                )
                if self.supervision.restart_backoff_seconds:
                    time.sleep(self.supervision.restart_backoff_seconds)
                continue
            raise RuntimeError(f"Generation driver exited with code {code}")

        result = self._validate_generation_result(generation)
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            self._check_required_metrics(metrics, kind="generation")
            self._append_metrics("generation", generation, metrics)
        tx.update(
            {
                "status": "COMMITTED",
                "finished_at": utc_now(),
                "exit_code": 0,
                "restart_attempts": restarts,
                "result_path": str(
                    self._generation_result_path(generation).relative_to(self.paths.root)
                ),
                "artifact_hashes": result["validated_artifact_hashes"],
            }
        )
        atomic_write_json(tx_path, tx)
        self._write_state(last_committed_generation=generation, active_phase="commit")
        self._update_manifest(committed_generation=generation)
        self.events.emit("INFO", "Generation committed", generation=generation)
        if isinstance(metrics, Mapping):
            self._check_performance(generation, metrics)
        self._learning_stall_checks()

        # A soft-stop request means: finish the current safe unit, then do not
        # start another unit.  Arena remains pending and will run first on the
        # next explicit resume before more training starts.
        if self._stop_request() is None and generation % self.spec.arena_every_generations == 0:
            self._run_arena(generation)
        elif self._stop_request() is not None and generation % self.spec.arena_every_generations == 0:
            self.events.emit(
                "INFO",
                "Periodic Arena left pending because soft-stop was requested",
                generation=generation,
            )
        self._render_report()

    def _render_report(self, *, final: bool = False) -> None:
        super()._render_report(final=final)
        if not final or not self.paths.final_json.is_file():
            return
        payload = read_json(self.paths.final_json)
        state = self._state()
        manifest = self._load_manifest()
        orchestrator = manifest.get("orchestrator")
        events = self._event_rows()
        payload["final_reason"] = state.get("error") or state.get("state")
        payload["supervision"] = {
            key: getattr(self.supervision, key)
            for key in self.supervision.__dataclass_fields__
        }
        payload["automatic_generation_restarts"] = [
            event
            for event in events
            if event.get("message")
            == "Generation child exited non-zero; bounded automatic resume scheduled"
        ]
        payload["arena_generations"] = (
            list(orchestrator.get("arena_generations", []))
            if isinstance(orchestrator, Mapping)
            else []
        )
        health_path = self.paths.metrics / "health-latest.json"
        payload["last_health_snapshot"] = (
            read_json(health_path) if health_path.is_file() else None
        )
        atomic_write_json(self.paths.final_json, payload)
        with self.paths.final_md.open("a", encoding="utf-8") as handle:
            handle.write("\n## Supervision\n\n")
            handle.write(f"- Final reason: `{payload['final_reason']}`\n")
            handle.write(
                f"- Automatic generation restarts: **{len(payload['automatic_generation_restarts'])}**\n"
            )
            handle.write(f"- Arena generations: `{payload['arena_generations']}`\n")

    def status(self) -> dict[str, object]:
        status = super().status()
        state_name = str(status.get("state"))
        health_path = self.paths.metrics / "health-latest.json"
        health_snapshot = read_json(health_path) if health_path.is_file() else {}
        progress_age = health_snapshot.get("driver_progress_age_sec")
        liveness_age = health_snapshot.get("driver_liveness_age_sec")
        classification = "HEALTHY"
        if state_name == "RECOVERY_REQUIRED":
            classification = "CRITICAL"
        elif (
            isinstance(progress_age, (int, float))
            and float(progress_age) >= self.supervision.progress_critical_seconds
        ) or (
            isinstance(liveness_age, (int, float))
            and float(liveness_age) >= self.spec.health.heartbeat_critical_seconds
        ):
            classification = "CRITICAL"
        elif (
            isinstance(progress_age, (int, float))
            and float(progress_age) >= self.supervision.progress_warning_seconds
        ) or (
            isinstance(liveness_age, (int, float))
            and float(liveness_age) >= self.spec.health.heartbeat_warning_seconds
        ):
            classification = "WARNING"

        manifest = self._load_manifest()
        orchestrator = manifest.get("orchestrator")
        arena_generations = (
            [int(value) for value in orchestrator.get("arena_generations", [])]
            if isinstance(orchestrator, Mapping)
            else []
        )
        history = self._history_rows()
        generation_rows = [row for row in history if row.get("kind") == "generation"]
        latest_metrics = (
            generation_rows[-1].get("metrics")
            if generation_rows and isinstance(generation_rows[-1].get("metrics"), Mapping)
            else {}
        )
        speed: dict[str, float] = {}
        if isinstance(latest_metrics, Mapping):
            for name in ("games_per_hour", "moves_per_sec", "optimizer_updates_per_sec"):
                value = _metric_number(latest_metrics, name)
                if value is not None:
                    speed[name] = value

        stop = status.get("stop_request")
        safe_stop_eta_seconds: float | None = None
        if isinstance(stop, Mapping) and stop.get("target_deadline_at"):
            safe_stop_eta_seconds = max(
                0.0,
                (parse_utc(str(stop["target_deadline_at"])) - datetime.now(timezone.utc)).total_seconds(),
            )
        driver = health_snapshot.get("driver_health")
        status.update(
            {
                "health": classification,
                "progress": (
                    driver.get("progress") if isinstance(driver, Mapping) else None
                ),
                "progress_token": health_snapshot.get("driver_progress_token"),
                "progress_age_sec": progress_age,
                "driver_liveness_age_sec": liveness_age,
                "speed": speed,
                "last_arena_generation": arena_generations[-1] if arena_generations else None,
                "arena_generations": arena_generations,
                "safe_stop_eta_seconds": safe_stop_eta_seconds,
                "active_child": (
                    read_json(self.active_child_path)
                    if self.active_child_path.is_file()
                    else None
                ),
            }
        )
        return status


def format_production_status(status: Mapping[str, object]) -> str:
    lines = [
        f"Lineage: {status.get('lineage_id')}",
        f"Topology: {status.get('topology')}",
        f"State: {status.get('state')} / health {status.get('health', '-')}",
        f"Generation: {status.get('generation') or '-'} (committed {status.get('last_committed_generation')})",
        f"Phase: {status.get('phase') or '-'}",
        f"PID: {status.get('pid') or '-'}",
    ]
    progress = status.get("progress")
    if isinstance(progress, Mapping):
        completed = progress.get("completed")
        total = progress.get("total")
        unit = progress.get("unit", "units")
        lines.append(f"Progress: {completed}/{total} {unit}")
    elif status.get("progress_token") is not None:
        lines.append(f"Progress: {status.get('progress_token')}")
    progress_age = status.get("progress_age_sec")
    if isinstance(progress_age, (int, float)):
        lines.append(f"Progress age: {float(progress_age):.1f}s")
    speed = status.get("speed")
    if isinstance(speed, Mapping) and speed:
        rendered = ", ".join(
            f"{key}={float(value):.3g}" for key, value in speed.items()
        )
        lines.append(f"Speed: {rendered}")
    lines.append(f"Last Arena: {status.get('last_arena_generation') or '-'}")
    stop = status.get("stop_request")
    if isinstance(stop, Mapping):
        eta = status.get("safe_stop_eta_seconds")
        eta_text = f", target in {float(eta) / 60.0:.1f} min" if isinstance(eta, (int, float)) else ""
        lines.append(f"Soft stop: requested{eta_text}")
    else:
        lines.append("Soft stop: no")
    warning = status.get("latest_warning")
    if isinstance(warning, Mapping):
        lines.append(f"Latest warning: {warning.get('at')} {warning.get('message')}")
    lines.append(f"Checkpoints: {status.get('checkpoint_count')}")
    lines.append(f"Report: {status.get('report')}")
    return "\n".join(lines)


__all__ = [
    "ACTIVE_CHILD_SCHEMA",
    "CriticalHealthError",
    "DRIVER_HEARTBEAT_SCHEMA",
    "SupervisionPolicy",
    "UniversalProductionTrainingOrchestrator",
    "format_production_status",
]
