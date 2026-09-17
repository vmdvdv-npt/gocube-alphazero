"""Operator-selected hard-stop policy for unattended production training.

Only three runtime conditions are allowed to stop a healthy supervisor:
1. a valid periodic Arena shows the candidate losing more games than it wins;
2. critical disk/RAM exhaustion;
3. training-data / artifact integrity failure.

Everything else is advisory: emit WARNING/DEGRADED telemetry and keep the
current process or retry the failed child. This policy is intentionally
installed by the production CLI so detached supervisors inherit it too.
"""
from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Mapping, Sequence

NO_PROGRESS_ALERT_SECONDS = 30.0 * 60.0
_EFFECTIVELY_UNBOUNDED_RESTARTS = 2_147_483_647
_INSTALLED = False


def _metric(metrics: Mapping[str, object], dotted: str) -> float | None:
    current: object = metrics
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return float(current)
    return None


def _integrity_failure(exc: BaseException) -> bool:
    """Return True only for failures that make training state unsafe to trust."""
    text = str(exc).lower()
    markers = (
        "checkpoint",
        "sha256",
        " sha ",
        "hash mismatch",
        "artifact",
        "replay",
        "resume-state",
        "resume state",
        "reload verification",
        "catalog fingerprint",
        "catalog hash",
        "mutate training state",
        "training state mutated",
        "technical/invalid games",
        "invalid replay",
        "nan",
        "non-finite",
    )
    return any(marker in text for marker in markers)


def _arena_metrics(run: object, generation: int) -> Mapping[str, object] | None:
    try:
        rows = getattr(run, "_history_rows")()
    except Exception:
        return None
    for row in reversed(rows):
        if not isinstance(row, Mapping):
            continue
        if row.get("kind") != "arena" or int(row.get("generation", -1)) != generation:
            continue
        metrics = row.get("metrics")
        return metrics if isinstance(metrics, Mapping) else None
    return None


def _apply_model_degradation_gate(run: object, generation: int) -> bool:
    """Stop at the already-safe Arena boundary iff losses strictly exceed wins."""
    metrics = _arena_metrics(run, generation)
    if metrics is None:
        getattr(run, "events").emit(
            "WARNING",
            "Arena result has no readable metrics; degradation gate skipped and training continues",
            generation=generation,
            phase="arena",
        )
        return False

    wins = metrics.get("wins")
    losses = metrics.get("losses")
    draws = metrics.get("draws", 0)
    if not isinstance(wins, (int, float)) or isinstance(wins, bool):
        getattr(run, "events").emit(
            "WARNING",
            "Arena result has no numeric wins; degradation gate skipped and training continues",
            generation=generation,
            phase="arena",
        )
        return False
    if not isinstance(losses, (int, float)) or isinstance(losses, bool):
        getattr(run, "events").emit(
            "WARNING",
            "Arena result has no numeric losses; degradation gate skipped and training continues",
            generation=generation,
            phase="arena",
        )
        return False

    wins_i = int(wins)
    losses_i = int(losses)
    draws_i = int(draws) if isinstance(draws, (int, float)) and not isinstance(draws, bool) else 0
    if losses_i <= wins_i:
        return False

    getattr(run, "events").emit(
        "CRITICAL",
        (
            "Model degradation hard-stop: latest candidate lost more Arena games than it won "
            f"(M{generation} W/L/D {wins_i}/{losses_i}/{draws_i})"
        ),
        generation=generation,
        phase="arena",
        wins=wins_i,
        losses=losses_i,
        draws=draws_i,
    )
    # Arena is already a safe boundary. A durable soft-stop request prevents the
    # next generation from starting and preserves normal resumability semantics.
    getattr(run, "request_soft_stop")(
        getattr(run, "spec").soft_stop.minimum_minutes,
        reason="arena-model-degradation",
    )
    return True


def install_operator_policy() -> None:
    """Install the operator-selected warning-vs-hard-stop policy once per process."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import orchestrator as orch
    from . import production_orchestrator as prod

    original_learning_stall_checks = orch.ProductionTrainingOrchestrator._learning_stall_checks
    original_run_arena = orch.ProductionTrainingOrchestrator._run_arena
    original_run_generation = prod.UniversalProductionTrainingOrchestrator._run_generation

    def advisory_check_performance(
        self: object, generation: int, metrics: Mapping[str, object]
    ) -> None:
        checks = getattr(self, "spec").performance.get("checks")
        if not isinstance(checks, list):
            return
        for item in checks:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("metric", ""))
            value = _metric(metrics, name)
            baseline = item.get("baseline")
            if value is None or not isinstance(baseline, (int, float)) or float(baseline) <= 0:
                continue
            warning_ratio = float(item.get("warning_ratio", 0.85))
            fail_ratio = float(item.get("fail_ratio", 0.70))
            ratio = value / float(baseline)
            if ratio < warning_ratio:
                former = "fail-closed" if ratio < fail_ratio else "warning"
                getattr(self, "events").emit(
                    "WARNING",
                    (
                        f"Performance regression {name}: {value:.4g} "
                        f"({ratio:.1%} of baseline); execution continues"
                    ),
                    generation=generation,
                    phase="generation",
                    metric=name,
                    value=value,
                    ratio=ratio,
                    previous_policy=former,
                )

    def advisory_required_metrics(
        self: object, metrics: Mapping[str, object], *, kind: str
    ) -> None:
        key = "required_generation_metrics" if kind == "generation" else "required_arena_metrics"
        required = getattr(self, "spec").payload.get(key)
        if not isinstance(required, list):
            return
        missing = [str(name) for name in required if _metric(metrics, str(name)) is None]
        if missing:
            getattr(self, "events").emit(
                "WARNING",
                f"{kind.capitalize()} metrics missing; execution continues: {', '.join(missing)}",
                phase=kind,
                missing_metrics=missing,
            )

    def advisory_learning_stall_checks(self: object) -> list[dict[str, object]]:
        checks = getattr(self, "spec").learning.get("stall_checks")
        if not isinstance(checks, list):
            return []
        changed: list[tuple[dict[str, object], object]] = []
        for item in checks:
            if isinstance(item, dict):
                changed.append((item, item.get("policy")))
                item["policy"] = "warning"
        try:
            return original_learning_stall_checks(self)
        finally:
            for item, previous in changed:
                if previous is None:
                    item.pop("policy", None)
                else:
                    item["policy"] = previous

    def advisory_health(self: object, snapshot: Mapping[str, object]) -> None:
        disk = float(snapshot["disk_free_gb"])
        ram = snapshot.get("ram_free_gb")
        spec = getattr(self, "spec")

        # Resource exhaustion remains a real hard-stop.
        if disk <= spec.health.min_disk_free_gb_critical:
            raise prod.CriticalHealthError(f"disk free critically low: {disk:.2f} GiB")
        if disk <= spec.health.min_disk_free_gb_warning:
            getattr(self, "_warning")(
                "WARNING",
                f"Disk free space low: {disk:.1f} GiB; execution continues",
                disk_free_gb=disk,
            )

        if ram is not None:
            ram_value = float(ram)
            if ram_value <= spec.health.min_ram_free_gb_critical:
                raise prod.CriticalHealthError(
                    f"RAM available critically low: {ram_value:.2f} GiB"
                )
            if ram_value <= spec.health.min_ram_free_gb_warning:
                getattr(self, "_warning")(
                    "WARNING",
                    f"RAM available low: {ram_value:.1f} GiB; execution continues",
                    ram_free_gb=ram_value,
                )

        liveness_age = snapshot.get("driver_liveness_age_sec")
        if isinstance(liveness_age, (int, float)) and not isinstance(liveness_age, bool):
            age = float(liveness_age)
            if age >= spec.health.heartbeat_warning_seconds:
                getattr(self, "_warning")(
                    "WARNING",
                    "Driver heartbeat is stale; execution continues",
                    liveness_age_seconds=age,
                )

        progress_age = snapshot.get("driver_progress_age_sec")
        if isinstance(progress_age, (int, float)) and not isinstance(progress_age, bool):
            age = float(progress_age)
            if age >= NO_PROGRESS_ALERT_SECONDS:
                token = str(snapshot.get("driver_progress_token") or "unknown")
                getattr(self, "_warning")(
                    "WARNING",
                    (
                        "No semantic progress for 30 minutes; execution continues "
                        f"(last progress: {token})"
                    ),
                    progress_age_seconds=age,
                    progress_token=token,
                )

        driver = snapshot.get("driver_health")
        if isinstance(driver, Mapping):
            if driver.get("schema") not in (None, prod.DRIVER_HEARTBEAT_SCHEMA):
                getattr(self, "_warning")(
                    "WARNING",
                    "Driver heartbeat schema mismatch; execution continues",
                )
            expected = driver.get("workers_expected")
            alive = driver.get("workers_alive")
            if isinstance(expected, int) and isinstance(alive, int) and alive < expected:
                getattr(self, "_warning")(
                    "WARNING",
                    f"Worker health degraded: {alive}/{expected} alive; execution continues",
                    workers_expected=expected,
                    workers_alive=alive,
                )
            if driver.get("inference_alive") is False:
                getattr(self, "_warning")(
                    "WARNING",
                    "Inference owner reports not alive; execution continues",
                )
            if driver.get("gpu_stuck") is True:
                getattr(self, "_warning")(
                    "WARNING",
                    "Driver reports stuck GPU progress; execution continues",
                )
            errors = driver.get("errors")
            if isinstance(errors, list) and errors:
                getattr(self, "_warning")(
                    "WARNING",
                    f"Driver reported runtime error; execution continues: {errors[-1]}",
                )

    def advisory_run_child(
        self: object,
        command: Sequence[str],
        *,
        generation: int,
        resume: bool,
        phase: str,
    ) -> int:
        from .orchestrator import _render_command, atomic_write_json, parse_utc, utc_now
        import subprocess

        rendered = _render_command(
            command,
            generation=generation,
            generation04=f"{generation:04d}",
            lineage_id=getattr(self, "lineage_id"),
            run_root=str(getattr(self, "paths").root),
            profile_path=str(getattr(self, "spec").profile_path),
        )
        paths = getattr(self, "paths")
        paths.driver_heartbeat.unlink(missing_ok=True)
        getattr(self, "events").emit(
            "INFO", f"Starting {phase}", generation=generation, argv=rendered
        )
        process = subprocess.Popen(
            rendered,
            cwd=getattr(self, "repo_root"),
            env=getattr(self, "_driver_env")(generation, resume=resume, phase=phase),
            start_new_session=True,
        )
        atomic_write_json(
            getattr(self, "active_child_path"),
            {
                "schema": prod.ACTIVE_CHILD_SCHEMA,
                "pid": process.pid,
                "process_group": process.pid,
                "generation": generation,
                "phase": phase,
                "started_at": utc_now(),
                "argv": rendered,
            },
        )
        started = time.monotonic()
        startup_warning_sent = False
        try:
            while True:
                code = process.poll()
                if code is not None:
                    return int(code)
                state = getattr(self, "_state")()
                getattr(self, "_heartbeat")(state)
                snapshot = getattr(self, "_health_snapshot")(process)
                atomic_write_json(paths.metrics / "health-latest.json", snapshot)
                try:
                    # Resource checks apply even before the driver publishes its first heartbeat.
                    getattr(self, "_emit_health_warnings")(snapshot)
                except prod.CriticalHealthError as exc:
                    # Only disk/RAM exhaustion can reach this branch under this policy.
                    getattr(self, "_terminate_child_group")(process, reason=str(exc))
                    raise

                if not paths.driver_heartbeat.is_file():
                    elapsed = time.monotonic() - started
                    if (
                        elapsed >= getattr(self, "supervision").startup_ack_timeout_seconds
                        and not startup_warning_sent
                    ):
                        startup_warning_sent = True
                        getattr(self, "_warning")(
                            "WARNING",
                            "Driver startup heartbeat missing; execution continues",
                            generation=generation,
                            phase=phase,
                            elapsed_seconds=elapsed,
                        )

                stop = getattr(self, "_stop_request")()
                if stop is not None:
                    target = parse_utc(str(stop["target_deadline_at"]))
                    if datetime.now(timezone.utc) > target:
                        getattr(self, "_warning")(
                            "WARNING",
                            "Soft-stop target window exceeded; active safe unit is still allowed to finish",
                            generation=generation,
                            phase=phase,
                        )
                time.sleep(getattr(self, "spec").health.poll_seconds)
        finally:
            try:
                if process.poll() is None or getattr(self, "_process_group_exists")(process.pid):
                    getattr(self, "_terminate_child_group")(
                        process,
                        reason="supervisor monitor exited before child process group was fully reaped",
                    )
            finally:
                getattr(self, "active_child_path").unlink(missing_ok=True)

    def retry_generation_without_operational_stop(self: object, generation: int) -> None:
        supervision = getattr(self, "supervision")
        previous_limit = supervision.max_generation_restarts
        object.__setattr__(
            supervision,
            "max_generation_restarts",
            max(previous_limit, _EFFECTIVELY_UNBOUNDED_RESTARTS),
        )
        try:
            while True:
                try:
                    original_run_generation(self, generation)
                    return
                except prod.CriticalHealthError:
                    # Critical disk/RAM exhaustion is an explicit hard-stop.
                    raise
                except Exception as exc:
                    if _integrity_failure(exc):
                        # Corrupt/untrustworthy training state is an explicit hard-stop.
                        raise
                    getattr(self, "events").emit(
                        "WARNING",
                        (
                            "Generation failed operationally; retrying without stopping lineage: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        generation=generation,
                        phase="generation",
                    )
                    delay = float(getattr(self, "supervision").restart_backoff_seconds)
                    if delay > 0:
                        time.sleep(delay)
        finally:
            object.__setattr__(supervision, "max_generation_restarts", previous_limit)

    def arena_with_operator_policy(self: object, generation: int) -> None:
        try:
            original_run_arena(self, generation)
        except prod.CriticalHealthError:
            # Critical disk/RAM exhaustion is an explicit hard-stop.
            raise
        except Exception as exc:
            if _integrity_failure(exc):
                raise
            getattr(self, "events").emit(
                "WARNING",
                f"Arena failed operationally; training continues: {type(exc).__name__}: {exc}",
                generation=generation,
                phase="arena",
            )
            return
        _apply_model_degradation_gate(self, generation)

    orch.ProductionTrainingOrchestrator._check_performance = advisory_check_performance
    orch.ProductionTrainingOrchestrator._check_required_metrics = advisory_required_metrics
    orch.ProductionTrainingOrchestrator._learning_stall_checks = advisory_learning_stall_checks
    orch.ProductionTrainingOrchestrator._run_arena = arena_with_operator_policy
    prod.UniversalProductionTrainingOrchestrator._emit_health_warnings = advisory_health
    prod.UniversalProductionTrainingOrchestrator._run_child = advisory_run_child
    prod.UniversalProductionTrainingOrchestrator._run_generation = retry_generation_without_operational_stop
    _INSTALLED = True


__all__ = [
    "NO_PROGRESS_ALERT_SECONDS",
    "install_operator_policy",
    "_apply_model_degradation_gate",
    "_integrity_failure",
]
