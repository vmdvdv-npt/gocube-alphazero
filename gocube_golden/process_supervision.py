"""Shared durable process-supervision primitives.

This module contains mechanics that must have one implementation while V1 is
being retired: atomic JSON persistence, owned child sessions/process groups,
active-child records, and heartbeat timestamp parsing.  Policy and lifecycle
decisions stay in the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


_REAL_POPEN_TYPE = subprocess.Popen


class ProcessOwnershipError(RuntimeError):
    """A child did not create or no longer owns its recorded process group."""


def _fsync_dir(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, payload: Mapping[str, object] | Sequence[object]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_active_child(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one active-child identity atomically."""
    atomic_write_json(path, payload)


def read_active_child(path: Path) -> dict[str, object] | None:
    """Read an active-child identity, or return None when no child is active."""
    return read_json(path) if path.is_file() else None


def clear_active_child(path: Path) -> None:
    path.unlink(missing_ok=True)


def timestamp_seconds(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    raise ValueError("timestamp is required")


def timestamp_age_seconds(value: object, *, now: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, (time.time() if now is None else float(now)) - timestamp_seconds(value))
    except (TypeError, ValueError, OverflowError):
        return None


def heartbeat_timestamp(path: Path, field: str) -> float | None:
    """Read one liveness/progress timestamp, with mtime fallback for raw files."""
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        try:
            return path.stat().st_mtime
        except OSError:
            return None
    raw = payload.get(field)
    if raw is None and field == "progress_at":
        raw = payload.get("progressed_at")
    if raw is None and not (field == "progress_at" and "liveness_at" in payload):
        raw = payload.get("at")
    try:
        return timestamp_seconds(raw) if raw is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def heartbeat_age_seconds(path: Path, field: str, *, now: float | None = None) -> float | None:
    return timestamp_age_seconds(heartbeat_timestamp(path, field), now=now)


def process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(int(process_group), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_group_owned_by(pid: int, process_group: int) -> bool:
    if int(process_group) <= 1 or int(process_group) == os.getpgrp():
        return False
    try:
        return os.getpgid(int(pid)) == int(process_group)
    except (ProcessLookupError, PermissionError):
        # The leader can exit while descendants still own the recorded group.
        return True


def wait_for_process_group_exit(
    process_group: int,
    timeout: float,
    *,
    sleeper: Callable[[float], None] = time.sleep,
    exists: Callable[[int], bool] = process_group_exists,
) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout))
    while exists(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        sleeper(min(0.01, remaining))
    return True


def _terminate_direct_process(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    timeout: float,
) -> None:
    try:
        process.terminate()
        process.wait(timeout=max(0.0, timeout))
    except (OSError, subprocess.TimeoutExpired):
        pass


def process_group_for(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> int:
    try:
        return os.getpgid(int(process.pid))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise ProcessOwnershipError("child process group cannot be inspected") from exc


def start_owned_child(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[bytes] | subprocess.Popen[str]] = subprocess.Popen,
    ownership_timeout: float = 5.0,
) -> subprocess.Popen[bytes] | subprocess.Popen[str]:
    """Start a child in a private session and verify its process-group owner."""
    process = popen(
        argv,
        cwd=cwd,
        env=None if env is None else dict(env),
        start_new_session=True,
    )
    try:
        process_group = process_group_for(process)
    except ProcessOwnershipError:
        # Lightweight fake processes used by unit tests have no OS PID.  Real
        # subprocess.Popen objects must always be inspectable.
        if isinstance(process, _REAL_POPEN_TYPE):
            _terminate_direct_process(process, ownership_timeout)
            raise
        return process
    if process_group != int(process.pid):
        _terminate_direct_process(process, ownership_timeout)
        raise ProcessOwnershipError("child did not create a private process group")
    return process


def terminate_process_group(
    process: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None,
    *,
    owner_pid: int | None = None,
    process_group: int | None = None,
    term_timeout: float,
    kill_timeout: float,
    reason: str,
    sleeper: Callable[[float], None] = time.sleep,
    exists: Callable[[int], bool] = process_group_exists,
) -> None:
    """Terminate exactly one verified process group, then fail closed."""
    if process is not None:
        owner_pid = int(process.pid)
    if owner_pid is None and process_group is None:
        raise ProcessOwnershipError("process-group owner is required")
    if process_group is None:
        assert owner_pid is not None
        process_group = int(owner_pid)
    process_group = int(process_group)
    if process_group <= 1 or process_group == os.getpgrp():
        raise ProcessOwnershipError("refusing to terminate an unscoped process group")
    if owner_pid is not None and not process_group_owned_by(owner_pid, process_group):
        raise ProcessOwnershipError("recorded process group is not owned by its child")
    if not exists(process_group):
        if process is not None:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
        return

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process is not None and process.poll() is None:
        try:
            process.wait(timeout=max(0.0, term_timeout))
        except subprocess.TimeoutExpired:
            pass
    if wait_for_process_group_exit(process_group, term_timeout, sleeper=sleeper, exists=exists):
        return

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process is not None and process.poll() is None:
        try:
            process.wait(timeout=max(0.0, kill_timeout))
        except subprocess.TimeoutExpired as exc:
            raise ProcessOwnershipError(
                f"child process {process.pid} survived scoped termination: {reason}"
            ) from exc
    if not wait_for_process_group_exit(process_group, kill_timeout, sleeper=sleeper, exists=exists):
        raise ProcessOwnershipError(
            f"child process group {process_group} survived scoped termination: {reason}"
        )


__all__ = [
    "ProcessOwnershipError",
    "atomic_write_json",
    "atomic_write_text",
    "clear_active_child",
    "heartbeat_age_seconds",
    "heartbeat_timestamp",
    "process_group_exists",
    "process_group_for",
    "process_group_owned_by",
    "read_active_child",
    "read_json",
    "start_owned_child",
    "terminate_process_group",
    "timestamp_age_seconds",
    "timestamp_seconds",
    "wait_for_process_group_exit",
    "write_active_child",
]
