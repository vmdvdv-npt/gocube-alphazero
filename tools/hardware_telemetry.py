#!/usr/bin/env python3
"""Low-overhead system telemetry for long GoCube experiments.

The sampler is intentionally independent of the training process. It observes
system CPU/RAM and NVIDIA GPU counters so an orchestration process can tag
SELFPLAY/TRAIN/ARENA/BENCHMARK phases without modifying search or training
semantics.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator
import shutil


WSL_NVIDIA_SMI_PATH = "/usr/lib/wsl/lib/nvidia-smi"


def _read_cpu_ticks() -> tuple[int, int] | None:
    try:
        line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    fields = line.split()
    if not fields or fields[0] != "cpu":
        return None
    values = [int(value) for value in fields[1:]]
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def _read_memory() -> dict[str, float]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            token = raw.strip().split()[0]
            values[key] = int(token)
    except (OSError, ValueError, IndexError):
        return {}
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return {}
    used = max(0, total - available)
    return {
        "ram_used_gib": used / (1024.0 * 1024.0),
        "ram_total_gib": total / (1024.0 * 1024.0),
        "ram_used_percent": 100.0 * used / total,
    }


def resolve_nvidia_smi() -> tuple[str | None, str]:
    """Resolve nvidia-smi using the production WSL precedence order.

    An explicit override is authoritative: if it is present but unusable we
    report that fact instead of silently falling through to another binary.
    """
    override = os.environ.get("GOCUBE_NVIDIA_SMI")
    if override is not None:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate), "env_override"
        return None, "override_missing_or_not_executable"
    path = shutil.which("nvidia-smi")
    if path:
        return path, "path"
    candidate = Path(WSL_NVIDIA_SMI_PATH)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate), "wsl_fallback"
    return None, "not_found"


def _query_nvidia_smi() -> tuple[list[dict[str, object]], dict[str, object]]:
    command_path, resolution = resolve_nvidia_smi()
    if command_path is None:
        status = "nvidia_smi_missing"
        if resolution == "override_missing_or_not_executable":
            status = "override_unavailable"
        return [], {
            "status": status,
            "resolver": resolution,
            "path": None,
            "error": f"nvidia-smi resolver status: {resolution}",
        }
    command = [
        command_path,
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.free,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], {
            "status": "command_failed",
            "resolver": resolution,
            "path": command_path,
            "error": str(exc),
        }
    if result.returncode != 0:
        stderr = str(result.stderr or "").strip()
        return [], {
            "status": "command_failed",
            "resolver": resolution,
            "path": command_path,
            "error": stderr or f"nvidia-smi exited with {result.returncode}",
        }
    rows: list[dict[str, object]] = []
    malformed = 0
    for raw in result.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if not raw.strip():
            continue
        if len(parts) != 8:
            malformed += 1
            continue
        try:
            index = int(parts[0])
            name = parts[1]
            util = float(parts[2])
            used = float(parts[3])
            free = float(parts[4])
            total = float(parts[5])
            temp = float(parts[6])
            power = float(parts[7])
        except ValueError:
            malformed += 1
            continue
        rows.append(
            {
                "index": index,
                "name": name,
                "gpu_util_percent": util,
                "gpu_memory_used_mib": used,
                "gpu_memory_free_mib": free,
                "gpu_memory_total_mib": total,
                "gpu_temperature_c": temp,
                "gpu_power_w": power,
            }
        )
    if not rows:
        status = "parse_failed" if malformed else "gpu_not_found"
    else:
        status = "parse_failed" if malformed else "ok"
    return rows, {
        "status": status,
        "resolver": resolution,
        "path": command_path,
        "error": f"ignored {malformed} malformed nvidia-smi row(s)" if malformed else None,
    }


def _read_nvidia_smi() -> list[dict[str, object]]:
    """Compatibility wrapper returning only the parsed GPU rows."""
    rows, _status = _query_nvidia_smi()
    return rows


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


class HardwareTelemetry:
    """Background JSONL sampler with phase tagging."""

    def __init__(self, path: str | os.PathLike[str], interval_s: float = 1.0):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.path = Path(path)
        self.interval_s = float(interval_s)
        self._phase = "IDLE"
        self._phase_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous_cpu: tuple[int, int] | None = None

    @property
    def phase_name(self) -> str:
        with self._phase_lock:
            return self._phase

    def set_phase(self, phase: str) -> None:
        normalized = str(phase).strip().upper()
        if not normalized:
            raise ValueError("phase must not be empty")
        with self._phase_lock:
            self._phase = normalized

    def start(self) -> "HardwareTelemetry":
        if self._thread is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gocube-hw-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 3.0))
        self._thread = None

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self.phase_name
        self.set_phase(name)
        self.start()
        try:
            yield
        finally:
            self.set_phase(previous)

    def _sample(self) -> dict[str, object]:
        sample: dict[str, object] = {
            "time": time.time(),
            "phase": self.phase_name,
            "loadavg_1m": None,
            "loadavg_5m": None,
            "loadavg_15m": None,
        }
        try:
            load = os.getloadavg()
            sample["loadavg_1m"], sample["loadavg_5m"], sample["loadavg_15m"] = load
        except (AttributeError, OSError):
            pass

        current = _read_cpu_ticks()
        if current is not None and self._previous_cpu is not None:
            total_delta = current[0] - self._previous_cpu[0]
            idle_delta = current[1] - self._previous_cpu[1]
            if total_delta > 0:
                sample["cpu_util_percent"] = 100.0 * (1.0 - idle_delta / total_delta)
        self._previous_cpu = current
        sample.update(_read_memory())
        gpus, gpu_status = _query_nvidia_smi()
        sample["gpus"] = gpus
        sample["gpu_telemetry_status"] = gpu_status["status"]
        sample["gpu_telemetry_resolver"] = gpu_status["resolver"]
        sample["nvidia_smi_path"] = gpu_status["path"]
        if gpu_status.get("error"):
            sample["gpu_telemetry_error"] = gpu_status["error"]
        return sample

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            sample = self._sample()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
            remaining = self.interval_s - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def summary(self) -> dict[str, object]:
        if not self.path.exists():
            return {"samples": 0, "phases": {}}
        by_phase: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        sample_count = 0
        gpu_samples = 0
        gpu_statuses: dict[str, int] = defaultdict(int)
        gpu_resolvers: dict[str, int] = defaultdict(int)
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_count += 1
            gpu_statuses[str(row.get("gpu_telemetry_status", "unknown"))] += 1
            gpu_resolvers[str(row.get("gpu_telemetry_resolver", "unknown"))] += 1
            phase = str(row.get("phase", "UNKNOWN"))
            for key in ("cpu_util_percent", "ram_used_gib", "ram_used_percent", "loadavg_1m"):
                value = row.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    by_phase[phase][key].append(float(value))
            gpus = row.get("gpus")
            if isinstance(gpus, list) and gpus:
                gpu_samples += 1
                util_values = [
                    float(item["gpu_util_percent"])
                    for item in gpus
                    if isinstance(item, dict) and isinstance(item.get("gpu_util_percent"), (int, float))
                ]
                memory_values = [
                    float(item["gpu_memory_used_mib"])
                    for item in gpus
                    if isinstance(item, dict) and isinstance(item.get("gpu_memory_used_mib"), (int, float))
                ]
                temperature_values = [
                    float(item["gpu_temperature_c"])
                    for item in gpus
                    if isinstance(item, dict) and isinstance(item.get("gpu_temperature_c"), (int, float))
                ]
                power_values = [
                    float(item["gpu_power_w"])
                    for item in gpus
                    if isinstance(item, dict) and isinstance(item.get("gpu_power_w"), (int, float))
                ]
                if util_values:
                    by_phase[phase]["gpu_util_percent"].append(max(util_values))
                if memory_values:
                    by_phase[phase]["gpu_memory_used_mib"].append(sum(memory_values))
                if temperature_values:
                    by_phase[phase]["gpu_temperature_c"].append(max(temperature_values))
                if power_values:
                    by_phase[phase]["gpu_power_w"].append(sum(power_values))

        phases: dict[str, object] = {}
        for phase, metrics in sorted(by_phase.items()):
            phases[phase] = {
                key: {
                    "mean": statistics.fmean(values) if values else None,
                    "p50": _percentile(values, 0.50),
                    "p95": _percentile(values, 0.95),
                    "max": max(values) if values else None,
                    "samples": len(values),
                }
                for key, values in sorted(metrics.items())
            }
        return {
            "samples": sample_count,
            "gpu_samples": gpu_samples,
            "gpu_telemetry_statuses": dict(sorted(gpu_statuses.items())),
            "gpu_telemetry_resolvers": dict(sorted(gpu_resolvers.items())),
            "phases": phases,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sample system/GPU telemetry to JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", default="BENCHMARK")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    telemetry = HardwareTelemetry(args.output, args.interval)
    telemetry.set_phase(args.phase)
    telemetry.start()
    try:
        time.sleep(args.seconds)
    finally:
        telemetry.stop()
    print(json.dumps(telemetry.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
