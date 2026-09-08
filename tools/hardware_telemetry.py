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
import shutil
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator


def nvidia_smi_command() -> list[str] | None:
    """Return the available NVIDIA management command.

    WSL exposes ``nvidia-smi`` in ``/usr/lib/wsl/lib`` without adding that
    directory to PATH.  Keeping discovery here lets both the preflight and
    the background sampler use the same telemetry path without treating a
    usable CUDA device as a missing GPU merely because PATH is minimal.
    """

    candidates = []
    resolved = shutil.which("nvidia-smi")
    if resolved:
        candidates.append(resolved)
    candidates.append("/usr/lib/wsl/lib/nvidia-smi")
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return [candidate]
    return None


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
    memory = {
        "ram_used_gib": used / (1024.0 * 1024.0),
        "ram_total_gib": total / (1024.0 * 1024.0),
        "ram_used_percent": 100.0 * used / total,
    }
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    if swap_total is not None and swap_free is not None:
        swap_used = max(0, swap_total - swap_free)
        memory.update(
            {
                "swap_used_gib": swap_used / (1024.0 * 1024.0),
                "swap_total_gib": swap_total / (1024.0 * 1024.0),
                "swap_used_percent": 100.0 * swap_used / swap_total if swap_total else 0.0,
            }
        )
    return memory


def _read_nvidia_smi() -> list[dict[str, float | int]]:
    executable = nvidia_smi_command()
    if executable is None:
        return []
    command = [
        *executable,
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, float | int]] = []
    for raw in result.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 6:
            continue
        try:
            index = int(parts[0])
            util = float(parts[1])
            used = float(parts[2])
            total = float(parts[3])
            temp = float(parts[4])
            power = float(parts[5])
        except ValueError:
            continue
        rows.append(
            {
                "index": index,
                "gpu_util_percent": util,
                "gpu_memory_used_mib": used,
                "gpu_memory_total_mib": total,
                "gpu_temperature_c": temp,
                "gpu_power_w": power,
            }
        )
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
        sample["gpus"] = _read_nvidia_smi()
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
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_count += 1
            phase = str(row.get("phase", "UNKNOWN"))
            for key in (
                "cpu_util_percent",
                "ram_used_gib",
                "ram_total_gib",
                "ram_used_percent",
                "swap_used_gib",
                "swap_total_gib",
                "swap_used_percent",
                "loadavg_1m",
            ):
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
                if util_values:
                    by_phase[phase]["gpu_util_percent"].append(max(util_values))
                if memory_values:
                    by_phase[phase]["gpu_memory_used_mib"].append(sum(memory_values))
                temperature_values = [
                    float(item["gpu_temperature_c"])
                    for item in gpus
                    if isinstance(item, dict)
                    and isinstance(item.get("gpu_temperature_c"), (int, float))
                ]
                power_values = [
                    float(item["gpu_power_w"])
                    for item in gpus
                    if isinstance(item, dict)
                    and isinstance(item.get("gpu_power_w"), (int, float))
                ]
                if temperature_values:
                    by_phase[phase]["gpu_temperature_c"].append(max(temperature_values))
                if power_values:
                    by_phase[phase]["gpu_power_w"].append(sum(power_values))

        phases: dict[str, object] = {}
        for phase, metrics in sorted(by_phase.items()):
            phases[phase] = {
                key: {
                    "mean": statistics.fmean(values) if values else None,
                    "p95": _percentile(values, 0.95),
                    "max": max(values) if values else None,
                    "samples": len(values),
                }
                for key, values in sorted(metrics.items())
            }
        peaks = {}
        for key in ("ram_used_gib", "ram_used_percent", "swap_used_gib", "swap_used_percent"):
            values = [
                value
                for metrics in by_phase.values()
                for value in metrics.get(key, [])
            ]
            peaks[key] = max(values) if values else None
        capacities = {}
        for key in ("ram_total_gib", "swap_total_gib"):
            values = [
                value
                for metrics in by_phase.values()
                for value in metrics.get(key, [])
            ]
            capacities[key] = max(values) if values else None
        return {
            "samples": sample_count,
            "gpu_samples": gpu_samples,
            "capacities": capacities,
            "peaks": peaks,
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
