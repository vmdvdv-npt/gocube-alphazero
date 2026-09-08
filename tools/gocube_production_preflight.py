from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from tools.gocube_experiment_storage import MIN_FREE_RESERVE_BYTES
from tools.hardware_telemetry import nvidia_smi_command
from alphazero.envs.gocube.b_experiment_contract import (
    ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES,
    BExperimentContract,
    diff_effective_configs,
    preflight_b_experiment,
    resolve_b_effective_configs_separate_processes,
    validate_b0_b1_effective_configs,
    validate_b_experiment_contract,
    validate_b_experiment_preflight,
)


PREFLIGHT_SCHEMA_VERSION = 1
EXPECTED_REPO = Path("/home/codex/projects/gocube-alphazero")
EXPECTED_USER = "codex"
EXPECTED_HOSTNAME = "Legion"
EXPECTED_LOGICAL_CPUS = 16
MIN_RAM_BYTES = 32 * 1024**3
# B05 is a bounded integration proof and has a separately approved memory
# floor.  The shared production preflight keeps the stricter MIN_RAM_BYTES
# default; only validate_b05_production_preflight opts into this threshold.
B05_MIN_RAM_BYTES = 29 * 1024**3
EXPECTED_GPU_TOKEN = "RTX 3060"
MIN_GPU_VRAM_BYTES = 5 * 1024**3
REQUIRED_BRANCH = "main"
REQUIRED_UPSTREAM = "origin/main"
EXPECTED_REPOSITORY_ID = "vmdvdv-npt/gocube-alphazero"
SUPERVISED_ENV = "GOCUBE_SWEEP_SUPERVISED"
UNIT_ENV = "GOCUBE_SWEEP_UNIT"


def _run_text(command: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or f"exit {process.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(command)}: {detail}")
    return process.stdout.strip()


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return _run_text(["git", *args], cwd=repo, check=check)


def _current_user() -> str:
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError, OSError):
        return getpass.getuser()


def _physical_memory_bytes() -> int:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except (AttributeError, OSError, ValueError):
        return 0


def _affinity_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return int(os.cpu_count() or 0)


def _origin_matches_project(url: str) -> bool:
    normalized = str(url).strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    normalized = normalized.lower()
    target = EXPECTED_REPOSITORY_ID.lower()
    return normalized.endswith(f"github.com/{target}") or normalized.endswith(f"github.com:{target}")


def _pip_freeze() -> list[str]:
    try:
        output = _run_text([sys.executable, "-m", "pip", "freeze", "--all"])
    except RuntimeError as pip_error:
        # The project environment is uv-managed and may intentionally omit
        # pip.  uv's interpreter-targeted freeze is equivalent for the
        # reproducibility fingerprint; retain the original hard failure when
        # neither inventory path is available.
        uv = shutil.which("uv")
        if uv is None:
            output = ""
        else:
            try:
                output = _run_text([uv, "pip", "freeze", "--python", sys.executable])
            except RuntimeError:
                output = ""
    if not output:
        # Some systemd user environments intentionally have a minimal PATH
        # that hides uv as well.  The interpreter's installed distribution
        # metadata is still an exact, deterministic package inventory for
        # the preflight fingerprint.
        try:
            from importlib.metadata import distributions

            return sorted(
                f"{distribution.metadata['Name']}=={distribution.version}"
                for distribution in distributions()
                if distribution.metadata.get("Name") and distribution.version
            )
        except Exception:
            raise pip_error
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def _nvidia_driver_version() -> str | None:
    executable = nvidia_smi_command()
    if executable is None:
        return None
    try:
        output = _run_text(
            [*executable, "--query-gpu=driver_version", "--format=csv,noheader"]
        )
    except RuntimeError:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else None


def _linger_status(user: str) -> tuple[bool, str | None]:
    try:
        value = _run_text(["loginctl", "show-user", user, "-p", "Linger", "--value"])
    except RuntimeError as exc:
        return False, str(exc)
    return value.strip().lower() == "yes", None


def is_supervised_invocation() -> bool:
    return os.environ.get(SUPERVISED_ENV) == "1" and bool(os.environ.get("INVOCATION_ID"))


def _git_source_snapshot(repo: Path, *, verify_remote: bool) -> dict[str, object]:
    git_root = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()
    branch = _git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False) or "DETACHED"
    head = _git(repo, "rev-parse", "HEAD")
    origin = _git(repo, "remote", "get-url", "origin")
    tracked_status = _git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    upstream = _git(
        repo,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    ) or None
    upstream_sha = _git(repo, "rev-parse", "@{upstream}", check=False) or None
    remote_main_sha = None
    if verify_remote:
        remote = _git(repo, "ls-remote", "origin", "refs/heads/main")
        lines = [line.strip() for line in remote.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Could not resolve origin/main from GitHub")
        remote_main_sha = lines[0].split()[0]
    return {
        "repo_path": str(repo.resolve()),
        "git_root": str(git_root),
        "branch": branch,
        "head_sha": head,
        "origin_url": origin,
        "tracked_tree_clean": not bool(tracked_status),
        "tracked_status": tracked_status,
        "upstream": upstream,
        "upstream_sha": upstream_sha,
        "remote_main_sha": remote_main_sha,
        "remote_verified": bool(verify_remote),
    }


def _runtime_snapshot(repo: Path) -> dict[str, object]:
    freeze = _pip_freeze()
    freeze_hash = hashlib.sha256(("\n".join(freeze) + "\n").encode("utf-8")).hexdigest()
    try:
        os_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        os_release = {}
    return {
        "user": _current_user(),
        "python_executable": str(Path(sys.executable).absolute()),
        "python_prefix": str(Path(sys.prefix).absolute()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "pip_freeze": freeze,
        "pip_freeze_sha256": freeze_hash,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "os_id": os_release.get("ID"),
        "os_version_id": os_release.get("VERSION_ID"),
        "expected_venv": str((repo / ".venv").absolute()),
    }


def _hardware_snapshot(requested_device: str) -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    selected_device = "cuda" if requested_device == "auto" and cuda_available else str(requested_device)
    gpu: dict[str, object] | None = None
    if cuda_available and cuda_count > 0:
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "index": 0,
            "name": str(torch.cuda.get_device_name(0)),
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": [int(props.major), int(props.minor)],
            "driver_version": _nvidia_driver_version(),
        }
    return {
        "hostname": socket.gethostname(),
        "logical_cpu_count": int(os.cpu_count() or 0),
        "affinity_cpu_count": _affinity_cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "requested_device": str(requested_device),
        "selected_device": selected_device,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_count,
        "gpu": gpu,
    }


def _supervision_snapshot() -> dict[str, object]:
    user = _current_user()
    linger, linger_error = _linger_status(user)
    return {
        "supervised": is_supervised_invocation(),
        "invocation_id": os.environ.get("INVOCATION_ID"),
        "unit": os.environ.get(UNIT_ENV),
        "linger_enabled": linger,
        "linger_error": linger_error,
    }


def environment_fingerprint_payload(report: dict[str, Any]) -> dict[str, object]:
    source = report["source"]
    runtime = report["runtime"]
    hardware = report["hardware"]
    gpu = hardware.get("gpu") or {}
    return {
        "repository": EXPECTED_REPOSITORY_ID,
        "source_commit": source.get("head_sha"),
        "python_version": runtime.get("python_version"),
        "python_implementation": runtime.get("python_implementation"),
        "torch_version": runtime.get("torch_version"),
        "torch_cuda_version": runtime.get("torch_cuda_version"),
        "cudnn_version": runtime.get("cudnn_version"),
        "pip_freeze_sha256": runtime.get("pip_freeze_sha256"),
        "platform_system": runtime.get("platform_system"),
        "platform_release": runtime.get("platform_release"),
        "platform_machine": runtime.get("platform_machine"),
        "os_id": runtime.get("os_id"),
        "os_version_id": runtime.get("os_version_id"),
        "hostname": str(hardware.get("hostname", "")).lower(),
        "logical_cpu_count": hardware.get("logical_cpu_count"),
        "affinity_cpu_count": hardware.get("affinity_cpu_count"),
        "physical_memory_bytes": hardware.get("physical_memory_bytes"),
        "gpu_name": gpu.get("name"),
        "gpu_total_memory_bytes": gpu.get("total_memory_bytes"),
        "gpu_compute_capability": gpu.get("compute_capability"),
        "nvidia_driver_version": gpu.get("driver_version"),
    }


def attach_environment_fingerprint(report: dict[str, Any]) -> dict[str, Any]:
    critical = environment_fingerprint_payload(report)
    digest = hashlib.sha256(
        json.dumps(critical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report["critical_environment"] = critical
    report["environment_fingerprint_sha256"] = digest
    return report


def collect_production_preflight(
    repo: Path,
    cli,
    *,
    verify_remote: bool,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    usage = shutil.disk_usage(repo)
    report: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "checked_at_epoch": time.time(),
        "source": _git_source_snapshot(repo, verify_remote=verify_remote),
        "runtime": _runtime_snapshot(repo),
        "hardware": _hardware_snapshot(str(cli.device)),
        "disk": {
            "filesystem_total_bytes": int(usage.total),
            "filesystem_used_bytes": int(usage.used),
            "filesystem_free_bytes": int(usage.free),
            "minimum_free_reserve_bytes": int(MIN_FREE_RESERVE_BYTES),
        },
        "supervision": _supervision_snapshot(),
    }
    return attach_environment_fingerprint(report)


def _validate_common(report: dict[str, Any], *, min_ram_bytes: int = MIN_RAM_BYTES) -> None:
    if int(report.get("schema_version", -1)) != PREFLIGHT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported production preflight schema")
    source = report["source"]
    runtime = report["runtime"]
    hardware = report["hardware"]
    disk = report["disk"]
    supervision = report["supervision"]

    if Path(str(source.get("repo_path"))).resolve() != EXPECTED_REPO.resolve():
        raise RuntimeError(
            f"Production sweep must run from {EXPECTED_REPO}; got {source.get('repo_path')}"
        )
    if Path(str(source.get("git_root"))).resolve() != EXPECTED_REPO.resolve():
        raise RuntimeError("Git root does not match the production repository path")
    if not _origin_matches_project(str(source.get("origin_url", ""))):
        raise RuntimeError("origin does not point to vmdvdv-npt/gocube-alphazero")
    if not bool(source.get("tracked_tree_clean")):
        raise RuntimeError("Tracked Git tree is dirty; commit/stash changes before production sweep")

    if str(runtime.get("user")) != EXPECTED_USER:
        raise RuntimeError(f"Production sweep must run as user {EXPECTED_USER}")
    prefix = Path(str(runtime.get("python_prefix"))).resolve()
    if prefix != (EXPECTED_REPO / ".venv").resolve():
        raise RuntimeError("Production sweep must run under the project .venv Python")
    executable_parent = Path(str(runtime.get("python_executable"))).parent.resolve()
    if executable_parent != (EXPECTED_REPO / ".venv" / "bin").resolve():
        raise RuntimeError("Python executable is not the project .venv/bin/python")

    if str(hardware.get("hostname", "")).lower() != EXPECTED_HOSTNAME.lower():
        raise RuntimeError(f"Production sweep must run on host {EXPECTED_HOSTNAME}")
    if int(hardware.get("logical_cpu_count", 0)) != EXPECTED_LOGICAL_CPUS:
        raise RuntimeError(
            f"Expected {EXPECTED_LOGICAL_CPUS} logical CPUs; got {hardware.get('logical_cpu_count')}"
        )
    if int(hardware.get("affinity_cpu_count", 0)) != EXPECTED_LOGICAL_CPUS:
        raise RuntimeError(
            f"Process CPU affinity must expose all {EXPECTED_LOGICAL_CPUS} CPUs"
        )
    if int(hardware.get("physical_memory_bytes", 0)) < int(min_ram_bytes):
        raise RuntimeError(
            "Insufficient physical RAM for the Legion production profile: "
            f"{int(hardware.get('physical_memory_bytes', 0))} < {int(min_ram_bytes)} bytes"
        )
    if str(hardware.get("selected_device")) != "cuda":
        raise RuntimeError("Production sweep requires CUDA; CPU mode is test/debug only")
    if not bool(hardware.get("cuda_available")) or int(hardware.get("cuda_device_count", 0)) < 1:
        raise RuntimeError("CUDA is unavailable for the production sweep")
    gpu = hardware.get("gpu") or {}
    if EXPECTED_GPU_TOKEN.lower() not in str(gpu.get("name", "")).lower():
        raise RuntimeError(
            f"Unexpected production GPU: {gpu.get('name')!r}; expected {EXPECTED_GPU_TOKEN}"
        )
    if int(gpu.get("total_memory_bytes", 0)) < MIN_GPU_VRAM_BYTES:
        raise RuntimeError("Production GPU exposes less than the required VRAM")
    if not gpu.get("driver_version"):
        raise RuntimeError("NVIDIA driver version could not be recorded")
    if not runtime.get("torch_cuda_version"):
        raise RuntimeError("Installed PyTorch is not a CUDA build")

    if int(disk.get("filesystem_free_bytes", 0)) < int(disk.get("minimum_free_reserve_bytes", 0)):
        raise RuntimeError("Free disk is below the production safety reserve")

    if not bool(supervision.get("supervised")) or not supervision.get("invocation_id"):
        raise RuntimeError("Production sweep must run as the supervised systemd user service")
    if not bool(supervision.get("linger_enabled")):
        raise RuntimeError(
            "systemd user lingering is disabled; the sweep would not reliably survive SSH logout"
        )


def validate_new_production_preflight(report: dict[str, Any]) -> None:
    _validate_common(report)
    source = report["source"]
    if source.get("branch") != REQUIRED_BRANCH:
        raise RuntimeError(f"New production sweep must start on branch {REQUIRED_BRANCH}")
    if source.get("upstream") != REQUIRED_UPSTREAM:
        raise RuntimeError(f"main must track {REQUIRED_UPSTREAM}")
    if source.get("upstream_sha") != source.get("head_sha"):
        raise RuntimeError("Local main differs from its origin/main tracking ref")
    if not bool(source.get("remote_verified")):
        raise RuntimeError("New production sweep must verify GitHub origin/main")
    if source.get("remote_main_sha") != source.get("head_sha"):
        raise RuntimeError("Local HEAD is not the current GitHub origin/main commit")


def validate_b05_production_preflight(
    report: dict[str, Any],
    *,
    base_ref: str = REQUIRED_UPSTREAM,
) -> None:
    """Validate a B05 checkout using the shared production preflight.

    The long-running production sweep intentionally requires ``main``.  B05
    is a short integration proof and must run from its review branch, but only
    when that branch is rooted at the freshly fetched remote default branch.
    All hardware, runtime, supervision, disk, and clean-tree checks remain
    the same as the production preflight above.
    """

    _validate_common(report, min_ram_bytes=B05_MIN_RAM_BYTES)
    source = report["source"]
    if not bool(source.get("remote_verified")):
        raise RuntimeError("B05 must verify the freshly fetched GitHub origin/main")
    remote_main_sha = source.get("remote_main_sha")
    if not isinstance(remote_main_sha, str) or len(remote_main_sha) != 40:
        raise RuntimeError("B05 preflight has no authoritative remote main SHA")
    try:
        base_sha = _git(Path(str(source["repo_path"])), "rev-parse", base_ref)
        merge_base = _git(
            Path(str(source["repo_path"])),
            "merge-base",
            "HEAD",
            base_ref,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"B05 cannot resolve its authoritative base {base_ref}: {exc}") from exc
    if base_sha != remote_main_sha:
        raise RuntimeError(
            f"{base_ref} is stale: local {base_sha}, fetched GitHub origin/main {remote_main_sha}"
        )
    if merge_base != remote_main_sha:
        raise RuntimeError(
            "B05 source is not based on the freshly fetched origin/main commit: "
            f"merge-base={merge_base}, origin/main={remote_main_sha}"
        )
    branch = str(source.get("branch", ""))
    if branch == REQUIRED_BRANCH:
        if source.get("head_sha") != remote_main_sha:
            raise RuntimeError("B05 main checkout is not at the current GitHub origin/main commit")
    elif not branch.startswith("codex/"):
        raise RuntimeError(f"B05 must run on main or a codex review branch; got {branch!r}")


def validate_resume_production_preflight(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> None:
    if int(baseline.get("schema_version", -1)) != PREFLIGHT_SCHEMA_VERSION:
        raise RuntimeError("Saved production preflight schema is not resumable")
    _validate_common(current)
    saved_head = (baseline.get("source") or {}).get("head_sha")
    current_head = (current.get("source") or {}).get("head_sha")
    if current_head != saved_head:
        raise RuntimeError(
            f"Resume source commit changed: current {current_head}, expected {saved_head}"
        )
    saved_fingerprint = baseline.get("environment_fingerprint_sha256")
    current_fingerprint = current.get("environment_fingerprint_sha256")
    if not saved_fingerprint or current_fingerprint != saved_fingerprint:
        raise RuntimeError(
            "Production environment changed since the experiment started; "
            "resume is fail-closed to preserve reproducibility"
        )


def _state_has_prior_work(state: dict[str, Any]) -> bool:
    return bool(
        int(state.get("resume_count", 0)) > 0
        or state.get("bootstrap")
        or state.get("health_gate")
        or state.get("performance_benchmark")
        or state.get("parameters")
        or state.get("completed_actions")
    )


def _session_check(report: dict[str, Any]) -> dict[str, object]:
    source = report["source"]
    hardware = report["hardware"]
    gpu = hardware.get("gpu") or {}
    disk = report["disk"]
    supervision = report["supervision"]
    return {
        "checked_at_epoch": report["checked_at_epoch"],
        "head_sha": source.get("head_sha"),
        "branch": source.get("branch"),
        "upstream": source.get("upstream"),
        "remote_main_sha": source.get("remote_main_sha"),
        "environment_fingerprint_sha256": report.get("environment_fingerprint_sha256"),
        "hostname": hardware.get("hostname"),
        "gpu_name": gpu.get("name"),
        "driver_version": gpu.get("driver_version"),
        "filesystem_free_bytes": disk.get("filesystem_free_bytes"),
        "systemd_unit": supervision.get("unit"),
        "invocation_id": supervision.get("invocation_id"),
    }


def apply_production_preflight(experiment) -> dict[str, Any]:
    state = experiment.state
    saved = state.get("production_preflight")
    try:
        if saved is None and _state_has_prior_work(state):
            raise RuntimeError(
                "Existing experiment has no production preflight provenance and cannot be safely resumed"
            )
        if saved is not None:
            if int(saved.get("schema_version", -1)) != PREFLIGHT_SCHEMA_VERSION:
                raise RuntimeError("Saved production preflight wrapper schema is not resumable")
            baseline = saved.get("baseline")
            if not isinstance(baseline, dict):
                raise RuntimeError("Saved production preflight has no baseline environment")
            current = collect_production_preflight(experiment.repo, experiment.cli, verify_remote=False)
            validate_resume_production_preflight(baseline, current)
            saved.setdefault("checks", []).append(_session_check(current))
            saved["latest_check"] = _session_check(current)
            preflight_state = saved
        else:
            current = collect_production_preflight(experiment.repo, experiment.cli, verify_remote=True)
            validate_new_production_preflight(current)
            check = _session_check(current)
            preflight_state = {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "baseline": current,
                "checks": [check],
                "latest_check": check,
            }
        state["production_preflight"] = preflight_state
        baseline = preflight_state["baseline"]
        state["source_commit"] = baseline["source"]["head_sha"]
        state["environment_fingerprint_sha256"] = baseline["environment_fingerprint_sha256"]
        state.pop("preflight_error", None)
        experiment._save_state()
        return current
    except Exception as exc:
        state["status"] = "FAILED"
        state["preflight_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at_epoch": time.time(),
        }
        experiment._save_state()
        raise


def _safe_unit_name(experiment_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(experiment_id)).strip("-.")
    safe = safe or "experiment"
    return f"gocube-sweep-{safe}"[:120]


def _supervised_cli_args(cli) -> list[str]:
    args = [
        "--experiment-id", str(cli.experiment_id),
        "--device", str(cli.device),
        "--arena-batch-wait-ms", str(cli.arena_batch_wait_ms),
        "--telemetry-interval", str(cli.telemetry_interval),
        "--heldout-positions", str(cli.heldout_positions),
        "--health-gate-min-win-rate", str(cli.health_gate_min_win_rate),
        "--benchmark-games", str(cli.benchmark_games),
        "--seed", str(cli.seed),
    ]
    if cli.bootstrap_run:
        args.extend(["--bootstrap-run", str(cli.bootstrap_run)])
    if cli.skip_performance_benchmark:
        args.append("--skip-performance-benchmark")
    return args


def build_systemd_run_command(repo: Path, cli) -> tuple[str, list[str]]:
    repo = Path(repo).resolve()
    unit = _safe_unit_name(cli.experiment_id)
    command = [
        "systemd-run",
        "--user",
        "--collect",
        "--unit", unit,
        f"--working-directory={repo}",
        "--property=Restart=on-abnormal",
        "--property=RestartSec=10s",
        "--property=TimeoutStopSec=30s",
        "--property=KillMode=mixed",
        f"--setenv={SUPERVISED_ENV}=1",
        f"--setenv={UNIT_ENV}={unit}",
        "--setenv=PYTHONUNBUFFERED=1",
        str(repo / ".venv" / "bin" / "python"),
        str(repo / "tools" / "c4_overnight_experiment.py"),
        *_supervised_cli_args(cli),
    ]
    return unit, command


def _assert_supervised_launch_ready(repo: Path) -> None:
    repo = Path(repo).resolve()
    if repo != EXPECTED_REPO.resolve():
        raise RuntimeError(f"Production sweep must be launched from {EXPECTED_REPO}")
    if _current_user() != EXPECTED_USER:
        raise RuntimeError(f"Production sweep must be launched as {EXPECTED_USER}")
    if socket.gethostname().lower() != EXPECTED_HOSTNAME.lower():
        raise RuntimeError(f"Production sweep must be launched on {EXPECTED_HOSTNAME}")
    if Path(sys.prefix).resolve() != (repo / ".venv").resolve():
        raise RuntimeError("Launch the sweep with the project .venv/bin/python")
    for executable in ("systemd-run", "systemctl", "loginctl"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required supervision command is missing: {executable}")
    linger, error = _linger_status(EXPECTED_USER)
    if not linger:
        detail = f" ({error})" if error else ""
        raise RuntimeError(
            f"systemd user lingering is disabled for {EXPECTED_USER}{detail}; "
            f"enable it before the overnight run so logout cannot stop the sweep"
        )


def launch_under_systemd(repo: Path, cli) -> int:
    repo = Path(repo).resolve()
    _assert_supervised_launch_ready(repo)
    unit, command = build_systemd_run_command(repo, cli)
    active = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        cwd=str(repo),
        check=False,
    )
    if active.returncode == 0:
        raise RuntimeError(f"Sweep service is already active: {unit}")
    process = subprocess.run(command, cwd=str(repo), check=False)
    if process.returncode != 0:
        raise RuntimeError(f"systemd-run failed for {unit} with exit code {process.returncode}")
    print(f"Started supervised sweep service: {unit}")
    print(f"Logs: journalctl --user -fu {unit}")
    return 0
