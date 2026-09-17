"""Production supervisor for long-running AlphaZero training lineages.

The orchestrator deliberately owns *execution*, not scientific semantics.
A profile-specific generation driver receives an immutable canonical profile
and is responsible for self-play/replay/training/checkpoint semantics.  This
module owns lineage storage, durable state, resume, health supervision,
periodic Arena scheduling, soft-stop, reporting, and fail-closed validation of
the driver's published generation result.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .artifact_catalog import ArtifactCatalog, ARTIFACT_CATALOG_SCHEMA
from .run_storage import (
    active_lineage_dir,
    create_lineage,
    ensure_evaluation_layout,
    evaluation_dir,
    evaluation_id_for_comparison,
)


ORCHESTRATOR_SCHEMA = "gocube-production-training-orchestrator-v1"
DRIVER_RESULT_SCHEMA = "gocube-generation-driver-result-v1"
ARENA_RESULT_SCHEMA = "gocube-arena-driver-result-v1"
FINAL_STATES = frozenset({"COMPLETED", "SOFT_STOPPED"})
RESUMABLE_STOP_STATES = frozenset({"COMPLETED", "SOFT_STOPPED", "RECOVERY_REQUIRED"})
RUNTIME_STATES = frozenset({"CREATED", "RUNNING", "SOFT_STOP_REQUESTED", "RECOVERY_REQUIRED", *FINAL_STATES})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def _git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Cannot determine git HEAD for training lineage")
    return result.stdout.strip()


def _safe_relative_path(value: str, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe relative path: {value!r}")
    return path


def _resolve_repo_path(repo_root: Path, value: str, *, label: str) -> Path:
    relative = _safe_relative_path(value, label=label)
    path = (repo_root / relative).resolve()
    root = repo_root.resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{label} escapes repository root")
    return path


def _render_token(value: str, *, generation: int, lineage_id: str, run_root: Path) -> str:
    return str(value).format(
        generation=generation,
        generation04=f"{generation:04d}",
        lineage_id=lineage_id,
        run_root=str(run_root),
    )


def _render_command(command: Sequence[str], **kwargs: object) -> list[str]:
    return [str(token).format(**kwargs) for token in command]


@dataclass(frozen=True)
class HealthPolicy:
    poll_seconds: float = 10.0
    heartbeat_warning_seconds: float = 180.0
    heartbeat_critical_seconds: float = 900.0
    min_disk_free_gb_warning: float = 15.0
    min_disk_free_gb_critical: float = 5.0
    min_ram_free_gb_warning: float = 2.0
    min_ram_free_gb_critical: float = 0.5

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "HealthPolicy":
        source = dict(value or {})
        result = cls(**{key: source[key] for key in cls.__dataclass_fields__ if key in source})
        if result.poll_seconds <= 0:
            raise ValueError("health.poll_seconds must be positive")
        if not 0 < result.heartbeat_warning_seconds < result.heartbeat_critical_seconds:
            raise ValueError("heartbeat warning must be below critical threshold")
        if not 0 <= result.min_disk_free_gb_critical <= result.min_disk_free_gb_warning:
            raise ValueError("disk critical threshold must be <= warning threshold")
        if not 0 <= result.min_ram_free_gb_critical <= result.min_ram_free_gb_warning:
            raise ValueError("RAM critical threshold must be <= warning threshold")
        return result


@dataclass(frozen=True)
class SoftStopPolicy:
    default_minutes: int = 60
    minimum_minutes: int = 30
    maximum_minutes: int = 100

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "SoftStopPolicy":
        source = dict(value or {})
        result = cls(**{key: int(source[key]) for key in cls.__dataclass_fields__ if key in source})
        if not (1 <= result.minimum_minutes <= result.default_minutes <= result.maximum_minutes):
            raise ValueError("soft_stop requires minimum <= default <= maximum")
        return result

    def validate(self, minutes: int) -> int:
        minutes = int(minutes)
        if not self.minimum_minutes <= minutes <= self.maximum_minutes:
            raise ValueError(
                f"soft-stop window must be {self.minimum_minutes}..{self.maximum_minutes} minutes"
            )
        return minutes


@dataclass(frozen=True)
class OrchestratorSpec:
    path: Path
    payload: Mapping[str, object]
    topology: str
    profile_path: Path
    profile_payload: Mapping[str, object]
    profile_fingerprint: str
    config_fingerprint: str
    generation_command: tuple[str, ...]
    generation_resume_command: tuple[str, ...] | None
    arena_command: tuple[str, ...] | None
    arena_every_generations: int
    arena_required: bool
    arena_preset_fingerprint: str | None
    arena_startset_fingerprint: str | None
    health: HealthPolicy
    soft_stop: SoftStopPolicy
    performance: Mapping[str, object]
    learning: Mapping[str, object]

    @classmethod
    def load(cls, path: str | Path, *, repo_root: Path) -> "OrchestratorSpec":
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        config_path = config_path.resolve()
        payload = read_json(config_path)
        if payload.get("schema") != ORCHESTRATOR_SCHEMA:
            raise ValueError(f"Unsupported orchestrator schema in {config_path}")
        topology = str(payload.get("topology", "")).strip()
        if not topology or "/" in topology or "\\" in topology or topology in {".", ".."}:
            raise ValueError("orchestrator topology must be an explicit safe storage topology")
        profile_ref = str(payload.get("profile_path", ""))
        if not profile_ref or profile_ref.lower().endswith(".xlsx"):
            raise ValueError("profile_path must reference the canonical JSON preset, never an XLSX export")
        profile_path = _resolve_repo_path(repo_root, profile_ref, label="profile_path")
        profile_payload = read_json(profile_path)
        declared = str(payload.get("expected_profile_fingerprint", "")).strip()
        embedded = str(
            profile_payload.get("profile_fingerprint")
            or profile_payload.get("content_fingerprint")
            or sha256_file(profile_path)
        )
        if declared and declared != embedded:
            raise ValueError(
                f"Canonical profile fingerprint mismatch: expected {declared}, got {embedded}"
            )
        execution = payload.get("execution")
        if not isinstance(execution, Mapping):
            raise ValueError("orchestrator execution block is required")
        generation_command = execution.get("generation_command")
        if not isinstance(generation_command, list) or not generation_command:
            raise ValueError("execution.generation_command must be a non-empty argv list")
        if not all(isinstance(item, str) and item for item in generation_command):
            raise ValueError("execution.generation_command tokens must be non-empty strings")
        resume_command = execution.get("generation_resume_command")
        if resume_command is not None and (
            not isinstance(resume_command, list)
            or not resume_command
            or not all(isinstance(item, str) and item for item in resume_command)
        ):
            raise ValueError("execution.generation_resume_command must be an argv list")
        arena = payload.get("arena")
        arena_map = dict(arena) if isinstance(arena, Mapping) else {}
        arena_command = arena_map.get("command")
        if arena_command is not None and (
            not isinstance(arena_command, list)
            or not arena_command
            or not all(isinstance(item, str) and item for item in arena_command)
        ):
            raise ValueError("arena.command must be an argv list")
        every = int(arena_map.get("every_generations", 5))
        if not 5 <= every <= 10:
            raise ValueError("arena.every_generations must be between 5 and 10")
        required = bool(arena_map.get("required", True))
        arena_preset_fingerprint = str(arena_map.get("preset_fingerprint", "")).strip() or None
        arena_startset_fingerprint = str(arena_map.get("startset_fingerprint", "")).strip() or None
        if required and arena_command is None:
            raise ValueError("required periodic Arena needs arena.command")
        if required and (arena_preset_fingerprint is None or arena_startset_fingerprint is None):
            raise ValueError("required periodic Arena needs frozen preset_fingerprint and startset_fingerprint")
        normalized = dict(payload)
        normalized["resolved_profile_fingerprint"] = embedded
        config_fingerprint = sha256_bytes(canonical_json(normalized).encode("utf-8"))
        return cls(
            path=config_path,
            payload=payload,
            topology=topology,
            profile_path=profile_path,
            profile_payload=profile_payload,
            profile_fingerprint=embedded,
            config_fingerprint=config_fingerprint,
            generation_command=tuple(generation_command),
            generation_resume_command=(tuple(resume_command) if isinstance(resume_command, list) else None),
            arena_command=(tuple(arena_command) if isinstance(arena_command, list) else None),
            arena_every_generations=every,
            arena_required=required,
            arena_preset_fingerprint=arena_preset_fingerprint,
            arena_startset_fingerprint=arena_startset_fingerprint,
            health=HealthPolicy.from_mapping(payload.get("health") if isinstance(payload.get("health"), Mapping) else None),
            soft_stop=SoftStopPolicy.from_mapping(payload.get("soft_stop") if isinstance(payload.get("soft_stop"), Mapping) else None),
            performance=dict(payload.get("performance", {})) if isinstance(payload.get("performance"), Mapping) else {},
            learning=dict(payload.get("learning", {})) if isinstance(payload.get("learning"), Mapping) else {},
        )


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    runtime: Path
    runtime_state: Path
    heartbeat: Path
    driver_heartbeat: Path
    lock: Path
    control: Path
    stop_request: Path
    logs: Path
    events: Path
    reports: Path
    report_md: Path
    final_json: Path
    final_md: Path
    metrics: Path
    generations: Path
    artifact_catalog: Path

    @classmethod
    def for_lineage(cls, topology: str, lineage_id: str) -> "RunPaths":
        root = active_lineage_dir(topology, lineage_id)
        return cls(
            root=root,
            manifest=root / "manifest.json",
            runtime=root / "runtime",
            runtime_state=root / "runtime" / "state.json",
            heartbeat=root / "runtime" / "orchestrator-heartbeat.json",
            driver_heartbeat=root / "runtime" / "driver-heartbeat.json",
            lock=root / "runtime" / "orchestrator.lock",
            control=root / "control",
            stop_request=root / "control" / "soft-stop.json",
            logs=root / "logs",
            events=root / "logs" / "orchestrator-events.jsonl",
            reports=root / "reports",
            report_md=root / "reports" / "training-report.md",
            final_json=root / "reports" / "final-report.json",
            final_md=root / "reports" / "final-report.md",
            metrics=root / "metrics",
            generations=root / "runtime" / "generations",
            artifact_catalog=root / "runtime" / "artifact-catalog.json",
        )


class EventSink:
    def __init__(self, paths: RunPaths, *, terminal: bool = True) -> None:
        self.paths = paths
        self.terminal = bool(terminal)

    def emit(self, level: str, message: str, **details: object) -> None:
        event = {"at": utc_now(), "level": str(level).upper(), "message": str(message), **details}
        self.paths.events.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
        if self.terminal:
            prefix = f"[{event['level']}]"
            print(f"{prefix} {message}", flush=True)


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "host": socket.gethostname(), "started_at": utc_now()}
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    existing = read_json(self.path)
                except Exception:
                    raise RuntimeError(f"Cannot validate existing orchestrator lock: {self.path}")
                same_host = existing.get("host") == socket.gethostname()
                pid = int(existing.get("pid", -1))
                if same_host and pid > 0 and not self._pid_alive(pid):
                    self.path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(
                    f"Training lineage already has an active orchestrator lock: pid={pid} host={existing.get('host')}"
                )
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self.acquired = True
                return
        raise RuntimeError("Could not acquire orchestrator lock")

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _memory_available_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].endswith(":"):
            try:
                values[fields[0][:-1]] = int(fields[1])
            except ValueError:
                continue
    available_kib = values.get("MemAvailable")
    return (available_kib / 1024 / 1024) if available_kib is not None else None


def _latest_event(events_path: Path, levels: set[str] | None = None) -> dict[str, object] | None:
    if not events_path.is_file():
        return None
    latest: dict[str, object] | None = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and (levels is None or str(event.get("level")) in levels):
            latest = event
    return latest


class ProductionTrainingOrchestrator:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        spec: OrchestratorSpec,
        lineage_id: str,
        terminal: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.spec = spec
        self.lineage_id = str(lineage_id)
        self.paths = RunPaths.for_lineage(spec.topology, self.lineage_id)
        self.events = EventSink(self.paths, terminal=terminal)

    def _ensure_layout(self) -> None:
        for path in (
            self.paths.runtime,
            self.paths.control,
            self.paths.logs,
            self.paths.reports,
            self.paths.metrics,
            self.paths.generations,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def create(self, *, parent_checkpoint: Mapping[str, object] | None = None) -> None:
        if self.paths.root.exists():
            raise FileExistsError(f"Lineage already exists: {self.paths.root}")
        initial_generation = 0
        if parent_checkpoint is not None and parent_checkpoint.get("generation") is not None:
            try:
                initial_generation = int(parent_checkpoint["generation"])
            except (TypeError, ValueError) as exc:
                raise ValueError("parent checkpoint generation must be an integer") from exc
            if initial_generation < 0:
                raise ValueError("parent checkpoint generation must be non-negative")
        manifest: dict[str, object] = {
            "lineage_id": self.lineage_id,
            "topology": self.spec.topology,
            "status": "ACTIVE",
            "parent_checkpoint": dict(parent_checkpoint) if parent_checkpoint is not None else None,
            "git_commit": _git_sha(self.repo_root),
            "config_fingerprint": self.spec.config_fingerprint,
            "created_at": utc_now(),
            "checkpoint_hashes": {},
            "orchestrator": {
                "schema": ORCHESTRATOR_SCHEMA,
                "spec_path": str(self.spec.path.relative_to(self.repo_root)),
                "profile_path": str(self.spec.profile_path.relative_to(self.repo_root)),
                "profile_fingerprint": self.spec.profile_fingerprint,
                "arena_every_generations": self.spec.arena_every_generations,
                "generation_origin": initial_generation,
                "last_committed_generation": initial_generation,
                "arena_generations": [],
                "runtime_state": "CREATED",
                "artifact_catalog": {
                    "schema": ARTIFACT_CATALOG_SCHEMA,
                    "path": "runtime/artifact-catalog.json",
                },
            },
        }
        create_lineage(
            self.spec.topology,
            self.lineage_id,
            manifest=manifest,
            extra_directories=("runtime", "control", "reports"),
        )
        self._ensure_layout()
        catalog = ArtifactCatalog.initialize(
            self.paths.artifact_catalog,
            lineage_id=self.lineage_id,
            root=self.paths.root,
        )
        manifest = read_json(self.paths.manifest)
        orchestrator = dict(manifest["orchestrator"])  # type: ignore[arg-type]
        catalog_record = dict(orchestrator["artifact_catalog"])  # type: ignore[arg-type]
        catalog_record["fingerprint"] = catalog.fingerprint
        orchestrator["artifact_catalog"] = catalog_record
        manifest["orchestrator"] = orchestrator
        atomic_write_json(self.paths.manifest, manifest)
        state = {
            "schema": ORCHESTRATOR_SCHEMA,
            "lineage_id": self.lineage_id,
            "topology": self.spec.topology,
            "state": "CREATED",
            "generation_origin": initial_generation,
            "last_committed_generation": initial_generation,
            "active_generation": None,
            "active_phase": None,
            "pid": None,
            "started_at": None,
            "updated_at": utc_now(),
            "stop_requested": False,
        }
        atomic_write_json(self.paths.runtime_state, state)
        self.events.emit("INFO", "Created training lineage", lineage_id=self.lineage_id)

    def _recover_interrupted_commit(self) -> bool:
        """Finish a durable generation commit whose tail writes were interrupted.

        A generation transaction is written as ``COMMITTED`` only after its
        result has been validated.  The catalog and manifest are then updated
        in separate atomic writes, so a process can die with a valid committed
        transaction and catalog but an older manifest.  Recovery treats the
        transaction plus catalog identities as the commit journal and makes
        the remaining state/manifest writes idempotently.

        The method deliberately verifies only the affected generations.  It
        never turns recovery into a scan or rehash of the whole lineage, but
        it does fail closed when the journal, catalog, or published result do
        not agree.
        """
        manifest = read_json(self.paths.manifest)
        orchestrator = manifest.get("orchestrator")
        if not isinstance(orchestrator, Mapping):
            return False
        catalog_record = orchestrator.get("artifact_catalog")
        if not isinstance(catalog_record, Mapping):
            return False
        catalog = ArtifactCatalog.load(
            self.paths.artifact_catalog,
            root=self.paths.root,
        )
        try:
            manifest_generation = int(orchestrator.get("last_committed_generation", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Manifest committed generation is malformed") from exc

        committed: list[tuple[int, Path, dict[str, object]]] = []
        if self.paths.generations.is_dir():
            transaction_paths = sorted(self.paths.generations.glob("generation-????.json"))
        else:
            transaction_paths = []
        for transaction_path in transaction_paths:
            transaction = read_json(transaction_path)
            if transaction.get("status") != "COMMITTED":
                continue
            try:
                generation = int(transaction.get("generation", -1))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Committed generation transaction is malformed: {transaction_path}"
                ) from exc
            if generation <= manifest_generation:
                continue
            if transaction.get("schema") != ORCHESTRATOR_SCHEMA:
                raise ValueError(
                    f"Committed generation transaction schema mismatch: {transaction_path}"
                )
            if transaction.get("profile_fingerprint") != self.spec.profile_fingerprint:
                raise ValueError(
                    f"Committed generation transaction profile mismatch: {transaction_path}"
                )
            artifact_hashes = transaction.get("artifact_hashes")
            if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
                raise ValueError(
                    f"Committed generation transaction has no artifact identities: {transaction_path}"
                )
            normalized: dict[str, object] = {}
            for raw_path, raw_digest in artifact_hashes.items():
                relative = _safe_relative_path(
                    str(raw_path), label="committed transaction artifact path"
                ).as_posix()
                digest = str(raw_digest)
                if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
                    raise ValueError(
                        f"Committed transaction artifact SHA is malformed: {relative}"
                    )
                if relative in normalized:
                    raise ValueError(
                        f"Committed transaction contains duplicate artifact path: {relative}"
                    )
                normalized[relative] = digest
            committed.append((generation, transaction_path, {**transaction, "artifact_hashes": normalized}))

        committed.sort(key=lambda item: item[0])
        recovered_generations = {generation for generation, _path, _tx in committed}
        raw_generations = catalog.payload.get("generations")
        if not isinstance(raw_generations, Mapping):
            raise ValueError("Artifact catalog generations are malformed")
        for raw_generation in raw_generations:
            try:
                catalog_generation = int(raw_generation)
            except (TypeError, ValueError) as exc:
                raise ValueError("Artifact catalog generation key is malformed") from exc
            if catalog_generation > manifest_generation and catalog_generation not in recovered_generations:
                raise ValueError(
                    "Artifact catalog has an unjournaled generation that cannot be recovered"
                )

        for generation, transaction_path, transaction in committed:
            expected_hashes = transaction["artifact_hashes"]
            assert isinstance(expected_hashes, Mapping)
            generations = catalog.payload.get("generations")
            assert isinstance(generations, Mapping)
            generation_record = generations.get(str(generation))
            if generation_record is None:
                # The process may have died after the transaction write but
                # before catalog registration.  Re-validate this one result
                # and publish its already-declared identities idempotently.
                result = self._validate_generation_result(generation)
                validated = result.get("validated_artifact_hashes")
                if not isinstance(validated, Mapping):
                    raise ValueError(
                        f"Committed generation {generation} has no validated artifact identities"
                    )
                normalized_validated = {
                    _safe_relative_path(
                        str(path), label="validated artifact path"
                    ).as_posix(): str(digest)
                    for path, digest in validated.items()
                }
                if normalized_validated != dict(expected_hashes):
                    raise ValueError(
                        f"Committed generation {generation} transaction/result artifact drift"
                    )
                self._record_generation_artifacts(generation, result, transaction_path)
                catalog = ArtifactCatalog.load(
                    self.paths.artifact_catalog,
                    root=self.paths.root,
                )
                generation_record = catalog.payload.get("generations", {}).get(str(generation))
            if not isinstance(generation_record, Mapping):
                raise ValueError(f"Artifact catalog record is malformed for generation {generation}")
            if int(generation_record.get("generation", -1)) != generation:
                raise ValueError(f"Artifact catalog generation identity mismatch: {generation}")
            artifact_paths = generation_record.get("artifact_paths")
            if not isinstance(artifact_paths, list):
                raise ValueError(f"Artifact catalog paths are malformed for generation {generation}")
            normalized_paths = {
                _safe_relative_path(
                    str(path), label="catalog generation artifact path"
                ).as_posix()
                for path in artifact_paths
            }
            if normalized_paths != set(expected_hashes):
                raise ValueError(f"Artifact catalog artifact set drift for generation {generation}")
            catalog_transaction = generation_record.get("transaction")
            if not isinstance(catalog_transaction, Mapping):
                raise ValueError(f"Artifact catalog transaction is missing for generation {generation}")
            expected_transaction_path = transaction_path.relative_to(self.paths.root).as_posix()
            if (
                catalog_transaction.get("path") != expected_transaction_path
                or catalog_transaction.get("status") != "COMMITTED"
                or catalog_transaction.get("sha256") != sha256_file(transaction_path)
                or int(catalog_transaction.get("size_bytes", -1)) != transaction_path.stat().st_size
            ):
                raise ValueError(f"Artifact catalog transaction drift for generation {generation}")
            for path, digest in expected_hashes.items():
                identity = catalog.identity(path)
                if identity.get("sha256") != digest:
                    raise ValueError(f"Artifact catalog identity drift for generation {generation}: {path}")
            catalog.verify(tuple(sorted(expected_hashes)))

        if not committed:
            # There is no durable commit journal that explains the catalog
            # change.  Leave the original strict drift error to the caller.
            return False

        latest_generation = max(
            [manifest_generation, *(generation for generation, _path, _tx in committed)]
        )
        expected_generations = set(range(manifest_generation + 1, latest_generation + 1))
        if recovered_generations != expected_generations:
            raise ValueError("Committed generation journal has a gap and cannot be recovered")
        current_state = self._state()
        try:
            state_generation = int(current_state.get("last_committed_generation", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Runtime committed generation is malformed") from exc
        if state_generation > latest_generation:
            raise ValueError("Runtime state is ahead of the recoverable committed generation")
        state = self._write_state(
            state="RECOVERY_REQUIRED",
            last_committed_generation=latest_generation,
            active_generation=None,
            active_phase=None,
            pid=None,
            error=(
                "Recovered interrupted generation commit; explicit resume is required "
                f"for generations {', '.join(str(generation) for generation, _path, _tx in committed)}"
            ),
        )
        recovered_manifest = read_json(self.paths.manifest)
        recovered_orchestrator = recovered_manifest.get("orchestrator")
        if not isinstance(recovered_orchestrator, dict):
            raise ValueError("Lineage manifest lacks orchestrator state during commit recovery")
        recovered_orchestrator["last_committed_generation"] = latest_generation
        recovered_orchestrator["runtime_state"] = state.get("state")
        recovered_catalog_record = dict(recovered_orchestrator.get("artifact_catalog", {}))
        recovered_catalog_record["schema"] = ARTIFACT_CATALOG_SCHEMA
        recovered_catalog_record["path"] = "runtime/artifact-catalog.json"
        recovered_catalog_record["fingerprint"] = catalog.fingerprint
        recovered_orchestrator["artifact_catalog"] = recovered_catalog_record
        recovered_manifest["orchestrator"] = recovered_orchestrator
        checkpoint_hashes = dict(recovered_manifest.get("checkpoint_hashes", {}))
        for _generation, _transaction_path, transaction in committed:
            artifact_hashes = transaction["artifact_hashes"]
            assert isinstance(artifact_hashes, Mapping)
            for path, digest in artifact_hashes.items():
                if str(path).startswith("checkpoints/"):
                    checkpoint_hashes[str(path)] = str(digest)
        recovered_manifest["checkpoint_hashes"] = checkpoint_hashes
        atomic_write_json(self.paths.manifest, recovered_manifest)
        self.events.emit(
            "WARNING",
            "Recovered interrupted generation commit; explicit resume is required",
            generations=sorted(recovered_generations),
        )
        return True

    def _load_manifest(self, *, expected_catalog_fingerprint: str | None = None) -> dict[str, object]:
        manifest = read_json(self.paths.manifest)
        if manifest.get("lineage_id") != self.lineage_id or manifest.get("topology") != self.spec.topology:
            raise ValueError("Lineage manifest identity does not match requested run")
        if manifest.get("status") != "ACTIVE":
            raise ValueError(f"Cannot run non-ACTIVE lineage: {manifest.get('status')}")
        if manifest.get("config_fingerprint") != self.spec.config_fingerprint:
            raise ValueError("Orchestrator/config fingerprint drift on resume")
        orchestrator = manifest.get("orchestrator")
        if not isinstance(orchestrator, Mapping):
            raise ValueError("Lineage manifest lacks orchestrator state")
        if orchestrator.get("profile_fingerprint") != self.spec.profile_fingerprint:
            raise ValueError("Canonical profile fingerprint drift on resume")
        catalog_record = orchestrator.get("artifact_catalog")
        if isinstance(catalog_record, Mapping):
            catalog = ArtifactCatalog.load(
                self.paths.artifact_catalog,
                root=self.paths.root,
            )
            recorded_fingerprint = (
                expected_catalog_fingerprint
                if expected_catalog_fingerprint is not None
                else catalog_record.get("fingerprint")
            )
            if recorded_fingerprint != catalog.fingerprint:
                raise ValueError("Committed artifact catalog fingerprint drift")
        return manifest

    def _state(self) -> dict[str, object]:
        return read_json(self.paths.runtime_state)

    def _write_state(self, **updates: object) -> dict[str, object]:
        state = self._state()
        state.update(updates)
        state["updated_at"] = utc_now()
        atomic_write_json(self.paths.runtime_state, state)
        self._heartbeat(state)
        return state

    def _heartbeat(self, state: Mapping[str, object]) -> None:
        atomic_write_json(
            self.paths.heartbeat,
            {
                "at": utc_now(),
                "pid": os.getpid(),
                "lineage_id": self.lineage_id,
                "state": state.get("state"),
                "generation": state.get("active_generation"),
                "phase": state.get("active_phase"),
            },
        )

    def prepare_resume(self) -> None:
        """Explicitly reopen the same lineage after a safe/recoverable stop."""
        self._ensure_layout()
        with RunLock(self.paths.lock):
            self._recover_interrupted_commit()
            self._load_manifest()
            state = self._state()
            current = str(state.get("state"))
            if current not in RESUMABLE_STOP_STATES and current != "CREATED":
                raise RuntimeError(f"Lineage is not in an explicit-resume state: {current}")
            self.paths.stop_request.unlink(missing_ok=True)
            self._write_state(
                state="CREATED",
                active_generation=None,
                active_phase=None,
                pid=None,
                stop_requested=False,
                error=None,
            )
            self.events.emit("INFO", "Lineage explicitly prepared for resume", previous_state=current)

    def request_soft_stop(self, minutes: int | None = None, *, reason: str = "operator") -> dict[str, object]:
        # Never let a mistyped `stop` command create a manifest-less ghost run.
        self._load_manifest()
        selected = self.spec.soft_stop.default_minutes if minutes is None else int(minutes)
        selected = self.spec.soft_stop.validate(selected)
        now = datetime.now(timezone.utc)
        payload: dict[str, object] = {
            "schema": ORCHESTRATOR_SCHEMA,
            "requested_at": now.isoformat(),
            "requested_by": str(reason),
            "target_deadline_at": (now + timedelta(minutes=selected)).isoformat(),
            "window_minutes": selected,
            "mode": "finish-current-safe-boundary-no-hard-kill",
        }
        self.paths.control.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.paths.stop_request, payload)
        return payload

    def _stop_request(self) -> dict[str, object] | None:
        if not self.paths.stop_request.is_file():
            return None
        return read_json(self.paths.stop_request)

    def _generation_tx_path(self, generation: int) -> Path:
        return self.paths.generations / f"generation-{generation:04d}.json"

    def _generation_result_path(self, generation: int) -> Path:
        return self.paths.generations / f"generation-{generation:04d}-driver-result.json"

    def _arena_result_path(self, generation: int) -> Path:
        manifest = read_json(self.paths.manifest)
        parent = manifest.get("parent_checkpoint")
        arena_payload = self.spec.payload.get("arena")
        arena_config = arena_payload.get("driver_config") if isinstance(arena_payload, Mapping) else None
        reference_gap = arena_config.get("reference_gap") if isinstance(arena_config, Mapping) else None
        external_parent: Mapping[str, object] | None = None
        if isinstance(parent, Mapping) and reference_gap is not None:
            try:
                reference_generation = int(generation) - int(reference_gap)
                parent_generation = int(parent.get("generation", -1))
            except (TypeError, ValueError):
                reference_generation = -1
                parent_generation = -2
            if (
                parent_generation == reference_generation
                and str(parent.get("lineage_id", ""))
                and str(parent.get("lineage_id")) != self.lineage_id
            ):
                external_parent = parent
        if external_parent is not None:
            evaluation_id = evaluation_id_for_comparison(
                candidate_lineage_id=self.lineage_id,
                candidate_generation=int(generation),
                reference_lineage_id=str(external_parent["lineage_id"]),
                reference_generation=int(external_parent["generation"]),
            )
            root = evaluation_dir(self.spec.topology, evaluation_id)
            ensure_evaluation_layout(root)
            return root / "result.json"
        return self.paths.root / "arena" / f"generation-{generation:04d}" / "result.json"

    def _record_generation_artifacts(
        self,
        generation: int,
        result: Mapping[str, object],
        transaction_path: Path,
    ) -> str:
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("Generation result artifacts are required for catalog commit")
        catalog = ArtifactCatalog.load(
            self.paths.artifact_catalog,
            root=self.paths.root,
        )
        transaction = {
            "path": str(transaction_path.relative_to(self.paths.root)),
            "sha256": sha256_file(transaction_path),
            "size_bytes": transaction_path.stat().st_size,
            "status": "COMMITTED",
        }
        return catalog.register_generation(generation, artifacts, transaction=transaction)

    def _validate_artifact(self, item: Mapping[str, object]) -> tuple[str, str]:
        raw_path = str(item.get("path", ""))
        relative = _safe_relative_path(raw_path, label="driver artifact path")
        path = (self.paths.root / relative).resolve()
        if self.paths.root.resolve() not in path.parents:
            raise ValueError(f"Driver artifact escapes lineage: {raw_path}")
        if not path.is_file():
            raise ValueError(f"Required driver artifact is missing: {relative}")
        size = int(item.get("size_bytes", path.stat().st_size))
        if size != path.stat().st_size:
            raise ValueError(f"Driver artifact size mismatch: {relative}")
        actual = sha256_file(path)
        declared = str(item.get("sha256", actual))
        if declared != actual:
            raise ValueError(f"Driver artifact hash mismatch: {relative}")
        return str(relative), actual

    def _validate_generation_result(self, generation: int) -> dict[str, object]:
        result_path = self._generation_result_path(generation)
        if not result_path.is_file():
            raise ValueError(f"Generation driver did not publish result: {result_path}")
        result = read_json(result_path)
        if result.get("schema") != DRIVER_RESULT_SCHEMA:
            raise ValueError("Generation driver result schema mismatch")
        if int(result.get("generation", -1)) != generation or result.get("status") != "COMPLETED":
            raise ValueError("Generation driver did not complete the requested generation")
        if result.get("profile_fingerprint") != self.spec.profile_fingerprint:
            raise ValueError("Generation result profile fingerprint mismatch")
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("Generation result requires a non-empty artifacts list")
        checked: dict[str, str] = {}
        for item in artifacts:
            if not isinstance(item, Mapping):
                raise ValueError("Generation result artifact entry is malformed")
            relative, digest = self._validate_artifact(item)
            checked[relative] = digest
        if result.get("checkpoint_reload_verified") is not True:
            raise ValueError("Generation result must prove checkpoint reload verification")
        checkpoint = result.get("checkpoint")
        replay = result.get("replay")
        resume_state = result.get("resume_state")
        for label, value in (("checkpoint", checkpoint), ("replay", replay), ("resume_state", resume_state)):
            if not isinstance(value, Mapping) or not value.get("path"):
                raise ValueError(f"Generation result requires {label}.path")
            relative = str(value["path"])
            if relative not in checked:
                raise ValueError(f"{label} artifact must be present in validated artifacts: {relative}")
        components = resume_state.get("components") if isinstance(resume_state, Mapping) else None
        required_components = {"model", "optimizer", "replay", "generation", "rng"}
        if not isinstance(components, list) or not required_components.issubset({str(v) for v in components}):
            raise ValueError("Resume-state proof must include model, optimizer, replay, generation, and RNG")
        technical = int(result.get("technical_games", 0))
        invalid = int(result.get("invalid_games", 0))
        if technical != 0 or invalid != 0:
            raise ValueError(
                f"Generation {generation} contains technical/invalid games: technical={technical} invalid={invalid}"
            )
        metrics = result.get("metrics")
        if metrics is not None and not isinstance(metrics, Mapping):
            raise ValueError("Generation metrics must be a mapping")
        result["validated_artifact_hashes"] = checked
        return result

    def _validate_arena_result(self, generation: int) -> dict[str, object]:
        result_path = self._arena_result_path(generation)
        if not result_path.is_file():
            raise ValueError(f"Arena driver did not publish result: {result_path}")
        result = read_json(result_path)
        if result.get("schema") != ARENA_RESULT_SCHEMA:
            raise ValueError("Arena driver result schema mismatch")
        if int(result.get("generation", -1)) != generation or result.get("status") != "COMPLETED":
            raise ValueError("Arena result does not match requested generation")
        if result.get("profile_fingerprint") != self.spec.profile_fingerprint:
            raise ValueError("Arena result profile fingerprint mismatch")
        if int(result.get("technical_games", 0)) != 0 or int(result.get("invalid_games", 0)) != 0:
            raise ValueError("Arena technical/invalid outcomes are fail-closed")
        if result.get("training_mutated") is not False:
            raise ValueError("Arena must explicitly prove that it did not mutate training state")
        if self.spec.arena_preset_fingerprint is not None and result.get("preset_fingerprint") != self.spec.arena_preset_fingerprint:
            raise ValueError("Arena frozen preset fingerprint mismatch")
        if self.spec.arena_startset_fingerprint is not None and result.get("startset_fingerprint") != self.spec.arena_startset_fingerprint:
            raise ValueError("Arena frozen startset fingerprint mismatch")
        return result

    def _driver_env(self, generation: int, *, resume: bool, phase: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "AZ_ORCHESTRATOR_SCHEMA": ORCHESTRATOR_SCHEMA,
                "AZ_LINEAGE_ID": self.lineage_id,
                "AZ_TOPOLOGY": self.spec.topology,
                "AZ_RUN_ROOT": str(self.paths.root),
                "AZ_PROFILE_PATH": str(self.spec.profile_path),
                "AZ_PROFILE_FINGERPRINT": self.spec.profile_fingerprint,
                "AZ_GENERATION": str(generation),
                "AZ_GENERATION04": f"{generation:04d}",
                "AZ_RESUME": "1" if resume else "0",
                "AZ_PHASE": phase,
                "AZ_SOFT_STOP_REQUEST_PATH": str(self.paths.stop_request),
                "AZ_DRIVER_HEARTBEAT_PATH": str(self.paths.driver_heartbeat),
                "AZ_GENERATION_RESULT_PATH": str(self._generation_result_path(generation)),
                "AZ_ARENA_RESULT_PATH": str(self._arena_result_path(generation)),
            }
        )
        return env

    def _health_snapshot(self, child: subprocess.Popen[bytes] | subprocess.Popen[str]) -> dict[str, object]:
        disk = shutil.disk_usage(self.paths.root)
        disk_gb = disk.free / (1024 ** 3)
        ram_gb = _memory_available_gb()
        heartbeat_age: float | None = None
        driver_health: dict[str, object] | None = None
        if self.paths.driver_heartbeat.is_file():
            heartbeat_age = max(0.0, time.time() - self.paths.driver_heartbeat.stat().st_mtime)
            try:
                driver_health = read_json(self.paths.driver_heartbeat)
            except Exception:
                driver_health = {"parse_error": True}
        return {
            "at": utc_now(),
            "child_pid": child.pid,
            "disk_free_gb": disk_gb,
            "ram_free_gb": ram_gb,
            "driver_heartbeat_age_sec": heartbeat_age,
            "driver_health": driver_health,
        }

    def _emit_health_warnings(self, snapshot: Mapping[str, object]) -> None:
        disk = float(snapshot["disk_free_gb"])
        ram = snapshot.get("ram_free_gb")
        heartbeat_age = snapshot.get("driver_heartbeat_age_sec")
        if disk <= self.spec.health.min_disk_free_gb_critical:
            self.events.emit("CRITICAL", f"Disk free space critically low: {disk:.2f} GiB")
            if not self.paths.stop_request.exists():
                self.request_soft_stop(reason="critical-disk")
        elif disk <= self.spec.health.min_disk_free_gb_warning:
            self.events.emit("WARNING", f"Disk free space low: {disk:.2f} GiB")
        if ram is not None:
            ram_value = float(ram)
            if ram_value <= self.spec.health.min_ram_free_gb_critical:
                self.events.emit("CRITICAL", f"RAM available critically low: {ram_value:.2f} GiB")
                if not self.paths.stop_request.exists():
                    self.request_soft_stop(reason="critical-ram")
            elif ram_value <= self.spec.health.min_ram_free_gb_warning:
                self.events.emit("WARNING", f"RAM available low: {ram_value:.2f} GiB")
        if heartbeat_age is not None:
            age = float(heartbeat_age)
            if age >= self.spec.health.heartbeat_critical_seconds:
                self.events.emit("CRITICAL", f"Driver heartbeat stale for {age:.0f}s; no hard kill issued")
            elif age >= self.spec.health.heartbeat_warning_seconds:
                self.events.emit("WARNING", f"Driver heartbeat stale for {age:.0f}s")
        driver_health = snapshot.get("driver_health")
        if isinstance(driver_health, Mapping):
            expected = driver_health.get("workers_expected")
            alive = driver_health.get("workers_alive")
            if isinstance(expected, int) and isinstance(alive, int) and alive < expected:
                self.events.emit("CRITICAL", f"Worker health degraded: {alive}/{expected} alive")
                if not self.paths.stop_request.exists():
                    self.request_soft_stop(reason="critical-worker-health")
            if driver_health.get("inference_alive") is False:
                self.events.emit("CRITICAL", "Inference owner heartbeat reports not alive")
                if not self.paths.stop_request.exists():
                    self.request_soft_stop(reason="critical-inference-health")
            errors = driver_health.get("errors")
            if isinstance(errors, list) and errors:
                self.events.emit("CRITICAL", f"Driver reported runtime errors: {errors[-1]}")
                if not self.paths.stop_request.exists():
                    self.request_soft_stop(reason="critical-driver-error")

    def _run_child(self, command: Sequence[str], *, generation: int, resume: bool, phase: str) -> int:
        rendered = _render_command(
            command,
            generation=generation,
            generation04=f"{generation:04d}",
            lineage_id=self.lineage_id,
            run_root=str(self.paths.root),
            profile_path=str(self.spec.profile_path),
        )
        self.events.emit("INFO", f"Starting {phase}", generation=generation, argv=rendered)
        process = subprocess.Popen(
            rendered,
            cwd=self.repo_root,
            env=self._driver_env(generation, resume=resume, phase=phase),
            start_new_session=True,
        )
        while True:
            code = process.poll()
            if code is not None:
                return int(code)
            state = self._state()
            self._heartbeat(state)
            snapshot = self._health_snapshot(process)
            atomic_write_json(self.paths.metrics / "health-latest.json", snapshot)
            self._emit_health_warnings(snapshot)
            stop = self._stop_request()
            if stop is not None:
                target = parse_utc(str(stop["target_deadline_at"]))
                if datetime.now(timezone.utc) > target:
                    self.events.emit(
                        "WARNING",
                        "Soft-stop target window exceeded; current phase is still allowed to finish safely",
                        generation=generation,
                        phase=phase,
                    )
            time.sleep(self.spec.health.poll_seconds)

    def _append_metrics(self, kind: str, generation: int, metrics: Mapping[str, object]) -> None:
        path = self.paths.metrics / "history.jsonl"
        payload = {"at": utc_now(), "kind": kind, "generation": generation, "metrics": dict(metrics)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _lookup_number(payload: Mapping[str, object], dotted: str) -> float | None:
        current: object = payload
        for key in dotted.split("."):
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            return float(current)
        return None

    def _check_performance(self, generation: int, metrics: Mapping[str, object]) -> None:
        checks = self.spec.performance.get("checks")
        if not isinstance(checks, list):
            return
        for item in checks:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("metric", ""))
            value = self._lookup_number(metrics, name)
            baseline = item.get("baseline")
            warning_ratio = float(item.get("warning_ratio", 0.85))
            fail_ratio = float(item.get("fail_ratio", 0.70))
            policy = str(item.get("policy", "warning"))
            if value is None or not isinstance(baseline, (int, float)) or float(baseline) <= 0:
                continue
            ratio = value / float(baseline)
            if ratio < fail_ratio:
                level = "CRITICAL" if policy == "fail-closed" else "WARNING"
                self.events.emit(level, f"Performance regression {name}: {value:.4g} ({ratio:.1%} of baseline)", generation=generation)
                if policy == "fail-closed":
                    raise RuntimeError(f"Performance fail-closed threshold breached for {name}")
            elif ratio < warning_ratio:
                self.events.emit("WARNING", f"Performance degraded {name}: {value:.4g} ({ratio:.1%} of baseline)", generation=generation)

    def _history_rows(self) -> list[dict[str, object]]:
        history_path = self.paths.metrics / "history.jsonl"
        rows: list[dict[str, object]] = []
        if not history_path.is_file():
            return rows
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _event_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if not self.paths.events.is_file():
            return rows
        for line in self.paths.events.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _check_required_metrics(self, metrics: Mapping[str, object], *, kind: str) -> None:
        key = "required_generation_metrics" if kind == "generation" else "required_arena_metrics"
        required = self.spec.payload.get(key)
        if not isinstance(required, list):
            return
        missing = [
            str(name) for name in required
            if self._lookup_number(metrics, str(name)) is None
        ]
        if missing:
            raise ValueError(f"{kind} metrics missing required numeric values: {', '.join(missing)}")

    def _learning_stall_checks(self) -> list[dict[str, object]]:
        checks = self.spec.learning.get("stall_checks")
        if not isinstance(checks, list):
            return []
        history = self._history_rows()
        findings: list[dict[str, object]] = []
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            kind = str(check.get("kind", "generation"))
            metric = str(check.get("metric", ""))
            window = max(2, int(check.get("window", 2)))
            minimum = float(check.get("minimum_delta", 0.0))
            direction = str(check.get("direction", "increase"))
            rows = [row for row in history if row.get("kind") == kind][-window:]
            values: list[float] = []
            for row in rows:
                payload = row.get("metrics")
                if isinstance(payload, Mapping):
                    value = self._lookup_number(payload, metric)
                    if value is not None:
                        values.append(value)
            if len(values) < window:
                continue
            raw_delta = values[-1] - values[0]
            progress = raw_delta if direction == "increase" else -raw_delta
            stalled = progress < minimum
            finding = {
                "kind": kind, "metric": metric, "window": window,
                "first": values[0], "last": values[-1],
                "progress_delta": progress, "minimum_delta": minimum,
                "status": "STALL" if stalled else "OK",
            }
            findings.append(finding)
            if stalled:
                policy = str(check.get("policy", "warning"))
                level = "CRITICAL" if policy == "fail-closed" else "WARNING"
                self.events.emit(
                    level,
                    f"Learning progress stalled for {kind}.{metric}: {progress:.4g} < required {minimum:.4g}",
                )
                if policy == "fail-closed":
                    raise RuntimeError(f"Learning-stall fail-closed threshold breached for {kind}.{metric}")
        return findings

    def _learning_summary(self) -> dict[str, object]:
        history_path = self.paths.metrics / "history.jsonl"
        if not history_path.is_file():
            return {"status": "NO_DATA"}
        rows: list[dict[str, object]] = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("kind") == "generation":
                rows.append(row)
        fields = self.spec.learning.get("metrics")
        metric_names = [str(value) for value in fields] if isinstance(fields, list) else []
        deltas: dict[str, object] = {}
        if len(rows) >= 2:
            first_metrics = rows[0].get("metrics")
            last_metrics = rows[-1].get("metrics")
            if isinstance(first_metrics, Mapping) and isinstance(last_metrics, Mapping):
                for name in metric_names:
                    first = self._lookup_number(first_metrics, name)
                    last = self._lookup_number(last_metrics, name)
                    if first is not None and last is not None:
                        deltas[name] = {"first": first, "last": last, "delta": last - first}
        return {"status": "OK", "generations": len(rows), "deltas": deltas}

    def _update_manifest(
        self,
        *,
        committed_generation: int,
        arena_generation: int | None = None,
        artifact_hashes: Mapping[str, object] | None = None,
        catalog_fingerprint: str | None = None,
    ) -> None:
        manifest = self._load_manifest(expected_catalog_fingerprint=catalog_fingerprint)
        orchestrator = dict(manifest["orchestrator"])  # type: ignore[arg-type]
        orchestrator["last_committed_generation"] = int(committed_generation)
        orchestrator["runtime_state"] = self._state().get("state")
        arena_generations = [int(v) for v in orchestrator.get("arena_generations", [])]
        if arena_generation is not None and arena_generation not in arena_generations:
            arena_generations.append(int(arena_generation))
            arena_generations.sort()
        orchestrator["arena_generations"] = arena_generations
        if catalog_fingerprint is not None:
            catalog_record = dict(orchestrator.get("artifact_catalog", {}))
            catalog_record["schema"] = ARTIFACT_CATALOG_SCHEMA
            catalog_record["path"] = "runtime/artifact-catalog.json"
            catalog_record["fingerprint"] = str(catalog_fingerprint)
            orchestrator["artifact_catalog"] = catalog_record
        manifest["orchestrator"] = orchestrator
        checkpoint_hashes = dict(manifest.get("checkpoint_hashes", {}))
        if artifact_hashes is not None:
            for path, digest in artifact_hashes.items():
                if str(path).startswith("checkpoints/"):
                    checkpoint_hashes[str(path)] = str(digest)
        manifest["checkpoint_hashes"] = checkpoint_hashes
        atomic_write_json(self.paths.manifest, manifest)

    def _render_report(self, *, final: bool = False) -> None:
        state = self._state()
        manifest = self._load_manifest()
        orchestrator = manifest.get("orchestrator")
        latest_warning = _latest_event(self.paths.events, {"WARNING", "CRITICAL"})
        learning = self._learning_summary()
        history = self._history_rows()
        events = self._event_rows()
        warnings = [event for event in events if event.get("level") in {"WARNING", "CRITICAL"}]
        restarts = [event for event in events if event.get("message") == "Resuming interrupted generation"]
        lines = [
            f"# Training lineage `{self.lineage_id}`",
            "",
            f"- Topology: `{self.spec.topology}`",
            f"- Profile: `{self.spec.profile_payload.get('profile_id', self.spec.profile_path.name)}`",
            f"- Profile fingerprint: `{self.spec.profile_fingerprint}`",
            f"- Config fingerprint: `{self.spec.config_fingerprint}`",
            f"- Git commit: `{manifest.get('git_commit')}`",
            f"- State: **{state.get('state')}**",
            f"- Last committed generation: **{state.get('last_committed_generation', 0)}**",
            f"- Arena cadence: every **{self.spec.arena_every_generations}** generations",
            f"- Warnings / critical events: **{len(warnings)}**",
            f"- Interrupted-generation resumes: **{len(restarts)}**",
        ]
        if isinstance(orchestrator, Mapping):
            lines.append(f"- Arena generations: `{orchestrator.get('arena_generations', [])}`")
            catalog_record = orchestrator.get("artifact_catalog")
            if isinstance(catalog_record, Mapping):
                lines.append(
                    f"- Artifact catalog: `{catalog_record.get('path')}`"
                    f" ({catalog_record.get('fingerprint')})"
                )
        if latest_warning is not None:
            lines.append(f"- Latest warning: `{latest_warning.get('at')}` — {latest_warning.get('message')}")
        lines.extend(["", "## Learning velocity", "", "```json", json.dumps(learning, indent=2, sort_keys=True), "```", ""])
        atomic_write_text(self.paths.report_md, "\n".join(lines))
        if final:
            final_payload = {
                "schema": ORCHESTRATOR_SCHEMA,
                "lineage_id": self.lineage_id,
                "topology": self.spec.topology,
                "state": state.get("state"),
                "last_committed_generation": state.get("last_committed_generation", 0),
                "profile_fingerprint": self.spec.profile_fingerprint,
                "config_fingerprint": self.spec.config_fingerprint,
                "git_commit": manifest.get("git_commit"),
                "checkpoint_hashes": manifest.get("checkpoint_hashes", {}),
                "learning_velocity": learning,
                "latest_warning": latest_warning,
                "warning_events": warnings,
                "restart_events": restarts,
                "metric_history": history,
                "generation_metrics": [row for row in history if row.get("kind") == "generation"],
                "arena_metrics": [row for row in history if row.get("kind") == "arena"],
                "artifact_catalog": (
                    dict(orchestrator.get("artifact_catalog", {}))
                    if isinstance(orchestrator, Mapping)
                    and isinstance(orchestrator.get("artifact_catalog"), Mapping)
                    else None
                ),
                "finished_at": utc_now(),
            }
            atomic_write_json(self.paths.final_json, final_payload)
            atomic_write_text(self.paths.final_md, "\n".join(lines + ["## Final", "", f"Finished: `{final_payload['finished_at']}`", ""]))

    def _run_arena(self, generation: int) -> None:
        if self.spec.arena_command is None:
            if self.spec.arena_required:
                raise RuntimeError("Periodic Arena is required but no Arena command is configured")
            return
        result_path = self._arena_result_path(generation)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_state(active_phase="arena")
        code = self._run_child(self.spec.arena_command, generation=generation, resume=False, phase="arena")
        if code != 0:
            raise RuntimeError(f"Arena driver exited with code {code}")
        result = self._validate_arena_result(generation)
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            self._check_required_metrics(metrics, kind="arena")
            self._append_metrics("arena", generation, metrics)
            performance_status = str(metrics.get("performance_status", "HEALTHY"))
            if performance_status == "WARNING":
                self.events.emit(
                    "WARNING",
                    "Arena performance below healthy target but above hard minimum; Arena accepted",
                    generation=generation,
                    mean_inference_batch_rows=metrics.get("inference_mean_batch_rows"),
                    performance_gate=metrics.get("performance_gate"),
                )
            elif performance_status == "CRITICAL":
                raise RuntimeError("Arena performance policy reported CRITICAL")
        self._update_manifest(committed_generation=generation, arena_generation=generation)
        self.events.emit("INFO", "Arena completed", generation=generation)

    def _ensure_pending_arena(self, committed_generation: int) -> None:
        if not self._arena_due(committed_generation):
            return
        manifest = self._load_manifest()
        orchestrator = manifest.get("orchestrator")
        completed: set[int] = set()
        if isinstance(orchestrator, Mapping):
            completed = {int(value) for value in orchestrator.get("arena_generations", [])}
        if committed_generation not in completed:
            self.events.emit(
                "WARNING",
                "Required Arena is pending from a previously committed generation; running it before further training",
                generation=committed_generation,
            )
            self._run_arena(committed_generation)

    def _generation_origin(self) -> int:
        state = self._state()
        try:
            return max(0, int(state.get("generation_origin", 0)))
        except (TypeError, ValueError) as exc:
            raise ValueError("Runtime generation origin is malformed") from exc

    def _arena_due(self, generation: int) -> bool:
        origin = self._generation_origin()
        return (
            int(generation) > origin
            and (int(generation) - origin) % self.spec.arena_every_generations == 0
        )

    def _run_generation(self, generation: int) -> None:
        tx_path = self._generation_tx_path(generation)
        resume = False
        command = self.spec.generation_command
        if tx_path.is_file():
            tx = read_json(tx_path)
            status = tx.get("status")
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
        tx = {
            "schema": ORCHESTRATOR_SCHEMA,
            "generation": generation,
            "status": "RUNNING",
            "started_at": utc_now(),
            "resume": resume,
            "profile_fingerprint": self.spec.profile_fingerprint,
        }
        atomic_write_json(tx_path, tx)
        self._write_state(active_generation=generation, active_phase="generation")
        code = self._run_child(command, generation=generation, resume=resume, phase="generation")
        if code != 0:
            tx.update({"status": "FAILED", "finished_at": utc_now(), "exit_code": code})
            atomic_write_json(tx_path, tx)
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
                "result_path": str(self._generation_result_path(generation).relative_to(self.paths.root)),
                "artifact_hashes": result["validated_artifact_hashes"],
            }
        )
        atomic_write_json(tx_path, tx)
        catalog_fingerprint = self._record_generation_artifacts(
            generation,
            result,
            tx_path,
        )
        self._write_state(last_committed_generation=generation, active_phase="commit")
        self._update_manifest(
            committed_generation=generation,
            artifact_hashes=result["validated_artifact_hashes"],
            catalog_fingerprint=catalog_fingerprint,
        )
        self.events.emit("INFO", "Generation committed", generation=generation)
        if isinstance(metrics, Mapping):
            self._check_performance(generation, metrics)
        self._learning_stall_checks()
        if self._arena_due(generation):
            self._run_arena(generation)
        self._render_report()

    def run(self, *, max_generations: int | None = None) -> None:
        self._ensure_layout()
        with RunLock(self.paths.lock):
            self._recover_interrupted_commit()
            self._load_manifest()
            previous_handlers: dict[int, object] = {}
            def _soft_signal(signum: int, _frame: object) -> None:
                if not self.paths.stop_request.exists():
                    self.request_soft_stop(reason=f"signal-{signum}")
                    self.events.emit(
                        "WARNING",
                        f"Signal {signum} converted to durable soft-stop request; active phase continues",
                    )
            for signum in (signal.SIGINT, signal.SIGTERM):
                try:
                    previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, _soft_signal)
                except (ValueError, OSError):
                    pass
            initial = self._state()
            if initial.get("state") in RESUMABLE_STOP_STATES:
                raise RuntimeError(
                    f"Run is stopped in state {initial.get('state')}; use explicit resume before restarting"
                )
            self._write_state(
                state="RUNNING",
                pid=os.getpid(),
                started_at=initial.get("started_at") or utc_now(),
                active_phase="startup",
            )
            self.events.emit("INFO", "Training orchestrator started", pid=os.getpid())
            try:
                while True:
                    state = self._state()
                    committed = int(state.get("last_committed_generation", 0))
                    stop = self._stop_request()
                    if stop is not None:
                        self._write_state(
                            state="SOFT_STOPPED",
                            stop_requested=True,
                            active_generation=None,
                            active_phase=None,
                            pid=None,
                        )
                        self._update_manifest(committed_generation=committed)
                        self.events.emit(
                            "INFO",
                            "Soft stop reached a safe generation boundary",
                            last_committed_generation=committed,
                        )
                        self._render_report(final=True)
                        return
                    self._ensure_pending_arena(committed)
                    if max_generations is not None and committed >= int(max_generations):
                        self._write_state(
                            state="COMPLETED",
                            active_generation=None,
                            active_phase=None,
                            pid=None,
                        )
                        self._update_manifest(committed_generation=committed)
                        self.events.emit("INFO", "Requested generation limit completed", generation=committed)
                        self._render_report(final=True)
                        return
                    self._run_generation(committed + 1)
            except KeyboardInterrupt:
                self.request_soft_stop(reason="keyboard-interrupt")
                self.events.emit(
                    "WARNING",
                    "Keyboard interrupt converted to durable soft-stop request; current safe boundary preserved",
                )
                self._write_state(state="SOFT_STOPPED", stop_requested=True, pid=None)
                self._render_report(final=True)
                return
            except Exception as exc:
                state = self._state()
                self._write_state(state="RECOVERY_REQUIRED", active_phase=None, pid=None, error=str(exc))
                self.events.emit("CRITICAL", f"Orchestrator failed closed and requires explicit resume: {exc}")
                self._render_report(final=True)
                raise

    def status(self) -> dict[str, object]:
        manifest = self._load_manifest()
        state = self._state()
        heartbeat_age: float | None = None
        if self.paths.heartbeat.is_file():
            heartbeat_age = max(0.0, time.time() - self.paths.heartbeat.stat().st_mtime)
        latest_warning = _latest_event(self.paths.events, {"WARNING", "CRITICAL"})
        return {
            "lineage_id": self.lineage_id,
            "topology": self.spec.topology,
            "state": state.get("state"),
            "generation": state.get("active_generation"),
            "phase": state.get("active_phase"),
            "last_committed_generation": state.get("last_committed_generation", 0),
            "generation_origin": state.get("generation_origin", 0),
            "next_generation": int(state.get("last_committed_generation", 0)) + 1,
            "parent_checkpoint": manifest.get("parent_checkpoint"),
            "pid": state.get("pid"),
            "heartbeat_age_sec": heartbeat_age,
            "stop_request": self._stop_request(),
            "latest_warning": latest_warning,
            "profile_fingerprint": self.spec.profile_fingerprint,
            "config_fingerprint": self.spec.config_fingerprint,
            "checkpoint_count": len(manifest.get("checkpoint_hashes", {})),
            "report": str(self.paths.report_md),
        }


def format_status(status: Mapping[str, object]) -> str:
    lines = [
        f"Lineage: {status.get('lineage_id')}",
        f"Topology: {status.get('topology')}",
        f"State: {status.get('state')}",
        f"Generation: {status.get('generation') or '-'} (committed {status.get('last_committed_generation')})",
        f"Phase: {status.get('phase') or '-'}",
        f"PID: {status.get('pid') or '-'}",
    ]
    heartbeat = status.get("heartbeat_age_sec")
    lines.append(f"Heartbeat: {float(heartbeat):.1f}s ago" if heartbeat is not None else "Heartbeat: -")
    stop = status.get("stop_request")
    if isinstance(stop, Mapping):
        lines.append(
            f"Soft stop: requested, target {stop.get('target_deadline_at')} ({stop.get('window_minutes')} min)"
        )
    else:
        lines.append("Soft stop: no")
    warning = status.get("latest_warning")
    if isinstance(warning, Mapping):
        lines.append(f"Latest warning: {warning.get('at')} {warning.get('message')}")
    lines.append(f"Checkpoints: {status.get('checkpoint_count')}")
    lines.append(f"Report: {status.get('report')}")
    return "\n".join(lines)


__all__ = [
    "ARENA_RESULT_SCHEMA",
    "DRIVER_RESULT_SCHEMA",
    "HealthPolicy",
    "ORCHESTRATOR_SCHEMA",
    "OrchestratorSpec",
    "ProductionTrainingOrchestrator",
    "RunPaths",
    "SoftStopPolicy",
    "atomic_write_json",
    "format_status",
    "sha256_file",
]
