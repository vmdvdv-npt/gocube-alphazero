"""Strict immutable run-spec boundary for universal production orchestration.

Every run-specific decision is explicit in one immutable JSON document.  The
orchestrator never chooses topology, board size, workload, Arena cadence,
execution tuning, health thresholds, seeds or performance gates implicitly.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .orchestrator import (
    HealthPolicy,
    OrchestratorSpec,
    SoftStopPolicy,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .production_orchestrator import (
    SupervisionPolicy,
    UniversalProductionTrainingOrchestrator,
)

RUN_SPEC_SCHEMA = "gocube-production-run-spec-v3"
RUN_SPEC_FILENAME = "run-spec.json"


def run_spec_fingerprint(payload: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _require_keys(value: Mapping[str, object], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} is missing explicit fields: {', '.join(missing)}")


def _argv(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty argv list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} tokens must be non-empty strings")
    return tuple(value)


def _safe_component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


def _safe_lineage_id(value: str) -> str:
    return _safe_component(value, "lineage id")


def _resolve_profile(
    repo_root: Path, payload: Mapping[str, object]
) -> tuple[Path, dict[str, object], str]:
    profile_ref = str(payload.get("profile_path", "")).strip()
    if not profile_ref or profile_ref.lower().endswith(".xlsx"):
        raise ValueError("profile_path must reference JSON and must never use XLSX")
    relative = Path(profile_ref)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("profile_path must be repo-relative")
    profile_path = (repo_root / relative).resolve()
    if repo_root.resolve() not in profile_path.parents:
        raise ValueError("profile_path escapes repository root")
    profile = read_json(profile_path)
    if profile.get("profile_id") == "gocube-torus9-golden-v3":
        # Torus9 has a machine-checked scientific contract.  Resolve it here
        # as well as in the child driver so a stale embedded fingerprint cannot
        # make an invalid one-shot run-spec look acceptable.
        from .torus9_contract import load_torus9_current_profile

        profile = load_torus9_current_profile(profile_path)
        actual = str(profile["profile_fingerprint"])
    else:
        actual = str(
            profile.get("profile_fingerprint")
            or profile.get("content_fingerprint")
            or sha256_file(profile_path)
        )
    expected = str(payload.get("expected_profile_fingerprint", "")).strip()
    if not expected:
        raise ValueError(
            "expected_profile_fingerprint is required; implicit profile acceptance is forbidden"
        )
    if expected != actual:
        raise ValueError(
            f"profile fingerprint mismatch: expected {expected}, got {actual}"
        )
    return profile_path, profile, actual


def _health(value: object) -> HealthPolicy:
    block = _mapping(value, "health")
    fields = tuple(HealthPolicy.__dataclass_fields__.keys())
    _require_keys(block, fields, "health")
    return HealthPolicy.from_mapping(block)


def _soft_stop(value: object) -> SoftStopPolicy:
    block = _mapping(value, "soft_stop")
    fields = tuple(SoftStopPolicy.__dataclass_fields__.keys())
    _require_keys(block, fields, "soft_stop")
    return SoftStopPolicy.from_mapping(block)


def _performance(value: object) -> dict[str, object]:
    block = _mapping(value, "performance")
    _require_keys(block, ("checks",), "performance")
    checks = block["checks"]
    if not isinstance(checks, list):
        raise ValueError("performance.checks must be an explicit list (empty is allowed)")
    for index, item in enumerate(checks):
        check = _mapping(item, f"performance.checks[{index}]")
        _require_keys(
            check,
            ("metric", "baseline", "warning_ratio", "fail_ratio", "policy"),
            f"performance.checks[{index}]",
        )
        baseline = float(check["baseline"])
        warning_ratio = float(check["warning_ratio"])
        fail_ratio = float(check["fail_ratio"])
        if baseline <= 0:
            raise ValueError("performance baseline must be positive")
        if not 0 <= fail_ratio <= warning_ratio:
            raise ValueError(
                "performance ratios require 0 <= fail_ratio <= warning_ratio"
            )
        if str(check["policy"]) not in {"warning", "fail-closed"}:
            raise ValueError("performance policy must be warning or fail-closed")
    return block


def _learning(value: object) -> dict[str, object]:
    block = _mapping(value, "learning")
    _require_keys(block, ("metrics", "stall_checks"), "learning")
    if not isinstance(block["metrics"], list):
        raise ValueError("learning.metrics must be an explicit list")
    checks = block["stall_checks"]
    if not isinstance(checks, list):
        raise ValueError("learning.stall_checks must be an explicit list (empty is allowed)")
    for index, item in enumerate(checks):
        check = _mapping(item, f"learning.stall_checks[{index}]")
        _require_keys(
            check,
            ("kind", "metric", "window", "minimum_delta", "direction", "policy"),
            f"learning.stall_checks[{index}]",
        )
        if int(check["window"]) < 2:
            raise ValueError("learning stall window must be >= 2")
        if str(check["direction"]) not in {"increase", "decrease"}:
            raise ValueError("learning stall direction must be increase or decrease")
        if str(check["policy"]) not in {"warning", "fail-closed"}:
            raise ValueError("learning stall policy must be warning or fail-closed")
    return block


def _metric_list(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an explicit list (empty is allowed)")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{key} entries must be non-empty strings")
    return list(value)


def _validate_adapter(value: object) -> dict[str, object]:
    block = _mapping(value, "adapter")
    _require_keys(block, ("id", "transport"), "adapter")
    _safe_component(block["id"], "adapter.id")
    if block["transport"] != "process":
        raise ValueError("production adapter.transport currently must be 'process'")
    return block


@dataclass(frozen=True)
class StrictRunSpec:
    source_path: Path
    payload: Mapping[str, object]
    fingerprint: str
    orchestrator_spec: OrchestratorSpec
    supervision: SupervisionPolicy
    board_size: int
    adapter_id: str

    @classmethod
    def load(cls, path: str | Path, *, repo_root: str | Path) -> "StrictRunSpec":
        root = Path(repo_root).resolve()
        source = Path(path)
        if not source.is_absolute():
            source = (root / source).resolve()
        payload = read_json(source)
        if payload.get("schema") != RUN_SPEC_SCHEMA:
            raise ValueError(f"Unsupported run-spec schema: {payload.get('schema')!r}")
        _require_keys(
            payload,
            (
                "schema",
                "topology",
                "board_size",
                "adapter",
                "profile_path",
                "expected_profile_fingerprint",
                "generation",
                "arena",
                "health",
                "supervision",
                "soft_stop",
                "performance",
                "learning",
                "required_generation_metrics",
                "required_arena_metrics",
            ),
            "run spec",
        )
        topology = _safe_component(payload["topology"], "topology")
        board_size = int(payload["board_size"])
        if board_size <= 0:
            raise ValueError("board_size must be positive")
        adapter = _validate_adapter(payload["adapter"])
        adapter_id = str(adapter["id"])
        profile_path, profile, profile_fp = _resolve_profile(root, payload)

        generation = _mapping(payload["generation"], "generation")
        _require_keys(
            generation, ("command", "resume_command", "driver_config"), "generation"
        )
        generation_command = _argv(generation["command"], "generation.command")
        resume_command = _argv(generation["resume_command"], "generation.resume_command")
        generation_config = _mapping(
            generation["driver_config"], "generation.driver_config"
        )
        _require_keys(generation_config, ("games",), "generation.driver_config")
        if int(generation_config["games"]) <= 0:
            raise ValueError("generation.driver_config.games must be positive")

        arena = _mapping(payload["arena"], "arena")
        _require_keys(
            arena,
            (
                "enabled",
                "required",
                "every_generations",
                "command",
                "driver_config",
                "startset",
            ),
            "arena",
        )
        enabled = arena["enabled"]
        required = arena["required"]
        if not isinstance(enabled, bool) or not isinstance(required, bool):
            raise ValueError("arena.enabled and arena.required must be booleans")
        if required and not enabled:
            raise ValueError("arena.required cannot be true when arena.enabled is false")
        every = int(arena["every_generations"])
        if every <= 0:
            raise ValueError("arena.every_generations must be positive")
        arena_command: tuple[str, ...] | None = None
        if enabled:
            arena_command = _argv(arena["command"], "arena.command")
        elif arena["command"] not in (None, []):
            raise ValueError("arena.command must be null/empty when arena is disabled")
        arena_driver_config = _mapping(arena["driver_config"], "arena.driver_config")
        arena_startset = _mapping(arena["startset"], "arena.startset")

        health = _health(payload["health"])
        supervision = SupervisionPolicy.from_mapping(
            _mapping(payload["supervision"], "supervision")
        )
        soft_stop = _soft_stop(payload["soft_stop"])
        performance = _performance(payload["performance"])
        learning = _learning(payload["learning"])
        _metric_list(payload, "required_generation_metrics")
        _metric_list(payload, "required_arena_metrics")

        fingerprint = run_spec_fingerprint(payload)
        orchestrator_spec = OrchestratorSpec(
            path=root / ".runtime-run-spec-v3.json",
            payload=deepcopy(payload),
            topology=topology,
            profile_path=profile_path,
            profile_payload=profile,
            profile_fingerprint=profile_fp,
            config_fingerprint=fingerprint,
            generation_command=generation_command,
            generation_resume_command=resume_command,
            arena_command=arena_command,
            arena_every_generations=every,
            arena_required=bool(enabled and required),
            arena_preset_fingerprint=(
                run_spec_fingerprint(arena_driver_config) if enabled else None
            ),
            arena_startset_fingerprint=(
                run_spec_fingerprint(arena_startset) if enabled else None
            ),
            health=health,
            soft_stop=soft_stop,
            performance=performance,
            learning=learning,
        )
        return cls(
            source_path=source,
            payload=deepcopy(payload),
            fingerprint=fingerprint,
            orchestrator_spec=orchestrator_spec,
            supervision=supervision,
            board_size=board_size,
            adapter_id=adapter_id,
        )


def discover_active_lineage(repo_root: str | Path, lineage_id: str) -> Path:
    root = Path(repo_root).resolve()
    lineage = _safe_lineage_id(lineage_id)
    matches = [
        path.parent for path in (root / "runs").glob(f"*/active/{lineage}/manifest.json")
    ]
    if not matches:
        raise FileNotFoundError(f"Active lineage not found: {lineage}")
    if len(matches) != 1:
        raise RuntimeError(f"Lineage id is ambiguous across topologies: {lineage}")
    return matches[0]


def load_persisted_run_spec(
    *, repo_root: str | Path, lineage_id: str
) -> StrictRunSpec:
    root = Path(repo_root).resolve()
    lineage_root = discover_active_lineage(root, lineage_id)
    manifest = read_json(lineage_root / "manifest.json")
    run_spec_meta = manifest.get("run_spec")
    if not isinstance(run_spec_meta, Mapping):
        raise ValueError("Lineage manifest has no immutable run_spec record")
    saved = lineage_root / RUN_SPEC_FILENAME
    spec = StrictRunSpec.load(saved, repo_root=root)
    expected = str(run_spec_meta.get("fingerprint", ""))
    if expected != spec.fingerprint or manifest.get("config_fingerprint") != spec.fingerprint:
        raise ValueError("Persisted run-spec fingerprint does not match lineage manifest")
    return spec


class StrictProductionTrainingOrchestrator(UniversalProductionTrainingOrchestrator):
    """Production entrypoint bound to an immutable lineage-owned run spec."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        run_spec: StrictRunSpec,
        lineage_id: str,
        terminal: bool = True,
    ) -> None:
        self.strict_run_spec = run_spec
        super().__init__(
            repo_root=repo_root,
            spec=run_spec.orchestrator_spec,
            lineage_id=lineage_id,
            terminal=terminal,
            supervision=run_spec.supervision,
        )

    @property
    def saved_run_spec_path(self) -> Path:
        return self.paths.root / RUN_SPEC_FILENAME

    def create(self, *, parent_checkpoint: Mapping[str, object] | None = None) -> None:
        super().create(parent_checkpoint=parent_checkpoint)
        atomic_write_json(self.saved_run_spec_path, self.strict_run_spec.payload)
        manifest = read_json(self.paths.manifest)
        orchestrator = manifest.get("orchestrator")
        if not isinstance(orchestrator, dict):
            raise ValueError("New lineage manifest lacks orchestrator block")
        orchestrator["spec_path"] = RUN_SPEC_FILENAME
        orchestrator["adapter_id"] = self.strict_run_spec.adapter_id
        orchestrator["board_size"] = self.strict_run_spec.board_size
        manifest["run_spec"] = {
            "schema": RUN_SPEC_SCHEMA,
            "path": RUN_SPEC_FILENAME,
            "fingerprint": self.strict_run_spec.fingerprint,
        }
        manifest["config_fingerprint"] = self.strict_run_spec.fingerprint
        atomic_write_json(self.paths.manifest, manifest)

    def _load_manifest(
        self, *, expected_catalog_fingerprint: str | None = None
    ) -> dict[str, object]:
        manifest = super()._load_manifest(
            expected_catalog_fingerprint=expected_catalog_fingerprint
        )
        if not self.saved_run_spec_path.is_file():
            raise ValueError("Immutable lineage run-spec is missing")
        persisted = read_json(self.saved_run_spec_path)
        actual = run_spec_fingerprint(persisted)
        record = manifest.get("run_spec")
        if not isinstance(record, Mapping):
            raise ValueError("Manifest lacks immutable run_spec record")
        if actual != self.strict_run_spec.fingerprint or record.get("fingerprint") != actual:
            raise ValueError("Immutable run-spec drift detected")
        return manifest

    def _driver_env(self, generation: int, *, resume: bool, phase: str) -> dict[str, str]:
        env = super()._driver_env(generation, resume=resume, phase=phase)
        env["AZ_RUN_SPEC_PATH"] = str(self.saved_run_spec_path.resolve())
        env["AZ_RUN_SPEC_FINGERPRINT"] = self.strict_run_spec.fingerprint
        env["AZ_ADAPTER_ID"] = self.strict_run_spec.adapter_id
        env["AZ_BOARD_SIZE"] = str(self.strict_run_spec.board_size)
        return env

    def status(self) -> dict[str, object]:
        status = super().status()
        status["run_spec_fingerprint"] = self.strict_run_spec.fingerprint
        status["run_spec_path"] = str(self.saved_run_spec_path)
        status["adapter_id"] = self.strict_run_spec.adapter_id
        status["board_size"] = self.strict_run_spec.board_size
        return status


__all__ = [
    "RUN_SPEC_FILENAME",
    "RUN_SPEC_SCHEMA",
    "StrictProductionTrainingOrchestrator",
    "StrictRunSpec",
    "discover_active_lineage",
    "load_persisted_run_spec",
    "run_spec_fingerprint",
]
