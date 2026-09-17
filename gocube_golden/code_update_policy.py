"""Explicit per-orchestrator application-code rollover and provenance policy.

A lineage represents one training history, not one application commit. The
creation commit remains preserved, while each generation records the exact
code revision and resolved parameters used for that attempt. The behavior is
opt-in per orchestrator instance through the production child-lifecycle API.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .orchestrator import atomic_write_json, read_json
from .provenance import capture_code_identity

if TYPE_CHECKING:
    from .production_orchestrator import ChildLifecycleContext


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_snapshot(repo_root: Path) -> dict[str, object]:
    code = capture_code_identity(repo_root)
    if not code.working_tree_clean:
        raise ValueError("Production training refuses a dirty working tree")
    return {
        "git_commit_sha": code.git_commit_sha,
        "git_tree_sha": code.git_tree_sha,
        "working_tree_clean": code.working_tree_clean,
    }


def _advance_manifest_code_pin(
    run: object,
    *,
    code: Mapping[str, object],
    generation: int,
    phase: str,
) -> None:
    paths = getattr(run, "paths")
    manifest_path = Path(getattr(paths, "manifest"))
    manifest = read_json(manifest_path)
    current = str(code["git_commit_sha"])
    previous = str(manifest.get("git_commit", ""))

    if previous == current:
        return

    initial = str(manifest.get("lineage_initial_git_commit") or previous)
    history_value = manifest.get("code_revision_history", [])
    history = [
        dict(item)
        for item in history_value
        if isinstance(item, Mapping)
    ] if isinstance(history_value, list) else []
    if not history or str(history[-1].get("git_commit_sha")) != current:
        history.append(
            {
                "git_commit_sha": current,
                "git_tree_sha": str(code["git_tree_sha"]),
                "first_seen_generation": int(generation),
                "phase": str(phase),
                "recorded_at": _utc_now(),
                "reason": "application-code update within same training lineage",
            }
        )

    manifest["lineage_initial_git_commit"] = initial
    manifest["git_commit"] = current
    manifest["current_git_tree"] = str(code["git_tree_sha"])
    manifest["code_revision_history"] = history
    atomic_write_json(manifest_path, manifest)


def _resolved_run_payload(run: object) -> Mapping[str, object]:
    strict = getattr(run, "strict_run_spec", None)
    payload = getattr(strict, "payload", None)
    if isinstance(payload, Mapping):
        return payload
    spec = getattr(run, "spec")
    payload = getattr(spec, "payload", None)
    return payload if isinstance(payload, Mapping) else {}


def _generation_provenance_path(run: object, generation: int) -> Path:
    paths = getattr(run, "paths")
    root = Path(getattr(paths, "root"))
    return root / "provenance" / "generations" / f"generation-{int(generation):04d}.json"


def _record_generation_attempt(
    run: object,
    *,
    generation: int,
    code: Mapping[str, object],
) -> Path:
    spec = getattr(run, "spec")
    run_payload = _resolved_run_payload(run)
    generation_payload = run_payload.get("generation", {})
    strict = getattr(run, "strict_run_spec", None)
    strict_fingerprint = getattr(strict, "fingerprint", None)

    tx_path = getattr(run, "_generation_tx_path")(int(generation))
    restart_attempts = 0
    if Path(tx_path).is_file():
        tx = read_json(Path(tx_path))
        restart_attempts = int(tx.get("restart_attempts", 0))

    path = _generation_provenance_path(run, generation)
    existing: dict[str, object] = read_json(path) if path.is_file() else {}
    history_value = existing.get("attempts", [])
    history = [
        dict(item)
        for item in history_value
        if isinstance(item, Mapping)
    ] if isinstance(history_value, list) else []

    attempt = {
        "attempt_index": len(history),
        "orchestrator_restart_attempts": restart_attempts,
        "status": "CHILD_RUNNING",
        "recorded_at": _utc_now(),
        "code": dict(code),
        "run_spec_fingerprint": strict_fingerprint
        or getattr(spec, "config_fingerprint", None),
        "config_fingerprint": getattr(spec, "config_fingerprint", None),
        "profile_fingerprint": getattr(spec, "profile_fingerprint", None),
        "effective_parameters": {
            "profile": deepcopy(dict(getattr(spec, "profile_payload"))),
            "generation": deepcopy(
                dict(generation_payload)
                if isinstance(generation_payload, Mapping)
                else generation_payload
            ),
        },
    }
    history.append(attempt)
    payload = {
        "schema": "gocube-training-generation-provenance-v1",
        "lineage_id": str(getattr(run, "lineage_id")),
        "topology": str(getattr(spec, "topology")),
        "generation": int(generation),
        "status": "CHILD_RUNNING",
        "latest_attempt": attempt,
        "attempts": history,
    }
    atomic_write_json(path, payload)
    return path


def _finish_generation_attempt(path: Path, exit_code: int) -> None:
    if not path.is_file():
        return
    payload = read_json(path)
    status = "CHILD_COMPLETED" if int(exit_code) == 0 else "CHILD_FAILED"
    attempts = payload.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], Mapping):
        final = dict(attempts[-1])
        finished_at = _utc_now()
        final["status"] = status
        final["child_exit_code"] = int(exit_code)
        final["child_finished_at"] = finished_at
        final["attempt_finished_at"] = finished_at
        attempts = [*attempts[:-1], final]
        payload["attempts"] = attempts
        payload["latest_attempt"] = final
    payload["status"] = status
    atomic_write_json(path, payload)


def _abort_generation_attempt(path: Path, error: BaseException) -> None:
    if not path.is_file():
        return
    payload = read_json(path)
    attempts = payload.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], Mapping):
        final = dict(attempts[-1])
        final["status"] = "SUPERVISOR_ABORTED"
        final["attempt_finished_at"] = _utc_now()
        final["supervisor_error_type"] = type(error).__name__
        final["supervisor_error_message"] = str(error)
        attempts = [*attempts[:-1], final]
        payload["attempts"] = attempts
        payload["latest_attempt"] = final
    payload["status"] = "SUPERVISOR_ABORTED"
    atomic_write_json(path, payload)


class CodeUpdateProvenancePolicy:
    """Opt-in lifecycle policy for clean code rollover and generation provenance."""

    def __init__(self) -> None:
        self._generation_provenance: dict[tuple[int, int], Path] = {}

    @staticmethod
    def _key(context: "ChildLifecycleContext") -> tuple[int, int]:
        return (id(context.orchestrator), int(context.generation))

    def before_child_start(self, context: "ChildLifecycleContext") -> None:
        run = context.orchestrator
        repo_root = Path(getattr(run, "repo_root"))
        # Synthetic/unit-test orchestrators commonly use temporary non-Git
        # roots. The policy remains a production-checkout concern.
        if not (repo_root / ".git").exists():
            return

        code = _code_snapshot(repo_root)
        _advance_manifest_code_pin(
            run,
            code=code,
            generation=int(context.generation),
            phase=str(context.phase),
        )
        if context.phase == "generation":
            self._generation_provenance[self._key(context)] = _record_generation_attempt(
                run,
                generation=int(context.generation),
                code=code,
            )

    def after_child_finish(
        self,
        context: "ChildLifecycleContext",
        exit_code: int,
    ) -> None:
        if context.phase != "generation":
            return
        provenance = self._generation_provenance.pop(self._key(context), None)
        if provenance is not None:
            _finish_generation_attempt(provenance, int(exit_code))

    def after_child_abort(
        self,
        context: "ChildLifecycleContext",
        error: BaseException,
    ) -> None:
        if context.phase != "generation":
            return
        provenance = self._generation_provenance.pop(self._key(context), None)
        if provenance is not None:
            _abort_generation_attempt(provenance, error)


__all__ = [
    "CodeUpdateProvenancePolicy",
]
