"""Immutable Git runtime checkouts for Orchestrator V2 production children.

The developer checkout is intentionally not part of a running production
child's execution environment.  A child receives a full Git commit, and this
module materializes that commit as a detached worktree in a global runtime
area outside Run Storage.  Worktrees are retained so a supervisor restart can
reconstruct the same execution environment without copying source files.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping


class RuntimeIntegrityError(RuntimeError):
    """The pinned execution revision cannot be trusted or materialized."""


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeIntegrityError(
            f"Git runtime operation failed in {repo_root}: {detail.strip()}"
        ) from exc
    return result.stdout.strip()


def resolve_execution_commit(repo_root: str | Path, value: object) -> str:
    """Resolve and validate a real commit object, returning its full SHA."""

    root = Path(repo_root).resolve()
    if not isinstance(value, str) or not value.strip():
        raise RuntimeIntegrityError("execution_code_commit is missing")
    try:
        resolved = _run_git(root, "rev-parse", "--verify", f"{value.strip()}^{{commit}}")
    except RuntimeIntegrityError:
        raise RuntimeIntegrityError(
            f"pinned execution commit cannot be resolved from Git: {value!r}"
        ) from None
    if len(resolved) != 40 or any(char not in "0123456789abcdef" for char in resolved):
        raise RuntimeIntegrityError(f"Git returned an invalid execution commit: {resolved!r}")
    return resolved


def _tree_sha(repo_root: Path, commit: str) -> str:
    tree = _run_git(repo_root, "rev-parse", f"{commit}^{{tree}}")
    if len(tree) != 40 or any(char not in "0123456789abcdef" for char in tree):
        raise RuntimeIntegrityError(f"Git returned an invalid execution tree: {tree!r}")
    return tree


def _default_runtime_root(repo_root: Path) -> Path:
    configured = os.environ.get("AZ_ORCHESTRATOR_RUNTIME_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    identity = hashlib.sha256(str(repo_root).encode("utf-8")).hexdigest()[:16]
    return (Path(tempfile.gettempdir()) / "gocube-orchestrator-v2" / identity).resolve()


@dataclass(frozen=True)
class ImmutableRuntime:
    repo_root: Path
    commit: str
    tree: str
    path: Path

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build a child environment with project imports rooted at this worktree."""

        env = dict(os.environ if base is None else base)
        source_root = self.repo_root.resolve()
        retained: list[str] = []
        for raw in env.get("PYTHONPATH", "").split(os.pathsep):
            if not raw:
                continue
            candidate = Path(raw).resolve()
            try:
                candidate.relative_to(source_root)
            except ValueError:
                retained.append(raw)
        env["PYTHONPATH"] = os.pathsep.join([str(self.path), *retained])
        env["AZ_ORCHESTRATOR_RUNTIME_COMMIT"] = self.commit
        env["AZ_ORCHESTRATOR_RUNTIME_TREE"] = self.tree
        return env


class ImmutableRuntimeManager:
    """Materialize and validate detached worktrees for one Git repository."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        runtime_root: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.runtime_root = (
            Path(runtime_root).resolve()
            if runtime_root is not None
            else _default_runtime_root(self.repo_root)
        )
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.runtime_root / ".lock"

    def ensure(self, value: object) -> ImmutableRuntime:
        commit = resolve_execution_commit(self.repo_root, value)
        tree = _tree_sha(self.repo_root, commit)
        path = self.runtime_root / commit
        with self._lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                if not path.exists():
                    subprocess.run(
                        ["git", "worktree", "add", "--detach", str(path), commit],
                        cwd=self.repo_root,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                actual = resolve_execution_commit(path, "HEAD")
                if actual != commit:
                    raise RuntimeIntegrityError(
                        f"runtime worktree resolved to {actual}, expected {commit}"
                    )
                dirty = _run_git(path, "status", "--porcelain", "--untracked-files=all")
                if dirty:
                    raise RuntimeIntegrityError(
                        f"immutable runtime worktree is dirty: {path}"
                    )
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr or str(exc)
                raise RuntimeIntegrityError(
                    f"cannot create immutable runtime worktree for {commit}: {detail.strip()}"
                ) from exc
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return ImmutableRuntime(self.repo_root, commit, tree, path)


def execution_commit_from_lineage(lineage_root: str | Path) -> str:
    """Read the durable execution pin, supporting pre-rollover manifests."""

    path = Path(lineage_root).resolve() / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIntegrityError(f"cannot read lineage manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeIntegrityError(f"lineage manifest is not an object: {path}")
    value = payload.get("execution_code_commit") or payload.get("git_commit")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeIntegrityError(
            f"lineage manifest has no durable execution code commit: {path}"
        )
    return value


def execution_report_from_lineage(lineage_root: str | Path) -> dict[str, object]:
    """Return the compact execution identity used by operator reporting."""

    path = Path(lineage_root).resolve() / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIntegrityError(f"cannot read lineage manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeIntegrityError(f"lineage manifest is not an object: {path}")
    commit = execution_commit_from_lineage(lineage_root)
    initial = payload.get("lineage_initial_git_commit") or commit
    if not isinstance(initial, str):
        initial = commit
    return {
        "execution_code_commit": commit,
        "initial_lineage_code_commit": initial,
        "rollover": payload.get("execution_rollover", "no"),
    }


def validate_runtime_head(runtime_root: str | Path, expected_commit: object) -> None:
    """Fail closed if a child is not actually running from its pinned worktree."""

    root = Path(runtime_root).resolve()
    actual = resolve_execution_commit(root, "HEAD")
    expected = resolve_execution_commit(root, expected_commit)
    if actual != expected:
        raise RuntimeIntegrityError(
            f"child runtime commit mismatch: actual={actual}, expected={expected}"
        )


__all__ = [
    "ImmutableRuntime",
    "ImmutableRuntimeManager",
    "RuntimeIntegrityError",
    "execution_commit_from_lineage",
    "execution_report_from_lineage",
    "resolve_execution_commit",
    "validate_runtime_head",
]
