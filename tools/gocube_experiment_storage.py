from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from pathlib import Path

from tools import gocube_experiment_runner as _runner


COW_CLONE_MODE = "hardlink-pkl-copy-metadata-v1"
DISK_SAFETY_FACTOR = 2.0
MIN_FREE_RESERVE_BYTES = 5 * 1024**3
_ITERATION_RE = re.compile(r"iteration-(\d+)")


def _path_iteration(path: Path) -> int | None:
    for part in path.parts:
        match = _ITERATION_RE.search(part)
        if match:
            return int(match.group(1))
    return None


def _tree_logical_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += int(item.stat().st_size)
        except FileNotFoundError:
            continue
    return total


def iteration_footprint_bytes(repo: Path, run_name: str, iteration: int) -> int:
    iteration = int(iteration)
    total = 0
    checkpoint = repo / "checkpoint" / run_name / f"iteration-{iteration:04d}.pkl"
    if checkpoint.is_file():
        total += int(checkpoint.stat().st_size)
    data_root = repo / "data" / run_name
    if data_root.is_dir():
        for item in data_root.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    relative = item.relative_to(data_root)
                    if _path_iteration(relative) == iteration:
                        total += int(item.stat().st_size)
            except FileNotFoundError:
                continue
    return total


def storage_preflight_report(
    repo: Path,
    *,
    run_name: str,
    iteration: int,
    parallel_branches: int,
    new_iterations_per_branch: int,
    reserve_bytes: int = MIN_FREE_RESERVE_BYTES,
    safety_factor: float = DISK_SAFETY_FACTOR,
) -> dict[str, object]:
    if parallel_branches < 1 or new_iterations_per_branch < 1:
        raise ValueError("storage preflight branch/iteration counts must be positive")
    footprint = iteration_footprint_bytes(repo, run_name, iteration)
    if footprint <= 0:
        raise RuntimeError(
            f"Cannot estimate storage: no iteration-{int(iteration):04d} footprint for {run_name}"
        )
    usage = shutil.disk_usage(repo)
    estimated_new = int(footprint) * int(parallel_branches) * int(new_iterations_per_branch)
    working = int(math.ceil(float(estimated_new) * float(safety_factor)))
    required = int(working) + int(reserve_bytes)
    return {
        "run": run_name,
        "iteration": int(iteration),
        "iteration_footprint_bytes": int(footprint),
        "parallel_branches": int(parallel_branches),
        "new_iterations_per_branch": int(new_iterations_per_branch),
        "estimated_new_unique_bytes": int(estimated_new),
        "safety_factor": float(safety_factor),
        "working_bytes": int(working),
        "reserve_bytes": int(reserve_bytes),
        "required_free_bytes": int(required),
        "filesystem_total_bytes": int(usage.total),
        "filesystem_free_bytes": int(usage.free),
        "ok": int(usage.free) >= int(required),
    }


def clone_history_cow(source: Path, destination: Path, *, max_iteration: int) -> dict[str, int]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    stats = {
        "hardlinked_files": 0,
        "hardlinked_bytes": 0,
        "copied_files": 0,
        "copied_bytes": 0,
        "skipped_future_entries": 0,
    }

    source = source.resolve()

    def ignore(directory: str, names: list[str]):
        directory_path = Path(directory).resolve()
        relative_dir = directory_path.relative_to(source)
        ignored = []
        for name in names:
            relative = relative_dir / name
            found = _path_iteration(relative)
            if found is not None and found > int(max_iteration):
                ignored.append(name)
                stats["skipped_future_entries"] += 1
        return set(ignored)

    def copy_file(src: str, dst: str):
        src_path = Path(src)
        size = int(src_path.stat().st_size)
        if src_path.suffix.lower() == ".pkl":
            try:
                os.link(src, dst, follow_symlinks=False)
            except (OSError, TypeError) as exc:
                raise RuntimeError(
                    "Storage-efficient sweep requires same-filesystem hardlinks for immutable .pkl history; "
                    f"could not link {src_path}"
                ) from exc
            stats["hardlinked_files"] += 1
            stats["hardlinked_bytes"] += size
            return dst
        shutil.copy2(src, dst, follow_symlinks=False)
        stats["copied_files"] += 1
        stats["copied_bytes"] += size
        return dst

    shutil.copytree(
        source,
        destination,
        symlinks=True,
        copy_function=copy_file,
        ignore=ignore,
    )
    return stats


class StorageEfficientExperiment(_runner.Experiment):
    """Crash-resumable sweep with copy-on-write history and bounded candidate retention."""

    def _storage_state(self) -> dict[str, object]:
        storage = self.state.setdefault("storage", {})
        storage.setdefault("schema_version", 1)
        storage.setdefault("clone_mode", COW_CLONE_MODE)
        storage.setdefault("preflights", {})
        storage.setdefault("clones", {})
        storage.setdefault("pruned_namespaces", {})
        return storage

    def _record_storage_preflight(
        self,
        key: str,
        *,
        parent_run: str,
        parent_iteration: int,
        parallel_branches: int,
        new_iterations_per_branch: int,
    ) -> None:
        report = storage_preflight_report(
            self.repo,
            run_name=parent_run,
            iteration=parent_iteration,
            parallel_branches=parallel_branches,
            new_iterations_per_branch=new_iterations_per_branch,
        )
        storage = self._storage_state()
        storage["preflights"][str(key)] = {**report, "checked_at_epoch": time.time()}
        self._save_state()
        if not bool(report["ok"]):
            gib = 1024**3
            raise RuntimeError(
                "Insufficient free disk for the next sweep phase: "
                f"free={report['filesystem_free_bytes'] / gib:.2f} GiB, "
                f"required={report['required_free_bytes'] / gib:.2f} GiB "
                "including safety reserve."
            )

    def _ensure_clone(
        self,
        *,
        parent_run: str,
        parent_iteration: int,
        target_run: str,
        kind: str,
    ) -> None:
        checkpoint_dest = self.repo / "checkpoint" / target_run
        data_dest = self.repo / "data" / target_run
        checkpoint_partial = self.repo / "checkpoint" / f".{target_run}.partial"
        data_partial = self.repo / "data" / f".{target_run}.partial"
        expected_core = {
            "experiment_id": self.cli.experiment_id,
            "kind": kind,
            "parent_run": parent_run,
            "parent_iteration": int(parent_iteration),
        }

        if checkpoint_partial.exists() or data_partial.exists():
            self._quarantine_namespace(target_run, "stale partial clone from interrupted copy")
        if checkpoint_dest.exists() != data_dest.exists():
            self._quarantine_namespace(target_run, "one-sided candidate namespace after interrupted clone")
        if checkpoint_dest.exists() and data_dest.exists():
            provenance_path = checkpoint_dest / _runner.PROVENANCE_FILENAME
            if not provenance_path.is_file():
                self._quarantine_namespace(target_run, "candidate namespace has no sweep provenance")
            else:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                for key, value in expected_core.items():
                    if provenance.get(key) != value:
                        raise RuntimeError(f"Candidate namespace provenance mismatch: {target_run}")
                schema = int(provenance.get("schema_version", -1))
                if schema not in (1, 2):
                    raise RuntimeError(f"Unsupported candidate provenance schema: {target_run}")
                if schema == 2 and provenance.get("clone_mode") != COW_CLONE_MODE:
                    raise RuntimeError(f"Candidate clone mode mismatch: {target_run}")
                if schema == 1:
                    self._storage_state().setdefault("legacy_full_copy_namespaces", {})[target_run] = {
                        "detected_at_epoch": time.time(),
                        "parent_run": parent_run,
                        "parent_iteration": int(parent_iteration),
                    }
                    self._save_state()
                return

        checkpoint_source = self.repo / "checkpoint" / parent_run
        data_source = self.repo / "data" / parent_run
        if not checkpoint_source.is_dir() or not data_source.is_dir():
            raise FileNotFoundError(f"Parent run is incomplete: {parent_run}")

        checkpoint_stats = clone_history_cow(
            checkpoint_source,
            checkpoint_partial,
            max_iteration=parent_iteration,
        )
        data_stats = clone_history_cow(
            data_source,
            data_partial,
            max_iteration=parent_iteration,
        )
        manifest = checkpoint_partial / "gocube-run.json"
        if manifest.exists():
            manifest.unlink()
        provenance = {
            "schema_version": 2,
            **expected_core,
            "clone_mode": COW_CLONE_MODE,
            "max_parent_iteration": int(parent_iteration),
            "created_at_epoch": time.time(),
        }
        _runner.atomic_json(checkpoint_partial / _runner.PROVENANCE_FILENAME, provenance)
        os.replace(checkpoint_partial, checkpoint_dest)
        os.replace(data_partial, data_dest)

        hardlinked_bytes = int(checkpoint_stats["hardlinked_bytes"]) + int(data_stats["hardlinked_bytes"])
        copied_bytes = int(checkpoint_stats["copied_bytes"]) + int(data_stats["copied_bytes"])
        self._storage_state()["clones"][target_run] = {
            "parent_run": parent_run,
            "parent_iteration": int(parent_iteration),
            "kind": kind,
            "clone_mode": COW_CLONE_MODE,
            "hardlinked_bytes": hardlinked_bytes,
            "copied_metadata_bytes": copied_bytes,
            "hardlinked_files": int(checkpoint_stats["hardlinked_files"]) + int(data_stats["hardlinked_files"]),
            "copied_metadata_files": int(checkpoint_stats["copied_files"]) + int(data_stats["copied_files"]),
            "created_at_epoch": time.time(),
        }
        self._save_state()

    def _remove_owned_namespace(self, run_name: str, reason: str) -> bool:
        prefix = f"{self.cli.experiment_id}-"
        if not str(run_name).startswith(prefix):
            raise RuntimeError(f"Refusing to prune non-experiment namespace: {run_name}")
        storage = self._storage_state()
        pruned = storage["pruned_namespaces"]
        paths = [
            self.repo / "checkpoint" / run_name,
            self.repo / "data" / run_name,
            self.repo / "runs" / run_name,
        ]
        existing = [path for path in paths if path.exists()]
        if not existing:
            return False
        logical_bytes = sum(_tree_logical_bytes(path) for path in existing)
        removed = []
        for path in existing:
            shutil.rmtree(path)
            removed.append(str(path.relative_to(self.repo)))
        pruned[run_name] = {
            "reason": reason,
            "paths": removed,
            "logical_bytes_removed": int(logical_bytes),
            "note": "logical bytes may include hardlinked history and are not a physical-free-space claim",
            "pruned_at_epoch": time.time(),
        }
        return True

    def _cleanup_benchmark_namespaces(self) -> None:
        if not isinstance(self.state.get("performance_benchmark"), dict):
            return
        changed = False
        for wait_ms in _runner.SELFPLAY_BENCHMARK_WAITS_MS:
            run_name = (
                f"{self.cli.experiment_id}-bench-selfplay-"
                f"{_runner._impl._safe(wait_ms)}ms"
            )
            changed |= self._remove_owned_namespace(
                run_name,
                "benchmark metrics persisted in experiment state",
            )
        if changed:
            self._save_state()

    def _prune_completed_candidate_namespaces(self, bootstrap_run: str | None = None) -> None:
        records = self.state.get("parameters") or []
        if not records:
            return
        if bootstrap_run is None and isinstance(self.state.get("bootstrap"), dict):
            bootstrap_run = str(self.state["bootstrap"]["run"])

        current = records[-1].get("champion_after") or {}
        keep = {str(current.get("run"))} if current.get("run") else set()
        if bootstrap_run:
            keep.add(str(bootstrap_run))
        promoted = [record for record in records if bool((record.get("winner") or {}).get("promoted"))]
        if promoted:
            parent = promoted[-1].get("parent") or {}
            if parent.get("run"):
                keep.add(str(parent["run"]))

        changed = False
        for record in records:
            for candidate in record.get("candidates") or []:
                run_name = str(candidate.get("run") or "")
                if not run_name:
                    continue
                retained = run_name in keep
                candidate["artifacts_retained"] = retained
                if retained:
                    candidate["artifact_status"] = "retained"
                    continue
                candidate["artifact_status"] = "pruned-after-evaluation"
                changed |= self._remove_owned_namespace(
                    run_name,
                    "candidate evidence persisted; namespace not required for champion lineage",
                )
        if changed:
            self._save_state()

    def performance_benchmark(self, bootstrap_run: str):
        if not self.cli.skip_performance_benchmark:
            self._record_storage_preflight(
                "performance-benchmark",
                parent_run=bootstrap_run,
                parent_iteration=_runner.BOOTSTRAP_ITERATION,
                parallel_branches=len(_runner.SELFPLAY_BENCHMARK_WAITS_MS),
                new_iterations_per_branch=1,
            )
        return super().performance_benchmark(bootstrap_run)

    def _restore_completed_stages(self, bootstrap_run: str):
        result = super()._restore_completed_stages(bootstrap_run)
        self._cleanup_benchmark_namespaces()
        self._prune_completed_candidate_namespaces(bootstrap_run)
        return result

    def run_parameter(
        self,
        spec: dict[str, object],
        parent_run: str,
        parent_iteration: int,
        active_overrides: dict[str, float | int],
    ):
        self._record_storage_preflight(
            f"stage:{spec['id']}",
            parent_run=parent_run,
            parent_iteration=parent_iteration,
            parallel_branches=4,
            new_iterations_per_branch=_runner.CANDIDATE_ITERATIONS,
        )
        result = super().run_parameter(
            spec,
            parent_run,
            parent_iteration,
            active_overrides,
        )
        bootstrap = None
        if isinstance(self.state.get("bootstrap"), dict):
            bootstrap = str(self.state["bootstrap"]["run"])
        self._prune_completed_candidate_namespaces(bootstrap)
        return result
