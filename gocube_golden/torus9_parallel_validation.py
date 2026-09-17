"""Process-parallel semantic validation for newly produced Torus9 replay rows.

Historical replay trust remains owned by :mod:`torus9_run_owned_training`.
This module changes only the validation execution strategy for the new batch
about to be appended to replay: semantic checks run in bounded spawned worker
processes, and replay mutation still happens only after every row passes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import math
import multiprocessing
import os
import time
from typing import Mapping, Sequence

from training_engine import value_fingerprint

from .torus9_run_owned_training import (
    Torus9TrainingAdapter as _RunOwnedTorus9TrainingAdapter,
)
from .torus9_training import Torus9RollingReplay


_PARALLEL_VALIDATION_MIN_ROWS = 256
_PARALLEL_VALIDATION_MAX_WORKERS = 8
_VALIDATION_CHUNKS_PER_WORKER = 4
_VALIDATION_MIN_CHUNK_ROWS = 64
_VALIDATION_MAX_CHUNK_ROWS = 512

_worker_adapter: _RunOwnedTorus9TrainingAdapter | None = None


def _init_validation_worker(profile: Mapping[str, object]) -> None:
    """Create one semantic validator per spawned worker process."""
    global _worker_adapter
    _worker_adapter = _RunOwnedTorus9TrainingAdapter(profile=profile)


def _validate_replay_chunk(
    rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[tuple[str, str], ...], float]:
    """Fully validate one chunk and return content fingerprints for the parent."""
    adapter = _worker_adapter
    if adapter is None:
        raise RuntimeError("Torus9 replay validation worker was not initialized")
    validated: list[tuple[str, str]] = []
    fingerprint_process_seconds = 0.0
    for row in rows:
        adapter._validate_sample_semantics(row)
        row_id = str(row.get("replay_row_id", ""))
        if not row_id:
            raise ValueError("Parallel Torus9 replay validation requires replay_row_id")
        fingerprint_started = time.process_time()
        fingerprint = value_fingerprint(row)
        fingerprint_process_seconds += time.process_time() - fingerprint_started
        validated.append((row_id, fingerprint))
    return tuple(validated), fingerprint_process_seconds


class Torus9TrainingAdapter(_RunOwnedTorus9TrainingAdapter):
    """Run-owned adapter with bounded process-parallel validation of new replay."""

    @staticmethod
    def _parallel_validation_worker_count(row_count: int) -> int:
        row_count = int(row_count)
        if row_count < _PARALLEL_VALIDATION_MIN_ROWS:
            return 1
        logical_cpus = max(1, int(os.cpu_count() or 1))
        cpu_workers = max(1, logical_cpus // 2)
        useful_workers = max(1, math.ceil(row_count / _VALIDATION_MIN_CHUNK_ROWS))
        return max(1, min(_PARALLEL_VALIDATION_MAX_WORKERS, cpu_workers, useful_workers))

    @staticmethod
    def _validation_chunks(
        samples: Sequence[Mapping[str, object]], workers: int
    ) -> tuple[tuple[Mapping[str, object], ...], ...]:
        total = len(samples)
        if total == 0:
            return ()
        target_chunks = max(1, int(workers) * _VALIDATION_CHUNKS_PER_WORKER)
        chunk_rows = math.ceil(total / target_chunks)
        chunk_rows = max(
            _VALIDATION_MIN_CHUNK_ROWS,
            min(_VALIDATION_MAX_CHUNK_ROWS, chunk_rows),
        )
        return tuple(
            tuple(samples[start : start + chunk_rows])
            for start in range(0, total, chunk_rows)
        )

    def _validate_new_rows_serial(
        self, samples: Sequence[Mapping[str, object]]
    ) -> dict[str, str]:
        fingerprints: dict[str, str] = {}
        fingerprint_process_seconds = 0.0
        total = len(samples)
        for position, sample in enumerate(samples, 1):
            self._validate_sample_semantics(sample)
            row_id = str(sample.get("replay_row_id", ""))
            if not row_id:
                raise ValueError("Torus9 replay validation requires replay_row_id")
            fingerprint_started = time.process_time()
            fingerprints[row_id] = value_fingerprint(sample)
            fingerprint_process_seconds += time.process_time() - fingerprint_started
            if position == total or position % 256 == 0:
                self._report_progress(
                    "replay",
                    position,
                    max(1, total),
                    "rows",
                    "validation",
                )
        if self._diagnostic_timing is not None:
            self._diagnostic_timing["replay_new_row_fingerprint_process_cpu_sec"] = fingerprint_process_seconds
        return fingerprints

    def _validate_new_rows_parallel(
        self,
        samples: Sequence[Mapping[str, object]],
        *,
        workers: int,
    ) -> dict[str, str]:
        chunks = self._validation_chunks(samples, workers)
        fingerprints: dict[str, str] = {}
        completed = 0
        fingerprint_process_seconds = 0.0
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=context,
            initializer=_init_validation_worker,
            initargs=(dict(self.profile),),
        ) as executor:
            for result, chunk_fingerprint_cpu in executor.map(_validate_replay_chunk, chunks):
                fingerprint_process_seconds += float(chunk_fingerprint_cpu)
                for row_id, fingerprint in result:
                    if row_id in fingerprints:
                        raise ValueError(
                            "Parallel Torus9 replay validation returned duplicate row ID"
                        )
                    fingerprints[row_id] = fingerprint
                completed += len(result)
                self._report_progress(
                    "replay",
                    completed,
                    max(1, len(samples)),
                    "rows",
                    "validation",
                )
        if completed != len(samples):
            raise RuntimeError(
                "Parallel Torus9 replay validation did not return the complete batch"
            )
        if self._diagnostic_timing is not None:
            self._diagnostic_timing["replay_new_row_fingerprint_process_cpu_sec"] = fingerprint_process_seconds
        return fingerprints

    @staticmethod
    def _validate_fingerprint_result(
        samples: Sequence[Mapping[str, object]], fingerprints: Mapping[str, str]
    ) -> None:
        expected = tuple(str(sample.get("replay_row_id", "")) for sample in samples)
        if any(not row_id for row_id in expected):
            raise ValueError("Torus9 replay validation requires replay_row_id")
        if len(expected) != len(set(expected)):
            raise ValueError("Torus9 replay validation received duplicate row IDs")
        if set(fingerprints) != set(expected):
            raise RuntimeError(
                "Torus9 replay validation fingerprints do not cover the complete batch"
            )

    def update_replay(
        self,
        replay: Torus9RollingReplay,
        generation: int,
        samples: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        generation = int(generation)

        # Identity/provenance remains an authoritative parent-process boundary.
        # No worker may mutate replay, and no semantic validation begins until
        # every row is proven to belong to this generation with its canonical ID.
        identity_started = time.perf_counter()
        for position, sample in enumerate(samples):
            self._validate_stamped_sample_identity(sample, generation, position)
        identity_elapsed = time.perf_counter() - identity_started

        workers = self._parallel_validation_worker_count(len(samples))
        started = time.perf_counter()
        fallback_reason: str | None = None
        if workers > 1:
            try:
                fingerprints = self._validate_new_rows_parallel(samples, workers=workers)
                mode = "process-parallel"
            except (BrokenProcessPool, OSError) as exc:
                # Multiprocessing is an execution optimization, not a scientific
                # requirement. Infrastructure-level pool failure may safely fall
                # back to the same authoritative serial validator; semantic
                # validation errors (ValueError) still propagate fail-closed.
                fallback_reason = type(exc).__name__
                fingerprints = self._validate_new_rows_serial(samples)
                workers = 1
                mode = "serial-fallback"
        else:
            fingerprints = self._validate_new_rows_serial(samples)
            mode = "serial"
        elapsed = time.perf_counter() - started
        self._validate_fingerprint_result(samples, fingerprints)

        # Fail-closed transaction boundary: replay is changed only after every
        # semantic validator succeeded. Validation cache publication follows
        # replay mutation so a failed append cannot leave false trusted state.
        metrics = replay.append_generation(generation, samples)
        cache_update_started = time.perf_counter()
        self._validated_sample_fingerprints.update(fingerprints)
        cache_update_elapsed = time.perf_counter() - cache_update_started

        if self._diagnostic_timing is not None:
            timing: dict[str, object] = {
                "replay_new_identity_and_provenance_wall_time_sec": identity_elapsed,
                "replay_new_semantic_validation_wall_time_sec": elapsed,
                "replay_new_semantic_validation_workers": int(workers),
                "replay_new_semantic_validation_mode": mode,
                "replay_new_semantic_validation_rows": len(samples),
                "replay_new_semantic_validation_chunks": (
                    len(self._validation_chunks(samples, workers))
                    if mode == "process-parallel"
                    else (1 if samples else 0)
                ),
                "replay_new_validation_cache_update_wall_time_sec": cache_update_elapsed,
            }
            if fallback_reason is not None:
                timing["replay_new_semantic_validation_fallback_reason"] = fallback_reason
            self._diagnostic_timing.update(timing)
        return metrics


__all__ = ["Torus9TrainingAdapter"]
