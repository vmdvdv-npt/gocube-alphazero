"""Compatibility facade for process self-play execution.

The implementation lives at repository level so the Golden scientific package
keeps its source dependency boundary free of process-runtime imports.

The facade also owns the production fail-closed process cleanup contract.  The
repository-level engine predates the production supervisor and historically
used a best-effort ``terminate()``/``join()`` helper.  Production callers must
not leave ``selfplay-search-*`` workers alive after an aborted engine run, so
the exported class strengthens that helper with bounded SIGKILL escalation.
"""

from __future__ import annotations

import inspect
from selfplay_engine import (
    GameFinished,
    InferenceNeed,
    InferenceClient,
    InferenceTransportError,
    SharedInferenceResult,
    SharedMemorySpec,
    SelfPlayEngine as _BaseSelfPlayEngine,
    SelfPlayEngineConfig,
    SelfPlayEngineError,
)

from typing import Any, Callable, Mapping, MutableMapping, NamedTuple, Protocol, Sequence, runtime_checkable


@runtime_checkable
class CooperativeSelfPlayAdapter(Protocol):
    """Explicit scientific-to-execution contract for cooperative self-play."""

    @property
    def worker_context(self) -> object:
        ...

    @property
    def shared_memory(self) -> SharedMemorySpec:
        ...

    @property
    def infer_shared_batch(self) -> Callable[[Any], object]:
        ...

    @property
    def worker_game_factory(self) -> Callable[[object, str, InferenceClient], object]:
        ...

    @property
    def record_metrics(self) -> Callable[[object], Mapping[str, object]]:
        ...


class CooperativeSelfPlayResult(NamedTuple):
    records: tuple[object, ...]
    telemetry: dict[str, object]


def run_cooperative_selfplay(
    game_ids: Sequence[str],
    *,
    adapter: CooperativeSelfPlayAdapter,
    engine_config: SelfPlayEngineConfig,
    telemetry: MutableMapping[str, object] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    worker_diagnostics_factory: Callable[[], object] | None = None,
    active_games_per_worker: int | None = None,
    total_active_contexts: int | None = None,
) -> CooperativeSelfPlayResult:
    """Run any cooperative scientific adapter through the one shared engine."""

    if not isinstance(adapter, CooperativeSelfPlayAdapter):
        raise TypeError("adapter does not implement the cooperative self-play contract")
    raw_telemetry: MutableMapping[str, object] = telemetry if telemetry is not None else {}
    engine = SelfPlayEngine(engine_config)
    run_kwargs: dict[str, object] = {
        "worker_play": None,
        "worker_context": adapter.worker_context,
        "infer_batch": None,
        "record_metrics": adapter.record_metrics,
        "telemetry": raw_telemetry,
        "progress_callback": progress_callback,
        "shared_memory": adapter.shared_memory,
        "infer_shared_batch": adapter.infer_shared_batch,
        "worker_game_factory": adapter.worker_game_factory,
        "active_games_per_worker": active_games_per_worker,
        "total_active_contexts": total_active_contexts,
    }
    if worker_diagnostics_factory is not None:
        if "worker_diagnostics_factory" not in inspect.signature(engine.run).parameters:
            raise TypeError("SelfPlayEngine.run does not support worker diagnostics")
        run_kwargs["worker_diagnostics_factory"] = worker_diagnostics_factory
    records = engine.run(game_ids, **run_kwargs)
    return CooperativeSelfPlayResult(tuple(records), dict(raw_telemetry))


class SelfPlayEngine(_BaseSelfPlayEngine):
    """Production facade with deterministic worker-process teardown."""

    @staticmethod
    def _terminate(processes):
        started = [process for process in processes if process.pid is not None]
        for process in started:
            if process.is_alive():
                process.terminate()

        for process in started:
            process.join(timeout=5.0)

        survivors = [process for process in started if process.is_alive()]
        for process in survivors:
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
            else:  # pragma: no cover - compatibility fallback for old Python.
                process.terminate()

        for process in survivors:
            process.join(timeout=5.0)

        still_alive = [process for process in started if process.is_alive()]
        if still_alive:
            detail = ", ".join(
                f"pid={process.pid}:name={process.name}" for process in still_alive
            )
            raise SelfPlayEngineError(
                "self-play worker process survived bounded teardown: " + detail
            )


__all__ = [
    "InferenceClient",
    "InferenceTransportError",
    "InferenceNeed",
    "GameFinished",
    "SharedMemorySpec",
    "SharedInferenceResult",
    "SelfPlayEngine",
    "SelfPlayEngineConfig",
    "SelfPlayEngineError",
    "CooperativeSelfPlayAdapter",
    "CooperativeSelfPlayResult",
    "run_cooperative_selfplay",
]
