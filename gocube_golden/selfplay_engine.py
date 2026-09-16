"""Compatibility facade for process self-play execution.

The implementation lives at repository level so the Golden scientific package
keeps its source dependency boundary free of process-runtime imports.

The facade also owns the production fail-closed process cleanup contract.  The
repository-level engine predates the production supervisor and historically
used a best-effort ``terminate()``/``join()`` helper.  Production callers must
not leave ``selfplay-search-*`` workers alive after an aborted engine run, so
the exported class strengthens that helper with bounded SIGKILL escalation.
"""
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
]
