"""Pickle-safe diagnostics for failures in multiprocessing workers."""

from __future__ import annotations


class SelfPlayWorkerError(RuntimeError):
    """A self-play child reported an exception to the parent process."""

    def __init__(self, payload: dict[str, object]):
        self.payload = dict(payload)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        payload = self.payload
        return (
            "Self-play worker failure: "
            f"iteration={payload.get('iteration')}, "
            f"worker={payload.get('worker_id')}, "
            f"game slot={payload.get('game_slot')}, "
            f"game id={payload.get('game_id')}, "
            f"stage={payload.get('stage')}, "
            f"exception type={payload.get('exception_type')}, "
            f"exception message={payload.get('exception_message')}\n"
            "child traceback:\n"
            f"{payload.get('traceback', '')}"
        )


def unexpected_worker_exit_payload(
    *, worker_id: int, iteration: int, exitcode: int, pid: int | None = None,
    game_slot: int | None = None,
    game_id: str | None = None,
) -> dict[str, object]:
    """Build a diagnostic payload when a child dies without reporting Python error."""

    return {
        "worker_id": int(worker_id),
        "pid": pid,
        "iteration": int(iteration),
        "game_slot": game_slot,
        "game_id": game_id,
        "stage": "shutdown",
        "exception_type": "WorkerExit",
        "exception_message": (
            f"Self-play worker {int(worker_id)} exited unexpectedly with exit code {int(exitcode)}"
        ),
        "traceback": "",
    }
