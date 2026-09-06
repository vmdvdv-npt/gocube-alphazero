from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass
from typing import Callable, Iterable

from .register_run import register_run


@dataclass(frozen=True)
class KnownRun:
    run_name: str
    topology: str
    size: int
    rule_set: str = "chinese"
    komi: float = 0.5


# Legacy runs trained before immutable GoCube metadata/manifests were written
# automatically. New training runs already create their own manifest.
KNOWN_LEGACY_RUNS = (
    KnownRun("gocube-cube4-stage4-v1", "cube", 4),
    KnownRun("torus-9x9-30iter", "torus", 9),
)


def ensure_known_runs(
    checkpoint_dir: str,
    *,
    runs: Iterable[KnownRun] = KNOWN_LEGACY_RUNS,
    emit: Callable[[str], None] = print,
) -> tuple[str, ...]:
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    ready: list[str] = []
    for run in runs:
        run_dir = os.path.join(checkpoint_dir, run.run_name)
        if not os.path.isdir(run_dir):
            emit(f"GoCube launcher: skip missing legacy run {run.run_name}")
            continue

        path = register_run(
            checkpoint_dir=checkpoint_dir,
            run_name=run.run_name,
            topology=run.topology,
            size=run.size,
            rule_set=run.rule_set,
            komi=run.komi,
            force=False,
        )
        ready.append(path)
        emit(f"GoCube launcher: ready {run.run_name}")

    return tuple(ready)


def assert_port_available(host: str, port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"GoCube AlphaZero service cannot start: {host}:{port} is already in use"
            ) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="One-command GoCube AlphaZero development service launcher"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-origin", action="append", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    cli = parse_args(argv)
    ensure_known_runs(cli.checkpoint_dir)
    assert_port_available(cli.host, cli.port)

    server_argv = [
        "--checkpoint-dir",
        cli.checkpoint_dir,
        "--host",
        cli.host,
        "--port",
        str(cli.port),
        "--device",
        cli.device,
    ]
    for origin in cli.allow_origin or ():
        server_argv.extend(("--allow-origin", origin))

    # Import only when launching the actual service so lightweight launcher tests
    # do not need to instantiate the inference stack.
    from .server import main as server_main

    server_main(server_argv)


if __name__ == "__main__":
    main()
