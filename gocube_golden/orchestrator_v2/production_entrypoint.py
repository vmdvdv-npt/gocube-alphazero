"""Explicit production wiring for Orchestrator V2.

The coordinators deliberately accept an injected notifier and never construct
one themselves.  This module is the production boundary that loads a JSON
plan, creates the existing fail-open ``TelegramNotifier``, injects it, runs
the selected V2 coordinator, and flushes queued legacy notifications.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import __main__
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from ..run_storage import ACTIVE, RUNS_ROOT
from ..telegram_notifier import TelegramError, TelegramNotifier, flush_all, telegram_test
from .artifact_resolver import ArtifactResolver
from .arena_runner import ArenaRunnerV2
from .continuous_training import ContinuousTrainingConfig, ContinuousTrainingRunnerV2
from .experiment_plan import ExperimentConfig
from .experiment_runner import ExperimentRunnerV2
from .version import mark_v2_process
from tools.arena_engine import DEFAULT_MASTER_SEED


def load_v2_config(path: str | Path) -> dict[str, object]:
    """Load one operator-owned V2 JSON plan without changing its values."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load V2 config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("V2 config must be a JSON object")
    return dict(payload)


def _notification_paths(root: Path) -> object:
    """Adapt an existing V2 root to the path interface of TelegramNotifier."""
    return SimpleNamespace(
        root=root,
        manifest=root / "manifest.json",
        runtime=root / "runtime",
        runtime_state=root / "runtime" / "state.json",
        logs=root / "logs",
        metrics=root / "metrics",
    )


def _require_file_backed_entrypoint() -> None:
    """Reject production runs whose ``spawn`` parent has no importable file.

    Arena workers use ``multiprocessing``'s ``spawn`` context.  A coordinator
    started with ``python -`` or ``python -c`` gives child processes a
    synthetic ``<stdin>``/``-`` main path, so every worker dies while trying
    to reconstruct the parent before the startup barrier.  Fail before any
    generation or Arena artifact is started and require the normal module or
    script entrypoint instead.
    """
    main_file = getattr(__main__, "__file__", None)
    if not isinstance(main_file, str) or main_file in {"", "-", "<stdin>"}:
        raise RuntimeError(
            "Orchestrator V2 production runs require a file-backed entrypoint; "
            "invoke production_entrypoint.py (or python -m "
            "gocube_golden.orchestrator_v2.production_entrypoint), not stdin/c."
        )
    if not Path(main_file).is_file():
        raise RuntimeError(
            "Orchestrator V2 production entrypoint is not an importable file: "
            f"{main_file!r}"
        )


def _continuous_config(payload: Mapping[str, object]) -> ContinuousTrainingConfig:
    raw = payload.get("continuous", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("continuous config must be an object")
    required = (
        "parent_checkpoint",
        "lineage_id",
        "effective_config",
        "arena_cadence",
        "arena_config",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"continuous config is missing: {', '.join(missing)}")
    return ContinuousTrainingConfig(
        parent_checkpoint=raw["parent_checkpoint"],  # type: ignore[arg-type]
        lineage_id=str(raw["lineage_id"]),
        effective_config=raw["effective_config"],  # type: ignore[arg-type]
        generations=None if raw.get("generations") is None else int(raw["generations"]),
        arena_cadence=int(raw["arena_cadence"]),
        arena_config=raw["arena_config"],  # type: ignore[arg-type]
        arena_master_seed=int(raw.get("arena_master_seed", DEFAULT_MASTER_SEED)),
        arena_startset=raw.get("arena_startset"),  # type: ignore[arg-type]
        arena_profile=str(raw.get("arena_profile", "torus9")),
        arena_scientific_contract=raw.get("arena_scientific_contract"),  # type: ignore[arg-type]
        arena_execution_contract=raw.get("arena_execution_contract"),  # type: ignore[arg-type]
        arena_workload=dict(raw.get("arena_workload", {})),  # type: ignore[arg-type]
        arena_reference_gap=(
            None if raw.get("arena_reference_gap") is None else int(raw["arena_reference_gap"])
        ),
        allow_code_rollover=raw.get("allow_code_rollover", False),  # type: ignore[arg-type]
        self_play_concurrency_sweep=raw.get("self_play_concurrency_sweep"),  # type: ignore[arg-type]
    )


def _experiment_config(payload: Mapping[str, object]) -> ExperimentConfig:
    raw = payload.get("experiment", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("experiment config must be an object")
    return ExperimentConfig.from_dict(raw)


def run_continuous_from_config(
    payload: Mapping[str, object],
    *,
    runs_root: str | Path | None = None,
    allow_code_rollover: bool | None = None,
    **runner_kwargs: Any,
) -> object:
    """Run a production continuous plan with explicitly injected Telegram."""
    _require_file_backed_entrypoint()
    mark_v2_process()
    config = _continuous_config(payload)
    if allow_code_rollover is not None:
        if type(allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        config = replace(config, allow_code_rollover=allow_code_rollover)
    resolver = runner_kwargs.pop("resolver", None) or ArtifactResolver(runs_root)
    root = resolver.runs_root / config.topology / ACTIVE / config.lineage_id
    notifier = TelegramNotifier(_notification_paths(root))
    try:
        runner = ContinuousTrainingRunnerV2(
            config,
            resolver=resolver,
            notifier=notifier,
            **runner_kwargs,
        )
        return runner.run()
    finally:
        flush_all()


def run_experiment_from_config(
    payload: Mapping[str, object],
    *,
    runs_root: str | Path | None = None,
    allow_code_rollover: bool | None = None,
    **runner_kwargs: Any,
) -> object:
    """Run a production A/B or A/B→C plan with explicitly injected Telegram."""
    _require_file_backed_entrypoint()
    mark_v2_process()
    config = _experiment_config(payload)
    if allow_code_rollover is not None:
        if type(allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        config = replace(config, allow_code_rollover=allow_code_rollover)
    resolver = runner_kwargs.pop("resolver", None) or ArtifactResolver(runs_root)
    experiment_root = runner_kwargs.get("experiment_root")
    root = (
        Path(experiment_root).resolve()
        if experiment_root is not None
        else resolver.runs_root / config.topology / "experiments" / config.experiment_id
    )
    notifier = TelegramNotifier(_notification_paths(root))
    arena_runner = runner_kwargs.pop("arena_runner", None) or ArenaRunnerV2()
    try:
        runner = ExperimentRunnerV2(
            config,
            arena_runner=arena_runner,
            resolver=resolver,
            notifier=notifier,
            **runner_kwargs,
        )
        return runner.run()
    finally:
        flush_all()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    telegram = subparsers.add_parser("telegram-test", help="send one explicit transport test")
    telegram.set_defaults(kind="telegram")
    for name in ("continuous", "experiment"):
        command = subparsers.add_parser(name, help=f"run a V2 {name} JSON plan")
        command.add_argument("config", type=Path)
        command.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
        command.add_argument(
            "--allow-code-rollover",
            action="store_true",
            default=None,
            help="explicitly allow a clean application-code rollover when resuming",
        )
        command.set_defaults(kind=name)
    args = parser.parse_args(argv)
    if args.kind == "telegram":
        try:
            telegram_test()
        except TelegramError as exc:
            raise SystemExit(f"Telegram test failed: {exc}") from None
        print("Telegram test message sent.")
        return 0

    payload = load_v2_config(args.config)
    if args.kind == "continuous":
        run_continuous_from_config(
            payload,
            runs_root=args.runs_root,
            allow_code_rollover=args.allow_code_rollover,
        )
    else:
        run_experiment_from_config(
            payload,
            runs_root=args.runs_root,
            allow_code_rollover=args.allow_code_rollover,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_v2_config",
    "run_continuous_from_config",
    "run_experiment_from_config",
]
