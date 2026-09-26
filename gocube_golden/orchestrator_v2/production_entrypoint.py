"""Single declarative production boundary for Orchestrator V2."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import __main__
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from ..run_storage import ACTIVE, RUNS_ROOT
from ..telegram_notifier import TelegramError, TelegramNotifier, flush_all, telegram_test
from .arena_runner import ArenaRunRequest, ArenaRunnerV2
from .artifact_resolver import ArtifactResolver
from .continuous_training import ContinuousTrainingConfig, ContinuousTrainingRunnerV2
from .contracts import StartsetRef
from .execution_permit import _production_authority
from .experiment_plan import ExperimentConfig
from .experiment_runner import ExperimentRunnerV2
from .immutable_runtime import execution_commit_from_lineage, resolve_execution_commit
from .komi_calibration import KomiCalibrationConfig
from .komi_calibration_production import ProductionKomiCalibrationRunnerV2
from .operator_messages import format_action_started
from .run_spec import RunMode, RunSpecV2, supervision_policy_for
from .supervisor import SupervisorPolicy
from .topology_binding import get_topology_binding
from .version import mark_v2_process
from .workflow import WorkflowRunner, WorkflowSpec
from tools.arena_engine import ArenaExecutionConfig, DEFAULT_MASTER_SEED


def load_v2_config(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load V2 config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("V2 config must be a JSON object")
    return dict(payload)


def _notification_paths(root: Path) -> object:
    return SimpleNamespace(
        root=root,
        manifest=root / "manifest.json",
        runtime=root / "runtime",
        runtime_state=root / "runtime" / "state.json",
        logs=root / "logs",
        metrics=root / "metrics",
    )


def _require_file_backed_entrypoint() -> None:
    main_file = getattr(__main__, "__file__", None)
    if not isinstance(main_file, str) or main_file in {"", "-", "<stdin>"}:
        raise RuntimeError(
            "Orchestrator V2 production runs require a file-backed entrypoint; "
            "invoke python -m gocube_golden.orchestrator_v2.production_entrypoint."
        )
    if not Path(main_file).is_file():
        raise RuntimeError(f"Orchestrator V2 production entrypoint is not an importable file: {main_file!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _entrypoint_code_identity() -> str:
    return resolve_execution_commit(_repo_root(), "HEAD")


def _authority(*, mode: str, topology: str, run_id: str):
    mark_v2_process()
    return _production_authority(
        mode=mode,
        topology=topology,
        run_id=run_id,
        code_identity=_entrypoint_code_identity(),
    )


def _continuous_config(payload: Mapping[str, object]) -> ContinuousTrainingConfig:
    raw = payload.get("continuous", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("continuous config must be an object")
    required = ("parent_checkpoint", "lineage_id", "effective_config", "arena_cadence", "arena_config")
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
        arena_profile=None if raw.get("arena_profile") is None else str(raw["arena_profile"]),
        arena_scientific_contract=raw.get("arena_scientific_contract"),  # type: ignore[arg-type]
        arena_execution_contract=raw.get("arena_execution_contract"),  # type: ignore[arg-type]
        arena_workload=dict(raw.get("arena_workload", {})),  # type: ignore[arg-type]
        arena_reference_gap=None if raw.get("arena_reference_gap") is None else int(raw["arena_reference_gap"]),
        allow_code_rollover=raw.get("allow_code_rollover", False),  # type: ignore[arg-type]
        self_play_concurrency_sweep=raw.get("self_play_concurrency_sweep"),  # type: ignore[arg-type]
        supervision=dict(raw.get("supervision", {})),  # type: ignore[arg-type]
    )


def _experiment_config(payload: Mapping[str, object]) -> ExperimentConfig:
    raw = payload.get("experiment", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("experiment config must be an object")
    return ExperimentConfig.from_dict(raw)


def _komi_calibration_config(payload: Mapping[str, object]) -> KomiCalibrationConfig:
    return KomiCalibrationConfig.from_dict(payload)


def _request_parent_soft_stop(parent: object) -> None:
    owner_root = Path(getattr(parent, "owner_root")).resolve()
    control = owner_root / "control" / "soft-stop.json"
    payload = {
        "schema": "gocube-continuous-training-soft-stop-v1",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": "komi-calibration",
        "mode": "finish-current-generation-no-next-generation",
    }
    atomic_write_text(control, canonical_json(payload) + "\n")


_ARENA_SEARCH_FIELDS = frozenset(
    {
        "simulations",
        "mcts_simulations",
        "cpuct",
        "fpu",
        "watchdog",
        "technical_move_limit",
        "komi",
        "root_noise",
        "temperature",
        "fast_search",
        "resign",
        "deterministic_tie_break",
    }
)


def _arena_config(value: object) -> ArenaExecutionConfig:
    if isinstance(value, ArenaExecutionConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("arena_config must be an object")
    raw = dict(value)
    aliases = {
        "inference_batch_cap": "inference_batch_rows",
        "inference_wait": "inference_batch_wait_ms",
        "inference_wait_ms": "inference_batch_wait_ms",
        "contexts": "games_per_worker",
    }
    for source, target in aliases.items():
        if source in raw:
            if target in raw:
                raise ValueError(f"arena_config specifies both {source} and {target}")
            raw[target] = raw.pop(source)
    unknown = set(raw) - set(ArenaExecutionConfig.__dataclass_fields__) - _ARENA_SEARCH_FIELDS
    unknown -= {"search"}
    if unknown:
        raise ValueError(
            "arena_config contains unsupported fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    raw.pop("search", None)
    for field in _ARENA_SEARCH_FIELDS:
        raw.pop(field, None)
    return ArenaExecutionConfig(**raw)


def _arena_search(raw: Mapping[str, object], config: ArenaExecutionConfig) -> dict[str, object]:
    nested = raw.get("search", raw.get("evaluation", {}))
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise ValueError("Arena search/evaluation config must be an object")
    if "search" in raw and isinstance(raw["search"], Mapping):
        unknown = set(raw["search"]) - _ARENA_SEARCH_FIELDS
        if unknown:
            raise ValueError(
                "Arena search contains unsupported fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
    values: dict[str, object] = {}
    for key in _ARENA_SEARCH_FIELDS:
        if key in nested:
            values[key] = nested[key]
        elif key in raw:
            values[key] = raw[key]
    if "simulations" not in values and "mcts_simulations" in values:
        values["simulations"] = values.pop("mcts_simulations")
    if "watchdog" not in values and "technical_move_limit" in values:
        values["watchdog"] = values.pop("technical_move_limit")
    return values


def _torus_profile_with_search(profile: str, search: Mapping[str, object]) -> str:
    from tools.arena_profiles import get_profile

    parsed = get_profile(profile)
    komi = float(search.get("komi", getattr(parsed, "komi", 0.5)))
    simulations = int(search.get("simulations", getattr(parsed, "simulations", 64)))
    cpuct = float(search.get("cpuct", getattr(parsed, "cpuct", 1.25)))
    fpu = float(search.get("fpu", getattr(parsed, "fpu", 0.0)))
    watchdog = int(
        search.get(
            "watchdog",
            search.get("technical_move_limit", getattr(parsed, "watchdog", 500)),
        )
    )
    channel_suffix = "|5ch" if getattr(parsed, "observation_shape", (6,))[0] == 5 else ""
    return (
        f"torus9|komi={komi:g}|simulations={simulations}"
        f"|cpuct={cpuct:g}|fpu={fpu:g}|watchdog={watchdog}{channel_suffix}"
    )


def _standalone_arena_request(raw: Mapping[str, object], resolver: ArtifactResolver) -> ArenaRunRequest:
    candidate_raw = raw.get("candidate_checkpoint", raw.get("candidate"))
    reference_raw = raw.get("reference_checkpoint", raw.get("reference"))
    if candidate_raw is None or reference_raw is None:
        raise ValueError("Arena run-spec requires candidate_checkpoint and reference_checkpoint")
    candidate = resolver.checkpoint(candidate_raw)  # type: ignore[arg-type]
    reference = resolver.checkpoint(reference_raw)  # type: ignore[arg-type]
    if candidate.topology != reference.topology:
        raise ValueError("Arena candidate/reference topologies differ")
    binding = get_topology_binding(candidate.topology)
    raw_config = raw.get("arena_config", raw.get("config"))
    if raw_config is None:
        raw_config = {
            key: raw[key]
            for key in ArenaExecutionConfig.__dataclass_fields__
            if key in raw
        }
    config = _arena_config(raw_config)
    seed = int(raw.get("master_seed", DEFAULT_MASTER_SEED))
    profile = str(raw.get("profile") or binding.default_arena_profile(candidate.effective_config.config))
    search_source = dict(raw)
    if isinstance(raw_config, Mapping):
        nested_search = raw_config.get("search")
        if nested_search is not None:
            search_source["search"] = nested_search
        for key in _ARENA_SEARCH_FIELDS:
            if key in raw_config and key not in search_source:
                search_source[key] = raw_config[key]
    search = _arena_search(search_source, config)
    if candidate.topology == "torus9":
        profile = _torus_profile_with_search(profile, search)
    binding.validate_arena_profile(profile, candidate.effective_config.config)
    startset_raw = raw.get("startset")
    startset = (
        binding.arena_startset(master_seed=seed, games=int(config.games))
        if startset_raw is None
        else StartsetRef.from_dict(startset_raw)  # type: ignore[arg-type]
    )
    return ArenaRunRequest(
        candidate=candidate,
        reference=reference,
        master_seed=seed,
        startset=startset,
        config=config,
        profile=profile,
        workload=dict(raw.get("workload", {})),  # type: ignore[arg-type]
        scientific_contract=raw.get("scientific_contract"),  # type: ignore[arg-type]
        execution_contract=raw.get("execution_contract"),  # type: ignore[arg-type]
        candidate_label=None if raw.get("candidate_label") is None else str(raw["candidate_label"]),
        reference_label=None if raw.get("reference_label") is None else str(raw["reference_label"]),
        comparison=None if raw.get("comparison") is None else str(raw["comparison"]),
        execution_code_commit=execution_commit_from_lineage(candidate.owner_root),
    )


def run_arena_from_config(payload: Mapping[str, object], *, runs_root: str | Path | None = None, arena_runner: object | None = None) -> object:
    _require_file_backed_entrypoint()
    raw = payload.get("arena", payload.get("evaluation", payload))
    if not isinstance(raw, Mapping):
        raise ValueError("arena/evaluation config must be an object")
    resolver = ArtifactResolver(runs_root)
    request = _standalone_arena_request(raw, resolver)
    runner = arena_runner or ArenaRunnerV2()
    if hasattr(runner, "supervisor_policy"):
        setattr(runner, "supervisor_policy", supervision_policy_for(raw, "arena"))
    evaluation_id = runner._evaluation_id(request) if isinstance(runner, ArenaRunnerV2) else str(raw.get("evaluation_id", "evaluation"))
    root = resolver.runs_root / request.candidate.topology / "evaluations" / evaluation_id
    notifier = TelegramNotifier(_notification_paths(root))
    if hasattr(runner, "notifier"):
        setattr(runner, "notifier", notifier)
    try:
        with _authority(mode="arena", topology=request.candidate.topology, run_id=evaluation_id):
            return runner.run(request)  # type: ignore[attr-defined]
    finally:
        flush_all()


def run_calibration_from_config(payload: Mapping[str, object], *, runs_root: str | Path | None = None, arena_runner: object | None = None) -> list[object]:
    _require_file_backed_entrypoint()
    raw = payload.get("calibration", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("calibration config must be an object")
    arms = raw.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("calibration run-spec requires a non-empty arms list")
    calibration_id = str(raw.get("calibration_id", "calibration"))
    results: list[object] = []
    for index, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            raise ValueError("each calibration arm must be an object")
        merged = dict(raw)
        merged.pop("arms", None)
        merged.update(arm)
        merged.setdefault("comparison", f"{calibration_id}:arm:{index}")
        resolver = ArtifactResolver(runs_root)
        request = _standalone_arena_request(merged, resolver)
        runner = arena_runner or ArenaRunnerV2()
        if hasattr(runner, "supervisor_policy"):
            setattr(runner, "supervisor_policy", supervision_policy_for(raw, "calibration"))
        evaluation_id = runner._evaluation_id(request) if isinstance(runner, ArenaRunnerV2) else f"{calibration_id}-arm-{index}"
        root = resolver.runs_root / request.candidate.topology / "evaluations" / evaluation_id
        notifier = TelegramNotifier(_notification_paths(root))
        if hasattr(runner, "notifier"):
            setattr(runner, "notifier", notifier)
        try:
            try:
                notifier.send_now(
                    f"calibration-arm:{evaluation_id}",
                    format_action_started(
                        "CALIBRATION ARM STARTED",
                        calibration=calibration_id,
                        arm=arm.get("id", index),
                        topology=request.candidate.topology,
                        profile=request.profile,
                        games=request.config.games,
                    ),
                )
            except BaseException:
                pass
            with _authority(mode="calibration", topology=request.candidate.topology, run_id=evaluation_id):
                results.append(runner.run(request))  # type: ignore[attr-defined]
        finally:
            flush_all()
    return results


def run_komi_calibration_from_config(payload: Mapping[str, object], *, runs_root: str | Path | None = None, allow_code_rollover: bool | None = None, **runner_kwargs: Any) -> object:
    _require_file_backed_entrypoint()
    config = _komi_calibration_config(payload)
    if allow_code_rollover is not None:
        if type(allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        config = replace(config, allow_code_rollover=allow_code_rollover)
    resolver = runner_kwargs.pop("resolver", None) or ArtifactResolver(runs_root)
    experiment_root = runner_kwargs.get("experiment_root")
    root = Path(experiment_root).resolve() if experiment_root is not None else resolver.runs_root / "torus9" / "evaluations" / config.calibration_id
    notifier = TelegramNotifier(_notification_paths(root))
    arena_runner = runner_kwargs.pop("arena_runner", None) or ArenaRunnerV2(notifier=notifier)
    if hasattr(arena_runner, "supervisor_policy"):
        policy_source = payload.get("calibration", payload)
        if not isinstance(policy_source, Mapping):
            raise ValueError("calibration config must be an object")
        setattr(arena_runner, "supervisor_policy", supervision_policy_for(policy_source, "calibration"))
    child_training = runner_kwargs.pop("child_training", None)
    calibration_config = config
    if child_training is None:
        def child_training(*, parent, config, output_lineage, first_generation):
            del first_generation
            production_arena = calibration_config.production_arena_config or ArenaExecutionConfig()
            selected = float(config.config.arena.get("komi", config.config.self_play.get("komi", 0.5)))
            profile = calibration_config.production_arena_profile
            if profile == "torus9":
                contract = calibration_config.arena_contract
                profile = (
                    f"torus9|komi={selected:g}|simulations={contract.simulations}"
                    f"|cpuct={contract.cpuct:g}|fpu={contract.fpu:g}"
                    f"|watchdog={contract.watchdog}|5ch"
                )
            continuous = ContinuousTrainingConfig(
                parent_checkpoint=parent.ref,
                lineage_id=output_lineage.lineage_id,
                effective_config=config.config,
                generations=calibration_config.generations,
                arena_cadence=calibration_config.production_arena_cadence,
                arena_config=production_arena,
                arena_master_seed=calibration_config.production_arena_master_seed,
                arena_profile=profile,
                arena_workload={"model_gating": "off", "komi": selected},
                allow_code_rollover=calibration_config.allow_code_rollover,
            )
            return ContinuousTrainingRunnerV2(continuous, resolver=resolver, arena_runner=arena_runner, notifier=notifier).run().final_checkpoint
    stop_parent = runner_kwargs.pop("stop_parent", None) or _request_parent_soft_stop
    try:
        with _authority(mode="komi-calibration", topology="torus9", run_id=config.calibration_id):
            runner = ProductionKomiCalibrationRunnerV2(
                config,
                arena_runner=arena_runner,
                resolver=resolver,
                experiment_root=root,
                child_training=child_training,
                stop_parent=stop_parent,
                notifier=notifier,
                **runner_kwargs,
            )
            return runner.run()
    finally:
        flush_all()


def run_continuous_from_config(payload: Mapping[str, object], *, runs_root: str | Path | None = None, allow_code_rollover: bool | None = None, **runner_kwargs: Any) -> object:
    _require_file_backed_entrypoint()
    config = _continuous_config(payload)
    if allow_code_rollover is not None:
        if type(allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        config = replace(config, allow_code_rollover=allow_code_rollover)
    resolver = runner_kwargs.pop("resolver", None) or ArtifactResolver(runs_root)
    root = resolver.runs_root / config.topology / ACTIVE / config.lineage_id
    notifier = TelegramNotifier(_notification_paths(root))
    try:
        with _authority(mode="continuous", topology=config.topology, run_id=config.lineage_id):
            runner = ContinuousTrainingRunnerV2(config, resolver=resolver, notifier=notifier, **runner_kwargs)
            return runner.run()
    finally:
        flush_all()


def run_experiment_from_config(payload: Mapping[str, object], *, runs_root: str | Path | None = None, allow_code_rollover: bool | None = None, **runner_kwargs: Any) -> object:
    _require_file_backed_entrypoint()
    config = _experiment_config(payload)
    if allow_code_rollover is not None:
        if type(allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        config = replace(config, allow_code_rollover=allow_code_rollover)
    resolver = runner_kwargs.pop("resolver", None) or ArtifactResolver(runs_root)
    experiment_root = runner_kwargs.get("experiment_root")
    root = Path(experiment_root).resolve() if experiment_root is not None else resolver.runs_root / config.topology / "experiments" / config.experiment_id
    notifier = TelegramNotifier(_notification_paths(root))
    arena_runner = runner_kwargs.pop("arena_runner", None) or ArenaRunnerV2(notifier=notifier)
    try:
        with _authority(mode="experiment", topology=config.topology, run_id=config.experiment_id):
            runner = ExperimentRunnerV2(config, arena_runner=arena_runner, resolver=resolver, notifier=notifier, **runner_kwargs)
            policy_source = payload.get("experiment", payload)
            if not isinstance(policy_source, Mapping):
                raise ValueError("experiment config must be an object")
            policy = supervision_policy_for(policy_source, "arena")
            selected_arena_runner = getattr(runner, "arena_runner", None)
            if hasattr(selected_arena_runner, "supervisor_policy"):
                setattr(selected_arena_runner, "supervisor_policy", policy)
            selected_train_one = getattr(runner, "train_one", None)
            if hasattr(selected_train_one, "supervisor_policy"):
                setattr(selected_train_one, "supervisor_policy", supervision_policy_for(policy_source, "generation"))
            return runner.run()
    finally:
        flush_all()


def run_workflow_from_config(
    payload: Mapping[str, object],
    *,
    runs_root: str | Path | None = None,
    workflow_handlers: Mapping[str, Any] | None = None,
    **runner_kwargs: Any,
) -> object:
    """Run a durable composition of existing V2 actions."""
    _require_file_backed_entrypoint()
    raw = payload.get("workflow", payload.get("scenario", payload))
    if not isinstance(raw, Mapping):
        raise ValueError("workflow/scenario config must be an object")
    spec = WorkflowSpec.from_dict(raw)
    root = (
        Path(runs_root or RUNS_ROOT).resolve()
        / spec.topology
        / "orchestration"
        / "workflows"
        / spec.workflow_id
    )
    handlers = dict(workflow_handlers or {})

    def action_config(config: Mapping[str, object]) -> dict[str, object]:
        selected = dict(config)
        if spec.supervision and "supervision" not in selected:
            selected["supervision"] = dict(spec.supervision)
        return selected

    # These adapters intentionally call the already-existing entrypoints.  A
    # workflow is not a second implementation of Arena/training/calibration.
    handlers.setdefault(
        "arena",
        lambda *, config, **_: run_arena_from_config(
            {"arena": action_config(config)}, runs_root=runs_root, **runner_kwargs
        ),
    )
    handlers.setdefault(
        "calibration",
        lambda *, config, **_: run_calibration_from_config(
            {"calibration": action_config(config)}, runs_root=runs_root, **runner_kwargs
        ),
    )
    handlers.setdefault(
        "continuous_training",
        lambda *, config, **_: run_continuous_from_config(
            {"continuous": action_config(config)}, runs_root=runs_root, **runner_kwargs
        ),
    )
    handlers.setdefault(
        "experiment",
        lambda *, config, **_: run_experiment_from_config(
            {"experiment": action_config(config)}, runs_root=runs_root, **runner_kwargs
        ),
    )
    try:
        with _authority(mode="workflow", topology=spec.topology, run_id=spec.workflow_id):
            return WorkflowRunner(spec, root=root, handlers=handlers).run()
    finally:
        flush_all()


def run_spec(payload: Mapping[str, object], *, runs_root: str | Path | None = None, **runner_kwargs: Any) -> object:
    spec = RunSpecV2.from_dict(payload)
    wrapped = {spec.mode.value: dict(spec.payload)}
    if spec.mode is RunMode.CONTINUOUS:
        return run_continuous_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    if spec.mode in {RunMode.ARENA, RunMode.EVALUATION}:
        return run_arena_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    if spec.mode is RunMode.EXPERIMENT:
        return run_experiment_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    if spec.mode is RunMode.CALIBRATION:
        return run_calibration_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    if spec.mode in {RunMode.WORKFLOW, RunMode.SCENARIO}:
        return run_workflow_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    raise AssertionError(spec.mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    telegram = subparsers.add_parser("telegram-test", help="send one explicit transport test")
    telegram.set_defaults(kind="telegram")
    command = subparsers.add_parser("run", help="run one declarative Orchestrator V2 run-spec")
    command.add_argument("config", type=Path)
    command.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    command.set_defaults(kind="run")
    for name in ("continuous", "experiment", "komi-calibration", "workflow"):
        command = subparsers.add_parser(name, help=f"legacy-compatible V2 {name} JSON plan")
        command.add_argument("config", type=Path)
        command.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
        command.add_argument("--allow-code-rollover", action="store_true", default=None)
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
    if args.kind == "run":
        run_spec(payload, runs_root=args.runs_root)
    elif args.kind == "continuous":
        run_continuous_from_config(payload, runs_root=args.runs_root, allow_code_rollover=args.allow_code_rollover)
    elif args.kind == "experiment":
        run_experiment_from_config(payload, runs_root=args.runs_root, allow_code_rollover=args.allow_code_rollover)
    else:
        run_komi_calibration_from_config(payload, runs_root=args.runs_root, allow_code_rollover=args.allow_code_rollover)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_v2_config", "run_arena_from_config", "run_calibration_from_config", "run_continuous_from_config",
    "run_experiment_from_config", "run_komi_calibration_from_config", "run_workflow_from_config", "run_spec",
]
