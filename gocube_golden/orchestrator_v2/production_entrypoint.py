"""Single declarative production boundary for Orchestrator V2."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import __main__
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

from ..process_supervision import atomic_write_text
from ..provenance import canonical_json
from ..run_storage import ACTIVE, RUNS_ROOT
from ..telegram_notifier import TelegramError, telegram_test
from ..notifications import coerce_event_sink, create_telegram_dispatcher, flush_all, migrate_legacy_storage, operator_event
from .arena_runner import ArenaRunRequest, ArenaRunnerV2
from .artifact_resolver import ArtifactResolver
from .continuous_training import ContinuousTrainingConfig, ContinuousTrainingRunnerV2
from .contracts import StartsetRef
from .generation_runner import OutputLineage
from .execution_permit import _production_authority
from .experiment_plan import ExperimentConfig
from ..scenarios.calibration import CalibrationArm, CalibrationRunner
from ..scenarios.experiment.runner import ExperimentRunnerV2
from .immutable_runtime import execution_commit_from_lineage, resolve_execution_commit
from .komi_calibration import KomiCalibrationConfig
from ..scenarios.komi.runner import ProductionKomiCalibrationRunnerV2
from .run_spec import RunMode, RunSpecV2, supervision_policy_for
from .supervisor import SupervisorPolicy
from .topology_binding import get_topology_binding
from .version import mark_v2_process
from .workflow import WorkflowRunner, WorkflowSpec
from .torus9_production import Torus9ProductionLineage
from ..performance_tuning import (
    Mode as TuningMode,
    PerformanceTuningRunner,
    Plan as TuningPlan,
    ProductionTrainOneExecutor,
    scientific_contract_fingerprint,
)
from tools.arena_engine import ArenaExecutionConfig, DEFAULT_MASTER_SEED


def TelegramNotifier(paths: object) -> object:
    """Compatibility composition name; returns the sole structured service."""
    return create_telegram_dispatcher(Path(getattr(paths, "root")))


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
    effective_config = raw["effective_config"]
    # Workflow selection commonly resolves a scalar komi while the training
    # action still receives the base effective config.  Apply it to the real
    # immutable config contract here; a free-standing ``config.komi`` field
    # must never be silently ignored by the training runner.
    if "komi" in raw:
        from .komi_calibration import effective_config_with_komi
        from ._continuous_training_core import _effective_config as normalize_effective_config

        try:
            effective_config = normalize_effective_config(effective_config)  # type: ignore[arg-type]
            effective_config = effective_config_with_komi(
                effective_config, float(raw["komi"])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"continuous.komi cannot be applied to effective_config: {exc}") from exc
    selected_profile = raw.get("selected_profile", raw.get("execution_profile"))
    if isinstance(selected_profile, str):
        try:
            selected_profile = json.loads(Path(selected_profile).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load selected_profile: {selected_profile}") from exc
    return ContinuousTrainingConfig(
        parent_checkpoint=raw["parent_checkpoint"],  # type: ignore[arg-type]
        lineage_id=str(raw["lineage_id"]),
        effective_config=effective_config,  # type: ignore[arg-type]
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
        execution_profile=selected_profile,  # type: ignore[arg-type]
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
    unknown -= {"search", "evaluation"}
    if unknown:
        raise ValueError(
            "arena_config contains unsupported fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    raw.pop("search", None)
    raw.pop("evaluation", None)
    for field in _ARENA_SEARCH_FIELDS:
        raw.pop(field, None)
    return ArenaExecutionConfig(**raw)


def _arena_search(raw: Mapping[str, object], config: ArenaExecutionConfig) -> dict[str, object]:
    if "search" in raw and "evaluation" in raw:
        raise ValueError("Arena config specifies both search and evaluation")
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
    for key in nested:
        if key in raw and key not in {"search", "evaluation"}:
            raise ValueError(
                f"Arena search field {key!r} conflicts with its top-level value"
            )
    values: dict[str, object] = {}
    for key in _ARENA_SEARCH_FIELDS:
        if key in nested:
            values[key] = nested[key]
        elif key in raw:
            values[key] = raw[key]
    if "simulations" not in values and "mcts_simulations" in values:
        values["simulations"] = values.pop("mcts_simulations")
    elif "simulations" in values and "mcts_simulations" in values:
        raise ValueError("Arena search specifies both simulations and mcts_simulations")
    if "watchdog" not in values and "technical_move_limit" in values:
        values["watchdog"] = values.pop("technical_move_limit")
    elif "watchdog" in values and "technical_move_limit" in values:
        raise ValueError("Arena search specifies both watchdog and technical_move_limit")
    fixed = {
        "root_noise": False,
        "temperature": 0.0,
        "fast_search": False,
        "resign": False,
        "deterministic_tie_break": True,
    }
    for key, expected in fixed.items():
        if key not in values:
            continue
        actual = values[key]
        if key == "temperature":
            import math
            try:
                valid = math.isfinite(float(actual)) and float(actual) == 0.0
            except (TypeError, ValueError):
                valid = False
        else:
            valid = type(actual) is bool and actual is expected
        if not valid:
            raise ValueError(
                f"Arena search.{key} is not supported by the Arena engine; "
                f"the only supported value is {expected!r}"
            )
    return values


def _torus_profile_with_search(profile: str, search: Mapping[str, object]) -> str:
    from tools.arena_profiles import get_profile

    parsed = get_profile(profile)
    fixed = {
        "root_noise": False,
        "temperature": 0.0,
        "fast_search": False,
        "resign": False,
        "deterministic_tie_break": True,
    }
    for key, expected in fixed.items():
        if key not in search:
            continue
        actual = search[key]
        if key == "temperature":
            import math
            try:
                valid = math.isfinite(float(actual)) and float(actual) == 0.0
            except (TypeError, ValueError):
                valid = False
        else:
            valid = type(actual) is bool and actual is expected
        if not valid:
            raise ValueError(
                f"Arena search.{key} is not supported by the Torus9 engine; "
                f"the only supported value is {expected!r}"
            )
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
    profile_was_explicit = raw.get("profile") is not None
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
    elif search:
        if "komi" in search:
            raise ValueError(
                "Cube Arena does not expose komi as a search parameter; "
                "put rules compatibility in the effective config"
            )
        if profile_was_explicit:
            raise ValueError(
                "Cube Arena config specifies both an explicit profile and independent search values"
            )
        from gocube_golden.cube_arena_contract_v2 import CubeArenaSearchConfig
        from tools.arena_profiles import get_profile

        default_profile = get_profile(profile)
        defaults = default_profile.search_config
        cube_search = CubeArenaSearchConfig(
            simulations=int(search.get("simulations", defaults.simulations)),
            cpuct=float(search.get("cpuct", defaults.cpuct)),
            fpu=float(search.get("fpu", defaults.fpu)),
            watchdog=int(search.get("watchdog", defaults.watchdog)),
            deterministic_tie_break=True,
        )
        profile = type(default_profile)(
            size=int(getattr(default_profile, "size")),
            search_config=cube_search,
        ).profile_id
    binding.validate_arena_profile(profile, candidate.effective_config.config)
    requested_execution_commit = raw.get("execution_code_commit")
    if requested_execution_commit is None:
        execution_code_commit = execution_commit_from_lineage(candidate.owner_root)
    else:
        # Historical converted checkpoints can be referenced by a current
        # production workflow without mutating their lineage manifest.  The
        # explicit pin is still resolved against this repository before any
        # child is spawned; ImmutableRuntimeManager repeats the same check at
        # the process boundary and materializes the exact immutable worktree.
        execution_code_commit = resolve_execution_commit(
            _repo_root(), requested_execution_commit
        )
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
        execution_code_commit=execution_code_commit,
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
    if hasattr(runner, "event_sink"):
        setattr(runner, "event_sink", notifier)
    try:
        with _authority(mode="arena", topology=request.candidate.topology, run_id=evaluation_id):
            return runner.run(request)  # type: ignore[attr-defined]
    finally:
        flush_all()


def _calibration_result_record(
    result: object,
    *,
    request: ArenaRunRequest,
    arm: Mapping[str, object],
    arm_index: int,
) -> dict[str, object]:
    """Persist the small, JSON-only result contract used by workflow refs."""
    if isinstance(result, Mapping):
        summary_value = result.get("summary", result)
        summary = dict(summary_value) if isinstance(summary_value, Mapping) else {}
        identity_value = result.get("identity", request.identity if hasattr(request, "identity") else {})
        evaluation_id = result.get("evaluation_id", arm.get("evaluation_id", f"calibration-arm-{arm_index}"))
        evaluation_fingerprint = result.get("evaluation_fingerprint")
        validity = str(result.get("validity", summary.get("validity", "INVALID"))).upper()
    else:
        summary_value = getattr(result, "summary", {})
        summary = dict(summary_value) if isinstance(summary_value, Mapping) else {}
        raw_identity = getattr(result, "identity", None)
        identity_value = raw_identity.to_dict() if hasattr(raw_identity, "to_dict") else raw_identity
        evaluation_id = getattr(result, "evaluation_id", f"calibration-arm-{arm_index}")
        evaluation_fingerprint = getattr(result, "evaluation_fingerprint", None)
        validity = str(getattr(result, "validity", summary.get("validity", "INVALID"))).upper()
    if not isinstance(identity_value, Mapping):
        identity_value = {}
    if evaluation_fingerprint is None:
        evaluation_fingerprint = identity_value.get("fingerprint")
    scientific = dict(request.scientific_contract or {})
    if not scientific:
        from tools.arena_profiles import get_profile
        scientific = dict(get_profile(request.profile).scientific_contract(request.config))
    komi = scientific.get("komi", arm.get("komi"))
    wld = summary.get("W/L/D")
    if isinstance(wld, (list, tuple)) and len(wld) == 3:
        black_wins, white_wins, draws = (int(wld[0]), int(wld[1]), int(wld[2]))
    else:
        black_wins = int(summary.get("black_wins", summary.get("candidate_wins", 0)))
        white_wins = int(summary.get("white_wins", summary.get("reference_wins", 0)))
        draws = int(summary.get("draws", 0))
    valid_games = int(summary.get("valid_games", summary.get("games_valid", black_wins + white_wins + draws)))
    black_win_rate = (black_wins / valid_games) if valid_games else float("nan")
    if not math.isfinite(black_win_rate):
        metrics: dict[str, object] = {"valid_games": valid_games}
    else:
        metrics = {
            "games": int(summary.get("games", summary.get("games_requested", valid_games))),
            "valid_games": valid_games,
            "black_wins": black_wins,
            "white_wins": white_wins,
            "draws": draws,
            "black_win_rate": black_win_rate,
            "bias": abs(black_win_rate - 0.5),
        }
    return {
        "schema": "gocube-orchestrator-v2-calibration-result-v1",
        "arm_index": arm_index,
        "arm_id": arm.get("id", arm_index),
        "parameters": scientific,
        "komi": komi,
        "evaluation_id": str(evaluation_id),
        "result_id": str(evaluation_id),
        "evaluation_fingerprint": evaluation_fingerprint,
        "identity": dict(identity_value),
        "validity": validity,
        "metrics": metrics,
        "summary": summary,
        # This is a compact config contract, never a model/replay artifact.
        "effective_config": request.candidate.effective_config.config.to_dict(),
    }


def run_calibration_from_config(payload: Mapping[str, object], *, runs_root: str | Path | None = None, arena_runner: object | None = None) -> list[dict[str, object]]:
    _require_file_backed_entrypoint()
    raw = payload.get("calibration", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("calibration config must be an object")
    arms = raw.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("calibration run-spec requires a non-empty arms list")
    calibration_id = str(raw.get("calibration_id", "calibration"))
    scenario_arms: list[CalibrationArm] = []
    for index, arm in enumerate(arms):
        if not isinstance(arm, Mapping):
            raise ValueError("each calibration arm must be an object")
        scenario_arms.append(CalibrationArm(str(arm.get("id", index)), dict(arm)))

    def execute_arm(scenario_arm: CalibrationArm, index: int) -> dict[str, object]:
        arm = scenario_arm.request
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
        if hasattr(runner, "event_sink"):
            setattr(runner, "event_sink", notifier)
        try:
            try:
                coerce_event_sink(notifier).publish(
                    operator_event(
                        "CALIBRATION_STARTED",
                        topology=request.candidate.topology,
                        owner_type="experiment",
                        owner_id=calibration_id,
                        action_id=f"{calibration_id}:arm:{evaluation_id}",
                        payload={
                            "calibration_id": calibration_id,
                            "arm": arm.get("id", index),
                            "profile": request.profile,
                            "games": request.config.games,
                        },
                        correlation_id=calibration_id,
                    )
                )
            except Exception:
                pass
            with _authority(mode="calibration", topology=request.candidate.topology, run_id=evaluation_id):
                result = runner.run(request)  # type: ignore[attr-defined]
                return _calibration_result_record(
                    result,
                    request=request,
                    arm=arm,
                    arm_index=index,
                )
        finally:
            flush_all()

    return CalibrationRunner(scenario_arms, execute_arm).run()


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


def run_performance_tuning_from_config(payload: Mapping[str, object], *, runs_root: str | Path | None = None, **runner_kwargs: Any) -> object:
    """Run the explicit, budgeted performance-tuning mode.

    The plan prepares one normal production lineage through the existing
    lineage factory.  ``dry_run`` validates and prints the plan without
    creating the lineage, starting a child, or publishing an event.
    """

    _require_file_backed_entrypoint()
    raw = payload.get("performance_tuning", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("performance_tuning config must be an object")
    required = ("tuning_id", "parent_checkpoint", "effective_config", "baseline", "modes", "measurement_budget")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError("performance_tuning config is missing: " + ", ".join(missing))
    resolver = runner_kwargs.pop("resolver", None) or ArtifactResolver(runs_root)
    parent = resolver.checkpoint(raw["parent_checkpoint"])  # type: ignore[arg-type]
    from ._continuous_training_core import _effective_config as normalize_effective_config

    effective_config = normalize_effective_config(raw["effective_config"])  # type: ignore[arg-type]
    topology = str(raw.get("topology", effective_config.topology))
    if topology != effective_config.topology or topology != parent.topology:
        raise ValueError("performance_tuning topology does not match the parent and effective config")
    baseline = TuningMode.from_dict(raw["baseline"])  # type: ignore[arg-type]
    raw_modes = raw["modes"]
    if not isinstance(raw_modes, list):
        raise ValueError("performance_tuning.modes must be a list")
    modes = tuple(TuningMode.from_dict(item, label=f"modes[{index}]") for index, item in enumerate(raw_modes))
    workers = raw.get("workers", effective_config.execution.get("workers"))
    if type(workers) is not int or workers <= 0:
        raise ValueError("performance_tuning workers must be a positive integer")
    lineage_id = str(raw.get("lineage_id", raw.get("owner_id", f"tuning-{raw['tuning_id']}")))
    owner_id = str(raw.get("owner_id", lineage_id))
    predicted_root = resolver.runs_root / topology / ACTIVE / lineage_id
    raw_contract = raw.get("measurement_contract", {})
    if not isinstance(raw_contract, Mapping):
        raise ValueError("performance_tuning.measurement_contract must be an object")
    measurement_contract = dict(raw_contract)
    measurement_contract.setdefault("expected_games", effective_config.self_play.get("games_per_iteration", effective_config.self_play.get("games")))
    plan = TuningPlan(
        tuning_id=str(raw["tuning_id"]),
        topology=topology,
        parent_checkpoint=parent.ref,
        baseline=baseline,
        modes=modes,
        workers=workers,
        scientific_config_fingerprint=scientific_contract_fingerprint(effective_config),
        measurement_budget=raw["measurement_budget"],
        measurement_contract=measurement_contract,
        owner_id=owner_id,
        owner_root=predicted_root,
        execution_code_commit=None if raw.get("execution_code_commit") is None else str(raw["execution_code_commit"]),
        device_characteristics=dict(raw.get("device_characteristics", {})),  # type: ignore[arg-type]
        finish_behavior=str(raw.get("finish_behavior", "export_profile")),
    )
    dry_run = raw.get("dry_run", False)
    if type(dry_run) is not bool:
        raise ValueError("performance_tuning.dry_run must be a boolean")
    if dry_run:
        return PerformanceTuningRunner(plan, execute_generation=lambda **_kwargs: None).dry_run()
    if plan.finish_behavior == "continue_training":
        raise ValueError("standalone performance_tuning currently supports finish_behavior=export_profile only")
    lineage_factory = runner_kwargs.pop("lineage_factory", None) or Torus9ProductionLineage(resolver.runs_root)
    prepare_args = {
        "topology": topology,
        "lineage_id": lineage_id,
        "parent": parent,
        "effective_config": effective_config,
        "experiment_id": f"performance-tuning-{plan.tuning_id}",
        "arm_id": "performance-tuning",
        "allow_code_rollover": bool(raw.get("allow_code_rollover", False)),
    }
    root, resolved_config = lineage_factory.prepare(**prepare_args)
    manifest_path = Path(root) / "manifest.json"
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("performance tuning owner manifest cannot be read") from exc
    if not isinstance(manifest_payload, Mapping):
        raise RuntimeError("performance tuning owner manifest is malformed")
    execution_commit = manifest_payload.get("execution_code_commit", manifest_payload.get("git_commit"))
    if not isinstance(execution_commit, str) or not execution_commit:
        raise RuntimeError("performance tuning owner manifest has no execution code commit")
    if plan.execution_code_commit is not None and plan.execution_code_commit != execution_commit:
        raise ValueError("performance_tuning execution_code_commit does not match owner lineage")
    if plan.execution_code_commit is None:
        plan = replace(plan, execution_code_commit=execution_commit)
    output_lineage = OutputLineage(topology, lineage_id, root)
    train_one = runner_kwargs.pop("train_one", None) or ProductionTrainOne(
        resolver=resolver,
        supervisor_policy=supervision_policy_for(raw, "generation"),
    )
    executor = runner_kwargs.pop("execute_generation", None) or ProductionTrainOneExecutor(
        train_one=train_one,
        config=resolved_config,
        output_lineage=output_lineage,
        workers=workers,
        resolver=resolver,
    )

    def read_metrics(_result: object, generation: int) -> Mapping[str, object]:
        summary_path = root / f"iter-{generation:02d}-summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return summary if isinstance(summary, Mapping) else {}

    event_sink = runner_kwargs.pop("event_sink", None)
    if event_sink is None:
        notifier = TelegramNotifier(_notification_paths(root))
        event_sink = notifier
    try:
        with _authority(mode="performance-tuning", topology=topology, run_id=plan.tuning_id):
            runner = PerformanceTuningRunner(
                plan,
                execute_generation=executor,
                owner_root=root,
                metrics_reader=read_metrics,
                event_sink=event_sink,
            )
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


def _workflow_checkpoint_ref(value: object) -> dict[str, object]:
    candidate = getattr(value, "ref", value)
    if hasattr(candidate, "to_dict") and callable(getattr(candidate, "to_dict")):
        payload = candidate.to_dict()
    elif isinstance(candidate, Mapping):
        payload = dict(candidate)
    else:
        raise ValueError("workflow action returned no checkpoint reference")
    if not isinstance(payload, dict) or not {"topology", "lineage_id", "checkpoint_id"} <= set(payload):
        raise ValueError("workflow action returned an incomplete checkpoint reference")
    return payload


def _workflow_arena_result(result: object, *, resolver: ArtifactResolver) -> dict[str, object]:
    summary_value = getattr(result, "summary", None)
    summary = dict(summary_value) if isinstance(summary_value, Mapping) else {}
    identity = getattr(result, "identity", None)
    candidate = getattr(identity, "candidate", None)
    reference = getattr(identity, "reference", None)
    if candidate is None or reference is None:
        raise ValueError("workflow Arena result has no checkpoint identity")
    candidate_ref = _workflow_checkpoint_ref(candidate)
    reference_ref = _workflow_checkpoint_ref(reference)
    candidate_node = resolver.checkpoint(candidate)
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("workflow Arena result has no scientific metrics")
    return {
        "schema": "gocube-orchestrator-v2-arena-result-v1",
        "evaluation_id": str(getattr(result, "evaluation_id", "")),
        "evaluation_fingerprint": str(getattr(result, "evaluation_fingerprint", "")),
        "validity": str(getattr(result, "validity", "INVALID")),
        "candidate_checkpoint": candidate_ref,
        "reference_checkpoint": reference_ref,
        "effective_config": candidate_node.effective_config.config.to_dict(),
        "identity": identity.to_dict() if hasattr(identity, "to_dict") else {},
        "metrics": dict(metrics),
        "summary": summary,
        "output_dir": str(getattr(result, "output_dir", "")),
    }


def _workflow_training_result(result: object) -> dict[str, object]:
    final_checkpoint = getattr(result, "final_checkpoint", None)
    if final_checkpoint is None:
        raise ValueError("workflow training result has no final checkpoint")
    effective = getattr(final_checkpoint, "effective_config", None)
    effective_config = getattr(effective, "config", effective)
    if not hasattr(effective_config, "to_dict"):
        raise ValueError("workflow training result has no effective config contract")
    arenas: list[dict[str, object]] = []
    for arena in getattr(result, "arenas", ()):
        arenas.append(
            {
                "evaluation_id": str(getattr(arena, "evaluation_id", "")),
                "evaluation_fingerprint": str(getattr(arena, "evaluation_fingerprint", "")),
                "validity": str(getattr(arena, "validity", "INVALID")),
                "summary": dict(getattr(arena, "summary", {})),
            }
        )
    checkpoint = _workflow_checkpoint_ref(final_checkpoint)
    return {
        "schema": "gocube-orchestrator-v2-training-result-v1",
        "state": str(getattr(result, "state", "UNKNOWN")),
        "checkpoint": checkpoint,
        "final_checkpoint": checkpoint,
        "effective_config": effective_config.to_dict(),
        "lineage_root": str(getattr(result, "lineage_root", "")),
        "soft_stop_requested": bool(getattr(result, "soft_stop_requested", False)),
        "arenas": arenas,
    }


def _workflow_spec_from_payload(payload: Mapping[str, object]) -> WorkflowSpec:
    raw = payload.get("workflow", payload.get("scenario", payload))
    if not isinstance(raw, Mapping):
        raise ValueError("workflow/scenario config must be an object")
    return WorkflowSpec.from_dict(raw)


def _launch_durable_workflow_controller(
    config_path: Path,
    *,
    runs_root: Path,
) -> dict[str, object]:
    """Start a detached controller which owns the workflow until completion."""
    payload = load_v2_config(config_path)
    spec = _workflow_spec_from_payload(payload)
    root = runs_root.resolve() / spec.topology / "orchestration" / "workflows" / spec.workflow_id
    controller_path = root / "runtime" / "controller.json"
    if controller_path.is_file():
        try:
            current = json.loads(controller_path.read_text(encoding="utf-8"))
            pid = int(current["pid"])
            process_group = int(current["process_group"])
            if os.getpgid(pid) == process_group:
                raise RuntimeError(
                    f"workflow {spec.workflow_id!r} already has a live controller {pid}"
                )
        except (OSError, KeyError, TypeError, ValueError, ProcessLookupError):
            pass
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "controller.log"
    command = [
        sys.executable,
        "-m",
        "gocube_golden.orchestrator_v2.production_entrypoint",
        "workflow",
        str(config_path.resolve()),
        "--runs-root",
        str(runs_root.resolve()),
        "--controller",
    ]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=_repo_root(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return {
        "state": "STARTED",
        "workflow_id": spec.workflow_id,
        "controller_pid": int(process.pid),
        "controller_process_group": int(os.getpgid(process.pid)),
        "controller_root": str(root),
        "log": str(log_path),
    }


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

    workflow_resolver = runner_kwargs.get("resolver") or ArtifactResolver(runs_root)
    arena_kwargs = {
        key: runner_kwargs[key]
        for key in ("arena_runner",)
        if key in runner_kwargs
    }

    def arena_action(*, config, **_):
        result = run_arena_from_config(
            {"arena": action_config(config)}, runs_root=runs_root, **arena_kwargs
        )
        return _workflow_arena_result(result, resolver=workflow_resolver)

    def calibration_action(*, config, **_):
        return run_calibration_from_config(
            {"calibration": action_config(config)}, runs_root=runs_root, **arena_kwargs
        )

    def training_action(*, config, **_):
        result = run_continuous_from_config(
            {"continuous": action_config(config)}, runs_root=runs_root, **runner_kwargs
        )
        return _workflow_training_result(result)

    # These adapters intentionally call the already-existing entrypoints.  A
    # workflow is not a second implementation of Arena/training/calibration.
    handlers.setdefault("arena", arena_action)
    handlers.setdefault("calibration", calibration_action)
    handlers.setdefault("continuous_training", training_action)
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
    if spec.mode is RunMode.PERFORMANCE_TUNING:
        return run_performance_tuning_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    if spec.mode in {RunMode.ARENA, RunMode.EVALUATION}:
        return run_arena_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    if spec.mode is RunMode.EXPERIMENT:
        return run_experiment_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    if spec.mode is RunMode.CALIBRATION:
        return run_calibration_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    if spec.mode in {RunMode.WORKFLOW, RunMode.SCENARIO}:
        return run_workflow_from_config(wrapped, runs_root=runs_root, **runner_kwargs)
    raise AssertionError(spec.mode)


def drain_notifications(root: str | Path, *, timeout: float = 7.0) -> dict[str, int]:
    """Retry one owner root without starting training, Arena, or a workflow."""
    dispatcher = create_telegram_dispatcher(Path(root).resolve())
    legacy = migrate_legacy_storage(Path(root).resolve(), dispatcher.store)
    requeued = dispatcher.requeue_blocked()
    dispatcher.flush(timeout)
    states = [dispatcher.store.read_delivery(event.event_id) for event in dispatcher.store.iter_events()]
    pending = sum(1 for state in states if state is not None and state.status != "DELIVERED")
    delivered = sum(1 for state in states if state is not None and state.status == "DELIVERED")
    dispatcher.close(timeout=max(0.0, timeout - 0.01))
    return {"requeued": requeued, "pending": pending, "delivered": delivered, "legacy_migrated": legacy["migrated"], "legacy_unknown": legacy["unknown"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    telegram = subparsers.add_parser("telegram-test", help="send one explicit transport test")
    telegram.set_defaults(kind="telegram")
    drain = subparsers.add_parser("notifications-drain", help="retry one saved notification root")
    drain.add_argument("root", type=Path)
    drain.add_argument("--timeout", type=float, default=7.0)
    drain.set_defaults(kind="drain")
    command = subparsers.add_parser("run", help="run one declarative Orchestrator V2 run-spec")
    command.add_argument("config", type=Path)
    command.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    command.set_defaults(kind="run")
    for name in ("continuous", "performance-tuning", "experiment", "komi-calibration", "workflow"):
        command = subparsers.add_parser(name, help=f"legacy-compatible V2 {name} JSON plan")
        command.add_argument("config", type=Path)
        command.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
        command.add_argument("--allow-code-rollover", action="store_true", default=None)
        if name == "workflow":
            command.add_argument("--controller", action="store_true", help=argparse.SUPPRESS)
        command.set_defaults(kind=name)
    args = parser.parse_args(argv)
    if args.kind == "telegram":
        try:
            telegram_test()
        except TelegramError as exc:
            raise SystemExit(f"Telegram test failed: {exc}") from None
        print("Telegram test message sent.")
        return 0
    if args.kind == "drain":
        print(json.dumps(drain_notifications(args.root, timeout=args.timeout), sort_keys=True))
        return 0
    payload = load_v2_config(args.config)
    result: object
    if args.kind == "run":
        result = run_spec(payload, runs_root=args.runs_root)
    elif args.kind == "continuous":
        result = run_continuous_from_config(payload, runs_root=args.runs_root, allow_code_rollover=args.allow_code_rollover)
    elif args.kind == "experiment":
        result = run_experiment_from_config(payload, runs_root=args.runs_root, allow_code_rollover=args.allow_code_rollover)
    elif args.kind == "performance-tuning":
        result = run_performance_tuning_from_config(payload, runs_root=args.runs_root)
    elif args.kind == "workflow":
        if getattr(args, "controller", False):
            result = run_workflow_from_config(payload, runs_root=args.runs_root)
        else:
            result = _launch_durable_workflow_controller(
                args.config,
                runs_root=args.runs_root,
            )
    else:
        result = run_komi_calibration_from_config(payload, runs_root=args.runs_root, allow_code_rollover=args.allow_code_rollover)
    if hasattr(result, "to_dict") and callable(getattr(result, "to_dict")):
        result = result.to_dict()
    if isinstance(result, Mapping):
        print(json.dumps(dict(result), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_v2_config", "run_arena_from_config", "run_calibration_from_config", "run_continuous_from_config", "run_performance_tuning_from_config",
    "run_experiment_from_config", "run_komi_calibration_from_config", "run_workflow_from_config", "run_spec", "drain_notifications",
]
