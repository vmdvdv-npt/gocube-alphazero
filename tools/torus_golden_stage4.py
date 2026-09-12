#!/usr/bin/env python3
"""Run the controlled Stage 3 -> Stage 4 Golden Torus experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Any, Mapping, Sequence
from multiprocessing import get_context

import torch

from gocube_golden.arena import MappedResult, SequentialGoldenArena, TerminationReason, write_records_jsonl
from gocube_golden.neural import GoldenGraphNetV1, GoldenNeuralEvaluator, count_parameters, model_hash
from gocube_golden.players import SearchPlayer
from gocube_golden.provenance import CodeIdentity, PlayerIdentity, capture_code_identity, derive_seed, file_sha256, sha256_fingerprint
from gocube_golden.result import Winner
from gocube_golden.stage3_contract import load_profile as load_stage3_profile
from gocube_golden.stage4 import (
    EVALUATION_ACCEPTED_TOTAL,
    EVALUATION_CONTRACT_ID,
    EVALUATION_MASTER_SEED,
    EVALUATION_PREFIX_LENGTHS,
    HOEFFDING_ALPHA,
    STAGE4_ARENA_MASTER_SEED,
    STAGE4_MODEL_INIT_SEED,
    STAGE4_PROFILE_ID,
    STAGE4_SELFPLAY_MASTER_SEED,
    audit_selfplay_records,
    diagnostic_subset,
    evaluate_model_samples,
    evaluation_start_fingerprint,
    fixed_observation_samples,
    freeze_evaluation_v2,
    high_reuse_schedule,
    load_frozen_starts,
    low_reuse_schedule,
    parameter_diagnostics,
    policy_value_entropy,
    runtime_telemetry,
    state_from_start_row,
    summarize_pair_records,
    train_batch_schedule,
)
from gocube_golden.search import SequentialPUCT
from gocube_golden.state import initial_state
from gocube_golden.training import (
    GoldenSelfPlayRunner,
    GoldenTrainingSample,
    SelfPlayGameRecord,
    SelfPlayPosition,
    build_replay_samples,
    load_checkpoint,
    run_selfplay_games,
    save_checkpoint,
    state_identity,
    write_jsonl,
    z_target,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_STAGE3_RUN = ROOT / "runs" / "torus-golden-stage3" / "torus-golden-stage3-seed1-v5"
OLD_STAGE3_LABELS = ("M0", "M1", "M2", "M3", "M4")

_PROCESS_MODEL = None
_PROCESS_CONFIG: dict[str, object] = {}


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stage4_profile() -> dict[str, Any]:
    path = ROOT / "configs" / "gocube" / "torus_golden_training_v2_data_rich.json"
    profile = load_json(path)
    if profile.get("profile_id") != STAGE4_PROFILE_ID:
        raise ValueError("Stage 4 profile id drift")
    profile["profile_fingerprint"] = sha256_fingerprint(profile)
    profile["config_sha256"] = file_sha256(path)
    return profile


def checkpoint_metadata(
    semantic_profile: Mapping[str, Any],
    *,
    run_id: str,
    label: str,
    parent: str | None,
    code,
    model: GoldenGraphNetV1,
    device: torch.device,
    completed_games: int,
    valid_replay_positions: int,
    optimizer_updates: int,
    train_samples_consumed: int,
    model_init_seed: int,
    stage4_profile: Mapping[str, Any],
    arm: str | None = None,
) -> dict[str, object]:
    return {
        "checkpoint_schema_version": 1,
        "checkpoint_label": label,
        "architecture_id": model.architecture_id,
        "architecture_config": model.architecture_config,
        "rules_profile_id": semantic_profile["frozen_identities"]["rules_profile_id"],
        "rules_fingerprint": semantic_profile["frozen_identities"]["rules_fingerprint"],
        "topology_fingerprint": semantic_profile["frozen_identities"]["topology_fingerprint"],
        "board_size": [5, 5],
        "point_id_order_identity": "row-major-yx:point_id=y*width+x",
        "komi": 0.5,
        "observation_schema_id": semantic_profile["observation"]["schema_id"],
        "observation_schema_version": semantic_profile["observation"]["schema_version"],
        "observation_fingerprint": semantic_profile["observation"]["fingerprint"],
        "target_contract_id": semantic_profile["target"]["contract_id"],
        "target_contract_version": semantic_profile["target"]["contract_version"],
        "target_fingerprint": semantic_profile["target"]["fingerprint"],
        "value_head_semantics": "side-to-move:[WIN,DRAW,LOSS]",
        "network_heads_and_shapes": {"policy": [26], "value": [3]},
        # The Stage-3 profile owns the frozen network/rules/search semantics;
        # the additional Stage-4 passport owns only data/training scheduling.
        "training_profile_id": semantic_profile["profile_id"],
        "training_profile_fingerprint": semantic_profile["profile_fingerprint"],
        "stage4_training_profile_id": stage4_profile["profile_id"],
        "stage4_training_profile_fingerprint": stage4_profile["profile_fingerprint"],
        "selfplay_contract_id": semantic_profile["self_play"]["contract_id"],
        "selfplay_contract_fingerprint": semantic_profile["self_play"]["fingerprint"],
        "parent_or_source_run_identity": parent or run_id,
        "parent_checkpoint_label": parent,
        "run_id": run_id,
        "ablation_arm": arm,
        "completed_games": completed_games,
        "valid_replay_positions": valid_replay_positions,
        "optimizer_updates": optimizer_updates,
        "train_samples_consumed": train_samples_consumed,
        "model_initialization_seed": model_init_seed,
        "git_commit": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "git_worktree_clean": code.working_tree_clean,
        "device": str(device),
        "model_parameter_count": count_parameters(model),
        "model_hash": model_hash(model),
    }


def checkpoint_info(path: Path, *, label: str, device: torch.device) -> dict[str, object]:
    metadata = load_json(path.with_suffix(".metadata.json"))
    model = GoldenGraphNetV1().to(device)
    loaded = load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]}, device=device)
    if model_hash(model) != loaded["model_hash"]:
        raise ValueError(f"Checkpoint model identity changed after load: {path}")
    return {"label": label, "path": str(path), "metadata": metadata, "artifact_sha256": file_sha256(path), "model_hash": model_hash(model)}


def make_checkpoint_player(label: str, info: Mapping[str, object], model: torch.nn.Module, evaluator: GoldenNeuralEvaluator) -> SearchPlayer:
    metadata = info["metadata"]
    artifact = str(info["artifact_sha256"])
    evaluator.checkpoint_path = str(info["path"])
    evaluator.checkpoint_metadata = metadata
    identity = PlayerIdentity(
        logical_player_id=label,
        player_kind="checkpoint",
        source_identity=f"stage4-checkpoint:{label}:{metadata['model_hash']}",
        model_file_sha256=artifact,
        checkpoint_metadata_fingerprint=sha256_fingerprint(metadata),
        observation_contract_id=str(metadata["observation_schema_id"]),
        observation_fingerprint=str(metadata["observation_fingerprint"]),
        target_contract_id=str(metadata["target_contract_id"]),
        target_fingerprint=str(metadata["target_fingerprint"]),
        value_semantics=str(metadata["value_head_semantics"]),
        policy_semantics="root-visits-over-legal-actions:[25-points+PASS]",
    )
    identity.validate()
    return SearchPlayer(label, SequentialPUCT(), evaluator, identity=identity)


def _runner_factory(model: torch.nn.Module, *, run_id: str, label: str, artifact: str, seed: int, semantic_profile: Mapping[str, Any], code, device: torch.device):
    evaluator = GoldenNeuralEvaluator(model, device=device)

    def factory(game_id: str) -> GoldenSelfPlayRunner:
        return GoldenSelfPlayRunner(
            model,
            run_id=run_id,
            profile_fingerprint=str(semantic_profile["profile_fingerprint"]),
            model_checkpoint_label=label,
            checkpoint_artifact_hash=artifact,
            master_seed=seed,
            code_identity=code,
            device=device,
            evaluator=evaluator,
        )

    return factory, evaluator


def _process_worker_init(checkpoint_path: str, expected_model_hash: str, run_id: str, label: str, artifact: str, seed: int, profile_fingerprint: str, code_commit: str, code_tree: str, code_clean: bool, device_name: str) -> None:
    """Load one immutable checkpoint per worker; PUCT remains sequential per game."""
    global _PROCESS_MODEL, _PROCESS_CONFIG
    torch.set_num_threads(1)
    device = torch.device(device_name)
    model = GoldenGraphNetV1().to(device)
    load_checkpoint(checkpoint_path, model=model, expected={"model_hash": expected_model_hash}, device=device)
    if model_hash(model) != expected_model_hash:
        raise RuntimeError("Process self-play worker loaded the wrong model hash")
    _PROCESS_MODEL = model
    _PROCESS_CONFIG = {
        "run_id": run_id,
        "label": label,
        "artifact": artifact,
        "seed": seed,
        "profile_fingerprint": profile_fingerprint,
        "code": CodeIdentity(code_commit, code_tree, code_clean),
        "device": device,
    }


def _process_play_game(game_id: str) -> SelfPlayGameRecord:
    if _PROCESS_MODEL is None:
        raise RuntimeError("Self-play process worker was not initialized")
    evaluator = GoldenNeuralEvaluator(_PROCESS_MODEL, device=_PROCESS_CONFIG["device"])
    runner = GoldenSelfPlayRunner(
        _PROCESS_MODEL,
        run_id=str(_PROCESS_CONFIG["run_id"]),
        profile_fingerprint=str(_PROCESS_CONFIG["profile_fingerprint"]),
        model_checkpoint_label=str(_PROCESS_CONFIG["label"]),
        checkpoint_artifact_hash=str(_PROCESS_CONFIG["artifact"]),
        master_seed=int(_PROCESS_CONFIG["seed"]),
        code_identity=_PROCESS_CONFIG["code"],
        device=_PROCESS_CONFIG["device"],
        evaluator=evaluator,
    )
    return runner.play_game(game_id)


def run_games(model: torch.nn.Module, *, run_id: str, label: str, artifact: str, checkpoint_path: str | None, seed: int, semantic_profile: Mapping[str, Any], code, device: torch.device, game_ids: Sequence[str], workers: int) -> tuple[tuple[SelfPlayGameRecord, ...], int]:
    if workers > 1:
        if checkpoint_path is None:
            raise ValueError("Process self-play requires an immutable checkpoint path")
        ordered_ids = tuple(sorted(str(game_id) for game_id in game_ids))
        context_name = "spawn" if device.type == "cuda" else "fork"
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=get_context(context_name),
            initializer=_process_worker_init,
            initargs=(checkpoint_path, model_hash(model), run_id, label, artifact, seed, str(semantic_profile["profile_fingerprint"]), code.git_commit_sha, code.git_tree_sha, code.working_tree_clean, str(device)),
        ) as pool:
            records = tuple(pool.map(_process_play_game, ordered_ids))
        return records, sum(record.nn_evaluations for record in records)
    factory, evaluator = _runner_factory(model, run_id=run_id, label=label, artifact=artifact, seed=seed, semantic_profile=semantic_profile, code=code, device=device)
    before = evaluator.nn_evaluations
    records = run_selfplay_games(factory, game_ids, workers=workers)
    return records, evaluator.nn_evaluations - before


def run_comparison(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    candidate_label: str,
    reference_label: str,
    starts: Sequence[Mapping[str, object]],
    arena_seed: int,
    code,
    device: torch.device,
    canonical: bool,
    output_dir: Path,
) -> dict[str, object]:
    candidate_model = GoldenGraphNetV1().to(device)
    reference_model = GoldenGraphNetV1().to(device)
    load_checkpoint(Path(str(candidate["path"])), model=candidate_model, expected={"model_hash": candidate["model_hash"]}, device=device)
    load_checkpoint(Path(str(reference["path"])), model=reference_model, expected={"model_hash": reference["model_hash"]}, device=device)
    if model_hash(candidate_model) != candidate["model_hash"] or model_hash(reference_model) != reference["model_hash"]:
        raise ValueError(f"Arena checkpoint identity changed for {slug}")
    candidate_evaluator = GoldenNeuralEvaluator(candidate_model, device=device)
    reference_evaluator = GoldenNeuralEvaluator(reference_model, device=device)
    player_a = make_checkpoint_player(candidate_label, candidate, candidate_model, candidate_evaluator)
    player_b = make_checkpoint_player(reference_label, reference, reference_model, reference_evaluator)
    arena = SequentialGoldenArena(
        master_seed=arena_seed,
        run_id=f"{run_id}-{slug}",
        code_identity=code,
        require_canonical_code=canonical,
    )
    requested = {
        "candidate": {key: value for key, value in candidate.items() if key != "metadata"},
        "reference": {key: value for key, value in reference.items() if key != "metadata"},
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "starts_frozen_before_comparison": True,
        "pair_count_declared_before_first_game": len(starts),
        "arena_seed": arena_seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "requested-checkpoints.json", requested)
    for row in starts:
        start_id = str(row["start_id"])
        arena.play_pair(
            pair_id=f"{slug}--{start_id}",
            player_A=player_a,
            player_B=player_b,
            start_state=state_from_start_row(row),
            start_trace=tuple(int(action) for action in row["trace"]),
        )
    records = arena.records
    write_records_jsonl(output_dir / "games.jsonl", records)
    summary = summarize_pair_records(records, candidate_label=candidate_label, reference_label=reference_label, starts=starts)

    empty_row = {"start_id": "empty-board", "prefix_length": 0, "trace": [], "state": state_identity(initial_state())}
    control_arena = SequentialGoldenArena(
        master_seed=derive_seed(arena_seed, "empty-board-control"),
        run_id=f"{run_id}-{slug}-empty-board",
        code_identity=code,
        require_canonical_code=canonical,
    )
    control_arena.play_pair(
        pair_id=f"{slug}--empty-board",
        player_A=player_a,
        player_B=player_b,
        start_state=initial_state(),
        start_trace=(),
    )
    control_records = control_arena.records
    write_records_jsonl(output_dir / "empty-board-games.jsonl", control_records)
    control_summary = summarize_pair_records(control_records, candidate_label=candidate_label, reference_label=reference_label, starts=(empty_row,))
    summary["empty_board_control"] = control_summary
    summary["requested_checkpoints"] = requested
    summary["arena_settings"] = {
        "simulations": 64,
        "cpuct": 1.25,
        "fpu": 0.0,
        "noise": False,
        "temperature": 0.0,
        "fast": False,
        "resign": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def records_from_jsonl(path: Path) -> tuple[SelfPlayGameRecord, ...]:
    records: list[SelfPlayGameRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        positions = tuple(SelfPlayPosition(
            ply=int(item["ply"]),
            state=item["state"],
            side_to_move=str(item["side_to_move"]),
            root_visits=tuple(int(value) for value in item["root_visits"]),
            pi=tuple(float(value) for value in item["pi"]),
            selected_action=item["selected_action"],
            search_seed=int(item["search_seed"]),
            model_hash=str(item["model_hash"]),
        ) for item in row["positions"])
        records.append(SelfPlayGameRecord(
            run_id=str(row["run_id"]),
            game_id=str(row["game_id"]),
            profile_id=str(row["profile_id"]),
            profile_fingerprint=str(row["profile_fingerprint"]),
            selfplay_contract_id=str(row["selfplay_contract_id"]),
            selfplay_contract_fingerprint=str(row["selfplay_contract_fingerprint"]),
            model_checkpoint_label=str(row["model_checkpoint_label"]),
            model_hash=str(row["model_hash"]),
            checkpoint_artifact_hash=str(row["checkpoint_artifact_hash"]),
            git_commit=str(row["git_commit"]),
            git_tree=str(row["git_tree"]),
            git_worktree_clean=bool(row["git_worktree_clean"]),
            master_seed=int(row["master_seed"]),
            game_seed=int(row["game_seed"]),
            start_state=row["start_state"],
            positions=positions,
            final_action_trace=tuple(row["final_action_trace"]),
            formal_result=row["formal_result"],
            technical_termination=row["technical_termination"],
            error=row.get("error"),
            nn_evaluations=int(row.get("nn_evaluations", 0)),
        ))
    return tuple(records)


def benchmark_devices(starts: Sequence[Mapping[str, object]], requested: str) -> dict[str, object]:
    devices = [requested] if requested != "auto" else ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows: dict[str, object] = {}
    benchmark_states = [initial_state()] + [state_from_start_row(row) for row in starts[:15]]
    for name in devices:
        device = torch.device(name)
        seed_everything(STAGE4_MODEL_INIT_SEED)
        model = GoldenGraphNetV1().to(device)
        evaluator = GoldenNeuralEvaluator(model, device=device)
        for state in benchmark_states[:2]:
            evaluator.evaluate(state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        calls = 0
        for index in range(32):
            evaluator.evaluate(benchmark_states[index % len(benchmark_states)])
            calls += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        rows[name] = {"device": name, "batch_size": 1, "calls": calls, "wall_sec": elapsed, "calls_per_sec": calls / elapsed if elapsed else None}
    selected = min(rows, key=lambda name: float(rows[name]["wall_sec"]))
    return {"requested": requested, "candidates": rows, "selected": selected, "selection_rule": "lowest wall time on fixed batch-1 inference workload"}


def arm_report(
    *,
    arm: str,
    model: GoldenGraphNetV1,
    initial_model: GoldenGraphNetV1,
    train_samples: Sequence[GoldenTrainingSample],
    holdout_samples: Sequence[GoldenTrainingSample],
    schedule_report: Mapping[str, object],
    checkpoint: Mapping[str, object],
    fixed_observations: Sequence[GoldenTrainingSample],
) -> dict[str, object]:
    result = {
        "arm": arm,
        "unique_training_games": None,
        "unique_training_positions": len(train_samples),
        "train_sample_budget": schedule_report["exact_samples_consumed"],
        "reuse_ratio_sampled_over_positions": float(schedule_report["exact_samples_consumed"]) / len(train_samples),
        "optimizer_updates": schedule_report["updates"],
        "actual_batch_sizes": schedule_report["batch_sizes"],
        "train_metrics": evaluate_model_samples(model, train_samples),
        "holdout_metrics": evaluate_model_samples(model, holdout_samples),
        "initial_train_metrics": evaluate_model_samples(initial_model, train_samples),
        "initial_holdout_metrics": evaluate_model_samples(initial_model, holdout_samples),
        "parameter_delta_from_m0": parameter_diagnostics(initial_model.state_dict(), model),
        "diagnostic_policy_value": policy_value_entropy(model, fixed_observations),
        "checkpoint": dict(checkpoint),
        "training_schedule": schedule_report,
    }
    return result


def run_ablation(
    *,
    run_dir: Path,
    run_id: str,
    semantic_profile: Mapping[str, Any],
    stage4_profile: Mapping[str, Any],
    code,
    device: torch.device,
    m0_model: GoldenGraphNetV1,
    m0_info: Mapping[str, object],
    chunk1_records: Sequence[SelfPlayGameRecord],
    starts: Sequence[Mapping[str, object]],
    canonical: bool,
) -> dict[str, object]:
    game_map = {record.game_id: record for record in chunk1_records}
    ordered_games = tuple(sorted(game_map))
    if len(ordered_games) != 128:
        raise ValueError("Ablation fixed dataset requires exactly 128 games")
    train_game_ids = ordered_games[:96]
    holdout_game_ids = ordered_games[96:]
    samples_by_game = {game_id: build_replay_samples(game_map[game_id]) for game_id in ordered_games}
    train_samples = tuple(sample for game_id in train_game_ids for sample in samples_by_game[game_id])
    holdout_samples = tuple(sample for game_id in holdout_game_ids for sample in samples_by_game[game_id])
    arm_a_samples = tuple(sample for game_id in train_game_ids[:16] for sample in samples_by_game[game_id])
    fixed_observations = fixed_observation_samples(starts)
    ablation_dir = run_dir / "ablation"
    write_jsonl(ablation_dir / "fixed-m0-selfplay" / "games.jsonl", (record.to_dict() for record in chunk1_records))
    write_jsonl(ablation_dir / "fixed-m0-selfplay" / "training-games.jsonl", (game_map[game_id].to_dict() for game_id in train_game_ids))
    write_jsonl(ablation_dir / "fixed-m0-selfplay" / "holdout-games.jsonl", (game_map[game_id].to_dict() for game_id in holdout_game_ids))
    write_json(ablation_dir / "fixed-m0-selfplay" / "split.json", {
        "rule": "sort game_id; first 96 training, final 32 holdout",
        "training_game_ids": train_game_ids,
        "holdout_game_ids": holdout_game_ids,
        "training_positions": len(train_samples),
        "holdout_positions": len(holdout_samples),
    })

    arm_specs = {
        "A-small-data-high-reuse": (arm_a_samples, high_reuse_schedule(len(arm_a_samples), seed=derive_seed(STAGE4_MODEL_INIT_SEED, "ablation-order")), "first 16 training games"),
        "B-large-data-high-reuse": (train_samples, high_reuse_schedule(len(train_samples), seed=derive_seed(STAGE4_MODEL_INIT_SEED, "ablation-order")), "all 96 training games"),
        "C-large-data-low-reuse": (train_samples, low_reuse_schedule(len(train_samples), seed=derive_seed(STAGE4_MODEL_INIT_SEED, "ablation-order")), "all 96 training games, one epoch without replacement"),
    }
    reports: dict[str, object] = {}
    arm_infos: dict[str, dict[str, object]] = {}
    for arm, (samples, schedule, data_rule) in arm_specs.items():
        model = GoldenGraphNetV1().to(device)
        model.load_state_dict({name: value.detach().clone() for name, value in m0_model.state_dict().items()})
        initial_model = GoldenGraphNetV1().to(device)
        initial_model.load_state_dict({name: value.detach().clone() for name, value in m0_model.state_dict().items()})
        optimizer, schedule_report = train_batch_schedule(
            model, samples, schedule,
            learning_rate=stage4_profile["training"]["learning_rate"],
            weight_decay=stage4_profile["training"]["weight_decay"],
        )
        checkpoint_path = ablation_dir / arm / "checkpoint.pt"
        metadata = checkpoint_metadata(
            semantic_profile,
            run_id=run_id,
            label=arm,
            parent=str(m0_info["model_hash"]),
            code=code,
            model=model,
            device=device,
            completed_games=128,
            valid_replay_positions=len(train_samples),
            optimizer_updates=int(schedule_report["updates"]),
            train_samples_consumed=int(schedule_report["exact_samples_consumed"]),
            model_init_seed=STAGE4_MODEL_INIT_SEED,
            stage4_profile=stage4_profile,
            arm=arm,
        )
        metadata = save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, metadata=metadata)
        info = {"label": arm, "path": str(checkpoint_path), "metadata": metadata, "artifact_sha256": metadata["artifact_sha256"], "model_hash": metadata["model_hash"]}
        arm_infos[arm] = info
        report = arm_report(
            arm=arm,
            model=model,
            initial_model=initial_model,
            train_samples=samples,
            holdout_samples=holdout_samples,
            schedule_report={**schedule_report, "data_rule": data_rule},
            checkpoint=info,
            fixed_observations=fixed_observations,
        )
        report["unique_training_games"] = 16 if arm.startswith("A-") else 96
        write_json(ablation_dir / arm / "metrics.json", report)
        reports[arm] = report

    m0_for_arena = dict(m0_info)
    arena_results = {
        "A_vs_M0": run_comparison(
            run_dir=run_dir, run_id=run_id, slug="ablation-a-vs-m0", candidate=arm_infos["A-small-data-high-reuse"], reference=m0_for_arena,
            candidate_label="A", reference_label="M0'", starts=diagnostic_subset(starts), arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "ablation-a-m0"), code=code, device=device, canonical=canonical, output_dir=ablation_dir / "arena" / "a-vs-m0",
        ),
        "B_vs_M0": run_comparison(
            run_dir=run_dir, run_id=run_id, slug="ablation-b-vs-m0", candidate=arm_infos["B-large-data-high-reuse"], reference=m0_for_arena,
            candidate_label="B", reference_label="M0'", starts=diagnostic_subset(starts), arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "ablation-b-m0"), code=code, device=device, canonical=canonical, output_dir=ablation_dir / "arena" / "b-vs-m0",
        ),
        "C_vs_M0": run_comparison(
            run_dir=run_dir, run_id=run_id, slug="ablation-c-vs-m0", candidate=arm_infos["C-large-data-low-reuse"], reference=m0_for_arena,
            candidate_label="C", reference_label="M0'", starts=diagnostic_subset(starts), arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "ablation-c-m0"), code=code, device=device, canonical=canonical, output_dir=ablation_dir / "arena" / "c-vs-m0",
        ),
        "B_vs_C": run_comparison(
            run_dir=run_dir, run_id=run_id, slug="ablation-b-vs-c", candidate=arm_infos["B-large-data-high-reuse"], reference=arm_infos["C-large-data-low-reuse"],
            candidate_label="B", reference_label="C", starts=diagnostic_subset(starts), arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "ablation-b-c"), code=code, device=device, canonical=canonical, output_dir=ablation_dir / "arena" / "b-vs-c",
        ),
    }
    result = {
        "fixed_dataset": {
            "games": 128,
            "training_games": 96,
            "holdout_games": 32,
            "training_positions": len(train_samples),
            "holdout_positions": len(holdout_samples),
            "split_fingerprint": sha256_fingerprint({"training": train_game_ids, "holdout": holdout_game_ids}),
        },
        "arms": reports,
        "arena": arena_results,
        "same_m0_model_hash": m0_info["model_hash"],
        "same_ordering_seed": derive_seed(STAGE4_MODEL_INIT_SEED, "ablation-order"),
        "holdout_is_never_used_for_optimizer": True,
    }
    write_json(ablation_dir / "report.json", result)
    return result


def run_smoke(*, run_id: str, device_name: str) -> dict[str, object]:
    run_dir = ROOT / "runs" / "torus-golden-stage4" / run_id
    if run_dir.exists():
        raise ValueError(f"Smoke run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    semantic_profile = load_stage3_profile()
    stage4 = stage4_profile()
    code = capture_code_identity(ROOT)
    device = torch.device(device_name)
    seed_everything(STAGE4_MODEL_INIT_SEED)
    model = GoldenGraphNetV1().to(device)
    # Smoke artifacts are deliberately separate and never feed the canonical run.
    info_meta = checkpoint_metadata(semantic_profile, run_id=run_id, label="M0'", parent=None, code=code, model=model, device=device, completed_games=0, valid_replay_positions=0, optimizer_updates=0, train_samples_consumed=0, model_init_seed=STAGE4_MODEL_INIT_SEED, stage4_profile=stage4)
    m0_path = run_dir / "checkpoints" / "M0.pt"
    info_meta = save_checkpoint(m0_path, model=model, optimizer=None, metadata=info_meta)
    m0_info = {"label": "M0'", "path": str(m0_path), "metadata": info_meta, "artifact_sha256": info_meta["artifact_sha256"], "model_hash": info_meta["model_hash"]}
    records, nn_calls = run_games(model, run_id=run_id, label="M0'", artifact=str(info_meta["artifact_sha256"]), checkpoint_path=None, seed=STAGE4_SELFPLAY_MASTER_SEED, semantic_profile=semantic_profile, code=code, device=device, game_ids=tuple(f"smoke-game-{i:02d}" for i in range(4)), workers=1)
    samples = tuple(sample for record in records for sample in build_replay_samples(record))
    optimizer, schedule = train_batch_schedule(model, samples, low_reuse_schedule(len(samples), batch_size=64, seed=19))
    m1_meta = checkpoint_metadata(semantic_profile, run_id=run_id, label="M1'", parent=str(m0_info["model_hash"]), code=code, model=model, device=device, completed_games=4, valid_replay_positions=len(samples), optimizer_updates=int(schedule["updates"]), train_samples_consumed=int(schedule["exact_samples_consumed"]), model_init_seed=STAGE4_MODEL_INIT_SEED, stage4_profile=stage4)
    m1_path = run_dir / "checkpoints" / "M1.pt"
    m1_meta = save_checkpoint(m1_path, model=model, optimizer=optimizer, metadata=m1_meta)
    m1_info = {"label": "M1'", "path": str(m1_path), "metadata": m1_meta, "artifact_sha256": m1_meta["artifact_sha256"], "model_hash": m1_meta["model_hash"]}
    from gocube_golden.stage4 import generate_evaluation_v2
    starts = list(generate_evaluation_v2())[:1]
    comparison = run_comparison(run_dir=run_dir, run_id=run_id, slug="smoke-pair", candidate=m1_info, reference=m0_info, candidate_label="M1'", reference_label="M0'", starts=starts, arena_seed=19, code=code, device=device, canonical=False, output_dir=run_dir / "arena" / "smoke-pair")
    result = {"run_id": run_id, "smoke": True, "games": len(records), "positions": len(samples), "nn_evaluations": nn_calls, "low_reuse_training": True, "arena_pair": comparison}
    write_json(run_dir / "report.json", result)
    return result


def old_checkpoint_info(label: str, device: torch.device) -> dict[str, object]:
    path = OLD_STAGE3_RUN / "checkpoints" / f"{label}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return checkpoint_info(path, label=label, device=device)


def stage3_retrospective(*, run_dir: Path, run_id: str, semantic_profile: Mapping[str, Any], starts: Sequence[Mapping[str, object]], code, device: torch.device, canonical: bool) -> dict[str, object]:
    subset = diagnostic_subset(starts)
    comparisons = (("M1", "M0"), ("M2", "M1"), ("M3", "M2"), ("M4", "M3"), ("M4", "M1"), ("M4", "M0"))
    result: dict[str, object] = {}
    infos = {label: old_checkpoint_info(label, device) for label in OLD_STAGE3_LABELS}
    for candidate_label, reference_label in comparisons:
        slug = f"{candidate_label.lower()}-vs-{reference_label.lower()}"
        result[f"{candidate_label}-{reference_label}"] = run_comparison(
            run_dir=run_dir,
            run_id=run_id,
            slug=f"retrospective-{slug}",
            candidate=infos[candidate_label],
            reference=infos[reference_label],
            candidate_label=candidate_label,
            reference_label=reference_label,
            starts=subset,
            arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "stage3-retrospective", slug),
            code=code,
            device=device,
            canonical=canonical,
            output_dir=run_dir / "arena" / "retrospective" / slug,
        )
    return {"subset_size": len(subset), "subset_fingerprint": evaluation_start_fingerprint(subset), "comparisons": result}


def main_run(args: argparse.Namespace) -> dict[str, object]:
    semantic_profile = load_stage3_profile()
    stage4 = stage4_profile()
    code = capture_code_identity(ROOT)
    canonical = not args.smoke
    if canonical and not code.working_tree_clean:
        raise RuntimeError("Canonical Stage 4 requires a clean committed tree")
    run_dir = ROOT / "runs" / "torus-golden-stage4" / args.run_id
    if run_dir.exists():
        raise ValueError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    if args.workers != 16 and canonical:
        raise ValueError("Canonical Stage 4 requires exactly 16 self-play workers")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.workers > 1:
        torch.set_num_threads(1)

    # Freeze the evaluation corpus before any comparison is run.
    evaluation_manifest = freeze_evaluation_v2(run_dir, master_seed=EVALUATION_MASTER_SEED, code=code)
    starts = load_frozen_starts(run_dir)
    subset = diagnostic_subset(starts)
    device_report = benchmark_devices(starts, args.device)
    device = torch.device(str(device_report["selected"]))
    if device.type == "cuda":
        torch.set_num_threads(1)

    seed_everything(STAGE4_MODEL_INIT_SEED)
    m0_model = GoldenGraphNetV1().to(device)
    m0_meta = checkpoint_metadata(semantic_profile, run_id=args.run_id, label="M0'", parent=None, code=code, model=m0_model, device=device, completed_games=0, valid_replay_positions=0, optimizer_updates=0, train_samples_consumed=0, model_init_seed=STAGE4_MODEL_INIT_SEED, stage4_profile=stage4)
    m0_path = run_dir / "checkpoints" / "M0.pt"
    m0_meta = save_checkpoint(m0_path, model=m0_model, optimizer=None, metadata=m0_meta)
    m0_info = {"label": "M0'", "path": str(m0_path), "metadata": m0_meta, "artifact_sha256": m0_meta["artifact_sha256"], "model_hash": m0_meta["model_hash"]}

    # The equivalence gate uses the current committed implementation and fixed
    # game IDs. It must pass before the 512-game run.
    eq_ids = ("equivalence-game-00", "equivalence-game-01")
    serial, _ = run_games(m0_model, run_id=args.run_id, label="M0'", artifact=str(m0_meta["artifact_sha256"]), checkpoint_path=None, seed=STAGE4_SELFPLAY_MASTER_SEED, semantic_profile=semantic_profile, code=code, device=device, game_ids=eq_ids, workers=1)
    parallel, _ = run_games(m0_model, run_id=args.run_id, label="M0'", artifact=str(m0_meta["artifact_sha256"]), checkpoint_path=str(m0_path), seed=STAGE4_SELFPLAY_MASTER_SEED, semantic_profile=semantic_profile, code=code, device=device, game_ids=eq_ids, workers=args.workers)
    from gocube_golden.training import compare_selfplay_evidence
    compare_selfplay_evidence(serial, parallel)
    write_json(run_dir / "equivalence-gate.json", {"passed": True, "game_ids": eq_ids, "workers": args.workers, "trace_root_visits_pi_result_z_exact": True, "inference_batch_size": 1, "inference_coalescing": False})

    # Phase B is completed before the new M0' is trained beyond chunk 1.
    retrospective = stage3_retrospective(run_dir=run_dir, run_id=args.run_id, semantic_profile=semantic_profile, starts=starts, code=code, device=device, canonical=canonical)

    # Generate exactly one immutable 128-game M0' dataset. It is used both by
    # the predeclared A/B/C ablation and as canonical chunk 1.
    chunk1_ids = tuple(f"chunk-01-game-{index:03d}" for index in range(128))
    run_started = time.perf_counter()
    start_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    chunk1_records, chunk1_nn = run_games(m0_model, run_id=args.run_id, label="M0'", artifact=str(m0_meta["artifact_sha256"]), checkpoint_path=str(m0_path), seed=STAGE4_SELFPLAY_MASTER_SEED, semantic_profile=semantic_profile, code=code, device=device, game_ids=chunk1_ids, workers=args.workers)
    for record in chunk1_records:
        if record.technical_termination is not None or record.model_hash != m0_info["model_hash"]:
            raise RuntimeError("Canonical Stage 4 chunk 1 is invalid")
    chunk1_samples = tuple(sample for record in chunk1_records for sample in build_replay_samples(record))
    write_jsonl(run_dir / "selfplay" / "chunk-01-games.jsonl", (record.to_dict() for record in chunk1_records))
    write_jsonl(run_dir / "replay" / "chunk-01.jsonl", (sample.to_dict() for sample in chunk1_samples))
    ablation = run_ablation(run_dir=run_dir, run_id=args.run_id, semantic_profile=semantic_profile, stage4_profile=stage4, code=code, device=device, m0_model=m0_model, m0_info=m0_info, chunk1_records=chunk1_records, starts=starts, canonical=canonical)

    # Main Stage 4 lineage. Chunk 1 deliberately reuses the immutable fixed
    # dataset, while chunks 2-4 are generated only after their parent model.
    main_optimizer = torch.optim.Adam(m0_model.parameters(), lr=stage4["training"]["learning_rate"], weight_decay=stage4["training"]["weight_decay"])
    cumulative = list(chunk1_samples)
    checkpoints: dict[str, dict[str, object]] = {"M0'": m0_info}
    training_reports: list[dict[str, object]] = []
    replay_metrics: dict[str, object] = {}
    chunk_records: list[SelfPlayGameRecord] = list(chunk1_records)
    current_label = "M0'"
    current_info = m0_info
    update_count = 0
    sample_count = 0
    current_chunk_telemetry = runtime_telemetry(run_started, start_cpu, start_rss, completed_games=128, positions=len(chunk1_samples), nn_evaluations=chunk1_nn)

    def train_chunk(chunk_index: int, new_samples: Sequence[GoldenTrainingSample], source_records: Sequence[SelfPlayGameRecord], nn_evaluations: int) -> dict[str, object]:
        nonlocal update_count, sample_count, current_label, current_info, m0_model, current_chunk_telemetry, main_optimizer
        if not new_samples:
            raise RuntimeError(f"Stage 4 chunk {chunk_index} produced no replay positions")
        before = {name: value.detach().clone() for name, value in m0_model.state_dict().items()}
        if chunk_index > 1:
            cumulative.extend(new_samples)
        cumulative.sort(key=lambda sample: (sample.game_id, sample.ply))
        new_positions = len(new_samples)
        if new_positions > len(cumulative):
            raise RuntimeError("Stage 4 cumulative replay is smaller than the new-position budget")
        ordering_seed = derive_seed(STAGE4_MODEL_INIT_SEED, "main-replay-sampling", chunk_index)
        selected = random.Random(ordering_seed).sample(range(len(cumulative)), new_positions)
        batches = tuple(tuple(selected[offset:offset + 64]) for offset in range(0, len(selected), 64))
        main_samples = tuple(cumulative)
        main_optimizer, schedule = train_batch_schedule(
            m0_model,
            main_samples,
            batches,
            learning_rate=stage4["training"]["learning_rate"],
            weight_decay=stage4["training"]["weight_decay"],
            optimizer=main_optimizer,
            update_offset=update_count,
            sample_offset=sample_count,
        )
        update_count = int(schedule["updates"])
        sample_count = int(schedule["exact_samples_consumed"])
        label = f"M{chunk_index}'"
        metadata = checkpoint_metadata(semantic_profile, run_id=args.run_id, label=label, parent=str(current_info["model_hash"]), code=code, model=m0_model, device=device, completed_games=chunk_index * 128, valid_replay_positions=len(cumulative), optimizer_updates=update_count, train_samples_consumed=sample_count, model_init_seed=STAGE4_MODEL_INIT_SEED, stage4_profile=stage4)
        path = run_dir / "checkpoints" / f"M{chunk_index}.pt"
        metadata = save_checkpoint(path, model=m0_model, optimizer=main_optimizer, metadata=metadata)
        info = {"label": label, "path": str(path), "metadata": metadata, "artifact_sha256": metadata["artifact_sha256"], "model_hash": metadata["model_hash"]}
        if info["model_hash"] == current_info["model_hash"]:
            raise RuntimeError(f"Stage 4 checkpoint {label} is identical to its parent")
        checkpoints[label] = info
        replay_metrics[label] = {
            "metric_class": "TRAINING/REPLAY METRIC NOT INDEPENDENT STRENGTH EVIDENCE",
            "cumulative_positions": len(cumulative),
            **evaluate_model_samples(m0_model, cumulative),
        }
        current_chunk_telemetry = runtime_telemetry(run_started, start_cpu, start_rss, completed_games=chunk_index * 128, positions=sum(len(record.positions) for record in chunk_records), nn_evaluations=sum(record.nn_evaluations for record in chunk_records) + nn_evaluations, training_wall_time=None)
        report = {
            "chunk": chunk_index,
            "source_checkpoint": current_label,
            "source_model_hash": current_info["model_hash"],
            "games": 128,
            "valid_games": 128,
            "technical": 0,
            "new_positions": new_positions,
            "cumulative_replay_positions": len(cumulative),
            "train_sample_budget": new_positions,
            "exact_samples_consumed_this_phase": new_positions,
            "exact_samples_consumed_cumulative": sample_count,
            "sample_ratio_this_phase": new_positions / new_positions,
            "optimizer_updates_this_phase": len(batches),
            "optimizer_updates_cumulative": update_count,
            "actual_batch_sizes": schedule["batch_sizes"],
            "final_batch_size": schedule["final_batch_size"],
            "sampling_seed": ordering_seed,
            "sampling_without_replacement": True,
            "parameter_delta": parameter_diagnostics(before, m0_model),
            "replay_metric": replay_metrics[label],
            "checkpoint": info,
            "telemetry": current_chunk_telemetry,
        }
        write_json(run_dir / "training" / f"chunk-{chunk_index:02d}-metrics.json", report)
        training_reports.append(report)
        current_label, current_info = label, info
        return report

    train_chunk(1, chunk1_samples, chunk1_records, chunk1_nn)
    for chunk_index in range(2, 5):
        game_ids = tuple(f"chunk-{chunk_index:02d}-game-{index:03d}" for index in range(128))
        records, nn_calls = run_games(m0_model, run_id=args.run_id, label=current_label, artifact=str(current_info["artifact_sha256"]), checkpoint_path=str(current_info["path"]), seed=derive_seed(STAGE4_SELFPLAY_MASTER_SEED, "chunk", chunk_index), semantic_profile=semantic_profile, code=code, device=device, game_ids=game_ids, workers=args.workers)
        for record in records:
            if record.technical_termination is not None or record.model_hash != current_info["model_hash"]:
                raise RuntimeError(f"Canonical Stage 4 chunk {chunk_index} is invalid")
        new_samples = tuple(sample for record in records for sample in build_replay_samples(record))
        write_jsonl(run_dir / "selfplay" / f"chunk-{chunk_index:02d}-games.jsonl", (record.to_dict() for record in records))
        write_jsonl(run_dir / "replay" / f"chunk-{chunk_index:02d}.jsonl", (sample.to_dict() for sample in new_samples))
        chunk_records.extend(records)
        train_chunk(chunk_index, new_samples, records, nn_calls)

    if set(checkpoints) != {"M0'", "M1'", "M2'", "M3'", "M4'"}:
        raise RuntimeError("Stage 4 checkpoint lineage is incomplete")
    if len({info["model_hash"] for info in checkpoints.values()}) != 5:
        raise RuntimeError("Stage 4 checkpoint model hashes are not all distinct")

    all_audit = audit_selfplay_records(chunk_records, expected_model_hash_by_label={label: info["model_hash"] for label, info in checkpoints.items()})
    stage3_records = tuple(records_from_jsonl(path) for path in sorted((OLD_STAGE3_RUN / "selfplay").glob("chunk-*-games.jsonl")))
    historical_stage3 = audit_selfplay_records(tuple(record for group in stage3_records for record in group))
    fixed_observations = fixed_observation_samples(starts)
    diagnostics = {
        label: policy_value_entropy(GoldenGraphNetV1().to(device), fixed_observations)
        for label in ()
    }
    # Load each actual checkpoint for the fixed observation corpus.
    diagnostics = {}
    for label, info in checkpoints.items():
        model = GoldenGraphNetV1().to(device)
        load_checkpoint(Path(str(info["path"])), model=model, expected={"model_hash": info["model_hash"]}, device=device)
        diagnostics[label] = policy_value_entropy(model, fixed_observations)

    # Confirmatory and progression Arena evaluations use only the frozen V2 set.
    primary = {
        "M1'-M0'": run_comparison(run_dir=run_dir, run_id=args.run_id, slug="m1-prime-vs-m0-prime", candidate=checkpoints["M1'"], reference=checkpoints["M0'"], candidate_label="M1'", reference_label="M0'", starts=starts, arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "m1-prime-m0-prime"), code=code, device=device, canonical=canonical, output_dir=run_dir / "arena" / "m1-prime-vs-m0-prime"),
        "M4'-M0'": run_comparison(run_dir=run_dir, run_id=args.run_id, slug="m4-prime-vs-m0-prime", candidate=checkpoints["M4'"], reference=checkpoints["M0'"], candidate_label="M4'", reference_label="M0'", starts=starts, arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "m4-prime-m0-prime"), code=code, device=device, canonical=canonical, output_dir=run_dir / "arena" / "m4-prime-vs-m0-prime"),
        "M4'-M1'": run_comparison(run_dir=run_dir, run_id=args.run_id, slug="m4-prime-vs-m1-prime", candidate=checkpoints["M4'"], reference=checkpoints["M1'"], candidate_label="M4'", reference_label="M1'", starts=starts, arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "m4-prime-m1-prime"), code=code, device=device, canonical=canonical, output_dir=run_dir / "arena" / "m4-prime-vs-m1-prime"),
    }
    progression = {
        f"M{index}'-M{index - 1}'": run_comparison(run_dir=run_dir, run_id=args.run_id, slug=f"progression-m{index}-prime-vs-m{index - 1}-prime", candidate=checkpoints[f"M{index}'"], reference=checkpoints[f"M{index - 1}'"], candidate_label=f"M{index}'", reference_label=f"M{index - 1}'", starts=subset, arena_seed=derive_seed(STAGE4_ARENA_MASTER_SEED, "progression", index), code=code, device=device, canonical=canonical, output_dir=run_dir / "arena" / "progression" / f"m{index}-prime-vs-m{index - 1}-prime")
        for index in (2, 3, 4)
    }

    tech_checks = [summary.get("technical", 0) for summary in list(primary.values()) + list(progression.values())]
    pipeline_validity = all(value == 0 for value in tech_checks) and len(checkpoints) == 5 and len({info["model_hash"] for info in checkpoints.values()}) == 5 and all_audit["technical_games"] == 0
    hypothesis = build_hypothesis_matrix(retrospective, primary, ablation, diagnostics, all_audit, historical_stage3)
    final_report = {
        "stage": "STAGE 3->4 CONTROLLED LEARNING DIAGNOSIS + INDEPENDENT CONFIRMATION",
        "run_id": args.run_id,
        "run_kind": "canonical" if canonical else "smoke",
        "source_commit": code.git_commit_sha,
        "source_tree": code.git_tree_sha,
        "git_worktree_clean_at_run_start": code.working_tree_clean,
        "semantic_profile_id": semantic_profile["profile_id"],
        "semantic_profile_fingerprint": semantic_profile["profile_fingerprint"],
        "stage4_profile_id": stage4["profile_id"],
        "stage4_profile_fingerprint": stage4["profile_fingerprint"],
        "evaluation_contract_id": EVALUATION_CONTRACT_ID,
        "evaluation_manifest": evaluation_manifest,
        "evaluation_v2_full_pairs": EVALUATION_ACCEPTED_TOTAL,
        "evaluation_v2_prefix_lengths": EVALUATION_PREFIX_LENGTHS,
        "evaluation_v2_diagnostic_pairs": len(subset),
        "evaluation_v2_diagnostic_subset_fingerprint": evaluation_start_fingerprint(subset),
        "device_benchmark": device_report,
        "selected_device": str(device),
        "checkpoint_lineage": {label: {key: value for key, value in info.items() if key != "metadata"} for label, info in checkpoints.items()},
        "equivalence_gate": load_json(run_dir / "equivalence-gate.json"),
        "stage3_retrospective": retrospective,
        "ablation": ablation,
        "training": training_reports,
        "replay_metrics": replay_metrics,
        "selfplay_audit_stage4": all_audit,
        "selfplay_outcome_comparison_stage3": historical_stage3,
        "policy_diagnostics": diagnostics,
        "arena_primary": primary,
        "arena_progression": progression,
        "pipeline_validity": "PASS" if pipeline_validity else "FAIL",
        "hypothesis_matrix": hypothesis,
        "global_verdict": global_verdict(pipeline_validity, hypothesis),
        "limitations": [
            "Golden replay audit checks artifact consistency against the same Golden rules implementation; it is not an independent rules oracle.",
            "Evaluation V2 exact semantic dedupe is inferential; the formally correct 200-element full-history symmetry canonicalizer is diagnostic-only because prefix length 2 cannot supply eight distinct symmetry classes.",
            "A/B/C Arena is diagnostic on 16 start pairs; the three primary Stage 4 comparisons use the pre-frozen 64-pair corpus.",
        ],
        "artifact_root": str(run_dir),
        "ci": "not run until final push",
    }
    write_json(run_dir / "final-report.json", final_report)
    write_text_report(run_dir / "final-report.md", final_report)
    write_json(run_dir / "manifest.json", {**final_report, "canonical": canonical, "canonical_requirements": {"selfplay_games": 512, "technical_training_games": 0, "primary_pairs_each": 64, "empty_board_control": True}})
    return final_report


def _mean(summary: Mapping[str, object]) -> float | None:
    value = summary.get("mean_pair_score")
    return float(value) if value is not None else None


def build_hypothesis_matrix(retrospective: Mapping[str, object], primary: Mapping[str, Mapping[str, object]], ablation: Mapping[str, object], diagnostics: Mapping[str, object], audit: Mapping[str, object], stage3_audit: Mapping[str, object]) -> dict[str, object]:
    old = retrospective["comparisons"]
    m1_m0 = _mean(old["M1-M0"])
    adjacent = [_mean(old[key]) for key in ("M2-M1", "M3-M2", "M4-M3")]
    h4 = m1_m0 is not None and m1_m0 > 0.60 and all(value is not None and abs(value - 0.5) <= max(0.15, (m1_m0 - 0.5) / 2.0) for value in adjacent)
    ablation_arms = ablation["arms"]
    a = ablation_arms["A-small-data-high-reuse"]
    b = ablation_arms["B-large-data-high-reuse"]
    c = ablation_arms["C-large-data-low-reuse"]
    arena = ablation["arena"]
    b_vs_a_holdout = float(b["holdout_metrics"]["policy_ce"]) < float(a["holdout_metrics"]["policy_ce"]) - 0.02 or float(b["holdout_metrics"]["value_ce"]) < float(a["holdout_metrics"]["value_ce"]) - 0.02
    b_vs_a_arena = (_mean(arena["B_vs_M0"]) or 0.0) > (_mean(arena["A_vs_M0"]) or 0.0) + 0.05
    h6 = b_vs_a_holdout or b_vs_a_arena
    b_train_better = float(b["train_metrics"]["policy_ce"]) < float(c["train_metrics"]["policy_ce"]) - 0.01 or float(b["train_metrics"]["value_ce"]) < float(c["train_metrics"]["value_ce"]) - 0.01
    b_not_better_generalization = float(b["holdout_metrics"]["policy_ce"]) >= float(c["holdout_metrics"]["policy_ce"]) - 0.02 and float(b["holdout_metrics"]["value_ce"]) >= float(c["holdout_metrics"]["value_ce"]) - 0.02 and (_mean(arena["B_vs_C"]) or 0.5) >= 0.45
    h5 = b_train_better and b_not_better_generalization
    c_matches_b = float(c["holdout_metrics"]["policy_ce"]) <= float(b["holdout_metrics"]["policy_ce"]) + 0.02 and float(c["holdout_metrics"]["value_ce"]) <= float(b["holdout_metrics"]["value_ce"]) + 0.02 and (_mean(arena["B_vs_C"]) or 0.5) <= 0.55
    h7 = c_matches_b
    old_color_artifact = True
    v2_m4_m1 = primary["M4'-M1'"]
    color_gap = abs(float(v2_m4_m1["candidate_as_black_score"]) - float(v2_m4_m1["candidate_as_white_score"])) if v2_m4_m1["candidate_as_black_score"] is not None and v2_m4_m1["candidate_as_white_score"] is not None else 1.0
    v2_split = float(v2_m4_m1["split_pair_fraction"] or 0.0)
    h3 = "CONFIRMED" if old_color_artifact and (v2_split >= 0.5 or color_gap >= 0.25) else "INCONCLUSIVE"
    v2_m4_m0 = primary["M4'-M0'"]
    both_colors_positive = float(v2_m4_m0["candidate_as_black_score"] or 0.0) > 0.5 and float(v2_m4_m0["candidate_as_white_score"] or 0.0) > 0.5
    h8 = "REPLICATED" if (_mean(v2_m4_m0) or 0.0) >= 0.60 and both_colors_positive else ("NOT REPLICATED" if (_mean(v2_m4_m0) or 0.0) <= 0.50 else "INCONCLUSIVE")
    h9 = "DEMONSTRATED" if (_mean(v2_m4_m1) or 0.0) >= 0.60 and float(v2_m4_m1["candidate_as_black_score"] or 0.0) > 0.5 and float(v2_m4_m1["candidate_as_white_score"] or 0.0) > 0.5 else ("NOT DEMONSTRATED" if (_mean(v2_m4_m1) or 0.0) <= 0.50 else "INCONCLUSIVE")
    def row(verdict: str, evidence: object, limitation: str) -> dict[str, object]:
        return {"verdict": verdict, "evidence": evidence, "limitation": limitation}
    return {
        "H1": row("CONFIRMED", {"old_protocol": "nested deterministic prefix trajectory", "old_start_count": 8, "new_protocol": "64 independent legal random prefixes with exact semantic dedupe", "new_full_set_fingerprint": "see evaluation-v2/manifest.json"}, "The old dependence is a protocol fact; symmetry classes are diagnostic-only in V2 at prefix length 2."),
        "H2": row("CONFIRMED", {"old_zero_variance_interval": [0.5, 0.5], "new_primary_interval": "Hoeffding bounded-mean", "alpha": HOEFFDING_ALPHA}, "Hoeffding is intentionally conservative and does not model any residual dependence beyond the frozen start-pair unit."),
        "H3": row(h3, {"old_color_artifact": old_color_artifact, "v2_split_pair_fraction": v2_split, "v2_color_gap": color_gap, "v2_summary": v2_m4_m1}, "This diagnoses benchmark color masking, not intrinsic komi or first-player strength."),
        "H4": row("CONFIRMED" if h4 else "INCONCLUSIVE", {"M1-M0": m1_m0, "later_adjacent": adjacent}, "The Stage 3 subset is diagnostic rather than a powered confirmatory Arena."),
        "H5": row("CONFIRMED" if h5 else ("REFUTED" if b_train_better and not b_not_better_generalization else "INCONCLUSIVE"), {"B_train": b["train_metrics"], "C_train": c["train_metrics"], "B_holdout": b["holdout_metrics"], "C_holdout": c["holdout_metrics"], "B_vs_C_arena": arena["B_vs_C"]}, "Causal interpretation requires the same 96-game dataset, which this ablation preserves."),
        "H6": row("CONFIRMED" if h6 else "INCONCLUSIVE", {"A": {"holdout": a["holdout_metrics"], "arena": arena["A_vs_M0"]}, "B": {"holdout": b["holdout_metrics"], "arena": arena["B_vs_M0"]}}, "A/B isolates unique-data volume at the same 25,600-sample budget."),
        "H7": row("CONFIRMED" if h7 else "INCONCLUSIVE", {"B_vs_C": arena["B_vs_C"], "B_sample_budget": b["train_sample_budget"], "C_sample_budget": c["train_sample_budget"]}, "The practical recommendation is conservative when Arena differences are small."),
        "H8": row(h8, v2_m4_m0, "Independent-seed replication is judged on full 64-pair V2 plus color split."),
        "H9": row(h9, v2_m4_m1, "M4' vs M1' is the primary incremental-strength test; diagnostic progression is secondary."),
    }


def global_verdict(pipeline_validity: bool, matrix: Mapping[str, Mapping[str, object]]) -> str:
    if not pipeline_validity:
        return "RUN INVALID"
    if matrix["H8"]["verdict"] == "REPLICATED" and matrix["H9"]["verdict"] == "DEMONSTRATED":
        return "LEARNING SYSTEM CONFIRMED"
    if matrix["H8"]["verdict"] == "REPLICATED":
        return "LEARNING CONFIRMED, INCREMENTAL GROWTH STILL UNCLEAR"
    return "LEARNING NOT REPLICATED"


def write_text_report(path: Path, report: Mapping[str, object]) -> None:
    matrix = report["hypothesis_matrix"]
    lines = [
        "# Stage 3→4 Controlled Learning Diagnosis + Independent Confirmation",
        "",
        "| Hypothesis | Verdict |",
        "|---|---|",
    ]
    for key in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"):
        lines.append(f"| {key} | {matrix[key]['verdict']} |" )
    lines.extend([
        "",
        f"Global verdict: **{report['global_verdict']}**",
        "",
        f"Pipeline validity: **{report['pipeline_validity']}**",
        f"Source commit: `{report['source_commit']}`",
        f"Source tree: `{report['source_tree']}`",
        f"Selected device: `{report['selected_device']}`",
        "",
        "## Plain answers",
        "",
        f"1. Golden neural pipeline technically works: **{'YES' if report['pipeline_validity'] == 'PASS' else 'NO'}**",
        f"2. Model learns meaningful playing strength from random initialization: **{'YES' if matrix['H8']['verdict'] == 'REPLICATED' else 'NO' if matrix['H8']['verdict'] == 'NOT REPLICATED' else 'INCONCLUSIVE'}**",
        f"3. Stage 3 M1→M4 plateau was real: **{'YES' if matrix['H4']['verdict'] == 'CONFIRMED' else 'NO' if matrix['H4']['verdict'] == 'REFUTED' else 'INCONCLUSIVE'}**",
        f"4. High optimizer reuse on low fresh-data volume contributed to it: **{'YES' if matrix['H5']['verdict'] == 'CONFIRMED' else 'NO' if matrix['H5']['verdict'] == 'REFUTED' else 'INCONCLUSIVE'}**",
        f"5. New data-rich low-reuse profile should become baseline for next stages: **{'YES' if matrix['H6']['verdict'] == 'CONFIRMED' or matrix['H7']['verdict'] == 'CONFIRMED' else 'NO' if matrix['H6']['verdict'] == 'REFUTED' and matrix['H7']['verdict'] == 'REFUTED' else 'INCONCLUSIVE'}**",
        "",
        "## Artifact root",
        "",
        str(report["artifact_root"]),
        "",
        "Replay limitation: Golden replay audit checks artifact consistency against the same Golden rules implementation; it is not an independent rules oracle.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.smoke:
            device = "cpu" if args.device == "auto" else args.device
            result = run_smoke(run_id=args.run_id, device_name=device)
        else:
            result = main_run(args)
    except Exception as exc:
        print(f"RUN INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
