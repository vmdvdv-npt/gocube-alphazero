#!/usr/bin/env python3
"""Run the Golden Torus Stage-3 neural proof.

The default canonical path uses independent self-play games in parallel while
keeping one sequential 64-simulation Golden PUCT inside every game.  Inference
is deliberately batch-one and non-coalesced; the equivalence gate compares the
parallel execution with the serial reference before the first canonical chunk.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from gocube_golden.arena import MappedResult, SequentialGoldenArena, TerminationReason, write_records_jsonl
from gocube_golden.neural import (
    GoldenGraphNetV1,
    GoldenNeuralEvaluator,
    OBSERVATION_FINGERPRINT,
    count_parameters,
    model_hash,
    build_observation,
)
from gocube_golden.players import SearchPlayer
from gocube_golden.provenance import (
    PlayerIdentity,
    capture_code_identity,
    derive_seed,
    file_sha256,
    sha256_fingerprint,
)
from gocube_golden.rules import apply_action, legal_actions
from gocube_golden.search import SequentialPUCT
from gocube_golden.stage3_contract import (
    PROFILE_ID,
    TARGET_CONTRACT_ID,
    TARGET_FINGERPRINT,
    load_profile,
    profile_fingerprint,
)
from gocube_golden.state import PASS, initial_state
from gocube_golden.topology import TORUS_5X5
from gocube_golden.training import (
    DEFAULT_SELFPLAY_CONTRACT,
    GoldenSelfPlayRunner,
    GoldenTrainer,
    GoldenTrainingSample,
    SelfPlayGameRecord,
    build_replay_samples,
    compare_selfplay_evidence,
    load_checkpoint,
    run_selfplay_games,
    save_checkpoint,
    state_identity,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is not available")
    return device


def git_clean() -> bool:
    return capture_code_identity(ROOT).working_tree_clean


def checkpoint_metadata(
    profile: Mapping[str, Any],
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
) -> dict[str, object]:
    return {
        "checkpoint_schema_version": 1,
        "checkpoint_label": label,
        "architecture_id": model.architecture_id,
        "architecture_config": model.architecture_config,
        "rules_profile_id": profile["frozen_identities"]["rules_profile_id"],
        "rules_fingerprint": profile["frozen_identities"]["rules_fingerprint"],
        "topology_fingerprint": profile["frozen_identities"]["topology_fingerprint"],
        "board_size": [5, 5],
        "point_id_order_identity": "row-major-yx:point_id=y*width+x",
        "komi": 0.5,
        "observation_schema_id": profile["observation"]["schema_id"],
        "observation_schema_version": profile["observation"]["schema_version"],
        "observation_fingerprint": profile["observation"]["fingerprint"],
        "target_contract_id": profile["target"]["contract_id"],
        "target_contract_version": profile["target"]["contract_version"],
        "target_fingerprint": profile["target"]["fingerprint"],
        "value_head_semantics": "side-to-move:[WIN,DRAW,LOSS]",
        "network_heads_and_shapes": {"policy": [26], "value": [3]},
        "training_profile_id": profile["profile_id"],
        "training_profile_fingerprint": profile["profile_fingerprint"],
        "selfplay_contract_id": profile["self_play"]["contract_id"],
        "selfplay_contract_fingerprint": DEFAULT_SELFPLAY_CONTRACT.fingerprint,
        "parent_or_source_run_identity": parent or run_id,
        "parent_checkpoint_label": parent,
        "run_id": run_id,
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


def start_set() -> tuple[dict[str, object], ...]:
    starts: list[dict[str, object]] = []
    for length in range(9):
        state = initial_state()
        trace: list[int | str] = []
        for _ in range(length):
            choices = tuple(action for action in legal_actions(state) if action != PASS)
            if not choices:
                raise RuntimeError("Could not construct legal Arena prefix")
            action = min(choices)
            trace.append(action)
            state = apply_action(state, action).after
        starts.append({"prefix_length": length, "trace": trace, "state": state_identity(state)})
    return tuple(starts)


def state_from_start(row: Mapping[str, object]):
    state = initial_state()
    for action in row["trace"]:  # type: ignore[index]
        state = apply_action(state, action).after
    if state_identity(state) != row["state"]:
        raise RuntimeError("Persisted Arena start set failed state reconstruction")
    return state


def start_set_fingerprint(starts: Sequence[Mapping[str, object]]) -> str:
    return sha256_fingerprint(starts)


def _runner_factory(
    model,
    *,
    run_id: str,
    profile_fp: str,
    label: str,
    artifact_hash: str,
    seed: int,
    code,
    device,
):
    shared_evaluator = GoldenNeuralEvaluator(model, device=device)

    def factory(_game_id: str) -> GoldenSelfPlayRunner:
        return GoldenSelfPlayRunner(
            model,
            run_id=run_id,
            profile_fingerprint=profile_fp,
            model_checkpoint_label=label,
            checkpoint_artifact_hash=artifact_hash,
            master_seed=seed,
            code_identity=code,
            device=device,
            evaluator=shared_evaluator,
        )
    return factory


def validate_chunk(
    records: Sequence[SelfPlayGameRecord],
    *,
    expected_games: int,
    expected_model_hash: str,
    require_clean: bool = True,
) -> tuple[GoldenTrainingSample, ...]:
    if len(records) != expected_games:
        raise RuntimeError(f"Canonical chunk requires {expected_games} games, got {len(records)}")
    if len({record.game_id for record in records}) != expected_games:
        raise RuntimeError("Canonical chunk contains duplicate or missing game ids")
    samples: list[GoldenTrainingSample] = []
    for record in sorted(records, key=lambda item: item.game_id):
        record.validate(require_clean=require_clean)
        if record.technical_termination is not None:
            raise RuntimeError(f"Canonical chunk has technical self-play game {record.game_id}: {record.technical_termination}")
        if record.model_hash != expected_model_hash:
            raise RuntimeError(f"Canonical chunk model hash drift in {record.game_id}")
        samples.extend(build_replay_samples(record))
    samples.sort(key=lambda sample: (sample.game_id, sample.ply))
    return tuple(samples)


def parameter_delta(before: Mapping[str, torch.Tensor], model: torch.nn.Module) -> float:
    total = 0.0
    for name, after in model.state_dict().items():
        total += float(torch.sum((after.detach().cpu() - before[name].detach().cpu()) ** 2))
    return math.sqrt(total)


def _make_checkpoint_player(label: str, path: Path, model, metadata: Mapping[str, object], artifact_hash: str, evaluator):
    evaluator.checkpoint_path = str(path)
    evaluator.checkpoint_metadata = metadata
    identity = PlayerIdentity(
        logical_player_id=label,
        player_kind="checkpoint",
        source_identity=f"stage3-checkpoint:{label}:{metadata['model_hash']}",
        model_file_sha256=artifact_hash,
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


def evaluate_pair_set(
    *,
    run_dir: Path,
    run_id: str,
    profile: Mapping[str, Any],
    m4_path: Path,
    m1_path: Path,
    m4_metadata: Mapping[str, object],
    m1_metadata: Mapping[str, object],
    m4_artifact: str,
    m1_artifact: str,
    starts: Sequence[Mapping[str, object]],
    arena_seed: int,
    device: torch.device,
    canonical: bool,
    opponent_label: str,
    output_name: str,
) -> dict[str, object]:
    model_a = GoldenGraphNetV1()
    model_b = GoldenGraphNetV1()
    load_checkpoint(m4_path, model=model_a, expected={"model_hash": m4_metadata["model_hash"]}, device=device)
    load_checkpoint(m1_path, model=model_b, expected={"model_hash": m1_metadata["model_hash"]}, device=device)
    loaded_a_hash = model_hash(model_a)
    loaded_b_hash = model_hash(model_b)
    if loaded_a_hash != m4_metadata["model_hash"] or loaded_b_hash != m1_metadata["model_hash"]:
        raise RuntimeError("Arena checkpoint model identity changed after load")
    if loaded_a_hash == loaded_b_hash:
        raise RuntimeError("Arena refuses self-comparison of identical model hashes")
    evaluator_a = GoldenNeuralEvaluator(model_a, device=device)
    evaluator_b = GoldenNeuralEvaluator(model_b, device=device)
    player_a = _make_checkpoint_player("M4", m4_path, model_a, m4_metadata, m4_artifact, evaluator_a)
    player_b = _make_checkpoint_player(opponent_label, m1_path, model_b, m1_metadata, m1_artifact, evaluator_b)
    arena = SequentialGoldenArena(master_seed=arena_seed, run_id=f"{run_id}-m4-v-{opponent_label.lower()}", require_canonical_code=canonical)
    requested = {
        "A": {"path": str(m4_path), "artifact_sha256": m4_artifact, "model_hash": loaded_a_hash},
        "B": {"path": str(m1_path), "artifact_sha256": m1_artifact, "model_hash": loaded_b_hash},
    }
    _write_json(run_dir / "arena" / output_name / "requested-checkpoints.json", requested)
    # Eight pairs are fixed before the first game; start index 0 is the empty board.
    for index, row in enumerate(starts[:8]):
        arena.play_pair(
            pair_id=f"m4-v-m1-pair-{index:02d}",
            player_A=player_a,
            player_B=player_b,
            start_state=state_from_start(row),
            start_trace=tuple(row["trace"]),  # type: ignore[arg-type]
        )
    records = arena.records
    write_records_jsonl(run_dir / "arena" / output_name / "games.jsonl", records)
    return arena_report(records, "M4", opponent_label, starts[:8], requested)


def arena_report(records, left_label: str, right_label: str, starts, requested) -> dict[str, object]:
    if any(record.termination_reason != TerminationReason.DOUBLE_PASS for record in records):
        raise RuntimeError("Golden Arena evaluation contains technical failures")
    pair_scores: list[float] = []
    left_wins = left_losses = draws = 0
    left_black = left_white = 0
    for record in records:
        if record.mapped_result == MappedResult.A_WIN:
            score = 1.0
            left_wins += 1
        elif record.mapped_result == MappedResult.B_WIN:
            score = 0.0
            left_losses += 1
        else:
            score = 0.5
            draws += 1
        if record.black_player == "A":
            left_black += int(record.mapped_result == MappedResult.A_WIN)
        else:
            left_white += int(record.mapped_result == MappedResult.A_WIN)
    for index in range(0, len(records), 2):
        pair_scores.append((
            (1.0 if records[index].mapped_result == MappedResult.A_WIN else 0.5 if records[index].mapped_result == MappedResult.DRAW else 0.0)
            + (1.0 if records[index + 1].mapped_result == MappedResult.A_WIN else 0.5 if records[index + 1].mapped_result == MappedResult.DRAW else 0.0)
        ) / 2.0)
    mean = statistics.fmean(pair_scores) if pair_scores else float("nan")
    deviation = statistics.stdev(pair_scores) if len(pair_scores) > 1 else 0.0
    half = 1.96 * deviation / math.sqrt(len(pair_scores)) if pair_scores else float("nan")
    return {
        "left": left_label,
        "right": right_label,
        "pairs": len(pair_scores),
        "games": len(records),
        "left_wins": left_wins,
        "left_losses": left_losses,
        "draws": draws,
        "technical": 0,
        "left_as_black_wins": left_black,
        "left_as_white_wins": left_white,
        "empty_board_games": 2,
        "prefix_start_games": max(0, len(records) - 2),
        "mean_pair_score": mean,
        "95_percent_interval": [max(0.0, mean - half), min(1.0, mean + half)],
        "requested_checkpoints": requested,
    }


def functional_preflight(
    model,
    device: torch.device,
    profile: Mapping[str, Any],
    run_id: str,
    code,
    artifact_hash: str,
    checkpoint_path: Path,
    checkpoint_metadata: Mapping[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    evaluator = GoldenNeuralEvaluator(model, device=device)
    inference_started = time.perf_counter()
    evaluation = evaluator.evaluate(initial_state())
    inference_latency = time.perf_counter() - inference_started
    search_started = time.perf_counter()
    search_result = SequentialPUCT().search(initial_state(), evaluator, seed=17)
    search_latency = time.perf_counter() - search_started
    runner = GoldenSelfPlayRunner(
        model,
        run_id=f"{run_id}-preflight",
        profile_fingerprint=profile["profile_fingerprint"],
        model_checkpoint_label="M0",
        checkpoint_artifact_hash=artifact_hash,
        master_seed=derive_seed(profile["seeds"]["selfplay_master_seed"], "preflight"),
        code_identity=code,
        device=device,
    )
    game_started = time.perf_counter()
    game = runner.play_game("preflight-game-00")
    game_time = time.perf_counter() - game_started
    if game.technical_termination is not None:
        raise RuntimeError("Preflight self-play game was technical: " + game.technical_termination)
    samples = build_replay_samples(game)
    arena_evaluator = GoldenNeuralEvaluator(model, device=device)
    arena_player = _make_checkpoint_player(
        "M0", checkpoint_path, model, checkpoint_metadata, artifact_hash, arena_evaluator
    )
    preflight_arena = SequentialGoldenArena(
        master_seed=derive_seed(profile["seeds"]["arena_master_seed"], "preflight"),
        run_id=f"{run_id}-preflight-arena",
        require_canonical_code=code.working_tree_clean,
    )
    preflight_arena.play_pair(
        pair_id="preflight-pair",
        player_A=arena_player,
        player_B=arena_player,
    )
    trainer = GoldenTrainer(model)
    train_started = time.perf_counter()
    metrics = trainer.train(samples, updates=1, batch_size=min(4, len(samples)), seed=19)
    train_time = time.perf_counter() - train_started
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_count": os.cpu_count(),
        "parameter_count": count_parameters(model),
        "single_state_inference_latency_sec": inference_latency,
        "search_move_latency_sec": search_latency,
        "one_selfplay_game_wall_sec": game_time,
        "one_training_batch_wall_sec": train_time,
        "one_training_total_loss": metrics[-1].total_loss,
        "one_arena_pair_games": len(preflight_arena.records),
        "one_arena_pair_technical": preflight_arena.summary().technical_failures,
        "nn_evaluations": evaluator.nn_evaluations + runner.evaluator.nn_evaluations,
        "search_evaluator_calls": search_result.evaluator_calls,
        "elapsed_sec": time.perf_counter() - started,
        "observation_fingerprint": OBSERVATION_FINGERPRINT,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    profile = load_profile()
    if args.run_id is None:
        raise ValueError("--run-id is required")
    device = choose_device(args.device)
    canonical = not args.smoke
    run_dir = ROOT / "runs" / "torus-golden-stage3" / args.run_id
    if run_dir.exists() and not args.resume:
        raise ValueError(f"Run directory already exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    code = capture_code_identity(ROOT)
    if canonical and not code.working_tree_clean:
        raise RuntimeError("Canonical Stage-3 M0 requires a clean committed tree")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if canonical and args.workers != 16:
        raise ValueError("Canonical Stage-3 requires selfplay_workers=16")
    # Avoid nested BLAS scheduling when independent games are threaded.
    if args.workers > 1 and device.type == "cpu":
        torch.set_num_threads(1)

    if args.resume:
        raise NotImplementedError("Resume is intentionally fail-closed until a complete chunk boundary is present")

    seed_everything(profile["seeds"]["model_init_seed"])
    model = GoldenGraphNetV1()
    profile_fp = profile["profile_fingerprint"]
    m0_path = run_dir / "checkpoints" / "M0.pt"
    m0_metadata = checkpoint_metadata(
        profile, run_id=args.run_id, label="M0", parent=None, code=code, model=model,
        device=device, completed_games=0, valid_replay_positions=0, optimizer_updates=0,
        train_samples_consumed=0, model_init_seed=profile["seeds"]["model_init_seed"],
    )
    m0_metadata = save_checkpoint(m0_path, model=model, optimizer=None, metadata=m0_metadata)
    m0_artifact = str(m0_metadata["artifact_sha256"])
    preflight = functional_preflight(
        model,
        device,
        profile,
        args.run_id,
        code,
        m0_artifact,
        m0_path,
        m0_metadata,
    )
    _write_json(run_dir / "preflight.json", preflight)
    # The functional preflight deliberately performs one optimizer update.  It
    # is never allowed to mutate canonical M0, so restore the exact artifact
    # before the equivalence gate and chunk 1.
    load_checkpoint(
        m0_path,
        model=model,
        expected={"model_hash": m0_metadata["model_hash"]},
        device=device,
    )
    if model_hash(model) != m0_metadata["model_hash"]:
        raise RuntimeError("M0 changed during preflight")
    starts = start_set()
    starts_payload = {"starts": starts, "fingerprint": start_set_fingerprint(starts), "generated_before_evaluation": True}
    _write_json(run_dir / "arena" / "start_set.json", starts_payload)

    equivalence_count = 1 if args.smoke else 2
    eq_ids = tuple(f"equivalence-game-{index:02d}" for index in range(equivalence_count))
    serial = run_selfplay_games(
        _runner_factory(model, run_id=args.run_id, profile_fp=profile_fp, label="M0", artifact_hash=m0_artifact, seed=profile["seeds"]["selfplay_master_seed"], code=code, device=device),
        eq_ids, workers=1,
    )
    parallel = run_selfplay_games(
        _runner_factory(model, run_id=args.run_id, profile_fp=profile_fp, label="M0", artifact_hash=m0_artifact, seed=profile["seeds"]["selfplay_master_seed"], code=code, device=device),
        eq_ids, workers=args.workers if not args.smoke else 1,
    )
    compare_selfplay_evidence(serial, parallel)
    _write_json(run_dir / "equivalence-gate.json", {"passed": True, "games": equivalence_count, "workers": args.workers, "inference_batch_size": 1, "inference_coalescing": False})

    games_per_chunk = 2 if args.smoke else profile["training"]["games_per_chunk"]
    chunks = 1 if args.smoke else profile["training"]["chunks"]
    updates_per_chunk = 2 if args.smoke else profile["training"]["optimizer_updates_per_chunk"]
    batch_size = 4 if args.smoke else profile["training"]["batch_size"]
    workers = 1 if args.smoke else args.workers
    trainer = GoldenTrainer(model, learning_rate=profile["training"]["learning_rate"], weight_decay=profile["training"]["weight_decay"])
    cumulative: list[GoldenTrainingSample] = []
    chunk_reports: list[dict[str, object]] = []
    parent_label = "M0"
    parent_artifact = m0_artifact
    parent_hash = model_hash(model)
    total_games = 0
    for chunk_index in range(1, chunks + 1):
        game_ids = tuple(f"chunk-{chunk_index:02d}-game-{index:02d}" for index in range(games_per_chunk))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        records = run_selfplay_games(
            _runner_factory(model, run_id=args.run_id, profile_fp=profile_fp, label=parent_label, artifact_hash=parent_artifact, seed=derive_seed(profile["seeds"]["selfplay_master_seed"], chunk_index), code=code, device=device),
            game_ids, workers=workers,
        )
        chunk_samples = validate_chunk(
            records,
            expected_games=games_per_chunk,
            expected_model_hash=parent_hash,
            require_clean=canonical,
        )
        total_games += len(records)
        cumulative.extend(chunk_samples)
        cumulative.sort(key=lambda sample: (sample.game_id, sample.ply))
        selfplay_path = run_dir / "selfplay" / f"chunk-{chunk_index:02d}-games.jsonl"
        write_jsonl(selfplay_path, (record.to_dict() for record in records))
        replay_path = run_dir / "replay" / f"chunk-{chunk_index:02d}.jsonl"
        write_jsonl(replay_path, (sample.to_dict() for sample in chunk_samples))
        metrics = trainer.train(cumulative, updates=updates_per_chunk, batch_size=batch_size, seed=derive_seed(profile["seeds"]["model_init_seed"], "train", chunk_index))
        label = f"M{chunk_index}"
        metadata = checkpoint_metadata(
            profile, run_id=args.run_id, label=label, parent=parent_hash, code=code, model=model,
            device=device, completed_games=total_games, valid_replay_positions=len(cumulative),
            optimizer_updates=trainer.update_count, train_samples_consumed=trainer.samples_consumed,
            model_init_seed=profile["seeds"]["model_init_seed"],
        )
        checkpoint_path = run_dir / "checkpoints" / f"{label}.pt"
        metadata = save_checkpoint(checkpoint_path, model=model, optimizer=trainer.optimizer, metadata=metadata)
        artifact = str(metadata["artifact_sha256"])
        new_hash = str(metadata["model_hash"])
        if new_hash == parent_hash:
            raise RuntimeError(f"Checkpoint lineage did not change at {label}")
        last = metrics[-1]
        report = {
            "chunk": chunk_index,
            "source_checkpoint": parent_label,
            "source_model_hash": parent_hash,
            "games": len(records),
            "valid_games": len(records),
            "technical": 0,
            "replay_positions": len(chunk_samples),
            "cumulative_replay_positions": len(cumulative),
            "optimizer_updates": trainer.update_count,
            "train_samples_consumed": trainer.samples_consumed,
            "policy_loss": last.policy_loss,
            "value_loss": last.value_loss,
            "total_loss": last.total_loss,
            "parameter_delta_from_parent": parameter_delta(before, model),
            "checkpoint": label,
            "model_hash": new_hash,
            "artifact_sha256": artifact,
        }
        _write_json(run_dir / "training" / f"chunk-{chunk_index:02d}-metrics.json", report)
        chunk_reports.append(report)
        parent_label, parent_artifact, parent_hash = label, artifact, new_hash

    # M0 is never overwritten; all five checkpoint identities are now present.
    mpaths = {label: run_dir / "checkpoints" / f"{label}.pt" for label in ("M0", "M1", "M2", "M3", "M4") if (run_dir / "checkpoints" / f"{label}.pt").is_file()}
    if not args.smoke and set(mpaths) != {"M0", "M1", "M2", "M3", "M4"}:
        raise RuntimeError("Canonical run did not produce M0 through M4")
    report: dict[str, object] = {
        "stage": "STAGE 3 GOLDEN TORUS NEURAL PROOF",
        "run_id": args.run_id,
        "run_kind": "smoke" if args.smoke else "canonical",
        "profile_id": PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "source_commit": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "device": str(device),
        "torch": torch.__version__,
        "architecture": GoldenGraphNetV1.architecture_id,
        "parameter_count": count_parameters(model),
        "selfplay_workers": workers,
        "inference_mode": "torch.inference_mode",
        "inference_batch_size": 1,
        "inference_coalescing": False,
        "self_play_contract": profile["self_play"],
        "arena_contract": profile["arena"],
        "preflight": preflight,
        "equivalence_gate": {"passed": True, "games": equivalence_count},
        "chunks": chunk_reports,
        "start_set_fingerprint": starts_payload["fingerprint"],
    }
    if args.smoke:
        report["verdict"] = "SMOKE ONLY — NO LEARNING VERDICT"
        _write_json(run_dir / "report.json", report)
        _write_json(run_dir / "manifest.json", {**report, "canonical": False})
        return report

    m4_meta = json.loads((run_dir / "checkpoints" / "M4.metadata.json").read_text(encoding="utf-8"))
    m1_meta = json.loads((run_dir / "checkpoints" / "M1.metadata.json").read_text(encoding="utf-8"))
    arena_m4_m1 = evaluate_pair_set(
        run_dir=run_dir, run_id=args.run_id, profile=profile,
        m4_path=mpaths["M4"], m1_path=mpaths["M1"], m4_metadata=m4_meta, m1_metadata=m1_meta,
        m4_artifact=str(m4_meta["artifact_sha256"]), m1_artifact=str(m1_meta["artifact_sha256"]),
        starts=starts, arena_seed=profile["seeds"]["arena_master_seed"], device=device, canonical=True,
        opponent_label="M1", output_name="m4-vs-m1",
    )
    # The diagnostic M4-vs-M0 uses the identical pre-generated start set and protocol.
    m0_meta = json.loads((run_dir / "checkpoints" / "M0.metadata.json").read_text(encoding="utf-8"))
    arena_m4_m0 = evaluate_pair_set(
        run_dir=run_dir, run_id=args.run_id, profile=profile,
        m4_path=mpaths["M4"], m1_path=mpaths["M0"], m4_metadata=m4_meta, m1_metadata=m0_meta,
        m4_artifact=str(m4_meta["artifact_sha256"]), m1_artifact=str(m0_meta["artifact_sha256"]),
        starts=starts, arena_seed=derive_seed(profile["seeds"]["arena_master_seed"], "m4-v-m0"), device=device, canonical=True,
        opponent_label="M0", output_name="m4-vs-m0",
    )
    report["arena"] = {"m4_vs_m1": arena_m4_m1, "m4_vs_m0": arena_m4_m0}
    report["verdict"] = (
        "LEARNING PROOF POSITIVE"
        if arena_m4_m1["mean_pair_score"] > 0.5 and arena_m4_m0["mean_pair_score"] > 0.5
        else "LEARNING NOT DEMONSTRATED"
    )
    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "manifest.json", {**report, "canonical": True, "canonical_sample_sizes_declared_before_arena": {"m4_vs_m1_pairs": 8, "m4_vs_m0_pairs": 8}})
    lines = [
        "STAGE 3 GOLDEN TORUS NEURAL PROOF",
        "",
        f"Run: {args.run_id}",
        f"Profile: {PROFILE_ID}",
        f"Profile fingerprint: {profile_fp}",
        f"Source commit: {code.git_commit_sha}",
        f"Device: {device}",
        f"Workers: {workers}; inference batch: 1; coalescing: false",
        "",
        f"M4 vs M1 mean pair score: {arena_m4_m1['mean_pair_score']:.6f}",
        f"M4 vs M0 mean pair score: {arena_m4_m0['mean_pair_score']:.6f}",
        f"Verdict: {report['verdict']}",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", default="proof-standard")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.profile != "proof-standard":
        raise SystemExit("Only immutable profile proof-standard is available")
    if args.resume is not None:
        args.run_id = args.resume.name
        args.resume = True
    try:
        result = run(args)
    except Exception as exc:
        print(f"RUN INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
