#!/usr/bin/env python3
"""Reproducible two-seed Golden Torus 5x5 auxiliary-head experiment.

The default invocation is the complete experiment described in
``docs/TORUS5_AUX_HEADS_NIGHT_AB_20260913.md``.  ``--smoke`` exercises the
same contracts with tiny budgets and never writes a scientific PASS report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
from multiprocessing import get_context
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.arena import write_records_jsonl
from gocube_golden.arena_process import CheckpointPlayerSpec, PairTask, ProcessParallelGoldenArena
from gocube_golden.neural import (
    AUXILIARY_VARIANTS,
    AuxiliaryGoldenGraphNet,
    GoldenGraphNetV1,
    GoldenNeuralEvaluator,
    instantiate_model_from_metadata,
    model_hash,
)
from gocube_golden.provenance import capture_code_identity, derive_seed, file_sha256, sha256_fingerprint
from gocube_golden.result import Winner
from gocube_golden.stage3_contract import load_profile as load_stage3_profile
from gocube_golden.stage4 import (
    EVALUATION_MASTER_SEED,
    audit_selfplay_records,
    diagnostic_subset,
    evaluation_start_fingerprint,
    first_move_statistics,
    freeze_evaluation_v2,
    load_frozen_starts,
    low_reuse_schedule,
    state_from_start_row,
    summarize_pair_records,
)
from gocube_golden.training import (
    GoldenSelfPlayRunner,
    GoldenTrainingSample,
    SelfPlayGameRecord,
    SelfPlayPosition,
    build_replay_samples,
    load_checkpoint,
    run_selfplay_games,
    save_checkpoint,
    train_variant_batch_schedule,
    write_jsonl,
)
from gocube_golden.stage4 import train_batch_schedule


RUN_DATE = "20260913"
DEFAULT_RUN_ID = f"torus5-aux-heads-night-ab-{RUN_DATE}"
DEFAULT_RUN_ROOT = ROOT / "runs" / "torus5-aux-heads"
VARIANTS = AUXILIARY_VARIANTS
SEED_CONFIG = {
    "A": {
        "model": 2026091401,
        "selfplay": 2026091402,
        "arena": 2026091403,
        "control": 2026091404,
    },
    "B": {
        "model": 2026091501,
        "selfplay": 2026091502,
        "arena": 2026091503,
        "control": 2026091504,
    },
}
TRAINING_BATCH_SIZE = 64
SELFPLAY_GAMES = 128
PRIMARY_PAIRS = 64
DIAGNOSTIC_PAIRS = 16
CONTROL_GAMES = 256
WORKERS = 16
SCORE_NORMALIZATION = 25.5

_SELFPLAY_MODEL: torch.nn.Module | None = None
_SELFPLAY_CONFIG: dict[str, object] = {}


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return str(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def branch_name() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)


def make_model(variant: str, seed: int) -> torch.nn.Module:
    """Initialize all arms from one deterministic shared-parameter seed."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown experiment variant: {variant}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if variant == "wdl":
            model: torch.nn.Module = GoldenGraphNetV1()
        else:
            model = AuxiliaryGoldenGraphNet(variant=variant)
    model.eval()
    return model


def shared_parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("ownership_head.") or name.startswith("score_head."):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def stage4_profile() -> dict[str, Any]:
    path = ROOT / "configs" / "gocube" / "torus_golden_training_v2_data_rich.json"
    profile = read_json(path)
    profile["profile_fingerprint"] = sha256_fingerprint(profile)
    profile["config_sha256"] = file_sha256(path)
    return profile


def checkpoint_metadata(
    semantic_profile: Mapping[str, Any],
    stage4: Mapping[str, Any],
    *,
    run_id: str,
    seed_label: str,
    variant: str,
    label: str,
    parent: str | None,
    code,
    model: torch.nn.Module,
    completed_games: int,
    valid_replay_positions: int,
    optimizer_updates: int,
    train_samples_consumed: int,
    model_init_seed: int,
) -> dict[str, object]:
    heads = {"policy": [26], "value": [3]}
    if variant != "wdl":
        heads.update({"ownership": [25, 3], "score": [1]})
    return {
        "checkpoint_schema_version": 1,
        "checkpoint_label": label,
        "model_variant": variant,
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
        "network_heads_and_shapes": heads,
        "auxiliary_target_contracts": {
            "source": "golden-referee-final-state-v1",
            "ownership": "golden-ownership-final-state-side-to-move-v1",
            "score": "golden-score-final-margin-side-to-move-v1",
            "score_normalization": SCORE_NORMALIZATION,
        },
        "training_profile_id": semantic_profile["profile_id"],
        "training_profile_fingerprint": semantic_profile["profile_fingerprint"],
        "stage4_training_profile_id": stage4["profile_id"],
        "stage4_training_profile_fingerprint": stage4["profile_fingerprint"],
        "selfplay_contract_id": semantic_profile["self_play"]["contract_id"],
        "selfplay_contract_fingerprint": semantic_profile["self_play"]["fingerprint"],
        "parent_or_source_run_identity": parent or run_id,
        "parent_checkpoint_label": parent,
        "run_id": run_id,
        "seed_label": seed_label,
        "ablation_arm": variant,
        "completed_games": completed_games,
        "valid_replay_positions": valid_replay_positions,
        "optimizer_updates": optimizer_updates,
        "train_samples_consumed": train_samples_consumed,
        "model_initialization_seed": model_init_seed,
        "shared_parameter_hash": shared_parameter_hash(model),
        "git_commit": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "git_worktree_clean": code.working_tree_clean,
        "device": "cpu",
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_hash": model_hash(model),
    }


def model_from_checkpoint(path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    metadata = read_json(path.with_suffix(".metadata.json"))
    model = instantiate_model_from_metadata(metadata).to("cpu")
    loaded = load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]}, device="cpu")
    if model_hash(model) != loaded["model_hash"]:
        raise ValueError(f"Checkpoint model hash changed on load: {path}")
    return model, loaded


def save_model_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    saved = save_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.is_file() or saved.get("artifact_sha256") != file_sha256(path):
        raise ValueError(f"Checkpoint artifact identity failed: {path}")
    return saved


def _selfplay_worker_init(
    checkpoint_path: str,
    expected_model_hash: str,
    run_id: str,
    seed_namespace: str,
    label: str,
    artifact: str,
    seed: int,
    profile_fingerprint: str,
    code_commit: str,
    code_tree: str,
    code_clean: bool,
) -> None:
    global _SELFPLAY_MODEL, _SELFPLAY_CONFIG
    seed_everything(0)
    metadata = read_json(Path(checkpoint_path).with_suffix(".metadata.json"))
    model = instantiate_model_from_metadata(metadata).to("cpu")
    load_checkpoint(checkpoint_path, model=model, expected={"model_hash": expected_model_hash}, device="cpu")
    if model_hash(model) != expected_model_hash:
        raise RuntimeError("Self-play worker loaded an unexpected checkpoint")
    _SELFPLAY_MODEL = model
    _SELFPLAY_CONFIG = {
        "run_id": run_id,
        "seed_namespace": seed_namespace,
        "label": label,
        "artifact": artifact,
        "seed": seed,
        "profile_fingerprint": profile_fingerprint,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "code_clean": code_clean,
    }


def _selfplay_worker_game(game_id: str) -> SelfPlayGameRecord:
    if _SELFPLAY_MODEL is None:
        raise RuntimeError("Self-play process worker was not initialized")
    evaluator = GoldenNeuralEvaluator(_SELFPLAY_MODEL, device="cpu")
    runner = GoldenSelfPlayRunner(
        _SELFPLAY_MODEL,
        run_id=str(_SELFPLAY_CONFIG["run_id"]),
        seed_namespace=str(_SELFPLAY_CONFIG["seed_namespace"]),
        profile_fingerprint=str(_SELFPLAY_CONFIG["profile_fingerprint"]),
        model_checkpoint_label=str(_SELFPLAY_CONFIG["label"]),
        checkpoint_artifact_hash=str(_SELFPLAY_CONFIG["artifact"]),
        master_seed=int(_SELFPLAY_CONFIG["seed"]),
        code_identity=type(capture_code_identity())(
            str(_SELFPLAY_CONFIG["code_commit"]),
            str(_SELFPLAY_CONFIG["code_tree"]),
            bool(_SELFPLAY_CONFIG["code_clean"]),
        ),
        device="cpu",
        evaluator=evaluator,
    )
    return runner.play_game(game_id)


def run_process_selfplay(
    *,
    checkpoint_path: Path,
    metadata: Mapping[str, object],
    run_id: str,
    seed_namespace: str,
    label: str,
    master_seed: int,
    code,
    game_ids: Sequence[str],
    workers: int,
) -> tuple[SelfPlayGameRecord, ...]:
    ordered = tuple(sorted(str(game_id) for game_id in game_ids))
    if workers <= 1:
        model, _ = model_from_checkpoint(checkpoint_path)
        evaluator = GoldenNeuralEvaluator(model, device="cpu")
        runner = GoldenSelfPlayRunner(
            model,
            run_id=run_id,
            seed_namespace=seed_namespace,
            profile_fingerprint=str(metadata["training_profile_fingerprint"]),
            model_checkpoint_label=label,
            checkpoint_artifact_hash=str(metadata["artifact_sha256"]),
            master_seed=master_seed,
            code_identity=code,
            evaluator=evaluator,
        )
        return tuple(run_selfplay_games(lambda _: runner, ordered, workers=1))
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=get_context("spawn"),
        initializer=_selfplay_worker_init,
        initargs=(
            str(checkpoint_path),
            str(metadata["model_hash"]),
            run_id,
            seed_namespace,
            label,
            str(metadata["artifact_sha256"]),
            master_seed,
            str(metadata["training_profile_fingerprint"]),
            code.git_commit_sha,
            code.git_tree_sha,
            code.working_tree_clean,
        ),
    ) as pool:
        return tuple(pool.map(_selfplay_worker_game, ordered))


def corpus_fingerprint(records: Sequence[SelfPlayGameRecord]) -> str:
    return sha256_fingerprint([record.to_dict() for record in records])


def write_immutable_corpus(directory: Path, records: Sequence[SelfPlayGameRecord], samples: Sequence[GoldenTrainingSample], manifest: Mapping[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    games_path = directory / "games.jsonl"
    replay_path = directory / "replay.jsonl"
    manifest_path = directory / "manifest.json"
    if games_path.exists() or replay_path.exists() or manifest_path.exists():
        if not manifest_path.exists() or read_json(manifest_path).get("corpus_fingerprint") != manifest.get("corpus_fingerprint"):
            raise ValueError(f"Immutable corpus already exists with a different fingerprint: {directory}")
        return
    write_jsonl(games_path, (record.to_dict() for record in records))
    write_jsonl(replay_path, (sample.to_dict() for sample in samples))
    write_json(manifest_path, manifest)


def audit_corpus(records: Sequence[SelfPlayGameRecord], *, label: str, expected_model_hash: str) -> tuple[dict[str, object], tuple[GoldenTrainingSample, ...]]:
    audit = audit_selfplay_records(records, expected_model_hash_by_label={label: expected_model_hash})
    if audit["technical_games"] != 0 or audit["valid_games"] != len(records):
        raise RuntimeError(f"Technical self-play games are not permitted: {audit}")
    samples = tuple(sample for record in records for sample in build_replay_samples(record))
    if not samples or any(sample.ownership_target is None or sample.score_target is None for sample in samples):
        raise RuntimeError("Golden auxiliary targets are missing from a valid replay corpus")
    return audit, samples


def _make_tiny_sample() -> GoldenTrainingSample:
    from gocube_golden.training import ownership_target, score_target, state_identity, z_target
    from gocube_golden.rules import apply_action
    from gocube_golden.neural import build_observation
    state = apply_action(apply_action(__import__("gocube_golden").initial_state(), "PASS").after, "PASS").after
    # Use a formal empty-board draw fixture for all semantic heads.
    live = __import__("gocube_golden").initial_state()
    observation = build_observation(live)
    visits = tuple([1] * 26)
    pi = tuple(1.0 / 26.0 for _ in range(26))
    return GoldenTrainingSample(
        run_id="smoke",
        game_id="smoke-game",
        ply=1,
        state=state_identity(live),
        side_to_move="BLACK",
        observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
        legal_action_mask=tuple([True] * 26),
        root_visits=visits,
        pi=pi,
        z=z_target("WHITE", live.side_to_move),
        model_hash="sha256:" + "0" * 64,
        selfplay_contract_fingerprint="sha256:" + "0" * 64,
        ownership_target=ownership_target(state, live.side_to_move),
        score_target=score_target(state, live.side_to_move),
        auxiliary_target_source="golden-referee-final-state-v1",
    )


def fail_fast_smoke(semantic_profile: Mapping[str, Any], stage4: Mapping[str, Any], code, run_dir: Path) -> dict[str, object]:
    from gocube_golden.training import ownership_target, score_target
    from gocube_golden.rules import apply_action
    from gocube_golden.state import BLACK, WHITE, EMPTY, research_state_from_stones
    from gocube_golden.neural import build_observation

    empty = [EMPTY] * 25
    final = apply_action(apply_action(research_state_from_stones(empty, side_to_move=BLACK), "PASS").after, "PASS").after
    if set(ownership_target(final, BLACK)) != {2} or score_target(final, BLACK) != -0.5:
        raise RuntimeError("Ownership/score target semantic smoke failed")
    stones = list(empty)
    stones[0] = BLACK
    stones[1] = WHITE
    mixed = apply_action(apply_action(research_state_from_stones(stones, side_to_move=WHITE), "PASS").after, "PASS").after
    if ownership_target(mixed, BLACK)[0] != 0 or ownership_target(mixed, WHITE)[0] != 1:
        raise RuntimeError("Ownership perspective inversion smoke failed")
    shared = {shared_parameter_hash(make_model(variant, SEED_CONFIG["A"]["model"])) for variant in VARIANTS}
    if len(shared) != 1:
        raise RuntimeError(f"Shared-M0 identity smoke failed: {shared}")
    sample = _make_tiny_sample()
    checkpoints: list[Path] = []
    for variant in VARIANTS:
        model = make_model(variant, SEED_CONFIG["A"]["model"])
        schedule = ((0,),)
        if variant == "wdl":
            optimizer, metrics = train_batch_schedule(model, (sample,), schedule, learning_rate=0.001, weight_decay=0.0)
        else:
            optimizer, metrics = train_variant_batch_schedule(model, (sample,), schedule, variant=variant, learning_rate=0.001, weight_decay=0.0)
        with torch.inference_mode():
            policy, value = model(torch.tensor([sample.observation], dtype=torch.float32))
            if not bool(torch.isfinite(policy).all()) or not bool(torch.isfinite(value).all()):
                raise RuntimeError("Tiny training produced non-finite baseline outputs")
            if variant != "wdl":
                outputs = model.forward_auxiliary(torch.tensor([sample.observation], dtype=torch.float32))
                if any(not bool(torch.isfinite(output).all()) for output in outputs):
                    raise RuntimeError("Tiny training produced non-finite auxiliary outputs")
        metadata = checkpoint_metadata(semantic_profile, stage4, run_id="smoke", seed_label="A", variant=variant, label="M1", parent="M0", code=code, model=model, completed_games=0, valid_replay_positions=1, optimizer_updates=int(metrics["updates"]), train_samples_consumed=int(metrics["exact_samples_consumed"]), model_init_seed=SEED_CONFIG["A"]["model"])
        path = run_dir / "smoke" / f"{variant.replace('+', '_')}.pt"
        saved = save_model_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)
        restored, _ = model_from_checkpoint(path)
        if model_hash(restored) != saved["model_hash"]:
            raise RuntimeError("Auxiliary checkpoint round-trip failed")
        checkpoints.append(path)
    wdl_path = checkpoints[0]
    wdl_model, wdl_metadata = model_from_checkpoint(wdl_path)
    smoke_records = run_process_selfplay(
        checkpoint_path=wdl_path,
        metadata=wdl_metadata,
        run_id="smoke",
        seed_namespace="smoke-selfplay",
        label="M1",
        master_seed=991,
        code=code,
        game_ids=("smoke-selfplay-00", "smoke-selfplay-01"),
        workers=1,
    )
    smoke_audit, smoke_samples = audit_corpus(smoke_records, label="M1", expected_model_hash=str(wdl_metadata["model_hash"]))
    candidate_spec = CheckpointPlayerSpec.from_checkpoint(
        str(wdl_path), player_id="smoke", metadata=wdl_metadata,
        artifact_sha256=str(wdl_metadata["artifact_sha256"]),
    )
    smoke_arena = ProcessParallelGoldenArena(
        player_A=candidate_spec,
        player_B=candidate_spec,
        workers=2,
        master_seed=992,
        run_id="smoke-arena",
        code_identity=code,
        mp_context="spawn",
    )
    smoke_arena_records = smoke_arena.play_pair(pair_id="smoke-pair")
    if any(record.is_technical for record in smoke_arena_records):
        raise RuntimeError("Tiny process-parallel Arena smoke produced a technical game")
    return {
        "status": "PASS",
        "variants": VARIANTS,
        "shared_parameter_hash": next(iter(shared)),
        "checkpoints": [str(path) for path in checkpoints],
        "selfplay": {"audit": smoke_audit, "positions": len(smoke_samples)},
        "arena": {"games": len(smoke_arena_records), "technical": sum(record.is_technical for record in smoke_arena_records)},
    }


def load_frozen_evaluation(run_dir: Path) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    source = ROOT / "runs" / "torus-golden-stage4" / "torus-golden-stage4-seed2-v4"
    if (source / "evaluation-v2" / "starts.jsonl").is_file():
        starts = load_frozen_starts(source)
        manifest = read_json(source / "evaluation-v2" / "manifest.json")
        return starts, {"source": str(source), **manifest}
    frozen_dir = run_dir / "frozen-evaluation"
    freeze_evaluation_v2(frozen_dir, master_seed=EVALUATION_MASTER_SEED, code=capture_code_identity())
    starts = load_frozen_starts(frozen_dir)
    return starts, {"source": str(frozen_dir), **read_json(frozen_dir / "evaluation-v2" / "manifest.json")}


def train_one_arm(
    *,
    variant: str,
    seed_label: str,
    model_seed: int,
    semantic_profile: Mapping[str, Any],
    stage4: Mapping[str, Any],
    code,
    run_id: str,
    source_model: torch.nn.Module,
    source_info: Mapping[str, object],
    samples: Sequence[GoldenTrainingSample],
    schedule_seed: int,
    output_dir: Path,
    phase: str,
) -> tuple[torch.nn.Module, dict[str, object], dict[str, object]]:
    model = source_model
    if shared_parameter_hash(model) != source_info["shared_parameter_hash"]:
        raise RuntimeError(f"Shared M0 parameter identity drift for {seed_label}/{variant}")
    schedule = low_reuse_schedule(len(samples), batch_size=TRAINING_BATCH_SIZE, seed=schedule_seed)
    started = time.perf_counter()
    if variant == "wdl":
        optimizer, training = train_batch_schedule(
            model, samples, schedule,
            learning_rate=float(stage4["training"]["learning_rate"]),
            weight_decay=float(stage4["training"]["weight_decay"]),
        )
    else:
        optimizer, training = train_variant_batch_schedule(
            model, samples, schedule, variant=variant,
            learning_rate=float(stage4["training"]["learning_rate"]),
            weight_decay=float(stage4["training"]["weight_decay"]),
        )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    label = phase
    metadata = checkpoint_metadata(
        semantic_profile, stage4, run_id=run_id, seed_label=seed_label, variant=variant,
        label=label, parent=str(source_info["model_hash"]), code=code, model=model,
        completed_games=128 if phase == "M1" else 256,
        valid_replay_positions=len(samples), optimizer_updates=int(training["updates"]),
        train_samples_consumed=int(training["exact_samples_consumed"]), model_init_seed=model_seed,
    )
    path = checkpoint_dir / f"{variant.replace('+', '_')}-{label}.pt"
    saved = save_model_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)
    report = {
        "variant": variant,
        "phase": phase,
        "checkpoint": {"path": str(path), "model_hash": saved["model_hash"], "artifact_sha256": saved["artifact_sha256"]},
        "training": training,
        "wall_time_sec": time.perf_counter() - started,
        "valid_replay_positions": len(samples),
    }
    write_json(output_dir / f"{variant.replace('+', '_')}-{phase}-training.json", report)
    return model, {**saved, "path": str(path), "shared_parameter_hash": shared_parameter_hash(model)}, report


def arena_comparison(
    *,
    run_id: str,
    slug: str,
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    candidate_label: str,
    reference_label: str,
    starts: Sequence[Mapping[str, object]],
    arena_seed: int,
    code,
    workers: int,
    output_dir: Path,
) -> dict[str, object]:
    candidate_spec = CheckpointPlayerSpec.from_checkpoint(
        str(candidate["path"]), player_id=candidate_label, device="cpu",
        metadata=candidate["metadata"], artifact_sha256=str(candidate["artifact_sha256"]),
    )
    reference_spec = CheckpointPlayerSpec.from_checkpoint(
        str(reference["path"]), player_id=reference_label, device="cpu",
        metadata=reference["metadata"], artifact_sha256=str(reference["artifact_sha256"]),
    )
    tasks = tuple(
        PairTask(
            pair_id=f"{slug}--{row['start_id']}",
            start_state=state_from_start_row(row),
            start_trace=tuple(int(action) for action in row["trace"]),
        )
        for row in starts
    )
    started = time.perf_counter()
    arena = ProcessParallelGoldenArena(
        player_A=candidate_spec, player_B=reference_spec, workers=workers,
        master_seed=arena_seed, run_id=f"{run_id}-{slug}", code_identity=code,
        mp_context="spawn",
    )
    records = arena.play_pairs(tasks)
    summary = summarize_pair_records(records, candidate_label=candidate_label, reference_label=reference_label, starts=starts)
    if summary["technical"] != 0:
        raise RuntimeError(f"Technical Arena games are not draws: {slug}: {summary}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records_jsonl(output_dir / "games.jsonl", records)
    summary.update({
        "slug": slug,
        "starts_fingerprint": evaluation_start_fingerprint(starts),
        "workers": workers,
        "wall_time_sec": time.perf_counter() - started,
        "noise": False,
        "temperature": 0.0,
        "technical_excluded": True,
    })
    write_json(output_dir / "summary.json", summary)
    return summary


def ranking_from_primary(primary: Mapping[str, Mapping[str, object]], variants: Sequence[str]) -> list[dict[str, object]]:
    points = {variant: 0.0 for variant in variants}
    games = {variant: 0 for variant in variants}
    for summary in primary.values():
        candidate = str(summary["candidate"])
        reference = str(summary["reference"])
        wl, losses, draws = (int(value) for value in summary["W/L/D"])
        points[candidate] += wl + 0.5 * draws
        points[reference] += losses + 0.5 * draws
        games[candidate] += wl + losses + draws
        games[reference] += wl + losses + draws
    rows = [
        {"variant": variant, "points": points[variant], "games": games[variant], "mean_pair_score": points[variant] / games[variant] if games[variant] else None}
        for variant in variants
    ]
    return sorted(rows, key=lambda row: (-float(row["mean_pair_score"] or 0.0), str(row["variant"])))


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    run_dir = Path(args.run_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    semantic_profile = load_stage3_profile()
    stage4 = stage4_profile()
    code = capture_code_identity()
    if semantic_profile["komi"] != 0.5 or stage4["frozen_semantics"]["komi"] != 0.5:
        raise RuntimeError("Golden experiment refuses a non-0.5 komi profile")
    starts, evaluation_manifest = load_frozen_evaluation(run_dir)
    write_json(run_dir / "frozen-evaluation-manifest.json", evaluation_manifest)
    if len(starts) != 64:
        raise RuntimeError("The frozen Golden evaluation corpus must contain 64 starts")
    smoke = fail_fast_smoke(semantic_profile, stage4, code, run_dir)
    if args.smoke:
        report = {"status": "SMOKE_PASS", "run_id": args.run_id, "komi": 0.5, "fail_fast": smoke, "evaluation_manifest": evaluation_manifest}
        write_json(run_dir / "final-report.json", report)
        return report

    experiment_started = time.perf_counter()
    all_seed_reports: dict[str, Any] = {}
    final_checkpoints: dict[str, Mapping[str, object]] = {}
    first_move_buckets: dict[str, object] = {}
    canonical_stage4 = ROOT / "runs" / "torus-golden-stage4" / "torus-golden-stage4-seed2-v4"
    if (canonical_stage4 / "selfplay").is_dir():
        from tools.torus_golden_stage4 import records_from_jsonl
        canonical_records = tuple(
            record
            for path in sorted((canonical_stage4 / "selfplay").glob("chunk-*-games.jsonl"))
            for record in records_from_jsonl(path)
        )
        if canonical_records:
            first_move_buckets["canonical-Stage4-seed2"] = first_move_statistics(
                canonical_records, source="canonical-Stage4-seed2"
            )
    for seed_label, seeds in SEED_CONFIG.items():
        seed_dir = run_dir / f"seed-{seed_label}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        m0 = make_model("wdl", seeds["model"])
        m0_metadata = checkpoint_metadata(semantic_profile, stage4, run_id=args.run_id, seed_label=seed_label, variant="wdl", label="M0", parent=None, code=code, model=m0, completed_games=0, valid_replay_positions=0, optimizer_updates=0, train_samples_consumed=0, model_init_seed=seeds["model"])
        m0_path = seed_dir / "checkpoints" / "M0.pt"
        m0_saved = save_model_checkpoint(m0_path, model=m0, optimizer=None, metadata=m0_metadata)
        m0_info = {**m0_saved, "path": str(m0_path), "metadata": read_json(m0_path.with_suffix(".metadata.json")), "shared_parameter_hash": shared_parameter_hash(m0)}
        game_ids = tuple(f"seed-{seed_label}-I1-game-{index:03d}" for index in range(SELFPLAY_GAMES))
        i1_records = run_process_selfplay(checkpoint_path=m0_path, metadata=m0_info["metadata"], run_id=args.run_id, seed_namespace=f"{seed_label}-I1", label="M0", master_seed=seeds["selfplay"], code=code, game_ids=game_ids, workers=args.workers)
        i1_audit, i1_samples = audit_corpus(i1_records, label="M0", expected_model_hash=str(m0_info["model_hash"]))
        i1_manifest = {"run_id": args.run_id, "seed": seed_label, "phase": "I1", "model_label": "M0", "games": len(i1_records), "positions": len(i1_samples), "technical": i1_audit["technical_games"], "corpus_fingerprint": corpus_fingerprint(i1_records), "immutable": True, "komi": 0.5}
        write_immutable_corpus(seed_dir / "selfplay" / "I1-shared", i1_records, i1_samples, i1_manifest)
        first_move_buckets[f"seed-{seed_label}/M0/I1"] = first_move_statistics(i1_records, source=f"seed-{seed_label}/M0/I1")

        arms: dict[str, Any] = {}
        arm_m0_shared_hashes: dict[str, str] = {}
        for variant in VARIANTS:
            arm_dir = seed_dir / variant.replace("+", "_")
            arm_m0 = make_model(variant, seeds["model"])
            arm_m0_info = {**m0_info, "shared_parameter_hash": shared_parameter_hash(arm_m0)}
            arm_m0_shared_hashes[variant] = str(arm_m0_info["shared_parameter_hash"])
            model1, info1, train1 = train_one_arm(variant=variant, seed_label=seed_label, model_seed=seeds["model"], semantic_profile=semantic_profile, stage4=stage4, code=code, run_id=args.run_id, source_model=arm_m0, source_info=arm_m0_info, samples=i1_samples, schedule_seed=derive_seed(seeds["model"], "I1-training-order"), output_dir=arm_dir, phase="M1")
            info1 = {**info1, "metadata": read_json(Path(str(info1["path"])).with_suffix(".metadata.json"))}
            diag = arena_comparison(run_id=args.run_id, slug=f"{seed_label}-{variant}-M1-vs-M0", candidate=info1, reference=m0_info, candidate_label="M1", reference_label="M0", starts=diagnostic_subset(starts), arena_seed=derive_seed(seeds["arena"], variant, "M1-M0"), code=code, workers=args.workers, output_dir=arm_dir / "arena" / "M1-vs-M0")
            i2_ids = tuple(f"seed-{seed_label}-{variant}-I2-game-{index:03d}" for index in range(SELFPLAY_GAMES))
            i2_records = run_process_selfplay(checkpoint_path=Path(str(info1["path"])), metadata=info1["metadata"], run_id=args.run_id, seed_namespace=f"{seed_label}-{variant}-I2", label="M1", master_seed=derive_seed(seeds["selfplay"], variant, "I2"), code=code, game_ids=i2_ids, workers=args.workers)
            i2_audit, i2_samples = audit_corpus(i2_records, label="M1", expected_model_hash=str(info1["model_hash"]))
            i2_manifest = {"run_id": args.run_id, "seed": seed_label, "phase": "I2", "model_label": "M1", "variant": variant, "games": len(i2_records), "positions": len(i2_samples), "technical": i2_audit["technical_games"], "corpus_fingerprint": corpus_fingerprint(i2_records), "immutable": True, "komi": 0.5}
            write_immutable_corpus(arm_dir / "selfplay" / "I2", i2_records, i2_samples, i2_manifest)
            first_move_buckets[f"seed-{seed_label}/{variant}/M1-I2"] = first_move_statistics(i2_records, source=f"seed-{seed_label}/{variant}/M1-I2")
            model2, info2, train2 = train_one_arm(variant=variant, seed_label=seed_label, model_seed=seeds["model"], semantic_profile=semantic_profile, stage4=stage4, code=code, run_id=args.run_id, source_model=model1, source_info=info1, samples=i2_samples, schedule_seed=derive_seed(seeds["model"], "I2-training-order"), output_dir=arm_dir, phase="M2")
            info2 = {**info2, "metadata": read_json(Path(str(info2["path"])).with_suffix(".metadata.json"))}
            arms[variant] = {"M1": info1, "M2": info2, "training_M1": train1, "training_M2": train2, "M1_vs_M0": diag, "I1": i1_manifest, "I2": i2_manifest}
            final_checkpoints[f"{seed_label}/{variant}"] = info2

        primary: dict[str, object] = {}
        for index, left in enumerate(VARIANTS):
            for right in VARIANTS[index + 1:]:
                slug = f"{seed_label}-{left}-vs-{right}-M2".replace("+", "-")
                primary[slug] = arena_comparison(run_id=args.run_id, slug=slug, candidate=arms[left]["M2"], reference=arms[right]["M2"], candidate_label=left, reference_label=right, starts=starts, arena_seed=derive_seed(seeds["arena"], "primary", left, right), code=code, workers=args.workers, output_dir=seed_dir / "arena" / slug)
        progression: dict[str, object] = {}
        for variant in VARIANTS:
            progression[f"{variant}-M2-vs-M1"] = arena_comparison(run_id=args.run_id, slug=f"{seed_label}-{variant}-M2-vs-M1", candidate=arms[variant]["M2"], reference=arms[variant]["M1"], candidate_label="M2", reference_label="M1", starts=diagnostic_subset(starts), arena_seed=derive_seed(seeds["arena"], "progression", variant, 2), code=code, workers=args.workers, output_dir=seed_dir / "arena" / "progression" / variant / "M2-vs-M1")
            progression[f"{variant}-M2-vs-M0"] = arena_comparison(run_id=args.run_id, slug=f"{seed_label}-{variant}-M2-vs-M0", candidate=arms[variant]["M2"], reference=m0_info, candidate_label="M2", reference_label="M0", starts=diagnostic_subset(starts), arena_seed=derive_seed(seeds["arena"], "progression", variant, 0), code=code, workers=args.workers, output_dir=seed_dir / "arena" / "progression" / variant / "M2-vs-M0")
        ranking = ranking_from_primary(primary, VARIANTS)
        all_seed_reports[seed_label] = {"m0": m0_info, "arms": arms, "primary": primary, "progression": progression, "ranking": ranking, "shared_m0_hash": m0_info["shared_parameter_hash"], "arm_m0_shared_hashes": arm_m0_shared_hashes}

    control_stats: dict[str, object] = {}
    for key, checkpoint in final_checkpoints.items():
        seed_label, variant = key.split("/", 1)
        seeds = SEED_CONFIG[seed_label]
        model_path = Path(str(checkpoint["path"]))
        metadata = checkpoint["metadata"]
        ids = tuple(f"control-{seed_label}-{variant}-game-{index:03d}" for index in range(CONTROL_GAMES))
        controls = run_process_selfplay(checkpoint_path=model_path, metadata=metadata, run_id=args.run_id, seed_namespace=f"{seed_label}-{variant}-M2-control", label="M2", master_seed=seeds["control"], code=code, game_ids=ids, workers=args.workers)
        audit, _ = audit_corpus(controls, label="M2", expected_model_hash=str(checkpoint["model_hash"]))
        control_stats[key] = {"protocol": "self-play first-move estimate", "audit": audit, "statistics": first_move_statistics(controls, source=f"same-model-control/{key}"), "games": len(controls), "technical": audit["technical_games"]}
        write_jsonl(run_dir / "first-move" / f"{seed_label}-{variant.replace('+', '_')}-M2-control.jsonl", (record.to_dict() for record in controls))

    combined_primary: dict[str, dict[str, object]] = {}
    for seed_report in all_seed_reports.values():
        for slug, summary in seed_report["primary"].items():
            combined_primary[slug] = summary
    combined_ranking = ranking_from_primary(combined_primary, VARIANTS)
    seed_rankings = {seed: report["ranking"] for seed, report in all_seed_reports.items()}
    winner_consistency = seed_rankings["A"][0]["variant"] == seed_rankings["B"][0]["variant"]
    confidence = "CLEAR WINNER" if winner_consistency and float(seed_rankings["A"][0]["mean_pair_score"] or 0) - float(seed_rankings["A"][1]["mean_pair_score"] or 0) >= 0.10 else "LIKELY WINNER" if winner_consistency else "INCONCLUSIVE"
    first_move_summary = aggregate_first_move_statistics(control_stats)
    report = {
        "status": "PASS",
        "run_id": args.run_id,
        "source_commit": code.git_commit_sha,
        "source_tree": code.git_tree_sha,
        "branch": branch_name(),
        "komi": 0.5,
        "variants": list(VARIANTS),
        "seeds": SEED_CONFIG,
        "fail_fast": smoke,
        "evaluation_manifest": evaluation_manifest,
        "seed_reports": all_seed_reports,
        "combined_ranking": combined_ranking,
        "confidence": confidence,
        "primary_games": 6 * PRIMARY_PAIRS * len(SEED_CONFIG) * 2,
        "total_arena_games": (6 * PRIMARY_PAIRS * len(SEED_CONFIG) * 2) + (len(VARIANTS) * len(SEED_CONFIG) * DIAGNOSTIC_PAIRS * 2) + (len(VARIANTS) * len(SEED_CONFIG) * 2 * DIAGNOSTIC_PAIRS * 2),
        "total_selfplay_games": (len(SEED_CONFIG) * SELFPLAY_GAMES) + (len(VARIANTS) * len(SEED_CONFIG) * SELFPLAY_GAMES) + (len(VARIANTS) * len(SEED_CONFIG) * CONTROL_GAMES),
        "first_move_selfplay": first_move_buckets,
        "first_move_controls": control_stats,
        "first_move_summary": first_move_summary,
        "first_move_conclusion": "INCONCLUSIVE",
        "telemetry": {"wall_time_sec": time.perf_counter() - experiment_started, "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, "platform": platform.platform(), "workers": args.workers},
        "technical_games_expected": 0,
        "ci": "not run until final push",
    }
    report["first_move_conclusion"] = derive_first_move_conclusion(control_stats)
    write_json(run_dir / "final-report.json", report)
    write_human_report(run_dir / "final-report.md", report)
    return report


def derive_first_move_conclusion(control_stats: Mapping[str, Mapping[str, object]]) -> str:
    if len(control_stats) != 8:
        return "INCONCLUSIVE"
    def wilson(wins: int, games: int) -> tuple[float, float] | None:
        if games <= 0:
            return None
        z = 1.959963984540054
        rate = wins / games
        denominator = 1.0 + z * z / games
        centre = (rate + z * z / (2.0 * games)) / denominator
        radius = z * math.sqrt(rate * (1.0 - rate) / games + z * z / (4.0 * games * games)) / denominator
        return max(0.0, centre - radius), min(1.0, centre + radius)

    rates = [float(item["statistics"]["black_win_rate"]) for item in control_stats.values() if item["statistics"]["black_win_rate"] is not None]
    if len(rates) != 8:
        return "INCONCLUSIVE"
    aggregate_wins = sum(int(item["statistics"]["black_wins"]) for item in control_stats.values())
    aggregate_games = sum(int(item["statistics"]["games"]) for item in control_stats.values())
    aggregate_ci = wilson(aggregate_wins, aggregate_games)
    seed_cis = []
    for seed in ("A", "B"):
        rows = [item for key, item in control_stats.items() if key.startswith(seed + "/")]
        seed_cis.append(wilson(sum(int(item["statistics"]["black_wins"]) for item in rows), sum(int(item["statistics"]["games"]) for item in rows)))
    if aggregate_ci is None or any(interval is None for interval in seed_cis):
        return "INCONCLUSIVE"
    if all(float(interval[0]) > 0.5 for interval in seed_cis) and float(aggregate_ci[0]) > 0.5 and all(rate > 0.5 for rate in rates):
        return "FIRST-MOVE ADVANTAGE: DETECTED"
    if any(float(interval[0]) > 0.5 for interval in seed_cis) and any(float(interval[1]) < 0.5 for interval in seed_cis):
        return "INCONCLUSIVE"
    if float(aggregate_ci[1]) <= 0.5:
        return "NO DETECTABLE FIRST-MOVE ADVANTAGE"
    return "INCONCLUSIVE"


def aggregate_first_move_statistics(control_stats: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Combine the eight predeclared controls without discarding model strata."""
    wins = sum(int(item["statistics"]["black_wins"]) for item in control_stats.values())
    losses = sum(int(item["statistics"]["white_wins"]) for item in control_stats.values())
    draws = sum(int(item["statistics"]["draws"]) for item in control_stats.values())
    games = wins + losses + draws
    rate = wins / games if games else None
    z = 1.959963984540054
    if games:
        denominator = 1.0 + z * z / games
        centre = (rate + z * z / (2.0 * games)) / denominator
        radius = z * math.sqrt(rate * (1.0 - rate) / games + z * z / (4.0 * games * games)) / denominator
        interval = [max(0.0, centre - radius), min(1.0, centre + radius)]
    else:
        interval = None
    raw_means = [float(item["statistics"]["raw_black_area_minus_white_area"]["mean"]) for item in control_stats.values()]
    margin_means = [float(item["statistics"]["final_margin_black_after_komi_0_5"]["mean"]) for item in control_stats.values()]
    return {
        "games": games,
        "black_wins": wins,
        "white_wins": losses,
        "draws": draws,
        "black_win_rate": rate,
        "black_win_rate_95_percent_ci": interval,
        "raw_black_area_minus_white_area_mean_across_controls": sum(raw_means) / len(raw_means) if raw_means else None,
        "final_margin_black_after_komi_0_5_mean_across_controls": sum(margin_means) / len(margin_means) if margin_means else None,
        "technical_games": sum(int(item["technical"]) for item in control_stats.values()),
        "strata": {key: item["statistics"] for key, item in control_stats.items()},
        "interpretation": "aggregate across eight independent same-model controls; per-model strata retained",
    }


def write_human_report(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "# TORUS 5x5 AUXILIARY-HEAD A/B",
        "",
        f"Status: **{report['status']}**",
        f"Source commit: `{report['source_commit']}`",
        f"Experiment run: `{report['run_id']}`",
        "Komi: **0.5**",
        "",
        "## Final M2 ranking",
        "",
        "| Rank | Variant | Mean score | Games |",
        "|---:|---|---:|---:|",
    ]
    for index, row in enumerate(report["combined_ranking"], 1):
        lines.append(f"| {index} | {row['variant']} | {row['mean_pair_score']:.4f} | {row['games']} |")
    lines.extend([
        "",
        f"Confidence: **{report['confidence']}**",
        "",
        "## FIRST-MOVE ADVANTAGE",
        "",
        f"**{report['first_move_conclusion']}**",
        "",
        f"Aggregate Black win rate: {report['first_move_summary']['black_win_rate']:.4f} (95% CI {report['first_move_summary']['black_win_rate_95_percent_ci'][0]:.4f}–{report['first_move_summary']['black_win_rate_95_percent_ci'][1]:.4f})",
        f"Raw Black area advantage mean: {report['first_move_summary']['raw_black_area_minus_white_area_mean_across_controls']:.4f}",
        f"Final Black margin mean at komi=0.5: {report['first_move_summary']['final_margin_black_after_komi_0_5_mean_across_controls']:.4f}",
        "",
        "Controls use the frozen Golden self-play protocol as a stochastic same-model estimate; technical games are excluded from WDL/statistics.",
        "",
        f"Total Arena games: {report['total_arena_games']} (primary: {report['primary_games']})",
        f"Total self-play games: {report['total_selfplay_games']}",
        f"Total first-move control games: {sum(int(item['games']) for item in report['first_move_controls'].values())}",
        f"Wall time: {report['telemetry']['wall_time_sec']:.1f}s",
        "",
        "Full machine-readable evidence is in `final-report.json`; checkpoint, corpus, Arena, and control artifacts are under the run directory.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be in 1..16")
    try:
        report = run_experiment(args)
    except Exception as exc:
        run_dir = Path(args.run_root) / args.run_id
        failure = {"status": "FAIL", "run_id": args.run_id, "komi": 0.5, "error": f"{type(exc).__name__}: {exc}"}
        write_json(run_dir / "failure-report.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
