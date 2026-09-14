#!/usr/bin/env python3
"""Run the controlled Torus9 M8 -> M10 WDL/ownership ablation.

The two rounds deliberately share D9 and D10.  D10 is generated once from the
frozen M8 reference as requested, so neither branch can influence the second
corpus.  No M9 Arena is created; the only Arena is M10-B versus M10-A.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.provenance import capture_code_identity, derive_seed, file_sha256
from gocube_golden.torus9 import (
    Torus9GraphNet,
    Torus9OwnershipGraphNet,
    Torus9OwnershipTrainer,
    Torus9RollingReplay,
    Torus9SelfPlayGameRecord,
    generate_torus9_evaluation_starts,
    run_torus9_batched_arena,
    run_torus9_selfplay_games,
    torus9_build_ownership_replay_samples,
    torus9_checkpoint_info,
    torus9_checkpoint_metadata,
    torus9_load_checkpoint,
    torus9_ownership_target,
    torus9_restore_optimizer_state,
    torus9_save_checkpoint,
    torus9_state_from_identity,
    validate_torus9_replay_sample,
    write_json,
    write_jsonl,
)
from gocube_golden.torus9_contract import (
    TORUS9_ARCHITECTURE_ID,
    TORUS9_BATCH_SIZE,
    TORUS9_BLOCKS,
    TORUS9_HIDDEN,
    TORUS9_KOMI,
    TORUS9_MAX_REPLAY_POSITIONS,
    TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    TORUS9_PROFILE_ID,
    TORUS9_ROLLING_GENERATIONS,
    TORUS9_WORKERS,
    load_torus9_profile,
    profile_fingerprint,
)


DEFAULT_RUN_ID = "torus9-wdl-ownership-ab-20260913-v1"
DEFAULT_RUN_ROOT = ROOT / "runs" / "torus9-ownership-ab"
DEFAULT_STABLE_RUN = ROOT / "runs" / "torus9-stable-learning-v2" / "torus9-stable-learning-20260913-v1"
DEFAULT_M8 = DEFAULT_STABLE_RUN / "canonical" / "checkpoints" / "M8.pt"
CORPUS_GAMES = 64
ARENA_PAIRS = 32
ARENA_BATCH_SIZE = 8
ARENA_INFERENCE_BATCH_WAIT_MS = 6.0
M8_OPTIMIZER_STEPS = 8 * TORUS9_OPTIMIZER_STEPS_PER_ITERATION
M8_TRAIN_SAMPLES = 8 * TORUS9_OPTIMIZER_STEPS_PER_ITERATION * TORUS9_BATCH_SIZE
OWNERSHIP_HEAD_SEED = 2026091411
D9_SEED = 2026091412
D10_SEED = 2026091413
TRAINING_SEEDS = {9: 2026091414, 10: 2026091415}


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


def canonical(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def branch_name() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shared_parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("ownership_head."):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def load_m8(path: Path) -> tuple[Torus9GraphNet, dict[str, Any]]:
    if not path.is_file() or not path.with_suffix(".metadata.json").is_file():
        raise FileNotFoundError(f"Stable Torus9 M8 checkpoint/sidecar is missing: {path}")
    sidecar = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    model = Torus9GraphNet(hidden=TORUS9_HIDDEN, blocks=TORUS9_BLOCKS)
    metadata = torus9_load_checkpoint(path, model=model, expected={"model_hash": sidecar["model_hash"], "checkpoint_label": "M8"})
    if metadata.get("architecture_id") != TORUS9_ARCHITECTURE_ID or metadata.get("architecture_config", {}).get("heads") != {"policy": [82], "value": [3]}:
        raise ValueError("The supplied M8 is not the stable Torus9 policy/WDL checkpoint")
    if metadata.get("auxiliary_heads") is not False or metadata.get("komi") != TORUS9_KOMI:
        raise ValueError("The supplied M8 has auxiliary-head or komi drift")
    if int(metadata.get("optimizer_updates", -1)) != M8_OPTIMIZER_STEPS:
        raise ValueError("The supplied M8 does not carry the stable 640-step optimizer lineage")
    if int(metadata.get("train_samples_consumed", -1)) != M8_TRAIN_SAMPLES:
        raise ValueError("The supplied M8 does not carry the stable 40,960-sample lineage")
    return model.eval(), dict(metadata)


def make_auxiliary_from_m8(m8: torch.nn.Module, *, seed: int) -> Torus9OwnershipGraphNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = Torus9OwnershipGraphNet(hidden=TORUS9_HIDDEN, blocks=TORUS9_BLOCKS)
    source = m8.state_dict()
    target = model.state_dict()
    for name, value in source.items():
        if name not in target:
            raise ValueError(f"M8 parameter is absent from the ownership model: {name}")
        target[name].copy_(value)
    if shared_parameter_hash(model) != shared_parameter_hash(m8):
        raise ValueError("M8 -> ownership model shared parameter identity drift")
    return model.eval()


def frozen_arena_starts(stable_run: Path, *, output: Path) -> tuple[list[dict[str, Any]], dict[str, object]]:
    source = stable_run / "evaluation" / "starts.jsonl"
    if source.is_file():
        all_starts = read_jsonl(source)
        if len(all_starts) != 64:
            raise ValueError("Stable Torus9 evaluation corpus must contain 64 starts")
        selected: list[dict[str, Any]] = []
        counts: Counter[int] = Counter()
        for row in all_starts:
            prefix = int(row["prefix_length"])
            if counts[prefix] < 4:
                selected.append(dict(row))
                counts[prefix] += 1
    else:
        selected = list(generate_torus9_evaluation_starts(master_seed=2026091401, accepted_per_stratum=4))
    if len(selected) != ARENA_PAIRS or sorted(Counter(int(row["prefix_length"]) for row in selected).values()) != [4] * 8:
        raise ValueError("Torus9 A/B Arena corpus must contain exactly 4 starts in each of 8 strata")
    corpus_fp = fingerprint(selected)
    for row in selected:
        row["ab_corpus_fingerprint"] = corpus_fp
    manifest = {
        "contract_id": "torus9-ownership-ab-frozen-arena-corpus-v1",
        "source": str(source),
        "starts": len(selected),
        "prefix_lengths": [2, 4, 6, 8, 10, 12, 14, 16],
        "accepted_per_stratum": 4,
        "color_swapped_pair_semantics": True,
        "fingerprint": corpus_fp,
        "created_before_results": True,
        "komi": TORUS9_KOMI,
    }
    write_jsonl(output, selected)
    write_json(output.with_name("manifest.json"), manifest)
    return selected, manifest


def audit_corpus(records: Sequence[Torus9SelfPlayGameRecord], *, m8_hash: str, label: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    technical = [record for record in records if record.technical_termination is not None]
    if technical:
        raise RuntimeError(f"{label} contains technical self-play games; corpus is not usable: {[(r.game_id, r.technical_termination) for r in technical]}")
    rows: list[dict[str, object]] = []
    record_by_id = {record.game_id: record for record in records}
    final_states: dict[str, Any] = {}
    for record in records:
        record.validate()
        if record.model_hash != m8_hash:
            raise ValueError(f"{label} model source drift in {record.game_id}")
        final_state = torus9_state_from_identity(record.start_state)
        from gocube_golden.rules import apply_action
        for action in record.final_action_trace:
            final_state = apply_action(final_state, action).after
        final_states[record.game_id] = final_state
        rows.extend(torus9_build_ownership_replay_samples(record))
    for row in rows:
        validate_torus9_replay_sample(row)
        if row.get("ownership_target") is None:
            raise ValueError(f"{label} ownership target missing")
        record = record_by_id[str(row["game_id"])]
        final_state = final_states[record.game_id]
        expected = torus9_ownership_target(final_state, str(row["side_to_move"]))
        if tuple(int(value) for value in row["ownership_target"]) != expected:  # type: ignore[arg-type]
            raise ValueError(f"{label} ownership target is not exact referee output for {row['game_id']}:{row['ply']}")
    audit = {
        "games": len(records),
        "valid_games": len(records),
        "technical_games": 0,
        "positions": len(rows),
        "outcomes": dict(Counter(record.formal_result for record in records)),
        "model_hash": m8_hash,
        "ownership_target_contract": "golden-ownership-final-state-side-to-move-v1",
        "technical_excluded_from_training": True,
    }
    return audit, rows


def write_corpus(root: Path, label: str, records: Sequence[Torus9SelfPlayGameRecord], rows: Sequence[Mapping[str, object]], audit: Mapping[str, object], *, source_seed: int) -> dict[str, object]:
    directory = root / "selfplay" / label
    directory.mkdir(parents=True, exist_ok=True)
    games_path = directory / "games.jsonl"
    replay_path = directory / "replay.jsonl"
    record_dicts = [record.to_dict() for record in records]
    replay_dicts = list(rows)
    write_jsonl(games_path, record_dicts)
    write_jsonl(replay_path, replay_dicts)
    manifest = {
        "run_id": root.name,
        "corpus": label,
        "source_model_label": "M8",
        "source_model_hash": audit["model_hash"],
        "master_seed": source_seed,
        "games": len(records),
        "positions": len(rows),
        "technical_games": audit["technical_games"],
        "corpus_fingerprint": fingerprint(record_dicts),
        "replay_fingerprint": fingerprint(replay_dicts),
        "immutable": True,
        "komi": TORUS9_KOMI,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_existing_corpus(root: Path, *, label: str, seed: int, m8_hash: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Resume from a complete immutable corpus produced by this runner."""
    directory = root / "selfplay" / label
    manifest_path = directory / "manifest.json"
    replay_path = directory / "replay.jsonl"
    if not manifest_path.is_file() or not replay_path.is_file():
        raise FileNotFoundError(f"Cannot reuse incomplete {label} corpus")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("immutable") is not True or manifest.get("master_seed") != seed:
        raise ValueError(f"Existing {label} corpus is not the requested immutable seed")
    if manifest.get("source_model_hash") != m8_hash:
        raise ValueError(f"Existing {label} corpus has a different M8 source")
    rows = read_jsonl(replay_path)
    if fingerprint(rows) != manifest.get("replay_fingerprint"):
        raise ValueError(f"Existing {label} replay fingerprint drift")
    if int(manifest.get("positions", -1)) != len(rows):
        raise ValueError(f"Existing {label} replay position count drift")
    for row in rows:
        validate_torus9_replay_sample(row)
    manifest["audit"] = {
        "games": int(manifest.get("games", 0)),
        "valid_games": int(manifest.get("games", 0)),
        "technical_games": int(manifest.get("technical_games", 0)),
        "positions": len(rows),
        "model_hash": m8_hash,
        "ownership_target_contract": "golden-ownership-final-state-side-to-move-v1",
        "technical_excluded_from_training": True,
        "reused_after_prior_successful_audit": True,
    }
    return manifest, rows


def generate_corpus(root: Path, *, label: str, seed: int, m8: torch.nn.Module, m8_path: Path, m8_metadata: Mapping[str, object], profile_fp: str, code) -> tuple[dict[str, object], list[dict[str, object]]]:
    ids = tuple(f"{label}-M8-game-{index:04d}" for index in range(CORPUS_GAMES))
    records = run_torus9_selfplay_games(
        m8,
        checkpoint_path=m8_path,
        run_id=root.name,
        label="M8",
        artifact=file_sha256(m8_path),
        master_seed=seed,
        profile_fp=profile_fp,
        game_ids=ids,
        workers=TORUS9_WORKERS,
        code_identity=code,
        device="cpu",
    )
    audit, rows = audit_corpus(records, m8_hash=str(m8_metadata["model_hash"]), label=label)
    manifest = write_corpus(root, label, records, rows, audit, source_seed=seed)
    manifest["audit"] = audit
    return manifest, rows


def restore_m8_optimizer(path: Path, model: Torus9OwnershipGraphNet, trainer: Torus9OwnershipTrainer, *, source_parameter_count: int) -> int:
    step = torus9_restore_optimizer_state(path, model=model, optimizer=trainer.optimizer, source_parameter_count=source_parameter_count)
    trainer.update_count = step
    trainer.samples_consumed = M8_TRAIN_SAMPLES
    if trainer.assert_optimizer_continuity() != M8_OPTIMIZER_STEPS:
        raise ValueError("M8 optimizer continuation did not restore step 640")
    return step


def train_round(
    *,
    root: Path,
    arm: str,
    model: Torus9OwnershipGraphNet,
    trainer: Torus9OwnershipTrainer,
    replay: Torus9RollingReplay,
    generation: int,
    rows: Sequence[Mapping[str, object]],
    source_label: str,
    source_model_hash: str,
    model_seed: int,
    profile_fp: str,
    code,
) -> tuple[dict[str, object], dict[str, object]]:
    before = len(replay.rows)
    replay_metrics = replay.append_generation(generation, rows)
    if list(replay.rows[before:]) != [dict(row, source_generation=generation, replay_row_id=row.get("replay_row_id", f"M{generation}:{row.get('game_id')}:{row.get('ply')}")) for row in rows]:
        # The replay's deterministic stamping is part of the shared-data gate.
        raise ValueError(f"{arm} replay changed the common {source_label} corpus")
    write_jsonl(root / "arms" / arm / "replay" / f"after-{generation}.jsonl", list(replay.rows))
    started = time.perf_counter()
    training = trainer.train_fixed_budget(replay.rows, seed=TRAINING_SEEDS[generation])
    elapsed = time.perf_counter() - started
    if training["optimizer_steps"] != TORUS9_OPTIMIZER_STEPS_PER_ITERATION or training["samples_consumed"] != TORUS9_OPTIMIZER_STEPS_PER_ITERATION * TORUS9_BATCH_SIZE or any(size != TORUS9_BATCH_SIZE for size in training["batch_sizes"]):
        raise AssertionError(f"{arm} round {generation} fixed optimizer budget drift")
    label = f"M{generation}-{arm}"
    checkpoint_path = root / "arms" / arm / "checkpoints" / f"{label}.pt"
    metadata = torus9_checkpoint_metadata(
        model=model,
        run_id=root.name,
        label=label,
        parent=source_label,
        model_seed=model_seed,
        code=code,
        profile_fp=profile_fp,
        completed_games=(512 + (generation - 8) * CORPUS_GAMES),
        replay_positions=len(replay.rows),
        optimizer_updates=int(training["optimizer_updates_total"]),
        samples_consumed=int(training["samples_consumed_total"]),
        ownership_loss_enabled=trainer.ownership_loss_enabled,
    )
    metadata.update({
        "ablation_arm": arm,
        "source_model_hash": source_model_hash,
        "source_corpus": source_label,
        "ownership_head_initialization_seed": OWNERSHIP_HEAD_SEED,
        "optimizer_continuation_from_m8": True,
        "optimizer_semantics": "Adam state step 640 copied for shared params; new ownership params start at step 640 with zero moments; zero-weight A path advances the same state schedule",
    })
    saved = torus9_save_checkpoint(checkpoint_path, model=model, optimizer=trainer.optimizer, metadata=metadata)
    round_report = {
        "arm": arm,
        "round": generation,
        "checkpoint": {"path": str(checkpoint_path), "model_hash": saved["model_hash"], "artifact_sha256": file_sha256(checkpoint_path)},
        "replay": replay_metrics,
        "training": training,
        "training_wall_time_sec": elapsed,
        "source_corpus": source_label,
        "source_model_hash": source_model_hash,
        "ownership_loss_enabled": trainer.ownership_loss_enabled,
        "ownership_loss_weight": 1.0 if trainer.ownership_loss_enabled else 0.0,
    }
    write_json(root / "arms" / arm / f"{label}-training.json", round_report)
    return round_report, {"path": checkpoint_path, "metadata": saved, "model_hash": saved["model_hash"], "artifact_sha256": file_sha256(checkpoint_path)}


def verdict(summary: Mapping[str, object]) -> str:
    if int(summary.get("technical_games", 0)) != 0:
        return "INCONCLUSIVE"
    wins, losses, draws = (int(value) for value in summary["W/L/D"])  # type: ignore[index]
    if wins >= 45 and wins > losses:
        return "B WINNER"
    if wins >= 42 and wins > losses:
        return "B HAS USEFUL EVIDENCE"
    return "INCONCLUSIVE"


def human_report(path: Path, report: Mapping[str, object]) -> None:
    arena = report["arena"]
    lines = [
        "# TORUS 9x9 WDL + OWNERSHIP A/B",
        "",
        f"Status: **{report['status']}**",
        f"Source branch: `{report['branch']}`",
        f"Source commit: `{report['source_commit']}`",
        f"Starting checkpoint: `{report['m8']['path']}` / **M8**",
        "Komi: **0.5**",
        "",
        "## Controlled design",
        "",
        "A = WDL-only loss (ownership head present with weight 0.0); B = WDL + ownership loss (weight 1.0).",
        "D9 and D10 are each generated once from frozen M8 and reused byte-for-byte by both arms.",
        "No M9 Arena was run. The sole Arena is M10-B vs M10-A.",
        "",
        "## Final Arena",
        "",
        "| Comparison | W/L/D | Valid pairs | Technical games | Mean pair score | Verdict |",
        "|---|---:|---:|---:|---:|---|",
        f"| M10-B vs M10-A | {arena['W/L/D']} | {arena['pairs_valid']} | {arena['technical_games']} | {arena['mean_pair_score']:.4f} | {report['verdict']} |",
        "",
        "## Contract",
        "",
        f"Arena: 64 games / 32 paired starts / {report['arena_protocol']['workers']} workers / watchdog {report['arena_protocol']['watchdog']}; noise OFF, temperature OFF, fast OFF.",
        f"Training: 80 optimizer steps × batch 64 per round; Adam lr=0.001, wd=0; optimizer step 640 -> 800.",
        f"Arena execution: batched={report['arena_protocol']['batched']}, arena_batch_size={report['arena_protocol']['arena_batch_size']}, inference_batch_wait_ms={report['arena_protocol']['inference_batch_wait_ms']}, observed mean inference rows={report['arena_protocol']['mean_inference_batch_rows']:.2f}.",
        "",
        "Technical games are never converted into W/L/D; any technical Arena result makes the scientific verdict INCONCLUSIVE.",
        "",
        f"Machine-readable evidence: `{report['run_dir']}/final-report.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.workers != TORUS9_WORKERS:
        raise ValueError("The controlled Torus9 A/B is pinned to workers=16")
    if args.komi != TORUS9_KOMI:
        raise ValueError("Any komi other than 0.5 is forbidden")
    profile = load_torus9_profile()
    profile_fp = profile_fingerprint(profile)
    code = capture_code_identity(ROOT)
    root = Path(args.run_root) / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    m8, m8_metadata = load_m8(Path(args.m8))
    m8_path = Path(args.m8)
    if m8_metadata.get("profile_id") != TORUS9_PROFILE_ID:
        raise ValueError("M8 profile identity drift")
    write_json(root / "contract.json", {
        "status": "PASS",
        "starting_point": "merged PR88 / origin/main",
        "m8": {"path": str(m8_path), "model_hash": m8_metadata["model_hash"], "artifact_sha256": file_sha256(m8_path), "metadata_profile_fingerprint": m8_metadata.get("profile_fingerprint")},
        "current_profile_fingerprint": profile_fp,
        "komi": TORUS9_KOMI,
        "architecture": {"architecture_id": TORUS9_ARCHITECTURE_ID, "hidden": TORUS9_HIDDEN, "blocks": TORUS9_BLOCKS},
        "training": {"optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": TORUS9_BATCH_SIZE, "steps_per_round": TORUS9_OPTIMIZER_STEPS_PER_ITERATION, "samples_per_round": TORUS9_OPTIMIZER_STEPS_PER_ITERATION * TORUS9_BATCH_SIZE},
    })
    arena_starts, arena_manifest = frozen_arena_starts(Path(args.stable_run), output=root / "evaluation" / "starts.jsonl")
    if args.smoke:
        return {"status": "SMOKE_PASS", "run_id": args.run_id, "komi": TORUS9_KOMI, "m8": str(m8_path), "arena_starts": len(arena_starts), "profile_fingerprint": profile_fp}

    started = time.perf_counter()
    if args.reuse_corpus:
        d9_manifest, d9_rows = load_existing_corpus(root, label="D9", seed=D9_SEED, m8_hash=str(m8_metadata["model_hash"]))
        d10_manifest, d10_rows = load_existing_corpus(root, label="D10", seed=D10_SEED, m8_hash=str(m8_metadata["model_hash"]))
    else:
        d9_manifest, d9_rows = generate_corpus(root, label="D9", seed=D9_SEED, m8=m8, m8_path=m8_path, m8_metadata=m8_metadata, profile_fp=profile_fp, code=code)
        d10_manifest, d10_rows = generate_corpus(root, label="D10", seed=D10_SEED, m8=m8, m8_path=m8_path, m8_metadata=m8_metadata, profile_fp=profile_fp, code=code)
    if d9_manifest["corpus_fingerprint"] == d10_manifest["corpus_fingerprint"]:
        raise ValueError("D9 and D10 must be independent fixed corpora")
    if fingerprint(d9_rows) == fingerprint(d10_rows):
        raise ValueError("D9 and D10 replay rows unexpectedly match")

    arms: dict[str, dict[str, object]] = {}
    initial_hashes: dict[str, str] = {}
    round_reports: dict[str, list[dict[str, object]]] = {}
    for arm, enabled in (("A", False), ("B", True)):
        model = make_auxiliary_from_m8(m8, seed=OWNERSHIP_HEAD_SEED)
        trainer = Torus9OwnershipTrainer(model, ownership_loss_enabled=enabled)
        restore_m8_optimizer(m8_path, model, trainer, source_parameter_count=len(tuple(m8.parameters())))
        initial_hashes[arm] = shared_parameter_hash(model)
        if initial_hashes[arm] != shared_parameter_hash(m8):
            raise ValueError(f"Initial shared model identity drift in arm {arm}")
        replay = Torus9RollingReplay(generations=TORUS9_ROLLING_GENERATIONS, maximum_positions=TORUS9_MAX_REPLAY_POSITIONS)
        first, m9_info = train_round(root=root, arm=arm, model=model, trainer=trainer, replay=replay, generation=9, rows=d9_rows, source_label="D9", source_model_hash=str(m8_metadata["model_hash"]), model_seed=OWNERSHIP_HEAD_SEED, profile_fp=profile_fp, code=code)
        second, m10_info = train_round(root=root, arm=arm, model=model, trainer=trainer, replay=replay, generation=10, rows=d10_rows, source_label="D10", source_model_hash=str(m8_metadata["model_hash"]), model_seed=OWNERSHIP_HEAD_SEED, profile_fp=profile_fp, code=code)
        round_reports[arm] = [first, second]
        arms[arm] = {"M9": m9_info, "M10": m10_info, "rounds": round_reports[arm], "initial_shared_parameter_hash": initial_hashes[arm]}
    if initial_hashes["A"] != initial_hashes["B"]:
        raise ValueError("A/B initial shared parameters are not identical")
    if round_reports["A"][0]["replay"]["fresh_positions"] != round_reports["B"][0]["replay"]["fresh_positions"] or round_reports["A"][1]["replay"]["fresh_positions"] != round_reports["B"][1]["replay"]["fresh_positions"]:
        raise ValueError("A/B common replay generation sizes drifted")

    b_path = Path(str(arms["B"]["M10"]["path"]))
    a_path = Path(str(arms["A"]["M10"]["path"]))
    if not args.batched:
        raise ValueError("The controlled Torus 9×9 A/B Arena requires --batched")
    arena = run_torus9_batched_arena(
        run_id=root.name,
        comparison="M10-B-vs-M10-A",
        candidate_path=b_path,
        reference_path=a_path,
        candidate_label="M10-B",
        reference_label="M10-A",
        starts=arena_starts,
        master_seed=2026091416,
        output_dir=root / "arena" / "M10-B-vs-M10-A",
        workers=TORUS9_WORKERS,
        arena_batch_size=ARENA_BATCH_SIZE,
        inference_batch_wait_ms=ARENA_INFERENCE_BATCH_WAIT_MS,
        device="cpu",
    )
    mean_batch_rows = float(arena.get("mean_inference_batch_rows", 0.0))
    if mean_batch_rows < 16.0:
        raise RuntimeError(f"Arena performance-degraded: mean_inference_batch_rows={mean_batch_rows:.3f} < 16")
    report: dict[str, object] = {
        "status": "PASS" if int(arena["technical_games"]) == 0 else "FAIL",
        "run_id": root.name,
        "run_dir": str(root),
        "branch": branch_name(),
        "source_commit": code.git_commit_sha,
        "source_tree": code.git_tree_sha,
        "source_worktree_clean": code.working_tree_clean,
        "base": "merged PR88 / origin/main",
        "komi": TORUS9_KOMI,
        "m8": {"path": str(m8_path), "model_hash": m8_metadata["model_hash"], "artifact_sha256": file_sha256(m8_path), "metadata": m8_metadata},
        "profile_id": TORUS9_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "corpora": {"D9": d9_manifest, "D10": d10_manifest, "same_source_model": True, "same_rows_reused_by_A_and_B": True},
        "arms": arms,
        "no_m9_arena": True,
        "arena_protocol": {"games": 64, "paired_starts": ARENA_PAIRS, "workers": TORUS9_WORKERS, "batched": True, "arena_batch_size": ARENA_BATCH_SIZE, "inference_batch_wait_ms": ARENA_INFERENCE_BATCH_WAIT_MS, "mean_inference_batch_rows": mean_batch_rows, "max_inference_batch_rows": arena.get("max_inference_batch_rows"), "watchdog": 1000, "simulations": 64, "noise": False, "temperature": 0.0, "fast": False, "komi": TORUS9_KOMI, "technical_fail_closed": True},
        "evaluation": arena_manifest,
        "arena": arena,
        "verdict": verdict(arena),
        "telemetry": {"wall_time_sec": time.perf_counter() - started, "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, "platform": platform.platform(), "workers": TORUS9_WORKERS},
        "training_contract": {"optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": TORUS9_BATCH_SIZE, "steps_per_round": TORUS9_OPTIMIZER_STEPS_PER_ITERATION, "samples_per_round": TORUS9_OPTIMIZER_STEPS_PER_ITERATION * TORUS9_BATCH_SIZE, "m8_step": M8_OPTIMIZER_STEPS, "m10_step": M8_OPTIMIZER_STEPS + 2 * TORUS9_OPTIMIZER_STEPS_PER_ITERATION, "optimizer_semantics_same_in_A_and_B": True},
    }
    write_json(root / "final-report.json", report)
    human_report(root / "final-report.md", report)
    failure_path = root / "failure-report.json"
    if failure_path.is_file():
        failure_path.unlink()
    return report


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--stable-run", type=Path, default=DEFAULT_STABLE_RUN)
    parser.add_argument("--m8", type=Path, default=DEFAULT_M8)
    parser.add_argument("--workers", type=int, default=TORUS9_WORKERS)
    parser.add_argument("--komi", type=float, default=TORUS9_KOMI)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batched", action="store_true", help="run the required coalesced-inference Arena")
    parser.add_argument("--reuse-corpus", action="store_true", help="reuse complete immutable D9/D10 corpora from a prior run")
    args = parser.parse_args(argv)
    try:
        report = run_experiment(args)
    except Exception as exc:
        failure = {"status": "FAIL", "run_id": args.run_id, "komi": args.komi, "error": f"{type(exc).__name__}: {exc}"}
        failure_path = Path(args.run_root) / args.run_id / "failure-report.json"
        write_json(failure_path, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
