#!/usr/bin/env python3
"""Run the controlled Torus9 Dirichlet-alpha then score-head experiments."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import resource
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
    Torus9OwnershipScoreGraphNet,
    Torus9OwnershipTrainer,
    Torus9OwnershipScoreTrainer,
    Torus9RollingReplay,
    Torus9SelfPlayGameRecord,
    Torus9SelfPlaySearchContract,
    run_torus9_batched_arena,
    run_torus9_selfplay_games,
    torus9_build_ownership_replay_samples,
    torus9_build_ownership_score_replay_samples,
    torus9_checkpoint_info,
    torus9_checkpoint_metadata,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    torus9_restore_optimizer_state,
    torus9_save_checkpoint,
    torus9_score_target,
    torus9_state_from_identity,
    validate_torus9_replay_sample,
    write_json,
    write_jsonl,
)
from gocube_golden.torus9_contract import (
    TORUS9_BATCH_SIZE,
    TORUS9_HIDDEN,
    TORUS9_BLOCKS,
    TORUS9_KOMI,
    TORUS9_MAX_REPLAY_POSITIONS,
    TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    TORUS9_PROFILE_ID,
    TORUS9_ROLLING_GENERATIONS,
    TORUS9_WORKERS,
    load_torus9_profile,
    profile_fingerprint,
)
from gocube_golden.rules import apply_action
from gocube_golden.state import initial_state


DEFAULT_RUN_ID = "torus9-alpha-score-ab-20260913-v1"
DEFAULT_RUN_ROOT = ROOT / "runs" / "torus9-alpha-score-ab"
DEFAULT_PARENT_RUN = ROOT / "runs" / "torus9-ownership-ab" / "torus9-wdl-ownership-ab-20260913-v1"
DEFAULT_M10_B = DEFAULT_PARENT_RUN / "arms" / "B" / "checkpoints" / "M10-B.pt"
DEFAULT_STARTS = DEFAULT_PARENT_RUN / "evaluation" / "starts.jsonl"
CORPUS_GAMES = 64
ARENA_PAIRS = 32
ARENA_BATCH_SIZE = 8
ARENA_WAIT_MS = 6.0
M10_STEP = 800
M10_SAMPLES = 51200
TRAINING_SEEDS = {
    "alpha-A": (2026091511, 2026091512),
    "alpha-B": (2026091521, 2026091522),
    "score-A": (2026091531, 2026091532),
    "score-B": (2026091541, 2026091542),
}
CORPUS_SEEDS = {
    "D11-A": 2026091501,
    "D12-A": 2026091502,
    "D11-B": 2026091503,
    "D12-B": 2026091504,
    "D13": 2026091505,
    "D14": 2026091506,
}
ARENA_SEEDS = {"alpha": 2026091571, "score": 2026091572}
ALPHAS = {"A": 0.30, "B": 0.11}


def jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "value"):
        return jsonable(value.value)
    return str(value)


def canonical(value: object) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_checkpoint(path: Path) -> tuple[Torus9GraphNet, dict[str, Any]]:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    model = torus9_model_from_metadata(metadata)
    loaded = torus9_load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]})
    return model.eval(), dict(loaded)


def copy_ownership_model(source: Torus9OwnershipGraphNet) -> Torus9OwnershipGraphNet:
    clone = Torus9OwnershipGraphNet(hidden=TORUS9_HIDDEN, blocks=TORUS9_BLOCKS)
    clone.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    return clone.eval()


def make_score_model(source: Torus9OwnershipGraphNet, *, seed: int) -> Torus9OwnershipScoreGraphNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Torus9OwnershipScoreGraphNet(hidden=TORUS9_HIDDEN, blocks=TORUS9_BLOCKS)
    target = model.state_dict()
    for name, value in source.state_dict().items():
        target[name].copy_(value)
    return model.eval()


def load_frozen_starts(source: Path, output: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(source)
    if len(rows) != 32:
        raise ValueError("Arena corpus must contain exactly 32 paired starts")
    counts = Counter(int(row["prefix_length"]) for row in rows)
    if sorted(counts.values()) != [4] * 8:
        raise ValueError("Arena corpus must contain four starts in each of eight strata")
    corpus_fp = fingerprint(rows)
    for row in rows:
        row["experiment_corpus_fingerprint"] = corpus_fp
    write_jsonl(output, rows)
    write_json(output.with_name("manifest.json"), {
        "contract_id": "torus9-alpha-score-ab-frozen-arena-corpus-v1",
        "source": str(source),
        "starts": len(rows),
        "prefix_lengths": sorted(counts),
        "accepted_per_stratum": 4,
        "color_swapped_pair_semantics": True,
        "fingerprint": corpus_fp,
        "created_before_results": True,
        "komi": TORUS9_KOMI,
    })
    return rows


def audit_records(records: Sequence[Torus9SelfPlayGameRecord], *, expected_hash: str, score: bool = False) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(records) != CORPUS_GAMES:
        raise ValueError(f"Expected {CORPUS_GAMES} self-play games, got {len(records)}")
    if any(record.model_hash != expected_hash for record in records):
        raise ValueError("Self-play source model hash drift")
    if any(record.technical_termination is not None for record in records):
        raise ValueError("Technical self-play games are excluded from training corpora")
    rows: list[dict[str, object]] = []
    for record in records:
        record.validate()
        rows.extend(torus9_build_ownership_score_replay_samples(record) if score else torus9_build_ownership_replay_samples(record))
    for row in rows:
        validate_torus9_replay_sample(row)
    return {
        "games": len(records),
        "valid_games": len(records),
        "technical_games": 0,
        "positions": len(rows),
        "outcomes": dict(Counter(record.formal_result for record in records)),
        "model_hash": expected_hash,
        "technical_excluded_from_training": True,
        "score_targets": score,
    }, rows


def write_corpus(root: Path, label: str, records: Sequence[Torus9SelfPlayGameRecord], rows: Sequence[Mapping[str, object]], audit: Mapping[str, object], *, source: str, seed: int) -> dict[str, object]:
    directory = root / "selfplay" / label
    games = [record.to_dict() for record in records]
    replay = [dict(row) for row in rows]
    write_jsonl(directory / "games.jsonl", games)
    write_jsonl(directory / "replay.jsonl", replay)
    manifest = {
        "run_id": root.name,
        "corpus": label,
        "source_model_label": source,
        "source_model_hash": audit["model_hash"],
        "master_seed": seed,
        "games": len(games),
        "positions": len(replay),
        "technical_games": audit["technical_games"],
        "corpus_fingerprint": fingerprint(games),
        "replay_fingerprint": fingerprint(replay),
        "immutable": True,
        "komi": TORUS9_KOMI,
        "score_targets": audit["score_targets"],
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def generate_corpus(root: Path, *, label: str, model: Torus9GraphNet, checkpoint: Path, source_label: str, seed: int, alpha: float, code, score_targets: bool = False) -> tuple[dict[str, object], list[dict[str, object]]]:
    ids = tuple(f"{label}-{source_label}-game-{index:04d}" for index in range(CORPUS_GAMES))
    contract = Torus9SelfPlaySearchContract(dirichlet_alpha=alpha)
    records = run_torus9_selfplay_games(
        model,
        checkpoint_path=checkpoint,
        run_id=root.name,
        label=source_label,
        artifact=file_sha256(checkpoint),
        master_seed=seed,
        profile_fp=profile_fingerprint(load_torus9_profile()),
        game_ids=ids,
        workers=TORUS9_WORKERS,
        code_identity=code,
        device="cpu",
        contract=contract,
    )
    audit, rows = audit_records(records, expected_hash=model_hash_from_metadata(checkpoint), score=score_targets)
    manifest = write_corpus(root, label, records, rows, audit, source=source_label, seed=seed)
    manifest.update({"dirichlet_alpha": alpha, "selfplay_contract_fingerprint": contract.fingerprint})
    write_json(root / "selfplay" / label / "manifest.json", manifest)
    return manifest, rows


def model_hash_from_metadata(path: Path) -> str:
    return str(json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))["model_hash"])


def load_old_replay(parent: Path, label: str) -> list[dict[str, object]]:
    path = parent / "selfplay" / label / "replay.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Missing immutable parent replay {path}")
    for row in rows:
        validate_torus9_replay_sample(row)
    return rows


def initialize_ownership_training(checkpoint: Path) -> tuple[Torus9OwnershipGraphNet, Torus9OwnershipTrainer, dict[str, Any]]:
    model = Torus9OwnershipGraphNet(hidden=TORUS9_HIDDEN, blocks=TORUS9_BLOCKS)
    trainer = Torus9OwnershipTrainer(model, ownership_loss_enabled=True)
    metadata = torus9_load_checkpoint(checkpoint, model=model, optimizer=trainer.optimizer, expected={"model_hash": model_hash_from_metadata(checkpoint)})
    trainer.update_count = int(metadata["optimizer_updates"])
    trainer.samples_consumed = int(metadata["train_samples_consumed"])
    trainer.assert_optimizer_continuity()
    return model.eval(), trainer, dict(metadata)


def train_alpha_arm(root: Path, arm: str, m10: Path, parent_replay: Mapping[str, Sequence[Mapping[str, object]]], alpha: float, code) -> dict[str, Any]:
    model, trainer, source_metadata = initialize_ownership_training(m10)
    replay = Torus9RollingReplay(generations=TORUS9_ROLLING_GENERATIONS, maximum_positions=TORUS9_MAX_REPLAY_POSITIONS)
    for generation, label in ((9, "D9"), (10, "D10")):
        replay.append_generation(generation, parent_replay[label])
    reports: dict[str, Any] = {"alpha": alpha, "arm": arm, "source": {"label": "M10-B", "model_hash": source_metadata["model_hash"], "optimizer_updates": source_metadata["optimizer_updates"]}}
    checkpoint = m10
    for index, generation in enumerate((11, 12)):
        label = f"D{generation}-{arm}"
        manifest, rows = generate_corpus(root, label=label, model=model, checkpoint=checkpoint, source_label=checkpoint.stem, seed=CORPUS_SEEDS[label], alpha=alpha, code=code)
        write_json(root / "arms" / arm / "corpora" / f"{label}.json", manifest)
        replay_metrics = replay.append_generation(generation, rows)
        train = trainer.train_fixed_budget(replay.rows, seed=TRAINING_SEEDS[f"alpha-{arm}"][index])
        checkpoint_label = f"M{generation}-{arm}"
        path = root / "arms" / arm / "checkpoints" / f"{checkpoint_label}.pt"
        metadata = torus9_checkpoint_metadata(
            model=model, run_id=root.name, label=checkpoint_label, parent=checkpoint.stem,
            model_seed=int(source_metadata.get("model_initialization_seed", 0)), code=code,
            profile_fp=profile_fingerprint(load_torus9_profile()), completed_games=640 + (generation - 10) * CORPUS_GAMES,
            replay_positions=len(replay.rows), optimizer_updates=trainer.update_count, samples_consumed=trainer.samples_consumed,
            ownership_loss_enabled=True,
        )
        metadata.update({"ablation_arm": arm, "dirichlet_alpha": alpha, "source_corpus": label, "optimizer_continuation": "M10 step 800"})
        saved = torus9_save_checkpoint(path, model=model, optimizer=trainer.optimizer, metadata=metadata)
        round_report = {"checkpoint": {"path": str(path), "model_hash": saved["model_hash"], "artifact_sha256": file_sha256(path)}, "corpus": manifest, "replay": replay_metrics, "training": train}
        write_json(root / "arms" / arm / f"{checkpoint_label}-training.json", round_report)
        reports[checkpoint_label] = round_report
        checkpoint = path
    return reports


def train_score_arm(root: Path, arm: str, winner_checkpoint: Path, d13: Sequence[Mapping[str, object]], d14: Sequence[Mapping[str, object]], score_enabled: bool, code) -> dict[str, Any]:
    source_model, source_metadata = load_checkpoint(winner_checkpoint)
    if not isinstance(source_model, Torus9OwnershipGraphNet) or isinstance(source_model, Torus9OwnershipScoreGraphNet):
        raise ValueError("M12-WINNER must be the Torus9 WDL+ownership checkpoint")
    if score_enabled:
        model = make_score_model(source_model, seed=2026091581)
        trainer = Torus9OwnershipScoreTrainer(model, score_loss_enabled=True)
        inherited = torus9_restore_optimizer_state(winner_checkpoint, model=model, optimizer=trainer.optimizer, source_parameter_count=len(tuple(source_model.parameters())))
    else:
        model = copy_ownership_model(source_model)
        trainer = Torus9OwnershipTrainer(model, ownership_loss_enabled=True)
        torus9_load_checkpoint(winner_checkpoint, model=model, optimizer=trainer.optimizer, expected={"model_hash": source_metadata["model_hash"]})
        inherited = int(source_metadata["optimizer_updates"])
    trainer.update_count = inherited
    trainer.samples_consumed = int(source_metadata["train_samples_consumed"])
    trainer.assert_optimizer_continuity()
    replay = Torus9RollingReplay(generations=TORUS9_ROLLING_GENERATIONS, maximum_positions=TORUS9_MAX_REPLAY_POSITIONS)
    reports: dict[str, Any] = {"arm": arm, "score_loss_enabled": score_enabled, "source": {"label": "M12-WINNER", "model_hash": source_metadata["model_hash"], "optimizer_updates": inherited}}
    checkpoint = winner_checkpoint
    for generation, rows in ((13, d13), (14, d14)):
        replay_metrics = replay.append_generation(generation, rows)
        train = trainer.train_fixed_budget(replay.rows, seed=TRAINING_SEEDS[f"score-{arm}"][generation - 13])
        checkpoint_label = f"M{generation}-{arm}"
        path = root / "score" / arm / "checkpoints" / f"{checkpoint_label}.pt"
        metadata = torus9_checkpoint_metadata(
            model=model, run_id=root.name, label=checkpoint_label, parent=checkpoint.stem,
            model_seed=int(source_metadata.get("model_initialization_seed", 0)), code=code,
            profile_fp=profile_fingerprint(load_torus9_profile()), completed_games=768 + (generation - 12) * CORPUS_GAMES,
            replay_positions=len(replay.rows), optimizer_updates=trainer.update_count, samples_consumed=trainer.samples_consumed,
            ownership_loss_enabled=True, score_loss_enabled=score_enabled,
        )
        metadata.update({"ablation_arm": arm, "score_target_contract_id": "golden-score-final-margin-side-to-move-v1", "score_target_source": "golden-referee-final-state-v1", "score_target_perspective": "side-to-move", "optimizer_continuation": "M12 step 800; score params initialized at step 800"})
        saved = torus9_save_checkpoint(path, model=model, optimizer=trainer.optimizer, metadata=metadata)
        round_report = {"checkpoint": {"path": str(path), "model_hash": saved["model_hash"], "artifact_sha256": file_sha256(path)}, "replay": replay_metrics, "training": train}
        write_json(root / "score" / arm / f"{checkpoint_label}-training.json", round_report)
        reports[checkpoint_label] = round_report
        checkpoint = path
    return reports


def verdict(summary: Mapping[str, object]) -> str:
    if int(summary.get("technical_games", 0)) or int(summary.get("pairs_valid", 0)) != ARENA_PAIRS:
        return "INCONCLUSIVE"
    if float(summary.get("mean_inference_batch_rows", 16.0)) < 16.0:
        return "INCONCLUSIVE_PERFORMANCE_DEGRADED"
    wins, losses, _ = (int(value) for value in summary["W/L/D"])  # type: ignore[index]
    if wins >= 42 and wins > losses:
        return "B WINNER"
    if losses >= 42 and losses > wins:
        return "A WINNER"
    return "INCONCLUSIVE"


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.workers != TORUS9_WORKERS:
        raise ValueError("Controlled Torus9 experiments are pinned to workers=16")
    if args.komi != TORUS9_KOMI:
        raise ValueError("Komi must remain 0.5")
    profile = load_torus9_profile()
    code = capture_code_identity()
    run_dir = Path(args.run_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        live = initial_state(topology=__import__("gocube_golden").TORUS_9X9, komi=0.5)
        final = apply_action(apply_action(live, "PASS").after, "PASS").after
        if torus9_score_target(final, "BLACK") != -0.5 or torus9_score_target(final, "WHITE") != 0.5:
            raise ValueError("Score target sign smoke failed")
        report = {"status": "SMOKE_PASS", "source_commit": code.git_commit_sha, "branch": current_branch(), "score_target": {"black": -0.5, "white": 0.5}}
        write_json(run_dir / "final-report.json", report)
        return report

    started = time.perf_counter()
    if not args.m10_b.is_file():
        raise FileNotFoundError(f"M10-B checkpoint is missing: {args.m10_b}")
    m10_model, m10_metadata = load_checkpoint(args.m10_b)
    if not isinstance(m10_model, Torus9OwnershipGraphNet) or isinstance(m10_model, Torus9OwnershipScoreGraphNet) or int(m10_metadata["optimizer_updates"]) != M10_STEP:
        raise ValueError("Starting checkpoint is not immutable M10-B WDL+ownership at optimizer step 800")
    starts = load_frozen_starts(args.starts, run_dir / "evaluation" / "starts.jsonl")
    old_parent_replay = {label: load_old_replay(DEFAULT_PARENT_RUN, label) for label in ("D9", "D10")}
    alpha_reports = {}
    for arm, alpha in ALPHAS.items():
        alpha_reports[arm] = train_alpha_arm(run_dir, arm, args.m10_b, old_parent_replay, alpha, code)
    alpha_arena = run_torus9_batched_arena(
        run_id=run_dir.name, comparison="M12-B-alpha-0.11-vs-M12-A-alpha-0.30",
        candidate_path=Path(alpha_reports["B"]["M12-B"]["checkpoint"]["path"]), reference_path=Path(alpha_reports["A"]["M12-A"]["checkpoint"]["path"]),
        candidate_label="M12-B(alpha=0.11)", reference_label="M12-A(alpha=0.30)", starts=starts,
        master_seed=ARENA_SEEDS["alpha"], output_dir=run_dir / "arena" / "M12-B-vs-M12-A", workers=TORUS9_WORKERS,
        arena_batch_size=ARENA_BATCH_SIZE, inference_batch_wait_ms=ARENA_WAIT_MS, device="cpu",
    )
    alpha_verdict = verdict(alpha_arena)
    selected_arm = "B" if alpha_verdict == "B WINNER" else "A"
    selected_alpha = ALPHAS[selected_arm]
    winner_checkpoint = Path(alpha_reports[selected_arm][f"M12-{selected_arm}"]["checkpoint"]["path"])
    winner_model, winner_metadata = load_checkpoint(winner_checkpoint)
    if not isinstance(winner_model, Torus9OwnershipGraphNet) or isinstance(winner_model, Torus9OwnershipScoreGraphNet):
        raise ValueError("Selected M12 winner is not WDL+ownership")

    d13_manifest, d13 = generate_corpus(run_dir, label="D13", model=winner_model, checkpoint=winner_checkpoint, source_label="M12-WINNER", seed=CORPUS_SEEDS["D13"], alpha=selected_alpha, code=code, score_targets=True)
    d14_manifest, d14 = generate_corpus(run_dir, label="D14", model=winner_model, checkpoint=winner_checkpoint, source_label="M12-WINNER", seed=CORPUS_SEEDS["D14"], alpha=selected_alpha, code=code, score_targets=True)
    write_json(run_dir / "score" / "common-corpora.json", {"D13": d13_manifest, "D14": d14_manifest, "same_immutable_rows_used_by_A_and_B": True})
    score_reports = {
        "A": train_score_arm(run_dir, "A", winner_checkpoint, d13, d14, False, code),
        "B": train_score_arm(run_dir, "B", winner_checkpoint, d13, d14, True, code),
    }
    score_arena = run_torus9_batched_arena(
        run_id=run_dir.name, comparison="M14-B-plus-score-vs-M14-A-no-score",
        candidate_path=Path(score_reports["B"]["M14-B"]["checkpoint"]["path"]), reference_path=Path(score_reports["A"]["M14-A"]["checkpoint"]["path"]),
        candidate_label="M14-B(+score)", reference_label="M14-A(no score)", starts=starts,
        master_seed=ARENA_SEEDS["score"], output_dir=run_dir / "arena" / "M14-B-vs-M14-A", workers=TORUS9_WORKERS,
        arena_batch_size=ARENA_BATCH_SIZE, inference_batch_wait_ms=ARENA_WAIT_MS, device="cpu",
    )
    report = {
        "status": "PASS",
        "experiment_id": "torus9-alpha-score-ab-v1",
        "source_commit": code.git_commit_sha,
        "branch": current_branch(),
        "profile_id": TORUS9_PROFILE_ID,
        "profile_fingerprint": profile_fingerprint(profile),
        "starting_checkpoint": {"label": "M10-B", "path": str(args.m10_b), "model_hash": m10_metadata["model_hash"], "optimizer_updates": M10_STEP},
        "experiment_1_dirichlet_alpha": {"arms": alpha_reports, "arena": alpha_arena, "verdict": alpha_verdict, "selected_arm": selected_arm, "selected_alpha": selected_alpha, "M12-WINNER": str(winner_checkpoint)},
        "experiment_2_score": {"common_corpora": {"D13": d13_manifest, "D14": d14_manifest, "same_immutable_rows_used_by_A_and_B": True}, "arms": score_reports, "arena": score_arena, "verdict": verdict(score_arena)},
        "contracts": {"no_intermediate_M11_M13_arena": True, "no_adaptive_extensions": True, "technical_games_never_wdl": True, "score_target": "official final margin_black; side-to-move sign", "score_target_normalization": 81.5},
        "telemetry": {"wall_time_sec": time.perf_counter() - started, "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, "platform": platform.platform(), "workers": TORUS9_WORKERS},
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "final-report.json", report)
    write_human_report(run_dir / "final-report.md", report)
    return report


def current_branch() -> str:
    import subprocess
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def write_human_report(path: Path, report: Mapping[str, object]) -> None:
    alpha = report["experiment_1_dirichlet_alpha"]
    score = report["experiment_2_score"]
    a = alpha["arena"]
    s = score["arena"]
    path.write_text("\n".join([
        "# Torus9 sequential controlled A/B experiments", "",
        f"Status: **{report['status']}**", f"Source commit: `{report['source_commit']}`", "Komi: **0.5**", "",
        "1. `M12-B(α=0.11) vs M12-A(α=0.30) = " + " / ".join(str(x) for x in a["W/L/D"]) + "`",
        f"   Selected alpha: **{alpha['selected_alpha']}** ({alpha['verdict']})",
        "2. `M14-B(+score) vs M14-A(no score) = " + " / ".join(str(x) for x in s["W/L/D"]) + "`",
        f"   Score verdict: **{score['verdict']}**", "",
        f"Arena performance: alpha mean batch rows {a.get('mean_inference_batch_rows'):.3f}; score {s.get('mean_inference_batch_rows'):.3f}.",
        f"Machine-readable evidence: `{report['run_dir']}/final-report.json`", "",
    ]) + "\n", encoding="utf-8")


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--m10-b", type=Path, default=DEFAULT_M10_B)
    parser.add_argument("--starts", type=Path, default=DEFAULT_STARTS)
    parser.add_argument("--workers", type=int, default=TORUS9_WORKERS)
    parser.add_argument("--komi", type=float, default=TORUS9_KOMI)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_experiment(args)
    except Exception as exc:
        run_dir = Path(args.run_root) / args.run_id
        failure = {"status": "FAIL", "run_id": args.run_id, "error": f"{type(exc).__name__}: {exc}"}
        write_json(run_dir / "failure-report.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
