"""Operator-owned fixed-mixture transfer diagnostic for frozen policy targets.

Only orchestration lives here: model updates and loss semantics remain owned by
the existing ordinary trainer.  The diagnostic overrides its batch sampler to
provide a precomputed, paired 56/8 schedule to both arms.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SNAPSHOTS = (0, 128, 256, 640)
BOOTSTRAP_REPLICATES = 5000
SAMPLE_SEED = 2026101216
BOOTSTRAP_SEED = 2026101217


def _sha(path: Path) -> str:
    from ..provenance import file_sha256
    return file_sha256(path)


def validate_operator_payload(value: object, *, training: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("policy_target_mix must be an object")
    allowed = {"offline_replay", "pilot_root", "pilot_manifest_sha256"}
    if set(value) != allowed:
        raise ValueError("policy_target_mix requires offline_replay, pilot_root, pilot_manifest_sha256 only")
    selectors = value["offline_replay"]
    if not isinstance(selectors, list) or len(selectors) != 1:
        raise ValueError("policy_target_mix requires exactly the D270 source checkpoint")
    from .operator_job import _selector
    selectors = [_selector(x, "policy_target_mix.offline_replay checkpoint") for x in selectors]
    if training["iterations"] != 0 or training["learning_rate"] != 0.000025 \
            or training["batch_size"] != 64 or training["updates_per_iteration"] != 640 \
            or training["gradient_clip"] != 8.0 or training["replay_generations"] != 5:
        raise ValueError("policy_target_mix requires the frozen LR=2.5e-5, batch=64, clip=8, 640-update, replay=5 protocol")
    root = Path(str(value["pilot_root"])).expanduser().resolve()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("policy_target_mix.pilot_root must be an existing absolute artifact directory")
    expected = str(value["pilot_manifest_sha256"])
    if not expected.startswith("sha256:") or len(expected) != 71:
        raise ValueError("policy_target_mix.pilot_manifest_sha256 must be a SHA-256 identity")
    return {"offline_replay": selectors, "pilot_root": str(root),
            "pilot_manifest_sha256": expected, "mixture": {"main": 56, "panel": 8},
            "snapshots": list(SNAPSHOTS), "bootstrap_replicates": BOOTSTRAP_REPLICATES}


def resolve_source_window(selector: str, *, resolver) -> dict[str, Any]:
    """Read the registered D270 rolling manifest without inheriting its weights."""
    from .operator_job import resolve_checkpoint
    node = resolve_checkpoint(selector, resolver=resolver)
    if node.generation != 270:
        raise ValueError("policy_target_mix replay source must be registered D270")
    root = node.owner_root.resolve()
    provenance = json.loads((root / "metadata/provenance-v2" / (node.checkpoint_id + ".json")).read_text())
    artifacts = {}
    for key in ("fresh_replay", "rolling_replay"):
        identity = provenance["artifact_identities"][key]
        path = (root / identity["path"]).resolve()
        if not path.is_relative_to(root) or _sha(path) != identity["sha256"]:
            raise ValueError("D270 replay manifest failed source identity verification")
        artifacts[key] = {"path": str(path), "sha256": identity["sha256"]}
    fresh = json.loads(Path(artifacts["fresh_replay"]["path"]).read_text())
    buckets = [json.loads(line) for line in Path(artifacts["rolling_replay"]["path"]).read_text().splitlines()]
    if not buckets or buckets[-1] != fresh or [int(row["generation"]) for row in buckets] != list(range(266, 271)):
        raise ValueError("Registered D270 rolling replay is not exactly D266-D270")
    for bucket in buckets:
        for shard in bucket["shards"]:
            if _sha(Path(shard["path"])) != shard["sha"]:
                raise ValueError("D270 historical replay shard SHA mismatch")
    return {"generation": 270, "source_checkpoint": node.ref.to_dict(), **artifacts,
            "buckets": buckets, "fresh_bucket": fresh}


def _pilot_inputs(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(spec["pilot_root"])
    manifest_path = root / "manifest.json"
    if _sha(manifest_path) != spec["pilot_manifest_sha256"]:
        raise ValueError("r2 pilot manifest SHA changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "gocube-d270-policy-target-pilot-v1" or manifest.get("status") != "COMPLETED":
        raise ValueError("policy_target_mix requires a completed r2 pilot manifest")
    required = {
        "split": "split-and-sample.json", "train": "targets/train-targets.jsonl",
        "validation": "targets/validation-targets.jsonl", "test": "targets/test-targets.jsonl",
    }
    paths = {}
    for key, relative in required.items():
        path = root / relative
        identity = manifest.get("outputs", {}).get(relative)
        if not isinstance(identity, str) or _sha(path) != identity:
            raise ValueError(f"r2 input artifact SHA mismatch: {relative}")
        paths[key] = path
    split = json.loads(paths["split"].read_text())
    def read_rows(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    target_rows = {key: read_rows(paths[key]) for key in ("train", "validation", "test")}
    positions = split["positions"]
    for key, expected in (("train", 512), ("validation", 128), ("test", 128)):
        if len(positions[key]) != expected or len(target_rows[key]) != expected:
            raise ValueError(f"r2 {key} panel does not have the frozen {expected} rows")
        if len({(r["game_id"], int(r["ply"])) for r in target_rows[key]}) != expected:
            raise ValueError(f"r2 {key} targets contain duplicate positions")
    whole = split["whole_game_split"]
    heldout = set(whole["validation"]) | set(whole["test"])
    if heldout & set(whole["train"]):
        raise ValueError("r2 whole-game splits overlap")
    for key in ("train", "validation", "test"):
        if not {str(row["game_id"]) for row in positions[key]}.issubset(set(whole[key])):
            raise ValueError(f"r2 {key} panel includes a game outside its frozen whole-game split")
    return {"root": root, "manifest": manifest, "manifest_sha256": spec["pilot_manifest_sha256"],
            "split": split, "positions": positions, "targets": target_rows,
            "heldout_games": heldout, "paths": paths}


def check_artifacts(spec: Mapping[str, Any], *, offline: Sequence[Mapping[str, Any]],
                    parent_path: str | Path, training_seed: int) -> dict[str, Any]:
    pilot = _pilot_inputs(spec)
    if len(offline) != 1 or offline[0]["generation"] != 270:
        raise ValueError("policy_target_mix must use exactly D270, whose historical rolling window is D266-D270")
    if len(offline[0]["buckets"]) != 5 or [int(b["generation"]) for b in offline[0]["buckets"]] != list(range(266, 271)):
        raise ValueError("D270 offline input does not resolve to the D266-D270 rolling window")
    if not Path(parent_path).is_file():
        raise ValueError("policy_target_mix parent checkpoint is unavailable")
    parent_meta = json.loads(Path(parent_path).with_suffix(".metadata.json").read_text())
    from ..torus9_five_channel_training import load_replay
    from ..torus9_five_channel_training import OrdinaryTrainer
    parent_trainer = OrdinaryTrainer(parent_path, learning_rate=2.5e-5, seed=training_seed,
                                     gradient_clip=8.0, batch_size=64, device="cpu")
    if parent_trainer.update != int(parent_meta.get("ordinary_update", -1)):
        raise ValueError("parent optimizer update clock differs from its metadata")
    del parent_trainer
    games = load_replay(offline[0]["buckets"], split="train")
    game_ids = {str(game["game_id"]) for game in games}
    heldout = pilot["heldout_games"]
    if not heldout.issubset(game_ids):
        raise ValueError("r2 validation/test games are not all present in the D266-D270 replay window")
    train_positions = {(str(row["game_id"]), int(row["eligible_index"]))
                       for row in pilot["positions"]["train"]}
    if len(train_positions) != 512:
        raise ValueError("r2 train panel does not identify 512 unique replay rows")
    for split in ("train", "validation", "test"):
        _flat_panel(games, pilot["positions"][split], pilot["targets"][split], updated=True)
    replay_rows = _flatten(games)
    filtered_rows = [row for row in replay_rows if row[2] not in heldout and (row[2], row[3]) not in train_positions]
    filtered = [dict(_target_row(g, i, {}, None), original_pi=g["pi"][i].clone(), new_pi=g["pi"][i].clone())
                for g, i, gid, _ in filtered_rows]
    if any(row[2] in heldout for row in filtered_rows):
        raise ValueError("r2 validation/test game exclusion failed")
    if len(filtered) < 512:
        raise ValueError("filtered main replay is too small for the fixed 512-position diagnostic panel")
    panel = _flat_panel(games, pilot["positions"]["train"], pilot["targets"]["train"], updated=False)
    parent_update = int(parent_meta.get("ordinary_update", -1))
    if parent_update < 0:
        raise ValueError("policy_target_mix parent lacks an ordinary optimizer update clock")
    schedule, per_update, order_hash = _schedule(filtered, panel, seed=training_seed,
                                                first_update=parent_update, updates=640)
    draw_counts = {(row["game_id"], row["index"]): 0 for row in panel}
    for batch in schedule:
        for source, row in batch:
            if source == "panel":
                draw_counts[(row["game_id"], row["index"])] += 1
    return {
        "parent_checkpoint_sha256": _sha(Path(parent_path)),
        "parent_model_hash": parent_meta.get("model_hash"),
        "parent_ordinary_update": parent_meta.get("ordinary_update"),
        "full_adam_state_verified": True,
        "offline_generations": [int(b["generation"]) for b in offline[0]["buckets"]],
        "pilot_manifest_sha256": pilot["manifest_sha256"],
        "target_artifacts": {k: _sha(v) for k, v in pilot["paths"].items()},
        "train_panel_positions": 512, "validation_positions": 128, "test_positions": 128,
        "main_replay_positions_before_filter": len(replay_rows),
        "main_replay_positions_after_filter": len(filtered_rows),
        "heldout_game_exclusion_verified": True,
        "special_train_rows_excluded_from_main_replay": len(train_positions),
        "mixture_per_update": {"main_replay": 56, "r2_train_panel": 8},
        "sample_order_sha256": order_hash,
        "panel_draws": sum(draw_counts.values()),
        "panel_position_draws": {"mean": statistics.fmean(draw_counts.values()),
                                  "min": min(draw_counts.values()), "max": max(draw_counts.values()),
                                  "unique_positions": len(draw_counts)},
        "heldout_game_ids": len(pilot["heldout_games"]), "training_seed": int(training_seed),
        "parent_ordinary_update": parent_update,
        "snapshots": list(SNAPSHOTS), "mixture": {"main": 56, "panel": 8},
        "planned_updates": 640, "planned_samples": 40960,
        "test_evaluation": "parent and endpoint 640 only",
    }


def _flatten(games: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], int, str, int]]:
    rows = []
    for game in games:
        for index in range(len(game["score"])):
            rows.append((game, index, str(game["game_id"]), index))
    return rows


def _target_row(game: Mapping[str, Any], index: int, data: Mapping[str, Any], policy_key: str | None = None) -> dict[str, Any]:
    result = {key: game[key][index] for key in ("observation", "pi", "z", "ownership", "score", "legal")}
    result["game_id"] = str(game["game_id"])
    result["index"] = int(index)
    if policy_key is not None:
        import torch
        result["pi"] = torch.tensor(data[policy_key], dtype=torch.float32)
    return result


def _flat_panel(games: Sequence[Mapping[str, Any]], position_manifest: Sequence[Mapping[str, Any]],
                target_rows: Sequence[Mapping[str, Any]], *, updated: bool) -> list[dict[str, Any]]:
    if len(position_manifest) != len(target_rows):
        raise ValueError("r2 position and policy target row counts differ")
    by_game = {str(g["game_id"]): g for g in games}
    import torch
    rows = []
    for pos, target in zip(position_manifest, target_rows):
        game_id = str(pos["game_id"])
        index = int(pos["eligible_index"])
        if str(target.get("game_id")) != game_id or int(target.get("ply", -1)) != int(pos["ply"]):
            raise ValueError(f"r2 target file row identity mismatch: {game_id}:{pos.get('ply')}")
        game = by_game.get(game_id)
        if game is None or not 0 <= index < len(game["score"]):
            raise ValueError(f"r2 selected row is absent from D270 replay: {game_id}:{index}")
        if not torch.allclose(game["pi"][index].float(), torch.tensor(target["original_pi"], dtype=torch.float32), atol=1e-6, rtol=0):
            raise ValueError(f"r2 original policy target mismatch: {game_id}:{index}")
        row = _target_row(game, index, target, "new_pi" if updated else None)
        row["original_pi"] = game["pi"][index].clone()
        row["new_pi"] = torch.tensor(target["new_pi"], dtype=torch.float32)
        rows.append(row)
    return rows


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _schedule(main_rows: Sequence[dict[str, Any]], special_rows: Sequence[dict[str, Any]], *,
              seed: int, first_update: int, updates: int):
    schedule, update_hashes, all_ids = [], [], []
    for offset in range(updates):
        rng = random.Random(seed + first_update + offset + 1)
        batch = [("main", main_rows[rng.randrange(len(main_rows))]) for _ in range(56)]
        batch.extend(("panel", special_rows[rng.randrange(len(special_rows))]) for _ in range(8))
        rng.shuffle(batch)
        identities = [(source, row["game_id"], row["index"]) for source, row in batch]
        digest = "sha256:" + hashlib.sha256(_canonical(identities)).hexdigest()
        schedule.append(batch)
        update_hashes.append(digest)
        all_ids.extend(identities)
    return schedule, update_hashes, "sha256:" + hashlib.sha256(_canonical(all_ids)).hexdigest()


def _evaluate(model, rows: Sequence[Mapping[str, Any]], *, device: str, batch_size: int = 64) -> tuple[dict[str, float], dict[str, list[float]]]:
    import torch
    import torch.nn.functional as F
    from ..torus9_adaptation import AdaptationModel
    model = model.to(device).eval()
    names = ("policy_ce_original", "policy_kl_original", "policy_ce_new", "policy_kl_new",
             "wdl_ce", "wdl_brier", "ownership_ce", "score_mse_normalized",
             "score_mse_points", "score_mae_points")
    per_row = {key: [] for key in names}
    games = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            obs = torch.stack([r["observation"] for r in batch]).to(device)
            pi_logits, wdl_logits, own_logits, score_hat = model.forward_auxiliary(obs)
            log_pi = F.log_softmax(pi_logits, dim=-1)
            wdl_log = F.log_softmax(wdl_logits, dim=-1)
            wdl_prob = wdl_log.exp()
            z = torch.stack([r["z"] for r in batch]).to(device)
            own = torch.stack([r["ownership"] for r in batch]).to(device).long()
            score = torch.stack([r["score"] for r in batch]).to(device).reshape(-1)
            own_ce = F.cross_entropy(own_logits.reshape(-1, 3), own.reshape(-1), reduction="none").reshape(len(batch), -1).mean(-1)
            score_delta = score_hat.reshape(-1) - score
            values = {
                "wdl_ce": -(z * wdl_log).sum(-1),
                "wdl_brier": ((wdl_prob - z) ** 2).sum(-1),
                "ownership_ce": own_ce,
                "score_mse_normalized": score_delta ** 2,
                "score_mse_points": score_delta.square() * (81.5 ** 2),
                "score_mae_points": score_delta.abs() * 81.5,
            }
            for key in ("original", "new"):
                target = torch.stack([r[key + "_pi"] for r in batch]).to(device)
                target = target / target.sum(-1, keepdim=True)
                values["policy_ce_" + key] = -(target * log_pi).sum(-1)
                values["policy_kl_" + key] = (target * (target.clamp_min(1e-30).log() - log_pi)).sum(-1)
            for i, row in enumerate(batch):
                games.append(str(row["game_id"]))
                for key in names:
                    per_row[key].append(float(values[key][i].cpu()))
    return {key: statistics.fmean(values) for key, values in per_row.items()}, {"game_id": games, **per_row}


def _cluster_bootstrap(left: Mapping[str, Any], right: Mapping[str, Any], metric: str, *, seed: int) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for game, a, b in zip(left["game_id"], left[metric], right[metric]):
        groups.setdefault(game, []).append(float(a) - float(b))
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("paired bootstrap needs at least two games")
    per_game = [statistics.fmean(groups[key]) for key in keys]
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(per_game[rng.randrange(len(per_game))] for _ in per_game)
                       for _ in range(BOOTSTRAP_REPLICATES))
    def q(p):
        pos = (len(estimates) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return estimates[lo] + (estimates[hi] - estimates[lo]) * (pos - lo)
    return {"mean": statistics.fmean(per_game), "ci95": [q(.025), q(.975)],
            "clusters": len(keys), "replicates": BOOTSTRAP_REPLICATES, "seed": seed,
            "method": "paired-game-cluster-bootstrap-v1"}


def run(config: Mapping[str, Any], *, runs_root: str | Path) -> dict[str, Any]:
    from ..artifact_resolver import ArtifactResolver
    from ..provenance import file_sha256
    from ..torus9_five_channel_training import OrdinaryTrainer, load_replay
    from ..process_supervision import atomic_write_json, atomic_write_text
    from ..neural import model_hash
    import torch

    pilot = _pilot_inputs(config["policy_target_mix"])
    offline = config["offline_replay"]
    # Recheck every immutable reference immediately before reading it.
    check_artifacts(config["policy_target_mix"], offline=offline,
                    parent_path=config["parent_path"], training_seed=int(config["effective_config"]["execution"]["training_master_seed"]))
    output = Path(runs_root).resolve() / "torus9" / "diagnostics" / config["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = "cuda"
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("policy_target_mix requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    games = load_replay(offline[0]["buckets"], split="train")
    all_rows = _flatten(games)
    replay_positions_before_filter = len(all_rows)
    heldout = pilot["heldout_games"]
    if {gid for _, _, gid, _ in all_rows} & heldout:
        all_rows = [r for r in all_rows if r[2] not in heldout]
    # Every selected r2 row is removed from the main replay, then sampled only
    # by the dedicated eight-row panel stream.
    selected_keys = {(str(p["game_id"]), int(p["eligible_index"])) for p in pilot["positions"]["train"]}
    main_rows = [dict(_target_row(g, i, {}, None), original_pi=g["pi"][i].clone(), new_pi=g["pi"][i].clone())
                 for g, i, gid, _ in all_rows if gid not in heldout and (gid, i) not in selected_keys]
    panel_b = _flat_panel(games, pilot["positions"]["train"], pilot["targets"]["train"], updated=True)
    panel_a = _flat_panel(games, pilot["positions"]["train"], pilot["targets"]["train"], updated=False)
    validation = _flat_panel(games, pilot["positions"]["validation"], pilot["targets"]["validation"], updated=True)
    test = _flat_panel(games, pilot["positions"]["test"], pilot["targets"]["test"], updated=True)
    if len(selected_keys) != 512 or len(panel_a) != 512 or len(panel_b) != 512 or not main_rows:
        raise ValueError("filtered replay/panel mixture did not resolve to 512 unique rows and a nonempty main replay")
    if {r["game_id"] for r in validation} & {r["game_id"] for r in test}:
        raise ValueError("r2 validation/test game split overlaps")
    rng = random.Random(SAMPLE_SEED)
    fixed_main = [main_rows[i] for i in rng.sample(range(len(main_rows)), 512)]
    seed = int(config["effective_config"]["execution"]["training_master_seed"])
    schedule, update_hashes, total_hash = _schedule(main_rows, panel_a, seed=seed,
        first_update=int(torch.load(config["parent_path"], map_location="cpu", weights_only=False)["ordinary_update"]), updates=640)
    if total_hash != config["preflight"]["sample_order_sha256"]:
        raise ValueError("sample order changed between operator preflight and execution")
    panel_draw_counts = {(r["game_id"], r["index"]): 0 for r in panel_a}
    for batch in schedule:
        for source, row in batch:
            if source == "panel":
                panel_draw_counts[(row["game_id"], row["index"])] += 1
    if sum(panel_draw_counts.values()) != 640 * 8:
        raise ValueError("panel sample draw count differs from the fixed 8/64 mixture")
    schedule_identity = {"sample_order_sha256": total_hash, "per_update_sha256": update_hashes,
                         "A_sample_order_sha256": total_hash, "B_sample_order_sha256": total_hash,
                         "sample_seed": seed, "rows_per_batch": {"main": 56, "panel": 8}, "updates": 640,
                         "main_draws": 640 * 56, "panel_draws": 640 * 8,
                         "panel_position_draws": {"mean": statistics.fmean(panel_draw_counts.values()),
                                                   "min": min(panel_draw_counts.values()),
                                                   "max": max(panel_draw_counts.values()),
                                                   "unique_positions": len(panel_draw_counts)}}
    # For exact pairing and traceability, use a small deterministic fixed panel
    # from the filtered replay in addition to the selected r2 train positions.
    train_rows = {"A": panel_a, "B": panel_b}
    parent = config["parent"]
    src = {"parent_checkpoint_sha256": parent["sha256"],
           "parent_training_seed": seed,
           "offline_buckets": [{"generation": b["generation"], "shards": b["shards"]} for b in offline[0]["buckets"]],
           "pilot_manifest_sha256": pilot["manifest_sha256"],
           "pilot_target_sha256": {k: _sha(v) for k, v in pilot["paths"].items()},
           "excluded_validation_test_game_ids": sorted(heldout),
           "excluded_special_train_rows": [{"game_id": str(p["game_id"]),
                                             "eligible_index": int(p["eligible_index"]),
                                             "ply": int(p["ply"])}
                                            for p in pilot["positions"]["train"]],
           "main_replay_positions_before_filter": replay_positions_before_filter,
           "main_replay_positions_after_exclusions": len(main_rows),
           "panel_rows": len(panel_a), "fixed_filtered_replay_panel": 512}
    report = {"schema": "gocube-policy-target-mixture-v1", "status": "RUNNING", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "parent": parent, "source_identity": src, "sample_order": schedule_identity,
              "protocol": {"learning_rate": 2.5e-5, "batch_size": 64, "gradient_clip": 8.0,
                           "updates": 640, "snapshots": list(SNAPSHOTS), "main_share": 56, "panel_share": 8,
                           "loss_weights": "inherited", "optimizer": "full Adam inherited",
                           "test_policy": "evaluate only parent and fixed endpoint 640 after training; no snapshot selection"},
              "elapsed_seconds": None}
    atomic_write_json(output / "source-and-sample.json", {"source_identity": src, "sample_order": schedule_identity})
    heartbeat_path = output / "runtime" / "heartbeat.json"

    # Save diagnostic snapshots outside active/archive; these files are never
    # registered as production checkpoints.
    def save_snapshot(arm: str, step: int, trainer) -> dict[str, Any]:
        path = output / "checkpoints" / f"{arm}-u{step:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": "gocube-policy-target-mixture-snapshot-v1", "diagnostic_only": True,
                    "arm": arm, "updates_from_parent": step, "parent": parent,
                    "model_state_dict": trainer.model.state_dict(), "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "ordinary_update": trainer.update, "model_hash": model_hash(trainer.model)}, path)
        return {"path": str(path), "sha256": file_sha256(path), "model_hash": model_hash(trainer.model)}

    def eval_set(model, rows):
        return _evaluate(model, rows, device=device)

    def adam_hash(state):
        digest = hashlib.sha256()
        for group in state["param_groups"]:
            digest.update(_canonical({k: v for k, v in group.items() if k != "params"}))
            digest.update(_canonical(group["params"]))
        for key in sorted(state["state"]):
            digest.update(_canonical(int(key)))
            for name, value in sorted(state["state"][key].items()):
                digest.update(name.encode() + b"\0")
                if torch.is_tensor(value):
                    tensor = value.detach().cpu().contiguous()
                    digest.update(str(tensor.dtype).encode() + _canonical(list(tensor.shape)) + tensor.numpy().tobytes())
                else:
                    digest.update(_canonical(value))
        return digest.hexdigest()

    class FixedMixtureTrainer(OrdinaryTrainer):
        def __init__(self, *args, mixture_schedule, **kwargs):
            self.mixture_schedule = mixture_schedule
            super().__init__(*args, **kwargs)
            self.parent_update = self.update

        def batch(self, _games, update):
            rows = self.mixture_schedule[int(update) - self.parent_update - 1]
            return {key: torch.stack([row[key] for row in rows]).to(next(self.model.parameters()).device)
                    for key in ("observation", "pi", "z", "ownership", "score")}

    trainers = {}
    snapshots = {"A": {}, "B": {}}
    evaluation = {"train": {}, "validation": {}, "filtered_replay_panel": {}, "test": {}}
    arm_training_seconds = {"A": 0.0, "B": 0.0}
    for arm in ("A", "B"):
        arm_rows = train_rows[arm]
        by_key = {(r["game_id"], r["index"]): r for r in main_rows}
        by_key.update({(r["game_id"], r["index"]): r for r in arm_rows})
        arm_schedule = [[by_key[(row["game_id"], row["index"])] for _, row in batch]
                        for batch in schedule]
        trainer = FixedMixtureTrainer(config["parent_path"], learning_rate=2.5e-5,
            seed=seed, gradient_clip=8.0, batch_size=64, device=device,
            mixture_schedule=arm_schedule)
        if model_hash(trainer.model) != config["preflight"]["parent_model_hash"]:
            raise ValueError("parent model hash changed between check and launch")
        initial_adam = adam_hash(trainer.optimizer.state_dict())
        if arm == "A":
            base_adam_hash = initial_adam
        elif initial_adam != base_adam_hash:
            raise ValueError("A/B branches did not load byte-identical parent Adam")

        trainers[arm] = trainer
        snapshots[arm][0] = save_snapshot(arm, 0, trainer)
        evaluation["train"][arm] = {"0": eval_set(trainer.model, train_rows[arm])}
        evaluation["validation"][arm] = {"0": eval_set(trainer.model, validation)}
        evaluation["filtered_replay_panel"][arm] = {"0": eval_set(trainer.model, fixed_main)}
    parent_update = trainers["A"].update
    if trainers["A"].update != trainers["B"].update:
        raise ValueError("A/B update clocks differ")
    # The sampler closure references parent_update; it was set after init above.
    for update_index in range(1, 641):
        for arm in ("A", "B"):
            arm_started = time.perf_counter()
            metrics = trainers[arm].step(main_rows)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            arm_training_seconds[arm] += time.perf_counter() - arm_started
            if metrics["update"] != parent_update + update_index:
                raise ValueError("ordinary optimizer clock drift")
        if update_index in SNAPSHOTS[1:]:
            for arm in ("A", "B"):
                trainer = trainers[arm]
                snapshots[arm][update_index] = save_snapshot(arm, update_index, trainer)
                for split, rows in (("train", train_rows[arm]), ("validation", validation), ("filtered_replay_panel", fixed_main)):
                    evaluation[split][arm][str(update_index)] = eval_set(trainer.model, rows)
        atomic_write_json(heartbeat_path, {
            "status": "RUNNING", "update": update_index, "updates_total": 640,
            "elapsed_seconds": time.perf_counter() - started,
            "sample_order_sha256": total_hash, "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    # Test access occurs only now, after the designated terminal update.
    for name, model in (("parent", trainers["A"].model), ("A", trainers["A"].model), ("B", trainers["B"].model)):
        if name == "parent":
            from ..torus9_five_channel_training import OrdinaryTrainer as _Trainer
            parent_trainer = _Trainer(config["parent_path"], learning_rate=2.5e-5, seed=seed,
                                      gradient_clip=8.0, batch_size=64, device=device)
            evaluation["test"][name] = eval_set(parent_trainer.model, test)
            del parent_trainer
        else:
            evaluation["test"][name] = eval_set(model, test)
    del trainers

    test_rows = evaluation["test"]
    metrics = ("policy_kl_new", "policy_ce_new", "policy_kl_original", "policy_ce_original",
               "wdl_ce", "wdl_brier", "ownership_ce", "score_mse_normalized", "score_mae_points")
    validation_contrasts = {}
    parent_validation = evaluation["validation"]["A"]["0"][1]
    for step in SNAPSHOTS:
        arm_a = evaluation["validation"]["A"][str(step)][1]
        arm_b = evaluation["validation"]["B"][str(step)][1]
        validation_contrasts[str(step)] = {
            metric: {
                "A_minus_parent": _cluster_bootstrap(arm_a, parent_validation, metric,
                                                       seed=BOOTSTRAP_SEED + 100 + step + i),
                "B_minus_parent": _cluster_bootstrap(arm_b, parent_validation, metric,
                                                       seed=BOOTSTRAP_SEED + 200 + step + i),
                "B_minus_A": _cluster_bootstrap(arm_b, arm_a, metric,
                                                  seed=BOOTSTRAP_SEED + 300 + step + i),
            }
            for i, metric in enumerate(metrics)
        }
    report["validation"] = {
        "snapshots": list(SNAPSHOTS),
        "model_means": {arm: {step: evaluation["validation"][arm][str(step)][0]
                               for step in SNAPSHOTS} for arm in ("A", "B")},
        "contrasts_by_snapshot": validation_contrasts,
    }
    contrasts = {}
    for metric in metrics:
        contrasts[metric] = {f"{left}_minus_{right}": _cluster_bootstrap(test_rows[left][1], test_rows[right][1], metric,
                                seed=BOOTSTRAP_SEED + i)
                             for i, (left, right) in enumerate((("A", "parent"), ("B", "parent"), ("B", "A")))}
    report["test"] = {"model_means": {name: test_rows[name][0] for name in ("parent", "A", "B")}, "contrasts": contrasts}
    report["snapshots"] = snapshots
    report["evaluation"] = evaluation
    report["training_seconds_by_arm"] = arm_training_seconds
    report["A_adam_start_sha256"] = "sha256:" + base_adam_hash
    report["B_adam_start_sha256"] = "sha256:" + base_adam_hash
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "COMPLETED"
    primary = contrasts["policy_kl_new"]["B_minus_A"]
    b_parent_policy = contrasts["policy_kl_new"]["B_minus_parent"]
    wdl_harm = (contrasts["wdl_ce"]["B_minus_parent"]["ci95"][0] > 0
                and contrasts["wdl_brier"]["B_minus_parent"]["ci95"][0] > 0)
    score_harm = (contrasts["score_mse_normalized"]["B_minus_parent"]["ci95"][0] > 0
                  and contrasts["score_mae_points"]["B_minus_parent"]["ci95"][0] > 0)
    if primary["ci95"][1] < 0 and b_parent_policy["ci95"][1] < 0:
        classification = "do_not_promote_value_harm" if wdl_harm or score_harm else "candidate_with_uncertainty"
    else:
        classification = "do_not_expand_automatically"
    report["decision"] = {
        "classification": classification,
        "primary_test_B_minus_A_policy_kl_new": primary,
        "B_minus_parent_policy_kl_new": b_parent_policy,
        "wdl_harm_rule": "both B-parent WDL CE and Brier 95% CI lower bounds > 0",
        "score_harm_rule": "both B-parent normalized score MSE and score MAE 95% CI lower bounds > 0",
        "interpretation": "Loss transfer is not playing strength. Intervals containing zero are inconclusive and do not establish safety or absence of harm.",
        "arena_included": False, "automatic_promotion": False,
    }
    atomic_write_json(output / "report.json", report)
    atomic_write_json(heartbeat_path, {"status": "COMPLETED", "update": 640, "updates_total": 640,
                                       "elapsed_seconds": report["elapsed_seconds"], "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    lines = ["# Policy-target mixture experiment", "", f"Status: {report['status']}. Elapsed: {report['elapsed_seconds']:.1f} s.",
             "", "Test was evaluated only at the preassigned endpoint 640, including the parent. Negative policy KL deltas favor the left model. Cluster intervals are paired by game, unadjusted across metrics, and do not establish non-inferiority when they include zero.", "",
             "## Test means and paired intervals", "", "| Metric | Parent | A | B | A−parent (95% CI) | B−parent (95% CI) | B−A (95% CI) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for metric in metrics:
        means = report["test"]["model_means"]
        def cell(comp):
            v = contrasts[metric][comp]
            return f"{v['mean']:.6f} [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]"
        lines.append(f"| {metric} | {means['parent'][metric]:.6f} | {means['A'][metric]:.6f} | {means['B'][metric]:.6f} | {cell('A_minus_parent')} | {cell('B_minus_parent')} | {cell('B_minus_A')} |")
    lines += ["", "## Validation contrasts at update 640", "", "| Metric | A−parent (95% CI) | B−parent (95% CI) | B−A (95% CI) |", "|---|---:|---:|---:|"]
    for metric in metrics:
        values = validation_contrasts["640"][metric]
        def validation_cell(name):
            v = values[name]
            return f"{v['mean']:.6f} [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]"
        lines.append(f"| {metric} | {validation_cell('A_minus_parent')} | {validation_cell('B_minus_parent')} | {validation_cell('B_minus_A')} |")
    lines += ["", "## Limits", "", "This is a mixed-data experiment. It does not isolate replay mixture fraction or update count versus the earlier pilot. Improved loss does not establish playing strength. The fixed filtered-replay panel is part of training and is not an independent holdout. No Arena or promotion was run.", ""]
    atomic_write_text(output / "REPORT.md", "\n".join(lines))
    return {"state": "COMPLETED", "output": str(output), "report_sha256": _sha(output / "report.json"),
            "elapsed_seconds": report["elapsed_seconds"]}
