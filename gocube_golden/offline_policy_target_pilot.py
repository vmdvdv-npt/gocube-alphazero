"""Bounded D270 policy-target relabel/transfer diagnostic.

This module is invoked only through the documented Orchestrator V2 production
entrypoint. It reads immutable source artifacts, writes an isolated diagnostic
directory, and never creates a production lineage, self-play record, or Arena.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .neural import model_hash
from .provenance import file_sha256, sha256_fingerprint
from .search import Evaluation, SearchSettings, SequentialPUCT
from .search_adapter import GoldenSearchAdapter
from .torus9_adaptation import AdaptationModel, FINGERPRINT as TARGET_FINGERPRINT
from .torus9_five_channel_training import OrdinaryTrainer
from .torus9_monolith import torus9_state_from_identity
from .torus9_m137_5ch import build_m137_five_channel_observation as build_observation


SCHEMA = "gocube-d270-policy-target-pilot-v1"
PARENT_LINEAGE = "az23-m266-m269-offline-lr-ab-20261010-v1-ab-m267-m269-lr-B"
PARENT_CHECKPOINT = "M269"
SOURCE_LINEAGE = "torus9-c2560-seed2-online-m262-arena1-s64-20261008-v1"
SOURCE_GENERATION = 270
SEED = 2026101207
SEARCH_SIMULATIONS = 1600
UPDATES = 128
BATCH_SIZE = 64
LEARNING_RATE = 2.5e-5
GRADIENT_CLIP = 8.0
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026101208
PHASES = ((1, 8, "opening"), (9, 24, "early"), (25, 64, "middle"), (65, 10**9, "late"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: object) -> str:
    data = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _phase(ply: int) -> str:
    for low, high, name in PHASES:
        if low <= ply <= high:
            return name
    raise ValueError(f"Unsupported ply: {ply}")


def _checked_payload(payload: Mapping[str, object]) -> dict[str, Any]:
    allowed = {
        "schema", "runs_root", "output", "parent_checkpoint", "parent_metadata",
        "parent_provenance", "parent_rolling_replay", "parent_checkpoint_sha256",
        "source_fresh_replay", "source_fresh_sha256", "device",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("Unknown pilot config fields: " + ", ".join(sorted(unknown)))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}")
    if payload.get("device", "cuda") != "cuda":
        raise ValueError("The bounded pilot is pinned to the existing CUDA training device")
    result = dict(payload)
    for key in ("runs_root", "output", "parent_checkpoint", "parent_metadata", "parent_provenance", "parent_rolling_replay", "source_fresh_replay"):
        if key not in result or not isinstance(result[key], str):
            raise ValueError(f"{key} must be an absolute path")
        path = Path(result[key])
        if not path.is_absolute():
            raise ValueError(f"{key} must be absolute")
        result[key] = path.resolve()
    for key in ("parent_checkpoint_sha256", "source_fresh_sha256"):
        value = result.get(key)
        if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
            raise ValueError(f"{key} must be a sha256:<64 hex> identity")
        try:
            int(value.removeprefix("sha256:"), 16)
        except ValueError as exc:
            raise ValueError(f"{key} must be a sha256:<64 hex> identity") from exc
    runs_root = result["runs_root"]
    output = result["output"]
    if runs_root == output or runs_root in output.parents:
        raise ValueError("Diagnostic output must be outside runs_root")
    if output.exists():
        raise FileExistsError(f"Diagnostic output must be a new exclusive directory: {output}")
    if not any(path.parts[-2:] == ("active", PARENT_LINEAGE) for path in result["parent_checkpoint"].parents):
        raise ValueError("Parent checkpoint must be AZ23 B/M269")
    for key in ("parent_metadata", "parent_provenance", "parent_rolling_replay"):
        if not any(path.parts[-2:] == ("active", PARENT_LINEAGE) for path in result[key].parents):
            raise ValueError(f"{key} must resolve inside the AZ23 B/M269 lineage")
    if not any(path.parts[-2:] == ("active", SOURCE_LINEAGE) for path in result["source_fresh_replay"].parents):
        raise ValueError("Source must be D270 from the registered R23 production lineage")
    return result


def _file_identity(path: Path, expected: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"SHA mismatch for immutable input: {path}")
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def _validate_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    parent_path: Path = config["parent_checkpoint"]
    parent_metadata_path: Path = config["parent_metadata"]
    parent_provenance_path: Path = config["parent_provenance"]
    rolling_path: Path = config["parent_rolling_replay"]
    source_fresh_path: Path = config["source_fresh_replay"]
    identities = {
        "parent_checkpoint": _file_identity(parent_path, str(config["parent_checkpoint_sha256"])),
        "parent_metadata": _file_identity(parent_metadata_path),
        "parent_provenance": _file_identity(parent_provenance_path),
        "parent_rolling_replay": _file_identity(rolling_path),
        "source_fresh_replay": _file_identity(source_fresh_path),
    }
    metadata = json.loads(parent_metadata_path.read_text())
    parent_raw = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_ckpt_meta = parent_raw.get("metadata", {})
    if metadata.get("model_hash") != parent_ckpt_meta.get("model_hash"):
        raise ValueError("Parent checkpoint metadata disagrees with the checkpoint payload")
    if metadata.get("model_hash") != "sha256:" + str(metadata.get("model_hash", "")).removeprefix("sha256:"):
        raise ValueError("Malformed parent model hash")
    if metadata.get("ordinary_schema") != "torus9-five-channel-ordinary-training-v1":
        raise ValueError("Parent must have complete ordinary Adam state")
    if metadata.get("ordinary_update") is None or parent_raw.get("optimizer_state_dict") is None:
        raise ValueError("Parent checkpoint is missing its full optimizer state")
    if int(metadata.get("generation", 269)) != 269 or parent_ckpt_meta.get("parent_checkpoint", {}).get("generation") != 268:
        raise ValueError("Parent must be AZ23 B/M269")
    architecture = parent_ckpt_meta.get("architecture_config", {})
    heads = architecture.get("heads", {}) if isinstance(architecture, Mapping) else {}
    if (parent_ckpt_meta.get("komi") != 1.5
            or not isinstance(heads, Mapping)
            or set(heads) != {"policy", "value", "ownership", "score"}):
        raise ValueError("Parent checkpoint model/target contract mismatch")
    if parent_ckpt_meta.get("target_fingerprint") != TARGET_FINGERPRINT:
        raise ValueError("Parent target contract mismatch")
    if not isinstance(parent_raw.get("clock_origin"), Mapping) or not isinstance(parent_raw.get("seed"), int):
        raise ValueError("Parent does not contain resumable ordinary training state")
    parent_seed = int(parent_raw["seed"])
    parent_update = int(parent_raw["ordinary_update"])
    if int(metadata["ordinary_update"]) != parent_update:
        raise ValueError("Parent checkpoint Adam update disagrees with its metadata")
    del parent_raw
    # Constructing the existing trainer on CPU verifies named Adam groups,
    # tensor shapes, finiteness, and every per-parameter clock without changing
    # the immutable checkpoint.
    adam_probe = OrdinaryTrainer(parent_path, learning_rate=LEARNING_RATE,
                                 seed=parent_seed, gradient_clip=GRADIENT_CLIP,
                                 batch_size=BATCH_SIZE, device="cpu")
    if adam_probe.update != parent_update:
        raise ValueError("Parent trainer did not restore the complete Adam update")
    del adam_probe

    provenance = json.loads(parent_provenance_path.read_text())
    if provenance.get("schema") != "gocube-orchestrator-v2-production-provenance-v2" or not provenance.get("checkpoint_reload_verified"):
        raise ValueError("Parent provenance is not a verified V2 checkpoint")
    if provenance.get("checkpoint", {}).get("checkpoint_id") != PARENT_CHECKPOINT:
        raise ValueError("Parent provenance checkpoint id mismatch")
    if identities["parent_checkpoint"]["sha256"] != provenance["artifact_identities"]["checkpoint"]["sha256"]:
        raise ValueError("Parent checkpoint SHA does not match its V2 provenance")
    if identities["parent_metadata"]["sha256"] != provenance["artifact_identities"]["checkpoint_metadata"]["sha256"]:
        raise ValueError("Parent checkpoint metadata SHA does not match its V2 provenance")
    provenance_rolling = provenance["artifact_identities"]["rolling_replay"]
    if Path(provenance_rolling["path"]).name != rolling_path.name or provenance_rolling["sha256"] != identities["parent_rolling_replay"]["sha256"]:
        raise ValueError("Parent rolling replay does not match its V2 provenance")
    parent_buckets = [json.loads(line) for line in rolling_path.read_text().splitlines() if line.strip()]
    parent_generations = [int(row["generation"]) for row in parent_buckets]
    if not parent_generations or max(parent_generations) >= SOURCE_GENERATION:
        raise ValueError("D270 is present in the AZ23 B/M269 training replay")
    for bucket in parent_buckets:
        for shard in bucket["shards"]:
            _file_identity(Path(shard["path"]), str(shard["sha"]))

    fresh = json.loads(source_fresh_path.read_text())
    source_id = _file_identity(source_fresh_path, str(config["source_fresh_sha256"]))
    if int(fresh.get("generation", -1)) != SOURCE_GENERATION:
        raise ValueError("D270 source manifest generation mismatch")
    source_run_root = source_fresh_path.parent.parent
    source_provenance_path = source_run_root / "metadata" / "provenance-v2" / "M270.json"
    source_provenance = json.loads(source_provenance_path.read_text())
    source_provenance_id = _file_identity(source_provenance_path)
    if source_provenance.get("schema") != "gocube-orchestrator-v2-production-provenance-v2":
        raise ValueError("D270 source lacks verified V2 artifact provenance")
    if source_provenance.get("checkpoint", {}).get("generation") != SOURCE_GENERATION:
        raise ValueError("D270 source provenance generation mismatch")
    if source_provenance.get("artifact_identities", {}).get("fresh_replay", {}).get("sha256") != source_id["sha256"]:
        raise ValueError("D270 fresh manifest SHA does not match source provenance")
    source_shards = []
    game_total = 0
    source_model_hashes = set()
    for shard in fresh.get("shards", []):
        shard_path = Path(shard["path"]).resolve()
        shard_id = _file_identity(shard_path, "sha256:" + str(shard["sha"]).removeprefix("sha256:"))
        raw_path = shard_path.with_suffix(".games.jsonl.gz")
        raw_id = _file_identity(raw_path)
        provenance_replay = source_provenance["artifact_identities"].get(f"replay_shard_{len(source_shards)}")
        if provenance_replay is None or provenance_replay.get("sha256") != shard_id["sha256"]:
            raise ValueError("D270 learner shard SHA does not match source provenance")
        raw = torch.load(shard_path, map_location="cpu", weights_only=False)
        if raw.get("generation") != SOURCE_GENERATION or raw.get("contract") != TARGET_FINGERPRINT:
            raise ValueError("D270 replay shard generation/contract mismatch")
        if str(raw.get("raw_games_sha", "")) != raw_id["sha256"]:
            raise ValueError("D270 raw game and learner replay identities disagree")
        game_ids = [str(game["game_id"]) for game in raw.get("games", [])]
        if len(game_ids) != int(shard["games"]) or len(game_ids) != len(set(game_ids)):
            raise ValueError("D270 learner shard game count mismatch")
        game_total += len(game_ids)
        source_model_hashes.add(str(raw.get("actor_hash")))
        source_shards.append({"learner": shard_id, "raw_games": raw_id, "game_ids": game_ids,
                              "actor_hash": raw.get("actor_hash")})
    if game_total != 1536 or len(source_shards) != 12:
        raise ValueError("D270 source is expected to contain exactly 1,536 games in 12 shards")
    if len(source_model_hashes) != 1:
        raise ValueError("D270 learner shards have mixed actor identities")
    actor_hash = next(iter(source_model_hashes))
    if actor_hash != "sha256:" + str(fresh.get("actor_hash", actor_hash)).removeprefix("sha256:") and fresh.get("actor_hash") is not None:
        raise ValueError("D270 actor identity mismatch")
    identities["d270_shards"] = source_shards
    identities["source_provenance"] = source_provenance_id
    identities["d270_actor_model_hash"] = actor_hash
    return {
        "identities": identities,
        "parent_metadata": metadata,
        "parent_training_state": {"seed": parent_seed, "ordinary_update": parent_update},
        "parent_model_hash": metadata["model_hash"],
        "parent_replay_generations": parent_generations,
        "d270_manifest": fresh,
        "d270_game_count": game_total,
        "d270_actor_model_hash": actor_hash,
    }


def _read_game_candidates(source_shards: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    games: dict[str, dict[str, Any]] = {}
    for shard in source_shards:
        raw_path = Path(shard["raw_games"]["path"])
        expected_sha = str(shard["raw_games"]["sha256"])
        if file_sha256(raw_path) != expected_sha:
            raise ValueError("D270 raw game artifact changed after preflight")
        with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                game_id = str(record["game_id"])
                if game_id in games:
                    raise ValueError("Duplicate D270 game id")
                if record.get("model_checkpoint_label") != "M269":
                    raise ValueError("D270 raw actor checkpoint label drift")
                if record.get("model_hash") != shard.get("actor_hash"):
                    raise ValueError("D270 raw game actor hash disagrees with its learner shard")
                if record.get("technical_termination") is not None or record.get("formal_result") not in ("BLACK", "WHITE", "DRAW"):
                    raise ValueError("D270 contains a non-formal game")
                eligible_index = 0
                per_game: list[dict[str, Any]] = []
                for position in record.get("positions", []):
                    if not position.get("training_eligible"):
                        continue
                    state = position["state"]
                    ply = int(position["ply"])
                    item = {
                        "game_id": game_id,
                        "ply": ply,
                        "eligible_index": eligible_index,
                        "phase": _phase(ply),
                        "side": str(position["side_to_move"]),
                        "state": state,
                        "historical_pi": position["pi"],
                        "historical_visits": position["root_visits"],
                        "search_mode": position["search_mode"],
                        "search_simulations": position["search_simulations"],
                        "model_hash": record["model_hash"],
                        "formal_result": record["formal_result"],
                    }
                    if len(item["historical_pi"]) != 82 or sum(item["historical_visits"]) <= 0:
                        raise ValueError("Invalid stored D270 policy target")
                    if not math.isclose(sum(map(float, item["historical_pi"])), 1.0, abs_tol=1e-5):
                        raise ValueError("D270 stored policy is not normalized")
                    candidates.append(item)
                    per_game.append(item)
                    eligible_index += 1
                games[game_id] = {
                    "game_id": game_id,
                    "formal_result": record["formal_result"],
                    "positions": per_game,
                    "start_state": record["start_state"],
                    "action_trace": record["final_action_trace"],
                }
    if len(games) != 1536 or not candidates:
        raise ValueError("D270 raw game record count or eligible position pool mismatch")
    return candidates, games


def _split_games(game_ids: Sequence[str], seed: int = SEED) -> dict[str, list[str]]:
    ordered = sorted(str(item) for item in game_ids)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    train_n = int(math.floor(n * 0.70))
    validation_n = int(math.floor(n * 0.15))
    test_n = n - train_n - validation_n
    return {
        "train": ordered[:train_n],
        "validation": ordered[train_n:train_n + validation_n],
        "test": ordered[train_n + validation_n:train_n + validation_n + test_n],
    }


def _select_positions(candidates: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[str]], *, seed: int = SEED,
                      counts: Mapping[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
    requested = dict(counts or {"train": 512, "validation": 128, "test": 128})
    selected: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        allowed_games = set(splits[split_name])
        buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in candidates:
            if row["game_id"] in allowed_games:
                buckets.setdefault((str(row["phase"]), str(row["side"])), []).append(row)
        rng = random.Random(seed + {"train": 0, "validation": 1, "test": 2}[split_name])
        for bucket in buckets.values():
            rng.shuffle(bucket)
        keys = [(phase, side) for _, _, phase in PHASES for side in ("BLACK", "WHITE")]
        chosen: list[dict[str, Any]] = []
        used_games: set[str] = set()
        cursors = {key: 0 for key in keys}
        while len(chosen) < requested[split_name]:
            progressed = False
            for key in keys:
                rows = buckets.get(key, [])
                while cursors[key] < len(rows) and str(rows[cursors[key]]["game_id"]) in used_games:
                    cursors[key] += 1
                if cursors[key] < len(rows) and len(chosen) < requested[split_name]:
                    row = rows[cursors[key]]
                    cursors[key] += 1
                    used_games.add(str(row["game_id"]))
                    chosen.append(dict(row))
                    progressed = True
            if not progressed:
                raise ValueError(f"D270 cannot supply {requested[split_name]} unique eligible games in all phase/color strata")
        selected[split_name] = chosen
    all_games = [str(row["game_id"]) for rows in selected.values() for row in rows]
    if len(all_games) != len(set(all_games)):
        raise ValueError("Position selection reused a game across splits")
    return selected


def _prepare(payload: Mapping[str, object]) -> dict[str, Any]:
    config = _checked_payload(payload)
    sources = _validate_sources(config)
    candidates, games = _read_game_candidates(sources["identities"]["d270_shards"])
    splits = _split_games(list(games))
    selected = _select_positions(candidates, splits)
    for split, rows in selected.items():
        if len(rows) != {"train": 512, "validation": 128, "test": 128}[split]:
            raise AssertionError("Selected position count drift")
    prepared = {"config": config, "sources": sources, "candidates": candidates, "games": games,
                "splits": splits, "selected": selected}
    # Validate learner/raw alignment and full-state reconstruction before any
    # output directory is created. This also makes --check a useful preflight.
    prepared["original_targets"] = _load_selected_targets(prepared)
    return prepared


def check(payload: Mapping[str, object]) -> dict[str, Any]:
    prepared = _prepare(payload)
    splits = prepared["splits"]
    selected = prepared["selected"]
    return {
        "schema": SCHEMA,
        "status": "READY",
        "parent_model_hash": prepared["sources"]["parent_model_hash"],
        "parent_replay_generations": prepared["sources"]["parent_replay_generations"],
        "d270_actor_model_hash": prepared["sources"]["d270_actor_model_hash"],
        "whole_game_split_counts": {key: len(value) for key, value in splits.items()},
        "selected_position_counts": {key: len(value) for key, value in selected.items()},
        "selected_strata": {key: _strata(value) for key, value in selected.items()},
        "planned_simulations": SEARCH_SIMULATIONS * sum(map(len, selected.values())),
        "full_state_target_alignment": "verified",
        "output": str(prepared["config"]["output"]),
    }


def _strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['phase']}:{row['side']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _manifest_selection(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        split: [
            {key: row[key] for key in ("game_id", "ply", "eligible_index", "phase", "side", "search_mode", "search_simulations", "model_hash")}
            | {"state_sha256": "sha256:" + hashlib.sha256(_canonical_bytes(row["state"])).hexdigest()}
            for row in values
        ]
        for split, values in rows.items()
    }


def _load_selected_targets(prepared: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    selected = prepared["selected"]
    lookup = {str(row["game_id"]): row for rows in selected.values() for row in rows}
    targets: dict[str, list[dict[str, Any]]] = {key: [] for key in selected}
    seen: set[str] = set()
    adapter = GoldenSearchAdapter()
    for shard in prepared["sources"]["identities"]["d270_shards"]:
        path = Path(shard["learner"]["path"])
        raw = torch.load(path, map_location="cpu", weights_only=False)
        for game in raw["games"]:
            game_id = str(game["game_id"])
            row = lookup.get(game_id)
            if row is None:
                continue
            split = next(name for name, values in selected.items() if game_id in {str(x["game_id"]) for x in values})
            index = int(row["eligible_index"])
            if not 0 <= index < len(game["score"]):
                raise ValueError("D270 eligible position index drift between raw and learner data")
            state = torus9_state_from_identity(row["state"], expected_komi=1.5)
            legal_context = adapter.prepare_legal_actions(state)
            observation = build_observation(state, legal_context=legal_context).cpu()
            stored_observation = game["observation"][index].cpu().float()
            if observation.shape != stored_observation.shape or not torch.allclose(
                observation, stored_observation, atol=1e-6, rtol=0
            ):
                raise ValueError("D270 raw full state does not match its stored training observation")
            stored_legal = tuple(bool(value) for value in game["legal"][index].tolist())
            if legal_context.action_mask != stored_legal:
                raise ValueError("D270 raw full state does not match its stored legal-action mask")
            if state.side_to_move.name != str(row["side"]):
                raise ValueError("D270 raw full state side-to-move label mismatch")
            historical_pi = torch.tensor(row["historical_pi"], dtype=torch.float32)
            if not torch.allclose(game["pi"][index].float(), historical_pi, atol=1e-6, rtol=0):
                raise ValueError("Raw and original replay policy targets disagree")
            if not torch.equal(game["visits"][index].long(), torch.tensor(row["historical_visits"], dtype=torch.long)):
                raise ValueError("Raw and original replay visits disagree")
            item = {
                "game_id": game_id,
                "ply": int(row["ply"]),
                "eligible_index": index,
                "phase": str(row["phase"]),
                "side": str(row["side"]),
                "state": row["state"],
                "observation": game["observation"][index].clone(),
                "original_pi": game["pi"][index].clone(),
                "new_pi": None,
                "z": game["z"][index].clone(),
                "ownership": game["ownership"][index].clone(),
                "score": game["score"][index].clone(),
                "legal": game["legal"][index].clone(),
                "visits": game["visits"][index].clone(),
                "historical_search_mode": row["search_mode"],
                "historical_search_simulations": row["search_simulations"],
            }
            targets[split].append(item)
            seen.add(game_id)
        del raw
    expected = {str(row["game_id"]) for rows in selected.values() for row in rows}
    if seen != expected:
        raise ValueError("Selected D270 positions are missing from original learner replay")
    for split, rows in targets.items():
        rows.sort(key=lambda row: next(i for i, original in enumerate(selected[split]) if original["game_id"] == row["game_id"]))
    return targets


class _AdaptationSearchEvaluator:
    def __init__(self, model: AdaptationModel, device: str) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    def evaluate_prepared(self, state: Any, legal_context: Any) -> Evaluation:
        observation = build_observation(state, legal_context=legal_context).to(self.device)
        with torch.inference_mode():
            policy_logits, wdl_logits, _, _ = self.model.forward_auxiliary(observation.unsqueeze(0))
            policy = torch.softmax(policy_logits[0], dim=0)
            wdl = torch.softmax(wdl_logits[0], dim=0)
        return Evaluation(
            policy=tuple(float(x) for x in policy.detach().cpu()),
            wdl=tuple(float(x) for x in wdl.detach().cpu()),
        )


def _reanalysis_targets(rows: Sequence[dict[str, Any]], parent_checkpoint: Path, *, device: str, seed: int,
                       progress: Any = None) -> tuple[list[dict[str, Any]], float]:
    trainer = OrdinaryTrainer(parent_checkpoint, learning_rate=LEARNING_RATE, seed=seed,
                              gradient_clip=GRADIENT_CLIP, batch_size=BATCH_SIZE, device=device)
    model = trainer.model.eval()
    evaluator = _AdaptationSearchEvaluator(model, device)
    settings = SearchSettings(simulations=SEARCH_SIMULATIONS, cpuct=1.25, fpu=0.0, root_noise=False,
                              deterministic_tie_break=True, tree_reuse=False)
    adapter = GoldenSearchAdapter()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, row in enumerate(rows, 1):
        state = torus9_state_from_identity(row["state"], expected_komi=1.5)
        context = adapter.prepare_legal_actions(state)
        prior_eval = evaluator.evaluate_prepared(state, context)
        result = SequentialPUCT(settings, adapter=adapter).search(state, evaluator, seed=0)
        if result.simulations != SEARCH_SIMULATIONS or sum(result.root_visits) != SEARCH_SIMULATIONS:
            raise ValueError("MCTS did not execute the requested fresh-tree budget")
        if result.legal_action_mask != context.action_mask:
            raise ValueError("MCTS root legal mask differs from saved full state")
        row["new_pi"] = torch.tensor(result.pi, dtype=torch.float32)
        legal_prior = [0.0] * 82
        legal_indices = [81 if action == "PASS" else int(action) for action in context.actions]
        total_legal_prior = sum(float(prior_eval.policy[index]) for index in legal_indices)
        if total_legal_prior <= 0:
            raise ValueError("M269 prior has no legal probability mass")
        for index in legal_indices:
            legal_prior[index] = float(prior_eval.policy[index]) / total_legal_prior
        results.append({
            "game_id": row["game_id"], "ply": row["ply"], "phase": row["phase"],
            "side": row["side"], "state_sha256": "sha256:" + hashlib.sha256(_canonical_bytes(row["state"])).hexdigest(),
            "prior": list(prior_eval.policy), "legal_prior": legal_prior,
            "visits": list(result.root_visits),
            "q": list(result.root_q), "policy": list(result.pi),
            "simulations_requested": SEARCH_SIMULATIONS, "simulations_completed": result.simulations,
            "actual_root_visit_sum": int(sum(result.root_visits)),
            "evaluator_calls": int(result.evaluator_calls),
            "search_implementation_fingerprint": result.implementation_fingerprint,
            "root_noise": False, "fresh_tree": True,
            "historical_search_mode": row["historical_search_mode"],
            "historical_search_simulations": row["historical_search_simulations"],
        })
        if progress is not None:
            progress(number, len(rows))
    return results, time.perf_counter() - started


def _as_game_rows(rows: Sequence[Mapping[str, Any]], *, target_key: str) -> list[dict[str, Any]]:
    games = []
    for index, row in enumerate(rows):
        games.append({
            "game_id": str(row["game_id"]),
            "observation": row["observation"].unsqueeze(0),
            "pi": row[target_key].unsqueeze(0),
            "z": row["z"].unsqueeze(0),
            "ownership": row["ownership"].unsqueeze(0),
            "score": row["score"].reshape(1),
            "legal": row["legal"].unsqueeze(0),
            "visits": row["visits"].unsqueeze(0),
            "contract": TARGET_FINGERPRINT,
            "split": "train",
            "_sample_index": index,
        })
    return games


def _sample_order_hash(seed: int, first_update: int, updates: int, count: int) -> tuple[str, list[str]]:
    ids: list[str] = []
    hashes: list[str] = []
    for offset in range(updates):
        update = first_update + offset + 1
        rng = random.Random(seed + update)
        selected = [rng.randrange(count) for _ in range(BATCH_SIZE)]
        encoded = _canonical_bytes(selected)
        hashes.append("sha256:" + hashlib.sha256(encoded).hexdigest())
        ids.extend(str(index) for index in selected)
    return "sha256:" + hashlib.sha256(_canonical_bytes(ids)).hexdigest(), hashes


def _state_fingerprint(value: object) -> str:
    digest = hashlib.sha256()
    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            digest.update(_canonical_bytes(list(tensor.shape)) + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda x: str(x)):
                digest.update(_canonical_bytes(str(key)) + b"\0")
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                visit(child)
        else:
            digest.update(_canonical_bytes(item) + b"\0")
    visit(value)
    return "sha256:" + digest.hexdigest()


def _distribution_metrics(logits: torch.Tensor, targets: torch.Tensor, legal: torch.Tensor) -> dict[str, float]:
    log_policy = F.log_softmax(logits, dim=-1)
    policy = log_policy.exp()
    log_legal = log_policy.masked_fill(~legal, -torch.inf)
    legal_policy = torch.softmax(log_legal, dim=-1)
    target = targets.clamp_min(0)
    target = target / target.sum(dim=-1, keepdim=True)
    ce = -(target * log_policy).sum(dim=-1)
    kl = (target * (target.clamp_min(1e-30).log() - log_policy)).sum(dim=-1)
    legal_kl = (target * (target.clamp_min(1e-30).log() - legal_policy.clamp_min(1e-30).log())).sum(dim=-1)
    return {
        "policy_ce_raw": float(ce.mean().cpu()),
        "policy_kl_raw": float(kl.mean().cpu()),
        "policy_kl_legal_renormalized": float(legal_kl.mean().cpu()),
        "illegal_policy_mass": float((policy * ~legal).sum(dim=-1).mean().cpu()),
    }


def _evaluate_model(model: AdaptationModel, rows: Sequence[Mapping[str, Any]], *, device: str,
                    target_keys: Sequence[str], batch_size: int = 64) -> dict[str, Any]:
    model = model.to(device).eval()
    by_target: dict[str, list[dict[str, float]]] = {key: [] for key in target_keys}
    per_row: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            observation = torch.stack([row["observation"] for row in batch_rows]).to(device)
            policy_logits, wdl_logits, ownership_logits, score_predictions = model.forward_auxiliary(observation)
            wdl_prob = torch.softmax(wdl_logits, dim=-1)
            for key in target_keys:
                target_pi = torch.stack([row[key] for row in batch_rows]).to(device)
                stats = _distribution_metrics(policy_logits, target_pi, torch.stack([row["legal"] for row in batch_rows]).to(device))
                by_target[key].append(stats)
                for offset, row in enumerate(batch_rows):
                    record = per_row.setdefault(str(row["game_id"]), {"game_id": str(row["game_id"]), "ply": int(row["ply"]), "policy": {}})
                    lp = F.log_softmax(policy_logits[offset], dim=-1)
                    t = target_pi[offset].clamp_min(0)
                    t = t / t.sum()
                    record["policy"][key] = {
                        "ce": float((-(t * lp).sum()).cpu()),
                        "kl": float((t * (t.clamp_min(1e-30).log() - lp)).sum().cpu()),
                    }
                    record["policy"][key]["probabilities"] = torch.softmax(policy_logits[offset], dim=-1).cpu().tolist()
            z = torch.stack([row["z"] for row in batch_rows]).to(device)
            ownership = torch.stack([row["ownership"] for row in batch_rows]).to(device).long()
            score = torch.stack([row["score"] for row in batch_rows]).to(device).reshape(-1)
            wdl_ce = -(z * F.log_softmax(wdl_logits, dim=-1)).sum(dim=-1)
            wdl_brier = ((wdl_prob - z) ** 2).sum(dim=-1)
            own_ce = F.cross_entropy(ownership_logits.reshape(-1, 3), ownership.reshape(-1), reduction="none").reshape(len(batch_rows), -1).mean(dim=-1)
            score_error = (score_predictions.reshape(-1) - score) * 81.5
            for offset, row in enumerate(batch_rows):
                record = per_row[str(row["game_id"])]
                record["wdl_ce"] = float(wdl_ce[offset].cpu())
                record["wdl_brier"] = float(wdl_brier[offset].cpu())
                record["ownership_ce"] = float(own_ce[offset].cpu())
                record["score_mse_points"] = float((score_error[offset] ** 2).cpu())
                record["score_mae_points"] = float(score_error[offset].abs().cpu())
                record["wdl_probabilities"] = wdl_prob[offset].cpu().tolist()
                record["ownership_probabilities"] = torch.softmax(ownership_logits[offset], dim=-1).cpu().tolist()
                record["score_prediction_points"] = float(score_predictions[offset].reshape(()) .cpu() * 81.5)
    summary: dict[str, Any] = {}
    for key in target_keys:
        summary[f"policy_to_{key}"] = {
            name: statistics.fmean(batch[name] for batch in by_target[key])
            for name in by_target[key][0]
        }
    rows_values = list(per_row.values())
    for metric in ("wdl_ce", "wdl_brier", "ownership_ce", "score_mse_points", "score_mae_points"):
        summary[metric] = statistics.fmean(float(row[metric]) for row in rows_values)
    return {"summary": summary, "per_game": per_row}


def _bootstrap_cluster(differences: Mapping[str, float], *, seed: int = BOOTSTRAP_SEED,
                       replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    groups = sorted(differences)
    if len(groups) < 2:
        raise ValueError("Paired game-cluster bootstrap requires at least two games")
    values = [float(differences[group]) for group in groups]
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        estimates.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    estimates.sort()
    def quantile(q: float) -> float:
        index = (len(estimates) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return estimates[low] + (estimates[high] - estimates[low]) * (index - low)
    return {"method": "paired-game-cluster-bootstrap-v1", "replicates": replicates,
            "seed": seed, "clusters": len(groups), "mean": statistics.fmean(values),
            "ci95": [quantile(.025), quantile(.975)], "p_b_le_a": sum(x <= 0 for x in estimates) / replicates}


def _set_policy(game_rows: Sequence[dict[str, Any]], *, new: bool) -> list[dict[str, Any]]:
    key = "new_pi" if new else "original_pi"
    return _as_game_rows(game_rows, target_key=key)


def _clone_model_from_checkpoint(checkpoint: Path, device: str, seed: int) -> tuple[AdaptationModel, dict[str, Any]]:
    trainer = OrdinaryTrainer(checkpoint, learning_rate=LEARNING_RATE, seed=seed,
                              gradient_clip=GRADIENT_CLIP, batch_size=BATCH_SIZE, device=device)
    return trainer.model, {"update": trainer.update, "seed": trainer.seed}


def _write_targets(output: Path, targets: Mapping[str, Sequence[Mapping[str, Any]]], reanalysis: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    identities = {}
    for split, rows in targets.items():
        path = output / "targets" / f"{split}-targets.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                search = reanalysis[split][index]
                item = {
                    "game_id": row["game_id"], "ply": row["ply"], "side": row["side"], "phase": row["phase"],
                    "state": row["state"], "original_pi": row["original_pi"].tolist(),
                    "original_wdl": row["z"].tolist(),
                    "original_ownership": row["ownership"].tolist(),
                    "original_score": float(row["score"].reshape(())),
                    "legal_action_mask": [bool(value) for value in row["legal"].tolist()],
                    "original_visits": row["visits"].tolist(),
                    "new_pi": row["new_pi"].tolist(), "new_visits": search["visits"],
                    "prior": search["prior"], "legal_prior": search["legal_prior"], "q": search["q"],
                    "simulations_requested": search["simulations_requested"],
                    "simulations_completed": search["simulations_completed"],
                    "actual_root_visit_sum": search["actual_root_visit_sum"],
                    "evaluator_calls": search["evaluator_calls"],
                    "search_implementation_fingerprint": search["search_implementation_fingerprint"],
                    "historical_search_mode": row["historical_search_mode"],
                    "historical_search_simulations": row["historical_search_simulations"],
                }
                stream.write(json.dumps(_jsonable(item), ensure_ascii=False, sort_keys=True) + "\n")
        identities[str(path)] = file_sha256(path)
    return identities


def run(payload: Mapping[str, object], *, config_path: Path | None = None) -> dict[str, Any]:
    from .orchestrator_v2.execution_permit import require_engine_execution
    require_engine_execution(__name__, action="training", topology="torus9")
    run_started = time.perf_counter()
    prepared = _prepare(payload)
    config = prepared["config"]
    output: Path = config["output"]
    output.mkdir(parents=True, exist_ok=False)
    started = run_started
    source_ids = prepared["sources"]["identities"]
    splits = prepared["splits"]
    selected = prepared["selected"]
    selection = {
        "schema": SCHEMA + ".split-v1", "seed": SEED, "whole_game_split": splits,
        "positions": _manifest_selection(selected), "strata": {key: _strata(value) for key, value in selected.items()},
        "source_identity": source_ids,
    }
    selection_sha = _write_json(output / "split-and-sample.json", selection)
    run_manifest: dict[str, Any] = {
        "schema": SCHEMA, "status": "PREPARED", "selection_sha256": selection_sha,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        "source_identity": source_ids,
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, run_manifest)

    def mark_progress(phase: str, **details: Any) -> None:
        run_manifest.update(
            status="RUNNING", phase=phase,
            updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        run_manifest.update(details)
        _write_json(manifest_path, run_manifest)

    mark_progress("loading_original_targets")
    original_targets = prepared["original_targets"]
    parent_path: Path = config["parent_checkpoint"]
    device = "cuda"
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the pinned diagnostic interface")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mark_progress("timing_pilot")

    # Pilot uses the first deterministic two positions from each phase/color
    # stratum, all already present in the frozen train selection.
    pilot_rows = []
    stratum_counts: dict[tuple[str, str], int] = {}
    for row in selected["train"]:
        key = (str(row["phase"]), str(row["side"]))
        if stratum_counts.get(key, 0) < 2:
            pilot_rows.append(row)
            stratum_counts[key] = stratum_counts.get(key, 0) + 1
    if len(pilot_rows) != 16:
        raise ValueError("Frozen train sample cannot supply the required 16-position timing pilot")
    all_search_rows: dict[str, list[dict[str, Any]]] = {}
    # Use original replay tensors for labels and observations; attach the exact
    # saved raw full state for reanalysis.
    prepared_by_id = {row["game_id"]: row for rows in selected.values() for row in rows}
    for split, rows in original_targets.items():
        for row in rows:
            row["state"] = prepared_by_id[row["game_id"]]["state"]
        all_search_rows[split] = rows
    targets_by_id = {str(row["game_id"]): row for rows in all_search_rows.values() for row in rows}
    pilot_target_rows = [targets_by_id[str(row["game_id"])] for row in pilot_rows]
    parent_seed = int(prepared["sources"]["parent_training_state"]["seed"])
    pilot_records, pilot_seconds = _reanalysis_targets(
        pilot_target_rows, parent_path, device=device, seed=parent_seed,
        progress=lambda done, total: print(f"[D270 pilot search] {done}/{total}", flush=True),
    )
    pilot_search_sha = _write_json(output / "pilot-search.json", {
        "results": pilot_records, "elapsed_seconds": pilot_seconds,
        "position_count": len(pilot_records), "simulations": len(pilot_records) * SEARCH_SIMULATIONS,
        "seconds_per_position": pilot_seconds / len(pilot_records),
        "selection_sha256": selection_sha,
    })
    estimated_full_search_seconds = pilot_seconds / len(pilot_records) * sum(map(len, selected.values()))
    mark_progress("reanalysis", pilot_search_sha256=pilot_search_sha,
                  timing_pilot_seconds=pilot_seconds,
                  estimated_full_search_seconds=estimated_full_search_seconds)

    reanalysis_by_split: dict[str, list[dict[str, Any]]] = {key: [] for key in selected}
    pilot_by_game = {str(row["game_id"]): result for row, result in zip(pilot_target_rows, pilot_records)}
    search_seconds = pilot_seconds
    for split in ("train", "validation", "test"):
        pending = [row for row in all_search_rows[split] if str(row["game_id"]) not in pilot_by_game]
        records, elapsed = _reanalysis_targets(
            pending, parent_path, device=device, seed=parent_seed,
            progress=lambda done, total, split=split: print(f"[D270 {split} reanalysis] {done}/{total}", flush=True),
        )
        search_seconds += elapsed
        reanalysis_by_split[split] = [pilot_by_game.get(str(row["game_id"])) for row in all_search_rows[split]]
        pending_map = {str(row["game_id"]): result for row, result in zip(pending, records)}
        reanalysis_by_split[split] = [pilot_by_game.get(str(row["game_id"]), pending_map.get(str(row["game_id"]))) for row in all_search_rows[split]]
        if any(row is None for row in reanalysis_by_split[split]):
            raise ValueError("Search result set does not cover the final frozen split sample")
        for row, result in zip(all_search_rows[split], reanalysis_by_split[split]):
            row["new_pi"] = torch.tensor(result["policy"], dtype=torch.float32)
    target_ids = _write_targets(output, all_search_rows, reanalysis_by_split)
    mark_progress("training", target_artifacts=target_ids)

    evaluate_targets = ("original_pi", "new_pi")
    metrics: dict[str, Any] = {"selection_sha256": selection_sha, "targets": target_ids, "splits": {}}
    model_outputs: dict[str, dict[str, dict[str, Any]]] = {"parent": {}, "A": {}, "B": {}}

    train_old = _set_policy(all_search_rows["train"], new=False)
    train_new = _set_policy(all_search_rows["train"], new=True)
    seed = parent_seed
    parent_update = int(prepared["sources"]["parent_training_state"]["ordinary_update"])
    order_hash, per_update_order = _sample_order_hash(seed, parent_update, UPDATES, len(train_old))
    if ([row["game_id"] for row in train_old] != [row["game_id"] for row in train_new]
            or len(train_old) != len(train_new)):
        raise AssertionError("Control and experiment replay order shape mismatch")
    trained_models: dict[str, AdaptationModel] = {}
    arm_reports: dict[str, dict[str, Any]] = {}
    common_adam_fingerprint: str | None = None
    for arm, train_games in (("A", train_old), ("B", train_new)):
        trainer = OrdinaryTrainer(parent_path, learning_rate=LEARNING_RATE, seed=seed,
                                  gradient_clip=GRADIENT_CLIP, batch_size=BATCH_SIZE, device=device)
        before_model_hash = model_hash(trainer.model)
        if before_model_hash != prepared["sources"]["parent_model_hash"]:
            raise ValueError("Arm did not start from the exact common M269 model")
        before_adam_fingerprint = _state_fingerprint(trainer.optimizer.state_dict())
        if common_adam_fingerprint is None:
            common_adam_fingerprint = before_adam_fingerprint
        elif before_adam_fingerprint != common_adam_fingerprint:
            raise ValueError("A/B arms did not start from the exact same full Adam state")
        metrics_rows = []
        arm_started = time.perf_counter()
        for update in range(UPDATES):
            metrics_rows.append(trainer.step(train_games))
        torch.cuda.synchronize()
        arm_seconds = time.perf_counter() - arm_started
        checkpoint_path = output / "checkpoints" / f"{arm}-after-128.pt"
        trainer.save(checkpoint_path, config_hash=sha256_fingerprint({"schema": SCHEMA, "arm": arm, "selection": selection_sha}),
                     parent={"lineage_id": PARENT_LINEAGE, "checkpoint_id": PARENT_CHECKPOINT,
                             "path": str(parent_path), "sha256": source_ids["parent_checkpoint"]["sha256"]},
                     replay_buckets=[])
        arm_report = {
            "arm": arm, "policy_target": "original" if arm == "A" else "reanalysis-1600",
            "parent_model_hash": before_model_hash, "final_model_hash": model_hash(trainer.model),
            "initial_adam_state_sha256": before_adam_fingerprint,
            "updates": UPDATES, "batch_size": BATCH_SIZE, "processed_samples": UPDATES * BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "gradient_clip": GRADIENT_CLIP,
            "training_seconds": arm_seconds, "sample_order_sha256": order_hash,
            "sample_order_per_update_sha256": per_update_order,
            "checkpoint": _file_identity(checkpoint_path),
            "checkpoint_metadata": _file_identity(checkpoint_path.with_suffix(".metadata.json")),
            "mean_losses": {key: statistics.fmean(float(row["losses"][key]) for row in metrics_rows) for key in metrics_rows[0]["losses"]},
            "final_update": metrics_rows[-1],
        }
        arm_reports[arm] = arm_report
        trained_models[arm] = trainer.model.cpu()
        del trainer
    mark_progress("evaluation", training_checkpoints={
        arm: details["checkpoint"]["sha256"] for arm, details in arm_reports.items()
    })

    # Hold test evaluation until both fixed 128-update branches are complete.
    parent_model, _ = _clone_model_from_checkpoint(parent_path, device, seed)
    for split, rows in all_search_rows.items():
        model_outputs["parent"][split] = _evaluate_model(parent_model, rows, device=device, target_keys=evaluate_targets)
        for arm in ("A", "B"):
            model_outputs[arm][split] = _evaluate_model(trained_models[arm], rows, device=device, target_keys=evaluate_targets)
    del parent_model, trained_models
    metrics.update(arm_reports)

    # Parent metrics for validation and test; do not read test outcomes while
    # selecting the split, target sample, update count, or any hyperparameter.
    for split in ("train", "validation", "test"):
        model_metrics = {name: model_outputs[name][split] for name in ("parent", "A", "B")}
        split_metrics: dict[str, Any] = {name: value["summary"] for name, value in model_metrics.items()}
        for arm in ("A", "B"):
            arm_summary = model_metrics[arm]["summary"]
            parent_summary = model_metrics["parent"]["summary"]
            deltas: dict[str, Any] = {}
            for target_name in ("policy_to_original_pi", "policy_to_new_pi"):
                deltas[target_name] = {
                    metric: arm_summary[target_name][metric] - parent_summary[target_name][metric]
                    for metric in arm_summary[target_name]
                }
            for metric in ("wdl_ce", "wdl_brier", "ownership_ce", "score_mse_points", "score_mae_points"):
                deltas[metric] = arm_summary[metric] - parent_summary[metric]
            split_metrics[f"{arm}_minus_parent_metric_deltas"] = deltas
        for arm in ("A", "B"):
            changes = []
            for game_id, parent_row in model_metrics["parent"]["per_game"].items():
                arm_row = model_metrics[arm]["per_game"][game_id]
                p = torch.tensor(parent_row["policy"]["original_pi"]["probabilities"], dtype=torch.float64)
                q = torch.tensor(arm_row["policy"]["original_pi"]["probabilities"], dtype=torch.float64)
                changes.append({
                    "parent_to_arm_kl": float((p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum()),
                    "mean_absolute_probability_change": float((p - q).abs().mean()),
                    "top_action_agreement": float(int(p.argmax() == q.argmax())),
                })
            split_metrics[f"{arm}_minus_parent_policy_change"] = {
                key: statistics.fmean(item[key] for item in changes) for key in changes[0]
            }
        if split == "test":
            differences = {}
            for game_id, row in model_metrics["A"]["per_game"].items():
                b = model_metrics["B"]["per_game"][game_id]
                differences[game_id] = float(b["policy"]["new_pi"]["kl"] - row["policy"]["new_pi"]["kl"])
            split_metrics["primary_B_minus_A_policy_kl_to_new_targets"] = _bootstrap_cluster(differences)
        metrics["splits"][split] = split_metrics

    test_metrics = metrics["splits"]["test"]
    primary = test_metrics["primary_B_minus_A_policy_kl_to_new_targets"]
    b_parent_new_kl = (
        test_metrics["B"]["policy_to_new_pi"]["policy_kl_raw"]
        - test_metrics["parent"]["policy_to_new_pi"]["policy_kl_raw"]
    )
    side_heads = ("wdl_ce", "wdl_brier", "ownership_ce", "score_mse_points", "score_mae_points")
    no_observed_mean_side_head_worsening = all(
        test_metrics["B_minus_parent_metric_deltas"][metric] <= 0
        and test_metrics["B_minus_parent_metric_deltas"][metric]
            <= test_metrics["A_minus_parent_metric_deltas"][metric]
        for metric in side_heads
    )
    primary_clear_gain = primary["mean"] < 0 and primary["ci95"][1] < 0
    train_b_minus_a = (
        metrics["splits"]["train"]["B"]["policy_to_new_pi"]["policy_kl_raw"]
        - metrics["splits"]["train"]["A"]["policy_to_new_pi"]["policy_kl_raw"]
    )
    if primary_clear_gain and b_parent_new_kl < 0 and no_observed_mean_side_head_worsening:
        decision = "candidate_for_separate_game_usefulness_check"
        decision_text = (
            "B has a clear test policy-KL gain over A, lower test KL than the parent, "
            "and no observed mean worsening in the other heads. This is only a candidate "
            "for a separate playing-strength/usefulness evaluation."
        )
    elif train_b_minus_a < 0 and not primary_clear_gain:
        decision = "train_gain_without_clear_test_generalization"
        decision_text = "B fits the new targets better on train, but the test result does not show a clear gain."
    elif not primary_clear_gain:
        decision = "do_not_scale_relabeling"
        decision_text = "The primary test contrast does not show a clear B-over-A gain; do not scale relabeling."
    else:
        decision = "mixed_result_no_game_usefulness_claim"
        decision_text = "Policy transfer improved, but the parent comparison or other-head checks do not support the candidate rule."
    metrics["decision"] = {
        "classification": decision,
        "text": decision_text,
        "primary_test_B_minus_A_ci_upper": primary["ci95"][1],
        "test_B_minus_parent_policy_kl_to_new_targets": b_parent_new_kl,
        "test_other_heads_no_observed_mean_worsening": no_observed_mean_side_head_worsening,
        "train_B_minus_A_policy_kl_to_new_targets": train_b_minus_a,
        "target_quality_or_playing_strength_established": False,
    }

    with (output / "model-evaluations.jsonl").open("w", encoding="utf-8") as stream:
        for split, rows in all_search_rows.items():
            for row in rows:
                game_id = str(row["game_id"])
                item = {"split": split, "game_id": game_id, "ply": int(row["ply"]), "side": row["side"], "phase": row["phase"]}
                for model_name in ("parent", "A", "B"):
                    item[model_name] = {name: model_outputs[model_name][split]["per_game"][game_id][name]
                                        for name in ("policy", "wdl_ce", "wdl_brier", "ownership_ce", "score_mse_points", "score_mae_points")}
                stream.write(json.dumps(item, sort_keys=True) + "\n")
    metrics["search"] = {
        "simulations_per_position": SEARCH_SIMULATIONS,
        "requested_positions": {key: len(value) for key, value in selected.items()},
        "actual_simulations": sum(len(value) for value in selected.values()) * SEARCH_SIMULATIONS,
        "pilot_positions": len(pilot_records), "pilot_seconds": pilot_seconds,
        "estimated_full_search_seconds": estimated_full_search_seconds,
        "search_seconds": search_seconds,
    }
    metrics["common_parent"] = {
        "lineage_id": PARENT_LINEAGE, "checkpoint_id": PARENT_CHECKPOINT,
        "model_hash": prepared["sources"]["parent_model_hash"],
        "ordinary_update": parent_update, "seed": seed, "adam_state_loaded": True,
    }
    metrics["design_note"] = (
        "This tests the full target update. Relative to historical D270 targets, model, simulation budget, root noise, and tree state differ. "
        "Transfer to the M269 search targets does not establish target quality or playing strength. D270 has prior investigative exposure."
    )
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["source_identity"] = source_ids
    repo_root = Path(__file__).resolve().parents[1]
    code_files = (
        "gocube_golden/offline_policy_target_pilot.py",
        "gocube_golden/orchestrator_v2/production_entrypoint.py",
        "gocube_golden/orchestrator_v2/operator_guide.py",
        "gocube_golden/torus9_five_channel_training.py",
        "gocube_golden/torus9_adaptation.py",
        "gocube_golden/torus9_m137_5ch.py",
        "gocube_golden/search.py",
        "gocube_golden/search_adapter.py",
        "docs/orchestrator-v2/ORCHESTRATOR_V2_LAUNCH_GUIDE.md",
    )
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
    worktree_status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True).stdout
    metrics["code_identity"] = {
        "commit": commit,
        "working_tree_clean": not worktree_status.strip(),
        "files": {path: file_sha256(repo_root / path) for path in code_files},
        "config": None if config_path is None else _file_identity(config_path),
    }
    metrics["selection_sha256"] = selection_sha
    report_sha = _write_json(output / "report.json", metrics)
    _write_markdown(output / "report.md", metrics, report_sha)
    _write_json(output / "manifest.json", {
        "schema": SCHEMA, "status": "COMPLETED", "report_sha256": report_sha,
        "selection_sha256": selection_sha, "source_identity": source_ids,
        "elapsed_seconds": metrics["elapsed_seconds"], "outputs": _output_identities(output),
    })
    return {"status": "COMPLETED", "output": str(output), "report_sha256": report_sha,
            "elapsed_seconds": metrics["elapsed_seconds"]}


def _output_identities(output: Path) -> dict[str, str]:
    result = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            result[str(path.relative_to(output))] = file_sha256(path)
    return result


def _write_markdown(path: Path, report: Mapping[str, Any], report_sha: str) -> None:
    primary = report["splits"]["test"]["primary_B_minus_A_policy_kl_to_new_targets"]
    lines = [
        "# D270 policy-target pilot", "",
        f"Report SHA-256: `{report_sha}`.",
        f"Elapsed: {report['elapsed_seconds'] / 60:.2f} minutes.",
        "",
        "## Primary result", "",
        f"Test B−A KL to the new M269/1600 targets: {primary['mean']:.8f}; "
        f"paired game-cluster 95% bootstrap CI [{primary['ci95'][0]:.8f}, {primary['ci95'][1]:.8f}] "
        f"({primary['replicates']} replicates, seed {primary['seed']}).",
        "Negative values favor B.", "",
        "## Transfer metrics", "",
        "| Split | Model | Orig CE | Orig KL | New CE | New KL | WDL CE | WDL Brier | Ownership CE | Score MSE | Score MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        for model in ("parent", "A", "B"):
            item = report["splits"][split][model]
            lines.append(
                f"| {split} | {model} | {item['policy_to_original_pi']['policy_ce_raw']:.6f} "
                f"| {item['policy_to_original_pi']['policy_kl_raw']:.6f} "
                f"| {item['policy_to_new_pi']['policy_ce_raw']:.6f} "
                f"| {item['policy_to_new_pi']['policy_kl_raw']:.6f} | {item['wdl_ce']:.6f} "
                f"| {item['wdl_brier']:.6f} | {item['ownership_ce']:.6f} "
                f"| {item['score_mse_points']:.4f} | {item['score_mae_points']:.4f} |"
            )
    lines += [
        "", "## Test B − parent changes", "",
        "| Metric | B − parent |",
        "|---|---:|",
        f"| Policy output KL (parent‖B) | {report['splits']['test']['B_minus_parent_policy_change']['parent_to_arm_kl']:.8f} |",
        f"| Policy output mean absolute probability change | {report['splits']['test']['B_minus_parent_policy_change']['mean_absolute_probability_change']:.8f} |",
        f"| Policy top-action agreement | {report['splits']['test']['B_minus_parent_policy_change']['top_action_agreement']:.4f} |",
        f"| Target KL to original | {report['splits']['test']['B_minus_parent_metric_deltas']['policy_to_original_pi']['policy_kl_raw']:.8f} |",
        f"| Target KL to new | {report['splits']['test']['B_minus_parent_metric_deltas']['policy_to_new_pi']['policy_kl_raw']:.8f} |",
        f"| WDL CE | {report['splits']['test']['B_minus_parent_metric_deltas']['wdl_ce']:.8f} |",
        f"| WDL Brier | {report['splits']['test']['B_minus_parent_metric_deltas']['wdl_brier']:.8f} |",
        f"| Ownership CE | {report['splits']['test']['B_minus_parent_metric_deltas']['ownership_ce']:.8f} |",
        f"| Score MSE (points²) | {report['splits']['test']['B_minus_parent_metric_deltas']['score_mse_points']:.8f} |",
        f"| Score MAE (points) | {report['splits']['test']['B_minus_parent_metric_deltas']['score_mae_points']:.8f} |",
        "", "## Decision", "",
        report["decision"]["text"],
        "",
        "## Interpretation", "",
        "Better fit to the M269 search targets demonstrates transfer into the network only; it does not establish that those targets are better or that playing strength increased.",
        "The target update jointly changes the teacher model, search budget, root noise, and fresh-tree state relative to historical D270 targets. D270 was used in earlier investigation and is not a fully independent dataset.",
        "", "## Reproducibility", "",
        f"- Selection SHA-256: `{report['selection_sha256']}`.",
        f"- Parent: AZ23 B/M269 `{report['common_parent']['model_hash']}` with full Adam, update {report['common_parent']['ordinary_update']}.",
        f"- Search: {report['search']['simulations_per_position']} fresh-tree simulations, root noise off; {report['search']['actual_simulations']} total simulations.",
        f"- Source and artifact SHAs: see `report.json` and `manifest.json`.",
        "- No Arena, new self-play, automatic model promotion, or production replay/checkpoint modification.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--check", action="store_true", help="verify source/target alignment and frozen sample plan without writing, search, or training")
    args = parser.parse_args(argv)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    result = check(payload) if args.check else run(payload, config_path=args.config.resolve())
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
