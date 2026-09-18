#!/usr/bin/env python3
"""Autonomous six-arm M88 training + Arena campaign on production Orchestrator V1."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.code_update_policy import CodeUpdateProvenancePolicy
from gocube_golden.operator_policy import install_operator_policy
from gocube_golden.run_spec import StrictProductionTrainingOrchestrator, StrictRunSpec
from gocube_golden.run_storage import evaluation_dir
from gocube_golden.torus9_contract import load_torus9_current_profile
from tools import torus9_staged_sims_harness_impl as h
from tools.torus9_m88_nightly_support import ALLOWED_LEARNING_RATES, install_nightly_profile_validation

DEFAULT_CONFIG = "configs/gocube/torus9_m88_nightly_ab_v1.json"
DRIVER = "tools/torus9_m88_nightly_driver.py"
BASE_COMMIT = "24d05394746d8008bb2d804a04367ab12f697270"


@dataclass(frozen=True)
class Arm:
    arm_id: str
    group: str
    games: int
    iterations: int
    lr: float
    profile_path: str
    steps: int = 160

    @property
    def total_games(self) -> int:
        return self.games * self.iterations

    @property
    def total_steps(self) -> int:
        return self.steps * self.iterations


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def repo_file(value: object) -> Path:
    rel = Path(str(value))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe repository path: {rel}")
    path = (ROOT / rel).resolve()
    if ROOT.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Missing repository file: {rel}")
    return path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config_path = repo_file(path)
    config = read_json(config_path)
    if config.get("schema") != "gocube-torus9-m88-nightly-ab-v1":
        raise ValueError("Unsupported campaign schema")
    if config.get("base_code_commit") != BASE_COMMIT:
        raise ValueError("Campaign base commit drift")
    parent = config.get("parent")
    if not isinstance(parent, Mapping) or str(parent.get("lineage_id")) != "torus9-post-ab-m80-g128-20260918-v1" or int(parent.get("generation", -1)) != 88:
        raise ValueError("Campaign parent must be post-M80 M88")
    fixed = config.get("fixed")
    if not isinstance(fixed, Mapping):
        raise ValueError("fixed must be an object")
    expected = {
        "self_play_mcts_simulations": 128,
        "optimizer_steps_per_iteration": 160,
        "replay_generations": 6,
        "replay_cap": 40000,
        "batch_size": 64,
        "komi": 0.5,
    }
    for key, wanted in expected.items():
        if float(fixed.get(key, -1)) != float(wanted):
            raise ValueError(f"Fixed campaign parameter drift: {key}")
    config["_path"] = config_path.relative_to(ROOT).as_posix()
    arms_from_config(config)
    return config


def arms_from_config(config: Mapping[str, object]) -> dict[str, Arm]:
    raw_arms = config.get("arms")
    if not isinstance(raw_arms, list):
        raise ValueError("arms must be a list")
    arms: dict[str, Arm] = {}
    for raw in raw_arms:
        if not isinstance(raw, Mapping):
            raise ValueError("arm must be an object")
        arm = Arm(
            arm_id=str(raw["id"]),
            group=str(raw["group"]),
            games=int(raw["games_per_iteration"]),
            iterations=int(raw["iterations"]),
            lr=float(raw["learning_rate"]),
            profile_path=str(raw["profile_path"]),
        )
        if arm.arm_id in arms or arm.games <= 0 or arm.iterations <= 0 or arm.lr not in ALLOWED_LEARNING_RATES:
            raise ValueError(f"Invalid arm: {arm.arm_id}")
        repo_file(arm.profile_path)
        arms[arm.arm_id] = arm
    if len(arms) != 6:
        raise ValueError("Campaign must contain exactly six arms")
    games = [a for a in arms.values() if a.group == "games"]
    lrs = [a for a in arms.values() if a.group == "learning-rate"]
    if {(a.games, a.iterations, a.steps) for a in games} != {(128, 6, 160), (192, 4, 160), (384, 2, 160)}:
        raise ValueError("Games sweep must be 128x6 / 192x4 / 384x2 with 160 steps")
    if {a.total_games for a in games} != {768}:
        raise ValueError("Games sweep must produce 768 games per arm")
    if {(a.games, a.iterations, a.lr) for a in lrs} != {(128, 3, 0.0001), (128, 3, 0.0003), (128, 3, 0.001)}:
        raise ValueError("LR sweep must be 128 games x3 at 1e-4 / 3e-4 / 1e-3")
    return arms


def make_run_spec(config: Mapping[str, object], arm: Arm) -> StrictRunSpec:
    payload = deepcopy(read_json(repo_file(config["run_spec_template_path"])))
    profile = load_torus9_current_profile(repo_file(arm.profile_path))
    if float(profile["training"]["learning_rate"]) != arm.lr or int(profile["self_play"]["mcts_simulations"]) != 128:
        raise ValueError(f"Profile drift for {arm.arm_id}")
    payload["profile_path"] = arm.profile_path
    payload["expected_profile_fingerprint"] = str(profile["profile_fingerprint"])
    generation = dict(payload["generation"])
    generation["command"] = [".venv/bin/python", DRIVER, "generation", "--generation", "{generation}"]
    generation["resume_command"] = [".venv/bin/python", DRIVER, "generation", "--generation", "{generation}", "--resume"]
    driver_config = dict(generation["driver_config"])
    driver_config["games"] = arm.games
    driver_config["optimizer_steps_per_iteration"] = arm.steps
    generation["driver_config"] = driver_config
    payload["generation"] = generation
    payload["experiment"] = {
        "kind": "gocube-torus9-m88-nightly-ab-v1",
        "arm_id": arm.arm_id,
        "group": arm.group,
        "games_per_iteration": arm.games,
        "iterations": arm.iterations,
        "optimizer_steps_per_iteration": arm.steps,
        "learning_rate": arm.lr,
        "total_self_play_games": arm.total_games,
        "total_optimizer_steps": arm.total_steps,
        "shared_parent_generation": 88,
    }
    with tempfile.TemporaryDirectory(prefix="m88-nightly-spec-") as tmp:
        path = Path(tmp) / "run-spec.json"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return StrictRunSpec.load(path, repo_root=ROOT)


def run_arm(campaign_id: str, config: Mapping[str, object], arm: Arm, parent: Mapping[str, object]) -> dict[str, object]:
    lineage_id = f"{campaign_id}-{arm.arm_id}"
    run = StrictProductionTrainingOrchestrator(
        repo_root=ROOT,
        run_spec=make_run_spec(config, arm),
        lineage_id=lineage_id,
        terminal=True,
        child_lifecycle_policy=CodeUpdateProvenancePolicy(),
    )
    target = int(parent["generation"]) + arm.iterations
    print(f"[TRAIN] {arm.arm_id}: games={arm.games} iters={arm.iterations} steps=160 lr={arm.lr:g} -> M{target}", flush=True)
    if run.paths.root.exists():
        manifest = read_json(run.paths.manifest)
        if not h._same_parent(manifest.get("parent_checkpoint"), parent):
            raise ValueError(f"{lineage_id}: parent mismatch")
        status = run.status()
        committed = int(status.get("last_committed_generation", 0))
        if committed < target:
            state = str(status.get("state"))
            if state in {"SOFT_STOPPED", "RECOVERY_REQUIRED", "COMPLETED"}:
                run.prepare_resume()
            elif state != "CREATED":
                raise RuntimeError(f"{lineage_id}: unsafe resume state {state!r}")
            run.run(max_generations=target)
    else:
        run.create(parent_checkpoint=parent)
        run.run(max_generations=target)
    if int(run.status().get("last_committed_generation", 0)) != target:
        raise RuntimeError(f"{lineage_id} did not reach M{target}")
    manifest = read_json(run.paths.manifest)
    rel = f"checkpoints/M{target}.pt"
    hashes = manifest.get("checkpoint_hashes")
    if not isinstance(hashes, Mapping) or rel not in hashes:
        raise ValueError(f"{lineage_id}: final checkpoint identity missing")
    return {
        "arm_id": arm.arm_id,
        "group": arm.group,
        "games_per_iteration": arm.games,
        "iterations": arm.iterations,
        "optimizer_steps_per_iteration": arm.steps,
        "learning_rate": arm.lr,
        "lineage_id": lineage_id,
        "generation": target,
        "checkpoint": str(run.paths.root / rel),
        "sha256": str(hashes[rel]),
        "budget": {
            "self_play_games": arm.total_games,
            "optimizer_steps": arm.total_steps,
            "sample_exposures": arm.total_steps * 64,
        },
    }


def run_arenas(config: Mapping[str, object], trained: Mapping[str, Mapping[str, object]], parent: Mapping[str, object]) -> dict[str, dict[str, object]]:
    arena = config.get("arena")
    if not isinstance(arena, Mapping) or not isinstance(arena.get("execution"), Mapping) or not isinstance(arena.get("scientific_contract"), Mapping):
        raise ValueError("Malformed Arena config")
    spec = {
        "kind": "gocube-torus9-m88-nightly-ab-v1",
        "arena_execution": dict(arena["execution"]),
        "arena_scientific_contract": dict(arena["scientific_contract"]),
    }
    all_results: dict[str, Mapping[str, object]] = dict(trained)
    all_results["parent"] = {
        "lineage_id": str(parent["lineage_id"]),
        "generation": int(parent["generation"]),
        "checkpoint": str(parent["path"]),
        "sha256": str(parent.get("artifact_sha256") or parent["sha256"]),
    }
    games, seed = int(arena["games"]), int(arena["master_seed"])
    results: dict[str, dict[str, object]] = {}
    raw_evals = config.get("evaluations")
    if not isinstance(raw_evals, list):
        raise ValueError("evaluations must be a list")
    for raw in raw_evals:
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation must be an object")
        eid, candidate, reference = str(raw["id"]), str(raw["candidate"]), str(raw["reference"])
        print(f"[ARENA] {eid}: {games} games", flush=True)
        results[eid] = h._compare(
            spec,
            h.Evaluation(eid, candidate, reference, games, seed),
            all_results,
        )
    return results


def write_report(campaign_id: str, config: Mapping[str, object], parent: Mapping[str, object], trained: Mapping[str, Mapping[str, object]], arenas: Mapping[str, Mapping[str, object]]) -> Path:
    output = evaluation_dir("torus9", f"{campaign_id}-campaign")
    output.mkdir(parents=True, exist_ok=True)
    h._write_json(output / "summary.json", {
        "schema": "gocube-torus9-m88-nightly-ab-result-v1",
        "campaign_id": campaign_id,
        "base_code_commit": BASE_COMMIT,
        "config_path": config["_path"],
        "shared_parent": {
            "lineage_id": parent["lineage_id"],
            "generation": parent["generation"],
            "checkpoint": parent["path"],
            "sha256": parent.get("artifact_sha256") or parent["sha256"],
        },
        "arms": dict(trained),
        "evaluations": dict(arenas),
    })
    lines = [
        "# Torus9 M88 nightly A/B", "",
        f"Campaign: `{campaign_id}`", f"Parent: `{parent['lineage_id']}` M{parent['generation']}", "",
        "## Training arms", "",
        "| Arm | Games/iter | Iters | Steps/iter | LR | Final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm_id, result in trained.items():
        lines.append(f"| {arm_id} | {result['games_per_iteration']} | {result['iterations']} | 160 | {result['learning_rate']} | M{result['generation']} |")
    lines += ["", "## Arena", "", "| Comparison | W/L/D |", "|---|---:|"]
    for eid, result in arenas.items():
        wld = result.get("W/L/D")
        text = "/".join(str(int(x)) for x in wld) if isinstance(wld, list) and len(wld) == 3 else "N/A"
        lines.append(f"| {eid} | {text} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def run_campaign(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    install_nightly_profile_validation()
    install_operator_policy()
    config = load_config(config_path)
    parent_cfg = config["parent"]
    assert isinstance(parent_cfg, Mapping)
    parent = h._source_parent_reference(str(parent_cfg["lineage_id"]), int(parent_cfg["generation"]))
    campaign_id = str(config["campaign_id"])
    print(f"Campaign {campaign_id}; parent M88; training sims=128; Arena keeps the standard 64-sim contract.", flush=True)
    trained = {
        arm.arm_id: run_arm(campaign_id, config, arm, parent)
        for arm in arms_from_config(config).values()
    }
    arenas = run_arenas(config, trained, parent)
    report = write_report(campaign_id, config, parent, trained, arenas)
    print(f"[DONE] {report / 'summary.md'}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    run_campaign(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
