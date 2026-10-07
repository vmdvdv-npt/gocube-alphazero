"""Strict operator parameters compiled into the existing V2 workflow.

This module does not run engines, create execution authority or send messages.
All generated actions use the existing continuous, Arena and experiment runners.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import math
from pathlib import Path
import re

from ..artifact_graph import CheckpointNode, EffectiveConfig
from ..artifact_resolver import ArtifactResolver
from ..provenance import file_sha256
from ..torus9_pcr import resolve_search_mode
from ..torus9_five_channel_training import SCHEMA as DRIVER, validate_config
from .continuous_training import ContinuousTrainingConfig
from .experiment_plan import ExperimentConfig
from .workflow import WorkflowSpec

SCHEMA = "gocube-operator-job-v1"
_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TRAINING_DEFAULTS = {
    "iterations": 5, "games_per_iteration": 384, "mcts_simulations": 200,
    "learning_rate": 0.00005, "updates_per_iteration": 160,
    "batch_size": 64, "gradient_clip": 1.0, "replay_generations": 6,
}
ARENA_DEFAULTS = {"every_iterations": 5, "games": 192, "mcts_simulations": 128}
EXECUTION_DEFAULTS = {"device": "cuda", "workers": 16}
ARENA_RUN_MASTER_SEED = 2026092902


def _object(value, allowed, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{label}: unknown fields: {', '.join(sorted(unknown))}")
    return dict(value)


def _name(value, label):
    if not isinstance(value, str) or not _COMPONENT.fullmatch(value):
        raise ValueError(f"{label} must be a simple name (letters, digits, dash, underscore, dot)")
    return value


def _selector(value, label):
    if not isinstance(value, str) or len(value.split("/")) != 2:
        raise ValueError(f"{label} must be lineage/checkpoint_id, not a path or executable")
    lineage, checkpoint = value.split("/")
    _name(lineage, f"{label} lineage")
    _name(checkpoint, f"{label} checkpoint")
    return value


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _tree_reuse(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _training(value, *, defaults=TRAINING_DEFAULTS, allow_iterations=True):
    allowed = (set(TRAINING_DEFAULTS) | {"replay_sampling"}) - (set() if allow_iterations else {"iterations"})
    result = {**defaults, **_object(value, allowed, "training")}
    for key, number in result.items():
        if key == "replay_sampling":
            from ..policy_surprise import sampling_setting
            result[key] = sampling_setting(number)
            continue
        if key == "iterations" and number is None:
            # The public operator job uses JSON null for the V2 continuous
            # runner's existing unbounded generation budget.
            continue
        elif key == "learning_rate":
            if type(number) not in (float, int) or not math.isfinite(number) or number <= 0:
                raise ValueError("learning_rate must be a finite positive number")
        elif key == "gradient_clip":
            if type(number) not in (float, int) or not math.isfinite(number) or number <= 0:
                raise ValueError("gradient_clip must be a finite positive number")
        else:
            _integer(number, key, minimum=0 if key == "iterations" else 1)
    return result


def _arena(value, *, defaults=ARENA_DEFAULTS):
    result = {**defaults, **_object(value, {*ARENA_DEFAULTS, "tree_reuse"}, "arena")}
    for key, number in result.items():
        if key == "tree_reuse":
            _tree_reuse(number, "arena.tree_reuse")
            continue
        _integer(number, "arena." + key)
    if result["games"] % 2 or result["games"] < 64:
        raise ValueError("Production arena requires an even games count >= 64")
    return result


def _arena_runs(value, *, defaults):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("arenas must be a list")
    normalized = []
    names = set()
    allowed = {"id", "candidate", "reference", "games", "mcts_simulations", "master_seed", "tree_reuse"}
    for index, raw_item in enumerate(value):
        item = _object(raw_item, allowed, f"arenas[{index}]")
        arena_id = _name(item.get("id"), f"arenas[{index}].id")
        if arena_id in names:
            raise ValueError(f"Duplicate Arena id: {arena_id}")
        names.add(arena_id)
        candidate = _selector(item.get("candidate"), f"arenas[{index}].candidate")
        reference = _selector(item.get("reference"), f"arenas[{index}].reference")
        games = _integer(item.get("games", defaults["games"]), f"arenas[{index}].games")
        if games % 2 or games < 64:
            raise ValueError(f"arenas[{index}].games must be an even integer >= 64")
        simulations = _integer(
            item.get("mcts_simulations", defaults["mcts_simulations"]),
            f"arenas[{index}].mcts_simulations",
        )
        seed = _integer(
            item.get("master_seed", ARENA_RUN_MASTER_SEED),
            f"arenas[{index}].master_seed",
            minimum=0,
        )
        reuse = _tree_reuse(item.get("tree_reuse", defaults.get("tree_reuse", False)), f"arenas[{index}].tree_reuse")
        normalized.append(
            {
                "id": arena_id,
                "candidate": candidate,
                "reference": reference,
                "games": games,
                "mcts_simulations": simulations,
                "master_seed": seed,
                **({"tree_reuse": reuse} if "tree_reuse" in item or "tree_reuse" in defaults else {}),
            }
        )
    return normalized


def parse_job(value):
    raw = _object(
        value,
        {
            "schema", "run_id", "topology", "parent", "training", "arena",
            "execution", "ab_tests", "arenas", "self_play", "winner_selection",
        },
        "job",
    )
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"job.schema must be {SCHEMA}")
    _name(raw.get("run_id"), "run_id")
    if raw.get("topology", "torus9") != "torus9":
        raise ValueError(
            "The simple operator job currently supports Torus9 5CH; "
            "other topologies use the existing V2 run-spec"
        )

    self_play = None
    if 'self_play' in raw:
        self_play = _object(raw['self_play'], {'search_mode', 'pcr', 'master_seed', 'tree_reuse'}, 'self_play')
        if 'tree_reuse' in self_play:
            _tree_reuse(self_play['tree_reuse'], 'self_play.tree_reuse')
        if 'master_seed' in self_play:
            _integer(self_play['master_seed'], 'self_play.master_seed', minimum=0)
        resolve_search_mode(self_play)
    arena = _arena(raw.get("arena", {}))
    arenas = _arena_runs(raw.get("arenas", []), defaults=arena)
    winner_selection = None
    if raw.get("winner_selection") is not None:
        selection = _object(raw["winner_selection"],
                            {"candidate", "reference", "games", "mcts_simulations", "master_seed", "tree_reuse"},
                            "winner_selection")
        winner_selection = _arena_runs(
            [{"id": "winner-selection", **selection}], defaults=arena)[0]
        if winner_selection["candidate"] == winner_selection["reference"]:
            raise ValueError("winner_selection must compare two distinct checkpoints")
        if raw.get("parent") is not None:
            raise ValueError("winner_selection replaces parent; do not specify both")
        if raw.get("ab_tests"):
            raise ValueError("winner_selection cannot be combined with ab_tests")

    # Arena-only jobs must not accidentally inherit the historical five-iteration
    # training default merely because the training section was omitted.
    arena_only_shape = bool(arenas) and "training" not in raw and "ab_tests" not in raw
    training_defaults = (
        {**TRAINING_DEFAULTS, "iterations": 0}
        if arena_only_shape
        else TRAINING_DEFAULTS
    )
    training = _training(raw.get("training", {}), defaults=training_defaults)
    if "replay_sampling" in training:
        raise ValueError("replay_sampling must be an offline arm setting")
    if winner_selection is not None and training["iterations"] == 0:
        raise ValueError("winner_selection requires positive training iterations or null")

    execution = {
        **EXECUTION_DEFAULTS,
        **_object(raw.get("execution", {}), EXECUTION_DEFAULTS, "execution"),
    }
    if execution != EXECUTION_DEFAULTS:
        raise ValueError("Production Torus9 job requires device=cuda and workers=16")

    tests = raw.get("ab_tests", [])
    if not isinstance(tests, list):
        raise ValueError("ab_tests must be a list")
    normalized_tests, names = [], set()
    for test in tests:
        item = _object(
            test,
            {"id", "iterations", "A", "B", "arena", "arms", "offline_replay", "training_seed"},
            "ab_test",
        )
        name = _name(item.get("id"), "ab_test.id")
        if name in names:
            raise ValueError("Duplicate A/B test id")
        names.add(name)
        iterations = _integer(item.get("iterations"), "ab_test.iterations")
        offline = item.get("offline_replay")
        training_seed = None
        if "training_seed" in item:
            if offline is None:
                raise ValueError("ab_test.training_seed is supported only for offline A/B tests")
            training_seed = _integer(item["training_seed"], "ab_test.training_seed", minimum=0)
        arm_values = item.get("arms")
        if arm_values is not None:
            if offline is None or "A" in item or "B" in item:
                raise ValueError("named arms are supported only for offline A/B tests")
            if not isinstance(arm_values, dict) or len(arm_values) < 2:
                raise ValueError("offline arms must contain at least two variants")
            for arm_name in arm_values:
                _name(arm_name, "offline arm id")
        else:
            arm_values = {arm: item.get(arm) for arm in ("A", "B")}
        if offline is not None:
            if training["iterations"] != 0:
                raise ValueError("offline A/B tests require training.iterations=0")
            if not isinstance(offline, list) or len(offline) != iterations:
                raise ValueError("offline_replay must specify one checkpoint per iteration")
            offline = [_selector(v, "offline_replay checkpoint") for v in offline]
        if arm_values is None or any(v is None for v in arm_values.values()):
            raise ValueError("Every A/B test must specify A and B training overrides")
        arms = {
            arm: _training(
                arm_values[arm],
                defaults={**training, "iterations": iterations},
                allow_iterations=False,
            )
            for arm in arm_values
        }
        for arm in arms.values():
            arm.pop("iterations")
            if "replay_sampling" in arm and offline is None:
                raise ValueError("replay_sampling is supported only for offline experiments")
        test_arena = _object(
            item.get("arena", {}),
            {"games", "mcts_simulations", "tree_reuse", "master_seed"},
            "ab_test.arena",
        )
        arena_master_seed = None
        if "master_seed" in test_arena:
            arena_master_seed = _integer(
                test_arena.pop("master_seed"),
                "ab_test.arena.master_seed",
                minimum=0,
            )
        checked_arena = _arena(test_arena, defaults=arena)
        checked_arena.pop("every_iterations")
        if arena_master_seed is not None:
            checked_arena["master_seed"] = arena_master_seed
        normalized_test = {
            "id": name,
            "iterations": iterations,
            "arena": checked_arena,
            **({"offline_replay": offline, "arms": arms} if offline is not None else arms),
        }
        if training_seed is not None:
            normalized_test["training_seed"] = training_seed
        normalized_tests.append(normalized_test)

    if training["iterations"] is None and normalized_tests:
        raise ValueError("Unbounded training cannot be followed by A/B tests")

    parent = raw.get("parent")
    training_requested = training["iterations"] != 0
    needs_parent = training_requested or bool(normalized_tests)
    if needs_parent and winner_selection is None:
        parent = _selector(parent, "parent")
    elif parent is not None:
        parent = _selector(parent, "parent")

    if not training_requested and not normalized_tests and not arenas:
        raise ValueError(
            "Job has no work: specify training iterations, A/B tests, or arenas"
        )

    return {
        **({"winner_selection": {k: v for k, v in winner_selection.items() if k != "id"}}
           if winner_selection is not None else {}),
        **({"self_play": self_play} if self_play is not None else {}),
        "schema": SCHEMA,
        "run_id": raw["run_id"],
        "topology": "torus9",
        "parent": parent,
        "training": training,
        "arena": arena,
        "execution": execution,
        "ab_tests": normalized_tests,
        "arenas": arenas,
    }


def resolve_checkpoint(selector, *, resolver):
    lineage, checkpoint = selector.split("/")
    matches = []
    for state in ("active", "archive"):
        root = resolver.runs_root / "torus9" / state / lineage
        path = root / "metadata" / "checkpoints" / (checkpoint + ".json")
        if path.is_file():
            matches.append(CheckpointNode.from_dict(json.loads(path.read_text())).checkpoint)
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint {selector!r} must identify exactly one registered checkpoint"
        )
    ref = matches[0]
    if (
        ref.topology != "torus9"
        or ref.lineage_id != lineage
        or ref.checkpoint_id != checkpoint
    ):
        raise ValueError("Checkpoint selector disagrees with registered graph identity")
    return resolver.checkpoint(ref)


def resolve_parent(selector, *, resolver):
    # Compatibility name retained for callers/tests that patch the historical helper.
    return resolve_checkpoint(selector, resolver=resolver)


def _require_five_channel_checkpoint(node, label):
    config = node.effective_config.config
    if config.topology != "torus9":
        raise ValueError(f"{label} must be a Torus9 checkpoint")
    if config.compatibility.get("input_channels") != 5:
        raise ValueError(f"{label} must use the current 5-channel Torus9 network")


def _base_config(parent):
    """Inherit ordinary config, or bind the completed adaptation's replay."""
    effective = parent.effective_config.config
    if effective.extensions.get("training_driver") == DRIVER:
        config = effective.to_dict()
        # Offline replay is an experiment input, never an inherited training mode.
        for key in ("offline_ab_replay", "policy_surprise_spec", "policy_surprise_cache"):
            config["extensions"].pop(key, None)
        config["replay"].pop("sampling", None)
        return config
    root = parent.owner_root
    metadata_path = parent.path.with_suffix(".metadata.json")
    meta = json.loads(metadata_path.read_text())
    if (
        meta.get("adaptation_update") != 2400
        or meta.get("observation_shape") != [5, 81]
        or meta.get("komi") != 1.5
        or meta.get("artifact_sha256") != parent.ref.sha256
    ):
        raise ValueError("Parent must be a completed komi=1.5 5CH adaptation or ordinary checkpoint")
    state = json.loads((root / "state.json").read_text())
    requested = set(meta["replay_ids"])
    shards = [s for s in state["shards"] if s["sha"] in requested]
    if {s["sha"] for s in shards} != requested:
        raise ValueError("Adaptation replay references are incomplete")

    def shard_ref(shard):
        path = Path(shard["path"]).resolve()
        path.relative_to(root.resolve())
        if file_sha256(path) != shard["sha"]:
            raise ValueError("Adaptation replay SHA mismatch")
        return {"path": str(path), "sha": shard["sha"]}

    buckets = [
        {
            "generation": g,
            "shards": [shard_ref(s) for s in shards if s["generation"] == g],
        }
        for g in sorted({s["generation"] for s in shards})
    ]
    validation = [
        {
            "generation": 0,
            "shards": [
                shard_ref(s) for s in state["shards"] if s["generation"] == 0
            ],
        }
    ]
    return EffectiveConfig(
        topology="torus9",
        compatibility={
            "input_channels": 5,
            "observation_shape": [5, 81],
            "architecture_id": meta["architecture_id"],
            "target_fingerprint": meta["target_fingerprint"],
        },
        self_play={"komi": 1.5},
        training={
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "l2_sp": False,
            "gradient_clip": 1.0,
            "lr_scheduler": None,
        },
        replay={"cap": None, "policy": "rolling-recent-generations"},
        execution={
            "device": "cuda",
            "workers": 16,
            "active_games_per_worker": 4,
            "inference_batch_cap": 64,
            "inference_batch_wait_ms": 1.0,
            "training_master_seed": int(
                json.loads((root / "config.json").read_text())["seed"]
            ),
            "selfplay_master_seed": 2026092901,
        },
        extensions={
            "training_driver": DRIVER,
            "adaptation_parent": parent.ref.to_dict(),
            "initial_replay_buckets": buckets,
            "validation_buckets": validation,
        },
    ).to_dict()


def _effective(base, training, arena, self_play=None, offline=None):
    cfg = copy.deepcopy(base)
    # A new job defaults to fixed even when its parent used PCR.
    cfg['self_play'].pop('pcr', None)
    cfg['self_play'].pop('search_mode', None)
    # Reuse always requires explicit opt-in for the new job.
    cfg['self_play'].pop('tree_reuse', None)
    if self_play is not None:
        settings = copy.deepcopy(self_play)
        if 'master_seed' in settings:
            # The engine already owns this seed in the execution contract.
            # Keep the learner seed and inherited Adam state unchanged.
            cfg['execution']['selfplay_master_seed'] = settings.pop('master_seed')
        cfg['self_play'].update(settings)
    cfg["self_play"].update(
        games_per_iteration=training["games_per_iteration"],
        mcts_simulations=training["mcts_simulations"],
    )
    cfg["training"].update(
        learning_rate=training["learning_rate"],
        batch_size=training["batch_size"],
        optimizer_steps_per_iteration=training["updates_per_iteration"],
        gradient_clip=training["gradient_clip"],
    )
    cfg["replay"].pop("sampling", None)
    if "replay_sampling" in training:
        cfg["replay"]["sampling"] = training["replay_sampling"]
    cfg["replay"].update(generations=training["replay_generations"], cap=None)
    cfg["execution"].update(EXECUTION_DEFAULTS)
    cfg["arena"] = {
        "komi": 1.5,
        "games": arena["games"],
        "simulations": arena["mcts_simulations"],
        "every_generations": arena["every_iterations"],
        "reference_gap": arena["every_iterations"],
        "cpuct": 1.25,
        "fpu": 0.0,
        "watchdog": 1000,
        "diagnostic_only": True,
        "gating": False,
        **({"tree_reuse": arena["tree_reuse"]} if "tree_reuse" in arena else {}),
    }
    if offline is not None:
        cfg["extensions"]["offline_ab_replay"] = offline
    result = EffectiveConfig.from_dict(cfg)
    validate_config(result)
    return result.to_dict()


def _offline_effective(base, training, arena, offline, self_play=None, training_seed=None):
    cfg = _effective(base, training, arena, self_play, offline=offline)
    if offline is not None:
        if any(len(row["buckets"]) != training["replay_generations"] for row in offline):
            raise ValueError("offline replay window differs from requested replay_generations")
        cfg["extensions"]["offline_ab_replay"] = offline
    if training_seed is not None:
        cfg["execution"]["training_master_seed"] = training_seed
    return cfg


def _arena_execution(arena):
    return {
        "games": arena["games"],
        "workers": 16,
        "games_per_worker": 4,
        "inference_batch_rows": 64,
        "inference_batch_wait_ms": 1.0,
        "device": "cuda",
        "strict_production": True,
        "monitoring_acceptance": False,
        "min_mean_inference_batch_rows": 0.0,
        "min_effective_cpu_cores": 0.0,
        "early_gate_enabled": False,
    }


def _arena_profile(arena):
    return (
        f"torus9|komi=1.5|simulations={arena['mcts_simulations']}"
        "|cpuct=1.25|fpu=0|watchdog=1000|5ch"
        + ("|tree_reuse=true" if arena.get("tree_reuse", False) else "")
    )


def _arena_step(job, item, candidate, reference, previous):
    arena = {
        "games": item["games"],
        "mcts_simulations": item["mcts_simulations"],
        **({"tree_reuse": item["tree_reuse"]} if "tree_reuse" in item else {}),
    }
    step_id = "arena-" + item["id"]
    config = {
        "candidate_checkpoint": candidate.ref.to_dict(),
        "reference_checkpoint": reference.ref.to_dict(),
        "arena_config": _arena_execution(arena),
        "profile": _arena_profile(arena),
        "master_seed": item["master_seed"],
        "candidate_label": candidate.ref.checkpoint_id,
        "reference_label": reference.ref.checkpoint_id,
        "comparison": f"{job['run_id']}:{item['id']}",
        "workload": {
            "mode": "diagnostic-only",
            "gating": "off",
            "paired_starts": True,
            "color_swap": True,
            "operator_job_id": job["run_id"],
            "arena_id": item["id"],
        },
    }
    return {
        "step_id": step_id,
        "action": "arena",
        "dependencies": [previous] if previous else [],
        "config": config,
    }


def compile_job(value, *, runs_root=None, resolver=None):
    """Read/verify input identities and return a deterministic standard workflow."""
    job = parse_job(value)
    resolver = resolver or ArtifactResolver(runs_root)
    steps = []
    previous = None

    parent = None
    base = None
    selection = job.get("winner_selection")
    if selection is not None:
        item = {"id": "winner-selection", **selection}
        candidates = {}
        for role in ("candidate", "reference"):
            node = resolve_checkpoint(item[role], resolver=resolver)
            _require_five_channel_checkpoint(node, "Winner selection " + role)
            candidates[role] = node
        step = _arena_step(job, item, candidates["candidate"], candidates["reference"], None)
        steps.append(step)
        choices = {}
        for role, node in candidates.items():
            effective = _effective(_base_config(node), job["training"], job["arena"], job.get("self_play"))
            choices[role] = {"checkpoint": node.ref.to_dict(), "effective_config": effective}
            ContinuousTrainingConfig(
                parent_checkpoint=node.ref.to_dict(), lineage_id=job["run_id"],
                effective_config=effective, generations=job["training"]["iterations"],
                arena_cadence=job["arena"]["every_iterations"],
                arena_config=_arena_execution(job["arena"]),
                arena_profile=_arena_profile(job["arena"]),
            )
        steps.append({"step_id": "winner", "action": "select",
                      "dependencies": [step["step_id"]], "config": {
                          "rule": "arena-winner",
                          "arena_result": {"$ref": step["step_id"] + ".outputs"},
                          "expected_games": item["games"], "candidates": choices}})
        previous = "winner"
    training_requested = job["training"]["iterations"] != 0
    if training_requested or job["ab_tests"]:
        arena = job["arena"]
        if selection is not None:
            main_config = {"$ref": "winner.outputs.selected_effective_config"}
            parent_ref = {"$ref": "winner.outputs.selected"}
        else:
            parent = resolve_parent(job["parent"], resolver=resolver)
            base = _base_config(parent)
            if base["compatibility"].get("input_channels") != 5:
                raise ValueError("Simple jobs cannot launch retired 6-channel training")
            main_config = _effective(base, job["training"], arena, job.get("self_play"))
            parent_ref = parent.ref.to_dict()
        if training_requested:
            config = {
                "parent_checkpoint": parent_ref,
                "lineage_id": job["run_id"],
                "effective_config": main_config,
                "generations": job["training"]["iterations"],
                "arena_cadence": arena["every_iterations"],
                "arena_reference_gap": arena["every_iterations"],
                "arena_config": _arena_execution(arena),
                "arena_profile": _arena_profile(arena),
                "arena_master_seed": ARENA_RUN_MASTER_SEED,
                "arena_workload": {
                    "mode": "diagnostic-only",
                    "gating": "off",
                    "paired_starts": True,
                    "color_swap": True,
                },
            }
            if selection is None:
                ContinuousTrainingConfig(**config)
            steps.append(
                {
                    "step_id": "training",
                    "action": "continuous_training",
                    **({"dependencies": [previous]} if previous else {}),
                    "config": config,
                }
            )
            parent_ref = {"$ref": "training.outputs.checkpoint"}
            previous = "training"

        for test in job["ab_tests"]:
            step_id = "ab-" + test["id"]
            test_arena = {
                **test["arena"],
                "every_iterations": test["iterations"],
            }
            offline = None
            if "offline_replay" in test:
                from .offline_replay import resolve_offline_replay
                offline = resolve_offline_replay(test["offline_replay"], parent=parent, resolver=resolver)
            surprise_spec = None
            weights = {arm["replay_sampling"]["weight"] for arm in test.get("arms", {key:test[key] for key in ("A", "B") if key in test}).values()
                       if arm.get("replay_sampling", {}).get("mode") == "policy_surprise"}
            if weights:
                if len(weights) != 1:
                    raise ValueError("One historical surprise weight per experiment is supported")
                from ..policy_surprise import resolve_spec
                surprise_spec = resolve_spec(offline, parent=parent, resolver=resolver, weight=next(iter(weights)))
            config = {
                "experiment_id": job["run_id"] + "-" + step_id,
                "topology": "torus9",
                "parent": parent_ref,
                "arms": [
                    {
                        "arm_id": arm,
                        "generations": test["iterations"],
                        "lineage_id": job["run_id"] + "-" + step_id + "-" + arm,
                        "config": _offline_effective(
                            base,
                            test.get("arms", test)[arm],
                            test_arena,
                            offline,
                            job.get("self_play"),
                            training_seed=test.get("training_seed"),
                        ),
                    }
                    for arm in test.get("arms", {"A": {}, "B": {}})
                ],
                "arena": {
                    "config": _arena_execution(test_arena),
                    "profile": _arena_profile(test_arena),
                    "master_seed": test_arena.get("master_seed", ARENA_RUN_MASTER_SEED),
                    "winner_rule": "candidate_if_wins_gt_losses_else_reference",
                },
            }
            if surprise_spec is not None:
                for arm in config["arms"]:
                    arm["config"]["extensions"]["policy_surprise_spec"] = surprise_spec
            # Validate all concrete budgets/contracts before training. Only the
            # future checkpoint identity is replaced here, never an arm setting.
            ExperimentConfig.from_dict({**config, "parent": parent.ref.to_dict()})
            steps.append(
                {
                    "step_id": step_id,
                    "action": "experiment",
                    "dependencies": [previous] if previous else [],
                    "config": config,
                }
            )
            previous = step_id

    # Arena-only (or post-training) comparisons are compiled into ordinary V2
    # workflow Arena actions. The operator plan stores immutable checkpoint
    # references with hashes; no checkpoint is copied into the evaluation.
    for item in job["arenas"]:
        candidate = resolve_checkpoint(item["candidate"], resolver=resolver)
        reference = resolve_checkpoint(item["reference"], resolver=resolver)
        _require_five_channel_checkpoint(candidate, "Arena candidate")
        _require_five_channel_checkpoint(reference, "Arena reference")
        if candidate.ref.topology != reference.ref.topology:
            raise ValueError("Arena candidate/reference topologies differ")
        step = _arena_step(job, item, candidate, reference, previous)
        steps.append(step)
        previous = step["step_id"]

    result = {
        "schema": "gocube-orchestrator-v2-run-spec-v1",
        "mode": "workflow",
        "workflow": {
            "workflow_id": job["run_id"],
            "topology": "torus9",
            "steps": steps,
        },
    }
    WorkflowSpec.from_dict(result["workflow"])
    return result
