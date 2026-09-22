from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import (
    ArtifactRef,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)
from gocube_golden.artifact_resolver import ResolvedArtifact, ResolvedEffectiveConfig
from gocube_golden.orchestrator_v2 import OutputLineage, ResolvedGenerationInput
from gocube_golden.torus9_contract import TORUS9_OPTIMIZER_STEPS_PER_ITERATION
from tools import torus9_run_driver as base
from tools import torus9_staged_sims_driver as staged


ROOT = Path(__file__).resolve().parents[1]


def test_standard_driver_uses_default_bindings() -> None:
    bindings = base.DEFAULT_DRIVER_BINDINGS
    assert bindings.training_adapter_factory is base.Torus9TrainingAdapter
    assert bindings.optimizer_steps_per_iteration == TORUS9_OPTIMIZER_STEPS_PER_ITERATION
    assert bindings.scientific_validator is base._validate_scientific_bindings


def test_standard_main_passes_default_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_generation(args: object, *, bindings: base.DriverBindings) -> dict[str, object]:
        seen["bindings"] = bindings
        return {"status": "COMPLETED"}

    monkeypatch.setattr(base, "run_generation", fake_generation)
    assert base.main(["generation", "--generation", "1"]) == 0
    assert seen["bindings"] is base.DEFAULT_DRIVER_BINDINGS


def test_v2_generation_config_accepts_backfilled_execution_seeds() -> None:
    config = base._v2_generation_config(
        {
            "self_play": {"games_per_iteration": 128, "mcts_simulations": 128},
            "training": {
                "optimizer_steps_per_iteration": 160,
                "learning_rate": 0.0003,
                "optimizer": "Adam",
                "batch_size": 64,
            },
            "replay": {"generations": 6, "cap": 40_000},
            "execution": {
                "device": "cuda",
                "workers": 16,
                "active_games_per_worker": 4,
                "active_contexts": 64,
                "inference_batch_cap": 64,
                "inference_batch_wait_ms": 1,
                "coalescing": True,
                "model_init_seed": 202609131001,
                "selfplay_master_seed": 202609131002,
                "training_master_seed": 202609131003,
            },
            "extensions": {},
        }
    )

    assert config["optimizer_steps_per_iteration"] == 160
    assert config["model_init_seed"] == 202609131001
    assert config["selfplay_master_seed"] == 202609131002
    assert config["training_master_seed"] == 202609131003


def test_v2_generation_config_applies_execution_only_concurrency_override() -> None:
    effective = {
        "self_play": {"games_per_iteration": 384, "mcts_simulations": 200},
        "training": {
            "optimizer_steps_per_iteration": 160,
            "learning_rate": 0.0001,
            "optimizer": "Adam",
            "batch_size": 64,
        },
        "replay": {"generations": 2, "cap": None},
        "execution": {
            "device": "cuda",
            "workers": 16,
            "active_games_per_worker": 4,
            "active_contexts": 64,
            "inference_batch_cap": 64,
            "inference_batch_wait_ms": 1,
            "coalescing": True,
            "model_init_seed": 1,
            "selfplay_master_seed": 2,
            "training_master_seed": 3,
        },
        "extensions": {},
    }
    config = base._v2_generation_config(
        effective,
        {
            "active_games_per_worker": 6,
            "total_active_contexts": 96,
        },
    )

    assert config["workers"] == 16
    assert config["active_games_per_worker"] == 6
    assert config["total_active_contexts"] == 96
    assert config["games"] == 384
    assert config["mcts_simulations"] == 200
    assert config["replay_generations"] == 2
    assert config["replay_cap"] is None


def _resolved_v2_generation_input(
    tmp_path: Path,
    execution_overrides: dict[str, object] | None,
) -> ResolvedGenerationInput:
    effective = EffectiveConfig(
        topology="torus9",
        compatibility={"profile_id": "test"},
        self_play={"games_per_iteration": 384, "mcts_simulations": 200},
        training={
            "optimizer_steps_per_iteration": 160,
            "learning_rate": 0.0001,
            "optimizer": "Adam",
            "batch_size": 64,
        },
        replay={"generations": 2, "cap": None},
        execution={
            "device": "cpu",
            "workers": 16,
            "active_games_per_worker": 4,
            "active_contexts": 64,
            "inference_batch_cap": 64,
            "inference_batch_wait_ms": 1,
            "coalescing": True,
            "model_init_seed": 1,
            "selfplay_master_seed": 2,
            "training_master_seed": 3,
        },
        extensions={},
    )
    effective_path = tmp_path / "metadata" / "effective.json"
    effective_artifact = ResolvedArtifact(
        ArtifactRef("metadata/effective.json", "sha256:" + "a" * 64),
        effective_path,
        tmp_path,
        "torus9",
        "child",
        "ACTIVE",
        {"immutable_verified": True},
    )
    resolved_effective = ResolvedEffectiveConfig(
        EffectiveConfigRef(effective_artifact.ref, effective.fingerprint),
        effective_artifact,
        effective,
    )
    parent_path = tmp_path / "parent.pt"
    parent_path.write_bytes(b"parent")
    parent_ref = CheckpointRef(
        "torus9",
        "parent",
        "M0",
        0,
        "checkpoints/M0.pt",
        "sha256:" + "b" * 64,
    )
    return ResolvedGenerationInput(
        parent_checkpoint=SimpleNamespace(path=parent_path, ref=parent_ref, generation=0),  # type: ignore[arg-type]
        generation=1,
        effective_config=resolved_effective,
        output_lineage=OutputLineage("torus9", "child", tmp_path / "lineage"),
        execution_overrides=execution_overrides,
    )


@pytest.mark.parametrize(
    ("execution_overrides", "expected_active_games", "expected_contexts"),
    (
        (None, 4, 64),
        ({"active_games_per_worker": 6, "total_active_contexts": 96}, 6, 96),
    ),
)
def test_v2_production_path_forwards_actual_selfplay_execution_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_overrides: dict[str, object] | None,
    expected_active_games: int,
    expected_contexts: int,
) -> None:
    """Exercise run_generation_v2 through the real call into self-play."""
    generation_input = _resolved_v2_generation_input(tmp_path, execution_overrides)
    effective_fingerprint = generation_input.effective_config.fingerprint
    captured: dict[str, object] = {}

    class ReachedSelfPlayBoundary(RuntimeError):
        pass

    def fake_prepare_state_v2(**kwargs: object):
        captured["config"] = dict(kwargs["config"])  # type: ignore[arg-type]
        return (
            SimpleNamespace(),
            SimpleNamespace(model=object(), completed_games=0),
            generation_input.parent_checkpoint.path,
        )

    def fake_selfplay(_model: object, **kwargs: object):
        captured["selfplay"] = kwargs
        raise ReachedSelfPlayBoundary

    monkeypatch.setattr(base, "_prepare_state_v2", fake_prepare_state_v2)
    monkeypatch.setattr(base, "_resolve_v2_replay_sources", lambda *_args: ((), (), (), None))
    monkeypatch.setattr(base, "_validate_code_pin", lambda _root: object())
    monkeypatch.setattr(base, "run_torus9_selfplay_games", fake_selfplay)

    with pytest.raises(ReachedSelfPlayBoundary):
        base.run_generation_v2(generation_input)

    selfplay = captured["selfplay"]
    assert isinstance(selfplay, dict)
    assert selfplay["workers"] == 16
    assert selfplay["active_games_per_worker"] == expected_active_games
    assert selfplay["total_active_contexts"] == expected_contexts
    assert selfplay["execution_override_reason"] == (
        "per-generation self-play concurrency sweep"
        if execution_overrides is not None
        else "immutable production run-spec"
    )
    assert generation_input.effective_config.fingerprint == effective_fingerprint
    assert generation_input.effective_config.config.execution["active_games_per_worker"] == 4
    assert generation_input.effective_config.config.execution["active_contexts"] == 64

    config = captured["config"]
    assert isinstance(config, dict)
    assert {
        key: config[key]
        for key in (
            "games",
            "mcts_simulations",
            "learning_rate",
            "optimizer",
            "optimizer_steps_per_iteration",
            "batch_size",
            "replay_generations",
            "replay_cap",
        )
    } == {
        "games": 384,
        "mcts_simulations": 200,
        "learning_rate": 0.0001,
        "optimizer": "Adam",
        "optimizer_steps_per_iteration": 160,
        "batch_size": 64,
        "replay_generations": 2,
        "replay_cap": None,
    }


def test_resume_state_uses_resolved_lineage_without_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AZ_LINEAGE_ID", raising=False)
    checkpoint = tmp_path / "checkpoints" / "M94.pt"
    replay = tmp_path / "replay" / "rolling-after-94.jsonl"
    checkpoint.parent.mkdir(parents=True)
    replay.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    replay.write_bytes(b"replay")

    loaded_state = SimpleNamespace(
        optimizer_updates=12,
        samples_consumed=34,
        rolling_replay=SimpleNamespace(last_generation=94),
    )
    config = {
        "model_init_seed": 101,
        "selfplay_master_seed": 202,
        "training_master_seed": 303,
    }

    resume_path = base._resume_state(
        root=tmp_path,
        lineage_id="torus9-v2-parity-m93-20260918-v1",
        generation=94,
        checkpoint=checkpoint,
        replay=replay,
        loaded_state=loaded_state,
        config=config,
    )
    payload = json.loads(resume_path.read_text(encoding="utf-8"))

    assert payload["rng"]["training_seed"] == base.derive_seed(
        303,
        "torus9-v2-parity-m93-20260918-v1",
        "training",
        94,
    )


@pytest.mark.parametrize(
    ("games", "optimizer_steps"),
    ((64, 80), (128, 160), (192, 240)),
)
def test_staged_configuration_builds_explicit_bindings_without_mutating_base(
    monkeypatch: pytest.MonkeyPatch,
    games: int,
    optimizer_steps: int,
) -> None:
    sentinel_spec = object()
    sentinel_factory = lambda **_kwargs: None  # noqa: E731
    captured: dict[str, int] = {}

    monkeypatch.setattr(base, "_load_run_spec", lambda: sentinel_spec)
    monkeypatch.setattr(
        base,
        "_generation_config",
        lambda spec: {
            "games": games,
            "optimizer_steps_per_iteration": optimizer_steps,
        },
    )

    def fake_adapter_type(steps: int):
        captured["optimizer_steps"] = int(steps)
        return sentinel_factory

    monkeypatch.setattr(staged, "_adapter_type", fake_adapter_type)
    before_adapter = base.Torus9TrainingAdapter
    before_steps = base.TORUS9_OPTIMIZER_STEPS_PER_ITERATION
    before_validator = base._validate_scientific_bindings

    bindings = staged.configure_from_run_spec()

    assert captured["optimizer_steps"] == optimizer_steps
    assert bindings.training_adapter_factory is sentinel_factory
    assert bindings.optimizer_steps_per_iteration == optimizer_steps
    assert bindings.scientific_validator is staged._validate_experiment_bindings
    assert base.Torus9TrainingAdapter is before_adapter
    assert base.TORUS9_OPTIMIZER_STEPS_PER_ITERATION == before_steps
    assert base._validate_scientific_bindings is before_validator


def test_staged_main_passes_bindings_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = base.DriverBindings(
        training_adapter_factory=base.Torus9TrainingAdapter,
        optimizer_steps_per_iteration=160,
        scientific_validator=staged._validate_experiment_bindings,
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(staged, "configure_from_run_spec", lambda: sentinel)

    def fake_main(argv: object = None, *, bindings: base.DriverBindings) -> int:
        seen["argv"] = argv
        seen["bindings"] = bindings
        return 0

    monkeypatch.setattr(base, "main", fake_main)
    argv = ["generation", "--generation", "48"]
    assert staged.main(argv) == 0
    assert seen == {"argv": argv, "bindings": sentinel}


def test_staged_scientific_validator_keeps_experiment_assumptions() -> None:
    profile = {
        "self_play": {"mcts_simulations": 128},
        "training": {"learning_rate": 0.0003, "batch_size": 64},
        "replay": {"generations": 6, "cap": 40_000},
        "network": {"hidden": 80, "blocks": 8},
        "seeds": {
            "model_init_seed": 101,
            "selfplay_master_seed": 202,
            "training_master_seed": 303,
        },
    }
    config = {
        "games": 128,
        "optimizer_steps_per_iteration": 160,
        "workers": 16,
        "active_games_per_worker": 4,
        "total_active_contexts": 64,
        "inference_batch_cap": 64,
        "inference_batch_wait_ms": 1.0,
        "coalescing": True,
        "model_init_seed": 101,
        "selfplay_master_seed": 202,
        "training_master_seed": 303,
    }

    staged._validate_experiment_bindings(profile, config)

    bad_network = {**profile, "network": {"hidden": 80, "blocks": 7}}
    with pytest.raises(ValueError, match="80x8"):
        staged._validate_experiment_bindings(bad_network, config)

    bad_execution = {**config, "total_active_contexts": 63}
    with pytest.raises(ValueError, match="execution drift"):
        staged._validate_experiment_bindings(profile, bad_execution)


def test_pr136_runtime_dependencies_are_not_monkey_patched() -> None:
    checks = {
        ROOT / "tools" / "torus9_staged_sims_driver.py": {"_base"},
        ROOT / "gocube_golden" / "code_update_policy.py": {
            "UniversalProductionTrainingOrchestrator"
        },
    }
    violations: list[str] = []
    for path, forbidden_roots in checks.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets.append(node.target)
            elif isinstance(node, ast.AugAssign):
                targets.append(node.target)
            for target in targets:
                if not isinstance(target, ast.Attribute):
                    continue
                owner = target.value
                if isinstance(owner, ast.Name) and owner.id in forbidden_roots:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: {owner.id}.{target.attr}"
                    )
    assert violations == []
