from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
