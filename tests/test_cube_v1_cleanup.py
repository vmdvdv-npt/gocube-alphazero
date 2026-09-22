from __future__ import annotations

import importlib.util
from pathlib import Path

import gocube_golden


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MODULES = (
    "gocube_golden.cube_contract",
    "gocube_golden.cube_evaluation",
    "gocube_golden.cube_neural",
    "gocube_golden.cube_selfplay",
    "gocube_golden.cube_topology",
    "gocube_golden.cube_training",
    "gocube_golden.cube_training_adapter",
)
HISTORICAL_ROOT_SYMBOLS = (
    "CUBE4_TOPOLOGY",
    "CUBE_ACTION_COUNT",
    "CUBE_SELFPLAY_CONTRACT_ID",
    "CUBE_TARGET_CONTRACT_ID",
    "CUBE_TARGET_FINGERPRINT",
    "CubeSelfPlayAdapter",
    "CubeTrainingAdapter",
    "GoldenCubeGraphNetV1",
    "GoldenCubeNeuralEvaluator",
    "build_cube_action_mask",
    "build_cube_observation",
    "cube_model_hash",
    "run_cube_selfplay_games",
    "run_cube_selfplay_games_shared",
    "run_cube_training_iteration",
)
CURRENT_CUBE_MODULES = (
    "gocube_golden.cube_game_contract_v2",
    "gocube_golden.cube_family",
    "gocube_golden.cube_observation_v2",
    "gocube_golden.cube_network_v2",
)
FORBIDDEN_PRODUCTION_TOKENS = (
    "GoldenCubeGraphNetV1",
    "GoldenCubeNeuralEvaluator",
    "CubeSelfPlayAdapter",
    "CubeTrainingAdapter",
    "CUBE4_TOPOLOGY",
    "gocube_golden.cube_contract",
    "gocube_golden.cube_evaluation",
    "gocube_golden.cube_neural",
    "gocube_golden.cube_selfplay",
    "gocube_golden.cube_topology",
    "gocube_golden.cube_training",
    "gocube_golden.cube_training_adapter",
    "gocube-cube4-golden-training-v1",
    "cube4_golden_training_v1.json",
)


def test_historical_cube_v1_runtime_is_absent_from_production_api():
    for module_name in HISTORICAL_MODULES:
        assert importlib.util.find_spec(module_name) is None
    for symbol in HISTORICAL_ROOT_SYMBOLS:
        assert not hasattr(gocube_golden, symbol)


def test_current_cube_stage1_to_stage4_modules_remain_importable():
    for module_name in CURRENT_CUBE_MODULES:
        assert importlib.util.find_spec(module_name) is not None


def test_historical_cube_v1_training_profile_is_absent():
    assert not (ROOT / "configs" / "gocube" / "cube4_golden_training_v1.json").exists()


def test_production_source_has_no_historical_cube_v1_runtime_references():
    roots = (
        ROOT / "gocube_golden",
        ROOT / "alphazero" / "envs" / "gocube" / "integration",
    )
    failures = []
    for scan_root in roots:
        for path in sorted(scan_root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            matches = [token for token in FORBIDDEN_PRODUCTION_TOKENS if token in source]
            if matches:
                failures.append((str(path.relative_to(ROOT)), matches))
    assert failures == []
