from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphazero.envs.gocube.b_experiment_contract import (
    B05_DRY_RUN_DEFAULT_SETTINGS,
    B05_DRY_RUN_TARGET,
    build_b_experiment_contract,
)
from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.katago_train import parse_args as katago_parse_args
from alphazero.envs.gocube.reproducible_manifest import effective_config
from alphazero.envs.gocube.production_training import SampleBudgetTarget
from tools import analyze_gocube_b_evaluation, gocube_b05, gocube_b_experiment
from tools.hardware_telemetry import HardwareTelemetry


def test_b05_contract_is_explicitly_non_scientific_and_keeps_batch_1024():
    contract = build_b_experiment_contract(
        heldout_suite_hash="a" * 64,
        scientific_target=SampleBudgetTarget("cumulative_new_samples", B05_DRY_RUN_TARGET),
        dry_run_settings=B05_DRY_RUN_DEFAULT_SETTINGS,
    )
    assert contract.non_scientific_dry_run is True
    assert contract.training_batch_size == 1024
    assert contract.worker_count == 2
    assert contract.generation_chunk_size == 4
    assert contract.scientific_sample_target["target"] == B05_DRY_RUN_TARGET


def test_b05_launcher_resume_preserves_hardened_path_and_batch():
    args = gocube_b_experiment.parse_args(
        [
            "--treatment", "B1", "--dry-run", "--b05-dry-run",
            "--b05-segment", "--b05-resume-segment", "--allow-existing-run",
        ]
    )
    command = gocube_b_experiment.training_command(args, python="python")
    assert "alphazero.envs.gocube.hardened_train" in command
    assert command[command.index("--train-batch-size") + 1] == "1024"
    assert "--b05-dry-run" in command
    assert "--b05-resume-segment" in command
    assert "--allow-existing-run" in command


def test_b05_resume_ignores_only_iteration_safety_ceiling_in_manifest():
    common = [
        "--model-profile", "baseline", "--topology", "cube", "--size", "4",
        "--workers", "2", "--sims", "50", "--arena-sims", "50",
        "--games-per-iteration", "4", "--train-batch-size", "1024",
        "--fast-game-prob", "0.25", "--train-samples-per-new-sample", "1",
        "--cumulative-new-samples-target", "512", "--run-name", "manifest-b05",
        "--no-arena", "--experiment-contract-id", "gocube-b-experiment-contract-v1",
        "--experiment-contract-sha256", "a" * 64, "--b05-dry-run", "--b05-segment",
    ]
    initial_cli = katago_parse_args(["--iterations", "1", *common])
    resume_cli = katago_parse_args(
        ["--iterations", "2", *common, "--b05-resume-segment", "--allow-existing-run"]
    )
    initial_game, initial_args = build_hardened_training_args(initial_cli)
    resume_game, resume_args = build_hardened_training_args(resume_cli)
    assert effective_config(initial_args, initial_game) == effective_config(resume_args, resume_game)

    production_cli = katago_parse_args(["--iterations", "1"])
    production_resume_cli = katago_parse_args(["--iterations", "2"])
    production_game, production_args = build_hardened_training_args(production_cli)
    production_resume_game, production_resume_args = build_hardened_training_args(production_resume_cli)
    assert effective_config(production_args, production_game)["numIters"] != effective_config(
        production_resume_args, production_resume_game
    )["numIters"]


def test_scientific_analyzer_rejects_b05_marker_before_other_fields():
    with pytest.raises(ValueError, match="B05 non-scientific"):
        analyze_gocube_b_evaluation.validate_seed_evaluation(
            {"schema_version": 1, "non_scientific_dry_run": True},
            evaluation_schedule={},
            experiment_contract_sha256="a" * 64,
        )


def test_hardware_telemetry_summarizes_temperature_and_power(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        json.dumps(
            {
                "phase": "TRAIN",
                "cpu_util_percent": 12,
                "gpus": [
                    {
                        "gpu_util_percent": 80,
                        "gpu_memory_used_mib": 1000,
                        "gpu_temperature_c": 61,
                        "gpu_power_w": 42,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = HardwareTelemetry(path).summary()
    assert summary["phases"]["TRAIN"]["gpu_temperature_c"]["max"] == 61
    assert summary["phases"]["TRAIN"]["gpu_power_w"]["max"] == 42


def test_hardware_telemetry_reports_peak_ram_and_swap(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        json.dumps(
            {
                "phase": "TRAIN",
                "ram_used_gib": 12.5,
                "ram_total_gib": 30.945,
                "ram_used_percent": 40.0,
                "swap_used_gib": 0.75,
                "swap_total_gib": 8.0,
                "swap_used_percent": 9.375,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = HardwareTelemetry(path).summary()
    assert summary["peaks"]["ram_used_gib"] == 12.5
    assert summary["peaks"]["swap_used_gib"] == 0.75
    assert summary["capacities"]["swap_total_gib"] == 8.0
    assert summary["phases"]["TRAIN"]["swap_used_percent"]["max"] == 9.375


def test_komi_audit_has_no_applicable_production_7_5_literal():
    audit = gocube_b05._komi_audit(Path(__file__).resolve().parents[1])
    assert audit["canonical_komi"] == 0.5
    assert audit["applicable_production_matches"] == []
