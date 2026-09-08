from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

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


def test_b05_checkpoint_metadata_is_json_safe(tmp_path):
    run_name = "json-safe"
    checkpoint_dir = tmp_path / "checkpoint" / run_name
    checkpoint_dir.mkdir(parents=True)
    args = {
        "gocube_b05_dry_run": True,
        "gocube_komi": 0.5,
        "gocube_topology": "cube",
        "gocube_size": 4,
        "train_batch_size": 1024,
        "gocube_experiment_contract_id": "gocube-b-experiment-contract-v1",
        "gocube_model_profile": "baseline",
        "nnet_type": type,
    }
    torch.save({"args": args, "state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_dir / "iteration-0001.pkl")
    metadata = gocube_b05._checkpoint_metadata(tmp_path, run_name)
    json.dumps(metadata)
    assert metadata["args"]["nnet_type"] == "builtins.type"


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


def test_throughput_summary_uses_median_and_preserves_canonical_workers():
    def repeat(workers, repeat, positions_per_second):
        return {
            "workers": workers,
            "repeat": repeat,
            "games": gocube_b05.THROUGHPUT_GAMES,
            "positions_per_second": positions_per_second,
        }

    summary = gocube_b05._summarize_throughput(
        [repeat(8, 1, 40.0), repeat(8, 2, 50.0), repeat(8, 3, 60.0)],
        [repeat(16, 1, 30.0), repeat(16, 2, 35.0), repeat(16, 3, 40.0)],
    )
    assert summary["benchmark"]["games_per_repeat"] == 64
    assert summary["workers_8"]["positions_per_second"]["median"] == 50.0
    assert summary["workers_16"]["positions_per_second"]["median"] == 35.0
    assert summary["comparison"]["relative_delta_percent_16_vs_8"] == pytest.approx(-30.0)
    assert summary["comparison"]["stable_workers_16_regression"] is True
    assert summary["comparison"]["canonical_workers"] == 16
    assert summary["comparison"]["canonical_workers_changed"] is False


def test_storage_extrapolation_scales_full_target_and_reports_logs(tmp_path):
    run_name = "storage-b0"
    checkpoint = tmp_path / "checkpoint" / run_name
    data = tmp_path / "data" / run_name
    runs = tmp_path / "runs" / run_name
    report_root = tmp_path / "report"
    (data / "records" / "iteration-0001").mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    runs.mkdir(parents=True)
    (report_root / "logs").mkdir(parents=True)
    for filename, size in (
        ("iteration-0000.pkl", 10),
        ("iteration-0001.pkl", 20),
        ("iteration-0002.pkl", 20),
        ("run-manifest.json", 5),
    ):
        (checkpoint / filename).write_bytes(b"x" * size)
    for filename, size in (
        ("iteration-0001-data.pkl", 100),
        ("iteration-0001-policy.pkl", 10),
        ("iteration-0001-complete.json", 5),
        ("training-progress.json", 5),
    ):
        (data / filename).write_bytes(b"x" * size)
    for index in (1, 2, 3, 4):
        (data / "records" / "iteration-0001" / f"C4-{index:06d}.json").write_bytes(b"x" * 25)
    (data / "records" / "iteration-0001" / "iteration-manifest.json").write_bytes(b"x" * 5)
    (runs / "events.out.tfevents.test").write_bytes(b"x" * 6)
    log = report_root / "logs" / "b0.log"
    log.write_bytes(b"x" * 8)

    details = {
        "run_name": run_name,
        "latest_iteration": 2,
        "counters": {
            "new_samples_accepted": 200,
            "selfplay_games_completed": 8,
        },
        "log_paths": [str(log)],
    }
    result = gocube_b05._storage_extrapolation(
        tmp_path,
        report_root,
        {gocube_b05.B0_TREATMENT: details, gocube_b05.B1_TREATMENT: details},
    )
    estimate = result["estimates"]["3+3"]
    assert result["scientific_target_new_samples_per_treatment"] == 40_000_000
    assert result["retention_policy"]["replay_window_max_iterations"] == 20
    assert result["measured_b05"][gocube_b05.B0_TREATMENT]["measured_command_log_bytes"] == 8
    assert estimate["estimated_new_bytes"] > estimate["measured_tiny_pair_artifact_bytes"] * 3
    assert estimate["estimated_unpruned_current_launcher_bytes"] >= estimate["estimated_new_bytes"]
    assert result["full_run_per_treatment"][gocube_b05.B0_TREATMENT]["retained_policy_bytes"]["logs"] > 0
    assert result["reserve_policy"]["used_for_full_run_estimate"] is False
    assert result["estimates"]["5+5"]["reserve_bytes"] == 1 * 1024**3
