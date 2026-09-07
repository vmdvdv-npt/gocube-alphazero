from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from alphazero.search_contract import SearchOutput
from tools import c4_adaptive_parameter_experiment as adaptive
from tools import c4_overnight_experiment as overnight_entrypoint
from tools import gocube_checkpoint_arena as checkpoint_arena
from tools import gocube_overnight_safety as overnight_safety
from tools.hardware_telemetry import HardwareTelemetry


def test_adaptive_runner_keeps_fixed_production_contract_and_has_no_deadline():
    assert adaptive.WORKERS == 16
    assert adaptive.REGULAR_SIMS == 50
    assert adaptive.FAST_SIMS == 20
    assert adaptive.GAMES_PER_ITERATION == 256
    assert adaptive.TRAIN_BATCH_SIZE == 1024
    assert adaptive.SELFPLAY_BATCH_WAIT_MS == 1.0
    assert adaptive.EXPECTED_KOMI == 0.5
    assert [spec["id"] for spec in adaptive.PARAMETER_SPECS] == ["P1", "P2", "P3", "P4", "P5"]
    assert [spec["flag"] for spec in adaptive.PARAMETER_SPECS] == [
        "--chosen-move-temperature-halflife",
        "--root-dirichlet-noise-weight",
        "--fast-game-prob",
        "--train-samples-per-new-sample",
        "--replay-window-iters",
    ]
    source = Path(adaptive.__file__).read_text(encoding="utf-8")
    assert "--max-hours" not in source


def test_training_command_carries_cumulative_sweep_overrides():
    experiment = object.__new__(adaptive.Experiment)
    experiment.python = Path(".venv/bin/python")
    command = experiment.training_command(
        run_name="candidate",
        target_iteration=11,
        sweep_overrides={
            "--chosen-move-temperature-halflife": 19.0,
            "--root-dirichlet-noise-weight": 0.25,
        },
        resume=True,
    )
    joined = " ".join(command)
    assert "--workers 16" in joined
    assert "--sims 50" in joined
    assert "--games-per-iteration 256" in joined
    assert "--train-batch-size 1024" in joined
    assert "--inference-batch-wait-ms 1.0" in joined
    assert "--chosen-move-temperature-halflife 19.0" in joined
    assert "--root-dirichlet-noise-weight 0.25" in joined
    assert "--allow-existing-run" in command


def test_clone_run_namespace_copies_checkpoint_and_replay_without_mutating_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parent = "parent"
    target = "candidate"
    (tmp_path / "checkpoint" / parent).mkdir(parents=True)
    (tmp_path / "data" / parent).mkdir(parents=True)
    (tmp_path / "checkpoint" / parent / "iteration-0007.pkl").write_bytes(b"checkpoint")
    (tmp_path / "checkpoint" / parent / "gocube-run.json").write_text(
        json.dumps({"runName": parent}), encoding="utf-8"
    )
    (tmp_path / "data" / parent / "iteration-0007-data.pkl").write_bytes(b"data")

    adaptive.clone_run_namespace(parent, target)

    assert (tmp_path / "checkpoint" / target / "iteration-0007.pkl").read_bytes() == b"checkpoint"
    assert (tmp_path / "data" / target / "iteration-0007-data.pkl").read_bytes() == b"data"
    assert not (tmp_path / "checkpoint" / target / "gocube-run.json").exists()
    assert (tmp_path / "checkpoint" / parent / "iteration-0007.pkl").read_bytes() == b"checkpoint"
    with pytest.raises(FileExistsError):
        adaptive.clone_run_namespace(parent, target)


def test_frozen_heldout_suite_is_fresh_and_never_requires_training_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = "bootstrap"
    checkpoint_dir = tmp_path / "checkpoint" / run
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {
            "args": {"gocube_rules_fingerprint": "rules", "gocube_komi": 0.5},
            "state_dict": {},
        },
        checkpoint_dir / "iteration-0007.pkl",
    )

    def fake_generate(**kwargs):
        assert kwargs["run_name"] == run
        assert not (tmp_path / "data" / run).exists()
        selected = [
            {
                "position_id": f"H{index + 1:03d}",
                "origin": overnight_safety.HELDOUT_PROVENANCE,
                "rollout_seed": 100 + index,
                "rollout_length": 20,
                "prefix_length": 4,
                "prefix_actions": [index, index + 1, index + 2, index + 3],
            }
            for index in range(4)
        ]
        return selected, {
            "source_checkpoint": str(checkpoint_dir / "iteration-0007.pkl"),
            "search_sims": 50,
            "action_temperature": 1.0,
            "root_noise": False,
            "root_policy_temperature": False,
            "device": "cpu",
        }

    monkeypatch.setattr(overnight_safety, "_generate_rollout_positions", fake_generate)
    suite_path = tmp_path / "suite.json"
    first = adaptive.build_frozen_heldout_suite(
        run_name=run,
        iteration=7,
        output_path=suite_path,
        positions=4,
        seed=123,
        device="cpu",
    )
    second = adaptive.build_frozen_heldout_suite(
        run_name=run,
        iteration=7,
        output_path=suite_path,
        positions=4,
        seed=999,
        device="cpu",
    )

    assert first == second
    assert first["schema_version"] == 2
    assert first["provenance"] == overnight_safety.HELDOUT_PROVENANCE
    assert first["training_data_excluded"] is True
    assert first["komi"] == 0.5
    assert first["rules_fingerprint"] == "rules"
    assert len(first["positions"]) == 4
    assert all(position["origin"] == overnight_safety.HELDOUT_PROVENANCE for position in first["positions"])
    assert all("source_record" not in position for position in first["positions"])


def test_checkpoint_arena_accepts_only_fresh_heldout_suite_contract():
    saved = {"gocube_rules_fingerprint": "abc"}
    payload = {
        "schema_version": 2,
        "provenance": overnight_safety.HELDOUT_PROVENANCE,
        "training_data_excluded": True,
        "komi": 0.5,
        "rules_fingerprint": "abc",
        "positions": [{
            "origin": overnight_safety.HELDOUT_PROVENANCE,
            "prefix_actions": [1, 2, 3],
        }],
    }
    positions = checkpoint_arena._validate_heldout_suite(payload, saved)
    assert positions[0]["prefix_actions"] == [1, 2, 3]
    assert checkpoint_arena.HELDOUT_SCHEMA_VERSION == 2

    legacy = dict(payload, schema_version=1)
    with pytest.raises(ValueError, match="fresh-only schema"):
        checkpoint_arena._validate_heldout_suite(legacy, saved)

    bad = dict(payload, komi=0.6)
    with pytest.raises(ValueError, match="komi 0.5"):
        checkpoint_arena._validate_heldout_suite(bad, saved)


def test_progress_metric_parser_uses_final_carriage_return_redraw():
    line = (
        "Generating Samples Sample Time: 1.250s | Infer Batch: 2.0\r"
        "Generating Samples Sample Time: 0.800s | Infer Batch: 4.0\r"
        "Generating Samples Sample Time: 0.500s | Infer Batch: 8.0\n"
    )
    assert overnight_safety.extract_last_progress_metrics(line) == (0.5, 8.0)
    assert "extract_last_progress_metrics" in inspect.getsource(overnight_entrypoint.Experiment.stream_command)


def test_checkpoint_arena_resolves_device_and_reports_wilson_interval(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert checkpoint_arena._resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError):
        checkpoint_arena._resolve_device("cuda")
    low, high = checkpoint_arena._wilson_interval(0.5, 128)
    assert 0.4 < low < 0.5 < high < 0.6


def test_copy_search_output_splits_one_coalesced_forward_across_workers():
    policy_tensors = [torch.zeros((1, 3)), torch.zeros((1, 3))]
    value_tensors = [torch.zeros((1, 3)), torch.zeros((1, 3))]
    score_tensors = [torch.zeros((1, 1)), torch.zeros((1, 1))]
    ownership_tensors = [torch.zeros((1, 2, 3)), torch.zeros((1, 2, 3))]
    output = SearchOutput(
        policy=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        value=torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]),
        score=torch.tensor([[1.5], [-2.5]]),
        ownership=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
    )

    checkpoint_arena._copy_search_output(
        output,
        [(0, 1), (1, 1)],
        policy_tensors,
        value_tensors,
        score_tensors,
        ownership_tensors,
    )

    assert torch.equal(policy_tensors[0], output.policy[0:1])
    assert torch.equal(policy_tensors[1], output.policy[1:2])
    assert torch.equal(score_tensors[0], output.score[0:1])
    assert torch.equal(ownership_tensors[1], output.ownership[1:2])


def test_hardware_telemetry_summary_is_phase_separated(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    rows = [
        {
            "time": 1.0,
            "phase": "SELFPLAY",
            "cpu_util_percent": 80.0,
            "ram_used_gib": 10.0,
            "gpus": [{"gpu_util_percent": 60.0, "gpu_memory_used_mib": 1000.0}],
        },
        {
            "time": 2.0,
            "phase": "SELFPLAY",
            "cpu_util_percent": 90.0,
            "ram_used_gib": 11.0,
            "gpus": [{"gpu_util_percent": 70.0, "gpu_memory_used_mib": 1100.0}],
        },
        {
            "time": 3.0,
            "phase": "ARENA",
            "cpu_util_percent": 50.0,
            "ram_used_gib": 9.0,
            "gpus": [{"gpu_util_percent": 40.0, "gpu_memory_used_mib": 900.0}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    summary = HardwareTelemetry(path).summary()
    assert summary["samples"] == 3
    assert summary["phases"]["SELFPLAY"]["cpu_util_percent"]["mean"] == pytest.approx(85.0)
    assert summary["phases"]["ARENA"]["gpu_util_percent"]["max"] == pytest.approx(40.0)
