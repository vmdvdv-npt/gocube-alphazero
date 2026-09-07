from __future__ import annotations

import torch

from tools import c4_overnight_experiment as overnight


def test_selfplay_benchmark_resume_forces_runtime_wait_through_explicit_resume_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint" / "bench"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {"args": {"gocube_chosen_move_temperature_halflife": 19.0}},
        checkpoint_dir / "iteration-0007.pkl",
    )

    experiment = object.__new__(overnight.Experiment)
    experiment.python = overnight.Path(".venv/bin/python")
    experiment.selfplay_wait_ms = 1.0
    command = experiment.training_command(
        run_name="bench",
        target_iteration=8,
        resume=True,
        inference_wait_ms=0.5,
    )

    joined = " ".join(command)
    assert "--inference-batch-wait-ms 0.5" in joined
    assert "--chosen-move-temperature-halflife 19.0" in joined
    assert "--allow-existing-run" in command
