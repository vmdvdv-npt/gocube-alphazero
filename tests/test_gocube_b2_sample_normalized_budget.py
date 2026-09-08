from __future__ import annotations

import json

import pytest

from alphazero.envs.gocube.b_experiment_contract import (
    DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET,
    build_b_experiment_contract,
)
from alphazero.envs.gocube.production_training import (
    CumulativeTrainingCounters,
    SampleBudgetTarget,
    build_replay_training_plan,
    load_training_progress,
    recover_cumulative_training_counters,
    write_training_progress,
)
from alphazero.envs.gocube.katago_train import KataGoSearchCoach
from tools import gocube_b_experiment
from tools._c4_overnight_runtime import render_markdown_report


def test_same_generation_chunk_with_different_realized_samples_has_different_budget():
    first = build_replay_training_plan(
        new_selfplay_samples=1_000,
        replay_window_samples=4_000,
        train_samples_per_new_sample=1.0,
        batch_size=256,
    )
    second = build_replay_training_plan(
        new_selfplay_samples=2_000,
        replay_window_samples=8_000,
        train_samples_per_new_sample=1.0,
        batch_size=256,
    )
    assert first.new_selfplay_samples == 1_000
    assert second.new_selfplay_samples == 2_000
    assert first.sample_clock_increment == 1_000
    assert second.sample_clock_increment == 2_000
    assert first.planned_optimizer_steps != second.planned_optimizer_steps


def test_different_game_chunks_can_reach_the_same_sample_milestone():
    target = SampleBudgetTarget(SampleBudgetTarget.NEW_SAMPLES, 3_000)
    short_run = CumulativeTrainingCounters(selfplay_games_completed=128, new_samples_accepted=3_000)
    long_run = CumulativeTrainingCounters(selfplay_games_completed=256, new_samples_accepted=3_000)
    assert target.current(short_run) == target.current(long_run) == target.target
    assert target.status(short_run)["reached"]
    assert target.status(long_run)["reached"]


def test_progress_artifact_and_resume_preserve_all_cumulative_counters(tmp_path):
    counters = CumulativeTrainingCounters(
        selfplay_games_completed=256,
        positions_generated=12_000,
        saved_replay_samples=13_000,
        new_samples_accepted=13_000,
        optimizer_steps=51,
        optimizer_examples_seen=13_056,
    )
    write_training_progress(
        tmp_path,
        "run",
        counters=counters,
        latest_iteration=4,
    )
    loaded = load_training_progress(tmp_path, "run")
    assert loaded is not None
    assert loaded["cumulative"] == counters.as_dict()
    assert loaded["cumulative_counters"] == counters.as_dict()
    assert loaded["cumulative_optimizer_steps"] == 51
    assert loaded["cumulative_optimizer_examples_seen"] == 13_056
    resumed = recover_cumulative_training_counters(
        tmp_path,
        "run",
        optimizer_steps=51,
        optimizer_examples_seen=13_056,
    )
    assert resumed.as_dict() == counters.as_dict()


def test_manifest_recovery_does_not_count_replay_window_as_new_samples(tmp_path):
    manifest_path = tmp_path / "run" / "records" / "iteration-0002" / "iteration-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "iteration": 2,
                "aggregate_metrics": {
                    "sample_accounting": {
                        "selfplay_games_completed": 256,
                        "positions_generated": 5_000,
                        "saved_replay_samples": 700,
                        "new_samples_accepted": 700,
                    },
                    "training": {
                        "replay_window_samples": 4_200,
                        "actual_optimizer_steps": 7,
                        "actual_training_samples": 768,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    recovered = recover_cumulative_training_counters(tmp_path, "run")
    assert recovered.selfplay_games_completed == 256
    assert recovered.positions_generated == 5_000
    assert recovered.saved_replay_samples == 700
    assert recovered.new_samples_accepted == 700
    assert recovered.optimizer_steps == 7
    assert recovered.optimizer_examples_seen == 768


def _write_iteration_manifest(tmp_path, iteration, *, games=256, samples=700, examples=768):
    path = tmp_path / "run" / "records" / f"iteration-{iteration:04d}" / "iteration-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": iteration,
                "aggregate_metrics": {
                    "sample_accounting": {
                        "selfplay_games_completed": games,
                        "positions_generated": samples,
                        "saved_replay_samples": samples,
                        "new_samples_accepted": samples,
                    },
                    "training": {
                        "actual_optimizer_steps": 7,
                        "actual_training_samples": examples,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_resume_reconciles_lagging_progress_to_the_loaded_checkpoint(tmp_path):
    write_training_progress(
        tmp_path,
        "run",
        counters=CumulativeTrainingCounters(
            selfplay_games_completed=4 * 256,
            positions_generated=4_000,
            saved_replay_samples=4_000,
            new_samples_accepted=4_000,
        ),
        latest_iteration=4,
    )
    _write_iteration_manifest(tmp_path, 5, games=256, samples=900)

    recovered = recover_cumulative_training_counters(
        tmp_path,
        "run",
        checkpoint_iteration=5,
        optimizer_steps=51,
        optimizer_examples_seen=5_568,
    )

    assert recovered.selfplay_games_completed == 5 * 256
    assert recovered.positions_generated == 4_900
    assert recovered.saved_replay_samples == 4_900
    assert recovered.new_samples_accepted == 4_900
    assert recovered.optimizer_steps == 51
    assert recovered.optimizer_examples_seen == 5_568


def test_resume_repairs_progress_artifact_to_the_loaded_checkpoint(tmp_path):
    write_training_progress(
        tmp_path,
        "run",
        counters=CumulativeTrainingCounters(new_samples_accepted=4_000),
        latest_iteration=4,
    )
    _write_iteration_manifest(tmp_path, 5, games=256, samples=900)
    coach = object.__new__(KataGoSearchCoach)
    coach.args = type("Args", (), {"data": str(tmp_path), "run_name": "run"})()
    coach.train_net = type(
        "Network",
        (),
        {"total_optimizer_updates": 51, "total_training_samples": 5_568},
    )()

    coach._initialize_training_accounting(resumed_checkpoint_iteration=5)

    repaired = load_training_progress(tmp_path, "run")
    assert repaired is not None
    assert repaired["latest_iteration"] == 5
    assert repaired["cumulative"]["new_samples_accepted"] == 4_900
    assert repaired["cumulative_optimizer_steps"] == 51
    assert repaired["cumulative_optimizer_examples_seen"] == 5_568


def test_resume_fails_closed_when_progress_is_ahead_of_checkpoint(tmp_path):
    write_training_progress(
        tmp_path,
        "run",
        counters=CumulativeTrainingCounters(
            selfplay_games_completed=6 * 256,
            new_samples_accepted=6_000,
        ),
        latest_iteration=6,
    )

    with pytest.raises(ValueError, match="ahead of the selected checkpoint"):
        recover_cumulative_training_counters(
            tmp_path,
            "run",
            checkpoint_iteration=5,
            optimizer_steps=50,
            optimizer_examples_seen=5_000,
        )


def test_resume_fails_closed_on_unreadable_manifest_needed_through_checkpoint(tmp_path):
    write_training_progress(
        tmp_path,
        "run",
        counters=CumulativeTrainingCounters(new_samples_accepted=4_000),
        latest_iteration=4,
    )
    manifest = tmp_path / "run" / "records" / "iteration-0005" / "iteration-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="iteration 5 manifest is missing or unreadable"):
        recover_cumulative_training_counters(
            tmp_path,
            "run",
            checkpoint_iteration=5,
            optimizer_steps=50,
            optimizer_examples_seen=5_000,
        )


def test_manifest_fallback_fails_closed_on_unreadable_json(tmp_path):
    manifest = tmp_path / "run" / "records" / "iteration-0001" / "iteration-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="iteration 1 manifest is missing or unreadable"):
        recover_cumulative_training_counters(tmp_path, "run")


def test_training_ratio_uses_actual_new_samples_and_not_replay_window():
    plan = build_replay_training_plan(
        new_selfplay_samples=200,
        replay_window_samples=10_000,
        train_samples_per_new_sample=3.0,
        batch_size=128,
    )
    assert plan.planned_training_samples == 600
    assert plan.planned_optimizer_steps == 5


def test_last_generation_overshoot_is_explicit():
    target = SampleBudgetTarget(SampleBudgetTarget.NEW_SAMPLES, 1_000)
    status = target.status(
        CumulativeTrainingCounters(new_samples_accepted=1_125),
        before=800,
        generation_chunk_games=256,
    )
    assert status["reached"] is True
    assert status["overshot"] is True
    assert status["overshoot"] == 125
    assert status["generation_chunk_games"] == 256


def test_b_report_contains_sample_normalized_accounting_for_each_treatment():
    counters = {
        "selfplay_games_completed": 256,
        "positions_generated": 8_000,
        "saved_replay_samples": 8_500,
        "new_samples_accepted": 8_500,
        "optimizer_steps": 9,
        "optimizer_examples_seen": 9_216,
    }
    report = render_markdown_report(
        {
            "status": "COMPLETE",
            "fixed_contract": {"games_per_iteration": 256},
            "totals": {},
            "parameters": [
                {
                    "id": "B0",
                    "name": "baseline",
                    "candidates": [{
                        "label": "B0",
                        "value": 0,
                        "screen": {},
                        "training_budget": {
                            "counters": counters,
                            "budget": {
                                "kind": "cumulative_new_samples",
                                "target": 8_375,
                                "after": 8_500,
                                "overshoot": 125,
                                "overshot": True,
                            },
                            "latest_iteration_metrics": {
                                "average_game_length": 91.5,
                                "no_result_games": 3,
                                "episode_move_limit_games": 2,
                            },
                        },
                    }],
                    "winner": {},
                },
                {
                    "id": "B1",
                    "name": "g1",
                    "candidates": [{"label": "B1", "value": 1, "screen": {}, "training_budget": {"counters": counters}}],
                    "winner": {},
                },
            ],
            "cumulative_counters": counters,
            "scientific_budget": {"kind": "cumulative_new_samples", "increment": 8_500},
        }
    )
    for label in ("B0", "B1"):
        assert f"- {label}" in report
    for field, value in counters.items():
        assert f"{field}={value}" in report or f"cumulative {field}: {value}" in report
    assert "overshoot=125" in report
    assert "average_length=91.5" in report
    assert "no_results=3" in report
    assert "move_limit=2" in report


def test_b_launcher_pins_one_shared_default_target_or_explicit_target():
    default_args = gocube_b_experiment.parse_args(["--treatment", "B0", "--dry-run"])
    assert default_args.scientific_target.kind == SampleBudgetTarget.NEW_SAMPLES
    assert default_args.scientific_target.target == DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET
    default_command = gocube_b_experiment.training_command(default_args, python="python")
    assert "--cumulative-new-samples-target" in default_command
    assert str(DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET) in default_command

    explicit_args = gocube_b_experiment.parse_args(
        ["--treatment", "B1", "--cumulative-optimizer-examples-target", "12000"]
    )
    assert explicit_args.scientific_target.kind == SampleBudgetTarget.OPTIMIZER_EXAMPLES
    explicit_command = gocube_b_experiment.training_command(explicit_args, python="python")
    assert "--cumulative-optimizer-examples-target" in explicit_command
    assert "12000" in explicit_command


def test_b_contract_records_the_scientific_target():
    contract = build_b_experiment_contract(source_git_sha="0" * 40, heldout_suite_hash="1" * 64)
    assert contract.scientific_sample_target["kind"] == SampleBudgetTarget.NEW_SAMPLES
    assert contract.scientific_sample_target["target"] == DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET
