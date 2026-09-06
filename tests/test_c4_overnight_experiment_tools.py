from __future__ import annotations

import importlib.util
import py_compile
import subprocess
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "c4_overnight_experiment.py"
HARDENED = ROOT / "tools" / "c4_overnight_hardened.py"
EVALUATOR = ROOT / "tools" / "evaluate_gocube_checkpoints.py"
LAUNCHER = ROOT / "tools" / "launch_c4_overnight.sh"
RESUMER = ROOT / "tools" / "resume_c4_overnight.sh"
REPORTER = ROOT / "tools" / "run_with_github_reports.sh"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("c4_overnight_experiment", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tools_have_valid_python_and_bash_syntax():
    py_compile.compile(str(RUNNER), doraise=True)
    py_compile.compile(str(HARDENED), doraise=True)
    py_compile.compile(str(EVALUATOR), doraise=True)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    subprocess.run(["bash", "-n", str(RESUMER)], check=True)
    subprocess.run(["bash", "-n", str(REPORTER)], check=True)


def test_branch_matrix_is_exactly_the_seven_planned_one_factor_variants():
    module = load_runner_module()
    assert module.WORKERS == 16
    assert module.PARENT_ITERATION == 5
    assert module.FAST_SIMS == 20
    assert module.BRANCHES == {
        "A": {"sims": 100, "pfast": 0.25, "games": 256, "axis": "control", "label": "baseline"},
        "B": {"sims": 100, "pfast": 0.00, "games": 256, "axis": "pfast", "label": "no fast search"},
        "C": {"sims": 100, "pfast": 0.50, "games": 256, "axis": "pfast", "label": "more fast search"},
        "D": {"sims": 50, "pfast": 0.25, "games": 256, "axis": "sims", "label": "shallower regular search"},
        "E": {"sims": 200, "pfast": 0.25, "games": 256, "axis": "sims", "label": "deeper regular search"},
        "F": {"sims": 100, "pfast": 0.25, "games": 128, "axis": "games", "label": "faster feedback loop"},
        "G": {"sims": 100, "pfast": 0.25, "games": 512, "axis": "games", "label": "more data per update"},
    }
    assert module.AXIS_PAIRS == {"pfast": ("B", "C"), "sims": ("D", "E"), "games": ("F", "G")}
    assert {key: cfg["games"] // module.WORKERS for key, cfg in module.BRANCHES.items()} == {
        "A": 16, "B": 16, "C": 16, "D": 16, "E": 16, "F": 8, "G": 32,
    }


def test_report_contract_contains_every_metric_needed_for_morning_decision():
    source = RUNNER.read_text(encoding="utf-8")
    required = (
        "checkpoint_sha256",
        "wall_seconds",
        "regular_decisions",
        "fast_decisions",
        "realized_fast_fraction",
        "base_positions",
        "latest_iteration_samples",
        "window_samples",
        "optimizer_steps_actual",
        "examples_seen",
        "effective_passes",
        "learning_rate",
        "average_game_length",
        "no_result_games",
        "training_valid_fraction",
        "process_batch_size",
        "candidate_score_rate",
        "candidate_score_ci95_approx",
        "metrics.csv",
        "evaluations.csv",
        "experiment.json",
        "state.json",
        "artifacts.json",
        "events.jsonl",
        "summary.md",
    )
    for token in required:
        assert token in source


def test_summary_calls_games_axis_coupling_out_explicitly():
    module = load_runner_module()
    experiment = module.Experiment.__new__(module.Experiment)
    experiment.args = SimpleNamespace(frozen_commit=module.FROZEN_TRAINING_COMMIT)
    experiment.source_run = module.SOURCE_RUN_DEFAULT
    experiment.state = {
        "experiment_id": "synthetic",
        "status": "DONE",
        "started_at": "2026-09-06T00:00:00+04:00",
        "deadline_at": "2026-09-06T08:00:00+04:00",
        "parent_checkpoint_sha256": "abc",
        "branches": {
            key: {"latest_iteration": 7, "status": "ACTIVE", "run_name": f"synthetic-{key.lower()}"}
            for key in module.BRANCHES
        },
        "metrics": [],
        "evaluations": [
            {
                "candidate_branch": "B",
                "candidate_iteration": 7,
                "reference_id": "A@7",
                "games_requested": 40,
                "candidate_wins": 24,
                "reference_wins": 16,
                "draws": 0,
                "no_results": 0,
                "candidate_score_rate": 0.60,
                "candidate_score_ci95_approx": [0.52, 0.68],
                "elapsed_seconds": 120.0,
            }
        ],
    }
    summary = experiment.summary_markdown()
    assert "process_batch_size = games_per_iteration / workers" in summary
    assert "F/G measure the practical outer-loop package" in summary
    assert "Strong evidence: branch B beats same-depth baseline A" in summary
    assert "loss curves alone" in summary


def test_evaluator_is_fixed_noise_free_balanced_batched_arena():
    source = EVALUATOR.read_text(encoding="utf-8")
    for token in (
        "prepare_evaluation_args",
        "play_balanced_batched_match",
        '"batched": True',
        '"balanced_colors": True',
        '"arena_batch_size"',
        '"arena_workers"',
        '"fast_probability": 0.0',
        '"root_noise": False',
        '"root_temperature": False',
        '"action_temperature": 0.0',
    ):
        assert token in source
    assert "use_batched_mcts=False" not in source


def test_reporter_watches_and_mirrors_structured_publish_directory():
    source = REPORTER.read_text(encoding="utf-8")
    for token in (
        'PUBLISH_DIR="$RUN_DIR/publish"',
        "publish_signature",
        "report_signature",
        "copy_structured_publish",
        'cp -a "$PUBLISH_DIR/." "$dest/"',
        "watch_reports",
    ):
        assert token in source


def test_launcher_uses_hardened_detached_runner_and_import_path():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "systemd-run --user" in source
    assert "powercfg" in source  # removes the legacy block from disposable preflight only
    assert "temporary preflight still contains Windows power-policy checks" in source
    assert "training_reports/" in source
    assert 'setenv="PYTHONPATH=$REPO_ROOT"' in source
    assert 'setenv="GOCUBE_NIGHT_TOOLING_COMMIT=$TOOLING_COMMIT"' in source
    assert '"$TMP_DIR/c4_overnight_hardened.py"' in source
    assert '--state=running,activating' in source
    assert "NIGHT LAUNCH PASS" in source


def test_resume_script_is_transactional_and_uses_hardened_runner():
    source = RESUMER.read_text(encoding="utf-8")
    assert 'STATE="training_reports/$EXP_ID/state-private.json"' in source
    assert 'state["status"] = "RUNNING"' not in source
    assert 'state.pop("fatal_error", None)' not in source
    assert '--state=running,activating' in source
    assert '"$TMP_DIR/c4_overnight_hardened.py"' in source
    assert '--experiment-id "$EXP_ID"' in source
    assert 'setenv="PYTHONPATH=$REPO_ROOT"' in source
    assert 'setenv="GOCUBE_NIGHT_TOOLING_COMMIT=$TOOLING_COMMIT"' in source
    assert "NIGHT RESUME PASS" in source


def test_hardened_runner_covers_interruption_recovery_windows():
    source = HARDENED.read_text(encoding="utf-8")
    required = (
        "iteration_quarantined",
        "fork_quarantined",
        "evaluation_recovered",
        "evaluation_output_quarantined",
        "_later_checkpoint_exists",
        "_later_data_exists",
        "_validate_existing_fork",
        "games_effective",
        "candidate SHA mismatch",
        "reference SHA mismatch",
        "tooling_history",
        "GOCUBE_NIGHT_TOOLING_COMMIT",
    )
    for token in required:
        assert token in source
