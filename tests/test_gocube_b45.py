from __future__ import annotations

import json
from pathlib import Path

from tools import gocube_b45


def test_b45_stage_order_is_common_prep_then_b4_then_b5():
    assert gocube_b45.PIPELINE_STAGE_ORDER == ("COMMON_PREP", "B4", "B5")


def test_b45_default_is_fresh_dated_run_namespace():
    args = gocube_b45.parse_args([])
    assert args.run_name == "gocube-b45-rerun-20260909"
    assert args.report_dir == "training_reports/gocube-b45-rerun-20260909"


def test_b45_systemd_wraps_github_reporting_and_does_not_attach_to_ssh(tmp_path):
    repo = tmp_path
    args = gocube_b45.parse_args(["--run-name", "gate-test"])
    unit, command = gocube_b45._systemd_command(repo, args)
    assert unit == "gocube-b45-gate-test"
    assert "--wait" not in command
    assert "--pipe" not in command
    assert str(repo / "tools" / "run_with_github_reports.sh") in command
    assert "gate-test" in command
    assert str(repo / ".venv" / "bin" / "python") in command
    assert str(repo / "tools" / "gocube_b45.py") in command


def test_b45_archives_only_b05_runtime_namespaces(tmp_path):
    repo = tmp_path / "repo"
    report = tmp_path / "report"
    for category in ("checkpoint", "data", "runs"):
        (repo / category / "gocube-b05-b0").mkdir(parents=True)
        (repo / category / "gocube-b05-b0" / "marker").write_text("old", encoding="utf-8")
        (repo / category / "keep-me").mkdir(parents=True)
    moved = gocube_b45._archive_previous_b05_runtime(repo, report)
    assert len(moved) == 3
    for category in ("checkpoint", "data", "runs"):
        assert not (repo / category / "gocube-b05-b0").exists()
        archived = list((repo / "training_reports" / "_b05_runtime_archive").glob(f"*/{category}/gocube-b05-b0/marker"))
        assert len(archived) == 1
        assert (repo / category / "keep-me").is_dir()


def test_b45_b4_gate_requires_complete_32_game_non_scientific_artifact(tmp_path):
    artifact = tmp_path / "b4.json"
    evaluation = {
        "games": [{} for _ in range(gocube_b45.B_HELDOUT_SUITE_POSITION_COUNT * 2)],
        "non_scientific_dry_run": True,
        "training_seed": 0,
        "scientific_milestone": gocube_b45.EVALUATION_MILESTONE,
        "position_count": gocube_b45.B_HELDOUT_SUITE_POSITION_COUNT,
    }
    record = {
        "experiment_contract_sha256": "a" * 64,
        "experiment_contract": {"evaluation_milestones": {}},
    }
    result = gocube_b45._validate_b4(evaluation, record, artifact)
    assert result["status"] == "PASS"
    assert result["number_of_games"] == 32
    assert result["scientific_analyzer_accepts"] is False


def test_b45_report_is_mirrored_to_publish_for_github_reporter(tmp_path):
    report = {
        "status": "RUNNING",
        "stages": {"COMMON_PREP": "PASS", "B4": "RUNNING", "B5": "NOT_RUN"},
    }
    artifact = tmp_path / "artifacts" / "b4.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
    gocube_b45._write_report(tmp_path, report, b4_artifact=artifact)
    assert (tmp_path / "publish" / "b45-report.json").is_file()
    assert (tmp_path / "publish" / "b45-report.md").is_file()
    assert json.loads((tmp_path / "publish" / "b4-result.json").read_text(encoding="utf-8")) == {"ok": True}
