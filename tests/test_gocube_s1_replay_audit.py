from pathlib import Path

from tools.audit_gocube_s1_replays import audit


def test_persisted_replay_audit_is_read_only_and_reports_all_current_records():
    root = Path(__file__).resolve().parents[1] / "data"
    report = audit([root])

    assert report["read_only"] is True
    assert report["s1_contract"]["replay_format_version"] == 4
    assert report["overall"]["total"] == sum(
        bucket["total"] for bucket in report["by_start_type"].values()
    )
    assert report["overall"]["total"] == sum(report["contract_versions"].values())
    assert (
        report["overall"]["reconstructible"]
        + report["overall"]["not_reconstructible"]
        == report["overall"]["total"]
    )
    assert (
        sum(report["overall"]["score_delta_distribution"].values())
        == report["overall"]["total"]
    )
    assert report["overall"]["winner_changed"] <= report["overall"]["total"]
    assert report["overall"]["ownership_changed"] <= report["overall"]["total"]
    assert report["by_start_type"]["synthetic_cleanup"]["total"] == 0
    assert report["by_start_type"]["fork"]["total"] == 0
