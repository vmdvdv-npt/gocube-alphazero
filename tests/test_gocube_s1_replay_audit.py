from pathlib import Path

from tools.audit_gocube_s1_replays import audit


def test_persisted_replay_audit_is_read_only_and_reports_all_current_records():
    root = Path(__file__).resolve().parents[1] / "data"
    report = audit([root])

    assert report["read_only"] is True
    assert report["s1_contract"]["replay_format_version"] == 3
    assert report["contract_versions"] == {"2": 4}
    assert report["overall"]["total"] == 4
    assert report["overall"]["reconstructible"] == 4
    assert report["overall"]["not_reconstructible"] == 0
    assert report["overall"]["score_delta_distribution"] == {"+0.000000": 4}
    assert report["overall"]["winner_changed"] == 0
    assert report["overall"]["ownership_changed"] == 4
    assert report["by_start_type"]["synthetic_cleanup"]["total"] == 0
    assert report["by_start_type"]["fork"]["total"] == 0
