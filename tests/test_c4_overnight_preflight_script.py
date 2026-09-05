from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "preflight_c4_overnight.sh"


def test_preflight_script_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_preflight_script_keeps_required_safety_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    required_fragments = (
        "EXPECTED_HEAD=${GOCUBE_PREFLIGHT_EXPECTED_HEAD:-85c87a7cfd467a4d3f4b2844253fb63d746d672a}",
        "EXPECTED_CHECKPOINT_SHA=${GOCUBE_PREFLIGHT_EXPECTED_CHECKPOINT_SHA:-64cf800460f6090880c7818cbeff80123257dcd14c79689f108cc5523fb58722}",
        "EXPECTED_MANIFEST_SHA=${GOCUBE_PREFLIGHT_EXPECTED_MANIFEST_SHA:-909252d5d793c446b163837019098cb60036a5fa6c18b390ee154bcb9ff3414a}",
        "WORKERS=${GOCUBE_PREFLIGHT_WORKERS:-16}",
        "git push --dry-run",
        "snapshot_source > \"$SOURCE_SNAPSHOT_BEFORE\"",
        "cmp -s \"$SOURCE_SNAPSHOT_BEFORE\" \"$SOURCE_SNAPSHOT_AFTER\"",
        "training_reports/.c4-preflight",
        "--workers \"$WORKERS\"",
        "--games-per-iteration \"$WORKERS\"",
        "--train-steps-per-iteration 1",
        "--no-arena",
        "iteration-0002.pkl",
        "iteration-0003.pkl",
        "PREFLIGHT PASS",
    )
    for fragment in required_fragments:
        assert fragment in text
