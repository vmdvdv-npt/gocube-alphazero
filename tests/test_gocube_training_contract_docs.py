from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_training_docs_state_the_same_production_contract():
    docs = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "docs/TRAINING_LOOP_CONTRACT.md",
            "docs/KATAGO_RULE_REFERENCE.md",
            "docs/PRODUCTION_TRAINING_HARDENING.md",
        )
    )
    assert "sample-clock-v2" in docs
    assert "replay format v3" in docs.lower()
    assert "win-loss-noresult-s1-v2" in docs
    assert "katago-boardhistory-clear-v1" in docs
    assert "f6bc4b19a1686caa2d088b56251e8c11c8be6d51" in docs
    assert "komi `0.5`" in docs
    assert "tools/run_cube4_katago_from_scratch.sh" in docs
