from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_engine import _write_jsonl, _write_jsonl_with_identity


def test_rolling_writer_returns_sha_of_the_bytes_it_writes(tmp_path: Path) -> None:
    rows = (
        {"source_generation": 1, "value": [3, 2, 1]},
        {"source_generation": 2, "value": {"b": 2, "a": 1}},
    )
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"

    _write_jsonl(old_path, rows)
    returned = _write_jsonl_with_identity(new_path, rows)

    assert new_path.read_bytes() == old_path.read_bytes()
    expected = "sha256:" + hashlib.sha256(new_path.read_bytes()).hexdigest()
    assert returned == expected


def test_training_engine_does_not_rehash_rolling_tmp_after_single_pass_write(
    tmp_path: Path,
) -> None:
    # Keep this test independent of the full Torus9 stack: the generic engine
    # owns the serialization and the adapter spy exposes every fallback hash.
    from tests.test_training_engine_stage3 import _FakeAdapter, _fake_state
    from training_engine import TrainingEngine

    adapter = _FakeAdapter()
    TrainingEngine(adapter).run_iteration(
        state=_fake_state(),
        generation=1,
        output_dir=tmp_path,
        run_id="fake-run",
        records=({"valid": True},),
        training_seed=17,
        device="cpu",
    )

    assert not any(path.name.startswith(".rolling-after-") for path in adapter.hash_calls)
    marker = json.loads((tmp_path / "generation-01.complete.json").read_text())
    rolling = tmp_path / "replay" / "rolling-after-01.jsonl"
    assert marker["rolling_replay_sha256"] == (
        "sha256:" + hashlib.sha256(rolling.read_bytes()).hexdigest()
    )
