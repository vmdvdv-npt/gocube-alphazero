from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path

import pytest

from training_engine import CheckpointContext, TrainingEngine, TrainingState
from gocube_golden.torus9_training import (
    Torus9CurrentGraphNet,
    Torus9RollingReplay,
    Torus9TrainingAdapter,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    load_torus9_current_profile,
)
from tools.torus9_golden_learning import _profile_comparison


@dataclass
class _FakeAdapter:
    fail_at: str | None = None

    def validate_state(self, state):
        if state.optimizer_updates < 0:
            raise ValueError("invalid fake state")

    def build_samples(self, records):
        return tuple(records)

    def validate_sample(self, sample):
        if sample.get("valid") is not True:
            raise ValueError("invalid fake sample")

    def stamp_samples(self, samples, generation):
        return tuple({**sample, "source_generation": generation, "replay_row_id": f"M{generation}:{i}"} for i, sample in enumerate(samples))

    def update_replay(self, replay, generation, samples):
        if self.fail_at == "replay":
            raise RuntimeError("replay failure")
        replay.extend(copy.deepcopy(list(samples)))
        return {"generation_added": generation, "rolling_buffer_positions": len(replay)}

    def replay_rows(self, replay):
        return tuple(replay)

    def validate_replay(self, rows):
        if not rows:
            raise ValueError("empty replay")

    def train(self, state, rows, seed):
        if self.fail_at == "train":
            raise RuntimeError("train failure")
        state.optimizer_updates += 1
        state.samples_consumed += len(rows)
        return {
            "optimizer_steps": 1,
            "samples_consumed": len(rows),
            "sampled_replay_row_ids": tuple(row["replay_row_id"] for row in rows),
            "seed": seed,
        }

    def prepare_checkpoint(self, state, context, training_metrics):
        if self.fail_at == "prepare":
            raise RuntimeError("prepare failure")
        return {
            "checkpoint_schema_version": 1,
            "checkpoint_label": context.label,
            "model_hash": "sha256:" + "1" * 64,
            "optimizer_updates": state.optimizer_updates,
        }

    def save_checkpoint(self, path, state, metadata):
        if self.fail_at == "save":
            raise RuntimeError("save failure")
        path.write_text("checkpoint", encoding="utf-8")
        path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return metadata

    def verify_checkpoint(self, path, state, metadata):
        if self.fail_at == "verify":
            raise RuntimeError("verify failure")
        assert path.is_file()

    def artifact_hash(self, path):
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def snapshot_state(self, state):
        return (list(state.rolling_replay), state.optimizer_updates, state.samples_consumed)

    def restore_state(self, state, snapshot):
        replay, updates, consumed = snapshot
        state.rolling_replay[:] = replay
        state.optimizer_updates = updates
        state.samples_consumed = consumed

    def sync_state(self, state):
        return None


def _fake_state():
    return TrainingState(
        model=object(),
        optimizer=object(),
        optimizer_updates=0,
        samples_consumed=0,
        current_generation=0,
        rolling_replay=[],
        profile_identity={"profile_id": "fake"},
        target_identity={"fingerprint": "fake"},
    )


def test_generic_engine_commits_all_artifacts_and_state(tmp_path: Path):
    adapter = _FakeAdapter()
    state = _fake_state()
    result = TrainingEngine(adapter).run_iteration(
        state=state,
        generation=1,
        output_dir=tmp_path,
        run_id="fake-run",
        records=({"valid": True},),
        training_seed=17,
        device="cpu",
    )
    assert state.current_generation == 1
    assert state.optimizer_updates == 1
    assert result.summary["training"]["optimizer_steps"] == 1
    assert (tmp_path / "replay/iter-01-fresh.jsonl").is_file()
    assert (tmp_path / "replay/rolling-after-01.jsonl").is_file()
    assert (tmp_path / "checkpoints/M1.pt").is_file()
    assert (tmp_path / "checkpoints/M1.metadata.json").is_file()
    assert (tmp_path / "generation-01.complete.json").is_file()
    assert not list(tmp_path.rglob("*.tmp*"))


@pytest.mark.parametrize("failure", ["replay", "train", "save", "verify"])
def test_generic_engine_failure_does_not_publish_or_commit(failure: str, tmp_path: Path):
    adapter = _FakeAdapter(fail_at=failure)
    state = _fake_state()
    with pytest.raises(RuntimeError, match="failure"):
        TrainingEngine(adapter).run_iteration(
            state=state,
            generation=1,
            output_dir=tmp_path,
            run_id="fake-run",
            records=({"valid": True},),
            training_seed=17,
            device="cpu",
        )
    assert state.current_generation == 0
    assert state.optimizer_updates == 0
    assert state.samples_consumed == 0
    assert state.rolling_replay == []
    assert not (tmp_path / "generation-01.complete.json").exists()
    assert not (tmp_path / "checkpoints/M1.pt").exists()
    assert not list(tmp_path.rglob("*.tmp*"))


def test_current_replay_rejects_duplicate_and_non_monotonic_generation():
    replay = Torus9RollingReplay(generations=3, maximum_positions=10)
    replay.append_generation(1, [{"replay_row_id": "r1"}])
    with pytest.raises(ValueError, match="strictly"):
        replay.append_generation(1, [{"replay_row_id": "r2"}])
    with pytest.raises(ValueError, match="duplicate"):
        replay.append_generation(2, [{"replay_row_id": "r1"}])


def test_current_adapter_is_profile_locked_and_uses_explicit_state():
    profile = load_torus9_current_profile()
    adapter = Torus9TrainingAdapter(profile=profile)
    assert adapter.profile_identity["profile_id"] == TORUS9_CURRENT_PROFILE_ID
    assert adapter.target_identity["fingerprint"] == TORUS9_CURRENT_TARGET_FINGERPRINT
    assert adapter.base_commit == TORUS9_GOLDEN_LINEAGE_BASE_COMMIT
    model = Torus9CurrentGraphNet()
    state = adapter.create_state(model, run_id="state-test")
    assert state.model is model
    assert state.optimizer is state.adapter_state.optimizer
    assert state.optimizer_updates == 0
    assert state.samples_consumed == 0


def test_current_adapter_rejects_non_current_profile():
    profile = load_torus9_current_profile()
    profile["profile_id"] = "retired-profile"
    with pytest.raises(ValueError, match="current profile"):
        Torus9TrainingAdapter(profile=profile)


def test_current_launcher_profile_comparison_uses_canonical_komi():
    comparison = _profile_comparison(load_torus9_current_profile())
    assert comparison["checked"]["komi"] == 0.5
    assert "legacy_komi_sentinel" not in comparison["checked"]
