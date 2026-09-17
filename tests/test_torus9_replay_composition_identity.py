from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

import training_engine
from training_engine import TrainingEngine, TrainingState
from gocube_golden.provenance import file_sha256
from gocube_golden.torus9_training import (
    TORUS9_REPLAY_COMPOSITION_IDENTITY_SCHEMA,
    TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA,
    TORUS9_REPLAY_SELECTION_CONTRACT,
    Torus9RollingReplay,
    replay_generation_identity_from_artifact,
    replay_generation_identity_from_rows,
)


def _sha(char: str) -> str:
    return "sha256:" + char * 64


def _rows(generation: int, count: int) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "source_generation": generation,
            "replay_row_id": f"M{generation}:game:{position}:0",
            "payload": position,
        }
        for position in range(count)
    )


def _identity(generation: int, count: int, char: str) -> dict[str, object]:
    return {
        "schema": TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA,
        "generation": generation,
        "sha256": _sha(char),
        "row_count": count,
    }


def _replay(*, generations: int = 3, maximum_positions: int = 10) -> Torus9RollingReplay:
    replay = Torus9RollingReplay(
        generations=generations,
        maximum_positions=maximum_positions,
    )
    for generation, char in ((1, "1"), (2, "2"), (3, "3")):
        rows = _rows(generation, 2)
        replay.append_generation(
            generation,
            rows,
            generation_identity=_identity(generation, len(rows), char),
        )
    return replay


def test_replay_composition_identity_is_compact_and_policy_bound() -> None:
    replay = _replay()
    descriptor = replay.replay_identity_descriptor()

    assert descriptor["schema"] == TORUS9_REPLAY_COMPOSITION_IDENTITY_SCHEMA
    assert descriptor["contract"] == {
        "selection": TORUS9_REPLAY_SELECTION_CONTRACT,
        "generations": 3,
        "maximum_positions": 10,
    }
    assert descriptor["represented_generations"] == [1, 2, 3]
    assert descriptor["position_counts"] == {"1": 2, "2": 2, "3": 2}
    assert descriptor["new_generation_artifact_sha256"] == _sha("3")

    changed_window = _replay(generations=4)
    changed_cap = _replay(maximum_positions=9)
    changed_artifact = _replay()
    changed_artifact._generation_identities[2]["sha256"] = _sha("4")
    assert changed_window.replay_identity_descriptor()["fingerprint"] != descriptor["fingerprint"]
    assert changed_cap.replay_identity_descriptor()["fingerprint"] != descriptor["fingerprint"]
    assert changed_artifact.replay_identity_descriptor()["fingerprint"] != descriptor["fingerprint"]


def test_persisted_generation_identity_coverage_is_fail_closed() -> None:
    rows = _rows(7, 2)
    identity = _identity(7, len(rows), "7")
    replay = Torus9RollingReplay.from_persisted_rows(
        rows,
        generations=3,
        maximum_positions=10,
        generation_identities=[identity],
    )
    assert replay.generation_identities == (
        {**identity, "retained_row_count": 2},
    )

    with pytest.raises(ValueError, match="coverage"):
        Torus9RollingReplay.from_persisted_rows(
            rows,
            generations=3,
            maximum_positions=10,
            generation_identities=[],
        )

    with pytest.raises(ValueError, match="retained row count"):
        Torus9RollingReplay.from_persisted_rows(
            rows,
            generations=3,
            maximum_positions=10,
            generation_identities=[{**identity, "retained_row_count": 1}],
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        Torus9RollingReplay.from_persisted_rows(
            (*_rows(7, 2), *_rows(8, 2)),
            generations=3,
            maximum_positions=10,
            generation_identities=[_identity(8, 2, "8"), _identity(7, 2, "7")],
        )


def test_new_generation_identity_matches_the_published_jsonl_bytes(tmp_path: Path) -> None:
    rows = _rows(8, 2)
    path = tmp_path / "iter-08-fresh.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    identity = replay_generation_identity_from_rows(8, rows)
    assert identity["sha256"] == file_sha256(path)
    assert replay_generation_identity_from_artifact(
        8,
        file_sha256(path),
        len(rows),
    ) == identity


@dataclass
class _IdentityAdapter:
    def validate_state(self, state: TrainingState) -> None:
        return None

    def build_samples(self, records):
        return tuple(records)

    def validate_sample(self, sample):
        return None

    def stamp_samples(self, samples, generation):
        return tuple(
            {
                **sample,
                "source_generation": generation,
                "replay_row_id": f"M{generation}:{position}",
            }
            for position, sample in enumerate(samples)
        )

    def update_replay(self, replay, generation, samples):
        replay.extend(samples)
        return {"generation_added": generation}

    def replay_rows(self, replay):
        return tuple(replay)

    def validate_replay(self, rows):
        assert rows

    def replay_identity(self, replay):
        return {
            "schema": TORUS9_REPLAY_COMPOSITION_IDENTITY_SCHEMA,
            "fingerprint": _sha("a"),
            "components": [],
            "contract": {"selection": "test"},
        }

    def train(self, state, rows, seed):
        state.optimizer_updates += 1
        return {
            "optimizer_steps": 1,
            "samples_consumed": 1,
            "sampled_replay_row_ids": (rows[0]["replay_row_id"],),
        }

    def prepare_checkpoint(self, state, context, training_metrics):
        return {
            "checkpoint_schema_version": 1,
            "checkpoint_label": context.label,
            "model_hash": _sha("b"),
            "optimizer_updates": state.optimizer_updates,
        }

    def save_checkpoint(self, path, state, metadata):
        path.write_text("checkpoint", encoding="utf-8")
        path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return metadata

    def verify_checkpoint(self, path, state, metadata):
        assert path.is_file()

    def artifact_hash(self, path):
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def snapshot_state(self, state):
        return (list(state.rolling_replay), state.optimizer_updates)

    def restore_state(self, state, snapshot):
        replay, updates = snapshot
        state.rolling_replay[:] = replay
        state.optimizer_updates = updates

    def sync_state(self, state):
        return None


def test_engine_uses_adapter_identity_without_full_replay_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy full replay fingerprint must not run")

    monkeypatch.setattr(training_engine, "sequence_fingerprint", forbidden)
    state = TrainingState(
        model=object(),
        optimizer=object(),
        optimizer_updates=0,
        samples_consumed=0,
        current_generation=0,
        rolling_replay=[],
        profile_identity={},
        target_identity={},
    )
    result = TrainingEngine(_IdentityAdapter()).run_iteration(
        state=state,
        generation=1,
        output_dir=tmp_path,
        run_id="identity-test",
        records=({"valid": True},),
        training_seed=1,
        device="cpu",
    )

    marker = json.loads((tmp_path / "generation-01.complete.json").read_text(encoding="utf-8"))
    assert marker["replay_fingerprint"] == _sha("a")
    assert marker["replay_identity_schema"] == TORUS9_REPLAY_COMPOSITION_IDENTITY_SCHEMA
    phase = result.training_metrics["phase_timing"]
    assert abs(float(phase["replay_update_and_validation_unaccounted_sec"])) < 0.01


class _PhysicalIdentityAdapter(_IdentityAdapter):
    def __init__(self) -> None:
        self.hashed_paths: list[Path] = []

    def update_replay(self, replay, generation, samples):
        return replay.append_generation(generation, samples)

    def replay_rows(self, replay):
        return replay.rows

    def replay_identity(self, replay):
        return replay.replay_identity_descriptor()

    def set_replay_generation_artifact_identity(self, replay, generation, sha256, row_count):
        replay.set_generation_artifact_identity(generation, sha256, row_count)

    def snapshot_state(self, state):
        return (tuple(state.rolling_replay.rows), state.optimizer_updates)

    def restore_state(self, state, snapshot):
        rows, updates = snapshot
        restored = Torus9RollingReplay.from_persisted_rows(rows)
        state.rolling_replay._rows = list(restored.rows)
        state.rolling_replay._last_generation = restored.last_generation
        state.optimizer_updates = updates

    def artifact_hash(self, path):
        self.hashed_paths.append(path)
        return super().artifact_hash(path)


def test_engine_hashes_fresh_artifact_during_its_single_write(tmp_path: Path) -> None:
    adapter = _PhysicalIdentityAdapter()
    state = TrainingState(
        model=object(),
        optimizer=object(),
        optimizer_updates=0,
        samples_consumed=0,
        current_generation=0,
        rolling_replay=Torus9RollingReplay(generations=3, maximum_positions=10),
        profile_identity={},
        target_identity={},
    )
    TrainingEngine(adapter).run_iteration(
        state=state,
        generation=1,
        output_dir=tmp_path,
        run_id="physical-identity-test",
        records=({"valid": True},),
        training_seed=1,
        device="cpu",
    )

    fresh = tmp_path / "replay/iter-01-fresh.jsonl"
    marker = json.loads((tmp_path / "generation-01.complete.json").read_text(encoding="utf-8"))
    assert marker["fresh_replay_sha256"] == file_sha256(fresh)
    assert fresh not in adapter.hashed_paths
