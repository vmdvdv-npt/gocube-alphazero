from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from training_engine import TrainingEngine, TrainingState


class _CountingAdapter:
    """Minimal adapter that makes validation ownership observable."""

    def __init__(self) -> None:
        self.validate_calls = 0
        self.construction_only_calls = 0

    def validate_state(self, state: TrainingState) -> None:
        assert isinstance(state.rolling_replay, list)

    def build_samples(self, records: Sequence[object]) -> Sequence[Mapping[str, object]]:
        rows = tuple({"record": str(record), "value": 1} for record in records)
        for row in rows:
            self.validate_sample(row)
        return rows

    def build_samples_for_replay(
        self, records: Sequence[object]
    ) -> Sequence[Mapping[str, object]]:
        self.construction_only_calls += 1
        raise AssertionError("TrainingEngine must not use construction-only capability yet")

    def validate_sample(self, sample: Mapping[str, object]) -> None:
        self.validate_calls += 1
        if int(sample.get("value", 0)) != 1:
            raise ValueError("bad sample")

    def stamp_samples(
        self, samples: Sequence[Mapping[str, object]], generation: int
    ) -> Sequence[Mapping[str, object]]:
        return tuple(
            {
                **dict(sample),
                "source_generation": int(generation),
                "replay_row_id": f"M{generation}:{index}",
            }
            for index, sample in enumerate(samples)
        )

    def update_replay(
        self,
        replay: list[dict[str, object]],
        generation: int,
        samples: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        for sample in samples:
            self.validate_sample(sample)
        replay.extend(dict(sample) for sample in samples)
        return {"generation_added": int(generation), "rolling_buffer_positions": len(replay)}

    def replay_rows(self, replay: list[dict[str, object]]) -> Sequence[Mapping[str, object]]:
        return tuple(dict(row) for row in replay)

    def validate_replay(self, rows: Sequence[Mapping[str, object]]) -> None:
        assert rows
        assert all(row.get("replay_row_id") for row in rows)

    def train(
        self,
        state: TrainingState,
        rows: Sequence[Mapping[str, object]],
        seed: int,
    ) -> Mapping[str, object]:
        return {
            "optimizer_steps": 1,
            "samples_consumed": 1,
            "sampled_replay_row_ids": [str(rows[0]["replay_row_id"])],
        }

    def prepare_checkpoint(self, state, context, training_metrics):
        return {"model_hash": "sha256:" + "0" * 64}

    def save_checkpoint(self, path: Path, state, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint\n", encoding="utf-8")
        path.with_suffix(".metadata.json").write_text(
            json.dumps(dict(metadata), sort_keys=True) + "\n", encoding="utf-8"
        )
        return dict(metadata)

    def verify_checkpoint(self, path: Path, state, metadata) -> None:
        assert path.is_file()
        assert path.with_suffix(".metadata.json").is_file()

    def artifact_hash(self, path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    def snapshot_state(self, state: TrainingState):
        return {
            "rolling_replay": copy.deepcopy(state.rolling_replay),
            "current_generation": state.current_generation,
            "completed_games": state.completed_games,
            "parent_checkpoint_identity": copy.deepcopy(state.parent_checkpoint_identity),
        }

    def restore_state(self, state: TrainingState, snapshot) -> None:
        state.rolling_replay[:] = copy.deepcopy(snapshot["rolling_replay"])
        state.current_generation = int(snapshot["current_generation"])
        state.completed_games = int(snapshot["completed_games"])
        state.parent_checkpoint_identity = copy.deepcopy(snapshot["parent_checkpoint_identity"])

    def sync_state(self, state: TrainingState) -> None:
        return None


def _state() -> TrainingState:
    return TrainingState(
        model=object(),
        optimizer=object(),
        optimizer_updates=0,
        samples_consumed=0,
        current_generation=0,
        rolling_replay=[],
        profile_identity={"profile": "test"},
        target_identity={"target": "test"},
    )


def test_record_built_samples_are_not_revalidated_by_engine(tmp_path: Path) -> None:
    adapter = _CountingAdapter()
    result = TrainingEngine(adapter).run_iteration(
        state=_state(),
        generation=1,
        output_dir=tmp_path,
        run_id="validation-contract",
        training_seed=1,
        records=("game-0",),
    )

    # Exactly one validation belongs to build_samples and one belongs to
    # update_replay immediately before replay mutation. The generic engine
    # must not add duplicate full passes between those adapter boundaries.
    assert adapter.validate_calls == 2
    assert adapter.construction_only_calls == 0
    assert result.fresh_positions == 1
    assert result.summary["iteration"] == 1


def test_direct_samples_keep_pre_validation(tmp_path: Path) -> None:
    adapter = _CountingAdapter()
    TrainingEngine(adapter).run_iteration(
        state=_state(),
        generation=1,
        output_dir=tmp_path,
        run_id="direct-sample-validation",
        training_seed=1,
        samples=({"record": "external", "value": 1},),
    )

    # External rows are still validated once before stamping, then the stamped
    # row is validated again by update_replay before state mutation.
    assert adapter.validate_calls == 2
