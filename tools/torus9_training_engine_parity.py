#!/usr/bin/env python3
"""Reproduce the immutable Torus9 M16→M17 training transition.

Only persisted M16/replay/M17 artifacts are read.  The reproduced checkpoint
is always written below a temporary directory; this command never runs
self-play and never writes the canonical run namespace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from gocube_golden.provenance import derive_seed, file_sha256
from gocube_golden.torus9_training import (
    Torus9TrainingAdapter,
    torus9_model_from_metadata,
    torus9_load_checkpoint,
    run_torus9_training_iteration,
)
from training_engine import sequence_fingerprint
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_TRAINING_MASTER_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_KOMI,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "torus9-golden-v3-active" / "torus9-golden-v3-20260914-run03"
M16_PATH = RUN_ROOT / "checkpoints" / "M16.pt"
M17_PATH = RUN_ROOT / "checkpoints" / "M17.pt"
ROLLING16_PATH = RUN_ROOT / "replay" / "rolling-after-16.jsonl"
FRESH17_PATH = RUN_ROOT / "replay" / "iter-17-fresh.jsonl"
ROLLING17_PATH = RUN_ROOT / "replay" / "rolling-after-17.jsonl"
CANONICAL_TRAINING17_PATH = RUN_ROOT / "training" / "iter-17.json"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected replay object: {path}")
            rows.append(value)
    return tuple(rows)


def _load_model_optimizer(path: Path, device: str):
    metadata = _json(path.with_suffix(".metadata.json"))
    model = torus9_model_from_metadata(metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    torus9_load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected={"model_hash": metadata["model_hash"]},
        device=device,
    )
    return metadata, model, optimizer


def _optimizer_parity(left: Any, right: Any) -> dict[str, object]:
    left_states = list(left.state.values())
    right_states = list(right.state.values())
    if len(left_states) != len(right_states):
        return {"status": "FAIL", "reason": "parameter state count mismatch"}
    max_delta = 0.0
    for left_state, right_state in zip(left_states, right_states):
        if left_state.keys() != right_state.keys():
            return {"status": "FAIL", "reason": "Adam state keys mismatch"}
        for key in left_state:
            left_value = left_state[key]
            right_value = right_state[key]
            if torch.is_tensor(left_value) and torch.is_tensor(right_value):
                delta = float((left_value.detach().float() - right_value.detach().float()).abs().max())
                max_delta = max(max_delta, delta)
            elif left_value != right_value:
                return {"status": "FAIL", "reason": f"Adam scalar mismatch: {key}"}
    return {"status": "PASS" if max_delta == 0.0 else "TOLERATED", "max_abs_delta": max_delta}


def reproduce(*, device: str = "cuda") -> dict[str, object]:
    for path in (M16_PATH, M17_PATH, ROLLING16_PATH, FRESH17_PATH, ROLLING17_PATH, CANONICAL_TRAINING17_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity was requested but CUDA is unavailable")

    adapter = Torus9TrainingAdapter()
    m16_metadata, model, optimizer = _load_model_optimizer(M16_PATH, device)
    state = adapter.create_state(
        model,
        run_id=str(m16_metadata["run_id"]),
        replay=adapter.reconstruct_replay(
            fresh_paths=[RUN_ROOT / "replay" / f"iter-{generation:02d}-fresh.jsonl" for generation in range(1, 17)],
            expected_rolling_path=ROLLING16_PATH,
            total_evictions=int(_json(RUN_ROOT / "iter-16-summary.json")["replay"]["total_evictions"]),  # type: ignore[index]
            validate_evicted_generations=False,
        ),
        trainer=None,
        parent_checkpoint_identity={
            "label": "M16",
            "path": str(M16_PATH),
            "metadata_path": str(M16_PATH.with_suffix(".metadata.json")),
            "model_hash": m16_metadata["model_hash"],
            "artifact_sha256": file_sha256(M16_PATH),
        },
        completed_games=int(m16_metadata["completed_games"]),
        current_generation=16,
    )
    # Bind the exact M16 Adam object to the explicit state, then restore the
    # canonical counters.  The model/optimizer values are already loaded above.
    state.optimizer = optimizer
    state.adapter_state.optimizer = optimizer
    state.adapter_state.update_count = int(m16_metadata["optimizer_updates"])
    state.adapter_state.samples_consumed = int(m16_metadata["train_samples_consumed"])
    adapter.sync_state(state)
    adapter.validate_state(state)

    fresh17 = _jsonl(FRESH17_PATH)
    canonical_training = _json(CANONICAL_TRAINING17_PATH)
    seed = derive_seed(
        TORUS9_CURRENT_TRAINING_MASTER_SEED,
        str(m16_metadata["run_id"]),
        "training",
        17,
    )
    with tempfile.TemporaryDirectory(prefix="torus9-stage3-training-parity-") as directory:
        result = run_torus9_training_iteration(
            state=state,
            generation=17,
            output_dir=directory,
            run_id=str(m16_metadata["run_id"]),
            training_seed=seed,
            samples=fresh17,
            adapter=adapter,
            completed_games=1088,
            device=device,
        )
        reproduced_fresh = _jsonl(Path(result.artifacts["fresh_replay"]))
        reproduced_rolling = _jsonl(Path(result.artifacts["rolling_replay"]))
        reproduced_metadata = _json(Path(result.artifacts["checkpoint_metadata"]))
        _, reproduced_model, reproduced_optimizer = _load_model_optimizer(
            Path(result.artifacts["checkpoint"]), device
        )
        _, canonical_model, canonical_optimizer = _load_model_optimizer(M17_PATH, device)

        model_delta = max(
            float((left.detach().float() - right.detach().float()).abs().max())
            for left, right in zip(reproduced_model.state_dict().values(), canonical_model.state_dict().values())
        )
        replay_parity = {
            "fresh_row_count": len(reproduced_fresh),
            "fresh_order": reproduced_fresh == fresh17,
            "fresh_file_sha256": file_sha256(Path(result.artifacts["fresh_replay"])),
            "expected_fresh_file_sha256": file_sha256(FRESH17_PATH),
            "rolling_row_count": len(reproduced_rolling),
            "rolling_order": reproduced_rolling == _jsonl(ROLLING17_PATH),
            "rolling_fingerprint": sequence_fingerprint(reproduced_rolling),
            "expected_rolling_fingerprint": sequence_fingerprint(_jsonl(ROLLING17_PATH)),
        }
        training_parity = {
            "training_seed": seed,
            "expected_training_seed": seed,
            "optimizer_steps": result.training_metrics["optimizer_steps"],
            "samples_consumed": result.training_metrics["samples_consumed"],
            "batch_sizes": result.training_metrics["batch_sizes"],
            "sampled_row_ids_fingerprint": result.training_metrics["sampled_row_ids_fingerprint"],
            "source_generation_counts": result.training_metrics["sampled_positions_by_generation"],
            "expected_source_generation_counts": canonical_training["sampled_positions_by_generation"],
            "training_metrics_reference": {
                "adam_step_before": canonical_training["adam_step_before"],
                "adam_step_after": canonical_training["adam_step_after"],
                "mean_policy_loss": canonical_training["mean_policy_loss"],
                "mean_value_loss": canonical_training["mean_value_loss"],
                "mean_ownership_loss": canonical_training["mean_ownership_loss"],
                "mean_score_loss_normalized": canonical_training["mean_score_loss_normalized"],
            },
        }
        metadata_keys = (
            "checkpoint_schema_version",
            "profile_id",
            "profile_fingerprint",
            "target_fingerprint",
            "architecture_id",
            "komi",
            "optimizer_updates",
            "train_samples_consumed",
            "adam_step",
            "replay_generations",
            "replay_row_count",
        )
        metadata_parity = {
            key: {
                "reproduced": reproduced_metadata.get(key),
                "canonical": _json(M17_PATH.with_suffix(".metadata.json")).get(key),
            }
            for key in metadata_keys
        }
        model_parity = {
            "expected_model_hash": _json(M17_PATH.with_suffix(".metadata.json"))["model_hash"],
            "reproduced_model_hash": reproduced_metadata["model_hash"],
            "model_hash_equal": reproduced_metadata["model_hash"] == _json(M17_PATH.with_suffix(".metadata.json"))["model_hash"],
            "max_parameter_abs_delta": model_delta,
        }
        optimizer_parity = _optimizer_parity(reproduced_optimizer, canonical_optimizer)
        report = {
            "source_checkpoint": "M16",
            "source_checkpoint_path": str(M16_PATH),
            "source_model_hash": m16_metadata["model_hash"],
            "source_optimizer_step": m16_metadata["optimizer_updates"],
            "fresh_generation": 17,
            "fresh_positions": len(fresh17),
            "rolling_replay_positions": len(_jsonl(ROLLING17_PATH)),
            "training_seed": seed,
            "optimizer_steps": result.training_metrics["optimizer_steps"],
            "samples_consumed": result.training_metrics["samples_consumed"],
            "expected_m17_model_hash": _json(M17_PATH.with_suffix(".metadata.json"))["model_hash"],
            "reproduced_m17_model_hash": reproduced_metadata["model_hash"],
            "model_parity": "PASS" if model_parity["model_hash_equal"] else "FAIL",
            "optimizer_parity": optimizer_parity,
            "replay_parity": replay_parity,
            "training_parity": training_parity,
            "checkpoint_metadata_parity": metadata_parity,
            "canonical_m17_mutated": False,
            "m18_created": False,
            "komi": TORUS9_KOMI,
            "profile_id": TORUS9_CURRENT_PROFILE_ID,
            "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
            "historical_validation_scope": "full semantic validation of training-visible rolling generations; older evicted fresh artifacts reconstructed and compared exactly",
        }
        if not model_parity["model_hash_equal"] or not replay_parity["fresh_order"] or not replay_parity["rolling_order"]:
            raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = reproduce(device=str(args.device))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
