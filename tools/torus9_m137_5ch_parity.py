"""Bounded numerical and behavioral parity proof for the M137 5-channel model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

import torch

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.neural import model_hash
from gocube_golden.provenance import file_sha256
from gocube_golden.rules import prepare_legal_actions
from gocube_golden.scoring import score_terminal
from gocube_golden.search import SequentialPUCT
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import rules_fingerprint_for
from gocube_golden.torus9_contract import TORUS9_POINT_COUNT
from gocube_golden.torus9_m137_5ch import (
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    M137_FIVE_CHANNEL_FORMULA,
    Torus9M137FiveChannelEvaluator,
    build_m137_five_channel_observation,
    convert_m137_model,
    load_canonical_m137,
    save_converted_checkpoint,
)
from gocube_golden.torus9_monolith import (
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_TOPOLOGY_FINGERPRINT,
    Torus9NeuralEvaluator,
    build_torus9_observation,
    torus9_state_from_identity,
)
from gocube_golden.topology import TORUS_9X9


SOURCE_LINEAGE = "torus9-m125-continuous-v2-gen6-20260922-v1"
SOURCE_CHECKPOINT_LABEL = "M137"
SOURCE_CHECKPOINT_SHA256 = "sha256:71cfc78dab3fe217b3c435a765790efe6f6c4fa42a7d21479f3fd909adf341fe"
SOURCE_MODEL_ARCHITECTURE = "GoldenGraphNetV2-Torus9"
DERIVED_ARTIFACT_ID = "torus9-m137-5ch-derived-20260925-v1"
DEFAULT_SOURCE_CHECKPOINT = (
    Path("runs") / "torus9" / "active" / SOURCE_LINEAGE / "checkpoints" / "M137.pt"
)
DEFAULT_REPLAY = Path("runs") / "torus9" / "active" / SOURCE_LINEAGE / "replay" / "iter-137-fresh.jsonl"
DEFAULT_ARTIFACT_DIR = Path("runs") / "torus9" / "active" / DERIVED_ARTIFACT_ID
DEFAULT_POSITION_COUNT = 4096
DEFAULT_MCTS_COUNT = 256
MCTS_SEED = 202609250137


@dataclass
class ErrorAccumulator:
    chunks: list[torch.Tensor]
    nonfinite_values: int = 0
    compared_values: int = 0

    @classmethod
    def create(cls) -> "ErrorAccumulator":
        return cls(chunks=[])

    def add(self, left: torch.Tensor, right: torch.Tensor) -> None:
        left_cpu = left.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        right_cpu = right.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if left_cpu.shape != right_cpu.shape:
            raise ValueError(f"Parity tensor shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
        left_finite = torch.isfinite(left_cpu)
        right_finite = torch.isfinite(right_cpu)
        diff = (left_cpu - right_cpu).abs()
        finite_diff = torch.isfinite(diff)
        self.compared_values += int(left_cpu.numel())
        self.nonfinite_values += int((~left_finite).sum()) + int((~right_finite).sum())
        if bool(finite_diff.any()):
            self.chunks.append(diff[finite_diff])

    def summary(self) -> dict[str, object]:
        if not self.chunks:
            return {
                "max_abs_error": None,
                "mean_abs_error": None,
                "p99_abs_error": None,
                "nonfinite_values": self.nonfinite_values,
                "compared_values": self.compared_values,
            }
        total_finite = sum(int(chunk.numel()) for chunk in self.chunks)
        exact_max = max(float(chunk.max()) for chunk in self.chunks)
        exact_sum = sum(float(chunk.sum()) for chunk in self.chunks)
        # torch.quantile rejects very large input tensors.  Select evenly spaced
        # values from the complete deterministic stream for a bounded p99 sample;
        # max and mean above remain exact over every finite comparison.
        sample_budget = 1_000_000
        sampled: list[torch.Tensor] = []
        offset = 0
        for chunk in self.chunks:
            count = int(chunk.numel())
            take = max(1, min(count, (count * sample_budget + total_finite - 1) // total_finite))
            indices = torch.linspace(0, count - 1, steps=take, dtype=torch.long)
            sampled.append(chunk[indices])
            offset += count
        values = torch.cat(sampled)
        return {
            "max_abs_error": exact_max,
            "mean_abs_error": exact_sum / total_finite,
            "p99_abs_error": float(torch.quantile(values, 0.99)),
            "nonfinite_values": self.nonfinite_values,
            "compared_values": self.compared_values,
        }


def _read_bounded_rows(path: Path, limit: int) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise ValueError(f"Replay has only {len(rows)} rows; {limit} are required")
    return rows


def _prepare_positions(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[object], list[torch.Tensor], list[torch.Tensor]]:
    states: list[object] = []
    legacy_observations: list[torch.Tensor] = []
    converted_observations: list[torch.Tensor] = []
    for row in rows:
        state = torus9_state_from_identity(row["state"], expected_komi=0.5)  # type: ignore[arg-type]
        if state.is_terminal:
            raise ValueError("Parity corpus contains a terminal position")
        context = prepare_legal_actions(state)
        legacy = build_torus9_observation(state, legal_context=context)
        converted = build_m137_five_channel_observation(state, legal_context=context)
        if tuple(legacy.shape) != (6, TORUS9_POINT_COUNT) or not torch.all(legacy[5] == 0.5):
            raise ValueError("Legacy M137 parity observation is not [6,81] with komi plane 0.5")
        if tuple(converted.shape) != (5, TORUS9_POINT_COUNT):
            raise ValueError("M137-derived parity observation is not [5,81]")
        replay_observation = row.get("observation")
        if replay_observation is not None:
            serialized = torch.tensor(replay_observation, dtype=torch.float32)  # type: ignore[arg-type]
            if not torch.equal(serialized, legacy):
                raise ValueError("Canonical replay observation does not match the reconstructed M137 position")
        states.append(state)
        legacy_observations.append(legacy)
        converted_observations.append(converted)
    return states, legacy_observations, converted_observations


def _run_numerical_parity(
    source_model: torch.nn.Module,
    converted_model: torch.nn.Module,
    legacy_observations: list[torch.Tensor],
    converted_observations: list[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, object]]:
    accumulators = {
        "first_projection": ErrorAccumulator.create(),
        "encoded_nodes": ErrorAccumulator.create(),
        "policy_logits": ErrorAccumulator.create(),
        "wdl_logits": ErrorAccumulator.create(),
        "ownership_logits": ErrorAccumulator.create(),
        "score_output": ErrorAccumulator.create(),
    }
    source_model.eval()
    converted_model.eval()
    with torch.inference_mode():
        for start in range(0, len(legacy_observations), batch_size):
            old = torch.stack(legacy_observations[start : start + batch_size]).to(device)
            new = torch.stack(converted_observations[start : start + batch_size]).to(device)
            source_projection = source_model.input_projection(old.transpose(1, 2))
            converted_projection = converted_model.input_projection(new.transpose(1, 2))
            source_nodes = source_model.encode(old)
            converted_nodes = converted_model.encode(new)
            source_outputs = source_model.forward_auxiliary(old)
            converted_outputs = converted_model.forward_auxiliary(new)
            accumulators["first_projection"].add(source_projection, converted_projection)
            accumulators["encoded_nodes"].add(source_nodes, converted_nodes)
            for name, source_output, converted_output in zip(
                ("policy_logits", "wdl_logits", "ownership_logits", "score_output"),
                source_outputs,
                converted_outputs,
            ):
                accumulators[name].add(source_output, converted_output)
    return {name: accumulator.summary() for name, accumulator in accumulators.items()}


def _run_mcts_parity(
    states: list[object],
    source_model: torch.nn.Module,
    converted_model: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, object]:
    settings = SearchSettings(
        simulations=64,
        cpuct=1.25,
        fpu=0.0,
        root_noise=False,
        fast_search=False,
        resign=False,
        root_policy_temperature=False,
        move_temperature=0.0,
        deterministic_tie_break=True,
    )
    adapter = GoldenSearchAdapter()
    source_evaluator = Torus9NeuralEvaluator(source_model, device=device)
    converted_evaluator = Torus9M137FiveChannelEvaluator(converted_model, device=device)
    selected_matches = 0
    root_visits_matches = 0
    root_q_accumulator = ErrorAccumulator.create()
    mismatch_details: list[dict[str, object]] = []
    for index, state in enumerate(states):
        seed = MCTS_SEED + index
        source_result = SequentialPUCT(settings, adapter=adapter).search(state, source_evaluator, seed=seed)  # type: ignore[arg-type]
        converted_result = SequentialPUCT(settings, adapter=adapter).search(state, converted_evaluator, seed=seed)  # type: ignore[arg-type]
        selected_equal = source_result.action == converted_result.action
        visits_equal = source_result.root_visits == converted_result.root_visits
        selected_matches += int(selected_equal)
        root_visits_matches += int(visits_equal)
        source_q = torch.tensor(
            [float(value) if value is not None else float("nan") for value in source_result.root_q],
            dtype=torch.float32,
        )
        converted_q = torch.tensor(
            [float(value) if value is not None else float("nan") for value in converted_result.root_q],
            dtype=torch.float32,
        )
        root_q_accumulator.add(source_q, converted_q)
        if not selected_equal or not visits_equal:
            mismatch_details.append(
                {
                    "index": index,
                    "legacy_action": source_result.action,
                    "converted_action": converted_result.action,
                    "legacy_root_visits": list(source_result.root_visits),
                    "converted_root_visits": list(converted_result.root_visits),
                    "legacy_root_q": list(source_result.root_q),
                    "converted_root_q": list(converted_result.root_q),
                }
            )
    return {
        "positions": len(states),
        "simulations_per_position": settings.simulations,
        "selected_action_matches": selected_matches,
        "root_visits_matches": root_visits_matches,
        "selected_action_parity": f"{selected_matches}/{len(states)}",
        "root_q_error": root_q_accumulator.summary(),
        "mismatch_details": mismatch_details,
        "settings": settings.__dict__,
    }


def _run_komi_independence(
    state: object,
    converted_model: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, object]:
    base_state = state
    variants = []
    observations = []
    for komi in (0.5, 2.5, 4.5):
        identity = {
            "stones": [int(stone) for stone in base_state.stones],  # type: ignore[attr-defined]
            "side_to_move": int(base_state.side_to_move),  # type: ignore[attr-defined]
            "superko_history": [list(position) for position in base_state.superko_history],  # type: ignore[attr-defined]
            "consecutive_passes": int(base_state.consecutive_passes),  # type: ignore[attr-defined]
            "topology_id": base_state.topology.topology_id,  # type: ignore[attr-defined]
            "topology_fingerprint": base_state.topology.fingerprint,  # type: ignore[attr-defined]
            "rules_id": base_state.rules_id,  # type: ignore[attr-defined]
            "rules_fingerprint": rules_fingerprint_for(TORUS_9X9, komi),
            "komi": komi,
            "history_provenance": base_state.history_provenance,  # type: ignore[attr-defined]
        }
        variant = torus9_state_from_identity(identity, expected_komi=komi)
        variants.append(variant)
        observations.append(build_m137_five_channel_observation(variant))
    observation_equal = all(torch.equal(observations[0], item) for item in observations[1:])
    with torch.inference_mode():
        outputs = [converted_model.forward_auxiliary(item.unsqueeze(0).to(device)) for item in observations]
    neural_errors: list[float] = []
    for reference, candidate in zip(outputs[0], outputs[1]):
        neural_errors.append(float((reference - candidate).abs().max()))
    for reference, candidate in zip(outputs[0], outputs[2]):
        neural_errors.append(float((reference - candidate).abs().max()))
    terminal_scores = []
    for variant in variants:
        terminal = replace(
            variant,
            consecutive_passes=2,
            rules_fingerprint=rules_fingerprint_for(TORUS_9X9, variant.komi),
        )
        terminal_scores.append(float(score_terminal(terminal).margin_black))
    return {
        "komi_values": [0.5, 2.5, 4.5],
        "observation_equal": observation_equal,
        "max_neural_output_abs_error": max(neural_errors),
        "terminal_score_margins_black": terminal_scores,
        "terminal_score_differs": len(set(terminal_scores)) == 3,
    }


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(args: argparse.Namespace) -> dict[str, object]:
    source_checkpoint = args.source_checkpoint.resolve()
    replay = args.replay.resolve()
    artifact_dir = args.artifact_dir.resolve()
    source_sha = file_sha256(source_checkpoint)
    if source_sha != SOURCE_CHECKPOINT_SHA256:
        raise ValueError(f"Canonical M137 SHA mismatch: expected {SOURCE_CHECKPOINT_SHA256}, got {source_sha}")
    source_model, source_metadata = load_canonical_m137(source_checkpoint, device="cpu")
    if source_metadata.get("checkpoint_label") != SOURCE_CHECKPOINT_LABEL:
        raise ValueError("Parity source is not checkpoint M137")
    if source_metadata.get("architecture_id") != SOURCE_MODEL_ARCHITECTURE:
        raise ValueError("Parity source architecture is not GoldenGraphNetV2-Torus9")
    if source_metadata.get("observation_shape") != [6, 81] or float(source_metadata.get("komi", -1.0)) != 0.5:
        raise ValueError("Canonical M137 metadata observation or komi contract drift")
    rows = _read_bounded_rows(replay, args.position_count)
    states, legacy_observations, converted_observations = _prepare_positions(rows)
    converted_model = convert_m137_model(source_model)
    converter_commit = _git_commit()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    converted_checkpoint = artifact_dir / "M137-5CH.pt"
    converted_metadata = save_converted_checkpoint(
        converted_checkpoint,
        model=converted_model,
        source_checkpoint=source_checkpoint,
        source_metadata=source_metadata,
        converter_git_commit=converter_commit,
    )
    device = torch.device(args.device)
    source_model.to(device)
    converted_model.to(device)
    numerical = _run_numerical_parity(
        source_model,
        converted_model,
        legacy_observations,
        converted_observations,
        device=device,
        batch_size=args.batch_size,
    )
    mcts = _run_mcts_parity(
        states[: args.mcts_count],
        source_model,
        converted_model,
        device=device,
    )
    komi_independence = _run_komi_independence(states[0], converted_model, device=device)
    artifact_manifest = {
        "schema": "torus9-derived-model-artifact-v1",
        "artifact_id": DERIVED_ARTIFACT_ID,
        "parent_lineage": SOURCE_LINEAGE,
        "parent_checkpoint": SOURCE_CHECKPOINT_LABEL,
        "source_checkpoint_path": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "source_architecture": SOURCE_MODEL_ARCHITECTURE,
        "target_architecture": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "source_model_hash": source_metadata.get("model_hash"),
        "converted_model_hash": model_hash(converted_model),
        "converter_git_commit": converter_commit,
        "conversion_formula": M137_FIVE_CHANNEL_FORMULA,
        "optimizer_conversion": "not_performed",
        "training_ready": False,
        "replay_copied": False,
        "checkpoint": {
            "path": str(converted_checkpoint),
            "artifact_sha256": converted_metadata["artifact_sha256"],
            "metadata_path": str(converted_checkpoint.with_suffix(".metadata.json")),
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finite_error_pass = all(
        item["nonfinite_values"] == 0 and float(item["max_abs_error"]) <= 1e-5
        for item in numerical.values()
    )
    behavioral_pass = (
        mcts["selected_action_matches"] == args.mcts_count
        and mcts["root_visits_matches"] == args.mcts_count
        and mcts["root_q_error"]["nonfinite_values"] == 0
        and float(mcts["root_q_error"]["max_abs_error"]) <= 1e-5
    )
    komi_pass = (
        komi_independence["observation_equal"]
        and komi_independence["terminal_score_differs"]
        and float(komi_independence["max_neural_output_abs_error"]) <= 1e-7
    )
    result = {
        "status": "PARITY PASS" if finite_error_pass and behavioral_pass and komi_pass else "PARITY FAIL",
        "source": {
            "lineage": SOURCE_LINEAGE,
            "checkpoint": SOURCE_CHECKPOINT_LABEL,
            "checkpoint_sha256": source_sha,
            "model_hash": source_metadata.get("model_hash"),
            "architecture": source_metadata.get("architecture_id"),
            "observation_shape": source_metadata.get("observation_shape"),
        },
        "target": {
            "architecture": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
            "model_hash": model_hash(converted_model),
            "checkpoint": str(converted_checkpoint),
            "optimizer_conversion": "not_performed",
            "training_ready": False,
        },
        "conversion": {
            "formula": M137_FIVE_CHANNEL_FORMULA,
            "converter_git_commit": converter_commit,
            "source_checkpoint_unchanged": source_sha == SOURCE_CHECKPOINT_SHA256,
        },
        "numerical_parity": {
            "positions": len(rows),
            "selection": "first bounded rows from canonical M137-compatible iter-137-fresh.jsonl",
            "heads": numerical,
        },
        "behavioral_parity": mcts,
        "komi_independence": komi_independence,
        "acceptance": {
            "numerical_max_abs_error_le_1e-5": finite_error_pass,
            "mcts_selected_action_and_root_distribution_parity": behavioral_pass,
            "komi_independent_neural_input_and_distinct_referee_score": komi_pass,
        },
        "artifact_manifest": str(artifact_dir / "manifest.json"),
    }
    (artifact_dir / "parity_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--position-count", type=int, default=DEFAULT_POSITION_COUNT)
    parser.add_argument("--mcts-count", type=int, default=DEFAULT_MCTS_COUNT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.position_count < args.mcts_count or args.mcts_count <= 0:
        parser.error("position-count must be >= mcts-count > 0")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PARITY PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
