"""Stage 3 to Stage 4 experimental contracts and evidence helpers.

This module deliberately lives beside, rather than inside, the Stage 3 runner.
The Stage 3 profile, checkpoints, replay and original Arena are read-only
inputs.  Stage 4 adds a frozen independent-start corpus, conservative pair
statistics, explicit training schedules, and audit/diagnostic helpers.
"""

from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import resource
import time
from typing import Any, Iterable, Mapping, Sequence

torch = importlib.import_module("torch")
F = importlib.import_module("torch.nn.functional")

from .arena import MappedResult, TerminationReason
from .neural import GoldenGraphNetV1, GoldenNeuralEvaluator, build_observation, count_parameters, model_hash
from .provenance import capture_code_identity, derive_seed, file_sha256, sha256_fingerprint
from .result import Winner
from .rules import apply_action, legal_actions
from .state import PASS, GoldenState, initial_state
from .topology import TORUS_5X5
from .training import (
    GoldenSelfPlayRunner,
    GoldenTrainingSample,
    SelfPlayGameRecord,
    build_replay_samples,
    load_checkpoint,
    run_selfplay_games,
    save_checkpoint,
    state_from_identity,
    state_identity,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONTRACT_ID = "golden-evaluation-startset-v2"
EVALUATION_MASTER_SEED = 2026091301
EVALUATION_PREFIX_LENGTHS = (2, 4, 6, 8, 10, 12, 14, 16)
EVALUATION_ACCEPTED_PER_STRATUM = 8
EVALUATION_ACCEPTED_TOTAL = 64
STAGE4_PROFILE_ID = "gocube-torus-golden-training-v2-data-rich"
STAGE4_MODEL_INIT_SEED = 2026091302
STAGE4_SELFPLAY_MASTER_SEED = 2026091303
STAGE4_ARENA_MASTER_SEED = 2026091304
HOEFFDING_ALPHA = 0.05


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def hoeffding_interval(pair_scores: Sequence[float], *, alpha: float = HOEFFDING_ALPHA) -> tuple[float, float]:
    """Distribution-free bounded-mean interval for independent pair scores."""
    if not pair_scores:
        raise ValueError("Hoeffding interval requires at least one pair score")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Hoeffding alpha must be between zero and one")
    if any(not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0 for score in pair_scores):
        raise ValueError("Pair scores must be finite values in [0,1]")
    mean = sum(float(score) for score in pair_scores) / len(pair_scores)
    epsilon = math.sqrt(math.log(2.0 / alpha) / (2.0 * len(pair_scores)))
    return (max(0.0, mean - epsilon), min(1.0, mean + epsilon))


def _coord(point: int) -> tuple[int, int]:
    return point % 5, point // 5


def _point(x: int, y: int) -> int:
    return (y % 5) * 5 + (x % 5)


def torus_automorphism_permutations() -> tuple[tuple[int, ...], ...]:
    """Return all 200 translations composed with the 8 D4 automorphisms."""
    permutations: list[tuple[int, ...]] = []
    for reflected in (False, True):
        for rotation in range(4):
            for tx in range(5):
                for ty in range(5):
                    mapped: list[int] = []
                    for point in range(25):
                        x, y = _coord(point)
                        if reflected:
                            x = -x
                        for _ in range(rotation):
                            x, y = -y, x
                        mapped.append(_point(x + tx, y + ty))
                    permutation = tuple(mapped)
                    if permutation not in permutations:
                        permutations.append(permutation)
    if len(permutations) != 200:
        raise RuntimeError(f"Expected 200 Torus automorphisms, got {len(permutations)}")
    for permutation in permutations:
        if sorted(permutation) != list(range(25)):
            raise RuntimeError("Torus automorphism is not a point permutation")
        for point, neighbors in enumerate(TORUS_5X5.adjacency):
            if {permutation[n] for n in neighbors} != {TORUS_5X5.neighbors(permutation[point])[i] for i in range(4)}:
                raise RuntimeError("Torus automorphism does not preserve adjacency")
    return tuple(permutations)


TORUS_AUTOMORPHISMS = torus_automorphism_permutations()


def _transform_state_identity(identity: Mapping[str, object], permutation: Sequence[int]) -> dict[str, object]:
    """Transform every board in full superko history, not only the live board."""
    transformed = dict(identity)
    stones = tuple(int(value) for value in identity["stones"])  # type: ignore[index]
    history = tuple(tuple(int(value) for value in board) for board in identity["superko_history"])  # type: ignore[index]
    new_stones = [0] * 25
    for old, new in enumerate(permutation):
        new_stones[new] = stones[old]
    transformed["stones"] = new_stones
    transformed["superko_history"] = [
        [board[old] for old in _inverse_permutation(permutation)] for board in history
    ]
    return transformed


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for old, new in enumerate(permutation):
        inverse[new] = old
    return tuple(inverse)


def symmetry_canonical_state_identity(identity: Mapping[str, object]) -> dict[str, object]:
    transformed = [_transform_state_identity(identity, permutation) for permutation in TORUS_AUTOMORPHISMS]
    return min(transformed, key=canonical_json)


def state_identity_fingerprint(identity: Mapping[str, object]) -> str:
    return sha256_fingerprint(identity)


def symmetry_identity_fingerprint(identity: Mapping[str, object]) -> str:
    return sha256_fingerprint(symmetry_canonical_state_identity(identity))


def candidate_start_seed(master_seed: int, prefix_length: int, candidate_index: int) -> int:
    return derive_seed(master_seed, prefix_length, candidate_index)


def _random_candidate(master_seed: int, prefix_length: int, candidate_index: int) -> tuple[dict[str, object], tuple[int, ...], int]:
    seed = candidate_start_seed(master_seed, prefix_length, candidate_index)
    rng = random.Random(seed)
    state = initial_state()
    trace: list[int] = []
    for _ in range(prefix_length):
        choices = tuple(action for action in legal_actions(state) if action != PASS)
        if not choices:
            raise ValueError("no legal non-pass placement available")
        action = rng.choice(choices)
        state = apply_action(state, action).after
        trace.append(int(action))
    if state.is_terminal:
        raise ValueError("candidate prefix reached a terminal state")
    return state_identity(state), tuple(trace), seed


def generate_evaluation_v2(
    *,
    master_seed: int = EVALUATION_MASTER_SEED,
    prefix_lengths: Sequence[int] = EVALUATION_PREFIX_LENGTHS,
    accepted_per_stratum: int = EVALUATION_ACCEPTED_PER_STRATUM,
) -> tuple[dict[str, object], ...]:
    if tuple(prefix_lengths) != EVALUATION_PREFIX_LENGTHS:
        raise ValueError("Evaluation V2 prefix strata are frozen to lengths 2..16")
    if accepted_per_stratum != EVALUATION_ACCEPTED_PER_STRATUM:
        raise ValueError("Evaluation V2 accepted count per stratum is frozen to eight")
    exact_seen: set[str] = set()
    symmetry_seen: set[str] = set()
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for prefix_length in prefix_lengths:
        count = 0
        candidate_index = 0
        while count < accepted_per_stratum:
            try:
                identity, trace, seed = _random_candidate(master_seed, prefix_length, candidate_index)
                exact = state_identity_fingerprint(identity)
                symmetry = symmetry_identity_fingerprint(identity)
                reason = None
                if exact in exact_seen:
                    reason = "exact_semantic_duplicate"
                if reason is not None:
                    rejected.append({
                        "prefix_length": prefix_length,
                        "candidate_index": candidate_index,
                        "candidate_seed": seed,
                        "reason": reason,
                        "exact_identity_fingerprint": exact,
                        "symmetry_identity_fingerprint": symmetry,
                        "symmetry_duplicate_diagnostic": symmetry in symmetry_seen,
                    })
                else:
                    start_id = f"prefix-{prefix_length:02d}-accepted-{count:02d}"
                    accepted.append({
                        "start_id": start_id,
                        "prefix_length": prefix_length,
                        "candidate_index": candidate_index,
                        "candidate_seed": seed,
                        "trace": list(trace),
                        "state": identity,
                        "exact_identity_fingerprint": exact,
                        "symmetry_identity_fingerprint": symmetry,
                        "symmetry_duplicate_diagnostic": symmetry in symmetry_seen,
                    })
                    exact_seen.add(exact)
                    symmetry_seen.add(symmetry)
                    count += 1
            except Exception as exc:
                rejected.append({
                    "prefix_length": prefix_length,
                    "candidate_index": candidate_index,
                    "candidate_seed": candidate_start_seed(master_seed, prefix_length, candidate_index),
                    "reason": f"candidate_rejected:{type(exc).__name__}",
                    "error": str(exc),
                })
            candidate_index += 1
            if candidate_index > 10000:
                raise RuntimeError(f"Could not accept eight independent starts at prefix length {prefix_length}")
    if len(accepted) != EVALUATION_ACCEPTED_TOTAL:
        raise RuntimeError(f"Expected 64 Evaluation V2 starts, got {len(accepted)}")
    # Rejection evidence is attached to the manifest by freeze_evaluation_v2.
    for row in accepted:
        row["_rejected_candidates_before_acceptance"] = [
            item for item in rejected
            if item["prefix_length"] == row["prefix_length"] and int(item["candidate_index"]) < int(row["candidate_index"])
        ]
    return tuple(accepted)


def evaluation_start_fingerprint(starts: Sequence[Mapping[str, object]]) -> str:
    return sha256_fingerprint([
        {key: value for key, value in row.items() if key != "_rejected_candidates_before_acceptance"}
        for row in starts
    ])


def diagnostic_subset(starts: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    selected: list[dict[str, object]] = []
    for prefix_length in EVALUATION_PREFIX_LENGTHS:
        stratum = [row for row in starts if int(row["prefix_length"]) == prefix_length]
        selected.extend(stratum[:2])
    if len(selected) != 16:
        raise ValueError("Evaluation V2 diagnostic subset must contain two starts per stratum")
    return tuple(selected)


def freeze_evaluation_v2(run_dir: Path, *, master_seed: int = EVALUATION_MASTER_SEED, code=None) -> dict[str, object]:
    code = code or capture_code_identity(ROOT)
    starts = generate_evaluation_v2(master_seed=master_seed)
    subset = diagnostic_subset(starts)
    evaluation_dir = run_dir / "evaluation-v2"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(evaluation_dir / "starts.jsonl", starts)
    _write_json(evaluation_dir / "diagnostic-subset.json", {
        "contract_id": EVALUATION_CONTRACT_ID,
        "selection": "first two accepted start IDs in each frozen prefix-length stratum",
        "starts": subset,
        "subset_fingerprint": evaluation_start_fingerprint(subset),
    })
    rejected_by_id = {
        (int(item["prefix_length"]), int(item["candidate_index"])): item
        for row in starts
        for item in row.get("_rejected_candidates_before_acceptance", ())
    }
    manifest = {
        "contract_id": EVALUATION_CONTRACT_ID,
        "generator_version": "independent-legal-nonpass-prefix-v2",
        "master_seed": master_seed,
        "prefix_lengths": list(EVALUATION_PREFIX_LENGTHS),
        "accepted_per_stratum": EVALUATION_ACCEPTED_PER_STRATUM,
        "accepted_starts": len(starts),
        "rejected_candidates": len(rejected_by_id),
        "rejected_candidate_evidence": list(rejected_by_id.values()),
        "dedupe_rules": {
            "exact": "full Golden state identity including full positional-superko history",
            "symmetry": "diagnostic-only: 200 full-history Torus automorphisms; current board and every history board transformed",
            "symmetry_rejection": False,
            "symmetry_limitation": "full-history symmetry classes are reported but not rejected because prefix-length-2 has fewer than eight possible classes; exact semantic dedupe remains mandatory",
            "board_only": False,
        },
        "start_fingerprints": [row["exact_identity_fingerprint"] for row in starts],
        "symmetry_start_fingerprints": [row["symmetry_identity_fingerprint"] for row in starts],
        "full_set_fingerprint": evaluation_start_fingerprint(starts),
        "diagnostic_subset_fingerprint": evaluation_start_fingerprint(subset),
        "empty_board_control": {
            "inferential": False,
            "state": state_identity(initial_state()),
            "trace": [],
        },
        "git_sha": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "git_worktree_clean": code.working_tree_clean,
        "candidate_seed_derivation": "derive_seed(master_seed, prefix_length, candidate_index)",
        "start_generation": "legal non-pass placements from empty Golden state",
    }
    _write_json(evaluation_dir / "manifest.json", manifest)
    return manifest


def load_frozen_starts(run_dir: Path) -> tuple[dict[str, object], ...]:
    path = run_dir / "evaluation-v2" / "starts.jsonl"
    rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(rows) != EVALUATION_ACCEPTED_TOTAL:
        raise ValueError("Frozen Evaluation V2 corpus does not contain 64 starts")
    manifest = json.loads((run_dir / "evaluation-v2" / "manifest.json").read_text(encoding="utf-8"))
    if manifest["full_set_fingerprint"] != evaluation_start_fingerprint(rows):
        raise ValueError("Frozen Evaluation V2 fingerprint mismatch")
    for row in rows:
        state = state_from_identity(row["state"])
        if state.is_terminal or len(row["trace"]) != int(row["prefix_length"]):
            raise ValueError("Frozen Evaluation V2 start is invalid")
        replayed = initial_state()
        for action in row["trace"]:
            replayed = apply_action(replayed, int(action)).after
        if state_identity(replayed) != row["state"]:
            raise ValueError("Frozen Evaluation V2 start trace does not reconstruct state")
    return rows


def state_from_start_row(row: Mapping[str, object]) -> GoldenState:
    state = state_from_identity(row["state"])  # type: ignore[arg-type]
    trace = tuple(int(action) for action in row["trace"])  # type: ignore[index]
    replayed = initial_state()
    for action in trace:
        replayed = apply_action(replayed, action).after
    if replayed.state_key != state.state_key:
        raise ValueError("Start row state and trace disagree")
    return state


def pair_score(mapped: MappedResult) -> float:
    if mapped == MappedResult.A_WIN:
        return 1.0
    if mapped == MappedResult.DRAW:
        return 0.5
    if mapped == MappedResult.B_WIN:
        return 0.0
    raise ValueError(f"Unknown mapped Arena result {mapped!r}")


def summarize_pair_records(
    records: Sequence[Any],
    *,
    candidate_label: str,
    reference_label: str,
    starts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_pair: dict[str, list[Any]] = {}
    for record in records:
        by_pair.setdefault(record.pair_id, []).append(record)
    pair_rows: list[dict[str, object]] = []
    pair_scores: list[float] = []
    candidate_black_scores: list[float] = []
    candidate_white_scores: list[float] = []
    left_wins = left_losses = draws = technical = 0
    black_wins = white_wins = 0
    for pair_id, games in by_pair.items():
        if len(games) != 2:
            raise ValueError(f"Pair {pair_id} does not contain exactly two games")
        games = sorted(games, key=lambda record: record.black_player)
        if any(record.termination_reason != TerminationReason.DOUBLE_PASS for record in games):
            technical += sum(record.termination_reason != TerminationReason.DOUBLE_PASS for record in games)
            continue
        scores = [pair_score(record.mapped_result) for record in games]
        score = sum(scores) / 2.0
        pair_scores.append(score)
        wins = sum(value == 1.0 for value in scores)
        losses = sum(value == 0.0 for value in scores)
        draws_in_pair = 2 - wins - losses
        left_wins += wins
        left_losses += losses
        draws += draws_in_pair
        candidate_black = next(record for record in games if record.black_player == "A")
        candidate_white = next(record for record in games if record.black_player == "B")
        candidate_black_scores.append(pair_score(candidate_black.mapped_result))
        candidate_white_scores.append(pair_score(candidate_white.mapped_result))
        black_wins += int(candidate_black.absolute_rule_result == Winner.BLACK)
        black_wins += int(candidate_white.absolute_rule_result == Winner.BLACK)
        white_wins += int(candidate_black.absolute_rule_result == Winner.WHITE)
        white_wins += int(candidate_white.absolute_rule_result == Winner.WHITE)
        pair_rows.append({
            "pair_id": pair_id,
            "pair_score": score,
            "candidate_black_score": candidate_black_scores[-1],
            "candidate_white_score": candidate_white_scores[-1],
            "wins": wins,
            "losses": losses,
            "draws": draws_in_pair,
        })
    if technical:
        interval = None
    else:
        interval = list(hoeffding_interval(pair_scores))
    start_lookup = {str(row["start_id"]): row for row in starts}
    stratum: dict[str, dict[str, Any]] = {}
    for row in pair_rows:
        start_id = row["pair_id"].split("--", 1)[-1]
        start = start_lookup.get(start_id)
        if start is None:
            continue
        key = str(start["prefix_length"])
        bucket = stratum.setdefault(key, {"pairs": 0, "pair_scores": []})
        bucket["pairs"] += 1
        bucket["pair_scores"].append(row["pair_score"])
    for bucket in stratum.values():
        bucket["mean_pair_score"] = sum(bucket["pair_scores"]) / len(bucket["pair_scores"])
    split_pairs = sum(row["wins"] == 1 and row["losses"] == 1 for row in pair_rows)
    two_zero = sum(row["wins"] == 2 for row in pair_rows)
    zero_two = sum(row["losses"] == 2 for row in pair_rows)
    draw_pairs = sum(row["draws"] > 0 for row in pair_rows)
    games = len(records)
    return {
        "candidate": candidate_label,
        "reference": reference_label,
        "pairs": len(pair_scores),
        "games": games,
        "technical": technical,
        "mean_pair_score": (sum(pair_scores) / len(pair_scores)) if pair_scores else None,
        "pair_score_distribution": pair_scores,
        "pair_rows": pair_rows,
        "candidate_wins": left_wins,
        "candidate_losses": left_losses,
        "draws": draws,
        "W/L/D": [left_wins, left_losses, draws],
        "candidate_as_black_score": (sum(candidate_black_scores) / len(candidate_black_scores)) if candidate_black_scores else None,
        "candidate_as_white_score": (sum(candidate_white_scores) / len(candidate_white_scores)) if candidate_white_scores else None,
        "candidate_as_black_wins": sum(value == 1.0 for value in candidate_black_scores),
        "candidate_as_white_wins": sum(value == 1.0 for value in candidate_white_scores),
        "overall_black_wins": black_wins,
        "overall_white_wins": white_wins,
        "overall_black_win_fraction": black_wins / games if games else None,
        "two_zero_pairs": two_zero,
        "split_one_one_pairs": split_pairs,
        "zero_two_pairs": zero_two,
        "draw_containing_pairs": draw_pairs,
        "split_pair_fraction": split_pairs / len(pair_scores) if pair_scores else None,
        "color_controlled_discriminative_power": (
            "LOW COLOR-CONTROLLED DISCRIMINATIVE POWER" if pair_scores and split_pairs / len(pair_scores) >= 0.75 else "not flagged"
        ),
        "prefix_stratum_breakdown": stratum,
        "95_percent_hoeffding_interval": interval,
        "interval_method": "Hoeffding bounded independent start-pair mean; primary",
        "old_zero_variance_ci_used": False,
    }


def _tensor_batch(samples: Sequence[GoldenTrainingSample], indices: Sequence[int], device: torch.device):
    observations = torch.tensor([samples[index].observation for index in indices], dtype=torch.float32, device=device)
    policies = torch.tensor([samples[index].pi for index in indices], dtype=torch.float32, device=device)
    values = torch.tensor([samples[index].z for index in indices], dtype=torch.float32, device=device)
    return observations, policies, values


def evaluate_model_samples(model: torch.nn.Module, samples: Sequence[GoldenTrainingSample]) -> dict[str, float | int | None]:
    if not samples:
        return {"samples": 0, "policy_ce": None, "value_ce": None, "value_accuracy": None, "policy_entropy": None, "value_entropy": None}
    device = next(model.parameters()).device
    observations = torch.tensor([sample.observation for sample in samples], dtype=torch.float32, device=device)
    policies = torch.tensor([sample.pi for sample in samples], dtype=torch.float32, device=device)
    values = torch.tensor([sample.z for sample in samples], dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        policy_logits, value_logits = model(observations)
        policy = torch.softmax(policy_logits, dim=1)
        value = torch.softmax(value_logits, dim=1)
        policy_ce = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        value_ce = -(values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
        accuracy = (value.argmax(dim=1) == values.argmax(dim=1)).float().mean()
        policy_entropy = -(policy * policy.clamp_min(1e-12).log()).sum(dim=1).mean()
        value_entropy = -(value * value.clamp_min(1e-12).log()).sum(dim=1).mean()
    return {
        "samples": len(samples),
        "policy_ce": float(policy_ce.cpu()),
        "value_ce": float(value_ce.cpu()),
        "value_accuracy": float(accuracy.cpu()),
        "policy_entropy": float(policy_entropy.cpu()),
        "value_entropy": float(value_entropy.cpu()),
    }


def train_batch_schedule(
    model: torch.nn.Module,
    samples: Sequence[GoldenTrainingSample],
    batch_indices: Sequence[Sequence[int]],
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
    update_offset: int = 0,
    sample_offset: int = 0,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    if not samples or not batch_indices:
        raise ValueError("Training schedule requires samples and at least one batch")
    if any(not batch for batch in batch_indices):
        raise ValueError("Training schedule cannot contain an empty batch")
    for sample in samples:
        sample.validate()
    device = next(model.parameters()).device
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    update_rows: list[dict[str, object]] = []
    consumed = int(sample_offset)
    updates = int(update_offset)
    for batch in batch_indices:
        observations, policies, values = _tensor_batch(samples, batch, device)
        policy_logits, value_logits = model(observations)
        policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        value_loss = -(values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
        total_loss = policy_loss + value_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("Stage 4 training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError("Stage 4 training produced non-finite gradient")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError("Stage 4 training produced non-finite parameter")
        updates += 1
        consumed += len(batch)
        update_rows.append({
            "update": updates,
            "batch_size": len(batch),
            "cumulative_samples": consumed,
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "gradient_norm": float(grad_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
    return optimizer, {
        "updates": updates,
        "exact_samples_consumed": consumed,
        "batch_count": len(batch_indices),
        "batch_sizes": [len(batch) for batch in batch_indices],
        "final_batch_size": len(batch_indices[-1]),
        "first_update": update_rows[0],
        "last_update": update_rows[-1],
        "updates_detail": update_rows,
    }


def high_reuse_schedule(sample_count: int, *, sample_budget: int = 25600, batch_size: int = 64, seed: int) -> tuple[tuple[int, ...], ...]:
    if sample_count <= 0 or sample_budget <= 0 or batch_size <= 0 or sample_budget % batch_size:
        raise ValueError("High-reuse schedule requires positive divisible sample budget")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return tuple(
        tuple(int(index) for index in torch.randint(0, sample_count, (batch_size,), generator=generator).tolist())
        for _ in range(sample_budget // batch_size)
    )


def low_reuse_schedule(sample_count: int, *, batch_size: int = 64, seed: int) -> tuple[tuple[int, ...], ...]:
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("Low-reuse schedule requires positive sample count and batch size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = [int(index) for index in torch.randperm(sample_count, generator=generator).tolist()]
    return tuple(tuple(permutation[offset:offset + batch_size]) for offset in range(0, sample_count, batch_size))


def clone_model_from(model: GoldenGraphNetV1, device: torch.device) -> GoldenGraphNetV1:
    clone = GoldenGraphNetV1().to(device)
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    return clone


def parameter_diagnostics(before: Mapping[str, torch.Tensor], model: torch.nn.Module) -> dict[str, float]:
    squared = 0.0
    baseline_squared = 0.0
    for name, after in model.state_dict().items():
        old = before[name].detach().cpu()
        new = after.detach().cpu()
        squared += float(torch.sum((new - old) ** 2))
        baseline_squared += float(torch.sum(old ** 2))
    delta = math.sqrt(squared)
    return {"l2_delta": delta, "relative_l2_delta": delta / math.sqrt(baseline_squared) if baseline_squared else None}


def fixed_observation_samples(starts: Sequence[Mapping[str, object]], *, run_id: str = "diagnostic") -> tuple[GoldenTrainingSample, ...]:
    """Build a deterministic observation corpus for entropy/parameter diagnostics."""
    result: list[GoldenTrainingSample] = []
    for index, row in enumerate(starts):
        state = state_from_start_row(row)
        observation = build_observation(state)
        # Targets are deliberately neutral placeholders; only the observation
        # and model output entropies are consumed by this diagnostic corpus.
        result.append(GoldenTrainingSample(
            run_id=run_id,
            game_id=f"diagnostic-{index:03d}",
            ply=1,
            state=state_identity(state),
            side_to_move=state.side_to_move.name,
            observation=tuple(tuple(float(value) for value in row_values) for row_values in observation.tolist()),
            legal_action_mask=tuple(bool(action in legal_actions(state)) for action in range(25)) + (PASS in legal_actions(state),),
            root_visits=tuple(1 if action in legal_actions(state) else 0 for action in range(25)) + (1,),
            pi=tuple((1.0 / len(legal_actions(state)) if action in legal_actions(state) else 0.0) for action in range(25)) + (1.0 / len(legal_actions(state)) if PASS in legal_actions(state) else 0.0,),
            z=(1.0, 0.0, 0.0),
            model_hash="sha256:" + "0" * 64,
            selfplay_contract_fingerprint="sha256:" + "0" * 64,
        ))
    return tuple(result)


def policy_value_entropy(model: torch.nn.Module, observations: Sequence[GoldenTrainingSample]) -> dict[str, float]:
    if not observations:
        return {"policy_entropy": float("nan"), "value_entropy": float("nan"), "pass_probability": float("nan"), "top1_probability": float("nan"), "legal_probability_mass_before_mask": float("nan"), "samples": 0}
    device = next(model.parameters()).device
    inputs = torch.tensor([sample.observation for sample in observations], dtype=torch.float32, device=device)
    legal_masks = torch.tensor([sample.legal_action_mask for sample in observations], dtype=torch.bool, device=device)
    model.eval()
    with torch.inference_mode():
        policy_logits, value_logits = model(inputs)
        probabilities = torch.softmax(policy_logits, dim=1)
        values = torch.softmax(value_logits, dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1).mean()
        value_entropy = -(values * values.clamp_min(1e-12).log()).sum(dim=1).mean()
        pass_probability = probabilities[:, 25].mean()
        top1 = probabilities.max(dim=1).values.mean()
        legal_mass = (probabilities * legal_masks).sum(dim=1).mean()
    return {
        "policy_entropy": float(entropy.cpu()),
        "value_entropy": float(value_entropy.cpu()),
        "pass_probability": float(pass_probability.cpu()),
        "top1_probability": float(top1.cpu()),
        "legal_probability_mass_before_mask": float(legal_mass.cpu()),
        "samples": len(observations),
    }


def audit_selfplay_records(records: Sequence[SelfPlayGameRecord], *, expected_model_hash_by_label: Mapping[str, str] | None = None) -> dict[str, object]:
    outcomes = {"BLACK": 0, "WHITE": 0, "DRAW": 0}
    lengths: list[int] = []
    pass_count = 0
    short_10 = short_20 = 0
    positions = 0
    technical = 0
    z_errors: list[str] = []
    for record in records:
        record.validate(require_clean=False)
        if record.technical_termination is not None:
            technical += 1
            continue
        if expected_model_hash_by_label and record.model_hash != expected_model_hash_by_label.get(record.model_checkpoint_label):
            raise ValueError(f"Self-play source model hash mismatch for {record.game_id}")
        state = initial_state()
        replay_samples = build_replay_samples(record)
        for position, action in zip(record.positions, record.final_action_trace):
            if state_identity(state) != position.state:
                raise ValueError(f"Self-play state trace mismatch for {record.game_id} ply {position.ply}")
            if action != position.selected_action or action not in legal_actions(state):
                raise ValueError(f"Self-play selected action mismatch for {record.game_id} ply {position.ply}")
            state = apply_action(state, action).after
            positions += 1
        if record.formal_result not in outcomes:
            raise ValueError(f"Self-play formal result missing for {record.game_id}")
        if not state.is_terminal:
            raise ValueError(f"Self-play final state is not terminal for {record.game_id}")
        outcomes[record.formal_result] += 1
        length = len(record.final_action_trace)
        lengths.append(length)
        pass_count += sum(action == PASS for action in record.final_action_trace)
        short_10 += int(length < 10)
        short_20 += int(length < 20)
        for position, replay_sample in zip(record.positions, replay_samples):
            # Recompute through the public target helper to keep perspective
            # checking explicit and independent from stored z.
            from .training import z_target
            expected_z = z_target(record.formal_result, position.side_to_move)
            if tuple(float(value) for value in expected_z) != tuple(float(value) for value in replay_sample.z):
                z_errors.append(f"{record.game_id}:{position.ply}")
    if z_errors:
        raise ValueError("z perspective audit failed: " + ",".join(z_errors[:5]))
    lengths_sorted = sorted(lengths)
    def percentile(frac: float) -> float | None:
        if not lengths_sorted:
            return None
        return float(lengths_sorted[min(len(lengths_sorted) - 1, int(frac * len(lengths_sorted)))])
    return {
        "games": len(records),
        "valid_games": len(records) - technical,
        "technical_games": technical,
        "positions": positions,
        "outcomes": outcomes,
        "game_length": {
            "mean": sum(lengths) / len(lengths) if lengths else None,
            "median": percentile(0.5),
            "p10": percentile(0.1),
            "p90": percentile(0.9),
        },
        "pass_actions": pass_count,
        "pass_frequency": pass_count / sum(lengths) if lengths else None,
        "games_under_10_moves": short_10,
        "games_under_20_moves": short_20,
        "z_perspective": "PASS: BLACK-to-move WIN / WHITE-to-move LOSS and vice versa; no exceptions",
        "independence_limitation": "Golden replay audit checks artifact consistency against the same Golden rules implementation; it is not an independent rules oracle.",
    }


def runtime_telemetry(start_wall: float, start_cpu: float, start_rss: int, *, completed_games: int, positions: int, nn_evaluations: int, training_wall: float | None = None) -> dict[str, object]:
    wall = max(0.0, time.perf_counter() - start_wall)
    cpu = max(0.0, resource.getrusage(resource.RUSAGE_SELF).ru_utime - start_cpu)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "wall_time_sec": wall,
        "cpu_time_sec": cpu,
        "cpu_utilization_percent_process": 100.0 * cpu / wall if wall else None,
        "ram_peak_mb": float(rss) / 1024.0,
        "games_completed": completed_games,
        "games_per_hour": completed_games * 3600.0 / wall if wall else None,
        "positions": positions,
        "positions_per_sec": positions / wall if wall else None,
        "nn_evaluations": nn_evaluations,
        "nn_evaluations_per_sec": nn_evaluations / wall if wall else None,
        "training_wall_time_sec": training_wall,
        "initial_ru_maxrss_kb": start_rss,
    }
