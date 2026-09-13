"""Frozen independent evaluation starts and pair scheduling for Cube V1."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .cube_training import cube_initial_state, cube_state_identity
from .cube_topology import CUBE4_TOPOLOGY
from .provenance import CodeIdentity, derive_seed, sha256_fingerprint
from .rules import apply_action, legal_actions
from .state import PASS

CUBE_EVALUATION_ID = "cube-golden-evaluation-v1"
CUBE_EVALUATION_MASTER_SEED = 2026091403
CUBE_EVALUATION_PREFIX_LENGTHS = (4, 8, 12, 16, 24, 32, 40, 48)
CUBE_EVALUATION_PER_STRATUM = 8


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate(master_seed: int, prefix_length: int, candidate_index: int) -> dict[str, object]:
    seed = derive_seed(master_seed, "cube-evaluation", prefix_length, candidate_index)
    rng = random.Random(seed)
    state = cube_initial_state()
    trace: list[int] = []
    for _ in range(prefix_length):
        choices = tuple(action for action in legal_actions(state) if action != PASS)
        if not choices:
            raise ValueError("Cube evaluation generation ran out of legal non-pass actions")
        action = int(rng.choice(choices))
        state = apply_action(state, action).after
        trace.append(action)
    if state.is_terminal:
        raise ValueError("Cube evaluation prefix unexpectedly reached a terminal state")
    identity = cube_state_identity(state)
    return {
        "start_id": f"cube-eval-p{prefix_length:02d}-{candidate_index:04d}",
        "prefix_length": prefix_length,
        "candidate_index": candidate_index,
        "seed": seed,
        "trace": trace,
        "state": identity,
        "state_fingerprint": sha256_fingerprint(identity),
    }


def generate_cube_evaluation(
    *,
    master_seed: int = CUBE_EVALUATION_MASTER_SEED,
    prefix_lengths: Sequence[int] = CUBE_EVALUATION_PREFIX_LENGTHS,
    accepted_per_stratum: int = CUBE_EVALUATION_PER_STRATUM,
) -> tuple[dict[str, object], ...]:
    starts, _ = _generate_cube_evaluation_with_diagnostics(
        master_seed=master_seed,
        prefix_lengths=prefix_lengths,
        accepted_per_stratum=accepted_per_stratum,
    )
    return starts


def _generate_cube_evaluation_with_diagnostics(
    *,
    master_seed: int,
    prefix_lengths: Sequence[int],
    accepted_per_stratum: int,
) -> tuple[tuple[dict[str, object], ...], int]:
    if tuple(prefix_lengths) != CUBE_EVALUATION_PREFIX_LENGTHS or accepted_per_stratum != CUBE_EVALUATION_PER_STRATUM:
        raise ValueError("Cube Evaluation V1 strata are frozen")
    accepted: list[dict[str, object]] = []
    seen: set[str] = set()
    rejected_duplicates = 0
    for prefix_length in prefix_lengths:
        count = 0
        candidate_index = 0
        while count < accepted_per_stratum:
            row = _candidate(master_seed, prefix_length, candidate_index)
            candidate_index += 1
            identity_key = canonical_json(row["state"])
            if identity_key in seen:
                rejected_duplicates += 1
                continue
            seen.add(identity_key)
            row["stratum_index"] = count
            accepted.append(row)
            count += 1
    return tuple(accepted), rejected_duplicates


def cube_evaluation_fingerprint(starts: Sequence[Mapping[str, object]]) -> str:
    return sha256_fingerprint(
        {
            "evaluation_id": CUBE_EVALUATION_ID,
            "topology_fingerprint": CUBE4_TOPOLOGY.fingerprint,
            "starts": list(starts),
        }
    )


def diagnostic_cube_subset(starts: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    result = []
    for prefix_length in CUBE_EVALUATION_PREFIX_LENGTHS:
        result.extend(row for row in starts if int(row["prefix_length"]) == prefix_length)
    selected = tuple(row for row in result if int(row["stratum_index"]) < 2)
    if len(selected) != 16:
        raise ValueError("Cube diagnostic subset must contain two starts per stratum")
    return selected


def freeze_cube_evaluation(
    run_dir: str | Path,
    *,
    code_identity: CodeIdentity,
    master_seed: int = CUBE_EVALUATION_MASTER_SEED,
) -> dict[str, object]:
    root = Path(run_dir) / "evaluation"
    starts, rejected_duplicates = _generate_cube_evaluation_with_diagnostics(
        master_seed=master_seed,
        prefix_lengths=CUBE_EVALUATION_PREFIX_LENGTHS,
        accepted_per_stratum=CUBE_EVALUATION_PER_STRATUM,
    )
    manifest = {
        "evaluation_id": CUBE_EVALUATION_ID,
        "master_seed": master_seed,
        "topology_fingerprint": CUBE4_TOPOLOGY.fingerprint,
        "prefix_lengths": list(CUBE_EVALUATION_PREFIX_LENGTHS),
        "accepted_per_stratum": CUBE_EVALUATION_PER_STRATUM,
        "accepted": len(starts),
        "rejected_exact_semantic_duplicates": rejected_duplicates,
        "exact_semantic_identity": [
            "current_board", "side_to_move", "consecutive_passes", "full_positional_superko_history", "rules", "komi"
        ],
        "corpus_fingerprint": cube_evaluation_fingerprint(starts),
        "diagnostic_subset_fingerprint": cube_evaluation_fingerprint(diagnostic_cube_subset(starts)),
        "source_commit": code_identity.git_commit_sha,
        "source_tree": code_identity.git_tree_sha,
        "source_clean": code_identity.working_tree_clean,
        "immutable_after_first_primary_arena": True,
    }
    _write_json(root / "manifest.json", manifest)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "starts.jsonl").open("w", encoding="utf-8") as handle:
        for row in starts:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(root / "diagnostic-subset.json", list(diagnostic_cube_subset(starts)))
    return manifest


def load_frozen_cube_starts(run_dir: str | Path) -> tuple[dict[str, object], ...]:
    root = Path(run_dir) / "evaluation"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = tuple(json.loads(line) for line in (root / "starts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    if len(rows) != 64 or manifest.get("corpus_fingerprint") != cube_evaluation_fingerprint(rows):
        raise ValueError("Frozen Cube Evaluation V1 corpus fingerprint/count mismatch")
    for row in rows:
        if int(row["prefix_length"]) not in CUBE_EVALUATION_PREFIX_LENGTHS:
            raise ValueError("Frozen Cube evaluation contains an unknown stratum")
    return rows
