#!/usr/bin/env python3
"""Final Stage 7 acceptance harness for the Torus9 Golden pipeline.

The harness is orchestration only.  It deliberately delegates rules, search,
self-play, replay, training, checkpoint loading, Arena, and Protocol V1 to
their current public boundaries.  A run is isolated from the historical
Golden namespace and is limited to M0..M6.

The normal end-to-end command is::

    .venv/bin/python tools/torus9_stage7_e2e_validation.py all \
      --run-id torus9-golden-stage7-20260915-run01

``all`` launches the M0..M2 and M2..M6 phases as separate Python processes.
The individual commands are useful when a long Legion run must be operated
in two sessions: ``new``, ``resume``, ``arena``, ``protocol``, and ``report``.
No command creates M7 or writes to the historical run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gocube_golden.provenance import (
    CodeIdentity,
    capture_code_identity,
    derive_seed,
    file_sha256,
)
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9SelfPlayGameRecord,
    Torus9SelfPlaySearchContract,
    Torus9TrainingAdapter,
    run_torus9_selfplay_games,
    run_torus9_training_iteration,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_ARENA_MASTER_SEED,
    TORUS9_CURRENT_BLOCKS,
    TORUS9_CURRENT_HIDDEN,
    TORUS9_CURRENT_MODEL_INIT_SEED,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_CURRENT_TRAINING_MASTER_SEED,
    TORUS9_KOMI,
    TORUS9_MAX_REPLAY_POSITIONS,
    TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    TORUS9_ROLLING_GENERATIONS,
    TORUS9_RULES_FINGERPRINT,
    TORUS9_TRAINING_SAMPLES_PER_ITERATION,
    current_torus9_profile_fingerprint,
    current_torus9_selfplay_contract_fingerprint,
    load_torus9_current_profile,
)
from gocube_golden.neural import model_hash
from gocube_golden.execution_reference import LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE
from training_engine import sequence_fingerprint, value_fingerprint


# This is the freshly fetched post-#108 main that the Stage 7 branch starts
# from.  The branch/source commit is recorded separately in every report.
BASE_SHA = "d0b81a51f46ef8a22dca75fac04d07cfad26d9eb"
REFERENCE_RUN_ID = "torus9-golden-v3-20260914-run03"
REFERENCE_M17_SHA = "86722afe70fefd1d4a2a408e47c3492c7b6da43e86f283888d8815da5b037e53"
DEFAULT_RUN_ID = "torus9-golden-stage7-20260915-run01"
VALIDATION_HORIZON = 6
SHORT_RUN_STOP_AFTER = 2
RESUME_STOP_AFTER = 6
SELFPLAY_GAMES = 64
ARENA_FIXED_GAMES = 64
ARENA_STRENGTH_INITIAL_GAMES = 256
ARENA_STRENGTH_TOTAL_GAMES = 512
ARENA_NON_INFERIORITY_MARGIN = 0.05
BOOTSTRAP_REPLICATES = 20_000
CANONICAL_CHECKPOINTS = ("M0", "M5", "M6", "M10", "M17")
CANONICAL_ARENA_COMPARISON = "M10-vs-M5"
DEFAULT_REFERENCE_RUN = (
    ROOT / "runs" / "torus9-golden-v3-active" / REFERENCE_RUN_ID
)


@dataclass(frozen=True)
class Stage7ExecutionConfig:
    """The validated PR #108 Legion self-play execution preset."""

    workers: int = 16
    active_games_per_worker: int = 4
    active_contexts: int = 64
    inference_batch_cap: int = 64
    inference_batch_wait_ms: float = 1.0
    worker_local_wait_ms: float = 0.0
    shared_memory: bool = True
    central_model_owner: str = "parent"
    device: str = "cuda"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        expected = Stage7ExecutionConfig()
        if self != expected:
            raise ValueError(
                "Stage 7 self-play execution is fixed to the validated Legion "
                f"preset: {expected.as_dict()}"
            )


EXECUTION = Stage7ExecutionConfig()


@dataclass(frozen=True)
class Stage7ArenaConfig:
    """Current Arena settings required by the Stage 7 acceptance contract."""

    games: int
    simulations: int = 64
    workers: int = 16
    batched: bool = True
    arena_batch_size: int = 8
    central_inference_batch_cap: int = 64
    inference_batch_wait_ms: float = 6.0
    cpuct: float = 1.25
    fpu: float = 0.0
    root_noise: bool = False
    temperature: float = 0.0
    fast_search: bool = False
    resign: bool = False
    komi: float = 0.5
    paired_color_swap: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        if self.games < 2 or self.games % 2:
            raise ValueError("Stage 7 Arena games must be a positive even count")
        expected = {
            "simulations": 64,
            "workers": 16,
            "batched": True,
            "arena_batch_size": 8,
            "central_inference_batch_cap": 64,
            "inference_batch_wait_ms": 6.0,
            "cpuct": 1.25,
            "fpu": 0.0,
            "root_noise": False,
            "temperature": 0.0,
            "fast_search": False,
            "resign": False,
            "komi": 0.5,
            "paired_color_swap": True,
        }
        actual = self.as_dict()
        if any(actual[key] != value for key, value in expected.items()):
            raise ValueError(f"Stage 7 Arena contract drift: {actual}")


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return tuple(rows)


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _safe_run_id(run_id: str) -> str:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("Stage 7 run ID must be a single non-empty path component")
    return run_id


def default_run_dir(run_id: str = DEFAULT_RUN_ID) -> Path:
    return ROOT / "runs" / "torus9-stage7" / _safe_run_id(run_id)


def _assert_isolated(run_dir: Path, canonical_run: Path) -> Path:
    run_dir = run_dir.resolve()
    canonical_run = canonical_run.resolve()
    if run_dir == canonical_run:
        raise ValueError("Stage 7 output cannot be the canonical run namespace")
    try:
        run_dir.relative_to(canonical_run)
    except ValueError:
        return run_dir
    raise ValueError("Stage 7 output cannot be nested in the canonical run namespace")


@contextmanager
def _run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".stage7.lock").open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another Stage 7 process owns {run_dir}") from exc
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _git_is_ancestor(ancestor: str, repo: Path) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _current_branch(repo: Path) -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _gpu_identity() -> dict[str, object]:
    devices: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_vram_gib": properties.total_memory / 1024**3,
                "capability": [properties.major, properties.minor],
            }
        )
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "devices": devices,
        "utilization_source": (
            "nvidia-smi" if shutil.which("nvidia-smi") else "unavailable"
        ),
    }


def hardware_preflight() -> dict[str, object]:
    gpu = _gpu_identity()
    checks = {
        "cuda_available": bool(gpu["cuda_available"]),
        "gpu_visible": int(gpu["device_count"]) > 0,
        "cpu_workers": int(os.cpu_count() or 0) >= EXECUTION.workers,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cpu": {
            "logical_cpus": int(os.cpu_count() or 0),
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
        },
        "pytorch_version": torch.__version__,
        "gpu": gpu,
    }


def _checkpoint_metadata(path: Path) -> dict[str, object]:
    return _read_json(path.with_suffix(".metadata.json"))


def _checkpoint_identity(path: Path) -> dict[str, object]:
    metadata = _checkpoint_metadata(path)
    return {
        "label": metadata.get("checkpoint_label"),
        "path": str(path),
        "metadata_path": str(path.with_suffix(".metadata.json")),
        "model_hash": metadata.get("model_hash"),
        "artifact_sha256": file_sha256(path),
    }


def _load_model(path: Path, device: str) -> tuple[torch.nn.Module, dict[str, object]]:
    metadata = _checkpoint_metadata(path)
    model = torus9_model_from_metadata(metadata).to(device)
    torus9_load_checkpoint(
        path,
        model=model,
        expected={"model_hash": metadata["model_hash"]},
        device=device,
    )
    actual = model_hash(model)
    if actual != metadata.get("model_hash"):
        raise ValueError(f"Model content hash mismatch for {path}")
    model.eval()
    return model, metadata


def canonical_snapshot(canonical_run: Path) -> dict[str, object]:
    canonical_run = canonical_run.resolve()
    checkpoint_dir = canonical_run / "checkpoints"
    entries: dict[str, object] = {}
    for label in CANONICAL_CHECKPOINTS:
        checkpoint = checkpoint_dir / f"{label}.pt"
        metadata = checkpoint.with_suffix(".metadata.json")
        if not checkpoint.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"Canonical {label} artifact is missing")
        entries[label] = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "metadata_sha256": file_sha256(metadata),
            "model_hash": _checkpoint_metadata(checkpoint).get("model_hash"),
            "checkpoint_size": checkpoint.stat().st_size,
            "metadata_size": metadata.stat().st_size,
        }
    m17_sha = str(entries["M17"]["checkpoint_sha256"])  # type: ignore[index]
    if m17_sha.removeprefix("sha256:") != REFERENCE_M17_SHA:
        raise ValueError(f"Canonical M17 SHA mismatch: {m17_sha}")
    forbidden = sorted(
        str(path.relative_to(canonical_run))
        for path in canonical_run.rglob("M18*")
        if path.is_file()
    )
    if forbidden:
        raise ValueError(f"Canonical M18 artifacts exist: {forbidden}")
    return {
        "run_id": REFERENCE_RUN_ID,
        "path": str(canonical_run),
        "checkpoints": entries,
        "m17_sha256": m17_sha,
        "m18_artifacts": forbidden,
    }


def _profile_contract() -> tuple[dict[str, object], str, Torus9SelfPlaySearchContract]:
    profile = load_torus9_current_profile()
    profile_fp = current_torus9_profile_fingerprint(profile)
    contract = Torus9SelfPlaySearchContract()
    contract.validate()
    if contract.fingerprint != profile["self_play"]["fingerprint"]:  # type: ignore[index]
        raise ValueError("Current self-play contract fingerprint drift")
    if profile_fp != "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775":
        raise ValueError("Current Torus9 profile fingerprint drift")
    if TORUS9_KOMI != 0.5:
        raise ValueError("Current Torus9 komi must be exactly 0.5")
    return profile, profile_fp, contract


def _preflight(*, run_dir: Path, canonical_run: Path, code: CodeIdentity) -> dict[str, object]:
    EXECUTION.validate()
    profile, profile_fp, contract = _profile_contract()
    canonical = canonical_snapshot(canonical_run)
    hardware = hardware_preflight()
    checks = {
        "post_108_base": _git_is_ancestor(BASE_SHA, ROOT),
        "profile": profile_fp == profile["profile_fingerprint"],
        "komi": TORUS9_KOMI == 0.5,
        "contract": contract.fingerprint == profile["self_play"]["fingerprint"],  # type: ignore[index]
        "canonical_artifacts": bool(canonical["m17_sha256"]),
        "hardware": hardware["status"] == "PASS",
    }
    # Existing unrelated untracked artifacts are kept out of the run namespace
    # by policy.  The exact status remains recorded in provenance; executable
    # source identity is frozen by the branch commit before the run starts.
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "code_identity": _jsonable(code),
        "source_worktree_clean": code.working_tree_clean,
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "rules_fingerprint": TORUS9_RULES_FINGERPRINT,
        "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        "selfplay_contract_fingerprint": contract.fingerprint,
        "execution": EXECUTION.as_dict(),
        "execution_reference": LEGION_TORUS9_SELFPLAY_PERFORMANCE_REFERENCE.as_dict(),
        "profile": {
            "path": str(ROOT / "configs/gocube/torus9_golden_current_v3.json"),
            "horizon": VALIDATION_HORIZON,
        },
        "canonical": canonical,
        "hardware": hardware,
    }


def _m0_initialization_parity(
    *, model: Torus9CurrentGraphNet, new_m0: Path, canonical_run: Path
) -> dict[str, object]:
    old_m0 = canonical_run / "checkpoints" / "M0.pt"
    old_model, old_metadata = _load_model(old_m0, "cpu")
    new_metadata = _checkpoint_metadata(new_m0)
    new_model, _ = _load_model(new_m0, "cpu")
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    if tuple(old_state) != tuple(new_state):
        raise ValueError("NEW M0 tensor names differ from OLD M0")
    shapes = {name: list(value.shape) for name, value in new_state.items()}
    equality = all(torch.equal(old_state[name], new_state[name]) for name in old_state)
    scientific_keys = (
        "profile_id",
        "profile_fingerprint",
        "rules_fingerprint",
        "target_fingerprint",
        "selfplay_contract_fingerprint",
        "architecture_id",
        "topology_fingerprint",
        "observation_fingerprint",
        "komi",
    )
    scientific_equal = all(old_metadata.get(key) == new_metadata.get(key) for key in scientific_keys)
    return {
        "old_checkpoint": str(old_m0),
        "new_checkpoint": str(new_m0),
        "old_model_hash": old_metadata.get("model_hash"),
        "new_model_hash": new_metadata.get("model_hash"),
        "model_hash_equal": model_hash(old_model) == model_hash(new_model),
        "tensor_equal": equality,
        "tensor_names": list(new_state),
        "tensor_shapes": shapes,
        "scientific_fingerprint_equal": scientific_equal,
        "scientific_fingerprint": {
            key: new_metadata.get(key) for key in scientific_keys
        },
        "profile_fingerprint": new_metadata.get("profile_fingerprint"),
        "komi": new_metadata.get("komi"),
        "architecture": new_metadata.get("architecture_id"),
    }


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "run-manifest.json"


def _load_manifest(run_dir: Path) -> dict[str, object]:
    return _read_json(_manifest_path(run_dir))


def _generation_paths(run_dir: Path, generation: int) -> dict[str, Path]:
    return {
        "fresh": run_dir / "replay" / f"iter-{generation:02d}-fresh.jsonl",
        "rolling": run_dir / "replay" / f"rolling-after-{generation:02d}.jsonl",
        "checkpoint": run_dir / "checkpoints" / f"M{generation}.pt",
        "metadata": run_dir / "checkpoints" / f"M{generation}.metadata.json",
        "training": run_dir / "training" / f"iter-{generation:02d}.json",
        "summary": run_dir / f"iter-{generation:02d}-summary.json",
        "marker": run_dir / f"generation-{generation:02d}.complete.json",
        "games": run_dir / "selfplay" / f"iter-{generation:02d}-games.jsonl",
    }


def _replay_identity(rows: Sequence[Mapping[str, object]], evictions: int = 0) -> dict[str, object]:
    row_ids = tuple(str(row.get("replay_row_id", "")) for row in rows)
    generations = tuple(sorted({int(row["source_generation"]) for row in rows}))
    if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("Replay rows have missing or duplicate IDs")
    return {
        "generations": list(generations),
        "row_count": len(rows),
        "row_ids_fingerprint": value_fingerprint(row_ids),
        "rows_fingerprint": sequence_fingerprint(rows),
        "total_evictions": int(evictions),
        "cap": TORUS9_MAX_REPLAY_POSITIONS,
    }


def _validate_manifest(
    manifest: Mapping[str, object], *, run_dir: Path, canonical_run: Path
) -> tuple[dict[str, object], str, Torus9SelfPlaySearchContract]:
    if manifest.get("schema") != "torus9-stage7-run-v2":
        raise ValueError("Unsupported Stage 7 manifest schema")
    if manifest.get("run_id") != run_dir.name:
        raise ValueError("Stage 7 run ID does not match its directory")
    if manifest.get("base_sha") != BASE_SHA:
        raise ValueError("Stage 7 base SHA drift")
    if manifest.get("profile_id") != TORUS9_CURRENT_PROFILE_ID:
        raise ValueError("Stage 7 profile ID drift")
    profile, profile_fp, contract = _profile_contract()
    if manifest.get("profile_fingerprint") != profile_fp:
        raise ValueError("Stage 7 profile fingerprint drift")
    if manifest.get("rules_fingerprint") != TORUS9_RULES_FINGERPRINT:
        raise ValueError("Stage 7 rules fingerprint drift")
    if manifest.get("target_fingerprint") != TORUS9_CURRENT_TARGET_FINGERPRINT:
        raise ValueError("Stage 7 target fingerprint drift")
    if manifest.get("selfplay_contract_fingerprint") != contract.fingerprint:
        raise ValueError("Stage 7 self-play contract drift")
    if manifest.get("execution") != EXECUTION.as_dict():
        raise ValueError("Stage 7 execution preset drift")
    for key, expected in (
        ("model_init_seed", TORUS9_CURRENT_MODEL_INIT_SEED),
        ("selfplay_master_seed", TORUS9_CURRENT_SELFPLAY_MASTER_SEED),
        ("training_master_seed", TORUS9_CURRENT_TRAINING_MASTER_SEED),
        ("arena_master_seed", TORUS9_CURRENT_ARENA_MASTER_SEED),
        ("validation_horizon", VALIDATION_HORIZON),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"Stage 7 {key} drift")
    if manifest.get("golden_standard_untouched") is not True:
        raise ValueError("Stage 7 historical scope marker is missing")
    if canonical_snapshot(canonical_run) != manifest.get("canonical_before"):
        raise ValueError("Historical canonical artifacts changed")
    source = manifest.get("source_code_identity")
    if not isinstance(source, Mapping):
        raise ValueError("Stage 7 source code identity is missing")
    if source.get("git_commit_sha") != manifest.get("source_commit"):
        raise ValueError("Stage 7 source commit identity drift")
    m0 = run_dir / "checkpoints" / "M0.pt"
    m0_meta = _checkpoint_metadata(m0)
    m0_identity = manifest.get("m0")
    if not isinstance(m0_identity, Mapping):
        raise ValueError("Stage 7 M0 identity is missing")
    if (
        m0_meta.get("checkpoint_label") != "M0"
        or m0_meta.get("run_id") != run_dir.name
        or m0_meta.get("model_hash") != m0_identity.get("model_hash")
        or file_sha256(m0) != m0_identity.get("artifact_sha256")
        or m0_meta.get("optimizer_updates") != 0
        or m0_meta.get("train_samples_consumed") != 0
    ):
        raise ValueError("Stage 7 M0 identity or zero clock drift")
    return profile, profile_fp, contract


def _validate_frozen_source(manifest: Mapping[str, object], code: CodeIdentity) -> None:
    """Reject continuation on an executable source state different from M0."""
    if (
        code.git_commit_sha != manifest.get("source_commit")
        or code.git_tree_sha != manifest.get("source_tree")
    ):
        raise RuntimeError(
            "Stage 7 executable source changed after the acceptance run started; "
            "discard this evidence and rerun from M0"
        )


def _validate_generation_marker(
    run_dir: Path, generation: int, *, run_id: str
) -> dict[str, object]:
    paths = _generation_paths(run_dir, generation)
    marker = _read_json(paths["marker"])
    for key, expected in (
        ("run_id", run_id),
        ("generation", generation),
        ("label", f"M{generation}"),
    ):
        if marker.get(key) != expected:
            raise ValueError(f"Stage 7 M{generation} marker {key} drift")
    for key, path_key in (
        ("fresh_replay_sha256", "fresh"),
        ("rolling_replay_sha256", "rolling"),
        ("checkpoint_sha256", "checkpoint"),
        ("checkpoint_metadata_sha256", "metadata"),
        ("training_metrics_sha256", "training"),
        ("summary_sha256", "summary"),
        ("selfplay_games_sha256", "games"),
    ):
        path = paths[path_key]
        if not path.is_file() or marker.get(key) != file_sha256(path):
            raise ValueError(f"Stage 7 M{generation} artifact hash drift: {path}")
    metadata = _checkpoint_metadata(paths["checkpoint"])
    training = _read_json(paths["training"])
    summary = _read_json(paths["summary"])
    rolling = _read_jsonl(paths["rolling"])
    identity = summary.get("replay_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"Stage 7 M{generation} replay identity is missing")
    if (
        metadata.get("checkpoint_label") != f"M{generation}"
        or metadata.get("model_hash") != marker.get("model_hash")
        or metadata.get("replay_fingerprint") != sequence_fingerprint(rolling)
        or marker.get("replay_fingerprint") != sequence_fingerprint(rolling)
        or identity.get("rows_fingerprint") != sequence_fingerprint(rolling)
        or identity.get("row_count") != len(rolling)
        or training.get("optimizer_steps") != TORUS9_OPTIMIZER_STEPS_PER_ITERATION
        or training.get("samples_consumed") != TORUS9_TRAINING_SAMPLES_PER_ITERATION
        or training.get("training_seed")
        != derive_seed(TORUS9_CURRENT_TRAINING_MASTER_SEED, run_id, "training", generation)
    ):
        raise ValueError(f"Stage 7 M{generation} checkpoint/training/replay drift")
    selfplay = summary.get("self_play")
    if (
        not isinstance(selfplay, Mapping)
        or selfplay.get("games") != SELFPLAY_GAMES
        or selfplay.get("valid_games") != SELFPLAY_GAMES
        or selfplay.get("technical_games") != 0
    ):
        raise ValueError(f"Stage 7 M{generation} self-play validity drift")
    for key in ("loss", "policy_loss", "value_loss", "ownership_loss", "score_loss"):
        value = training.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
            raise ValueError(f"Stage 7 M{generation} contains non-finite {key}")
    return summary


def _committed_generation(run_dir: Path) -> int:
    manifest = _load_manifest(run_dir)
    last = 0
    for generation in range(1, VALIDATION_HORIZON + 1):
        marker = _generation_paths(run_dir, generation)["marker"]
        if not marker.is_file():
            break
        _validate_generation_marker(run_dir, generation, run_id=str(manifest["run_id"]))
        last = generation
    for generation in range(last + 1, VALIDATION_HORIZON + 1):
        if _generation_paths(run_dir, generation)["marker"].is_file():
            raise ValueError("Stage 7 generation marker gap")
    for generation in range(VALIDATION_HORIZON + 1, 8):
        if any(path.exists() for path in _generation_paths(run_dir, generation).values()):
            raise ValueError(f"Stage 7 created forbidden M{generation}")
    if int(manifest.get("last_committed_generation", 0)) != last:
        raise ValueError("Stage 7 manifest commit cursor drift")
    return last


def _validate_records(
    records: Sequence[Torus9SelfPlayGameRecord],
    *,
    run_id: str,
    generation: int,
    source_checkpoint: Path,
    profile_fp: str,
    code: CodeIdentity,
    telemetry: Mapping[str, object],
) -> None:
    expected_ids = {
        f"{run_id}-M{generation:02d}-game-{index:04d}"
        for index in range(SELFPLAY_GAMES)
    }
    if {record.game_id for record in records} != expected_ids or len(records) != SELFPLAY_GAMES:
        raise ValueError(f"M{generation} did not return exactly 64 unique games")
    expected_model = _checkpoint_metadata(source_checkpoint).get("model_hash")
    expected_artifact = file_sha256(source_checkpoint)
    for record in records:
        record.validate()
        if record.technical_termination is not None:
            raise ValueError(f"M{generation} contains technical game {record.game_id}")
        if record.profile_id != TORUS9_CURRENT_PROFILE_ID or record.profile_fingerprint != profile_fp:
            raise ValueError(f"M{generation} scientific profile drift")
        if record.selfplay_contract_id != "torus9-golden-current-selfplay-search-v1":
            raise ValueError(f"M{generation} search contract drift")
        if record.selfplay_contract_fingerprint != current_torus9_selfplay_contract_fingerprint():
            raise ValueError(f"M{generation} search fingerprint drift")
        if record.model_hash != expected_model or record.checkpoint_artifact_hash != expected_artifact:
            raise ValueError(f"M{generation} source checkpoint lineage drift")
        if record.git_commit != code.git_commit_sha or record.git_tree != code.git_tree_sha:
            raise ValueError(f"M{generation} source code provenance drift")
        if record.master_seed != TORUS9_CURRENT_SELFPLAY_MASTER_SEED:
            raise ValueError(f"M{generation} self-play seed drift")
        if record.game_seed != derive_seed(TORUS9_CURRENT_SELFPLAY_MASTER_SEED, run_id, record.game_id, "game"):
            raise ValueError(f"M{generation} game seed drift")
        if record.start_state.get("komi") != TORUS9_KOMI or record.nn_evaluations <= 0:
            raise ValueError(f"M{generation} record komi/evaluation drift")
    if telemetry.get("execution_reference_status") != "validated_recommended":
        raise ValueError(f"M{generation} did not use the validated Legion preset")
    reference = telemetry.get("execution_reference")
    if not isinstance(reference, Mapping) or reference.get("effective_context_ceiling") != 64:
        raise ValueError(f"M{generation} did not use the full 64-context workload")
    if telemetry.get("shared_memory_transport") is not True:
        raise ValueError(f"M{generation} did not use shared-memory self-play")
    if telemetry.get("central_inference_owner_pid") != os.getpid():
        raise ValueError(f"M{generation} central inference owner is not parent")
    if len(set(telemetry.get("worker_pids", []))) != EXECUTION.workers:
        raise ValueError(f"M{generation} worker PID count drift")


def _training_summary_builder(
    *, generation: int, records: Sequence[Torus9SelfPlayGameRecord], wall: float,
    telemetry: Mapping[str, object], state: Any
):
    def build(base: Mapping[str, object]) -> Mapping[str, object]:
        rolling_rows = tuple(state.rolling_replay.rows)
        replay = base.get("replay")
        training = base.get("training")
        if not isinstance(replay, Mapping) or not isinstance(training, Mapping):
            raise ValueError("TrainingEngine returned malformed metrics")
        moves = sum(len(record.final_action_trace) for record in records)
        performance = telemetry.get("performance_reference")
        return {
            **dict(base),
            "self_play": {
                "games": len(records),
                "valid_games": len(records),
                "technical_games": 0,
                "moves": moves,
                "wall_time_sec": wall,
                "moves_per_sec": moves / wall if wall else 0.0,
                "inference_rows_per_sec": telemetry.get("inference_rows_per_sec", 0.0),
                "worker_pids": list(telemetry.get("worker_pids", [])),
                "effective_context_ceiling": telemetry.get("execution_reference", {}).get("effective_context_ceiling") if isinstance(telemetry.get("execution_reference"), Mapping) else None,
                "execution_reference_status": telemetry.get("execution_reference_status"),
                "performance": performance,
                "batch": {
                    "mean": telemetry.get("mean_inference_batch_rows", 0.0),
                    "p50": telemetry.get("p50_inference_batch_rows", 0),
                    "p95": telemetry.get("p95_inference_batch_rows", 0),
                    "max": telemetry.get("max_inference_batch_rows", 0),
                },
                "execution": {
                    **EXECUTION.as_dict(),
                    "central_inference_owner_pid": telemetry.get("central_inference_owner_pid"),
                    "shared_memory_transport": telemetry.get("shared_memory_transport"),
                },
            },
            "replay_identity": _replay_identity(
                rolling_rows,
                int(replay.get("total_evictions", 0)),
            ),
        }
    return build


def _run_generation(
    *, run_dir: Path, manifest: Mapping[str, object], state: Any,
    adapter: Torus9TrainingAdapter, contract: Torus9SelfPlaySearchContract,
    generation: int, code: CodeIdentity
) -> dict[str, object]:
    if not 1 <= generation <= VALIDATION_HORIZON:
        raise ValueError(f"Stage 7 generation outside M1..M{VALIDATION_HORIZON}")
    source = run_dir / "checkpoints" / ("M0.pt" if generation == 1 else f"M{generation - 1}.pt")
    if not source.is_file():
        raise FileNotFoundError(f"Missing Stage 7 source checkpoint: {source}")
    model = state.model
    telemetry: dict[str, object] = {}
    started = time.perf_counter()
    records = run_torus9_selfplay_games(
        model,
        run_id=str(manifest["run_id"]),
        label=f"M{generation - 1}",
        artifact=file_sha256(source),
        master_seed=TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
        profile_fp=str(manifest["profile_fingerprint"]),
        profile_id=TORUS9_CURRENT_PROFILE_ID,
        game_ids=[
            f"{manifest['run_id']}-M{generation:02d}-game-{index:04d}"
            for index in range(SELFPLAY_GAMES)
        ],
        workers=EXECUTION.workers,
        code_identity=code,
        device=EXECUTION.device,
        contract=contract,
        coalescing=True,
        inference_batch_cap=EXECUTION.inference_batch_cap,
        inference_batch_wait_ms=EXECUTION.inference_batch_wait_ms,
        active_games_per_worker=EXECUTION.active_games_per_worker,
        total_active_contexts=EXECUTION.active_contexts,
        inference_telemetry=telemetry,
        execution_activity=telemetry,
        execution_reference_interactive=False,
    )
    wall = time.perf_counter() - started
    _validate_records(
        records, run_id=str(manifest["run_id"]), generation=generation,
        source_checkpoint=source, profile_fp=str(manifest["profile_fingerprint"]),
        code=code, telemetry=telemetry,
    )
    paths = _generation_paths(run_dir, generation)
    paths["games"].parent.mkdir(parents=True, exist_ok=True)
    with paths["games"].open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    result = run_torus9_training_iteration(
        state=state,
        generation=generation,
        output_dir=run_dir,
        run_id=str(manifest["run_id"]),
        records=records,
        training_seed=derive_seed(
            TORUS9_CURRENT_TRAINING_MASTER_SEED,
            str(manifest["run_id"]), "training", generation,
        ),
        completed_games=generation * SELFPLAY_GAMES,
        code_identity=code,
        device=EXECUTION.device,
        adapter=adapter,
        summary_builder=_training_summary_builder(
            generation=generation, records=records, wall=wall,
            telemetry=telemetry, state=state,
        ),
    )
    summary = dict(result.summary)
    summary["source_checkpoint"] = _checkpoint_identity(source)
    summary["scientific_fingerprint"] = {
        "profile": manifest["profile_fingerprint"],
        "rules": manifest["rules_fingerprint"],
        "target": manifest["target_fingerprint"],
        "selfplay": manifest["selfplay_contract_fingerprint"],
        "komi": TORUS9_KOMI,
    }
    # The generic TrainingEngine has already atomically committed the marker;
    # add the game-file hash and the performance evidence to that marker only.
    marker_path = paths["marker"]
    marker = _read_json(marker_path)
    marker.update({
        "selfplay_games_sha256": file_sha256(paths["games"]),
        "selfplay_games": SELFPLAY_GAMES,
        "technical_games": 0,
        "performance_status": (
            telemetry.get("performance_reference", {}).get("status")
            if isinstance(telemetry.get("performance_reference"), Mapping)
            else None
        ),
    })
    _atomic_write(marker_path, marker)
    _atomic_write(paths["summary"], summary)
    marker["summary_sha256"] = file_sha256(paths["summary"])
    _atomic_write(marker_path, marker)
    _validate_generation_marker(run_dir, generation, run_id=str(manifest["run_id"]))
    return summary


def _new_manifest(
    *, run_id: str, source: CodeIdentity, preflight: Mapping[str, object],
    canonical: Mapping[str, object], m0_parity: Mapping[str, object], m0: Path
) -> dict[str, object]:
    profile, profile_fp, contract = _profile_contract()
    return {
        "schema": "torus9-stage7-run-v2",
        "run_id": run_id,
        "branch": _current_branch(ROOT),
        "base_sha": BASE_SHA,
        "source_commit": source.git_commit_sha,
        "source_tree": source.git_tree_sha,
        "source_code_identity": _jsonable(source),
        "source_worktree_clean": source.working_tree_clean,
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": profile_fp,
        "rules_fingerprint": TORUS9_RULES_FINGERPRINT,
        "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        "selfplay_contract_fingerprint": contract.fingerprint,
        "execution": EXECUTION.as_dict(),
        "model_init_seed": TORUS9_CURRENT_MODEL_INIT_SEED,
        "selfplay_master_seed": TORUS9_CURRENT_SELFPLAY_MASTER_SEED,
        "training_master_seed": TORUS9_CURRENT_TRAINING_MASTER_SEED,
        "arena_master_seed": TORUS9_CURRENT_ARENA_MASTER_SEED,
        "evaluation_master_seed": 202609131005,
        "validation_horizon": VALIDATION_HORIZON,
        "golden_standard_untouched": True,
        "preflight": dict(preflight),
        "canonical_before": dict(canonical),
        "m0_parity": dict(m0_parity),
        "m0": _checkpoint_identity(m0),
        "last_committed_generation": 0,
        "training_processes": [{"pid": os.getpid(), "phase": "M0-M2"}],
        "resume": None,
        "arena": None,
        "protocol": None,
        "profile_snapshot": profile,
    }


def _initialize_m0(run_dir: Path, run_id: str, code: CodeIdentity) -> tuple[Any, Torus9TrainingAdapter, dict[str, object]]:
    profile, profile_fp, _ = _profile_contract()
    torch.manual_seed(TORUS9_CURRENT_MODEL_INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORUS9_CURRENT_MODEL_INIT_SEED)
    model = Torus9CurrentGraphNet().to(EXECUTION.device)
    adapter = Torus9TrainingAdapter(profile=profile, code_identity=code, base_commit=BASE_SHA)
    state = adapter.create_state(model, run_id=run_id)
    m0 = run_dir / "checkpoints" / "M0.pt"
    adapter.save_initial_checkpoint(
        m0, state, run_id=run_id, label="M0", completed_games=0,
        device=EXECUTION.device, code_identity=code,
    )
    return state, adapter, {"profile_fingerprint": profile_fp}


def _validate_finished_training(run_dir: Path, *, expected_last: int) -> None:
    last = _committed_generation(run_dir)
    if last != expected_last:
        raise ValueError(f"Stage 7 expected M{expected_last}, found M{last}")
    manifest = _load_manifest(run_dir)
    _validate_checkpoint_chain(run_dir, expected_last, str(manifest["run_id"]))
    for generation in range(1, expected_last + 1):
        _validate_generation_marker(run_dir, generation, run_id=str(manifest["run_id"]))


def _validate_checkpoint_chain(run_dir: Path, last: int, run_id: str) -> None:
    previous: dict[str, object] | None = None
    for generation in range(1, last + 1):
        metadata = _checkpoint_metadata(run_dir / "checkpoints" / f"M{generation}.pt")
        if metadata.get("run_id") != run_id or metadata.get("checkpoint_label") != f"M{generation}":
            raise ValueError(f"Stage 7 checkpoint identity drift at M{generation}")
        if metadata.get("profile_fingerprint") != "sha256:36911d01c04e8c77a99146c86b053a68126725998c207332d8e18df269bb1775":
            raise ValueError(f"Stage 7 checkpoint scientific fingerprint drift at M{generation}")
        if metadata.get("komi") != TORUS9_KOMI:
            raise ValueError(f"Stage 7 checkpoint komi drift at M{generation}")
        parent = metadata.get("parent_checkpoint_identity")
        if generation == 1:
            if parent is not None:
                raise ValueError("Stage 7 M1 must have no parent checkpoint")
        elif (
            not isinstance(parent, Mapping)
            or parent.get("label") != f"M{generation - 1}"
            or previous is None
            or parent.get("model_hash") != previous.get("model_hash")
            or parent.get("artifact_sha256")
            != file_sha256(run_dir / "checkpoints" / f"M{generation - 1}.pt")
        ):
            raise ValueError(f"Stage 7 checkpoint lineage drift at M{generation}")
        if metadata.get("optimizer_updates") != generation * TORUS9_OPTIMIZER_STEPS_PER_ITERATION:
            raise ValueError(f"Stage 7 optimizer clock drift at M{generation}")
        if metadata.get("train_samples_consumed") != generation * TORUS9_TRAINING_SAMPLES_PER_ITERATION:
            raise ValueError(f"Stage 7 sample clock drift at M{generation}")
        previous = metadata


def _replay_audit(run_dir: Path, last: int) -> dict[str, object]:
    per_generation: dict[str, object] = {}
    for generation in range(1, last + 1):
        paths = _generation_paths(run_dir, generation)
        rows = _read_jsonl(paths["rolling"])
        identity = _replay_identity(rows)
        expected = list(range(max(1, generation - TORUS9_ROLLING_GENERATIONS + 1), generation + 1))
        if identity["generations"] != expected:
            raise ValueError(f"Stage 7 replay window drift at M{generation}")
        if int(identity["row_count"]) > TORUS9_MAX_REPLAY_POSITIONS:
            raise ValueError(f"Stage 7 replay cap drift at M{generation}")
        all_fresh_rows: list[dict[str, object]] = []
        expected_rows: list[dict[str, object]] = []
        for source_generation in expected:
            fresh = _read_jsonl(
                _generation_paths(run_dir, source_generation)["fresh"]
            )
            if any(
                int(row.get("source_generation", 0)) != source_generation
                for row in fresh
            ):
                raise ValueError(
                    f"Stage 7 fresh replay generation stamp drift at M{source_generation}"
                )
            expected_rows.extend(fresh)
        for source_generation in range(1, generation + 1):
            all_fresh_rows.extend(
                _read_jsonl(_generation_paths(run_dir, source_generation)["fresh"])
            )
        # The rolling artifact is the actual training-visible content.  For
        # the present 20,000-row cap the equality must be exact; if a future
        # profile reaches the cap, retain only the same tail as the adapter.
        expected_rows = expected_rows[-TORUS9_MAX_REPLAY_POSITIONS:]
        if rows != expected_rows:
            raise ValueError(f"Stage 7 replay content drift at M{generation}")
        metadata = _checkpoint_metadata(paths["checkpoint"])
        if metadata.get("replay_generations") != expected:
            raise ValueError(f"Stage 7 checkpoint replay generations drift at M{generation}")
        summary = _read_json(paths["summary"])
        replay_metrics = summary.get("replay")
        if not isinstance(replay_metrics, Mapping):
            raise ValueError(f"Stage 7 replay metrics missing at M{generation}")
        expected_total_evictions = max(0, len(all_fresh_rows) - len(rows))
        if int(replay_metrics.get("total_evictions", -1)) != expected_total_evictions:
            raise ValueError(f"Stage 7 replay eviction count drift at M{generation}")
        if int(replay_metrics.get("rolling_buffer_positions", -1)) != len(rows):
            raise ValueError(f"Stage 7 replay size drift at M{generation}")
        per_generation[f"M{generation}"] = identity
    final = per_generation[f"M{last}"]
    return {
        "status": "PASS",
        "window": "rolling last 3 generations",
        "cap": TORUS9_MAX_REPLAY_POSITIONS,
        "per_generation": per_generation,
        "final_generations": final["generations"],  # type: ignore[index]
        "stale_generations": [],
        "duplicate_ids": False,
        "resume_duplication": False,
    }


def run_new(*, run_dir: Path, canonical_run: Path, run_id: str, code: CodeIdentity) -> dict[str, object]:
    run_dir = _assert_isolated(run_dir, canonical_run)
    with _run_lock(run_dir):
        existing = [path for path in run_dir.iterdir() if path.name != ".stage7.lock"]
        if existing:
            raise FileExistsError(f"Stage 7 fresh run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "replay", "selfplay", "training"):
            (run_dir / name).mkdir()
        preflight = _preflight(run_dir=run_dir, canonical_run=canonical_run, code=code)
        _atomic_write(run_dir / "preflight.json", preflight)
        if preflight["status"] != "PASS":
            raise RuntimeError(f"Stage 7 preflight failed: {preflight}")
        state, adapter, _ = _initialize_m0(run_dir, run_id, code)
        m0_parity = _m0_initialization_parity(
            model=state.model, new_m0=run_dir / "checkpoints" / "M0.pt", canonical_run=canonical_run,
        )
        if not all((
            m0_parity["model_hash_equal"],
            m0_parity["tensor_equal"],
            m0_parity["scientific_fingerprint_equal"],
            m0_parity["architecture"] == "GoldenGraphNetV2-Torus9",
            m0_parity["komi"] == 0.5,
        )):
            raise RuntimeError(f"Stage 7 M0 parity failed: {m0_parity}")
        profile, profile_fp, contract = _profile_contract()
        manifest = _new_manifest(
            run_id=run_id, source=code, preflight=preflight,
            canonical=preflight["canonical"], m0_parity=m0_parity,
            m0=run_dir / "checkpoints" / "M0.pt",
        )
        _atomic_write(_manifest_path(run_dir), manifest)
        for generation in range(1, SHORT_RUN_STOP_AFTER + 1):
            _run_generation(
                run_dir=run_dir, manifest=manifest, state=state, adapter=adapter,
                contract=contract, generation=generation, code=code,
            )
            manifest = _load_manifest(run_dir)
            manifest["last_committed_generation"] = generation
            _atomic_write(_manifest_path(run_dir), manifest)
        _validate_finished_training(run_dir, expected_last=SHORT_RUN_STOP_AFTER)
        manifest = _load_manifest(run_dir)
        manifest["m2_boundary"] = {
            "checkpoint": _checkpoint_identity(run_dir / "checkpoints" / "M2.pt"),
            "optimizer_updates": 2 * TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
            "samples_consumed": 2 * TORUS9_TRAINING_SAMPLES_PER_ITERATION,
            "replay": _replay_identity(_read_jsonl(run_dir / "replay" / "rolling-after-02.jsonl")),
        }
        _atomic_write(_manifest_path(run_dir), manifest)
        return manifest


def _resume_evidence_before(run_dir: Path) -> dict[str, object]:
    return {
        "m2_checkpoint_sha256_before": file_sha256(run_dir / "checkpoints" / "M2.pt"),
        "m2_metadata_sha256_before": file_sha256(run_dir / "checkpoints" / "M2.metadata.json"),
        "m1_games_sha256_before": file_sha256(run_dir / "selfplay" / "iter-01-games.jsonl"),
        "m2_games_sha256_before": file_sha256(run_dir / "selfplay" / "iter-02-games.jsonl"),
        "m1_training_sha256_before": file_sha256(run_dir / "training" / "iter-01.json"),
        "m2_training_sha256_before": file_sha256(run_dir / "training" / "iter-02.json"),
        "replay_before": _replay_identity(_read_jsonl(run_dir / "replay" / "rolling-after-02.jsonl")),
        "process_before": os.getpid(),
    }


def run_resume(*, run_dir: Path, canonical_run: Path, code: CodeIdentity) -> dict[str, object]:
    run_dir = _assert_isolated(run_dir, canonical_run)
    with _run_lock(run_dir):
        manifest = _load_manifest(run_dir)
        profile, profile_fp, contract = _validate_manifest(manifest, run_dir=run_dir, canonical_run=canonical_run)
        _validate_frozen_source(manifest, code)
        if _committed_generation(run_dir) != SHORT_RUN_STOP_AFTER:
            raise ValueError("Stage 7 resume must start exactly after committed M2")
        before = _resume_evidence_before(run_dir)
        adapter = Torus9TrainingAdapter(profile=profile, code_identity=code, base_commit=BASE_SHA)
        state = adapter.load_state(
            run_dir / "checkpoints" / "M2.pt",
            replay_path=run_dir / "replay" / "rolling-after-02.jsonl",
            device=EXECUTION.device,
        )
        if state.current_generation != 2 or state.optimizer_updates != 160 or state.samples_consumed != 10240:
            raise ValueError("Stage 7 M2 optimizer/generation state was not restored")
        if _replay_identity(state.rolling_replay.rows) != before["replay_before"]:
            raise ValueError("Stage 7 M2 replay state was not restored")
        for generation in range(3, RESUME_STOP_AFTER + 1):
            _run_generation(
                run_dir=run_dir, manifest=manifest, state=state, adapter=adapter,
                contract=contract, generation=generation, code=code,
            )
            manifest = _load_manifest(run_dir)
            manifest["last_committed_generation"] = generation
            _atomic_write(_manifest_path(run_dir), manifest)
        after = {
            "m2_checkpoint_sha256_after": file_sha256(run_dir / "checkpoints" / "M2.pt"),
            "m2_metadata_sha256_after": file_sha256(run_dir / "checkpoints" / "M2.metadata.json"),
            "m1_games_sha256_after": file_sha256(run_dir / "selfplay" / "iter-01-games.jsonl"),
            "m2_games_sha256_after": file_sha256(run_dir / "selfplay" / "iter-02-games.jsonl"),
            "m1_training_sha256_after": file_sha256(run_dir / "training" / "iter-01.json"),
            "m2_training_sha256_after": file_sha256(run_dir / "training" / "iter-02.json"),
            "replay_after": _replay_identity(state.rolling_replay.rows),
            "optimizer_restored": {
                "generation": state.current_generation,
                "optimizer_updates": state.optimizer_updates,
                "adam_step": state.optimizer_updates,
                "samples_consumed": state.samples_consumed,
            },
            "process_after": os.getpid(),
        }
        evidence = {
            **before,
            **after,
            "m1_m2_games_rerun": False,
            "m1_m2_training_rerun": False,
            "next_generation": "M3",
        }
        evidence["resume_result"] = "PASS" if (
            evidence["process_before"] != evidence["process_after"]
            and evidence["m2_checkpoint_sha256_before"] == evidence["m2_checkpoint_sha256_after"]
            and evidence["m2_metadata_sha256_before"] == evidence["m2_metadata_sha256_after"]
            and evidence["m1_games_sha256_before"] == evidence["m1_games_sha256_after"]
            and evidence["m2_games_sha256_before"] == evidence["m2_games_sha256_after"]
            and evidence["m1_training_sha256_before"] == evidence["m1_training_sha256_after"]
            and evidence["m2_training_sha256_before"] == evidence["m2_training_sha256_after"]
            and evidence["replay_before"] == evidence["replay_after"]
            and evidence["next_generation"] == "M3"
        ) else "FAIL"
        if evidence["resume_result"] != "PASS":
            raise RuntimeError(f"Stage 7 resume failed: {evidence}")
        _validate_finished_training(run_dir, expected_last=VALIDATION_HORIZON)
        manifest = _load_manifest(run_dir)
        manifest["resume"] = evidence
        manifest["training_processes"] = list(manifest.get("training_processes", [])) + [{"pid": os.getpid(), "phase": "M3-M6"}]
        _atomic_write(_manifest_path(run_dir), manifest)
        return manifest


def _arena_snapshot(summary: Mapping[str, object], games_path: Path) -> dict[str, object]:
    wld = summary.get("W/L/D")
    if not isinstance(wld, list) or len(wld) != 3:
        wld = [summary.get("wins", 0), summary.get("losses", 0), summary.get("draws", 0)]
    telemetry = summary.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    rows = _read_jsonl(games_path)
    lengths = [len(row.get("action_trace", [])) for row in rows if row.get("technical_termination") is None]
    return {
        "games": int(summary.get("games", len(rows))),
        "W/L/D": [int(wld[0]), int(wld[1]), int(wld[2])],
        "score": (int(wld[0]) + 0.5 * int(wld[2])) / len(rows) if rows else None,
        "technical_games": int(summary.get("technical_games", 0)),
        "moves": telemetry.get("moves"),
        "moves_per_sec": telemetry.get("moves_per_sec"),
        "mean_inference_batch_rows": telemetry.get("mean_inference_batch_rows"),
        "p50_inference_batch_rows": telemetry.get("p50_inference_batch_rows"),
        "p95_inference_batch_rows": telemetry.get("p95_inference_batch_rows"),
        "max_inference_batch_rows": telemetry.get("max_inference_batch_rows"),
        "effective_cpu_cores": telemetry.get("effective_cpu_cores"),
        "gpu_utilization_avg_pct": telemetry.get("gpu_utilization_avg_pct"),
        "gpu_utilization_peak_pct": telemetry.get("gpu_utilization_peak_pct"),
        "average_game_length": sum(lengths) / len(lengths) if lengths else None,
        "median_game_length": sorted(lengths)[len(lengths) // 2] if lengths else None,
        "performance_status": telemetry.get("performance_status"),
        "execution": summary.get("execution"),
    }


def _combine_arena_snapshots(snapshots: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not snapshots:
        raise ValueError("Cannot combine an empty Arena result")
    wld = [0, 0, 0]
    games = 0
    technical = 0
    moves = 0
    for snapshot in snapshots:
        current_wld = snapshot.get("W/L/D")
        if not isinstance(current_wld, Sequence) or len(current_wld) != 3:
            raise ValueError("Malformed Arena W/L/D result")
        wld = [left + int(right) for left, right in zip(wld, current_wld)]
        games += int(snapshot.get("games", 0))
        technical += int(snapshot.get("technical_games", 0))
        moves += int(snapshot.get("moves") or 0)
    return {
        "games": games,
        "W/L/D": wld,
        "score": (wld[0] + 0.5 * wld[2]) / games if games else None,
        "technical_games": technical,
        "moves": moves,
        "components": [dict(snapshot) for snapshot in snapshots],
        "performance_status": (
            "PERFORMANCE_DEGRADED"
            if any(snapshot.get("performance_status") == "PERFORMANCE_DEGRADED" for snapshot in snapshots)
            else "PASS"
        ),
    }


def _validate_arena_summary(
    summary: Mapping[str, object], *, games: int, candidate: Path, reference: Path
) -> None:
    if int(summary.get("games", -1)) != games:
        raise ValueError("Arena returned the wrong game count")
    wld = summary.get("W/L/D")
    if not isinstance(wld, list) or len(wld) != 3 or sum(int(value) for value in wld) != games:
        raise ValueError("Arena W/L/D does not cover the requested games")
    if int(summary.get("technical_games", -1)) != 0:
        raise ValueError("Arena returned technical games")
    contract = summary.get("scientific_contract")
    required_contract = {
        "games": games,
        "komi": 0.5,
        "simulations": 64,
        "cpuct": 1.25,
        "fpu": 0.0,
        "noise": False,
        "temperature": 0.0,
        "fast_search": False,
        "resign": False,
        "paired_starts_color_swap": True,
        "technical_fail_closed": True,
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in required_contract.items()
    ):
        raise ValueError("Arena scientific contract drift")
    execution = summary.get("execution")
    if not isinstance(execution, Mapping) or any(
        execution.get(key) != value
        for key, value in {
            "workers": 16,
            "games_per_worker": 4,
            "inference_batch_rows": 64,
            "inference_batch_wait_ms": 6.0,
            "device": "cuda",
        }.items()
    ):
        raise ValueError("Arena execution contract drift")
    if summary.get("candidate_artifact_sha256") != file_sha256(candidate):
        raise ValueError("Arena candidate artifact changed during evaluation")
    if summary.get("reference_artifact_sha256") != file_sha256(reference):
        raise ValueError("Arena reference artifact changed during evaluation")
    if summary.get("candidate_model_hash") != _checkpoint_metadata(candidate).get("model_hash"):
        raise ValueError("Arena candidate model identity drift")
    if summary.get("reference_model_hash") != _checkpoint_metadata(reference).get("model_hash"):
        raise ValueError("Arena reference model identity drift")
    telemetry = summary.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("Arena execution telemetry is missing")
    if len(set(telemetry.get("worker_pids", []))) != 16:
        raise ValueError("Arena worker PID count drift")
    if any(telemetry.get("worker_cuda_initialized_after_run", [])):
        raise ValueError("Arena initialized CUDA inside a search worker")


def _run_current_arena(
    *, name: str, candidate: Path, reference: Path, run_dir: Path,
    games: int, seed: int, comparison: str
) -> tuple[dict[str, object], Path]:
    from tools.arena_engine import ArenaExecutionConfig, run_arena
    from tools.arena_profiles.torus9 import Torus9ArenaProfile

    output = run_dir / "arena" / name
    if (output / "summary.json").is_file():
        summary = _read_json(output / "summary.json")
    else:
        config = Stage7ArenaConfig(games=games)
        config.validate()
        summary = run_arena(
            profile=Torus9ArenaProfile(),
            candidate_path=candidate,
            reference_path=reference,
            output_dir=output,
            candidate_label=candidate.stem,
            reference_label=reference.stem,
            run_id=f"{run_dir.name}-{name}",
            comparison=comparison,
            master_seed=seed,
            # ``arena_batch_size=8`` is the fixed logical Stage-7 contract;
            # the current parent broker cap is 64 rows, as in the shared rail.
            config=ArenaExecutionConfig(
                games=games, workers=16, games_per_worker=4,
                inference_batch_rows=64, inference_batch_wait_ms=6.0,
                device="cuda", strict_production=False,
                min_mean_inference_batch_rows=16.0,
            ),
            expected_candidate_model_hash=_checkpoint_metadata(candidate)["model_hash"],
            expected_candidate_artifact_sha256=file_sha256(candidate),
            expected_reference_model_hash=_checkpoint_metadata(reference)["model_hash"],
            expected_reference_artifact_sha256=file_sha256(reference),
        )
    _validate_arena_summary(summary, games=games, candidate=candidate, reference=reference)
    return summary, output


def _fixed_weight_status(snapshot: Mapping[str, object]) -> str:
    if int(snapshot.get("technical_games", 0)) != 0:
        return "FAIL"
    score = snapshot.get("score")
    if not isinstance(score, (int, float)):
        return "FAIL"
    if float(score) >= 0.70:
        return "PASS"
    if float(score) >= 0.60:
        return "INCONCLUSIVE"
    return "FAIL"


def _paired_blocks(games_paths: Sequence[Path]) -> tuple[float, ...]:
    rows: list[dict[str, object]] = []
    for path in games_paths:
        rows.extend(_read_jsonl(path))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row.get("technical_termination") is not None:
            raise ValueError("Technical Arena outcome cannot enter paired bootstrap")
        grouped.setdefault(str(row["pair_id"]), []).append(row)
    blocks: list[float] = []
    for pair_id, pair in sorted(grouped.items()):
        if len(pair) != 2 or {bool(row.get("candidate_black")) for row in pair} != {True, False}:
            raise ValueError(f"Incomplete or non-color-swapped Arena pair: {pair_id}")
        scores = [
            1.0 if row.get("mapped_result") == "A_WIN" else
            0.5 if row.get("mapped_result") == "DRAW" else 0.0
            for row in pair
        ]
        blocks.append(sum(scores) / 2.0)
    return tuple(blocks)


def paired_block_bootstrap(
    blocks: Sequence[float], *, seed: int, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, object]:
    if not blocks or any(not math.isfinite(float(value)) for value in blocks):
        raise ValueError("Paired bootstrap requires finite non-empty blocks")
    rng = random.Random(int(seed))
    values = [float(value) for value in blocks]
    means: list[float] = []
    count = len(values)
    for _ in range(int(replicates)):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    lower_index = min(len(means) - 1, max(0, math.ceil(0.05 * len(means)) - 1))
    upper_index = min(len(means) - 1, max(0, math.ceil(0.95 * len(means)) - 1))
    observed = sum(values) / count
    diff = observed - 0.5
    lower_diff = means[lower_index] - 0.5
    upper_diff = means[upper_index] - 0.5
    if lower_diff >= -ARENA_NON_INFERIORITY_MARGIN:
        status = "PASS"
    elif upper_diff < -ARENA_NON_INFERIORITY_MARGIN:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    return {
        "method": "paired-block-bootstrap-v1",
        "seed": int(seed),
        "replicates": int(replicates),
        "paired_blocks": count,
        "observed_new_score": observed,
        "observed_difference_vs_old": diff,
        "one_sided_95_lower_difference": lower_diff,
        "one_sided_95_upper_difference": upper_diff,
        "non_inferiority_margin": ARENA_NON_INFERIORITY_MARGIN,
        "pass_threshold_difference": -ARENA_NON_INFERIORITY_MARGIN,
        "status": status,
    }


def run_arena_stage7(*, run_dir: Path, canonical_run: Path, code: CodeIdentity) -> dict[str, object]:
    run_dir = _assert_isolated(run_dir, canonical_run)
    with _run_lock(run_dir):
        manifest = _load_manifest(run_dir)
        _validate_manifest(manifest, run_dir=run_dir, canonical_run=canonical_run)
        _validate_frozen_source(manifest, code)
        _validate_finished_training(run_dir, expected_last=VALIDATION_HORIZON)
        historical_m10 = canonical_run / "checkpoints" / "M10.pt"
        historical_m5 = canonical_run / "checkpoints" / "M5.pt"
        historical_m6 = canonical_run / "checkpoints" / "M6.pt"
        new_m6 = run_dir / "checkpoints" / "M6.pt"
        fixed_summary, fixed_output = _run_current_arena(
            name="fixed-M10-vs-M5-64", candidate=historical_m10,
            reference=historical_m5, run_dir=run_dir, games=ARENA_FIXED_GAMES,
            seed=TORUS9_CURRENT_ARENA_MASTER_SEED, comparison=CANONICAL_ARENA_COMPARISON,
        )
        fixed_initial = _arena_snapshot(fixed_summary, fixed_output / "games.jsonl")
        fixed_extension: dict[str, object] | None = None
        fixed_components = [fixed_initial]
        fixed_status = _fixed_weight_status(fixed_initial)
        if fixed_status == "INCONCLUSIVE":
            extension_summary, extension_output = _run_current_arena(
                name="fixed-M10-vs-M5-extension-64", candidate=historical_m10,
                reference=historical_m5, run_dir=run_dir, games=ARENA_FIXED_GAMES,
                seed=derive_seed(TORUS9_CURRENT_ARENA_MASTER_SEED, run_dir.name, "fixed-extension-64"),
                comparison=CANONICAL_ARENA_COMPARISON,
            )
            fixed_extension = {
                "summary_path": str(extension_output / "summary.json"),
                "games_path": str(extension_output / "games.jsonl"),
                "result": _arena_snapshot(extension_summary, extension_output / "games.jsonl"),
                "seed": derive_seed(TORUS9_CURRENT_ARENA_MASTER_SEED, run_dir.name, "fixed-extension-64"),
            }
            fixed_components.append(fixed_extension["result"])  # type: ignore[arg-type]
        fixed = _combine_arena_snapshots(fixed_components)
        fixed_status = _fixed_weight_status(fixed)
        strength_summary, strength_output = _run_current_arena(
            name="new-M6-vs-old-M6-256", candidate=new_m6,
            reference=historical_m6, run_dir=run_dir,
            games=ARENA_STRENGTH_INITIAL_GAMES,
            seed=202609131005,
            comparison="NEW-M6-vs-OLD-M6",
        )
        paths = [strength_output / "games.jsonl"]
        blocks = _paired_blocks(paths)
        bootstrap = paired_block_bootstrap(
            blocks, seed=derive_seed(202609131005, run_dir.name, "paired-bootstrap", 256)
        )
        extension: dict[str, object] | None = None
        if bootstrap["status"] == "INCONCLUSIVE":
            extension_summary, extension_output = _run_current_arena(
                name="new-M6-vs-old-M6-extension-256", candidate=new_m6,
                reference=historical_m6, run_dir=run_dir,
                games=ARENA_STRENGTH_INITIAL_GAMES,
                seed=derive_seed(202609131005, run_dir.name, "arena-extension-256"),
                comparison="NEW-M6-vs-OLD-M6-extension",
            )
            paths.append(extension_output / "games.jsonl")
            blocks = _paired_blocks(paths)
            bootstrap = paired_block_bootstrap(
                blocks, seed=derive_seed(202609131005, run_dir.name, "paired-bootstrap", 512)
            )
            extension = {
                "summary_path": str(extension_output / "summary.json"),
                "games_path": str(extension_output / "games.jsonl"),
                "result": _arena_snapshot(extension_summary, extension_output / "games.jsonl"),
                "seed": derive_seed(202609131005, run_dir.name, "arena-extension-256"),
            }
        strength = _arena_snapshot(strength_summary, strength_output / "games.jsonl")
        total_strength_games = len(blocks) * 2
        if total_strength_games not in (ARENA_STRENGTH_INITIAL_GAMES, ARENA_STRENGTH_TOTAL_GAMES):
            raise ValueError(f"Unexpected NEW M6 vs OLD M6 game count: {total_strength_games}")
        if (extension is None) != (total_strength_games == ARENA_STRENGTH_INITIAL_GAMES):
            raise ValueError("Strength Arena extension/game-count mismatch")
        total_wld = [0, 0, 0]
        for path in paths:
            summary = _read_json(path.parent / "summary.json")
            wld = summary.get("W/L/D", [0, 0, 0])
            for index in range(3):
                total_wld[index] += int(wld[index])
        arena = {
            "schema": "torus9-stage7-arena-v3",
            "contract": Stage7ArenaConfig(games=ARENA_FIXED_GAMES).as_dict(),
            "fixed_weight_old_m10_vs_old_m5": {
                "historical_reference_summary": str(canonical_run / "arena" / "M10-vs-M5" / "summary.json"),
                "historical_reference_result": _arena_snapshot(
                    _read_json(canonical_run / "arena" / "M10-vs-M5" / "summary.json"),
                    canonical_run / "arena" / "M10-vs-M5" / "games.jsonl",
                ),
                "current_summary_path": str(fixed_output / "summary.json"),
                "current_games_path": str(fixed_output / "games.jsonl"),
                "initial_64": fixed_initial,
                "extension": fixed_extension,
                "result": fixed,
                "status": fixed_status,
            },
            "new_m6_vs_old_m6": {
                "new_checkpoint": _checkpoint_identity(new_m6),
                "old_checkpoint": _checkpoint_identity(historical_m6),
                "initial_256": {
                    "summary_path": str(strength_output / "summary.json"),
                    "games_path": str(strength_output / "games.jsonl"),
                    "result": strength,
                    "seed": 202609131005,
                },
                "extension": extension,
                "total_games": len(blocks) * 2,
                "W/L/D": total_wld,
                "paired_block_bootstrap": bootstrap,
                "status": bootstrap["status"],
            },
        }
        manifest = _load_manifest(run_dir)
        manifest["arena"] = arena
        _atomic_write(_manifest_path(run_dir), manifest)
        _atomic_write(run_dir / "arena" / "stage7-arena-summary.json", arena)
        return arena


def run_protocol_stage7(*, run_dir: Path, canonical_run: Path, code: CodeIdentity) -> dict[str, object]:
    from alphazero.envs.gocube.integration.catalog import CheckpointCatalog
    from alphazero.envs.gocube.integration.golden_generation import replay_protocol_game
    from alphazero.envs.gocube.integration.golden_models import GoldenCheckpointLoader
    from alphazero.envs.gocube.integration.service import GoCubeAlphaZeroService

    run_dir = _assert_isolated(run_dir, canonical_run)
    with _run_lock(run_dir):
        manifest = _load_manifest(run_dir)
        _validate_manifest(manifest, run_dir=run_dir, canonical_run=canonical_run)
        _validate_frozen_source(manifest, code)
        _validate_finished_training(run_dir, expected_last=VALIDATION_HORIZON)
        source = run_dir / "checkpoints" / "M6.pt"
        catalog_root = run_dir / "protocol" / "catalog"
        target = catalog_root / "checkpoints" / "M6.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
            shutil.copy2(source.with_suffix(".metadata.json"), target.with_suffix(".metadata.json"))
        catalog = CheckpointCatalog(str(catalog_root))
        checkpoint_id = f"{run_dir.name}@6"
        descriptor = catalog.get(checkpoint_id)
        if descriptor is None:
            raise RuntimeError("Protocol V1 catalog discovery did not find NEW M6")
        loader = GoldenCheckpointLoader(catalog, device="cpu")
        _, playable = loader.load(checkpoint_id)
        service = GoCubeAlphaZeroService(str(catalog_root), catalog=catalog, loader=loader)
        health = service.health()
        listing = service.checkpoints()
        response = service.generate_game({
            "protocolVersion": 1,
            "blackCheckpointId": checkpoint_id,
            "whiteCheckpointId": checkpoint_id,
            "mctsSims": 1,
        })
        game = response["game"]
        replayed = replay_protocol_game(game)
        source_metadata = _checkpoint_metadata(source)
        integration_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "alphazero/envs/gocube/integration").glob("*.py")
        )
        forbidden = ("NNetWrapper", "GenericPlayers", "SelfPlayAgent", "Coach")
        checks = {
            "catalog_discovery": descriptor is not None,
            "catalog_lists_checkpoint": any(item.get("id") == checkpoint_id for item in listing["checkpoints"]),
            "health_protocol_v1": health.get("protocolVersion") == 1,
            "loader_model_hash": playable.metadata.get("model_hash") == source_metadata.get("model_hash"),
            "topology_torus9": game.get("topology") == "torus" and game.get("size") == 9,
            "komi": game.get("komi") == 0.5 and descriptor.komi == 0.5,
            "legal_moves": len(game.get("moves", [])) > 0,
            "captures_fields": all("captured" in move for move in game.get("moves", [])),
            "passes": sum(move.get("action", {}).get("type") == "pass" for move in game.get("moves", [])) == 2,
            "terminal_result": bool(replayed.is_terminal),
            "checkpoint_identity": checkpoint_id == f"{run_dir.name}@6",
            "no_legacy_runtime": not any(token in integration_sources for token in forbidden),
        }
        result = {
            "schema": "torus9-stage7-protocol-v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "checkpoint_id": checkpoint_id,
            "model_hash": source_metadata.get("model_hash"),
            "game": {
                "moves": len(game.get("moves", [])),
                "passes": sum(move.get("action", {}).get("type") == "pass" for move in game.get("moves", [])),
                "topology": game.get("topology"),
                "size": game.get("size"),
                "komi": game.get("komi"),
                "winner": game.get("result", {}).get("winner"),
            },
            "public_path": "GoCubeAlphaZeroService -> CheckpointCatalog/GoldenCheckpointLoader -> GoldenGameGenerator",
        }
        if result["status"] != "PASS":
            raise RuntimeError(f"Stage 7 Protocol V1 smoke failed: {result}")
        manifest = _load_manifest(run_dir)
        manifest["protocol"] = result
        _atomic_write(_manifest_path(run_dir), manifest)
        _atomic_write(run_dir / "protocol" / "stage7-protocol-summary.json", result)
        return result


def _status_from_manifest(manifest: Mapping[str, object], *, canonical_run: Path) -> dict[str, str]:
    run_dir_value = manifest.get("run_dir")
    rows = (
        [
            _read_json(_generation_paths(Path(str(run_dir_value)), generation)["summary"])
            for generation in range(1, VALIDATION_HORIZON + 1)
        ]
        if run_dir_value
        else []
    )
    expected_scientific = {
        "profile": manifest.get("profile_fingerprint"),
        "rules": manifest.get("rules_fingerprint"),
        "target": manifest.get("target_fingerprint"),
        "selfplay": manifest.get("selfplay_contract_fingerprint"),
        "komi": TORUS9_KOMI,
    }
    scientific = "PASS" if all(
        isinstance(row.get("scientific_fingerprint"), Mapping)
        and dict(row["scientific_fingerprint"]) == expected_scientific  # type: ignore[index]
        for row in rows
    ) and len(rows) == VALIDATION_HORIZON else "FAIL"
    m0 = manifest.get("m0_parity")
    m0_status = "PASS" if isinstance(m0, Mapping) and all(m0.get(key) for key in ("model_hash_equal", "tensor_equal", "scientific_fingerprint_equal")) and m0.get("komi") == 0.5 else "FAIL"
    valid = all(
        isinstance(row.get("self_play"), Mapping)
        and row["self_play"].get("games") == 64  # type: ignore[index]
        and row["self_play"].get("technical_games") == 0  # type: ignore[index]
        and row["self_play"].get("effective_context_ceiling") == 64  # type: ignore[index]
        and row["self_play"].get("execution_reference_status") == "validated_recommended"  # type: ignore[index]
        for row in rows
    ) and len(rows) == VALIDATION_HORIZON
    training = all(
        isinstance(row.get("training"), Mapping)
        and row["training"].get("optimizer_steps") == 80  # type: ignore[index]
        and row["training"].get("samples_consumed") == 5120  # type: ignore[index]
        for row in rows
    ) and len(rows) == VALIDATION_HORIZON
    replay_audit = manifest.get("replay_audit")
    replay = "PASS" if isinstance(replay_audit, Mapping) and replay_audit.get("status") == "PASS" else "FAIL"
    resume = manifest.get("resume")
    resume_status = "PASS" if isinstance(resume, Mapping) and resume.get("resume_result") == "PASS" else "FAIL"
    arena = manifest.get("arena")
    fixed = "FAIL"
    strength = "FAIL"
    execution = "FAIL"
    execution_degraded = False
    if any(
        isinstance(row.get("self_play"), Mapping)
        and isinstance(row["self_play"].get("performance"), Mapping)  # type: ignore[index]
        and row["self_play"]["performance"].get("status") == "PERFORMANCE_DEGRADED"  # type: ignore[index]
        for row in rows
    ):
        execution_degraded = True
    if isinstance(arena, Mapping):
        fixed_result = arena.get("fixed_weight_old_m10_vs_old_m5")
        strength_result = arena.get("new_m6_vs_old_m6")
        if isinstance(fixed_result, Mapping):
            fixed = str(fixed_result.get("status", "FAIL"))
            result = fixed_result.get("result")
            if isinstance(result, Mapping) and result.get("performance_status") == "PERFORMANCE_DEGRADED":
                execution_degraded = True
        if isinstance(strength_result, Mapping):
            strength = str(strength_result.get("status", "FAIL"))
        if isinstance(fixed_result, Mapping) and isinstance(fixed_result.get("result"), Mapping):
            execution = "PERFORMANCE_DEGRADED" if execution_degraded else "PASS"
    elif rows and execution_degraded:
        execution = "PERFORMANCE_DEGRADED"
    elif rows:
        execution = "PASS"
    protocol = manifest.get("protocol")
    protocol_status = "PASS" if isinstance(protocol, Mapping) and protocol.get("status") == "PASS" else "FAIL"
    canonical_before = manifest.get("canonical_before")
    historical = "PASS" if (
        isinstance(canonical_before, Mapping)
        and canonical_snapshot(canonical_run) == canonical_before
        and not canonical_before.get("m18_artifacts")
    ) else "FAIL"
    statuses = {
        "scientific_contract": scientific,
        "m0_initialization_parity": m0_status,
        "selfplay_validity": "PASS" if valid else "FAIL",
        "training_and_optimizer": "PASS" if training else "FAIL",
        "resume_and_checkpoint": resume_status,
        "replay_window": replay,
        "execution": execution,
        "fixed_weight_runtime_non_regression": fixed,
        "training_strength_non_regression": strength,
        "protocol_v1": protocol_status,
        "historical_artifact_integrity": historical,
    }
    correctness = [value for key, value in statuses.items() if key != "execution"]
    statuses["overall"] = "PASS" if all(value == "PASS" for value in correctness) else "FAIL"
    return statuses


def generate_report(*, run_dir: Path, canonical_run: Path, code: CodeIdentity) -> dict[str, object]:
    run_dir = _assert_isolated(run_dir, canonical_run)
    with _run_lock(run_dir):
        manifest = _load_manifest(run_dir)
        _validate_manifest(manifest, run_dir=run_dir, canonical_run=canonical_run)
        _validate_frozen_source(manifest, code)
        _validate_finished_training(run_dir, expected_last=VALIDATION_HORIZON)
        replay = _replay_audit(run_dir, VALIDATION_HORIZON)
        manifest["replay_audit"] = replay
        manifest["run_dir"] = str(run_dir)
        _atomic_write(_manifest_path(run_dir), manifest)
        statuses = _status_from_manifest(manifest, canonical_run=canonical_run)
        rows: list[dict[str, object]] = []
        for generation in range(1, VALIDATION_HORIZON + 1):
            summary = _read_json(_generation_paths(run_dir, generation)["summary"])
            selfplay = summary["self_play"]
            training = summary["training"]
            replay_row = summary["replay"]
            checkpoint = summary["checkpoint"]
            rows.append({
                "label": f"M{generation}",
                "games": selfplay["games"],  # type: ignore[index]
                "moves": selfplay["moves"],  # type: ignore[index]
                "samples": training["samples_consumed"],  # type: ignore[index]
                "optimizer_steps": training["optimizer_steps"],  # type: ignore[index]
                "fresh_positions": summary["fresh_positions"],
                "replay_positions": replay_row.get("rolling_buffer_positions"),  # type: ignore[union-attr]
                "replay_generations": summary["replay_identity"]["generations"],  # type: ignore[index]
                "losses": {key: training.get(key) for key in ("loss", "policy_loss", "value_loss", "ownership_loss", "score_loss")},  # type: ignore[union-attr]
                "moves_per_sec": selfplay["moves_per_sec"],  # type: ignore[index]
                "inference_rows_per_sec": selfplay["inference_rows_per_sec"],  # type: ignore[index]
                "batch": selfplay["batch"],  # type: ignore[index]
                "worker_pids": selfplay["worker_pids"],  # type: ignore[index]
                "technical": selfplay["technical_games"],  # type: ignore[index]
                "model_hash": checkpoint["model_hash"],  # type: ignore[index]
            })
        report = {
            "schema": "torus9-stage7-final-report-v1",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "main_sha": BASE_SHA,
            "stage7_branch": manifest.get("branch", "codex/stage7-golden-acceptance"),
            "stage7_source_sha": manifest.get("source_commit"),
            "stage7_head_sha_at_report": code.git_commit_sha,
            "run_id": manifest["run_id"],
            "run_dir": str(run_dir),
            "profile_fingerprint": manifest["profile_fingerprint"],
            "scientific_fingerprint": {
                "profile": manifest["profile_fingerprint"],
                "rules": manifest["rules_fingerprint"],
                "target": manifest["target_fingerprint"],
                "selfplay": manifest["selfplay_contract_fingerprint"],
                "komi": TORUS9_KOMI,
            },
            "hardware": manifest["preflight"]["hardware"],  # type: ignore[index]
            "m0": manifest["m0_parity"],
            "iterations": rows,
            "resume_m2_to_m3": manifest.get("resume"),
            "replay_audit": replay,
            "arena": manifest.get("arena"),
            "protocol_v1": manifest.get("protocol"),
            "old_m17_sha_before": manifest["canonical_before"]["m17_sha256"],  # type: ignore[index]
            "old_m17_sha_after": canonical_snapshot(canonical_run)["m17_sha256"],
            "canonical_m18": "absent" if not manifest["canonical_before"]["m18_artifacts"] else "present",  # type: ignore[index]
            "statuses": statuses,
            "overall": statuses["overall"],
            "golden_standard_spreadsheet": "NOT TOUCHED",
            "preconditions": {
                "post_108_main": manifest["preflight"]["checks"]["post_108_base"],  # type: ignore[index]
                "source_worktree_clean_at_start": manifest.get("source_worktree_clean"),
                "source_identity_frozen": True,
                "historical_run_read_only": True,
            },
        }
        _atomic_write(run_dir / "final-report.json", report)
        _atomic_write(ROOT / "docs" / "TORUS9_GOLDEN_STAGE7_FINAL_20260915.json", report)
        _atomic_write_text(ROOT / "docs" / "TORUS9_GOLDEN_STAGE7_FINAL_20260915.md", _render_markdown(report))
        return report


def _render_markdown(report: Mapping[str, object]) -> str:
    statuses = report["statuses"]
    lines = [
        "# Torus9 Golden Stage 7 final acceptance",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Main SHA: `{report['main_sha']}`  ",
        f"Stage-7 source SHA: `{report['stage7_source_sha']}`",
        "",
        f"Overall: **{report['overall']}**",
        "",
        "## Iterations",
        "",
        "| M | games | moves | samples | optimizer steps | replay generations | moves/s | batch mean/p50/p95/max | technical |",
        "|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in report["iterations"]:  # type: ignore[union-attr]
        batch = row["batch"]
        lines.append(
            f"| {row['label']} | {row['games']} | {row['moves']} | {row['samples']} | {row['optimizer_steps']} | {row['replay_generations']} | {float(row['moves_per_sec']):.3f} | {batch['mean']}/{batch['p50']}/{batch['p95']}/{batch['max']} | {row['technical']} |"
        )
    lines.extend(["", "## Independent statuses", ""])
    for key, value in report["statuses"].items():  # type: ignore[union-attr]
        lines.append(f"- `{key}`: **{value}**")
    lines.extend([
        "",
        "Historical M17 before/after: "
        f"`{report['old_m17_sha_before']}` / `{report['old_m17_sha_after']}`; "
        f"canonical M18: **{report['canonical_m18']}**.",
        "",
    ])
    return "\n".join(lines)


def _run_child(command: str, *, run_id: str, run_dir: Path, canonical_run: Path) -> None:
    command_line = [
        sys.executable, str(Path(__file__).resolve()), command,
        "--run-id", run_id, "--run-dir", str(run_dir),
        "--canonical-run", str(canonical_run),
    ]
    subprocess.run(command_line, cwd=ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "new", "resume", "arena", "protocol", "report", "all"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--canonical-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = _safe_run_id(args.run_id)
    run_dir = (args.run_dir or default_run_dir(run_id)).resolve()
    canonical_run = args.canonical_run.resolve()
    code = capture_code_identity(ROOT)
    if args.command == "preflight":
        result = _preflight(run_dir=run_dir, canonical_run=canonical_run, code=code)
    elif args.command == "new":
        result = run_new(run_dir=run_dir, canonical_run=canonical_run, run_id=run_id, code=code)
    elif args.command == "resume":
        result = run_resume(run_dir=run_dir, canonical_run=canonical_run, code=code)
    elif args.command == "arena":
        result = run_arena_stage7(run_dir=run_dir, canonical_run=canonical_run, code=code)
    elif args.command == "protocol":
        result = run_protocol_stage7(run_dir=run_dir, canonical_run=canonical_run, code=code)
    elif args.command == "report":
        result = generate_report(run_dir=run_dir, canonical_run=canonical_run, code=code)
    else:
        _run_child("new", run_id=run_id, run_dir=run_dir, canonical_run=canonical_run)
        _run_child("resume", run_id=run_id, run_dir=run_dir, canonical_run=canonical_run)
        _run_child("arena", run_id=run_id, run_dir=run_dir, canonical_run=canonical_run)
        _run_child("protocol", run_id=run_id, run_dir=run_dir, canonical_run=canonical_run)
        result = generate_report(run_dir=run_dir, canonical_run=canonical_run, code=capture_code_identity(ROOT))
    print(json.dumps(_jsonable(result), sort_keys=True))
    if args.command != "preflight" and isinstance(result, Mapping) and result.get("overall") == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
