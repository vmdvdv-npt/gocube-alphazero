#!/usr/bin/env python3
"""Production Torus9 adapter driven only by an immutable run spec.

The driver owns Torus9 scientific adaptation; it does not choose production
execution policy.  Worker counts, concurrency, batching, device, seeds,
heartbeat interval, Arena workload and Arena execution gates are all required
in the lineage-owned run spec.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256
from gocube_golden.run_spec import StrictRunSpec, run_spec_fingerprint
from gocube_golden.run_storage import (
    ensure_evaluation_layout,
    evaluation_id_for_comparison,
    resolve_checkpoint,
)
from gocube_golden.artifact_catalog import ArtifactCatalog, ARTIFACT_VALIDATION_SCHEMA
from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9SelfPlaySearchContract,
    Torus9TrainingAdapter,
    run_torus9_selfplay_games,
    run_torus9_training_iteration,
    torus9_load_checkpoint,
)
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)
from gocube_golden.torus9_training import TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA
from tools.arena_engine import (
    ArenaExecutionConfig,
    classify_arena_performance,
    run_arena as run_arena_engine,
)
from tools.arena_profiles.torus9 import PROFILE as TORUS9_ARENA_PROFILE


GENERATION_RESULT_SCHEMA = "gocube-generation-driver-result-v1"
ARENA_RESULT_SCHEMA = "gocube-arena-driver-result-v1"
HEARTBEAT_SCHEMA = "gocube-training-driver-heartbeat-v2"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _required(value: Mapping[str, object], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} is missing explicit fields: {', '.join(missing)}")


def _positive_int(value: object, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_float(value: object, label: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _validate_device(device: str) -> None:
    parsed = torch.device(device)
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run spec requested CUDA but CUDA is unavailable")


def _load_run_spec() -> StrictRunSpec:
    raw_path = os.environ.get("AZ_RUN_SPEC_PATH")
    expected = os.environ.get("AZ_RUN_SPEC_FINGERPRINT")
    if not raw_path or not expected:
        raise ValueError("Torus9 driver requires AZ_RUN_SPEC_PATH and AZ_RUN_SPEC_FINGERPRINT")
    spec = StrictRunSpec.load(Path(raw_path), repo_root=ROOT)
    if spec.fingerprint != expected:
        raise ValueError("Immutable run-spec identity mismatch")
    if spec.orchestrator_spec.topology != "torus9" or spec.board_size != 9:
        raise ValueError("Torus9 driver requires topology=torus9 and board_size=9")
    return spec


def _generation_config(spec: StrictRunSpec) -> dict[str, object]:
    generation = _mapping(spec.payload["generation"], "generation")
    config = _mapping(generation["driver_config"], "generation.driver_config")
    _required(
        config,
        (
            "games",
            "device",
            "workers",
            "active_games_per_worker",
            "total_active_contexts",
            "inference_batch_cap",
            "inference_batch_wait_ms",
            "coalescing",
            "heartbeat_interval_seconds",
            "model_init_seed",
            "selfplay_master_seed",
            "training_master_seed",
        ),
        "generation.driver_config",
    )
    workers = _positive_int(config["workers"], "generation.workers")
    active = _positive_int(
        config["active_games_per_worker"], "generation.active_games_per_worker"
    )
    contexts = _positive_int(
        config["total_active_contexts"], "generation.total_active_contexts"
    )
    if contexts > workers * active:
        raise ValueError("generation.total_active_contexts exceeds configured lane capacity")
    if not isinstance(config["coalescing"], bool):
        raise ValueError("generation.coalescing must be boolean")
    device = str(config["device"]).strip()
    if not device:
        raise ValueError("generation.device must be explicit")
    heartbeat = float(config["heartbeat_interval_seconds"])
    if heartbeat <= 0:
        raise ValueError("generation.heartbeat_interval_seconds must be positive")
    return {
        **config,
        "games": _positive_int(config["games"], "generation.games"),
        "device": device,
        "workers": workers,
        "active_games_per_worker": active,
        "total_active_contexts": contexts,
        "inference_batch_cap": _positive_int(
            config["inference_batch_cap"], "generation.inference_batch_cap"
        ),
        "inference_batch_wait_ms": _nonnegative_float(
            config["inference_batch_wait_ms"], "generation.inference_batch_wait_ms"
        ),
        "heartbeat_interval_seconds": heartbeat,
        "model_init_seed": int(config["model_init_seed"]),
        "selfplay_master_seed": int(config["selfplay_master_seed"]),
        "training_master_seed": int(config["training_master_seed"]),
    }


def _arena_config(spec: StrictRunSpec) -> tuple[dict[str, object], dict[str, object]]:
    arena = _mapping(spec.payload["arena"], "arena")
    if arena.get("enabled") is not True:
        raise ValueError("Arena driver invoked while arena.enabled is false")
    config = _mapping(arena["driver_config"], "arena.driver_config")
    startset = _mapping(arena["startset"], "arena.startset")
    _required(
        config,
        (
            "comparison_mode",
            "reference_gap",
            "games",
            "master_seed",
            "heartbeat_interval_seconds",
            "execution",
        ),
        "arena.driver_config",
    )
    if config["comparison_mode"] != "candidate-vs-prior-generation":
        raise ValueError("Torus9 periodic Arena requires explicit candidate-vs-prior-generation mode")
    execution = _mapping(config["execution"], "arena.driver_config.execution")
    _required(
        execution,
        (
            "workers",
            "games_per_worker",
            "inference_batch_rows",
            "inference_batch_wait_ms",
            "device",
            "strict_production",
            "min_mean_inference_batch_rows",
            "min_effective_cpu_cores",
            "early_gate_enabled",
            "early_gate_min_forwards",
            "early_gate_min_wall_sec",
        ),
        "arena.driver_config.execution",
    )
    games = _positive_int(config["games"], "arena.games")
    if games % 2:
        raise ValueError("arena.games must be even for paired color-swapped starts")
    heartbeat = float(config["heartbeat_interval_seconds"])
    if heartbeat <= 0:
        raise ValueError("arena.heartbeat_interval_seconds must be positive")
    normalized_execution = {
        "workers": _positive_int(execution["workers"], "arena.execution.workers"),
        "games_per_worker": _positive_int(
            execution["games_per_worker"], "arena.execution.games_per_worker"
        ),
        "inference_batch_rows": _positive_int(
            execution["inference_batch_rows"], "arena.execution.inference_batch_rows"
        ),
        "inference_batch_wait_ms": _nonnegative_float(
            execution["inference_batch_wait_ms"],
            "arena.execution.inference_batch_wait_ms",
        ),
        "device": str(execution["device"]).strip(),
        "strict_production": execution["strict_production"],
        "min_mean_inference_batch_rows": float(
            execution["min_mean_inference_batch_rows"]
        ),
        "min_effective_cpu_cores": float(execution["min_effective_cpu_cores"]),
        "early_gate_enabled": execution["early_gate_enabled"],
        "early_gate_min_forwards": _positive_int(
            execution["early_gate_min_forwards"], "arena.execution.early_gate_min_forwards"
        ),
        "early_gate_min_wall_sec": _nonnegative_float(
            execution["early_gate_min_wall_sec"],
            "arena.execution.early_gate_min_wall_sec",
        ),
    }
    if not normalized_execution["device"]:
        raise ValueError("arena.execution.device must be explicit")
    if not isinstance(normalized_execution["strict_production"], bool):
        raise ValueError("arena.execution.strict_production must be boolean")
    if not isinstance(normalized_execution["early_gate_enabled"], bool):
        raise ValueError("arena.execution.early_gate_enabled must be boolean")
    if float(normalized_execution["min_mean_inference_batch_rows"]) < 0:
        raise ValueError("arena.execution.min_mean_inference_batch_rows must be non-negative")
    if float(normalized_execution["min_effective_cpu_cores"]) < 0:
        raise ValueError("arena.execution.min_effective_cpu_cores must be non-negative")
    _required(startset, ("schema", "generator", "master_seed", "pairs"), "arena.startset")
    master_seed = int(config["master_seed"])
    if int(startset["master_seed"]) != master_seed:
        raise ValueError("arena.startset.master_seed must match arena.driver_config.master_seed")
    if int(startset["pairs"]) != games // 2:
        raise ValueError("arena.startset.pairs must equal arena.games / 2")
    return (
        {
            **config,
            "reference_gap": _positive_int(config["reference_gap"], "arena.reference_gap"),
            "games": games,
            "master_seed": master_seed,
            "heartbeat_interval_seconds": heartbeat,
            "execution": normalized_execution,
        },
        startset,
    )


def _validate_code_pin(root: Path) -> CodeIdentity:
    manifest = _read_json(root / "manifest.json")
    current = capture_code_identity(ROOT)
    expected = str(manifest.get("git_commit", ""))
    if expected != current.git_commit_sha:
        raise ValueError(
            "Training lineage git commit drift: "
            f"manifest={expected!r} current={current.git_commit_sha!r}"
        )
    if not current.working_tree_clean:
        raise ValueError("Production training refuses a dirty working tree")
    return current


class _Heartbeat:
    """Separate process liveness from semantic progress."""

    def __init__(self, path: Path, generation: int, interval: float) -> None:
        self.path = path
        self.generation = int(generation)
        self.interval = float(interval)
        self.phase = "startup"
        self.progress_token = "startup"
        self.progress_at = time.time()
        self.progress: dict[str, object] | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._last_write_monotonic = 0.0
        self._minimum_write_interval = min(1.0, max(0.1, self.interval / 2.0))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def advance(
        self,
        phase: str,
        *,
        token: str | None = None,
        completed: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        subphase: str | None = None,
    ) -> None:
        with self._lock:
            old_phase = self.phase
            self.phase = str(phase)
            if completed is not None or total is not None:
                progress_unit = unit or "units"
                self.progress = {
                    "completed": completed,
                    "total": total,
                    "unit": progress_unit,
                }
                if subphase is not None:
                    self.progress["subphase"] = str(subphase)
                default_token = f"{self.phase}:{completed}/{total} {progress_unit}"
            else:
                self.progress = None
                default_token = self.phase
            self.progress_token = str(token) if token is not None else default_token
            self.progress_at = time.time()
        self.write(force=old_phase != self.phase or completed == total)

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            self.error = f"{type(exc).__name__}: {exc}"
            self.phase = "failed"
        self.write(force=True)

    def write(self, *, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            monotonic_now = time.monotonic()
            if (
                not force
                and self._last_write_monotonic
                and monotonic_now - self._last_write_monotonic < self._minimum_write_interval
            ):
                return
            payload: dict[str, object] = {
                "schema": HEARTBEAT_SCHEMA,
                "liveness_at": now,
                "progress_at": self.progress_at,
                "progress_token": self.progress_token,
                "pid": os.getpid(),
                "generation": self.generation,
                "phase": self.phase,
            }
            if self.progress is not None:
                progress = dict(self.progress)
                payload["progress"] = progress
                # Keep the scalar form available to simple status consumers;
                # ``progress`` remains the backwards-compatible envelope.
                payload.update(
                    {
                        "completed": progress.get("completed"),
                        "total": progress.get("total"),
                        "unit": progress.get("unit"),
                    }
                )
            if self.error is not None:
                payload["errors"] = [self.error]
            self._last_write_monotonic = monotonic_now
        _atomic_json(self.path, payload)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()

    def __enter__(self) -> "_Heartbeat":
        self.write()
        self._thread = threading.Thread(
            target=self._loop,
            name="torus9-production-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type: object, exc: BaseException | None, _tb: object) -> None:
        if exc is not None:
            self.fail(exc)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 2.0))
        self.write()


def _environment(generation: int) -> tuple[Path, str, Path, str, CodeIdentity]:
    env_generation = int(os.environ["AZ_GENERATION"])
    if env_generation != int(generation):
        raise ValueError("Generation argument disagrees with AZ_GENERATION")
    if os.environ.get("AZ_TOPOLOGY") != "torus9":
        raise ValueError("Torus9 driver requires AZ_TOPOLOGY=torus9")
    root = Path(os.environ["AZ_RUN_ROOT"]).resolve()
    lineage_id = os.environ["AZ_LINEAGE_ID"]
    profile_path = Path(os.environ["AZ_PROFILE_PATH"]).resolve()
    expected_fingerprint = os.environ["AZ_PROFILE_FINGERPRINT"]
    if not root.is_dir():
        raise ValueError(f"Lineage root does not exist: {root}")
    canonical_parent = (ROOT / "runs" / "torus9" / "active").resolve()
    try:
        root.relative_to(canonical_parent)
    except ValueError as exc:
        raise ValueError("Torus9 driver refuses a run root outside canonical active storage") from exc
    return root, lineage_id, profile_path, expected_fingerprint, _validate_code_pin(root)


def _load_profile(profile_path: Path, expected_fingerprint: str) -> dict[str, object]:
    # The shared loader proves the payload, derived section fingerprints and
    # Golden snapshot before the embedded fingerprint is consulted.  Keeping
    # this boundary in torus9_contract avoids a second, inevitably incomplete,
    # hand-maintained validator in the production driver.
    profile = load_torus9_current_profile(profile_path)
    actual = current_torus9_profile_fingerprint(profile)
    if actual != expected_fingerprint:
        raise ValueError("Torus9 canonical profile fingerprint drift")
    return dict(profile)


def _validate_scientific_bindings(
    profile: Mapping[str, object], config: Mapping[str, object]
) -> None:
    self_play = _mapping(profile.get("self_play"), "profile.self_play")
    if int(self_play["games_per_iteration"]) != int(config["games"]):
        raise ValueError(
            "run-spec generation.games must match the immutable scientific profile"
        )
    seeds = _mapping(profile.get("seeds"), "profile.seeds")
    for config_key, profile_key in (
        ("model_init_seed", "model_init_seed"),
        ("selfplay_master_seed", "selfplay_master_seed"),
        ("training_master_seed", "training_master_seed"),
    ):
        if int(config[config_key]) != int(seeds[profile_key]):
            raise ValueError(
                f"run-spec {config_key} disagrees with the referenced scientific profile"
            )


def _contract(profile: Mapping[str, object]) -> Torus9SelfPlaySearchContract:
    settings = _mapping(profile["self_play"], "profile.self_play")
    temperature_plies = settings.get("temperature_plies")
    if not isinstance(temperature_plies, Sequence) or len(temperature_plies) != 2:
        raise ValueError("Torus9 temperature_plies profile is malformed")
    return Torus9SelfPlaySearchContract(
        contract_id=str(settings["contract_id"]),
        simulations=int(settings["mcts_simulations"]),
        cpuct=float(settings["cpuct"]),
        fpu=float(settings["fpu"]),
        temperature_until_ply=int(temperature_plies[1]),
        temperature_after=float(settings["temperature_after"]),
        dirichlet_epsilon=float(settings["dirichlet_epsilon"]),
        dirichlet_alpha=float(settings["dirichlet_alpha"]),
        watchdog=int(settings["watchdog"]),
    )


def _seed_model(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resume_replay_generation_identities(
    *,
    root: Path,
    metadata: Mapping[str, object],
    base_identity: Mapping[str, object] | None,
    catalog: ArtifactCatalog | None,
    parent_reference: Mapping[str, object] | None,
) -> list[dict[str, object]] | None:
    """Resolve compact fresh-generation identities without reading replay rows."""
    for source in (base_identity, metadata):
        if not isinstance(source, Mapping):
            continue
        for key in ("generation_identities", "replay_generation_identities", "replay_identity_components"):
            value = source.get(key)
            if isinstance(value, list):
                return [dict(component) for component in value if isinstance(component, Mapping)]

    raw_generations = metadata.get("replay_generations")
    if not isinstance(raw_generations, list):
        return None
    generations = [int(value) for value in raw_generations]
    parent_entries: dict[int, Mapping[str, object]] = {}
    if isinstance(parent_reference, Mapping):
        references = parent_reference.get("replay_references")
        if isinstance(references, list):
            for raw in references:
                if isinstance(raw, Mapping) and raw.get("generation") is not None:
                    parent_entries[int(raw["generation"])] = raw

    components: list[dict[str, object]] = []
    for generation in generations:
        known: Mapping[str, object] | None = parent_entries.get(generation)
        relative = f"replay/iter-{generation:02d}-fresh.jsonl"
        if catalog is not None:
            candidate = catalog.entries.get(relative)
            if isinstance(candidate, Mapping):
                known = candidate
        if known is None:
            return None
        component: dict[str, object] = {
            "schema": TORUS9_REPLAY_GENERATION_IDENTITY_SCHEMA,
            "generation": generation,
            "sha256": known.get("sha256"),
        }
        if known.get("row_count") is not None:
            component["row_count"] = known["row_count"]
        components.append(component)
    return components


def _prepare_state(
    *,
    root: Path,
    lineage_id: str,
    generation: int,
    profile: Mapping[str, object],
    config: Mapping[str, object],
    device: str,
    code_identity: CodeIdentity,
    timing: Mapping[str, object] | None = None,
) -> tuple[Torus9TrainingAdapter, Any, Path]:
    adapter = Torus9TrainingAdapter(
        profile=profile,
        code_identity=code_identity,
        base_commit=TORUS9_GOLDEN_LINEAGE_BASE_COMMIT,
    )
    if generation == 1:
        _seed_model(int(config["model_init_seed"]))
        model = Torus9CurrentGraphNet().to(device)
        state = adapter.create_state(model, run_id=lineage_id)
        m0 = root / "checkpoints" / "M0.pt"
        if not m0.exists():
            metadata = adapter.save_initial_checkpoint(
                m0,
                state,
                run_id=lineage_id,
                label="M0",
                completed_games=0,
                device=device,
                code_identity=code_identity,
            )
        else:
            metadata = _read_json(m0.with_suffix(".metadata.json"))
            torus9_load_checkpoint(
                m0,
                model=model,
                expected={
                    "model_hash": metadata["model_hash"],
                    "profile_id": TORUS9_CURRENT_PROFILE_ID,
                    "profile_fingerprint": adapter.profile_fingerprint,
                },
                device=device,
            )
        state.parent_checkpoint_identity = {
            "label": "M0",
            "path": str(m0),
            "metadata_path": str(m0.with_suffix(".metadata.json")),
            "model_hash": metadata["model_hash"],
            "artifact_sha256": file_sha256(m0),
        }
        if isinstance(timing, dict):
            timing["restore_previous_state_wall_time_sec"] = 0.0
            timing["replay_validation_mode"] = "initial-empty"
        return adapter, state, m0

    local_checkpoint = root / "checkpoints" / f"M{generation - 1}.pt"
    local_replay = root / "replay" / f"rolling-after-{generation - 1:02d}.jsonl"
    previous_checkpoint = local_checkpoint
    previous_replay = local_replay
    parent_reference: Mapping[str, object] | None = None
    parent_replay_paths: tuple[Path, ...] = ()
    manifest = _read_json(root / "manifest.json")
    candidate = manifest.get("parent_checkpoint")
    if candidate is not None:
        if not isinstance(candidate, Mapping):
            raise ValueError("Parent checkpoint reference must be an object")
        try:
            parent_generation = int(candidate.get("generation", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Parent checkpoint generation is malformed") from exc
        if parent_generation == generation - 1:
            parent_reference = candidate

    if parent_reference is not None:
        parent_checkpoint_value = candidate.get("path")
        parent_replay_value = candidate.get("replay_path")
        if not parent_checkpoint_value or not parent_replay_value:
            raise ValueError("Parent checkpoint reference must include path and replay_path")
        previous_checkpoint = Path(str(parent_checkpoint_value)).resolve()
        previous_replay = Path(str(parent_replay_value)).resolve()
        parent_reference = candidate
        if not previous_checkpoint.is_file() or not previous_replay.is_file():
            raise FileNotFoundError("Referenced parent checkpoint or replay is missing")
        expected_checkpoint_sha = str(
            candidate.get("artifact_sha256") or candidate.get("sha256") or ""
        )
        if expected_checkpoint_sha and file_sha256(previous_checkpoint) != expected_checkpoint_sha:
            raise ValueError("Referenced parent checkpoint SHA-256 mismatch")
        expected_metadata_sha = str(candidate.get("metadata_sha256") or "")
        metadata_path = previous_checkpoint.with_suffix(".metadata.json")
        if expected_metadata_sha and file_sha256(metadata_path) != expected_metadata_sha:
            raise ValueError("Referenced parent checkpoint metadata SHA-256 mismatch")
        expected_replay_sha = str(candidate.get("replay_sha256") or "")
        if expected_replay_sha and file_sha256(previous_replay) != expected_replay_sha:
            raise ValueError("Referenced parent replay SHA-256 mismatch")
        # Bootstrap a full rolling-6 window from immutable parent fresh-replay
        # artifacts. These are references only; no parent dataset is copied.
        parent_root = previous_checkpoint.parents[1]
        discovered = _parent_replay_reference_paths(parent_root, generation)
        if all(path.is_file() for path in discovered):
            parent_replay_paths = discovered
            existing_references = {
                int(item["generation"]): str(item.get("sha256", ""))
                for item in candidate.get("replay_references", [])
                if isinstance(item, Mapping) and item.get("generation") is not None
            }
            replay_references = []
            for value, path in zip(range(max(1, generation - 6), generation), discovered):
                digest = file_sha256(path)
                expected = existing_references.get(value)
                if expected and expected != digest:
                    raise ValueError(
                        f"Referenced parent fresh replay M{value} SHA-256 mismatch"
                    )
                replay_references.append(
                    {"generation": value, "path": str(path), "sha256": digest}
                )
            manifest["parent_checkpoint"]["replay_references"] = replay_references
            _atomic_json(root / "manifest.json", manifest)
    elif not local_checkpoint.is_file() or not local_replay.is_file():
        if isinstance(candidate, Mapping):
            raise ValueError(
                "Parent checkpoint is not the immediately preceding generation: "
                f"parent=M{int(candidate.get('generation', -1))}, requested=M{generation}"
            )
        raise FileNotFoundError(
            f"Cannot start M{generation}; previous committed checkpoint/replay is missing"
        )
    previous_summary = (
        root / f"iter-{generation - 1:02d}-summary.json"
        if parent_reference is None
        else None
    )
    evictions = 0
    if previous_summary is not None and previous_summary.is_file():
        replay_metrics = _read_json(previous_summary).get("replay")
        if isinstance(replay_metrics, Mapping):
            evictions = int(replay_metrics.get("total_evictions", 0))
    if parent_reference is not None:
        evictions = int(parent_reference.get("total_evictions", 0))
    load_timing = timing if isinstance(timing, dict) else None
    replay_identity: Mapping[str, object] | None = None
    catalog_path = root / "runtime" / "artifact-catalog.json"
    catalog: ArtifactCatalog | None = None
    if parent_reference is not None and not parent_replay_paths:
        replay_identity = {
            "sha256": parent_reference.get("replay_sha256"),
            "size_bytes": previous_replay.stat().st_size,
            "row_count": parent_reference.get("replay_row_count"),
            "canonical_replay_fingerprint": parent_reference.get("replay_fingerprint"),
            "validation_schema": parent_reference.get(
                "replay_validation_schema", ARTIFACT_VALIDATION_SCHEMA
            ),
        }
    elif catalog_path.is_file() and previous_replay.resolve().is_relative_to(root.resolve()):
        catalog = ArtifactCatalog.load(catalog_path, root=root)
        replay_identity = catalog.identity(_relative(root, previous_replay))
    previous_metadata = _read_json(previous_checkpoint.with_suffix(".metadata.json"))
    if replay_identity is not None:
        enriched_identity = dict(replay_identity)
        generation_identities = _resume_replay_generation_identities(
            root=root,
            metadata=previous_metadata,
            base_identity=replay_identity,
            catalog=catalog,
            parent_reference=(
                parent_reference
                if parent_reference is not None
                else (candidate if isinstance(candidate, Mapping) else None)
            ),
        )
        if generation_identities is not None:
            enriched_identity["generation_identities"] = generation_identities
        for key in ("replay_identity_schema", "replay_identity_contract"):
            if previous_metadata.get(key) is not None:
                enriched_identity[key] = previous_metadata[key]
        replay_identity = enriched_identity
    restore_started = time.perf_counter()
    state = adapter.load_state(
        previous_checkpoint,
        replay_path=previous_replay,
        replay_paths=parent_replay_paths or None,
        device=device,
        # The external canonical M17 predates the Stage-3 metadata extension.
        # Its checkpoint/replay/profile identities were validated above, so
        # permit that legacy metadata shape without weakening child lineage
        # checkpoint validation or replay artifact hash checks.
        allow_reference=parent_reference is not None,
        total_evictions=evictions,
        replay_artifact_identity=replay_identity,
        load_timing=load_timing,
    )
    if isinstance(load_timing, dict):
        load_timing["restore_previous_state_wall_time_sec"] = time.perf_counter() - restore_started
        load_timing["replay_validation_mode"] = "catalog-evidence" if replay_identity is not None else "cold-full-validation"
    return adapter, state, previous_checkpoint


def _parent_replay_reference_paths(parent_root: Path, generation: int) -> tuple[Path, ...]:
    """Return the exact six pre-fork fresh replay generations for M48+."""
    first = max(1, int(generation) - 6)
    return tuple(
        parent_root / "replay" / f"iter-{value:02d}-fresh.jsonl"
        for value in range(first, int(generation))
    )


def _generation_paths(root: Path, generation: int) -> tuple[Path, ...]:
    return (
        root / "selfplay" / f"iter-{generation:02d}-games.jsonl",
        root / "replay" / f"iter-{generation:02d}-fresh.jsonl",
        root / "replay" / f"rolling-after-{generation:02d}.jsonl",
        root / "checkpoints" / f"M{generation}.pt",
        root / "checkpoints" / f"M{generation}.metadata.json",
        root / "training" / f"iter-{generation:02d}.json",
        root / f"iter-{generation:02d}-summary.json",
    )


def _cleanup_uncommitted_generation(root: Path, generation: int) -> None:
    marker = root / f"generation-{generation:02d}.complete.json"
    if marker.exists():
        return
    for path in _generation_paths(root, generation):
        path.unlink(missing_ok=True)
    for pattern in (
        f"replay/.iter-{generation:02d}.tmp-*",
        f"replay/.rolling-after-{generation:02d}.tmp*",
        f"checkpoints/.M{generation}.tmp*",
        f"training/.iter-{generation:02d}.tmp*",
        f".iter-{generation:02d}-summary.tmp*",
        f".generation-{generation:02d}.complete.tmp*",
    ):
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _local_checkpoint_reference(
    *,
    root: Path,
    lineage_id: str,
    generation: int,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Describe a checkpoint owned by the current lineage."""
    relative = f"checkpoints/M{int(generation)}.pt"
    path = (root / relative).resolve()
    checkpoint_hashes = manifest.get("checkpoint_hashes")
    digest: object | None = None
    if isinstance(checkpoint_hashes, Mapping):
        digest = checkpoint_hashes.get(relative)
    if digest is None:
        catalog_path = root / "runtime" / "artifact-catalog.json"
        if catalog_path.is_file():
            catalog = ArtifactCatalog.load(catalog_path, root=root)
            digest = catalog.identity(relative).get("sha256")
    if digest is None and path.is_file():
        digest = file_sha256(path)
    return {
        "topology": "torus9",
        "lineage_id": lineage_id,
        "checkpoint_id": f"M{int(generation)}",
        "label": f"M{int(generation)}",
        "generation": int(generation),
        "path": str(path),
        "sha256": str(digest or ""),
        "artifact_sha256": str(digest or ""),
        "source_type": "lineage-owned",
    }


def _arena_checkpoint_references(
    *,
    root: Path,
    lineage_id: str,
    generation: int,
    reference_generation: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return independently resolvable candidate/reference declarations."""
    manifest = _read_json(root / "manifest.json")
    candidate = _local_checkpoint_reference(
        root=root,
        lineage_id=lineage_id,
        generation=generation,
        manifest=manifest,
    )
    parent = manifest.get("parent_checkpoint")
    if isinstance(parent, Mapping) and int(parent.get("generation", -1)) == int(reference_generation):
        reference = dict(parent)
        reference.setdefault("topology", "torus9")
        reference.setdefault("checkpoint_id", reference.get("label", f"M{reference_generation}"))
        reference.setdefault("source_type", "external-parent")
    else:
        reference = _local_checkpoint_reference(
            root=root,
            lineage_id=lineage_id,
            generation=reference_generation,
            manifest=manifest,
        )
    return candidate, reference


def _artifact(root: Path, path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(root, path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_with_known_identity(
    root: Path,
    path: Path,
    identity: Mapping[str, object] | None,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Use a commit-record identity when the file was already hashed."""
    if identity is None:
        result = _artifact(root, path)
    else:
        if not path.is_file():
            raise FileNotFoundError(path)
        size = int(identity.get("size_bytes", -1))
        if size != path.stat().st_size:
            raise ValueError(f"Cached artifact size mismatch: {path}")
        result = {
            "path": _relative(root, path),
            "sha256": str(identity["sha256"]),
            "size_bytes": size,
        }
    if extra:
        result.update(dict(extra))
    return result


def _resume_state(
    *,
    root: Path,
    generation: int,
    checkpoint: Path,
    replay: Path,
    loaded_state: Any,
    config: Mapping[str, object],
    artifact_identities: Mapping[str, Mapping[str, object]] | None = None,
) -> Path:
    path = root / "runtime" / "resume" / f"generation-{generation:04d}.json"
    payload = {
        "schema": "torus9-orchestrator-resume-v2",
        "generation": generation,
        "components": ["model", "optimizer", "replay", "generation", "rng"],
        "checkpoint": _artifact_with_known_identity(
            root,
            checkpoint,
            artifact_identities.get(_relative(root, checkpoint)) if artifact_identities else None,
        ),
        "replay": _artifact_with_known_identity(
            root,
            replay,
            artifact_identities.get(_relative(root, replay)) if artifact_identities else None,
        ),
        "optimizer_updates": int(loaded_state.optimizer_updates),
        "samples_consumed": int(loaded_state.samples_consumed),
        "replay_last_generation": int(loaded_state.rolling_replay.last_generation),
        "rng": {
            "model_init_seed": int(config["model_init_seed"]),
            "selfplay_master_seed": int(config["selfplay_master_seed"]),
            "training_master_seed": int(config["training_master_seed"]),
            "training_seed": derive_seed(
                int(config["training_master_seed"]),
                os.environ["AZ_LINEAGE_ID"],
                "training",
                generation,
            ),
            "policy": "all game/search/training randomness derives from explicit immutable run-spec seeds",
        },
    }
    _atomic_json(path, payload)
    return path


def _generation_metrics(
    *, selfplay_metrics: Mapping[str, object], training: Mapping[str, object]
) -> dict[str, object]:
    training_wall = float(training.get("training_wall_time_sec", 0.0))
    optimizer_steps = int(training.get("optimizer_steps", 0))
    metrics = dict(selfplay_metrics)
    timing: dict[str, object] = {}
    selfplay_timing = selfplay_metrics.get("timing")
    if isinstance(selfplay_timing, Mapping):
        timing.update(dict(selfplay_timing))
    training_timing = training.get("phase_timing")
    if isinstance(training_timing, Mapping):
        normalized_training_timing = dict(training_timing)
        # Keep the canonical phase timing name alongside the historical
        # adapter name so required-metric paths remain stable across reports.
        normalized_training_timing.setdefault(
            "training_wall_time_sec", training_wall
        )
        timing["training"] = normalized_training_timing
    metrics.update(
        {
            "training_time_sec": training_wall,
            "optimizer_updates_per_sec": (
                optimizer_steps / training_wall if training_wall else 0.0
            ),
            "loss": {
                "policy": float(training.get("mean_policy_loss", 0.0)),
                "value": float(training.get("mean_value_loss", 0.0)),
                "ownership": float(training.get("mean_ownership_loss", 0.0)),
                "score": float(training.get("mean_score_loss_normalized", 0.0)),
                "total": float(training.get("mean_total_loss", 0.0)),
            },
            "learning": {
                "optimizer_updates_total": int(training.get("optimizer_updates_total", 0)),
                "samples_consumed_total": int(training.get("samples_consumed_total", 0)),
                "mean_parameter_delta": float(training.get("mean_parameter_delta", 0.0)),
                "mean_gradient_norm": float(training.get("mean_gradient_norm", 0.0)),
            },
            "timing": timing,
        }
    )
    return metrics


def _publish_generation_result(
    *,
    root: Path,
    generation: int,
    profile_fingerprint: str,
    selfplay_metrics: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    checkpoint = root / "checkpoints" / f"M{generation}.pt"
    replay = root / "replay" / f"rolling-after-{generation:02d}.jsonl"
    summary_path = root / f"iter-{generation:02d}-summary.json"
    marker = root / f"generation-{generation:02d}.complete.json"
    selfplay_path = root / "selfplay" / f"iter-{generation:02d}-games.jsonl"
    profile_path = Path(os.environ["AZ_PROFILE_PATH"]).resolve()
    marker_payload = _read_json(marker)
    cached_identities: dict[str, Mapping[str, object]] = {}
    marker_fields = {
        "checkpoint": "checkpoint_sha256",
        "checkpoint_metadata": "checkpoint_metadata_sha256",
        "replay": "rolling_replay_sha256",
        "fresh_replay": "fresh_replay_sha256",
        "training": "training_metrics_sha256",
        "summary": "summary_sha256",
        "selfplay": "selfplay_artifact_sha256",
    }
    marker_paths = {
        "checkpoint": checkpoint,
        "checkpoint_metadata": checkpoint.with_suffix(".metadata.json"),
        "replay": replay,
        "fresh_replay": root / "replay" / f"iter-{generation:02d}-fresh.jsonl",
        "training": root / "training" / f"iter-{generation:02d}.json",
        "summary": summary_path,
        "selfplay": selfplay_path,
    }
    for name, field in marker_fields.items():
        path = marker_paths[name]
        if field in marker_payload and path.is_file():
            cached_identities[_relative(root, path)] = {
                "sha256": marker_payload[field],
                "size_bytes": path.stat().st_size,
            }
    replay_identity: dict[str, object] = {
        **cached_identities.get(_relative(root, replay), {}),
        "row_count": marker_payload.get("replay_row_count"),
        "canonical_replay_fingerprint": marker_payload.get("replay_fingerprint"),
        "validation_schema": marker_payload.get("validation_schema", ARTIFACT_VALIDATION_SCHEMA),
    }
    if marker_payload.get("replay_identity_schema") is not None:
        replay_identity["replay_identity_schema"] = marker_payload["replay_identity_schema"]
    if marker_payload.get("replay_identity_components") is not None:
        replay_identity["generation_identities"] = marker_payload["replay_identity_components"]
    if marker_payload.get("replay_identity_contract") is not None:
        replay_identity["replay_identity_contract"] = marker_payload["replay_identity_contract"]
    adapter = Torus9TrainingAdapter(profile=load_torus9_current_profile(profile_path))
    reload_timing: dict[str, object] = {}
    loaded = adapter.load_state(
        checkpoint,
        replay_path=replay,
        device="cpu",
        replay_artifact_identity=replay_identity,
        load_timing=reload_timing,
    )
    if int(loaded.current_generation) != generation:
        raise ValueError("Reloaded Torus9 checkpoint generation mismatch")
    resume = _resume_state(
        root=root,
        generation=generation,
        checkpoint=checkpoint,
        replay=replay,
        loaded_state=loaded,
        config=config,
        artifact_identities=cached_identities,
    )
    summary = _read_json(summary_path)
    training = summary.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Training summary is missing training metrics")
    expected_selfplay_hash = selfplay_metrics.get("selfplay_artifact_sha256")
    cached_selfplay = cached_identities.get(_relative(root, selfplay_path))
    if cached_selfplay is not None:
        if expected_selfplay_hash != cached_selfplay.get("sha256"):
            raise ValueError("Persisted self-play artifact hash disagrees with commit marker")
        if int(cached_selfplay.get("size_bytes", -1)) != selfplay_path.stat().st_size:
            raise ValueError("Persisted self-play artifact size disagrees with commit marker")
    elif expected_selfplay_hash != file_sha256(selfplay_path):
        # Backward-compatible recovery for markers produced before the
        # self-play identity was added; new commits always take the durable
        # identity path above.
        raise ValueError("Persisted self-play artifact hash disagrees with committed summary")
    artifact_paths = (
        checkpoint,
        checkpoint.with_suffix(".metadata.json"),
        replay,
        root / "replay" / f"iter-{generation:02d}-fresh.jsonl",
        root / "training" / f"iter-{generation:02d}.json",
        summary_path,
        marker,
        selfplay_path,
        resume,
    )
    artifacts = []
    for path in artifact_paths:
        extra: Mapping[str, object] | None = None
        if path == replay:
            extra = {
                "row_count": marker_payload.get("replay_row_count", len(loaded.rolling_replay.rows)),
                "source_generations": marker_payload.get("replay_generations", []),
                "canonical_replay_fingerprint": marker_payload.get("replay_fingerprint"),
                "validation_schema": marker_payload.get("validation_schema", ARTIFACT_VALIDATION_SCHEMA),
            }
            if marker_payload.get("replay_identity_schema") is not None:
                extra = {
                    **extra,
                    "replay_identity_schema": marker_payload["replay_identity_schema"],
                }
            if marker_payload.get("replay_identity_components") is not None:
                extra = {
                    **extra,
                    "generation_identities": marker_payload["replay_identity_components"],
                }
            if marker_payload.get("replay_identity_contract") is not None:
                extra = {
                    **extra,
                    "replay_identity_contract": marker_payload["replay_identity_contract"],
                }
        known = cached_identities.get(_relative(root, path))
        if path == selfplay_path and known is None:
            known = {
                "sha256": selfplay_metrics.get("selfplay_artifact_sha256"),
                "size_bytes": path.stat().st_size,
            }
        artifacts.append(_artifact_with_known_identity(root, path, known, extra=extra))
    if generation == 1:
        initial_checkpoint = root / "checkpoints" / "M0.pt"
        initial_metadata = initial_checkpoint.with_suffix(".metadata.json")
        artifacts.extend((_artifact(root, initial_checkpoint), _artifact(root, initial_metadata)))
    payload = {
        "schema": GENERATION_RESULT_SCHEMA,
        "generation": generation,
        "status": "COMPLETED",
        "profile_fingerprint": profile_fingerprint,
        "checkpoint_reload_verified": True,
        "checkpoint": {"path": _relative(root, checkpoint)},
        "replay": {"path": _relative(root, replay)},
        "resume_state": {
            "path": _relative(root, resume),
            "components": ["model", "optimizer", "replay", "generation", "rng"],
        },
        "artifacts": artifacts,
        "technical_games": int(selfplay_metrics.get("technical_games", 0)),
        "invalid_games": int(selfplay_metrics.get("invalid_games", 0)),
        "metrics": _generation_metrics(
            selfplay_metrics=selfplay_metrics, training=training
        ),
    }
    publication_started = time.perf_counter()
    result_path = Path(os.environ["AZ_GENERATION_RESULT_PATH"])
    _atomic_json(result_path, payload)
    publication_elapsed = time.perf_counter() - publication_started
    metrics_payload = payload.get("metrics")
    if isinstance(metrics_payload, dict):
        timing_payload = metrics_payload.get("timing")
        if isinstance(timing_payload, dict):
            timing_payload["result_publication_wall_time_sec"] = publication_elapsed
            # The result is intentionally tiny; persist the final measured
            # publication timing without touching replay/checkpoint artifacts.
            _atomic_json(result_path, payload)
    return payload


def run_generation(args: argparse.Namespace) -> dict[str, object]:
    spec = _load_run_spec()
    config = _generation_config(spec)
    _validate_device(str(config["device"]))
    root, lineage_id, profile_path, expected_fingerprint, code = _environment(args.generation)
    profile = _load_profile(profile_path, expected_fingerprint)
    _validate_scientific_bindings(profile, config)
    heartbeat_path = Path(os.environ["AZ_DRIVER_HEARTBEAT_PATH"])
    marker = root / f"generation-{args.generation:02d}.complete.json"

    with _Heartbeat(
        heartbeat_path,
        args.generation,
        interval=float(config["heartbeat_interval_seconds"]),
    ) as heartbeat:
        manifest = _read_json(root / "manifest.json")
        if marker.is_file():
            heartbeat.advance("recover-published-generation")
            summary = _read_json(root / f"iter-{args.generation:02d}-summary.json")
            persisted = summary.get("orchestrator_selfplay")
            if not isinstance(persisted, Mapping):
                raise ValueError("Committed generation lacks orchestrator self-play metrics")
            return _publish_generation_result(
                root=root,
                generation=args.generation,
                profile_fingerprint=expected_fingerprint,
                selfplay_metrics=persisted,
                config=config,
            )
        if args.resume:
            heartbeat.advance("cleanup-uncommitted-generation")
            _cleanup_uncommitted_generation(root, args.generation)
            Path(os.environ["AZ_GENERATION_RESULT_PATH"]).unlink(missing_ok=True)
            (root / "runtime" / "resume" / f"generation-{args.generation:04d}.json").unlink(missing_ok=True)
        elif any(path.exists() for path in _generation_paths(root, args.generation)):
            raise FileExistsError("Generation has uncommitted artifacts; use orchestrator resume")

        heartbeat.advance("load-previous-state", subphase="restore")
        generation_timing: dict[str, object] = {}
        restore_started = time.perf_counter()
        adapter, state, previous_checkpoint = _prepare_state(
            root=root,
            lineage_id=lineage_id,
            generation=args.generation,
            profile=profile,
            config=config,
            device=str(config["device"]),
            code_identity=code,
            timing=generation_timing,
        )
        generation_timing.setdefault(
            "restore_previous_state_wall_time_sec",
            time.perf_counter() - restore_started,
        )
        games = int(config["games"])
        game_ids = [
            f"{lineage_id}-generation-{args.generation:04d}-game-{index:04d}"
            for index in range(games)
        ]
        inference: dict[str, object] = {}
        heartbeat.advance(
            "self-play",
            token=f"self-play:0/{games} games",
            completed=0,
            total=games,
            unit="games",
            subphase="games",
        )

        def selfplay_progress(completed: int, total: int) -> None:
            heartbeat.advance(
                "self-play",
                token=f"self-play:{completed}/{total} games",
                completed=completed,
                total=total,
                unit="games",
                subphase="games",
            )

        started = time.perf_counter()
        records = run_torus9_selfplay_games(
            state.model,
            run_id=lineage_id,
            label=f"M{args.generation - 1}",
            artifact=file_sha256(previous_checkpoint),
            master_seed=int(config["selfplay_master_seed"]),
            profile_fp=expected_fingerprint,
            game_ids=game_ids,
            workers=int(config["workers"]),
            code_identity=code,
            device=str(config["device"]),
            contract=_contract(profile),
            coalescing=bool(config["coalescing"]),
            inference_batch_cap=int(config["inference_batch_cap"]),
            inference_batch_wait_ms=float(config["inference_batch_wait_ms"]),
            active_games_per_worker=int(config["active_games_per_worker"]),
            total_active_contexts=int(config["total_active_contexts"]),
            inference_telemetry=inference,
            execution_activity=inference,
            execution_override_reason="immutable production run-spec",
            execution_reference_interactive=False,
            progress_callback=selfplay_progress,
        )
        selfplay_wall = time.perf_counter() - started
        generation_timing["self_play_wall_time_sec"] = selfplay_wall
        if len(records) != games or {record.game_id for record in records} != set(game_ids):
            raise RuntimeError("Torus9 self-play did not return exactly the requested game set")
        technical_games = sum(record.technical_termination is not None for record in records)
        if technical_games:
            raise RuntimeError(f"Torus9 self-play produced {technical_games} technical games")
        for record in records:
            record.validate()
        selfplay_path = root / "selfplay" / f"iter-{args.generation:02d}-games.jsonl"
        _atomic_jsonl(selfplay_path, [record.to_dict() for record in records])
        moves = sum(len(record.final_action_trace) for record in records)
        selfplay_metrics = {
            "games": games,
            "technical_games": 0,
            "invalid_games": 0,
            "moves": moves,
            "games_per_hour": games * 3600.0 / selfplay_wall if selfplay_wall else 0.0,
            "moves_per_sec": moves / selfplay_wall if selfplay_wall else 0.0,
            "selfplay_time_sec": selfplay_wall,
            "inference": {
                "mean_batch_rows": float(inference.get("mean_batch_rows", 0.0)),
                "p95_batch_rows": float(inference.get("p95_batch_rows", 0.0)),
                "max_batch_rows": int(inference.get("max_batch_rows", 0)),
                "rows_per_sec": float(inference.get("rows_per_sec", 0.0)),
            },
            "selfplay_artifact_path": _relative(root, selfplay_path),
            "selfplay_artifact_sha256": file_sha256(selfplay_path),
            "timing": generation_timing,
        }
        heartbeat.advance("self-play-complete", completed=games, total=games, unit="games")
        heartbeat.advance(
            "replay",
            token="replay:0/1 target-build",
            completed=0,
            total=1,
            unit="phase",
            subphase="target-build",
        )
        adapter.set_progress_callback(heartbeat.advance)
        adapter.set_diagnostic_timing(generation_timing)
        heartbeat.advance(
            "training",
            token=f"training:0/{TORUS9_OPTIMIZER_STEPS_PER_ITERATION} optimizer_steps",
            completed=0,
            total=TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
            unit="optimizer_steps",
            subphase="optimizer",
        )
        run_torus9_training_iteration(
            state=state,
            generation=args.generation,
            output_dir=root,
            run_id=lineage_id,
            records=records,
            training_seed=derive_seed(
                int(config["training_master_seed"]),
                lineage_id,
                "training",
                args.generation,
            ),
            completed_games=int(state.completed_games) + games,
            code_identity=code,
            device=str(config["device"]),
            adapter=adapter,
            summary_extra={"orchestrator_selfplay": selfplay_metrics},
            progress_callback=heartbeat.advance,
        )
        heartbeat.advance(
            "training",
            token=f"training:{TORUS9_OPTIMIZER_STEPS_PER_ITERATION}/{TORUS9_OPTIMIZER_STEPS_PER_ITERATION} optimizer_steps",
            completed=TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
            total=TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
            unit="optimizer_steps",
            subphase="optimizer",
        )
        heartbeat.advance("reload-verification")
        payload = _publish_generation_result(
            root=root,
            generation=args.generation,
            profile_fingerprint=expected_fingerprint,
            selfplay_metrics=selfplay_metrics,
            config=config,
        )
        heartbeat.advance("completed", token=f"generation-M{args.generation}-completed")
        return payload


def _training_snapshot(
    root: Path,
    generation: int | None = None,
    *,
    tracked_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Capture bounded Arena mutation evidence.

    The commit catalog already identifies immutable artifacts.  Arena only
    snapshots its transaction metadata, runtime state, and the explicitly
    selected candidate/reference files instead of walking all history.
    """
    paths = [
        root / "manifest.json",
        root / "runtime" / "state.json",
    ]
    if generation is not None:
        paths.append(root / "runtime" / "generations" / f"generation-{generation:04d}.json")
    paths.extend(Path(path) for path in tracked_paths)
    snapshot: dict[str, object] = {}
    for path in sorted({path.resolve() for path in paths}):
        if path.is_file():
            try:
                key = _relative(root, path)
            except ValueError:
                # External reference checkpoints are intentionally outside the
                # training lineage.  Snapshot them by canonical absolute path
                # so mutation checks remain bounded without pretending that
                # the file is lineage-owned.
                key = f"external::{path}"
            snapshot[key] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    catalog_path = root / "runtime" / "artifact-catalog.json"
    if catalog_path.is_file():
        catalog_stat = catalog_path.stat()
        # The catalog itself is already authenticated by its manifest
        # fingerprint and loaded once before Arena.  Record cheap filesystem
        # identity here without hashing its growing historical contents.
        snapshot[_relative(root, catalog_path)] = {
            "size_bytes": catalog_stat.st_size,
            "mtime_ns": catalog_stat.st_mtime_ns,
        }
    return snapshot


def _validate_existing_arena(
    *,
    output: Path,
    summary: Mapping[str, object],
    candidate: Path,
    reference: Path,
    config: Mapping[str, object],
    candidate_sha256: str | None = None,
    reference_sha256: str | None = None,
) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Existing Arena output has no manifest")
    manifest = _read_json(manifest_path)
    execution = summary.get("execution")
    expected_execution = config["execution"]
    if int(summary.get("technical_games", 0)) != 0 or int(summary.get("invalid_games", 0)) != 0:
        raise ValueError("Existing Arena technical/invalid outcomes are fail-closed")
    if (
        summary.get("candidate_artifact_sha256")
        != (candidate_sha256 or file_sha256(candidate))
        or summary.get("reference_artifact_sha256")
        != (reference_sha256 or file_sha256(reference))
        or int(summary.get("games", -1)) != int(config["games"])
        or summary.get("arena_profile") != "torus9"
        or int(manifest.get("master_seed", -1)) != int(config["master_seed"])
        or not isinstance(execution, Mapping)
        or dict(execution) != {
            key: expected_execution[key]
            for key in (
                "workers",
                "games_per_worker",
                "inference_batch_rows",
                "inference_batch_wait_ms",
                "device",
                "strict_production",
            )
        }
    ):
        raise ValueError("Existing Arena output does not match immutable run-spec policy")


def _apply_arena_performance_policy(
    summary: Mapping[str, object],
    config: ArenaExecutionConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    """Reclassify a completed Arena summary under the current gate policy.

    This is intentionally used for both freshly produced and already complete
    summaries.  It lets recovery commit a finished Arena without replaying
    games after a policy-only change.
    """
    normalized = dict(summary)
    telemetry_value = normalized.get("telemetry")
    if not isinstance(telemetry_value, Mapping):
        raise ValueError("Arena summary requires telemetry for performance policy validation")
    telemetry = dict(telemetry_value)
    mean_batch = float(telemetry.get("mean_inference_batch_rows", 0.0))
    policy = classify_arena_performance(mean_batch, config)
    prior_failures = telemetry.get("performance_failures", [])
    prior_warnings = telemetry.get("performance_warnings", [])
    hard_failures = [
        str(value)
        for value in prior_failures
        if str(value) != "mean_inference_batch_rows"
    ]
    hard_failures.extend(str(value) for value in policy["hard_failures"])
    warnings = [
        str(value)
        for value in prior_warnings
        if str(value) != "mean_inference_batch_rows"
    ]
    warnings.extend(str(value) for value in policy["warnings"])
    status = (
        "CRITICAL"
        if hard_failures
        else (
            str(policy["status"])
            if str(policy["status"]) == "SEVERE_WARNING"
            else ("WARNING" if warnings else "HEALTHY")
        )
    )
    telemetry["performance_status"] = status
    telemetry["performance_failures"] = sorted(set(hard_failures))
    telemetry["performance_warnings"] = sorted(set(warnings))
    telemetry["performance_gate"] = {
        "mean_inference_batch_rows": mean_batch,
        "severe_warning_threshold": policy["severe_warning_threshold"],
        "healthy_minimum": policy["healthy_minimum"],
    }
    normalized["telemetry"] = telemetry
    return normalized, policy


def run_arena(args: argparse.Namespace) -> dict[str, object]:
    spec = _load_run_spec()
    config, startset = _arena_config(spec)
    execution = _mapping(config["execution"], "arena.execution")
    _validate_device(str(execution["device"]))
    root, lineage_id, profile_path, expected_fingerprint, _code = _environment(args.generation)
    _load_profile(profile_path, expected_fingerprint)
    reference_gap = int(config["reference_gap"])
    if args.generation < reference_gap:
        raise ValueError("Arena generation/reference gap is invalid")
    reference_generation = int(args.generation) - reference_gap
    candidate_declared, reference_declared = _arena_checkpoint_references(
        root=root,
        lineage_id=lineage_id,
        generation=int(args.generation),
        reference_generation=reference_generation,
    )
    runs_root = root.parents[2]
    candidate_identity = resolve_checkpoint(
        candidate_declared,
        topology="torus9",
        runs_root=runs_root,
    )
    reference_identity = resolve_checkpoint(
        reference_declared,
        topology="torus9",
        runs_root=runs_root,
    )
    candidate = candidate_identity.path
    reference = reference_identity.path
    catalog_path = root / "runtime" / "artifact-catalog.json"
    if not catalog_path.is_file():
        raise ValueError("Torus9 Arena requires the committed artifact catalog")
    catalog = ArtifactCatalog.load(catalog_path, root=root)
    candidate_relative = _relative(root, candidate)
    verified_artifacts = catalog.verify((candidate_relative,))
    candidate_sha256 = verified_artifacts[candidate_relative]
    if reference_identity.lineage_id == lineage_id:
        reference_relative = _relative(root, reference)
        verified_artifacts.update(catalog.verify((reference_relative,)))
        reference_sha256 = verified_artifacts[reference_relative]
    else:
        reference_relative = None
        reference_sha256 = reference_identity.sha256

    heartbeat_path = Path(os.environ["AZ_DRIVER_HEARTBEAT_PATH"])
    with _Heartbeat(
        heartbeat_path,
        args.generation,
        interval=float(config["heartbeat_interval_seconds"]),
    ) as heartbeat:
        heartbeat.advance(
            "arena-snapshot",
            completed=0,
            total=1,
            unit="phase",
            subphase="snapshot",
        )
        before = _training_snapshot(
            root,
            args.generation,
            tracked_paths=(candidate, reference),
        )
        result_path = Path(os.environ["AZ_ARENA_RESULT_PATH"])
        output = result_path.parent.resolve()
        if reference_identity.lineage_id != lineage_id:
            ensure_evaluation_layout(output)
        summary_path = output / "summary.json"
        if output.exists() and not summary_path.is_file():
            # Keep the orchestrator-owned directory but remove only incomplete
            # Arena engine payloads; never touch checkpoints/replay/training.
            for child in output.iterdir():
                if child == result_path:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        output.mkdir(parents=True, exist_ok=True)
        arena_execution = ArenaExecutionConfig(
            games=int(config["games"]),
            workers=int(execution["workers"]),
            games_per_worker=int(execution["games_per_worker"]),
            inference_batch_rows=int(execution["inference_batch_rows"]),
            inference_batch_wait_ms=float(execution["inference_batch_wait_ms"]),
            device=str(execution["device"]),
            strict_production=bool(execution["strict_production"]),
            min_mean_inference_batch_rows=float(execution["min_mean_inference_batch_rows"]),
            min_effective_cpu_cores=float(execution["min_effective_cpu_cores"]),
            early_gate_enabled=bool(execution["early_gate_enabled"]),
            early_gate_min_forwards=int(execution["early_gate_min_forwards"]),
            early_gate_min_wall_sec=float(execution["early_gate_min_wall_sec"]),
        )
        if summary_path.is_file():
            summary = _read_json(summary_path)
            _validate_existing_arena(
                output=output,
                summary=summary,
                candidate=candidate,
                reference=reference,
                config=config,
                candidate_sha256=candidate_sha256,
                reference_sha256=reference_sha256,
            )
            summary, performance_policy = _apply_arena_performance_policy(summary, arena_execution)
            _atomic_json(summary_path, summary)
        else:
            heartbeat.advance(
                "arena",
                token=f"arena:0/{int(config['games'])} games",
                completed=0,
                total=int(config["games"]),
                unit="games",
                subphase="games",
            )

            def arena_progress(completed: int, total: int) -> None:
                heartbeat.advance(
                    "arena",
                    token=f"arena:{completed}/{total} games",
                    completed=completed,
                    total=total,
                    unit="games",
                    subphase="games",
                )

            summary = run_arena_engine(
                profile=TORUS9_ARENA_PROFILE,
                candidate_path=candidate,
                reference_path=reference,
                output_dir=output,
                candidate_label=f"M{args.generation}",
                reference_label=f"M{reference_generation}",
                run_id=f"{lineage_id}-periodic-M{args.generation:04d}-vs-M{reference_generation:04d}",
                comparison=f"periodic-M{args.generation}-vs-M{reference_generation}",
                master_seed=int(config["master_seed"]),
                config=arena_execution,
                expected_candidate_artifact_sha256=candidate_sha256,
                expected_reference_artifact_sha256=reference_sha256,
                expected_candidate_model_hash=candidate_declared.get("model_hash"),
                expected_reference_model_hash=reference_declared.get("model_hash"),
                progress_callback=arena_progress,
            )
            summary, performance_policy = _apply_arena_performance_policy(summary, arena_execution)
            _atomic_json(summary_path, summary)
        if performance_policy["status"] in {"WARNING", "SEVERE_WARNING"}:
            _atomic_json(
                output / "performance-warning.json",
                {
                    "status": str(performance_policy["status"]),
                    "reasons": list(performance_policy["warnings"]),
                    "performance_gate": dict(performance_policy),
                    "summary": str(summary_path),
                },
            )
        heartbeat.advance("arena-verify")
        # Recheck the selected immutable artifacts after Arena.  Unrelated
        # ancient files are intentionally outside this bounded operation.
        if reference_relative is None:
            if file_sha256(reference) != reference_sha256:
                raise ValueError(
                    "External Arena reference checkpoint changed during evaluation: "
                    f"{reference}"
                )
        else:
            catalog.verify((candidate_relative, reference_relative))
        after = _training_snapshot(
            root,
            args.generation,
            tracked_paths=(candidate, reference),
        )
        if before != after:
            raise ValueError("Arena mutated lineage-owned training metadata or checkpoints")
        games = int(summary["games"])
        wins = int(summary["wins"])
        draws = int(summary["draws"])
        telemetry = summary.get("telemetry")
        telemetry_map = telemetry if isinstance(telemetry, Mapping) else {}
        arena_block = _mapping(spec.payload["arena"], "arena")
        cross_lineage = reference_identity.lineage_id != candidate_identity.lineage_id
        comparison = f"periodic-M{args.generation}-vs-M{reference_generation}"
        evaluation_id = (
            evaluation_id_for_comparison(
                candidate_lineage_id=candidate_identity.lineage_id,
                candidate_generation=candidate_identity.generation or int(args.generation),
                reference_lineage_id=reference_identity.lineage_id,
                reference_generation=reference_identity.generation,
            )
            if cross_lineage
            else None
        )
        provenance = {
            "schema": "gocube-checkpoint-evaluation-provenance-v1",
            "evaluation_id": evaluation_id,
            "comparison": comparison,
            "candidate": candidate_identity.as_reference(),
            "reference": reference_identity.as_reference(),
            "arena_config": dict(arena_block["driver_config"]),
            "startset": dict(startset),
            "startset_fingerprint": run_spec_fingerprint(startset),
            "master_seed": int(config["master_seed"]),
            "profile_fingerprint": expected_fingerprint,
            "performance_policy": dict(performance_policy),
            "training_mutated": False,
        }
        _atomic_json(output / "provenance.json", provenance)
        engine_manifest_path = output / "manifest.json"
        if engine_manifest_path.is_file():
            engine_manifest = _read_json(engine_manifest_path)
            engine_manifest["evaluation_id"] = evaluation_id
            engine_manifest["checkpoint_references"] = {
                "candidate": candidate_identity.as_reference(),
                "reference": reference_identity.as_reference(),
            }
            engine_manifest["provenance"] = str(output / "provenance.json")
            _atomic_json(engine_manifest_path, engine_manifest)
        payload = {
            "schema": ARENA_RESULT_SCHEMA,
            "generation": int(args.generation),
            "status": "COMPLETED",
            "profile_fingerprint": expected_fingerprint,
            "technical_games": int(summary["technical_games"]),
            "invalid_games": 0,
            "training_mutated": False,
            "preset_fingerprint": run_spec_fingerprint(
                _mapping(arena_block["driver_config"], "arena.driver_config")
            ),
            "startset_fingerprint": run_spec_fingerprint(startset),
            "evaluation_output": str(output),
            "evaluation_id": evaluation_id,
            "candidate_reference": candidate_identity.as_reference(),
            "reference_reference": reference_identity.as_reference(),
            "candidate_generation": int(args.generation),
            "reference_generation": reference_generation,
            "metrics": {
                "games": games,
                "wins": wins,
                "losses": int(summary["losses"]),
                "draws": draws,
                "win_rate": (wins + 0.5 * draws) / games if games else 0.0,
                "games_per_hour": float(telemetry_map.get("games_per_hour", 0.0)),
                "moves_per_sec": float(telemetry_map.get("moves_per_sec", 0.0)),
                "inference_mean_batch_rows": float(
                    telemetry_map.get("mean_inference_batch_rows", 0.0)
                ),
                "effective_cpu_cores": float(
                    telemetry_map.get("effective_cpu_cores", 0.0)
                ),
                "performance_status": str(telemetry_map.get("performance_status", "HEALTHY")),
                "performance_warnings": list(telemetry_map.get("performance_warnings", [])),
                "performance_gate": dict(performance_policy),
            },
        }
        _atomic_json(result_path, payload)
        heartbeat.advance("completed", token=f"arena-M{args.generation}-completed")
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generation = sub.add_parser("generation")
    generation.add_argument("--generation", type=int, required=True)
    generation.add_argument("--resume", action="store_true")
    generation.set_defaults(func=run_generation)
    arena = sub.add_parser("arena")
    arena.add_argument("--generation", type=int, required=True)
    arena.set_defaults(func=run_arena)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generation <= 0:
        raise SystemExit("--generation must be positive")
    result = args.func(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
