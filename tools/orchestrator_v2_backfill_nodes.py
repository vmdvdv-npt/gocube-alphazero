"""One-time materialization of the proven Torus9 legacy V2 metadata.

This is intentionally a small, fixed-scope backfill.  It does not discover
lineages, infer parent edges, inspect model/replay contents, or modify the
existing physical artifacts.  The output is ordinary Stage 1 metadata:
EffectiveConfig V2, migration provenance, and CheckpointNode V2.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Sequence

# Keep the requested ``.venv/bin/python tools/...`` invocation self-contained;
# Python otherwise puts only ``tools/`` on sys.path for a script entrypoint.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gocube_golden.artifact_catalog import ArtifactCatalog
from gocube_golden.orchestrator import atomic_write_json
from gocube_golden.orchestrator_v2.artifact_resolver import NODE_DIRECTORY
from gocube_golden.orchestrator_v2.contracts import (
    ArtifactRef,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
)
from gocube_golden.torus9_contract import current_torus9_content_fingerprint, profile_fingerprint


TOPOLOGY = "torus9"
CHECKPOINT_METADATA_DIR = Path("checkpoints")
PROVENANCE_DIRECTORY = Path("metadata") / "provenance-v2"
CONFIG_DIRECTORY = Path("metadata") / "effective-config-v2"
CHECKPOINT_ID_RE = re.compile(r"^M([0-9]+)$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
BACKFILL_PROVENANCE_SCHEMA = "gocube-orchestrator-v2-backfill-provenance-v1"


@dataclass(frozen=True)
class ScopeEntry:
    lineage_id: str
    first_generation: int
    last_generation: int

    def __post_init__(self) -> None:
        if self.first_generation < 0 or self.last_generation < self.first_generation:
            raise ValueError("invalid backfill generation range")

    @property
    def count(self) -> int:
        return self.last_generation - self.first_generation + 1


# This is the complete and deliberately non-discoverable migration scope.
TARGET_SCOPE: tuple[ScopeEntry, ...] = (
    ScopeEntry("torus9-golden-v3-production-20260917-m17", 18, 47),
    ScopeEntry("torus9-golden-v3-plateau-exit-m47-lr3e4-r6-40k-s128-20260917-v2", 48, 54),
    ScopeEntry("torus9-golden-v3-plateau-exit-m54-lr3e4-r6-40k-s128-20260917-v3", 55, 80),
    ScopeEntry("torus9-staged-cadence-m80-20260918-v1-g64", 81, 86),
    ScopeEntry("torus9-staged-cadence-m80-20260918-v1-g128", 81, 83),
    ScopeEntry("torus9-staged-cadence-m80-20260918-v1-g192", 81, 82),
    ScopeEntry("torus9-post-ab-m80-g128-20260918-v1", 84, 93),
)
TARGET_COUNT = sum(entry.count for entry in TARGET_SCOPE)


class BackfillError(ValueError):
    """A mandatory migration fact is absent or inconsistent."""


@dataclass(frozen=True)
class NodePlan:
    lineage_root: Path
    checkpoint: CheckpointRef
    node: CheckpointNode
    config_payload: dict[str, object]
    provenance_payload: dict[str, object]
    node_payload: dict[str, object]
    config_path: Path
    provenance_path: Path
    node_path: Path


@dataclass(frozen=True)
class BackfillReport:
    target_checkpoints: int
    ready: int
    conflicts: int
    writes: int
    problems: tuple[str, ...]


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BackfillError(f"{label} must be an object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackfillError(f"{label} is missing")
    return value


def _require_sha(value: object, label: str) -> str:
    text = _require_string(value, label)
    if not SHA256_RE.fullmatch(text):
        raise BackfillError(f"{label} is not a canonical SHA-256: {text!r}")
    return text


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackfillError(f"cannot read {label}: {path}") from exc
    return dict(_require_mapping(value, label))


def _safe_relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise BackfillError(f"{label} escapes its expected root: {path}") from exc


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    # Match gocube_golden.orchestrator.atomic_write_json exactly, so a dry-run
    # can calculate the future ArtifactRef SHA without writing the file.
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _json_sha(payload: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(payload)).hexdigest()


def _copy_json(value: object, label: str) -> dict[str, object]:
    return dict(_require_mapping(value, label))


def _merge(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = dict(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(_require_mapping(result[key], key), value)
        else:
            result[key] = value
    return result


def _load_profile(path: Path, cache: dict[Path, dict[str, object]]) -> dict[str, object]:
    path = path.resolve()
    if path in cache:
        return cache[path]
    raw = _read_json(path, "historical profile")
    base_ref = raw.get("base_profile_path")
    if base_ref is not None:
        base_path = (path.parent / _require_string(base_ref, "base_profile_path")).resolve()
        profile = _merge(_load_profile(base_path, cache), {k: v for k, v in raw.items() if k != "base_profile_path"})
        profile["content_fingerprint"] = current_torus9_content_fingerprint(profile)
        profile["profile_fingerprint"] = profile_fingerprint(profile)
    else:
        profile = dict(raw)
    cache[path] = profile
    return profile


def _profile_path(
    profile_ref: object,
    *,
    repo_root: Path,
    runs_root: Path,
) -> Path:
    reference = _require_string(profile_ref, "run-spec profile_path")
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate.resolve()
    candidates = ((repo_root / candidate).resolve(), (runs_root / candidate).resolve())
    for path in candidates:
        if path.is_file():
            return path
    raise BackfillError(f"historical profile is missing: {reference}")


def _generation_from_id(value: str, label: str) -> int:
    match = CHECKPOINT_ID_RE.fullmatch(value)
    if match is None:
        raise BackfillError(f"{label} must be an M<number> identity: {value!r}")
    return int(match.group(1))


def _path_parts_under_runs(path: Path, runs_root: Path, label: str) -> tuple[str, str, str, str]:
    relative = Path(_safe_relative(path, runs_root, label))
    parts = relative.parts
    if len(parts) != 5 or parts[1] not in {"active", "archive"} or parts[3] != "checkpoints":
        raise BackfillError(f"{label} is not a lineage checkpoint path: {path}")
    topology, namespace, lineage_id, _checkpoints, filename = parts
    if not filename.endswith(".pt"):
        raise BackfillError(f"{label} is not a checkpoint artifact path: {path}")
    return topology, namespace, lineage_id, filename


def _reference_path(
    identity: Mapping[str, object],
    *,
    runs_root: Path,
) -> tuple[Path, str, str]:
    raw_path = _require_string(identity.get("path"), "parent path")
    declared = Path(raw_path)
    if declared.is_absolute():
        path = declared.resolve()
    else:
        # Relative legacy references are accepted only with an explicit owner.
        owner = _require_string(identity.get("lineage_id"), "parent lineage_id")
        path = (runs_root / TOPOLOGY / "active" / owner / declared).resolve()
    topology, namespace, lineage_id, filename = _path_parts_under_runs(path, runs_root, "parent path")
    if topology != TOPOLOGY:
        raise BackfillError(f"parent topology is not {TOPOLOGY}: {topology}")
    declared_lineage = identity.get("lineage_id")
    if declared_lineage is not None and declared_lineage != lineage_id:
        raise BackfillError("parent lineage_id conflicts with its durable path")
    if filename != f"{_require_string(identity.get('label') or identity.get('checkpoint_id'), 'parent label')}.pt":
        raise BackfillError("parent label conflicts with its durable checkpoint path")
    return path, namespace, lineage_id


def _checkpoint_ref_from_explicit_parent(
    identity_value: object,
    *,
    runs_root: Path,
    child_generation: int,
) -> tuple[CheckpointRef, Path]:
    identity = _require_mapping(identity_value, "parent_checkpoint_identity")
    path, namespace, lineage_id = _reference_path(identity, runs_root=runs_root)
    checkpoint_id = _require_string(identity.get("checkpoint_id") or identity.get("label"), "parent checkpoint_id")
    generation_value = identity.get("generation")
    generation = int(generation_value) if isinstance(generation_value, int) and not isinstance(generation_value, bool) else _generation_from_id(checkpoint_id, "parent checkpoint_id")
    if _generation_from_id(checkpoint_id, "parent checkpoint_id") != generation:
        raise BackfillError("parent checkpoint_id and generation conflict")
    if generation + 1 != child_generation:
        raise BackfillError(
            f"explicit parent generation does not immediately precede child: parent=M{generation}, child=M{child_generation}"
        )
    sha = _require_sha(identity.get("artifact_sha256") or identity.get("sha256"), "parent checkpoint SHA")
    metadata_path_value = identity.get("metadata_path")
    if metadata_path_value is None:
        raise BackfillError("parent metadata_path is missing")
    metadata_path = Path(_require_string(metadata_path_value, "parent metadata_path"))
    if not metadata_path.is_absolute():
        metadata_path = (runs_root / TOPOLOGY / namespace / lineage_id / metadata_path).resolve()
    else:
        metadata_path = metadata_path.resolve()
    _path_parts_under_runs(metadata_path.with_name("M" + str(generation) + ".pt"), runs_root, "parent metadata_path")
    if not metadata_path.is_file():
        raise BackfillError(f"parent metadata file is missing: {metadata_path}")
    if not path.is_file():
        raise BackfillError(f"parent checkpoint artifact is missing: {path}")
    relative = _safe_relative(path, runs_root / TOPOLOGY / namespace / lineage_id, "parent checkpoint")
    return CheckpointRef(TOPOLOGY, lineage_id, checkpoint_id, generation, relative, sha), metadata_path


def _child_checkpoint_ref(
    metadata: Mapping[str, object],
    complete: Mapping[str, object],
    manifest: Mapping[str, object],
    catalog: ArtifactCatalog,
    *,
    lineage_root: Path,
    generation: int,
) -> CheckpointRef:
    checkpoint_id = _require_string(complete.get("label") or metadata.get("checkpoint_label"), "checkpoint label")
    if checkpoint_id != metadata.get("checkpoint_label"):
        raise BackfillError("checkpoint label differs between metadata and generation commit")
    if _generation_from_id(checkpoint_id, "checkpoint label") != generation:
        raise BackfillError("checkpoint label differs from fixed migration scope")
    if complete.get("generation") != generation:
        raise BackfillError("generation commit number differs from fixed migration scope")
    sha = _require_sha(complete.get("checkpoint_sha256"), "checkpoint SHA")
    relative = Path("checkpoints") / f"{checkpoint_id}.pt"
    physical = (lineage_root / relative).resolve()
    if not physical.is_file():
        raise BackfillError(f"checkpoint artifact is missing: {physical}")
    checkpoint_hashes = _require_mapping(manifest.get("checkpoint_hashes"), "manifest checkpoint_hashes")
    manifest_sha = _require_sha(checkpoint_hashes.get(relative.as_posix()), "manifest checkpoint SHA")
    if manifest_sha != sha:
        raise BackfillError("checkpoint SHA conflicts between generation commit and lineage manifest")
    try:
        catalog_sha = _require_sha(catalog.identity(relative.as_posix()).get("sha256"), "catalog checkpoint SHA")
    except ValueError as exc:
        raise BackfillError(f"checkpoint is not committed in artifact catalog: {relative}") from exc
    if catalog_sha != sha:
        raise BackfillError("checkpoint SHA conflicts with artifact catalog")
    return CheckpointRef(TOPOLOGY, lineage_root.name, checkpoint_id, generation, relative.as_posix(), sha)


def _fresh_replay_ref(
    complete: Mapping[str, object],
    catalog: ArtifactCatalog,
    *,
    lineage_root: Path,
    generation: int,
) -> ArtifactRef:
    sha = _require_sha(complete.get("fresh_replay_sha256"), "fresh replay SHA")
    relative = Path("replay") / f"iter-{generation}-fresh.jsonl"
    physical = (lineage_root / relative).resolve()
    if not physical.is_file():
        raise BackfillError(f"fresh replay artifact is missing: {physical}")
    try:
        identity = catalog.identity(relative.as_posix())
    except ValueError as exc:
        raise BackfillError(f"fresh replay is not committed in artifact catalog: {relative}") from exc
    catalog_sha = _require_sha(identity.get("sha256"), "catalog fresh replay SHA")
    if catalog_sha != sha:
        raise BackfillError("fresh replay SHA conflicts with artifact catalog")
    generations = _require_mapping(catalog.payload.get("generations"), "catalog generations")
    generation_record = _require_mapping(generations.get(str(generation)), f"catalog generation {generation}")
    paths = generation_record.get("artifact_paths")
    if not isinstance(paths, list) or relative.as_posix() not in paths:
        raise BackfillError(f"catalog generation {generation} does not commit its fresh replay")
    return ArtifactRef(relative.as_posix(), sha)


def _profile_sections(
    metadata: Mapping[str, object],
    manifest: Mapping[str, object],
    run_spec: Mapping[str, object],
    profile: Mapping[str, object] | None,
) -> dict[str, object]:
    scientific = _require_mapping(metadata.get("scientific_contract"), "scientific_contract")
    board_size = metadata.get("board_size")
    compatibility: dict[str, object] = {
        "topology": TOPOLOGY,
        "board_size": board_size,
        "topology_id": _require_string(metadata.get("topology_id"), "topology_id"),
        "topology_fingerprint": _require_sha(metadata.get("topology_fingerprint"), "topology_fingerprint"),
        "rules": {
            "profile_id": _require_string(metadata.get("rules_profile_id"), "rules_profile_id"),
            "fingerprint": _require_sha(metadata.get("rules_fingerprint"), "rules_fingerprint"),
        },
        "observation": {
            "schema_id": _require_string(metadata.get("observation_schema_id"), "observation_schema_id"),
            "schema_version": metadata.get("observation_schema_version"),
            "fingerprint": _require_sha(metadata.get("observation_fingerprint"), "observation_fingerprint"),
            "shape": metadata.get("observation_shape"),
        },
        "target": {
            "contract_id": _require_string(metadata.get("target_contract_id"), "target_contract_id"),
            "contract_version": metadata.get("target_contract_version"),
            "fingerprint": _require_sha(metadata.get("target_fingerprint"), "target_fingerprint"),
        },
        "network": {
            "architecture_id": _require_string(metadata.get("architecture_id"), "architecture_id"),
            "fingerprint": _require_sha(metadata.get("architecture_fingerprint"), "architecture_fingerprint"),
            "parameter_count": metadata.get("model_parameter_count"),
            "heads_and_shapes": metadata.get("network_heads_and_shapes"),
        },
    }
    if profile is not None:
        profile_fingerprint = _require_sha(profile.get("profile_fingerprint"), "historical profile fingerprint")
        if profile_fingerprint != _require_sha(metadata.get("profile_fingerprint"), "checkpoint profile fingerprint"):
            raise BackfillError("historical profile fingerprint conflicts with checkpoint metadata")
        if isinstance(profile.get("topology"), Mapping):
            compatibility["topology_details"] = dict(profile["topology"])  # type: ignore[index]
        for section in ("rules", "observation", "target", "network"):
            if isinstance(profile.get(section), Mapping):
                compatibility[section] = _merge(
                    _require_mapping(compatibility.get(section, {}), f"compatibility.{section}"),
                    _require_mapping(profile[section], f"profile.{section}"),
                )
        compatibility["profile_id"] = _require_string(profile.get("profile_id"), "profile_id")
        compatibility["profile_fingerprint"] = profile_fingerprint

    generation_config = _require_mapping(run_spec.get("generation"), "run-spec generation")
    driver_config = _require_mapping(generation_config.get("driver_config"), "run-spec generation.driver_config")
    self_play = _copy_json(profile.get("self_play", {}) if profile else {}, "profile self_play")
    raw_operator_tunables = manifest.get("operator_tunables")
    operator_tunables = (
        {}
        if raw_operator_tunables is None
        else _require_mapping(raw_operator_tunables, "manifest operator_tunables")
    )
    mcts_simulations = self_play.get("mcts_simulations", operator_tunables.get("self_play_mcts_simulations"))
    self_play.update(
        {
            "contract_id": _require_string(metadata.get("selfplay_contract_id"), "selfplay_contract_id"),
            "contract_fingerprint": _require_sha(metadata.get("selfplay_contract_fingerprint"), "selfplay_contract_fingerprint"),
            "mcts_simulations": mcts_simulations,
            "games_per_iteration": driver_config.get("games"),
            "komi": metadata.get("komi"),
        }
    )
    if self_play.get("mcts_simulations") is None:
        raise BackfillError("effective config is missing self-play mcts_simulations")

    training = _copy_json(profile.get("training", {}) if profile else {}, "profile training")
    training.update(
        {
            "optimizer": scientific.get("optimizer"),
            "learning_rate": scientific.get("learning_rate"),
            "weight_decay": scientific.get("weight_decay"),
            "batch_size": scientific.get("batch_size"),
            "optimizer_steps_per_iteration": scientific.get("optimizer_steps"),
            "samples_consumed_per_iteration": scientific.get("samples_consumed"),
            "model_gating": scientific.get("model_gating"),
            "ownership_loss": scientific.get("ownership_loss"),
            "score_loss": scientific.get("score_loss"),
        }
    )
    replay = _copy_json(profile.get("replay", {}) if profile else {}, "profile replay")
    replay.update(
        {
            "policy": metadata.get("replay_policy"),
            "generations": scientific.get("replay_generations"),
            "cap": scientific.get("replay_cap"),
        }
    )
    execution = dict(driver_config)
    arena = _copy_json(run_spec.get("arena", {}), "run-spec arena")
    supervision = _copy_json(run_spec.get("supervision", {}), "run-spec supervision")
    extensions: dict[str, object] = {
        "profile_id": profile.get("profile_id") if profile else metadata.get("profile_id"),
        "profile_fingerprint": metadata.get("profile_fingerprint"),
        "profile_path": run_spec.get("profile_path"),
        "run_spec_schema": run_spec.get("schema"),
        "run_spec_fingerprint": _require_mapping(manifest.get("run_spec"), "manifest run_spec").get("fingerprint"),
        "lineage_config_fingerprint": manifest.get("config_fingerprint"),
    }
    if isinstance(metadata.get("cadence_experiment"), Mapping):
        extensions["cadence_experiment"] = dict(metadata["cadence_experiment"])  # type: ignore[index]
    return {
        "schema": "gocube-effective-config-v2",
        "version": 2,
        "topology": TOPOLOGY,
        "compatibility": compatibility,
        "self_play": self_play,
        "training": training,
        "replay": replay,
        "execution": execution,
        "arena": arena,
        "supervision": supervision,
        "extensions": extensions,
    }


def _source_path(path: Path, *, repo_root: Path, runs_root: Path) -> str:
    for root in (repo_root, runs_root.parent):
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return str(path.resolve())


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise BackfillError("cannot determine migration commit")
    return result.stdout.strip()


def _build_plan(
    *,
    runs_root: Path,
    repo_root: Path,
    lineage_id: str,
    generation: int,
    migration_commit: str,
    profile_cache: dict[Path, dict[str, object]],
) -> NodePlan:
    lineage_root = (runs_root / TOPOLOGY / "active" / lineage_id).resolve()
    if not lineage_root.is_dir():
        raise BackfillError(f"target lineage is missing: {lineage_root}")
    metadata_path = lineage_root / CHECKPOINT_METADATA_DIR / f"M{generation}.metadata.json"
    complete_path = lineage_root / f"generation-{generation}.complete.json"
    manifest_path = lineage_root / "manifest.json"
    run_spec_path = lineage_root / "run-spec.json"
    catalog_path = lineage_root / "runtime" / "artifact-catalog.json"
    metadata = _read_json(metadata_path, "checkpoint metadata")
    complete = _read_json(complete_path, "generation commit")
    manifest = _read_json(manifest_path, "lineage manifest")
    run_spec = _read_json(run_spec_path, "run-spec")
    try:
        catalog = ArtifactCatalog.load(catalog_path, root=lineage_root)
    except (OSError, ValueError) as exc:
        raise BackfillError(f"cannot load artifact catalog: {catalog_path}") from exc

    checkpoint = _child_checkpoint_ref(metadata, complete, manifest, catalog, lineage_root=lineage_root, generation=generation)
    parent, parent_metadata_path = _checkpoint_ref_from_explicit_parent(
        metadata.get("parent_checkpoint_identity"),
        runs_root=runs_root,
        child_generation=generation,
    )
    fresh_replay = _fresh_replay_ref(complete, catalog, lineage_root=lineage_root, generation=generation)

    profile: dict[str, object] | None = None
    profile_ref = run_spec.get("profile_path")
    if profile_ref is not None:
        profile = _load_profile(_profile_path(profile_ref, repo_root=repo_root, runs_root=runs_root), profile_cache)
    config_payload = _profile_sections(metadata, manifest, run_spec, profile)
    config = EffectiveConfig.from_dict(config_payload)
    config_json_path = CONFIG_DIRECTORY / f"{config.fingerprint}.json"
    config_abs_path = lineage_root / config_json_path
    config_ref = EffectiveConfigRef(ArtifactRef(config_json_path.as_posix(), _json_sha(config.to_dict())), config.fingerprint)

    provenance_payload: dict[str, object] = {
        "schema": BACKFILL_PROVENANCE_SCHEMA,
        "version": 1,
        "checkpoint": checkpoint.to_dict(),
        "source_metadata_files": [
            _source_path(metadata_path, repo_root=repo_root, runs_root=runs_root),
            _source_path(complete_path, repo_root=repo_root, runs_root=runs_root),
            _source_path(manifest_path, repo_root=repo_root, runs_root=runs_root),
            _source_path(run_spec_path, repo_root=repo_root, runs_root=runs_root),
            _source_path(catalog_path, repo_root=repo_root, runs_root=runs_root),
            _source_path(parent_metadata_path, repo_root=repo_root, runs_root=runs_root),
        ],
        "immediate_parent": parent.to_dict(),
        "fresh_replay": fresh_replay.to_dict(),
        "effective_config_fingerprint": config.fingerprint,
        "migration_commit": migration_commit,
    }
    provenance_path = PROVENANCE_DIRECTORY / f"{checkpoint.checkpoint_id}.json"
    provenance_ref = ArtifactRef(provenance_path.as_posix(), _json_sha(provenance_payload))
    node = CheckpointNode(
        checkpoint=checkpoint,
        genesis=False,
        parent=parent,
        fresh_replay=fresh_replay,
        effective_config=config_ref,
        provenance=provenance_ref,
    )
    return NodePlan(
        lineage_root=lineage_root,
        checkpoint=checkpoint,
        node=node,
        config_payload=config.to_dict(),
        provenance_payload=provenance_payload,
        node_payload=node.to_dict(),
        config_path=config_abs_path,
        provenance_path=lineage_root / provenance_path,
        node_path=lineage_root / NODE_DIRECTORY / f"{checkpoint.checkpoint_id}.json",
    )


def _destination_state(path: Path, payload: Mapping[str, object]) -> str:
    if not path.exists():
        return "missing"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "conflict"
    return "same" if existing == dict(payload) else "conflict"


def _publish(path: Path, payload: Mapping[str, object]) -> None:
    state = _destination_state(path, payload)
    if state == "same":
        return
    if state == "conflict":
        raise BackfillError(f"destination conflicts with existing V2 metadata: {path}")
    atomic_write_json(path, payload)


def run_backfill(
    runs_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    apply: bool = False,
    targets: Sequence[ScopeEntry] = TARGET_SCOPE,
    migration_commit: str | None = None,
) -> BackfillReport:
    runs_root_path = Path(runs_root).resolve()
    repo_root_path = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    commit = migration_commit or _git_commit(repo_root_path)
    profile_cache: dict[Path, dict[str, object]] = {}
    plans: list[NodePlan] = []
    problems: list[str] = []
    target_count = sum(entry.count for entry in targets)
    for entry in targets:
        for generation in range(entry.first_generation, entry.last_generation + 1):
            label = f"{entry.lineage_id}/M{generation}"
            try:
                plans.append(
                    _build_plan(
                        runs_root=runs_root_path,
                        repo_root=repo_root_path,
                        lineage_id=entry.lineage_id,
                        generation=generation,
                        migration_commit=commit,
                        profile_cache=profile_cache,
                    )
                )
            except (BackfillError, TypeError, ValueError, OSError) as exc:
                problems.append(f"{label}: {exc}")

    destinations: list[tuple[Path, Mapping[str, object]]] = []
    for plan in plans:
        destinations.extend(
            (
                (plan.config_path, plan.config_payload),
                (plan.provenance_path, plan.provenance_payload),
                (plan.node_path, plan.node_payload),
            )
        )
    # A shared config artifact is expected when a lineage keeps one effective
    # configuration across generations.  De-duplicate exact destination/payload
    # pairs before counting writes or checking conflicts.
    unique: dict[Path, Mapping[str, object]] = {}
    for path, payload in destinations:
        prior = unique.get(path)
        if prior is not None and dict(prior) != dict(payload):
            problems.append(f"{path}: plans disagree about immutable destination content")
        unique[path] = payload

    conflicts = 0
    missing = 0
    for path, payload in unique.items():
        state = _destination_state(path, payload)
        if state == "conflict":
            conflicts += 1
            problems.append(f"{path}: conflicting existing V2 metadata")
        elif state == "missing":
            missing += 1

    complete_plans = len(plans) == target_count and not problems
    ready = len(plans) if complete_plans and conflicts == 0 else 0
    writes = missing if apply and complete_plans and conflicts == 0 else 0
    if apply and complete_plans and conflicts == 0:
        for path, payload in unique.items():
            _publish(path, payload)

    return BackfillReport(target_count, ready, conflicts, writes, tuple(problems))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only (the default)")
    mode.add_argument("--apply", action="store_true", help="publish missing V2 JSON metadata")
    parser.add_argument("--runs-root", type=Path, default=None, help="override the repository runs/ root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runs_root = args.runs_root or (Path(__file__).resolve().parents[1] / "runs")
    try:
        report = run_backfill(runs_root, apply=bool(args.apply))
    except (BackfillError, OSError, ValueError) as exc:
        print(f"Backfill failed: {exc}")
        return 1
    print(f"Target checkpoints: {report.target_checkpoints}")
    print(f"Ready: {report.ready}")
    print(f"Conflicts: {report.conflicts}")
    print(f"Writes: {report.writes}")
    for problem in report.problems:
        print(f"Problem: {problem}")
    return 0 if report.ready == report.target_checkpoints and not report.problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
