from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace

from alphazero.utils import get_iter_file

from .contract import ContractError, ResolvedGoCubeContract
from .errors import CheckpointCatalogCollision
from .manifest import ManifestError, RunManifest, load_run_manifest

_CHECKPOINT_RE = re.compile(r"^iteration-(\d{4,})\.pkl$")
_GOLDEN_CHECKPOINT_RE = re.compile(r"^M(\d+)\.pt$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GOLDEN_TERMINAL_ADJUDICATOR = "golden-graph-area-v1"


@dataclass(frozen=True)
class CheckpointDescriptor:
    checkpoint_id: str
    run_name: str
    iteration: int
    topology: str
    size: int
    rule_set: str
    komi: float
    terminal_adjudicator: str
    path: str
    model_contract: dict[str, object] | None = None
    metadata_error: str | None = None
    backend_kind: str = "legacy_nnet"
    checkpoint_format: str = "legacy_pickle"
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    architecture_id: str | None = None
    rules_fingerprint: str | None = None
    observation_fingerprint: str | None = None
    target_fingerprint: str | None = None
    metadata_path: str | None = None

    @classmethod
    def from_manifest(
        cls, manifest: RunManifest, *, iteration: int, path: str
    ) -> "CheckpointDescriptor":
        return cls(
            checkpoint_id=f"{manifest.run_name}@{iteration}",
            run_name=manifest.run_name,
            iteration=iteration,
            topology=manifest.topology,
            size=manifest.size,
            rule_set=manifest.rule_set,
            komi=float(manifest.komi),
            terminal_adjudicator=manifest.terminal_adjudicator,
            path=path,
            model_contract=manifest.model_contract,
        )

    def to_api(self) -> dict[str, object]:
        return {
            "id": self.checkpoint_id,
            "runName": self.run_name,
            "iteration": self.iteration,
            "topology": self.topology,
            "size": self.size,
            "ruleSet": self.rule_set,
            "komi": self.komi,
            "terminalAdjudicator": self.terminal_adjudicator,
            **({"modelContract": self.model_contract} if self.model_contract is not None else {}),
        }


class CheckpointCatalog:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)

    @staticmethod
    def iteration_from_filename(filename: str) -> int | None:
        match = _CHECKPOINT_RE.fullmatch(filename)
        if not match:
            return None
        iteration = int(match.group(1))
        if filename != get_iter_file(iteration):
            return None
        return iteration

    @classmethod
    def checkpoint_files(cls, run_dir: str) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        try:
            entries = os.listdir(run_dir)
        except FileNotFoundError:
            return found
        for filename in entries:
            iteration = cls.iteration_from_filename(filename)
            if iteration is None:
                continue
            path = os.path.join(run_dir, filename)
            if not os.path.isfile(path):
                continue
            try:
                if os.path.getsize(path) <= 0:
                    continue
            except OSError:
                continue
            found.append((iteration, path))
        found.sort(key=lambda item: item[0])
        return found

    def list(self) -> list[CheckpointDescriptor]:
        descriptors: list[CheckpointDescriptor] = []
        try:
            run_names = sorted(os.listdir(self.checkpoint_dir))
        except FileNotFoundError:
            return descriptors

        for run_name in run_names:
            run_dir = os.path.join(self.checkpoint_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            try:
                manifest = load_run_manifest(run_dir)
            except ManifestError:
                continue
            for iteration, path in self.checkpoint_files(run_dir):
                descriptor = CheckpointDescriptor.from_manifest(
                    manifest,
                    iteration=iteration,
                    path=path,
                )
                descriptors.append(self._enrich_descriptor(descriptor, run_dir))

        descriptors.extend(self._golden_descriptors())
        descriptors.sort(key=lambda item: (item.run_name, item.iteration, item.backend_kind))
        by_id: dict[str, CheckpointDescriptor] = {}
        for descriptor in descriptors:
            previous = by_id.get(descriptor.checkpoint_id)
            if previous is not None and previous.backend_kind != descriptor.backend_kind:
                raise CheckpointCatalogCollision(
                    f"Checkpoint id collision between {previous.backend_kind} and "
                    f"{descriptor.backend_kind}: {descriptor.checkpoint_id}"
                )
            if previous is not None and previous.path != descriptor.path:
                raise CheckpointCatalogCollision(
                    f"Duplicate checkpoint id with different artifacts: {descriptor.checkpoint_id}"
                )
            by_id[descriptor.checkpoint_id] = descriptor
        return descriptors

    @staticmethod
    def _golden_metadata(path: str) -> object | None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    @classmethod
    def _validate_golden_metadata(cls, metadata: object) -> dict[str, object] | None:
        """Return normalized supported identity, or None for unsupported data."""

        if not isinstance(metadata, dict):
            return None
        profile_id = metadata.get("profile_id", metadata.get("training_profile_id"))
        profile_fingerprint = metadata.get(
            "profile_fingerprint", metadata.get("training_profile_fingerprint")
        )
        model_hash = metadata.get("model_hash")
        if (
            not isinstance(profile_id, str)
            or not isinstance(profile_fingerprint, str)
            or not isinstance(model_hash, str)
            or not _SHA256_RE.fullmatch(model_hash)
        ):
            return None

        try:
            if profile_id == "gocube-torus9-golden-v3":
                from gocube_golden.torus9_contract import (
                    TORUS9_ACTION_COUNT,
                    TORUS9_CURRENT_ARCHITECTURE_ID,
                    TORUS9_CURRENT_PROFILE_FINGERPRINT,
                    TORUS9_CURRENT_PROFILE_ID,
                    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
                    TORUS9_CURRENT_TARGET_FINGERPRINT,
                    TORUS9_KOMI,
                    TORUS9_OBSERVATION_FINGERPRINT,
                    TORUS9_OBSERVATION_SCHEMA_ID,
                    TORUS9_OBSERVATION_SCHEMA_VERSION,
                    TORUS9_POINT_COUNT,
                    TORUS9_RULES_FINGERPRINT,
                    TORUS9_TARGET_CONTRACT_ID,
                    current_torus9_selfplay_contract_fingerprint,
                )
                from gocube_golden.torus9 import TORUS9_TOPOLOGY_ID, TORUS9_TOPOLOGY_FINGERPRINT
                expected = {
                    "checkpoint_schema_version": 1,
                    "profile_id": TORUS9_CURRENT_PROFILE_ID,
                    "profile_fingerprint": TORUS9_CURRENT_PROFILE_FINGERPRINT,
                    "architecture_id": TORUS9_CURRENT_ARCHITECTURE_ID,
                    "topology_id": TORUS9_TOPOLOGY_ID,
                    "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
                    "board_size": [9, 9],
                    "point_id_order_identity": "row-major-yx:point_id=y*width+x",
                    "komi": TORUS9_KOMI,
                    "observation_schema_id": TORUS9_OBSERVATION_SCHEMA_ID,
                    "observation_schema_version": TORUS9_OBSERVATION_SCHEMA_VERSION,
                    "observation_fingerprint": TORUS9_OBSERVATION_FINGERPRINT,
                    "observation_shape": [6, TORUS9_POINT_COUNT],
                    "target_contract_id": TORUS9_TARGET_CONTRACT_ID,
                    "target_contract_version": 1,
                    "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
                    "rules_profile_id": "graph-area-v1",
                    "rules_fingerprint": TORUS9_RULES_FINGERPRINT,
                    "network_heads_and_shapes": {
                        "policy": [TORUS9_ACTION_COUNT],
                        "value": [3],
                        "ownership": [TORUS9_POINT_COUNT, 3],
                        "score": [1],
                    },
                    "auxiliary_heads": True,
                    "selfplay_contract_id": TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
                    "selfplay_contract_fingerprint": current_torus9_selfplay_contract_fingerprint(),
                }
                topology, size = "torus", 9
                architecture_id = TORUS9_CURRENT_ARCHITECTURE_ID
                rules_fingerprint = TORUS9_RULES_FINGERPRINT
                observation_fingerprint = TORUS9_OBSERVATION_FINGERPRINT
                target_fingerprint = TORUS9_CURRENT_TARGET_FINGERPRINT
            elif profile_id == "gocube-cube4-golden-training-v1":
                from gocube_golden.cube_contract import CUBE_PROFILE_ID, load_profile
                from gocube_golden.cube_neural import CUBE_OBSERVATION_FINGERPRINT, CUBE_OBSERVATION_SCHEMA_ID
                from gocube_golden.cube_topology import (
                    CUBE4_GEOMETRY_FINGERPRINT,
                    CUBE4_TOPOLOGY_FINGERPRINT,
                    CUBE4_TOPOLOGY_ID,
                )
                from gocube_golden.cube_training import CUBE_TARGET_CONTRACT_ID, CUBE_TARGET_FINGERPRINT
                profile = load_profile()
                expected = {
                    "checkpoint_schema_version": 1,
                    "training_profile_id": CUBE_PROFILE_ID,
                    "training_profile_fingerprint": profile.get("profile_fingerprint"),
                    "architecture_id": "GoldenCubeGraphNetV1",
                    "topology_id": CUBE4_TOPOLOGY_ID,
                    "topology_fingerprint": CUBE4_TOPOLOGY_FINGERPRINT,
                    "geometry_fingerprint": CUBE4_GEOMETRY_FINGERPRINT,
                    "point_count": 96,
                    "action_count": 97,
                    "point_ordering_fingerprint": profile["topology"]["point_ordering_fingerprint"],
                    "rules_id": "graph-area-v1",
                    "rules_fingerprint": profile["rules"]["fingerprint"],
                    "komi": 0.5,
                    "observation_schema_id": CUBE_OBSERVATION_SCHEMA_ID,
                    "observation_fingerprint": CUBE_OBSERVATION_FINGERPRINT,
                    "target_contract_id": CUBE_TARGET_CONTRACT_ID,
                    "target_fingerprint": CUBE_TARGET_FINGERPRINT,
                    "network_heads_and_shapes": {"policy": [97], "value": [3]},
                }
                topology, size = "cube", 4
                architecture_id = "GoldenCubeGraphNetV1"
                rules_fingerprint = str(profile["rules"]["fingerprint"])
                observation_fingerprint = CUBE_OBSERVATION_FINGERPRINT
                target_fingerprint = CUBE_TARGET_FINGERPRINT
            else:
                return None
            for key, value in expected.items():
                if metadata.get(key) != value:
                    return None
            if profile_id == "gocube-cube4-golden-training-v1":
                if metadata.get("profile_id", profile_id) != profile_id:
                    return None
                if metadata.get("profile_fingerprint", profile_fingerprint) != profile_fingerprint:
                    return None
        except (KeyError, TypeError, ValueError, ImportError):
            return None

        run_name = metadata.get("run_id")
        label = metadata.get("checkpoint_label")
        if (
            not isinstance(run_name, str)
            or not run_name
            or os.path.basename(run_name) != run_name
            or not isinstance(label, str)
        ):
            return None
        label_match = _GOLDEN_CHECKPOINT_RE.fullmatch(label + ".pt")
        if label_match is None:
            return None
        return {
            "run_name": run_name,
            "iteration": int(label_match.group(1)),
            "topology": topology,
            "size": size,
            "profile_id": profile_id,
            "profile_fingerprint": profile_fingerprint,
            "architecture_id": architecture_id,
            "rules_fingerprint": rules_fingerprint,
            "observation_fingerprint": observation_fingerprint,
            "target_fingerprint": target_fingerprint,
            "komi": 0.5,
        }

    def _golden_descriptors(self) -> list[CheckpointDescriptor]:
        descriptors: list[CheckpointDescriptor] = []
        candidates: list[str] = []
        try:
            for root, _directories, filenames in os.walk(self.checkpoint_dir):
                candidates.extend(
                    os.path.join(root, filename)
                    for filename in filenames
                    if _GOLDEN_CHECKPOINT_RE.fullmatch(filename)
                )
        except OSError:
            return descriptors

        for path in sorted(candidates):
            try:
                if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                    continue
            except OSError:
                continue
            metadata_path = os.path.splitext(path)[0] + ".metadata.json"
            identity = self._validate_golden_metadata(self._golden_metadata(metadata_path))
            if identity is None:
                continue
            filename_match = _GOLDEN_CHECKPOINT_RE.fullmatch(os.path.basename(path))
            if filename_match is None or identity["iteration"] != int(filename_match.group(1)):
                continue
            descriptors.append(
                CheckpointDescriptor(
                    checkpoint_id=f"{identity['run_name']}@{identity['iteration']}",
                    run_name=str(identity["run_name"]),
                    iteration=int(identity["iteration"]),
                    topology=str(identity["topology"]),
                    size=int(identity["size"]),
                    # V1 has no graph-area tag.  This is only the narrow
                    # area/komi/double-pass UI projection; Golden fingerprints
                    # remain authoritative in the internal descriptor.
                    rule_set="chinese",
                    komi=float(identity["komi"]),
                    terminal_adjudicator=GOLDEN_TERMINAL_ADJUDICATOR,
                    path=os.path.abspath(path),
                    backend_kind="golden",
                    checkpoint_format="golden_pt",
                    profile_id=str(identity["profile_id"]),
                    profile_fingerprint=str(identity["profile_fingerprint"]),
                    architecture_id=str(identity["architecture_id"]),
                    rules_fingerprint=str(identity["rules_fingerprint"]),
                    observation_fingerprint=str(identity["observation_fingerprint"]),
                    target_fingerprint=str(identity["target_fingerprint"]),
                    metadata_path=os.path.abspath(metadata_path),
                )
            )
        return descriptors

    @staticmethod
    def _read_json(path: str) -> object | None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    @classmethod
    def _artifact_contract(
        cls, path: str, *, kind: str
    ) -> tuple[ResolvedGoCubeContract | None, str | None]:
        data = cls._read_json(path)
        if not isinstance(data, dict):
            return None, None
        value = data.get("modelContract") if kind == "compact" else data.get("model_contract")
        if value is None and kind == "effective":
            value = data.get("modelContract")
        if value is None:
            return None, None
        if not isinstance(value, dict):
            return None, f"{os.path.basename(path)} model contract must be an object"
        try:
            return ResolvedGoCubeContract.from_dict(value), None
        except ContractError as exc:
            return None, f"Invalid {os.path.basename(path)} model contract: {exc}"

    @classmethod
    def _artifact_projection_error(
        cls, path: str, contract: ResolvedGoCubeContract | None
    ) -> str | None:
        if contract is None:
            return None
        data = cls._read_json(path)
        if not isinstance(data, dict):
            return None
        projections = {
            "topology": contract.topology_kind,
            "size": contract.topology_size,
            "point_count": contract.point_count,
            "observation_schema": contract.observation_schema,
            "observation_shape": list(contract.observation_shape),
            "action_schema": contract.action_schema,
            "action_size": contract.action_size,
            "terminal_adjudicator": contract.terminal_adjudicator_id,
            "terminal_adjudicator_id": contract.terminal_adjudicator_id,
            "rules_fingerprint": contract.rules_fingerprint,
            "komi": contract.komi,
        }
        for key, expected in projections.items():
            if key not in data:
                continue
            actual = data[key]
            if actual != expected:
                return (
                    f"Conflicting GoCube model contract metadata in {os.path.basename(path)}: "
                    f"{key} saved={actual!r}, expected={expected!r}"
                )
        return None

    @classmethod
    def _enrich_descriptor(cls, descriptor: CheckpointDescriptor, run_dir: str) -> CheckpointDescriptor:
        """Carry rich metadata to the loader and retain conflicts fail-closed."""

        compact = None
        if descriptor.model_contract is not None:
            try:
                compact = ResolvedGoCubeContract.from_dict(descriptor.model_contract)
            except ContractError as exc:
                return replace(descriptor, metadata_error=f"Invalid compact model contract: {exc}")
        rich, rich_error = cls._artifact_contract(os.path.join(run_dir, "run-manifest.json"), kind="rich")
        effective, effective_error = cls._artifact_contract(
            os.path.join(run_dir, "effective-config.json"), kind="effective"
        )
        if rich_error or effective_error:
            return replace(descriptor, metadata_error=rich_error or effective_error)
        projection_error = cls._artifact_projection_error(
            os.path.join(run_dir, "run-manifest.json"), rich
        ) or cls._artifact_projection_error(
            os.path.join(run_dir, "effective-config.json"), effective
        )
        if projection_error:
            return replace(descriptor, metadata_error=projection_error)
        sources = [("gocube-run.json", compact), ("run-manifest.json", rich), ("effective-config.json", effective)]
        present = [(name, value) for name, value in sources if value is not None]
        for index, (name, value) in enumerate(present):
            for other_name, other in present[index + 1:]:
                differences = value.differences(other)
                if differences:
                    field, (left, right) = next(iter(differences.items()))
                    return replace(
                        descriptor,
                        model_contract=(compact or value).to_dict(),
                        metadata_error=(
                            f"Conflicting GoCube model contract metadata in {name} and "
                            f"{other_name}: {field} saved={left!r}, expected={right!r}"
                        ),
                    )
        chosen = compact or rich or effective
        return replace(descriptor, model_contract=chosen.to_dict() if chosen else None)

    def get(self, checkpoint_id: str) -> CheckpointDescriptor | None:
        if not isinstance(checkpoint_id, str):
            return None
        for descriptor in self.list():
            if descriptor.checkpoint_id == checkpoint_id:
                return descriptor
        return None
