from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import CheckpointCatalogCollision

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
    metadata_error: str | None = None
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    architecture_id: str | None = None
    rules_fingerprint: str | None = None
    observation_fingerprint: str | None = None
    target_fingerprint: str | None = None
    metadata_path: str | None = None

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
        }


class CheckpointCatalog:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)

    def list(self) -> list[CheckpointDescriptor]:
        descriptors = self._golden_descriptors()
        descriptors.sort(key=lambda item: (item.run_name, item.iteration))
        by_id: dict[str, CheckpointDescriptor] = {}
        for descriptor in descriptors:
            previous = by_id.get(descriptor.checkpoint_id)
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

    @staticmethod
    def _lineage_id_for_checkpoint(path: str) -> str | None:
        """Use storage lineage identity when metadata kept a legacy run id."""
        manifest_path = Path(path).parent.parent / "manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        lineage_id = manifest.get("lineage_id") if isinstance(manifest, dict) else None
        return lineage_id if isinstance(lineage_id, str) and lineage_id else None

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
            lineage_id = self._lineage_id_for_checkpoint(path)
            if lineage_id is not None:
                identity["run_name"] = lineage_id
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
                    # V1 has no graph-area tag. This is only the narrow
                    # area/komi/double-pass UI projection; Golden fingerprints
                    # remain authoritative in the internal descriptor.
                    rule_set="chinese",
                    komi=float(identity["komi"]),
                    terminal_adjudicator=GOLDEN_TERMINAL_ADJUDICATOR,
                    path=os.path.abspath(path),
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

    def get(self, checkpoint_id: str) -> CheckpointDescriptor | None:
        if not isinstance(checkpoint_id, str):
            return None
        for descriptor in self.list():
            if descriptor.checkpoint_id == checkpoint_id:
                return descriptor
        return None
