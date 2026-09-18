from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import CheckpointCatalogCollision

_GOLDEN_CHECKPOINT_RE = re.compile(r"^M(\d+)\.pt$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_LINEAGE_STATUSES = frozenset({"ACTIVE", "ARCHIVED", "DISCARDED"})
GOLDEN_TERMINAL_ADJUDICATOR = "golden-graph-area-v1"
GOLDEN_CHECKPOINT_FORMAT = "golden_pt"
DEFAULT_PUBLICATION_MANIFEST = (
    Path(__file__).resolve().parents[4] / "configs/gocube/production_checkpoint_publication.json"
)


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
    lineage_status: str = "ACTIVE"
    metadata_error: str | None = None
    profile_id: str | None = None
    profile_fingerprint: str | None = None
    architecture_id: str | None = None
    rules_fingerprint: str | None = None
    observation_fingerprint: str | None = None
    target_fingerprint: str | None = None
    metadata_path: str | None = None
    serving_contract_data: Mapping[str, object] | None = None
    published: bool = False
    publication_reason: str | None = None

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
            "lineageStatus": self.lineage_status,
        }


class CheckpointCatalog:
    def __init__(self, checkpoint_dir: str, publication_manifest: str | None = None):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        selected_manifest = Path(publication_manifest) if publication_manifest else DEFAULT_PUBLICATION_MANIFEST
        self.publication_manifest = os.path.abspath(str(selected_manifest))
        self._publication_entries = self._read_publication_manifest(selected_manifest)

    @staticmethod
    def _read_publication_manifest(path: Path) -> tuple[dict[str, object], ...]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ()
        if not isinstance(payload, dict) or payload.get("schema") != "gocube-production-checkpoint-publication-v1":
            return ()
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            return ()
        entries: list[dict[str, object]] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            lineage_id = raw.get("lineage_id")
            topology = raw.get("topology")
            ranges = raw.get("checkpoint_ranges")
            if not isinstance(lineage_id, str) or not lineage_id or not isinstance(topology, str):
                continue
            normalized_ranges: list[tuple[int, int]] = []
            if isinstance(ranges, list):
                for item in ranges:
                    if (
                        isinstance(item, list)
                        and len(item) == 2
                        and all(isinstance(value, int) and not isinstance(value, bool) for value in item)
                        and item[0] <= item[1]
                    ):
                        normalized_ranges.append((int(item[0]), int(item[1])))
            if normalized_ranges:
                entries.append(
                    {
                        "lineage_id": lineage_id,
                        "topology": topology,
                        "checkpoint_ranges": tuple(normalized_ranges),
                    }
                )
        return tuple(entries)

    def publication_decision(
        self,
        *,
        topology: str,
        lineage_id: str | None,
        iteration: int,
        lineage_status: str,
    ) -> tuple[bool, str]:
        if lineage_status == "DISCARDED":
            return False, "lineage status is DISCARDED"
        if lineage_id is None:
            return False, "checkpoint has no canonical lineage_id"
        if not self._publication_entries:
            return False, f"publication manifest is missing or invalid: {self.publication_manifest}"
        publication_topology = {"torus": "torus9", "cube": "cube4"}.get(topology, topology)
        same_lineage = [
            entry
            for entry in self._publication_entries
            if entry["topology"] == publication_topology and entry["lineage_id"] == lineage_id
        ]
        if not same_lineage:
            return False, f"lineage {lineage_id!r} is not listed in publication manifest"
        for entry in same_lineage:
            if any(start <= iteration <= end for start, end in entry["checkpoint_ranges"]):
                return True, f"published by {self.publication_manifest}"
        return False, f"iteration M{iteration} is outside the published ranges for {lineage_id!r}"

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
            or not _SHA256_RE.fullmatch(profile_fingerprint)
            or not isinstance(model_hash, str)
            or not _SHA256_RE.fullmatch(model_hash)
        ):
            return None

        try:
            if profile_id == "gocube-torus9-golden-v3":
                from gocube_golden.torus9_contract import (
                    TORUS9_ACTION_COUNT,
                    TORUS9_CURRENT_ARCHITECTURE_ID,
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
            "serving_contract": {
                "checkpoint_format": GOLDEN_CHECKPOINT_FORMAT,
                "checkpoint_schema_version": metadata.get("checkpoint_schema_version"),
                "topology": topology,
                "size": size,
                "rule_set": "chinese",
                "terminal_adjudicator": GOLDEN_TERMINAL_ADJUDICATOR,
                "architecture_id": architecture_id,
                "architecture_config": metadata.get("architecture_config"),
                "topology_id": metadata.get("topology_id"),
                "topology_fingerprint": metadata.get("topology_fingerprint"),
                "board_size": metadata.get("board_size"),
                "point_id_order_identity": metadata.get("point_id_order_identity"),
                "komi": 0.5,
                "observation_schema_id": metadata.get("observation_schema_id"),
                "observation_schema_version": metadata.get("observation_schema_version"),
                "observation_fingerprint": observation_fingerprint,
                "observation_shape": metadata.get("observation_shape"),
                "target_contract_id": metadata.get("target_contract_id"),
                "target_contract_version": metadata.get("target_contract_version"),
                "target_fingerprint": target_fingerprint,
                "rules_profile_id": metadata.get("rules_profile_id", metadata.get("rules_id")),
                "rules_fingerprint": rules_fingerprint,
                "network_heads_and_shapes": metadata.get("network_heads_and_shapes"),
                "auxiliary_heads": metadata.get("auxiliary_heads"),
            },
        }

    @staticmethod
    def _lineage_identity_for_checkpoint(path: str) -> tuple[str | None, str]:
        """Read canonical lineage identity/status when a storage manifest is present."""
        manifest_path = Path(path).parent.parent / "manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, "ACTIVE"
        if not isinstance(manifest, dict):
            return None, "ACTIVE"
        lineage_id = manifest.get("lineage_id")
        status = manifest.get("status")
        normalized_lineage_id = lineage_id if isinstance(lineage_id, str) and lineage_id else None
        normalized_status = status if isinstance(status, str) and status in _LINEAGE_STATUSES else "ACTIVE"
        return normalized_lineage_id, normalized_status

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
            lineage_id, lineage_status = self._lineage_identity_for_checkpoint(path)
            if lineage_id is not None:
                identity["run_name"] = lineage_id
            filename_match = _GOLDEN_CHECKPOINT_RE.fullmatch(os.path.basename(path))
            if filename_match is None or identity["iteration"] != int(filename_match.group(1)):
                continue
            published, publication_reason = self.publication_decision(
                topology=str(identity["topology"]),
                lineage_id=lineage_id or str(identity["run_name"]),
                iteration=int(identity["iteration"]),
                lineage_status=lineage_status,
            )
            if not published:
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
                    lineage_status=lineage_status,
                    profile_id=str(identity["profile_id"]),
                    profile_fingerprint=str(identity["profile_fingerprint"]),
                    architecture_id=str(identity["architecture_id"]),
                    rules_fingerprint=str(identity["rules_fingerprint"]),
                    observation_fingerprint=str(identity["observation_fingerprint"]),
                    target_fingerprint=str(identity["target_fingerprint"]),
                    metadata_path=os.path.abspath(metadata_path),
                    serving_contract_data=identity.get("serving_contract"),
                    published=published,
                    publication_reason=publication_reason,
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


def serving_contract(descriptor: CheckpointDescriptor) -> dict[str, object]:
    """Return only the artifact identity required by the serving boundary."""
    if descriptor.serving_contract_data is not None:
        return dict(descriptor.serving_contract_data)
    return {
        "checkpoint_format": GOLDEN_CHECKPOINT_FORMAT,
        "topology": descriptor.topology,
        "size": descriptor.size,
        "rule_set": descriptor.rule_set,
        "terminal_adjudicator": descriptor.terminal_adjudicator,
        "architecture_id": descriptor.architecture_id,
        "rules_fingerprint": descriptor.rules_fingerprint,
        "observation_fingerprint": descriptor.observation_fingerprint,
        "target_fingerprint": descriptor.target_fingerprint,
        "komi": descriptor.komi,
    }


def is_runtime_compatible(a: CheckpointDescriptor, b: CheckpointDescriptor) -> bool:
    return serving_contract(a) == serving_contract(b)
