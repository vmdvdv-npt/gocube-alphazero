"""Publish a byte-identical diagnostic snapshot as an evaluation-only V2 owner.

No engine, replay publication, training promotion, or lifecycle event is involved.
The source report and snapshot are pinned; normal Arena handles its own lifecycle.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

from ..artifact_graph import CheckpointNode, EffectiveConfig
from ..artifact_resolver import ArtifactResolver
from ..provenance import file_sha256
from .operator_job import _name, _selector, resolve_checkpoint


def register_diagnostic_checkpoint(raw, *, runs_root, check_only=False):
    required = {"schema", "lineage_id", "parent", "report", "arm", "updates"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("Diagnostic registration requires exactly " + ", ".join(sorted(required)))
    if raw["schema"] != "gocube-diagnostic-checkpoint-registration-v1":
        raise ValueError("Unsupported diagnostic registration schema")
    lineage = _name(raw["lineage_id"], "lineage_id")
    _name(raw["arm"], "arm")
    if type(raw["updates"]) is not int or raw["updates"] < 0:
        raise ValueError("updates must be a nonnegative integer")
    reference = raw["report"]
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("report requires path and sha256")
    report_path = Path(reference["path"])
    if not report_path.is_absolute() or not re.fullmatch(r"sha256:[0-9a-f]{64}", reference["sha256"]):
        raise ValueError("report requires an absolute path and sha256: digest")
    if file_sha256(report_path) != reference["sha256"]:
        raise ValueError("Diagnostic report SHA mismatch")
    report = json.loads(report_path.read_text())
    if report.get("status") != "COMPLETED":
        raise ValueError("Diagnostic report must be COMPLETED")
    resolver = ArtifactResolver(runs_root)
    parent = resolve_checkpoint(_selector(raw["parent"], "parent"), resolver=resolver)
    if report.get("parent") != parent.ref.to_dict():
        raise ValueError("Diagnostic report parent disagrees with registered parent")
    snapshot = report["snapshots"][raw["arm"]][str(raw["updates"])]
    source = Path(snapshot["path"]).resolve()
    source.relative_to(report_path.resolve().parent)
    if source.suffix != ".pt" or file_sha256(source) != snapshot["sha256"]:
        raise ValueError("Diagnostic snapshot SHA/path mismatch")
    # Validate the checkpoint contract and every tensor against the parent's
    # architecture, on CPU only; no inference, games, or optimizer steps.
    import torch
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (payload.get("diagnostic_only") is not True
            or payload.get("parent") != parent.ref.to_dict()
            or payload.get("arm") != raw["arm"]
            or payload.get("updates_from_parent") != raw["updates"]
            or payload.get("model_hash") != snapshot["model_hash"]):
        raise ValueError("Diagnostic snapshot identity disagrees with report")
    original_metadata = json.loads(parent.path.with_suffix(".metadata.json").read_text())
    from ..torus9_adaptation import AdaptationModel, FINGERPRINT
    from ..neural import model_hash
    architecture = original_metadata["architecture_config"]
    if architecture.get("training_contract") != FINGERPRINT:
        raise ValueError("Only current Torus9 5CH adaptation architecture is supported")
    model = AdaptationModel()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("Diagnostic checkpoint contains nonfinite tensors")
    if model_hash(model) != snapshot["model_hash"]:
        raise ValueError("Diagnostic model tensor hash mismatch")
    checkpoint_id = raw["arm"] + "-u" + str(raw["updates"])
    root = Path(runs_root).resolve() / "torus9" / "archive" / lineage
    ref = {"topology": "torus9", "lineage_id": lineage,
           "checkpoint_id": checkpoint_id, "generation": parent.generation,
           "path": "checkpoints/" + checkpoint_id + ".pt", "sha256": snapshot["sha256"]}
    result = {"state": "VALIDATED", "selector": lineage + "/" + checkpoint_id,
              "checkpoint": ref, "evaluation_only": True}
    active_root = Path(runs_root).resolve() / "torus9" / "active" / lineage
    if active_root.exists():
        raise ValueError("Diagnostic lineage already exists in active storage")
    if root.exists():
        if json.loads((root / "registration.json").read_text()) != raw:
            raise ValueError("Diagnostic lineage already has different registration")
        resolver.checkpoint(ref)
    if check_only:
        return result
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ("." + lineage + ".registration.lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (Path(runs_root) / "torus9" / "active" / lineage).exists():
            raise ValueError("Diagnostic lineage already exists in active storage")
        if root.exists():
            if json.loads((root / "registration.json").read_text()) != raw:
                raise ValueError("Diagnostic lineage already has different registration")
            resolver.checkpoint(ref)
            return {**result, "state": "REGISTERED"}
        temp = Path(tempfile.mkdtemp(prefix="." + lineage + "-", dir=root.parent))
        try:
            def write(relative, value):
                path = temp / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
                return {"path": relative, "sha256": file_sha256(path)}
            target = temp / ref["path"]
            target.parent.mkdir(parents=True)
            shutil.copyfile(source, target)
            if file_sha256(target) != ref["sha256"] or file_sha256(report_path) != reference["sha256"]:
                raise ValueError("Diagnostic source changed during registration")
            metadata = {k: original_metadata[k] for k in (
                "architecture_config", "architecture_id", "observation_shape", "komi"
            )}
            metadata.update(model_hash=snapshot["model_hash"], artifact_sha256=ref["sha256"],
                            diagnostic_only=True, evaluation_only=True,
                            source_parent=parent.ref.to_dict())
            metadata_ref = write("checkpoints/" + checkpoint_id + ".metadata.json", metadata)
            config = parent.effective_config.config.to_dict()
            config["extensions"] = {"diagnostic_evaluation_only": True}
            config["training"] = {}
            config["replay"] = {}
            effective = EffectiveConfig.from_dict(config)
            config_ref = write("metadata/effective-config.json", effective.to_dict())
            provenance_ref = write("metadata/provenance.json", {
                "schema": "gocube-diagnostic-evaluation-provenance-v1", "registration": raw,
                "source_snapshot": snapshot, "source_parent": parent.ref.to_dict(),
                "arena_metadata": metadata_ref,
                "note": "Evaluation-only root; no fabricated replay or training ancestry."})
            node = {"schema": "gocube-checkpoint-node-v2", "version": 2,
                    "checkpoint": ref, "genesis": True, "parent": None, "fresh_replay": None,
                    "effective_config": {"artifact": config_ref, "fingerprint": effective.fingerprint},
                    "provenance": provenance_ref}
            CheckpointNode.from_dict(node)
            write("metadata/checkpoints/" + checkpoint_id + ".json", node)
            write("registration.json", raw)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            write("manifest.json", {"lineage_id": lineage, "topology": "torus9",
                  "status": "ARCHIVED", "parent_checkpoint": parent.ref.to_dict(),
                  "git_commit": commit, "config_fingerprint": effective.fingerprint,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "checkpoint_hashes": {ref["path"]: ref["sha256"]}, "evaluation_only": True})
            for path in temp.rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            os.rename(temp, root)
            descriptor = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    resolver.checkpoint(ref)
    return {**result, "state": "REGISTERED"}
