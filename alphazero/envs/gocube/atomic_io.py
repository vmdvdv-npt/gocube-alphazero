from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import uuid
from glob import glob

import torch

from .contract_versions import (
    OWNERSHIP_TARGET_SEMANTICS,
    REPLAY_FORMAT_VERSION,
    SCORE_INITIALIZATION_CONTRACT,
    SCORE_TARGET_SEMANTICS,
    TARGET_PROVENANCE_ENCODING,
    TARGET_PROVENANCE_SEMANTICS,
    TERMINATION_CONTRACT,
    VALUE_TARGET_SEMANTICS,
)
from .replay_provenance import (
    REPLAY_TARGET_PROVENANCE_SUFFIX,
    validate_target_provenance_tensor,
)


RECOVERY_CONTRACT = "gocube-atomic-recovery-v1"
REPLAY_MARKER_SCHEMA_VERSION = REPLAY_FORMAT_VERSION
REPLAY_TENSOR_SUFFIXES = (
    "-data.pkl",
    "-policy.pkl",
    "-value.pkl",
    "-score.pkl",
    "-score-mask.pkl",
    "-ownership.pkl",
    "-ownership-mask.pkl",
)
REPLAY_ARTIFACT_SUFFIXES = REPLAY_TENSOR_SUFFIXES + (REPLAY_TARGET_PROVENANCE_SUFFIX,)
_CHECKPOINT_RE = re.compile(r"^iteration-(\d+)\.pkl$")


def _fsync_directory(path: str | os.PathLike[str]) -> None:
    directory = os.fspath(path) or "."
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_file(path: str | os.PathLike[str]) -> None:
    fd = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def temporary_sibling(path: str | os.PathLike[str], *, tag: str = "tmp") -> str:
    path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(path)
    base = os.path.basename(path)
    return os.path.join(parent, f".{base}.{tag}-{os.getpid()}-{uuid.uuid4().hex}")


def promote_staged_file(staged: str, target: str) -> None:
    """Durably promote a fully-written same-filesystem staging file."""

    staged = os.path.abspath(staged)
    target = os.path.abspath(target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fsync_file(staged)
    os.replace(staged, target)
    _fsync_directory(os.path.dirname(target))


def atomic_json_write(payload: object, path: str | os.PathLike[str]) -> None:
    target = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = temporary_sibling(target, tag="json")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(os.path.dirname(target))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(payload: object, path: str | os.PathLike[str], *, pickle_protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    target = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = temporary_sibling(target, tag="torch")
    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle, pickle_protocol=pickle_protocol)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(os.path.dirname(target))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def replay_marker_path(iteration_base: str | os.PathLike[str]) -> str:
    return os.fspath(iteration_base) + "-complete.json"


def remove_replay_marker(iteration_base: str | os.PathLike[str]) -> None:
    marker = replay_marker_path(iteration_base)
    if os.path.exists(marker):
        os.unlink(marker)
        _fsync_directory(os.path.dirname(os.path.abspath(marker)))


def write_replay_marker(iteration_base: str, *, iteration: int, row_count: int) -> str:
    row_count = int(row_count)
    if row_count < 0:
        raise ValueError("Replay marker row count must be non-negative")
    _load_target_provenance_sidecar(iteration_base, expected_rows=row_count)
    marker = replay_marker_path(iteration_base)
    atomic_json_write(
        {
            "schema_version": REPLAY_MARKER_SCHEMA_VERSION,
            "replay_format_version": REPLAY_FORMAT_VERSION,
            "recovery_contract": RECOVERY_CONTRACT,
            "score_initialization_contract": SCORE_INITIALIZATION_CONTRACT,
            "value_target_semantics": VALUE_TARGET_SEMANTICS,
            "score_target_semantics": SCORE_TARGET_SEMANTICS,
            "ownership_target_semantics": OWNERSHIP_TARGET_SEMANTICS,
            "target_provenance_semantics": TARGET_PROVENANCE_SEMANTICS,
            "target_provenance_encoding": TARGET_PROVENANCE_ENCODING,
            "target_provenance_sidecar": REPLAY_TARGET_PROVENANCE_SUFFIX,
            "target_provenance_rows": row_count,
            "termination_contract": TERMINATION_CONTRACT,
            "iteration": int(iteration),
            "row_count": row_count,
            "tensor_suffixes": list(REPLAY_TENSOR_SUFFIXES),
        },
        marker,
    )
    return marker


def load_replay_marker(iteration_base: str | os.PathLike[str]) -> dict[str, object]:
    marker = replay_marker_path(iteration_base)
    with open(marker, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Replay completion marker must be an object: {marker}")
    if payload.get("schema_version") == 1 or payload.get("replay_format_version") == 1:
        raise ValueError(
            "Replay v1 cannot be used with win/loss/no-result training contract: "
            "NO_RESULT policy/value rows were not retained and cannot be reconstructed. "
            "Use a fresh run or an explicit legacy evaluation path."
        )
    if payload.get("schema_version") != REPLAY_MARKER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported replay completion marker: {marker}")
    if payload.get("replay_format_version") != REPLAY_FORMAT_VERSION:
        raise ValueError(f"Replay format version mismatch: {marker}")
    if payload.get("recovery_contract") != RECOVERY_CONTRACT:
        raise ValueError(f"Replay marker does not use {RECOVERY_CONTRACT}: {marker}")
    suffixes = tuple(payload.get("tensor_suffixes", ()))
    if suffixes != REPLAY_TENSOR_SUFFIXES:
        raise ValueError(f"Replay marker tensor set mismatch: {marker}")
    semantic_fields = {
        "value_target_semantics": VALUE_TARGET_SEMANTICS,
        "score_target_semantics": SCORE_TARGET_SEMANTICS,
        "ownership_target_semantics": OWNERSHIP_TARGET_SEMANTICS,
        "score_initialization_contract": SCORE_INITIALIZATION_CONTRACT,
        "target_provenance_semantics": TARGET_PROVENANCE_SEMANTICS,
        "target_provenance_encoding": TARGET_PROVENANCE_ENCODING,
        "termination_contract": TERMINATION_CONTRACT,
    }
    for key, expected in semantic_fields.items():
        if payload.get(key) != expected:
            raise ValueError(f"Replay marker {key} mismatch: {marker}")
    try:
        row_count = int(payload.get("row_count", -1))
        provenance_rows = int(payload.get("target_provenance_rows", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Replay marker row count is invalid: {marker}") from exc
    if row_count < 0:
        raise ValueError(f"Replay marker row count is invalid: {marker}")
    if payload.get("target_provenance_sidecar") != REPLAY_TARGET_PROVENANCE_SUFFIX:
        raise ValueError(f"Replay marker target provenance sidecar mismatch: {marker}")
    if provenance_rows != row_count:
        raise ValueError(f"Replay marker target provenance row count mismatch: {marker}")
    _load_target_provenance_sidecar(iteration_base, expected_rows=row_count)
    return payload


def _trusted_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_target_provenance_sidecar(
    iteration_base: str | os.PathLike[str],
    *,
    expected_rows: int | None = None,
) -> torch.Tensor:
    path = os.fspath(iteration_base) + REPLAY_TARGET_PROVENANCE_SUFFIX
    try:
        tensor = _trusted_torch_load(path)
    except FileNotFoundError as exc:
        raise ValueError(f"Replay target provenance sidecar is missing: {path}") from exc
    except Exception as exc:
        raise ValueError(f"Replay target provenance sidecar is unreadable: {path}") from exc
    try:
        return validate_target_provenance_tensor(tensor, expected_rows=expected_rows)
    except ValueError as exc:
        raise ValueError(f"Invalid replay target provenance sidecar {path}: {exc}") from exc


def load_replay_target_provenance(
    iteration_base: str | os.PathLike[str],
    *,
    marker: dict[str, object] | None = None,
) -> torch.Tensor:
    """Load the row-aligned provenance sidecar for a current replay artifact."""

    if marker is None:
        marker = load_replay_marker(iteration_base)
    if not isinstance(marker, dict):
        raise ValueError("Replay marker must be an object")
    if marker.get("schema_version") != REPLAY_MARKER_SCHEMA_VERSION:
        raise ValueError("Replay target provenance requires the current marker schema")
    if marker.get("replay_format_version") != REPLAY_FORMAT_VERSION:
        raise ValueError("Replay target provenance requires the current replay format")
    if marker.get("recovery_contract") != RECOVERY_CONTRACT:
        raise ValueError("Replay target provenance recovery contract mismatch")
    if tuple(marker.get("tensor_suffixes", ())) != REPLAY_TENSOR_SUFFIXES:
        raise ValueError("Replay target provenance tensor set mismatch")
    if marker.get("target_provenance_encoding") != TARGET_PROVENANCE_ENCODING:
        raise ValueError("Replay target provenance encoding mismatch")
    if marker.get("target_provenance_sidecar") != REPLAY_TARGET_PROVENANCE_SUFFIX:
        raise ValueError("Replay target provenance sidecar mismatch")
    if marker.get("target_provenance_semantics") != TARGET_PROVENANCE_SEMANTICS:
        raise ValueError("Replay target provenance semantics mismatch")
    if marker.get("termination_contract") != TERMINATION_CONTRACT:
        raise ValueError("Replay termination contract mismatch")
    try:
        row_count = int(marker.get("row_count", -1))
        provenance_rows = int(marker.get("target_provenance_rows", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Replay target provenance row count is invalid") from exc
    if provenance_rows != row_count:
        raise ValueError("Replay target provenance row count mismatch")
    return _load_target_provenance_sidecar(
        iteration_base,
        expected_rows=row_count,
    )


def checkpoint_iteration(path: str | os.PathLike[str]) -> int | None:
    match = _CHECKPOINT_RE.match(os.path.basename(os.fspath(path)))
    return int(match.group(1)) if match else None


def checkpoint_is_structurally_valid(path: str | os.PathLike[str]) -> bool:
    try:
        checkpoint = _trusted_torch_load(os.fspath(path))
    except Exception:
        return False
    return isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict)


def find_last_valid_contiguous_checkpoint(folder: str | os.PathLike[str]) -> tuple[int | None, list[int]]:
    """Return the last valid 0..N checkpoint and trailing iterations ignored.

    A corrupt or missing iteration ends the resumable prefix. Later files are
    deliberately ignored so a crash cannot make ``len(glob(...))`` skip ahead.
    Full training/search-contract validation still happens when the selected
    checkpoint is loaded by the network wrapper.
    """

    folder = os.fspath(folder)
    found: dict[int, str] = {}
    for path in glob(os.path.join(folder, "iteration-*.pkl")):
        iteration = checkpoint_iteration(path)
        if iteration is not None:
            found[iteration] = path
    if not found:
        return None, []

    last_valid = None
    expected = 0
    while expected in found and checkpoint_is_structurally_valid(found[expected]):
        last_valid = expected
        expected += 1

    if last_valid is None:
        raise RuntimeError(
            f"No valid iteration-0000.pkl in checkpoint namespace {os.path.abspath(folder)}"
        )
    ignored = sorted(iteration for iteration in found if iteration > last_valid)
    return last_valid, ignored


def make_staging_directory(parent: str | os.PathLike[str], *, prefix: str) -> str:
    parent = os.path.abspath(os.fspath(parent))
    os.makedirs(parent, exist_ok=True)
    path = os.path.join(parent, f".{prefix}.staging-{os.getpid()}-{uuid.uuid4().hex}")
    os.makedirs(path)
    return path


def cleanup_staging_directory(path: str | os.PathLike[str]) -> None:
    shutil.rmtree(os.fspath(path), ignore_errors=True)
