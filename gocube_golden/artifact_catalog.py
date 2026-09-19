"""Durable catalog of immutable artifacts committed by a training lineage.

The catalog is deliberately small and append-oriented.  A generation driver
publishes already-validated artifact identities, and the supervisor records
them once at the generation commit boundary.  Consumers can then verify the
few artifacts relevant to an operation without walking all historical files.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_CATALOG_SCHEMA = "gocube-training-artifact-catalog-v1"
ARTIFACT_CATALOG_VERSION = 1
ARTIFACT_VALIDATION_SCHEMA = "torus9-replay-validation-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_relative(value: object) -> str:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Artifact catalog path must be safe and relative: {value!r}")
    return path.as_posix()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


class ArtifactCatalog:
    """Read and append immutable artifact identities for one lineage."""

    def __init__(self, path: str | Path, *, root: str | Path | None = None) -> None:
        self.path = Path(path)
        self.root = Path(root).resolve() if root is not None else self.path.parent.parent.resolve()
        self._payload: dict[str, Any] = {}

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        *,
        lineage_id: str,
        root: str | Path,
    ) -> "ArtifactCatalog":
        catalog = cls(path, root=root)
        catalog._payload = {
            "schema": ARTIFACT_CATALOG_SCHEMA,
            "version": ARTIFACT_CATALOG_VERSION,
            "lineage_id": str(lineage_id),
            "validation_schema": ARTIFACT_VALIDATION_SCHEMA,
            "entries": {},
            "generations": {},
        }
        catalog._write()
        return catalog

    @classmethod
    def load(cls, path: str | Path, *, root: str | Path | None = None) -> "ArtifactCatalog":
        catalog = cls(path, root=root)
        try:
            payload = json.loads(catalog.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read artifact catalog: {catalog.path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Artifact catalog must be a JSON object")
        catalog._payload = payload
        catalog._validate()
        return catalog

    @property
    def payload(self) -> Mapping[str, object]:
        return self._payload

    @property
    def entries(self) -> Mapping[str, Mapping[str, object]]:
        entries = self._payload.get("entries")
        if not isinstance(entries, Mapping):
            raise ValueError("Artifact catalog entries are malformed")
        return entries  # type: ignore[return-value]

    @property
    def fingerprint(self) -> str:
        return str(self._payload.get("catalog_fingerprint", self._computed_fingerprint()))

    def _computed_fingerprint(self) -> str:
        return _fingerprint(
            {
                key: value
                for key, value in self._payload.items()
                if key != "catalog_fingerprint"
            }
        )

    def _validate(self) -> None:
        if self._payload.get("schema") != ARTIFACT_CATALOG_SCHEMA:
            raise ValueError("Artifact catalog schema mismatch")
        if self._payload.get("version") != ARTIFACT_CATALOG_VERSION:
            raise ValueError("Artifact catalog version mismatch")
        if not isinstance(self._payload.get("lineage_id"), str) or not self._payload["lineage_id"]:
            raise ValueError("Artifact catalog lineage_id is missing")
        if self._payload.get("validation_schema") != ARTIFACT_VALIDATION_SCHEMA:
            raise ValueError("Artifact catalog validation schema mismatch")
        if not isinstance(self._payload.get("entries"), Mapping):
            raise ValueError("Artifact catalog entries are malformed")
        if not isinstance(self._payload.get("generations"), Mapping):
            raise ValueError("Artifact catalog generations are malformed")
        stored = self._payload.get("catalog_fingerprint")
        if stored != self._computed_fingerprint():
            raise ValueError("Artifact catalog fingerprint mismatch")
        for raw_path, identity in self.entries.items():
            path = _safe_relative(raw_path)
            if not isinstance(identity, Mapping):
                raise ValueError(f"Artifact catalog identity is malformed: {path}")
            if not str(identity.get("sha256", "")).startswith("sha256:"):
                raise ValueError(f"Artifact catalog SHA is missing: {path}")
            if int(identity.get("size_bytes", -1)) < 0:
                raise ValueError(f"Artifact catalog size is malformed: {path}")

    def _write(self) -> None:
        self._payload["catalog_fingerprint"] = self._computed_fingerprint()
        self._validate()
        _atomic_json(self.path, self._payload)

    def identity(self, path: str | Path) -> Mapping[str, object]:
        relative = _safe_relative(path)
        identity = self.entries.get(relative)
        if identity is None:
            raise ValueError(f"Artifact is not committed in catalog: {relative}")
        return identity

    def register_generation(
        self,
        generation: int,
        artifacts: Iterable[Mapping[str, object]],
        *,
        transaction: Mapping[str, object] | None = None,
    ) -> str:
        """Record a committed generation without rehashing its artifacts.

        Artifact hashes come from the generation result, which the generic
        supervisor has just verified against disk.  Rehashing here would
        recreate the replay overhead this catalog is intended to remove.
        """
        entries = dict(self.entries)
        generation_paths: list[str] = []
        for raw in artifacts:
            if not isinstance(raw, Mapping):
                raise ValueError("Artifact catalog entry must be an object")
            relative = _safe_relative(raw.get("path", ""))
            sha = str(raw.get("sha256", ""))
            size = int(raw.get("size_bytes", -1))
            if not sha.startswith("sha256:") or len(sha) != len("sha256:") + 64:
                raise ValueError(f"Artifact catalog SHA is malformed: {relative}")
            if size < 0:
                raise ValueError(f"Artifact catalog size is malformed: {relative}")
            destination = (self.root / relative).resolve()
            if self.root not in destination.parents or not destination.is_file():
                raise ValueError(f"Committed artifact is missing or escapes lineage: {relative}")
            if destination.stat().st_size != size:
                raise ValueError(f"Committed artifact size changed before catalog registration: {relative}")
            identity = dict(raw)
            identity.update({"path": relative, "sha256": sha, "size_bytes": size})
            existing = entries.get(relative)
            if existing is not None and dict(existing) != identity:
                raise ValueError(f"Immutable artifact identity changed in catalog: {relative}")
            entries[relative] = identity
            generation_paths.append(relative)

        generations = dict(self._payload.get("generations", {}))
        key = str(int(generation))
        existing_generation = generations.get(key)
        generation_record: dict[str, object] = {
            "generation": int(generation),
            "artifact_paths": sorted(set(generation_paths)),
        }
        if transaction is not None:
            generation_record["transaction"] = dict(transaction)
        if existing_generation is not None and dict(existing_generation) != generation_record:
            raise ValueError(f"Generation {generation} catalog record changed")
        generations[key] = generation_record
        self._payload["entries"] = entries
        self._payload["generations"] = generations
        self._write()
        return self.fingerprint

    def discard_generation(self, generation: int) -> None:
        """Remove pre-fence evidence for an interrupted generation."""
        key = str(int(generation))
        generations = dict(self._payload.get("generations", {}))
        record = generations.pop(key, None)
        if record is None:
            return
        referenced: set[str] = set()
        for other in generations.values():
            if isinstance(other, Mapping):
                paths = other.get("artifact_paths", ())
                if isinstance(paths, Sequence):
                    referenced.update(str(path) for path in paths)
        entries = dict(self.entries)
        paths = record.get("artifact_paths", ()) if isinstance(record, Mapping) else ()
        if isinstance(paths, Sequence):
            for path in paths:
                if str(path) not in referenced:
                    entries.pop(str(path), None)
        self._payload["entries"] = entries
        self._payload["generations"] = generations
        self._write()

    def verify(self, paths: Sequence[str | Path]) -> dict[str, str]:
        """Verify only selected committed files and return their SHA values."""
        verified: dict[str, str] = {}
        for raw_path in paths:
            relative = _safe_relative(raw_path)
            identity = self.identity(relative)
            destination = (self.root / relative).resolve()
            if self.root not in destination.parents or not destination.is_file():
                raise ValueError(f"Committed artifact is missing: {relative}")
            if destination.stat().st_size != int(identity["size_bytes"]):
                raise ValueError(f"Committed artifact size mismatch: {relative}")
            actual = sha256_file(destination)
            if actual != identity["sha256"]:
                raise ValueError(f"Committed artifact hash mismatch: {relative}")
            verified[relative] = actual
        return verified


__all__ = [
    "ARTIFACT_CATALOG_SCHEMA",
    "ARTIFACT_CATALOG_VERSION",
    "ARTIFACT_VALIDATION_SCHEMA",
    "ArtifactCatalog",
    "sha256_file",
]
