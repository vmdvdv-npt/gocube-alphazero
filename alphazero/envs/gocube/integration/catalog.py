from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace

from alphazero.utils import get_iter_file

from .contract import ContractError, ResolvedGoCubeContract
from .manifest import ManifestError, RunManifest, load_run_manifest

_CHECKPOINT_RE = re.compile(r"^iteration-(\d{4,})\.pkl$")


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

        descriptors.sort(key=lambda item: (item.run_name, item.iteration))
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
