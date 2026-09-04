from __future__ import annotations

import os
import re
from dataclasses import dataclass

from alphazero.utils import get_iter_file

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
                descriptors.append(
                    CheckpointDescriptor.from_manifest(
                        manifest,
                        iteration=iteration,
                        path=path,
                    )
                )

        descriptors.sort(key=lambda item: (item.run_name, item.iteration))
        return descriptors

    def get(self, checkpoint_id: str) -> CheckpointDescriptor | None:
        if not isinstance(checkpoint_id, str):
            return None
        for descriptor in self.list():
            if descriptor.checkpoint_id == checkpoint_id:
                return descriptor
        return None
