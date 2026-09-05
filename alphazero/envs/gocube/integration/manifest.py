from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from alphazero.envs.gocube.game import legacy_game_class
from alphazero.envs.gocube.katago_v3 import (
    KATAGO_JAPANESE_ADJUDICATOR_V3,
    KATAGO_REFERENCE_COMMIT,
    KATAGO_RULES_VERSION,
    OBSERVATION_SCHEMA_V3,
)
from alphazero.envs.gocube.terminal import (
    CONSERVATIVE_AREA_ADJUDICATOR_V1,
    JAPANESE_CLEANUP_ADJUDICATOR_V2,
)

MANIFEST_FILENAME = "gocube-run.json"
RUN_MANIFEST_VERSION = 3
SUPPORTED_MANIFEST_VERSIONS = (1, 2, 3)


class ManifestError(ValueError):
    pass


class ManifestExistsError(ManifestError):
    pass


@dataclass(frozen=True)
class RunManifest:
    version: int
    run_name: str
    topology: str
    size: int
    rule_set: str
    komi: float
    terminal_adjudicator: str
    observation_schema: str | None = None
    rules_fingerprint: str | None = None
    katago_rules_version: int | None = None
    katago_reference_commit: str | None = None

    @classmethod
    def create(
        cls,
        *,
        run_name: str,
        topology: str,
        size: int,
        rule_set: str = "japanese",
        komi: float = 7.5,
        terminal_adjudicator: str | None = None,
    ) -> "RunManifest":
        if terminal_adjudicator is None:
            terminal_adjudicator = (
                CONSERVATIVE_AREA_ADJUDICATOR_V1
                if rule_set == "chinese"
                else KATAGO_JAPANESE_ADJUDICATOR_V3
            )
        if terminal_adjudicator == CONSERVATIVE_AREA_ADJUDICATOR_V1:
            version = 1
        elif terminal_adjudicator == JAPANESE_CLEANUP_ADJUDICATOR_V2:
            version = 2
        elif terminal_adjudicator == KATAGO_JAPANESE_ADJUDICATOR_V3:
            version = 3
        else:
            raise ManifestError(f"Unsupported terminalAdjudicator: {terminal_adjudicator!r}")

        observation_schema = None
        fingerprint = None
        rules_version = None
        reference_commit = None
        if version == 3:
            game_cls = legacy_game_class(topology, size, terminal_adjudicator)
            observation_schema = game_cls.OBSERVATION_SCHEMA
            fingerprint = game_cls.rules_fingerprint()
            rules_version = KATAGO_RULES_VERSION
            reference_commit = KATAGO_REFERENCE_COMMIT

        return cls(
            version=version,
            run_name=run_name,
            topology=topology,
            size=size,
            rule_set=rule_set,
            komi=komi,
            terminal_adjudicator=terminal_adjudicator,
            observation_schema=observation_schema,
            rules_fingerprint=fingerprint,
            katago_rules_version=rules_version,
            katago_reference_commit=reference_commit,
        ).validated()

    @classmethod
    def from_dict(cls, data: object, *, directory_name: str | None = None) -> "RunManifest":
        if not isinstance(data, dict):
            raise ManifestError("Run manifest must be a JSON object")

        required = {
            "version", "runName", "topology", "size", "ruleSet", "komi", "terminalAdjudicator",
        }
        if data.get("version") == 3:
            required |= {
                "observationSchema", "rulesFingerprint", "katagoRulesVersion", "katagoReferenceCommit",
            }
        missing = sorted(required - set(data))
        if missing:
            raise ManifestError(f"Run manifest is missing fields: {', '.join(missing)}")

        manifest = cls(
            version=data["version"],
            run_name=data["runName"],
            topology=data["topology"],
            size=data["size"],
            rule_set=data["ruleSet"],
            komi=data["komi"],
            terminal_adjudicator=data["terminalAdjudicator"],
            observation_schema=data.get("observationSchema"),
            rules_fingerprint=data.get("rulesFingerprint"),
            katago_rules_version=data.get("katagoRulesVersion"),
            katago_reference_commit=data.get("katagoReferenceCommit"),
        ).validated()

        if directory_name is not None and manifest.run_name != directory_name:
            raise ManifestError(
                f"Run manifest runName {manifest.run_name!r} does not match directory {directory_name!r}"
            )
        return manifest

    def validated(self) -> "RunManifest":
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise ManifestError("Run manifest version must be an integer")
        if self.version not in SUPPORTED_MANIFEST_VERSIONS:
            raise ManifestError(f"Unsupported run manifest version: {self.version}")
        if not isinstance(self.run_name, str) or not self.run_name or self.run_name in {".", ".."}:
            raise ManifestError("Run manifest runName must be a non-empty directory name")
        if os.path.basename(self.run_name) != self.run_name:
            raise ManifestError("Run manifest runName must not contain path separators")
        if not isinstance(self.topology, str):
            raise ManifestError("Run manifest topology must be a string")
        if self.topology not in {"cube", "torus"}:
            raise ManifestError(f"Unsupported topology: {self.topology!r}")
        if not isinstance(self.size, int) or isinstance(self.size, bool):
            raise ManifestError("Run manifest size must be an integer")
        if not isinstance(self.rule_set, str):
            raise ManifestError("Run manifest ruleSet must be a string")
        if self.rule_set not in {"chinese", "japanese"}:
            raise ManifestError(f"Unsupported ruleSet: {self.rule_set!r}")
        if not isinstance(self.komi, (int, float)) or isinstance(self.komi, bool) or not math.isfinite(self.komi):
            raise ManifestError("Run manifest komi must be a finite number")
        if not isinstance(self.terminal_adjudicator, str):
            raise ManifestError("Run manifest terminalAdjudicator must be a string")

        expected = {
            1: CONSERVATIVE_AREA_ADJUDICATOR_V1,
            2: JAPANESE_CLEANUP_ADJUDICATOR_V2,
            3: KATAGO_JAPANESE_ADJUDICATOR_V3,
        }[self.version]
        if self.terminal_adjudicator != expected:
            raise ManifestError(
                f"Manifest version {self.version} requires terminalAdjudicator {expected!r}"
            )
        if self.version == 1 and self.rule_set != "chinese":
            raise ManifestError("Manifest version 1 is reserved for legacy Chinese V1 runs")
        if self.version in (2, 3) and self.rule_set != "japanese":
            raise ManifestError(f"Manifest version {self.version} is reserved for Japanese runs")

        try:
            game_cls = legacy_game_class(self.topology, self.size, self.terminal_adjudicator)
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc
        if game_cls.RULESET != self.rule_set:
            raise ManifestError("Manifest ruleSet does not match game class")
        if float(game_cls.KOMI) != float(self.komi):
            raise ManifestError("Manifest komi does not match enabled game komi")
        if game_cls.TERMINAL_ADJUDICATOR_ID != self.terminal_adjudicator:
            raise ManifestError("Manifest terminal adjudicator does not match enabled game class")

        if self.version == 3:
            if self.observation_schema != OBSERVATION_SCHEMA_V3:
                raise ManifestError("Manifest observation schema does not match V3")
            if not isinstance(self.rules_fingerprint, str) or self.rules_fingerprint != game_cls.rules_fingerprint():
                raise ManifestError("Manifest rules fingerprint does not match V3 rules")
            if self.katago_rules_version != KATAGO_RULES_VERSION:
                raise ManifestError("Manifest KataGo rules version does not match V3")
            if self.katago_reference_commit != KATAGO_REFERENCE_COMMIT:
                raise ManifestError("Manifest KataGo reference commit does not match V3")
        return self

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": self.version,
            "runName": self.run_name,
            "topology": self.topology,
            "size": self.size,
            "ruleSet": self.rule_set,
            "komi": float(self.komi),
            "terminalAdjudicator": self.terminal_adjudicator,
        }
        if self.version == 3:
            data.update({
                "observationSchema": self.observation_schema,
                "rulesFingerprint": self.rules_fingerprint,
                "katagoRulesVersion": self.katago_rules_version,
                "katagoReferenceCommit": self.katago_reference_commit,
            })
        return data


def manifest_path(run_dir: str) -> str:
    return os.path.join(run_dir, MANIFEST_FILENAME)


def load_run_manifest(run_dir: str) -> RunManifest:
    path = manifest_path(run_dir)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ManifestError(f"Missing run manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in run manifest: {path}") from exc
    return RunManifest.from_dict(data, directory_name=os.path.basename(os.path.normpath(run_dir)))


def write_run_manifest(run_dir: str, manifest: RunManifest, *, force: bool = False) -> str:
    manifest = manifest.validated()
    directory_name = os.path.basename(os.path.normpath(run_dir))
    if manifest.run_name != directory_name:
        raise ManifestError(
            f"Run manifest runName {manifest.run_name!r} does not match directory {directory_name!r}"
        )
    os.makedirs(run_dir, exist_ok=True)
    path = manifest_path(run_dir)
    if os.path.exists(path):
        try:
            existing = load_run_manifest(run_dir)
        except ManifestError:
            if not force:
                raise
        else:
            if existing == manifest:
                return path
            if not force:
                raise ManifestExistsError(f"Refusing to overwrite incompatible existing run manifest: {path}")
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path


def ensure_training_manifest(checkpoint_dir: str, run_name: str, game_cls) -> RunManifest:
    manifest = RunManifest.create(
        run_name=run_name,
        topology=game_cls.topology_kind(),
        size=game_cls.board_size(),
        rule_set=game_cls.RULESET,
        komi=game_cls.KOMI,
        terminal_adjudicator=game_cls.TERMINAL_ADJUDICATOR_ID,
    )
    run_dir = os.path.join(checkpoint_dir, run_name)
    write_run_manifest(run_dir, manifest, force=False)
    return manifest
