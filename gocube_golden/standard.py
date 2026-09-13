"""Versioned Golden Standard resolution for the standalone Torus 5x5 line.

The older Golden Torus runners predate a canonical ``current`` alias and each
carry parts of their configuration in code.  This module is the small,
explicit resolution boundary for new runs: one current alias, one concrete v2
profile, and opt-in access to preserved legacy profiles.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .provenance import capture_code_identity


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "gocube"
CATALOG_PATH = CONFIG_ROOT / "golden_standards.json"
CURRENT_ALIAS = "gocube-torus5-golden-current"
CURRENT_PRESET_ID = "gocube-torus5-golden-v2"
UNIVERSAL_PROFILE_ID = "gocube-universal-training-v1"
RUN_MANIFEST_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class GoldenStandardError(ValueError):
    """Base error for invalid or unsafe Golden Standard resolution."""


class LegacyGoldenStandardError(GoldenStandardError):
    """Raised when a deprecated preset is selected without explicit consent."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenStandardError(f"Cannot read Golden Standard config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GoldenStandardError(f"Golden Standard config must be a JSON object: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _torus9_universal_projection(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only board-independent settings from the current Torus9 profile."""

    self_play = profile["self_play"]
    arena = profile["arena"]
    return {
        "self_play": {
            "search_implementation_id": "golden-sequential-puct-v1",
            "simulations": self_play["simulations"],
            "cpuct": self_play["cpuct"],
            "fpu": self_play["fpu"],
            "root_noise": self_play["root_noise"],
            "dirichlet_epsilon": self_play["dirichlet_epsilon"],
            "dirichlet_alpha": self_play["dirichlet_alpha"],
            "temperature_plies": self_play["temperature_plies"],
            "temperature_after": self_play["temperature_after"],
            "fast_search": self_play["fast_search"],
            "fast_sims": self_play["fast_sims"],
            "resign": self_play["resign"],
        },
        "arena": {
            "search_implementation_id": "golden-sequential-puct-v1",
            "simulations": arena["simulations"],
            "cpuct": arena["cpuct"],
            "fpu": arena["fpu"],
            "root_noise": arena["noise"],
            "temperature": arena["temperature"],
            "fast_search": arena["fast_search"],
            "resign": arena["resign"],
            "deterministic_tie_break": True,
            "one_game_per_process": arena["one_game_per_process"],
            "technical_fail_closed": arena["technical_fail_closed"],
        },
        # The stable Torus9 runner deliberately uses one inference row and no
        # coalescing.  These are execution semantics, not board geometry.
        "inference": {
            "self_play": {"batch_size": 1, "coalescing": False},
            "arena": {"batch_size": 1, "coalescing": False},
        },
        "training": {
            "optimizer": profile["training"]["optimizer"],
            "learning_rate": profile["training"]["learning_rate"],
            "weight_decay": profile["training"]["weight_decay"],
            "batch_size": profile["training"]["batch_size"],
            "optimizer_steps_per_iteration": profile["training"]["optimizer_steps_per_iteration"],
            "samples_consumed_per_iteration": profile["training"]["samples_consumed_per_iteration"],
            "sample_selection": profile["training"]["sample_selection"],
            "sample_exposure": profile["training"]["sample_exposure"],
            "scheduler": profile["training"]["scheduler"],
            "gating": profile["training"]["gating"],
        },
        "replay": copy.deepcopy(profile["replay"]),
        "iteration": {
            "games_per_iteration": profile["canonical_games_per_iteration"],
            "workers": self_play["workers"],
        },
    }


def load_universal_training_config() -> dict[str, Any]:
    path = CONFIG_ROOT / "universal_training_v1.json"
    config = _read_json(path)
    if config.get("profile_id") != UNIVERSAL_PROFILE_ID or config.get("schema_version") != 1:
        raise GoldenStandardError("Universal training profile identity/schema drift")

    # Import lazily so reading the catalog does not pull in the neural runtime.
    from .torus9_contract import load_torus9_profile, profile_fingerprint

    torus9 = load_torus9_profile()
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise GoldenStandardError("Universal training profile source metadata is missing")
    if source.get("profile_id") != torus9["profile_id"]:
        raise GoldenStandardError("Universal profile no longer points at the current Torus9 profile")
    if source.get("profile_fingerprint") != profile_fingerprint(torus9):
        raise GoldenStandardError("Universal profile source Torus9 fingerprint drift")
    expected = _torus9_universal_projection(torus9)
    if config.get("settings") != expected:
        raise GoldenStandardError(
            "Universal training settings drifted from current Torus9 settings; "
            "update the universal version intentionally"
        )
    return config


@dataclass(frozen=True)
class ResolvedGoldenPreset:
    requested_preset: str
    preset_id: str
    status: str
    source_path: Path
    config: dict[str, Any]
    fingerprint: str

    @property
    def is_legacy(self) -> bool:
        return self.status == "legacy"

    @property
    def resolved_config(self) -> dict[str, Any]:
        return copy.deepcopy(self.config)


def _catalog() -> dict[str, Any]:
    catalog = _read_json(CATALOG_PATH)
    if catalog.get("schema_version") != 1 or catalog.get("standard") != "torus-5x5":
        raise GoldenStandardError("Golden Standard catalog identity/schema drift")
    presets = catalog.get("presets")
    if not isinstance(presets, list) or not presets:
        raise GoldenStandardError("Golden Standard catalog has no presets")
    current = [entry for entry in presets if isinstance(entry, Mapping) and entry.get("status") == "current"]
    if len(current) != 1 or current[0].get("preset_id") != CURRENT_PRESET_ID:
        raise GoldenStandardError("Torus 5x5 must have exactly one current Golden Standard preset")
    if catalog.get("current_alias") != CURRENT_ALIAS:
        raise GoldenStandardError("Torus 5x5 current alias drift")
    if catalog.get("aliases", {}).get(CURRENT_ALIAS) != CURRENT_PRESET_ID:
        raise GoldenStandardError("Torus 5x5 current alias must resolve directly to v2")
    return catalog


def _resolve_alias(catalog: Mapping[str, Any], requested: str) -> str:
    aliases = catalog.get("aliases", {})
    seen: set[str] = set()
    value = requested
    while value in aliases:
        if value in seen:
            raise GoldenStandardError("Golden Standard alias cycle detected")
        seen.add(value)
        value = str(aliases[value])
    return value


def resolve_torus5_golden(
    preset: str | None = None,
    *,
    allow_legacy_config: bool = False,
) -> ResolvedGoldenPreset:
    """Resolve ``current`` or a concrete Torus5 preset.

    Legacy profiles remain readable for reproduction, but selecting one is an
    explicit operation and never happens through the default/current path.
    """

    catalog = _catalog()
    requested = "current" if preset is None else str(preset)
    preset_id = _resolve_alias(catalog, requested)
    entries = {
        str(entry["preset_id"]): entry
        for entry in catalog["presets"]
        if isinstance(entry, Mapping) and "preset_id" in entry
    }
    entry = entries.get(preset_id)
    if entry is None:
        raise GoldenStandardError(f"Unknown Torus 5x5 Golden Standard preset: {requested}")
    status = str(entry.get("status"))
    source_path = ROOT / str(entry.get("path", ""))
    if status == "legacy" and not allow_legacy_config:
        raise LegacyGoldenStandardError(
            f"{preset_id} is deprecated legacy configuration. "
            "Use --allow-legacy-config only to reproduce historical results."
        )
    if status not in {"current", "legacy"}:
        raise GoldenStandardError(f"Unsupported Golden Standard status for {preset_id}: {status!r}")
    raw = _read_json(source_path)
    if status == "legacy":
        resolved = {
            "preset_id": preset_id,
            "status": "legacy",
            "legacy_profile_id": entry.get("legacy_profile_id"),
            "legacy_source_path": str(source_path.relative_to(ROOT)),
            "legacy_source": raw,
        }
    else:
        universal = load_universal_training_config()
        if raw.get("preset_id") != preset_id or raw.get("extends") != UNIVERSAL_PROFILE_ID:
            raise GoldenStandardError("Current Torus 5x5 preset identity/inheritance drift")
        resolved = _deep_merge(
            {
                "preset_id": preset_id,
                "status": "current",
                "universal_profile_id": UNIVERSAL_PROFILE_ID,
                "universal": copy.deepcopy(universal["settings"]),
            },
            {
                "board": copy.deepcopy(raw["board"]),
                "lineage": {
                    "current_alias": CURRENT_ALIAS,
                    "concrete_preset": preset_id,
                    "universal_source_profile": universal["source"],
                },
            },
        )
        validate_current_torus5_config(resolved)
    return ResolvedGoldenPreset(
        requested_preset=requested,
        preset_id=preset_id,
        status=status,
        source_path=source_path,
        config=resolved,
        fingerprint=config_fingerprint(resolved),
    )


def validate_current_torus5_config(config: Mapping[str, Any]) -> None:
    if config.get("preset_id") != CURRENT_PRESET_ID or config.get("status") != "current":
        raise GoldenStandardError("Current Torus 5x5 preset identity drift")
    board = config.get("board")
    if not isinstance(board, Mapping):
        raise GoldenStandardError("Current Torus 5x5 board-specific section is missing")
    topology = board.get("topology", {})
    if not isinstance(topology, Mapping):
        raise GoldenStandardError("Current Torus 5x5 topology section is invalid")
    if topology.get("width") != 5 or topology.get("height") != 5 or topology.get("point_count") != 25:
        raise GoldenStandardError("Current Torus 5x5 topology dimensions drift")
    if topology.get("production_torus_factory") is not False:
        raise GoldenStandardError("Current Torus 5x5 must remain a standalone research topology")
    rules = board.get("rules", {})
    if not isinstance(rules, Mapping):
        raise GoldenStandardError("Current Torus 5x5 rules section is invalid")
    if rules.get("komi") != 0.5 or rules.get("komi") == 7.5:
        raise GoldenStandardError("Current Torus 5x5 komi must be exactly 0.5")
    network = board.get("network", {})
    if not isinstance(network, Mapping):
        raise GoldenStandardError("Current Torus 5x5 network section is invalid")
    if network.get("channels") != 48 or network.get("hidden") != 48 or network.get("blocks") != 6:
        raise GoldenStandardError("Current Torus 5x5 network must be 48 channels x 6 blocks")
    if network.get("heads") != {"policy": [26], "value": [3]}:
        raise GoldenStandardError("Current Torus 5x5 network head shapes drift")
    universal = config.get("universal")
    if not isinstance(universal, Mapping):
        raise GoldenStandardError("Current Torus 5x5 universal section is missing")
    training = universal.get("training")
    replay = universal.get("replay")
    if not isinstance(training, Mapping) or not isinstance(replay, Mapping):
        raise GoldenStandardError("Current Torus 5x5 universal training/replay sections are invalid")
    if training.get("batch_size") != 64:
        raise GoldenStandardError("Current Torus 5x5 universal training batch drift")
    if training.get("optimizer_steps_per_iteration") != 80:
        raise GoldenStandardError("Current Torus 5x5 universal optimizer budget drift")
    if replay.get("generations") != 3:
        raise GoldenStandardError("Current Torus 5x5 replay window drift")


def build_torus5_model(
    preset: str | None = None,
    *,
    allow_legacy_config: bool = False,
):
    """Construct the network declared by a resolved preset for smoke checks."""

    resolved = resolve_torus5_golden(preset, allow_legacy_config=allow_legacy_config)
    from .neural import GoldenGraphNetV1, Torus5GoldenGraphNetV2

    if resolved.is_legacy:
        network = resolved.config["legacy_source"].get("network", {})
        return GoldenGraphNetV1(
            hidden=int(network.get("hidden", 64)),
            blocks=int(network.get("blocks", 4)),
        )
    return Torus5GoldenGraphNetV2()


def build_run_metadata(
    *,
    run_name: str,
    preset: str | None = None,
    allow_legacy_config: bool = False,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    if not run_name or run_name in {".", ".."} or "/" in run_name:
        raise GoldenStandardError("Torus 5x5 run_name must be a simple non-empty name")
    resolved = resolve_torus5_golden(preset, allow_legacy_config=allow_legacy_config)
    code = capture_code_identity(ROOT)
    metadata: dict[str, Any] = {
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_name": run_name,
        "argv": list(argv or []),
        "git_sha": code.git_commit_sha,
        "git_tree_sha": code.git_tree_sha,
        "git_worktree_clean": code.working_tree_clean,
        "requested_preset": resolved.requested_preset,
        "resolved_preset_id": resolved.preset_id,
        "preset_status": resolved.status,
        "resolved_config": resolved.resolved_config,
        "resolved_config_sha256": resolved.fingerprint,
        "legacy_override_used": bool(resolved.is_legacy and allow_legacy_config),
    }
    if not resolved.is_legacy:
        board = resolved.config["board"]
        metadata.update(
            {
                "golden_standard": {
                    "standard": "torus-5x5",
                    "alias": CURRENT_ALIAS,
                    "concrete_version": CURRENT_PRESET_ID,
                    "config_fingerprint": resolved.fingerprint,
                },
                "board_size": [board["topology"]["width"], board["topology"]["height"]],
                "network_channels": board["network"]["channels"],
                "network_blocks": board["network"]["blocks"],
                "komi": board["rules"]["komi"],
                "universal_training": resolved.config["universal"]["training"],
                "universal_replay": resolved.config["universal"]["replay"],
                "universal_self_play": resolved.config["universal"]["self_play"],
                "universal_arena": resolved.config["universal"]["arena"],
                "inference": resolved.config["universal"]["inference"],
            }
        )
    return metadata


def validate_run_metadata(metadata: Mapping[str, Any]) -> None:
    required = {
        "manifest_schema_version", "run_name", "requested_preset", "resolved_preset_id",
        "resolved_config", "resolved_config_sha256", "git_sha", "git_tree_sha",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise GoldenStandardError("Run metadata is missing: " + ", ".join(missing))
    if metadata["manifest_schema_version"] != RUN_MANIFEST_SCHEMA_VERSION:
        raise GoldenStandardError("Unsupported Golden Standard run manifest schema")
    if not _SHA256_RE.fullmatch(str(metadata["resolved_config_sha256"])):
        raise GoldenStandardError("Run metadata resolved config fingerprint is malformed")
    if config_fingerprint(metadata["resolved_config"]) != metadata["resolved_config_sha256"]:
        raise GoldenStandardError("Run metadata resolved config fingerprint does not recompute")
    if metadata["resolved_preset_id"] == CURRENT_PRESET_ID:
        validate_current_torus5_config(metadata["resolved_config"])
        if metadata.get("network_channels") != 48 or metadata.get("network_blocks") != 6:
            raise GoldenStandardError("Run metadata network identity does not match current v2")
        if metadata.get("komi") != 0.5:
            raise GoldenStandardError("Run metadata komi does not match current v2")


def write_run_manifest(
    path: str | Path,
    *,
    run_name: str,
    preset: str | None = None,
    allow_legacy_config: bool = False,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing Golden Standard run manifest: {destination}")
    metadata = build_run_metadata(
        run_name=run_name,
        preset=preset,
        allow_legacy_config=allow_legacy_config,
        argv=argv,
    )
    validate_run_metadata(metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
