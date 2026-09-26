"""Durable Torus9 komi calibration and post-M137 handoff.

This module is deliberately separate from :mod:`experiment_runner`.  A komi
calibration is an evaluation of one immutable checkpoint under two rules
contracts, not an A/B training experiment and not a model-gating decision.

The runner persists every scientific boundary before doing work.  A process
restart therefore re-enters the state machine and never relies on an in-memory
"already ran" flag.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Protocol

from ..artifact_graph import ArtifactRef, CheckpointRef, EffectiveConfig
from ..process_supervision import atomic_write_text
from ..provenance import canonical_json, sha256_fingerprint
from .arena_runner import ArenaRunRequest, ArenaRunResult, ArenaRunnerV2
from .artifact_resolver import ArtifactResolver, ResolvedCheckpointNode, ResolvedEffectiveConfig
from .generation_runner import OutputLineage
from .experiment_runner import LineageFactory
from .torus9_production import Torus9ProductionLineage
from .version import ORCHESTRATOR_VERSION
from ..torus9_contract import TORUS9_ALLOWED_KOMI

from tools.arena_engine import ArenaExecutionConfig


KOMI_CALIBRATION_TYPE = "komi_calibration"
KOMI_CALIBRATION_SCHEMA = "gocube-komi-calibration-v1"
KOMI_CALIBRATION_STATE_SCHEMA = "gocube-komi-calibration-state-v1"
KOMI_CALIBRATION_INITIAL_GAMES = 1024
KOMI_CALIBRATION_EXTENSION_GAMES = 1024
KOMI_CALIBRATION_AMBIGUITY_THRESHOLD = 0.01
KOMI_CALIBRATION_PARENT_GENERATION = 137
KOMI_CALIBRATION_PARENT_CHECKPOINT_ID = "M137"
KOMI_CALIBRATION_CANDIDATES = (1.5, 2.5)
KOMI_CALIBRATION_ALLOWED_CANDIDATES = tuple(sorted(TORUS9_ALLOWED_KOMI))

WAITING_FOR_M137 = "WAITING_FOR_M137"
STOPPING_PARENT = "STOPPING_PARENT"
M137_PINNED = "M137_PINNED"
CALIBRATION_KOMI_1_5 = "CALIBRATION_KOMI_1_5"
CALIBRATION_KOMI_2_5 = "CALIBRATION_KOMI_2_5"
CALIBRATION_CANDIDATE = "CALIBRATION_CANDIDATE"
CALIBRATION_EXTENSION = "CALIBRATION_EXTENSION"
KOMI_SELECTED = "KOMI_SELECTED"
CHILD_LINEAGE_CREATED = "CHILD_LINEAGE_CREATED"
TRAINING_RESUMED = "TRAINING_RESUMED"
COMPLETE_HANDOFF = "COMPLETE_HANDOFF"
CALIBRATION_FAILED = "CALIBRATION_FAILED"

KOMI_CALIBRATION_STAGES = (
    WAITING_FOR_M137,
    STOPPING_PARENT,
    M137_PINNED,
    CALIBRATION_KOMI_1_5,
    CALIBRATION_KOMI_2_5,
    CALIBRATION_CANDIDATE,
    CALIBRATION_EXTENSION,
    KOMI_SELECTED,
    CHILD_LINEAGE_CREATED,
    TRAINING_RESUMED,
    COMPLETE_HANDOFF,
)


class KomiCalibrationError(RuntimeError):
    """Calibration cannot safely make or apply a scientific decision."""


class ChildTrainingHandoff(Protocol):
    def __call__(
        self,
        *,
        parent: ResolvedCheckpointNode,
        config: ResolvedEffectiveConfig,
        output_lineage: OutputLineage,
        first_generation: int,
    ) -> object: ...


class ParentSupplier(Protocol):
    def __call__(self) -> ResolvedCheckpointNode | None: ...


@dataclass(frozen=True)
class KomiCalibrationArenaContract:
    """The fully resolved scientific and execution contract for calibration."""

    simulations: int = 64
    cpuct: float = 1.25
    fpu: float = 0.0
    root_noise: bool = False
    temperature: float = 0.0
    fast_search: bool = False
    resign: bool = False
    watchdog: int = 1000
    deterministic_tie_break: bool = True
    batching: bool = True
    workers: int = 16
    active_contexts: int = 192
    inference_batch_cap: int = 64
    inference_wait_ms: float = 4.0

    def __post_init__(self) -> None:
        if type(self.simulations) is not int or self.simulations <= 0:
            raise ValueError("Komi calibration simulations must be a positive integer")
        if not math.isfinite(float(self.cpuct)) or float(self.cpuct) <= 0:
            raise ValueError("Komi calibration cpuct must be finite and positive")
        if not math.isfinite(float(self.fpu)):
            raise ValueError("Komi calibration fpu must be finite")
        if self.root_noise or float(self.temperature) != 0.0 or self.fast_search or self.resign:
            raise ValueError("Komi calibration search switches must be OFF/zero")
        if type(self.watchdog) is not int or self.watchdog <= 0:
            raise ValueError("Komi calibration watchdog must be a positive integer")
        if not self.deterministic_tie_break or not self.batching:
            raise ValueError("Komi calibration requires deterministic tie-break and batching")
        if type(self.workers) is not int or self.workers <= 0:
            raise ValueError("Komi calibration workers must be a positive integer")
        if type(self.active_contexts) is not int or self.active_contexts <= 0:
            raise ValueError("Komi calibration active_contexts must be a positive integer")
        if self.active_contexts < self.workers:
            raise ValueError("Komi calibration active_contexts must cover all workers")
        if type(self.inference_batch_cap) is not int or self.inference_batch_cap <= 0:
            raise ValueError("Komi calibration inference_batch_cap must be positive")
        if not math.isfinite(float(self.inference_wait_ms)) or float(self.inference_wait_ms) < 0:
            raise ValueError("Komi calibration inference wait must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "simulations": self.simulations,
            "cpuct": self.cpuct,
            "fpu": self.fpu,
            "root_noise": self.root_noise,
            "temperature": self.temperature,
            "fast_search": self.fast_search,
            "resign": self.resign,
            "watchdog": self.watchdog,
            "deterministic_tie_break": self.deterministic_tie_break,
            "batching": self.batching,
            "workers": self.workers,
            "active_contexts": self.active_contexts,
            "inference_batch_cap": self.inference_batch_cap,
            "inference_wait_ms": self.inference_wait_ms,
        }

    def scientific_contract(self, *, komi: float, games: int) -> dict[str, object]:
        return {
            "experiment_type": KOMI_CALIBRATION_TYPE,
            "profile": "torus9",
            "komi": float(komi),
            "games": int(games),
            **self.to_dict(),
            "technical_outcomes": "fail-closed / excluded from denominator",
            "paired_starts_color_swap": True,
        }

    def execution_config(self, games: int) -> ArenaExecutionConfig:
        if int(games) <= 0 or int(games) % 2:
            raise ValueError("Komi calibration games must be a positive even number")
        return ArenaExecutionConfig(
            games=int(games),
            workers=self.workers,
            games_per_worker=max(1, self.active_contexts // self.workers),
            inference_batch_rows=self.inference_batch_cap,
            inference_batch_wait_ms=self.inference_wait_ms,
            device="cuda",
            strict_production=True,
        )


def _safe_component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


def _komi(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("komi must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result not in KOMI_CALIBRATION_ALLOWED_CANDIDATES:
        raise ValueError("komi must be one of 1.5, 2.5, 3.5, or 4.5")
    return result


def _effective_config(value: EffectiveConfig | Mapping[str, object] | None) -> EffectiveConfig | None:
    if value is None:
        return None
    if isinstance(value, EffectiveConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("production_effective_config must be an EffectiveConfig or object")
    payload = dict(value)
    if "schema" not in payload:
        topology = str(payload.get("topology", "torus9"))
        payload = {
            "schema": "gocube-effective-config-v2",
            "version": 2,
            "topology": topology,
            "compatibility": payload.get("compatibility", {"topology": topology}),
            "self_play": payload.get("self_play", {}),
            "training": payload.get("training", {}),
            "replay": payload.get("replay", {}),
            "execution": payload.get("execution", {}),
            "arena": payload.get("arena", {}),
            "supervision": payload.get("supervision", {}),
            "extensions": payload.get("extensions", {}),
        }
    return EffectiveConfig.from_dict(payload)


def _set_nested_komi(section: dict[str, object], value: float) -> None:
    """Update existing rules/komi slots without touching unrelated config."""
    changed = False
    if "komi" in section:
        section["komi"] = value
        changed = True
    for key in ("rules", "game_rules", "rule_set"):
        nested = section.get(key)
        if isinstance(nested, Mapping):
            nested_copy = dict(nested)
            if "komi" in nested_copy:
                nested_copy["komi"] = value
                section[key] = nested_copy
                changed = True
    if not changed and section is not None:
        section["komi"] = value


def effective_config_with_komi(config: EffectiveConfig, komi: float) -> EffectiveConfig:
    """Return the child contract with only the rules/komi value changed."""
    selected = _komi(komi)
    payload = deepcopy(config.to_dict())
    compatibility = dict(payload.get("compatibility", {}))
    _set_nested_komi(compatibility, selected)
    payload["compatibility"] = compatibility
    for name in ("self_play", "arena"):
        section = dict(payload.get(name, {}))
        _set_nested_komi(section, selected)
        payload[name] = section
    extensions = dict(payload.get("extensions", {}))
    if "rules" in extensions or "komi" in extensions:
        _set_nested_komi(extensions, selected)
        payload["extensions"] = extensions
    return EffectiveConfig.from_dict(payload)


def _configuration_only_komi_changed(before: EffectiveConfig, after: EffectiveConfig) -> None:
    left = before.to_dict()
    right = after.to_dict()

    def strip(value: object, path: tuple[str, ...] = ()) -> object:
        if isinstance(value, Mapping):
            return {
                key: strip(item, path + (str(key),))
                for key, item in value.items()
                if not (str(key) == "komi" or (path and path[-1] in {"rules", "game_rules", "rule_set"} and str(key) == "komi"))
            }
        if isinstance(value, list):
            return [strip(item, path) for item in value]
        return value

    if strip(left) != strip(right):
        raise KomiCalibrationError("child effective config changes a parameter other than komi")


def frozen_calibration_startset_ref(*, master_seed: int, games: int = 2048):
    """Describe one immutable startset family shared by both candidates."""
    if int(games) < KOMI_CALIBRATION_INITIAL_GAMES or int(games) % 2:
        raise ValueError("calibration startset family must cover at least 1024 paired games")
    descriptor = {
        "generator": "torus9-komi-calibration-starts-v1",
        "master_seed": int(master_seed),
        "games": int(games),
        "pairs": int(games) // 2,
        "colour_symmetric": True,
        "continuation": "deterministic-seed-extension-v1",
    }
    fingerprint = sha256_fingerprint(descriptor)
    from .contracts import StartsetRef

    return StartsetRef(
        id="torus9-komi-calibration-starts-v1",
        artifact=ArtifactRef("startsets/torus9-komi-calibration-v1.json", fingerprint),
        fingerprint=fingerprint,
    )


@dataclass(frozen=True)
class KomiCalibrationConfig:
    calibration_id: str
    parent_checkpoint: CheckpointRef | Mapping[str, object]
    production_effective_config: EffectiveConfig | Mapping[str, object] | None = None
    child_lineage_id: str | None = None
    arena_master_seed: int = 202609131004
    arena_startset: object | None = None
    arena_contract: KomiCalibrationArenaContract = field(default_factory=KomiCalibrationArenaContract)
    initial_games: int = KOMI_CALIBRATION_INITIAL_GAMES
    extension_games: int = KOMI_CALIBRATION_EXTENSION_GAMES
    ambiguity_threshold: float = KOMI_CALIBRATION_AMBIGUITY_THRESHOLD
    production_arena_cadence: int = 1
    production_arena_config: ArenaExecutionConfig | Mapping[str, object] | None = None
    production_arena_master_seed: int = 202609131004
    production_arena_profile: str = "torus9"
    generations: int | None = None
    allow_code_rollover: bool = False
    wait_poll_seconds: float = 5.0
    candidates: tuple[float, ...] = KOMI_CALIBRATION_CANDIDATES
    reuse_results_path: str | Path | None = None
    reuse_candidates: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration_id", _safe_component(self.calibration_id, "calibration_id"))
        if isinstance(self.arena_contract, Mapping):
            object.__setattr__(
                self,
                "arena_contract",
                KomiCalibrationArenaContract(**dict(self.arena_contract)),
            )
        elif not isinstance(self.arena_contract, KomiCalibrationArenaContract):
            raise TypeError("arena_contract must be a KomiCalibrationArenaContract or object")
        candidates = tuple(float(value) for value in self.candidates)
        if len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise ValueError("komi calibration requires at least two unique candidates")
        if any(value not in KOMI_CALIBRATION_ALLOWED_CANDIDATES for value in candidates):
            raise ValueError("unsupported komi calibration candidate")
        if tuple(sorted(candidates)) != candidates:
            raise ValueError("komi calibration candidates must be sorted")
        object.__setattr__(self, "candidates", candidates)
        reuse_candidates = tuple(float(value) for value in self.reuse_candidates)
        if any(value not in candidates for value in reuse_candidates):
            raise ValueError("reused komi candidates must be present in candidates")
        if len(set(reuse_candidates)) != len(reuse_candidates):
            raise ValueError("reused komi candidates must be unique")
        object.__setattr__(self, "reuse_candidates", reuse_candidates)
        if self.reuse_results_path is not None:
            object.__setattr__(self, "reuse_results_path", str(Path(self.reuse_results_path).resolve()))
        parent = self.parent_checkpoint if isinstance(self.parent_checkpoint, CheckpointRef) else CheckpointRef.from_dict(self.parent_checkpoint)
        if parent.topology != "torus9":
            raise ValueError("komi calibration requires topology=torus9")
        object.__setattr__(self, "parent_checkpoint", parent)
        if type(self.initial_games) is not int or self.initial_games != 1024:
            raise ValueError("komi calibration initial_games must be exactly 1024")
        if type(self.extension_games) is not int or self.extension_games != 1024:
            raise ValueError("komi calibration extension_games must be exactly 1024")
        if float(self.ambiguity_threshold) != KOMI_CALIBRATION_AMBIGUITY_THRESHOLD:
            raise ValueError("komi calibration ambiguity threshold must be 0.01")
        if type(self.production_arena_cadence) is not int or self.production_arena_cadence <= 0:
            raise ValueError("production_arena_cadence must be positive")
        if self.production_effective_config is not None:
            config = _effective_config(self.production_effective_config)
            assert config is not None
            if config.topology != "torus9":
                raise ValueError("production_effective_config must be Torus9")
            object.__setattr__(self, "production_effective_config", config)
        if self.child_lineage_id is not None:
            object.__setattr__(self, "child_lineage_id", _safe_component(self.child_lineage_id, "child_lineage_id"))
        if isinstance(self.arena_startset, Mapping):
            from .contracts import StartsetRef

            object.__setattr__(self, "arena_startset", StartsetRef.from_dict(self.arena_startset))
        if self.arena_startset is not None and not hasattr(self.arena_startset, "fingerprint"):
            raise TypeError("arena_startset must be a StartsetRef or object")
        if isinstance(self.arena_master_seed, bool):
            raise ValueError("arena_master_seed must be an integer")
        object.__setattr__(self, "arena_master_seed", int(self.arena_master_seed))
        if type(self.allow_code_rollover) is not bool:
            raise ValueError("allow_code_rollover must be a boolean")
        if self.wait_poll_seconds < 0:
            raise ValueError("wait_poll_seconds must be non-negative")

    @property
    def topology(self) -> str:
        return "torus9"

    @property
    def resolved_child_lineage_id(self) -> str:
        return self.child_lineage_id or f"{self.calibration_id}-selected"

    @property
    def fingerprint(self) -> str:
        return sha256_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": KOMI_CALIBRATION_SCHEMA,
            "type": KOMI_CALIBRATION_TYPE,
            "calibration_id": self.calibration_id,
            "parent_checkpoint": self.parent_checkpoint.to_dict(),  # type: ignore[union-attr]
            "production_effective_config": None if self.production_effective_config is None else self.production_effective_config.to_dict(),  # type: ignore[union-attr]
            "child_lineage_id": self.resolved_child_lineage_id,
            "arena_master_seed": self.arena_master_seed,
            "arena_startset": None if self.arena_startset is None else self.arena_startset.to_dict(),  # type: ignore[union-attr]
            "arena_contract": self.arena_contract.to_dict(),
            "initial_games": self.initial_games,
            "extension_games": self.extension_games,
            "ambiguity_threshold": self.ambiguity_threshold,
            "production_arena_cadence": self.production_arena_cadence,
            "production_arena_config": _arena_config_dict(self.production_arena_config),
            "production_arena_master_seed": self.production_arena_master_seed,
            "production_arena_profile": self.production_arena_profile,
            "generations": self.generations,
        }
        if self.candidates != KOMI_CALIBRATION_CANDIDATES:
            payload["candidates"] = list(self.candidates)
        if self.reuse_results_path is not None:
            payload["reuse_results_path"] = self.reuse_results_path
        if self.reuse_candidates:
            payload["reuse_candidates"] = list(self.reuse_candidates)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "KomiCalibrationConfig":
        raw = value.get("komi_calibration", value)
        if not isinstance(raw, Mapping):
            raise ValueError("komi_calibration plan must be an object")
        raw_contract = raw.get("arena_contract", raw.get("arena", {}))
        if not isinstance(raw_contract, Mapping):
            raise ValueError("komi calibration arena_contract must be an object")
        contract_fields = set(KomiCalibrationArenaContract.__dataclass_fields__)
        unknown_contract = set(raw_contract) - contract_fields
        if unknown_contract:
            raise ValueError(
                "komi calibration arena_contract contains unsupported fields: "
                + ", ".join(sorted(map(str, unknown_contract)))
            )
        contract = KomiCalibrationArenaContract(**{key: raw_contract[key] for key in contract_fields if key in raw_contract})
        production = raw.get("production_effective_config", raw.get("effective_config"))
        arena_startset = raw.get("arena_startset", raw.get("startset"))
        production_arena = raw.get("production_arena_config")
        return cls(
            calibration_id=str(raw.get("calibration_id", raw.get("experiment_id", raw.get("id", "")))),
            parent_checkpoint=raw.get("parent_checkpoint", raw.get("parent", {})),  # type: ignore[arg-type]
            production_effective_config=production,  # type: ignore[arg-type]
            child_lineage_id=None if raw.get("child_lineage_id") is None else str(raw["child_lineage_id"]),
            arena_master_seed=int(raw.get("arena_master_seed", 202609131004)),
            arena_startset=arena_startset,
            arena_contract=contract,
            initial_games=int(raw.get("initial_games", KOMI_CALIBRATION_INITIAL_GAMES)),
            extension_games=int(raw.get("extension_games", KOMI_CALIBRATION_EXTENSION_GAMES)),
            ambiguity_threshold=float(raw.get("ambiguity_threshold", KOMI_CALIBRATION_AMBIGUITY_THRESHOLD)),
            production_arena_cadence=int(raw.get("production_arena_cadence", raw.get("arena_cadence", 1))),
            production_arena_config=production_arena,  # type: ignore[arg-type]
            production_arena_master_seed=int(raw.get("production_arena_master_seed", raw.get("arena_master_seed", 202609131004))),
            production_arena_profile=str(raw.get("production_arena_profile", "torus9")),
            generations=None if raw.get("generations") is None else int(raw["generations"]),
            allow_code_rollover=bool(raw.get("allow_code_rollover", False)),
            wait_poll_seconds=float(raw.get("wait_poll_seconds", 5.0)),
            candidates=tuple(float(value) for value in raw.get("candidates", KOMI_CALIBRATION_CANDIDATES)),
            reuse_results_path=raw.get("reuse_results_path"),
            reuse_candidates=tuple(float(value) for value in raw.get("reuse_candidates", ())),
        )


def _arena_config_dict(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, ArenaExecutionConfig):
        from dataclasses import asdict

        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("production_arena_config must be an ArenaExecutionConfig or object")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, canonical_json(dict(payload)) + "\n")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KomiCalibrationError(f"cannot read calibration state: {path}") from exc
    if not isinstance(payload, dict):
        raise KomiCalibrationError(f"calibration state is not an object: {path}")
    return payload


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _summary_stats(result: ArenaRunResult) -> dict[str, object]:
    summary = dict(result.summary)
    technical = int(summary.get("technical_games", summary.get("invalid_games", 0)) or 0)
    invalid = int(summary.get("invalid_games", 0) or 0)
    games = int(summary.get("games", result.identity.games) or 0)
    valid = int(summary.get("valid_games", games - technical - invalid) or 0)
    black_wins = summary.get("black_wins")
    white_wins = summary.get("white_wins")
    draws = summary.get("draws")
    if not all(isinstance(value, int) and value >= 0 for value in (black_wins, white_wins, draws)):
        black_wins = white_wins = draws = None
        games_path = result.output_dir / "games.jsonl"
        if games_path.is_file():
            for line in games_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise KomiCalibrationError("calibration games.jsonl is malformed") from exc
                if not isinstance(row, Mapping) or row.get("technical_termination") is not None:
                    technical += 1
                    continue
                if row.get("formal_result") == "BLACK":
                    black_wins = int(black_wins or 0) + 1
                elif row.get("formal_result") == "WHITE":
                    white_wins = int(white_wins or 0) + 1
                elif row.get("formal_result") == "DRAW":
                    draws = int(draws or 0) + 1
                else:
                    invalid += 1
        else:
            raise KomiCalibrationError("calibration result lacks black/white counters")
    black_wins, white_wins, draws = int(black_wins), int(white_wins), int(draws)
    valid = int(black_wins + white_wins + draws)
    if technical or invalid or valid <= 0:
        raise KomiCalibrationError("calibration contains technical/invalid outcomes")
    if valid != games - technical - invalid and games > 0:
        raise KomiCalibrationError("calibration valid-game count is inconsistent")
    rate = black_wins / valid
    stats: dict[str, object] = {
        "games": games,
        "valid_games": valid,
        "technical_games": technical,
        "invalid_games": invalid,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "black_win_rate": rate,
        "bias": abs(rate - 0.5),
        "confidence_interval": _wilson_interval(black_wins, valid),
        "confidence_interval_method": "wilson-95-percent",
        "raw_score_margin_histogram": summary.get("raw_score_margin_histogram", summary.get("margin_histogram")),
        "raw_summary": summary,
    }
    return stats


def _result_to_state(result: ArenaRunResult, stats: Mapping[str, object], *, komi: float, batch: int) -> dict[str, object]:
    return {
        "komi": komi,
        "batch": batch,
        "evaluation_id": result.evaluation_id,
        "evaluation_fingerprint": result.evaluation_fingerprint,
        "output_dir": str(result.output_dir),
        "identity": result.identity.to_dict(),
        "summary": dict(result.summary),
        "validity": result.validity,
        "stats": dict(stats),
    }


def _result_from_state(raw: Mapping[str, object]) -> ArenaRunResult:
    from .contracts import EvaluationIdentity

    try:
        summary = raw["summary"]
        if not isinstance(summary, Mapping):
            raise TypeError("summary")
        return ArenaRunResult(
            evaluation_id=str(raw["evaluation_id"]),
            evaluation_fingerprint=str(raw["evaluation_fingerprint"]),
            output_dir=Path(str(raw["output_dir"])),
            identity=EvaluationIdentity.from_dict(raw["identity"]),  # type: ignore[arg-type]
            summary=dict(summary),
            validity=str(raw["validity"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KomiCalibrationError("persisted calibration Arena evidence is malformed") from exc


@dataclass(frozen=True)
class KomiCalibrationResult:
    state: str
    selected_komi: float | None
    parent: ResolvedCheckpointNode
    child_lineage_id: str | None
    child_checkpoint: ResolvedCheckpointNode | None
    results: Mapping[float, Mapping[str, object]]


class KomiCalibrationRunnerV2:
    """Run and resume the M137 -> calibration -> selected child workflow."""

    def __init__(
        self,
        config: KomiCalibrationConfig,
        *,
        arena_runner: ArenaRunnerV2,
        resolver: ArtifactResolver | None = None,
        experiment_root: str | Path | None = None,
        lineage_factory: LineageFactory | None = None,
        child_training: ChildTrainingHandoff | None = None,
        parent_supplier: ParentSupplier | None = None,
        stop_parent: Callable[[ResolvedCheckpointNode], object] | None = None,
        notifier: object | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.arena_runner = arena_runner
        self.resolver = resolver or ArtifactResolver()
        self.lineage_factory = lineage_factory or Torus9ProductionLineage(self.resolver.runs_root)
        self.child_training = child_training
        self.parent_supplier = parent_supplier
        self.stop_parent = stop_parent
        self.notifier = notifier
        self.sleeper = sleeper
        self.experiment_root = Path(experiment_root).resolve() if experiment_root is not None else (
            self.resolver.runs_root / "torus9" / "evaluations" / config.calibration_id
        )

    @property
    def state_path(self) -> Path:
        return self.experiment_root / "state.json"

    @property
    def plan_path(self) -> Path:
        return self.experiment_root / "plan.json"

    def run(self) -> KomiCalibrationResult:
        state = self._load_or_create_state()
        if state.get("state") == COMPLETE_HANDOFF:
            parent = self._resolve_pinned_parent(state)
            return self._result(state, parent)
        if state.get("state") == CALIBRATION_FAILED:
            raise KomiCalibrationError(str(state.get("failure", "calibration failed closed")))

        # Once M137 is pinned, its identity is the durable boundary.  Resume
        # from that boundary instead of moving the state machine back through
        # WAITING_FOR_M137 and re-requesting a stop on every process restart.
        raw_parent = state.get("parent_checkpoint")
        if isinstance(raw_parent, Mapping):
            parent = self.resolver.checkpoint(raw_parent)
            self._validate_m137(parent)
        else:
            parent = self._obtain_m137(state)

        if state.get("state") in {WAITING_FOR_M137, STOPPING_PARENT}:
            self._transition(state, STOPPING_PARENT)
        if self.stop_parent is not None and not state.get("parent_stop_requested"):
            try:
                self.stop_parent(parent)
            except BaseException as exc:
                self._fail(state, f"parent graceful stop failed: {exc}")
            state["parent_stop_requested"] = True
            self._persist(state)
        if state.get("state") != M137_PINNED and state.get("state") in {
            WAITING_FOR_M137,
            STOPPING_PARENT,
        }:
            self._transition(state, M137_PINNED, parent=parent)
        elif state.get("parent_checkpoint") is None:
            self._transition(state, M137_PINNED, parent=parent)

        self._load_reused_candidates(state, parent)
        for komi in self.config.candidates:
            self._run_candidate(state, parent, komi, self._candidate_stage(komi), batch=1)
        selected = self._select_or_extend(state, parent)
        previous_selected = state.get("selected_komi")
        if previous_selected is not None and float(previous_selected) != selected:
            self._fail(state, "persisted komi winner changed during resume")
        state["selected_komi"] = selected
        if state.get("state") not in {
            KOMI_SELECTED,
            CHILD_LINEAGE_CREATED,
            TRAINING_RESUMED,
        }:
            self._transition(state, KOMI_SELECTED)
        try:
            child, child_config, root = self._create_child(state, parent, selected)
        except KomiCalibrationError:
            raise
        except BaseException as exc:
            self._fail(state, f"child lineage creation failed closed: {exc}")
        if child is not None:
            state["child_checkpoint"] = child.ref.to_dict()
        state["child_config"] = child_config.config.to_dict()
        state["child_lineage_id"] = child_config.artifact.owner_lineage_id
        state["child_root"] = str(root)
        if state.get("state") in {
            KOMI_SELECTED,
            CALIBRATION_EXTENSION,
            CALIBRATION_CANDIDATE,
            CALIBRATION_KOMI_2_5,
            CALIBRATION_KOMI_1_5,
            M137_PINNED,
        }:
            self._transition(state, CHILD_LINEAGE_CREATED)

        # Persist the handoff before invoking a potentially long-running child
        # runner.  If the process dies in M138, the child runner's own durable
        # state is the authority and this stage is safely resumable.
        if state.get("state") == CHILD_LINEAGE_CREATED:
            self._transition(state, TRAINING_RESUMED)
        if self.child_training is not None and not isinstance(state.get("child_checkpoint"), Mapping):
            try:
                returned = self.child_training(
                    parent=parent,
                    config=child_config,
                    output_lineage=OutputLineage(self.config.topology, self.config.resolved_child_lineage_id, root),
                    first_generation=parent.generation + 1,
                )
            except KomiCalibrationError:
                raise
            except BaseException as exc:
                self._fail(state, f"child training failed closed: {exc}")
            if isinstance(returned, ResolvedCheckpointNode):
                if returned.generation < parent.generation + 1 or returned.node.parent != parent.ref:
                    self._fail(state, "child training returned a checkpoint with the wrong M137 parent")
                state["child_checkpoint"] = returned.ref.to_dict()
            else:
                self._fail(state, "child training did not return a committed child checkpoint")
            self._persist(state)
        elif self.child_training is None:
            # The durable handoff is intentionally resumable by a separate
            # child process.  Do not claim COMPLETE_HANDOFF without evidence
            # that at least M138 was committed.
            return self._result(state, parent)
        if state.get("state") == TRAINING_RESUMED:
            self._transition(state, COMPLETE_HANDOFF)
        return self._result(state, parent)

    @staticmethod
    def _candidate_stage(komi: float) -> str:
        if komi == 1.5:
            return CALIBRATION_KOMI_1_5
        if komi == 2.5:
            return CALIBRATION_KOMI_2_5
        return CALIBRATION_CANDIDATE

    def _validate_candidate_evidence(
        self,
        raw: Mapping[str, object],
        parent: ResolvedCheckpointNode,
        komi: float,
    ) -> dict[str, object]:
        if raw.get("validity") != "VALID":
            self._fail({}, f"reused komi {komi:g} evidence is not VALID")
        if str(raw.get("evaluation_id", "")).strip() == "":
            self._fail({}, f"reused komi {komi:g} evaluation id is missing")
        if str(raw.get("evaluation_fingerprint", "")).strip() == "":
            self._fail({}, f"reused komi {komi:g} evaluation fingerprint is missing")
        identity = raw.get("identity")
        stats = raw.get("stats")
        if not isinstance(identity, Mapping) or not isinstance(stats, Mapping):
            self._fail({}, f"reused komi {komi:g} evidence is malformed")
        parent_ref = parent.ref.to_dict()
        if identity.get("candidate") != parent_ref or identity.get("reference") != parent_ref:
            self._fail({}, f"reused komi {komi:g} parent identity changed")
        expected_startset = self.config.arena_startset or frozen_calibration_startset_ref(
            master_seed=self.config.arena_master_seed
        )
        startset = identity.get("startset")
        if not isinstance(startset, Mapping) or startset.get("fingerprint") != expected_startset.fingerprint:
            self._fail({}, f"reused komi {komi:g} startset identity changed")
        expected_contract = self.config.arena_contract.scientific_contract(
            komi=komi, games=self.config.initial_games
        )
        if identity.get("scientific_contract") != expected_contract:
            self._fail({}, f"reused komi {komi:g} calibration contract changed")
        if int(raw.get("batch", 1)) != 1 or float(raw.get("komi", -1.0)) != komi:
            self._fail({}, f"reused komi {komi:g} batch identity is malformed")
        if (
            int(stats.get("games", 0)) != self.config.initial_games
            or int(stats.get("valid_games", 0)) != self.config.initial_games
            or int(stats.get("technical_games", 0)) != 0
            or int(stats.get("invalid_games", 0)) != 0
        ):
            self._fail({}, f"reused komi {komi:g} game counters are invalid")
        output_dir = Path(str(raw.get("output_dir", ""))).resolve()
        required = (
            "evaluation-identity.json",
            "games.jsonl",
            "manifest.json",
            "provenance.json",
            "summary.json",
        )
        if not output_dir.is_dir() or any(not (output_dir / name).is_file() for name in required):
            self._fail({}, f"reused komi {komi:g} evaluation metadata is incomplete")
        return deepcopy(dict(raw))

    def _load_reused_candidates(
        self, state: dict[str, object], parent: ResolvedCheckpointNode
    ) -> None:
        path_value = self.config.reuse_results_path
        if path_value is None or not self.config.reuse_candidates:
            return
        source = _read_json(Path(path_value))
        source_candidates = source.get("candidates")
        candidates = state.get("candidates")
        if not isinstance(source_candidates, Mapping) or not isinstance(candidates, dict):
            self._fail(state, "reused calibration results are malformed")
        candidate_batches = state.setdefault("candidate_batches", {})
        if not isinstance(candidate_batches, dict):
            self._fail(state, "reused calibration batch ledger is malformed")
        for komi in self.config.reuse_candidates:
            key = f"{komi:g}"
            raw = source_candidates.get(key)
            if not isinstance(raw, Mapping):
                self._fail(state, f"reused komi {key} result is missing")
            aggregate = self._validate_candidate_evidence(raw, parent, komi)
            batches = aggregate.get("batches")
            first = None
            if isinstance(batches, list):
                for item in batches:
                    if isinstance(item, Mapping) and int(item.get("batch", 0)) == 1:
                        first = self._validate_candidate_evidence(item, parent, komi)
                        break
            if first is None:
                first = aggregate
            candidates[key] = aggregate
            candidate_batches[key] = {"1": first}
        state["reused_candidates"] = list(self.config.reuse_candidates)
        self._persist(state)

    def _load_or_create_state(self) -> dict[str, object]:
        if self.state_path.is_file():
            state = _read_json(self.state_path)
            if state.get("schema") != KOMI_CALIBRATION_STATE_SCHEMA:
                raise KomiCalibrationError("unsupported komi calibration state schema")
            if state.get("calibration_id") != self.config.calibration_id or state.get("config_fingerprint") != self.config.fingerprint:
                raise KomiCalibrationError("komi calibration plan changed during resume")
            if state.get("state") not in set(KOMI_CALIBRATION_STAGES) | {CALIBRATION_FAILED}:
                raise KomiCalibrationError("komi calibration state is malformed")
            winner_path = self.experiment_root / "winner.json"
            if winner_path.is_file():
                winner = _read_json(winner_path)
                persisted = winner.get("selected_komi")
                selected = state.get("selected_komi")
                if persisted is not None and selected is not None and float(persisted) != float(selected):
                    raise KomiCalibrationError("persisted komi winner is inconsistent")
            return state
        now = _now()
        state = {
            "schema": KOMI_CALIBRATION_STATE_SCHEMA,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "type": KOMI_CALIBRATION_TYPE,
            "calibration_id": self.config.calibration_id,
            "config_fingerprint": self.config.fingerprint,
            "state": WAITING_FOR_M137,
            "parent_checkpoint": None,
            "parent_stop_requested": False,
            "candidates": {f"{komi:g}": None for komi in self.config.candidates},
            "selected_komi": None,
            "child_lineage_id": self.config.resolved_child_lineage_id,
            "created_at": now,
            "updated_at": now,
        }
        _write_json(self.plan_path, self.config.to_dict())
        _write_json(self.state_path, state)
        self._notify("KOMI_CALIBRATION_SCHEDULED", "Komi calibration scheduled: parent M137", "scheduled")
        return state

    def _obtain_m137(self, state: dict[str, object]) -> ResolvedCheckpointNode:
        self._transition(state, WAITING_FOR_M137)
        while True:
            candidate: ResolvedCheckpointNode | None = None
            try:
                candidate = self.resolver.checkpoint(self.config.parent_checkpoint)
            except Exception:
                if self.parent_supplier is not None:
                    candidate = self.parent_supplier()
                if candidate is None:
                    self.sleeper(self.config.wait_poll_seconds)
                    continue
            try:
                self._validate_m137(candidate)
                return candidate
            except KomiCalibrationError as exc:
                # A visible M138 or an integrity failure must never be treated
                # as a reason to move the parent pin forward.
                self._fail(state, str(exc))

    def _resolve_pinned_parent(self, state: Mapping[str, object]) -> ResolvedCheckpointNode:
        raw = state.get("parent_checkpoint")
        if not isinstance(raw, Mapping):
            raise KomiCalibrationError("completed calibration lacks pinned M137")
        parent = self.resolver.checkpoint(raw)
        self._validate_m137(parent)
        return parent

    @staticmethod
    def _validate_m137(parent: ResolvedCheckpointNode) -> None:
        if not isinstance(parent, ResolvedCheckpointNode):
            raise KomiCalibrationError("calibration parent did not resolve to a checkpoint node")
        if parent.topology != "torus9" or parent.generation != KOMI_CALIBRATION_PARENT_GENERATION or parent.checkpoint_id != KOMI_CALIBRATION_PARENT_CHECKPOINT_ID:
            raise KomiCalibrationError("calibration parent must be the committed checkpoint M137")
        marker = parent.owner_root / "generation-137.complete.json"
        if not marker.is_file():
            raise KomiCalibrationError("M137 completion marker is missing")
        try:
            payload = _read_json(marker)
            marker_lineage = payload.get("lineage_id")
            if payload.get("generation") != 137 or (
                marker_lineage is not None and marker_lineage != parent.lineage_id
            ):
                raise KomiCalibrationError("M137 completion marker identity mismatch")
            if payload.get("checkpoint_sha256") != parent.ref.sha256:
                raise KomiCalibrationError("M137 completion marker SHA mismatch")
            manifest = _read_json(parent.owner_root / "manifest.json")
            hashes = manifest.get("checkpoint_hashes")
            if not isinstance(hashes, Mapping) or hashes.get(parent.ref.path) != parent.ref.sha256:
                raise KomiCalibrationError("M137 manifest checkpoint SHA evidence is missing")
        except OSError as exc:
            raise KomiCalibrationError("M137 commit evidence cannot be read") from exc

    def _run_candidate(
        self,
        state: dict[str, object],
        parent: ResolvedCheckpointNode,
        komi: float,
        stage: str,
        *,
        batch: int,
    ) -> None:
        key = f"{komi:g}"
        candidates = state.get("candidates")
        if not isinstance(candidates, dict):
            self._fail(state, "calibration candidate state is malformed")
        existing = candidates.get(key)
        if isinstance(existing, Mapping) and int(existing.get("batch", 0)) >= batch:
            expected_startset = self.config.arena_startset or frozen_calibration_startset_ref(
                master_seed=self.config.arena_master_seed
            )
            identity = existing.get("identity")
            if not isinstance(identity, Mapping):
                self._fail(state, f"persisted komi {key} Arena identity is malformed")
            if identity.get("candidate") != parent.ref.to_dict() or identity.get("reference") != parent.ref.to_dict():
                self._fail(state, f"persisted komi {key} Arena parent identity changed")
            startset = identity.get("startset")
            if not isinstance(startset, Mapping) or startset.get("fingerprint") != expected_startset.fingerprint:
                self._fail(state, f"persisted komi {key} Arena startset fingerprint changed")
            scientific = identity.get("scientific_contract")
            if not isinstance(scientific, Mapping) or float(scientific.get("komi", -1.0)) != komi:
                self._fail(state, f"persisted komi {key} Arena contract changed")
            stats = existing.get("stats")
            expected_games = self.config.initial_games if int(existing.get("batch", 0)) == 1 else self.config.extension_games
            if not isinstance(stats, Mapping) or int(stats.get("valid_games", 0)) != expected_games:
                self._fail(state, f"persisted komi {key} Arena evidence is malformed")
            return
        self._transition(state, stage)
        self._notify(f"KOMI_{key.replace('.', '_')}_STARTED", f"Komi calibration: {key} started", f"started:{key}:{batch}")
        games = self.config.initial_games if batch == 1 else self.config.extension_games
        master_seed = int(self.config.arena_master_seed)
        continuation_offset_pairs = 0 if batch == 1 else self.config.initial_games // 2
        startset = self.config.arena_startset or frozen_calibration_startset_ref(master_seed=self.config.arena_master_seed)
        execution = self.config.arena_contract.execution_config(games)
        contract = self.config.arena_contract
        profile = (
            f"torus9-komi-calibration|{key}"
            f"|simulations={contract.simulations}"
            f"|cpuct={contract.cpuct:g}|fpu={contract.fpu:g}"
            f"|watchdog={contract.watchdog}|5ch"
        )
        request = ArenaRunRequest(
            candidate=parent,
            reference=parent,
            master_seed=master_seed,
            startset=startset,  # type: ignore[arg-type]
            config=execution,
            profile=profile,
            scientific_contract=self.config.arena_contract.scientific_contract(komi=komi, games=games),
            execution_contract={
                "engine": "process-central-inference-v1",
                "workers": 16,
                "active_contexts": 192,
                "inference_batch_cap": 64,
                "inference_wait_ms": 4.0,
                "batching": True,
            },
            workload={
                "experiment_type": KOMI_CALIBRATION_TYPE,
                "komi": komi,
                "pairs": games // 2,
                "paired_starts": True,
                "color_swap": True,
                "frozen_startset_fingerprint": startset.fingerprint,
                "continuation_batch": batch,
                "continuation_offset_pairs": continuation_offset_pairs,
                "frozen_family_games": KOMI_CALIBRATION_INITIAL_GAMES + KOMI_CALIBRATION_EXTENSION_GAMES,
            },
            candidate_label=f"M137-komi-{key}",
            reference_label=f"M137-komi-{key}",
            comparison=f"M137-komi-{key}",
        )
        try:
            result = self.arena_runner.run(request)
            if result.validity != "VALID":
                self._fail(state, f"komi {key} Arena validity={result.validity}")
            stats = _summary_stats(result)
        except KomiCalibrationError as exc:
            self._fail(state, str(exc))
        except BaseException as exc:
            self._fail(state, f"komi {key} Arena failed closed: {exc}")
        if int(stats["valid_games"]) != games:
            self._fail(state, f"komi {key} received {stats['valid_games']} valid games, expected {games}")
        candidates[key] = _result_to_state(result, stats, komi=komi, batch=batch)
        self._persist(state)
        self._notify(
            f"KOMI_{key.replace('.', '_')}_COMPLETED",
            f"Komi {key}: Black {float(stats['black_win_rate']) * 100:.1f}% / White {(1.0 - float(stats['black_win_rate'])) * 100:.1f}% "
            f"(execution={str(result.execution_code_commit or '')[:12] or 'synthetic'})",
            f"completed:{key}:{batch}:{result.evaluation_id}",
        )

    def _select_or_extend(self, state: dict[str, object], parent: ResolvedCheckpointNode) -> float:
        candidates = state.get("candidates")
        if not isinstance(candidates, Mapping):
            self._fail(state, "calibration candidates are malformed")
        configured = tuple(self.config.candidates)
        first = {komi: candidates.get(f"{komi:g}") for komi in configured}
        if any(not isinstance(value, Mapping) for value in first.values()):
            self._fail(state, "calibration candidates are incomplete")
        biases = {komi: float(first[komi]["stats"]["bias"]) for komi in configured}  # type: ignore[index]
        if configured != KOMI_CALIBRATION_CANDIDATES:
            minimum = min(biases.values())
            tied = [komi for komi in configured if biases[komi] == minimum]
            return 1.5 if 1.5 in tied else min(tied)
        if abs(biases[1.5] - biases[2.5]) < self.config.ambiguity_threshold:
            self._transition(state, CALIBRATION_EXTENSION)
            self._run_candidate(state, parent, 1.5, CALIBRATION_EXTENSION, batch=2)
            self._run_candidate(state, parent, 2.5, CALIBRATION_EXTENSION, batch=2)
            candidates = state["candidates"]
            assert isinstance(candidates, Mapping)
            first = {komi: candidates[f"{komi:g}"] for komi in KOMI_CALIBRATION_CANDIDATES}
            biases = {komi: float(first[komi]["stats"]["bias"]) for komi in KOMI_CALIBRATION_CANDIDATES}  # type: ignore[index]
        selected = 1.5 if biases[1.5] <= biases[2.5] else 2.5
        return selected

    def _create_child(
        self,
        state: dict[str, object],
        parent: ResolvedCheckpointNode,
        selected: float,
    ) -> tuple[ResolvedCheckpointNode | None, ResolvedEffectiveConfig, Path]:
        raw_child = state.get("child_checkpoint")
        if isinstance(raw_child, Mapping):
            child = self.resolver.checkpoint(raw_child)
            config = self.resolver.effective_config(child.node.effective_config, owner_root=child.owner_root, topology=child.topology, lineage_id=child.lineage_id, owner_status=child.owner_status)
            if child.node.parent != parent.ref:
                self._fail(state, "existing child lineage does not descend from M137")
            return child, config, child.owner_root
        base = self.config.production_effective_config or parent.effective_config.config
        child_config = effective_config_with_komi(base, selected)
        _configuration_only_komi_changed(base, child_config)
        args: dict[str, object] = {
            "topology": "torus9",
            "lineage_id": self.config.resolved_child_lineage_id,
            "parent": parent,
            "effective_config": child_config,
            "experiment_id": self.config.calibration_id,
            "arm_id": f"komi-{selected:g}",
        }
        if self.config.allow_code_rollover:
            args["allow_code_rollover"] = True
        root, resolved_config = self.lineage_factory.prepare(**args)  # type: ignore[arg-type]
        if resolved_config.config.fingerprint != child_config.fingerprint:
            self._fail(state, "child lineage resolved an incompatible effective config")
        if Path(root).resolve() != (self.resolver.runs_root / "torus9" / "active" / self.config.resolved_child_lineage_id).resolve():
            self._fail(state, "child lineage factory returned a non-canonical path")
        self._persist_parent_replay_references(root, parent, state)
        return None, resolved_config, Path(root).resolve()

    def _persist_parent_replay_references(
        self,
        root: Path,
        parent: ResolvedCheckpointNode,
        state: dict[str, object],
    ) -> None:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self._fail(state, "child lineage manifest is missing")
        manifest = _read_json(manifest_path)
        refs: list[dict[str, object]] = []
        try:
            artifacts = self.resolver.replay_window(parent, 6)
            for generation, artifact in zip(range(max(1, parent.generation - 5), parent.generation + 1), artifacts):
                refs.append({"generation": generation, "path": str(artifact.path), "sha256": artifact.ref.sha256})
        except Exception as exc:
            self._fail(state, f"M137 parent replay references are unavailable: {exc}")
        parent_payload = dict(manifest.get("parent_checkpoint", {}))
        parent_payload["replay_references"] = refs
        manifest["parent_checkpoint"] = parent_payload
        manifest["parent_replay_references"] = refs
        _write_json(manifest_path, manifest)

    def _transition(self, state: dict[str, object], stage: str, *, parent: ResolvedCheckpointNode | None = None) -> None:
        if stage not in KOMI_CALIBRATION_STAGES:
            raise ValueError(f"unsupported calibration stage: {stage}")
        state["state"] = stage
        if parent is not None:
            state["parent_checkpoint"] = parent.ref.to_dict()
            state["parent_sha256"] = parent.ref.sha256
        self._persist(state)

    def _persist(self, state: dict[str, object]) -> None:
        state["updated_at"] = _now()
        _write_json(self.state_path, state)
        candidates = state.get("candidates")
        if not isinstance(candidates, Mapping):
            candidates = {}
        refs = {
            "schema": f"{KOMI_CALIBRATION_SCHEMA}-refs",
            "parent_checkpoint": state.get("parent_checkpoint"),
            "parent_sha256": state.get("parent_sha256"),
            "startset_fingerprints": {
                key: value.get("identity", {}).get("startset", {}).get("fingerprint")
                for key, value in candidates.items()
                if isinstance(value, Mapping) and isinstance(value.get("identity"), Mapping)
            },
            "candidates": {
                key: {
                    "evaluation_id": value.get("evaluation_id"),
                    "evaluation_fingerprint": value.get("evaluation_fingerprint"),
                    "output_dir": value.get("output_dir"),
                }
                for key, value in candidates.items()
                if isinstance(value, Mapping)
            },
        }
        results = {
            "schema": f"{KOMI_CALIBRATION_SCHEMA}-results",
            "candidates": dict(candidates),
        }
        winner = {
            "schema": f"{KOMI_CALIBRATION_SCHEMA}-winner",
            "selected_komi": state.get("selected_komi"),
            "rule": "minimize_abs_black_win_rate_minus_0.5; exact_tie=1.5",
        }
        full_contract = {
            "schema": f"{KOMI_CALIBRATION_SCHEMA}-full-contract",
            "calibration": self.config.to_dict(),
            "candidates": {
                str(komi): self.config.arena_contract.scientific_contract(
                    komi=komi,
                    games=self.config.initial_games,
                )
                for komi in self.config.candidates
            },
            "selected_komi": state.get("selected_komi"),
            "selected_effective_config": state.get("child_config"),
        }
        _write_json(self.experiment_root / "refs.json", refs)
        _write_json(self.experiment_root / "results.json", results)
        _write_json(self.experiment_root / "winner.json", winner)
        _write_json(self.experiment_root / "full-contract.json", full_contract)

    def _fail(self, state: dict[str, object], reason: str) -> None:
        if state:
            state["state"] = CALIBRATION_FAILED
            state["failure"] = str(reason)
            self._persist(state)
        self._notify("KOMI_CALIBRATION_FAILED", f"Komi calibration failed closed: {reason}", f"failed:{reason}")
        raise KomiCalibrationError(str(reason))

    def _notify(self, key: str, message: str, suffix: str) -> None:
        notifier = self.notifier
        if notifier is None:
            return
        try:
            notifier.send_now(f"komi-calibration:{self.config.calibration_id}:{suffix}", message)
        except Exception:
            # Operator notification must not turn a durable scientific result
            # into a retryable Arena or training execution.
            return

    def _result(self, state: Mapping[str, object], parent: ResolvedCheckpointNode) -> KomiCalibrationResult:
        child = None
        raw_child = state.get("child_checkpoint")
        if isinstance(raw_child, Mapping):
            child = self.resolver.checkpoint(raw_child)
        raw_results = state.get("candidates")
        results = {}
        if isinstance(raw_results, Mapping):
            for key, value in raw_results.items():
                if isinstance(value, Mapping):
                    results[float(key)] = dict(value)
        selected = state.get("selected_komi")
        return KomiCalibrationResult(
            state=str(state.get("state")),
            selected_komi=None if selected is None else float(selected),
            parent=parent,
            child_lineage_id=None if state.get("child_lineage_id") is None else str(state["child_lineage_id"]),
            child_checkpoint=child,
            results=results,
        )


__all__ = [
    "CALIBRATION_EXTENSION",
    "CALIBRATION_FAILED",
    "CALIBRATION_KOMI_1_5",
    "CALIBRATION_KOMI_2_5",
    "CHILD_LINEAGE_CREATED",
    "COMPLETE_HANDOFF",
    "KOMI_CALIBRATION_CANDIDATES",
    "KOMI_CALIBRATION_SCHEMA",
    "KOMI_CALIBRATION_STATE_SCHEMA",
    "KOMI_CALIBRATION_TYPE",
    "KOMI_SELECTED",
    "KomiCalibrationArenaContract",
    "KomiCalibrationConfig",
    "KomiCalibrationError",
    "KomiCalibrationResult",
    "KomiCalibrationRunnerV2",
    "M137_PINNED",
    "STOPPING_PARENT",
    "TRAINING_RESUMED",
    "WAITING_FOR_M137",
    "effective_config_with_komi",
    "frozen_calibration_startset_ref",
]
