"""Immutable plan and winner contracts for ExperimentRunner V2.

This module contains only declarative experiment data.  Execution remains in
the existing ``ArmExecutionPath`` and ``ArenaRunnerV2`` seams.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum

from ..provenance import sha256_fingerprint
from .contracts import CheckpointRef, EffectiveConfig, StartsetRef

from tools.arena_engine import ArenaExecutionConfig


EXPERIMENT_WINNER_RULE = "candidate_if_wins_gt_losses_else_reference"


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one safe path component")
    return text


def _config_from_value(value: EffectiveConfig | Mapping[str, object]) -> EffectiveConfig:
    if isinstance(value, EffectiveConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("arm config must be EffectiveConfig or an object")
    payload = dict(value)
    if "schema" not in payload:
        topology = str(payload.get("topology", ""))
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


class WinnerRuleName(str, Enum):
    CANDIDATE_IF_WINS_GT_LOSSES_ELSE_REFERENCE = EXPERIMENT_WINNER_RULE


@dataclass(frozen=True)
class WinnerRule:
    """Explicit deterministic rule used to turn a valid Arena into a decision."""

    name: str = EXPERIMENT_WINNER_RULE

    def __post_init__(self) -> None:
        name = self.name.value if isinstance(self.name, WinnerRuleName) else str(self.name)
        if name != EXPERIMENT_WINNER_RULE:
            raise ValueError(f"unsupported winner rule: {self.name}")
        object.__setattr__(self, "name", name)

    @classmethod
    def from_value(cls, value: "WinnerRule | str | Mapping[str, object] | None") -> "WinnerRule":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, WinnerRuleName):
            return cls(value.value)
        if isinstance(value, Mapping):
            value = value.get("name", value.get("rule"))  # type: ignore[assignment]
        return cls(str(value))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name}

    def choose(
        self,
        *,
        candidate: CheckpointRef,
        reference: CheckpointRef,
        wins: int,
        losses: int,
    ) -> CheckpointRef:
        return candidate if wins > losses else reference


@dataclass(frozen=True)
class ExperimentArmConfig:
    """One arm's independent budget and effective scientific configuration."""

    arm_id: str
    generations: int
    config: EffectiveConfig | Mapping[str, object]
    lineage_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _component(self.arm_id, "arm_id"))
        if type(self.generations) is not int or self.generations < 0:
            raise ValueError("arm generations must be a non-negative integer")
        object.__setattr__(self, "config", _config_from_value(self.config))
        if self.lineage_id is not None:
            object.__setattr__(self, "lineage_id", _component(self.lineage_id, "lineage_id"))

    @property
    def effective_config(self) -> EffectiveConfig:
        return self.config  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExperimentArmConfig":
        if not isinstance(value, Mapping):
            raise ValueError("experiment arm must be an object")
        raw_config = value.get("config", value.get("effective_config"))
        if raw_config is None:
            raise ValueError("experiment arm config is required")
        return cls(
            arm_id=str(value.get("arm_id", value.get("id", ""))),
            generations=int(value.get("generations", value.get("iterations", 0))),
            config=raw_config,  # type: ignore[arg-type]
            lineage_id=None if value.get("lineage_id") is None else str(value["lineage_id"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "generations": self.generations,
            "lineage_id": self.lineage_id,
            "config": self.effective_config.to_dict(),
        }


def _arena_to_dict(
    *,
    config: ArenaExecutionConfig,
    master_seed: int,
    startset: StartsetRef | None,
    profile: str,
    scientific_contract: Mapping[str, object] | None,
    execution_contract: Mapping[str, object] | None,
    workload: Mapping[str, object],
    winner_rule: WinnerRule,
) -> dict[str, object]:
    return {
        "config": asdict(config),
        "master_seed": master_seed,
        "startset": None if startset is None else startset.to_dict(),
        "profile": profile,
        "scientific_contract": dict(scientific_contract or {}),
        "execution_contract": dict(execution_contract or {}),
        "workload": dict(workload),
        "winner_rule": winner_rule.to_dict(),
    }


@dataclass(frozen=True)
class ExperimentStage2Config:
    """Optional Stage 2: winner -> independent control and C lineages."""

    control_generations: int
    c: ExperimentArmConfig
    arena_config: ArenaExecutionConfig
    arena_master_seed: int
    arena_startset: StartsetRef | None = None
    arena_profile: str = "torus9"
    arena_scientific_contract: Mapping[str, object] | None = None
    arena_execution_contract: Mapping[str, object] | None = None
    arena_workload: Mapping[str, object] = field(default_factory=dict)
    winner_rule: WinnerRule | str | Mapping[str, object] = field(default_factory=WinnerRule)

    def __post_init__(self) -> None:
        if type(self.control_generations) is not int or self.control_generations < 0:
            raise ValueError("control_generations must be a non-negative integer")
        if self.c.arm_id != "C":
            raise ValueError("Stage 2 variant arm must have arm_id=C")
        self.arena_config.validate_base()
        if isinstance(self.arena_master_seed, bool):
            raise ValueError("arena_master_seed must be an integer")
        object.__setattr__(self, "arena_master_seed", int(self.arena_master_seed))
        object.__setattr__(self, "winner_rule", WinnerRule.from_value(self.winner_rule))
        if not isinstance(self.arena_workload, Mapping):
            raise ValueError("arena_workload must be an object")
        if not isinstance(self.arena_profile, str) or not self.arena_profile:
            raise ValueError("arena_profile must be a non-empty string")

    @property
    def c_arm(self) -> ExperimentArmConfig:
        return self.c

    def to_dict(self) -> dict[str, object]:
        return {
            "control_generations": self.control_generations,
            "c": self.c.to_dict(),
            "arena": _arena_to_dict(
                config=self.arena_config,
                master_seed=self.arena_master_seed,
                startset=self.arena_startset,
                profile=self.arena_profile,
                scientific_contract=self.arena_scientific_contract,
                execution_contract=self.arena_execution_contract,
                workload=self.arena_workload,
                winner_rule=self.winner_rule,  # type: ignore[arg-type]
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExperimentStage2Config":
        if not isinstance(value, Mapping):
            raise ValueError("stage2 config must be an object")
        raw_c = value.get("c", value.get("candidate"))
        if not isinstance(raw_c, Mapping):
            raise ValueError("stage2 C arm is required")
        raw_arena = value.get("arena")
        if not isinstance(raw_arena, Mapping):
            raise ValueError("stage2 arena config is required")
        raw_execution = raw_arena.get("config", raw_arena.get("execution"))
        if not isinstance(raw_execution, Mapping):
            raise ValueError("stage2 arena execution config is required")
        raw_startset = raw_arena.get("startset")
        return cls(
            control_generations=int(value.get("control_generations", value.get("control_budget", 0))),
            c=ExperimentArmConfig.from_dict(raw_c),
            arena_config=ArenaExecutionConfig(**dict(raw_execution)),
            arena_master_seed=int(raw_arena.get("master_seed", 0)),
            arena_startset=None if raw_startset is None else StartsetRef.from_dict(raw_startset),  # type: ignore[arg-type]
            arena_profile=str(raw_arena.get("profile", "torus9")),
            arena_scientific_contract=(
                None if raw_arena.get("scientific_contract") is None
                else dict(raw_arena["scientific_contract"])  # type: ignore[arg-type]
            ),
            arena_execution_contract=(
                None if raw_arena.get("execution_contract") is None
                else dict(raw_arena["execution_contract"])  # type: ignore[arg-type]
            ),
            arena_workload=dict(raw_arena.get("workload", {})),  # type: ignore[arg-type]
            winner_rule=raw_arena.get("winner_rule", EXPERIMENT_WINNER_RULE),  # type: ignore[arg-type]
        )


Stage2Config = ExperimentStage2Config


@dataclass(frozen=True)
class ExperimentConfig:
    """Stage 1 A/B plan with an optional winner-rooted Stage 2."""

    experiment_id: str
    topology: str
    parent: CheckpointRef | Mapping[str, object]
    arms: Sequence[ExperimentArmConfig]
    arena_config: ArenaExecutionConfig
    arena_master_seed: int
    arena_startset: StartsetRef | None = None
    arena_profile: str = "torus9"
    arena_scientific_contract: Mapping[str, object] | None = None
    arena_execution_contract: Mapping[str, object] | None = None
    arena_workload: Mapping[str, object] = field(default_factory=dict)
    winner_rule: WinnerRule | str | Mapping[str, object] = field(default_factory=WinnerRule)
    stage2: ExperimentStage2Config | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _component(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "topology", _component(self.topology, "topology"))
        parent = self.parent if isinstance(self.parent, CheckpointRef) else CheckpointRef.from_dict(self.parent)
        if parent.topology != self.topology:
            raise ValueError("experiment parent topology does not match experiment topology")
        object.__setattr__(self, "parent", parent)
        arms = tuple(self.arms)
        if len(arms) != 2 or {arm.arm_id for arm in arms} != {"A", "B"}:
            raise ValueError("ExperimentRunner V2 requires exactly the A and B arms")
        for arm in arms:
            if arm.effective_config.topology != self.topology:
                raise ValueError(f"arm {arm.arm_id} config topology does not match experiment")
        object.__setattr__(self, "arms", arms)
        self.arena_config.validate_base()
        if isinstance(self.arena_master_seed, bool):
            raise ValueError("arena_master_seed must be an integer")
        object.__setattr__(self, "arena_master_seed", int(self.arena_master_seed))
        object.__setattr__(self, "winner_rule", WinnerRule.from_value(self.winner_rule))
        if not isinstance(self.arena_workload, Mapping):
            raise ValueError("arena_workload must be an object")
        if self.stage2 is not None:
            if self.stage2.c.effective_config.topology != self.topology:
                raise ValueError("Stage 2 C config topology does not match experiment")
            if self.stage2.arena_profile != self.arena_profile and self.topology != "torus9":
                raise ValueError("non-torus9 Stage 2 must use the experiment Arena profile")

    @property
    def arm_a(self) -> ExperimentArmConfig:
        return next(arm for arm in self.arms if arm.arm_id == "A")

    @property
    def arm_b(self) -> ExperimentArmConfig:
        return next(arm for arm in self.arms if arm.arm_id == "B")

    @property
    def stage2_enabled(self) -> bool:
        return self.stage2 is not None

    def to_dict(self) -> dict[str, object]:
        arena = _arena_to_dict(
            config=self.arena_config,
            master_seed=self.arena_master_seed,
            startset=self.arena_startset,
            profile=self.arena_profile,
            scientific_contract=self.arena_scientific_contract,
            execution_contract=self.arena_execution_contract,
            workload=self.arena_workload,
            winner_rule=self.winner_rule,  # type: ignore[arg-type]
        )
        return {
            "schema": "gocube-experiment-runner-v2",
            "experiment_id": self.experiment_id,
            "topology": self.topology,
            "parent": self.parent.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
            "arena": arena,
            "stage2": None if self.stage2 is None else self.stage2.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_fingerprint(self.to_dict())

    @property
    def legacy_fingerprint(self) -> str:
        """Fingerprint emitted by the pre-Stage-2 ExperimentRunner V2.

        PR #155 persisted the same experiment schema before the explicit
        winner rule and optional Stage 2 fields existed.  Keeping this
        compatibility calculation here makes the migration exact rather than
        accepting an arbitrary stale fingerprint.
        """
        payload = self.to_dict()
        arena = payload["arena"]
        if not isinstance(arena, Mapping):
            raise ValueError("experiment arena payload is malformed")
        legacy_arena = dict(arena)
        legacy_arena.pop("winner_rule", None)
        payload["arena"] = legacy_arena
        payload.pop("stage2", None)
        return sha256_fingerprint(payload)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExperimentConfig":
        if not isinstance(value, Mapping):
            raise ValueError("experiment config must be an object")
        raw_arms = value.get("arms")
        if not isinstance(raw_arms, Sequence) or isinstance(raw_arms, (str, bytes)):
            raise ValueError("experiment arms must be a list")
        raw_arena = value.get("arena")
        if not isinstance(raw_arena, Mapping):
            raise ValueError("experiment arena config must be an object")
        raw_execution = raw_arena.get("config", raw_arena.get("execution"))
        if not isinstance(raw_execution, Mapping):
            raise ValueError("experiment arena execution config is required")
        raw_startset = raw_arena.get("startset")
        raw_stage2 = value.get("stage2")
        return cls(
            experiment_id=str(value.get("experiment_id", value.get("id", ""))),
            topology=str(value.get("topology", "")),
            parent=value.get("parent", value.get("parent_checkpoint", {})),  # type: ignore[arg-type]
            arms=tuple(ExperimentArmConfig.from_dict(raw) for raw in raw_arms),  # type: ignore[arg-type]
            arena_config=ArenaExecutionConfig(**dict(raw_execution)),
            arena_master_seed=int(raw_arena.get("master_seed", 0)),
            arena_startset=None if raw_startset is None else StartsetRef.from_dict(raw_startset),  # type: ignore[arg-type]
            arena_profile=str(raw_arena.get("profile", "torus9")),
            arena_scientific_contract=(
                None if raw_arena.get("scientific_contract") is None
                else dict(raw_arena["scientific_contract"])  # type: ignore[arg-type]
            ),
            arena_execution_contract=(
                None if raw_arena.get("execution_contract") is None
                else dict(raw_arena["execution_contract"])  # type: ignore[arg-type]
            ),
            arena_workload=dict(raw_arena.get("workload", {})),  # type: ignore[arg-type]
            winner_rule=raw_arena.get("winner_rule", EXPERIMENT_WINNER_RULE),  # type: ignore[arg-type]
            stage2=(None if raw_stage2 is None else ExperimentStage2Config.from_dict(raw_stage2)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class WinnerDecision:
    """Durable scientific decision made once from one valid Arena result."""

    stage: int
    evaluation_id: str
    evaluation_fingerprint: str
    candidate: CheckpointRef
    reference: CheckpointRef
    wins: int
    losses: int
    draws: int
    winner_rule: WinnerRule
    winner: CheckpointRef

    def __post_init__(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError("winner decision stage must be 1 or 2")
        if not self.evaluation_id or not self.evaluation_fingerprint:
            raise ValueError("winner decision requires evaluation identity")
        values = (self.wins, self.losses, self.draws)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("winner decision W/L/D must be non-negative integers")
        if self.winner not in (self.candidate, self.reference):
            raise ValueError("winner decision must select candidate or reference")
        object.__setattr__(self, "winner_rule", WinnerRule.from_value(self.winner_rule))

    @property
    def wld(self) -> tuple[int, int, int]:
        return self.wins, self.losses, self.draws

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "evaluation_id": self.evaluation_id,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "candidate_checkpoint": self.candidate.to_dict(),
            "reference_checkpoint": self.reference.to_dict(),
            "W/L/D": [self.wins, self.losses, self.draws],
            "winner_rule": self.winner_rule.to_dict(),
            "winner_checkpoint": self.winner.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WinnerDecision":
        if not isinstance(value, Mapping):
            raise ValueError("winner decision must be an object")
        wld = value.get("W/L/D")
        if not isinstance(wld, Sequence) or isinstance(wld, (str, bytes)) or len(wld) != 3:
            raise ValueError("winner decision requires W/L/D")
        return cls(
            stage=int(value["stage"]),
            evaluation_id=str(value["evaluation_id"]),
            evaluation_fingerprint=str(value["evaluation_fingerprint"]),
            candidate=CheckpointRef.from_dict(value["candidate_checkpoint"]),  # type: ignore[arg-type]
            reference=CheckpointRef.from_dict(value["reference_checkpoint"]),  # type: ignore[arg-type]
            wins=int(wld[0]),
            losses=int(wld[1]),
            draws=int(wld[2]),
            winner_rule=WinnerRule.from_value(value.get("winner_rule")),
            winner=CheckpointRef.from_dict(value["winner_checkpoint"]),  # type: ignore[arg-type]
        )


__all__ = [
    "EXPERIMENT_WINNER_RULE",
    "ExperimentArmConfig",
    "ExperimentConfig",
    "ExperimentStage2Config",
    "Stage2Config",
    "WinnerDecision",
    "WinnerRule",
    "WinnerRuleName",
]
