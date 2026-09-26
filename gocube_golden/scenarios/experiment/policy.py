"""Pure experiment decisions.

No policy function reads state, starts a process, sends a message, or touches
the network.  The exact historical rule is intentionally kept as-is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from ...provenance import sha256_fingerprint
from ...artifact_graph import CheckpointRef


EXPERIMENT_WINNER_RULE = "candidate_if_wins_gt_losses_else_reference"


class WinnerRuleName(str, Enum):
    CANDIDATE_IF_WINS_GT_LOSSES_ELSE_REFERENCE = EXPERIMENT_WINNER_RULE


@dataclass(frozen=True)
class WinnerRule:
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
class WinnerDecision:
    """A scientific choice bound to one immutable Arena evidence record."""

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

    @property
    def fingerprint(self) -> str:
        return sha256_fingerprint(self.to_dict())

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


__all__ = ["EXPERIMENT_WINNER_RULE", "WinnerDecision", "WinnerRule", "WinnerRuleName"]
