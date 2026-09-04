from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .core import (
    BLACK,
    ENDGAME,
    WHITE,
    FinalScore,
    GoState,
    GroupClassification,
    Topology,
    score_position,
)
from .endgame import EndgameGroupProposal, assisted_endgame_proposal

CONSERVATIVE_AREA_ADJUDICATOR_V1 = "gocube-conservative-area-v1"
JAPANESE_CLEANUP_ADJUDICATOR_V2 = "gocube-japanese-cleanup-v2"


class UnsupportedSelfPlayRuleset(ValueError):
    pass


@dataclass(frozen=True)
class TerminalGroupResolution:
    points: tuple[int, ...]
    status: str
    source: str
    evidence: Mapping[str, object] | None


@dataclass(frozen=True)
class TerminalAdjudication:
    adjudicator_id: str
    stage_a: tuple[EndgameGroupProposal, ...]
    classification: tuple[TerminalGroupResolution, ...]
    score: FinalScore | None
    fallback_count: int = 0
    unresolved_count: int = 0
    no_result: bool = False

    @property
    def winner(self) -> str:
        if self.no_result or self.score is None:
            return "draw"
        return self.score.winner

    @property
    def training_valid(self) -> bool:
        return not self.no_result and self.score is not None


def conservative_area_adjudicate(
    state: GoState,
    topology: Topology,
    *,
    ruleset: str,
    komi: float,
) -> TerminalAdjudication:
    """Legacy V1 total terminal adjudicator for Chinese/area self-play.

    This path is retained only so historical Chinese checkpoints remain loadable
    and reproducible. New GoCube training uses JAPANESE_CLEANUP_ADJUDICATOR_V2.
    V1's unresolved-as-alive fallback must not be used for new training runs.
    """
    if state.phase != ENDGAME or state.consecutive_passes < 2:
        raise ValueError("Terminal adjudication requires the state after two consecutive passes")
    if ruleset != "chinese":
        raise UnsupportedSelfPlayRuleset(
            f"{CONSERVATIVE_AREA_ADJUDICATOR_V1} supports Chinese area scoring only; "
            f"got {ruleset!r}"
        )

    stage_a = assisted_endgame_proposal(state, topology)
    resolved: list[TerminalGroupResolution] = []
    scoring_classification: list[GroupClassification] = []
    fallback_count = 0

    for proposal in stage_a:
        if proposal.status == "unresolved":
            fallback_count += 1
            status = "alive"
            source = "self-play-conservative"
            evidence: Mapping[str, object] = {
                "algorithm": CONSERVATIVE_AREA_ADJUDICATOR_V1,
                "decision": "retain-unresolved-as-alive",
                "reason": "legacy V1 compatibility only",
            }
        else:
            status = proposal.status
            source = proposal.source or "automatic"
            evidence = proposal.evidence or {}

        resolved.append(
            TerminalGroupResolution(
                points=proposal.points,
                status=status,
                source=source,
                evidence=evidence,
            )
        )
        scoring_classification.append(GroupClassification(proposal.points, status))

    score = score_position(
        state,
        topology,
        tuple(scoring_classification),
        ruleset="chinese",
        komi=komi,
    )
    return TerminalAdjudication(
        adjudicator_id=CONSERVATIVE_AREA_ADJUDICATOR_V1,
        stage_a=stage_a,
        classification=tuple(resolved),
        score=score,
        fallback_count=fallback_count,
    )


def _proposal_by_point(
    proposals: tuple[EndgameGroupProposal, ...],
) -> dict[int, EndgameGroupProposal]:
    result: dict[int, EndgameGroupProposal] = {}
    for proposal in proposals:
        for point in proposal.points:
            result[point] = proposal
    return result


def japanese_cleanup_adjudicate(
    scoring_state: GoState,
    cleanup_state: GoState,
    topology: Topology,
    *,
    ruleset: str,
    komi: float,
) -> TerminalAdjudication:
    """Japanese-like territory adjudication with a frozen scoring position.

    ``scoring_state`` is the board at the end of the main phase, before any
    service cleanup moves. ``cleanup_state`` is a proof board reached during
    cleanup. Cleanup can physically capture questionable groups, but its added
    stones and cleanup captures never change territory or prisoner points.

    Stage-A Benson/dead/seki proofs are preserved. An original unresolved group
    becomes dead only if cleanup physically captured it, or takes a resolved
    status from the current cleanup proof. If uncertainty remains, this returns
    a no-result instead of fabricating an alive status and a training reward.
    """
    if ruleset != "japanese":
        raise UnsupportedSelfPlayRuleset(
            f"{JAPANESE_CLEANUP_ADJUDICATOR_V2} supports Japanese territory scoring only; "
            f"got {ruleset!r}"
        )

    stage_a = assisted_endgame_proposal(scoring_state, topology)
    cleanup_stage_a = assisted_endgame_proposal(cleanup_state, topology)
    cleanup_by_point = _proposal_by_point(cleanup_stage_a)

    resolved: list[TerminalGroupResolution] = []
    scoring_classification: list[GroupClassification] = []
    unresolved_count = 0

    for proposal in stage_a:
        if proposal.status != "unresolved":
            status = proposal.status
            source = proposal.source or "automatic-main"
            evidence: Mapping[str, object] = proposal.evidence or {}
        else:
            color = int(scoring_state.board[proposal.points[0]])
            surviving = tuple(
                point for point in proposal.points
                if int(cleanup_state.board[point]) == color
            )
            if not surviving:
                status = "dead"
                source = "cleanup-captured"
                evidence = {
                    "algorithm": JAPANESE_CLEANUP_ADJUDICATOR_V2,
                    "proof": "original-group-physically-captured-during-cleanup",
                }
            else:
                current = cleanup_by_point.get(surviving[0])
                if current is not None and current.status != "unresolved":
                    status = current.status
                    source = current.source or "automatic-cleanup"
                    evidence = current.evidence or {
                        "algorithm": JAPANESE_CLEANUP_ADJUDICATOR_V2,
                        "proof": "resolved-on-cleanup-board",
                    }
                else:
                    status = "unresolved"
                    source = "cleanup-unresolved"
                    evidence = {
                        "algorithm": JAPANESE_CLEANUP_ADJUDICATOR_V2,
                        "decision": "no-training-reward",
                        "reason": "group status remains unproven after cleanup",
                    }
                    unresolved_count += 1

        resolved.append(
            TerminalGroupResolution(
                points=proposal.points,
                status=status,
                source=source,
                evidence=evidence,
            )
        )
        if status != "unresolved":
            scoring_classification.append(GroupClassification(proposal.points, status))

    if unresolved_count:
        return TerminalAdjudication(
            adjudicator_id=JAPANESE_CLEANUP_ADJUDICATOR_V2,
            stage_a=stage_a,
            classification=tuple(resolved),
            score=None,
            unresolved_count=unresolved_count,
            no_result=True,
        )

    score = score_position(
        scoring_state,
        topology,
        tuple(scoring_classification),
        ruleset="japanese",
        komi=komi,
    )
    return TerminalAdjudication(
        adjudicator_id=JAPANESE_CLEANUP_ADJUDICATOR_V2,
        stage_a=stage_a,
        classification=tuple(resolved),
        score=score,
    )


def ownership_target(
    adjudication: TerminalAdjudication,
    scoring_state: GoState,
    topology: Topology,
) -> np.ndarray:
    """Return point ownership as one-hot [black, white, neutral]."""
    if not adjudication.training_valid or adjudication.score is None:
        raise ValueError("Ownership target requires a valid scored terminal result")

    labels = np.full(topology.point_count, 2, dtype=np.int64)
    for group in adjudication.classification:
        if group.status not in ("alive", "seki"):
            continue
        color = int(scoring_state.board[group.points[0]])
        label = 0 if color == BLACK else 1
        for point in group.points:
            labels[point] = label

    score = adjudication.score
    labels[list(score.territory_points.black)] = 0
    labels[list(score.territory_points.white)] = 1
    labels[list(score.territory_points.neutral)] = 2
    labels[list(score.territory_points.seki)] = 2

    result = np.zeros((topology.point_count, 3), dtype=np.float32)
    result[np.arange(topology.point_count), labels] = 1.0
    return result


def normalized_score_target(
    adjudication: TerminalAdjudication,
    topology: Topology,
) -> np.ndarray:
    """Signed black-minus-white score, normalized to a stable [-1, 1] target."""
    if not adjudication.training_valid or adjudication.score is None:
        raise ValueError("Score target requires a valid scored terminal result")
    signed = adjudication.score.black - adjudication.score.white
    return np.asarray([np.clip(signed / topology.point_count, -1.0, 1.0)], dtype=np.float32)
