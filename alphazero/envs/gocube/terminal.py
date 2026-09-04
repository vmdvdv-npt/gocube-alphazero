from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .core import ENDGAME, FinalScore, GoState, GroupClassification, Topology, score_position
from .endgame import EndgameGroupProposal, assisted_endgame_proposal

CONSERVATIVE_AREA_ADJUDICATOR_V1 = "gocube-conservative-area-v1"


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
    score: FinalScore
    fallback_count: int

    @property
    def winner(self) -> str:
        return self.score.winner


def conservative_area_adjudicate(
    state: GoState,
    topology: Topology,
    *,
    ruleset: str,
    komi: float,
) -> TerminalAdjudication:
    """Total V1 terminal adjudicator for Chinese/area self-play.

    Stage A reproduces production GoCube automatic proof. Any group still
    unresolved is conservatively retained as alive for scoring rather than
    being virtually removed or called seki. This makes the terminal function
    total while requiring self-play to physically capture unproven dead stones
    before passing if it wants those stones removed from the final area score.

    Japanese/territory self-play is intentionally not enabled by this resolver;
    it requires a separately specified cleanup phase rather than pretending the
    same fallback preserves territory/prisoner semantics.
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
                "reason": "unproven stones must remain on the area-scoring board",
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
