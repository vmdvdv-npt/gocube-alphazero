"""Pure Cube V2 training-target construction from compact game records."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from .cube_family import cube_family_topology, deserialize_cube_state
from .cube_game_contract_v2 import concrete_game_fingerprint, concrete_game_identity
from .cube_observation_v2 import (
    SCHEMA_FINGERPRINT,
    CubeObservationContext,
    build_cube_observation,
    concrete_observation_identity,
    initial_cube_observation_context,
    advance_cube_observation_context,
)
from .cube_selfplay_v2 import CubeSelfPlayGameRecord, CubeSelfPlayPosition
from .cube_selfplay_contract import (
    CUBE_TARGET_CONTRACT_ID,
    CUBE_TARGET_FINGERPRINT,
    project_cube_ownership,
    project_cube_score,
    project_cube_wdl,
)
from .rules import prepare_legal_actions, apply_action
from .state import Stone


@dataclass(frozen=True)
class CubeTrainingSampleV2:
    observation: torch.Tensor
    policy: tuple[float, ...]
    wdl: tuple[int, int, int]
    ownership: tuple[str, ...]
    score_exact: float
    score_normalized: float
    legal_action_mask: tuple[bool, ...]
    selected_action: int
    ply: int
    side_to_move: str
    game_id: str
    size: int
    game_identity_fingerprint: str
    observation_fingerprint: str
    model_hash: str
    selfplay_semantics_fingerprint: str
    search_config_fingerprint: str
    target_contract_id: str = CUBE_TARGET_CONTRACT_ID
    target_contract_fingerprint: str = CUBE_TARGET_FINGERPRINT

    def __post_init__(self) -> None:
        topology = cube_family_topology(self.size)
        if tuple(self.observation.shape) != (30, topology.point_count):
            raise ValueError("Cube training observation shape drift")
        if self.observation.dtype != torch.float32 or not bool(torch.isfinite(self.observation).all()):
            raise ValueError("Cube training observation must be finite float32")
        if len(self.policy) != topology.action_count or len(self.legal_action_mask) != topology.action_count:
            raise ValueError("Cube training policy/action shape drift")
        if not math.isclose(sum(float(value) for value in self.policy), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Cube training policy is not normalized")
        if any(float(value) < 0.0 or not math.isfinite(float(value)) for value in self.policy):
            raise ValueError("Cube training policy is invalid")
        if any(not legal and float(value) != 0.0 for legal, value in zip(self.legal_action_mask, self.policy)):
            raise ValueError("Cube training policy assigns mass to an illegal action")
        if len(self.wdl) != 3 or tuple(self.wdl) not in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            raise ValueError("Cube training WDL target is invalid")
        if len(self.ownership) != topology.point_count or any(value not in ("OWN", "OPPONENT", "NEUTRAL") for value in self.ownership):
            raise ValueError("Cube training ownership target is invalid")
        if not math.isfinite(float(self.score_exact)) or not math.isfinite(float(self.score_normalized)):
            raise ValueError("Cube training score target is invalid")
        if self.target_contract_id != CUBE_TARGET_CONTRACT_ID or self.target_contract_fingerprint != CUBE_TARGET_FINGERPRINT:
            raise ValueError("Cube training target contract drift")

    @property
    def pi(self) -> tuple[float, ...]:
        return self.policy

    @property
    def margin_stm(self) -> float:
        return self.score_exact

    @property
    def normalized_score(self) -> float:
        return self.score_normalized

    @property
    def score(self) -> float:
        return self.score_exact


def _record_identity(record: CubeSelfPlayGameRecord) -> str:
    topology = cube_family_topology(record.size)
    contract = {
        "game_family": "cube",
        "size": record.size,
        "point_count": topology.point_count,
        "action_count": topology.action_count,
        "topology_id": record.topology_id,
        "topology_fingerprint": record.topology_fingerprint,
        "rules_fingerprint": record.rules_fingerprint,
        "komi": record.komi,
        "family_contract_fingerprint": record.game_contract_fingerprint,
    }
    return concrete_game_fingerprint(contract)


def _build_formal_samples(record: CubeSelfPlayGameRecord) -> tuple[CubeTrainingSampleV2, ...]:
    if record.formal_result is None or record.technical_termination is not None:
        return ()
    state = deserialize_cube_state(record.initial_state)
    context = initial_cube_observation_context(state)
    topology = cube_family_topology(record.size)
    observation_identity = concrete_observation_identity(topology)
    if record.observation_fingerprint != observation_identity["concrete_observation_fingerprint"]:
        raise ValueError("Cube record observation identity does not match target builder")
    if record.margin_black is None:
        raise ValueError("Formal Cube record has no final margin")
    denominator = topology.point_count + abs(float(record.komi))
    if denominator <= 0.0:
        raise ValueError("Cube score normalization denominator is invalid")
    samples: list[CubeTrainingSampleV2] = []
    for position, action in zip(record.positions, record.final_action_trace):
        legal = prepare_legal_actions(state)
        observation = build_cube_observation(state, context, legal_context=legal).detach().clone()
        if position.legal_action_mask != legal.action_mask:
            raise ValueError("Cube record legal mask drift during target replay")
        if position.selected_action != action:
            raise ValueError("Cube record selected action drift during target replay")
        side = state.side_to_move.name
        absolute_ownership = record.final_ownership
        ownership = project_cube_ownership(absolute_ownership, side)
        score_exact = project_cube_score(record.margin_black, side)
        wdl = project_cube_wdl(record.formal_result, side)
        sample = CubeTrainingSampleV2(
            observation=observation,
            policy=tuple(float(value) for value in position.pi),
            wdl=tuple(int(value) for value in wdl),
            ownership=tuple(ownership),
            score_exact=float(score_exact),
            score_normalized=float(score_exact) / denominator,
            legal_action_mask=tuple(bool(value) for value in legal.action_mask),
            selected_action=int(action),
            ply=int(position.ply),
            side_to_move=side,
            game_id=record.game_id,
            size=record.size,
            game_identity_fingerprint=_record_identity(record),
            observation_fingerprint=str(record.observation_fingerprint),
            model_hash=record.model_hash,
            selfplay_semantics_fingerprint=record.selfplay_semantics_fingerprint,
            search_config_fingerprint=record.search_config_fingerprint,
        )
        samples.append(sample)
        state = apply_action(state, record_action(action, record.size)).after
        context = advance_cube_observation_context(context, action, state)
    if not state.is_terminal:
        raise ValueError("Cube formal target replay did not reach DOUBLE_PASS")
    return tuple(samples)


def record_action(action: int, size: int):
    from .cube_game_contract_v2 import action_index_to_rules_action

    return action_index_to_rules_action(action, size)


def build_cube_training_samples(
    records: CubeSelfPlayGameRecord | Sequence[CubeSelfPlayGameRecord],
) -> tuple[CubeTrainingSampleV2, ...]:
    """Build in-memory samples; technical records intentionally yield none."""

    if isinstance(records, CubeSelfPlayGameRecord):
        selected = (records,)
    else:
        selected = tuple(records)
    samples: list[CubeTrainingSampleV2] = []
    for record in selected:
        if not isinstance(record, CubeSelfPlayGameRecord):
            raise TypeError("Cube target builder accepts CubeSelfPlayGameRecord values")
        record.validate(deep=True)
        samples.extend(_build_formal_samples(record))
    return tuple(samples)


build_cube_training_targets = build_cube_training_samples


__all__ = ["CubeTrainingSampleV2", "build_cube_training_samples", "build_cube_training_targets"]
