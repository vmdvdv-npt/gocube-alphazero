"""Standalone Torus 9×9 Golden contract, training line, and Arena helpers.

This module is intentionally dimension-explicit.  It shares the already
verified generic Golden rules and PUCT implementation, while keeping the 9×9
observation, replay, checkpoint, and provenance boundaries independent from
the frozen 5×5 line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from queue import Empty, Queue
import random
import threading
import time
from typing import Any, Callable, Mapping, MutableMapping, Sequence

torch = importlib.import_module("torch")
nn = torch.nn
F = importlib.import_module("torch.nn.functional")

from .arena_contract import SearchSettings
from .neural import model_hash
from .provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256, sha256_fingerprint
from .result import Winner, result_from_terminal
from .rules import IllegalMoveError, LegalActionContext, apply_action, legal_actions, prepare_legal_actions
from .scoring import Ownership, score_terminal
from .search import (
    Evaluation,
    SearchError,
    SearchResult,
    SequentialPUCT,
    _Edge,
    _Node,
    _child_to_parent_utility,
    _policy_for_legal,
    wdl_to_side_to_move_utility,
)
from .search_adapter import GoldenSearchAdapter
from .selfplay_policy import apply_root_dirichlet_noise, sample_action_from_search_result
from .state import BLACK, EMPTY, PASS, WHITE, GoldenState, Stone, initial_state, rules_fingerprint_for
from .topology import TORUS_5X5, TORUS_9X9, TORUS_9X9_TOPOLOGY_ID
from .torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_ARCHITECTURE_ID,
    TORUS9_ARENA_CONTRACT_FINGERPRINT,
    TORUS9_ARENA_CONTRACT_ID,
    TORUS9_ARENA_MOVE_LIMIT,
    TORUS9_BATCH_SIZE,
    TORUS9_BLOCKS,
    TORUS9_HIDDEN,
    TORUS9_KOMI,
    TORUS9_MOVE_LIMIT,
    TORUS9_OBSERVATION_FINGERPRINT,
    TORUS9_OBSERVATION_SCHEMA_ID,
    TORUS9_OBSERVATION_SCHEMA_VERSION,
    TORUS9_MAX_REPLAY_POSITIONS,
    TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    TORUS9_PASS_INDEX,
    TORUS9_POINT_COUNT,
    TORUS9_ROLLING_GENERATIONS,
    TORUS9_RULES_FINGERPRINT,
    TORUS9_TARGET_CONTRACT_ID,
    TORUS9_WORKERS,
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_BLOCKS,
    TORUS9_CURRENT_DIRICHLET_ALPHA,
    TORUS9_CURRENT_HIDDEN,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    current_torus9_selfplay_contract_fingerprint,
    profile_fingerprint,
)


TORUS9_VALUE_HEAD_SEMANTICS = "side-to-move:[WIN,DRAW,LOSS]"
TORUS9_TOPOLOGY_ID = TORUS_9X9_TOPOLOGY_ID
TORUS9_OBSERVATION_CHANNELS = (
    "own_stones",
    "opponent_stones",
    "side_to_move_color",
    "previous_pass",
    "legal_point_mask",
    "komi",
)
TORUS9_TOPOLOGY_FINGERPRINT = TORUS_9X9.fingerprint
TORUS9_OWNERSHIP_TARGET_CONTRACT_ID = "golden-ownership-final-state-side-to-move-v1"
TORUS9_AUXILIARY_TARGET_SOURCE = "golden-referee-final-state-v1"
TORUS9_OWNERSHIP_CLASSES = ("OWN", "OPPONENT", "NEUTRAL")
TORUS9_SCORE_TARGET_CONTRACT_ID = "golden-score-final-margin-side-to-move-v1"
TORUS9_SCORE_TARGET_NORMALIZATION = 81.5


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Stone):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, EnumValue):
        return value.value
    return str(value)


def _side_name(side_to_move: Stone | str) -> str:
    return side_to_move.name if isinstance(side_to_move, Stone) else str(side_to_move)


def _ownership_target(final_state: GoldenState, side_to_move: Stone | str) -> tuple[int, ...]:
    """Return exact final graph-area ownership in sample perspective."""
    if not final_state.is_terminal:
        raise ValueError("Ownership targets require a formal Golden terminal state")
    side = _side_name(side_to_move)
    if side not in ("BLACK", "WHITE"):
        raise ValueError("Ownership target perspective must be BLACK or WHITE")
    score = score_terminal(final_state)
    own = Ownership.BLACK if side == "BLACK" else Ownership.WHITE
    opponent = Ownership.WHITE if side == "BLACK" else Ownership.BLACK
    mapping = {own: 0, opponent: 1, Ownership.NEUTRAL: 2}
    return tuple(mapping[item] for item in score.ownership)


class EnumValue:
    value: str


def _process_context(name: str) -> Any:
    # Keep the Golden package's source-level dependency boundary free of a
    # direct multiprocessing import; the executor still uses the requested
    # fork/spawn context at runtime.
    return __import__("multiprocessing").get_context(name)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def torus9_state_identity(state: GoldenState) -> dict[str, object]:
    if state.topology.fingerprint != TORUS9_TOPOLOGY_FINGERPRINT:
        raise ValueError("Torus 9×9 state identity received another topology")
    return {
        "stones": [int(stone) for stone in state.stones],
        "side_to_move": int(state.side_to_move),
        "superko_history": [list(position) for position in state.superko_history],
        "consecutive_passes": state.consecutive_passes,
        "topology_id": state.topology.topology_id,
        "topology_fingerprint": state.topology.fingerprint,
        "rules_id": state.rules_id,
        "rules_fingerprint": state.rules_fingerprint,
        "komi": state.komi,
        "history_provenance": state.history_provenance,
    }


def torus9_state_from_identity(
    identity: Mapping[str, object],
    *,
    expected_komi: float | None = None,
) -> GoldenState:
    if identity.get("topology_id") != TORUS9_TOPOLOGY_ID or identity.get("topology_fingerprint") != TORUS9_TOPOLOGY_FINGERPRINT:
        raise ValueError("Torus 9×9 state topology identity drift")
    komi = float(identity.get("komi", -1.0))
    if expected_komi is not None and komi != float(expected_komi):
        raise ValueError("Torus 9×9 state komi identity drift")
    if identity.get("rules_fingerprint") != rules_fingerprint_for(TORUS_9X9, komi):
        raise ValueError("Torus 9×9 state rules/komi identity drift")
    return GoldenState(
        stones=tuple(Stone(int(value)) for value in identity["stones"]),  # type: ignore[index]
        side_to_move=Stone(int(identity["side_to_move"])),  # type: ignore[arg-type]
        superko_history=tuple(tuple(int(value) for value in row) for row in identity["superko_history"]),  # type: ignore[index]
        consecutive_passes=int(identity["consecutive_passes"]),
        topology=TORUS_9X9,
        rules_id=str(identity["rules_id"]),
        rules_fingerprint=str(identity["rules_fingerprint"]),
        komi=komi,
        history_provenance=str(identity["history_provenance"]),
    )


@dataclass(frozen=True)
class Torus9Observation:
    tensor: torch.Tensor
    action_mask: tuple[bool, ...]
    state_key: tuple[object, ...]

    def __post_init__(self) -> None:
        if tuple(self.tensor.shape) != (6, TORUS9_POINT_COUNT):
            raise ValueError("Torus 9×9 observation tensor must have shape [6,81]")
        if self.tensor.dtype != torch.float32 or not bool(torch.isfinite(self.tensor).all()):
            raise ValueError("Torus 9×9 observation must be finite float32")
        if len(self.action_mask) != TORUS9_ACTION_COUNT:
            raise ValueError("Torus 9×9 action mask must have length 82")


def build_torus9_observation_bundle(
    state: GoldenState,
    *,
    legal_context: LegalActionContext | None = None,
) -> Torus9Observation:
    if state.is_terminal:
        raise ValueError("Terminal Torus 9×9 states must never be observed")
    if state.topology.fingerprint != TORUS9_TOPOLOGY_FINGERPRINT:
        raise ValueError("Observation requires canonical Torus 9×9")
    context = legal_context if legal_context is not None else prepare_legal_actions(state)
    context.assert_compatible(state)
    if len(context.action_mask) != TORUS9_ACTION_COUNT:
        raise ValueError("Torus 9×9 legal-action mask length drift")
    own = state.side_to_move
    other = WHITE if own == BLACK else BLACK
    values = torch.empty((6, TORUS9_POINT_COUNT), dtype=torch.float32)
    build_torus9_observation_into(state, values, legal_context=context)
    return Torus9Observation(values, context.action_mask, state.state_key)


def build_torus9_observation_into(
    state: GoldenState,
    destination: torch.Tensor,
    *,
    legal_context: LegalActionContext | None = None,
) -> None:
    """Fill a preallocated float32 observation row, value-for-value.

    This is the execution form of ``build_torus9_observation``.  Keeping the
    canonical builder as a thin allocation wrapper makes the shared-memory
    path and the ordinary scientific path use exactly the same channel logic.
    """
    if state.is_terminal:
        raise ValueError("Terminal Torus 9×9 states must never be observed")
    if state.topology.fingerprint != TORUS9_TOPOLOGY_FINGERPRINT:
        raise ValueError("Observation requires canonical Torus 9×9")
    if tuple(destination.shape) != (6, TORUS9_POINT_COUNT) or destination.dtype != torch.float32:
        raise ValueError("Torus 9×9 destination must have shape [6,81] and float32 dtype")
    context = legal_context if legal_context is not None else prepare_legal_actions(state)
    context.assert_compatible(state)
    own = state.side_to_move
    other = WHITE if own == BLACK else BLACK
    destination.zero_()
    for point, stone in enumerate(state.stones):
        destination[0, point] = float(stone == own)
        destination[1, point] = float(stone == other)
    destination[2].fill_(1.0 if own == BLACK else -1.0)
    destination[3].fill_(1.0 if state.consecutive_passes == 1 else 0.0)
    destination[4].copy_(torch.tensor(context.action_mask[:TORUS9_POINT_COUNT], dtype=torch.float32))
    destination[5].fill_(float(state.komi))


def build_torus9_observation(state: GoldenState, *, legal_context: LegalActionContext | None = None) -> torch.Tensor:
    return build_torus9_observation_bundle(state, legal_context=legal_context).tensor


class Torus9GraphMessageLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.register_buffer("neighbors", torch.tensor(TORUS_9X9.adjacency, dtype=torch.long))
        self.self_linear = nn.Linear(hidden, hidden)
        self.neighbor_linear = nn.Linear(hidden, hidden)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        gathered = nodes[:, self.neighbors, :]
        return self.self_linear(nodes) + self.neighbor_linear(gathered.mean(dim=2))


class Torus9GraphResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.message = Torus9GraphMessageLayer(hidden)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        return nodes + self.output(F.relu(self.message(self.norm(nodes))))


class Torus9GraphNet(nn.Module):
    """Eight-hop Torus9 graph network with the policy boundary [81 points + PASS]."""

    architecture_id = TORUS9_ARCHITECTURE_ID

    def __init__(
        self,
        *,
        hidden: int = TORUS9_HIDDEN,
        blocks: int = TORUS9_BLOCKS,
        architecture_id: str | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0 or blocks <= 0:
            raise ValueError("Torus 9×9 hidden and blocks must be positive")
        self.architecture_id = architecture_id or TORUS9_ARCHITECTURE_ID
        self.topology_id = TORUS9_TOPOLOGY_ID
        self.topology_fingerprint = TORUS9_TOPOLOGY_FINGERPRINT
        self.hidden = int(hidden)
        self.blocks_count = int(blocks)
        self.input_projection = nn.Linear(6, hidden)
        self.blocks = nn.ModuleList(Torus9GraphResidualBlock(hidden) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(hidden)
        self.point_policy = nn.Linear(hidden, 1)
        self.pass_policy = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    @property
    def architecture_config(self) -> dict[str, object]:
        return {
            "architecture_id": self.architecture_id,
            "topology_id": self.topology_id,
            "topology_fingerprint": self.topology_fingerprint,
            "input_channels": 6,
            "point_count": TORUS9_POINT_COUNT,
            "hidden": self.hidden,
            "blocks": self.blocks_count,
            "heads": {"policy": [TORUS9_ACTION_COUNT], "value": [3]},
        }

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 2:
            observation = observation.unsqueeze(0)
        if tuple(observation.shape[1:]) != (6, TORUS9_POINT_COUNT):
            raise ValueError("Torus 9×9 network expects [batch,6,81]")
        nodes = self.input_projection(observation.transpose(1, 2))
        for block in self.blocks:
            nodes = block(nodes)
        return F.relu(self.output_norm(nodes))

    def policy_value_from_nodes(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the policy/WDL heads to an already encoded node tensor."""
        if nodes.ndim != 3 or nodes.shape[1:] != (TORUS9_POINT_COUNT, self.hidden):
            raise ValueError("Torus 9×9 encoded nodes must have shape [batch,81,hidden]")
        policy = torch.cat((self.point_policy(nodes).squeeze(-1), self.pass_policy(nodes.mean(dim=1))), dim=1)
        return policy, self.value_head(nodes.mean(dim=1))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.encode(observation)
        return self.policy_value_from_nodes(nodes)


class Torus9OwnershipGraphNet(Torus9GraphNet):
    """Torus9 policy/WDL network with the existing referee ownership head.

    The forward boundary intentionally remains the two-output policy/WDL
    boundary used by PUCT.  The ownership head is only exposed through
    ``forward_auxiliary`` for the controlled training ablation.  Both A and B
    therefore carry the same parameter set and optimizer parameter groups;
    A gives this head a zero loss weight while B gives it the existing unit
    weight used by the Golden auxiliary experiment.
    """

    def __init__(self, *, hidden: int = TORUS9_HIDDEN, blocks: int = TORUS9_BLOCKS) -> None:
        super().__init__(hidden=hidden, blocks=blocks)
        self.ownership_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    @property
    def architecture_config(self) -> dict[str, object]:
        config = super().architecture_config
        config["auxiliary_variant"] = "wdl+ownership"
        config["heads"] = {
            "policy": [TORUS9_ACTION_COUNT],
            "value": [3],
            "ownership": [TORUS9_POINT_COUNT, 3],
        }
        return config

    def forward_auxiliary(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = self.encode(observation)
        policy, value = self.policy_value_from_nodes(nodes)
        return policy, value, self.ownership_head(nodes)


class Torus9OwnershipScoreGraphNet(Torus9OwnershipGraphNet):
    """Torus9 WDL + ownership network with an optional score head.

    The score head is deliberately outside ``forward`` so adding it cannot
    change PUCT, self-play policy targets, or Arena inference semantics.
    """

    def __init__(self, *, hidden: int = TORUS9_HIDDEN, blocks: int = TORUS9_BLOCKS) -> None:
        super().__init__(hidden=hidden, blocks=blocks)
        self.score_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @property
    def architecture_config(self) -> dict[str, object]:
        config = super().architecture_config
        config["auxiliary_variant"] = "wdl+ownership+score"
        config["heads"] = {
            "policy": [TORUS9_ACTION_COUNT],
            "value": [3],
            "ownership": [TORUS9_POINT_COUNT, 3],
            "score": [1],
        }
        return config

    def forward_auxiliary(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, value, ownership = super().forward_auxiliary(observation)
        nodes = self.encode(observation)
        return policy, value, ownership, self.score_head(nodes.mean(dim=1)).squeeze(-1)


class Torus9CurrentGraphNet(Torus9OwnershipScoreGraphNet):
    """Current Golden Standard model: 80-wide, 8-block, WDL+auxiliary heads.

    The historical ``Torus9GraphNet`` defaults remain available for replaying
    the v2 line.  The current launcher must instantiate this class explicitly
    from the resolved current profile, so a legacy default cannot leak into a
    new run.
    """

    def __init__(
        self,
        *,
        hidden: int = TORUS9_CURRENT_HIDDEN,
        blocks: int = TORUS9_CURRENT_BLOCKS,
    ) -> None:
        if int(hidden) != TORUS9_CURRENT_HIDDEN or int(blocks) != TORUS9_CURRENT_BLOCKS:
            raise ValueError("Current Torus 9×9 model is fixed to GoldenGraphNetV2-Torus9 80×8")
        super().__init__(hidden=int(hidden), blocks=int(blocks))
        self.architecture_id = TORUS9_CURRENT_ARCHITECTURE_ID

    @property
    def architecture_config(self) -> dict[str, object]:
        config = super().architecture_config
        config["architecture_id"] = TORUS9_CURRENT_ARCHITECTURE_ID
        config["auxiliary_variant"] = "wdl+ownership+score"
        config["explicit_symmetry_augmentation"] = False
        return config

    def forward_auxiliary(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = self.encode(observation)
        policy, value = self.policy_value_from_nodes(nodes)
        ownership = self.ownership_head(nodes)
        score = self.score_head(nodes.mean(dim=1)).squeeze(-1)
        return policy, value, ownership, score


class Torus9NeuralEvaluator:
    def __init__(self, model: Torus9GraphNet, *, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.nn_evaluations = 0
        self.inference_batch_rows: list[int] = []

    def evaluate(self, state: GoldenState, *, legal_context: LegalActionContext | None = None) -> Evaluation:
        return self.evaluate_prepared(state, legal_context or prepare_legal_actions(state))

    def evaluate_prepared(self, state: GoldenState, legal_context: LegalActionContext) -> Evaluation:
        if state.is_terminal:
            raise SearchError("Torus 9×9 evaluator received terminal state")
        legal_context.assert_compatible(state)
        observation = build_torus9_observation(state, legal_context=legal_context).to(self.device)
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observation.unsqueeze(0))
            policy = torch.softmax(policy_logits[0], dim=0)
            wdl = torch.softmax(value_logits[0], dim=0)
        self.nn_evaluations += 1
        self.inference_batch_rows.append(1)
        policy_values = tuple(float(value) for value in policy.detach().cpu())
        wdl_values = tuple(float(value) for value in wdl.detach().cpu())
        if len(policy_values) != TORUS9_ACTION_COUNT or len(wdl_values) != 3:
            raise SearchError("Torus 9×9 neural head shape drift")
        if any(not math.isfinite(value) or value < 0.0 for value in policy_values + wdl_values):
            raise SearchError("Torus 9×9 neural output is non-finite")
        return Evaluation(policy=policy_values, wdl=wdl_values)

    def evaluate_prepared_batch(
        self,
        states: Sequence[GoldenState],
        legal_contexts: Sequence[LegalActionContext],
    ) -> tuple[Evaluation, ...]:
        """Evaluate many independent leaves in one neural-network call."""
        if len(states) != len(legal_contexts) or not states:
            raise SearchError("Torus 9×9 batched evaluator received mismatched or empty inputs")
        for state, context in zip(states, legal_contexts):
            if state.is_terminal:
                raise SearchError("Torus 9×9 batched evaluator received a terminal state")
            context.assert_compatible(state)
        observations = torch.stack(
            [build_torus9_observation(state, legal_context=context) for state, context in zip(states, legal_contexts)],
            dim=0,
        ).to(self.device)
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observations)
            policies = torch.softmax(policy_logits, dim=1)
            wdls = torch.softmax(value_logits, dim=1)
        self.nn_evaluations += len(states)
        self.inference_batch_rows.append(len(states))
        if tuple(policies.shape) != (len(states), TORUS9_ACTION_COUNT) or tuple(wdls.shape) != (len(states), 3):
            raise SearchError("Torus 9×9 batched neural head shape drift")
        evaluations: list[Evaluation] = []
        for policy, wdl in zip(policies.detach().cpu(), wdls.detach().cpu()):
            policy_values = tuple(float(value) for value in policy)
            wdl_values = tuple(float(value) for value in wdl)
            if any(not math.isfinite(value) or value < 0.0 for value in policy_values + wdl_values):
                raise SearchError("Torus 9×9 batched neural output is non-finite")
            evaluations.append(Evaluation(policy=policy_values, wdl=wdl_values))
        return tuple(evaluations)


@dataclass(frozen=True)
class Torus9SelfPlaySearchContract:
    contract_id: str = TORUS9_CURRENT_SELFPLAY_CONTRACT_ID
    simulations: int = 64
    cpuct: float = 1.25
    fpu: float = 0.0
    temperature_until_ply: int = 8
    temperature_after: float = 0.0
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = TORUS9_CURRENT_DIRICHLET_ALPHA
    watchdog: int = TORUS9_MOVE_LIMIT
    komi: float = TORUS9_KOMI

    @property
    def settings(self) -> SearchSettings:
        return SearchSettings(simulations=self.simulations, cpuct=self.cpuct, fpu=self.fpu, deterministic_tie_break=True)

    def validate(self) -> None:
        expected = Torus9SelfPlaySearchContract()
        actual = asdict(self)
        expected_values = asdict(expected)
        actual_komi = float(actual.pop("komi"))
        expected_values.pop("komi")
        if actual != expected_values or actual_komi not in {0.5, 1.5, 2.5}:
            raise ValueError("Torus 9×9 self-play search contract drift")

    @property
    def fingerprint(self) -> str:
        base = current_torus9_selfplay_contract_fingerprint(self.dirichlet_alpha)
        return base if float(self.komi) == TORUS9_KOMI else sha256_fingerprint({"base": base, "komi": float(self.komi)})


def _action_index(action: int | str) -> int:
    return TORUS9_PASS_INDEX if action == PASS else int(action)


class Torus9RootNoiseEvaluator:
    """Apply the canonical self-play Dirichlet transform at the root only."""

    def __init__(
        self,
        evaluator: object | None,
        root_state: GoldenState,
        *,
        seed: int,
        alpha: float = TORUS9_CURRENT_DIRICHLET_ALPHA,
    ) -> None:
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("Torus 9×9 Dirichlet alpha must be positive and finite")
        self.evaluator = evaluator
        self.root_state_key = root_state.state_key
        self.alpha = float(alpha)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def evaluate_prepared(
        self, state: GoldenState, legal_context: LegalActionContext
    ) -> Evaluation:
        if self.evaluator is None:
            raise RuntimeError("Root-noise transform has no evaluator")
        base = self.evaluator.evaluate_prepared(state, legal_context)
        return self.transform(base, state, legal_context)

    def transform(
        self, base: Evaluation, state: GoldenState, legal_context: LegalActionContext
    ) -> Evaluation:
        if state.state_key != self.root_state_key:
            return base
        legal_context.assert_compatible(state)
        policy = apply_root_dirichlet_noise(
            tuple(float(value) for value in base.policy),
            legal_context.actions,
            action_index=_action_index,
            epsilon=0.25,
            alpha=self.alpha,
            generator=self.generator,
        )
        return Evaluation(policy=policy, wdl=base.wdl)

    def evaluate(self, state: GoldenState) -> Evaluation:
        if self.evaluator is None:
            raise RuntimeError("Root-noise transform has no evaluator")
        return self.evaluate_prepared(state, prepare_legal_actions(state))


def graph_distance(topology: Any, source: int, target: int) -> int:
    """Return an unweighted shortest-path distance in a Golden topology."""
    if source == target:
        return 0
    frontier = [source]
    distances = {source: 0}
    while frontier:
        point = frontier.pop(0)
        for neighbor in topology.neighbors(point):
            if neighbor in distances:
                continue
            distance = distances[point] + 1
            if neighbor == target:
                return distance
            distances[neighbor] = distance
            frontier.append(neighbor)
    raise ValueError("Topology graph is disconnected")


def graph_diameter(topology: Any) -> int:
    return max(graph_distance(topology, source, target) for source in range(topology.point_count) for target in range(topology.point_count))


def _sample_action(result: Any, *, temperature: float, rng: random.Random) -> int | str:
    if len(result.root_visits) != TORUS9_ACTION_COUNT or not result.legal_actions:
        raise SearchError("Torus 9×9 search returned malformed root visits")
    return sample_action_from_search_result(
        result,
        temperature=temperature,
        rng=rng,
        action_index=_action_index,
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class Torus9SelfPlayPosition:
    ply: int
    state: dict[str, object]
    side_to_move: str
    root_visits: tuple[int, ...]
    pi: tuple[float, ...]
    selected_action: int | str
    search_seed: int
    model_hash: str

    def validate(self, expected_model_hash: str | None = None) -> None:
        state = torus9_state_from_identity(self.state)
        if self.side_to_move != state.side_to_move.name or self.ply <= 0:
            raise ValueError("Torus 9×9 self-play position provenance drift")
        context = prepare_legal_actions(state)
        if len(self.root_visits) != TORUS9_ACTION_COUNT or len(self.pi) != TORUS9_ACTION_COUNT:
            raise ValueError("Torus 9×9 self-play target shape drift")
        if sum(self.root_visits) <= 0 or not math.isclose(sum(self.pi), 1.0, abs_tol=1e-6):
            raise ValueError("Torus 9×9 policy target is not normalized")
        for index, value in enumerate(self.pi):
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError("Torus 9×9 policy target is invalid")
            action = PASS if index == TORUS9_PASS_INDEX else index
            if action not in context.actions and value != 0.0:
                raise ValueError("Illegal Torus 9×9 policy target is non-zero")
        if self.selected_action not in context.actions:
            raise ValueError("Torus 9×9 selected action is illegal")
        if expected_model_hash is not None and self.model_hash != expected_model_hash:
            raise ValueError("Torus 9×9 position model hash drift")


@dataclass(frozen=True)
class Torus9SelfPlayGameRecord:
    run_id: str
    game_id: str
    profile_id: str
    profile_fingerprint: str
    selfplay_contract_id: str
    selfplay_contract_fingerprint: str
    model_checkpoint_label: str
    model_hash: str
    checkpoint_artifact_hash: str
    git_commit: str
    git_tree: str
    git_worktree_clean: bool
    master_seed: int
    game_seed: int
    start_state: dict[str, object]
    positions: tuple[Torus9SelfPlayPosition, ...]
    final_action_trace: tuple[int | str, ...]
    formal_result: str | None
    technical_termination: str | None
    error: str | None = None
    nn_evaluations: int = 0

    def validate(self) -> None:
        if self.profile_id != TORUS9_CURRENT_PROFILE_ID or self.selfplay_contract_id != TORUS9_CURRENT_SELFPLAY_CONTRACT_ID:
            raise ValueError("Torus 9×9 self-play contract identity drift")
        if self.technical_termination is None and self.formal_result not in ("BLACK", "WHITE", "DRAW"):
            raise ValueError("Formal Torus 9×9 game lacks a result")
        if self.technical_termination is not None and self.formal_result is not None:
            raise ValueError("Technical Torus 9×9 game cannot have WDL result")
        if len(self.positions) != len(self.final_action_trace) and self.technical_termination is None:
            raise ValueError("Torus 9×9 trace length drift")
        for position in self.positions:
            position.validate(expected_model_hash=self.model_hash)

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def torus9_z_target(winner: str, side_to_move: str | Stone) -> tuple[float, float, float]:
    side = side_to_move.name if isinstance(side_to_move, Stone) else str(side_to_move)
    if winner == "DRAW":
        return (0.0, 1.0, 0.0)
    if winner not in ("BLACK", "WHITE") or side not in ("BLACK", "WHITE"):
        raise ValueError("Torus 9×9 WDL target requires formal result and side")
    return (1.0, 0.0, 0.0) if winner == side else (0.0, 0.0, 1.0)


def torus9_build_replay_samples(game: Torus9SelfPlayGameRecord) -> tuple[dict[str, object], ...]:
    game.validate()
    if game.technical_termination is not None or game.formal_result is None:
        raise ValueError("Technical Torus 9×9 self-play games are excluded from replay")
    state = torus9_state_from_identity(game.start_state)
    for action in game.final_action_trace:
        state = apply_action(state, action).after
    if not state.is_terminal:
        raise ValueError("Torus 9×9 replay did not reach DOUBLE_PASS")
    samples: list[dict[str, object]] = []
    for position in game.positions:
        sample_state = torus9_state_from_identity(position.state)
        context = prepare_legal_actions(sample_state)
        observation = build_torus9_observation(sample_state, legal_context=context)
        row = {
            "run_id": game.run_id,
            "game_id": game.game_id,
            "ply": position.ply,
            "state": position.state,
            "side_to_move": position.side_to_move,
            "observation": [[float(value) for value in channel] for channel in observation.tolist()],
            "legal_action_mask": list(context.action_mask),
            "root_visits": list(position.root_visits),
            "pi": list(position.pi),
            "z": list(torus9_z_target(game.formal_result, sample_state.side_to_move)),
            "model_hash": game.model_hash,
            "selfplay_contract_fingerprint": game.selfplay_contract_fingerprint,
            "observation_fingerprint": TORUS9_OBSERVATION_FINGERPRINT,
            "target_contract_id": TORUS9_TARGET_CONTRACT_ID,
            "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        }
        samples.append(row)
    return tuple(samples)


def torus9_ownership_target(final_state: GoldenState, side_to_move: str | Stone) -> tuple[int, ...]:
    """Return exact graph-area ownership in the sample side's perspective."""
    return _ownership_target(final_state, side_to_move)


def torus9_score_target(final_state: GoldenState, side_to_move: str | Stone) -> float:
    """Return the exact final graph-area margin in sample perspective."""
    if not final_state.is_terminal:
        raise ValueError("Torus 9×9 score target requires a formal terminal state")
    side = side_to_move.name if isinstance(side_to_move, Stone) else str(side_to_move)
    if side not in ("BLACK", "WHITE"):
        raise ValueError("Torus 9×9 score target requires BLACK or WHITE perspective")
    margin = float(score_terminal(final_state).margin_black)
    return margin if side == "BLACK" else -margin


def torus9_build_ownership_replay_samples(game: Torus9SelfPlayGameRecord) -> tuple[dict[str, object], ...]:
    """Add the existing Golden ownership target to a Torus9 replay corpus."""
    rows = list(torus9_build_replay_samples(game))
    final_state = torus9_state_from_identity(game.start_state)
    for action in game.final_action_trace:
        final_state = apply_action(final_state, action).after
    if not final_state.is_terminal:
        raise ValueError("Torus 9×9 ownership replay did not reach DOUBLE_PASS")
    for row in rows:
        sample_state = torus9_state_from_identity(row["state"])  # type: ignore[arg-type]
        row["ownership_target"] = list(torus9_ownership_target(final_state, sample_state.side_to_move))
        row["ownership_target_contract_id"] = TORUS9_OWNERSHIP_TARGET_CONTRACT_ID
        row["auxiliary_target_source"] = TORUS9_AUXILIARY_TARGET_SOURCE
    return tuple(rows)


def torus9_build_ownership_score_replay_samples(game: Torus9SelfPlayGameRecord) -> tuple[dict[str, object], ...]:
    """Add exact ownership and final-margin targets to Torus9 replay rows."""
    rows = list(torus9_build_ownership_replay_samples(game))
    final_state = torus9_state_from_identity(game.start_state)
    for action in game.final_action_trace:
        final_state = apply_action(final_state, action).after
    for row in rows:
        sample_state = torus9_state_from_identity(row["state"])  # type: ignore[arg-type]
        row["score_target"] = torus9_score_target(final_state, sample_state.side_to_move)
        row["score_target_contract_id"] = TORUS9_SCORE_TARGET_CONTRACT_ID
        row["score_target_normalization"] = TORUS9_SCORE_TARGET_NORMALIZATION
    return tuple(rows)


def validate_torus9_replay_sample(sample: Mapping[str, object], *, expected_target_fingerprint: str | None = None) -> None:
    """Validate one serialized policy+WDL replay row at the 9×9 boundary."""
    state = torus9_state_from_identity(sample["state"])  # type: ignore[arg-type]
    context = prepare_legal_actions(state)
    observation = sample["observation"]
    if len(observation) != 6 or any(len(row) != TORUS9_POINT_COUNT for row in observation):  # type: ignore[arg-type]
        raise ValueError("Torus 9×9 replay observation shape drift")
    if tuple(bool(value) for value in sample["legal_action_mask"]) != context.action_mask:  # type: ignore[arg-type]
        raise ValueError("Torus 9×9 replay legal mask drift")
    pi = tuple(float(value) for value in sample["pi"])  # type: ignore[arg-type]
    visits = tuple(int(value) for value in sample["root_visits"])  # type: ignore[arg-type]
    if len(pi) != TORUS9_ACTION_COUNT or len(visits) != TORUS9_ACTION_COUNT or sum(visits) <= 0:
        raise ValueError("Torus 9×9 replay policy target shape drift")
    if not math.isclose(sum(pi), 1.0, abs_tol=1e-6) or any(not math.isfinite(value) or value < 0.0 for value in pi):
        raise ValueError("Torus 9×9 replay policy target is invalid")
    z = tuple(float(value) for value in sample["z"])  # type: ignore[arg-type]
    if len(z) != 3 or any(not math.isfinite(value) or value < 0.0 for value in z) or not math.isclose(sum(z), 1.0, abs_tol=1e-6):
        raise ValueError("Torus 9×9 replay WDL target is invalid")
    accepted_target_fingerprints = {
        expected_target_fingerprint or TORUS9_CURRENT_TARGET_FINGERPRINT
    }
    if expected_target_fingerprint is None:
        accepted_target_fingerprints.add(TORUS9_CURRENT_TARGET_FINGERPRINT)
    if sample.get("observation_fingerprint") != TORUS9_OBSERVATION_FINGERPRINT or sample.get("target_contract_id") != TORUS9_TARGET_CONTRACT_ID or sample.get("target_fingerprint") not in accepted_target_fingerprints:
        raise ValueError("Torus 9×9 replay semantic fingerprint drift")
    if sample.get("ownership_target") is not None:
        ownership = tuple(int(value) for value in sample["ownership_target"])  # type: ignore[arg-type]
        if len(ownership) != TORUS9_POINT_COUNT or any(value not in range(3) for value in ownership):
            raise ValueError("Torus 9×9 ownership target shape or class drift")
        if sample.get("ownership_target_contract_id") != TORUS9_OWNERSHIP_TARGET_CONTRACT_ID or sample.get("auxiliary_target_source") != TORUS9_AUXILIARY_TARGET_SOURCE:
            raise ValueError("Torus 9×9 auxiliary target provenance drift")
    if sample.get("score_target") is not None:
        score = float(sample["score_target"])
        if not math.isfinite(score):
            raise ValueError("Torus 9×9 score target is non-finite")
        if sample.get("score_target_contract_id") != TORUS9_SCORE_TARGET_CONTRACT_ID or sample.get("score_target_normalization") != TORUS9_SCORE_TARGET_NORMALIZATION or sample.get("auxiliary_target_source") != TORUS9_AUXILIARY_TARGET_SOURCE:
            raise ValueError("Torus 9×9 score target provenance drift")


class Torus9RollingReplay:
    """Deterministic rolling replay window for the current Torus9 adapter."""

    def __init__(
        self,
        *,
        generations: int = TORUS9_ROLLING_GENERATIONS,
        maximum_positions: int | None = TORUS9_MAX_REPLAY_POSITIONS,
    ) -> None:
        if generations <= 0 or (
            maximum_positions is not None and maximum_positions <= 0
        ):
            raise ValueError("Torus 9×9 replay window settings are invalid")
        self.generations = int(generations)
        self.maximum_positions = (
            None if maximum_positions is None else int(maximum_positions)
        )
        self._rows: list[dict[str, object]] = []
        self._last_generation = 0
        self.total_evictions = 0

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        return tuple(self._rows)

    @staticmethod
    def _row_id(row: Mapping[str, object], generation: int, position: int) -> str:
        return str(row.get("replay_row_id", f"M{generation}:{row.get('game_id', position)}:{row.get('ply', position)}"))

    def append_generation(
        self, generation: int, samples: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        generation = int(generation)
        if generation <= 0 or generation < self._last_generation:
            raise ValueError("Torus 9×9 replay generations must be positive and monotonic")
        before = len(self._rows)
        stamped: list[dict[str, object]] = []
        for position, sample in enumerate(samples):
            row = dict(sample)
            row["source_generation"] = generation
            row["replay_row_id"] = self._row_id(row, generation, position)
            stamped.append(row)
        self._rows.extend(stamped)
        oldest_allowed = generation - self.generations + 1
        self._rows = [row for row in self._rows if int(row["source_generation"]) >= oldest_allowed]
        if self.maximum_positions is not None and len(self._rows) > self.maximum_positions:
            self._rows = self._rows[-self.maximum_positions:]
        evicted = before + len(stamped) - len(self._rows)
        self.total_evictions += evicted
        self._last_generation = generation
        counts: dict[str, int] = {}
        for row in self._rows:
            key = str(row["source_generation"])
            counts[key] = counts.get(key, 0) + 1
        return {
            "generation_added": generation,
            "fresh_positions": len(stamped),
            "rolling_buffer_positions": len(self._rows),
            "positions_per_generation": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
            "generations_represented": sorted(int(key) for key in counts),
            "eviction_count": evicted,
            "total_evictions": self.total_evictions,
        }


def _adam_step(optimizer: torch.optim.Optimizer) -> int:
    steps: set[int] = set()
    for state in optimizer.state.values():
        if "step" in state:
            steps.add(int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"]))
    if not steps:
        return 0
    if len(steps) != 1:
        raise RuntimeError(f"Torus 9×9 Adam state is discontinuous: steps={sorted(steps)}")
    return next(iter(steps))


def _parameter_l2(parameters: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.sum(value.detach().float() ** 2)) for value in parameters.values()))


def _parameter_delta(before: Mapping[str, torch.Tensor], model: nn.Module) -> tuple[float, float]:
    after = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    deltas = {name: after[name] - before[name] for name in before}
    total = _parameter_l2(deltas)
    by_layer: dict[str, dict[str, torch.Tensor]] = {}
    for name, delta in deltas.items():
        by_layer.setdefault(name.split(".", 1)[0], {})[name] = delta
    maximum = max((_parameter_l2(layer) for layer in by_layer.values()), default=0.0)
    return total, maximum


class _Torus9TrainingCore:
    def __init__(
        self,
        model: Torus9GraphNet,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_steps_per_iteration: int = TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    ) -> None:
        if learning_rate <= 0.0 or weight_decay != 0.0:
            raise ValueError("Torus 9×9 optimizer requires positive Adam learning rate and weight_decay=0")
        if optimizer_steps_per_iteration != TORUS9_OPTIMIZER_STEPS_PER_ITERATION:
            raise ValueError("Canonical Torus 9×9 optimizer budget is fixed at 80 updates")
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer_steps_per_iteration = int(optimizer_steps_per_iteration)
        self.update_count = 0
        self.samples_consumed = 0

    def assert_optimizer_continuity(self) -> int:
        step = _adam_step(self.optimizer)
        if step != self.update_count:
            raise RuntimeError(f"Torus 9×9 Adam continuation mismatch: state={step}, tracker={self.update_count}")
        return step

    @staticmethod
    def _sample_indices(size: int, *, seed: int, count: int) -> list[int]:
        if size <= 0:
            raise ValueError("Torus 9×9 trainer requires a non-empty rolling replay")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        if size >= count:
            return [int(index) for index in torch.randperm(size, generator=generator)[:count].tolist()]
        return [int(index) for index in torch.randint(size, (count,), generator=generator).tolist()]

    def train_fixed_budget(self, samples: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
        self.assert_optimizer_continuity()
        count = self.optimizer_steps_per_iteration * TORUS9_BATCH_SIZE
        indices = self._sample_indices(len(samples), seed=seed, count=count)
        device = next(self.model.parameters()).device
        updates: list[dict[str, object]] = []
        sampled_rows = [str(samples[index].get("replay_row_id", index)) for index in indices]
        source_counts: dict[str, int] = {}
        for index in indices:
            generation = str(samples[index].get("source_generation", "unknown"))
            source_counts[generation] = source_counts.get(generation, 0) + 1
        self.model.train()
        for update_index in range(self.optimizer_steps_per_iteration):
            batch_indices = indices[update_index * TORUS9_BATCH_SIZE:(update_index + 1) * TORUS9_BATCH_SIZE]
            if len(batch_indices) != TORUS9_BATCH_SIZE:
                raise AssertionError("Torus 9×9 fixed-budget trainer produced a partial batch")
            observations = torch.tensor([samples[index]["observation"] for index in batch_indices], dtype=torch.float32, device=device)
            policies = torch.tensor([samples[index]["pi"] for index in batch_indices], dtype=torch.float32, device=device)
            values = torch.tensor([samples[index]["z"] for index in batch_indices], dtype=torch.float32, device=device)
            before_parameters = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
            step_before = self.assert_optimizer_continuity()
            policy_logits, value_logits = self.model(observations)
            policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = -(values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
            total_loss = policy_loss + value_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Torus 9×9 training produced non-finite loss")
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradients = [parameter.grad.detach() for parameter in self.model.parameters() if parameter.grad is not None]
            grad_norm = torch.sqrt(sum(torch.sum(gradient.float() ** 2) for gradient in gradients)) if gradients else torch.tensor(0.0)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise FloatingPointError("Torus 9×9 training produced non-finite gradient")
            self.optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
                raise FloatingPointError("Torus 9×9 training produced non-finite parameter")
            self.update_count += 1
            step_after = self.assert_optimizer_continuity()
            if step_after != step_before + 1:
                raise RuntimeError("Torus 9×9 Adam step did not advance by one")
            parameter_delta, max_layer_delta = _parameter_delta(before_parameters, self.model)
            self.samples_consumed += TORUS9_BATCH_SIZE
            updates.append({
                "update": self.update_count,
                "batch_size": TORUS9_BATCH_SIZE,
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "gradient_norm": float(grad_norm),
                "parameter_delta": parameter_delta,
                "max_layer_delta": max_layer_delta,
                "adam_step_before": step_before,
                "adam_step_after": step_after,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            })
        unique_rows = len(set(sampled_rows))
        return {
            "updates": len(updates),
            "optimizer_steps": len(updates),
            "optimizer_updates_total": self.update_count,
            "replay_positions": len(samples),
            "samples": len(samples),
            "samples_consumed": count,
            "samples_consumed_total": self.samples_consumed,
            "unique_sample_rows": unique_rows,
            "reused_sample_rows": count - unique_rows,
            "effective_generations": sorted(int(key) for key in source_counts if key.isdigit()),
            "sampled_positions_by_generation": dict(sorted(source_counts.items(), key=lambda item: item[0])),
            "adam_step_before": updates[0]["adam_step_before"],
            "adam_step_after": updates[-1]["adam_step_after"],
            "batch_size": TORUS9_BATCH_SIZE,
            "batch_sizes": [int(row["batch_size"]) for row in updates],
            "mean_policy_loss": sum(float(row["policy_loss"]) for row in updates) / len(updates),
            "mean_value_loss": sum(float(row["value_loss"]) for row in updates) / len(updates),
            "mean_total_loss": sum(float(row["total_loss"]) for row in updates) / len(updates),
            "mean_gradient_norm": sum(float(row["gradient_norm"]) for row in updates) / len(updates),
            "mean_parameter_delta": sum(float(row["parameter_delta"]) for row in updates) / len(updates),
            "max_layer_delta": max(float(row["max_layer_delta"]) for row in updates),
            "updates_detail": updates,
        }


class Torus9OwnershipTrainer(_Torus9TrainingCore):
    """Fixed-budget Torus9 trainer for the WDL-only/ownership-loss A/B."""

    def __init__(
        self,
        model: Torus9OwnershipGraphNet,
        *,
        ownership_loss_enabled: bool,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_steps_per_iteration: int = TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    ) -> None:
        if not isinstance(model, Torus9OwnershipGraphNet):
            raise TypeError("Torus 9×9 ownership trainer requires Torus9OwnershipGraphNet")
        super().__init__(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_steps_per_iteration=optimizer_steps_per_iteration,
        )
        self.ownership_loss_enabled = bool(ownership_loss_enabled)

    def train_fixed_budget(self, samples: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
        self.assert_optimizer_continuity()
        count = self.optimizer_steps_per_iteration * TORUS9_BATCH_SIZE
        indices = self._sample_indices(len(samples), seed=seed, count=count)
        for sample in samples:
            validate_torus9_replay_sample(sample)
            if sample.get("ownership_target") is None:
                raise ValueError("Torus 9×9 ownership training requires ownership targets")
        device = next(self.model.parameters()).device
        updates: list[dict[str, object]] = []
        sampled_rows = [str(samples[index].get("replay_row_id", index)) for index in indices]
        source_counts: dict[str, int] = {}
        for index in indices:
            generation = str(samples[index].get("source_generation", "unknown"))
            source_counts[generation] = source_counts.get(generation, 0) + 1
        self.model.train()
        for update_index in range(self.optimizer_steps_per_iteration):
            batch_indices = indices[update_index * TORUS9_BATCH_SIZE:(update_index + 1) * TORUS9_BATCH_SIZE]
            if len(batch_indices) != TORUS9_BATCH_SIZE:
                raise AssertionError("Torus 9×9 fixed-budget trainer produced a partial batch")
            observations = torch.tensor([samples[index]["observation"] for index in batch_indices], dtype=torch.float32, device=device)
            policies = torch.tensor([samples[index]["pi"] for index in batch_indices], dtype=torch.float32, device=device)
            values = torch.tensor([samples[index]["z"] for index in batch_indices], dtype=torch.float32, device=device)
            ownership = torch.tensor([samples[index]["ownership_target"] for index in batch_indices], dtype=torch.long, device=device)
            before_parameters = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
            step_before = self.assert_optimizer_continuity()
            policy_logits, value_logits, ownership_logits = self.model.forward_auxiliary(observations)
            policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = -(values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
            ownership_loss = F.cross_entropy(ownership_logits.reshape(-1, 3), ownership.reshape(-1))
            ownership_weight = 1.0 if self.ownership_loss_enabled else 0.0
            # Keep a differentiable zero path in A so the new head has the
            # same Adam state/step semantics in both arms.
            total_loss = policy_loss + value_loss + ownership_weight * ownership_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Torus 9×9 auxiliary training produced non-finite loss")
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradients = [parameter.grad.detach() for parameter in self.model.parameters() if parameter.grad is not None]
            grad_norm = torch.sqrt(sum(torch.sum(gradient.float() ** 2) for gradient in gradients)) if gradients else torch.tensor(0.0)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise FloatingPointError("Torus 9×9 auxiliary training produced non-finite gradient")
            self.optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
                raise FloatingPointError("Torus 9×9 auxiliary training produced non-finite parameter")
            self.update_count += 1
            step_after = self.assert_optimizer_continuity()
            if step_after != step_before + 1:
                raise RuntimeError("Torus 9×9 Adam step did not advance by one")
            parameter_delta, max_layer_delta = _parameter_delta(before_parameters, self.model)
            self.samples_consumed += TORUS9_BATCH_SIZE
            updates.append({
                "update": self.update_count,
                "batch_size": TORUS9_BATCH_SIZE,
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "ownership_loss": float(ownership_loss.detach().cpu()),
                "ownership_loss_weight": ownership_weight,
                "total_loss": float(total_loss.detach().cpu()),
                "gradient_norm": float(grad_norm),
                "parameter_delta": parameter_delta,
                "max_layer_delta": max_layer_delta,
                "adam_step_before": step_before,
                "adam_step_after": step_after,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            })
        unique_rows = len(set(sampled_rows))
        return {
            "updates": len(updates),
            "optimizer_steps": len(updates),
            "optimizer_updates_total": self.update_count,
            "replay_positions": len(samples),
            "samples": len(samples),
            "samples_consumed": count,
            "samples_consumed_total": self.samples_consumed,
            "unique_sample_rows": unique_rows,
            "reused_sample_rows": count - unique_rows,
            "effective_generations": sorted(int(key) for key in source_counts if key.isdigit()),
            "sampled_positions_by_generation": dict(sorted(source_counts.items(), key=lambda item: item[0])),
            "adam_step_before": updates[0]["adam_step_before"],
            "adam_step_after": updates[-1]["adam_step_after"],
            "batch_size": TORUS9_BATCH_SIZE,
            "batch_sizes": [int(row["batch_size"]) for row in updates],
            "mean_policy_loss": sum(float(row["policy_loss"]) for row in updates) / len(updates),
            "mean_value_loss": sum(float(row["value_loss"]) for row in updates) / len(updates),
            "mean_ownership_loss": sum(float(row["ownership_loss"]) for row in updates) / len(updates),
            "ownership_loss_enabled": self.ownership_loss_enabled,
            "ownership_loss_weight": 1.0 if self.ownership_loss_enabled else 0.0,
            "mean_total_loss": sum(float(row["total_loss"]) for row in updates) / len(updates),
            "mean_gradient_norm": sum(float(row["gradient_norm"]) for row in updates) / len(updates),
            "mean_parameter_delta": sum(float(row["parameter_delta"]) for row in updates) / len(updates),
            "max_layer_delta": max(float(row["max_layer_delta"]) for row in updates),
            "updates_detail": updates,
        }


class Torus9OwnershipScoreTrainer(Torus9OwnershipTrainer):
    """Torus9 port of the proven Torus5 ownership+score trainer.

    The loss is exactly policy CE + WDL CE + ownership CE + normalized score
    MSE.  Only the model dimensions and the score normalization are adapted
    for the 9×9 graph; the target remains final referee margin in the sample's
    side-to-move perspective.
    """

    def __init__(
        self,
        model: Torus9OwnershipScoreGraphNet,
        *,
        score_loss_enabled: bool,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_steps_per_iteration: int = TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    ) -> None:
        if not isinstance(model, Torus9OwnershipScoreGraphNet):
            raise TypeError("Torus 9×9 score trainer requires Torus9OwnershipScoreGraphNet")
        _Torus9TrainingCore.__init__(
            self,
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_steps_per_iteration=optimizer_steps_per_iteration,
        )
        self.ownership_loss_enabled = True
        self.score_loss_enabled = bool(score_loss_enabled)

    def train_fixed_budget(
        self,
        samples: Sequence[Mapping[str, object]],
        *,
        seed: int,
        validate_samples: bool = True,
        timing: MutableMapping[str, object] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, object]:
        self.assert_optimizer_continuity()
        count = self.optimizer_steps_per_iteration * TORUS9_BATCH_SIZE
        indices = self._sample_indices(len(samples), seed=seed, count=count)
        if validate_samples:
            for sample in samples:
                validate_torus9_replay_sample(sample)
                if sample.get("ownership_target") is None or sample.get("score_target") is None:
                    raise ValueError("Torus 9×9 score training requires ownership and score targets")
        device = next(self.model.parameters()).device
        updates: list[dict[str, object]] = []
        stage_totals: dict[str, float] = {}
        timing_enabled = timing is not None

        def timed_stage(name: str, operation):
            if not timing_enabled:
                return operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage_totals[name] = stage_totals.get(name, 0.0) + (time.perf_counter() - started)
            return result

        sampled_rows = [str(samples[index].get("replay_row_id", index)) for index in indices]
        source_counts: dict[str, int] = {}
        for index in indices:
            generation = str(samples[index].get("source_generation", "unknown"))
            source_counts[generation] = source_counts.get(generation, 0) + 1
        self.model.train()
        for update_index in range(self.optimizer_steps_per_iteration):
            batch_indices = indices[update_index * TORUS9_BATCH_SIZE:(update_index + 1) * TORUS9_BATCH_SIZE]
            observations, policies, values, ownership, scores = timed_stage(
                "h2d_and_batch_construction_wall_time_sec",
                lambda: (
                    torch.tensor([samples[index]["observation"] for index in batch_indices], dtype=torch.float32, device=device),
                    torch.tensor([samples[index]["pi"] for index in batch_indices], dtype=torch.float32, device=device),
                    torch.tensor([samples[index]["z"] for index in batch_indices], dtype=torch.float32, device=device),
                    torch.tensor([samples[index]["ownership_target"] for index in batch_indices], dtype=torch.long, device=device),
                    torch.tensor([float(samples[index]["score_target"]) / TORUS9_SCORE_TARGET_NORMALIZATION for index in batch_indices], dtype=torch.float32, device=device),
                ),
            )
            before_parameters = timed_stage(
                "parameter_snapshot_wall_time_sec",
                lambda: {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()},
            )
            step_before = self.assert_optimizer_continuity()
            policy_logits, value_logits, ownership_logits, score_logits = timed_stage(
                "forward_wall_time_sec",
                lambda: self.model.forward_auxiliary(observations),
            )
            policy_loss, value_loss, ownership_loss, score_loss = timed_stage(
                "loss_wall_time_sec",
                lambda: (
                    -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean(),
                    -(values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean(),
                    F.cross_entropy(ownership_logits.reshape(-1, 3), ownership.reshape(-1)),
                    F.mse_loss(score_logits, scores),
                ),
            )
            score_weight = 1.0 if self.score_loss_enabled else 0.0
            total_loss = policy_loss + value_loss + ownership_loss + score_weight * score_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Torus 9×9 score training produced non-finite loss")
            timed_stage(
                "backward_wall_time_sec",
                lambda: (self.optimizer.zero_grad(set_to_none=True), total_loss.backward()),
            )
            gradients = [parameter.grad.detach() for parameter in self.model.parameters() if parameter.grad is not None]
            grad_norm = torch.sqrt(sum(torch.sum(gradient.float() ** 2) for gradient in gradients)) if gradients else torch.tensor(0.0)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise FloatingPointError("Torus 9×9 score training produced non-finite gradient")
            timed_stage("optimizer_wall_time_sec", self.optimizer.step)
            if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
                raise FloatingPointError("Torus 9×9 score training produced non-finite parameter")
            self.update_count += 1
            step_after = self.assert_optimizer_continuity()
            if step_after != step_before + 1:
                raise RuntimeError("Torus 9×9 Adam step did not advance by one")
            parameter_delta, max_layer_delta = _parameter_delta(before_parameters, self.model)
            self.samples_consumed += TORUS9_BATCH_SIZE
            updates.append({
                "update": self.update_count,
                "batch_size": TORUS9_BATCH_SIZE,
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "ownership_loss": float(ownership_loss.detach().cpu()),
                "score_loss_normalized": float(score_loss.detach().cpu()),
                "score_loss_weight": score_weight,
                "total_loss": float(total_loss.detach().cpu()),
                "gradient_norm": float(grad_norm),
                "parameter_delta": parameter_delta,
                "max_layer_delta": max_layer_delta,
                "adam_step_before": step_before,
                "adam_step_after": step_after,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            })
            if progress_callback is not None:
                progress_callback(
                    int(update_index + 1),
                    int(self.optimizer_steps_per_iteration),
                )
        if timing is not None:
            timing.update({
                "timing_clock": "perf_counter wall time",
                "cuda_synchronized": bool(device.type == "cuda"),
                "optimizer_steps": self.optimizer_steps_per_iteration,
                **stage_totals,
            })
        unique_rows = len(set(sampled_rows))
        return {
            "updates": len(updates),
            "optimizer_steps": len(updates),
            "optimizer_updates_total": self.update_count,
            "replay_positions": len(samples),
            "samples": len(samples),
            "samples_consumed": count,
            "samples_consumed_total": self.samples_consumed,
            "unique_sample_rows": unique_rows,
            "reused_sample_rows": count - unique_rows,
            "effective_generations": sorted(int(key) for key in source_counts if key.isdigit()),
            "sampled_positions_by_generation": dict(sorted(source_counts.items(), key=lambda item: item[0])),
            "adam_step_before": updates[0]["adam_step_before"],
            "adam_step_after": updates[-1]["adam_step_after"],
            "batch_size": TORUS9_BATCH_SIZE,
            "batch_sizes": [int(row["batch_size"]) for row in updates],
            "mean_policy_loss": sum(float(row["policy_loss"]) for row in updates) / len(updates),
            "mean_value_loss": sum(float(row["value_loss"]) for row in updates) / len(updates),
            "mean_ownership_loss": sum(float(row["ownership_loss"]) for row in updates) / len(updates),
            "ownership_loss_enabled": self.ownership_loss_enabled,
            "mean_score_loss_normalized": sum(float(row["score_loss_normalized"]) for row in updates) / len(updates),
            "score_loss_enabled": self.score_loss_enabled,
            "score_target_normalization": TORUS9_SCORE_TARGET_NORMALIZATION,
            "mean_total_loss": sum(float(row["total_loss"]) for row in updates) / len(updates),
            "mean_gradient_norm": sum(float(row["gradient_norm"]) for row in updates) / len(updates),
            "mean_parameter_delta": sum(float(row["parameter_delta"]) for row in updates) / len(updates),
            "max_layer_delta": max(float(row["max_layer_delta"]) for row in updates),
            "updates_detail": updates,
        }


def torus9_checkpoint_metadata(*, model: Torus9GraphNet, run_id: str, label: str, parent: str | None, model_seed: int, code: CodeIdentity, profile_fp: str, completed_games: int, replay_positions: int, optimizer_updates: int, samples_consumed: int, ownership_loss_enabled: bool | None = None, score_loss_enabled: bool | None = None, profile_id: str = TORUS9_CURRENT_PROFILE_ID, target_fingerprint: str = TORUS9_CURRENT_TARGET_FINGERPRINT, selfplay_contract_id: str = TORUS9_CURRENT_SELFPLAY_CONTRACT_ID, selfplay_contract_fingerprint: str | None = None, base_commit: str | None = None, komi: float = TORUS9_KOMI) -> dict[str, object]:
    auxiliary = isinstance(model, Torus9OwnershipGraphNet)
    heads = {
        "policy": [TORUS9_ACTION_COUNT],
        "value": [3],
    }
    if auxiliary:
        heads["ownership"] = [TORUS9_POINT_COUNT, 3]
    score_auxiliary = isinstance(model, Torus9OwnershipScoreGraphNet)
    if score_auxiliary:
        heads["score"] = [1]
    return {
        "checkpoint_schema_version": 1,
        "checkpoint_label": label,
        "run_id": run_id,
        "parent_or_source_run_identity": parent or run_id,
        "architecture_id": model.architecture_id,
        "architecture_config": model.architecture_config,
        "architecture_fingerprint": "sha256:" + hashlib.sha256(_canonical(model.architecture_config).encode("utf-8")).hexdigest(),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_hash": model_hash(model),
        "profile_id": profile_id,
        "profile_fingerprint": profile_fp,
        "base_commit": base_commit,
        "rules_profile_id": "graph-area-v1",
        "rules_fingerprint": rules_fingerprint_for(TORUS_9X9, float(komi)),
        "topology_id": TORUS9_TOPOLOGY_ID,
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "board_size": [9, 9],
        "point_id_order_identity": "row-major-yx:point_id=y*width+x",
        "komi": float(komi),
        "observation_schema_id": TORUS9_OBSERVATION_SCHEMA_ID,
        "observation_schema_version": TORUS9_OBSERVATION_SCHEMA_VERSION,
        "observation_fingerprint": TORUS9_OBSERVATION_FINGERPRINT,
        "observation_shape": [6, TORUS9_POINT_COUNT],
        "target_contract_id": TORUS9_TARGET_CONTRACT_ID,
        "target_contract_version": 1,
        "target_fingerprint": target_fingerprint,
        "value_head_semantics": TORUS9_VALUE_HEAD_SEMANTICS,
        "network_heads_and_shapes": heads,
        "model_initialization_seed": model_seed,
        "completed_games": completed_games,
        "valid_replay_positions": replay_positions,
        "optimizer_updates": optimizer_updates,
        "train_samples_consumed": samples_consumed,
        "git_commit": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "git_worktree_clean": code.working_tree_clean,
        "replay_policy": "rolling-recent-generations",
        "rolling_generations": TORUS9_ROLLING_GENERATIONS,
        "maximum_replay_positions": TORUS9_MAX_REPLAY_POSITIONS,
        "optimizer_steps_per_iteration": TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
        "samples_consumed_per_iteration": TORUS9_OPTIMIZER_STEPS_PER_ITERATION * TORUS9_BATCH_SIZE,
        "selfplay_contract_id": selfplay_contract_id,
        "selfplay_contract_fingerprint": selfplay_contract_fingerprint,
        "auxiliary_heads": auxiliary,
        "auxiliary_target_contracts": ({
            "source": TORUS9_AUXILIARY_TARGET_SOURCE,
            "ownership": TORUS9_OWNERSHIP_TARGET_CONTRACT_ID,
            "ownership_loss_weight": 1.0 if ownership_loss_enabled else 0.0,
            **({
                "score": TORUS9_SCORE_TARGET_CONTRACT_ID,
                "score_target_normalization": TORUS9_SCORE_TARGET_NORMALIZATION,
            } if score_auxiliary else {}),
        } if auxiliary else None),
        "ownership_loss_enabled": ownership_loss_enabled,
        "score_loss_enabled": score_loss_enabled,
    }


def torus9_save_checkpoint(path: Path, *, model: Torus9GraphNet, optimizer: torch.optim.Optimizer | None, metadata: Mapping[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(metadata)
    enriched["model_hash"] = model_hash(model)
    payload = {"checkpoint_schema_version": 1, "metadata": enriched, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None}
    torch.save(payload, path)
    path.with_suffix(".metadata.json").write_text(json.dumps(_jsonable(enriched), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return enriched


def torus9_model_from_metadata(metadata: Mapping[str, object]) -> Torus9GraphNet:
    architecture = metadata.get("architecture_config", {})
    if not isinstance(architecture, Mapping):
        raise ValueError("Torus 9×9 checkpoint architecture metadata is malformed")
    if metadata.get("architecture_id") != TORUS9_CURRENT_ARCHITECTURE_ID:
        raise ValueError("Only the current Golden Torus9 checkpoint architecture is supported")
    hidden = int(architecture.get("hidden", TORUS9_HIDDEN))
    blocks = int(architecture.get("blocks", TORUS9_BLOCKS))
    return Torus9CurrentGraphNet(hidden=hidden, blocks=blocks)


def torus9_load_checkpoint(path: Path, *, model: Torus9GraphNet, optimizer: torch.optim.Optimizer | None = None, expected: Mapping[str, object] | None = None, device: str | torch.device = "cpu") -> dict[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    metadata = dict(payload["metadata"])
    if metadata.get("architecture_id") != TORUS9_CURRENT_ARCHITECTURE_ID:
        raise ValueError("Only the current Golden Torus9 checkpoint architecture is supported")
    if not isinstance(model, Torus9CurrentGraphNet):
        raise ValueError("Current Golden Torus9 checkpoints require Torus9CurrentGraphNet")
    metadata_komi = float(metadata.get("komi", -1.0))
    if metadata.get("topology_fingerprint") != TORUS9_TOPOLOGY_FINGERPRINT or metadata.get("board_size") != [9, 9] or metadata.get("rules_fingerprint") != rules_fingerprint_for(TORUS_9X9, metadata_komi) or metadata_komi not in {0.5, 1.5, 2.5}:
        raise ValueError("Torus 9×9 checkpoint topology/komi mismatch")
    expected_heads = {"policy": [82], "value": [3]}
    auxiliary = isinstance(model, Torus9OwnershipGraphNet)
    score_auxiliary = isinstance(model, Torus9OwnershipScoreGraphNet)
    if auxiliary:
        expected_heads["ownership"] = [TORUS9_POINT_COUNT, 3]
    if score_auxiliary:
        expected_heads["score"] = [1]
    if metadata.get("network_heads_and_shapes") != expected_heads or metadata.get("auxiliary_heads") is not auxiliary:
        raise ValueError("Torus 9×9 checkpoint head contract mismatch")
    if metadata.get("architecture_config") != model.architecture_config:
        raise ValueError("Torus 9×9 checkpoint architecture does not match the supplied model")
    if expected:
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Torus 9×9 checkpoint metadata mismatch for {key}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    actual = model_hash(model)
    if actual != metadata.get("model_hash"):
        raise ValueError("Torus 9×9 checkpoint model hash mismatch")
    return metadata


def torus9_restore_optimizer_state(
    path: Path,
    *,
    model: Torus9OwnershipGraphNet,
    optimizer: torch.optim.Optimizer,
    source_parameter_count: int,
    device: str | torch.device = "cpu",
) -> int:
    """Continue M8 Adam state while initializing only the new head state.

    The M8 checkpoint has one optimizer parameter group for the policy/WDL
    network.  The ownership experiment uses the same group hyperparameters,
    copies those states by parameter order, and gives each new ownership
    parameter zero moments at the inherited Adam step.  This makes the
    difference between A and B a loss gradient, not a hidden optimizer reset.
    """
    if source_parameter_count <= 0 or source_parameter_count >= len(tuple(model.parameters())):
        raise ValueError("Invalid source parameter count for Torus 9×9 optimizer continuation")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    saved_optimizer = payload.get("optimizer_state_dict")
    if not isinstance(saved_optimizer, Mapping):
        raise ValueError("Torus 9×9 source checkpoint has no optimizer state")
    groups = saved_optimizer.get("param_groups")
    state = saved_optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(state, Mapping):
        raise ValueError("Torus 9×9 source optimizer state is malformed")
    source_ids = list(groups[0].get("params", ()))
    current_params = list(model.parameters())
    if len(source_ids) != source_parameter_count or len(current_params) <= source_parameter_count:
        raise ValueError("Torus 9×9 source/auxiliary optimizer parameter boundary drift")
    inherited_steps: set[int] = set()
    for parameter, source_id in zip(current_params[:source_parameter_count], source_ids):
        saved = state.get(source_id)
        if not isinstance(saved, Mapping):
            raise ValueError("Torus 9×9 source optimizer is missing a parameter state")
        copied: dict[str, object] = {}
        for key, value in saved.items():
            copied[str(key)] = value.detach().clone().to(parameter.device) if torch.is_tensor(value) else value
        optimizer.state[parameter] = copied
        step = copied.get("step")
        if step is not None:
            inherited_steps.add(int(step.item() if torch.is_tensor(step) else step))
    if len(inherited_steps) != 1:
        raise ValueError("Torus 9×9 source Adam state has inconsistent steps")
    inherited_step = next(iter(inherited_steps))
    for parameter in current_params[source_parameter_count:]:
        optimizer.state[parameter] = {
            "step": torch.tensor(float(inherited_step), device=parameter.device),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    saved_group = groups[0]
    current_group = optimizer.param_groups[0]
    for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
        if key in saved_group:
            current_group[key] = saved_group[key]
    return inherited_step


def torus9_checkpoint_info(path: Path) -> dict[str, object]:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    model = torus9_model_from_metadata(metadata)
    loaded = torus9_load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]})
    return {"path": str(path), "metadata": loaded, "model_hash": model_hash(model), "artifact_sha256": file_sha256(path)}


def torus9_hoeffding_interval(scores: Sequence[float], alpha: float = 0.05) -> list[float]:
    if not scores or not 0.0 < alpha < 1.0:
        raise ValueError("Torus 9×9 Arena interval requires non-empty scores")
    mean = sum(scores) / len(scores)
    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * len(scores)))
    return [max(0.0, mean - radius), min(1.0, mean + radius)]


def summarize_torus9_arena(records: Sequence[Mapping[str, object]], *, candidate_label: str, reference_label: str, pairs: int) -> dict[str, object]:
    valid = [row for row in records if row.get("technical_termination") is None]
    technical = [row for row in records if row.get("technical_termination") is not None]
    counts = {"W": sum(row.get("mapped_result") == "A_WIN" for row in valid), "L": sum(row.get("mapped_result") == "B_WIN" for row in valid), "D": sum(row.get("mapped_result") == "DRAW" for row in valid)}
    grouped: dict[str, list[Mapping[str, object]]] = {}
    all_grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in records:
        all_grouped.setdefault(str(row["pair_id"]), []).append(row)
    for row in valid:
        grouped.setdefault(str(row["pair_id"]), []).append(row)
    pair_scores = []
    technical_pairs: list[str] = []
    for pair_id, pair in sorted(all_grouped.items()):
        if any(row.get("technical_termination") is not None for row in pair):
            technical_pairs.append(pair_id)
            continue
        if len(pair) != 2:
            raise ValueError(f"Nontechnical Torus 9×9 Arena pair {pair_id} is incomplete")
        scores = [1.0 if row["mapped_result"] == "A_WIN" else 0.5 if row["mapped_result"] == "DRAW" else 0.0 for row in pair]
        pair_scores.append(sum(scores) / 2.0)
    if technical:
        interval = None
    else:
        interval = torus9_hoeffding_interval(pair_scores)
    black_wins = sum(row.get("formal_result") == "BLACK" for row in valid)
    white_wins = sum(row.get("formal_result") == "WHITE" for row in valid)
    draws = sum(row.get("formal_result") == "DRAW" for row in valid)
    black_rate = black_wins / len(valid) if valid else None
    if black_rate is None:
        black_ci = None
    else:
        z = 1.959963984540054
        denominator = 1.0 + z * z / len(valid)
        centre = (black_rate + z * z / (2.0 * len(valid))) / denominator
        radius = z * math.sqrt(black_rate * (1.0 - black_rate) / len(valid) + z * z / (4.0 * len(valid) * len(valid))) / denominator
        black_ci = [max(0.0, centre - radius), min(1.0, centre + radius)]
    margins = [float(row["margin_black"]) for row in valid if row.get("margin_black") is not None]
    return {
        "candidate": candidate_label,
        "reference": reference_label,
        "pairs_declared": pairs,
        "pairs_valid": len(pair_scores),
        "technical_pairs": technical_pairs,
        "games": len(records),
        "valid_games": len(valid),
        "technical_games": len(technical),
        "W/L/D": [counts["W"], counts["L"], counts["D"]],
        "black_wins": black_wins,
        "white_wins": white_wins,
        "black_win_rate": black_rate,
        "black_win_rate_95_percent_ci": black_ci,
        "black_win_rate_bias": None if black_rate is None else abs(black_rate - 0.5),
        "raw_score_margin_histogram": margins,
        "wins": counts["W"],
        "losses": counts["L"],
        "draws": counts["D"],
        "mean_pair_score": sum(pair_scores) / len(pair_scores) if pair_scores else None,
        "95_percent_hoeffding_interval": interval,
        "technical_by_reason": {str(reason): sum(row.get("technical_termination") == reason for row in technical) for reason in sorted({str(row.get("technical_termination")) for row in technical})},
        "arena_contract_fingerprint": TORUS9_ARENA_CONTRACT_FINGERPRINT,
        "technical_fail_closed": True,
    }


def _candidate_start(
    master_seed: int,
    prefix_length: int,
    candidate_index: int,
    *,
    komi: float = TORUS9_KOMI,
) -> dict[str, object]:
    seed = derive_seed(master_seed, prefix_length, candidate_index)
    rng = random.Random(seed)
    state = initial_state(topology=TORUS_9X9, komi=float(komi))
    trace: list[int] = []
    for _ in range(prefix_length):
        choices = tuple(action for action in legal_actions(state) if action != PASS)
        if not choices:
            raise ValueError("No legal non-pass action for frozen start")
        action = int(rng.choice(choices))
        state = apply_action(state, action).after
        trace.append(action)
    return {"prefix_length": prefix_length, "candidate_index": candidate_index, "candidate_seed": seed, "trace": trace, "state": torus9_state_identity(state)}


def generate_torus9_evaluation_starts(
    *,
    master_seed: int,
    accepted_per_stratum: int = 8,
    komi: float = TORUS9_KOMI,
) -> tuple[dict[str, object], ...]:
    accepted: list[dict[str, object]] = []
    seen: set[str] = set()
    for prefix_length in (2, 4, 6, 8, 10, 12, 14, 16):
        count = 0
        candidate_index = 0
        while count < accepted_per_stratum:
            row = _candidate_start(master_seed, prefix_length, candidate_index, komi=komi)
            exact = "sha256:" + hashlib.sha256(_canonical(row["state"]).encode("utf-8")).hexdigest()
            if exact not in seen:
                row.update({"start_id": f"prefix-{prefix_length:02d}-accepted-{count:02d}", "exact_identity_fingerprint": exact})
                accepted.append(row)
                seen.add(exact)
                count += 1
            candidate_index += 1
    corpus_fp = "sha256:" + hashlib.sha256(_canonical(accepted).encode("utf-8")).hexdigest()
    for row in accepted:
        row["corpus_fingerprint"] = corpus_fp
    return tuple(accepted)


def torus9_first_move_statistics(records: Sequence[Torus9SelfPlayGameRecord], *, source: str) -> dict[str, object]:
    black_wins = white_wins = draws = technical = 0
    raw: list[float] = []
    margins: list[float] = []
    for record in records:
        get = record.get if isinstance(record, Mapping) else lambda key: getattr(record, key)
        if get("technical_termination") is not None:
            technical += 1
            continue
        state = torus9_state_from_identity(get("start_state"))
        for action in get("final_action_trace"):
            state = apply_action(state, action).after
        if not state.is_terminal:
            raise ValueError("First-move statistics received nonterminal record")
        score = score_terminal(state)
        if score.komi != TORUS9_KOMI:
            raise ValueError("First-move statistics received non-0.5 komi")
        if get("formal_result") == "BLACK":
            black_wins += 1
        elif get("formal_result") == "WHITE":
            white_wins += 1
        else:
            draws += 1
        raw.append(float(score.black_area - score.white_area))
        margins.append(float(score.margin_black))
    games = black_wins + white_wins + draws
    rate = black_wins / games if games else None
    if games:
        z = 1.959963984540054
        denominator = 1.0 + z * z / games
        centre = (rate + z * z / (2.0 * games)) / denominator
        radius = z * math.sqrt(rate * (1.0 - rate) / games + z * z / (4.0 * games * games)) / denominator
        ci = [max(0.0, centre - radius), min(1.0, centre + radius)]
    else:
        ci = None
    return {
        "source": source,
        "komi": TORUS9_KOMI,
        "games": games,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "technical_games": technical,
        "black_win_rate": rate,
        "black_win_rate_95_percent_ci": ci,
        "raw_black_area_advantage_mean": sum(raw) / len(raw) if raw else None,
        "raw_black_area_advantage_values": raw,
        "final_black_margin_mean": sum(margins) / len(margins) if margins else None,
        "final_black_margin_values": margins,
    }


def torus9_contract_proof() -> dict[str, object]:
    """Run the cheap, independent contract assertions before long self-play."""
    topology = TORUS_9X9
    neighbors = topology.adjacency
    assert graph_diameter(TORUS_5X5) == 4
    assert graph_diameter(TORUS_9X9) == 8
    assert TORUS9_BLOCKS == 8
    assert topology.point_count == TORUS9_POINT_COUNT
    assert len({tuple(row) for row in neighbors}) > 1
    assert all(len(row) == 4 and len(set(row)) == 4 and point not in row for point, row in enumerate(neighbors))
    assert all(point in neighbors[neighbor] for point, row in enumerate(neighbors) for neighbor in row)
    reached = {0}
    frontier = [0]
    while frontier:
        point = frontier.pop()
        for neighbor in neighbors[point]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    assert len(reached) == TORUS9_POINT_COUNT
    for y in range(9):
        for x in range(9):
            point = topology.coordinate_to_point(x, y)
            assert topology.point_to_coordinate(point) == (x, y)
    assert set(neighbors[topology.coordinate_to_point(0, 0)]) == {topology.coordinate_to_point(0, 8), topology.coordinate_to_point(1, 0), topology.coordinate_to_point(0, 1), topology.coordinate_to_point(8, 0)}
    state = initial_state(topology=TORUS_9X9, komi=TORUS9_KOMI)
    context = prepare_legal_actions(state)
    observation = build_torus9_observation_bundle(state, legal_context=context)
    model = Torus9GraphNet()
    policy, value = model(observation.tensor.unsqueeze(0))
    assert tuple(policy.shape) == (1, TORUS9_ACTION_COUNT) and tuple(value.shape) == (1, 3)
    assert context.action_mask[TORUS9_PASS_INDEX] is True
    pass_one = apply_action(state, PASS).after
    terminal = apply_action(pass_one, PASS).after
    assert terminal.is_terminal and result_from_terminal(terminal).komi == TORUS9_KOMI
    return {
        "status": "PASS",
        "points": TORUS9_POINT_COUNT,
        "actions": TORUS9_ACTION_COUNT,
        "pass_index": TORUS9_PASS_INDEX,
        "komi": TORUS9_KOMI,
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "rules_fingerprint": TORUS9_RULES_FINGERPRINT,
        "observation_fingerprint": TORUS9_OBSERVATION_FINGERPRINT,
        "architecture": model.architecture_config,
        "architecture_fingerprint": "sha256:" + hashlib.sha256(_canonical(model.architecture_config).encode("utf-8")).hexdigest(),
        "receptive_field": {
            "torus5_diameter": graph_diameter(TORUS_5X5),
            "torus9_diameter": graph_diameter(TORUS_9X9),
            "message_passing_blocks": TORUS9_BLOCKS,
            "canonical_torus9_distant_dependency": graph_distance(TORUS_9X9, 0, 40) <= TORUS9_BLOCKS,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "fixtures": {"corner_like": [0, 8, 72, 80], "horizontal_wrap": True, "vertical_wrap": True, "central": 40},
        "no_ownership_or_score_head": True,
    }
