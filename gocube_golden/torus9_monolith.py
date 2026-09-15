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
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from pathlib import Path
import random
import resource
import threading
import time
from typing import Any, Mapping, MutableMapping, Sequence

torch = importlib.import_module("torch")
nn = torch.nn
F = importlib.import_module("torch.nn.functional")

from .arena_contract import SearchSettings
from .neural import model_hash
from .provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256
from .result import Winner, result_from_terminal
from .rules import IllegalMoveError, LegalActionContext, apply_action, legal_actions, prepare_legal_actions
from .scoring import score_terminal
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
from .state import BLACK, EMPTY, PASS, WHITE, GoldenState, Stone, initial_state
from .training import (
    GOLDEN_AUXILIARY_TARGET_SOURCE,
    OWNERSHIP_TARGET_CONTRACT_ID,
    ownership_target as golden_ownership_target,
)
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
    TORUS9_PROFILE_ID,
    TORUS9_ROLLING_GENERATIONS,
    TORUS9_RULES_FINGERPRINT,
    TORUS9_SELFPLAY_CONTRACT_FINGERPRINT,
    TORUS9_SELFPLAY_CONTRACT_ID,
    torus9_selfplay_contract_fingerprint,
    TORUS9_TARGET_CONTRACT_ID,
    TORUS9_TARGET_FINGERPRINT,
    TORUS9_WORKERS,
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_BLOCKS,
    TORUS9_CURRENT_DIRICHLET_ALPHA,
    TORUS9_CURRENT_HIDDEN,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    current_torus9_selfplay_contract_fingerprint,
    load_torus9_profile,
    profile_fingerprint,
)


TORUS9_VALUE_HEAD_SEMANTICS = "side-to-move:[WIN,DRAW,LOSS]"
TORUS9_LEGACY_ARCHITECTURE_ID = "GoldenGraphNetV1-Torus9"
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
TORUS9_OWNERSHIP_TARGET_CONTRACT_ID = OWNERSHIP_TARGET_CONTRACT_ID
TORUS9_AUXILIARY_TARGET_SOURCE = GOLDEN_AUXILIARY_TARGET_SOURCE
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


def torus9_state_from_identity(identity: Mapping[str, object]) -> GoldenState:
    if identity.get("topology_id") != TORUS9_TOPOLOGY_ID or identity.get("topology_fingerprint") != TORUS9_TOPOLOGY_FINGERPRINT:
        raise ValueError("Torus 9×9 state topology identity drift")
    if identity.get("rules_fingerprint") != TORUS9_RULES_FINGERPRINT or float(identity.get("komi", -1.0)) != TORUS9_KOMI:
        raise ValueError("Torus 9×9 state rules/komi identity drift")
    return GoldenState(
        stones=tuple(Stone(int(value)) for value in identity["stones"]),  # type: ignore[index]
        side_to_move=Stone(int(identity["side_to_move"])),  # type: ignore[arg-type]
        superko_history=tuple(tuple(int(value) for value in row) for row in identity["superko_history"]),  # type: ignore[index]
        consecutive_passes=int(identity["consecutive_passes"]),
        topology=TORUS_9X9,
        rules_id=str(identity["rules_id"]),
        rules_fingerprint=str(identity["rules_fingerprint"]),
        komi=float(identity["komi"]),
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
    destination[5].fill_(TORUS9_KOMI)


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


class Torus9ExecutionActivity:
    """Thread-safe activity counters for proving lane/forward concurrency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_mcts = 0
        self._active_inference_requests = 0
        self._max_active_mcts = 0
        self._max_active_inference_requests = 0

    def enter_mcts(self) -> None:
        with self._lock:
            self._active_mcts += 1
            self._max_active_mcts = max(self._max_active_mcts, self._active_mcts)

    def exit_mcts(self) -> None:
        with self._lock:
            self._active_mcts -= 1

    def enter_inference_request(self) -> None:
        with self._lock:
            self._active_inference_requests += 1
            self._max_active_inference_requests = max(
                self._max_active_inference_requests,
                self._active_inference_requests,
            )

    def exit_inference_request(self) -> None:
        with self._lock:
            self._active_inference_requests -= 1

    @property
    def telemetry(self) -> dict[str, object]:
        with self._lock:
            return {
                "max_active_mcts_lanes": self._max_active_mcts,
                "max_active_inference_requests": self._max_active_inference_requests,
            }


@dataclass
class _Torus9InferenceRequest:
    state: GoldenState
    legal_context: LegalActionContext
    ticket: int
    done: threading.Event
    result: Evaluation | None = None
    error: BaseException | None = None


class Torus9InferenceCoordinator:
    """Execution-only request coalescer shared by independent self-play lanes.

    A lane blocks on its own request until its row is returned.  The
    coordinator may combine requests from other lanes into one forward, but
    it never shares trees, state, RNG, or search results between lanes.
    """

    def __init__(
        self,
        model: Torus9GraphNet,
        *,
        device: str | torch.device = "cpu",
        batch_cap: int,
        wait_ms: float,
    ) -> None:
        if int(batch_cap) <= 0:
            raise ValueError("self-play inference batch cap must be positive")
        if float(wait_ms) < 0.0 or not math.isfinite(float(wait_ms)):
            raise ValueError("self-play inference batch wait ms must be finite and non-negative")
        self.batch_cap = int(batch_cap)
        self.wait_ms = float(wait_ms)
        self._evaluator = Torus9NeuralEvaluator(model, device=device)
        self._queue: Queue[object] = Queue()
        self._stop = object()
        self._ticket = 0
        self._ticket_lock = threading.Lock()
        self._closed = False
        self._batches: list[int] = []
        self._thread = threading.Thread(target=self._serve, name="torus9-inference-coordinator", daemon=True)
        self._thread.start()

    def _next_ticket(self) -> int:
        with self._ticket_lock:
            ticket = self._ticket
            self._ticket += 1
            return ticket

    def evaluate_prepared(self, state: GoldenState, legal_context: LegalActionContext) -> Evaluation:
        if self._closed:
            raise SearchError("Torus 9×9 inference coordinator is closed")
        request = _Torus9InferenceRequest(
            state=state,
            legal_context=legal_context,
            ticket=self._next_ticket(),
            done=threading.Event(),
        )
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise SearchError(f"Torus 9×9 inference request {request.ticket} failed: {request.error}") from request.error
        if request.result is None:
            raise SearchError(f"Torus 9×9 inference request {request.ticket} returned no result")
        return request.result

    def _serve(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._stop:
                return
            first = item
            if not isinstance(first, _Torus9InferenceRequest):
                continue
            batch = [first]
            deadline = time.monotonic() + self.wait_ms / 1000.0
            while len(batch) < self.batch_cap:
                timeout = max(0.0, deadline - time.monotonic()) if self.wait_ms > 0.0 else 0.0
                try:
                    item = self._queue.get(timeout=timeout)
                except Empty:
                    break
                if item is self._stop:
                    self._queue.put(self._stop)
                    break
                if isinstance(item, _Torus9InferenceRequest):
                    batch.append(item)
            self._dispatch(batch)

    def _dispatch(self, batch: Sequence[_Torus9InferenceRequest]) -> None:
        try:
            evaluations = self._evaluator.evaluate_prepared_batch(
                [request.state for request in batch],
                [request.legal_context for request in batch],
            )
            if len(evaluations) != len(batch):
                raise SearchError("Torus 9×9 coordinator returned the wrong row count")
            self._batches.append(len(batch))
            for request, evaluation in zip(batch, evaluations):
                request.result = evaluation
        except BaseException as exc:
            for request in batch:
                request.error = exc
        finally:
            for request in batch:
                request.done.set()

    @property
    def telemetry(self) -> dict[str, object]:
        rows = sum(self._batches)
        return {
            "mode": "coalesced",
            "batch_cap": self.batch_cap,
            "wait_ms": self.wait_ms,
            "forward_calls": len(self._batches),
            "total_rows": rows,
            "batch_rows": list(self._batches),
            "mean_batch_rows": rows / len(self._batches) if self._batches else 0.0,
            "max_batch_rows": max(self._batches, default=0),
            "max_concurrent_forwards": 1 if self._batches else 0,
            "lock_scope": "single_batched_model_forward_only",
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._stop)
        self._thread.join()


class Torus9UncoalescedInference:
    """Baseline execution backend: one model forward per request."""

    def __init__(self, model: Torus9GraphNet, *, device: str | torch.device = "cpu") -> None:
        self._evaluator = Torus9NeuralEvaluator(model, device=device)
        self._lock = threading.Lock()
        self._rows = 0
        self._max_concurrent_forwards = 0
        self._active_forwards = 0

    def evaluate_prepared(self, state: GoldenState, legal_context: LegalActionContext) -> Evaluation:
        with self._lock:
            self._rows += 1
            self._active_forwards += 1
            self._max_concurrent_forwards = max(self._max_concurrent_forwards, self._active_forwards)
            try:
                return self._evaluator.evaluate_prepared(state, legal_context)
            finally:
                self._active_forwards -= 1

    @property
    def telemetry(self) -> dict[str, object]:
        return {
            "mode": "uncoalesced",
            "batch_cap": 1,
            "wait_ms": 0.0,
            "forward_calls": self._rows,
            "total_rows": self._rows,
            "batch_rows": [1] * self._rows,
            "mean_batch_rows": 1.0 if self._rows else 0.0,
            "max_batch_rows": 1 if self._rows else 0,
            "max_concurrent_forwards": self._max_concurrent_forwards,
            "lock_scope": "single_model_forward_only",
        }


class _Torus9LaneEvaluator:
    """Per-lane accounting wrapper around either execution backend."""

    def __init__(
        self,
        backend: Torus9InferenceCoordinator | Torus9UncoalescedInference,
        activity: Torus9ExecutionActivity | None = None,
    ) -> None:
        self.backend = backend
        self.activity = activity
        self.nn_evaluations = 0

    def evaluate_prepared(self, state: GoldenState, legal_context: LegalActionContext) -> Evaluation:
        self.nn_evaluations += 1
        if self.activity is not None:
            self.activity.enter_inference_request()
        try:
            return self.backend.evaluate_prepared(state, legal_context)
        finally:
            if self.activity is not None:
                self.activity.exit_inference_request()


@dataclass
class _Torus9BatchedTree:
    state: GoldenState
    evaluator: Torus9NeuralEvaluator
    root: _Node
    evaluator_calls: int = 0


class Torus9BatchedPUCT:
    """Run independent Torus9 PUCT trees while coalescing leaf evaluations.

    Tree selection and backup are the same deterministic no-transposition PUCT
    operations as ``SequentialPUCT``.  Only the neural calls are coalesced
    across independent Arena games, which keeps the Arena protocol's search
    semantics fixed while making the inference batch observable.
    """

    def __init__(
        self,
        settings: SearchSettings | None = None,
        *,
        adapter: GoldenSearchAdapter | None = None,
        max_batch_rows: int | None = None,
        inference_batch_wait_ms: float = 0.0,
    ) -> None:
        self.settings = settings or SearchSettings()
        self.adapter = adapter or GoldenSearchAdapter()
        self.max_batch_rows = max_batch_rows
        if inference_batch_wait_ms < 0.0 or not math.isfinite(float(inference_batch_wait_ms)):
            raise ValueError("Torus 9×9 Arena inference batch wait must be finite and non-negative")
        self.inference_batch_wait_ms = float(inference_batch_wait_ms)
        self.inference_batch_rows: list[int] = []

    def _tie_key(self, state: GoldenState, action: int | str) -> int:
        return self.adapter.action_index(state, action)

    def _select(self, node: _Node) -> tuple[int | str, _Edge]:
        total_visits = sum(edge.visits for edge in node.edges.values())
        scale = math.sqrt(total_visits + 1.0)
        best_value = -float("inf")
        candidates: list[tuple[int | str, _Edge]] = []
        for action, edge in node.edges.items():
            q = edge.q if edge.visits else float(self.settings.fpu)
            score = q + float(self.settings.cpuct) * edge.prior * scale / (1.0 + edge.visits)
            if score > best_value + 1e-15:
                best_value = score
                candidates = [(action, edge)]
            elif abs(score - best_value) <= 1e-15:
                candidates.append((action, edge))
        if not candidates:
            raise SearchError("Batched Torus 9×9 PUCT could not select a legal edge")
        return min(candidates, key=lambda item: self._tie_key(node.state, item[0]))

    def _expand(self, node: _Node, evaluation: Evaluation) -> float:
        if self.adapter.is_terminal(node.state):
            return self.adapter.terminal_utility(node.state)
        utility = wdl_to_side_to_move_utility(evaluation.wdl)
        context = self.adapter.prepare_legal_actions(node.state)
        legal = context.actions
        if not legal:
            raise SearchError("Nonterminal Golden state exposed no legal actions")
        priors = _policy_for_legal(evaluation, node.state, legal, self.adapter)
        node.edges = {action: _Edge(prior=priors[action]) for action in legal}
        node.legal_context = context
        node.expanded = True
        return utility

    def _evaluate_entries(
        self,
        trees: Sequence[_Torus9BatchedTree],
        entries: Sequence[tuple[int, _Node]],
    ) -> dict[int, float]:
        utilities: dict[int, float] = {}
        groups: dict[int, list[tuple[int, _Node]]] = {}
        for tree_index, node in entries:
            groups.setdefault(id(trees[tree_index].evaluator), []).append((tree_index, node))
        for group_entries in groups.values():
            evaluator = trees[group_entries[0][0]].evaluator
            limit = self.max_batch_rows or len(group_entries)
            if limit <= 0:
                raise ValueError("Batched Torus 9×9 PUCT max_batch_rows must be positive")
            for offset in range(0, len(group_entries), limit):
                chunk = group_entries[offset:offset + limit]
                states = [node.state for _, node in chunk]
                contexts = [self.adapter.prepare_legal_actions(state) for state in states]
                # This is an execution-only scheduler wait.  It deliberately
                # occurs outside the model call and therefore cannot alter
                # search semantics, targets, or deterministic tie-breaking.
                if self.inference_batch_wait_ms > 0.0:
                    time.sleep(self.inference_batch_wait_ms / 1000.0)
                batch_method = getattr(evaluator, "evaluate_prepared_batch", None)
                if callable(batch_method):
                    evaluations = tuple(batch_method(states, contexts))
                else:
                    evaluations = tuple(evaluator.evaluate_prepared(state, context) for state, context in zip(states, contexts))
                if len(evaluations) != len(chunk):
                    raise SearchError("Batched Torus 9×9 evaluator returned the wrong row count")
                self.inference_batch_rows.append(len(chunk))
                for (tree_index, node), evaluation in zip(chunk, evaluations):
                    trees[tree_index].evaluator_calls += 1
                    utilities[tree_index] = self._expand(node, evaluation)
        return utilities

    def search(
        self,
        states: Sequence[GoldenState],
        evaluators: Sequence[Torus9NeuralEvaluator],
        *,
        seeds: Sequence[int] | None = None,
    ) -> tuple[SearchResult, ...]:
        del seeds  # deterministic tie-breaking is the Arena contract
        if not states or len(states) != len(evaluators):
            raise SearchError("Batched Torus 9×9 PUCT requires matching non-empty states/evaluators")
        trees: list[_Torus9BatchedTree] = []
        for state, evaluator in zip(states, evaluators):
            if state.is_terminal:
                raise SearchError("Batched Torus 9×9 search cannot start from a terminal state")
            trees.append(_Torus9BatchedTree(state=state, evaluator=evaluator, root=_Node(state)))
        self._evaluate_entries(trees, [(index, tree.root) for index, tree in enumerate(trees)])
        for _ in range(self.settings.simulations):
            pending: list[tuple[int, _Node, list[_Edge]]] = []
            terminal_backups: list[tuple[int, list[_Edge], float]] = []
            for index, tree in enumerate(trees):
                node = tree.root
                path: list[_Edge] = []
                while node.expanded:
                    action, edge = self._select(node)
                    if edge.child is None:
                        edge.child = _Node(self.adapter.apply_action(node.state, action))
                    path.append(edge)
                    node = edge.child
                if self.adapter.is_terminal(node.state):
                    terminal_backups.append((index, path, self.adapter.terminal_utility(node.state)))
                else:
                    pending.append((index, node, path))
            leaf_utilities = self._evaluate_entries(trees, [(index, node) for index, node, _ in pending])
            for index, path, utility in terminal_backups:
                for edge in reversed(path):
                    utility = _child_to_parent_utility(utility)
                    edge.visits += 1
                    edge.value_sum += utility
            for index, node, path in pending:
                utility = leaf_utilities[index]
                for edge in reversed(path):
                    utility = _child_to_parent_utility(utility)
                    edge.visits += 1
                    edge.value_sum += utility
        results: list[SearchResult] = []
        for tree in trees:
            if tree.root.legal_context is None:
                raise SearchError("Batched Torus 9×9 search root lacks legal context")
            legal = tree.root.legal_context.actions
            action_space = self.adapter.action_space(tree.state)
            visit_map = {action: tree.root.edges[action].visits for action in legal}
            root_visits = tuple(visit_map.get(action, 0) for action in action_space)
            total = sum(root_visits)
            if total <= 0:
                raise SearchError("Batched Torus 9×9 search produced zero root visits")
            root_q = tuple(
                tree.root.edges[action].q if action in tree.root.edges and tree.root.edges[action].visits else None
                for action in action_space
            )
            pi = tuple(count / total for count in root_visits)
            maximum = max(visit_map.values())
            candidates = [action for action, visits in visit_map.items() if visits == maximum]
            selected = min(candidates, key=lambda action: self._tie_key(tree.state, action))
            results.append(SearchResult(
                action=selected,
                legal_actions=legal,
                root_visits=root_visits,
                pi=pi,
                simulations=self.settings.simulations,
                evaluator_calls=tree.evaluator_calls,
                root_q=root_q,
                legal_action_mask=tree.root.legal_context.action_mask,
            ))
        return tuple(results)


class Torus9RootNoiseEvaluator:
    def __init__(self, evaluator: Torus9NeuralEvaluator, root_state: GoldenState, *, seed: int, alpha: float = 0.30) -> None:
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("Torus 9×9 Dirichlet alpha must be positive and finite")
        self.evaluator = evaluator
        self.root_state_key = root_state.state_key
        self.alpha = float(alpha)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def evaluate_prepared(self, state: GoldenState, legal_context: LegalActionContext) -> Evaluation:
        base = self.evaluator.evaluate_prepared(state, legal_context)
        return self.transform(base, state, legal_context)

    def transform(self, base: Evaluation, state: GoldenState, legal_context: LegalActionContext) -> Evaluation:
        if state.state_key != self.root_state_key:
            return base
        legal_indices = [TORUS9_PASS_INDEX if action == PASS else int(action) for action in legal_context.actions]
        prior = torch.tensor([base.policy[index] for index in legal_indices], dtype=torch.float64)
        total = float(prior.sum())
        prior = prior / total if total > 0.0 else torch.full_like(prior, 1.0 / len(prior))
        noise = torch._standard_gamma(torch.full((len(prior),), self.alpha, dtype=torch.float64), generator=self.generator)
        noise /= noise.sum()
        mixed = 0.75 * prior + 0.25 * noise
        policy = list(base.policy)
        for index, value in zip(legal_indices, mixed.tolist()):
            policy[index] = float(value)
        return Evaluation(policy=tuple(policy), wdl=base.wdl)

    def evaluate(self, state: GoldenState) -> Evaluation:
        return self.evaluate_prepared(state, prepare_legal_actions(state))


@dataclass(frozen=True)
class Torus9SelfPlaySearchContract:
    contract_id: str = TORUS9_SELFPLAY_CONTRACT_ID
    simulations: int = 64
    cpuct: float = 1.25
    fpu: float = 0.0
    temperature_until_ply: int = 8
    temperature_after: float = 0.0
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.30
    watchdog: int = TORUS9_MOVE_LIMIT

    @property
    def settings(self) -> SearchSettings:
        return SearchSettings(simulations=self.simulations, cpuct=self.cpuct, fpu=self.fpu, deterministic_tie_break=True)

    def validate(self) -> None:
        expected = Torus9SelfPlaySearchContract()
        candidate = asdict(self)
        baseline = asdict(expected)
        if self.contract_id not in {TORUS9_SELFPLAY_CONTRACT_ID, TORUS9_CURRENT_SELFPLAY_CONTRACT_ID}:
            raise ValueError("Unknown Torus 9×9 self-play contract id")
        candidate.pop("dirichlet_alpha")
        baseline.pop("dirichlet_alpha")
        candidate.pop("contract_id")
        baseline.pop("contract_id")
        if candidate != baseline or not math.isfinite(float(self.dirichlet_alpha)) or self.dirichlet_alpha <= 0.0:
            raise ValueError("Torus 9×9 self-play search contract drift")

    @property
    def fingerprint(self) -> str:
        if self.contract_id == TORUS9_CURRENT_SELFPLAY_CONTRACT_ID:
            return current_torus9_selfplay_contract_fingerprint(self.dirichlet_alpha)
        return torus9_selfplay_contract_fingerprint(self.dirichlet_alpha)


def _action_index(action: int | str) -> int:
    return TORUS9_PASS_INDEX if action == PASS else int(action)


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
    if temperature <= 0.0:
        maximum = max(result.root_visits[_action_index(action)] for action in result.legal_actions)
        return min((action for action in result.legal_actions if result.root_visits[_action_index(action)] == maximum), key=_action_index)
    weights = [float(result.root_visits[_action_index(action)]) ** (1.0 / temperature) for action in result.legal_actions]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return rng.choice(result.legal_actions)
    threshold = rng.random() * total
    for action, weight in zip(result.legal_actions, weights):
        threshold -= weight
        if threshold <= 0.0:
            return action
    return result.legal_actions[-1]


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
        if self.profile_id not in {TORUS9_PROFILE_ID, TORUS9_CURRENT_PROFILE_ID} or self.selfplay_contract_id not in {TORUS9_SELFPLAY_CONTRACT_ID, TORUS9_CURRENT_SELFPLAY_CONTRACT_ID}:
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


class Torus9SelfPlayRunner:
    def __init__(
        self,
        model: Torus9GraphNet,
        *,
        run_id: str,
        model_checkpoint_label: str,
        checkpoint_artifact_hash: str,
        master_seed: int,
        profile_fp: str,
        seed_namespace: str | None = None,
        code_identity: CodeIdentity | None = None,
        device: str | torch.device = "cpu",
        contract: Torus9SelfPlaySearchContract = Torus9SelfPlaySearchContract(),
        evaluator_override: Any | None = None,
        profile_id: str = TORUS9_PROFILE_ID,
        activity_tracker: Torus9ExecutionActivity | None = None,
    ) -> None:
        contract.validate()
        if model.topology_fingerprint != TORUS9_TOPOLOGY_FINGERPRINT:
            raise ValueError("Torus 9×9 runner received the wrong model")
        self.model = model
        self.run_id = run_id
        self.model_checkpoint_label = model_checkpoint_label
        self.model_hash = model_hash(model)
        self.checkpoint_artifact_hash = checkpoint_artifact_hash
        self.master_seed = int(master_seed)
        self.profile_fp = profile_fp
        self.seed_namespace = seed_namespace or run_id
        self.code_identity = code_identity or capture_code_identity()
        self.device = torch.device(device)
        self.contract = contract
        self.profile_id = str(profile_id)
        self.evaluator = evaluator_override or Torus9NeuralEvaluator(model, device=self.device)
        self.activity_tracker = activity_tracker
        self.adapter = GoldenSearchAdapter()

    def play_game(self, game_id: str) -> Torus9SelfPlayGameRecord:
        game_seed = derive_seed(self.master_seed, self.seed_namespace, game_id, "game")
        rng = random.Random(game_seed)
        state = initial_state(topology=TORUS_9X9, komi=TORUS9_KOMI)
        start = torus9_state_identity(state)
        positions: list[Torus9SelfPlayPosition] = []
        trace: list[int | str] = []
        formal: str | None = None
        technical: str | None = None
        error: str | None = None
        for ply in range(1, self.contract.watchdog + 1):
            search_seed = derive_seed(game_seed, ply, "search")
            try:
                context = prepare_legal_actions(state)
                evaluator = Torus9RootNoiseEvaluator(self.evaluator, state, seed=derive_seed(search_seed, "dirichlet"), alpha=self.contract.dirichlet_alpha)
                if self.activity_tracker is not None:
                    self.activity_tracker.enter_mcts()
                try:
                    result = SequentialPUCT(self.contract.settings, adapter=self.adapter).search(state, evaluator, seed=search_seed)
                finally:
                    if self.activity_tracker is not None:
                        self.activity_tracker.exit_mcts()
                action = _sample_action(result, temperature=1.0 if ply <= self.contract.temperature_until_ply else self.contract.temperature_after, rng=rng)
            except Exception as exc:
                technical, error = "ERROR_SEARCH", f"{type(exc).__name__}: {exc}"
                break
            positions.append(Torus9SelfPlayPosition(
                ply=ply,
                state=torus9_state_identity(state),
                side_to_move=state.side_to_move.name,
                root_visits=tuple(int(value) for value in result.root_visits),
                pi=tuple(float(value) for value in result.pi),
                selected_action=action,
                search_seed=search_seed,
                model_hash=self.model_hash,
            ))
            trace.append(action)
            try:
                state = apply_action(state, action).after
            except IllegalMoveError as exc:
                technical, error = "ERROR_ILLEGAL_PLAYER_ACTION", f"{type(exc).__name__}: {exc}"
                break
            if state.is_terminal:
                formal = result_from_terminal(state).winner.value
                break
        else:
            technical, error = "TRUNCATED_MOVE_LIMIT", "Torus 9×9 self-play watchdog reached 500 actions"
        record = Torus9SelfPlayGameRecord(
            run_id=self.run_id,
            game_id=str(game_id),
            profile_id=self.profile_id,
            profile_fingerprint=self.profile_fp,
            selfplay_contract_id=self.contract.contract_id,
            selfplay_contract_fingerprint=self.contract.fingerprint,
            model_checkpoint_label=self.model_checkpoint_label,
            model_hash=self.model_hash,
            checkpoint_artifact_hash=self.checkpoint_artifact_hash,
            git_commit=self.code_identity.git_commit_sha,
            git_tree=self.code_identity.git_tree_sha,
            git_worktree_clean=self.code_identity.working_tree_clean,
            master_seed=self.master_seed,
            game_seed=game_seed,
            start_state=start,
            positions=tuple(positions),
            final_action_trace=tuple(trace),
            formal_result=formal,
            technical_termination=technical,
            error=error,
            nn_evaluations=self.evaluator.nn_evaluations,
        )
        record.validate()
        return record


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
            "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT if game.profile_id == TORUS9_CURRENT_PROFILE_ID else TORUS9_TARGET_FINGERPRINT,
        }
        samples.append(row)
    return tuple(samples)


def torus9_ownership_target(final_state: GoldenState, side_to_move: str | Stone) -> tuple[int, ...]:
    """Return exact graph-area ownership in the sample side's perspective."""
    return golden_ownership_target(final_state, side_to_move)


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
        expected_target_fingerprint or TORUS9_TARGET_FINGERPRINT
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


def _selfplay_process_init(checkpoint_path: str, expected_hash: str, run_id: str, label: str, artifact: str, master_seed: int, profile_fp: str, code_commit: str, code_tree: str, code_clean: bool, device_name: str, contract: Torus9SelfPlaySearchContract) -> None:
    global _SELFPLAY_PROCESS_MODEL, _SELFPLAY_PROCESS_CONFIG
    torch.set_num_threads(1)
    device = torch.device(device_name)
    metadata = json.loads(Path(checkpoint_path).with_suffix(".metadata.json").read_text(encoding="utf-8"))
    model = torus9_model_from_metadata(metadata).to(device)
    torus9_load_checkpoint(Path(checkpoint_path), model=model, expected={"model_hash": expected_hash}, device=device)
    if model_hash(model) != expected_hash:
        raise RuntimeError("Torus 9×9 self-play worker loaded the wrong checkpoint")
    _SELFPLAY_PROCESS_MODEL = model
    _SELFPLAY_PROCESS_CONFIG = (run_id, label, artifact, master_seed, profile_fp, CodeIdentity(code_commit, code_tree, code_clean), device, contract)


_SELFPLAY_PROCESS_MODEL: Torus9GraphNet | None = None
_SELFPLAY_PROCESS_CONFIG: tuple[object, ...] | None = None


def _selfplay_process_game(game_id: str) -> Torus9SelfPlayGameRecord:
    if _SELFPLAY_PROCESS_MODEL is None or _SELFPLAY_PROCESS_CONFIG is None:
        raise RuntimeError("Torus 9×9 self-play process was not initialized")
    run_id, label, artifact, master_seed, profile_fp, code, device, contract = _SELFPLAY_PROCESS_CONFIG
    runner = Torus9SelfPlayRunner(
        _SELFPLAY_PROCESS_MODEL,
        run_id=str(run_id),
        model_checkpoint_label=str(label),
        checkpoint_artifact_hash=str(artifact),
        master_seed=int(master_seed),
        profile_fp=str(profile_fp),
        code_identity=code,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        contract=contract,  # type: ignore[arg-type]
    )
    return runner.play_game(str(game_id))


def run_torus9_selfplay_games(
    model: Torus9GraphNet,
    *,
    checkpoint_path: Path,
    run_id: str,
    label: str,
    artifact: str,
    master_seed: int,
    profile_fp: str,
    game_ids: Sequence[str],
    workers: int = TORUS9_WORKERS,
    code_identity: CodeIdentity | None = None,
    device: str | torch.device = "cpu",
    contract: Torus9SelfPlaySearchContract = Torus9SelfPlaySearchContract(),
    profile_id: str = TORUS9_PROFILE_ID,
    coalescing: bool = False,
    inference_batch_cap: int | None = None,
    inference_batch_wait_ms: float = 0.0,
    inference_telemetry: MutableMapping[str, object] | None = None,
    execution_activity: MutableMapping[str, object] | None = None,
) -> tuple[Torus9SelfPlayGameRecord, ...]:
    if workers <= 0 or len(set(game_ids)) != len(game_ids):
        raise ValueError("Torus 9×9 self-play workers/game IDs are invalid")
    code = code_identity or capture_code_identity()
    contract.validate()
    ids = tuple(sorted(str(game_id) for game_id in game_ids))
    if profile_id == TORUS9_CURRENT_PROFILE_ID:
        if not isinstance(model, Torus9CurrentGraphNet):
            raise ValueError("Current Torus 9×9 self-play requires the 80×8 auxiliary-head model")
        cap = 1 if inference_batch_cap is None else int(inference_batch_cap)
        if coalescing and cap <= 1:
            raise ValueError("Coalesced current self-play requires batch_cap > 1")
        backend: Torus9InferenceCoordinator | Torus9UncoalescedInference
        if coalescing:
            backend = Torus9InferenceCoordinator(
                model,
                device=device,
                batch_cap=cap,
                wait_ms=float(inference_batch_wait_ms),
            )
        else:
            backend = Torus9UncoalescedInference(model, device=device)
        activity = Torus9ExecutionActivity()

        def play_current(game_id: str) -> Torus9SelfPlayGameRecord:
            lane = _Torus9LaneEvaluator(backend, activity)
            runner = Torus9SelfPlayRunner(
                model,
                run_id=run_id,
                model_checkpoint_label=label,
                checkpoint_artifact_hash=artifact,
                master_seed=master_seed,
                profile_fp=profile_fp,
                code_identity=code,
                device=device,
                contract=contract,
                evaluator_override=lane,
                profile_id=profile_id,
                activity_tracker=activity,
            )
            return runner.play_game(game_id)

        try:
            with ThreadPoolExecutor(max_workers=min(int(workers), len(ids) or 1)) as pool:
                records = tuple(pool.map(play_current, ids))
        finally:
            if isinstance(backend, Torus9InferenceCoordinator):
                backend.close()
        if inference_telemetry is not None:
            inference_telemetry.update(backend.telemetry)
        if execution_activity is not None:
            execution_activity.update(activity.telemetry)
        return tuple(sorted(records, key=lambda record: record.game_id))
    if workers == 1:
        runner = Torus9SelfPlayRunner(model, run_id=run_id, model_checkpoint_label=label, checkpoint_artifact_hash=artifact, master_seed=master_seed, profile_fp=profile_fp, code_identity=code, device=device, contract=contract)
        return tuple(runner.play_game(game_id) for game_id in ids)
    context_name = "spawn" if torch.device(device).type == "cuda" else "fork"
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=_process_context(context_name),
        initializer=_selfplay_process_init,
        initargs=(str(checkpoint_path), model_hash(model), run_id, label, artifact, master_seed, profile_fp, code.git_commit_sha, code.git_tree_sha, code.working_tree_clean, str(device), contract),
    ) as pool:
        records = tuple(pool.map(_selfplay_process_game, ids))
    return tuple(sorted(records, key=lambda record: record.game_id))


class Torus9RollingReplay:
    """Small deterministic recent-generation replay window.

    Rows are appended in generation/game/ply order.  A generation is removed
    only after it falls outside the three-generation window; if a single
    window exceeds the cap, the oldest rows are removed.  This deliberately
    simple policy makes every retained row and every eviction auditable.
    """

    def __init__(
        self,
        *,
        generations: int = TORUS9_ROLLING_GENERATIONS,
        maximum_positions: int = TORUS9_MAX_REPLAY_POSITIONS,
    ) -> None:
        if generations <= 0 or maximum_positions <= 0:
            raise ValueError("Torus 9×9 replay window settings must be positive")
        self.generations = int(generations)
        self.maximum_positions = int(maximum_positions)
        self._rows: list[dict[str, object]] = []
        self._last_generation = 0
        self.total_evictions = 0

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        return tuple(self._rows)

    @staticmethod
    def _row_id(row: Mapping[str, object], generation: int, position: int) -> str:
        return str(row.get("replay_row_id", f"M{generation}:{row.get('game_id', position)}:{row.get('ply', position)}"))

    def append_generation(self, generation: int, samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
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
        if len(self._rows) > self.maximum_positions:
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
        layer = name.split(".", 1)[0]
        by_layer.setdefault(layer, {})[name] = delta
    maximum = max((_parameter_l2(layer) for layer in by_layer.values()), default=0.0)
    return total, maximum


class Torus9Trainer:
    def __init__(
        self,
        model: Torus9GraphNet,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_steps_per_iteration: int = TORUS9_OPTIMIZER_STEPS_PER_ITERATION,
    ) -> None:
        if learning_rate != 0.001 or weight_decay != 0.0:
            raise ValueError("Canonical Torus 9×9 optimizer is Adam(lr=0.001, weight_decay=0)")
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


class Torus9OwnershipTrainer(Torus9Trainer):
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
        Torus9Trainer.__init__(
            self,
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_steps_per_iteration=optimizer_steps_per_iteration,
        )
        self.ownership_loss_enabled = True
        self.score_loss_enabled = bool(score_loss_enabled)

    def train_fixed_budget(self, samples: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
        self.assert_optimizer_continuity()
        count = self.optimizer_steps_per_iteration * TORUS9_BATCH_SIZE
        indices = self._sample_indices(len(samples), seed=seed, count=count)
        for sample in samples:
            validate_torus9_replay_sample(sample)
            if sample.get("ownership_target") is None or sample.get("score_target") is None:
                raise ValueError("Torus 9×9 score training requires ownership and score targets")
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
            observations = torch.tensor([samples[index]["observation"] for index in batch_indices], dtype=torch.float32, device=device)
            policies = torch.tensor([samples[index]["pi"] for index in batch_indices], dtype=torch.float32, device=device)
            values = torch.tensor([samples[index]["z"] for index in batch_indices], dtype=torch.float32, device=device)
            ownership = torch.tensor([samples[index]["ownership_target"] for index in batch_indices], dtype=torch.long, device=device)
            scores = torch.tensor([float(samples[index]["score_target"]) / TORUS9_SCORE_TARGET_NORMALIZATION for index in batch_indices], dtype=torch.float32, device=device)
            before_parameters = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
            step_before = self.assert_optimizer_continuity()
            policy_logits, value_logits, ownership_logits, score_logits = self.model.forward_auxiliary(observations)
            policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = -(values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
            ownership_loss = F.cross_entropy(ownership_logits.reshape(-1, 3), ownership.reshape(-1))
            score_loss = F.mse_loss(score_logits, scores)
            score_weight = 1.0 if self.score_loss_enabled else 0.0
            total_loss = policy_loss + value_loss + ownership_loss + score_weight * score_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Torus 9×9 score training produced non-finite loss")
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradients = [parameter.grad.detach() for parameter in self.model.parameters() if parameter.grad is not None]
            grad_norm = torch.sqrt(sum(torch.sum(gradient.float() ** 2) for gradient in gradients)) if gradients else torch.tensor(0.0)
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise FloatingPointError("Torus 9×9 score training produced non-finite gradient")
            self.optimizer.step()
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


def torus9_checkpoint_metadata(*, model: Torus9GraphNet, run_id: str, label: str, parent: str | None, model_seed: int, code: CodeIdentity, profile_fp: str, completed_games: int, replay_positions: int, optimizer_updates: int, samples_consumed: int, ownership_loss_enabled: bool | None = None, score_loss_enabled: bool | None = None, profile_id: str = TORUS9_PROFILE_ID, target_fingerprint: str = TORUS9_TARGET_FINGERPRINT, selfplay_contract_id: str = TORUS9_SELFPLAY_CONTRACT_ID, selfplay_contract_fingerprint: str | None = None, base_commit: str | None = None) -> dict[str, object]:
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
        "rules_fingerprint": TORUS9_RULES_FINGERPRINT,
        "topology_id": TORUS9_TOPOLOGY_ID,
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "board_size": [9, 9],
        "point_id_order_identity": "row-major-yx:point_id=y*width+x",
        "komi": TORUS9_KOMI,
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
    hidden = int(architecture.get("hidden", TORUS9_HIDDEN))
    blocks = int(architecture.get("blocks", TORUS9_BLOCKS))
    if architecture.get("architecture_id") == TORUS9_CURRENT_ARCHITECTURE_ID:
        return Torus9CurrentGraphNet(hidden=hidden, blocks=blocks)
    heads = dict(metadata.get("network_heads_and_shapes", {}))
    if "score" in heads:
        return Torus9OwnershipScoreGraphNet(hidden=hidden, blocks=blocks)
    if bool(metadata.get("auxiliary_heads")) or "ownership" in heads:
        return Torus9OwnershipGraphNet(hidden=hidden, blocks=blocks)
    return Torus9GraphNet(hidden=hidden, blocks=blocks, architecture_id=str(architecture.get("architecture_id", TORUS9_ARCHITECTURE_ID)))


def torus9_load_checkpoint(path: Path, *, model: Torus9GraphNet, optimizer: torch.optim.Optimizer | None = None, expected: Mapping[str, object] | None = None, device: str | torch.device = "cpu") -> dict[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    metadata = dict(payload["metadata"])
    if metadata.get("topology_fingerprint") != TORUS9_TOPOLOGY_FINGERPRINT or metadata.get("board_size") != [9, 9] or metadata.get("komi") != TORUS9_KOMI:
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


def _arena_process_init(candidate_path: str, reference_path: str, candidate_hash: str, reference_hash: str, device_name: str) -> None:
    global _ARENA_CANDIDATE_MODEL, _ARENA_REFERENCE_MODEL, _ARENA_CANDIDATE_EVALUATOR, _ARENA_REFERENCE_EVALUATOR
    torch.set_num_threads(1)
    device = torch.device(device_name)
    candidate_info = json.loads(Path(candidate_path).with_suffix(".metadata.json").read_text(encoding="utf-8"))
    reference_info = json.loads(Path(reference_path).with_suffix(".metadata.json").read_text(encoding="utf-8"))
    candidate = torus9_model_from_metadata(candidate_info).to(device)
    reference = torus9_model_from_metadata(reference_info).to(device)
    torus9_load_checkpoint(Path(candidate_path), model=candidate, expected={"model_hash": candidate_hash}, device=device)
    torus9_load_checkpoint(Path(reference_path), model=reference, expected={"model_hash": reference_hash}, device=device)
    _ARENA_CANDIDATE_MODEL = candidate
    _ARENA_REFERENCE_MODEL = reference
    _ARENA_CANDIDATE_EVALUATOR = Torus9NeuralEvaluator(candidate, device=device)
    _ARENA_REFERENCE_EVALUATOR = Torus9NeuralEvaluator(reference, device=device)


def torus9_arena_termination_reason(state: GoldenState, action_count: int) -> str | None:
    """Return the Arena termination taxonomy without turning the watchdog into WDL."""

    if state.is_terminal:
        return "DOUBLE_PASS"
    if action_count >= TORUS9_ARENA_MOVE_LIMIT:
        return "TRUNCATED_MOVE_LIMIT"
    return None


_ARENA_CANDIDATE_MODEL: Torus9GraphNet | None = None
_ARENA_REFERENCE_MODEL: Torus9GraphNet | None = None
_ARENA_CANDIDATE_EVALUATOR: Torus9NeuralEvaluator | None = None
_ARENA_REFERENCE_EVALUATOR: Torus9NeuralEvaluator | None = None


def _arena_process_game(task: Mapping[str, object]) -> dict[str, object]:
    if _ARENA_CANDIDATE_MODEL is None or _ARENA_REFERENCE_MODEL is None or _ARENA_CANDIDATE_EVALUATOR is None or _ARENA_REFERENCE_EVALUATOR is None:
        raise RuntimeError("Torus 9×9 Arena process was not initialized")
    started = time.perf_counter()
    start_state = torus9_state_from_identity(task["state"])  # type: ignore[arg-type]
    state = start_state
    candidate_black = bool(task["candidate_black"])
    trace: list[dict[str, object]] = []
    technical: str | None = None
    error: str | None = None
    formal: str | None = None
    for ply in range(1, TORUS9_ARENA_MOVE_LIMIT + 1):
        candidate_turn = (state.side_to_move == BLACK and candidate_black) or (state.side_to_move == WHITE and not candidate_black)
        model = _ARENA_CANDIDATE_MODEL if candidate_turn else _ARENA_REFERENCE_MODEL
        evaluator = _ARENA_CANDIDATE_EVALUATOR if candidate_turn else _ARENA_REFERENCE_EVALUATOR
        try:
            result = SequentialPUCT(SearchSettings(simulations=64, cpuct=1.25, fpu=0.0, deterministic_tie_break=True), adapter=GoldenSearchAdapter()).search(state, evaluator, seed=derive_seed(int(task["game_seed"]), ply, "arena-search"))
            action = result.action
        except Exception as exc:
            technical, error = "ERROR_SEARCH", f"{type(exc).__name__}: {exc}"
            break
        try:
            state = apply_action(state, action).after
        except IllegalMoveError as exc:
            technical, error = "ERROR_ILLEGAL_PLAYER_ACTION", f"{type(exc).__name__}: {exc}"
            trace.append({"ply": ply, "side_to_move": state.side_to_move.name, "player": "candidate" if candidate_turn else "reference", "action": action, "legal": False, "error": error})
            break
        trace.append({"ply": ply, "side_to_move": (BLACK if state.side_to_move == WHITE else WHITE).name, "player": "candidate" if candidate_turn else "reference", "action": action, "legal": True})
        if torus9_arena_termination_reason(state, ply) == "DOUBLE_PASS":
            formal = result_from_terminal(state).winner.value
            break
    else:
        technical, error = "TRUNCATED_MOVE_LIMIT", f"Torus 9×9 Arena watchdog reached {TORUS9_ARENA_MOVE_LIMIT} actions"
    result_row: dict[str, object] = {
        "run_id": str(task["run_id"]),
        "comparison": str(task["comparison"]),
        "pair_id": str(task["pair_id"]),
        "game_id": str(task["game_id"]),
        "start_id": str(task["start_id"]),
        "candidate_black": candidate_black,
        "candidate_model_hash": str(task["candidate_hash"]),
        "reference_model_hash": str(task["reference_hash"]),
        "komi": TORUS9_KOMI,
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "start_state": task["state"],
        "start_trace": task["trace"],
        "action_trace": trace,
        "final_board": [int(stone) for stone in state.stones],
        "formal_result": formal,
        "technical_termination": technical,
        "error": error,
        "black_area": None,
        "white_area": None,
        "margin_black": None,
        "mapped_result": None,
        "wall_time_sec": time.perf_counter() - started,
    }
    if formal is not None:
        score = score_terminal(state)
        result_row.update({"black_area": score.black_area, "white_area": score.white_area, "margin_black": score.margin_black})
        if formal == "DRAW":
            result_row["mapped_result"] = "DRAW"
        elif (formal == "BLACK") == candidate_black:
            result_row["mapped_result"] = "A_WIN"
        else:
            result_row["mapped_result"] = "B_WIN"
    return result_row


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
        "wins": counts["W"],
        "losses": counts["L"],
        "draws": counts["D"],
        "mean_pair_score": sum(pair_scores) / len(pair_scores) if pair_scores else None,
        "95_percent_hoeffding_interval": interval,
        "technical_by_reason": {str(reason): sum(row.get("technical_termination") == reason for row in technical) for reason in sorted({str(row.get("technical_termination")) for row in technical})},
        "arena_contract_fingerprint": TORUS9_ARENA_CONTRACT_FINGERPRINT,
        "technical_fail_closed": True,
    }


def run_torus9_arena(
    *,
    run_id: str,
    comparison: str,
    candidate_path: Path,
    reference_path: Path,
    candidate_label: str,
    reference_label: str,
    starts: Sequence[Mapping[str, object]],
    master_seed: int,
    output_dir: Path,
    workers: int = TORUS9_WORKERS,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    candidate_meta = torus9_checkpoint_info(candidate_path)
    reference_meta = torus9_checkpoint_info(reference_path)
    tasks: list[dict[str, object]] = []
    for row in starts:
        pair_id = f"{comparison}--{row['start_id']}"
        for suffix, candidate_black in (("g1", True), ("g2", False)):
            game_id = f"{pair_id}--{suffix}"
            tasks.append({
                "run_id": run_id,
                "comparison": comparison,
                "pair_id": pair_id,
                "game_id": game_id,
                "start_id": row["start_id"],
                "state": row["state"],
                "trace": row["trace"],
                "candidate_black": candidate_black,
                "candidate_hash": candidate_meta["model_hash"],
                "reference_hash": reference_meta["model_hash"],
                "game_seed": derive_seed(master_seed, pair_id, game_id),
            })
    if workers <= 0:
        raise ValueError("Torus 9×9 Arena workers must be positive")
    context_name = "spawn" if torch.device(device).type == "cuda" else "fork"
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=_process_context(context_name), initializer=_arena_process_init, initargs=(str(candidate_path), str(reference_path), str(candidate_meta["model_hash"]), str(reference_meta["model_hash"]), str(device))) as pool:
        records = tuple(pool.map(_arena_process_game, tasks))
    records = tuple(sorted(records, key=lambda row: str(row["game_id"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "games.jsonl", records)
    summary = summarize_torus9_arena(records, candidate_label=candidate_label, reference_label=reference_label, pairs=len(starts))
    summary["comparison"] = comparison
    summary["frozen_corpus_fingerprint"] = str(starts[0].get("corpus_fingerprint", "")) if starts else None
    write_json(output_dir / "summary.json", summary)
    return summary


def run_torus9_batched_arena(
    *,
    run_id: str,
    comparison: str,
    candidate_path: Path,
    reference_path: Path,
    candidate_label: str,
    reference_label: str,
    starts: Sequence[Mapping[str, object]],
    master_seed: int,
    output_dir: Path,
    workers: int = TORUS9_WORKERS,
    arena_batch_size: int = 8,
    inference_batch_wait_ms: float = 6.0,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Run the fixed Torus9 Arena with coalesced neural inference batches.

    ``workers`` is the number of logical game lanes and ``arena_batch_size``
    is the maximum contribution per lane to one coalesced model call.  The
    deterministic PUCT trees are advanced together in the parent process so
    leaves from all active lanes can reach the same model batch.
    """
    if workers <= 0 or arena_batch_size <= 0 or inference_batch_wait_ms < 0.0:
        raise ValueError("Torus 9×9 batched Arena settings must be positive")
    candidate_meta = torus9_checkpoint_info(candidate_path)
    reference_meta = torus9_checkpoint_info(reference_path)
    candidate_info = json.loads(candidate_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    reference_info = json.loads(reference_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    candidate = torus9_model_from_metadata(candidate_info).to(device)
    reference = torus9_model_from_metadata(reference_info).to(device)
    torus9_load_checkpoint(candidate_path, model=candidate, expected={"model_hash": candidate_meta["model_hash"]}, device=device)
    torus9_load_checkpoint(reference_path, model=reference, expected={"model_hash": reference_meta["model_hash"]}, device=device)
    candidate.eval()
    reference.eval()
    candidate_evaluator = Torus9NeuralEvaluator(candidate, device=device)
    reference_evaluator = Torus9NeuralEvaluator(reference, device=device)
    tasks: list[dict[str, object]] = []
    for row in starts:
        pair_id = f"{comparison}--{row['start_id']}"
        for suffix, candidate_black in (("g1", True), ("g2", False)):
            game_id = f"{pair_id}--{suffix}"
            tasks.append({
                "run_id": run_id,
                "comparison": comparison,
                "pair_id": pair_id,
                "game_id": game_id,
                "start_id": row["start_id"],
                "state": row["state"],
                "trace": row["trace"],
                "candidate_black": candidate_black,
                "candidate_hash": candidate_meta["model_hash"],
                "reference_hash": reference_meta["model_hash"],
                "game_seed": derive_seed(master_seed, pair_id, game_id),
                "worker_id": len(tasks) % int(workers),
            })
    if not tasks:
        raise ValueError("Torus 9×9 batched Arena requires at least one game")
    games: list[dict[str, object]] = []
    for task in tasks:
        games.append({
            "task": task,
            "state": torus9_state_from_identity(task["state"]),
            "trace": [],
            "formal": None,
            "technical": None,
            "error": None,
        })
    old_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, min(int(workers), 16)))
    started = time.perf_counter()
    try:
        search = Torus9BatchedPUCT(
            SearchSettings(simulations=64, cpuct=1.25, fpu=0.0, deterministic_tie_break=True),
            adapter=GoldenSearchAdapter(),
            max_batch_rows=int(workers) * int(arena_batch_size),
            inference_batch_wait_ms=float(inference_batch_wait_ms),
        )
        for ply in range(1, TORUS9_ARENA_MOVE_LIMIT + 1):
            active = [index for index, game in enumerate(games) if game["formal"] is None and game["technical"] is None]
            if not active:
                break
            states = [games[index]["state"] for index in active]
            evaluators = []
            for index in active:
                task = games[index]["task"]
                state = games[index]["state"]
                candidate_turn = (state.side_to_move == BLACK and bool(task["candidate_black"])) or (state.side_to_move == WHITE and not bool(task["candidate_black"]))
                evaluators.append(candidate_evaluator if candidate_turn else reference_evaluator)
            results = search.search(
                states,
                evaluators,
                seeds=[derive_seed(int(games[index]["task"]["game_seed"]), ply, "arena-search") for index in active],
            )
            for index, result in zip(active, results):
                game = games[index]
                task = game["task"]
                state = game["state"]
                candidate_turn = (state.side_to_move == BLACK and bool(task["candidate_black"])) or (state.side_to_move == WHITE and not bool(task["candidate_black"]))
                action = result.action
                try:
                    next_state = apply_action(state, action).after
                except IllegalMoveError as exc:
                    game["technical"] = "ERROR_ILLEGAL_PLAYER_ACTION"
                    game["error"] = f"{type(exc).__name__}: {exc}"
                    game["trace"].append({"ply": ply, "side_to_move": state.side_to_move.name, "player": "candidate" if candidate_turn else "reference", "action": action, "legal": False, "error": game["error"]})
                    continue
                game["state"] = next_state
                game["trace"].append({"ply": ply, "side_to_move": (BLACK if next_state.side_to_move == WHITE else WHITE).name, "player": "candidate" if candidate_turn else "reference", "action": action, "legal": True})
                if torus9_arena_termination_reason(next_state, ply) == "DOUBLE_PASS":
                    game["formal"] = result_from_terminal(next_state).winner.value
        else:
            for game in games:
                if game["formal"] is None and game["technical"] is None:
                    game["technical"] = "TRUNCATED_MOVE_LIMIT"
                    game["error"] = f"Torus 9×9 Arena watchdog reached {TORUS9_ARENA_MOVE_LIMIT} actions"
    finally:
        torch.set_num_threads(old_threads)
    records: list[dict[str, object]] = []
    for game in games:
        task = game["task"]
        state = game["state"]
        formal = game["formal"]
        candidate_black = bool(task["candidate_black"])
        row: dict[str, object] = {
            "run_id": str(task["run_id"]),
            "comparison": str(task["comparison"]),
            "pair_id": str(task["pair_id"]),
            "game_id": str(task["game_id"]),
            "start_id": str(task["start_id"]),
            "worker_id": int(task["worker_id"]),
            "candidate_black": candidate_black,
            "candidate_model_hash": str(task["candidate_hash"]),
            "reference_model_hash": str(task["reference_hash"]),
            "komi": TORUS9_KOMI,
            "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
            "start_state": task["state"],
            "start_trace": task["trace"],
            "action_trace": game["trace"],
            "final_board": [int(stone) for stone in state.stones],
            "formal_result": formal,
            "technical_termination": game["technical"],
            "error": game["error"],
            "black_area": None,
            "white_area": None,
            "margin_black": None,
            "mapped_result": None,
            "wall_time_sec": time.perf_counter() - started,
        }
        if formal is not None:
            score = score_terminal(state)
            row.update({"black_area": score.black_area, "white_area": score.white_area, "margin_black": score.margin_black})
            if formal == "DRAW":
                row["mapped_result"] = "DRAW"
            elif (formal == "BLACK") == candidate_black:
                row["mapped_result"] = "A_WIN"
            else:
                row["mapped_result"] = "B_WIN"
        records.append(row)
    records = sorted(records, key=lambda row: str(row["game_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "games.jsonl", records)
    summary = summarize_torus9_arena(records, candidate_label=candidate_label, reference_label=reference_label, pairs=len(starts))
    batch_rows = list(search.inference_batch_rows)
    ordered_batch_rows = sorted(batch_rows)
    p50_index = min(len(ordered_batch_rows) - 1, max(0, math.ceil(0.50 * len(ordered_batch_rows)) - 1)) if ordered_batch_rows else 0
    p95_index = min(len(ordered_batch_rows) - 1, max(0, math.ceil(0.95 * len(ordered_batch_rows)) - 1)) if ordered_batch_rows else 0
    summary.update({
        "comparison": comparison,
        "frozen_corpus_fingerprint": str(starts[0].get("corpus_fingerprint", "")) if starts else None,
        "batched": True,
        "arena_batch_size": int(arena_batch_size),
        "inference_batch_wait_ms": float(inference_batch_wait_ms),
        "inference_calls": len(batch_rows),
        "inference_rows": sum(batch_rows),
        "mean_inference_batch_rows": sum(batch_rows) / len(batch_rows) if batch_rows else 0.0,
        "p50_inference_batch_rows": ordered_batch_rows[p50_index] if ordered_batch_rows else 0,
        "p95_inference_batch_rows": ordered_batch_rows[p95_index] if ordered_batch_rows else 0,
        "max_inference_batch_rows": max(batch_rows, default=0),
        "workers": int(workers),
        "logical_worker_lanes": int(workers),
    })
    write_json(output_dir / "summary.json", summary)
    return summary


def _candidate_start(master_seed: int, prefix_length: int, candidate_index: int) -> dict[str, object]:
    seed = derive_seed(master_seed, prefix_length, candidate_index)
    rng = random.Random(seed)
    state = initial_state(topology=TORUS_9X9, komi=TORUS9_KOMI)
    trace: list[int] = []
    for _ in range(prefix_length):
        choices = tuple(action for action in legal_actions(state) if action != PASS)
        if not choices:
            raise ValueError("No legal non-pass action for frozen start")
        action = int(rng.choice(choices))
        state = apply_action(state, action).after
        trace.append(action)
    return {"prefix_length": prefix_length, "candidate_index": candidate_index, "candidate_seed": seed, "trace": trace, "state": torus9_state_identity(state)}


def generate_torus9_evaluation_starts(*, master_seed: int, accepted_per_stratum: int = 8) -> tuple[dict[str, object], ...]:
    accepted: list[dict[str, object]] = []
    seen: set[str] = set()
    for prefix_length in (2, 4, 6, 8, 10, 12, 14, 16):
        count = 0
        candidate_index = 0
        while count < accepted_per_stratum:
            row = _candidate_start(master_seed, prefix_length, candidate_index)
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
