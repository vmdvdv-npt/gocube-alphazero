"""Standalone Torus 9×9 Golden contract, training line, and Arena helpers.

This module is intentionally dimension-explicit.  It shares the already
verified generic Golden rules and PUCT implementation, while keeping the 9×9
observation, replay, checkpoint, and provenance boundaries independent from
the frozen 5×5 line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .arena_contract import SearchSettings
from .neural import model_hash
from .provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256
from .result import Winner, result_from_terminal
from .rules import IllegalMoveError, LegalActionContext, apply_action, legal_actions, prepare_legal_actions
from .scoring import score_terminal
from .search import Evaluation, SearchError, SequentialPUCT
from .search_adapter import GoldenSearchAdapter
from .state import BLACK, EMPTY, PASS, WHITE, GoldenState, Stone, initial_state
from .topology import TORUS_9X9, TORUS_9X9_TOPOLOGY_ID
from .torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_ARENA_CONTRACT_FINGERPRINT,
    TORUS9_ARENA_CONTRACT_ID,
    TORUS9_BATCH_SIZE,
    TORUS9_BLOCKS,
    TORUS9_HIDDEN,
    TORUS9_KOMI,
    TORUS9_MOVE_LIMIT,
    TORUS9_OBSERVATION_FINGERPRINT,
    TORUS9_OBSERVATION_SCHEMA_ID,
    TORUS9_OBSERVATION_SCHEMA_VERSION,
    TORUS9_PASS_INDEX,
    TORUS9_POINT_COUNT,
    TORUS9_PROFILE_ID,
    TORUS9_RULES_FINGERPRINT,
    TORUS9_SELFPLAY_CONTRACT_FINGERPRINT,
    TORUS9_SELFPLAY_CONTRACT_ID,
    TORUS9_TARGET_CONTRACT_ID,
    TORUS9_TARGET_FINGERPRINT,
    TORUS9_WORKERS,
    load_torus9_profile,
    profile_fingerprint,
)


TORUS9_VALUE_HEAD_SEMANTICS = "side-to-move:[WIN,DRAW,LOSS]"
TORUS9_ARCHITECTURE_ID = "GoldenGraphNetV1-Torus9"
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
    values = torch.zeros((6, TORUS9_POINT_COUNT), dtype=torch.float32)
    for point, stone in enumerate(state.stones):
        values[0, point] = float(stone == own)
        values[1, point] = float(stone == other)
    values[2].fill_(1.0 if own == BLACK else -1.0)
    values[3].fill_(1.0 if state.consecutive_passes == 1 else 0.0)
    values[4] = torch.tensor(context.action_mask[:TORUS9_POINT_COUNT], dtype=torch.float32)
    values[5].fill_(TORUS9_KOMI)
    return Torus9Observation(values, context.action_mask, state.state_key)


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
    """GoldenGraphNetV1 capacity, with the 9×9 policy boundary [81 points + PASS]."""

    architecture_id = TORUS9_ARCHITECTURE_ID

    def __init__(self, *, hidden: int = TORUS9_HIDDEN, blocks: int = TORUS9_BLOCKS) -> None:
        super().__init__()
        if hidden <= 0 or blocks <= 0:
            raise ValueError("Torus 9×9 hidden and blocks must be positive")
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

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.encode(observation)
        policy = torch.cat((self.point_policy(nodes).squeeze(-1), self.pass_policy(nodes.mean(dim=1))), dim=1)
        return policy, self.value_head(nodes.mean(dim=1))


class Torus9NeuralEvaluator:
    def __init__(self, model: Torus9GraphNet, *, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.nn_evaluations = 0

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
        policy_values = tuple(float(value) for value in policy.detach().cpu())
        wdl_values = tuple(float(value) for value in wdl.detach().cpu())
        if len(policy_values) != TORUS9_ACTION_COUNT or len(wdl_values) != 3:
            raise SearchError("Torus 9×9 neural head shape drift")
        if any(not math.isfinite(value) or value < 0.0 for value in policy_values + wdl_values):
            raise SearchError("Torus 9×9 neural output is non-finite")
        return Evaluation(policy=policy_values, wdl=wdl_values)


class Torus9RootNoiseEvaluator:
    def __init__(self, evaluator: Torus9NeuralEvaluator, root_state: GoldenState, *, seed: int) -> None:
        self.evaluator = evaluator
        self.root_state_key = root_state.state_key
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def evaluate_prepared(self, state: GoldenState, legal_context: LegalActionContext) -> Evaluation:
        base = self.evaluator.evaluate_prepared(state, legal_context)
        if state.state_key != self.root_state_key:
            return base
        legal_indices = [TORUS9_PASS_INDEX if action == PASS else int(action) for action in legal_context.actions]
        prior = torch.tensor([base.policy[index] for index in legal_indices], dtype=torch.float64)
        total = float(prior.sum())
        prior = prior / total if total > 0.0 else torch.full_like(prior, 1.0 / len(prior))
        noise = torch._standard_gamma(torch.full((len(prior),), 0.30, dtype=torch.float64), generator=self.generator)
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
        if asdict(self) != asdict(Torus9SelfPlaySearchContract()):
            raise ValueError("Torus 9×9 self-play search contract drift")


def _action_index(action: int | str) -> int:
    return TORUS9_PASS_INDEX if action == PASS else int(action)


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
        if self.profile_id != TORUS9_PROFILE_ID or self.selfplay_contract_id != TORUS9_SELFPLAY_CONTRACT_ID:
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
        self.evaluator = Torus9NeuralEvaluator(model, device=self.device)
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
                evaluator = Torus9RootNoiseEvaluator(self.evaluator, state, seed=derive_seed(search_seed, "dirichlet"))
                result = SequentialPUCT(self.contract.settings, adapter=self.adapter).search(state, evaluator, seed=search_seed)
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
            profile_id=TORUS9_PROFILE_ID,
            profile_fingerprint=self.profile_fp,
            selfplay_contract_id=TORUS9_SELFPLAY_CONTRACT_ID,
            selfplay_contract_fingerprint=TORUS9_SELFPLAY_CONTRACT_FINGERPRINT,
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
            "target_fingerprint": TORUS9_TARGET_FINGERPRINT,
        }
        samples.append(row)
    return tuple(samples)


def validate_torus9_replay_sample(sample: Mapping[str, object]) -> None:
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
    if sample.get("observation_fingerprint") != TORUS9_OBSERVATION_FINGERPRINT or sample.get("target_contract_id") != TORUS9_TARGET_CONTRACT_ID or sample.get("target_fingerprint") != TORUS9_TARGET_FINGERPRINT:
        raise ValueError("Torus 9×9 replay semantic fingerprint drift")


def _selfplay_process_init(checkpoint_path: str, expected_hash: str, run_id: str, label: str, artifact: str, master_seed: int, profile_fp: str, code_commit: str, code_tree: str, code_clean: bool, device_name: str) -> None:
    global _SELFPLAY_PROCESS_MODEL, _SELFPLAY_PROCESS_CONFIG
    torch.set_num_threads(1)
    device = torch.device(device_name)
    model = Torus9GraphNet().to(device)
    torus9_load_checkpoint(Path(checkpoint_path), model=model, expected={"model_hash": expected_hash}, device=device)
    if model_hash(model) != expected_hash:
        raise RuntimeError("Torus 9×9 self-play worker loaded the wrong checkpoint")
    _SELFPLAY_PROCESS_MODEL = model
    _SELFPLAY_PROCESS_CONFIG = (run_id, label, artifact, master_seed, profile_fp, CodeIdentity(code_commit, code_tree, code_clean), device)


_SELFPLAY_PROCESS_MODEL: Torus9GraphNet | None = None
_SELFPLAY_PROCESS_CONFIG: tuple[object, ...] | None = None


def _selfplay_process_game(game_id: str) -> Torus9SelfPlayGameRecord:
    if _SELFPLAY_PROCESS_MODEL is None or _SELFPLAY_PROCESS_CONFIG is None:
        raise RuntimeError("Torus 9×9 self-play process was not initialized")
    run_id, label, artifact, master_seed, profile_fp, code, device = _SELFPLAY_PROCESS_CONFIG
    runner = Torus9SelfPlayRunner(
        _SELFPLAY_PROCESS_MODEL,
        run_id=str(run_id),
        model_checkpoint_label=str(label),
        checkpoint_artifact_hash=str(artifact),
        master_seed=int(master_seed),
        profile_fp=str(profile_fp),
        code_identity=code,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
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
) -> tuple[Torus9SelfPlayGameRecord, ...]:
    if workers <= 0 or len(set(game_ids)) != len(game_ids):
        raise ValueError("Torus 9×9 self-play workers/game IDs are invalid")
    code = code_identity or capture_code_identity()
    ids = tuple(sorted(str(game_id) for game_id in game_ids))
    if workers == 1:
        runner = Torus9SelfPlayRunner(model, run_id=run_id, model_checkpoint_label=label, checkpoint_artifact_hash=artifact, master_seed=master_seed, profile_fp=profile_fp, code_identity=code, device=device)
        return tuple(runner.play_game(game_id) for game_id in ids)
    context_name = "spawn" if torch.device(device).type == "cuda" else "fork"
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=get_context(context_name),
        initializer=_selfplay_process_init,
        initargs=(str(checkpoint_path), model_hash(model), run_id, label, artifact, master_seed, profile_fp, code.git_commit_sha, code.git_tree_sha, code.working_tree_clean, str(device)),
    ) as pool:
        records = tuple(pool.map(_selfplay_process_game, ids))
    return tuple(sorted(records, key=lambda record: record.game_id))


class Torus9Trainer:
    def __init__(self, model: Torus9GraphNet, *, learning_rate: float = 1e-3, weight_decay: float = 0.0) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.update_count = 0
        self.samples_consumed = 0

    def train_fresh_epoch(self, samples: Sequence[Mapping[str, object]], *, seed: int) -> dict[str, object]:
        if not samples:
            raise ValueError("Torus 9×9 trainer requires fresh replay samples")
        device = next(self.model.parameters()).device
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        order = [int(index) for index in torch.randperm(len(samples), generator=generator).tolist()]
        updates: list[dict[str, object]] = []
        self.model.train()
        for offset in range(0, len(order), TORUS9_BATCH_SIZE):
            indices = order[offset:offset + TORUS9_BATCH_SIZE]
            observations = torch.tensor([samples[index]["observation"] for index in indices], dtype=torch.float32, device=device)
            policies = torch.tensor([samples[index]["pi"] for index in indices], dtype=torch.float32, device=device)
            values = torch.tensor([samples[index]["z"] for index in indices], dtype=torch.float32, device=device)
            policy_logits, value_logits = self.model(observations)
            policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = -(values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
            total_loss = policy_loss + value_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Torus 9×9 training produced non-finite loss")
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float("inf"))
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise FloatingPointError("Torus 9×9 training produced non-finite gradient")
            self.optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
                raise FloatingPointError("Torus 9×9 training produced non-finite parameter")
            self.update_count += 1
            self.samples_consumed += len(indices)
            updates.append({
                "update": self.update_count,
                "batch_size": len(indices),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "gradient_norm": float(grad_norm),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            })
        return {
            "updates": len(updates),
            "optimizer_updates_total": self.update_count,
            "samples": len(samples),
            "samples_consumed_total": self.samples_consumed,
            "batch_size": TORUS9_BATCH_SIZE,
            "batch_sizes": [int(row["batch_size"]) for row in updates],
            "mean_policy_loss": sum(float(row["policy_loss"]) for row in updates) / len(updates),
            "mean_value_loss": sum(float(row["value_loss"]) for row in updates) / len(updates),
            "mean_total_loss": sum(float(row["total_loss"]) for row in updates) / len(updates),
            "updates_detail": updates,
        }


def torus9_checkpoint_metadata(*, model: Torus9GraphNet, run_id: str, label: str, parent: str | None, model_seed: int, code: CodeIdentity, profile_fp: str, completed_games: int, replay_positions: int, optimizer_updates: int, samples_consumed: int) -> dict[str, object]:
    return {
        "checkpoint_schema_version": 1,
        "checkpoint_label": label,
        "run_id": run_id,
        "parent_or_source_run_identity": parent or run_id,
        "architecture_id": model.architecture_id,
        "architecture_config": model.architecture_config,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_hash": model_hash(model),
        "profile_id": TORUS9_PROFILE_ID,
        "profile_fingerprint": profile_fp,
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
        "target_fingerprint": TORUS9_TARGET_FINGERPRINT,
        "value_head_semantics": TORUS9_VALUE_HEAD_SEMANTICS,
        "network_heads_and_shapes": {"policy": [TORUS9_ACTION_COUNT], "value": [3]},
        "model_initialization_seed": model_seed,
        "completed_games": completed_games,
        "valid_replay_positions": replay_positions,
        "optimizer_updates": optimizer_updates,
        "train_samples_consumed": samples_consumed,
        "git_commit": code.git_commit_sha,
        "git_tree": code.git_tree_sha,
        "git_worktree_clean": code.working_tree_clean,
        "fresh_data_ratio": 1.0,
        "auxiliary_heads": False,
    }


def torus9_save_checkpoint(path: Path, *, model: Torus9GraphNet, optimizer: torch.optim.Optimizer | None, metadata: Mapping[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(metadata)
    enriched["model_hash"] = model_hash(model)
    payload = {"checkpoint_schema_version": 1, "metadata": enriched, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None}
    torch.save(payload, path)
    path.with_suffix(".metadata.json").write_text(json.dumps(_jsonable(enriched), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return enriched


def torus9_load_checkpoint(path: Path, *, model: Torus9GraphNet, optimizer: torch.optim.Optimizer | None = None, expected: Mapping[str, object] | None = None, device: str | torch.device = "cpu") -> dict[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    metadata = dict(payload["metadata"])
    if metadata.get("topology_fingerprint") != TORUS9_TOPOLOGY_FINGERPRINT or metadata.get("board_size") != [9, 9] or metadata.get("komi") != TORUS9_KOMI:
        raise ValueError("Torus 9×9 checkpoint topology/komi mismatch")
    if metadata.get("network_heads_and_shapes") != {"policy": [82], "value": [3]} or metadata.get("auxiliary_heads") is not False:
        raise ValueError("Torus 9×9 checkpoint head contract mismatch")
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


def torus9_checkpoint_info(path: Path) -> dict[str, object]:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    model = Torus9GraphNet()
    loaded = torus9_load_checkpoint(path, model=model, expected={"model_hash": metadata["model_hash"]})
    return {"path": str(path), "metadata": loaded, "model_hash": model_hash(model), "artifact_sha256": file_sha256(path)}


def _arena_process_init(candidate_path: str, reference_path: str, candidate_hash: str, reference_hash: str, device_name: str) -> None:
    global _ARENA_CANDIDATE_MODEL, _ARENA_REFERENCE_MODEL, _ARENA_CANDIDATE_EVALUATOR, _ARENA_REFERENCE_EVALUATOR
    torch.set_num_threads(1)
    device = torch.device(device_name)
    candidate = Torus9GraphNet().to(device)
    reference = Torus9GraphNet().to(device)
    torus9_load_checkpoint(Path(candidate_path), model=candidate, expected={"model_hash": candidate_hash}, device=device)
    torus9_load_checkpoint(Path(reference_path), model=reference, expected={"model_hash": reference_hash}, device=device)
    _ARENA_CANDIDATE_MODEL = candidate
    _ARENA_REFERENCE_MODEL = reference
    _ARENA_CANDIDATE_EVALUATOR = Torus9NeuralEvaluator(candidate, device=device)
    _ARENA_REFERENCE_EVALUATOR = Torus9NeuralEvaluator(reference, device=device)


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
    for ply in range(1, TORUS9_MOVE_LIMIT + 1):
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
        if state.is_terminal:
            formal = result_from_terminal(state).winner.value
            break
    else:
        technical, error = "TRUNCATED_MOVE_LIMIT", "Torus 9×9 Arena watchdog reached 500 actions"
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
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=get_context(context_name), initializer=_arena_process_init, initargs=(str(candidate_path), str(reference_path), str(candidate_meta["model_hash"]), str(reference_meta["model_hash"]), str(device))) as pool:
        records = tuple(pool.map(_arena_process_game, tasks))
    records = tuple(sorted(records, key=lambda row: str(row["game_id"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "games.jsonl", records)
    summary = summarize_torus9_arena(records, candidate_label=candidate_label, reference_label=reference_label, pairs=len(starts))
    summary["comparison"] = comparison
    summary["frozen_corpus_fingerprint"] = str(starts[0].get("corpus_fingerprint", "")) if starts else None
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
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "fixtures": {"corner_like": [0, 8, 72, 80], "horizontal_wrap": True, "vertical_wrap": True, "central": 40},
        "no_ownership_or_score_head": True,
    }
