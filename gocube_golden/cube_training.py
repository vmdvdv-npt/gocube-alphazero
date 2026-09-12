"""Cube-specific self-play, replay and one-epoch training primitives.

The rules and PUCT implementation are the shared Golden implementations.  A
separate typed record layer is necessary because Cube has a different
observation/action contract and a 1920-action technical watchdog.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

torch = importlib.import_module("torch")
Tensor = torch.Tensor
nn = torch.nn
F = importlib.import_module("torch.nn.functional")

from .cube_neural import (
    CUBE_ACTION_COUNT,
    CUBE_OBSERVATION_FINGERPRINT,
    GoldenCubeNeuralEvaluator,
    SelfPlayCubeRootNoiseEvaluator,
    build_cube_action_mask,
    build_cube_observation,
    cube_count_parameters,
    cube_model_hash,
    GoldenCubeGraphNetV1,
)
from .cube_topology import CUBE4_TOPOLOGY
from .provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256
from .result import Winner, result_from_terminal
from .rules import IllegalMoveError, apply_action, legal_actions
from .search import SearchError, SearchResult, SequentialPUCT
from .state import BLACK, PASS, WHITE, GoldenState, Stone, initial_state

CUBE_SELFPLAY_SCHEMA_VERSION = 1
CUBE_REPLAY_SCHEMA_VERSION = 1
CUBE_CHECKPOINT_SCHEMA_VERSION = 1
CUBE_WATCHDOG = 20 * CUBE4_TOPOLOGY.point_count
CUBE_SELFPLAY_CONTRACT_ID = "gocube-cube4-golden-selfplay-v1"
CUBE_ARENA_CONTRACT_ID = "gocube-cube4-golden-arena-v1"
CUBE_TARGET_CONTRACT_ID = "gocube-cube4-wdl-side-to-move-v1"


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Stone):
        return int(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _canonical(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


CUBE_TARGET_FINGERPRINT = _fingerprint(
    {
        "contract_id": CUBE_TARGET_CONTRACT_ID,
        "contract_version": 1,
        "value_vector": ["WIN", "DRAW", "LOSS"],
        "value_perspective": "side-to-move",
        "utility": "P(WIN)-P(LOSS)",
        "child_to_parent_sign_flips": 1,
        "technical_results_are_targets": False,
    }
)


def cube_post_action_termination(state: GoldenState, action_count: int) -> str | None:
    """Return formal termination before technical watchdog termination."""

    if state.is_terminal:
        return "DOUBLE_PASS"
    if int(action_count) >= CUBE_WATCHDOG:
        return "TRUNCATED_MOVE_LIMIT"
    return None


def cube_state_identity(state: GoldenState) -> dict[str, object]:
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


def cube_initial_state(*, komi: float = 0.5) -> GoldenState:
    return initial_state(topology=CUBE4_TOPOLOGY, komi=komi)


def cube_state_from_identity(identity: Mapping[str, object]) -> GoldenState:
    expected = cube_initial_state(komi=float(identity.get("komi", 0.5)))
    if identity.get("topology_fingerprint") != expected.topology.fingerprint:
        raise ValueError("Cube state topology fingerprint drift")
    if identity.get("rules_fingerprint") != expected.rules_fingerprint:
        raise ValueError("Cube state rules fingerprint drift")
    history = tuple(tuple(int(value) for value in position) for position in identity["superko_history"])  # type: ignore[index]
    return GoldenState(
        stones=tuple(Stone(int(value)) for value in identity["stones"]),  # type: ignore[index]
        side_to_move=Stone(int(identity["side_to_move"])),  # type: ignore[arg-type]
        superko_history=history,
        consecutive_passes=int(identity["consecutive_passes"]),
        topology=CUBE4_TOPOLOGY,
        rules_id=str(identity["rules_id"]),
        rules_fingerprint=str(identity["rules_fingerprint"]),
        komi=float(identity["komi"]),
        history_provenance=str(identity["history_provenance"]),
    )


@dataclass(frozen=True)
class CubeSelfPlaySearchContract:
    contract_id: str = CUBE_SELFPLAY_CONTRACT_ID
    simulations: int = 64
    cpuct: float = 1.25
    fpu: float = 0.0
    root_noise: bool = True
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.30
    temperature_plies: tuple[int, int] = (1, 8)
    temperature_after: float = 0.0
    fast_search: bool = False
    resign: bool = False
    watchdog: int = CUBE_WATCHDOG
    komi: float = 0.5

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    def validate(self, *, canonical: bool = True) -> None:
        if canonical and asdict(self) != asdict(CubeSelfPlaySearchContract()):
            raise ValueError("Cube self-play contract drift")
        if self.simulations <= 0 or self.watchdog <= 0 or self.cpuct <= 0.0 or self.fpu != 0.0:
            raise ValueError("Cube self-play contract contains invalid search settings")

    @property
    def puct_settings(self):
        from .arena_contract import SearchSettings

        return SearchSettings(
            simulations=self.simulations,
            cpuct=self.cpuct,
            fpu=self.fpu,
            deterministic_tie_break=True,
        )


DEFAULT_CUBE_SELFPLAY_CONTRACT = CubeSelfPlaySearchContract()


def cube_action_index(action: int | str) -> int:
    return CUBE4_TOPOLOGY.pass_action if action == PASS else int(action)


def validate_cube_policy_target(
    state: GoldenState,
    pi: Sequence[float],
    visits: Sequence[int],
    *,
    expected_simulations: int | None = None,
) -> None:
    if len(pi) != CUBE_ACTION_COUNT or len(visits) != CUBE_ACTION_COUNT:
        raise ValueError("Cube policy target must have 97 actions")
    legal = set(legal_actions(state))
    for index, value in enumerate(pi):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("Cube policy target must be finite and non-negative")
        action = PASS if index == CUBE4_TOPOLOGY.pass_action else index
        if action not in legal and float(value) != 0.0:
            raise ValueError("Illegal Cube action has non-zero policy target")
    visit_total = sum(int(value) for value in visits)
    if visit_total <= 0 or (expected_simulations is not None and visit_total != expected_simulations):
        raise ValueError("Cube root visits do not match the search simulation contract")
    if not math.isclose(sum(float(value) for value in pi), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Cube policy target is not normalized")
    if any(int(value) < 0 for value in visits):
        raise ValueError("Cube root visits must be non-negative")


@dataclass(frozen=True)
class CubeSelfPlayPosition:
    ply: int
    state: dict[str, object]
    side_to_move: str
    legal_action_mask: tuple[bool, ...]
    root_visits: tuple[int, ...]
    pi: tuple[float, ...]
    selected_action: int | str
    search_seed: int
    model_hash: str

    def validate(self, *, expected_model_hash: str | None = None) -> None:
        current = cube_state_from_identity(self.state)
        if self.side_to_move != current.side_to_move.name:
            raise ValueError("Cube self-play side-to-move provenance drift")
        if current.komi != 0.5:
            raise ValueError("Cube self-play komi must be exactly 0.5")
        if tuple(self.legal_action_mask) != build_cube_action_mask(current):
            raise ValueError("Cube self-play legal action mask drift")
        validate_cube_policy_target(current, self.pi, self.root_visits)
        if self.selected_action not in legal_actions(current):
            raise ValueError("Cube self-play selected action is illegal")
        if expected_model_hash is not None and self.model_hash != expected_model_hash:
            raise ValueError("Cube self-play model hash drift")


@dataclass(frozen=True)
class CubeSelfPlayGameRecord:
    run_id: str
    game_id: str
    chunk_id: str
    profile_id: str
    profile_fingerprint: str
    selfplay_contract_id: str
    selfplay_contract_fingerprint: str
    topology_fingerprint: str
    geometry_fingerprint: str
    rules_fingerprint: str
    komi: float
    observation_fingerprint: str
    target_fingerprint: str
    model_checkpoint_label: str
    model_hash: str
    checkpoint_artifact_hash: str
    git_commit: str
    git_tree: str
    git_worktree_clean: bool
    master_seed: int
    game_seed: int
    start_state: dict[str, object]
    positions: tuple[CubeSelfPlayPosition, ...]
    final_action_trace: tuple[int | str, ...]
    formal_result: str | None
    technical_termination: str | None
    error: str | None = None
    nn_evaluations: int = 0

    def validate(
        self,
        *,
        require_clean: bool = False,
        expected_contract_fingerprint: str | None = None,
    ) -> None:
        if not self.profile_id or not self.profile_fingerprint.startswith("sha256:"):
            raise ValueError("Cube self-play profile identity is missing")
        if not self.chunk_id:
            raise ValueError("Cube self-play chunk identity is missing")
        expected_game_seed = derive_seed(
            self.master_seed, "cube-selfplay-game-v1", self.chunk_id, self.game_id
        )
        if self.game_seed != expected_game_seed:
            raise ValueError("Cube self-play game seed derivation drift")
        if self.selfplay_contract_id != CUBE_SELFPLAY_CONTRACT_ID:
            raise ValueError("Cube self-play contract id drift")
        if expected_contract_fingerprint is not None and self.selfplay_contract_fingerprint != expected_contract_fingerprint:
            raise ValueError("Cube self-play contract fingerprint drift")
        if self.topology_fingerprint != CUBE4_TOPOLOGY.fingerprint:
            raise ValueError("Cube self-play topology fingerprint drift")
        if self.geometry_fingerprint != CUBE4_TOPOLOGY.geometry_fingerprint:
            raise ValueError("Cube self-play geometry fingerprint drift")
        if self.rules_fingerprint != cube_initial_state(komi=self.komi).rules_fingerprint:
            raise ValueError("Cube self-play rules fingerprint drift")
        if self.komi != 0.5:
            raise ValueError("Cube self-play komi must be exactly 0.5")
        if self.observation_fingerprint != CUBE_OBSERVATION_FINGERPRINT:
            raise ValueError("Cube self-play observation fingerprint drift")
        if self.target_fingerprint != CUBE_TARGET_FINGERPRINT:
            raise ValueError("Cube self-play target fingerprint drift")
        if require_clean and not self.git_worktree_clean:
            raise ValueError("Canonical Cube self-play requires a clean tree")
        if (self.technical_termination is None) != (self.formal_result is not None):
            raise ValueError("Cube technical games cannot contain a formal result")
        if self.formal_result not in (None, "BLACK", "WHITE", "DRAW"):
            raise ValueError("Invalid Cube self-play result")
        state = cube_state_from_identity(self.start_state)
        if not state.is_canonical_live or state.is_terminal:
            raise ValueError("Cube self-play start state must be canonical-live and nonterminal")
        expected_start = cube_state_identity(state)
        if self.start_state != expected_start:
            raise ValueError("Cube self-play start state identity drift")
        replayed_actions: list[int | str] = []
        for expected_ply, position in enumerate(self.positions, start=1):
            if position.ply != expected_ply:
                raise ValueError("Cube self-play ply ordering drift")
            if position.state != cube_state_identity(state):
                raise ValueError("Cube self-play state transition drift")
            position.validate(expected_model_hash=self.model_hash)
            replayed_actions.append(position.selected_action)
            try:
                state = apply_action(state, position.selected_action).after
            except IllegalMoveError as exc:
                raise ValueError("Cube self-play action trace contains an illegal action") from exc
        if tuple(replayed_actions) != self.final_action_trace:
            raise ValueError("Cube self-play action trace drift")
        if self.technical_termination is None and len(self.final_action_trace) != len(self.positions):
            raise ValueError("Cube formal trace length does not match positions")
        if self.formal_result is not None:
            if not state.is_terminal:
                raise ValueError("Cube formal self-play result has no terminal state")
            if result_from_terminal(state).winner.value != self.formal_result:
                raise ValueError("Cube formal self-play result drift")
        elif state.is_terminal:
            raise ValueError("Cube terminal self-play trace is missing a formal result")
        if len(self.final_action_trace) > CUBE_WATCHDOG:
            raise ValueError("Cube self-play trace exceeded the watchdog")

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def cube_z_target(winner: str | Winner, side_to_move: Stone | str) -> tuple[float, float, float]:
    winner_name = winner.value if isinstance(winner, Winner) else str(winner)
    side_name = side_to_move.name if isinstance(side_to_move, Stone) else str(side_to_move)
    if winner_name == "DRAW":
        return (0.0, 1.0, 0.0)
    if winner_name not in ("BLACK", "WHITE") or side_name not in ("BLACK", "WHITE"):
        raise ValueError("Cube z target requires formal result and side-to-move")
    return (1.0, 0.0, 0.0) if winner_name == side_name else (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class CubeTrainingSample:
    run_id: str
    game_id: str
    ply: int
    state: dict[str, object]
    side_to_move: str
    observation: tuple[tuple[float, ...], ...]
    legal_action_mask: tuple[bool, ...]
    root_visits: tuple[int, ...]
    pi: tuple[float, ...]
    z: tuple[float, float, float]
    model_hash: str
    selfplay_contract_fingerprint: str
    observation_fingerprint: str = CUBE_OBSERVATION_FINGERPRINT
    target_contract_id: str = CUBE_TARGET_CONTRACT_ID
    target_fingerprint: str = CUBE_TARGET_FINGERPRINT

    def validate(
        self,
        *,
        expected_contract_fingerprint: str | None = None,
        expected_model_hash: str | None = None,
    ) -> None:
        state = cube_state_from_identity(self.state)
        if self.side_to_move != state.side_to_move.name:
            raise ValueError("Cube replay side-to-move drift")
        if state.komi != 0.5:
            raise ValueError("Cube replay komi must be exactly 0.5")
        if len(self.observation) != 15 or any(len(row) != 96 for row in self.observation):
            raise ValueError("Cube replay observation shape drift")
        generated_observation = build_cube_observation(state)
        stored_observation = torch.tensor(self.observation, dtype=torch.float32)
        if not torch.equal(stored_observation, generated_observation):
            raise ValueError("Cube replay observation values drift")
        if len(self.legal_action_mask) != CUBE_ACTION_COUNT or tuple(self.legal_action_mask) != build_cube_action_mask(state):
            raise ValueError("Cube replay legal mask drift")
        validate_cube_policy_target(state, self.pi, self.root_visits)
        if len(self.z) != 3 or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in self.z):
            raise ValueError("Cube replay z target is invalid")
        if not math.isclose(sum(self.z), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Cube replay z target is not normalized")
        if self.observation_fingerprint != CUBE_OBSERVATION_FINGERPRINT or self.target_contract_id != CUBE_TARGET_CONTRACT_ID or self.target_fingerprint != CUBE_TARGET_FINGERPRINT:
            raise ValueError("Cube replay semantic fingerprint drift")
        if expected_contract_fingerprint is not None and self.selfplay_contract_fingerprint != expected_contract_fingerprint:
            raise ValueError("Cube replay self-play contract fingerprint drift")
        if expected_model_hash is not None and self.model_hash != expected_model_hash:
            raise ValueError("Cube replay model hash drift")

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def build_cube_replay_samples(
    game: CubeSelfPlayGameRecord,
    *,
    expected_contract_fingerprint: str | None = None,
) -> tuple[CubeTrainingSample, ...]:
    game.validate(expected_contract_fingerprint=expected_contract_fingerprint)
    if game.technical_termination is not None or game.formal_result is None:
        raise ValueError("Technical Cube games are excluded from replay")
    samples = []
    for position in game.positions:
        state = cube_state_from_identity(position.state)
        observation = build_cube_observation(state)
        sample = CubeTrainingSample(
            run_id=game.run_id,
            game_id=game.game_id,
            ply=position.ply,
            state=position.state,
            side_to_move=position.side_to_move,
            observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
            legal_action_mask=build_cube_action_mask(state),
            root_visits=position.root_visits,
            pi=position.pi,
            z=cube_z_target(game.formal_result, state.side_to_move),
            model_hash=game.model_hash,
            selfplay_contract_fingerprint=game.selfplay_contract_fingerprint,
        )
        sample.validate(expected_contract_fingerprint=expected_contract_fingerprint)
        samples.append(sample)
    return tuple(samples)


def sample_cube_action_from_visits(result: SearchResult, *, temperature: float, rng: random.Random) -> int | str:
    if len(result.root_visits) != CUBE_ACTION_COUNT or not result.legal_actions:
        raise ValueError("Invalid Cube root visit result")
    indices = [cube_action_index(action) for action in result.legal_actions]
    if temperature <= 0.0:
        maximum = max(result.root_visits[index] for index in indices)
        return min(
            (action for action in result.legal_actions if result.root_visits[cube_action_index(action)] == maximum),
            key=cube_action_index,
        )
    weights = [float(result.root_visits[index]) ** (1.0 / float(temperature)) for index in indices]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return rng.choice(result.legal_actions)
    threshold = rng.random() * total
    for action, weight in zip(result.legal_actions, weights):
        threshold -= weight
        if threshold <= 0.0:
            return action
    return result.legal_actions[-1]


class CubeSelfPlayRunner:
    def __init__(
        self,
        model: nn.Module,
        *,
        run_id: str,
        profile_id: str,
        profile_fingerprint: str,
        model_checkpoint_label: str,
        checkpoint_artifact_hash: str,
        master_seed: int,
        chunk_id: str,
        contract: CubeSelfPlaySearchContract = DEFAULT_CUBE_SELFPLAY_CONTRACT,
        code_identity: CodeIdentity | None = None,
        device: str | torch.device = "cpu",
        evaluator: GoldenCubeNeuralEvaluator | None = None,
        allow_noncanonical_contract: bool = False,
    ) -> None:
        contract.validate(canonical=not allow_noncanonical_contract)
        self.model = model
        self.run_id = run_id
        self.profile_id = profile_id
        self.profile_fingerprint = profile_fingerprint
        self.model_checkpoint_label = model_checkpoint_label
        self.model_hash = cube_model_hash(model)
        self.checkpoint_artifact_hash = checkpoint_artifact_hash
        self.master_seed = int(master_seed)
        self.chunk_id = str(chunk_id)
        self.contract = contract
        self.code_identity = code_identity or capture_code_identity()
        self.device = torch.device(device)
        self.evaluator = evaluator or GoldenCubeNeuralEvaluator(model, device=self.device)

    def play_game(self, game_id: str) -> CubeSelfPlayGameRecord:
        game_seed = derive_seed(self.master_seed, "cube-selfplay-game-v1", self.chunk_id, game_id)
        rng = random.Random(game_seed)
        state = cube_initial_state(komi=self.contract.komi)
        start = cube_state_identity(state)
        positions: list[CubeSelfPlayPosition] = []
        trace: list[int | str] = []
        formal: str | None = None
        technical: str | None = None
        error: str | None = None
        for ply in range(1, self.contract.watchdog + 1):
            search_seed = derive_seed(game_seed, ply, "search")
            try:
                wrapped = SelfPlayCubeRootNoiseEvaluator(
                    self.evaluator,
                    state,
                    seed=derive_seed(search_seed, "dirichlet"),
                    epsilon=self.contract.dirichlet_epsilon,
                    alpha=self.contract.dirichlet_alpha,
                )
                result = SequentialPUCT(self.contract.puct_settings).search(state, wrapped, seed=search_seed)
                if result.simulations != self.contract.simulations or sum(result.root_visits) != self.contract.simulations:
                    raise SearchError("Cube root visits do not match the self-play simulation contract")
                action = sample_cube_action_from_visits(
                    result,
                    temperature=1.0 if ply <= self.contract.temperature_plies[1] else self.contract.temperature_after,
                    rng=rng,
                )
            except SearchError as exc:
                technical, error = "ERROR_SEARCH", f"{type(exc).__name__}: {exc}"
                break
            except Exception as exc:
                technical, error = "ERROR_SEARCH", f"{type(exc).__name__}: {exc}"
                break
            positions.append(CubeSelfPlayPosition(
                ply=ply,
                state=cube_state_identity(state),
                side_to_move=state.side_to_move.name,
                legal_action_mask=build_cube_action_mask(state),
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
            termination = cube_post_action_termination(state, ply)
            if termination == "DOUBLE_PASS":
                formal = result_from_terminal(state).winner.value
                break
            if termination == "TRUNCATED_MOVE_LIMIT":
                technical, error = "TRUNCATED_MOVE_LIMIT", f"Cube watchdog reached {self.contract.watchdog} actions"
                break
        else:
            technical, error = "TRUNCATED_MOVE_LIMIT", f"Cube watchdog reached {self.contract.watchdog} actions"
        record = CubeSelfPlayGameRecord(
            run_id=self.run_id,
            game_id=game_id,
            chunk_id=self.chunk_id,
            profile_id=self.profile_id,
            profile_fingerprint=self.profile_fingerprint,
            selfplay_contract_id=self.contract.contract_id,
            selfplay_contract_fingerprint=self.contract.fingerprint,
            topology_fingerprint=CUBE4_TOPOLOGY.fingerprint,
            geometry_fingerprint=CUBE4_TOPOLOGY.geometry_fingerprint,
            rules_fingerprint=state.rules_fingerprint,
            komi=self.contract.komi,
            observation_fingerprint=CUBE_OBSERVATION_FINGERPRINT,
            target_fingerprint=CUBE_TARGET_FINGERPRINT,
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


_PROCESS_MODEL: nn.Module | None = None
_PROCESS_CONFIG: dict[str, object] = {}


def _cube_process_worker_init(
    checkpoint_path: str,
    expected_hash: str,
    run_id: str,
    profile_id: str,
    profile_fingerprint: str,
    label: str,
    artifact: str,
    master_seed: int,
    chunk_id: str,
    code_commit: str,
    code_tree: str,
    code_clean: bool,
    device: str,
    contract: Mapping[str, object],
    allow_noncanonical_contract: bool,
) -> None:
    global _PROCESS_MODEL, _PROCESS_CONFIG
    torch.set_num_threads(1)
    worker_device = torch.device(device)
    model = GoldenCubeGraphNetV1().to(worker_device)
    cube_load_checkpoint(checkpoint_path, model=model, expected={"model_hash": expected_hash}, device=worker_device)
    if cube_model_hash(model) != expected_hash:
        raise RuntimeError("Cube worker loaded the wrong model hash")
    _PROCESS_MODEL = model
    _PROCESS_CONFIG = {
        "run_id": run_id,
        "profile_id": profile_id,
        "profile_fingerprint": profile_fingerprint,
        "label": label,
        "artifact": artifact,
        "master_seed": master_seed,
        "chunk_id": chunk_id,
        "device": str(worker_device),
        "contract": dict(contract),
        "allow_noncanonical_contract": allow_noncanonical_contract,
        "code": CodeIdentity(code_commit, code_tree, code_clean),
    }


def _cube_process_play_game(game_id: str) -> CubeSelfPlayGameRecord:
    if _PROCESS_MODEL is None:
        raise RuntimeError("Cube self-play process worker is not initialized")
    evaluator = GoldenCubeNeuralEvaluator(_PROCESS_MODEL, device=str(_PROCESS_CONFIG["device"]))
    runner = CubeSelfPlayRunner(
        _PROCESS_MODEL,
        run_id=str(_PROCESS_CONFIG["run_id"]),
        profile_id=str(_PROCESS_CONFIG["profile_id"]),
        profile_fingerprint=str(_PROCESS_CONFIG["profile_fingerprint"]),
        model_checkpoint_label=str(_PROCESS_CONFIG["label"]),
        checkpoint_artifact_hash=str(_PROCESS_CONFIG["artifact"]),
        master_seed=int(_PROCESS_CONFIG["master_seed"]),
        chunk_id=str(_PROCESS_CONFIG["chunk_id"]),
        contract=CubeSelfPlaySearchContract(**_PROCESS_CONFIG["contract"]),  # type: ignore[arg-type]
        allow_noncanonical_contract=bool(_PROCESS_CONFIG["allow_noncanonical_contract"]),
        code_identity=_PROCESS_CONFIG["code"],  # type: ignore[arg-type]
        evaluator=evaluator,
    )
    return runner.play_game(game_id)


def run_cube_selfplay_games(
    model: nn.Module,
    game_ids: Sequence[str],
    *,
    run_id: str,
    profile_id: str,
    profile_fingerprint: str,
    model_checkpoint_label: str,
    checkpoint_artifact_hash: str,
    master_seed: int,
    chunk_id: str = "default",
    code_identity: CodeIdentity,
    checkpoint_path: str | Path | None,
    workers: int = 1,
    device: str | torch.device = "cpu",
    contract: CubeSelfPlaySearchContract = DEFAULT_CUBE_SELFPLAY_CONTRACT,
    allow_noncanonical_contract: bool = False,
) -> tuple[CubeSelfPlayGameRecord, ...]:
    ordered = tuple(sorted(str(game_id) for game_id in game_ids))
    if len(ordered) != len(set(ordered)) or workers <= 0:
        raise ValueError("Cube self-play game IDs must be unique and workers positive")
    if workers == 1:
        evaluator = GoldenCubeNeuralEvaluator(model, device=device)
        runner = CubeSelfPlayRunner(
            model,
            run_id=run_id,
            profile_id=profile_id,
            profile_fingerprint=profile_fingerprint,
            model_checkpoint_label=model_checkpoint_label,
            checkpoint_artifact_hash=checkpoint_artifact_hash,
            master_seed=master_seed,
            chunk_id=chunk_id,
            contract=contract,
            allow_noncanonical_contract=allow_noncanonical_contract,
            code_identity=code_identity,
            device=device,
            evaluator=evaluator,
        )
        return tuple(runner.play_game(game_id) for game_id in ordered)
    if checkpoint_path is None:
        raise ValueError("Process Cube self-play requires an immutable checkpoint path")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=get_context("fork" if torch.device(device).type == "cpu" else "spawn"),
        initializer=_cube_process_worker_init,
        initargs=(
            str(checkpoint_path), cube_model_hash(model), run_id, profile_id,
            profile_fingerprint, model_checkpoint_label, checkpoint_artifact_hash,
            master_seed, chunk_id, code_identity.git_commit_sha, code_identity.git_tree_sha,
            code_identity.working_tree_clean, str(device), asdict(contract),
            allow_noncanonical_contract,
        ),
    ) as pool:
        records = tuple(pool.map(_cube_process_play_game, ordered))
    return records


def cube_compare_selfplay_evidence(
    serial: Sequence[CubeSelfPlayGameRecord],
    parallel: Sequence[CubeSelfPlayGameRecord],
) -> None:
    """Exact serial/process equivalence gate for fixed game IDs."""

    left = {record.game_id: record for record in serial}
    right = {record.game_id: record for record in parallel}
    if set(left) != set(right):
        raise ValueError("Cube self-play equivalence game ID sets differ")
    for game_id in sorted(left):
        a, b = left[game_id], right[game_id]
        if (a.formal_result, a.technical_termination, a.final_action_trace) != (
            b.formal_result, b.technical_termination, b.final_action_trace
        ):
            raise ValueError(f"Cube self-play equivalence action/result drift for {game_id}")
        if len(a.positions) != len(b.positions):
            raise ValueError(f"Cube self-play equivalence position-count drift for {game_id}")
        for left_position, right_position in zip(a.positions, b.positions):
            left_state = cube_state_from_identity(left_position.state)
            right_state = cube_state_from_identity(right_position.state)
            left_z = cube_z_target(a.formal_result or "DRAW", left_state.side_to_move) if a.formal_result else None
            right_z = cube_z_target(b.formal_result or "DRAW", right_state.side_to_move) if b.formal_result else None
            if (
                left_position.state,
                left_position.side_to_move,
                left_position.legal_action_mask,
                left_position.root_visits,
                left_position.pi,
                left_position.selected_action,
                left_z,
            ) != (
                right_position.state,
                right_position.side_to_move,
                right_position.legal_action_mask,
                right_position.root_visits,
                right_position.pi,
                right_position.selected_action,
                right_z,
            ):
                raise ValueError(f"Cube self-play equivalence target drift for {game_id} ply {left_position.ply}")


@dataclass(frozen=True)
class CubeTrainingUpdate:
    update: int
    policy_loss: float
    value_loss: float
    total_loss: float
    gradient_norm: float
    learning_rate: float
    batch_size: int
    cumulative_samples: int


def train_cube_batch_schedule(
    model: nn.Module,
    samples: Sequence[CubeTrainingSample],
    batches: Sequence[Sequence[int]],
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
    update_offset: int = 0,
    sample_offset: int = 0,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    if not samples or not batches:
        raise ValueError("Cube training requires samples and batches")
    for sample in samples:
        sample.validate()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    observations = torch.tensor([sample.observation for sample in samples], dtype=torch.float32)
    policies = torch.tensor([sample.pi for sample in samples], dtype=torch.float32)
    values = torch.tensor([sample.z for sample in samples], dtype=torch.float32)
    device = next(model.parameters()).device
    observations, policies, values = observations.to(device), policies.to(device), values.to(device)
    model.train()
    updates: list[CubeTrainingUpdate] = []
    consumed = int(sample_offset)
    for batch_number, batch in enumerate(batches, start=1):
        indices = torch.tensor(tuple(int(index) for index in batch), dtype=torch.long, device=device)
        if indices.numel() == 0:
            raise ValueError("Cube training contains an empty batch")
        policy_logits, value_logits = model(observations[indices])
        policy_loss = -(policies[indices] * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        value_loss = -(values[indices] * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
        total_loss = policy_loss + value_loss
        if not bool(torch.isfinite(total_loss).all()):
            raise FloatingPointError("Cube trainer encountered non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).all()):
            raise FloatingPointError("Cube trainer encountered non-finite gradient")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError("Cube trainer produced non-finite parameter")
        consumed += int(indices.numel())
        updates.append(CubeTrainingUpdate(
            update=update_offset + batch_number,
            policy_loss=float(policy_loss.detach()),
            value_loss=float(value_loss.detach()),
            total_loss=float(total_loss.detach()),
            gradient_norm=float(gradient_norm),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            batch_size=int(indices.numel()),
            cumulative_samples=consumed,
        ))
    expected_consumed = sum(len(batch) for batch in batches)
    return optimizer, {
        "updates": len(updates) + update_offset,
        "phase_updates": len(updates),
        "exact_samples_consumed": expected_consumed,
        "cumulative_samples": consumed,
        "batch_sizes": [item.batch_size for item in updates],
        "final_batch_size": updates[-1].batch_size,
        "metrics": [_jsonable(asdict(item)) for item in updates],
    }


def cube_replay_batches(cumulative_count: int, new_positions: int, *, seed: int, batch_size: int = 64) -> tuple[tuple[int, ...], ...]:
    if cumulative_count <= 0 or not 0 < new_positions <= cumulative_count or batch_size <= 0:
        raise ValueError("Cube replay budget is invalid")
    selected = random.Random(int(seed)).sample(range(cumulative_count), new_positions)
    return tuple(tuple(selected[offset:offset + batch_size]) for offset in range(0, len(selected), batch_size))


def cube_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def cube_save_checkpoint(path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer | None, metadata: Mapping[str, object]) -> dict[str, object]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_out = dict(metadata)
    metadata_out["model_hash"] = cube_model_hash(model)
    metadata_out["checkpoint_schema_version"] = CUBE_CHECKPOINT_SCHEMA_VERSION
    payload = {
        "checkpoint_schema_version": CUBE_CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata_out,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    metadata_out["artifact_sha256"] = file_sha256(destination)
    destination.with_suffix(".metadata.json").write_text(json.dumps(_jsonable(metadata_out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_out


def cube_load_checkpoint(path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, expected: Mapping[str, object] | None = None, device: str | torch.device = "cpu") -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Cube checkpoint does not exist: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    if payload.get("checkpoint_schema_version") != CUBE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Cube checkpoint schema mismatch")
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("model_hash") != cube_model_hash_from_state_dict(model, payload["model_state_dict"]):
        raise ValueError("Cube checkpoint model hash mismatch")
    if expected:
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Cube checkpoint metadata mismatch for {key}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        if payload.get("optimizer_state_dict") is None:
            raise ValueError("Cube checkpoint is missing optimizer state")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    sidecar = source.with_suffix(".metadata.json")
    if not sidecar.is_file():
        raise ValueError("Cube checkpoint metadata sidecar is missing")
    sidecar_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    artifact = file_sha256(source)
    if sidecar_metadata.get("artifact_sha256") != artifact or sidecar_metadata.get("model_hash") != metadata.get("model_hash"):
        raise ValueError("Cube checkpoint sidecar hash mismatch")
    metadata["artifact_sha256"] = artifact
    return metadata


def cube_model_hash_from_state_dict(model: nn.Module, state_dict: Mapping[str, Tensor]) -> str:
    clone = GoldenCubeGraphNetV1(topology=CUBE4_TOPOLOGY, hidden=model.hidden, blocks=model.blocks_count)
    clone.load_state_dict(state_dict, strict=True)
    return cube_model_hash(clone)
