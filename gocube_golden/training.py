"""Typed Golden self-play records, replay targets, trainer and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Mapping, Sequence

_torch = importlib.import_module("torch")
torch = _torch
Tensor = torch.Tensor
nn = torch.nn
F = importlib.import_module("torch.nn.functional")

from .arena_contract import GOLDEN_MOVE_LIMIT, SearchSettings
from .neural import (
    ACTION_COUNT,
    OBSERVATION_FINGERPRINT,
    PASS_INDEX,
    GoldenGraphNetV1,
    GoldenNeuralEvaluator,
    SelfPlayRootNoiseEvaluator,
    build_action_mask,
    build_observation,
    model_hash,
)
from .provenance import CodeIdentity, capture_code_identity, derive_seed, file_sha256
from .result import Winner, result_from_terminal
from .rules import IllegalMoveError, apply_action, legal_actions
from .search import SearchError, SearchResult, SequentialPUCT
from .stage3_contract import (
    PROFILE_ID,
    SELFPLAY_CONTRACT_ID,
    SELFPLAY_CONTRACT_FINGERPRINT,
    TARGET_CONTRACT_ID,
    TARGET_FINGERPRINT,
    profile_fingerprint,
    validate_checkpoint_metadata,
)
from .state import BLACK, PASS, WHITE, GoldenState, Stone, initial_state


SELFPLAY_SEARCH_IMPLEMENTATION_ID = "golden-sequential-puct-v1"
SELFPLAY_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1


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


def state_identity(state: GoldenState) -> dict[str, object]:
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


def state_from_identity(identity: Mapping[str, object]) -> GoldenState:
    if identity.get("topology_fingerprint") != initial_state().topology.fingerprint:
        raise ValueError("Self-play state topology fingerprint drift")
    if identity.get("rules_fingerprint") != initial_state().rules_fingerprint:
        raise ValueError("Self-play state rules fingerprint drift")
    history = tuple(tuple(int(value) for value in position) for position in identity["superko_history"])  # type: ignore[index]
    return GoldenState(
        stones=tuple(Stone(int(value)) for value in identity["stones"]),  # type: ignore[index]
        side_to_move=Stone(int(identity["side_to_move"])),  # type: ignore[arg-type]
        superko_history=history,
        consecutive_passes=int(identity["consecutive_passes"]),
        topology=initial_state().topology,
        rules_id=str(identity["rules_id"]),
        rules_fingerprint=str(identity["rules_fingerprint"]),
        komi=float(identity["komi"]),
        history_provenance=str(identity["history_provenance"]),
    )


@dataclass(frozen=True)
class SelfPlaySearchContract:
    contract_id: str = SELFPLAY_CONTRACT_ID
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
    watchdog: int = GOLDEN_MOVE_LIMIT
    komi: float = 0.5

    @property
    def fingerprint(self) -> str:
        result = "sha256:" + hashlib.sha256(_canonical(asdict(self)).encode("utf-8")).hexdigest()
        if result != SELFPLAY_CONTRACT_FINGERPRINT:
            raise RuntimeError("Golden self-play contract fingerprint constant drifted")
        return result

    def validate(self) -> None:
        if asdict(self) != asdict(SelfPlaySearchContract()):
            raise ValueError("Stage-3 self-play search contract drift")

    @property
    def puct_settings(self) -> SearchSettings:
        # Root exploration is intentionally outside Arena-owned SearchSettings.
        return SearchSettings(
            simulations=self.simulations,
            cpuct=self.cpuct,
            fpu=self.fpu,
            deterministic_tie_break=True,
        )


DEFAULT_SELFPLAY_CONTRACT = SelfPlaySearchContract()


def _action_index(action: int | str) -> int:
    return PASS_INDEX if action == PASS else int(action)


def _validate_policy_target(state: GoldenState, pi: Sequence[float], visits: Sequence[int]) -> None:
    if len(pi) != ACTION_COUNT or len(visits) != ACTION_COUNT:
        raise ValueError("Golden policy target must have 26 actions")
    legal = set(legal_actions(state))
    for index, value in enumerate(pi):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("Golden policy target must be finite and non-negative")
        action = PASS if index == PASS_INDEX else index
        if action not in legal and float(value) != 0.0:
            raise ValueError("Illegal action has non-zero Golden policy target")
    if sum(int(value) for value in visits) <= 0:
        raise ValueError("Golden policy target has zero root visits")
    if not math.isclose(sum(float(value) for value in pi), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Golden policy target is not normalized")
    if any(int(value) < 0 for value in visits):
        raise ValueError("Golden root visits must be non-negative")


@dataclass(frozen=True)
class SelfPlayPosition:
    ply: int
    state: dict[str, object]
    side_to_move: str
    root_visits: tuple[int, ...]
    pi: tuple[float, ...]
    selected_action: int | str
    search_seed: int
    model_hash: str

    def validate(self, *, expected_model_hash: str | None = None) -> None:
        current = state_from_identity(self.state)
        if self.side_to_move != current.side_to_move.name:
            raise ValueError("Self-play position side-to-move provenance drift")
        _validate_policy_target(current, self.pi, self.root_visits)
        if self.selected_action not in legal_actions(current):
            raise ValueError("Self-play selected action is illegal")
        if expected_model_hash is not None and self.model_hash != expected_model_hash:
            raise ValueError("Self-play position model hash drift")
        if self.ply <= 0 or not self.model_hash.startswith("sha256:"):
            raise ValueError("Self-play position identity is malformed")


@dataclass(frozen=True)
class SelfPlayGameRecord:
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
    positions: tuple[SelfPlayPosition, ...]
    final_action_trace: tuple[int | str, ...]
    formal_result: str | None
    technical_termination: str | None
    error: str | None = None
    nn_evaluations: int = 0

    def validate(self, *, require_clean: bool = False) -> None:
        if self.profile_id != PROFILE_ID or not self.profile_fingerprint.startswith("sha256:"):
            raise ValueError("Self-play training profile identity drift")
        if self.selfplay_contract_id != SELFPLAY_CONTRACT_ID:
            raise ValueError("Self-play contract id drift")
        if self.model_checkpoint_label == "" or not self.model_hash.startswith("sha256:"):
            raise ValueError("Self-play model identity is missing")
        if require_clean and not self.git_worktree_clean:
            raise ValueError("Canonical self-play requires a clean tree")
        technical = self.technical_termination is not None
        if technical != (self.formal_result is None):
            raise ValueError("Technical self-play games cannot contain a formal result")
        if self.formal_result not in (None, "BLACK", "WHITE", "DRAW"):
            raise ValueError("Invalid Golden self-play result")
        if technical and self.positions:
            # Positions remain useful diagnostics, but the replay builder will reject them.
            if not self.error:
                raise ValueError("Technical self-play game must preserve an error")
        for position in self.positions:
            position.validate(expected_model_hash=self.model_hash)
        if len(self.final_action_trace) != len(self.positions) and self.technical_termination is None:
            raise ValueError("Formal self-play trace length does not match positions")

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def z_target(winner: str | Winner, side_to_move: Stone | str) -> tuple[float, float, float]:
    winner_name = winner.value if isinstance(winner, Winner) else str(winner)
    side_name = side_to_move.name if isinstance(side_to_move, Stone) else str(side_to_move)
    if winner_name == "DRAW":
        return (0.0, 1.0, 0.0)
    if winner_name not in ("BLACK", "WHITE") or side_name not in ("BLACK", "WHITE"):
        raise ValueError("z_target requires a formal BLACK/WHITE/DRAW and a side-to-move")
    return (1.0, 0.0, 0.0) if winner_name == side_name else (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class GoldenTrainingSample:
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
    observation_fingerprint: str = OBSERVATION_FINGERPRINT
    target_contract_id: str = TARGET_CONTRACT_ID
    target_fingerprint: str = TARGET_FINGERPRINT

    def validate(self) -> None:
        current = state_from_identity(self.state)
        if self.side_to_move != current.side_to_move.name:
            raise ValueError("Replay side-to-move provenance drift")
        if len(self.observation) != 6 or any(len(row) != 25 for row in self.observation):
            raise ValueError("Replay observation shape drift")
        if len(self.legal_action_mask) != ACTION_COUNT:
            raise ValueError("Replay action mask shape drift")
        if tuple(self.legal_action_mask) != build_action_mask(current):
            raise ValueError("Replay legal action mask does not match Golden rules")
        _validate_policy_target(current, self.pi, self.root_visits)
        if len(self.z) != 3 or any(not math.isfinite(float(value)) or float(value) < 0 for value in self.z):
            raise ValueError("Replay z target is invalid")
        if not math.isclose(sum(self.z), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Replay z target is not normalized")
        if self.observation_fingerprint != OBSERVATION_FINGERPRINT or self.target_contract_id != TARGET_CONTRACT_ID or self.target_fingerprint != TARGET_FINGERPRINT:
            raise ValueError("Replay semantic fingerprint drift")

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def build_replay_samples(game: SelfPlayGameRecord) -> tuple[GoldenTrainingSample, ...]:
    game.validate()
    if game.technical_termination is not None:
        raise ValueError("Technical self-play games are excluded from training replay")
    if game.formal_result is None:
        raise ValueError("Replay requires a formal Golden result")
    samples: list[GoldenTrainingSample] = []
    for position in game.positions:
        state = state_from_identity(position.state)
        observation = build_observation(state)
        sample = GoldenTrainingSample(
            run_id=game.run_id,
            game_id=game.game_id,
            ply=position.ply,
            state=position.state,
            side_to_move=position.side_to_move,
            observation=tuple(tuple(float(value) for value in row) for row in observation.tolist()),
            legal_action_mask=build_action_mask(state),
            root_visits=position.root_visits,
            pi=position.pi,
            z=z_target(game.formal_result, state.side_to_move),
            model_hash=game.model_hash,
            selfplay_contract_fingerprint=game.selfplay_contract_fingerprint,
        )
        sample.validate()
        samples.append(sample)
    return tuple(samples)


class GoldenSelfPlayRunner:
    def __init__(
        self,
        model: nn.Module,
        *,
        run_id: str,
        profile_fingerprint: str,
        model_checkpoint_label: str,
        checkpoint_artifact_hash: str,
        master_seed: int,
        contract: SelfPlaySearchContract = DEFAULT_SELFPLAY_CONTRACT,
        code_identity: CodeIdentity | None = None,
        device: str | torch.device = "cpu",
        evaluator: GoldenNeuralEvaluator | None = None,
    ) -> None:
        contract.validate()
        self.model = model
        self.run_id = run_id
        self.profile_fingerprint = profile_fingerprint
        self.model_checkpoint_label = model_checkpoint_label
        self.model_hash = model_hash(model)
        self.checkpoint_artifact_hash = checkpoint_artifact_hash
        self.master_seed = int(master_seed)
        self.contract = contract
        self.code_identity = code_identity or capture_code_identity()
        self.device = torch.device(device)
        # The coordinator owns the one immutable model placement.  Parallel
        # game workers receive this evaluator rather than concurrently calling
        # model.to(device), which is not a safe operation.
        self.evaluator = evaluator or GoldenNeuralEvaluator(model, device=self.device)

    def play_game(self, game_id: str) -> SelfPlayGameRecord:
        game_seed = derive_seed(self.master_seed, self.run_id, game_id, "game")
        rng = random.Random(game_seed)
        state = initial_state()
        start = state_identity(state)
        positions: list[SelfPlayPosition] = []
        trace: list[int | str] = []
        technical: str | None = None
        error: str | None = None
        formal: str | None = None
        for ply in range(1, self.contract.watchdog + 1):
            search_seed = derive_seed(game_seed, ply, "search")
            try:
                wrapped = SelfPlayRootNoiseEvaluator(
                    self.evaluator, state, seed=derive_seed(search_seed, "dirichlet"),
                    epsilon=self.contract.dirichlet_epsilon,
                    alpha=self.contract.dirichlet_alpha,
                )
                result = SequentialPUCT(self.contract.puct_settings).search(
                    state, wrapped, seed=search_seed
                )
                action = sample_action_from_visits(
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
            positions.append(SelfPlayPosition(
                ply=ply,
                state=state_identity(state),
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
            technical, error = "TRUNCATED_MOVE_LIMIT", "Golden self-play watchdog reached 500 actions"
        record = SelfPlayGameRecord(
            run_id=self.run_id,
            game_id=game_id,
            profile_id=PROFILE_ID,
            profile_fingerprint=self.profile_fingerprint,
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


def compare_selfplay_evidence(
    sequential: Sequence[SelfPlayGameRecord],
    parallel: Sequence[SelfPlayGameRecord],
) -> None:
    """Equivalence gate for the only canonical optimization: independent games."""
    left = {record.game_id: record for record in sequential}
    right = {record.game_id: record for record in parallel}
    if set(left) != set(right):
        raise ValueError("Self-play equivalence gate game_id sets differ")
    for game_id in sorted(left):
        a = left[game_id]
        b = right[game_id]
        if a.formal_result != b.formal_result or a.technical_termination != b.technical_termination:
            raise ValueError(f"Self-play equivalence gate result drift for {game_id}")
        if a.final_action_trace != b.final_action_trace:
            raise ValueError(f"Self-play equivalence gate action trace drift for {game_id}")
        if len(a.positions) != len(b.positions):
            raise ValueError(f"Self-play equivalence gate position count drift for {game_id}")
        for left_position, right_position in zip(a.positions, b.positions):
            if left_position.root_visits != right_position.root_visits:
                raise ValueError(f"Self-play equivalence gate root visits drift for {game_id} ply {left_position.ply}")
            if left_position.pi != right_position.pi:
                raise ValueError(f"Self-play equivalence gate pi drift for {game_id} ply {left_position.ply}")
        if a.formal_result is not None:
            left_z = tuple(z_target(a.formal_result, position.side_to_move) for position in a.positions)
            right_z = tuple(z_target(b.formal_result, position.side_to_move) for position in b.positions)
            if left_z != right_z:
                raise ValueError(f"Self-play equivalence gate z drift for {game_id}")


def run_selfplay_games(
    runner_factory,
    game_ids: Sequence[str],
    *,
    workers: int = 1,
) -> tuple[SelfPlayGameRecord, ...]:
    """Run games independently, then return evidence in stable game_id order.

    Each worker constructs its own evaluator wrapper around the one immutable
    model owned by the caller.  No worker id, PID, completion order, or global
    RNG is consulted by the game seed schedule.
    """
    if workers <= 0:
        raise ValueError("Self-play workers must be positive")
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("Self-play game ids must be unique")
    ordered_ids = tuple(sorted(str(game_id) for game_id in game_ids))
    if workers == 1:
        return tuple(runner_factory(game_id).play_game(game_id) for game_id in ordered_ids)
    with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="golden-selfplay") as pool:
        futures = [pool.submit(runner_factory(game_id).play_game, game_id) for game_id in ordered_ids]
        records = [future.result() for future in futures]
    return tuple(sorted(records, key=lambda record: record.game_id))


def sample_action_from_visits(result: SearchResult, *, temperature: float, rng: random.Random) -> int | str:
    visits = tuple(int(value) for value in result.root_visits)
    legal = tuple(result.legal_actions)
    if len(visits) != ACTION_COUNT or not legal:
        raise ValueError("Invalid Golden root visit result")
    legal_indices = [_action_index(action) for action in legal]
    if temperature <= 0.0:
        maximum = max(visits[index] for index in legal_indices)
        return min((action for action in legal if visits[_action_index(action)] == maximum), key=_action_index)
    weights = [float(visits[index]) ** (1.0 / float(temperature)) for index in legal_indices]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return rng.choice(legal)
    threshold = rng.random() * total
    for action, weight in zip(legal, weights):
        threshold -= weight
        if threshold <= 0.0:
            return action
    return legal[-1]


@dataclass(frozen=True)
class TrainingUpdateMetrics:
    update: int
    policy_loss: float
    value_loss: float
    total_loss: float
    gradient_norm: float
    learning_rate: float
    batch_size: int
    cumulative_samples: int


class GoldenTrainer:
    def __init__(self, model: nn.Module, *, learning_rate: float = 1e-3, weight_decay: float = 0.0) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.update_count = 0
        self.samples_consumed = 0

    @staticmethod
    def _ensure_finite(value: Tensor, label: str) -> None:
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Golden trainer encountered non-finite {label}")

    def train(
        self,
        samples: Sequence[GoldenTrainingSample],
        *,
        updates: int = 200,
        batch_size: int = 128,
        seed: int = 0,
    ) -> tuple[TrainingUpdateMetrics, ...]:
        if not samples:
            raise ValueError("Golden trainer requires at least one replay sample")
        if updates <= 0 or batch_size <= 0:
            raise ValueError("Golden trainer updates and batch_size must be positive")
        for sample in samples:
            sample.validate()
        observations = torch.tensor([sample.observation for sample in samples], dtype=torch.float32)
        policies = torch.tensor([sample.pi for sample in samples], dtype=torch.float32)
        values = torch.tensor([sample.z for sample in samples], dtype=torch.float32)
        model_device = next(self.model.parameters()).device
        observations = observations.to(model_device)
        policies = policies.to(model_device)
        values = values.to(model_device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.model.train()
        metrics: list[TrainingUpdateMetrics] = []
        for _ in range(updates):
            indices = torch.randint(0, len(samples), (batch_size,), generator=generator)
            batch_observations = observations[indices]
            batch_policies = policies[indices]
            batch_values = values[indices]
            policy_logits, value_logits = self.model(batch_observations)
            policy_loss = -(batch_policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = -(batch_values * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
            total_loss = policy_loss + value_loss
            self._ensure_finite(policy_loss, "policy loss")
            self._ensure_finite(value_loss, "value loss")
            self._ensure_finite(total_loss, "total loss")
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float("inf"))
            self._ensure_finite(torch.as_tensor(gradient_norm), "gradient norm")
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    self._ensure_finite(parameter.grad, "gradient")
            self.optimizer.step()
            for parameter in self.model.parameters():
                self._ensure_finite(parameter, "model parameter")
            self.update_count += 1
            self.samples_consumed += batch_size
            metrics.append(TrainingUpdateMetrics(
                update=self.update_count,
                policy_loss=float(policy_loss.detach()),
                value_loss=float(value_loss.detach()),
                total_loss=float(total_loss.detach()),
                gradient_norm=float(gradient_norm),
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                batch_size=batch_size,
                cumulative_samples=self.samples_consumed,
            ))
        return tuple(metrics)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_model_hash = model_hash(model)
    metadata_out = dict(metadata)
    metadata_out["model_hash"] = expected_model_hash
    metadata_out["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata_out,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    artifact_hash = file_sha256(destination)
    metadata_out["artifact_sha256"] = artifact_hash
    metadata_path = destination.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(_jsonable(metadata_out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_out


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected: Mapping[str, object] | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Golden checkpoint does not exist: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Golden checkpoint schema mismatch")
    metadata = dict(payload.get("metadata") or {})
    validate_checkpoint_metadata(metadata)
    if metadata.get("model_hash") != model_hash_from_state_dict(model, payload["model_state_dict"]):
        raise ValueError("Golden checkpoint model hash mismatch")
    if expected:
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Golden checkpoint metadata mismatch for {key}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        if payload.get("optimizer_state_dict") is None:
            raise ValueError("Golden checkpoint is missing optimizer state")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    artifact_hash = file_sha256(source)
    sidecar = source.with_suffix(".metadata.json")
    if not sidecar.is_file():
        raise ValueError("Golden checkpoint metadata sidecar is missing")
    sidecar_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if sidecar_metadata.get("artifact_sha256") != artifact_hash:
        raise ValueError("Golden checkpoint artifact hash mismatch")
    if sidecar_metadata.get("model_hash") != metadata.get("model_hash"):
        raise ValueError("Golden checkpoint sidecar model hash mismatch")
    metadata["artifact_sha256"] = artifact_hash
    if metadata.get("artifact_sha256") not in (None, artifact_hash):
        raise ValueError("Golden checkpoint artifact hash mismatch")
    return metadata


def model_hash_from_state_dict(model: nn.Module, state_dict: Mapping[str, Tensor]) -> str:
    clone = type(model)(topology=initial_state().topology, hidden=model.hidden, blocks=model.blocks_count)
    clone.load_state_dict(state_dict, strict=True)
    return model_hash(clone)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
