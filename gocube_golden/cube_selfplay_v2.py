"""Cube V2 scientific self-play adapter over the shared execution rails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from .cube_family import CubeFamilyTopology, cube_family_topology, initial_cube_state
from .cube_game_contract_v2 import (
    FORMAL_DOUBLE_PASS,
    concrete_game_fingerprint,
    concrete_game_identity,
    load_contract,
    action_index_to_rules_action,
    completion_is_formal_result,
    project_ownership,
    project_score,
    project_wdl,
    validate_cube_size,
)
from .cube_network_v2 import (
    ARCHITECTURE_FINGERPRINT,
    ARCHITECTURE_ID,
    CubeGraphNetV2,
    build_cube_model_from_metadata,
    cube_graphnet_v2_model_hash,
    cube_model_metadata,
    validate_cube_model_metadata,
)
from .cube_observation_v2 import (
    CHANNEL_COUNT,
    SCHEMA_FINGERPRINT,
    CubeObservationContext,
    concrete_observation_identity,
    initial_cube_observation_context,
    write_cube_observation,
    advance_cube_observation_context,
)
from .cube_search import CubeSearchAdapter, CubeSearchPosition
from .cube_selfplay_contract import (
    CUBE_SEMANTICS_ID,
    CUBE_SELFPLAY_CONTRACT_ID,
    CUBE_SELFPLAY_SEMANTICS_FINGERPRINT,
    CUBE_TARGET_CONTRACT_ID,
    CUBE_TARGET_FINGERPRINT,
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    CubeSelfPlaySearchContract,
    cube_action_index,
    sample_cube_action_from_visits,
)
from .inference import BatchedPolicyWDLInferenceOwner
from .provenance import CodeIdentity, capture_code_identity, derive_seed
from .rules import IllegalMoveError, LegalActionContext, apply_action, prepare_legal_actions
from .scoring import score_terminal
from .search import Evaluation, SearchError, SearchEvaluationRequest, SearchResult, SequentialPUCTSession
from .selfplay_engine import (
    CooperativeSelfPlayAdapter,
    GameFinished,
    InferenceClient,
    InferenceNeed,
    InferenceTransportError,
    SharedInferenceResult,
    SharedMemorySpec,
    SelfPlayEngineConfig,
    run_cooperative_selfplay,
)
from .selfplay_policy import RootDirichletNoiseTransform
from .state import GoldenState, PASS, Stone


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Stone):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cube_game_seed(master_seed: int, run_id: str, game_id: str) -> int:
    return derive_seed(int(master_seed), "cube-selfplay-game-v2", str(run_id), str(game_id))


@dataclass(frozen=True)
class CubeSelfPlayWorkerContext:
    run_id: str
    game_id_namespace: str
    size: int
    model_checkpoint_label: str
    checkpoint_artifact_hash: str
    master_seed: int
    profile_id: str
    profile_fingerprint: str
    game_contract_fingerprint: str
    observation_fingerprint: str
    model_architecture_id: str
    model_architecture_fingerprint: str
    code_identity: CodeIdentity
    contract: CubeSelfPlaySearchContract
    expected_model_hash: str


def _write_cube_shared_observation(payload: object, destination: Any) -> None:
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise InferenceTransportError("Cube shared input payload must be (search_position, legal_context)")
    position, legal_context = payload
    if not isinstance(position, CubeSearchPosition) or not isinstance(legal_context, LegalActionContext):
        raise InferenceTransportError("Cube shared input payload has an invalid scientific boundary")
    legal_context.assert_compatible(position.game_state)
    write_cube_observation(
        destination,
        position.game_state,
        position.observation_context,
        legal_context=legal_context,
    )


def _decode_cube_shared_output(policy: Any, wdl: Any) -> Evaluation:
    try:
        policy_values = tuple(float(value) for value in policy.tolist())
        wdl_values = tuple(float(value) for value in wdl.tolist())
    except (AttributeError, TypeError, ValueError) as exc:
        raise InferenceTransportError("Cube shared output is not tensor-like") from exc
    if len(policy_values) < 2 or len(wdl_values) != 3:
        raise InferenceTransportError("Cube shared output head shape drift")
    values = policy_values + wdl_values
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise InferenceTransportError("Cube shared output contains invalid probabilities")
    if not math.isclose(sum(policy_values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise InferenceTransportError("Cube shared policy output is not normalized")
    if not math.isclose(sum(wdl_values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise InferenceTransportError("Cube shared WDL output is not normalized")
    return Evaluation(policy=policy_values, wdl=wdl_values)


@dataclass(frozen=True)
class CubeSelfPlayPosition:
    """Compact per-ply record; state/history are reconstructed by replay."""

    ply: int
    side_to_move: str
    root_visits: tuple[int, ...]
    pi: tuple[float, ...]
    legal_action_mask: tuple[bool, ...]
    selected_action: int
    search_seed: int

    def validate(self, *, action_count: int, expected_simulations: int | None = None) -> None:
        if isinstance(self.ply, bool) or not isinstance(self.ply, int) or self.ply <= 0:
            raise ValueError("Cube self-play position ply is invalid")
        if self.side_to_move not in ("BLACK", "WHITE"):
            raise ValueError("Cube self-play position side-to-move is invalid")
        if len(self.root_visits) != action_count or len(self.pi) != action_count or len(self.legal_action_mask) != action_count:
            raise ValueError("Cube self-play position action shape drift")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.root_visits):
            raise ValueError("Cube root visits are invalid")
        total = sum(self.root_visits)
        if total <= 0 or (expected_simulations is not None and total != expected_simulations):
            raise ValueError("Cube root visits do not match the search contract")
        if isinstance(self.selected_action, bool) or not isinstance(self.selected_action, int) or not 0 <= self.selected_action < action_count:
            raise ValueError("Cube selected action is invalid")
        if not self.legal_action_mask[self.selected_action]:
            raise ValueError("Cube selected action is illegal")
        pi_total = sum(float(value) for value in self.pi)
        if not math.isfinite(pi_total) or not math.isclose(pi_total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Cube policy target is not normalized")
        for index, value in enumerate(self.pi):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("Cube policy target contains an invalid value")
            if not self.legal_action_mask[index] and float(value) != 0.0:
                raise ValueError("Illegal Cube policy target is non-zero")


@dataclass(frozen=True)
class CubeSelfPlayGameRecord:
    """One compact formal or technical Cube V2 game record."""

    schema_version: int
    run_id: str
    game_id: str
    size: int
    topology_id: str
    topology_fingerprint: str
    geometry_fingerprint: str
    game_contract_fingerprint: str
    rules_fingerprint: str
    komi: float
    observation_fingerprint: str
    model_architecture_id: str
    model_architecture_fingerprint: str
    model_hash: str
    checkpoint_artifact_hash: str
    selfplay_semantics_id: str
    selfplay_semantics_fingerprint: str
    search_config: Mapping[str, object]
    search_config_fingerprint: str
    target_contract_id: str
    target_contract_fingerprint: str
    master_seed: int
    game_seed: int
    initial_state: Mapping[str, object]
    positions: tuple[CubeSelfPlayPosition, ...]
    final_action_trace: tuple[int, ...]
    completion: str | None
    formal_result: str | None
    technical_termination: str | None
    error: str | None
    final_ownership: tuple[str, ...]
    black_area: int | None
    white_area: int | None
    neutral_points: int | None
    margin_black: float | None

    def validate(self, *, deep: bool = True) -> None:
        topology = cube_family_topology(validate_cube_size(self.size))
        if self.schema_version != 2:
            raise ValueError("Cube self-play record schema drift")
        if self.topology_id != topology.topology_id or self.topology_fingerprint != topology.fingerprint or self.geometry_fingerprint != topology.geometry_fingerprint:
            raise ValueError("Cube self-play topology identity drift")
        if self.observation_fingerprint != concrete_observation_identity(topology)["concrete_observation_fingerprint"]:
            raise ValueError("Cube self-play observation identity drift")
        if self.model_architecture_id != ARCHITECTURE_ID or self.model_architecture_fingerprint != ARCHITECTURE_FINGERPRINT:
            raise ValueError("Cube self-play model architecture identity drift")
        if self.selfplay_semantics_fingerprint != CUBE_SELFPLAY_SEMANTICS_FINGERPRINT:
            raise ValueError("Cube self-play semantics identity drift")
        if self.selfplay_semantics_id != CUBE_SEMANTICS_ID:
            raise ValueError("Cube self-play semantics id drift")
        if self.target_contract_id != CUBE_TARGET_CONTRACT_ID or self.target_contract_fingerprint != CUBE_TARGET_FINGERPRINT:
            raise ValueError("Cube self-play target contract identity drift")
        if self.search_config_fingerprint != _fingerprint(self.search_config):
            raise ValueError("Cube self-play search config fingerprint drift")
        if len(self.positions) != len(self.final_action_trace):
            raise ValueError("Cube self-play trace/position length drift")
        for position in self.positions:
            position.validate(
                action_count=topology.action_count,
                expected_simulations=int(self.search_config["simulations"]),
            )
        for action in self.final_action_trace:
            if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < topology.action_count:
                raise ValueError("Cube durable action trace must use point/PASS integer indexing")
        formal = self.completion == FORMAL_DOUBLE_PASS
        if formal != (self.formal_result is not None):
            raise ValueError("Cube formal completion/result mismatch")
        if formal and self.technical_termination is not None:
            raise ValueError("Cube formal record cannot have a technical termination")
        if not formal and self.formal_result is not None:
            raise ValueError("Cube technical record cannot have a formal result")
        if deep:
            _validate_replayed_record(self, topology)

    @property
    def action_trace(self) -> tuple[int, ...]:
        return self.final_action_trace

    @property
    def final_absolute_result(self) -> dict[str, object] | None:
        if self.formal_result is None:
            return None
        return {
            "winner": self.formal_result,
            "black_area": self.black_area,
            "white_area": self.white_area,
            "neutral_points": self.neutral_points,
            "margin_black": self.margin_black,
            "ownership": self.final_ownership,
        }

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def _validate_replayed_record(record: CubeSelfPlayGameRecord, topology: CubeFamilyTopology) -> None:
    from .cube_family import deserialize_cube_state

    state = deserialize_cube_state(record.initial_state)
    context = initial_cube_observation_context(state)
    adapter = CubeSearchAdapter()
    for position, action in zip(record.positions, record.final_action_trace):
        legal = prepare_legal_actions(state)
        expected_mask = tuple(bool(value) for value in legal.action_mask)
        if position.side_to_move != state.side_to_move.name or position.legal_action_mask != expected_mask:
            raise ValueError("Cube self-play position replay identity drift")
        if position.selected_action != action:
            raise ValueError("Cube selected action/trace drift")
        if not expected_mask[action]:
            raise ValueError("Cube trace contains an illegal action")
        rules_action = action_index_to_rules_action(action, record.size)
        state = apply_action(state, rules_action).after
        context = advance_cube_observation_context(context, action, state)
        if state.is_terminal:
            break
    if record.completion == FORMAL_DOUBLE_PASS:
        if not state.is_terminal or record.final_action_trace[-2:] != (topology.pass_action, topology.pass_action):
            raise ValueError("Cube formal record did not end with DOUBLE_PASS")
        if record.formal_result not in ("BLACK", "WHITE", "DRAW"):
            raise ValueError("Cube formal result is invalid")
        if len(record.final_ownership) != topology.point_count:
            raise ValueError("Cube final ownership shape drift")
        if record.black_area is None or record.white_area is None or record.neutral_points is None or record.margin_black is None:
            raise ValueError("Cube formal score fields are incomplete")
    elif record.final_ownership or any(value is not None for value in (record.black_area, record.white_area, record.neutral_points, record.margin_black)):
        raise ValueError("Cube technical record must not contain formal score targets")


class _CubeCooperativeGame:
    """Scientific game flow suspended at common inference requests."""

    def __init__(self, context: CubeSelfPlayWorkerContext, game_id: str, _client: InferenceClient) -> None:
        context.contract.validate()
        self.context = context
        self.game_id = str(game_id)
        self.game_seed = cube_game_seed(context.master_seed, context.run_id, self.game_id)
        self.rng = random.Random(self.game_seed)
        state = initial_cube_state(size=context.size, komi=context.contract.komi)
        self.position = CubeSearchPosition(state, initial_cube_observation_context(state))
        self.search_adapter = CubeSearchAdapter()
        self.start_state = self._initial_state_identity(state)
        self.positions: list[CubeSelfPlayPosition] = []
        self.trace: list[int] = []
        self.completion: str | None = None
        self.formal: str | None = None
        self.technical: str | None = None
        self.error: str | None = None
        self.final_ownership: tuple[str, ...] = ()
        self.black_area: int | None = None
        self.white_area: int | None = None
        self.neutral_points: int | None = None
        self.margin_black: float | None = None
        self._session: SequentialPUCTSession | None = None
        self._nn_evaluations = 0

    @staticmethod
    def _initial_state_identity(state: GoldenState) -> Mapping[str, object]:
        from .cube_family import serialize_cube_state

        return serialize_cube_state(state)

    def _finish(self) -> CubeSelfPlayGameRecord:
        record = CubeSelfPlayGameRecord(
            schema_version=2,
            run_id=self.context.run_id,
            game_id=self.game_id,
            size=self.context.size,
            topology_id=self.position.topology.topology_id,
            topology_fingerprint=self.position.topology.fingerprint,
            geometry_fingerprint=self.position.topology.geometry_fingerprint,
            game_contract_fingerprint=self.context.game_contract_fingerprint,
            rules_fingerprint=self.position.game_state.rules_fingerprint,
            komi=self.context.contract.komi,
            observation_fingerprint=self.context.observation_fingerprint,
            model_architecture_id=self.context.model_architecture_id,
            model_architecture_fingerprint=self.context.model_architecture_fingerprint,
            model_hash=self.context.expected_model_hash,
            checkpoint_artifact_hash=self.context.checkpoint_artifact_hash,
            selfplay_semantics_id=CUBE_SEMANTICS_ID,
            selfplay_semantics_fingerprint=CUBE_SELFPLAY_SEMANTICS_FINGERPRINT,
            search_config=self.context.contract.concrete_search_config_identity(),
            search_config_fingerprint=self.context.contract.fingerprint,
            target_contract_id=CUBE_TARGET_CONTRACT_ID,
            target_contract_fingerprint=CUBE_TARGET_FINGERPRINT,
            master_seed=self.context.master_seed,
            game_seed=self.game_seed,
            initial_state=self.start_state,
            positions=tuple(self.positions),
            final_action_trace=tuple(self.trace),
            completion=self.completion,
            formal_result=self.formal,
            technical_termination=self.technical,
            error=self.error,
            final_ownership=self.final_ownership,
            black_area=self.black_area,
            white_area=self.white_area,
            neutral_points=self.neutral_points,
            margin_black=self.margin_black,
        )
        record.validate(deep=False)
        return record

    def _mark_formal_terminal(self) -> None:
        score = score_terminal(self.position.game_state)
        self.completion = FORMAL_DOUBLE_PASS
        if score.margin_black > 0.0:
            self.formal = "BLACK"
        elif score.margin_black < 0.0:
            self.formal = "WHITE"
        else:
            self.formal = "DRAW"
        self.final_ownership = tuple(item.value for item in score.ownership)
        self.black_area = int(score.black_area)
        self.white_area = int(score.white_area)
        self.neutral_points = int(score.neutral_points)
        self.margin_black = float(score.margin_black)

    def _mark_technical(self, reason: str, error: str | None = None) -> None:
        self.completion = f"TECHNICAL_{reason}"
        self.technical = reason
        self.error = error

    def _start_search(self) -> None:
        search_seed = derive_seed(self.game_seed, len(self.trace) + 1, "search")
        transform = None
        if self.context.contract.root_noise:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(derive_seed(search_seed, "dirichlet")))
            transform = RootDirichletNoiseTransform(
                self.position,
                action_index=lambda action: self.search_adapter.action_index(self.position, action),
                epsilon=self.context.contract.dirichlet_epsilon,
                alpha=self.context.contract.dirichlet_alpha,
                generator=generator,
            )
        self._session = SequentialPUCTSession(
            self.position,
            self.context.contract.puct_settings,
            adapter=self.search_adapter,
            seed=search_seed,
            evaluation_transform=transform,
        )

    def advance(self) -> InferenceNeed | GameFinished:
        if self.completion is not None:
            return GameFinished(self._finish())
        try:
            while True:
                if self._session is None:
                    if len(self.trace) >= self.context.contract.technical_move_limit:
                        self._mark_technical("MOVE_LIMIT", "Cube technical move limit reached")
                        return GameFinished(self._finish())
                    self._start_search()
                step = self._session.advance()
                if isinstance(step, SearchEvaluationRequest):
                    self._nn_evaluations += 1
                    return InferenceNeed((step.state, step.legal_context))
                if not isinstance(step, SearchResult):
                    raise SearchError("Cube cooperative search returned malformed result")
                if step.simulations != self.context.contract.simulations or sum(step.root_visits) != self.context.contract.simulations:
                    raise SearchError("Cube root visits do not match the self-play contract")
                ply = len(self.trace) + 1
                internal_action = sample_cube_action_from_visits(
                    step,
                    temperature=1.0 if self.context.contract.temperature_plies[0] <= ply <= self.context.contract.temperature_plies[1] else self.context.contract.temperature_after,
                    rng=self.rng,
                    point_count=self.position.topology.point_count,
                )
                canonical_action = self.search_adapter.action_index(self.position, internal_action)
                self.positions.append(
                    CubeSelfPlayPosition(
                        ply=ply,
                        side_to_move=self.position.game_state.side_to_move.name,
                        root_visits=tuple(int(value) for value in step.root_visits),
                        pi=tuple(float(value) for value in step.pi),
                        legal_action_mask=tuple(bool(value) for value in step.legal_action_mask),
                        selected_action=canonical_action,
                        search_seed=derive_seed(self.game_seed, ply, "search"),
                    )
                )
                self.trace.append(canonical_action)
                self.position = self.search_adapter.apply_action(self.position, internal_action)
                self._session = None
                if self.position.is_terminal:
                    self._mark_formal_terminal()
                    return GameFinished(self._finish())
                if len(self.trace) >= self.context.contract.technical_move_limit:
                    self._mark_technical("MOVE_LIMIT", "Cube technical move limit reached")
                    return GameFinished(self._finish())
        except Exception as exc:
            self._session = None
            self._mark_technical("WORKER_ERROR", f"{type(exc).__name__}: {exc}")
            return GameFinished(self._finish())

    def resume(self, evaluation: Evaluation) -> None:
        if self.completion is not None:
            return
        if self._session is None:
            raise InferenceTransportError("Cube cooperative game is not waiting for inference")
        try:
            self._session.resume(evaluation)
        except Exception as exc:
            self._session = None
            self._mark_technical("WORKER_ERROR", f"{type(exc).__name__}: {exc}")


def _make_cube_cooperative_game(
    context: CubeSelfPlayWorkerContext,
    game_id: str,
    client: InferenceClient,
) -> _CubeCooperativeGame:
    return _CubeCooperativeGame(context, game_id, client)


def _cube_record_metrics(record: object) -> Mapping[str, object]:
    if not isinstance(record, CubeSelfPlayGameRecord):
        raise TypeError("Cube worker returned the wrong record type")
    return {
        "moves": len(record.final_action_trace),
        "technical": record.technical_termination is not None,
    }


def _model_hash_without_metadata(model: CubeGraphNetV2) -> str:
    digest = hashlib.sha256()
    digest.update(ARCHITECTURE_FINGERPRINT.encode("ascii"))
    digest.update(str(model.size).encode("ascii"))
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def cube_model_hash_v2(model: CubeGraphNetV2) -> str:
    if getattr(model, "model_metadata", None) is not None:
        return cube_graphnet_v2_model_hash(model)
    return _model_hash_without_metadata(model)


def _validate_cube_model(model: object, topology: CubeFamilyTopology) -> CubeGraphNetV2:
    if not isinstance(model, CubeGraphNetV2):
        raise TypeError("Cube self-play requires CubeGraphNetV2")
    expected = concrete_observation_identity(topology)
    checks = {
        "architecture_id": getattr(model, "architecture_id", None) == ARCHITECTURE_ID,
        "architecture_fingerprint": getattr(model, "architecture_fingerprint", None) == ARCHITECTURE_FINGERPRINT,
        "size": getattr(model, "size", None) == topology.size,
        "point_count": getattr(model, "point_count", None) == topology.point_count,
        "action_count": getattr(model, "action_count", None) == topology.action_count,
        "topology_id": getattr(model, "topology_id", None) == topology.topology_id,
        "topology_fingerprint": getattr(model, "game_graph_fingerprint", None) == topology.fingerprint,
        "geometry_fingerprint": getattr(model, "geometry_fingerprint", None) == topology.geometry_fingerprint,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, ok in checks.items() if not ok)
        raise ValueError(f"Cube model identity mismatch: {failed}")
    metadata = getattr(model, "model_metadata", None)
    if metadata is not None:
        validate_cube_model_metadata(metadata)
        if metadata.get("observation_schema_fingerprint") != SCHEMA_FINGERPRINT or metadata.get("concrete_observation_fingerprint") != expected["concrete_observation_fingerprint"]:
            raise ValueError("Cube model observation identity mismatch")
    return model


class CubeCentralInferenceOwner:
    """Thin Cube-specific model boundary over the shared inference owner."""

    def __init__(self, model: CubeGraphNetV2, *, topology: CubeFamilyTopology, device: str | torch.device) -> None:
        _validate_cube_model(model, topology)
        self.model = model
        self.topology = topology
        self.device = torch.device(device)
        self._owner = BatchedPolicyWDLInferenceOwner(
            model,
            device=self.device,
            expected_observation_shape=(CHANNEL_COUNT, topology.point_count),
            expected_policy_size=topology.action_count,
            wdl_size=3,
            forward_policy_wdl_logits=lambda batch: model.infer_policy_wdl(batch),
        )

    def evaluate_shared_batch(self, observations: Any) -> SharedInferenceResult:
        return self._owner.evaluate_shared_batch(observations)


class CubeSelfPlayAdapter:
    """Scientific/profile adapter consumed by the generic cooperative runner."""

    def __init__(
        self,
        model: CubeGraphNetV2,
        *,
        size: int | None = None,
        run_id: str = "cube-stage5",
        model_checkpoint_label: str = "unpublished",
        checkpoint_artifact_hash: str = "",
        master_seed: int = 0,
        profile_id: str | None = None,
        profile_fingerprint: str | None = None,
        code_identity: CodeIdentity | None = None,
        contract: CubeSelfPlaySearchContract = DEFAULT_CUBE_SELFPLAY_CONTRACT,
        device: str | torch.device = "cpu",
    ) -> None:
        resolved_size = validate_cube_size(getattr(model, "size", size) if size is None else size)
        topology = cube_family_topology(resolved_size)
        _validate_cube_model(model, topology)
        contract.validate()
        game_contract = load_contract()
        game_identity = concrete_game_identity(
            game_contract,
            resolved_size,
            topology_id=topology.topology_id,
            topology_fingerprint=topology.fingerprint,
            rules_fingerprint=initial_cube_state(size=resolved_size, komi=contract.komi).rules_fingerprint,
            komi=contract.komi,
        )
        expected_profile_fingerprint = concrete_game_fingerprint(game_identity)
        if profile_fingerprint is not None and str(profile_fingerprint) != expected_profile_fingerprint:
            raise ValueError("Cube self-play profile/game identity fingerprint drift")
        self.model = model
        self.size = resolved_size
        self.topology = topology
        self.device = torch.device(device)
        self.owner = CubeCentralInferenceOwner(model, topology=topology, device=self.device)
        self.worker_context = CubeSelfPlayWorkerContext(
            run_id=str(run_id),
            game_id_namespace=str(run_id),
            size=resolved_size,
            model_checkpoint_label=str(model_checkpoint_label),
            checkpoint_artifact_hash=str(checkpoint_artifact_hash),
            master_seed=int(master_seed),
            profile_id=str(profile_id or f"cube{resolved_size}-v2"),
            profile_fingerprint=expected_profile_fingerprint,
            game_contract_fingerprint=str(game_identity["family_contract_fingerprint"]),
            observation_fingerprint=str(concrete_observation_identity(topology)["concrete_observation_fingerprint"]),
            model_architecture_id=ARCHITECTURE_ID,
            model_architecture_fingerprint=ARCHITECTURE_FINGERPRINT,
            code_identity=code_identity or capture_code_identity(),
            contract=contract,
            expected_model_hash=cube_model_hash_v2(model),
        )
        self.shared_memory = SharedMemorySpec(
            observation_shape=(CHANNEL_COUNT, topology.point_count),
            policy_size=topology.action_count,
            wdl_size=3,
            write_input=_write_cube_shared_observation,
            decode_output=_decode_cube_shared_output,
        )

    @property
    def infer_shared_batch(self):
        return self.owner.evaluate_shared_batch

    @property
    def worker_game_factory(self):
        return _make_cube_cooperative_game

    @property
    def record_metrics(self):
        return _cube_record_metrics


@dataclass(frozen=True)
class CubeSelfPlayExecutionConfig:
    """Execution-only knobs kept outside the scientific contract."""

    workers: int = 1
    active_games_per_worker: int = 1
    total_active_contexts: int | None = 1
    inference_batch_cap: int = 1
    inference_batch_wait_ms: float = 0.0
    device: str = "cpu"
    process_start_method: str | None = None
    inference_request_timeout_s: float = 300.0
    worker_join_timeout_s: float = 15.0

    def validate(self) -> None:
        if self.workers <= 0 or self.active_games_per_worker <= 0:
            raise ValueError("Cube execution workers and active games must be positive")
        if self.total_active_contexts is not None and self.total_active_contexts <= 0:
            raise ValueError("Cube execution active-context cap must be positive")
        if self.inference_batch_cap <= 0 or self.inference_batch_wait_ms < 0.0:
            raise ValueError("Cube execution batching settings are invalid")

    def to_engine_config(self) -> SelfPlayEngineConfig:
        self.validate()
        start_method = self.process_start_method or ("spawn" if torch.device(self.device).type == "cuda" else "fork")
        return SelfPlayEngineConfig(
            workers=int(self.workers),
            inference_batch_cap=int(self.inference_batch_cap),
            inference_batch_wait_ms=float(self.inference_batch_wait_ms),
            device=str(torch.device(self.device)),
            process_start_method=start_method,
            inference_request_timeout_s=float(self.inference_request_timeout_s),
            worker_join_timeout_s=float(self.worker_join_timeout_s),
            lanes_per_worker=1,
            active_games_per_worker=int(self.active_games_per_worker),
            total_active_contexts=self.total_active_contexts,
        )


def run_cube_selfplay_games(
    model: CubeGraphNetV2,
    game_ids: Sequence[str],
    *,
    size: int | None = None,
    run_id: str = "cube-stage5",
    model_checkpoint_label: str = "unpublished",
    checkpoint_artifact_hash: str = "",
    master_seed: int = 0,
    profile_id: str | None = None,
    profile_fingerprint: str | None = None,
    code_identity: CodeIdentity | None = None,
    contract: CubeSelfPlaySearchContract = DEFAULT_CUBE_SELFPLAY_CONTRACT,
    device: str | torch.device = "cpu",
    execution_config: CubeSelfPlayExecutionConfig | None = None,
    workers: int | None = None,
    active_games_per_worker: int | None = None,
    total_active_contexts: int | None = None,
    inference_batch_cap: int | None = None,
    inference_batch_wait_ms: float | None = None,
    inference_telemetry: MutableMapping[str, object] | None = None,
    execution_activity: MutableMapping[str, object] | None = None,
    progress_callback: Any | None = None,
) -> tuple[CubeSelfPlayGameRecord, ...]:
    adapter = CubeSelfPlayAdapter(
        model,
        size=size,
        run_id=run_id,
        model_checkpoint_label=model_checkpoint_label,
        checkpoint_artifact_hash=checkpoint_artifact_hash,
        master_seed=master_seed,
        profile_id=profile_id,
        profile_fingerprint=profile_fingerprint,
        code_identity=code_identity,
        contract=contract,
        device=device,
    )
    if execution_config is None:
        execution_config = CubeSelfPlayExecutionConfig(
            workers=1 if workers is None else int(workers),
            active_games_per_worker=1 if active_games_per_worker is None else int(active_games_per_worker),
            total_active_contexts=total_active_contexts if total_active_contexts is not None else (1 if active_games_per_worker is None else int(active_games_per_worker)),
            inference_batch_cap=1 if inference_batch_cap is None else int(inference_batch_cap),
            inference_batch_wait_ms=0.0 if inference_batch_wait_ms is None else float(inference_batch_wait_ms),
            device=str(torch.device(device)),
        )
    result = run_cooperative_selfplay(
        game_ids,
        adapter=adapter,
        engine_config=execution_config.to_engine_config(),
        telemetry=inference_telemetry,
        progress_callback=progress_callback,
        active_games_per_worker=execution_config.active_games_per_worker,
        total_active_contexts=execution_config.total_active_contexts,
    )
    if execution_activity is not None:
        execution_activity.update(result.telemetry)
    typed: list[CubeSelfPlayGameRecord] = []
    for record in result.records:
        if not isinstance(record, CubeSelfPlayGameRecord):
            raise RuntimeError("Cube self-play engine returned the wrong record type")
        record.validate(deep=False)
        typed.append(record)
    return tuple(typed)


run_cube_selfplay_games_shared = run_cube_selfplay_games


__all__ = [
    "CubeCentralInferenceOwner",
    "CubeSelfPlayAdapter",
    "CubeSelfPlayExecutionConfig",
    "CubeSelfPlayGameRecord",
    "CubeSelfPlayPosition",
    "CubeSelfPlaySearchContract",
    "CubeSelfPlayWorkerContext",
    "cube_game_seed",
    "cube_model_hash_v2",
    "run_cube_selfplay_games",
    "run_cube_selfplay_games_shared",
]
