"""Cube scientific adapter for the universal :mod:`selfplay_engine`.

The Cube profile owns rules, observations, root-noise, search seeds and game
records.  The repository-level engine owns process scheduling, shared-memory
transport, central batching and fail-closed lifecycle handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import random
import time
from typing import Any, Mapping, MutableMapping, Sequence

torch = importlib.import_module("torch")

from .cube_neural import (
    CUBE_ACTION_COUNT,
    CUBE_OBSERVATION_CHANNEL_COUNT,
    CUBE_POINT_COUNT,
    CUBE_OBSERVATION_FINGERPRINT,
    GoldenCubeGraphNetV1,
    apply_cube_root_dirichlet_noise,
    build_cube_observation_into,
    configure_single_thread_inference,
    cube_model_hash,
)
from .cube_contract import CUBE_PROFILE_ID, load_profile
from .cube_topology import CUBE4_TOPOLOGY
from .cube_training import (
    CUBE_SELFPLAY_CONTRACT_ID,
    CUBE_TARGET_FINGERPRINT,
    CUBE_WATCHDOG,
    CubeSelfPlayGameRecord,
    CubeSelfPlayPosition,
    CubeSelfPlaySearchContract,
    DEFAULT_CUBE_SELFPLAY_CONTRACT,
    cube_initial_state,
    cube_post_action_termination,
    cube_state_identity,
    sample_cube_action_from_visits,
)
from .provenance import CodeIdentity, capture_code_identity, derive_seed
from .result import result_from_terminal
from .rules import IllegalMoveError, apply_action
from .search import Evaluation, SearchEvaluationRequest, SearchError, SearchResult, SequentialPUCTSession
from .search_adapter import GoldenSearchAdapter
from .selfplay_engine import (
    GameFinished,
    InferenceClient,
    InferenceNeed,
    InferenceTransportError,
    SharedInferenceResult,
    SharedMemorySpec,
    SelfPlayEngine,
    SelfPlayEngineConfig,
    SelfPlayEngineError,
)
from .state import PASS, GoldenState


@dataclass(frozen=True)
class CubeSelfPlayExecutionConfig:
    """Execution-only knobs; none of these enter the Cube profile identity."""

    workers: int = 16
    active_games_per_worker: int = 4
    total_active_contexts: int | None = 64
    inference_batch_cap: int = 64
    inference_batch_wait_ms: float = 1.0
    device: str = "cpu"
    process_start_method: str | None = None
    inference_request_timeout_s: float = 300.0
    worker_join_timeout_s: float = 15.0

    def validate(self) -> None:
        if self.workers <= 0 or self.active_games_per_worker <= 0:
            raise ValueError("Cube self-play execution workers and active games must be positive")
        if self.total_active_contexts is not None and self.total_active_contexts <= 0:
            raise ValueError("Cube self-play total active contexts must be positive")
        if self.inference_batch_cap <= 0 or self.inference_batch_wait_ms < 0:
            raise ValueError("Cube self-play inference settings are invalid")


@dataclass(frozen=True)
class CubeSelfPlayWorkerContext:
    run_id: str
    model_checkpoint_label: str
    checkpoint_artifact_hash: str
    master_seed: int
    chunk_id: str
    profile_id: str
    profile_fingerprint: str
    code_identity: CodeIdentity
    contract: CubeSelfPlaySearchContract
    allow_noncanonical_contract: bool
    expected_model_hash: str


def _write_cube_shared_observation(payload: object, destination: Any) -> None:
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise InferenceTransportError("Cube shared input payload must be (state, legal_context)")
    state, legal_context = payload
    if not isinstance(state, GoldenState):
        raise InferenceTransportError("Cube shared input payload contains an invalid state")
    build_cube_observation_into(state, destination, legal_context=legal_context)


def _decode_cube_shared_output(policy: Any, wdl: Any) -> Evaluation:
    policy_values = tuple(float(value) for value in policy.tolist())
    wdl_values = tuple(float(value) for value in wdl.tolist())
    if len(policy_values) != CUBE_ACTION_COUNT or len(wdl_values) != 3:
        raise InferenceTransportError("Cube shared output head shape drift")
    values = policy_values + wdl_values
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise InferenceTransportError("Cube shared output contains invalid probabilities")
    if not math.isclose(sum(policy_values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise InferenceTransportError("Cube shared policy output is not normalized")
    if not math.isclose(sum(wdl_values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise InferenceTransportError("Cube shared WDL output is not normalized")
    return Evaluation(policy=policy_values, wdl=wdl_values)


class _CubeRootNoiseTransform:
    """Worker-side transform identical to ``SelfPlayCubeRootNoiseEvaluator``."""

    def __init__(self, root_state: GoldenState, *, seed: int, epsilon: float, alpha: float) -> None:
        if not 0.0 <= epsilon <= 1.0 or alpha <= 0.0:
            raise ValueError("Invalid Cube self-play Dirichlet parameters")
        self.root_state_key = root_state.state_key
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))

    def __call__(self, base: Evaluation, state: GoldenState, legal_context: Any) -> Evaluation:
        if state.state_key != self.root_state_key:
            return base
        legal_context.assert_compatible(state)
        legal = legal_context.actions
        if not legal:
            raise SearchError("Cube self-play root has no legal actions")
        try:
            policy = apply_cube_root_dirichlet_noise(
                tuple(float(value) for value in base.policy),
                legal,
                epsilon=self.epsilon,
                alpha=self.alpha,
                generator=self._generator,
            )
        except ValueError as exc:
            raise SearchError(str(exc)) from exc
        return Evaluation(policy=policy, wdl=base.wdl)


class _CubeCooperativeGame:
    """One sequential Cube PUCT tree suspended at neural evaluations."""

    def __init__(self, context: CubeSelfPlayWorkerContext, game_id: str, _client: InferenceClient) -> None:
        context.contract.validate(canonical=not context.allow_noncanonical_contract)
        configure_single_thread_inference()
        self.context = context
        self.game_id = str(game_id)
        self.game_seed = derive_seed(
            context.master_seed,
            "cube-selfplay-game-v1",
            context.chunk_id,
            self.game_id,
        )
        self.rng = random.Random(self.game_seed)
        self.state = cube_initial_state(komi=context.contract.komi)
        self.start_state = cube_state_identity(self.state)
        self.positions: list[CubeSelfPlayPosition] = []
        self.trace: list[int | str] = []
        self.formal: str | None = None
        self.technical: str | None = None
        self.error: str | None = None
        self._session: SequentialPUCTSession | None = None
        self._nn_evaluations = 0

    def _finish(self) -> CubeSelfPlayGameRecord:
        record = CubeSelfPlayGameRecord(
            run_id=self.context.run_id,
            game_id=self.game_id,
            chunk_id=self.context.chunk_id,
            profile_id=self.context.profile_id,
            profile_fingerprint=self.context.profile_fingerprint,
            selfplay_contract_id=self.context.contract.contract_id,
            selfplay_contract_fingerprint=self.context.contract.fingerprint,
            topology_fingerprint=CUBE4_TOPOLOGY.fingerprint,
            geometry_fingerprint=CUBE4_TOPOLOGY.geometry_fingerprint,
            rules_fingerprint=self.state.rules_fingerprint,
            komi=self.context.contract.komi,
            observation_fingerprint=CUBE_OBSERVATION_FINGERPRINT,
            target_fingerprint=CUBE_TARGET_FINGERPRINT,
            model_checkpoint_label=self.context.model_checkpoint_label,
            model_hash=self.context.expected_model_hash,
            checkpoint_artifact_hash=self.context.checkpoint_artifact_hash,
            git_commit=self.context.code_identity.git_commit_sha,
            git_tree=self.context.code_identity.git_tree_sha,
            git_worktree_clean=self.context.code_identity.working_tree_clean,
            master_seed=self.context.master_seed,
            game_seed=self.game_seed,
            start_state=self.start_state,
            positions=tuple(self.positions),
            final_action_trace=tuple(self.trace),
            formal_result=self.formal,
            technical_termination=self.technical,
            error=self.error,
            nn_evaluations=self._nn_evaluations,
        )
        record.validate(deep=False)
        return record

    def _start_search(self) -> None:
        search_seed = derive_seed(self.game_seed, len(self.trace) + 1, "search")
        transform = _CubeRootNoiseTransform(
            self.state,
            seed=derive_seed(search_seed, "dirichlet"),
            epsilon=self.context.contract.dirichlet_epsilon,
            alpha=self.context.contract.dirichlet_alpha,
        )
        self._session = SequentialPUCTSession(
            self.state,
            self.context.contract.puct_settings,
            adapter=GoldenSearchAdapter(),
            seed=search_seed,
            evaluation_transform=transform,
        )

    def advance(self) -> InferenceNeed | GameFinished:
        if self.technical is not None or self.formal is not None:
            return GameFinished(self._finish())
        try:
            while True:
                if self._session is None:
                    if len(self.trace) >= self.context.contract.watchdog:
                        self.technical = "TRUNCATED_MOVE_LIMIT"
                        self.error = f"Cube watchdog reached {self.context.contract.watchdog} actions"
                        return GameFinished(self._finish())
                    self._start_search()
                step = self._session.advance()
                if isinstance(step, SearchEvaluationRequest):
                    self._nn_evaluations += 1
                    return InferenceNeed((step.state, step.legal_context))
                if not isinstance(step, SearchResult):
                    raise RuntimeError("Cube cooperative search returned malformed result")
                if step.simulations != self.context.contract.simulations or sum(step.root_visits) != self.context.contract.simulations:
                    raise SearchError("Cube root visits do not match the self-play simulation contract")
                ply = len(self.trace) + 1
                search_seed = derive_seed(self.game_seed, ply, "search")
                action = sample_cube_action_from_visits(
                    step,
                    temperature=1.0 if ply <= self.context.contract.temperature_plies[1] else self.context.contract.temperature_after,
                    rng=self.rng,
                )
                self.positions.append(
                    CubeSelfPlayPosition(
                        ply=ply,
                        state=cube_state_identity(self.state),
                        side_to_move=self.state.side_to_move.name,
                        legal_action_mask=tuple(step.legal_action_mask),
                        root_visits=tuple(int(value) for value in step.root_visits),
                        pi=tuple(float(value) for value in step.pi),
                        selected_action=action,
                        search_seed=search_seed,
                        model_hash=self.context.expected_model_hash,
                    )
                )
                self.trace.append(action)
                try:
                    self.state = apply_action(self.state, action).after
                except IllegalMoveError as exc:
                    self.technical = "ERROR_ILLEGAL_PLAYER_ACTION"
                    self.error = f"{type(exc).__name__}: {exc}"
                    return GameFinished(self._finish())
                self._session = None
                termination = cube_post_action_termination(self.state, ply)
                if termination == "DOUBLE_PASS":
                    self.formal = result_from_terminal(self.state).winner.value
                    return GameFinished(self._finish())
                if termination == "TRUNCATED_MOVE_LIMIT":
                    self.technical = "TRUNCATED_MOVE_LIMIT"
                    self.error = f"Cube watchdog reached {self.context.contract.watchdog} actions"
                    return GameFinished(self._finish())
        except Exception as exc:
            self.technical = "ERROR_SEARCH"
            self.error = f"{type(exc).__name__}: {exc}"
            self._session = None
            return GameFinished(self._finish())

    def resume(self, evaluation: Evaluation) -> None:
        if self.technical is not None or self.formal is not None:
            return
        if self._session is None:
            raise InferenceTransportError("Cube cooperative game is not waiting for inference")
        try:
            self._session.resume(evaluation)
        except Exception as exc:
            self.technical = "ERROR_SEARCH"
            self.error = f"{type(exc).__name__}: {exc}"
            self._session = None


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


class CubeCentralInferenceOwner:
    """The sole parent-side Cube model owner for a self-play generation."""

    def __init__(self, model: GoldenCubeGraphNetV1, *, device: str | torch.device = "cpu") -> None:
        if not isinstance(model, GoldenCubeGraphNetV1):
            raise TypeError("Cube central inference requires GoldenCubeGraphNetV1")
        if model.topology_fingerprint != CUBE4_TOPOLOGY.fingerprint:
            raise ValueError("Cube central inference received the wrong topology model")
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, observations: Any) -> tuple[Any, Any, float, float, float, float]:
        if tuple(observations.shape[1:]) != (CUBE_OBSERVATION_CHANNEL_COUNT, CUBE_POINT_COUNT):
            raise ValueError("Cube shared observations must have shape [batch,15,96]")
        h2d_started = time.perf_counter()
        device_observations = observations.to(self.device, non_blocking=self.device.type == "cuda")
        h2d_finished = time.perf_counter()
        forward_started = time.perf_counter()
        with torch.inference_mode():
            policy_logits, value_logits = self.model(device_observations)
            policies = torch.softmax(policy_logits, dim=1)
            wdls = torch.softmax(value_logits, dim=1)
        forward_finished = time.perf_counter()
        rows = int(observations.shape[0])
        if tuple(policies.shape) != (rows, CUBE_ACTION_COUNT) or tuple(wdls.shape) != (rows, 3):
            raise ValueError("Cube central policy/value head shape drift")
        outputs = torch.cat((policies, wdls), dim=1)
        if not bool(torch.isfinite(outputs).all()) or bool((outputs < 0.0).any()):
            raise ValueError("Cube central inference produced invalid probabilities")
        return policies, wdls, h2d_started, h2d_finished, forward_started, forward_finished

    def evaluate_shared_batch(self, observations: Any) -> SharedInferenceResult:
        policies, wdls, h2d_started, h2d_finished, forward_started, forward_finished = self._forward(observations)
        return SharedInferenceResult(
            policy=policies,
            wdl=wdls,
            h2d_started_at=h2d_started,
            h2d_finished_at=h2d_finished,
            forward_started_at=forward_started,
            forward_finished_at=forward_finished,
        )

class CubeSelfPlayAdapter:
    """Cube profile boundary consumed by the generic ``SelfPlayEngine``."""

    def __init__(
        self,
        model: GoldenCubeGraphNetV1,
        *,
        run_id: str,
        model_checkpoint_label: str,
        checkpoint_artifact_hash: str,
        master_seed: int,
        chunk_id: str,
        profile_id: str,
        profile_fingerprint: str,
        code_identity: CodeIdentity | None = None,
        contract: CubeSelfPlaySearchContract = DEFAULT_CUBE_SELFPLAY_CONTRACT,
        device: str | torch.device = "cpu",
        allow_noncanonical_contract: bool = False,
    ) -> None:
        if profile_id != CUBE_PROFILE_ID:
            raise ValueError("Cube self-play supports only the current Golden profile")
        current_profile = load_profile()
        if profile_fingerprint != current_profile.get("profile_fingerprint"):
            raise ValueError("Cube self-play profile fingerprint drift")
        resolved_code = code_identity or capture_code_identity()
        contract.validate(canonical=not allow_noncanonical_contract)
        if contract.komi != 0.5:
            raise ValueError("Cube self-play requires komi 0.5")
        self.model = model
        self.device = torch.device(device)
        self.owner = CubeCentralInferenceOwner(model, device=device)
        self.worker_context = CubeSelfPlayWorkerContext(
            run_id=str(run_id),
            model_checkpoint_label=str(model_checkpoint_label),
            checkpoint_artifact_hash=str(checkpoint_artifact_hash),
            master_seed=int(master_seed),
            chunk_id=str(chunk_id),
            profile_id=str(profile_id),
            profile_fingerprint=str(profile_fingerprint),
            code_identity=resolved_code,
            contract=contract,
            allow_noncanonical_contract=allow_noncanonical_contract,
            expected_model_hash=cube_model_hash(model),
        )
        self.shared_memory = SharedMemorySpec(
            observation_shape=(CUBE_OBSERVATION_CHANNEL_COUNT, CUBE_POINT_COUNT),
            policy_size=CUBE_ACTION_COUNT,
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


def run_cube_selfplay_games_shared(
    model: GoldenCubeGraphNetV1,
    game_ids: Sequence[str],
    *,
    run_id: str,
    profile_id: str,
    profile_fingerprint: str,
    model_checkpoint_label: str,
    checkpoint_artifact_hash: str,
    master_seed: int,
    chunk_id: str,
    code_identity: CodeIdentity | None = None,
    workers: int = 16,
    active_games_per_worker: int = 4,
    total_active_contexts: int | None = 64,
    inference_batch_cap: int = 64,
    inference_batch_wait_ms: float = 1.0,
    device: str | torch.device = "cpu",
    contract: CubeSelfPlaySearchContract = DEFAULT_CUBE_SELFPLAY_CONTRACT,
    allow_noncanonical_contract: bool = False,
    inference_telemetry: MutableMapping[str, object] | None = None,
    execution_activity: MutableMapping[str, object] | None = None,
) -> tuple[CubeSelfPlayGameRecord, ...]:
    execution = CubeSelfPlayExecutionConfig(
        workers=int(workers),
        active_games_per_worker=int(active_games_per_worker),
        total_active_contexts=total_active_contexts,
        inference_batch_cap=int(inference_batch_cap),
        inference_batch_wait_ms=float(inference_batch_wait_ms),
        device=str(torch.device(device)),
    )
    execution.validate()
    ordered = tuple(sorted(str(game_id) for game_id in game_ids))
    if len(ordered) != len(set(ordered)):
        raise ValueError("Cube self-play game IDs must be unique")
    code = code_identity or capture_code_identity()
    adapter = CubeSelfPlayAdapter(
        model,
        run_id=run_id,
        model_checkpoint_label=model_checkpoint_label,
        checkpoint_artifact_hash=checkpoint_artifact_hash,
        master_seed=master_seed,
        chunk_id=chunk_id,
        profile_id=profile_id,
        profile_fingerprint=profile_fingerprint,
        code_identity=code,
        contract=contract,
        device=device,
        allow_noncanonical_contract=allow_noncanonical_contract,
    )
    start_method = "spawn" if torch.device(device).type == "cuda" else "fork"
    engine = SelfPlayEngine(
        SelfPlayEngineConfig(
            workers=execution.workers,
            inference_batch_cap=execution.inference_batch_cap,
            inference_batch_wait_ms=execution.inference_batch_wait_ms,
            device=execution.device,
            process_start_method=start_method,
            lanes_per_worker=1,
            active_games_per_worker=execution.active_games_per_worker,
            total_active_contexts=execution.total_active_contexts,
            inference_request_timeout_s=execution.inference_request_timeout_s,
            worker_join_timeout_s=execution.worker_join_timeout_s,
        )
    )
    telemetry: dict[str, object] = {}
    records = engine.run(
        ordered,
        worker_play=None,
        worker_context=adapter.worker_context,
        infer_batch=None,
        record_metrics=adapter.record_metrics,
        telemetry=telemetry,
        shared_memory=adapter.shared_memory,
        infer_shared_batch=adapter.infer_shared_batch,
        worker_game_factory=adapter.worker_game_factory,
        active_games_per_worker=execution.active_games_per_worker,
        total_active_contexts=execution.total_active_contexts,
    )
    typed: list[CubeSelfPlayGameRecord] = []
    for record in records:
        if not isinstance(record, CubeSelfPlayGameRecord):
            raise SelfPlayEngineError("Cube engine returned the wrong record type")
        record.validate(deep=False)
        typed.append(record)
    if inference_telemetry is not None:
        inference_telemetry.update(telemetry)
        inference_telemetry.update({
            "mode": "central-process-batched",
            "batch_cap": telemetry["inference_batch_cap"],
            "wait_ms": telemetry["inference_batch_wait_ms"],
            "forward_calls": telemetry["inference_forwards"],
            "total_rows": telemetry["inference_rows"],
            "batch_rows": telemetry["batch_rows"],
            "mean_batch_rows": telemetry["mean_inference_batch_rows"],
            "p50_batch_rows": telemetry["p50_inference_batch_rows"],
            "p95_batch_rows": telemetry["p95_inference_batch_rows"],
            "max_batch_rows": telemetry["max_inference_batch_rows"],
            "rows_per_sec": telemetry["inference_rows_per_sec"],
            "forwards_per_sec": telemetry["inference_forwards_per_sec"],
            "device": telemetry["inference_device"],
            "central_inference_owner_pid": telemetry["central_inference_owner_pid"],
            "max_concurrent_forwards": 1 if int(telemetry["inference_forwards"]) else 0,
            "lock_scope": "single_parent_inference_owner",
        })
    if execution_activity is not None:
        execution_activity.update(telemetry)
    return tuple(typed)


__all__ = [
    "CubeCentralInferenceOwner",
    "CubeSelfPlayAdapter",
    "CubeSelfPlayExecutionConfig",
    "CubeSelfPlayWorkerContext",
    "run_cube_selfplay_games_shared",
]
