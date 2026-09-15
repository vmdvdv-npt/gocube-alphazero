"""Torus9 scientific adapter for the universal process SelfPlayEngine.

The adapter owns Torus9 observation/model/search/game-record semantics. The
engine owns only execution; retired training paths are not used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Mapping, MutableMapping, Sequence

from .neural import model_hash
from .provenance import CodeIdentity, capture_code_identity, derive_seed
from .search import Evaluation
from .selfplay_engine import (
    GameFinished,
    InferenceNeed,
    InferenceClient,
    InferenceTransportError,
    SharedInferenceResult,
    SharedMemorySpec,
    SelfPlayEngine,
    SelfPlayEngineConfig,
)
from .torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_BLOCKS,
    TORUS9_CURRENT_BLOCKS,
    TORUS9_CURRENT_DIRICHLET_ALPHA,
    TORUS9_CURRENT_HIDDEN,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_SELFPLAY_CONTRACT_ID,
    TORUS9_HIDDEN,
    TORUS9_KOMI,
    TORUS9_PROFILE_ID,
    TORUS9_WORKERS,
    current_torus9_profile_fingerprint,
    load_torus9_current_profile,
)
from . import torus9_monolith as _t9
from .search import SearchEvaluationRequest, SequentialPUCTSession


torch = _t9.torch


@dataclass(frozen=True)
class Torus9SelfPlayWorkerContext:
    run_id: str
    model_checkpoint_label: str
    checkpoint_artifact_hash: str
    master_seed: int
    profile_fingerprint: str
    code_identity: CodeIdentity
    contract: _t9.Torus9SelfPlaySearchContract
    profile_id: str
    expected_model_hash: str


def _write_torus9_shared_observation(payload: object, destination: Any) -> None:
    """Adapter codec: write one prepared scientific observation into a slot."""
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise InferenceTransportError("Torus9 shared input payload must be (state, legal_context)")
    state, legal_context = payload
    _t9.build_torus9_observation_into(state, destination, legal_context=legal_context)


def _decode_torus9_shared_output(policy: Any, wdl: Any) -> Evaluation:
    policy_values = tuple(float(value) for value in policy.tolist())
    wdl_values = tuple(float(value) for value in wdl.tolist())
    if len(policy_values) != TORUS9_ACTION_COUNT or len(wdl_values) != 3:
        raise InferenceTransportError("Torus9 shared output head shape drift")
    if any(not math.isfinite(value) or value < 0.0 for value in policy_values + wdl_values):
        raise InferenceTransportError("Torus9 shared output contains invalid probabilities")
    return Evaluation(policy=policy_values, wdl=wdl_values)


class _Torus9CooperativeGame:
    """One Torus9 game stepped between neural evaluations.

    The session is the canonical Golden PUCT implementation.  This object
    only turns its evaluation suspension points into the generic execution
    engine's ``InferenceNeed`` messages and keeps the existing record logic.
    """

    def __init__(self, context: Torus9SelfPlayWorkerContext, game_id: str, _client: InferenceClient) -> None:
        context.contract.validate()
        self.context = context
        self.game_id = str(game_id)
        self.game_seed = derive_seed(context.master_seed, context.run_id, self.game_id, "game")
        self.rng = __import__("random").Random(self.game_seed)
        self.state = _t9.initial_state(topology=_t9.TORUS_9X9, komi=TORUS9_KOMI)
        self.start_state = _t9.torus9_state_identity(self.state)
        self.positions: list[_t9.Torus9SelfPlayPosition] = []
        self.trace: list[int | str] = []
        self.formal: str | None = None
        self.technical: str | None = None
        self.error: str | None = None
        self._session: SequentialPUCTSession | None = None
        self._root_noise: _t9.Torus9RootNoiseEvaluator | None = None
        self._nn_evaluations = 0

    def _finish(self) -> _t9.Torus9SelfPlayGameRecord:
        record = _t9.Torus9SelfPlayGameRecord(
            run_id=self.context.run_id,
            game_id=self.game_id,
            profile_id=self.context.profile_id,
            profile_fingerprint=self.context.profile_fingerprint,
            selfplay_contract_id=self.context.contract.contract_id,
            selfplay_contract_fingerprint=self.context.contract.fingerprint,
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
        record.validate()
        return record

    def _start_search(self) -> None:
        search_seed = derive_seed(self.game_seed, len(self.trace) + 1, "search")
        self._root_noise = _t9.Torus9RootNoiseEvaluator(
            None,
            self.state,
            seed=derive_seed(search_seed, "dirichlet"),
            alpha=self.context.contract.dirichlet_alpha,
        )
        self._session = SequentialPUCTSession(
            self.state,
            self.context.contract.settings,
            adapter=_t9.GoldenSearchAdapter(),
            seed=search_seed,
            evaluation_transform=self._root_noise.transform,
        )

    def advance(self) -> InferenceNeed | GameFinished:
        if self.technical is not None or self.formal is not None:
            return GameFinished(self._finish())
        try:
            while True:
                if self._session is None:
                    if len(self.trace) >= self.context.contract.watchdog:
                        self.technical = "TRUNCATED_MOVE_LIMIT"
                        self.error = "Torus 9×9 self-play watchdog reached 500 actions"
                        return GameFinished(self._finish())
                    self._start_search()
                step = self._session.advance()
                if isinstance(step, SearchEvaluationRequest):
                    self._nn_evaluations += 1
                    return InferenceNeed((step.state, step.legal_context))
                if not isinstance(step, _t9.SearchResult):
                    raise RuntimeError("Torus9 cooperative search returned malformed result")
                ply = len(self.trace) + 1
                search_seed = derive_seed(self.game_seed, ply, "search")
                action = _t9._sample_action(
                    step,
                    temperature=1.0 if ply <= self.context.contract.temperature_until_ply else self.context.contract.temperature_after,
                    rng=self.rng,
                )
                self.positions.append(_t9.Torus9SelfPlayPosition(
                    ply=ply,
                    state=_t9.torus9_state_identity(self.state),
                    side_to_move=self.state.side_to_move.name,
                    root_visits=tuple(int(value) for value in step.root_visits),
                    pi=tuple(float(value) for value in step.pi),
                    selected_action=action,
                    search_seed=search_seed,
                    model_hash=self.context.expected_model_hash,
                ))
                self.trace.append(action)
                try:
                    self.state = _t9.apply_action(self.state, action).after
                except _t9.IllegalMoveError as exc:
                    self.technical = "ERROR_ILLEGAL_PLAYER_ACTION"
                    self.error = f"{type(exc).__name__}: {exc}"
                    return GameFinished(self._finish())
                self._session = None
                self._root_noise = None
                if self.state.is_terminal:
                    self.formal = _t9.result_from_terminal(self.state).winner.value
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
            raise InferenceTransportError("Torus9 cooperative game is not waiting for inference")
        try:
            self._session.resume(evaluation)
        except Exception as exc:
            self.technical = "ERROR_SEARCH"
            self.error = f"{type(exc).__name__}: {exc}"
            self._session = None


def _torus9_record_metrics(record: object) -> Mapping[str, object]:
    if not isinstance(record, _t9.Torus9SelfPlayGameRecord):
        raise TypeError("Torus9 worker returned the wrong record type")
    return {
        "moves": len(record.final_action_trace),
        "technical": record.technical_termination is not None,
    }


class Torus9CentralInferenceOwner:
    """The only process that owns the Torus9 neural model/device."""

    def __init__(self, model: _t9.Torus9GraphNet, *, device: str | torch.device) -> None:
        if model.topology_fingerprint != _t9.TORUS9_TOPOLOGY_FINGERPRINT:
            raise ValueError("Torus9 central inference received the wrong topology model")
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _evaluate_batch_tensors(self, payloads: Sequence[object]):
        if not payloads:
            raise ValueError("Torus9 central inference requires a non-empty batch")
        observations = torch.stack(
            [torch.as_tensor(payload, dtype=torch.float32) for payload in payloads],
            dim=0,
        ).to(self.device)
        if tuple(observations.shape[1:]) != (6, _t9.TORUS9_POINT_COUNT):
            raise ValueError("Torus9 inference payload must have shape [batch,6,81]")
        with torch.inference_mode():
            policy_logits, value_logits = self.model(observations)
            policies = torch.softmax(policy_logits, dim=1)
            wdls = torch.softmax(value_logits, dim=1)
        if tuple(policies.shape) != (len(payloads), TORUS9_ACTION_COUNT):
            raise ValueError("Torus9 central policy head shape drift")
        if tuple(wdls.shape) != (len(payloads), 3):
            raise ValueError("Torus9 central WDL head shape drift")
        outputs = torch.cat((policies, wdls), dim=1)
        if not bool(torch.isfinite(outputs).all()) or bool((outputs < 0.0).any()):
            raise ValueError("Torus9 central inference produced invalid probabilities")
        return outputs

    def evaluate_shared_batch(self, observations: Any) -> SharedInferenceResult:
        """Forward a preassembled staging tensor without queue payload copies."""
        if tuple(observations.shape[1:]) != (6, _t9.TORUS9_POINT_COUNT):
            raise ValueError("Torus9 shared observations must have shape [batch,6,81]")
        h2d_started = time.perf_counter()
        device_observations = observations.to(self.device, non_blocking=self.device.type == "cuda")
        h2d_finished = time.perf_counter()
        forward_started = time.perf_counter()
        with torch.inference_mode():
            policy_logits, value_logits = self.model(device_observations)
            policies = torch.softmax(policy_logits, dim=1)
            wdls = torch.softmax(value_logits, dim=1)
        forward_finished = time.perf_counter()
        if tuple(policies.shape) != (int(observations.shape[0]), TORUS9_ACTION_COUNT):
            raise ValueError("Torus9 central policy head shape drift")
        if tuple(wdls.shape) != (int(observations.shape[0]), 3):
            raise ValueError("Torus9 central WDL head shape drift")
        outputs = torch.cat((policies, wdls), dim=1)
        if not bool(torch.isfinite(outputs).all()) or bool((outputs < 0.0).any()):
            raise ValueError("Torus9 central inference produced invalid probabilities")
        return SharedInferenceResult(
            policy=policies,
            wdl=wdls,
            h2d_started_at=h2d_started,
            h2d_finished_at=h2d_finished,
            forward_started_at=forward_started,
            forward_finished_at=forward_finished,
        )

    def evaluate_batch(self, payloads: Sequence[object]) -> tuple[Evaluation, ...]:
        """Return the legacy in-process Evaluation representation."""
        outputs = self._evaluate_batch_tensors(payloads)
        rows: list[Evaluation] = []
        for output in outputs.detach().cpu():
            values = tuple(float(value) for value in output)
            policy_values = values[:TORUS9_ACTION_COUNT]
            wdl_values = values[TORUS9_ACTION_COUNT:]
            rows.append(Evaluation(policy=policy_values, wdl=wdl_values))
        return tuple(rows)

def torus9_game_seed(master_seed: int, run_id: str, game_id: str) -> int:
    """Canonical scheduling-independent Torus9 game seed."""
    return derive_seed(int(master_seed), str(run_id), str(game_id), "game")


def _validate_current_scientific_boundary(
    model: _t9.Torus9GraphNet,
    *,
    profile_id: str,
    profile_fp: str,
    contract: _t9.Torus9SelfPlaySearchContract,
) -> None:
    if float(TORUS9_KOMI) != 0.5:
        raise RuntimeError("Current Torus9 self-play requires komi 0.5")
    if profile_id != TORUS9_CURRENT_PROFILE_ID:
        return
    profile = load_torus9_current_profile()
    expected_profile_fp = current_torus9_profile_fingerprint(profile)
    if profile_fp != expected_profile_fp:
        raise ValueError("Current Torus9 profile/lineage fingerprint drift")
    if not isinstance(model, _t9.Torus9CurrentGraphNet):
        raise ValueError("Current Torus9 self-play requires GoldenGraphNetV2-Torus9")
    if int(model.hidden) != TORUS9_CURRENT_HIDDEN or int(model.blocks_count) != TORUS9_CURRENT_BLOCKS:
        raise ValueError("Current Torus9 network must remain 80x8")
    if contract.contract_id != TORUS9_CURRENT_SELFPLAY_CONTRACT_ID:
        raise ValueError("Current Torus9 self-play contract id drift")
    if not math.isclose(float(contract.dirichlet_alpha), float(TORUS9_CURRENT_DIRICHLET_ALPHA), abs_tol=0.0):
        raise ValueError("Current Torus9 Dirichlet alpha drift")
    contract.validate()


class Torus9SelfPlayAdapter:
    """Scientific/profile side of the SelfPlayEngine boundary."""

    def __init__(
        self,
        model: _t9.Torus9GraphNet,
        *,
        run_id: str,
        label: str,
        artifact: str,
        master_seed: int,
        profile_fp: str,
        code_identity: CodeIdentity,
        device: str | torch.device,
        contract: _t9.Torus9SelfPlaySearchContract,
        profile_id: str,
    ) -> None:
        contract.validate()
        _validate_current_scientific_boundary(
            model,
            profile_id=profile_id,
            profile_fp=profile_fp,
            contract=contract,
        )
        self.model = model
        self.device = torch.device(device)
        self.owner = Torus9CentralInferenceOwner(model, device=self.device)
        self.worker_context = Torus9SelfPlayWorkerContext(
            run_id=str(run_id),
            model_checkpoint_label=str(label),
            checkpoint_artifact_hash=str(artifact),
            master_seed=int(master_seed),
            profile_fingerprint=str(profile_fp),
            code_identity=code_identity,
            contract=contract,
            profile_id=str(profile_id),
            expected_model_hash=model_hash(model),
        )
        self.shared_memory = SharedMemorySpec(
            observation_shape=(6, _t9.TORUS9_POINT_COUNT),
            policy_size=TORUS9_ACTION_COUNT,
            wdl_size=3,
            write_input=_write_torus9_shared_observation,
            decode_output=_decode_torus9_shared_output,
        )

    @property
    def infer_shared_batch(self):
        return self.owner.evaluate_shared_batch

    @property
    def worker_game_factory(self):
        return _make_torus9_cooperative_game

    @property
    def record_metrics(self):
        return _torus9_record_metrics


def _make_torus9_cooperative_game(
    context: Torus9SelfPlayWorkerContext,
    game_id: str,
    client: InferenceClient,
) -> _Torus9CooperativeGame:
    return _Torus9CooperativeGame(context, game_id, client)


# Keep the exact pre-Stage-2 public call shape. checkpoint_path is retained as
# provenance/API compatibility even though workers no longer load a model copy.
def run_torus9_selfplay_games(
    model: _t9.Torus9GraphNet,
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
    contract: _t9.Torus9SelfPlaySearchContract = _t9.Torus9SelfPlaySearchContract(),
    profile_id: str = TORUS9_PROFILE_ID,
    coalescing: bool = False,
    inference_batch_cap: int | None = None,
    inference_batch_wait_ms: float = 0.0,
    search_lanes_per_worker: int = 4,
    active_games_per_worker: int | None = None,
    total_active_contexts: int | None = None,
    inference_telemetry: MutableMapping[str, object] | None = None,
    execution_activity: MutableMapping[str, object] | None = None,
) -> tuple[_t9.Torus9SelfPlayGameRecord, ...]:
    del checkpoint_path
    if workers <= 0 or len(set(game_ids)) != len(game_ids):
        raise ValueError("Torus9 self-play workers/game IDs are invalid")
    code = code_identity or capture_code_identity()
    ids = tuple(sorted(str(game_id) for game_id in game_ids))
    adapter = Torus9SelfPlayAdapter(
        model,
        run_id=run_id,
        label=label,
        artifact=artifact,
        master_seed=master_seed,
        profile_fp=profile_fp,
        code_identity=code,
        device=device,
        contract=contract,
        profile_id=profile_id,
    )
    active_games = int(active_games_per_worker) if active_games_per_worker is not None else (int(search_lanes_per_worker) if coalescing else 1)
    if active_games <= 0:
        raise ValueError("Torus9 self-play requires positive active games per worker")
    batch_cap = (
        int(inference_batch_cap)
        if inference_batch_cap is not None
        else (max(16, int(workers) * active_games) if coalescing else 1)
    )
    if coalescing and batch_cap <= 1:
        raise ValueError("Coalesced Torus9 self-play requires batch_cap > 1")
    wait_ms = float(inference_batch_wait_ms) if coalescing else 0.0
    start_method = "spawn" if torch.device(device).type == "cuda" else "fork"
    engine = SelfPlayEngine(
        SelfPlayEngineConfig(
            workers=int(workers),
            inference_batch_cap=batch_cap,
            inference_batch_wait_ms=wait_ms,
            device=str(torch.device(device)),
            process_start_method=start_method,
            # ``lanes_per_worker`` is reported only for compatibility. The
            # shared production path uses cooperative game contexts.
            lanes_per_worker=1,
            active_games_per_worker=active_games,
            total_active_contexts=total_active_contexts,
        )
    )
    raw_telemetry: dict[str, object] = {}
    records = engine.run(
        ids,
        worker_play=None,
        worker_context=adapter.worker_context,
        infer_batch=None,
        record_metrics=adapter.record_metrics,
        telemetry=raw_telemetry,
        shared_memory=adapter.shared_memory,
        infer_shared_batch=adapter.infer_shared_batch,
        worker_game_factory=adapter.worker_game_factory,
        active_games_per_worker=active_games,
        total_active_contexts=total_active_contexts,
    )

    compatibility_inference = {
        "mode": "central-process-batched" if coalescing else "central-process-uncoalesced",
        "batch_cap": raw_telemetry["inference_batch_cap"],
        "wait_ms": raw_telemetry["inference_batch_wait_ms"],
        "forward_calls": raw_telemetry["inference_forwards"],
        "total_rows": raw_telemetry["inference_rows"],
        "batch_rows": raw_telemetry["batch_rows"],
        "mean_batch_rows": raw_telemetry["mean_inference_batch_rows"],
        "p50_batch_rows": raw_telemetry["p50_inference_batch_rows"],
        "p95_batch_rows": raw_telemetry["p95_inference_batch_rows"],
        "max_batch_rows": raw_telemetry["max_inference_batch_rows"],
        "rows_per_sec": raw_telemetry["inference_rows_per_sec"],
        "forwards_per_sec": raw_telemetry["inference_forwards_per_sec"],
        "device": raw_telemetry["inference_device"],
        "central_inference_owner_pid": raw_telemetry["central_inference_owner_pid"],
        "max_concurrent_forwards": 1 if int(raw_telemetry["inference_forwards"]) else 0,
        "lock_scope": "single_parent_inference_owner",
    }
    compatibility_execution = {
        **raw_telemetry,
        "max_active_mcts_lanes": raw_telemetry["peak_concurrent_search_workers"],
        "max_active_inference_requests": raw_telemetry["max_inference_batch_rows"],
    }
    if inference_telemetry is not None:
        inference_telemetry.update(compatibility_inference)
    if execution_activity is not None:
        execution_activity.update(compatibility_execution)
    typed_records: list[_t9.Torus9SelfPlayGameRecord] = []
    for record in records:
        if not isinstance(record, _t9.Torus9SelfPlayGameRecord):
            raise RuntimeError("Torus9 engine returned the wrong record type")
        record.validate()
        typed_records.append(record)
    return tuple(typed_records)
