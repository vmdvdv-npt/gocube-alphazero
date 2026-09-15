"""Torus9 scientific adapter for the universal process SelfPlayEngine.

The adapter owns Torus9 observation/model/search/game-record semantics.  The
engine owns only execution.  No legacy Coach/SelfPlayAgent/NNetWrapper path is
used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .neural import model_hash
from .provenance import CodeIdentity, capture_code_identity, derive_seed
from .search import Evaluation
from .selfplay_engine import (
    InferenceClient,
    InferenceTransportError,
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


class _RemoteTorus9Evaluator:
    """Worker-side Torus9 evaluator backed by central inference RPC."""

    def __init__(self, client: InferenceClient) -> None:
        self.client = client
        self.nn_evaluations = 0

    def evaluate_prepared(self, state: _t9.GoldenState, legal_context: _t9.LegalActionContext) -> Evaluation:
        legal_context.assert_compatible(state)
        observation = _t9.build_torus9_observation(state, legal_context=legal_context)
        payload = tuple(tuple(float(value) for value in row) for row in observation.tolist())
        result = self.client.request(payload)
        if not isinstance(result, Evaluation):
            raise InferenceTransportError(
                f"Torus9 central inference returned {type(result).__name__}, expected Evaluation"
            )
        if len(result.policy) != TORUS9_ACTION_COUNT or len(result.wdl) != 3:
            raise InferenceTransportError("Torus9 central inference returned malformed head shapes")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in result.policy + result.wdl):
            raise InferenceTransportError("Torus9 central inference returned invalid probabilities")
        self.nn_evaluations += 1
        return result


class _RemoteTorus9Runner(_t9.Torus9SelfPlayRunner):
    """Reuse the exact current game/search loop without a worker-local model."""

    def __init__(self, context: Torus9SelfPlayWorkerContext, evaluator: _RemoteTorus9Evaluator) -> None:
        context.contract.validate()
        self.model = None
        self.run_id = context.run_id
        self.model_checkpoint_label = context.model_checkpoint_label
        self.model_hash = context.expected_model_hash
        self.checkpoint_artifact_hash = context.checkpoint_artifact_hash
        self.master_seed = int(context.master_seed)
        self.profile_fp = context.profile_fingerprint
        self.seed_namespace = context.run_id
        self.code_identity = context.code_identity
        self.device = torch.device("cpu")
        self.contract = context.contract
        self.profile_id = context.profile_id
        self.evaluator = evaluator
        self.activity_tracker = None
        self.adapter = _t9.GoldenSearchAdapter()


def _play_torus9_worker_game(
    context: Torus9SelfPlayWorkerContext,
    game_id: str,
    client: InferenceClient,
) -> _t9.Torus9SelfPlayGameRecord:
    runner = _RemoteTorus9Runner(context, _RemoteTorus9Evaluator(client))
    return runner.play_game(game_id)


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

    def evaluate_batch(self, payloads: Sequence[object]) -> tuple[Evaluation, ...]:
        if not payloads:
            raise ValueError("Torus9 central inference requires a non-empty batch")
        observations = torch.tensor(payloads, dtype=torch.float32, device=self.device)
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
        rows: list[Evaluation] = []
        for policy, wdl in zip(policies.detach().cpu(), wdls.detach().cpu()):
            policy_values = tuple(float(value) for value in policy)
            wdl_values = tuple(float(value) for value in wdl)
            if any(not math.isfinite(value) or value < 0.0 for value in policy_values + wdl_values):
                raise ValueError("Torus9 central inference produced invalid probabilities")
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

    @property
    def worker_play(self):
        return _play_torus9_worker_game

    @property
    def infer_batch(self):
        return self.owner.evaluate_batch

    @property
    def record_metrics(self):
        return _torus9_record_metrics


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
    batch_cap = int(inference_batch_cap) if inference_batch_cap is not None else (max(16, int(workers)) if coalescing else 1)
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
        )
    )
    raw_telemetry: dict[str, object] = {}
    records = engine.run(
        ids,
        worker_play=adapter.worker_play,
        worker_context=adapter.worker_context,
        infer_batch=adapter.infer_batch,
        record_metrics=adapter.record_metrics,
        telemetry=raw_telemetry,
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
