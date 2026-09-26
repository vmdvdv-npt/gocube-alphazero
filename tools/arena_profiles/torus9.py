"""Torus 9x9 scientific adapter for the universal Arena engine.

Scientific state/scoring remains here. Cooperative lane execution and parent
inference mechanics are shared with every Arena topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.neural import model_hash
from gocube_golden.provenance import derive_seed, file_sha256, sha256_fingerprint
from gocube_golden.result import result_from_terminal
from gocube_golden.rules import IllegalMoveError, apply_action
from gocube_golden.scoring import score_terminal
from gocube_golden.search import SearchResult
from gocube_golden.search_adapter import GoldenSearchAdapter
from gocube_golden.state import BLACK, WHITE
from gocube_golden.torus9 import (
    TORUS9_TOPOLOGY_FINGERPRINT,
    build_torus9_observation,
    generate_torus9_evaluation_starts,
    summarize_torus9_arena,
    torus9_load_checkpoint,
    torus9_model_from_metadata,
    torus9_state_from_identity,
)
from gocube_golden.torus9_m137_5ch import (
    M137_FIVE_CHANNEL_ARCHITECTURE_ID,
    Torus9M137FiveChannelGraphNet,
    build_m137_five_channel_observation,
)
from gocube_golden.torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_ARENA_MOVE_LIMIT,
    TORUS9_ALLOWED_KOMI,
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_KOMI,
    TORUS9_POINT_COUNT,
)
from tools.arena_engine import ArenaExecutionConfig, CheckpointIdentity
from tools.arena_inference import infer_policy_wdl_batch
from tools.arena_worker import (
    CooperativeArenaCallbacks,
    run_cooperative_arena_worker,
)

PROFILE_ID = "torus9"
PRODUCTION_WORKERS = 16
PRODUCTION_MIN_GAMES = 192
PRODUCTION_MIN_BATCH_ROWS = 16


@dataclass
class _WorkerGame:
    task: Mapping[str, object]
    state: Any
    trace: list[dict[str, object]]
    ply: int
    started_at: float
    formal: str | None = None
    technical: str | None = None
    error: str | None = None


def _make_game(task: Mapping[str, object]) -> _WorkerGame:
    return _WorkerGame(
        task=task,
        state=torus9_state_from_identity(
            task["state"], expected_komi=float(task["komi"])  # type: ignore[arg-type]
        ),
        trace=[],
        ply=0,
        started_at=time.perf_counter(),
    )


def _candidate_turn(game: _WorkerGame) -> bool:
    state = game.state
    candidate_black = bool(game.task["candidate_black"])
    return (state.side_to_move == BLACK and candidate_black) or (
        state.side_to_move == WHITE and not candidate_black
    )


def _finish_game(game: _WorkerGame) -> dict[str, object]:
    task = game.task
    state = game.state
    formal = game.formal
    candidate_black = bool(task["candidate_black"])
    row: dict[str, object] = {
        "run_id": str(task["run_id"]),
        "comparison": str(task["comparison"]),
        "pair_id": str(task["pair_id"]),
        "game_id": str(task["game_id"]),
        "start_id": str(task["start_id"]),
        "worker_id": int(task["worker_id"]),
        "worker_pid": os.getpid(),
        "candidate_black": candidate_black,
        "candidate_model_hash": str(task["candidate_hash"]),
        "reference_model_hash": str(task["reference_hash"]),
        "komi": float(state.komi),
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "start_state": task["state"],
        "start_trace": task["trace"],
        "action_trace": game.trace,
        "final_board": [int(stone) for stone in state.stones],
        "formal_result": formal,
        "technical_termination": game.technical,
        "error": game.error,
        "black_area": None,
        "white_area": None,
        "margin_black": None,
        "mapped_result": None,
        "wall_time_sec": time.perf_counter() - game.started_at,
    }
    if formal is not None:
        score = score_terminal(state)
        row.update(
            {
                "black_area": score.black_area,
                "white_area": score.white_area,
                "margin_black": score.margin_black,
            }
        )
        if formal == "DRAW":
            row["mapped_result"] = "DRAW"
        elif (formal == "BLACK") == candidate_black:
            row["mapped_result"] = "A_WIN"
        else:
            row["mapped_result"] = "B_WIN"
    return row


def _select_starts(
    master_seed: int,
    pairs: int,
    *,
    komi: float = TORUS9_KOMI,
    offset_pairs: int = 0,
) -> tuple[dict[str, object], ...]:
    if pairs <= 0:
        raise ValueError("Arena requires at least one pair")
    if offset_pairs < 0:
        raise ValueError("Arena startset offset must be non-negative")
    per_stratum = max(1, (pairs + offset_pairs + 7) // 8)
    generated = generate_torus9_evaluation_starts(
        master_seed=master_seed,
        accepted_per_stratum=per_stratum,
        komi=komi,
    )
    selected: list[dict[str, object]] = []
    for offset in range(per_stratum):
        for stratum in range(8):
            index = stratum * per_stratum + offset
            if index < len(generated):
                selected.append(dict(generated[index]))
                if len(selected) == pairs + offset_pairs:
                    return tuple(selected[offset_pairs:])
    raise RuntimeError("Could not build requested Torus9 Arena startset")


def _state_for_komi(state: Mapping[str, object], komi: float) -> dict[str, object]:
    """Rebind only the referee-owned komi fields of a frozen legal start."""
    from gocube_golden.state import rules_fingerprint_for
    from gocube_golden.topology import TORUS_9X9

    result = dict(state)
    result["komi"] = float(komi)
    result["rules_fingerprint"] = rules_fingerprint_for(TORUS_9X9, float(komi))
    return result


def _load_frozen_starts(
    path: Path,
    *,
    master_seed: int,
    pair_indices: object,
    expected_fingerprint: str | None,
    komi: float,
) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read frozen Arena startset: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "torus9-frozen-startset-v1":
        raise ValueError("Arena frozen startset schema mismatch")
    if int(payload.get("master_seed", -1)) != int(master_seed):
        raise ValueError("Arena frozen startset master seed mismatch")
    pairs = payload.get("pairs")
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence) or not pairs:
        raise ValueError("Arena frozen startset pairs are malformed")
    body = {str(key): value for key, value in payload.items() if key != "fingerprint"}
    actual_fingerprint = sha256_fingerprint(body)
    declared_fingerprint = str(payload.get("fingerprint", ""))
    if declared_fingerprint != actual_fingerprint or (
        expected_fingerprint is not None and expected_fingerprint != actual_fingerprint
    ):
        raise ValueError("Arena frozen startset fingerprint mismatch")
    if pair_indices is None:
        selected_indices = list(range(len(pairs)))
    else:
        if isinstance(pair_indices, (str, bytes)) or not isinstance(pair_indices, Sequence):
            raise ValueError("Arena workload pair_indices must be a sequence")
        selected_indices = [int(value) for value in pair_indices]
    selected: list[dict[str, object]] = []
    for index in selected_indices:
        if index < 0 or index >= len(pairs):
            raise ValueError("Arena frozen startset pair index is out of range")
        raw = pairs[index]
        if not isinstance(raw, Mapping):
            raise ValueError("Arena frozen startset pair is malformed")
        start = dict(raw)
        state = start.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("Arena frozen startset pair has no state")
        start["state"] = _state_for_komi(state, komi)
        selected.append(start)
    return tuple(selected)


class Torus9ArenaProfile:
    profile_id = PROFILE_ID
    run_id_prefix = "torus9-arena"
    worker_process_prefix = "arena-torus9-worker"
    observation_shape = (6, TORUS9_POINT_COUNT)
    policy_size = TORUS9_ACTION_COUNT
    wdl_size = 3
    last_infer_timing: Mapping[str, float] = {}

    def __init__(
        self,
        *,
        komi: float = TORUS9_KOMI,
        profile_id: str = PROFILE_ID,
        simulations: int = 64,
        cpuct: float = 1.25,
        fpu: float = 0.0,
        watchdog: int = TORUS9_ARENA_MOVE_LIMIT,
        five_channel: bool = False,
    ) -> None:
        if not isinstance(komi, (int, float)) or isinstance(komi, bool) or float(komi) not in TORUS9_ALLOWED_KOMI:
            raise ValueError("Torus9 Arena komi must be one of 0.5, 1.5, 2.5, 3.5, or 4.5")
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
            raise ValueError("Torus9 Arena simulations must be a positive integer")
        if not isinstance(cpuct, (int, float)) or isinstance(cpuct, bool) or not math.isfinite(float(cpuct)) or float(cpuct) <= 0:
            raise ValueError("Torus9 Arena cpuct must be finite and positive")
        if not isinstance(fpu, (int, float)) or isinstance(fpu, bool) or not math.isfinite(float(fpu)):
            raise ValueError("Torus9 Arena fpu must be finite")
        if isinstance(watchdog, bool) or not isinstance(watchdog, int) or watchdog <= 0:
            raise ValueError("Torus9 Arena watchdog must be a positive integer")
        self.komi = float(komi)
        self.profile_id = str(profile_id)
        self.simulations = int(simulations)
        self.cpuct = float(cpuct)
        self.fpu = float(fpu)
        self.watchdog = int(watchdog)
        self._five_channel = bool(five_channel)
        self.observation_shape = (5 if self._five_channel else 6, TORUS9_POINT_COUNT)

    def matches_metadata(self, metadata: Mapping[str, object]) -> bool:
        architecture = metadata.get("architecture_config")
        architecture_topology = (
            architecture.get("topology_fingerprint")
            if isinstance(architecture, Mapping)
            else None
        )
        topology_fingerprint = metadata.get("topology_fingerprint", architecture_topology)
        legacy = (
            metadata.get("profile_id") == TORUS9_CURRENT_PROFILE_ID
            and metadata.get("architecture_id") == TORUS9_CURRENT_ARCHITECTURE_ID
            and topology_fingerprint == TORUS9_TOPOLOGY_FINGERPRINT
        )
        five_channel = (
            metadata.get("architecture_id") == M137_FIVE_CHANNEL_ARCHITECTURE_ID
            and topology_fingerprint == TORUS9_TOPOLOGY_FINGERPRINT
            and metadata.get("observation_shape") == [5, TORUS9_POINT_COUNT]
        )
        return legacy or five_channel

    def validate_execution_config(self, config: ArenaExecutionConfig) -> None:
        if config.strict_production:
            if config.games < PRODUCTION_MIN_GAMES and not config.monitoring_acceptance:
                raise ValueError(
                    f"Production Torus9 Arena requires at least {PRODUCTION_MIN_GAMES} games"
                )
            if config.workers != PRODUCTION_WORKERS:
                raise ValueError("Production Torus9 Arena requires exactly 16 OS workers")
            if torch.device(config.device).type != "cuda":
                raise ValueError("Production Torus9 Arena requires CUDA central inference")
            if config.inference_batch_rows < PRODUCTION_MIN_BATCH_ROWS:
                raise ValueError(
                    "Production Torus9 Arena inference_batch_rows must be >= 16"
                )

    def load_identity(self, path: Path) -> CheckpointIdentity:
        metadata_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Torus9 checkpoint or metadata: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not self.matches_metadata(metadata):
            raise ValueError("Checkpoint does not match Torus9 Arena profile")
        architecture = metadata.get("architecture_config")
        if not isinstance(architecture, Mapping):
            raise ValueError("Torus9 checkpoint architecture metadata is malformed")
        is_five_channel = metadata.get("architecture_id") == M137_FIVE_CHANNEL_ARCHITECTURE_ID
        if is_five_channel:
            if architecture.get("input_channels") != 5:
                raise ValueError("Torus9 5CH checkpoint input_channels drift")
            if metadata.get("observation_shape") != [5, TORUS9_POINT_COUNT]:
                raise ValueError("Torus9 5CH checkpoint observation shape drift")
        else:
            metadata_komi = float(metadata.get("komi", -1.0))
            allowed_metadata_komi = {0.5} if self.profile_id == PROFILE_ID else {0.5, self.komi}
            if metadata_komi not in allowed_metadata_komi:
                raise ValueError("Torus9 Arena checkpoint metadata komi is incompatible")
        if self._five_channel != is_five_channel:
            raise ValueError("Torus9 Arena profile/checkpoint channel contract mismatch")
        self._five_channel = is_five_channel
        self.observation_shape = (5 if is_five_channel else 6, TORUS9_POINT_COUNT)
        identity_model_hash = metadata.get("model_hash", metadata.get("converted_model_hash"))
        if not isinstance(identity_model_hash, str) or not identity_model_hash:
            raise ValueError("Torus9 checkpoint metadata has no model hash")
        return CheckpointIdentity(
            path=path,
            model_hash=identity_model_hash,
            artifact_sha256=file_sha256(path),
            architecture_config=dict(architecture),
            metadata=dict(metadata),
        )

    def load_parent_model(
        self,
        identity: CheckpointIdentity,
        device: torch.device,
    ) -> torch.nn.Module:
        if identity.metadata.get("architecture_id") == M137_FIVE_CHANNEL_ARCHITECTURE_ID:
            try:
                payload = torch.load(identity.path, map_location=device, weights_only=False)
            except TypeError:
                payload = torch.load(identity.path, map_location=device)
            if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
                raise ValueError("Torus9 5CH checkpoint payload is malformed")
            model = Torus9M137FiveChannelGraphNet().to(device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            if model_hash(model) != identity.model_hash:
                raise RuntimeError("Parent inference broker loaded the wrong Torus9 5CH checkpoint")
            model.eval()
            return model
        model = torus9_model_from_metadata(identity.metadata).to(device)
        torus9_load_checkpoint(
            identity.path,
            model=model,
            expected={"model_hash": identity.model_hash},
            device=device,
        )
        if model_hash(model) != identity.model_hash:
            raise RuntimeError("Parent inference broker loaded the wrong Torus9 checkpoint")
        model.eval()
        return model

    def build_tasks(
        self,
        *,
        run_id: str,
        comparison: str,
        candidate: CheckpointIdentity,
        reference: CheckpointIdentity,
        master_seed: int,
        games: int,
        workers: int,
        workload: Mapping[str, object] | None = None,
    ) -> tuple[list[dict[str, object]], int]:
        pairs = games // 2
        workload = workload or {}
        offset_pairs = int(workload.get("continuation_offset_pairs", 0))
        startset_path = workload.get("startset_path")
        pair_indices = workload.get("pair_indices")
        selected_game_ids = workload.get("game_ids")
        if startset_path is not None:
            starts = _load_frozen_starts(
                Path(str(startset_path)),
                master_seed=master_seed,
                pair_indices=pair_indices,
                expected_fingerprint=(
                    None
                    if workload.get("startset_fingerprint") is None
                    else str(workload["startset_fingerprint"])
                ),
                komi=self.komi,
            )
        else:
            starts = _select_starts(
                master_seed,
                pairs,
                komi=self.komi,
                offset_pairs=offset_pairs,
            )
        if selected_game_ids is not None:
            if isinstance(selected_game_ids, (str, bytes)) or not isinstance(selected_game_ids, Sequence):
                raise ValueError("Arena workload game_ids must be a sequence")
            selected = {str(value) for value in selected_game_ids}
        else:
            selected = None
        tasks: list[dict[str, object]] = []
        for row in starts:
            pair_id = str(row.get("pair_id") or f"{comparison}--{row['start_id']}")
            for suffix, candidate_black in (("g1", True), ("g2", False)):
                game_id = f"{pair_id}--{suffix}"
                if selected is not None and game_id not in selected:
                    continue
                tasks.append(
                    {
                        "run_id": run_id,
                        "comparison": comparison,
                        "pair_id": pair_id,
                        "game_id": game_id,
                        "start_id": row["start_id"],
                        "state": row["state"],
                        "trace": row["trace"],
                        "candidate_black": candidate_black,
                        "candidate_hash": candidate.model_hash,
                        "reference_hash": reference.model_hash,
                        "candidate_artifact_sha256": candidate.artifact_sha256,
                        "reference_artifact_sha256": reference.artifact_sha256,
                        "komi": self.komi,
                        "game_seed": derive_seed(master_seed, pair_id, game_id),
                        "worker_id": len(tasks) % workers,
                    }
                )
        return tasks, pairs

    def worker_main(
        self,
        worker_id: int,
        task_queue: Any,
        games_per_worker: int,
        worker_local_wait_ms: float,
        candidate_hash: str,
        reference_hash: str,
        input_slot: torch.Tensor,
        policy_slot: torch.Tensor,
        wdl_slot: torch.Tensor,
        request_queue: Any,
        response_queues: Any,
        start_event: Any,
    ) -> None:
        search_adapter = GoldenSearchAdapter()

        def apply_search_result(game: _WorkerGame, result: SearchResult) -> bool:
            state = game.state
            player_is_candidate = _candidate_turn(game)
            game.ply += 1
            action = result.action
            try:
                next_state = apply_action(state, action).after
            except IllegalMoveError as exc:
                game.technical = "ERROR_ILLEGAL_PLAYER_ACTION"
                game.error = f"{type(exc).__name__}: {exc}"
                game.trace.append(
                    {
                        "ply": game.ply,
                        "side_to_move": state.side_to_move.name,
                        "player": (
                            "candidate" if player_is_candidate else "reference"
                        ),
                        "action": action,
                        "legal": False,
                        "error": game.error,
                    }
                )
                return True

            game.state = next_state
            game.trace.append(
                {
                    "ply": game.ply,
                    "side_to_move": (
                        BLACK if next_state.side_to_move == WHITE else WHITE
                    ).name,
                    "player": "candidate" if player_is_candidate else "reference",
                    "action": action,
                    "legal": True,
                }
            )
            if next_state.is_terminal:
                game.formal = result_from_terminal(next_state).winner.value
                return True
            if game.ply >= self.watchdog:
                game.technical = "TRUNCATED_MOVE_LIMIT"
                game.error = (
                    "Torus 9x9 Arena watchdog reached "
                    f"{self.watchdog} actions"
                )
                return True
            return False

        def mark_search_error(game: _WorkerGame, exc: Exception) -> None:
            game.technical = "ERROR_SEARCH"
            game.error = f"{type(exc).__name__}: {exc}"

        callbacks = CooperativeArenaCallbacks(
            search_settings=SearchSettings(
                simulations=self.simulations,
                cpuct=self.cpuct,
                fpu=self.fpu,
                deterministic_tie_break=True,
            ),
            search_adapter=search_adapter,
            make_game=_make_game,
            game_id=lambda game: str(game.task["game_id"]),
            search_state=lambda game: game.state,
            search_seed=lambda game: derive_seed(
                int(game.task["game_seed"]),
                game.ply + 1,
                "arena-search",
            ),
            model_role=lambda game: (
                "candidate" if _candidate_turn(game) else "reference"
            ),
            build_observation=(
                self._build_five_channel_observation
                if self._five_channel
                else self._build_six_channel_observation
            ),
            apply_search_result=apply_search_result,
            mark_search_error=mark_search_error,
            finish_record=_finish_game,
        )
        # SequentialPUCTSession is owned by this common cooperative path.
        run_cooperative_arena_worker(
            callbacks=callbacks,
            worker_id=worker_id,
            task_queue=task_queue,
            games_per_worker=games_per_worker,
            worker_local_wait_ms=worker_local_wait_ms,
            candidate_hash=candidate_hash,
            reference_hash=reference_hash,
            input_slot=input_slot,
            policy_slot=policy_slot,
            wdl_slot=wdl_slot,
            request_queue=request_queue,
            response_queues=response_queues,
            start_event=start_event,
        )

    @staticmethod
    def _build_five_channel_observation(state: Any, legal_context: Any) -> torch.Tensor:
        return build_m137_five_channel_observation(state, legal_context=legal_context)

    @staticmethod
    def _build_six_channel_observation(state: Any, legal_context: Any) -> torch.Tensor:
        return build_torus9_observation(state, legal_context=legal_context)

    @staticmethod
    def forward_policy_wdl_logits(
        model: torch.nn.Module,
        batch: torch.Tensor,
    ) -> object:
        return model(batch)

    def infer_batch(
        self,
        model: torch.nn.Module,
        cpu_batch: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inferred = infer_policy_wdl_batch(
            model,
            cpu_batch,
            device,
            observation_shape=self.observation_shape,
            policy_size=self.policy_size,
            wdl_size=self.wdl_size,
            forward_key=self.profile_id,
            forward_policy_wdl_logits=self.forward_policy_wdl_logits,
        )
        self.last_infer_timing = inferred.timing
        return inferred.policy, inferred.wdl

    def summarize(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        candidate_label: str,
        reference_label: str,
        pairs: int,
    ) -> dict[str, object]:
        return summarize_torus9_arena(
            records,
            candidate_label=candidate_label,
            reference_label=reference_label,
            pairs=pairs,
        )

    def scientific_contract(
        self,
        config: ArenaExecutionConfig,
    ) -> Mapping[str, object]:
        return {
            "profile": self.profile_id,
            "games": config.games,
            "komi": self.komi,
            "simulations": self.simulations,
            "cpuct": self.cpuct,
            "fpu": self.fpu,
            "noise": False,
            "temperature": 0.0,
            "fast_search": False,
            "resign": False,
            "watchdog": self.watchdog,
            "paired_starts_color_swap": True,
            "deterministic_tie_break": True,
            "technical_fail_closed": True,
            "input_channels": 5 if self._five_channel else 6,
            "komi_observation_channel": not self._five_channel,
        }


PROFILE = Torus9ArenaProfile()
