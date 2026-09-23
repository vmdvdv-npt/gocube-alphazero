"""Torus 9x9 scientific adapter for the universal Arena engine.

Scientific state/scoring remains here. Cooperative lane execution and parent
inference mechanics are shared with every Arena topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from gocube_golden.arena_contract import SearchSettings
from gocube_golden.neural import model_hash
from gocube_golden.provenance import derive_seed, file_sha256
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
from gocube_golden.torus9_contract import (
    TORUS9_ACTION_COUNT,
    TORUS9_ARENA_MOVE_LIMIT,
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
        state=torus9_state_from_identity(task["state"]),  # type: ignore[arg-type]
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
        "komi": TORUS9_KOMI,
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


def _select_starts(master_seed: int, pairs: int) -> tuple[dict[str, object], ...]:
    if pairs <= 0:
        raise ValueError("Arena requires at least one pair")
    per_stratum = max(1, (pairs + 7) // 8)
    generated = generate_torus9_evaluation_starts(
        master_seed=master_seed,
        accepted_per_stratum=per_stratum,
    )
    selected: list[dict[str, object]] = []
    for offset in range(per_stratum):
        for stratum in range(8):
            index = stratum * per_stratum + offset
            if index < len(generated):
                selected.append(dict(generated[index]))
                if len(selected) == pairs:
                    return tuple(selected)
    raise RuntimeError("Could not build requested Torus9 Arena startset")


class Torus9ArenaProfile:
    profile_id = PROFILE_ID
    run_id_prefix = "torus9-arena"
    worker_process_prefix = "arena-torus9-worker"
    observation_shape = (6, TORUS9_POINT_COUNT)
    policy_size = TORUS9_ACTION_COUNT
    wdl_size = 3
    last_infer_timing: Mapping[str, float] = {}

    def matches_metadata(self, metadata: Mapping[str, object]) -> bool:
        return (
            metadata.get("profile_id") == TORUS9_CURRENT_PROFILE_ID
            and metadata.get("architecture_id") == TORUS9_CURRENT_ARCHITECTURE_ID
            and metadata.get("topology_fingerprint") == TORUS9_TOPOLOGY_FINGERPRINT
        )

    def validate_execution_config(self, config: ArenaExecutionConfig) -> None:
        if TORUS9_KOMI != 0.5:
            raise RuntimeError("Active Torus9 komi drifted from 0.5")
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
        if float(metadata.get("komi", -1.0)) != 0.5:
            raise ValueError("Torus9 Arena checkpoint komi must be 0.5")
        architecture = metadata.get("architecture_config")
        if not isinstance(architecture, Mapping):
            raise ValueError("Torus9 checkpoint architecture metadata is malformed")
        return CheckpointIdentity(
            path=path,
            model_hash=str(metadata["model_hash"]),
            artifact_sha256=file_sha256(path),
            architecture_config=dict(architecture),
            metadata=dict(metadata),
        )

    def load_parent_model(
        self,
        identity: CheckpointIdentity,
        device: torch.device,
    ) -> torch.nn.Module:
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
    ) -> tuple[list[dict[str, object]], int]:
        pairs = games // 2
        starts = _select_starts(master_seed, pairs)
        tasks: list[dict[str, object]] = []
        for row in starts:
            pair_id = f"{comparison}--{row['start_id']}"
            for suffix, candidate_black in (("g1", True), ("g2", False)):
                game_id = f"{pair_id}--{suffix}"
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
            if game.ply >= TORUS9_ARENA_MOVE_LIMIT:
                game.technical = "TRUNCATED_MOVE_LIMIT"
                game.error = (
                    "Torus 9x9 Arena watchdog reached "
                    f"{TORUS9_ARENA_MOVE_LIMIT} actions"
                )
                return True
            return False

        def mark_search_error(game: _WorkerGame, exc: Exception) -> None:
            game.technical = "ERROR_SEARCH"
            game.error = f"{type(exc).__name__}: {exc}"

        callbacks = CooperativeArenaCallbacks(
            search_settings=SearchSettings(
                simulations=64,
                cpuct=1.25,
                fpu=0.0,
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
            build_observation=lambda state, legal_context: build_torus9_observation(
                state,
                legal_context=legal_context,
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
            "profile": PROFILE_ID,
            "games": config.games,
            "komi": 0.5,
            "simulations": 64,
            "cpuct": 1.25,
            "fpu": 0.0,
            "noise": False,
            "temperature": 0.0,
            "fast_search": False,
            "resign": False,
            "watchdog": TORUS9_ARENA_MOVE_LIMIT,
            "paired_starts_color_swap": True,
            "deterministic_tie_break": True,
            "technical_fail_closed": True,
        }


PROFILE = Torus9ArenaProfile()
