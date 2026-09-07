from __future__ import annotations

import traceback
from dataclasses import dataclass

import numpy as np

from alphazero.SelfPlayAgent import SelfPlayAgent


@dataclass(frozen=True)
class ArenaWorkerAssignment:
    worker_id: int
    quota: int
    model_a_color: str
    player_to_index: tuple[int, int]


def _split_quota(total: int, count: int) -> list[int]:
    if count <= 0:
        if total:
            raise ValueError("cannot assign Arena games to an empty worker group")
        return []
    base, remainder = divmod(int(total), int(count))
    return [base + (1 if index < remainder else 0) for index in range(count)]


def arena_worker_assignments(total_games: int, workers: int) -> tuple[ArenaWorkerAssignment, ...]:
    """Return a fixed-color worker schedule with an exact global color split.

    Workers keep one model/color mapping for their whole lifetime, so the parent
    process can safely interpret results without racing a child-side color flip.
    The per-worker quotas make the aggregate schedule independent of worker
    speed. For an even number of games, model A receives exactly half black and
    half white games. For an odd number, the unavoidable imbalance is one game.
    """
    total_games = int(total_games)
    workers = int(workers)
    if total_games < 1:
        raise ValueError("Arena must contain at least one game")
    if workers < 1:
        raise ValueError("Arena must contain at least one worker")
    if workers == 1 and total_games > 1:
        raise ValueError(
            "Balanced batched Arena with more than one game requires at least two workers"
        )

    black_games = (total_games + 1) // 2
    white_games = total_games // 2
    black_workers = (workers + 1) // 2
    white_workers = workers - black_workers

    black_quotas = _split_quota(black_games, black_workers)
    white_quotas = _split_quota(white_games, white_workers)
    assignments: list[ArenaWorkerAssignment] = []

    for worker_id, quota in enumerate(black_quotas):
        assignments.append(ArenaWorkerAssignment(worker_id, quota, "black", (0, 1)))
    for offset, quota in enumerate(white_quotas):
        worker_id = black_workers + offset
        assignments.append(ArenaWorkerAssignment(worker_id, quota, "white", (1, 0)))

    if len(assignments) != workers:
        raise RuntimeError("Arena worker schedule did not cover every worker")
    if sum(item.quota for item in assignments) != total_games:
        raise RuntimeError("Arena worker quotas do not sum to the requested game count")
    scheduled_black = sum(item.quota for item in assignments if item.model_a_color == "black")
    scheduled_white = sum(item.quota for item in assignments if item.model_a_color == "white")
    if scheduled_black != black_games or scheduled_white != white_games:
        raise RuntimeError("Arena worker schedule does not satisfy the color-balance contract")
    return tuple(assignments)


class BalancedArenaSelfPlayAgent(SelfPlayAgent):
    """Checkpoint-Arena worker with deterministic color and game quota."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._is_arena:
            return
        if self.batch_size != 1:
            raise ValueError("Balanced checkpoint Arena requires exactly one active game per worker")
        assignments = arena_worker_assignments(self.args.gamesPerIteration, self.args.workers)
        assignment = assignments[int(self.id)]
        self._arena_game_quota = int(assignment.quota)
        # player_to_index[color] -> model index. Keep this mapping constant for
        # the worker so the parent process always interprets queued results with
        # the exact mapping that the child used during search.
        self.player_to_index = list(assignment.player_to_index)

    def run(self):
        if not self._is_arena:
            return super().run()
        try:
            np.random.seed()
            local_completed = 0
            while not self.stop_event.is_set() and local_completed < self._arena_game_quota:
                self._check_pause()
                sims = self._select_search_sims()
                for _ in range(sims):
                    if self.stop_event.is_set():
                        break
                    self.generateBatch()
                    if self.stop_event.is_set():
                        break
                    self.processBatch()
                if self.stop_event.is_set():
                    break

                # With checkpoint Arena batch_size=1, a finished game is
                # replaced by a new game object inside SelfPlayAgent.playMoves.
                # Object identity therefore gives an exact local completion
                # count without reading the racy global games_played counter.
                previous_game = self.games[0]
                self.playMoves()
                if self.games[0] is not previous_game:
                    local_completed += 1

            if local_completed != self._arena_game_quota and not self.stop_event.is_set():
                raise RuntimeError(
                    f"Arena worker {self.id} completed {local_completed} games, "
                    f"expected quota {self._arena_game_quota}"
                )
            with self.complete_count.get_lock():
                self.complete_count.value += 1
        except Exception:
            print(traceback.format_exc())


def _validate_color_balance(summary: dict[str, object], requested_games: int) -> None:
    by_color = summary.get("by_color")
    if not isinstance(by_color, dict):
        raise RuntimeError("Checkpoint Arena result is missing by-color statistics")
    black = by_color.get("black")
    white = by_color.get("white")
    if not isinstance(black, dict) or not isinstance(white, dict):
        raise RuntimeError("Checkpoint Arena result has invalid by-color statistics")
    black_games = int(black.get("games", -1))
    white_games = int(white.get("games", -1))
    expected_black = (int(requested_games) + 1) // 2
    expected_white = int(requested_games) // 2
    if black_games != expected_black or white_games != expected_white:
        raise RuntimeError(
            "Checkpoint Arena color-balance contract failed: "
            f"A-black={black_games}, A-white={white_games}, "
            f"expected {expected_black}/{expected_white}"
        )


def install_balanced_checkpoint_arena(module):
    """Install deterministic fixed-color workers into the checkpoint Arena module."""
    if getattr(module, "_gocube_balanced_arena_installed", False):
        return module

    original_coalesced = module._coalesced_batched_summary

    def balanced_coalesced(players, game_cls, eval_args, games, seed, wait_ms):
        summary = original_coalesced(players, game_cls, eval_args, games, seed, wait_ms)
        _validate_color_balance(summary, int(games))
        return summary

    module.SelfPlayAgent = BalancedArenaSelfPlayAgent
    module._coalesced_batched_summary = balanced_coalesced
    module._gocube_balanced_arena_installed = True
    return module
