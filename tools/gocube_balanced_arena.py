from __future__ import annotations

import traceback
from dataclasses import dataclass

from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.arena_bookkeeping import (
    arena_game_ids_by_worker,
    model_a_color_for_game_id,
    player_to_index_for_game_id,
)
from alphazero.envs.gocube.reproducibility import derive_worker_seed, seed_process


@dataclass(frozen=True)
class ArenaWorkerAssignment:
    worker_id: int
    quota: int
    game_ids: tuple[int, ...]

    @property
    def model_a_colors(self) -> tuple[str, ...]:
        return tuple(model_a_color_for_game_id(game_id) for game_id in self.game_ids)

    @property
    def player_to_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(player_to_index_for_game_id(game_id) for game_id in self.game_ids)

    @property
    def model_a_color(self) -> str:
        """Compatibility summary; active games use the per-game property."""

        colors = set(self.model_a_colors)
        if not colors:
            return "mixed"
        return next(iter(colors)) if len(colors) == 1 else "mixed"

    @property
    def player_to_index(self) -> tuple[int, int]:
        """Compatibility summary for callers that only have one game."""

        return self.player_to_indices[0] if self.player_to_indices else (0, 1)

def arena_worker_assignments(total_games: int, workers: int) -> tuple[ArenaWorkerAssignment, ...]:
    """Return a deterministic per-game schedule partitioned into worker quotas."""
    total_games = int(total_games)
    workers = int(workers)
    if total_games < 1:
        raise ValueError("Arena must contain at least one game")
    if workers < 1:
        raise ValueError("Arena must contain at least one worker")
    schedules = arena_game_ids_by_worker(total_games, workers)
    assignments = tuple(
        ArenaWorkerAssignment(worker_id, len(game_ids), tuple(game_ids))
        for worker_id, game_ids in enumerate(schedules)
    )
    if sum(item.quota for item in assignments) != total_games:
        raise RuntimeError("Arena worker quotas do not sum to the requested game count")
    return assignments


class BalancedArenaSelfPlayAgent(SelfPlayAgent):
    """Checkpoint-Arena worker with deterministic per-game quota and colors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._is_arena:
            return
        if self._arena_fixed_schedule:
            self._arena_game_quota = len(self._arena_game_ids)
        else:
            assignments = arena_worker_assignments(
                self.args.gamesPerIteration, self.args.workers
            )
            assignment = assignments[int(self.id)]
            self._arena_game_quota = int(assignment.quota)

    def run(self):
        if not self._is_arena:
            return super().run()
        try:
            master_seed = int(getattr(self.args, "gocube_arena_seed", 0))
            seed_process(derive_worker_seed(master_seed, self.iteration, int(self.id), 0, 0))
            local_completed = 0
            while (
                not self.stop_event.is_set()
                and local_completed < self._arena_game_quota
                and self._arena_active_slots
            ):
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

                completed_before = self._arena_completed_count
                self.playMoves()
                local_completed += self._arena_completed_count - completed_before

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
