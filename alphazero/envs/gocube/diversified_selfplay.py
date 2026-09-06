from __future__ import annotations

import numpy as np

from alphazero.envs.gocube.pinned_selfplay import PinnedSelfPlayAgent, _optional_arg
from alphazero.envs.gocube.selfplay_semantics import CLEANUP_1, CLEANUP_2


KATAGO_PINNED_DIVERSIFICATION_DEFAULTS = {
    "early_fork_game_prob": 0.04,
    "early_fork_game_expected_move_prop": 0.025,
    "fork_game_prob": 0.01,
    "fork_game_min_choices": 3,
    "early_fork_game_max_choices": 12,
    "fork_game_max_choices": 36,
    "init_games_with_policy": True,
    "policy_init_area_prop": 0.04,
    "policy_init_gamma_shape": 1.0,
    "policy_init_temperature": 1.0,
    "plain_fork_pool_capacity": 1000,
}

_SETUP_POLICY_INIT = "pinned_policy_init"
_SETUP_EARLY_FORK = "pinned_early_fork"
_SETUP_ORDINARY_FORK = "pinned_ordinary_fork"
_SETUP_PHASES = {_SETUP_POLICY_INIT, _SETUP_EARLY_FORK, _SETUP_ORDINARY_FORK}


def sample_policy_init_moves(rng, *, point_count: int, area_prop: float, gamma_shape: float) -> int:
    mean = float(point_count) * float(area_prop)
    if mean <= 0.0:
        return 0
    return int(np.floor(rng.gamma(float(gamma_shape), mean / float(gamma_shape))))


class DiversifiedPinnedSelfPlayAgent(PinnedSelfPlayAgent):
    """Pinned self-play plus KataGo early/plain forks and policy-initialized starts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        defaults = KATAGO_PINNED_DIVERSIFICATION_DEFAULTS
        self.early_fork_game_prob = float(_optional_arg(
            self.args, "gocube_early_fork_game_prob", defaults["early_fork_game_prob"]
        ))
        self.early_fork_game_expected_move_prop = float(_optional_arg(
            self.args,
            "gocube_early_fork_game_expected_move_prop",
            defaults["early_fork_game_expected_move_prop"],
        ))
        self.fork_game_prob = float(_optional_arg(
            self.args, "gocube_fork_game_prob", defaults["fork_game_prob"]
        ))
        self.fork_game_min_choices = int(_optional_arg(
            self.args, "gocube_fork_game_min_choices", defaults["fork_game_min_choices"]
        ))
        self.early_fork_game_max_choices = int(_optional_arg(
            self.args,
            "gocube_early_fork_game_max_choices",
            defaults["early_fork_game_max_choices"],
        ))
        self.fork_game_max_choices = int(_optional_arg(
            self.args, "gocube_fork_game_max_choices", defaults["fork_game_max_choices"]
        ))
        self.init_games_with_policy = bool(_optional_arg(
            self.args, "gocube_init_games_with_policy", defaults["init_games_with_policy"]
        ))
        self.policy_init_area_prop = float(_optional_arg(
            self.args, "gocube_policy_init_area_prop", defaults["policy_init_area_prop"]
        ))
        self.policy_init_gamma_shape = float(_optional_arg(
            self.args, "gocube_policy_init_gamma_shape", defaults["policy_init_gamma_shape"]
        ))
        self.policy_init_temperature = float(_optional_arg(
            self.args, "gocube_policy_init_temperature", defaults["policy_init_temperature"]
        ))
        self.plain_fork_pool_capacity = int(_optional_arg(
            self.args,
            "gocube_plain_fork_pool_capacity",
            defaults["plain_fork_pool_capacity"],
        ))
        self._validate_diversification_config()
        self._diverse_setup_mode = [None] * self.batch_size
        self._diverse_setup_choices = [0] * self.batch_size
        self._diverse_start_metadata = [None] * self.batch_size
        self._diverse_deferred_cleanup = [False] * self.batch_size

    def _validate_diversification_config(self):
        for name, value in (
            ("gocube_early_fork_game_prob", self.early_fork_game_prob),
            ("gocube_fork_game_prob", self.fork_game_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0,1]")
        if self.early_fork_game_expected_move_prop < 0.0:
            raise ValueError("gocube_early_fork_game_expected_move_prop must be non-negative")
        if self.fork_game_min_choices < 1:
            raise ValueError("gocube_fork_game_min_choices must be positive")
        if self.early_fork_game_max_choices < self.fork_game_min_choices:
            raise ValueError("early fork max choices must be >= fork min choices")
        if self.fork_game_max_choices < self.fork_game_min_choices:
            raise ValueError("fork max choices must be >= fork min choices")
        if self.policy_init_area_prop < 0.0:
            raise ValueError("gocube_policy_init_area_prop must be non-negative")
        if self.policy_init_gamma_shape <= 0.0:
            raise ValueError("gocube_policy_init_gamma_shape must be positive")
        if self.policy_init_temperature <= 0.0:
            raise ValueError("gocube_policy_init_temperature must be positive")
        if self.plain_fork_pool_capacity < 1:
            raise ValueError("gocube_plain_fork_pool_capacity must be positive")

    def _configure_pinned_game(self, index: int) -> bool:
        game = self.games[index]
        if hasattr(game, "configure_diversification"):
            game.configure_diversification(
                early_fork_prob=self.early_fork_game_prob,
                ordinary_fork_prob=self.fork_game_prob,
                early_expected_move_prop=self.early_fork_game_expected_move_prop,
                pool_capacity=self.plain_fork_pool_capacity,
            )
        used_seki = super()._configure_pinned_game(index)
        if used_seki and hasattr(self.games[index], "mark_diversified_training_start"):
            self.games[index].mark_diversified_training_start()
        return used_seki

    def _cleanup_is_eligible(self, index: int) -> bool:
        if (
            not getattr(self, "score_aware", False)
            or self._is_arena
            or self._is_warmup
            or self.cleanup_training_prob <= 0.0
        ):
            return False
        state = getattr(self.games[index], "semantic_state", None)
        return state is not None and getattr(state, "phase", None) == "main" and getattr(state, "terminal_kind", None) is None

    def _sample_cleanup_flag(self, index: int) -> bool:
        return self._cleanup_is_eligible(index) and np.random.random_sample() < self.cleanup_training_prob

    def _schedule_cleanup_plan(self, index: int) -> None:
        state = getattr(self.games[index], "semantic_state", None)
        if state is None or getattr(state, "phase", None) != "main" or getattr(state, "terminal_kind", None) is not None:
            return
        phase = CLEANUP_1 if np.random.random_sample() < 0.5 else CLEANUP_2
        point_count = int(self.game_cls.logical_topology().point_count)
        mean = float(point_count) * self.cleanup_training_prelude_area_prop
        moves = 0 if mean <= 0.0 else int(np.floor(np.random.gamma(
            self.cleanup_training_gamma_shape,
            mean / self.cleanup_training_gamma_shape,
        )))
        self._cleanup_slot_set("cleanup_training_phase", index, phase)
        self._cleanup_slot_set("cleanup_training_moves_left", index, moves)
        self._cleanup_slot_set("cleanup_training_prelude_total", index, moves)
        if hasattr(self.games[index], "set_plain_fork_generation_suppressed"):
            self.games[index].set_plain_fork_generation_suppressed(True)
        if moves <= 0:
            self._start_cleanup_training(index)

    def _begin_setup(self, index: int, *, phase: str, moves: int, metadata: dict, choices: int = 0) -> None:
        self._cleanup_slot_set("cleanup_training_phase", index, phase)
        self._cleanup_slot_set("cleanup_training_moves_left", index, int(moves))
        self._cleanup_slot_set("cleanup_training_prelude_total", index, int(moves))
        self._cleanup_slot_set("root_policy_cache", index, None)
        self._diverse_setup_mode[index] = phase
        self._diverse_setup_choices[index] = int(choices)
        self._diverse_start_metadata[index] = dict(metadata)
        self.histories[index] = []
        self.temps[index] = self.args.startTemp
        self.mcts[index] = self._get_mcts()
        self.next_reset[index] = 0
        if hasattr(self.games[index], "set_plain_fork_generation_suppressed"):
            self.games[index].set_plain_fork_generation_suppressed(True)

    def _sample_cleanup_training_plan(self, index):
        # Fork/policy setup happens first, while cleanup is sampled independently
        # and begins only after setup has completed.
        self._cancel_cleanup_training_plan(index)
        self._cleanup_slot_set("cleanup_training_metadata", index, None)
        self._diverse_setup_mode[index] = None
        self._diverse_setup_choices[index] = 0
        self._diverse_start_metadata[index] = None
        self._diverse_deferred_cleanup[index] = False

        used_seki = self._configure_pinned_game(index)
        deferred_cleanup = self._sample_cleanup_flag(index)
        self._diverse_deferred_cleanup[index] = deferred_cleanup

        if used_seki:
            if deferred_cleanup:
                self._schedule_cleanup_plan(index)
            return

        game = self.games[index]
        fork = game.maybe_start_plain_fork() if hasattr(game, "maybe_start_plain_fork") else None
        if fork is not None:
            mode = str(fork["mode"])
            depth = int(fork["fork_depth"])
            if mode == "early_fork":
                phase = _SETUP_EARLY_FORK
                max_choices = self.early_fork_game_max_choices
                self._telemetry_add("early_forks")
            else:
                phase = _SETUP_ORDINARY_FORK
                max_choices = self.fork_game_max_choices
                self._telemetry_add("ordinary_forks")
            choices = int(np.random.randint(self.fork_game_min_choices, max_choices + 1))
            self._telemetry_add("fork_depth_sum", depth)
            self._telemetry_add("fork_depth_count", 1)
            self._begin_setup(
                index,
                phase=phase,
                moves=1,
                choices=choices,
                metadata={"mode": mode, "fork_move_number": depth, "setup_moves": 1},
            )
            return

        point_count = int(self.game_cls.logical_topology().point_count)
        policy_moves = 0
        if self.init_games_with_policy:
            policy_moves = sample_policy_init_moves(
                np.random,
                point_count=point_count,
                area_prop=self.policy_init_area_prop,
                gamma_shape=self.policy_init_gamma_shape,
            )
        if policy_moves > 0:
            self._telemetry_add("policy_initialized_starts")
            self._begin_setup(
                index,
                phase=_SETUP_POLICY_INIT,
                moves=policy_moves,
                metadata={"mode": "policy_initialized", "setup_moves": int(policy_moves)},
            )
            return

        self._telemetry_add("normal_starts")
        if hasattr(game, "mark_diversified_training_start"):
            game.mark_diversified_training_start()
        if deferred_cleanup:
            self._schedule_cleanup_plan(index)

    def _cleanup_prelude_policy(self, index, fallback_policy):
        phase = self.cleanup_training_phase[index]
        if phase not in _SETUP_PHASES:
            return super()._cleanup_prelude_policy(index, fallback_policy)

        raw = self.root_policy_cache[index]
        if raw is None:
            raw = fallback_policy
        policy = np.asarray(raw, dtype=np.float64).reshape(-1).copy()
        valid = np.asarray(self.games[index].valid_moves(), dtype=np.uint8).reshape(-1)
        if policy.size != valid.size:
            return None
        policy[valid == 0] = 0.0
        policy[policy < 0.0] = 0.0
        legal = np.flatnonzero(valid)
        if legal.size == 0:
            return None

        if phase in (_SETUP_EARLY_FORK, _SETUP_ORDINARY_FORK):
            num_choices = min(int(self._diverse_setup_choices[index]), int(legal.size))
            if num_choices <= 0:
                return None
            candidates = np.random.choice(legal, size=num_choices, replace=False)
            best = int(candidates[int(np.argmax(policy[candidates]))])
            result = np.zeros_like(policy)
            result[best] = 1.0
            return result

        inv_temp = 1.0 / self.policy_init_temperature
        policy = np.power(policy, inv_temp)
        total = float(policy.sum())
        if not np.isfinite(total) or total <= 0.0:
            return None
        return policy / total

    def _reset_discarded_setup_game(self, index: int) -> None:
        self.games[index] = self.game_cls()
        self.histories[index] = []
        self.temps[index] = self.args.startTemp
        self.mcts[index] = self._get_mcts()
        self.next_reset[index] = 0
        self._cleanup_slot_set("cleanup_training_phase", index, None)
        self._cleanup_slot_set("cleanup_training_moves_left", index, 0)
        self._cleanup_slot_set("cleanup_training_prelude_total", index, 0)
        self._cleanup_slot_set("cleanup_training_metadata", index, None)
        self._cleanup_slot_set("root_policy_cache", index, None)
        if getattr(self, "recording_enabled", False):
            self.game_ids[index] = None
            self.move_histories[index] = []
            self.game_start_times[index] = None
        self._sample_cleanup_training_plan(index)

    def _finish_diversified_setup(self, index: int) -> bool:
        game = self.games[index]
        if game.win_state().any():
            # KataGo discards a fork/init setup that already ended the game.
            self._reset_discarded_setup_game(index)
            return True

        self._cleanup_slot_set("cleanup_training_phase", index, None)
        self._cleanup_slot_set("cleanup_training_moves_left", index, 0)
        self._cleanup_slot_set("cleanup_training_prelude_total", index, 0)
        self._cleanup_slot_set("root_policy_cache", index, None)
        self.histories[index] = []
        self.temps[index] = self.args.startTemp
        self.mcts[index] = self._get_mcts()
        self.next_reset[index] = 0
        if hasattr(game, "mark_diversified_training_start"):
            game.mark_diversified_training_start()
        metadata = self._diverse_start_metadata[index]
        if metadata is not None:
            state = getattr(game, "semantic_state", None)
            metadata = dict(metadata)
            metadata.update({
                "initial_player": int(game.player),
                "initial_board": [int(v) for v in np.asarray(state.board).reshape(-1)] if state is not None else None,
            })
            self._diverse_start_metadata[index] = metadata
            self._cleanup_slot_set("cleanup_training_metadata", index, metadata)
        if self._diverse_deferred_cleanup[index]:
            self._schedule_cleanup_plan(index)
        return True

    def _start_cleanup_training(self, index):
        phase = self.cleanup_training_phase[index]
        if phase in _SETUP_PHASES:
            return self._finish_diversified_setup(index)
        result = super()._start_cleanup_training(index)
        if result:
            metadata = self._diverse_start_metadata[index]
            if metadata:
                cleanup_meta = self.cleanup_training_metadata[index] or {}
                cleanup_meta = dict(cleanup_meta)
                cleanup_meta["source_start"] = dict(metadata)
                self._cleanup_slot_set("cleanup_training_metadata", index, cleanup_meta)
            if hasattr(self.games[index], "mark_diversified_training_start"):
                self.games[index].mark_diversified_training_start()
        return result
