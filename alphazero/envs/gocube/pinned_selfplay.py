from __future__ import annotations

import numpy as np

from alphazero.SelfPlayAgent import SelfPlayAgent

from .selfplay_semantics import KATAGO_PINNED_SELFPLAY_DEFAULTS, sample_policy_init_moves


POLICY_INIT_PRELUDE = "policy_init"
FORK_PRELUDE = "plain_fork"


def _optional_arg(args, name, default):
    if hasattr(args, "get"):
        return args.get(name, default)
    try:
        return getattr(args, name)
    except (AttributeError, KeyError):
        return default


class PinnedSelfPlayAgent(SelfPlayAgent):
    """Pinned KataGo self-play semantics plus fork/policy-init diversification."""

    def _configure_pinned_game(self, index: int):
        if self._is_arena or self._is_warmup or not getattr(self, "score_aware", False):
            return None

        game = self.games[index]
        if not hasattr(game, "configure_pinned_selfplay"):
            raise RuntimeError("Pinned KataGo self-play requires a pinned GoCube game class")

        defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
        auto_end_probability = float(_optional_arg(
            self.args, "gocube_pass_alive_auto_end_probability",
            defaults["pass_alive_auto_end_probability"],
        ))
        seki_probability = float(_optional_arg(
            self.args, "gocube_seki_fork_hack_probability",
            defaults["seki_fork_hack_probability"],
        ))
        early_probability = float(_optional_arg(
            self.args, "gocube_early_fork_game_probability",
            defaults["early_fork_game_probability"],
        ))
        ordinary_probability = float(_optional_arg(
            self.args, "gocube_fork_game_probability",
            defaults["fork_game_probability"],
        ))
        for name, value in (
            ("gocube_pass_alive_auto_end_probability", auto_end_probability),
            ("gocube_seki_fork_hack_probability", seki_probability),
            ("gocube_early_fork_game_probability", early_probability),
            ("gocube_fork_game_probability", ordinary_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0,1]")

        root_prune = bool(_optional_arg(
            self.args, "gocube_root_prune_useless_moves",
            defaults["root_prune_useless_moves"],
        ))
        expected_prop = float(_optional_arg(
            self.args, "gocube_early_fork_expected_move_prop",
            defaults["early_fork_expected_move_prop"],
        ))
        min_choices = int(_optional_arg(
            self.args, "gocube_fork_game_min_choices",
            defaults["fork_game_min_choices"],
        ))
        early_max = int(_optional_arg(
            self.args, "gocube_early_fork_game_max_choices",
            defaults["early_fork_game_max_choices"],
        ))
        ordinary_max = int(_optional_arg(
            self.args, "gocube_fork_game_max_choices",
            defaults["fork_game_max_choices"],
        ))
        if expected_prop < 0.0:
            raise ValueError("gocube_early_fork_expected_move_prop must be non-negative")
        if min_choices < 1 or early_max < min_choices or ordinary_max < min_choices:
            raise ValueError("fork candidate choice ranges are invalid")

        game.configure_pinned_selfplay(
            auto_end_pass_alive=np.random.random_sample() < auto_end_probability,
            root_prune_useless_moves=root_prune,
            seki_fork_hack_prob=seki_probability,
            early_fork_game_prob=early_probability,
            early_fork_expected_move_prop=expected_prop,
            fork_game_prob=ordinary_probability,
            fork_game_min_choices=min_choices,
            early_fork_game_max_choices=early_max,
            fork_game_max_choices=ordinary_max,
        )

        start = None
        if game.maybe_start_seki_fork(seki_probability):
            start = {"kind": "seki"}
        else:
            plain = game.maybe_start_plain_fork() if hasattr(game, "maybe_start_plain_fork") else None
            if plain is not None:
                start = {"kind": "plain", **plain}

        if start is not None:
            self.histories[index] = []
            self.temps[index] = self.args.startTemp
            self.mcts[index] = self._get_mcts()
            self.next_reset[index] = 0
            if hasattr(self, "root_policy_cache"):
                self.root_policy_cache[index] = None
        return start

    def _set_setup_prelude(self, index, *, mode, moves, metadata):
        self._cleanup_slot_set("cleanup_training_phase", index, mode)
        self._cleanup_slot_set("cleanup_training_moves_left", index, int(moves))
        self._cleanup_slot_set("cleanup_training_prelude_total", index, int(moves))
        self._cleanup_slot_set("cleanup_training_metadata", index, dict(metadata))
        self._cleanup_slot_set("root_policy_cache", index, None)

    def _sample_cleanup_training_plan(self, index):
        # KataGo picks any stored initial/fork position first. Cleanup training
        # is then sampled independently. Policy-init is disabled for fork games.
        start = self._configure_pinned_game(index)
        super()._sample_cleanup_training_plan(index)
        if self._cleanup_training_active(index):
            return
        if self._is_arena or self._is_warmup or not getattr(self, "score_aware", False):
            return

        if start is not None and start.get("kind") == "seki":
            return
        if start is not None and start.get("kind") == "plain":
            mode = str(start["mode"])
            depth = int(start["depth"])
            self._telemetry_add("early_forks" if mode == "early" else "ordinary_forks")
            self._telemetry_add("fork_depth_sum", depth)
            self._telemetry_add("fork_depth_count", 1)
            self._set_setup_prelude(
                index,
                mode=FORK_PRELUDE,
                moves=1,
                metadata={"mode": "plain_fork", "fork_mode": mode, "fork_depth": depth},
            )
            return

        self._telemetry_add("normal_starts")
        defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
        init_enabled = bool(_optional_arg(
            self.args, "gocube_init_games_with_policy", defaults["init_games_with_policy"]
        ))
        if not init_enabled:
            return
        area_prop = float(_optional_arg(
            self.args, "gocube_policy_init_area_prop", defaults["policy_init_area_prop"]
        ))
        gamma_shape = float(_optional_arg(
            self.args, "gocube_policy_init_gamma_shape", defaults["policy_init_gamma_shape"]
        ))
        if area_prop < 0.0 or gamma_shape <= 0.0:
            raise ValueError("policy-init area/gamma parameters are invalid")
        moves = sample_policy_init_moves(
            np.random, int(self.game_cls.logical_topology().point_count), area_prop, gamma_shape
        )
        if moves <= 0:
            return
        self._telemetry_add("policy_initialized_starts")
        self._set_setup_prelude(
            index,
            mode=POLICY_INIT_PRELUDE,
            moves=moves,
            metadata={"mode": "policy_init", "prelude_moves": moves},
        )

    def _setup_policy(self, index, fallback_policy, temperature, *, fork_mode=None):
        caches = getattr(self, "root_policy_cache", None)
        raw = caches[index] if caches is not None and index < len(caches) else None
        if raw is None:
            raw = fallback_policy
        policy = np.asarray(raw, dtype=np.float64).reshape(-1).copy()
        valid = np.asarray(self.games[index].valid_moves(), dtype=np.uint8).reshape(-1)
        if policy.size != valid.size:
            self._cancel_cleanup_training_plan(index)
            return None
        policy[valid == 0] = 0.0
        policy[policy < 0.0] = 0.0

        if fork_mode is not None:
            defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
            minimum = int(_optional_arg(
                self.args, "gocube_fork_game_min_choices", defaults["fork_game_min_choices"]
            ))
            maximum = int(_optional_arg(
                self.args,
                "gocube_early_fork_game_max_choices" if fork_mode == "early" else "gocube_fork_game_max_choices",
                defaults["early_fork_game_max_choices"] if fork_mode == "early" else defaults["fork_game_max_choices"],
            ))
            legal = np.flatnonzero(valid)
            if legal.size <= 0:
                self._cancel_cleanup_training_plan(index)
                return None
            count = min(legal.size, int(np.random.randint(minimum, maximum + 1)))
            chosen = np.random.choice(legal, size=count, replace=False)
            mask = np.zeros_like(policy)
            mask[chosen] = 1.0
            policy *= mask

        if float(policy.sum()) <= 0.0:
            self._cancel_cleanup_training_plan(index)
            return None
        inv_temp = 1.0 / float(temperature)
        policy = np.power(policy, inv_temp)
        total = float(policy.sum())
        if not np.isfinite(total) or total <= 0.0:
            self._cancel_cleanup_training_plan(index)
            return None
        return policy / total

    def _cleanup_prelude_policy(self, index, fallback_policy):
        phase = self.cleanup_training_phase[index]
        defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
        if phase == POLICY_INIT_PRELUDE:
            temperature = float(_optional_arg(
                self.args, "gocube_policy_init_temperature", defaults["policy_init_temperature"]
            ))
            return self._setup_policy(index, fallback_policy, temperature)
        if phase == FORK_PRELUDE:
            metadata = self.cleanup_training_metadata[index] or {}
            return self._setup_policy(
                index,
                fallback_policy,
                defaults["policy_init_temperature"],
                fork_mode=metadata.get("fork_mode", "ordinary"),
            )
        return super()._cleanup_prelude_policy(index, fallback_policy)

    def _start_cleanup_training(self, index):
        phase = self.cleanup_training_phase[index]
        if phase in (POLICY_INIT_PRELUDE, FORK_PRELUDE):
            state = getattr(self.games[index], "semantic_state", None)
            if state is None or getattr(state, "terminal_kind", None) is not None or getattr(state, "phase", None) != "main":
                self._cancel_cleanup_training_plan(index)
                return False
            # These setup moves are real BoardHistory/V3 transitions and remain
            # in ko/pass/cycle history, but are not MCTS training targets.
            self.histories[index] = []
            self.temps[index] = self.args.startTemp
            self.mcts[index] = self._get_mcts()
            self.next_reset[index] = 0
            self._cleanup_slot_set("root_policy_cache", index, None)
            self._cleanup_slot_set("cleanup_training_phase", index, None)
            self._cleanup_slot_set("cleanup_training_moves_left", index, 0)
            return True

        old_game = self.games[index]
        config = old_game.pinned_selfplay_config() if hasattr(old_game, "pinned_selfplay_config") else None
        result = super()._start_cleanup_training(index)
        if result and config is not None:
            self.games[index].configure_pinned_selfplay(
                auto_end_pass_alive=bool(config["auto_end_pass_alive"]),
                root_prune_useless_moves=bool(config["root_prune_useless_moves"]),
                seki_fork_hack_prob=float(config["seki_fork_hack_prob"]),
                started_from_seki_fork=bool(config["started_from_seki_fork"]),
                started_from_plain_fork=bool(config.get("started_from_plain_fork", False)),
                early_fork_game_prob=float(config["early_fork_game_prob"]),
                early_fork_expected_move_prop=float(config["early_fork_expected_move_prop"]),
                fork_game_prob=float(config["fork_game_prob"]),
                fork_game_min_choices=int(config["fork_game_min_choices"]),
                early_fork_game_max_choices=int(config["early_fork_game_max_choices"]),
                fork_game_max_choices=int(config["fork_game_max_choices"]),
            )
        return result
