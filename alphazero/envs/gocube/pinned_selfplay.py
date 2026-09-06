from __future__ import annotations

import numpy as np

from alphazero.SelfPlayAgent import SelfPlayAgent

from .selfplay_semantics import KATAGO_PINNED_SELFPLAY_DEFAULTS


def _optional_arg(args, name, default):
    if hasattr(args, "get"):
        return args.get(name, default)
    try:
        return getattr(args, name)
    except (AttributeError, KeyError):
        return default


class PinnedSelfPlayAgent(SelfPlayAgent):
    """Self-play agent that adds the remaining pinned KataGo game-level semantics."""

    def _configure_pinned_game(self, index: int) -> bool:
        if self._is_arena or self._is_warmup or not getattr(self, "score_aware", False):
            return False

        game = self.games[index]
        if not hasattr(game, "configure_pinned_selfplay"):
            raise RuntimeError("Pinned KataGo self-play requires a pinned GoCube game class")

        defaults = KATAGO_PINNED_SELFPLAY_DEFAULTS
        auto_end_probability = float(_optional_arg(
            self.args,
            "gocube_pass_alive_auto_end_probability",
            defaults["pass_alive_auto_end_probability"],
        ))
        if not 0.0 <= auto_end_probability <= 1.0:
            raise ValueError("gocube_pass_alive_auto_end_probability must be within [0,1]")

        root_prune = bool(_optional_arg(
            self.args,
            "gocube_root_prune_useless_moves",
            defaults["root_prune_useless_moves"],
        ))
        seki_probability = float(_optional_arg(
            self.args,
            "gocube_seki_fork_hack_probability",
            defaults["seki_fork_hack_probability"],
        ))
        if not 0.0 <= seki_probability <= 1.0:
            raise ValueError("gocube_seki_fork_hack_probability must be within [0,1]")

        game.configure_pinned_selfplay(
            auto_end_pass_alive=np.random.random_sample() < auto_end_probability,
            root_prune_useless_moves=root_prune,
            seki_fork_hack_prob=seki_probability,
        )
        used_seki_fork = game.maybe_start_seki_fork(seki_probability)
        if used_seki_fork:
            # The blank-game MCTS has no useful state to retain. Reset every
            # per-game search/training accumulator before searching the fork.
            self.histories[index] = []
            self.temps[index] = self.args.startTemp
            self.mcts[index] = self._get_mcts()
            self.next_reset[index] = 0
            if hasattr(self, "root_policy_cache"):
                self.root_policy_cache[index] = None
        return used_seki_fork

    def _sample_cleanup_training_plan(self, index):
        # KataGo chooses any initial/fork position first, then independently
        # applies its 4% cleanup/encore-training path when eligible.
        self._configure_pinned_game(index)
        return super()._sample_cleanup_training_plan(index)

    def _start_cleanup_training(self, index):
        # Base cleanup rebasing constructs a new game object. Preserve the
        # per-game 98%/2%, root-prune, and seki-fork selections across that clear.
        old_game = self.games[index]
        config = old_game.pinned_selfplay_config() if hasattr(old_game, "pinned_selfplay_config") else None
        result = super()._start_cleanup_training(index)
        if result and config is not None:
            self.games[index].configure_pinned_selfplay(
                auto_end_pass_alive=bool(config["auto_end_pass_alive"]),
                root_prune_useless_moves=bool(config["root_prune_useless_moves"]),
                seki_fork_hack_prob=float(config["seki_fork_hack_prob"]),
                started_from_seki_fork=bool(config["started_from_seki_fork"]),
            )
        return result
