from __future__ import annotations

import time

from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.utils import get_iter_file


def prepare_evaluation_args(
    saved_args,
    game_cls,
    sims,
    *,
    arena_batch_size=16,
    arena_workers=1,
    cuda=None,
):
    if int(sims) < 1:
        raise ValueError("sims must be at least 1")
    if int(arena_batch_size) < 2:
        raise ValueError("arena_batch_size must be at least 2")
    if int(arena_workers) < 1:
        raise ValueError("arena_workers must be at least 1")

    args = saved_args.copy()
    args.numMCTSSims = int(sims)
    args.arenaMCTSSims = int(sims)
    args.probFastSim = 0.0
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    args.add_root_noise = False
    args.add_root_temp = False
    args.startTemp = 0.0
    args.arenaTemp = 0.0
    args.arenaBatched = True
    args.arena_batch_size = int(arena_batch_size)
    args.workers = int(arena_workers)
    args.use_draws_for_winrate = True
    if cuda is not None:
        args.cuda = bool(cuda)
    return args


def load_evaluation_checkpoint(game_cls, folder, iteration, *, device=None):
    return NNetWrapper.from_checkpoint(
        game_cls,
        folder=folder,
        filename=get_iter_file(iteration),
        device=device,
        load_training_state=False if device is not None else True,
    )


def play_balanced_batched_match(candidate, reference, game_cls, args, games):
    """Play an exactly color-balanced two-player match using batched Arena MCTS.

    The match is split into two concurrent-game halves. In the first half the
    candidate is Black; in the second half the player list is reversed so the
    candidate is White. `shuffle_players=False` is intentional and is honored
    by the batched Arena path.
    """
    games = int(games)
    if game_cls.num_players() != 2:
        raise ValueError("balanced batched GoCube evaluation currently requires two players")
    if games < 2 or games % 2:
        raise ValueError("games must be a positive even number of at least 2")

    half = games // 2

    def run_half(first, second):
        half_args = args.copy()
        half_args.gamesPerIteration = half
        half_args.arena_batch_size = min(int(args.arena_batch_size), half)
        # Avoid launching more worker processes than there are concurrent game
        # slots to feed them. One worker with batch 16 already gives 16 games in
        # flight and one neural-network inference batch per search step.
        half_args.workers = min(int(args.workers), half)
        players = [
            MCTSPlayer(first, game_cls=game_cls, args=half_args),
            MCTSPlayer(second, game_cls=game_cls, args=half_args),
        ]
        arena = Arena(players, game_cls, use_batched_mcts=True, args=half_args)
        wins, draws, _ = arena.play_games(half, verbose=False, shuffle_players=False)
        return wins, int(draws), int(arena.no_results)

    started = time.time()
    black_wins, black_draws, black_no_results = run_half(candidate, reference)
    white_wins, white_draws, white_no_results = run_half(reference, candidate)
    elapsed = time.time() - started

    candidate_wins = int(black_wins[0]) + int(white_wins[1])
    reference_wins = int(black_wins[1]) + int(white_wins[0])
    draws = black_draws + white_draws
    no_results = black_no_results + white_no_results

    return {
        "wins": [candidate_wins, reference_wins],
        "draws": draws,
        "no_results": no_results,
        "candidate_black_games": half,
        "candidate_white_games": half,
        "arena_batch_size": min(int(args.arena_batch_size), half),
        "arena_workers": min(int(args.workers), half),
        "elapsed_seconds": elapsed,
    }
