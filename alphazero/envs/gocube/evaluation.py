from __future__ import annotations

from alphazero.NNetWrapper import NNetWrapper
from alphazero.utils import get_iter_file


def prepare_evaluation_args(saved_args, game_cls, sims):
    args = saved_args.copy()
    args.numMCTSSims = sims
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    args.add_root_noise = False
    args.add_root_temp = False
    args.startTemp = 0.0
    args.arenaTemp = 0.0
    return args


def load_evaluation_checkpoint(game_cls, folder, iteration, *, device=None):
    return NNetWrapper.from_checkpoint(
        game_cls,
        folder=folder,
        filename=get_iter_file(iteration),
        device=device,
        load_training_state=False if device is not None else True,
    )
