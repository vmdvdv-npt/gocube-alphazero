from __future__ import annotations

from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.production_contract import require_gocube_komi
from alphazero.utils import get_iter_file


def prepare_evaluation_args(saved_args, game_cls, sims):
    require_gocube_komi(game_cls.KOMI, context="GoCube evaluation")
    saved_komi = (
        saved_args.get("gocube_komi")
        if hasattr(saved_args, "get")
        else getattr(saved_args, "gocube_komi", None)
    )
    if saved_komi is not None:
        require_gocube_komi(saved_komi, context="Saved GoCube evaluation args")
    args = saved_args.copy()
    args.numMCTSSims = sims
    args.arenaMCTSSims = sims
    args.probFastSim = 0.0
    args._num_players = game_cls.num_players() + game_cls.has_draw()
    args.add_root_noise = False
    args.add_root_temp = False
    args.startTemp = 0.0
    args.arenaTemp = 0.0
    return args


def load_evaluation_checkpoint(game_cls, folder, iteration, *, device=None):
    require_gocube_komi(game_cls.KOMI, context="GoCube evaluation")
    model = NNetWrapper.from_checkpoint(
        game_cls,
        folder=folder,
        filename=get_iter_file(iteration),
        device=device,
        load_training_state=False if device is not None else True,
    )
    saved_args = getattr(model, "args", None)
    saved_komi = (
        saved_args.get("gocube_komi")
        if hasattr(saved_args, "get")
        else getattr(saved_args, "gocube_komi", None)
    ) if saved_args is not None else None
    if saved_komi is not None:
        require_gocube_komi(saved_komi, context="Saved GoCube evaluation args")
    return model
