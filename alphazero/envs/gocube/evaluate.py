import argparse
import os

import pyximport

pyximport.install()

from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.envs.gocube.evaluation import load_evaluation_checkpoint, prepare_evaluation_args
from alphazero.envs.gocube.game import game_class, legacy_game_class
from alphazero.envs.gocube.integration.manifest import load_run_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate two GoCube AlphaZero checkpoints without training")
    parser.add_argument("--topology", choices=("torus", "cube"), default="cube")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--candidate", type=int, default=5)
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sims", type=int, default=20)
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    return parser.parse_args()


def validate_cli(cli):
    if cli.candidate < 0 or cli.baseline < 0:
        raise ValueError("checkpoint iterations must be non-negative")
    if cli.games < 2 or cli.games % 2:
        raise ValueError("games must be a positive even number of at least 2")
    if cli.sims < 1:
        raise ValueError("sims must be at least 1")


def prepare_arena_args(saved_args, game_cls, sims):
    return prepare_evaluation_args(saved_args, game_cls, sims)


def load_checkpoint(game_cls, folder, iteration):
    return load_evaluation_checkpoint(game_cls, folder, iteration)


def resolve_game_class(checkpoint_dir, run_name, topology, size):
    folder = os.path.join(checkpoint_dir, run_name)
    try:
        manifest = load_run_manifest(folder)
    except Exception:
        return game_class(topology, size)
    if manifest.topology != topology or manifest.size != size:
        raise ValueError("CLI topology/size does not match run manifest")
    return legacy_game_class(manifest.topology, manifest.size, manifest.terminal_adjudicator)


def main():
    cli = parse_args()
    validate_cli(cli)
    game_cls = resolve_game_class(cli.checkpoint_dir, cli.run_name, cli.topology, cli.size)
    folder = os.path.join(cli.checkpoint_dir, cli.run_name)
    candidate = load_checkpoint(game_cls, folder, cli.candidate)
    baseline = load_checkpoint(game_cls, folder, cli.baseline)
    args = prepare_arena_args(candidate.args, game_cls, cli.sims)
    players = [MCTSPlayer(candidate, game_cls, args), MCTSPlayer(baseline, game_cls, args)]
    arena = Arena(players, game_cls, use_batched_mcts=False, args=args)
    wins, draws, winrates = arena.play_games(cli.games, shuffle_players=True)
    print()
    print(f"candidate iteration {cli.candidate}: {wins[0]} wins ({winrates[0]:.3f} score rate)")
    print(f"baseline iteration {cli.baseline}: {wins[1]} wins ({winrates[1]:.3f} score rate)")
    print(f"draws: {draws}")


if __name__ == "__main__":
    main()
