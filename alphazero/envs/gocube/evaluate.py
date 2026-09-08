import argparse
import os
from pathlib import Path

import pyximport

pyximport.install()

from alphazero.Arena import Arena
from alphazero.GenericPlayers import MCTSPlayer
from alphazero.envs.gocube.evaluation import (
    load_evaluation_checkpoint,
    prepare_evaluation_args,
)
from alphazero.envs.gocube.integration.contract import (
    ContractError,
    resolve_contract_for_descriptor,
    resolve_game_class_from_contract,
)
from alphazero.envs.gocube.observation import GoCubeObservationAdapter
from alphazero.envs.gocube.integration.manifest import load_run_manifest
from alphazero.utils import get_iter_file

from tools.gocube_checkpoint_arena_complete import (
    ARENA_SIMS,
    _authoritative_game_class,
    _load_network,
    _load_payload,
    _require_compatible_contracts,
    _resolve_checkpoint_contract,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate two GoCube AlphaZero checkpoints without training")
    parser.add_argument("--topology", choices=("torus", "cube"), default="cube")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--candidate", type=int, default=5)
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sims", type=int, default=ARENA_SIMS)
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
    except Exception as exc:
        raise ValueError(f"Cannot resolve evaluation contract: missing/invalid run manifest: {exc}") from exc
    if manifest.topology != topology or manifest.size != size:
        raise ValueError("CLI topology/size does not match run manifest")
    try:
        contract = resolve_contract_for_descriptor(manifest)
        return resolve_game_class_from_contract(contract)
    except ContractError as exc:
        raise ValueError(f"Cannot resolve evaluation model contract: {exc}") from exc


def main():
    cli = parse_args()
    validate_cli(cli)
    if cli.sims != ARENA_SIMS:
        raise ValueError(f"GoCube Arena requires exactly {ARENA_SIMS} simulations")
    folder = os.path.join(cli.checkpoint_dir, cli.run_name)
    manifest = load_run_manifest(folder)
    if manifest.topology != cli.topology or manifest.size != cli.size:
        raise ValueError("CLI topology/size does not match run manifest")
    candidate_path = Path(folder) / get_iter_file(cli.candidate)
    baseline_path = Path(folder) / get_iter_file(cli.baseline)
    candidate_payload = _load_payload(candidate_path)
    baseline_payload = _load_payload(baseline_path)
    candidate_contract, candidate_game_cls = _resolve_checkpoint_contract(
        candidate_payload["args"], "candidate"
    )
    baseline_contract, baseline_game_cls = _resolve_checkpoint_contract(
        baseline_payload["args"], "baseline"
    )
    _require_compatible_contracts(
        candidate_contract,
        baseline_contract,
        candidate_payload["args"],
        baseline_payload["args"],
    )
    game_cls = _authoritative_game_class(candidate_contract)
    candidate = _load_network(candidate_game_cls, candidate_path, "cpu")
    baseline = _load_network(baseline_game_cls, baseline_path, "cpu")
    args = prepare_arena_args(candidate.args, game_cls, cli.sims)
    args.probFastSim = 0.0
    args.add_root_noise = False
    args.add_root_temp = False
    args.startTemp = 0.0
    args.arenaTemp = 0.0
    players = [
        MCTSPlayer(
            candidate,
            game_cls,
            args,
            observation_adapter=GoCubeObservationAdapter(candidate_game_cls),
        ),
        MCTSPlayer(
            baseline,
            game_cls,
            args,
            observation_adapter=GoCubeObservationAdapter(baseline_game_cls),
        ),
    ]
    arena = Arena(players, game_cls, use_batched_mcts=False, args=args)
    wins, draws, winrates = arena.play_games(cli.games, shuffle_players=True)
    print()
    print(f"candidate iteration {cli.candidate}: {wins[0]} wins ({winrates[0]:.3f} score rate)")
    print(f"baseline iteration {cli.baseline}: {wins[1]} wins ({winrates[1]:.3f} score rate)")
    print(f"draws: {draws}")
    print(f"no-result: {arena.no_results}")


if __name__ == "__main__":
    main()
