#!/usr/bin/env python3
"""Slow, fail-closed reference Arena for GoCube checkpoints.

The Golden Arena intentionally avoids alphazero.Arena, multiprocessing,
batched inference, queues, worker routing, and production win_state() result
accounting. It plays one game at a time and independently adjudicates every
terminal state with golden_scoring.py.

Scope: this is an independent Arena/result oracle around the existing MCTS and
network implementations. It is not an independent implementation of MCTS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from alphazero.GenericPlayers import MCTSPlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.golden_scoring import (
    BLACK,
    WHITE,
    GOLDEN_KOMI,
    GoldenOutcome,
    adjudicate_v3_terminal,
    assert_production_agreement,
)
from alphazero.envs.gocube.integration.contract import (
    ContractError,
    EVALUATION_SHARED_ARG_KEYS,
    contract_compatibility_differences,
    evaluation_argument_differences,
    evaluation_contract_differences,
    resolve_game_class_from_contract,
    resolve_model_contract,
    resolve_model_contract_from_metadata,
    resolve_semantic_game_class_from_contract,
)
from alphazero.envs.gocube.katago_v3 import (
    CLEANUP_1,
    CLEANUP_2,
    MAIN,
    KATAGO_JAPANESE_ADJUDICATOR_V3,
    NO_RESULT,
    PASS_ALIVE,
    SCORED,
    episode_move_limit,
)
from alphazero.envs.gocube.observation import GoCubeObservationAdapter
from alphazero.envs.gocube.production_contract import require_gocube_komi
from alphazero.search_contract import KATAGO_PINNED_SEARCH_UTILITY_MODE
from alphazero.utils import const_temp_scaling, dotdict, get_iter_file

GOLDEN_ARENA_SCHEMA_VERSION = 1
GOLDEN_ARENA_SIMS = 50
DEFAULT_SEED = 20260910


class GoldenArenaError(RuntimeError):
    pass


def _get(metadata, key, default=None):
    if hasattr(metadata, "get"):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _checkpoint_path(run_name: str, iteration: int) -> Path:
    return Path("checkpoint") / run_name / get_iter_file(int(iteration))


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "args" not in payload or "state_dict" not in payload:
        raise GoldenArenaError(
            f"Checkpoint does not contain saved args/state_dict: {path}"
        )
    return payload


def _resolve_checkpoint(saved_args, label: str):
    try:
        contract = resolve_model_contract_from_metadata(saved_args)
        model_game_cls = resolve_game_class_from_contract(contract)
        recomputed = resolve_model_contract(model_game_cls, saved_args)
    except (ContractError, TypeError, ValueError) as exc:
        raise GoldenArenaError(
            f"Checkpoint {label} has invalid model contract: {exc}"
        ) from exc

    differences = contract_compatibility_differences(
        contract,
        recomputed,
        legacy_game_cls=model_game_cls,
        legacy_args=saved_args,
    )
    if differences:
        field, (saved, current) = next(iter(differences.items()))
        raise GoldenArenaError(
            f"Checkpoint {label} contract mismatch for {field}: "
            f"saved={saved!r}, current={current!r}"
        )
    return contract, model_game_cls


def _require_compatible(
    contract_a,
    contract_b,
    saved_a,
    saved_b,
) -> None:
    try:
        require_gocube_komi(contract_a.komi, context="Golden Arena checkpoint A")
        require_gocube_komi(contract_b.komi, context="Golden Arena checkpoint B")
        require_gocube_komi(
            _get(saved_a, "gocube_komi", float("nan")),
            context="Golden Arena checkpoint A args",
        )
        require_gocube_komi(
            _get(saved_b, "gocube_komi", float("nan")),
            context="Golden Arena checkpoint B args",
        )
    except (TypeError, ValueError) as exc:
        raise GoldenArenaError(str(exc)) from exc

    if (
        contract_a.terminal_adjudicator_id != KATAGO_JAPANESE_ADJUDICATOR_V3
        or contract_b.terminal_adjudicator_id != KATAGO_JAPANESE_ADJUDICATOR_V3
    ):
        raise GoldenArenaError(
            "Golden Arena v1 supports only gocube-katago-japanese-v3 checkpoints"
        )

    contract_diffs = evaluation_contract_differences(contract_a, contract_b)
    if contract_diffs:
        field, (a, b) = next(iter(contract_diffs.items()))
        raise GoldenArenaError(
            f"Checkpoints cannot share one semantic game for {field}: A={a!r}, B={b!r}"
        )

    arg_diffs = evaluation_argument_differences(saved_a, saved_b)
    if arg_diffs:
        field, (a, b) = next(iter(arg_diffs.items()))
        raise GoldenArenaError(
            f"Checkpoints have different search setting {field}: A={a!r}, B={b!r}"
        )


_REQUIRED_MCTS_ARGS = (
    "cpuct",
    "fpu_reduction",
    "min_discount",
    "search_utility_mode",
)


def _canonical_search_args(saved_a, saved_b, game_cls) -> dotdict:
    """Build a whitelist-only reference search config.

    No unrelated training setting is copied from checkpoint A. Search settings
    are included only if they belong to the explicit evaluation contract and
    were proved equal between A and B.
    """

    differences = evaluation_argument_differences(saved_a, saved_b)
    if differences:
        field, (a, b) = next(iter(differences.items()))
        raise GoldenArenaError(
            f"Cannot build Golden search config: {field} differs ({a!r} != {b!r})"
        )

    values: dict[str, object] = {}
    missing_required = []
    for key in EVALUATION_SHARED_ARG_KEYS:
        a = _get(saved_a, key, None)
        b = _get(saved_b, key, None)
        if a is None and b is None:
            continue
        if a != b:
            raise GoldenArenaError(
                f"Golden search config requires equal {key}: A={a!r}, B={b!r}"
            )
        values[key] = a

    for key in _REQUIRED_MCTS_ARGS:
        if key not in values:
            missing_required.append(key)
    if missing_required:
        raise GoldenArenaError(
            "Golden search config is missing required checkpoint settings: "
            + ", ".join(missing_required)
        )

    if values["search_utility_mode"] != KATAGO_PINNED_SEARCH_UTILITY_MODE:
        raise GoldenArenaError(
            "Golden Arena v1 requires the pinned KataGo search utility contract"
        )

    # Root randomness and move randomness are disabled. The MCTS still has an
    # internal deterministic tie-order stream, seeded per pair below.
    values.update(
        {
            "_num_players": game_cls.num_players() + game_cls.has_draw(),
            "numMCTSSims": GOLDEN_ARENA_SIMS,
            "arenaMCTSSims": GOLDEN_ARENA_SIMS,
            "probFastSim": 0.0,
            "add_root_noise": False,
            "add_root_temp": False,
            "root_noise_frac": 0.0,
            "root_policy_temp": 0.0,
            "startTemp": 0.0,
            "arenaTemp": 0.0,
            "temp_scaling_fn": const_temp_scaling,
            "use_draws_for_winrate": True,
        }
    )
    return dotdict(values)


def golden_game_plan(game_id: int, base_seed: int) -> dict[str, int | str]:
    game_id = int(game_id)
    if game_id < 0:
        raise GoldenArenaError("game_id must be non-negative")
    pair_id = game_id // 2
    # A paired color swap shares one RNG seed. This makes the pairing explicit
    # rather than letting worker/scheduling topology choose random streams.
    game_seed = int(base_seed) + pair_id
    model_a_color = "black" if game_id % 2 == 0 else "white"
    return {
        "game_id": game_id,
        "pair_id": pair_id,
        "seed": game_seed,
        "model_a_color": model_a_color,
    }


def _seed_reference_process(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed) & 0x7FFFFFFF)


def _move_digest(actions: list[int]) -> str:
    encoded = ",".join(str(int(action)) for action in actions).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _result_for_model_a(outcome: GoldenOutcome, model_a_color: str) -> str:
    if outcome is GoldenOutcome.NO_RESULT:
        return "no_result"
    if outcome is GoldenOutcome.DRAW:
        return "draw"
    if model_a_color not in ("black", "white"):
        raise GoldenArenaError(f"Invalid model A color: {model_a_color!r}")
    return "win" if outcome.value == model_a_color else "loss"


def _signed_margin_for_model_a(adjudication, model_a_color: str):
    if adjudication.score is None:
        return None
    black_minus_white = adjudication.score.black - adjudication.score.white
    return (
        float(black_minus_white)
        if model_a_color == "black"
        else float(-black_minus_white)
    )


class GoldenSequentialArena:
    """One-process, one-game-at-a-time reference orchestration."""

    def __init__(self, players, game_cls, *, base_seed: int):
        if len(players) != 2 or int(game_cls.num_players()) != 2:
            raise GoldenArenaError("Golden Arena v1 requires exactly two players")
        if not bool(getattr(game_cls, "GOCUBE_V3", False)):
            raise GoldenArenaError("Golden Arena v1 requires a GoCube V3 game")
        require_gocube_komi(
            getattr(game_cls, "KOMI", None),
            context="Golden Arena semantic game",
        )
        self.players = list(players)
        self.game_cls = game_cls
        self.base_seed = int(base_seed)
        self.move_limit = int(episode_move_limit(game_cls.logical_topology()))

    def play_one(self, game_id: int) -> dict[str, object]:
        plan = golden_game_plan(game_id, self.base_seed)
        _seed_reference_process(int(plan["seed"]))

        for player in self.players:
            player.reset()

        state = self.game_cls()
        initial_semantic = state.semantic_state
        if any(int(value) != 0 for value in initial_semantic.board):
            raise GoldenArenaError("Golden Arena must start from an empty board")
        if tuple(getattr(initial_semantic, "captures", ())) != (0, 0):
            raise GoldenArenaError("Golden Arena must start with zero captures")
        if float(getattr(initial_semantic, "white_bonus_score", float("nan"))) != 0.0:
            raise GoldenArenaError("Golden Arena must start with zero whiteBonusScore")

        model_a_is_black = plan["model_a_color"] == "black"
        player_for_color = (0, 1) if model_a_is_black else (1, 0)
        actions: list[int] = []

        # These score-relevant values are reconstructed independently from
        # observed state transitions instead of trusting the production
        # counters that final_v3_score consumes.
        reference_captures = [0, 0]
        reference_white_bonus = 0.0
        reference_second_cleanup_start = None

        while getattr(state.semantic_state, "terminal_kind", None) is None:
            if len(actions) >= self.move_limit:
                raise GoldenArenaError(
                    "Golden Arena reached the runtime safety limit without a formal "
                    f"terminal result: game_id={game_id}, moves={len(actions)}, "
                    f"limit={self.move_limit}"
                )

            color = int(state.player)
            if color not in (0, 1):
                raise GoldenArenaError(f"Invalid side to move: {color}")
            model_index = int(player_for_color[color])

            action = int(self.players[model_index](state))
            valid = state.valid_moves()
            if action < 0 or action >= len(valid) or not bool(valid[action]):
                raise GoldenArenaError(
                    f"Model {model_index} selected illegal action {action} "
                    f"in game_id={game_id}, turn={state.turns}"
                )

            before = state.semantic_state
            before_board = tuple(int(value) for value in before.board)
            before_phase = str(before.phase)
            moving_color = color

            for player in self.players:
                player.update(state, action)
            state.play_action(action)
            actions.append(action)

            after = state.semantic_state
            after_board = tuple(int(value) for value in after.board)
            board_changed = before_board != after_board

            if board_changed:
                opponent_stone = WHITE if moving_color == 0 else BLACK
                removed = sum(
                    before_board[point] == opponent_stone
                    and after_board[point] != opponent_stone
                    for point in range(len(before_board))
                )
                reference_captures[moving_color] += int(removed)
                if before_phase in (MAIN, CLEANUP_1):
                    reference_white_bonus += 1.0 if moving_color == 0 else -1.0

            if before_phase == CLEANUP_1 and after.phase == CLEANUP_2:
                reference_second_cleanup_start = bytes(after_board)

            # Pass-alive early termination is scored using the current coloring
            # as the formal start-color snapshot, independently mirroring the
            # Rules-V3 contract without reading the production snapshot.
            if (
                after.terminal_kind == SCORED
                and getattr(after, "termination_reason", None) == PASS_ALIVE
            ):
                reference_second_cleanup_start = bytes(after_board)

        terminal_kind = getattr(state.semantic_state, "terminal_kind", None)
        if terminal_kind not in (SCORED, NO_RESULT):
            raise GoldenArenaError(
                f"Unknown terminal kind {terminal_kind!r} in game_id={game_id}"
            )

        semantic = state.semantic_state
        semantic_captures = tuple(int(value) for value in semantic.captures)
        if semantic_captures != tuple(reference_captures):
            raise GoldenArenaError(
                "Capture counter mismatch before scoring: "
                f"reference={tuple(reference_captures)}, production={semantic_captures}"
            )
        if float(semantic.white_bonus_score) != float(reference_white_bonus):
            raise GoldenArenaError(
                "whiteBonusScore mismatch before scoring: "
                f"reference={reference_white_bonus}, "
                f"production={semantic.white_bonus_score}"
            )

        semantic_second_start = getattr(
            semantic, "second_cleanup_start_colors", None
        )
        if semantic_second_start is not None:
            semantic_second_start = bytes(semantic_second_start)
        if semantic_second_start != reference_second_cleanup_start:
            raise GoldenArenaError(
                "CLEANUP_2 start-color snapshot mismatch before scoring"
            )

        # Crucially, no call to state.win_state() participates in result
        # accounting. The absolute color winner is recomputed from the raw
        # final board plus independently reconstructed score-relevant state.
        adjudication = adjudicate_v3_terminal(
            board=tuple(int(value) for value in semantic.board),
            adjacency=self.game_cls.logical_topology().neighbors_by_index,
            terminal_kind=terminal_kind,
            white_bonus_score=reference_white_bonus,
            second_cleanup_start_colors=reference_second_cleanup_start,
            captures=tuple(reference_captures),
            komi=GOLDEN_KOMI,
        )
        assert_production_agreement(state, adjudication)
        model_a_result = _result_for_model_a(
            adjudication.outcome,
            str(plan["model_a_color"]),
        )

        record: dict[str, object] = {
            **plan,
            "model_b_color": "white" if model_a_is_black else "black",
            "moves": len(actions),
            "actions": [int(action) for action in actions],
            "move_digest": _move_digest(actions),
            "final_board": [int(value) for value in semantic.board],
            "reference_score_inputs": {
                "captures": [int(value) for value in reference_captures],
                "white_bonus_score": float(reference_white_bonus),
                "second_cleanup_start_colors": (
                    None
                    if reference_second_cleanup_start is None
                    else [int(value) for value in reference_second_cleanup_start]
                ),
            },
            "terminal_kind": adjudication.terminal_kind,
            "absolute_outcome": adjudication.outcome.value,
            "model_a_result": model_a_result,
            "model_a_margin": _signed_margin_for_model_a(
                adjudication, str(plan["model_a_color"])
            ),
            "production_score_agreed": True,
        }

        if adjudication.score is not None:
            score = adjudication.score
            record["score"] = {
                "black": score.black,
                "white": score.white,
                "komi": score.komi,
                "margin": score.margin,
                "winner": score.outcome.value,
                "black_territory": len(score.area.black_territory),
                "white_territory": len(score.area.white_territory),
                "neutral": len(score.area.neutral),
                "seki": len(score.area.seki),
                "black_area": len(score.area.black_area),
                "white_area": len(score.area.white_area),
                "captures": list(score.captures),
            }
        else:
            record["score"] = None
        return record

    def play(self, games: int) -> dict[str, object]:
        games = int(games)
        if games < 2 or games % 2:
            raise GoldenArenaError(
                "Golden Arena requires an even number of games >= 2 for paired colors"
            )

        records = [self.play_one(game_id) for game_id in range(games)]
        results = [str(record["model_a_result"]) for record in records]
        absolute = [str(record["absolute_outcome"]) for record in records]

        by_color = {}
        for color in ("black", "white"):
            subset = [
                str(record["model_a_result"])
                for record in records
                if record["model_a_color"] == color
            ]
            by_color[color] = {
                "games": len(subset),
                "wins": subset.count("win"),
                "losses": subset.count("loss"),
                "draws": subset.count("draw"),
                "no_results": subset.count("no_result"),
            }

        return {
            "games": games,
            "wins": results.count("win"),
            "losses": results.count("loss"),
            "draws": results.count("draw"),
            "no_results": results.count("no_result"),
            "black_wins": absolute.count("black"),
            "white_wins": absolute.count("white"),
            "absolute_draws": absolute.count("draw"),
            "by_color": by_color,
            "records": records,
        }


def _load_network(game_cls, path: Path) -> NNetWrapper:
    return NNetWrapper.from_checkpoint(
        game_cls,
        folder=str(path.parent),
        filename=path.name,
        device="cpu",
        load_training_state=False,
    )


def _write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise GoldenArenaError(
            f"Refusing to overwrite Golden Arena evidence: {path}"
        ) from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Slow sequential GoCube Golden Arena with independent terminal scoring"
        )
    )
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--iteration-a", required=True, type=int)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--iteration-b", required=True, type=int)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.iteration_a < 0 or args.iteration_b < 0:
        parser.error("checkpoint iterations must be non-negative")
    if args.games < 2 or args.games % 2:
        parser.error("--games must be an even number >= 2")

    # Canonical Golden execution is deliberately CPU-only and single-threaded.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    path_a = _checkpoint_path(args.run_a, args.iteration_a)
    path_b = _checkpoint_path(args.run_b, args.iteration_b)
    payload_a = _load_payload(path_a)
    payload_b = _load_payload(path_b)
    saved_a, saved_b = payload_a["args"], payload_b["args"]

    contract_a, model_game_a = _resolve_checkpoint(saved_a, "A")
    contract_b, model_game_b = _resolve_checkpoint(saved_b, "B")
    _require_compatible(contract_a, contract_b, saved_a, saved_b)

    try:
        game_cls = resolve_semantic_game_class_from_contract(contract_a)
    except (ContractError, TypeError, ValueError) as exc:
        raise GoldenArenaError(
            f"Cannot resolve the authoritative semantic game: {exc}"
        ) from exc

    search_args = _canonical_search_args(saved_a, saved_b, game_cls)
    network_a = _load_network(model_game_a, path_a)
    network_b = _load_network(model_game_b, path_b)
    players = [
        MCTSPlayer(
            network_a,
            game_cls=game_cls,
            args=search_args,
            observation_adapter=GoCubeObservationAdapter(model_game_a),
        ),
        MCTSPlayer(
            network_b,
            game_cls=game_cls,
            args=search_args,
            observation_adapter=GoCubeObservationAdapter(model_game_b),
        ),
    ]

    started = time.perf_counter()
    summary = GoldenSequentialArena(
        players,
        game_cls,
        base_seed=int(args.seed),
    ).play(int(args.games))
    elapsed = time.perf_counter() - started

    output = {
        "schema_version": GOLDEN_ARENA_SCHEMA_VERSION,
        "reference": "gocube-golden-arena-v1",
        "scope": {
            "independent_terminal_scoring": True,
            "independent_result_accounting": True,
            "independent_arena_orchestration": True,
            "independent_mcts": False,
            "independent_network": False,
        },
        "run_a": args.run_a,
        "iteration_a": int(args.iteration_a),
        "checkpoint_a_sha256": _checkpoint_sha256(path_a),
        "run_b": args.run_b,
        "iteration_b": int(args.iteration_b),
        "checkpoint_b_sha256": _checkpoint_sha256(path_b),
        "seed": int(args.seed),
        "games": summary["games"],
        "wins": summary["wins"],
        "losses": summary["losses"],
        "draws": summary["draws"],
        "no_results": summary["no_results"],
        "black_wins": summary["black_wins"],
        "white_wins": summary["white_wins"],
        "absolute_draws": summary["absolute_draws"],
        "by_color": summary["by_color"],
        "records": summary["records"],
        "wall_time_seconds": elapsed,
        "golden_contract": {
            "komi": GOLDEN_KOMI,
            "sims": GOLDEN_ARENA_SIMS,
            "device": "cpu",
            "torch_threads": 1,
            "workers": 1,
            "batched": False,
            "fast_search": False,
            "root_noise": False,
            "root_temperature": False,
            "move_temperature": 0.0,
            "paired_color_swaps": True,
            "same_seed_within_pair": True,
            "terminal_adjudicator": KATAGO_JAPANESE_ADJUDICATOR_V3,
            "semantic_game_class": (
                f"{game_cls.__module__}.{game_cls.__qualname__}"
            ),
            "rules_fingerprint": contract_a.rules_fingerprint,
            "search_contract": contract_a.search_contract_id,
            "search_args": {
                key: value
                for key, value in search_args.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        },
    }

    if args.output:
        output_path = Path(args.output)
    else:
        safe_a = args.run_a.replace("/", "-")
        safe_b = args.run_b.replace("/", "-")
        output_path = Path("arena-results") / (
            f"golden-{safe_a}-i{args.iteration_a:04d}-vs-"
            f"{safe_b}-i{args.iteration_b:04d}-"
            f"g{args.games}-seed{args.seed}.json"
        )

    _write_new_json(output_path, output)
    print(
        "Golden Arena A vs B: "
        f"{summary['wins']}W/{summary['losses']}L/"
        f"{summary['draws']}D/{summary['no_results']}NR; "
        f"Black {summary['black_wins']} - White {summary['white_wins']}"
    )
    print(f"JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
