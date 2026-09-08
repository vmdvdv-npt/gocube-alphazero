from __future__ import annotations

import operator

import numpy as np

from alphazero.envs.gocube.core import BLACK, EMPTY, PLAYING, WHITE
from alphazero.envs.gocube.evaluation import prepare_evaluation_args
from alphazero.envs.gocube.katago_v3 import RESULT_PROVENANCE_RUNTIME

from .catalog import CheckpointDescriptor
from .contract import ContractError, resolve_contract_for_descriptor, resolve_game_class_from_contract
from .errors import GenerationFailed


def serialize_action(game_cls, action: int) -> dict[str, object]:
    if action == game_cls.pass_action():
        return {"type": "pass"}
    return {"type": "place", "pointId": game_cls.point_id_for_action(action)}


def captured_point_ids(
    game_cls,
    pre_board: np.ndarray,
    post_board: np.ndarray,
    current_player: int,
) -> list[str]:
    opponent_stone = WHITE if current_player == 0 else BLACK
    topology = game_cls.logical_topology()
    return [
        topology.point_id(index)
        for index in range(topology.point_count)
        if int(pre_board[index]) == opponent_stone and int(post_board[index]) == EMPTY
    ]


def serialize_terminal(
    terminal,
    *,
    cleanup_move_count: int = 0,
    adjudicator_id: str | None = None,
) -> dict[str, object]:
    payload = {
        "winner": terminal.winner,
        "adjudicatorId": adjudicator_id or getattr(terminal, "adjudicator_id", None),
        "fallbackCount": getattr(terminal, "fallback_count", 0),
        "unresolvedCount": getattr(terminal, "unresolved_count", 0),
        "cleanupMoveCount": cleanup_move_count,
        "noResult": terminal.no_result,
        "terminationReason": getattr(terminal, "termination_reason", getattr(terminal, "reason", None)),
        "resultProvenance": getattr(terminal, "result_provenance", None),
        "runtimeForced": getattr(terminal, "result_provenance", None) == RESULT_PROVENANCE_RUNTIME,
        "score": None,
    }
    score = terminal.score
    if score is None:
        return payload

    payload["score"] = {
        "ruleSet": score.ruleset,
        "black": score.black,
        "white": score.white,
        "komi": score.komi,
        "winner": score.winner,
        "margin": score.margin,
        "captures": list(score.captures),
        "prisoners": None if score.prisoners is None else list(score.prisoners),
        "territory": {
            "black": score.territory.black,
            "white": score.territory.white,
            "neutral": score.territory.neutral,
            "seki": score.territory.seki,
        },
        "stonesOnBoard": {
            "black": score.stones_on_board.black,
            "white": score.stones_on_board.white,
        },
        "deadStones": {
            "black": score.dead_stones.black,
            "white": score.dead_stones.white,
        },
    }
    return payload


def _default_player_factory(model, game_cls, args):
    import pyximport

    pyximport.install()
    from alphazero.GenericPlayers import MCTSPlayer

    return MCTSPlayer(model, game_cls, args)


class GameGenerator:
    def __init__(self, *, player_factory=None, game_cls_resolver=None):
        self.player_factory = player_factory or _default_player_factory
        self.game_cls_resolver = game_cls_resolver

    def generate(
        self,
        *,
        black: CheckpointDescriptor,
        white: CheckpointDescriptor,
        black_model,
        white_model,
        mcts_sims: int,
    ) -> dict[str, object]:
        try:
            black_contract = resolve_contract_for_descriptor(black)
            white_contract = resolve_contract_for_descriptor(white)
            differences = black_contract.differences(white_contract)
            if differences:
                field, (black_value, white_value) = next(iter(differences.items()))
                raise GenerationFailed(
                    f"Game model contracts differ for {field}: "
                    f"black={black_value!r}, white={white_value!r}"
                )
            game_cls = (
                self.game_cls_resolver(black.topology, black.size, black.rule_set)
                if self.game_cls_resolver is not None
                else resolve_game_class_from_contract(black_contract)
            )
        except GenerationFailed:
            raise
        except ContractError as exc:
            raise GenerationFailed(f"Cannot resolve generation model contract: {exc}") from exc
        black_args = prepare_evaluation_args(black_model.args, game_cls, mcts_sims)
        white_args = prepare_evaluation_args(white_model.args, game_cls, mcts_sims)
        players = [
            self.player_factory(black_model, game_cls, black_args),
            self.player_factory(white_model, game_cls, white_args),
        ]
        for player in players:
            player.reset()

        game = game_cls()
        moves: list[dict[str, object]] = []
        cleanup_move_count = 0

        while True:
            winstate = game.win_state()
            if winstate.any():
                break

            pre_state = game.semantic_state
            current_player = pre_state.current_player
            pre_board = np.asarray(pre_state.board).copy()

            try:
                action = operator.index(players[current_player](game))
            except Exception as exc:
                raise GenerationFailed(f"MCTS player failed to choose an action: {exc}") from exc

            if action < 0 or action >= game_cls.action_size():
                raise GenerationFailed(f"MCTS returned out-of-range action {action}")
            valids = game.valid_moves()
            if int(valids[action]) != 1:
                raise GenerationFailed(f"MCTS returned illegal action {action}")

            for player in players:
                player.update(game, action)

            try:
                game.play_action(action)
            except Exception as exc:
                raise GenerationFailed(f"GoGame.play_action({action}) failed: {exc}") from exc

            # Cleanup is an internal proof phase for Japanese territory scoring,
            # not part of the player-facing game record. Keep the two main-phase
            # passes in replay, then suppress service cleanup placements/passes.
            # Historical GoState calls the main phase ``playing``; the
            # KataGo V3 state calls it ``main``.  Both are player-facing.
            if pre_state.phase not in (PLAYING, "main"):
                cleanup_move_count += 1
                continue

            post_board = np.asarray(game.semantic_state.board)
            captured = (
                []
                if action == game_cls.pass_action()
                else captured_point_ids(game_cls, pre_board, post_board, current_player)
            )
            moves.append(
                {
                    "moveNumber": len(moves) + 1,
                    "color": "black" if current_player == 0 else "white",
                    "action": serialize_action(game_cls, action),
                    "captured": captured,
                }
            )

        terminal = game.terminal_adjudication
        if terminal is None:
            raise GenerationFailed("Game reached a winning state without terminal adjudication")

        return {
            "topology": black.topology,
            "size": black.size,
            "ruleSet": black.rule_set,
            "komi": black.komi,
            "terminalAdjudicator": black.terminal_adjudicator,
            "mctsSims": mcts_sims,
            "black": {"checkpointId": black.checkpoint_id},
            "white": {"checkpointId": white.checkpoint_id},
            "moves": moves,
            "result": serialize_terminal(
                terminal,
                cleanup_move_count=cleanup_move_count,
                adjudicator_id=black.terminal_adjudicator,
            ),
        }
