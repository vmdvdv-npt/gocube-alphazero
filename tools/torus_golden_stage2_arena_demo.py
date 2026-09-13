#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden import BadPlayer, GoodPlayer, SequentialGoldenArena


def main() -> int:
    arena = SequentialGoldenArena(master_seed=20260912)
    games = arena.play_pair(
        pair_id="golden-stage2-demo",
        player_A=GoodPlayer("A-GOOD"),
        player_B=BadPlayer("B-BAD"),
    )
    print("Golden Sequential Arena V1")
    print(f"pair_id: {games[0].pair_id}")
    for index, game in enumerate(games, start=1):
        print(f"Game {index}: A={'Black' if game.black_player == 'A' else 'White'} "
              f"B={'White' if game.black_player == 'A' else 'Black'}")
        print("trace:", [item.action for item in game.action_trace])
        print("absolute winner:", game.absolute_rule_result.value)
        print("mapped winner:", game.mapped_result.value)
        print("margin_black:", game.margin_black)
    print("summary:", arena.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
