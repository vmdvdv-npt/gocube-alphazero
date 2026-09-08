#!/usr/bin/env python3
"""Build or verify the model-independent B4 frozen Cube-4 suite."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphazero.envs.gocube.b_evaluation import (
    B_HELDOUT_DEPTH_SCHEDULE,
    B_HELDOUT_GENERATOR_MASTER_SEED,
    B_HELDOUT_POSITION_IDS,
    B_HELDOUT_SUITE_SHA256,
    B_HELDOUT_SUITE_POSITION_COUNT,
    canonical_json_bytes,
    canonical_suite_path,
    semantic_state_fingerprint,
    suite_payload_from_positions,
    authoritative_suite_game_class,
    validate_frozen_suite,
)


def generate_positions() -> list[dict[str, object]]:
    """Generate four independent positions at each prescribed depth."""

    game_cls = authoritative_suite_game_class()
    topology = game_cls.logical_topology()
    positions: list[dict[str, object]] = []
    for index, target_ply in enumerate(B_HELDOUT_DEPTH_SCHEDULE):
        rng = random.Random(B_HELDOUT_GENERATOR_MASTER_SEED + index)
        game = game_cls()
        actions: list[int] = []
        for ply in range(int(target_ply)):
            if game.win_state().any():
                raise RuntimeError(
                    f"Frozen suite generation reached terminal before {B_HELDOUT_POSITION_IDS[index]} "
                    f"at ply {ply}"
                )
            valid = game.valid_moves()
            legal = [
                action
                for action in range(int(topology.point_count))
                if bool(valid[action])
            ]
            if not legal:
                raise RuntimeError(
                    f"Frozen suite generation has no legal non-pass point move at ply {ply}"
                )
            # PointId order is the authoritative canonical action order.  In
            # this topology the action ID is the stable PointId index; using
            # numeric order avoids accidentally sorting the textual labels.
            legal.sort()
            action = int(legal[rng.randrange(len(legal))])
            game.play_action(action)
            actions.append(action)
        if game.win_state().any():
            raise RuntimeError(
                f"Frozen suite generation reached terminal at target {B_HELDOUT_POSITION_IDS[index]}"
            )
        state = game.semantic_state
        positions.append(
            {
                "position_id": B_HELDOUT_POSITION_IDS[index],
                "target_ply": int(target_ply),
                "actions": actions,
                "action_point_ids": [topology.point_id(action) for action in actions],
                "side_to_move": "black" if game.player == 0 else "white",
                "player_to_move": int(game.player),
                "phase": str(state.phase),
                "move_count": int(game.turns),
                "semantic_state_fingerprint": semantic_state_fingerprint(state),
            }
        )
    if len(positions) != B_HELDOUT_SUITE_POSITION_COUNT:
        raise RuntimeError("Frozen suite generator produced the wrong number of positions")
    return positions


def build_payload() -> dict[str, object]:
    return suite_payload_from_positions(generate_positions())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(canonical_suite_path()))
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and require byte-identical canonical output",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    expected_bytes = canonical_json_bytes(build_payload())
    if args.check:
        if not output.is_file():
            raise SystemExit(f"canonical frozen suite does not exist: {output}")
        actual = output.read_bytes()
        if actual != expected_bytes:
            raise SystemExit("canonical frozen suite is not byte-identical to deterministic generator output")
        # This also exercises every replay/fingerprint check and the pinned
        # SHA gate when --check is run against the committed artifact.
        validate_frozen_suite(output)
        print(f"OK: {output} sha256={B_HELDOUT_SUITE_SHA256}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(expected_bytes)
    temporary.replace(output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
