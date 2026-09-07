from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


# GoCube has one komi contract. It is deliberately not configurable.
# Persisted checkpoints/manifests may repeat the value for provenance, but
# runtime code must validate against this constant rather than inventing a
# local default.
GOCUBE_KOMI = 0.5


@dataclass(frozen=True)
class Cube4ProductionContract:
    topology: str = "cube"
    size: int = 4
    rule_set: str = "japanese"
    komi: float = GOCUBE_KOMI
    workers: int = 16
    regular_sims: int = 50
    fast_sims: int = 20
    games_per_iteration: int = 256
    train_batch_size: int = 1024
    arena_sims: int = 50

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


CUBE4_PRODUCTION = Cube4ProductionContract()


def require_gocube_komi(value: object, *, context: str = "GoCube") -> float:
    try:
        komi = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} komi must be {GOCUBE_KOMI}, got {value!r}") from exc
    if not math.isfinite(komi) or not math.isclose(
        komi, GOCUBE_KOMI, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{context} komi must be {GOCUBE_KOMI}, got {komi!r}")
    return komi


def validate_cube4_checkpoint_args(args: Mapping[str, object]) -> dict[str, object]:
    """Validate the immutable Cube-4 production fields in saved args.

    Sweep parameters are intentionally absent here. This function owns only the
    fixed game/search/training envelope that must never drift between branches.
    """

    contract = CUBE4_PRODUCTION
    observed = {
        "topology": args.get("gocube_topology"),
        "size": int(args.get("gocube_size", -1)),
        "rule_set": args.get("gocube_rule_set"),
        "komi": require_gocube_komi(args.get("gocube_komi"), context="checkpoint"),
        "regular_sims": int(args.get("numMCTSSims", -1)),
        "fast_sims": int(args.get("numFastSims", -1)),
    }
    expected = {
        "topology": contract.topology,
        "size": contract.size,
        "rule_set": contract.rule_set,
        "komi": contract.komi,
        "regular_sims": contract.regular_sims,
        "fast_sims": contract.fast_sims,
    }
    mismatches = {
        key: (observed[key], expected[key])
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        detail = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"Cube-4 production checkpoint contract mismatch: {detail}")
    return observed
