from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping, Any

from .state import BASELINE_KOMI, RULES_PROFILE_ID, STAGE0_RULES_FINGERPRINT
from .topology import TORUS_5X5_TOPOLOGY_FINGERPRINT

ARENA_CONTRACT_ID = "golden-arena-search-v1"
SEARCH_IMPLEMENTATION_ID = "golden-sequential-puct-v1"
SEARCH_PATH = "B"
LEGACY_ARENA_WATCHDOG = 500
LEGACY_ARENA_BOARD_SIZE = 5
ARENA_WATCHDOG_SCALING_ID = "board-size-linear-v1"


def resolve_arena_watchdog(
    board_size: int | tuple[int, int],
    *,
    legacy_limit: int = LEGACY_ARENA_WATCHDOG,
    legacy_board_size: int = LEGACY_ARENA_BOARD_SIZE,
    extra_row_budget: int = 125,
) -> int:
    """Resolve the deterministic Arena technical boundary for a board.

    The frozen 5x5 Golden Arena remains 500 plies.  Larger boards receive
    125 additional plies per extra board row/column, using the larger
    dimension for rectangular boards.  This is intentionally a technical
    fail-closed boundary, not a result or a draw rule.
    """

    if isinstance(board_size, bool):
        raise ValueError("Arena board size must be a positive integer or pair")
    if isinstance(board_size, int):
        extent = board_size
    else:
        if len(board_size) != 2:
            raise ValueError("Arena board size pair must contain width and height")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in board_size):
            raise ValueError("Arena board dimensions must be positive integers")
        extent = max(board_size)
    if extent <= 0:
        raise ValueError("Arena board dimensions must be positive integers")
    for name, value in (
        ("legacy_limit", legacy_limit),
        ("legacy_board_size", legacy_board_size),
        ("extra_row_budget", extra_row_budget),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return legacy_limit + max(0, extent - legacy_board_size) * extra_row_budget


GOLDEN_MOVE_LIMIT = resolve_arena_watchdog(LEGACY_ARENA_BOARD_SIZE)


@dataclass(frozen=True)
class SearchSettings:
    simulations: int = 64
    cpuct: float = 1.25
    fpu: float = 0.0
    root_noise: bool = False
    fast_search: bool = False
    resign: bool = False
    root_policy_temperature: bool = False
    move_temperature: float = 0.0
    deterministic_tie_break: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.simulations, bool)
            or not isinstance(self.simulations, int)
            or self.simulations <= 0
        ):
            raise ValueError("Golden search simulations must be a positive integer")
        if not math.isfinite(float(self.cpuct)) or self.cpuct <= 0:
            raise ValueError("Golden search cpuct must be finite and positive")
        if not math.isfinite(float(self.fpu)):
            raise ValueError("Golden search FPU must be finite")
        if self.root_noise:
            raise ValueError("Golden Arena contract requires root noise OFF")
        if self.fast_search:
            raise ValueError("Golden Arena contract requires fast search OFF")
        if self.resign:
            raise ValueError("Golden Arena contract requires resign OFF")
        if self.root_policy_temperature:
            raise ValueError("Golden Arena contract requires root policy temperature OFF")
        if float(self.move_temperature) != 0.0:
            raise ValueError("Golden Arena contract requires move temperature 0")

    def evidence(self) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(asdict(self).items()))


def search_contract_payload(
    settings: SearchSettings | None = None,
) -> dict[str, object]:
    effective = settings or SearchSettings()
    return {
        "arena_contract_id": ARENA_CONTRACT_ID,
        "search_implementation_id": SEARCH_IMPLEMENTATION_ID,
        "search_settings": asdict(effective),
        "move_limit": GOLDEN_MOVE_LIMIT,
        "execution": {
            "batching": False,
            "workers": False,
            "production_arena_reuse": False,
        },
    }


def compute_search_contract_fingerprint(
    settings: SearchSettings | None = None,
) -> str:
    payload = json.dumps(
        search_contract_payload(settings),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


SEARCH_CONTRACT_FINGERPRINT = (
    "sha256:c13d3159e123865f4f091dc667ce94458ec9e6002e076ab82d78cffe8c46df1c"
)
if compute_search_contract_fingerprint() != SEARCH_CONTRACT_FINGERPRINT:
    raise RuntimeError("Golden Arena search contract fingerprint constant drifted")


@dataclass(frozen=True)
class GoldenArenaContract:
    contract_id: str = ARENA_CONTRACT_ID
    rules_id: str = RULES_PROFILE_ID
    rules_fingerprint: str = STAGE0_RULES_FINGERPRINT
    topology_fingerprint: str = TORUS_5X5_TOPOLOGY_FINGERPRINT
    komi: float = BASELINE_KOMI
    move_limit: int = GOLDEN_MOVE_LIMIT
    search: SearchSettings = SearchSettings()

    def __post_init__(self) -> None:
        if self.contract_id != ARENA_CONTRACT_ID:
            raise ValueError(f"Unsupported Golden Arena contract {self.contract_id!r}")
        if self.rules_id != RULES_PROFILE_ID:
            raise ValueError("Golden Arena rules profile drift")
        if self.rules_fingerprint != STAGE0_RULES_FINGERPRINT:
            raise ValueError("Golden Arena rules fingerprint drift")
        if self.topology_fingerprint != TORUS_5X5_TOPOLOGY_FINGERPRINT:
            raise ValueError("Golden Arena topology fingerprint drift")
        if float(self.komi) != BASELINE_KOMI:
            raise ValueError("Production Golden Arena contract komi must be 0.5")
        if self.move_limit != GOLDEN_MOVE_LIMIT:
            raise ValueError("Golden Arena execution watchdog must be exactly 500 actions")
        if self.search != SearchSettings():
            raise ValueError("Golden Arena V1 search settings are fixed by the Arena contract")

    def evidence(self) -> tuple[tuple[str, object], ...]:
        return (
            ("contract_id", self.contract_id),
            ("rules_id", self.rules_id),
            ("rules_fingerprint", self.rules_fingerprint),
            ("topology_fingerprint", self.topology_fingerprint),
            ("komi", self.komi),
            ("move_limit", self.move_limit),
            ("search_contract_fingerprint", compute_search_contract_fingerprint(self.search)),
            ("search", self.search.evidence()),
        )


# A checkpoint may describe model semantics, but these settings are Arena-owned.
_CHECKPOINT_FORBIDDEN_ARENA_KEYS = frozenset(
    {
        "arena_sims",
        "arena_simulations",
        "num" + "M" + "C" + "T" + "S" + "Sims",
        "cpuct",
        "fpu",
        "root_noise",
        "dirichlet_noise",
        "fast_search",
        "probFastSim",
        "move_temperature",
        "temperature",
        "root_policy_temperature",
        "resign",
    }
)


def reject_checkpoint_arena_overrides(metadata: Mapping[str, Any] | None) -> None:
    if not metadata:
        return
    found = sorted(key for key in metadata if key in _CHECKPOINT_FORBIDDEN_ARENA_KEYS)
    nested = metadata.get("arena") if isinstance(metadata, Mapping) else None
    if isinstance(nested, Mapping):
        found.extend(f"arena.{key}" for key in sorted(nested))
    if found:
        raise ValueError(
            "Checkpoint/train metadata is not allowed to override Golden Arena contract: "
            + ", ".join(found)
        )


DEFAULT_ARENA_CONTRACT = GoldenArenaContract()
