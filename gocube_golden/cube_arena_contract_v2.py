"""Caller-owned Cube V2 Arena scientific settings."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from .arena_contract import SEARCH_IMPLEMENTATION_ID, SearchSettings

CUBE_ARENA_CONTRACT_ID = "gocube-cube-arena-v2"
CUBE_ARENA_SCHEMA_VERSION = 1
CUBE_ARENA_RESULT_SCHEMA = "gocube-cube-arena-result-v2"
CUBE_ARENA_RESULT_SCHEMA_VERSION = 1


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CubeArenaSearchConfig:
    """Resolved scientific Arena parameters; no Golden-value whitelist."""

    simulations: int
    cpuct: float
    fpu: float
    watchdog: int
    deterministic_tie_break: bool = True

    def validate(self) -> None:
        if isinstance(self.simulations, bool) or not isinstance(self.simulations, int) or self.simulations <= 0:
            raise ValueError("Cube Arena simulations must be a positive integer")
        if not math.isfinite(float(self.cpuct)) or float(self.cpuct) <= 0.0:
            raise ValueError("Cube Arena cpuct must be positive and finite")
        if not math.isfinite(float(self.fpu)):
            raise ValueError("Cube Arena FPU must be finite")
        if isinstance(self.watchdog, bool) or not isinstance(self.watchdog, int) or self.watchdog <= 0:
            raise ValueError("Cube Arena watchdog must be a positive integer")
        if self.deterministic_tie_break is not True:
            raise ValueError("Cube Arena requires deterministic tie breaking")

    @property
    def search_settings(self) -> SearchSettings:
        self.validate()
        return SearchSettings(
            simulations=int(self.simulations),
            cpuct=float(self.cpuct),
            fpu=float(self.fpu),
            deterministic_tie_break=True,
        )

    def identity_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "contract_id": CUBE_ARENA_CONTRACT_ID,
            "schema_version": CUBE_ARENA_SCHEMA_VERSION,
            "search_implementation_id": SEARCH_IMPLEMENTATION_ID,
            "simulations": int(self.simulations),
            "cpuct": float(self.cpuct),
            "fpu": float(self.fpu),
            "noise": False,
            "temperature": 0.0,
            "fast_search": False,
            "resign": False,
            "watchdog": int(self.watchdog),
            "paired_starts_color_swap": True,
            "opening": "cube-evaluation-starts-v1",
            "empty_board_control_pair": True,
            "deterministic_tie_break": True,
            "technical_fail_closed": True,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.identity_payload())


__all__ = [
    "CUBE_ARENA_CONTRACT_ID",
    "CUBE_ARENA_SCHEMA_VERSION",
    "CUBE_ARENA_RESULT_SCHEMA",
    "CUBE_ARENA_RESULT_SCHEMA_VERSION",
    "CubeArenaSearchConfig",
]
