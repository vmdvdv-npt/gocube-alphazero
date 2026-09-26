"""Arena profile registry.

Profiles own scientific game/topology semantics. Execution concurrency and
batching are run-owned ArenaEngine parameters and are not Golden whitelists.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.arena_engine import ArenaExecutionConfig, ArenaProfile


class _RunOwnedExecutionMixin:
    def validate_execution_config(self, config: ArenaExecutionConfig) -> None:
        config.validate_base()


class _RunOwnedTorus9ProfileMixin(_RunOwnedExecutionMixin):
    """Remove historical komi-candidate whitelist from Torus semantics."""

    def __init__(
        self,
        *,
        komi: float = 0.5,
        profile_id: str = "torus9",
        simulations: int = 64,
        cpuct: float = 1.25,
        fpu: float = 0.0,
        watchdog: int = 500,
        five_channel: bool = False,
    ) -> None:
        if isinstance(komi, bool) or not isinstance(komi, (int, float)) or not math.isfinite(float(komi)):
            raise ValueError("Torus9 Arena komi must be a finite number")
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
            raise ValueError("Torus9 Arena simulations must be a positive integer")
        from tools.arena_profiles.torus9 import TORUS9_POINT_COUNT
        self.komi = float(komi)
        self.profile_id = str(profile_id)
        self.simulations = int(simulations)
        if not math.isfinite(float(cpuct)) or float(cpuct) <= 0:
            raise ValueError("Torus9 Arena cpuct must be finite and positive")
        if not math.isfinite(float(fpu)):
            raise ValueError("Torus9 Arena fpu must be finite")
        if isinstance(watchdog, bool) or not isinstance(watchdog, int) or watchdog <= 0:
            raise ValueError("Torus9 Arena watchdog must be a positive integer")
        self.cpuct = float(cpuct)
        self.fpu = float(fpu)
        self.watchdog = int(watchdog)
        self._five_channel = bool(five_channel)
        self.observation_shape = (5 if self._five_channel else 6, TORUS9_POINT_COUNT)


def _torus_profile(
    *,
    komi: float = 0.5,
    simulations: int = 64,
    cpuct: float = 1.25,
    fpu: float = 0.0,
    watchdog: int = 500,
    five_channel: bool = False,
    profile_id: str = "torus9",
) -> ArenaProfile:
    from tools.arena_profiles.torus9 import Torus9ArenaProfile

    class RunOwnedTorus9ArenaProfile(_RunOwnedTorus9ProfileMixin, Torus9ArenaProfile):
        pass

    return RunOwnedTorus9ArenaProfile(
        komi=komi,
        profile_id=profile_id,
        simulations=simulations,
        cpuct=cpuct,
        fpu=fpu,
        watchdog=watchdog,
        five_channel=five_channel,
    )


def _cube_profile(value: str) -> ArenaProfile:
    from tools.arena_profiles.cube_v2 import CubeV2ArenaProfile

    parsed = CubeV2ArenaProfile.from_profile_id(value)

    class RunOwnedCubeV2ArenaProfile(_RunOwnedExecutionMixin, CubeV2ArenaProfile):
        pass

    return RunOwnedCubeV2ArenaProfile(size=parsed.size, search_config=parsed.search_config)


def _profiles() -> dict[str, ArenaProfile]:
    profile = _torus_profile()
    return {profile.profile_id: profile}


def _parse_torus_fields(value: str) -> tuple[float, int, float, float, int, bool]:
    fields = value.split("|")
    komi = 0.5
    simulations = 64
    cpuct = 1.25
    fpu = 0.0
    watchdog = 500
    five_channel = False
    if value.startswith("torus9-komi-calibration|"):
        try:
            komi = float(fields[1])
            remaining = fields[2:]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Malformed Torus9 profile {value!r}") from exc
    else:
        remaining = fields[1:]
    try:
        for field in remaining:
            if field == "5ch":
                five_channel = True
            elif field.startswith("komi="):
                komi = float(field.split("=", 1)[1])
            elif field.startswith("simulations="):
                simulations = int(field.split("=", 1)[1])
            elif field.startswith("cpuct="):
                cpuct = float(field.split("=", 1)[1])
            elif field.startswith("fpu="):
                fpu = float(field.split("=", 1)[1])
            elif field.startswith("watchdog=") or field.startswith("technical_move_limit="):
                watchdog = int(field.split("=", 1)[1])
            elif field:
                raise ValueError(f"unknown Torus9 profile field {field!r}")
    except ValueError as exc:
        raise ValueError(f"Malformed Torus9 profile {value!r}") from exc
    return komi, simulations, cpuct, fpu, watchdog, five_channel


def get_profile(profile_id: str) -> ArenaProfile:
    value = str(profile_id)
    if value.startswith("torus9-komi-calibration|") or value.startswith("torus9|"):
        komi, simulations, cpuct, fpu, watchdog, five_channel = _parse_torus_fields(value)
        return _torus_profile(
            komi=komi,
            profile_id=value,
            simulations=simulations,
            cpuct=cpuct,
            fpu=fpu,
            watchdog=watchdog,
            five_channel=five_channel,
        )
    if value.startswith("cube-v2|"):
        return _cube_profile(value)
    profiles = _profiles()
    try:
        return profiles[value]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Arena profile {profile_id!r}; available={sorted(profiles)}; Cube V2 profiles are caller-configured"
        ) from exc


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(_profiles()))


def detect_profile(checkpoint_path: Path) -> ArenaProfile:
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Cannot auto-detect Arena profile without metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matches = [profile for profile in _profiles().values() if getattr(profile, "matches_metadata")(metadata)]
    if len(matches) != 1:
        if metadata.get("checkpoint_schema") == "gocube-cube-checkpoint-v2":
            raise ValueError(
                "Cube V2 Arena requires explicit caller-owned size/search configuration; use the Cube generation/Arena API instead of profile auto-detection"
            )
        raise ValueError(
            "Could not uniquely auto-detect Arena profile from checkpoint metadata; "
            f"matches={[profile.profile_id for profile in matches]}. Use --profile explicitly; available={list(available_profiles())}"
        )
    return matches[0]
