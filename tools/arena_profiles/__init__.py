"""Arena profile registry.

Profiles define game/topology semantics. They do not own multiprocessing,
inference brokering, lifecycle, or telemetry; those belong to arena_engine.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.arena_engine import ArenaProfile


def _profiles() -> dict[str, ArenaProfile]:
    from tools.arena_profiles.torus9 import PROFILE as torus9

    return {torus9.profile_id: torus9}


def get_profile(profile_id: str) -> ArenaProfile:
    value = str(profile_id)
    if value.startswith("torus9-komi-calibration|"):
        from tools.arena_profiles.torus9 import Torus9ArenaProfile

        try:
            komi = float(value.split("|", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Malformed Torus9 komi calibration profile {profile_id!r}") from exc
        return Torus9ArenaProfile(komi=komi, profile_id=value)
    if value.startswith("cube-v2|"):
        from tools.arena_profiles.cube_v2 import CubeV2ArenaProfile

        return CubeV2ArenaProfile.from_profile_id(value)
    profiles = _profiles()
    try:
        return profiles[value]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Arena profile {profile_id!r}; available={sorted(profiles)}; "
            "Cube V2 profiles are caller-configured"
        ) from exc


def available_profiles() -> tuple[str, ...]:
    # Dynamic Cube profiles deliberately do not advertise a default scientific
    # configuration: size/search values are caller-owned.
    return tuple(sorted(_profiles()))


def detect_profile(checkpoint_path: Path) -> ArenaProfile:
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Cannot auto-detect Arena profile without metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matches = [
        profile
        for profile in _profiles().values()
        if getattr(profile, "matches_metadata")(metadata)
    ]
    if len(matches) != 1:
        if metadata.get("checkpoint_schema") == "gocube-cube-checkpoint-v2":
            raise ValueError(
                "Cube V2 Arena requires explicit caller-owned size/search configuration; "
                "use the Cube generation/Arena API instead of profile auto-detection"
            )
        raise ValueError(
            "Could not uniquely auto-detect Arena profile from checkpoint metadata; "
            f"matches={[profile.profile_id for profile in matches]}. "
            f"Use --profile explicitly; available={list(available_profiles())}"
        )
    return matches[0]
