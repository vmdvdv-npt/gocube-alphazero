"""Topology-specific composition for Orchestrator V2.

Lifecycle remains topology-neutral. Topology bindings select scientific
bridges/profiles/startsets, while run-owned execution values remain parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..artifact_graph import EffectiveConfig
from .contracts import ArtifactRef, StartsetRef

SUPPORTED_TOPOLOGIES = ("torus9", "cube2", "cube3", "cube4", "cube5", "cube6", "cube7")


def _required(mapping: Mapping[str, object], names: tuple[str, ...], label: str) -> object:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise ValueError(f"{label} must be explicit in the resolved effective config")


def _cube_size(topology: str) -> int:
    if topology.startswith("cube") and topology[4:].isdigit():
        size = int(topology[4:])
        if 2 <= size <= 7:
            return size
    raise ValueError(f"unsupported Cube topology: {topology}")


def _cube_arena_search(config: EffectiveConfig):
    from ..cube_arena_contract_v2 import CubeArenaSearchConfig

    arena = config.arena
    return CubeArenaSearchConfig(
        simulations=int(_required(arena, ("simulations", "mcts_simulations"), "Cube Arena simulations")),
        cpuct=float(_required(arena, ("cpuct",), "Cube Arena cpuct")),
        fpu=float(_required(arena, ("fpu",), "Cube Arena fpu")),
        watchdog=int(_required(arena, ("watchdog", "technical_move_limit"), "Cube Arena watchdog")),
    )


def _torus_profile_id(config: EffectiveConfig) -> str:
    arena = config.arena
    simulations = int(arena.get("simulations", arena.get("mcts_simulations", 64)))
    if simulations <= 0:
        raise ValueError("Torus9 Arena simulations must be a positive integer")
    komi = float(arena.get("komi", config.self_play.get("komi", 0.5)))
    cpuct = float(arena.get("cpuct", 1.25))
    fpu = float(arena.get("fpu", 0.0))
    watchdog = int(arena.get("watchdog", arena.get("technical_move_limit", 500)))
    compatibility = config.compatibility
    channels = compatibility.get("input_channels")
    if channels is None:
        shape = compatibility.get("observation_shape")
        if isinstance(shape, (list, tuple)) and shape:
            channels = shape[0]
    profile = (
        f"torus9|komi={komi:g}|simulations={simulations}"
        f"|cpuct={cpuct:g}|fpu={fpu:g}|watchdog={watchdog}"
    )
    if channels == 5:
        profile += "|5ch"
    return profile


@dataclass(frozen=True)
class TopologyBinding:
    topology: str
    cube_size: int | None = None

    def production_path(self):
        if self.topology == "torus9":
            from .torus9_production import Torus9ProductionGenerationPath
            return Torus9ProductionGenerationPath()
        from .cube_production_recovery import CubeProductionGenerationPath
        return CubeProductionGenerationPath(size=int(self.cube_size))

    def default_arena_profile(self, config: EffectiveConfig) -> str:
        if config.topology != self.topology:
            raise ValueError("topology binding received an effective config for another topology")
        if self.topology == "torus9":
            return _torus_profile_id(config)
        from tools.arena_profiles.cube_v2 import CubeV2ArenaProfile
        return CubeV2ArenaProfile(size=int(self.cube_size), search_config=_cube_arena_search(config)).profile_id

    def validate_arena_profile(self, profile: str, config: EffectiveConfig) -> str:
        from tools.arena_profiles import get_profile

        if not isinstance(profile, str) or not profile:
            raise ValueError("arena_profile must be a non-empty string")
        resolved = get_profile(profile)
        if self.topology == "torus9":
            if profile == "torus9" or profile.startswith("torus9|") or profile.startswith("torus9-komi-calibration|"):
                # Search/rules values are owned by this evaluation run.  The
                # effective training config is used only for model-format and
                # lineage compatibility; it is not an Arena whitelist.
                return profile
            raise ValueError("Arena profile is not compatible with topology=torus9")
        expected = self.default_arena_profile(config)
        if profile != expected:
            raise ValueError(f"Arena profile {profile!r} is not compatible with {self.topology}; expected {expected!r}")
        if getattr(resolved, "size", None) != self.cube_size:
            raise ValueError("Cube Arena profile size does not match topology")
        return profile

    def arena_startset(self, *, master_seed: int, games: int) -> StartsetRef:
        if games <= 0 or games % 2:
            raise ValueError("Arena games must be a positive even number")
        if self.topology == "torus9":
            from .arena_runner import torus9_startset_ref
            return torus9_startset_ref(master_seed=master_seed, games=games)
        from ..cube_arena_startset_v1 import build_cube_arena_startset

        startset = build_cube_arena_startset(
            size=int(self.cube_size),
            master_seed=int(master_seed),
            pairs=int(games) // 2,
        )
        fingerprint = startset.fingerprint
        identity = f"{self.topology}-evaluation-starts-v1"
        return StartsetRef(
            id=identity,
            artifact=ArtifactRef(
                f"startsets/{identity}-{fingerprint.removeprefix('sha256:')[:16]}.json",
                fingerprint,
            ),
            fingerprint=fingerprint,
        )


def get_topology_binding(topology: str) -> TopologyBinding:
    value = str(topology)
    if value == "torus9":
        return TopologyBinding("torus9", None)
    if value in SUPPORTED_TOPOLOGIES:
        return TopologyBinding(value, _cube_size(value))
    raise ValueError(f"unsupported Orchestrator V2 topology: {value}")


def production_path_for(topology: str):
    return get_topology_binding(topology).production_path()


__all__ = ["SUPPORTED_TOPOLOGIES", "TopologyBinding", "get_topology_binding", "production_path_for"]
