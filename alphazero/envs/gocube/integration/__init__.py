"""Local Protocol V1 integration boundary between GoCube and AlphaZero.

Imports are lazy because the core network wrapper also persists the integration
model contract; eager service imports would create a wrapper/integration cycle.
"""

__all__ = [
    "CheckpointCatalog",
    "CheckpointDescriptor",
    "GoCubeAlphaZeroService",
    "PROTOCOL_VERSION",
    "RunManifest",
    "GoCubeModelContract",
    "ResolvedGoCubeContract",
    "resolve_model_contract",
    "resolve_model_contract_from_metadata",
    "resolve_semantic_game_class_from_contract",
]


def __getattr__(name):
    if name in {"CheckpointCatalog", "CheckpointDescriptor"}:
        from .catalog import CheckpointCatalog, CheckpointDescriptor
        return {"CheckpointCatalog": CheckpointCatalog, "CheckpointDescriptor": CheckpointDescriptor}[name]
    if name == "RunManifest":
        from .manifest import RunManifest
        return RunManifest
    if name in {"GoCubeAlphaZeroService", "PROTOCOL_VERSION"}:
        from .service import GoCubeAlphaZeroService, PROTOCOL_VERSION
        return {"GoCubeAlphaZeroService": GoCubeAlphaZeroService, "PROTOCOL_VERSION": PROTOCOL_VERSION}[name]
    if name in {
        "GoCubeModelContract", "ResolvedGoCubeContract", "resolve_model_contract",
        "resolve_model_contract_from_metadata", "resolve_semantic_game_class_from_contract",
    }:
        from .contract import (
            GoCubeModelContract,
            ResolvedGoCubeContract,
            resolve_model_contract,
            resolve_model_contract_from_metadata,
            resolve_semantic_game_class_from_contract,
        )
        return {
            "GoCubeModelContract": GoCubeModelContract,
            "ResolvedGoCubeContract": ResolvedGoCubeContract,
            "resolve_model_contract": resolve_model_contract,
            "resolve_model_contract_from_metadata": resolve_model_contract_from_metadata,
            "resolve_semantic_game_class_from_contract": resolve_semantic_game_class_from_contract,
        }[name]
    raise AttributeError(name)
