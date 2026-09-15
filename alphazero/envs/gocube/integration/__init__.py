"""Golden-only GoCube Protocol V1 integration boundary."""

__all__ = [
    "CheckpointCatalog",
    "CheckpointDescriptor",
    "GoCubeAlphaZeroService",
    "PROTOCOL_VERSION",
    "GoldenCheckpointLoader",
    "GoldenGameGenerator",
    "GoldenPlayableModel",
    "GoldenProtocolMapping",
]


def __getattr__(name):
    if name in {"CheckpointCatalog", "CheckpointDescriptor"}:
        from .catalog import CheckpointCatalog, CheckpointDescriptor
        return {"CheckpointCatalog": CheckpointCatalog, "CheckpointDescriptor": CheckpointDescriptor}[name]
    if name in {"GoCubeAlphaZeroService", "PROTOCOL_VERSION"}:
        from .service import GoCubeAlphaZeroService, PROTOCOL_VERSION
        return {"GoCubeAlphaZeroService": GoCubeAlphaZeroService, "PROTOCOL_VERSION": PROTOCOL_VERSION}[name]
    if name == "GoldenCheckpointLoader":
        from .golden_models import GoldenCheckpointLoader
        return GoldenCheckpointLoader
    if name == "GoldenPlayableModel":
        from .golden_models import GoldenPlayableModel
        return GoldenPlayableModel
    if name == "GoldenGameGenerator":
        from .golden_generation import GoldenGameGenerator
        return GoldenGameGenerator
    if name == "GoldenProtocolMapping":
        from .golden_mapping import GoldenProtocolMapping
        return GoldenProtocolMapping
    raise AttributeError(name)
