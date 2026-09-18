from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock

from .catalog import CheckpointCatalog, CheckpointDescriptor, is_runtime_compatible
from .errors import (
    CheckpointIncompatible,
    CheckpointNotFound,
    GenerationBusy,
    GenerationFailed,
    IntegrationError,
    InvalidRequest,
    UnsupportedProtocol,
)
from .golden_generation import GoldenGameGenerator
from .golden_models import GoldenCheckpointLoader

PROTOCOL_VERSION = 1
RUNTIME_IDENTITY_SCHEMA = "gocube-alphazero-runtime-identity-v1"


def _runtime_source_fingerprint(source_root: Path) -> str:
    digest = hashlib.sha256()
    roots = (source_root / "alphazero" / "envs" / "gocube" / "integration", source_root / "gocube_golden")
    paths = sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in paths:
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _file_sha256(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _compatible(a: CheckpointDescriptor, b: CheckpointDescriptor) -> bool:
    return is_runtime_compatible(a, b)


class GoCubeAlphaZeroService:
    def __init__(
        self,
        checkpoint_dir: str,
        *,
        device: str = "auto",
        catalog: CheckpointCatalog | None = None,
        loader: GoldenCheckpointLoader | None = None,
        generator: GoldenGameGenerator | None = None,
        publication_manifest: str | None = None,
    ):
        self.catalog = catalog or CheckpointCatalog(
            checkpoint_dir,
            publication_manifest=publication_manifest,
        )
        self.loader = loader or GoldenCheckpointLoader(
            self.catalog,
            device=device,
        )
        self.generator = generator or GoldenGameGenerator()
        self.device = self.loader.device
        self._generation_lock = Lock()
        source_root = Path(__file__).resolve().parents[4]
        self._runtime_identity = {
            "schema": RUNTIME_IDENTITY_SCHEMA,
            "sourceRoot": str(source_root),
            "sourceFingerprint": _runtime_source_fingerprint(source_root),
            "checkpointDir": self.catalog.checkpoint_dir,
            "publicationManifest": self.catalog.publication_manifest,
            "publicationManifestSha256": _file_sha256(self.catalog.publication_manifest),
        }

    def health(self) -> dict[str, object]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "status": "ok",
            "service": "gocube-alphazero",
            "device": self.device,
            "runtimeIdentity": dict(self._runtime_identity),
        }

    def checkpoints(self) -> dict[str, object]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "checkpoints": [item.to_api() for item in self.catalog.list()],
        }

    def _validate_game_request(self, request: object):
        if not isinstance(request, dict):
            raise InvalidRequest("Request body must be a JSON object")

        required = {
            "protocolVersion",
            "blackCheckpointId",
            "whiteCheckpointId",
            "mctsSims",
        }
        allowed = set(required)
        missing = sorted(required - set(request))
        unknown = sorted(set(request) - allowed)
        if missing:
            raise InvalidRequest(f"Missing request fields: {', '.join(missing)}")
        if unknown:
            raise InvalidRequest(f"Unknown request fields: {', '.join(unknown)}")

        protocol = request["protocolVersion"]
        if not isinstance(protocol, int) or isinstance(protocol, bool):
            raise InvalidRequest("protocolVersion must be an integer")
        if protocol != PROTOCOL_VERSION:
            raise UnsupportedProtocol(f"Unsupported protocolVersion: {protocol}")

        black_id = request["blackCheckpointId"]
        white_id = request["whiteCheckpointId"]
        if not isinstance(black_id, str) or not black_id:
            raise InvalidRequest("blackCheckpointId must be a non-empty string")
        if not isinstance(white_id, str) or not white_id:
            raise InvalidRequest("whiteCheckpointId must be a non-empty string")

        sims = request["mctsSims"]
        if not isinstance(sims, int) or isinstance(sims, bool) or sims < 1:
            raise InvalidRequest("mctsSims must be an integer >= 1")

        black = self.catalog.get(black_id)
        if black is None:
            raise CheckpointNotFound(f"Unknown checkpoint: {black_id}")
        white = self.catalog.get(white_id)
        if white is None:
            raise CheckpointNotFound(f"Unknown checkpoint: {white_id}")
        if not _compatible(black, white):
            raise CheckpointIncompatible(
                f"Checkpoints {black_id} and {white_id} do not share their exact "
                "topology/rules/model/search contract"
            )
        return black, white, sims

    def generate_game(self, request: object) -> dict[str, object]:
        black, white, sims = self._validate_game_request(request)
        if not self._generation_lock.acquire(blocking=False):
            raise GenerationBusy("Another game generation is already running")

        try:
            if black.checkpoint_id == white.checkpoint_id:
                _, model = self.loader.load(black.checkpoint_id)
                black_model = white_model = model
            else:
                _, black_model = self.loader.load(black.checkpoint_id)
                _, white_model = self.loader.load(white.checkpoint_id)

            game = self.generator.generate(
                black=black,
                white=white,
                black_model=black_model,
                white_model=white_model,
                mcts_sims=sims,
            )
            return {"protocolVersion": PROTOCOL_VERSION, "game": game}
        except IntegrationError:
            raise
        except Exception as exc:
            raise GenerationFailed(f"Game generation failed: {exc}") from exc
        finally:
            self._generation_lock.release()
