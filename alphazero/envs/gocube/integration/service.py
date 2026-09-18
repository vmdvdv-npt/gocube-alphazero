from __future__ import annotations

import hashlib
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Mapping, Sequence

from .catalog import CheckpointCatalog, CheckpointDescriptor, is_runtime_compatible
from .errors import (
    CheckpointIncompatible,
    CheckpointNotFound,
    GenerationBusy,
    GenerationFailed,
    IntegrationError,
    InvalidRequest,
    ServiceBusy,
    TerminalPosition,
    UnsupportedProtocol,
)
from .golden_generation import GoldenGameGenerator
from .golden_models import GoldenCheckpointLoader
from .golden_move import (
    INTERACTIVE_SEARCH_CONTRACT,
    GoldenMoveSelector,
    GoldenPositionContract,
    mapping_for_position,
    replay_action_history,
    validate_checkpoint_position_compatibility,
)
from .model_cache import BoundedModelCache

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
        move_selector: GoldenMoveSelector | None = None,
        publication_manifest: str | None = None,
        model_cache_size: int = 2,
        move_concurrency: int = 1,
    ):
        self.catalog = catalog or CheckpointCatalog(
            checkpoint_dir,
            publication_manifest=publication_manifest,
        )
        self.model_cache = BoundedModelCache(max_entries=model_cache_size)
        self.loader = loader or GoldenCheckpointLoader(
            self.catalog,
            device=device,
            cache=self.model_cache,
        )
        self.move_selector = move_selector or GoldenMoveSelector()
        self.generator = generator or GoldenGameGenerator(move_selector=self.move_selector)
        self.device = self.loader.device
        self._generation_lock = Lock()
        if isinstance(move_concurrency, bool) or not isinstance(move_concurrency, int) or move_concurrency < 1:
            raise ValueError("move concurrency must be an integer >= 1")
        self.move_concurrency = move_concurrency
        self._move_capacity = BoundedSemaphore(move_concurrency)
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
            "capabilities": {
                "generateGame": True,
                "selectMove": True,
            },
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

    def select_move(
        self,
        *,
        checkpoint_id: str,
        topology: str,
        size: int,
        rule_set: str,
        komi: float,
        history: Sequence[Mapping[str, object]],
        mcts_sims: int,
    ) -> dict[str, object]:
        """Select one move from a request-contained position with no game session state."""

        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise InvalidRequest("checkpointId must be a non-empty string")
        INTERACTIVE_SEARCH_CONTRACT.settings(mcts_sims)
        position = GoldenPositionContract(
            topology=topology,
            size=size,
            rule_set=rule_set,
            komi=komi,
        )
        mapping = mapping_for_position(position)
        state = replay_action_history(
            topology=position.topology,
            size=position.size,
            rule_set=position.rule_set,
            komi=position.komi,
            moves=history,
        )
        if state.is_terminal:
            raise TerminalPosition("History already represents a terminal Golden position")

        descriptor = self.catalog.get(checkpoint_id)
        if descriptor is None:
            raise CheckpointNotFound(f"Unknown checkpoint: {checkpoint_id}")

        validate_checkpoint_position_compatibility(
            position=position,
            state=state,
            descriptor=descriptor,
            mapping=mapping,
        )

        if not self._move_capacity.acquire(blocking=False):
            raise ServiceBusy("Move search capacity is exhausted")
        try:
            try:
                _loaded_descriptor, model = self.loader.load(checkpoint_id)
                selection = self.move_selector.select_move(
                    state=state,
                    descriptor=descriptor,
                    model=model,
                    mcts_sims=mcts_sims,
                )
                action = mapping.golden_action_to_protocol(selection.action)
            except IntegrationError:
                raise
            except Exception as exc:
                raise GenerationFailed(f"Move selection failed: {exc}") from exc
        finally:
            self._move_capacity.release()

        return {
            "checkpointId": checkpoint_id,
            "topology": position.topology,
            "size": position.size,
            "ruleSet": position.rule_set,
            "komi": position.komi,
            "color": "black" if int(state.side_to_move) == 1 else "white",
            "action": action,
            "mctsSims": selection.simulations,
            "searchProfileId": selection.search_profile_id,
        }

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
