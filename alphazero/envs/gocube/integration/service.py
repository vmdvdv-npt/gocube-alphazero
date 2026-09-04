from __future__ import annotations

from threading import Lock

from .catalog import CheckpointCatalog, CheckpointDescriptor
from .errors import (
    CheckpointIncompatible,
    CheckpointNotFound,
    GenerationBusy,
    GenerationFailed,
    IntegrationError,
    InvalidRequest,
    UnsupportedProtocol,
)
from .generation import GameGenerator
from .models import CheckpointModelLoader

PROTOCOL_VERSION = 1


def _compatible(a: CheckpointDescriptor, b: CheckpointDescriptor) -> bool:
    return (
        a.topology == b.topology
        and a.size == b.size
        and a.rule_set == b.rule_set
        and a.komi == b.komi
        and a.terminal_adjudicator == b.terminal_adjudicator
    )


class GoCubeAlphaZeroService:
    def __init__(
        self,
        checkpoint_dir: str,
        *,
        device: str = "auto",
        catalog: CheckpointCatalog | None = None,
        loader: CheckpointModelLoader | None = None,
        generator: GameGenerator | None = None,
    ):
        self.catalog = catalog or CheckpointCatalog(checkpoint_dir)
        self.loader = loader or CheckpointModelLoader(self.catalog, device=device)
        self.generator = generator or GameGenerator()
        self.device = self.loader.device
        self._generation_lock = Lock()

    def health(self) -> dict[str, object]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "status": "ok",
            "service": "gocube-alphazero",
            "device": self.device,
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
                f"Checkpoints {black_id} and {white_id} do not share topology/size/rules/komi/adjudicator"
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
