from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from gocube_golden.arena_contract import SEARCH_IMPLEMENTATION_ID
from gocube_golden.run_storage import RUNS_ROOT

from .errors import (
    CheckpointLoadFailed,
    GenerationBusy,
    GenerationFailed,
    IntegrationError,
    InvalidMoveHistory,
    InvalidRequest,
    PositionInvalid,
    PositionTerminal,
    SearchFailed,
    ServiceBusy,
    TerminalPosition,
    UnsupportedProtocol,
)
from .service import GoCubeAlphaZeroService, PROTOCOL_VERSION

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
MAX_REQUEST_BYTES = 64 * 1024


def _error_payload(error: IntegrationError) -> dict[str, object]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "error": {"code": error.code, "message": error.message},
    }


def _validate_move_request(request: object) -> dict[str, object]:
    if not isinstance(request, dict):
        raise InvalidRequest("Request body must be a JSON object")

    required = {"protocolVersion", "requestId", "checkpointId", "mctsSims", "position"}
    missing = sorted(required - set(request))
    unknown = sorted(set(request) - required)
    if missing:
        raise InvalidRequest(f"Missing request fields: {', '.join(missing)}")
    if unknown:
        raise InvalidRequest(f"Unknown request fields: {', '.join(unknown)}")

    protocol = request["protocolVersion"]
    if not isinstance(protocol, int) or isinstance(protocol, bool):
        raise InvalidRequest("protocolVersion must be an integer")
    if protocol != PROTOCOL_VERSION:
        raise UnsupportedProtocol(f"Unsupported protocolVersion: {protocol}")

    request_id = request["requestId"]
    if not isinstance(request_id, str):
        raise InvalidRequest("requestId must be a string")

    checkpoint_id = request["checkpointId"]
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise InvalidRequest("checkpointId must be a non-empty string")

    mcts_sims = request["mctsSims"]
    if not isinstance(mcts_sims, int) or isinstance(mcts_sims, bool) or mcts_sims < 1:
        raise InvalidRequest("mctsSims must be an integer >= 1")

    position = request["position"]
    if not isinstance(position, dict):
        raise InvalidRequest("position must be an object")
    position_required = {"topology", "size", "ruleSet", "komi", "moves"}
    missing_position = sorted(position_required - set(position))
    unknown_position = sorted(set(position) - position_required)
    if missing_position:
        raise InvalidRequest(f"Missing position fields: {', '.join(missing_position)}")
    if unknown_position:
        raise InvalidRequest(f"Unknown position fields: {', '.join(unknown_position)}")

    topology = position["topology"]
    if not isinstance(topology, str) or not topology:
        raise InvalidRequest("position.topology must be a non-empty string")
    size = position["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise InvalidRequest("position.size must be an integer >= 1")
    rule_set = position["ruleSet"]
    if not isinstance(rule_set, str) or not rule_set:
        raise InvalidRequest("position.ruleSet must be a non-empty string")
    komi = position["komi"]
    if isinstance(komi, bool) or not isinstance(komi, (int, float)) or not math.isfinite(float(komi)):
        raise InvalidRequest("position.komi must be a finite number")
    moves = position["moves"]
    if not isinstance(moves, list):
        raise InvalidRequest("position.moves must be an array")

    return {
        "requestId": request_id,
        "checkpointId": checkpoint_id,
        "mctsSims": mcts_sims,
        "moves": moves,
        "serviceArgs": {
            "checkpoint_id": checkpoint_id,
            "topology": topology,
            "size": size,
            "rule_set": rule_set,
            "komi": float(komi),
            "history": moves,
            "mcts_sims": mcts_sims,
        },
    }


def _move_error(error: IntegrationError) -> IntegrationError:
    if isinstance(error, InvalidMoveHistory):
        return PositionInvalid("Position history is invalid")
    if isinstance(error, TerminalPosition):
        return PositionTerminal("Position is terminal")
    if isinstance(error, GenerationBusy):
        return ServiceBusy("Move service is busy")
    if isinstance(error, (GenerationFailed, CheckpointLoadFailed)):
        return SearchFailed("Move search failed")
    return error


def _move_response(dto: dict[str, object], selected: dict[str, object]) -> dict[str, object]:
    moves = dto["moves"]
    assert isinstance(moves, list)
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": dto["requestId"],
        "checkpointId": dto["checkpointId"],
        "mctsSims": dto["mctsSims"],
        "moveNumber": len(moves) + 1,
        "color": selected["color"],
        "action": selected["action"],
        "search": {
            "simulations": selected["mctsSims"],
            "implementationId": SEARCH_IMPLEMENTATION_ID,
        },
    }


def make_handler(service: GoCubeAlphaZeroService, allowed_origins=DEFAULT_ALLOWED_ORIGINS):
    allowed = frozenset(allowed_origins)

    class Handler(BaseHTTPRequestHandler):
        server_version = "GoCubeAlphaZero/1"

        def log_message(self, format, *args):
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

        def _origin(self):
            return self.headers.get("Origin")

        def _origin_allowed(self) -> bool:
            origin = self._origin()
            return origin is None or origin in allowed

        def _send_json(self, status: int, payload: object):
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            origin = self._origin()
            if origin in allowed:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, error: IntegrationError):
            self._send_json(error.http_status, _error_payload(error))

        def _require_origin(self) -> bool:
            if self._origin_allowed():
                return True
            self._send_error(InvalidRequest(f"Origin is not allowed: {self._origin()}"))
            return False

        def do_OPTIONS(self):
            if not self._require_origin():
                return
            self.send_response(204)
            origin = self._origin()
            if origin in allowed:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self):
            if not self._require_origin():
                return
            path = urlsplit(self.path).path
            try:
                if path == "/v1/health":
                    self._send_json(200, service.health())
                elif path == "/v1/checkpoints":
                    self._send_json(200, service.checkpoints())
                else:
                    self._send_error(InvalidRequest(f"Unknown endpoint: {path}"))
            except IntegrationError as exc:
                self._send_error(exc)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                self._send_json(
                    500,
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "error": {"code": "generation_failed", "message": "Internal service failure"},
                    },
                )

        def do_POST(self):
            if not self._require_origin():
                return
            path = urlsplit(self.path).path
            if path not in {"/v1/games", "/v1/move"}:
                self._send_error(InvalidRequest(f"Unknown endpoint: {path}"))
                return

            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                self._send_error(InvalidRequest("Content-Type must be application/json"))
                return

            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_error(InvalidRequest("Content-Length must be an integer"))
                return
            if length < 0 or length > MAX_REQUEST_BYTES:
                self._send_error(InvalidRequest("Request body is too large"))
                return

            try:
                raw = self.rfile.read(length)
                request = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_error(InvalidRequest("Request body must contain valid UTF-8 JSON"))
                return

            if path == "/v1/games":
                try:
                    self._send_json(200, service.generate_game(request))
                except IntegrationError as exc:
                    self._send_error(exc)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    self._send_json(
                        500,
                        {
                            "protocolVersion": PROTOCOL_VERSION,
                            "error": {
                                "code": "generation_failed",
                                "message": "Internal game generation failure",
                            },
                        },
                    )
                return

            try:
                dto = _validate_move_request(request)
                selected = service.select_move(**dto["serviceArgs"])
                self._send_json(200, _move_response(dto, selected))
            except IntegrationError as exc:
                self._send_error(_move_error(exc))
            except Exception:
                traceback.print_exc(file=sys.stderr)
                self._send_error(SearchFailed("Internal move selection failure"))

    return Handler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local GoCube AlphaZero integration service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", default=str(RUNS_ROOT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--publication-manifest",
        default=None,
        help="Explicit checkpoint publication manifest; defaults to the repository production manifest.",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=None,
        help="Allowed browser Origin. Repeat for multiple origins.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    if not 1 <= cli.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    service = GoCubeAlphaZeroService(
        cli.checkpoint_dir,
        device=cli.device,
        publication_manifest=cli.publication_manifest,
    )
    origins = tuple(cli.allow_origin) if cli.allow_origin else DEFAULT_ALLOWED_ORIGINS
    server = ThreadingHTTPServer((cli.host, cli.port), make_handler(service, origins))
    server.daemon_threads = True
    print(
        f"GoCube AlphaZero Protocol V1 listening on http://{cli.host}:{cli.port} "
        f"(device={service.device}, checkpoint-dir={cli.checkpoint_dir})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
