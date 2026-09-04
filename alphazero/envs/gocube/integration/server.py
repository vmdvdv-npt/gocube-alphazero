from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .errors import IntegrationError, InvalidRequest
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
            if path != "/v1/games":
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

    return Handler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local GoCube AlphaZero integration service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    service = GoCubeAlphaZeroService(cli.checkpoint_dir, device=cli.device)
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
