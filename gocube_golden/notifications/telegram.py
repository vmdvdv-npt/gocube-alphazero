"""HTTP-only Telegram Bot API transport.

The dispatcher owns persistence and retry policy.  This module knows only how
to make one bounded ``sendMessage`` request and classify its response.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TOKEN_ENV = "GOCUBE_TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "GOCUBE_TELEGRAM_CHAT_ID"
ENV_FILE = Path.home() / ".config" / "gocube-alphazero" / "telegram.env"


def _env_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    result: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key in {TOKEN_ENV, CHAT_ID_ENV}:
            result[key] = value
    return result


def load_config(*, environ: Mapping[str, str] | None = None, env_file: Path | None = None) -> tuple[str, str] | None:
    env = os.environ if environ is None else environ
    file_values = _env_file(ENV_FILE if env_file is None else env_file)
    token = str(env.get(TOKEN_ENV, "") or file_values.get(TOKEN_ENV, "")).strip()
    chat_id = str(env.get(CHAT_ID_ENV, "") or file_values.get(CHAT_ID_ENV, "")).strip()
    return (token, chat_id) if token and chat_id else None


class TelegramTransportError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool, retry_after: float | None = None, detail: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = bool(retryable)
        self.retry_after = retry_after
        self.detail = detail or code


class TelegramTransport:
    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        timeout: float = 3.0,
        opener: Callable[..., Any] = urlopen,
        endpoint: str = "https://api.telegram.org",
    ) -> None:
        if not token or not chat_id:
            raise ValueError("Telegram token and chat_id are required")
        self._token = str(token)
        self._chat_id = str(chat_id)
        self.timeout = max(0.1, float(timeout))
        self._opener = opener
        self._endpoint = endpoint.rstrip("/")

    @staticmethod
    def _response(payload: Mapping[str, Any], status: int) -> dict[str, Any]:
        if status < 200 or status >= 300 or payload.get("ok") is not True:
            error_code = payload.get("error_code")
            params = payload.get("parameters")
            retry_after = params.get("retry_after") if isinstance(params, Mapping) else None
            retry_seconds = float(retry_after) if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool) else None
            code = f"HTTP_{int(error_code or status)}"
            # 400/401/403 are configuration or formatting errors; Telegram's
            # documentation uses ``ok=false`` plus error_code/description.
            retryable = int(error_code or status) == 429 or int(error_code or status) >= 500
            raise TelegramTransportError(code, retryable=retryable, retry_after=retry_seconds)
        result = payload.get("result")
        if not isinstance(result, Mapping):
            return {"ok": True}
        receipt: dict[str, Any] = {"ok": True}
        for key in ("message_id", "date", "chat"):
            if key in result and key != "chat":
                receipt[key] = result[key]
            elif key == "chat" and isinstance(result.get(key), Mapping):
                chat = result[key]
                receipt["chat_id"] = chat.get("id")
        return receipt

    def send(self, text: str) -> Mapping[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise TelegramTransportError("FORMAT_EMPTY", retryable=False)
        if len(text) > 4096:
            raise TelegramTransportError("FORMAT_TOO_LONG", retryable=False)
        body = json.dumps({"chat_id": self._chat_id, "text": text}, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self._endpoint}/bot{self._token}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                try:
                    payload = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TelegramTransportError("INVALID_RESPONSE", retryable=True) from exc
                if not isinstance(payload, Mapping):
                    raise TelegramTransportError("INVALID_RESPONSE", retryable=True)
                return self._response(payload, int(getattr(response, "status", 200)))
        except HTTPError as exc:
            # Never include exc or its URL in a diagnostic: the URL contains
            # the bot token.  The body is parsed only for safe retry_after.
            retry_after = None
            try:
                raw = exc.read().decode("utf-8")
                payload = json.loads(raw)
                params = payload.get("parameters") if isinstance(payload, Mapping) else None
                value = params.get("retry_after") if isinstance(params, Mapping) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    retry_after = float(value)
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
                pass
            status = int(getattr(exc, "code", 0) or 0)
            raise TelegramTransportError(
                f"HTTP_{status or 'ERROR'}",
                retryable=status == 429 or status >= 500 or status == 0,
                retry_after=retry_after,
            ) from None
        except (URLError, TimeoutError, OSError):
            raise TelegramTransportError("NETWORK_ERROR", retryable=True) from None


__all__ = ["CHAT_ID_ENV", "ENV_FILE", "TOKEN_ENV", "TelegramTransport", "TelegramTransportError", "load_config"]
