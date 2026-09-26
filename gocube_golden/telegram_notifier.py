"""Fail-open Telegram observability for production training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import weakref

from .process_supervision import atomic_write_json

TOKEN_ENV = "GOCUBE_TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "GOCUBE_TELEGRAM_CHAT_ID"
ENV_FILE = Path.home() / ".config" / "gocube-alphazero" / "telegram.env"


class TelegramError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key.strip() in {TOKEN_ENV, CHAT_ID_ENV}:
            out[key.strip()] = value
    return out


def load_config(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> tuple[str, str] | None:
    env = os.environ if environ is None else environ
    file_values = _env_file(ENV_FILE if env_file is None else env_file)
    token = str(env.get(TOKEN_ENV, "") or file_values.get(TOKEN_ENV, "")).strip()
    chat_id = str(env.get(CHAT_ID_ENV, "") or file_values.get(CHAT_ID_ENV, "")).strip()
    return (token, chat_id) if token and chat_id else None


def _send(token: str, chat_id: str, text: str) -> None:
    body = json.dumps({"chat_id": chat_id, "text": text}, ensure_ascii=False).encode()
    last = "delivery failed"
    for attempt in range(2):
        req = Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=3.0) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
                if 200 <= int(response.status) < 300 and payload.get("ok") is True:
                    return
                last = f"HTTP {response.status}"
        except HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code == 429 and attempt == 0:
                time.sleep(0.5)
        except (URLError, TimeoutError, OSError) as exc:
            # Never stringify URL-bearing exceptions: the URL contains the token.
            last = exc.__class__.__name__
        if attempt == 0:
            time.sleep(0.2)
    raise TelegramError(last)


def telegram_test() -> None:
    config = load_config()
    if config is None:
        raise TelegramError(f"missing {TOKEN_ENV}/{CHAT_ID_ENV}; expected {ENV_FILE}")
    _send(
        *config,
        f"GoCube AlphaZero\nTelegram notifications: OK\nHost: {socket.gethostname()}",
    )


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _lineage(paths: object) -> str:
    manifest = _json(Path(getattr(paths, "manifest")))
    return str(manifest.get("lineage_id") or Path(getattr(paths, "root")).name)


def _state(paths: object) -> dict[str, object]:
    return _json(Path(getattr(paths, "runtime_state")))


def _arena(paths: object, generation: int) -> tuple[str, str]:
    root = Path(getattr(paths, "root"))
    result = _json(root / "arena" / f"generation-{generation:04d}" / "result.json")
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else None
    if metrics is None:
        history = Path(getattr(paths, "metrics")) / "history.jsonl"
        try:
            lines = history.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, Mapping)
                and row.get("kind") == "arena"
                and int(row.get("generation", -1)) == generation
            ):
                value = row.get("metrics")
                metrics = value if isinstance(value, Mapping) else {}
                break
    metrics = dict(metrics or {})
    reference = result.get("reference_generation")
    if not isinstance(reference, (int, float)) or isinstance(reference, bool):
        manifest = _json(Path(getattr(paths, "manifest")))
        orchestrator = manifest.get("orchestrator")
        previous = []
        if isinstance(orchestrator, Mapping):
            previous = [
                int(x)
                for x in orchestrator.get("arena_generations", [])
                if isinstance(x, (int, float)) and int(x) < generation
            ]
        parent = manifest.get("parent_checkpoint")
        reference = (
            max(previous)
            if previous
            else (parent.get("generation") if isinstance(parent, Mapping) else None)
        )
    games = int(metrics.get("games", 0) or 0)
    technical = int(result.get("technical_games", 0) or 0)
    invalid = int(result.get("invalid_games", 0) or 0)
    comparison = (
        f"M{generation} vs M{int(reference)}" if reference is not None else f"M{generation}"
    )
    text = "\n".join(
        [
            "Arena completed — GoCube AlphaZero",
            f"Lineage: {_lineage(paths)}",
            comparison,
            "W/L/D: "
            f"{int(metrics.get('wins', 0) or 0)}/"
            f"{int(metrics.get('losses', 0) or 0)}/"
            f"{int(metrics.get('draws', 0) or 0)}",
            f"Valid: {max(0, games - technical - invalid)}",
            f"Technical: {technical}",
            "Mean batch: "
            f"{float(metrics.get('inference_mean_batch_rows', 0.0) or 0.0):.2f}",
        ]
    )
    performance = str(metrics.get("performance_status", "HEALTHY"))
    if performance != "HEALTHY":
        text += f"\nPerformance: {performance}"
    return f"arena:{generation}:{reference}", text


def build_notification(
    paths: object,
    level: str,
    message: str,
    details: Mapping[str, object],
) -> tuple[str, str] | None:
    upper = level.upper()
    state = _state(paths)
    if message == "Arena performance warning; Arena accepted":
        return None  # Included in the Arena completion message instead of duplicated.
    if message == "Arena completed":
        return _arena(paths, int(details.get("generation", 0)))
    if message == "Soft stop reached a safe generation boundary":
        generation = details.get(
            "last_committed_generation", state.get("last_committed_generation", 0)
        )
        return f"stop-complete:{generation}", "\n".join(
            [
                "Graceful stop completed — GoCube AlphaZero",
                f"Lineage: {_lineage(paths)}",
                f"Last committed generation: M{generation}",
                "Lineage remains resumable.",
            ]
        )
    if message == "Requested generation limit completed":
        generation = details.get("generation", state.get("last_committed_generation", 0))
        return f"completed:{generation}", "\n".join(
            [
                "Training stopped — GoCube AlphaZero",
                f"Lineage: {_lineage(paths)}",
                "Final state: COMPLETED",
                f"Last committed generation: M{generation}",
            ]
        )
    if upper == "WARNING" and "converted to durable soft-stop request" in message:
        return stop_requested_notification(paths, details.get("requested_at"))
    if upper == "WARNING":
        generation = details.get("generation", state.get("active_generation"))
        phase = details.get("phase", state.get("active_phase"))
        lines = [
            "WARNING / DEGRADED — GoCube AlphaZero",
            f"Lineage: {_lineage(paths)}",
        ]
        if generation is not None:
            lines.append(f"Generation: M{generation}")
        if phase:
            lines.append(f"Stage: {phase}")
        lines.append(message)
        if details.get("mean_inference_batch_rows") is not None:
            lines.append(
                f"Mean batch: {float(details['mean_inference_batch_rows']):.2f}"
            )
        return (
            "warning:"
            f"{message}:{generation}:{phase}:{details.get('mean_inference_batch_rows')}",
            "\n".join(lines),
        )
    if upper == "CRITICAL":
        committed = state.get("last_committed_generation", 0)
        active = state.get("active_generation")
        phase = state.get("active_phase")
        title = (
            "CRITICAL / RECOVERY_REQUIRED"
            if state.get("state") == "RECOVERY_REQUIRED"
            else "CRITICAL"
        )
        lines = [
            title + " — GoCube AlphaZero",
            f"Lineage: {_lineage(paths)}",
            f"Last committed generation: M{committed}",
        ]
        if active is not None:
            lines.append(f"Active generation: M{active}")
        if phase:
            lines.append(f"Stage: {phase}")
        lines.append(f"Reason: {message}")
        return (
            f"critical:{state.get('state')}:{message}:{committed}:{active}:{phase}",
            "\n".join(lines),
        )
    return None


def stop_requested_notification(
    paths: object,
    requested_at: object = None,
) -> tuple[str, str]:
    state = _state(paths)
    current = state.get("active_generation", state.get("last_committed_generation", 0))
    return f"stop-request:{requested_at or current}", "\n".join(
        [
            "Graceful stop requested — GoCube AlphaZero",
            f"Lineage: {_lineage(paths)}",
            f"Current generation: M{current}",
            "Waiting for safe boundary.",
        ]
    )


_instances: weakref.WeakSet = weakref.WeakSet()


class TelegramNotifier:
    retry_interval_seconds = 30.0
    # The structured EventSink compatibility adapter may emit start facts for
    # this real facade; test doubles do not opt in merely by exposing send_now.
    supports_structured_starts = True

    def __init__(self, paths: object) -> None:
        self.paths = paths
        self.config = load_config()
        self.delivered = Path(getattr(paths, "runtime")) / "telegram-notifications.jsonl"
        self.errors = Path(getattr(paths, "logs")) / "telegram-notifier-errors.jsonl"
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.outbox = Path(getattr(paths, "runtime")) / "telegram-outbox"
        self._delivery_lock = threading.RLock()
        self._worker_lock = threading.Lock()
        _instances.add(self)
        if self.config is not None and any(self.outbox.glob("*.json")):
            self._ensure_worker()

    def _seen(self, key: str) -> bool:
        try:
            lines = self.delivered.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return False
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("key") == key:
                return True
        return False

    def _pending_path(self, key: str) -> Path:
        return self.outbox / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def _record_error(self, key: str, exc: Exception) -> None:
        try:
            self.errors.parent.mkdir(parents=True, exist_ok=True)
            with self.errors.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "at": _now(), "key": key,
                    "error": str(exc) if isinstance(exc, TelegramError) else "delivery failed",
                    "type": exc.__class__.__name__,
                }) + "\n")
        except OSError:
            pass

    def _persist(self, key: str, text: str) -> bool:
        if self.config is None:
            return False
        with self._delivery_lock:
            if self._seen(key):
                return False
            try:
                atomic_write_json(self._pending_path(key), {"key": key, "text": text})
            except OSError as exc:
                self._record_error(key, exc)
                # Disk trouble must not prevent an immediate delivery attempt.
            return True

    def _deliver(self, key: str, text: str) -> bool:
        with self._delivery_lock:
            if self.config is None:
                return False
            try:
                if not self._seen(key):
                    _send(*self.config, text)
                    self.delivered.parent.mkdir(parents=True, exist_ok=True)
                    with self.delivered.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"at": _now(), "key": key}) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                self._pending_path(key).unlink(missing_ok=True)
                return True
            except Exception as exc:
                self._record_error(key, exc)
                return False

    def _retry_pending(self) -> None:
        for path in sorted(self.outbox.glob("*.json")):
            payload = _json(path)
            key, text = payload.get("key"), payload.get("text")
            if isinstance(key, str) and isinstance(text, str):
                self._deliver(key, text)

    def _worker(self) -> None:
        next_retry = time.monotonic()
        while True:
            try:
                key, text = self.queue.get(timeout=max(0.0, next_retry - time.monotonic()))
            except queue.Empty:
                pass
            else:
                try:
                    self._deliver(key, text)
                finally:
                    self.queue.task_done()
            if time.monotonic() >= next_retry:
                self._retry_pending()
                next_retry = time.monotonic() + self.retry_interval_seconds
            with self._worker_lock:
                if self.queue.empty() and not any(self.outbox.glob("*.json")):
                    self.thread = None
                    return

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(
                    target=self._worker, name="gocube-telegram", daemon=True,
                )
                self.thread.start()

    def enqueue(self, key: str, text: str) -> None:
        if self._persist(key, text):
            self.queue.put((key, text))
            self._ensure_worker()

    def send_now(self, key: str, text: str) -> None:
        if self._persist(key, text) and not self._deliver(key, text):
            self._ensure_worker()

    def flush(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        # A standalone command may exit before the periodic retry is due.
        # Give persisted failures one final bounded opportunity to send.
        if self.config is not None:
            for path in self.outbox.glob("*.json"):
                payload = _json(path)
                key, text = payload.get("key"), payload.get("text")
                if isinstance(key, str) and isinstance(text, str):
                    self.queue.put((key, text))
            if self.queue.unfinished_tasks:
                self._ensure_worker()
        while self.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.02)


_notifiers: dict[str, TelegramNotifier] = {}
_installed = False
_lock = threading.Lock()


def _notifier(paths: object) -> TelegramNotifier:
    key = str(Path(getattr(paths, "root")).resolve())
    with _lock:
        if key not in _notifiers:
            _notifiers[key] = TelegramNotifier(paths)
        return _notifiers[key]


def notify_stop_requested(paths: object, payload: Mapping[str, object]) -> None:
    try:
        key, text = stop_requested_notification(paths, payload.get("requested_at"))
        _notifier(paths).send_now(key, text)
    except BaseException:
        pass


def install() -> None:
    global _installed
    if _installed:
        return
    from .orchestrator import EventSink

    original = EventSink.emit

    def emit(self: object, level: str, message: str, **details: object) -> None:
        original(self, level, message, **details)
        try:
            item = build_notification(getattr(self, "paths"), level, message, details)
            if item:
                _notifier(getattr(self, "paths")).enqueue(*item)
        except BaseException:
            pass

    EventSink.emit = emit  # type: ignore[method-assign]
    _installed = True


def flush_all(timeout: float = 7.0) -> None:
    values = list(_instances)
    for notifier in values:
        notifier.flush(max(0.1, timeout / max(1, len(values))))
