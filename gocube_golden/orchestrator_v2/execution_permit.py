"""Capability-based authorization for Orchestrator V2 production execution.

The legacy ``AZ_ORCHESTRATOR_VERSION=V2`` marker is deliberately not an
authorization mechanism. A production entrypoint activates an in-process
orchestration authority. Only that authority can mint a short-lived signed
permit for a supervised child. The child permit is bound to the issuing
(supervisor) PID, action, topology, run/evaluation identity and code identity.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import time
import uuid
from typing import Iterator, Mapping

from ..provenance import canonical_json
from .version_constants import ORCHESTRATOR_VERSION

PERMIT_SCHEMA = "gocube-orchestrator-v2-execution-permit-v1"
PERMIT_ENV = "AZ_V2_EXECUTION_PERMIT"
PERMIT_KEY_ENV = "AZ_V2_EXECUTION_PERMIT_KEY"
DEFAULT_PERMIT_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class OrchestratorAuthority:
    launch_id: str
    mode: str
    topology: str
    run_id: str
    code_identity: str


_AUTHORITY: ContextVar[OrchestratorAuthority | None] = ContextVar(
    "gocube_orchestrator_v2_authority", default=None
)


def _component(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


@contextmanager
def _production_authority(
    *,
    mode: str,
    topology: str,
    run_id: str,
    code_identity: str,
    launch_id: str | None = None,
) -> Iterator[OrchestratorAuthority]:
    """Activate the private in-process capability owned by the V2 entrypoint."""
    authority = OrchestratorAuthority(
        launch_id=_component(launch_id or uuid.uuid4().hex, "launch_id"),
        mode=_component(mode, "mode"),
        topology=_component(topology, "topology"),
        run_id=_component(run_id, "run_id"),
        code_identity=_component(code_identity, "code_identity"),
    )
    token = _AUTHORITY.set(authority)
    try:
        yield authority
    finally:
        _AUTHORITY.reset(token)


def active_authority() -> OrchestratorAuthority | None:
    return _AUTHORITY.get()


def _signature(payload: Mapping[str, object], key: bytes) -> str:
    return hmac.new(
        key,
        canonical_json(dict(payload)).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _encode_key(key: bytes) -> str:
    return base64.urlsafe_b64encode(key).decode("ascii")


def _decode_key(value: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode(value.encode("ascii"))
    except Exception as exc:  # pragma: no cover - defensive decode boundary
        raise RuntimeError("Orchestrator V2 execution permit key is malformed") from exc
    if len(key) < 32:
        raise RuntimeError("Orchestrator V2 execution permit key is malformed")
    return key


def _permit_payload(
    *,
    authority: OrchestratorAuthority,
    action_type: str,
    topology: str,
    run_id: str,
    code_identity: str,
    ttl_seconds: float,
    attempt: int = 1,
) -> tuple[dict[str, object], bytes]:
    now = time.time()
    if not math.isfinite(float(ttl_seconds)) or float(ttl_seconds) <= 0:
        raise ValueError("permit ttl_seconds must be finite and positive")
    if topology != authority.topology:
        raise RuntimeError("Orchestrator V2 cannot mint a permit for a different topology")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("permit attempt must be a positive integer")
    key = secrets.token_bytes(32)
    payload: dict[str, object] = {
        "schema": PERMIT_SCHEMA,
        "launch_id": authority.launch_id,
        "action_id": uuid.uuid4().hex,
        "action_type": _component(action_type, "action_type"),
        "attempt": attempt,
        "topology": _component(topology, "topology"),
        "run_id": _component(run_id, "run_id"),
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "code_identity": _component(code_identity, "code_identity"),
        "supervisor_pid": os.getpid(),
        "issued_at": now,
        "expires_at": now + float(ttl_seconds),
        "nonce": secrets.token_hex(16),
    }
    payload["signature"] = _signature(payload, key)
    return payload, key


@contextmanager
def _child_execution_permit(
    *,
    action_type: str,
    topology: str,
    run_id: str,
    code_identity: str,
    ttl_seconds: float = DEFAULT_PERMIT_TTL_SECONDS,
    attempt: int = 1,
) -> Iterator[Mapping[str, object]]:
    """Mint a child capability and expose it only for supervised process spawn."""
    authority = active_authority()
    if authority is None:
        raise RuntimeError("Production execution cannot mint a permit outside Orchestrator V2.")
    payload, key = _permit_payload(
        authority=authority,
        action_type=action_type,
        topology=topology,
        run_id=run_id,
        code_identity=code_identity,
        ttl_seconds=ttl_seconds,
        attempt=attempt,
    )
    previous_permit = os.environ.get(PERMIT_ENV)
    previous_key = os.environ.get(PERMIT_KEY_ENV)
    os.environ[PERMIT_ENV] = canonical_json(payload)
    os.environ[PERMIT_KEY_ENV] = _encode_key(key)
    try:
        yield payload
    finally:
        if previous_permit is None:
            os.environ.pop(PERMIT_ENV, None)
        else:
            os.environ[PERMIT_ENV] = previous_permit
        if previous_key is None:
            os.environ.pop(PERMIT_KEY_ENV, None)
        else:
            os.environ[PERMIT_KEY_ENV] = previous_key


def _load_child_permit(
    *,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
    parent_pid: int | None = None,
) -> dict[str, object]:
    env = os.environ if environ is None else environ
    raw = env.get(PERMIT_ENV)
    raw_key = env.get(PERMIT_KEY_ENV)
    if not raw or not raw_key:
        raise RuntimeError("Production execution requires an Orchestrator V2 execution permit.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Orchestrator V2 execution permit is malformed") from exc
    if not isinstance(payload, dict) or payload.get("schema") != PERMIT_SCHEMA:
        raise RuntimeError("Orchestrator V2 execution permit schema mismatch")
    signature = payload.pop("signature", None)
    if not isinstance(signature, str) or not signature:
        raise RuntimeError("Orchestrator V2 execution permit has no signature")
    expected = _signature(payload, _decode_key(raw_key))
    if not hmac.compare_digest(signature, expected):
        raise RuntimeError("Orchestrator V2 execution permit signature mismatch")
    payload["signature"] = signature
    if payload.get("orchestrator_version") != ORCHESTRATOR_VERSION:
        raise RuntimeError("Orchestrator V2 execution permit version mismatch")
    current = time.time() if now is None else float(now)
    try:
        issued_at = float(payload["issued_at"])
        expires_at = float(payload["expires_at"])
        supervisor_pid = int(payload["supervisor_pid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Orchestrator V2 execution permit timing/PID is malformed") from exc
    if current < issued_at - 5.0 or current > expires_at:
        raise RuntimeError("Orchestrator V2 execution permit is expired or not yet valid")
    actual_parent = os.getppid() if parent_pid is None else int(parent_pid)
    if supervisor_pid != actual_parent:
        raise RuntimeError("Orchestrator V2 execution permit is not bound to this child process")
    return payload


def require_child_execution_permit(
    entrypoint: str,
    *,
    action_type: str | None = None,
    topology: str | None = None,
    run_id: str | None = None,
    code_identity: str | None = None,
    attempt: int | None = None,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
    parent_pid: int | None = None,
) -> Mapping[str, object]:
    try:
        payload = _load_child_permit(environ=environ, now=now, parent_pid=parent_pid)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{entrypoint}: production execution requires "
            "gocube_golden.orchestrator_v2.production_entrypoint (Orchestrator V2). "
            "Direct production execution is forbidden; submit this action through that entrypoint. "
            f"Permit rejected: {exc}"
        ) from exc
    expected = {
        "action_type": action_type,
        "topology": topology,
        "run_id": run_id,
        "code_identity": code_identity,
        "attempt": attempt,
    }
    for key, value in expected.items():
        if value is not None and str(payload.get(key)) != str(value):
            raise RuntimeError(f"{entrypoint}: Orchestrator V2 execution permit {key} mismatch")
    return payload


def require_orchestrator_execution(entrypoint: str) -> None:
    """Require either the live V2 authority or a valid supervised child permit."""
    if active_authority() is not None:
        return
    require_child_execution_permit(entrypoint)


def _test_authority(
    *, topology: str = "torus9", run_id: str = "test", code_identity: str = "test"
):
    """Unit-test-only authority; no filesystem or production bypass is implied."""
    return _production_authority(
        mode="test",
        topology=topology,
        run_id=run_id,
        code_identity=code_identity,
        launch_id="test-authority",
    )


__all__ = [
    "DEFAULT_PERMIT_TTL_SECONDS",
    "OrchestratorAuthority",
    "PERMIT_ENV",
    "PERMIT_KEY_ENV",
    "PERMIT_SCHEMA",
    "active_authority",
    "require_child_execution_permit",
    "require_orchestrator_execution",
]
