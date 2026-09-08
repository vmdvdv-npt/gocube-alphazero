"""Stable per-row provenance encoding for current GoCube replay artifacts."""

from __future__ import annotations

from typing import Any

import torch

from .contract_versions import TARGET_PROVENANCE_ENCODING
from .katago_v3 import (
    RESULT_PROVENANCE_FORMAL,
    RESULT_PROVENANCE_RULE_NO_RESULT,
    RESULT_PROVENANCE_RUNTIME,
    UNKNOWN_LEGACY_TERMINATION,
)


REPLAY_TARGET_PROVENANCE_SUFFIX = "-target-provenance.pkl"

TARGET_PROVENANCE_CODE_UNKNOWN = 0
TARGET_PROVENANCE_CODE_FORMAL = 1
TARGET_PROVENANCE_CODE_RULE_NO_RESULT = 2
TARGET_PROVENANCE_CODE_RUNTIME = 3

_PROVENANCE_TO_CODE = {
    UNKNOWN_LEGACY_TERMINATION: TARGET_PROVENANCE_CODE_UNKNOWN,
    RESULT_PROVENANCE_FORMAL: TARGET_PROVENANCE_CODE_FORMAL,
    RESULT_PROVENANCE_RULE_NO_RESULT: TARGET_PROVENANCE_CODE_RULE_NO_RESULT,
    RESULT_PROVENANCE_RUNTIME: TARGET_PROVENANCE_CODE_RUNTIME,
}
_CODE_TO_PROVENANCE = {
    code: provenance for provenance, code in _PROVENANCE_TO_CODE.items()
}
_VALID_CODES = frozenset(_CODE_TO_PROVENANCE)


def encode_target_provenance(
    provenance: str | None,
    *,
    allow_unknown: bool = False,
) -> int:
    """Encode an authoritative target provenance using the persisted mapping."""

    if provenance is None:
        if allow_unknown:
            return TARGET_PROVENANCE_CODE_UNKNOWN
        raise ValueError("Current S3 replay requires non-null target provenance")
    try:
        code = _PROVENANCE_TO_CODE[str(provenance)]
    except KeyError as exc:
        raise ValueError(f"Unknown target provenance: {provenance!r}") from exc
    if code == TARGET_PROVENANCE_CODE_UNKNOWN and not allow_unknown:
        raise ValueError("Current S3 replay cannot persist unknown target provenance")
    return code


def decode_target_provenance(code: int) -> str:
    """Decode one persisted code; reject values outside the versioned format."""

    try:
        return _CODE_TO_PROVENANCE[int(code)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown target provenance code: {code!r}") from exc


def validate_target_provenance_tensor(
    tensor: Any,
    *,
    expected_rows: int | None = None,
    allow_unknown: bool = False,
) -> torch.Tensor:
    """Validate and return a CPU ``uint8 [N]`` provenance sidecar."""

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Target provenance sidecar must be a torch.Tensor")
    if tensor.dtype != torch.uint8:
        raise ValueError(
            f"Target provenance sidecar must use torch.uint8, got {tensor.dtype}"
        )
    if tensor.ndim != 1:
        raise ValueError(
            f"Target provenance sidecar must have shape [N], got {tuple(tensor.shape)}"
        )
    if expected_rows is not None and int(tensor.numel()) != int(expected_rows):
        raise ValueError(
            "Target provenance sidecar row count mismatch: "
            f"sidecar={int(tensor.numel())}, expected={int(expected_rows)}"
        )
    values = {int(value) for value in tensor.detach().cpu().tolist()}
    invalid = values - _VALID_CODES
    if invalid:
        raise ValueError(f"Unknown target provenance codes: {sorted(invalid)}")
    if not allow_unknown and TARGET_PROVENANCE_CODE_UNKNOWN in values:
        raise ValueError("Current S3 replay cannot contain unknown provenance code 0")
    return tensor.detach().cpu()


def provenance_codes_to_semantics(tensor: torch.Tensor) -> tuple[str, ...]:
    """Return row-ordered semantic values after sidecar validation."""

    validated = validate_target_provenance_tensor(tensor, allow_unknown=True)
    return tuple(decode_target_provenance(int(code)) for code in validated.tolist())


__all__ = [
    "REPLAY_TARGET_PROVENANCE_SUFFIX",
    "TARGET_PROVENANCE_ENCODING",
    "TARGET_PROVENANCE_CODE_UNKNOWN",
    "TARGET_PROVENANCE_CODE_FORMAL",
    "TARGET_PROVENANCE_CODE_RULE_NO_RESULT",
    "TARGET_PROVENANCE_CODE_RUNTIME",
    "encode_target_provenance",
    "decode_target_provenance",
    "validate_target_provenance_tensor",
    "provenance_codes_to_semantics",
]
