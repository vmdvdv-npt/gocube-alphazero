from __future__ import annotations

import math


GOCUBE_DEFAULT_KOMI = 0.5
LEGACY_FORBIDDEN_KOMI = 7.5
_KOMI_ABS_TOL = 1e-12


class LegacyKomiError(ValueError):
    """A known stale legacy komi value reached a current GoCube path."""


def validate_gocube_komi(value: object, *, context: str = "GoCube") -> float:
    """Validate komi under the current project-wide policy.

    ``0.5`` is the current default/baseline, not the only value that may ever
    be used. Explicit finite alternatives are allowed so Torus fair-komi work
    and future rule experiments do not need to bypass a global hard-coded
    equality check.

    ``7.5`` is deliberately different: it is a known legacy ordinary-Go value
    in this repository's history. Seeing it in a current runtime/configuration
    path is treated as likely stale-artifact contamination and fails closed.
    """

    if isinstance(value, bool):
        raise ValueError(f"{context} requires a finite numeric komi, got {value!r}")
    try:
        komi = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} requires a finite numeric komi, got {value!r}"
        ) from exc
    if not math.isfinite(komi):
        raise ValueError(f"{context} requires a finite numeric komi, got {value!r}")
    if math.isclose(
        komi,
        LEGACY_FORBIDDEN_KOMI,
        rel_tol=0.0,
        abs_tol=_KOMI_ABS_TOL,
    ):
        raise LegacyKomiError(
            f"{context} received forbidden legacy komi {LEGACY_FORBIDDEN_KOMI}. "
            "This most likely indicates stale legacy configuration, checkpoint, "
            "or manifest contamination. Stop and contact the project owner before "
            "continuing; do not silently coerce the value to 0.5."
        )
    return komi
