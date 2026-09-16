"""Small guards for keeping generated runs out of protected namespaces."""

from __future__ import annotations

from pathlib import Path


def safe_run_id(run_id: str) -> str:
    """Validate a run identifier before using it as one directory component."""
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("Run ID must be a single non-empty path component")
    return run_id


def assert_isolated(output_dir: Path, protected_dir: Path) -> Path:
    """Reject output equal to or nested below a protected directory."""
    output_dir = output_dir.resolve()
    protected_dir = protected_dir.resolve()
    if output_dir == protected_dir:
        raise ValueError("Output cannot be the protected namespace")
    try:
        output_dir.relative_to(protected_dir)
    except ValueError:
        return output_dir
    raise ValueError("Output cannot be nested in the protected namespace")
