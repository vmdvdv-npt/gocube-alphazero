from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch

from alphazero.envs.gocube.contract import CUBE4_PRODUCTION, require_gocube_komi


DEFAULT_MIN_FREE_GIB = 10


def validate_runtime_constants(implementation: Any) -> dict[str, object]:
    """Fail closed if orchestration mirrors drift from the central contract."""

    contract = CUBE4_PRODUCTION
    observed = {
        "workers": int(implementation.WORKERS),
        "regular_sims": int(implementation.REGULAR_SIMS),
        "fast_sims": int(implementation.FAST_SIMS),
        "games_per_iteration": int(implementation.GAMES_PER_ITERATION),
        "train_batch_size": int(implementation.TRAIN_BATCH_SIZE),
        "arena_sims": int(implementation.ARENA_SIMS),
        "komi": require_gocube_komi(
            implementation.EXPECTED_KOMI, context="overnight implementation"
        ),
    }
    expected = {
        "workers": contract.workers,
        "regular_sims": contract.regular_sims,
        "fast_sims": contract.fast_sims,
        "games_per_iteration": contract.games_per_iteration,
        "train_batch_size": contract.train_batch_size,
        "arena_sims": contract.arena_sims,
        "komi": contract.komi,
    }
    mismatches = {
        key: (observed[key], expected[key])
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        detail = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise RuntimeError(f"Cube-4 runtime contract drift: {detail}")
    return observed


def preflight(cli: Any, implementation: Any, *, min_free_gib: int = DEFAULT_MIN_FREE_GIB) -> dict[str, object]:
    """Small current-runtime preflight with no frozen historical run assumptions."""

    contract = validate_runtime_constants(implementation)

    cpu_count = int(os.cpu_count() or 0)
    if cpu_count < CUBE4_PRODUCTION.workers:
        raise RuntimeError(
            f"Cube-4 sweep requires at least {CUBE4_PRODUCTION.workers} logical CPUs, got {cpu_count}"
        )

    disk = shutil.disk_usage(Path.cwd())
    free_gib = disk.free / (1024 ** 3)
    if free_gib < int(min_free_gib):
        raise RuntimeError(
            f"Cube-4 sweep requires at least {int(min_free_gib)} GiB free, got {free_gib:.1f} GiB"
        )

    requested_device = str(getattr(cli, "device", "auto"))
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    bootstrap_run = getattr(cli, "bootstrap_run", None)
    checkpoint_contract = None
    if bootstrap_run:
        checkpoint_contract = implementation.validate_production_checkpoint(
            str(bootstrap_run), int(implementation.BOOTSTRAP_ITERATION)
        )

    return {
        "contract": contract,
        "cpu_count": cpu_count,
        "free_gib": free_gib,
        "requested_device": requested_device,
        "cuda_available": cuda_available,
        "bootstrap_checkpoint": checkpoint_contract,
    }
