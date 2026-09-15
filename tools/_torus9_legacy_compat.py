"""Compatibility helpers used only by the frozen historical runner."""

from __future__ import annotations

from pathlib import Path

import torch

from gocube_golden.torus9 import (
    Torus9CurrentGraphNet,
    Torus9SelfPlaySearchContract,
    torus9_checkpoint_metadata,
    torus9_save_checkpoint,
)


def checkpoint(
    path: Path,
    model: Torus9CurrentGraphNet,
    optimizer: torch.optim.Optimizer | None,
    *,
    run_id: str,
    label: str,
    parent: str | None,
    code,
    profile_fp: str,
    contract: Torus9SelfPlaySearchContract,
    completed_games: int,
    replay_positions: int,
    optimizer_updates: int,
    samples_consumed: int,
    device: str,
    base_commit: str = "e7b088be4ad743f089d7895c867f4f28990019ed",
) -> dict[str, object]:
    metadata = torus9_checkpoint_metadata(
        model=model,
        run_id=run_id,
        label=label,
        parent=parent,
        model_seed=202609131001,
        code=code,
        profile_fp=profile_fp,
        completed_games=completed_games,
        replay_positions=replay_positions,
        optimizer_updates=optimizer_updates,
        samples_consumed=samples_consumed,
        ownership_loss_enabled=True,
        score_loss_enabled=True,
        profile_id="gocube-torus9-golden-v3",
        target_fingerprint="sha256:02ab244688534b271473302ab4edf00516b91d43fb91b8c9e592d2a8de63dfb5",
        selfplay_contract_id="torus9-golden-current-selfplay-search-v1",
        selfplay_contract_fingerprint=contract.fingerprint,
        base_commit=base_commit,
    )
    metadata.update({
        "adam_step": optimizer_updates,
        "model_init_seed": 202609131001,
        "device": device,
        "device_locked": True,
        "execution_only_parameters": {
            "self_play_inference_batch_cap": "not checkpoint semantic",
            "self_play_inference_batch_wait_ms": "not checkpoint semantic",
        },
    })
    return torus9_save_checkpoint(path, model=model, optimizer=optimizer, metadata=metadata)
