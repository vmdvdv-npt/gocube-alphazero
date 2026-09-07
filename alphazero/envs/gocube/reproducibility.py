from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch

from .contract_versions import SEED_DERIVATION_CONTRACT


def derive_worker_seed(
    master_seed: int,
    iteration: int,
    worker_id: int,
    game_slot: int,
    game_sequence_number: int,
) -> int:
    """Derive a stable child seed from the complete self-play coordinate."""

    fields = {
        "contract": SEED_DERIVATION_CONTRACT,
        "master_seed": int(master_seed),
        "iteration": int(iteration),
        "worker_id": int(worker_id),
        "game_slot": int(game_slot),
        "game_sequence_number": int(game_sequence_number),
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def seed_process(seed: int) -> None:
    """Seed process-local RNGs used by training and self-play."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed & 0x7FFFFFFFFFFFFFFF)
