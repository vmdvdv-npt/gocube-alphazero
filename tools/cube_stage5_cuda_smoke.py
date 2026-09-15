#!/usr/bin/env python3
"""Short Legion CUDA smoke for the Cube Stage-5 shared execution path.

The pass-biased fixture makes every game finish formally after two passes, so
the smoke validates process/shared-memory/batching plumbing without running a
long scientific game sweep. Its search settings are intentionally marked
noncanonical; this is an execution smoke, not training evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import sys
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gocube_golden.cube_contract import CUBE_PROFILE_ID, load_profile
from gocube_golden.cube_neural import GoldenCubeGraphNetV1, cube_model_hash
from gocube_golden.cube_training import DEFAULT_CUBE_SELFPLAY_CONTRACT, run_cube_selfplay_games
from gocube_golden.provenance import capture_code_identity, file_sha256


class _PassBiasedCubeModel(GoldenCubeGraphNetV1):
    """Keep the real Cube network and force a deterministic formal finish."""

    def forward(self, observations):
        policy_logits, value_logits = super().forward(observations)
        policy_logits = policy_logits.clone()
        policy_logits[:, 96] = 1000.0
        return policy_logits, value_logits


def run_smoke(*, output: Path, device: str = "cuda") -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Cube Stage-5 CUDA smoke requires CUDA")
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("Cube Stage-5 smoke must run on CUDA")
    profile = load_profile()
    model = _PassBiasedCubeModel().to(selected)
    contract = replace(
        DEFAULT_CUBE_SELFPLAY_CONTRACT,
        simulations=1,
        dirichlet_epsilon=0.0,
    )
    run_id = "cube-stage5-legion-cuda-smoke"
    code = capture_code_identity(ROOT)
    telemetry: dict[str, object] = {}
    game_ids = tuple(f"{run_id}-game-{index:02d}" for index in range(16))
    records = run_cube_selfplay_games(
        model,
        game_ids,
        run_id=run_id,
        profile_id=CUBE_PROFILE_ID,
        profile_fingerprint=profile["profile_fingerprint"],
        model_checkpoint_label="CUDA-SMOKE",
        checkpoint_artifact_hash=file_sha256(ROOT / "tools" / "cube_stage5_cuda_smoke.py"),
        master_seed=2026091515,
        chunk_id="cuda-smoke",
        code_identity=code,
        workers=16,
        active_games_per_worker=4,
        total_active_contexts=64,
        inference_batch_cap=64,
        inference_batch_wait_ms=1.0,
        device=selected,
        contract=contract,
        allow_noncanonical_contract=True,
        inference_telemetry=telemetry,
        execution_activity=telemetry,
    )
    for record in records:
        record.validate(deep=True)
    technical = sum(record.technical_termination is not None for record in records)
    if len(records) != len(game_ids) or technical != 0:
        raise RuntimeError("CUDA smoke did not produce 16 formal games")
    if telemetry.get("worker_processes_started") != 16:
        raise RuntimeError(f"CUDA smoke did not start 16 workers: {telemetry}")
    if telemetry.get("shared_memory_transport") is not True:
        raise RuntimeError("CUDA smoke did not use shared-memory transport")
    if telemetry.get("central_inference_owner_pid") != os.getpid():
        raise RuntimeError("CUDA smoke central inference owner is not the parent")
    if int(telemetry.get("max_inference_batch_rows", 0)) < 3:
        raise RuntimeError("CUDA smoke did not observe a real multi-row inference batch")
    report = {
        "smoke": "cube-stage5-shared-cuda-v1",
        "host": socket.gethostname(),
        "device": str(selected),
        "run_id": run_id,
        "games": len(records),
        "technical_games": technical,
        "formal_games": len(records) - technical,
        "workers": telemetry["worker_processes_started"],
        "worker_pids": telemetry["worker_pids"],
        "central_inference_owner_pid": telemetry["central_inference_owner_pid"],
        "shared_memory_transport": telemetry["shared_memory_transport"],
        "worker_model_replicas": False,
        "mean_inference_batch_rows": telemetry["mean_inference_batch_rows"],
        "max_inference_batch_rows": telemetry["max_inference_batch_rows"],
        "inference_rows": telemetry["inference_rows"],
        "model_hash": cube_model_hash(model),
        "scientific_contract": "noncanonical short pass-biased execution fixture",
        "telemetry": telemetry,
        "code": {
            "git_commit_sha": code.git_commit_sha,
            "git_tree_sha": code.git_tree_sha,
            "working_tree_clean": code.working_tree_clean,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "host": report["host"],
        "workers": report["workers"],
        "technical_games": report["technical_games"],
        "mean_batch": report["mean_inference_batch_rows"],
        "max_batch": report["max_inference_batch_rows"],
    }, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/cube-stage5-legion-cuda-smoke.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run_smoke(output=args.output, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
