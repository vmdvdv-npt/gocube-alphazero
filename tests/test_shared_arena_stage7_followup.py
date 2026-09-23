from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.arena_inference as arena_inference
import tools.arena_worker as arena_worker
import tools.arena_profiles.cube_v2_worker as cube_worker
from training_engine import CheckpointContext, value_fingerprint
from gocube_golden.cube_arena_contract_v2 import CubeArenaSearchConfig
from gocube_golden.cube_arena_v2 import run_cube_arena
from gocube_golden.cube_training_contract_v2 import CubeTrainingConfig
from gocube_golden.cube_training_v2 import create_cube_m0_state
from tools.arena_engine import ArenaExecutionConfig
from tools.arena_profiles.cube_v2 import CubeV2ArenaProfile
from tools.arena_profiles.torus9 import Torus9ArenaProfile


def _training_config() -> CubeTrainingConfig:
    return CubeTrainingConfig(
        learning_rate=0.001,
        batch_size=1,
        optimizer_steps=1,
        replay_generations=2,
        replay_cap=16,
    )


def _make_pass_model(state, *, salt: float) -> None:
    with torch.no_grad():
        for parameter in state.model.parameters():
            parameter.zero_()
        state.model.pass_head.bias.fill_(8.0)
        state.model.score_head[-1].bias.fill_(float(salt))
    state.model.eval()


def _publish_temp_m0(
    root: Path,
    *,
    size: int,
    lineage_id: str,
    seed: int,
    salt: float,
):
    lineage = root / lineage_id
    config = _training_config()
    adapter, state = create_cube_m0_state(
        size=size,
        config=config,
        seed=seed,
        device="cpu",
    )
    _make_pass_model(state, salt=salt)
    context = CheckpointContext(
        run_id=lineage_id,
        label="M0",
        parent_label=None,
        generation=0,
        training_seed=seed,
        fresh_positions=0,
        replay_positions=0,
        replay_generations=(),
        replay_fingerprint=value_fingerprint(()),
        sampled_row_ids_fingerprint=value_fingerprint(()),
        completed_games=0,
        parent_checkpoint_identity=None,
        code_identity=None,
        device="cpu",
    )
    metadata = adapter.prepare_checkpoint(state, context, {})
    checkpoint = lineage / "checkpoints" / "M0.pt"
    saved = adapter.save_checkpoint(checkpoint, state, metadata)
    reference = {
        "lineage_id": lineage_id,
        "checkpoint_id": "M0",
        "label": "M0",
        "path": str(checkpoint.resolve()),
        "metadata_path": str(checkpoint.with_suffix(".metadata.json").resolve()),
        "sha256": saved["checkpoint_sha256"],
        "artifact_sha256": saved["checkpoint_sha256"],
        "model_hash": saved["model_hash"],
        "generation": 0,
        "size": size,
    }
    state.parent_checkpoint_identity = dict(reference)
    return adapter, state, reference, lineage


class _TorusToyModel(torch.nn.Module):
    def __init__(self, policy_size: int) -> None:
        super().__init__()
        self.policy_size = int(policy_size)
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch: torch.Tensor):
        rows = int(batch.shape[0])
        return (
            torch.zeros(
                (rows, self.policy_size),
                dtype=batch.dtype,
                device=batch.device,
            )
            + self.anchor,
            torch.zeros((rows, 3), dtype=batch.dtype, device=batch.device)
            + self.anchor,
        )


class _CubeToyModel(torch.nn.Module):
    def __init__(self, policy_size: int) -> None:
        super().__init__()
        self.policy_size = int(policy_size)
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def infer_policy_wdl(self, batch: torch.Tensor):
        rows = int(batch.shape[0])
        return SimpleNamespace(
            policy_logits=(
                torch.zeros(
                    (rows, self.policy_size),
                    dtype=batch.dtype,
                    device=batch.device,
                )
                + self.anchor
            ),
            wdl_logits=(
                torch.zeros((rows, 3), dtype=batch.dtype, device=batch.device)
                + self.anchor
            ),
        )


def _cube_search() -> CubeArenaSearchConfig:
    return CubeArenaSearchConfig(
        simulations=1,
        cpuct=1.0,
        fpu=0.0,
        watchdog=4,
    )


def test_torus_and_cube_inference_really_pass_through_shared_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[arena_inference.BatchedPolicyWDLInferenceOwner] = []
    evaluated: list[arena_inference.BatchedPolicyWDLInferenceOwner] = []
    real_owner = arena_inference.BatchedPolicyWDLInferenceOwner

    class TrackingOwner(real_owner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def evaluate_shared_batch(self, observations):
            evaluated.append(self)
            return super().evaluate_shared_batch(observations)

    arena_inference.clear_arena_inference_owner_cache()
    monkeypatch.setattr(
        arena_inference,
        "BatchedPolicyWDLInferenceOwner",
        TrackingOwner,
    )

    torus = Torus9ArenaProfile()
    torus_policy, torus_wdl = torus.infer_batch(
        _TorusToyModel(torus.policy_size),
        torch.zeros((2, *torus.observation_shape), dtype=torch.float32),
        torch.device("cpu"),
    )
    assert torus_policy.shape == (2, torus.policy_size)
    assert torus_wdl.shape == (2, 3)

    cube = CubeV2ArenaProfile(size=2, search_config=_cube_search())
    cube_policy, cube_wdl = cube.infer_batch(
        _CubeToyModel(cube.policy_size),
        torch.zeros((2, *cube.observation_shape), dtype=torch.float32),
        torch.device("cpu"),
    )
    assert cube_policy.shape == (2, cube.policy_size)
    assert cube_wdl.shape == (2, 3)

    assert len(created) == 2
    assert evaluated == created
    assert all(isinstance(owner, real_owner) for owner in created)
    arena_inference.clear_arena_inference_owner_cache()


def test_profiles_have_only_model_specific_inference_boundary() -> None:
    common = inspect.getsource(arena_inference)
    torus_infer = inspect.getsource(Torus9ArenaProfile.infer_batch)
    cube_infer = inspect.getsource(CubeV2ArenaProfile.infer_batch)

    assert "BatchedPolicyWDLInferenceOwner" in common
    for source in (torus_infer, cube_infer):
        assert "torch.softmax" not in source
        assert "torch.inference_mode" not in source
        assert ".to(" not in source
        assert "infer_policy_wdl_batch" in source


def test_one_topology_neutral_cooperative_worker_path() -> None:
    common = inspect.getsource(arena_worker)
    cube = inspect.getsource(cube_worker)
    torus = inspect.getsource(Torus9ArenaProfile.worker_main)

    assert "SequentialPUCTSession" in common
    assert "gocube_golden.cube" not in common
    assert "gocube_golden.torus" not in common
    assert "SequentialPUCT(" not in cube
    assert "run_cooperative_arena_worker" in cube
    assert "run_cooperative_arena_worker" in torus


def test_cube_strict_production_remains_forbidden() -> None:
    profile = CubeV2ArenaProfile(size=2, search_config=_cube_search())
    with pytest.raises(ValueError, match="no production execution profile"):
        profile.validate_execution_config(
            ArenaExecutionConfig(
                games=4,
                workers=1,
                games_per_worker=4,
                inference_batch_rows=4,
                inference_batch_wait_ms=0.0,
                device="cpu",
                strict_production=True,
            )
        )


def test_cube_four_games_use_four_real_lanes_and_central_batching(
    tmp_path: Path,
) -> None:
    # Use the same bounded pass-biased fixture shape as the Stage-7 suite.
    _, _, candidate, _ = _publish_temp_m0(
        tmp_path,
        size=2,
        lineage_id="shared-worker-a",
        seed=9101,
        salt=0.01,
    )
    _, _, reference, _ = _publish_temp_m0(
        tmp_path,
        size=2,
        lineage_id="shared-worker-b",
        seed=9102,
        salt=0.02,
    )
    output = tmp_path / "shared-worker-arena"
    result = run_cube_arena(
        size=2,
        candidate_checkpoint=candidate,
        reference_checkpoint=reference,
        output_dir=output,
        search_config=_cube_search(),
        execution_config=ArenaExecutionConfig(
            games=4,
            workers=1,
            games_per_worker=4,
            inference_batch_rows=4,
            inference_batch_wait_ms=10.0,
            device="cpu",
            strict_production=False,
            early_gate_enabled=False,
            min_effective_cpu_cores=0.0,
        ),
        seed=9103,
        temporary_root=tmp_path,
    )

    assert result.games_requested == 4
    assert result.games_valid == 4
    assert result.technical == 0
    assert result.invalid == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    telemetry = summary["telemetry"]
    assert telemetry["observed_unique_lane_ids_per_worker"]["0"] == [0, 1, 2, 3]
    assert int(telemetry["max_inference_batch_rows"]) > 1

    rows = [
        json.loads(line)
        for line in (output / "games.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert all(row["formal_result"] in {"BLACK", "WHITE", "DRAW"} for row in rows)
    assert all(row["technical_termination"] is None for row in rows)
