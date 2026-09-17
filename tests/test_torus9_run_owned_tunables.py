from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import gocube_golden.torus9_run_owned_training as run_owned_training
from gocube_golden.torus9_monolith import TORUS9_TOPOLOGY_FINGERPRINT
from gocube_golden import (
    Torus9CurrentGraphNet,
    Torus9SelfPlaySearchContract,
    Torus9TrainingAdapter,
    load_torus9_current_profile,
)
from gocube_golden.run_spec import StrictRunSpec
from gocube_golden.torus9_contract import (
    TORUS9_CURRENT_ARCHITECTURE_ID,
    TORUS9_CURRENT_PROFILE_ID,
    TORUS9_CURRENT_TARGET_FINGERPRINT,
    TORUS9_KOMI,
    TORUS9_TARGET_CONTRACT_ID,
    current_torus9_content_fingerprint,
    profile_fingerprint,
    validate_torus9_current_profile,
)


def _run_profile(*, lr: float, replay_generations: int, replay_cap: int, simulations: int):
    profile = deepcopy(load_torus9_current_profile())
    profile["experiment"] = {"kind": "test-run-owned-tunables"}
    profile["training"]["learning_rate"] = lr
    profile["replay"]["window"] = f"rolling last {replay_generations} generations"
    profile["replay"]["generations"] = replay_generations
    profile["replay"]["cap"] = replay_cap
    profile["self_play"]["mcts_simulations"] = simulations
    profile["content_fingerprint"] = current_torus9_content_fingerprint(profile)
    profile["profile_fingerprint"] = profile_fingerprint(profile)
    return profile


def _parent_reference(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "parent-lineage"
    checkpoint = root / "checkpoints" / "M47.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-placeholder")
    checkpoint.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_label": "M47",
                "rolling_generations": 3,
                "maximum_replay_positions": 20000,
            }
        ),
        encoding="utf-8",
    )
    replay = root / "replay" / "rolling-after-47.jsonl"
    replay.parent.mkdir(parents=True)
    replay.write_text("", encoding="utf-8")
    return root, checkpoint, replay


@pytest.mark.parametrize(
    ("lr", "generations", "cap", "simulations"),
    [
        (0.0007, 2, 12345, 37),
        (0.00005, 9, 75000, 256),
        (0.003, 17, 150000, 511),
    ],
)
def test_run_owned_tunables_are_not_compared_with_golden(
    lr: float,
    generations: int,
    cap: int,
    simulations: int,
) -> None:
    profile = _run_profile(
        lr=lr,
        replay_generations=generations,
        replay_cap=cap,
        simulations=simulations,
    )
    validate_torus9_current_profile(profile)

    contract = Torus9SelfPlaySearchContract(simulations=simulations)
    contract.validate()
    assert contract.settings.simulations == simulations

    adapter = Torus9TrainingAdapter(profile=profile)
    state = adapter.create_state(Torus9CurrentGraphNet(), run_id="run-owned-test")
    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(lr)
    assert state.rolling_replay.generations == generations
    assert state.rolling_replay.maximum_positions == cap


def test_parent_adam_state_keeps_moments_but_run_lr_wins() -> None:
    profile = _run_profile(
        lr=0.0003,
        replay_generations=6,
        replay_cap=40000,
        simulations=128,
    )
    model = Torus9CurrentGraphNet()
    adapter = Torus9TrainingAdapter(profile=profile)
    state = adapter.create_state(model, run_id="adam-lr-override")

    parent_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    parent_optimizer.step()
    state.optimizer.load_state_dict(parent_optimizer.state_dict())

    first_parameter = next(iter(model.parameters()))
    moment_before = state.optimizer.state[first_parameter]["exp_avg"].detach().clone()
    step_before = int(state.optimizer.state[first_parameter]["step"].item())
    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)

    adapter._apply_effective_lr(state.optimizer)

    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.0003)
    assert int(state.optimizer.state[first_parameter]["step"].item()) == step_before
    assert torch.equal(state.optimizer.state[first_parameter]["exp_avg"], moment_before)


def test_expanded_parent_replay_requires_complete_fresh_window(tmp_path: Path) -> None:
    root, checkpoint, fallback = _parent_reference(tmp_path)
    for generation in (42, 43, 44, 46, 47):
        (root / "replay" / f"iter-{generation:02d}-fresh.jsonl").write_text(
            "", encoding="utf-8"
        )
    adapter = Torus9TrainingAdapter(
        profile=_run_profile(
            lr=0.0003,
            replay_generations=6,
            replay_cap=40000,
            simulations=128,
        )
    )

    with pytest.raises(FileNotFoundError, match="iter-45-fresh.jsonl"):
        adapter._reference_sources(checkpoint, (fallback,))


def test_expanded_parent_replay_uses_exact_m42_through_m47(tmp_path: Path) -> None:
    root, checkpoint, fallback = _parent_reference(tmp_path)
    for generation in range(42, 48):
        (root / "replay" / f"iter-{generation:02d}-fresh.jsonl").write_text(
            "", encoding="utf-8"
        )
    adapter = Torus9TrainingAdapter(
        profile=_run_profile(
            lr=0.0003,
            replay_generations=6,
            replay_cap=40000,
            simulations=128,
        )
    )

    sources = adapter._reference_sources(checkpoint, (fallback,))
    assert [path.name for path in sources] == [
        "iter-42-fresh.jsonl",
        "iter-43-fresh.jsonl",
        "iter-44-fresh.jsonl",
        "iter-45-fresh.jsonl",
        "iter-46-fresh.jsonl",
        "iter-47-fresh.jsonl",
    ]


def test_parent_rolling_replay_remains_valid_when_child_does_not_expand_scope(
    tmp_path: Path,
) -> None:
    _, checkpoint, fallback = _parent_reference(tmp_path)
    adapter = Torus9TrainingAdapter(
        profile=_run_profile(
            lr=0.0003,
            replay_generations=3,
            replay_cap=20000,
            simulations=128,
        )
    )

    assert adapter._reference_sources(checkpoint, (fallback,)) == (fallback,)


def test_replay6_bootstrap_evicts_to_40k_instead_of_rejecting_source_total(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = Torus9TrainingAdapter(
        profile=_run_profile(
            lr=0.0003,
            replay_generations=6,
            replay_cap=40000,
            simulations=128,
        )
    )
    sources = tuple(tmp_path / f"iter-{generation:02d}-fresh.jsonl" for generation in range(42, 48))
    for source in sources:
        source.write_text("", encoding="utf-8")

    def fake_read(source: Path):
        generation = int(source.name.split("-")[1])
        rows = [
            {
                "source_generation": generation,
                "replay_row_id": f"M{generation}:{index}",
            }
            for index in range(7000)
        ]
        return rows, {"sha256": f"sha256:{generation:064x}", "size_bytes": len(rows)}

    monkeypatch.setattr(run_owned_training._base, "_read_jsonl_with_identity", fake_read)
    monkeypatch.setattr(adapter, "validate_sample", lambda _row: None)

    replay, digests = adapter._rolling_from_sources(sources, total_evictions=0)

    assert len(digests) == 6
    assert len(replay.rows) == 40000
    assert replay.total_evictions == 2000
    assert replay.rows[0]["replay_row_id"] == "M42:2000"
    assert replay.rows[-1]["replay_row_id"] == "M47:6999"
    assert {int(row["source_generation"]) for row in replay.rows} == set(range(42, 48))


def test_torus9_run_spec_ignores_stale_expected_profile_fingerprint(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "configs" / "gocube" / "torus9_plateau_exit_production_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["expected_profile_fingerprint"] = "sha256:" + "0" * 64
    spec_path = tmp_path / "run-spec.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    spec = StrictRunSpec.load(spec_path, repo_root=repo_root)
    assert spec.orchestrator_spec.profile_payload["training"]["learning_rate"] == pytest.approx(0.0003)
    assert spec.orchestrator_spec.profile_payload["replay"]["generations"] == 6
    assert spec.orchestrator_spec.profile_payload["self_play"]["mcts_simulations"] == 128
    assert spec.orchestrator_spec.arena_every_generations == 5


def test_invalid_shapes_still_fail_without_golden_comparison() -> None:
    with pytest.raises(ValueError, match="positive"):
        Torus9SelfPlaySearchContract(simulations=0).validate()


def test_modern_external_parent_still_validates_stage3_metadata() -> None:
    parent_metadata = {
        "checkpoint_schema_version": 1,
        "checkpoint_label": "M47",
        "run_id": "parent",
        "profile_id": TORUS9_CURRENT_PROFILE_ID,
        "profile_fingerprint": "sha256:" + "1" * 64,
        "target_contract_id": TORUS9_TARGET_CONTRACT_ID,
        "target_fingerprint": TORUS9_CURRENT_TARGET_FINGERPRINT,
        "architecture_id": TORUS9_CURRENT_ARCHITECTURE_ID,
        "architecture_config": {},
        "architecture_fingerprint": "sha256:" + "2" * 64,
        "model_hash": "sha256:" + "3" * 64,
        "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
        "board_size": [9, 9],
        "komi": TORUS9_KOMI,
        "network_heads_and_shapes": {},
        "optimizer_updates": 3760,
        "train_samples_consumed": 240640,
        "adam_step": 3760,
        "replay_generations": [45, 46, 47],
        "replay_fingerprint": "sha256:" + "4" * 64,
        "sampled_row_ids_fingerprint": "sha256:" + "5" * 64,
        "training_seed": 123,
        "optimizer_parameter_order": ["weight"],
        "optimizer_parameter_groups": [{"lr": 0.001}],
    }
    adapter = Torus9TrainingAdapter(
        profile=_run_profile(
            lr=0.0003,
            replay_generations=6,
            replay_cap=40000,
            simulations=128,
        )
    )

    adapter._validate_checkpoint_metadata(
        parent_metadata,
        require_optimizer=True,
        require_stage3_fields=True,
        allow_profile_reference=True,
    )

    tampered = dict(parent_metadata)
    tampered["adam_step"] = int(tampered["optimizer_updates"]) - 1
    with pytest.raises(ValueError, match="Adam step mismatch"):
        adapter._validate_checkpoint_metadata(
            tampered,
            require_optimizer=True,
            require_stage3_fields=True,
            allow_profile_reference=True,
        )
