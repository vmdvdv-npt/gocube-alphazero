import json
from pathlib import Path

import torch

from gocube_golden.artifact_catalog import ArtifactCatalog
from gocube_golden.artifact_resolver import ArtifactResolver
from gocube_golden.provenance import file_sha256
from gocube_golden.torus9_m137_5ch import convert_m137_model
from gocube_golden.torus9_m137_5ch import M137_FIVE_CHANNEL_ARCHITECTURE_ID
from gocube_golden.torus9_monolith import TORUS9_TOPOLOGY_FINGERPRINT, Torus9CurrentGraphNet
from gocube_golden.torus9_new_komi import (
    NEW_KOMI_CONVERTED_MODEL_HASH,
    NEW_KOMI_LINEAGE_ID,
    NEW_KOMI_OPTIMIZER_CONVERSION,
    NEW_KOMI_REPLAY_POLICY,
    migrate_m137_adam,
    publish_new_komi_bootstrap_graph,
)


def _prime_adam(model: Torus9CurrentGraphNet) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    observation = torch.zeros((2, 6, 81), dtype=torch.float32)
    observation[:, 5, :].fill_(0.5)
    policy, value, ownership, score = model.forward_auxiliary(observation)
    loss = (
        policy.square().mean()
        + value.square().mean()
        + ownership.square().mean()
        + score.square().mean()
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return optimizer


def test_new_komi_identity_is_explicit_and_fresh_history() -> None:
    assert NEW_KOMI_LINEAGE_ID == "new_komi"
    assert NEW_KOMI_REPLAY_POLICY == "fresh-only-no-parent-history"
    assert "reset-folded-bias-moments" in NEW_KOMI_OPTIMIZER_CONVERSION


def test_new_komi_bootstrap_is_published_as_resolvable_v2_genesis(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    root = runs_root / "torus9" / "active" / NEW_KOMI_LINEAGE_ID
    checkpoint = root / "checkpoints" / "M137-5CH-bootstrap.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test-only immutable M137 5CH artifact")
    artifact_sha = file_sha256(checkpoint)
    metadata = {
        "architecture_id": M137_FIVE_CHANNEL_ARCHITECTURE_ID,
        "architecture_config": {
            "topology_id": "torus-9x9-row-major-v1",
            "topology_fingerprint": TORUS9_TOPOLOGY_FINGERPRINT,
            "input_channels": 5,
            "hidden": 80,
            "blocks": 8,
        },
        "observation_shape": [5, 81],
        "converted_model_hash": NEW_KOMI_CONVERTED_MODEL_HASH,
        "source_checkpoint": "M137",
        "source_checkpoint_sha256": "sha256:71cfc78dab3fe217b3c435a765790efe6f6c4fa42a7d21479f3fd909adf341fe",
        "conversion_formula": "W5 = W[:, 0:5]; b5 = b + 0.5 * W[:, 5]",
        "optimizer_conversion": "not-performed",
    }
    checkpoint.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "lineage_id": NEW_KOMI_LINEAGE_ID,
        "topology": "torus9",
        "status": "ACTIVE",
        "parent_checkpoint": {
            "lineage_id": "torus9-m125-continuous-v2-gen6-20260922-v1",
            "checkpoint_id": "M137",
            "generation": 137,
            "sha256": metadata["source_checkpoint_sha256"],
        },
        "bootstrap_checkpoint": {
            "path": "checkpoints/M137-5CH-bootstrap.pt",
            "sha256": artifact_sha,
            "model_hash": NEW_KOMI_CONVERTED_MODEL_HASH,
        },
        "checkpoint_hashes": {"M137-5CH-bootstrap": artifact_sha},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    ArtifactCatalog.initialize(
        root / "runtime" / "artifact-catalog.json",
        lineage_id=NEW_KOMI_LINEAGE_ID,
        root=root,
    )

    evidence = publish_new_komi_bootstrap_graph(root)
    resolved = ArtifactResolver(runs_root).checkpoint(evidence["checkpoint"])

    assert resolved.node.genesis is True
    assert resolved.ref.checkpoint_id == "M137-5CH-bootstrap"
    assert resolved.ref.generation == 137
    assert resolved.effective_config.config.compatibility["input_channels"] == 5
    assert resolved.effective_config.config.compatibility["model_hash"] == NEW_KOMI_CONVERTED_MODEL_HASH
    assert resolved.provenance.path.name == "M137-5CH-bootstrap.json"


def test_m137_adam_migration_preserves_exact_states_and_resets_only_folded_bias() -> None:
    torch.manual_seed(137)
    source = Torus9CurrentGraphNet()
    source_optimizer = _prime_adam(source)
    source_state = source_optimizer.state_dict()
    target = convert_m137_model(source)

    target_optimizer, report = migrate_m137_adam(
        source_model=source,
        target_model=target,
        source_optimizer_state=source_state,
    )
    target_state = target_optimizer.state_dict()

    source_names = [name for name, _ in source.named_parameters()]
    source_ids = source_state["param_groups"][0]["params"]
    target_ids = target_state["param_groups"][0]["params"]
    source_by_name = {
        name: source_state["state"][parameter_id]
        for name, parameter_id in zip(source_names, source_ids)
    }
    target_by_name = {
        name: target_state["state"][parameter_id]
        for name, parameter_id in zip(source_names, target_ids)
    }

    source_weight = source_by_name["input_projection.weight"]
    target_weight = target_by_name["input_projection.weight"]
    assert torch.equal(target_weight["exp_avg"], source_weight["exp_avg"][:, :5])
    assert torch.equal(target_weight["exp_avg_sq"], source_weight["exp_avg_sq"][:, :5])
    assert int(target_weight["step"].item()) == int(source_weight["step"].item())

    source_bias = source_by_name["input_projection.bias"]
    target_bias = target_by_name["input_projection.bias"]
    assert torch.count_nonzero(target_bias["exp_avg"]).item() == 0
    assert torch.count_nonzero(target_bias["exp_avg_sq"]).item() == 0
    assert int(target_bias["step"].item()) == int(source_bias["step"].item())

    unchanged_name = next(
        name
        for name in source_names
        if name not in {"input_projection.weight", "input_projection.bias"}
    )
    for key, value in source_by_name[unchanged_name].items():
        migrated = target_by_name[unchanged_name][key]
        if torch.is_tensor(value):
            assert torch.equal(migrated, value)
        else:
            assert migrated == value

    steps = {
        int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
        for state in target_state["state"].values()
        if "step" in state
    }
    assert len(steps) == 1
    assert report["reset_parameter_moments"] == ["input_projection.bias"]
    assert report["cropped_parameter_states"] == ["input_projection.weight"]
