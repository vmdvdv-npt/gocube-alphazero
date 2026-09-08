from pathlib import Path

import pytest

from alphazero.envs.gocube.hardened_train import build_hardened_training_args
from alphazero.envs.gocube.katago_train import parse_args
from alphazero.envs.gocube.reproducible_manifest import (
    create_reproducible_manifest,
    effective_config,
    validate_existing_reproducible_manifest,
)
from alphazero.envs.gocube.reproducibility import derive_worker_seed
from alphazero.envs.gocube.contract_versions import (
    REPLAY_FORMAT_VERSION,
    SCORE_INITIALIZATION_CONTRACT,
    TARGET_PROVENANCE_ENCODING,
    TARGET_PROVENANCE_SEMANTICS,
    TERMINATION_CONTRACT,
    VALUE_TARGET_SEMANTICS,
)


def test_seed_derivation_is_stable_and_coordinate_sensitive():
    baseline = derive_worker_seed(123, 2, 3, 4, 5)
    assert baseline == 10496753387030779032
    assert baseline == derive_worker_seed(123, 2, 3, 4, 5)
    assert baseline != derive_worker_seed(124, 2, 3, 4, 5)
    assert baseline != derive_worker_seed(123, 2, 3, 4, 6)


def test_existing_run_manifest_rejects_changed_seed(tmp_path):
    game_cls, args = build_hardened_training_args(parse_args([]))
    create_reproducible_manifest(
        checkpoint_dir=tmp_path,
        run_name="manifest-test",
        game_cls=game_cls,
        args=args,
        argv=["test"],
        allow_dirty_source=True,
    )

    validate_existing_reproducible_manifest(
        checkpoint_dir=tmp_path,
        run_name="manifest-test",
        game_cls=game_cls,
        args=args,
        allow_dirty_source=True,
    )
    args.master_seed += 1
    with pytest.raises(RuntimeError, match="immutable run metadata conflicts"):
        validate_existing_reproducible_manifest(
            checkpoint_dir=tmp_path,
            run_name="manifest-test",
            game_cls=game_cls,
            args=args,
            allow_dirty_source=True,
        )


def test_effective_config_contains_pinned_immutable_fields():
    game_cls, args = build_hardened_training_args(parse_args([]))
    config = effective_config(args, game_cls)
    assert config["komi"] == 0.5
    assert config["katago_reference_commit"] == game_cls.KATAGO_REFERENCE_COMMIT
    assert config["replay_format_version"] == REPLAY_FORMAT_VERSION == 4
    assert config["value_target_semantics"] == VALUE_TARGET_SEMANTICS
    assert config["score_initialization_contract"] == SCORE_INITIALIZATION_CONTRACT
    assert config["target_provenance_semantics"] == TARGET_PROVENANCE_SEMANTICS
    assert config["target_provenance_encoding"] == TARGET_PROVENANCE_ENCODING
    assert config["termination_contract"] == TERMINATION_CONTRACT
    assert config["sample_clock_contract"] == "sample-clock-v2"
