from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .atomic_io import atomic_json_write
from .contract_versions import (
    OWNERSHIP_TARGET_SEMANTICS,
    REPLAY_FORMAT_VERSION,
    SCORE_TARGET_SEMANTICS,
    SEED_DERIVATION_CONTRACT,
    TRAINING_CONTRACT_VERSION,
    VALUE_TARGET_SEMANTICS,
)
from .parameter_origins import classify_parameters
from .records import effective_parameter_snapshot
from .integration.contract import resolve_model_contract

RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "run-manifest.json"
EFFECTIVE_CONFIG_FILENAME = "effective-config.json"
ENVIRONMENT_FILENAME = "environment.txt"
_RESUME_MANIFEST_FIELDS = (
    "katago_reference_commit",
    "rules_fingerprint",
    "komi",
    "master_seed",
    "replay_format_version",
    "value_target_semantics",
    "score_target_semantics",
    "ownership_target_semantics",
    "sample_clock_contract",
    "training_contract_version",
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_artifact_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_json_artifact(payload: object) -> str:
    return hashlib.sha256(_json_artifact_bytes(payload)).hexdigest()


def effective_config(args: Any, game_cls) -> dict[str, object]:
    contract = resolve_model_contract(game_cls, args)
    config = effective_parameter_snapshot(args)
    config.update({
        "topology": game_cls.topology_kind(),
        "size": int(game_cls.board_size()),
        "point_count": int(game_cls.logical_topology().point_count),
        "komi": float(game_cls.KOMI),
        "ko_rule": "SIMPLE",
        "scoring_rule": "TERRITORY",
        "tax_rule": "SEKI",
        "suicide": "disabled",
        "button": "disabled",
        "white_handicap_bonus": 0,
        "katago_reference_commit": game_cls.KATAGO_REFERENCE_COMMIT,
        "rules_implementation_version": 3,
        "network_type": getattr(args, "nnet_type", None),
        "observation_shape": list(game_cls.observation_size()),
        "action_size": int(game_cls.action_size()),
        "value_output_size": 3,
        "replay_format_version": REPLAY_FORMAT_VERSION,
        "value_target_semantics": VALUE_TARGET_SEMANTICS,
        "score_target_semantics": SCORE_TARGET_SEMANTICS,
        "ownership_target_semantics": OWNERSHIP_TARGET_SEMANTICS,
        "sample_clock_contract": getattr(args, "gocube_training_contract", "sample-clock-v2"),
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "seed_derivation_contract": SEED_DERIVATION_CONTRACT,
    })
    return dict(sorted(config.items()))


def validate_existing_reproducible_manifest(
    *,
    checkpoint_dir: str | os.PathLike[str],
    run_name: str,
    game_cls,
    args: Any,
    allow_dirty_source: bool = False,
) -> dict[str, object]:
    """Validate immutable run identity before loading an existing run.

    Resume must never repair or replace the rich manifest.  The effective
    configuration is compared from its canonical JSON representation so a
    changed optimizer, topology, target contract, or seed fails before any
    checkpoint/replay state is opened.
    """

    root = Path(__file__).resolve().parents[3]
    checkpoint_run = Path(checkpoint_dir) / run_name
    manifest_path = checkpoint_run / RUN_MANIFEST_FILENAME
    config_path = checkpoint_run / EFFECTIVE_CONFIG_FILENAME
    if not manifest_path.exists() or not config_path.exists():
        raise RuntimeError(
            "Refusing resume: existing run is missing the immutable "
            f"{RUN_MANIFEST_FILENAME} or {EFFECTIVE_CONFIG_FILENAME} artifact"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Refusing resume: cannot read immutable run metadata: {exc}") from exc

    if manifest.get("manifest_schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("Refusing resume: unsupported run manifest schema")
    if manifest.get("run_name") != run_name:
        raise RuntimeError("Refusing resume: run manifest name does not match requested run")

    current_commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain")
    if dirty and not allow_dirty_source:
        raise RuntimeError(
            "Refusing production resume from dirty source tree. "
            "Use --allow-dirty-source only for an explicitly experimental run."
        )

    current_config = effective_config(args, game_cls)
    mismatches = []
    if saved_config != current_config:
        for key in sorted(set(saved_config) | set(current_config)):
            if saved_config.get(key) != current_config.get(key):
                mismatches.append(
                    f"effective_config.{key}: saved={saved_config.get(key)!r}, "
                    f"current={current_config.get(key)!r}"
                )
    expected_config_hash = _sha256_json_artifact(current_config)
    if manifest.get("effective_config_sha256") != expected_config_hash:
        mismatches.append("effective_config_sha256")
    if manifest.get("repository_commit") != current_commit:
        mismatches.append(
            f"repository_commit: saved={manifest.get('repository_commit')!r}, "
            f"current={current_commit!r}"
        )

    expected_manifest_values = {
        "katago_reference_commit": game_cls.KATAGO_REFERENCE_COMMIT,
        "rules_fingerprint": game_cls.rules_fingerprint(),
        "komi": float(game_cls.KOMI),
        "master_seed": int(getattr(args, "master_seed", 0)),
        "replay_format_version": REPLAY_FORMAT_VERSION,
        "value_target_semantics": VALUE_TARGET_SEMANTICS,
        "score_target_semantics": SCORE_TARGET_SEMANTICS,
        "ownership_target_semantics": OWNERSHIP_TARGET_SEMANTICS,
        "sample_clock_contract": getattr(args, "gocube_training_contract", "sample-clock-v2"),
        "training_contract_version": TRAINING_CONTRACT_VERSION,
    }
    for key in _RESUME_MANIFEST_FIELDS:
        if manifest.get(key) != expected_manifest_values[key]:
            mismatches.append(
                f"{key}: saved={manifest.get(key)!r}, current={expected_manifest_values[key]!r}"
            )
    if manifest.get("model_contract") is not None and manifest.get("model_contract") != current_config["model_contract"]:
        mismatches.append("model_contract: run-manifest conflicts with effective-config")
    if mismatches:
        raise RuntimeError(
            "Refusing resume: immutable run metadata conflicts: "
            + "; ".join(mismatches)
        )
    return manifest


def _environment_artifact(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        path.write_text(result.stdout, encoding="utf-8")
        return

    # Some project virtualenvs intentionally omit pip.  The manifest still
    # needs a deterministic environment record without bootstrapping or
    # modifying that environment.
    from importlib import metadata

    distributions = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    )
    path.write_text("\n".join(distributions) + "\n", encoding="utf-8")


def create_reproducible_manifest(
    *,
    checkpoint_dir: str | os.PathLike[str],
    run_name: str,
    game_cls,
    args: Any,
    argv: list[str] | None = None,
    allow_dirty_source: bool = False,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    checkpoint_run = Path(checkpoint_dir) / run_name
    checkpoint_run.mkdir(parents=True, exist_ok=True)

    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    dirty = _git(root, "status", "--porcelain")
    if dirty and not allow_dirty_source:
        raise RuntimeError(
            "Refusing production run from dirty source tree. "
            "Use --allow-dirty-source only for an explicitly experimental run."
        )

    contract = resolve_model_contract(game_cls, args)
    config = effective_config(args, game_cls)
    config_path = checkpoint_run / EFFECTIVE_CONFIG_FILENAME
    atomic_json_write(config, config_path)
    config_hash = _sha256_json_artifact(config)
    environment_path = checkpoint_run / ENVIRONMENT_FILENAME
    _environment_artifact(environment_path)

    source_patch_hash = None
    if dirty:
        patch_path = checkpoint_run / "source.patch"
        patch = _git(root, "diff") + _git(root, "diff", "--cached")
        patch_path.write_text(patch, encoding="utf-8")
        source_patch_hash = _sha256_file(patch_path)

    cuda_names = []
    if torch.cuda.is_available():
        cuda_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    manifest = {
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_name": run_name,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "canonical_entrypoint": "tools/run_cube4_katago_from_scratch.sh",
        "argv": list(sys.argv if argv is None else argv),
        "repository_commit": commit,
        "repository_branch": branch,
        "repository_dirty": bool(dirty),
        "source_patch_sha256": source_patch_hash,
        "katago_reference_commit": game_cls.KATAGO_REFERENCE_COMMIT,
        "rules_fingerprint": game_cls.rules_fingerprint(),
        "model_contract": contract.to_dict(),
        "topology": contract.topology_kind,
        "size": contract.topology_size,
        "point_count": contract.point_count,
        "observation_schema": contract.observation_schema,
        "observation_shape": list(contract.observation_shape),
        "action_schema": contract.action_schema,
        "action_size": contract.action_size,
        "network_architecture_id": contract.network_architecture_id,
        "search_contract_id": contract.search_contract_id,
        "terminal_adjudicator": contract.terminal_adjudicator_id,
        "python_executable": os.path.abspath(sys.executable),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": __import__("numpy").__version__,
        "cython_version": __import__("Cython").__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_names": cuda_names,
        "environment_file": ENVIRONMENT_FILENAME,
        "environment_sha256": _sha256_file(environment_path),
        "effective_config_file": EFFECTIVE_CONFIG_FILENAME,
        "effective_config_sha256": config_hash,
        "parameter_origins": classify_parameters(config),
        "master_seed": int(getattr(args, "master_seed", 0)),
        "seed_derivation_contract": SEED_DERIVATION_CONTRACT,
        "replay_format_version": REPLAY_FORMAT_VERSION,
        "value_target_semantics": VALUE_TARGET_SEMANTICS,
        "score_target_semantics": SCORE_TARGET_SEMANTICS,
        "ownership_target_semantics": OWNERSHIP_TARGET_SEMANTICS,
        "sample_clock_contract": getattr(args, "gocube_training_contract", "sample-clock-v2"),
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "komi": float(game_cls.KOMI),
    }
    atomic_json_write(manifest, checkpoint_run / RUN_MANIFEST_FILENAME)
    return manifest
