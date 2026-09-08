"""Immutable, fail-closed contract for the GoCube B0/B1 experiment.

The model, search, rules, and training contracts remain authoritative for
their own domains.  This module only composes their resolved identities into
one experiment specification and validates the common B0/B1 surface.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contract_versions import (
    OWNERSHIP_TARGET_SEMANTICS,
    REPLAY_FORMAT_VERSION,
    SCORE_INITIALIZATION_CONTRACT,
    SCORE_TARGET_SEMANTICS,
    TARGET_PROVENANCE_ENCODING,
    TARGET_PROVENANCE_SEMANTICS,
    TERMINATION_CONTRACT,
    TRAINING_CONTRACT_VERSION,
    VALUE_TARGET_SEMANTICS,
)
from .production_contract import CUBE4_PRODUCTION, GOCUBE_KOMI
from .production_training import (
    SampleBudgetTarget,
    build_sample_budget_target,
)


B_EXPERIMENT_CONTRACT_ID = "gocube-b-experiment-contract-v1"
B_EXPERIMENT_CONTRACT_VERSION = 1
B0_TREATMENT = "B0"
B1_TREATMENT = "B1"
B0_MODEL_PROFILE = "baseline"
B1_MODEL_PROFILE = "g1"
DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET = 40_000_000
EVALUATION_MILESTONE_FRACTIONS = (0.25, 0.50, 0.75, 1.0)
B_SEED_LIST = (0, 1, 2, 3, 4)
B_INITIAL_SEED_COUNT = 3
B_EXTENSION_SEED_COUNT = 5
B_MANDATORY_SEEDS = (0, 1, 2)
B_EXTENSION_SEEDS = (3, 4)
B_EXTENSION_SEED_CRITERION_ID = (
    "extend-to-five-seeds-only-if-mandatory-seed-bootstrap-ambiguity-or-variance-v1"
)


def _evaluation_milestones(scientific_target: SampleBudgetTarget) -> dict[str, object]:
    """Describe scientific checkpoints on the cumulative sample clock.

    Iteration numbers remain useful for operations and recovery, but they are
    deliberately not scientific comparison milestones because realized rows
    per generation chunk can differ between treatments.
    """

    target = int(scientific_target.target)
    return {
        "clock": scientific_target.kind,
        "counter": scientific_target.counter_key,
        "target": target,
        "milestone_fractions": list(EVALUATION_MILESTONE_FRACTIONS),
        "milestone_targets": [
            int(round(target * fraction)) for fraction in EVALUATION_MILESTONE_FRACTIONS
        ],
        "comparison_rule": "paired-at-equal-cumulative-sample-budget-v1",
        "operational_metadata": {
            "bootstrap_iteration": 7,
            "health_reference_iteration": 4,
            "arena_anchor_period_iterations": 10,
            "heldout_positions": 16,
        },
    }

# These are semantic paths in the JSON effective-config artifact.  A whole
# model contract is not whitelisted: rules/search/training changes inside it
# must still fail closed.
ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES = (
    "gocube_model_profile",
    "gocube_network_architecture",
    "gocube_structural_feature_schema",
    "gocube_structural_feature_channels",
    "gocube_observation_schema",
    "model_profile",
    "game_class_id",
    "network_architecture_id",
    "network_architecture_fingerprint",
    "observation_schema",
    "observation_shape",
    "model_contract.gameClassId",
    "model_contract.observationSchema",
    "model_contract.observationShape",
    "model_contract.networkArchitectureId",
    "model_contract.networkArchitectureFingerprint",
    # Run-specific identity/path fields are not treatment semantics.
    "run_name",
    "gocube_record_root",
    "checkpoint",
    "data",
)
B_EFFECTIVE_CONFIG_WHITELIST = ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES


class ExperimentContractError(ValueError):
    """Raised when a B experiment contract or preflight is unsafe."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def current_source_git_sha(repo: str | os.PathLike[str] | None = None) -> str:
    root = Path(repo).resolve() if repo is not None else _repo_root()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sha = result.stdout.strip()
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha.lower()):
        raise ExperimentContractError(f"Invalid source git SHA: {sha!r}")
    return sha


def hash_heldout_suite(path: str | os.PathLike[str]) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise ExperimentContractError(
            f"heldout suite must be a real frozen artifact file: {artifact}"
        )
    if artifact.stat().st_size == 0:
        raise ExperimentContractError("heldout suite artifact must not be empty")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_clean_source(repo: str | os.PathLike[str] | None = None) -> None:
    """Reject a B run whose source SHA would not describe the working tree."""

    root = Path(repo).resolve() if repo is not None else _repo_root()
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ExperimentContractError(
            "B experiment requires a clean committed source tree; "
            "uncommitted or untracked changes are present"
        )


def validate_extension_seed_decision(
    path: str | os.PathLike[str],
    *,
    contract_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the pre-registered decision that permits seeds 3 and 4."""

    decision_path = Path(path)
    try:
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentContractError(
            f"Cannot read extension-seed decision: {decision_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExperimentContractError("Extension-seed decision must be a JSON object")
    expected = {
        "approved": True,
        "decision": "extend_to_five",
        "criterion_id": B_EXTENSION_SEED_CRITERION_ID,
        "mandatory_seed_count": B_INITIAL_SEED_COUNT,
        "extension_seed_count": B_EXTENSION_SEED_COUNT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ExperimentContractError(
                f"Extension-seed decision {key} drift: expected {value!r}, "
                f"got {payload.get(key)!r}"
            )
    evidence = payload.get("criterion_evidence")
    if not isinstance(evidence, Mapping) or not (
        bool(evidence.get("ambiguity_detected"))
        or bool(evidence.get("variance_exceeded"))
    ):
        raise ExperimentContractError(
            "Extension-seed decision must record ambiguity or variance evidence"
        )
    if contract_sha256 is not None and payload.get("experiment_contract_sha256") != contract_sha256:
        raise ExperimentContractError(
            "Extension-seed decision is for a different experiment contract"
        )
    return dict(payload)


@dataclass(frozen=True)
class BExperimentContract:
    """Deeply immutable experiment specification for B0 versus B1."""

    contract_id: str
    contract_version: int
    source_git_sha: str
    topology: str
    board_size: int
    rules_id: str
    rules_fingerprint: str
    komi: float
    search_contract_id: str
    target_semantics: Mapping[str, object]
    termination_contract: str
    b0_model_profile: str
    b0_model_contract: Mapping[str, object]
    b0_structural_feature_schema: str | None
    b0_structural_feature_channels: int
    b1_model_profile: str
    b1_model_contract: Mapping[str, object]
    b1_structural_feature_schema: str | None
    b1_structural_feature_channels: int
    b0_observation_schema: str
    b1_observation_schema: str
    self_play_simulations: int
    fast_simulations: int
    fast_probability: float
    arena_simulations: int
    arena_temperature_noise_policy: Mapping[str, object]
    training_batch_size: int
    replay_window: Mapping[str, object]
    train_samples_per_new_sample: float
    optimizer_settings: Mapping[str, object]
    scheduler_settings: Mapping[str, object]
    worker_count: int
    generation_chunk_size: int
    scientific_sample_target: Mapping[str, object]
    seed_list: tuple[int, ...]
    initial_seed_count: int
    extension_seed_count: int
    mandatory_seed_list: tuple[int, ...]
    extension_seed_list: tuple[int, ...]
    extension_seed_activation_criterion: str
    evaluation_milestones: Mapping[str, object]
    heldout_suite_hash: str
    result_semantics: Mapping[str, object]
    no_result_evaluation_convention: str
    primary_endpoint: str
    statistical_method_identifier: str
    canonical_common_effective_config: Mapping[str, object]
    allowed_effective_config_differences: tuple[str, ...] = ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Mapping):
                object.__setattr__(self, field.name, _freeze(value))
            elif isinstance(value, list):
                object.__setattr__(self, field.name, tuple(_freeze(item) for item in value))
        if not isinstance(self.seed_list, tuple):
            object.__setattr__(self, "seed_list", tuple(int(seed) for seed in self.seed_list))
        for name in ("mandatory_seed_list", "extension_seed_list"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(int(seed) for seed in value))
        if not isinstance(self.allowed_effective_config_differences, tuple):
            object.__setattr__(
                self,
                "allowed_effective_config_differences",
                tuple(self.allowed_effective_config_differences),
            )

    @property
    def version(self) -> int:
        return self.contract_version

    @property
    def board(self) -> int:
        return self.board_size

    @property
    def train_batch_size(self) -> int:
        """Compatibility alias used by the production launcher vocabulary."""

        return self.training_batch_size

    @property
    def regular_sims(self) -> int:
        return self.self_play_simulations

    @property
    def fast_sims(self) -> int:
        return self.fast_simulations

    @property
    def arena_sims(self) -> int:
        return self.arena_simulations

    @property
    def arena_temperatures_noise_policy(self) -> Mapping[str, object]:
        return self.arena_temperature_noise_policy

    @property
    def no_result_convention(self) -> str:
        return self.no_result_evaluation_convention

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: _json_safe(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BExperimentContract":
        if not isinstance(payload, Mapping):
            raise ExperimentContractError("B experiment contract must be a JSON object")
        values = {}
        for field in dataclasses.fields(cls):
            if field.name in payload:
                values[field.name] = payload[field.name]
        missing = [
            field.name
            for field in dataclasses.fields(cls)
            if field.name not in values and field.default is dataclasses.MISSING
        ]
        if missing:
            raise ExperimentContractError(
                "B experiment contract is missing fields: " + ", ".join(missing)
            )
        try:
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise ExperimentContractError(f"Invalid B experiment contract: {exc}") from exc

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


def _resolved_profile_contracts(
    scientific_target: SampleBudgetTarget | None = None,
) -> tuple[object, object, object, object]:
    """Resolve B0/B1 from the real hardened builders, not hand-written copies."""

    from .hardened_train import build_hardened_training_args
    from .integration.contract import resolve_model_contract
    from .katago_train import parse_args
    from .reproducible_manifest import effective_config

    resolved = []
    target_args = []
    if scientific_target is not None:
        target_args = [
            (
                "--cumulative-new-samples-target"
                if scientific_target.kind == SampleBudgetTarget.NEW_SAMPLES
                else "--cumulative-optimizer-examples-target"
            ),
            str(scientific_target.target),
        ]
    for profile in (B0_MODEL_PROFILE, B1_MODEL_PROFILE):
        cli = parse_args(
            [
                "--model-profile",
                profile,
                "--topology",
                "cube",
                "--size",
                "4",
                "--workers",
                str(CUBE4_PRODUCTION.workers),
                "--sims",
                str(CUBE4_PRODUCTION.regular_sims),
                "--arena-sims",
                str(CUBE4_PRODUCTION.arena_sims),
                "--games-per-iteration",
                str(CUBE4_PRODUCTION.games_per_iteration),
                "--train-batch-size",
                str(CUBE4_PRODUCTION.train_batch_size),
                "--fast-game-prob",
                "0.25",
                "--no-arena",
                "--run-name",
                f"gocube-b-preflight-{profile}",
                *target_args,
            ]
        )
        game_cls, args = build_hardened_training_args(cli)
        resolved.append((resolve_model_contract(game_cls, args), effective_config(args, game_cls)))
    return resolved[0][0], resolved[0][1], resolved[1][0], resolved[1][1]


def build_b_experiment_contract(
    *,
    source_git_sha: str | None = None,
    heldout_suite_hash: str | None = None,
    repo: str | os.PathLike[str] | None = None,
    scientific_target: SampleBudgetTarget | None = None,
) -> BExperimentContract:
    """Build the canonical B contract from the live hardened config builders."""

    if scientific_target is None:
        scientific_target = SampleBudgetTarget(
            SampleBudgetTarget.NEW_SAMPLES,
            DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET,
        )
    b0_model, b0_config, b1_model, b1_config = _resolved_profile_contracts(scientific_target)
    if source_git_sha is None:
        source_git_sha = current_source_git_sha(repo)
    if heldout_suite_hash is None:
        raise ExperimentContractError(
            "heldout_suite_hash must come from a real frozen heldout-suite artifact"
        )
    rules_fingerprint = str(b0_model.rules_fingerprint)
    if rules_fingerprint != str(b1_model.rules_fingerprint):
        raise ExperimentContractError("B0 and B1 resolve different rules fingerprints")
    if b0_model.search_contract_id != b1_model.search_contract_id:
        raise ExperimentContractError("B0 and B1 resolve different search contracts")

    return BExperimentContract(
        contract_id=B_EXPERIMENT_CONTRACT_ID,
        contract_version=B_EXPERIMENT_CONTRACT_VERSION,
        source_git_sha=str(source_git_sha),
        topology="cube",
        board_size=4,
        rules_id=str(b0_model.terminal_adjudicator_id),
        rules_fingerprint=rules_fingerprint,
        komi=GOCUBE_KOMI,
        search_contract_id=str(b0_model.search_contract_id),
        target_semantics={
            "value": VALUE_TARGET_SEMANTICS,
            "score": SCORE_TARGET_SEMANTICS,
            "ownership": OWNERSHIP_TARGET_SEMANTICS,
            "provenance": TARGET_PROVENANCE_SEMANTICS,
            "provenance_encoding": TARGET_PROVENANCE_ENCODING,
            "replay_format_version": REPLAY_FORMAT_VERSION,
            "training_contract_version": TRAINING_CONTRACT_VERSION,
            "score_initialization": SCORE_INITIALIZATION_CONTRACT,
        },
        termination_contract=TERMINATION_CONTRACT,
        b0_model_profile=B0_MODEL_PROFILE,
        b0_model_contract=b0_model.to_dict(),
        b0_structural_feature_schema=b0_config.get("gocube_structural_feature_schema"),
        b0_structural_feature_channels=int(b0_config.get("gocube_structural_feature_channels", 0)),
        b1_model_profile=B1_MODEL_PROFILE,
        b1_model_contract=b1_model.to_dict(),
        b1_structural_feature_schema=b1_config.get("gocube_structural_feature_schema"),
        b1_structural_feature_channels=int(b1_config.get("gocube_structural_feature_channels", 0)),
        b0_observation_schema=str(b0_model.observation_schema),
        b1_observation_schema=str(b1_model.observation_schema),
        self_play_simulations=CUBE4_PRODUCTION.regular_sims,
        fast_simulations=CUBE4_PRODUCTION.fast_sims,
        fast_probability=CUBE4_PRODUCTION.fast_probability,
        arena_simulations=CUBE4_PRODUCTION.arena_sims,
        arena_temperature_noise_policy={
            "temperature": 0.0,
            "root_noise": False,
            "root_temperature": False,
            "batched": True,
            "chosen_move_temperature": 0.0,
        },
        training_batch_size=CUBE4_PRODUCTION.train_batch_size,
        replay_window={
            "mode": "production-schedule",
            "min_iterations": 4,
            "max_iterations": 20,
            "increment_iterations": 2,
        },
        train_samples_per_new_sample=CUBE4_PRODUCTION.train_samples_per_new_sample,
        optimizer_settings={
            "identifier": "torch.optim.SGD",
            "learning_rate": 0.01,
            "arguments": {"momentum": 0.9, "weight_decay": 1e-4},
        },
        scheduler_settings={
            "identifier": "gocube-sample-clock-v2",
            "warmup_samples": 2_000_000,
            "warmup_start_factor": 0.05,
            "milestone_samples": [20_000_000, 40_000_000],
            "decay_gamma": 0.1,
            "gradient_clip_norm": 5.0,
        },
        worker_count=CUBE4_PRODUCTION.workers,
        generation_chunk_size=CUBE4_PRODUCTION.games_per_iteration,
        scientific_sample_target={
            "clock": "cumulative-training-counter",
            "kind": scientific_target.kind,
            "counter": scientific_target.counter_key,
            "target": int(scientific_target.target),
            "generation_chunk_games": CUBE4_PRODUCTION.games_per_iteration,
        },
        seed_list=B_SEED_LIST,
        initial_seed_count=B_INITIAL_SEED_COUNT,
        extension_seed_count=B_EXTENSION_SEED_COUNT,
        mandatory_seed_list=B_MANDATORY_SEEDS,
        extension_seed_list=B_EXTENSION_SEEDS,
        extension_seed_activation_criterion=B_EXTENSION_SEED_CRITERION_ID,
        evaluation_milestones=_evaluation_milestones(scientific_target),
        heldout_suite_hash=str(heldout_suite_hash),
        result_semantics={
            "win": 1.0,
            "draw": 0.5,
            "no_result": 0.5,
            "loss": 0.0,
            "metric": "paired-position-score",
            "unit": "paired-starting-position",
            "reported_counts": ["wins", "losses", "draws", "no_results"],
        },
        no_result_evaluation_convention=(
            "NO_RESULT contributes 0.5 to the paired position score and remains in the denominator"
        ),
        primary_endpoint="heldout_paired_position_score",
        statistical_method_identifier=(
            "hierarchical-paired-bootstrap-seeds-to-starting-position-pairs-v1"
        ),
        canonical_common_effective_config=_strip_allowed_effective_config(b0_config),
    )


def _walk_differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_walk_differences(left[key], right[key], child))
        return paths
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [path]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(_walk_differences(left_item, right_item, f"{path}[{index}]"))
        return paths
    return [] if left == right else [path]


def _allowed_difference(path: str, allowed: tuple[str, ...]) -> bool:
    if path in allowed:
        return True
    # Shape is a single semantic field; list element paths are reported by the
    # recursive walker but inherit that field's whitelist entry.
    for candidate in allowed:
        if path.startswith(candidate + "["):
            return True
    if path in {"run_name", "checkpoint", "data", "gocube_record_root"}:
        return True
    return False


_OMIT = object()


def _strip_allowed_effective_config(
    value: Any,
    path: str = "",
    *,
    allowed: tuple[str, ...] = ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES,
) -> Any:
    """Remove treatment-specific and run-identity fields from a config.

    The resulting snapshot is the canonical common surface. Comparing every
    treatment against that snapshot catches a drift applied identically to B0
    and B1, which a B0-vs-B1 diff alone cannot see.
    """

    if path and _allowed_difference(path, allowed):
        return _OMIT
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            normalized = _strip_allowed_effective_config(item, child, allowed=allowed)
            if normalized is not _OMIT:
                result[str(key)] = normalized
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            normalized = _strip_allowed_effective_config(item, child, allowed=allowed)
            if normalized is not _OMIT:
                result.append(normalized)
        return result
    return value


def _effective_config_common_differences(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> list[str]:
    return _walk_differences(expected, _strip_allowed_effective_config(actual))


def diff_effective_configs(
    b0_effective_config: Mapping[str, object],
    b1_effective_config: Mapping[str, object],
    *,
    allowed: tuple[str, ...] = ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES,
) -> list[str]:
    """Return only semantic B0/B1 differences outside the whitelist."""

    all_differences = _walk_differences(b0_effective_config, b1_effective_config)
    return [path for path in all_differences if not _allowed_difference(path, allowed)]


# Descriptive aliases make the preflight API discoverable without exposing a
# second implementation.
compare_effective_configs = diff_effective_configs
effective_config_diff = diff_effective_configs


def _config_value(config: Mapping[str, object], *keys: str) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return None


def _require_config_value(
    config: Mapping[str, object],
    path: str,
    expected: Any,
    *,
    label: str,
) -> None:
    """Require one effective-config field to equal its canonical value."""

    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ExperimentContractError(
                f"{label} effective config is missing canonical field {path}"
            )
        value = value[part]
    same = value == expected
    if isinstance(expected, float) or isinstance(value, float):
        try:
            same = abs(float(value) - float(expected)) <= 1e-12
        except (TypeError, ValueError):
            same = False
    if not same:
        raise ExperimentContractError(
            f"{label} effective config {path} drift: expected {expected!r}, got {value!r}"
        )


def _validate_treatment_specific_config(
    config: Mapping[str, object],
    *,
    label: str,
    profile: str,
    model_contract: Mapping[str, object],
    observation_schema: str,
    structural_feature_schema: str | None,
    structural_feature_channels: int,
) -> None:
    """Validate the values allowed to differ between B0 and B1.

    Whitelisting a path only permits the B0/B1 difference. It does not permit
    an arbitrary value: every treatment-specific value must still match the
    model contract resolved for that treatment.
    """

    expected_model = _thaw(model_contract)
    expected_architecture = expected_model["networkArchitectureId"]
    expected_fingerprint = expected_model["networkArchitectureFingerprint"]
    expected_shape = expected_model["observationShape"]
    expected_game_class = expected_model["gameClassId"]
    for path, expected in (
        ("gocube_model_profile", profile),
        ("model_profile", profile),
        ("gocube_network_architecture", expected_architecture),
        ("network_architecture_id", expected_architecture),
        ("network_architecture_fingerprint", expected_fingerprint),
        ("gocube_structural_feature_schema", structural_feature_schema),
        ("gocube_structural_feature_channels", structural_feature_channels),
        ("gocube_observation_schema", observation_schema),
        ("observation_shape", expected_shape),
        ("game_class_id", expected_game_class),
        ("model_contract", expected_model),
    ):
        _require_config_value(config, path, expected, label=label)


def validate_b0_b1_effective_configs(
    b0_effective_config: Mapping[str, object],
    b1_effective_config: Mapping[str, object],
    *,
    contract: BExperimentContract | None = None,
) -> list[str]:
    """Validate common fixed settings and reject non-whitelisted drift."""

    if not isinstance(b0_effective_config, Mapping) or not isinstance(b1_effective_config, Mapping):
        raise ExperimentContractError("B0/B1 effective configs must be JSON objects")
    differences = diff_effective_configs(b0_effective_config, b1_effective_config)
    if differences:
        raise ExperimentContractError(
            "B0/B1 effective config differs outside the whitelist: " + ", ".join(differences)
        )
    if contract is not None:
        for label, config, profile, model_contract, observation_schema in (
            ("B0", b0_effective_config, contract.b0_model_profile, contract.b0_model_contract, contract.b0_observation_schema),
            ("B1", b1_effective_config, contract.b1_model_profile, contract.b1_model_contract, contract.b1_observation_schema),
        ):
            expected = {
                "profile": profile,
                "komi": contract.komi,
                "regular_sims": contract.self_play_simulations,
                "fast_sims": contract.fast_simulations,
                "arena_sims": contract.arena_simulations,
                "fast_probability": contract.fast_probability,
                "train_batch_size": contract.training_batch_size,
                "workers": contract.worker_count,
                "ratio": contract.train_samples_per_new_sample,
                "rules_fingerprint": contract.rules_fingerprint,
                "observation_schema": observation_schema,
                "model_contract": _thaw(model_contract),
            }
            actual = {
                "profile": _config_value(config, "gocube_model_profile", "model_profile"),
                "komi": _config_value(config, "gocube_komi", "komi"),
                "regular_sims": _config_value(config, "numMCTSSims"),
                "fast_sims": _config_value(config, "numFastSims"),
                "arena_sims": _config_value(config, "arenaMCTSSims"),
                "fast_probability": _config_value(config, "probFastSim"),
                "train_batch_size": _config_value(config, "train_batch_size"),
                "workers": _config_value(config, "workers"),
                "ratio": _config_value(config, "gocube_train_samples_per_new_sample"),
                "rules_fingerprint": _config_value(config, "gocube_rules_fingerprint", "rules_fingerprint"),
                "observation_schema": _config_value(config, "gocube_observation_schema", "observation_schema"),
                "model_contract": _config_value(config, "model_contract"),
            }
            for key, expected_value in expected.items():
                actual_value = actual[key]
                if key in {"komi", "fast_probability", "ratio"}:
                    try:
                        same = abs(float(actual_value) - float(expected_value)) <= 1e-12
                    except (TypeError, ValueError):
                        same = False
                else:
                    same = actual_value == expected_value
                if not same:
                    raise ExperimentContractError(
                        f"{label} effective config {key} drift: "
                        f"expected {expected_value!r}, got {actual_value!r}"
                    )
        common_differences = _effective_config_common_differences(
            contract.canonical_common_effective_config,
            b0_effective_config,
        )
        common_differences.extend(
            _effective_config_common_differences(
                contract.canonical_common_effective_config,
                b1_effective_config,
            )
        )
        if common_differences:
            raise ExperimentContractError(
                "B effective config differs from canonical common settings: "
                + ", ".join(sorted(set(common_differences)))
            )
        _validate_treatment_specific_config(
            b0_effective_config,
            label="B0",
            profile=contract.b0_model_profile,
            model_contract=contract.b0_model_contract,
            observation_schema=contract.b0_observation_schema,
            structural_feature_schema=contract.b0_structural_feature_schema,
            structural_feature_channels=contract.b0_structural_feature_channels,
        )
        _validate_treatment_specific_config(
            b1_effective_config,
            label="B1",
            profile=contract.b1_model_profile,
            model_contract=contract.b1_model_contract,
            observation_schema=contract.b1_observation_schema,
            structural_feature_schema=contract.b1_structural_feature_schema,
            structural_feature_channels=contract.b1_structural_feature_channels,
        )
    return differences


def validate_b_experiment_contract(
    contract: BExperimentContract | Mapping[str, object],
    *,
    b0_effective_config: Mapping[str, object] | None = None,
    b1_effective_config: Mapping[str, object] | None = None,
) -> BExperimentContract | Mapping[str, object]:
    """Validate the immutable B contract and optionally its resolved configs."""

    get = contract.get if isinstance(contract, Mapping) else lambda key, default=None: getattr(contract, key, default)
    required = {
        "contract_id": B_EXPERIMENT_CONTRACT_ID,
        "contract_version": B_EXPERIMENT_CONTRACT_VERSION,
        "topology": "cube",
        "board_size": 4,
        "rules_id": "gocube-katago-japanese-v3",
        "komi": GOCUBE_KOMI,
        "self_play_simulations": CUBE4_PRODUCTION.regular_sims,
        "fast_simulations": CUBE4_PRODUCTION.fast_sims,
        "fast_probability": CUBE4_PRODUCTION.fast_probability,
        "arena_simulations": CUBE4_PRODUCTION.arena_sims,
        "training_batch_size": CUBE4_PRODUCTION.train_batch_size,
        "train_samples_per_new_sample": CUBE4_PRODUCTION.train_samples_per_new_sample,
        "worker_count": CUBE4_PRODUCTION.workers,
        "generation_chunk_size": CUBE4_PRODUCTION.games_per_iteration,
        "b0_model_profile": B0_MODEL_PROFILE,
        "b1_model_profile": B1_MODEL_PROFILE,
        "termination_contract": TERMINATION_CONTRACT,
        "primary_endpoint": "heldout_paired_position_score",
        "statistical_method_identifier": (
            "hierarchical-paired-bootstrap-seeds-to-starting-position-pairs-v1"
        ),
        "no_result_evaluation_convention": (
            "NO_RESULT contributes 0.5 to the paired position score and remains in the denominator"
        ),
        "initial_seed_count": B_INITIAL_SEED_COUNT,
        "extension_seed_count": B_EXTENSION_SEED_COUNT,
        "mandatory_seed_list": list(B_MANDATORY_SEEDS),
        "extension_seed_list": list(B_EXTENSION_SEEDS),
        "extension_seed_activation_criterion": B_EXTENSION_SEED_CRITERION_ID,
    }
    for key, expected in required.items():
        actual = get(key, None)
        if key in {"komi", "fast_probability", "train_samples_per_new_sample"}:
            try:
                same = abs(float(actual) - float(expected)) <= 1e-12
            except (TypeError, ValueError):
                same = False
        elif key in {"mandatory_seed_list", "extension_seed_list"}:
            same = tuple(actual or ()) == tuple(expected)
        else:
            same = actual == expected
        if not same:
                raise ExperimentContractError(
                    f"B experiment contract field {key} drift: expected {expected!r}, got {actual!r}"
                )
    target_payload = get("scientific_sample_target", None)
    if not isinstance(target_payload, Mapping):
        raise ExperimentContractError("B experiment scientific_sample_target must be an object")
    try:
        scientific_target = SampleBudgetTarget(
            str(target_payload.get("kind")),
            int(target_payload.get("target")),
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(
            "B experiment scientific_sample_target is invalid"
        ) from exc
    if target_payload.get("counter") != scientific_target.counter_key:
        raise ExperimentContractError(
            "B experiment scientific_sample_target counter does not match its kind"
        )
    if int(target_payload.get("generation_chunk_games", -1)) != CUBE4_PRODUCTION.games_per_iteration:
        raise ExperimentContractError(
            "B experiment scientific_sample_target generation chunk drift"
        )
    source_sha = get("source_git_sha", "")
    if not isinstance(source_sha, str) or len(source_sha) != 40:
        raise ExperimentContractError("B experiment contract source_git_sha is missing or invalid")
    heldout_hash = get("heldout_suite_hash", "")
    if not isinstance(heldout_hash, str) or len(heldout_hash) != 64:
        raise ExperimentContractError("B experiment contract heldout_suite_hash is missing or invalid")
    canonical_fields = build_b_experiment_contract(
        source_git_sha=str(source_sha),
        heldout_suite_hash=str(heldout_hash),
        scientific_target=scientific_target,
    )
    for key in ("rules_fingerprint", "search_contract_id", "b0_observation_schema", "b1_observation_schema"):
        if not isinstance(get(key, None), str) or not get(key, ""):
            raise ExperimentContractError(f"B experiment contract field {key} is missing")
        if get(key) != getattr(canonical_fields, key):
            raise ExperimentContractError(
                f"B experiment contract field {key} differs from the current rules/model specification"
            )
    for key in ("b0_model_contract", "b1_model_contract", "target_semantics", "replay_window", "optimizer_settings", "scheduler_settings"):
        if not isinstance(get(key, None), Mapping):
            raise ExperimentContractError(f"B experiment contract field {key} must be an object")
    for key in (
        "canonical_common_effective_config",
    ):
        if not isinstance(get(key, None), Mapping):
            raise ExperimentContractError(f"B experiment contract field {key} must be an object")
    contract_object = contract if isinstance(contract, BExperimentContract) else BExperimentContract.from_dict(contract)
    canonical = canonical_fields
    actual_payload = contract_object.to_dict()
    if isinstance(contract, BExperimentContract) or isinstance(contract, Mapping):
        canonical_payload = canonical.to_dict()
        for key in actual_payload:
            if key in {"source_git_sha", "heldout_suite_hash"}:
                continue
            if actual_payload.get(key) != canonical_payload.get(key):
                raise ExperimentContractError(
                    f"B experiment contract field {key} differs from the canonical specification"
                )
    if b0_effective_config is not None or b1_effective_config is not None:
        if b0_effective_config is None or b1_effective_config is None:
            raise ExperimentContractError("Both B0 and B1 effective configs are required")
        validate_b0_b1_effective_configs(
            b0_effective_config,
            b1_effective_config,
            contract=contract_object,
        )
    return contract


def write_b_experiment_record(
    path: str | os.PathLike[str],
    contract: BExperimentContract,
    b0_effective_config: Mapping[str, object],
    b1_effective_config: Mapping[str, object],
) -> dict[str, object]:
    """Persist the single machine-readable pre-run comparison record."""

    validate_b_experiment_contract(
        contract,
        b0_effective_config=b0_effective_config,
        b1_effective_config=b1_effective_config,
    )
    payload = {
        "schema_version": 1,
        "experiment_contract": contract.to_dict(),
        "experiment_contract_sha256": contract.sha256(),
        "effective_configs": {"B0": _json_safe(b0_effective_config), "B1": _json_safe(b1_effective_config)},
        "effective_config_diff": diff_effective_configs(b0_effective_config, b1_effective_config),
        "allowed_effective_config_differences": list(ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return payload


def validate_b_experiment_record(path: str | os.PathLike[str]) -> dict[str, object]:
    """Read and verify the persisted pre-run contract record."""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentContractError(f"Cannot read B experiment record: {exc}") from exc
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ExperimentContractError("Unsupported B experiment record schema")
    contract_payload = payload.get("experiment_contract")
    contract = BExperimentContract.from_dict(contract_payload)
    if payload.get("experiment_contract_sha256") != contract.sha256():
        raise ExperimentContractError("B experiment contract hash mismatch")
    configs = payload.get("effective_configs")
    if not isinstance(configs, Mapping) or not isinstance(configs.get("B0"), Mapping) or not isinstance(configs.get("B1"), Mapping):
        raise ExperimentContractError("B experiment record is missing B0/B1 effective configs")
    validate_b_experiment_contract(
        contract,
        b0_effective_config=configs["B0"],
        b1_effective_config=configs["B1"],
    )
    return dict(payload)


def _resolve_effective_config(
    profile: str,
    scientific_target: SampleBudgetTarget | None = None,
) -> dict[str, object]:
    from .hardened_train import build_hardened_training_args
    from .katago_train import parse_args
    from .reproducible_manifest import effective_config

    if scientific_target is None:
        scientific_target = SampleBudgetTarget(
            SampleBudgetTarget.NEW_SAMPLES,
            DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET,
        )
    target_flag = (
        "--cumulative-new-samples-target"
        if scientific_target.kind == SampleBudgetTarget.NEW_SAMPLES
        else "--cumulative-optimizer-examples-target"
    )
    cli = parse_args(
        [
            "--model-profile",
            profile,
            "--topology",
            "cube",
            "--size",
            "4",
            "--workers",
            str(CUBE4_PRODUCTION.workers),
            "--sims",
            str(CUBE4_PRODUCTION.regular_sims),
            "--arena-sims",
            str(CUBE4_PRODUCTION.arena_sims),
            "--games-per-iteration",
            str(CUBE4_PRODUCTION.games_per_iteration),
            "--train-batch-size",
            str(CUBE4_PRODUCTION.train_batch_size),
            "--fast-game-prob",
            "0.25",
            "--no-arena",
            "--run-name",
            f"gocube-b-preflight-{profile}",
            target_flag,
            str(scientific_target.target),
        ]
    )
    game_cls, args = build_hardened_training_args(cli)
    return effective_config(args, game_cls)


def resolve_b_effective_configs_separate_processes(
    *,
    repo: str | os.PathLike[str] | None = None,
    scientific_target: SampleBudgetTarget | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve B0 and B1 in independent interpreters before diffing them."""

    root = Path(repo).resolve() if repo is not None else _repo_root()
    if scientific_target is None:
        scientific_target = SampleBudgetTarget(
            SampleBudgetTarget.NEW_SAMPLES,
            DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET,
        )
    configs = []
    for profile in (B0_MODEL_PROFILE, B1_MODEL_PROFILE):
        command = [
            sys.executable,
            "-m",
            "alphazero.envs.gocube.b_experiment_contract",
            "--resolve-effective-config",
            "--profile",
            profile,
            "--target-kind",
            scientific_target.kind,
            "--target",
            str(scientific_target.target),
        ]
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(root) if not existing_pythonpath else str(root) + os.pathsep + existing_pythonpath
        )
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise ExperimentContractError(
                f"Separate-process {profile} effective-config resolution failed: {detail}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExperimentContractError(
                f"Separate-process {profile} resolver did not emit JSON: {result.stdout!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise ExperimentContractError(f"Separate-process {profile} resolver returned a non-object")
        configs.append(payload)
    return configs[0], configs[1]


def resolve_b_effective_configs(
    *,
    repo: str | os.PathLike[str] | None = None,
    scientific_target: SampleBudgetTarget | None = None,
):
    """Short public alias; resolution remains separate-process by contract."""

    return resolve_b_effective_configs_separate_processes(
        repo=repo,
        scientific_target=scientific_target,
    )


def validate_effective_config_for_treatment(
    config: Mapping[str, object],
    treatment: str,
    *,
    contract: BExperimentContract | None = None,
) -> None:
    """Validate one resolved treatment config against the B contract."""

    selected = str(treatment).upper()
    if selected not in {B0_TREATMENT, B1_TREATMENT}:
        raise ExperimentContractError(f"Unknown B experiment treatment: {treatment!r}")
    if contract is None:
        raise ExperimentContractError(
            "validate_effective_config_for_treatment requires the immutable B contract"
        )
    if selected == B0_TREATMENT:
        expected_profile = contract.b0_model_profile
        expected_model = contract.b0_model_contract
        expected_schema = contract.b0_observation_schema
        expected_structural_schema = contract.b0_structural_feature_schema
        expected_structural_channels = contract.b0_structural_feature_channels
    else:
        expected_profile = contract.b1_model_profile
        expected_model = contract.b1_model_contract
        expected_schema = contract.b1_observation_schema
        expected_structural_schema = contract.b1_structural_feature_schema
        expected_structural_channels = contract.b1_structural_feature_channels
    profile = _config_value(config, "gocube_model_profile", "model_profile")
    if profile != expected_profile:
        raise ExperimentContractError(
            f"{selected} effective config selected profile {profile!r}; expected {expected_profile!r}"
        )
    validate_b0_b1_effective_configs(
        config,
        config,
        contract=dataclasses.replace(
            contract,
            b0_model_profile=expected_profile,
            b1_model_profile=expected_profile,
            b0_model_contract=expected_model,
            b1_model_contract=expected_model,
            b0_structural_feature_schema=expected_structural_schema,
            b1_structural_feature_schema=expected_structural_schema,
            b0_structural_feature_channels=expected_structural_channels,
            b1_structural_feature_channels=expected_structural_channels,
            b0_observation_schema=expected_schema,
            b1_observation_schema=expected_schema,
        ),
    )


def validate_b0_effective_config(
    config: Mapping[str, object], *, contract: BExperimentContract | None = None
) -> None:
    validate_effective_config_for_treatment(config, B0_TREATMENT, contract=contract)


def validate_b1_effective_config(
    config: Mapping[str, object], *, contract: BExperimentContract | None = None
) -> None:
    validate_effective_config_for_treatment(config, B1_TREATMENT, contract=contract)


canonical_b_experiment_contract = build_b_experiment_contract


def preflight_b_experiment(
    *,
    repo: str | os.PathLike[str] | None = None,
    contract_path: str | os.PathLike[str] | None = None,
    heldout_suite_path: str | os.PathLike[str] | None = None,
    scientific_target: SampleBudgetTarget | None = None,
) -> dict[str, object]:
    """Resolve, validate, and optionally persist the B experiment record."""

    root = Path(repo).resolve() if repo is not None else _repo_root()
    if heldout_suite_path is None:
        raise ExperimentContractError(
            "--heldout-suite is required for a real B experiment; only --dry-run may omit it"
        )
    require_clean_source(root)
    heldout_hash = hash_heldout_suite(heldout_suite_path)
    contract = build_b_experiment_contract(
        source_git_sha=current_source_git_sha(root),
        heldout_suite_hash=heldout_hash,
        repo=root,
        scientific_target=scientific_target,
    )
    b0, b1 = resolve_b_effective_configs_separate_processes(
        repo=root,
        scientific_target=scientific_target,
    )
    validate_b_experiment_contract(contract, b0_effective_config=b0, b1_effective_config=b1)
    if contract_path is None:
        return {
            "schema_version": 1,
            "experiment_contract": contract.to_dict(),
            "experiment_contract_sha256": contract.sha256(),
            "effective_configs": {"B0": b0, "B1": b1},
            "effective_config_diff": [],
            "allowed_effective_config_differences": list(ALLOWED_EFFECTIVE_CONFIG_DIFFERENCES),
        }
    return write_b_experiment_record(contract_path, contract, b0, b1)


validate_b_experiment_preflight = preflight_b_experiment


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolve-effective-config", action="store_true")
    parser.add_argument("--profile", choices=(B0_MODEL_PROFILE, B1_MODEL_PROFILE))
    parser.add_argument(
        "--target-kind",
        choices=(SampleBudgetTarget.NEW_SAMPLES, SampleBudgetTarget.OPTIMIZER_EXAMPLES),
        default=SampleBudgetTarget.NEW_SAMPLES,
    )
    parser.add_argument("--target", type=int, default=DEFAULT_B_CUMULATIVE_NEW_SAMPLES_TARGET)
    args = parser.parse_args(argv)
    if args.resolve_effective_config:
        if args.profile is None:
            parser.error("--profile is required with --resolve-effective-config")
        target = SampleBudgetTarget(args.target_kind, args.target)
        print(
            json.dumps(
                _resolve_effective_config(args.profile, target),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    parser.error("this module is a library; use --resolve-effective-config or the B launcher")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
