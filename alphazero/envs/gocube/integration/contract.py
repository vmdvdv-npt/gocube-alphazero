"""Authoritative GoCube model/game compatibility contract.

The training game class and effective network arguments are the source of
truth.  Integration code may resolve a class from this contract, but it must
not reconstruct one from topology and size alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..production_contract import require_gocube_komi

MODEL_CONTRACT_VERSION = 1
MODEL_CONTRACT_ID = "gocube-model-contract-v1"
ACTION_SCHEMA = "gocube-action-point-id-pass-v1"
_MISSING = object()


class ContractError(ValueError):
    """Raised when a GoCube model contract is missing or inconsistent."""


def _get(metadata: Any, key: str, default: Any = None) -> Any:
    if metadata is None:
        return default
    if isinstance(metadata, Mapping):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _provided(metadata: Any, key: str) -> Any:
    """Return a value while preserving the distinction between absent/None."""

    if metadata is None:
        return _MISSING
    if isinstance(metadata, Mapping):
        return metadata[key] if key in metadata else _MISSING
    return getattr(metadata, key, _MISSING)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def point_order_fingerprint(topology) -> str:
    """Fingerprint the canonical logical PointId order used by the NN."""

    return _fingerprint({"pointIds": list(topology.point_ids)})


def adjacency_fingerprint(topology) -> str:
    """Fingerprint adjacency in canonical point-index order."""

    return _fingerprint({
        "neighborsByIndex": [list(neighbors) for neighbors in topology.neighbors_by_index],
    })


def topology_fingerprint(topology) -> str:
    return _fingerprint({
        "kind": topology.kind,
        "size": int(topology.size),
        "pointCount": int(topology.point_count),
        "pointOrderFingerprint": point_order_fingerprint(topology),
        "adjacencyFingerprint": adjacency_fingerprint(topology),
    })


def _class_id(game_cls) -> str:
    explicit = getattr(game_cls, "GOCUBE_GAME_CLASS_ID", None)
    if explicit:
        return str(explicit)
    return f"{game_cls.__module__}.{game_cls.__qualname__}"


def _rules_implementation(game_cls) -> str:
    explicit = getattr(game_cls, "GOCUBE_RULES_IMPLEMENTATION", None)
    if explicit:
        return str(explicit)
    adjudicator = getattr(game_cls, "TERMINAL_ADJUDICATOR_ID", "")
    if adjudicator == "gocube-katago-japanese-v3":
        rules_version = getattr(game_cls, "KATAGO_RULES_VERSION", 3)
        implementation_version = getattr(game_cls, "KATAGO_RULES_IMPLEMENTATION_VERSION", None)
        suffix = "" if implementation_version is None else f"-implementation-v{implementation_version}"
        return f"gocube-katago-rules-v{rules_version}{suffix}"
    if adjudicator == "gocube-japanese-cleanup-v2":
        return "gocube-japanese-cleanup-v2"
    if adjudicator == "gocube-conservative-area-v1":
        return "gocube-conservative-area-v1"
    return str(adjudicator)


def _rules_fingerprint(game_cls, topology) -> str:
    explicit = getattr(game_cls, "rules_fingerprint", None)
    if explicit is not None:
        return str(explicit())
    # Historical Chinese V1 classes never exposed a rules fingerprint.  Keep
    # their compatibility identity deterministic without pretending it is the
    # current KataGo V3 fingerprint.
    return _fingerprint({
        "rulesImplementation": _rules_implementation(game_cls),
        "topology": topology.kind,
        "size": int(topology.size),
        "pointCount": int(topology.point_count),
        "komi": float(game_cls.KOMI),
    })


def _architecture_config(args: Any, game_cls=None) -> dict[str, object]:
    nnet_type = _get(args, "nnet_type", None)
    keys = (
        "nnet_type", "num_channels", "depth", "value_head_channels",
        "policy_head_channels", "input_fc_layers", "value_dense_layers",
        "policy_dense_layers", "score_dense_layers", "gocube_auxiliary_targets",
        "gocube_model_profile",
        "gocube_network_architecture", "gocube_structural_feature_schema",
        "gocube_structural_feature_channels",
    )
    result = (
        {key: _get(args, key) for key in keys if _get(args, key, None) is not None}
        if nnet_type is not None
        else {}
    )
    if game_cls is not None and hasattr(game_cls, "GOCUBE_MODEL_PROFILE"):
        result.setdefault("gocube_model_profile", str(game_cls.GOCUBE_MODEL_PROFILE))
    if game_cls is not None and hasattr(game_cls, "GOCUBE_NETWORK_ARCHITECTURE_ID"):
        result.setdefault(
            "gocube_network_architecture",
            str(game_cls.GOCUBE_NETWORK_ARCHITECTURE_ID),
        )
    if game_cls is not None and hasattr(game_cls, "STRUCTURAL_FEATURE_SCHEMA"):
        result.setdefault(
            "gocube_structural_feature_schema",
            getattr(game_cls, "STRUCTURAL_FEATURE_SCHEMA"),
        )
    if game_cls is not None and hasattr(game_cls, "STRUCTURAL_FEATURE_CHANNELS"):
        result.setdefault(
            "gocube_structural_feature_channels",
            int(getattr(game_cls, "STRUCTURAL_FEATURE_CHANNELS", 2)),
        )
    return result


def _validate_game_class_args_compatibility(game_cls, args: Any) -> None:
    """Reject contradictory class-vs-args architecture metadata early."""

    checks = (
        ("GOCUBE_MODEL_PROFILE", "gocube_model_profile", str),
        ("GOCUBE_NETWORK_ARCHITECTURE_ID", "gocube_network_architecture", str),
        ("STRUCTURAL_FEATURE_SCHEMA", "gocube_structural_feature_schema", None),
        ("STRUCTURAL_FEATURE_CHANNELS", "gocube_structural_feature_channels", int),
    )
    for class_key, args_key, converter in checks:
        if not hasattr(game_cls, class_key):
            continue
        expected = getattr(game_cls, class_key)
        actual = _provided(args, args_key)
        if actual is _MISSING:
            continue
        try:
            compared = converter(actual) if converter is not None else actual
        except (TypeError, ValueError):
            compared = actual
        if compared != expected:
            raise ContractError(
                f"GoCube game class/args conflict for {args_key}: "
                f"game_cls={expected!r}, args={actual!r}"
            )


def _architecture_id(args: Any, game_cls=None) -> str:
    configured = _get(args, "gocube_network_architecture", None)
    if configured:
        return str(configured)
    configured = getattr(game_cls, "GOCUBE_NETWORK_ARCHITECTURE_ID", None)
    if configured:
        return str(configured)
    nnet_type = _get(args, "nnet_type", None)
    return f"gocube-{nnet_type}-v1" if nnet_type else "gocube-network-legacy"


def _search_contract_id(args: Any) -> str:
    return str(
        _get(args, "gocube_katago_search_contract", None)
        or _get(args, "gocube_search_contract", None)
        or "gocube-search-contract-legacy"
    )


def _target_schema(args: Any) -> dict[str, object]:
    return {
        "value": _get(args, "gocube_value_target_semantics", "win-loss-noresult-v1"),
        "score": _get(args, "gocube_score_target_semantics", None),
        "ownership": _get(args, "gocube_ownership_target_semantics", None),
    }


@dataclass(frozen=True)
class ResolvedGoCubeContract:
    """Complete, serializable identity of a GoCube inference environment."""

    contract_id: str
    contract_version: int
    game_class_id: str
    rules_implementation: str
    observation_schema: str
    observation_shape: tuple[int, int, int]
    action_schema: str
    action_size: int
    topology_kind: str
    topology_size: int
    point_count: int
    point_order_fingerprint: str
    adjacency_fingerprint: str
    topology_fingerprint: str
    network_architecture_id: str
    network_architecture_fingerprint: str
    search_contract_id: str
    terminal_adjudicator_id: str
    rules_fingerprint: str
    komi: float
    targets_schema: dict[str, object]
    output_heads: int

    @property
    def topology(self) -> str:
        return self.topology_kind

    @property
    def size(self) -> int:
        return self.topology_size

    @property
    def observation_schema_id(self) -> str:
        return self.observation_schema

    @property
    def network_architecture(self) -> str:
        return self.network_architecture_id

    @property
    def search_contract(self) -> str:
        return self.search_contract_id

    def to_dict(self) -> dict[str, object]:
        return {
            "contractId": self.contract_id,
            "contractVersion": self.contract_version,
            "gameClassId": self.game_class_id,
            "rulesImplementation": self.rules_implementation,
            "observationSchema": self.observation_schema,
            "observationShape": list(self.observation_shape),
            "actionSchema": self.action_schema,
            "actionSize": self.action_size,
            "topologyKind": self.topology_kind,
            "topologySize": self.topology_size,
            "pointCount": self.point_count,
            "pointOrderFingerprint": self.point_order_fingerprint,
            "adjacencyFingerprint": self.adjacency_fingerprint,
            "topologyFingerprint": self.topology_fingerprint,
            "networkArchitectureId": self.network_architecture_id,
            "networkArchitectureFingerprint": self.network_architecture_fingerprint,
            "searchContractId": self.search_contract_id,
            "terminalAdjudicatorId": self.terminal_adjudicator_id,
            "rulesFingerprint": self.rules_fingerprint,
            "komi": float(self.komi),
            "targetsSchema": dict(self.targets_schema),
            "outputHeads": self.output_heads,
        }

    def to_checkpoint_fields(self) -> dict[str, object]:
        """Flat fields retained in checkpoint args for old tooling."""

        return {
            "gocube_model_contract_id": self.contract_id,
            "gocube_model_contract_version": self.contract_version,
            "gocube_game_class_id": self.game_class_id,
            "gocube_rules_implementation": self.rules_implementation,
            "gocube_observation_schema": self.observation_schema,
            "gocube_observation_shape": tuple(self.observation_shape),
            "gocube_action_schema": self.action_schema,
            "gocube_action_size": self.action_size,
            "gocube_topology": self.topology_kind,
            "gocube_size": self.topology_size,
            "gocube_point_count": self.point_count,
            "gocube_point_order_fingerprint": self.point_order_fingerprint,
            "gocube_adjacency_fingerprint": self.adjacency_fingerprint,
            "gocube_topology_fingerprint": self.topology_fingerprint,
            "gocube_network_architecture": self.network_architecture_id,
            "gocube_network_architecture_fingerprint": self.network_architecture_fingerprint,
            "gocube_search_contract": self.search_contract_id,
            "gocube_terminal_adjudicator": self.terminal_adjudicator_id,
            "gocube_rules_fingerprint": self.rules_fingerprint,
            "gocube_komi": float(self.komi),
            "gocube_targets_schema": dict(self.targets_schema),
            "gocube_output_heads": self.output_heads,
            # Keep the rich object alongside flat args so a future loader does
            # not have to infer which fields belonged to one contract version.
            "gocube_model_contract": self.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResolvedGoCubeContract":
        def required(name: str, *aliases: str):
            for key in (name, *aliases):
                if key in data:
                    return data[key]
            raise ContractError(f"Model contract is missing field: {name}")

        shape = required("observationShape", "observation_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ContractError("Model contract observationShape must contain three dimensions")
        targets = required("targetsSchema", "targets_schema")
        if not isinstance(targets, Mapping):
            raise ContractError("Model contract targetsSchema must be an object")
        try:
            result = cls(
                contract_id=str(required("contractId", "contract_id")),
                contract_version=int(required("contractVersion", "contract_version")),
                game_class_id=str(required("gameClassId", "game_class_id")),
                rules_implementation=str(required("rulesImplementation", "rules_implementation")),
                observation_schema=str(required("observationSchema", "observation_schema")),
                observation_shape=tuple(int(x) for x in shape),
                action_schema=str(required("actionSchema", "action_schema")),
                action_size=int(required("actionSize", "action_size")),
                topology_kind=str(required("topologyKind", "topology_kind", "topology")),
                topology_size=int(required("topologySize", "topology_size", "size")),
                point_count=int(required("pointCount", "point_count")),
                point_order_fingerprint=str(required("pointOrderFingerprint", "point_order_fingerprint")),
                adjacency_fingerprint=str(required("adjacencyFingerprint", "adjacency_fingerprint")),
                topology_fingerprint=str(required("topologyFingerprint", "topology_fingerprint")),
                network_architecture_id=str(required("networkArchitectureId", "network_architecture_id", "networkArchitecture")),
                network_architecture_fingerprint=str(required("networkArchitectureFingerprint", "network_architecture_fingerprint")),
                search_contract_id=str(required("searchContractId", "search_contract_id", "searchContract")),
                terminal_adjudicator_id=str(required("terminalAdjudicatorId", "terminal_adjudicator_id")),
                rules_fingerprint=str(required("rulesFingerprint", "rules_fingerprint")),
                komi=float(required("komi")),
                targets_schema=dict(targets),
                output_heads=int(required("outputHeads", "output_heads")),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Invalid model contract: {exc}") from exc
        if result.contract_id != MODEL_CONTRACT_ID:
            raise ContractError(f"Unsupported model contract id: {result.contract_id!r}")
        if result.contract_version != MODEL_CONTRACT_VERSION:
            raise ContractError(f"Unsupported model contract version: {result.contract_version}")
        if any(d <= 0 for d in result.observation_shape) or result.point_count <= 0:
            raise ContractError("Model contract contains invalid geometry")
        return result

    def differences(self, other: "ResolvedGoCubeContract") -> dict[str, tuple[object, object]]:
        fields = (
            "contract_id", "contract_version", "game_class_id", "rules_implementation",
            "observation_schema", "observation_shape", "action_schema", "action_size",
            "topology_kind", "topology_size", "point_count", "point_order_fingerprint",
            "adjacency_fingerprint", "topology_fingerprint", "network_architecture_id",
            "network_architecture_fingerprint", "search_contract_id",
            "terminal_adjudicator_id", "rules_fingerprint", "komi", "targets_schema",
            "output_heads",
        )
        return {
            field: (getattr(self, field), getattr(other, field))
            for field in fields
            if getattr(self, field) != getattr(other, field)
        }


# Model-specific observation and architecture identity are intentionally not
# part of this compatibility view.  The Arena/evaluation caller still has to
# load each model against its complete own contract; this helper only answers
# whether both models can share one semantic GoCube game timeline.
EVALUATION_SHARED_CONTRACT_FIELDS = (
    "contract_id",
    "contract_version",
    "rules_implementation",
    "action_schema",
    "action_size",
    "topology_kind",
    "topology_size",
    "point_count",
    "point_order_fingerprint",
    "adjacency_fingerprint",
    "topology_fingerprint",
    "search_contract_id",
    "terminal_adjudicator_id",
    "rules_fingerprint",
    "komi",
    "targets_schema",
    "output_heads",
)

# Search settings are intentionally kept outside ResolvedGoCubeContract's
# compact serialized shape for backward compatibility, but loaded evaluation
# models must still agree on their effective values.  Missing on both sides
# means the same runtime default is used; asymmetric presence is rejected.
EVALUATION_SHARED_ARG_KEYS = (
    "gocube_katago_search_contract",
    "gocube_search_contract",
    "gocube_katago_search_reference_commit",
    "search_utility_mode",
    "gocube_win_loss_utility_factor",
    "gocube_static_score_utility_factor",
    "gocube_dynamic_score_utility_factor",
    "gocube_dynamic_score_center_zero_weight",
    "gocube_dynamic_score_center_scale",
    "gocube_cpuct_exploration",
    "gocube_cpuct_exploration_log",
    "gocube_cpuct_exploration_base",
    "gocube_root_fpu_reduction",
    "gocube_fpu_parent_weight_by_visited_policy",
    "gocube_fpu_parent_weight_by_visited_policy_pow",
    "gocube_root_ending_bonus_points",
    "gocube_fill_dame_before_pass",
    "gocube_conservative_pass",
    "gocube_root_dirichlet_noise_total_concentration",
    "gocube_root_policy_temperature_early",
    "gocube_root_policy_temperature",
    "gocube_root_policy_temperature_halflife",
    "gocube_root_desired_per_child_visits_coeff",
    "gocube_value_weight_exponent",
    "gocube_use_lcb_for_selection",
    "gocube_lcb_stdevs",
    "gocube_min_visit_prop_for_lcb",
    "gocube_chosen_move_subtract",
    "gocube_chosen_move_prune",
    "cpuct",
    "fpu_reduction",
    "min_discount",
    "root_noise_frac",
    "root_policy_temp",
    "gocube_target_provenance_semantics",
    "gocube_target_provenance_encoding",
    "gocube_termination_contract",
    "gocube_katago_exploration_contract",
)


def evaluation_argument_differences(first: Any, second: Any) -> dict[str, tuple[object, object]]:
    """Compare effective evaluation/search args without profile fields."""

    missing = object()

    def value(metadata, key):
        if metadata is None:
            return missing
        if isinstance(metadata, Mapping):
            return metadata[key] if key in metadata else missing
        return getattr(metadata, key, missing)

    differences = {}
    for key in EVALUATION_SHARED_ARG_KEYS:
        first_value = value(first, key)
        second_value = value(second, key)
        if first_value is missing and second_value is missing:
            continue
        if first_value is missing or second_value is missing or first_value != second_value:
            differences[key] = (
                None if first_value is missing else first_value,
                None if second_value is missing else second_value,
            )
    return differences


def evaluation_contract_differences(
    first: ResolvedGoCubeContract,
    second: ResolvedGoCubeContract,
) -> dict[str, tuple[object, object]]:
    """Return only differences that prevent one semantic evaluation game."""

    return {
        field: (getattr(first, field), getattr(second, field))
        for field in EVALUATION_SHARED_CONTRACT_FIELDS
        if getattr(first, field) != getattr(second, field)
    }


def resolve_model_contract(game_cls, args: Any = None) -> ResolvedGoCubeContract:
    """Resolve the exact contract from the concrete training class and args."""

    try:
        require_gocube_komi(
            getattr(game_cls, "KOMI"),
            context="GoCube model contract",
        )
        topology = game_cls.logical_topology()
        observation_shape = tuple(int(x) for x in game_cls.observation_size())
        action_size = int(game_cls.action_size())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError(f"Cannot resolve GoCube contract from game class: {exc}") from exc
    if len(observation_shape) != 3:
        raise ContractError("GoCube observations must have three dimensions")
    _validate_game_class_args_compatibility(game_cls, args)
    architecture = _architecture_config(args, game_cls)
    target_schema = _target_schema(args)
    output_heads = 4 if bool(_get(args, "gocube_auxiliary_targets", False)) else 2
    # A concrete V3 game class has auxiliary heads even when an old caller did
    # not provide training args.  Metadata in args remains authoritative when
    # it is present.
    if _get(args, "gocube_auxiliary_targets", None) is None:
        output_heads = 4 if target_schema.get("score") or target_schema.get("ownership") else 2
    return ResolvedGoCubeContract(
        contract_id=MODEL_CONTRACT_ID,
        contract_version=MODEL_CONTRACT_VERSION,
        game_class_id=_class_id(game_cls),
        rules_implementation=_rules_implementation(game_cls),
        observation_schema=str(getattr(game_cls, "OBSERVATION_SCHEMA")),
        observation_shape=observation_shape,
        action_schema=str(getattr(game_cls, "ACTION_SCHEMA", ACTION_SCHEMA)),
        action_size=action_size,
        topology_kind=str(topology.kind),
        topology_size=int(topology.size),
        point_count=int(topology.point_count),
        point_order_fingerprint=point_order_fingerprint(topology),
        adjacency_fingerprint=adjacency_fingerprint(topology),
        topology_fingerprint=topology_fingerprint(topology),
        network_architecture_id=_architecture_id(args, game_cls),
        network_architecture_fingerprint=_fingerprint(architecture),
        search_contract_id=_search_contract_id(args),
        terminal_adjudicator_id=str(game_cls.TERMINAL_ADJUDICATOR_ID),
        rules_fingerprint=_rules_fingerprint(game_cls, topology),
        komi=float(game_cls.KOMI),
        targets_schema=target_schema,
        output_heads=output_heads,
    )


def _candidate_game_classes() -> tuple[type, ...]:
    # Imports stay local: this module is used by manifest code during startup.
    from alphazero.envs.gocube.diversified_game import (
        diversified_baseline_pinned_game_class,
        diversified_pinned_game_class,
        diversified_structural_pinned_game_class,
    )
    from alphazero.envs.gocube.game import (
        SUPPORTED_CHINESE_GAMES,
        SUPPORTED_JAPANESE_GAMES,
        SUPPORTED_JAPANESE_V2_GAMES,
    )
    from alphazero.envs.gocube.pinned_game import (
        structural_pinned_game_class,
        pinned_game_class,
    )

    result = []
    for mapping in (SUPPORTED_JAPANESE_GAMES, SUPPORTED_JAPANESE_V2_GAMES, SUPPORTED_CHINESE_GAMES):
        for base in mapping.values():
            wrapped = None
            try:
                wrapped = pinned_game_class(base)
            except ValueError:
                pass
            for candidate in (wrapped, base):
                if candidate is not None and candidate not in result:
                    result.append(candidate)
            if mapping is SUPPORTED_JAPANESE_GAMES:
                try:
                    baseline = diversified_baseline_pinned_game_class(base)
                except ValueError:
                    baseline = None
                if baseline is not None and baseline not in result:
                    result.append(baseline)
                try:
                    diversified = diversified_pinned_game_class(base)
                except ValueError:
                    diversified = None
                if diversified is not None and diversified not in result:
                    result.append(diversified)
                try:
                    structural_pinned = structural_pinned_game_class(base)
                    diversified_structural = diversified_structural_pinned_game_class(base)
                except ValueError:
                    structural_pinned = diversified_structural = None
                for candidate in (structural_pinned, diversified_structural):
                    if candidate is not None and candidate not in result:
                        result.append(candidate)
    return tuple(result)


def _metadata_contract_dict(metadata: Any) -> Mapping[str, object] | None:
    nested = _get(metadata, "gocube_model_contract", None)
    if nested is None:
        nested = _get(metadata, "modelContract", None)
    if nested is None:
        nested = _get(metadata, "model_contract", None)
    return nested if isinstance(nested, Mapping) else None


def _class_matches_metadata(game_cls, metadata: Any) -> bool:
    checks = (
        ("gocube_topology", game_cls.topology_kind()),
        ("gocube_size", game_cls.board_size()),
        ("gocube_point_count", game_cls.logical_topology().point_count),
        ("gocube_observation_schema", getattr(game_cls, "OBSERVATION_SCHEMA", None)),
        ("gocube_action_size", game_cls.action_size()),
        ("gocube_terminal_adjudicator", game_cls.TERMINAL_ADJUDICATOR_ID),
        ("gocube_rules_fingerprint", _rules_fingerprint(game_cls, game_cls.logical_topology())),
    )
    for key, expected in checks:
        actual = _get(metadata, key, None)
        if actual is not None and actual != expected:
            return False
    # Current production checkpoints carry explicit profile metadata.  Use it
    # when resolving a class from flat/legacy args as well; otherwise a B1
    # label combined with B0 geometry could silently resolve to a generic
    # pinned class and weaken the model-contract boundary.
    profile = _provided(metadata, "gocube_model_profile")
    if profile is not _MISSING and str(getattr(game_cls, "GOCUBE_MODEL_PROFILE", "")) != str(profile):
        return False
    architecture = _provided(metadata, "gocube_network_architecture")
    if architecture is not _MISSING and hasattr(game_cls, "GOCUBE_NETWORK_ARCHITECTURE_ID"):
        if str(architecture) != str(game_cls.GOCUBE_NETWORK_ARCHITECTURE_ID):
            return False
    structural_schema = _provided(metadata, "gocube_structural_feature_schema")
    if structural_schema is not _MISSING and hasattr(game_cls, "STRUCTURAL_FEATURE_SCHEMA"):
        if structural_schema != getattr(game_cls, "STRUCTURAL_FEATURE_SCHEMA"):
            return False
    structural_channels = _provided(metadata, "gocube_structural_feature_channels")
    if structural_channels is not _MISSING and hasattr(game_cls, "STRUCTURAL_FEATURE_CHANNELS"):
        try:
            structural_channels = int(structural_channels)
        except (TypeError, ValueError):
            return False
        if structural_channels != int(getattr(game_cls, "STRUCTURAL_FEATURE_CHANNELS")):
            return False
    return True


def resolve_game_class_from_contract(contract: ResolvedGoCubeContract):
    candidates = _candidate_game_classes()
    exact = [candidate for candidate in candidates if _class_id(candidate) == contract.game_class_id]
    if exact:
        candidate = exact[0]
        actual = resolve_model_contract(candidate, None)
        # Network/search fields are not class properties; compare only the
        # game/inference portion before returning the class.
        differences = {
            field: values for field, values in contract.differences(actual).items()
            if field not in {
                "network_architecture_id", "network_architecture_fingerprint",
                "search_contract_id", "targets_schema", "output_heads",
            }
        }
        if differences:
            field, (saved, expected) = next(iter(differences.items()))
            raise ContractError(
                f"GoCube contract mismatch for {field}: saved={saved!r}, expected={expected!r}"
            )
        return candidate
    matching = [candidate for candidate in candidates if _class_contract_matches(candidate, contract)]
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise ContractError(
            "No GoCube game class matches model contract "
            f"(topology={contract.topology_kind!r}, size={contract.topology_size}, "
            f"observation={contract.observation_schema!r}, "
            f"point_order_fingerprint={contract.point_order_fingerprint!r}, "
            f"adjacency_fingerprint={contract.adjacency_fingerprint!r})"
        )
    # Pinned and diversified classes intentionally share inference semantics;
    # legacy metadata has no class id, so use the pinned inference class.
    pinned = [candidate for candidate in matching if "pinned_game" in candidate.__module__]
    return pinned[0] if pinned else matching[0]


def _class_contract_matches(game_cls, contract: ResolvedGoCubeContract) -> bool:
    actual = resolve_model_contract(game_cls, None)
    return all(
        getattr(actual, field) == getattr(contract, field)
        for field in (
            "rules_implementation", "observation_schema", "observation_shape", "action_schema",
            "action_size", "topology_kind", "topology_size", "point_count",
            "point_order_fingerprint", "adjacency_fingerprint", "topology_fingerprint",
            "terminal_adjudicator_id", "rules_fingerprint", "komi",
        )
    )


def resolve_model_contract_from_metadata(metadata: Any, *, fallback: ResolvedGoCubeContract | None = None) -> ResolvedGoCubeContract:
    """Resolve a contract from rich/flat checkpoint or manifest metadata."""

    nested = _metadata_contract_dict(metadata)
    if nested is not None:
        contract = ResolvedGoCubeContract.from_dict(nested)
        # Flat args and nested contract are two representations of one source.
        # If both exist, a disagreement is corruption, not a precedence choice.
        for key, expected in contract.to_checkpoint_fields().items():
            if key == "gocube_model_contract":
                continue
            actual = _get(metadata, key, None)
            if actual is None:
                continue
            if isinstance(expected, tuple) and isinstance(actual, list):
                actual = tuple(actual)
            if actual != expected:
                raise ContractError(
                    f"GoCube contract metadata conflict for {key}: "
                    f"saved={actual!r}, expected={expected!r}"
                )
        return contract

    identity_keys = (
        "gocube_game_class_id", "gocube_topology", "gocube_size",
        "gocube_point_count", "gocube_observation_schema", "gocube_action_size",
        "gocube_terminal_adjudicator", "gocube_rules_fingerprint",
    )
    if fallback is not None and not any(_get(metadata, key, None) is not None for key in identity_keys):
        return fallback
    game_class_id = _get(metadata, "gocube_game_class_id", None)
    candidates = _candidate_game_classes()
    if game_class_id is not None:
        candidates = tuple(candidate for candidate in candidates if _class_id(candidate) == game_class_id)
    candidates = tuple(candidate for candidate in candidates if _class_matches_metadata(candidate, metadata))
    if not candidates:
        if fallback is not None:
            return fallback
        raise ContractError("Checkpoint metadata does not identify a supported GoCube inference contract")
    # Old V4 checkpoints predate the explicit class id.  Pinned is the
    # inference-equivalent class; new checkpoints always carry the exact id.
    pinned = [candidate for candidate in candidates if "pinned_game" in candidate.__module__]
    game_cls = pinned[0] if pinned else candidates[0]
    args_for_contract = metadata
    contract = resolve_model_contract(game_cls, args_for_contract)
    # A legacy checkpoint may not have the newer fingerprints.  Validate every
    # supplied field, but do not manufacture evidence for fields it never saved.
    for key, expected in contract.to_checkpoint_fields().items():
        if key in {"gocube_model_contract"} or _get(metadata, key, None) is None:
            continue
        actual = _get(metadata, key)
        if isinstance(expected, tuple) and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected:
            raise ContractError(
                f"GoCube contract metadata conflict for {key}: saved={actual!r}, expected={expected!r}"
            )
    return contract


def contract_from_dict(data: Mapping[str, object]) -> ResolvedGoCubeContract:
    return ResolvedGoCubeContract.from_dict(data)


def resolve_contract_for_descriptor(descriptor) -> ResolvedGoCubeContract:
    """Resolve a catalog descriptor, retaining explicit legacy semantics."""

    try:
        require_gocube_komi(
            _get(descriptor, "komi"),
            context=f"Checkpoint {_get(descriptor, 'checkpoint_id', 'descriptor')}",
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc
    nested = _get(descriptor, "model_contract", None)
    if nested is not None:
        if isinstance(nested, ResolvedGoCubeContract):
            return nested
        if isinstance(nested, Mapping):
            return ResolvedGoCubeContract.from_dict(nested)
        raise ContractError("Checkpoint descriptor model_contract must be an object")
    from alphazero.envs.gocube.game import legacy_game_class

    try:
        game_cls = legacy_game_class(
            _get(descriptor, "topology"),
            int(_get(descriptor, "size")),
            _get(descriptor, "terminal_adjudicator"),
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Cannot resolve descriptor contract: {exc}") from exc
    return resolve_model_contract(game_cls)


# Friendly aliases for callers/tests that use the shorter terminology.
ModelContract = ResolvedGoCubeContract
GoCubeModelContract = ResolvedGoCubeContract
contract_from_game_class = resolve_model_contract
resolve_game_contract = resolve_model_contract
resolve_game_class = resolve_game_class_from_contract
canonical_point_order_fingerprint = point_order_fingerprint
