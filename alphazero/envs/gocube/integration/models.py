from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Mapping

import torch

from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import legacy_game_class
from alphazero.envs.gocube.contract_versions import (
    TARGET_PROVENANCE_ENCODING,
    TARGET_PROVENANCE_SEMANTICS,
    TERMINATION_CONTRACT,
)
from alphazero.envs.gocube.production_contract import require_gocube_komi
from alphazero.search_contract import KATAGO_SEARCH_CONTRACT

from .catalog import CheckpointCatalog, CheckpointDescriptor
from .contract import (
    ContractError,
    ResolvedGoCubeContract,
    resolve_game_class_from_contract,
    resolve_model_contract_from_metadata,
    resolve_model_contract,
)
from .errors import CheckpointLoadFailed, CheckpointMetadataInvalid, CheckpointNotFound

DEVICE_CHOICES = ("auto", "cpu", "cuda")


def resolve_device(device: str) -> str:
    if device not in DEVICE_CHOICES:
        raise ValueError(f"Unsupported device {device!r}; expected one of {DEVICE_CHOICES}")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _metadata_value(metadata: Any, key: str, default=None):
    if metadata is None:
        return default
    if isinstance(metadata, Mapping):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _contract_error_field(field: str) -> str:
    return {
        "observation_schema": "gocube_observation_schema",
        "observation_shape": "gocube_observation_shape",
        "action_schema": "gocube_action_schema",
        "action_size": "gocube_action_size",
        "topology_kind": "gocube_topology",
        "topology_size": "gocube_size",
        "point_count": "gocube_point_count",
        "network_architecture_id": "gocube_network_architecture",
        "gocube_model_profile": "gocube_model_profile",
        "gocube_structural_feature_schema": "gocube_structural_feature_schema",
        "gocube_structural_feature_channels": "gocube_structural_feature_channels",
        "search_contract_id": "gocube_search_contract",
        "terminal_adjudicator_id": "gocube_terminal_adjudicator",
        "rules_fingerprint": "gocube_rules_fingerprint",
    }.get(field, field)


def _validate_saved_gocube_metadata(
    model,
    descriptor: CheckpointDescriptor,
    expected_contract: ResolvedGoCubeContract | None = None,
) -> None:
    args = getattr(model, "args", None)
    if args is None:
        return
    expected = {
        "gocube_topology": descriptor.topology,
        "gocube_size": descriptor.size,
        "gocube_rule_set": descriptor.rule_set,
        "gocube_komi": descriptor.komi,
        "gocube_terminal_adjudicator": descriptor.terminal_adjudicator,
    }
    for field, value in expected.items():
        actual = _metadata_value(args, field, None)
        if actual is not None and actual != value:
            raise CheckpointMetadataInvalid(
                f"Checkpoint {descriptor.checkpoint_id} saved metadata {field}={actual!r} "
                f"does not match run manifest value {value!r}"
            )
    if expected_contract is None:
        return
    for key, expected_value in expected_contract.to_checkpoint_fields().items():
        if key == "gocube_model_contract":
            continue
        actual = _metadata_value(args, key, None)
        if actual is None:
            # Checkpoints written before S2 have the original compact fields,
            # but not the newly added topology/network fingerprints.
            continue
        if isinstance(expected_value, tuple) and isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected_value:
            raise CheckpointMetadataInvalid(
                f"Checkpoint GoCube contract mismatch for {key}: "
                f"saved={actual!r}, expected={expected_value!r}"
            )
    if expected_contract.search_contract_id == KATAGO_SEARCH_CONTRACT:
        # S3 changes the meaning of a pinned search result.  These fields are
        # not network architecture fields, so they live beside the model
        # contract, but a current pinned loader must still require them.
        for key, expected_value in (
            ("gocube_target_provenance_semantics", TARGET_PROVENANCE_SEMANTICS),
            ("gocube_target_provenance_encoding", TARGET_PROVENANCE_ENCODING),
            ("gocube_termination_contract", TERMINATION_CONTRACT),
        ):
            actual = _metadata_value(args, key, None)
            if actual != expected_value:
                raise CheckpointMetadataInvalid(
                    f"Checkpoint missing or mismatched S3 contract field {key}: "
                    f"saved={actual!r}, expected={expected_value!r}"
                )


def _read_checkpoint_args(path: str):
    try:
        payload = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        # Keep the historical test-double/API path usable.  A real loader
        # still fails closed when NNetWrapper opens the missing weights file.
        return {}, None
    except Exception as exc:
        raise CheckpointLoadFailed(f"Cannot read checkpoint metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointMetadataInvalid("Checkpoint payload must be an object")
    return payload, payload.get("args")


def _descriptor_contract(descriptor: CheckpointDescriptor) -> ResolvedGoCubeContract | None:
    if descriptor.model_contract is None:
        return None
    try:
        return ResolvedGoCubeContract.from_dict(descriptor.model_contract)
    except ContractError as exc:
        raise CheckpointMetadataInvalid(f"Invalid descriptor model contract: {exc}") from exc


def _legacy_descriptor_contract(descriptor: CheckpointDescriptor) -> ResolvedGoCubeContract:
    try:
        cls = legacy_game_class(
            descriptor.topology,
            descriptor.size,
            descriptor.terminal_adjudicator,
        )
        return resolve_model_contract(cls)
    except (ValueError, ContractError) as exc:
        raise CheckpointMetadataInvalid(
            f"Cannot resolve legacy descriptor contract for {descriptor.checkpoint_id}: {exc}"
        ) from exc


def _validate_descriptor_against_contract(
    descriptor: CheckpointDescriptor,
    contract: ResolvedGoCubeContract,
) -> None:
    expected = {
        "topology": descriptor.topology,
        "size": descriptor.size,
        "rule_set": descriptor.rule_set,
        "komi": descriptor.komi,
        "terminal_adjudicator": descriptor.terminal_adjudicator,
    }
    actual = {
        "topology": contract.topology_kind,
        "size": contract.topology_size,
        "rule_set": {
            "gocube-katago-japanese-v3": "japanese",
            "gocube-japanese-cleanup-v2": "japanese",
            "gocube-conservative-area-v1": "chinese",
        }.get(contract.terminal_adjudicator_id),
        "komi": contract.komi,
        "terminal_adjudicator": contract.terminal_adjudicator_id,
    }
    for field, expected_value in expected.items():
        value = actual[field]
        if field == "komi":
            matches = float(value) == float(expected_value)
        else:
            matches = value == expected_value
        if not matches:
            raise CheckpointMetadataInvalid(
                f"Checkpoint GoCube contract mismatch for {field}: "
                f"saved={value!r}, expected={expected_value!r}"
            )


class ModelCache:
    def __init__(self, max_entries: int = 2):
        if max_entries < 1:
            raise ValueError("Model cache must contain at least one entry")
        self.max_entries = max_entries
        self._items: OrderedDict[tuple[str, str], object] = OrderedDict()
        self._lock = Lock()

    def get(self, key: tuple[str, str]):
        with self._lock:
            model = self._items.get(key)
            if model is not None:
                self._items.move_to_end(key)
            return model

    def put(self, key: tuple[str, str], model):
        evicted_key = None
        evicted = None
        with self._lock:
            self._items[key] = model
            self._items.move_to_end(key)
            if len(self._items) > self.max_entries:
                evicted_key, evicted = self._items.popitem(last=False)
        if evicted is not None:
            del evicted
            if evicted_key is not None and evicted_key[1] == "cuda":
                torch.cuda.empty_cache()
        return model

    def get_or_load(self, key: tuple[str, str], loader):
        existing = self.get(key)
        if existing is not None:
            return existing
        model = loader()
        existing = self.get(key)
        if existing is not None:
            return existing
        return self.put(key, model)


class CheckpointModelLoader:
    def __init__(
        self,
        catalog: CheckpointCatalog,
        *,
        device: str = "auto",
        cache: ModelCache | None = None,
    ):
        self.catalog = catalog
        self.device = resolve_device(device)
        self.cache = cache or ModelCache(max_entries=2)

    def descriptor(self, checkpoint_id: str) -> CheckpointDescriptor:
        descriptor = self.catalog.get(checkpoint_id)
        if descriptor is None:
            raise CheckpointNotFound(f"Unknown checkpoint: {checkpoint_id}")
        return descriptor

    def load(self, checkpoint_id: str):
        descriptor = self.descriptor(checkpoint_id)
        if descriptor.terminal_adjudicator == "gocube-katago-japanese-v3":
            require_gocube_komi(
                descriptor.komi,
                context=f"Checkpoint {descriptor.checkpoint_id}",
            )
        cache_key = (descriptor.checkpoint_id, self.device)

        def load_uncached():
            try:
                if descriptor.metadata_error:
                    raise CheckpointMetadataInvalid(descriptor.metadata_error)
                payload, saved_args = _read_checkpoint_args(descriptor.path)
                requested_contract = _descriptor_contract(descriptor)
                fallback = requested_contract or _legacy_descriptor_contract(descriptor)
                saved_contract = None
                if saved_args is not None:
                    try:
                        saved_contract = resolve_model_contract_from_metadata(
                            saved_args,
                            fallback=fallback,
                        )
                    except ContractError as exc:
                        raise CheckpointMetadataInvalid(
                            f"Checkpoint {descriptor.checkpoint_id} has invalid model contract: {exc}"
                        ) from exc
                contract = saved_contract or fallback
                if saved_args is not None:
                    _validate_descriptor_against_contract(descriptor, contract)
                if requested_contract is not None and contract.differences(requested_contract):
                    field, (saved, expected) = next(iter(contract.differences(requested_contract).items()))
                    raise CheckpointMetadataInvalid(
                        f"Checkpoint GoCube contract mismatch for {_contract_error_field(field)}: "
                        f"saved={saved!r}, expected={expected!r}"
                    )
                cls = resolve_game_class_from_contract(contract)
                if saved_args is not None:
                    computed = resolve_model_contract(cls, saved_args)
                    if contract.differences(computed):
                        field, (saved, expected) = next(iter(contract.differences(computed).items()))
                        raise CheckpointMetadataInvalid(
                            f"Checkpoint GoCube contract mismatch for {_contract_error_field(field)}: "
                            f"saved={saved!r}, expected={expected!r}"
                        )
                model = NNetWrapper.from_checkpoint(
                    cls,
                    folder="",
                    filename=descriptor.path,
                    device=self.device,
                    load_training_state=False,
                )
                _validate_saved_gocube_metadata(model, descriptor, contract)
                return model
            except (CheckpointMetadataInvalid, CheckpointLoadFailed):
                raise
            except ContractError as exc:
                raise CheckpointMetadataInvalid(
                    f"Checkpoint {descriptor.checkpoint_id} has incompatible model contract: {exc}"
                ) from exc
            except Exception as exc:
                raise CheckpointLoadFailed(
                    f"Failed to load checkpoint {descriptor.checkpoint_id}: {exc}"
                ) from exc

        return descriptor, self.cache.get_or_load(cache_key, load_uncached)
