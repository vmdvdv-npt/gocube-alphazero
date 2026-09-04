from __future__ import annotations

from collections import OrderedDict
from threading import Lock

import torch

from alphazero.NNetWrapper import NNetWrapper
from alphazero.envs.gocube.game import game_class

from .catalog import CheckpointCatalog, CheckpointDescriptor
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


def _validate_saved_gocube_metadata(model, descriptor: CheckpointDescriptor) -> None:
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
        if field in args and args[field] != value:
            raise CheckpointMetadataInvalid(
                f"Checkpoint {descriptor.checkpoint_id} saved metadata {field}={args[field]!r} "
                f"does not match run manifest value {value!r}"
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
        cache_key = (descriptor.checkpoint_id, self.device)

        def load_uncached():
            try:
                cls = game_class(descriptor.topology, descriptor.size)
                model = NNetWrapper.from_checkpoint(
                    cls,
                    folder="",
                    filename=descriptor.path,
                    device=self.device,
                    load_training_state=False,
                )
                _validate_saved_gocube_metadata(model, descriptor)
                return model
            except CheckpointMetadataInvalid:
                raise
            except Exception as exc:
                raise CheckpointLoadFailed(
                    f"Failed to load checkpoint {descriptor.checkpoint_id}: {exc}"
                ) from exc

        return descriptor, self.cache.get_or_load(cache_key, load_uncached)
