from types import SimpleNamespace

import pytest

from alphazero.envs.gocube.integration.catalog import CheckpointDescriptor
from alphazero.envs.gocube.integration.errors import CheckpointMetadataInvalid
from alphazero.envs.gocube.integration.models import (
    CheckpointModelLoader,
    ModelCache,
    resolve_device,
)
from alphazero.utils import dotdict


def descriptor(checkpoint_id="run@5"):
    return CheckpointDescriptor(
        checkpoint_id=checkpoint_id,
        run_name="run",
        iteration=int(checkpoint_id.split("@")[-1]),
        topology="cube",
        size=4,
        rule_set="chinese",
        komi=7.5,
        terminal_adjudicator="gocube-conservative-area-v1",
        path="/tmp/iteration-0005.pkl",
    )


class Catalog:
    def __init__(self, item):
        self.item = item

    def get(self, checkpoint_id):
        return self.item if checkpoint_id == self.item.checkpoint_id else None


def test_resolve_device_auto_and_explicit_cpu(monkeypatch):
    import alphazero.envs.gocube.integration.models as models

    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="CUDA was requested"):
        resolve_device("cuda")


def test_model_cache_is_bounded_and_reuses_entries():
    cache = ModelCache(max_entries=2)
    a, b, c = object(), object(), object()
    cache.put(("a", "cpu"), a)
    cache.put(("b", "cpu"), b)
    assert cache.get(("a", "cpu")) is a
    cache.put(("c", "cpu"), c)
    assert cache.get(("a", "cpu")) is a
    assert cache.get(("b", "cpu")) is None
    assert cache.get(("c", "cpu")) is c


def test_checkpoint_loader_uses_runtime_device_and_cache(monkeypatch):
    import alphazero.envs.gocube.integration.models as models

    item = descriptor()
    catalog = Catalog(item)
    calls = []
    model = SimpleNamespace(args=dotdict({}))

    def fake_from_checkpoint(game_cls, **kwargs):
        calls.append((game_cls, kwargs))
        return model

    monkeypatch.setattr(models.NNetWrapper, "from_checkpoint", fake_from_checkpoint)
    loader = CheckpointModelLoader(catalog, device="cpu")

    first = loader.load("run@5")[1]
    second = loader.load("run@5")[1]

    assert first is second is model
    assert len(calls) == 1
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["load_training_state"] is False


def test_checkpoint_loader_rejects_saved_metadata_mismatch(monkeypatch):
    import alphazero.envs.gocube.integration.models as models

    item = descriptor()
    catalog = Catalog(item)
    model = SimpleNamespace(args=dotdict({"gocube_topology": "torus"}))
    monkeypatch.setattr(models.NNetWrapper, "from_checkpoint", lambda *args, **kwargs: model)
    loader = CheckpointModelLoader(catalog, device="cpu")

    with pytest.raises(CheckpointMetadataInvalid, match="saved metadata"):
        loader.load("run@5")
