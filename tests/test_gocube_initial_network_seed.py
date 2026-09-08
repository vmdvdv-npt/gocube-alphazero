from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from alphazero.envs.gocube import hardened_train


def hash_state_dict(network):
    """Hash the complete numerical state of a network deterministically."""

    state_dict = network.nnet.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            raise TypeError(f"state_dict entry {key!r} is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


_SUBPROCESS_PROBE = r"""
import hashlib
import sys

import torch

from alphazero.envs.gocube.hardened_train import (
    AtomicSampleClockNNetWrapper,
    build_hardened_training_args,
)
from alphazero.envs.gocube.katago_train import parse_args
from alphazero.envs.gocube.reproducibility import seed_process


def hash_state_dict(network):
    state_dict = network.nnet.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


seed = int(sys.argv[1])
cli = parse_args(["--seed", str(seed), "--smoke", "--no-arena", "--run-name", "seed-probe"])
game_cls, args = build_hardened_training_args(cli)
seed_process(seed)
network = AtomicSampleClockNNetWrapper(game_cls, args)
print(hash_state_dict(network))
"""

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _initial_network_digest(seed):
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROBE, str(seed)],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    digest = result.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", digest), result.stdout
    return digest


def test_same_seed_produces_same_initial_network_in_independent_processes():
    hash_a = _initial_network_digest(123)
    hash_b = _initial_network_digest(123)

    assert hash_a == hash_b


def test_different_seeds_produce_different_initial_networks_in_independent_processes():
    hash_123 = _initial_network_digest(123)
    hash_124 = _initial_network_digest(124)

    assert hash_123 != hash_124


@pytest.mark.parametrize("resume", [False, True])
def test_hardened_main_seeds_before_network_construction(monkeypatch, resume):
    events = []
    cli = SimpleNamespace(allow_existing_run=resume, allow_dirty_source=False)
    args = SimpleNamespace(
        checkpoint="checkpoint",
        run_name="seed-order-test",
        master_seed=123456,
    )
    game_cls = object()

    monkeypatch.setattr(hardened_train, "parse_args", lambda _argv=None: cli)
    monkeypatch.setattr(hardened_train, "build_hardened_training_args", lambda _cli: (game_cls, args))
    monkeypatch.setattr(hardened_train, "assert_fresh_run", lambda _args: None)
    monkeypatch.setattr(hardened_train, "assert_resumable_run", lambda _args: None)
    monkeypatch.setattr(
        hardened_train,
        "create_reproducible_manifest",
        lambda **_kwargs: {"effective_config_sha256": "config-hash"},
    )
    monkeypatch.setattr(
        hardened_train,
        "validate_existing_reproducible_manifest",
        lambda **_kwargs: {"effective_config_sha256": "config-hash"},
    )
    monkeypatch.setattr(hardened_train, "print_hardened_configuration", lambda _args: None)
    monkeypatch.setattr(
        hardened_train,
        "ensure_training_manifest",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        hardened_train,
        "checkpoint_arg_overrides",
        lambda *_args: {},
    )

    def fake_seed(seed):
        events.append(("seed", seed))

    class FakeNetwork:
        def __init__(self, _game_cls, _args):
            events.append("network-created")

    class FakeCoach:
        def __init__(self, _game_cls, _network, _args):
            events.append("coach-created")

        def learn(self):
            events.append("learn")

    monkeypatch.setattr(hardened_train, "seed_process", fake_seed)
    monkeypatch.setattr(hardened_train, "AtomicSampleClockNNetWrapper", FakeNetwork)
    monkeypatch.setattr(hardened_train, "HardenedKataGoSearchCoach", FakeCoach)

    hardened_train.main([])

    assert events == [
        ("seed", 123456),
        "network-created",
        "coach-created",
        "learn",
    ]
