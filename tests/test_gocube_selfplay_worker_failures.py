from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
import torch
from torch import multiprocessing as mp

import alphazero.Coach as coach_module
from alphazero.Coach import Coach
from alphazero.envs.connect4.connect4 import Game as Connect4Game
from alphazero.envs.gocube.atomic_io import (
    REPLAY_TENSOR_SUFFIXES,
    atomic_torch_save,
    find_last_valid_contiguous_checkpoint,
    replay_marker_path,
)
from alphazero.envs.gocube.hardened_train import HardenedKataGoSearchCoach
from alphazero.worker_errors import SelfPlayWorkerError


class _Writer:
    def add_scalar(self, *_args, **_kwargs):
        pass


class _InferenceNetwork:
    def process(self, batch):
        return (
            torch.zeros(batch.size(0), Connect4Game.action_size()),
            torch.zeros(batch.size(0), Connect4Game.num_players() + 1),
        )


def _agent_args():
    return SimpleNamespace(
        workers=2,
        process_batch_size=1,
        cuda=False,
        run_name="",
        gamesPerIteration=1,
        numMCTSSims=1,
        numFastSims=1,
        numWarmupSims=1,
        probFastSim=0.0,
        startTemp=1.0,
        temp_scaling_fn=lambda temp, _turns, _max_turns: temp,
        arenaTemp=0.25,
        _num_players=3,
        root_noise_frac=0.0,
        root_policy_temp=1.0,
        min_discount=1.0,
        fpu_reduction=0.0,
        cpuct=1.25,
        gocube_recording_enabled=False,
        model_gating=False,
        add_root_noise=False,
        add_root_temp=False,
    )


def _runtime_coach():
    coach = object.__new__(Coach)
    coach.args = _agent_args()
    coach.game_cls = Connect4Game
    coach.model_iter = 37
    coach.warmup = False
    coach.agents = []
    coach.input_tensors = []
    coach.policy_tensors = []
    coach.value_tensors = []
    coach.batch_ready = []
    coach.stop_train = mp.Event()
    coach.pause_train = mp.Event()
    coach.stop_agents = mp.Event()
    coach.ready_queue = mp.Queue()
    coach.file_queue = mp.Queue()
    coach.result_queue = mp.Queue()
    coach.worker_error_queue = mp.Queue()
    coach.completed = mp.Value("i", 0)
    coach.games_played = mp.Value("i", 0)
    coach.last_worker_error = None
    coach.train_net = _InferenceNetwork()
    coach.self_play_net = coach.train_net
    coach.writer = _Writer()
    return coach


class _FailureAndInferenceWaitAgent(coach_module.SelfPlayAgent):
    def generateBatch(self):
        if self.id == 0:
            raise RuntimeError("TEST_SEARCH_FAILURE")
        return super().generateBatch()

    def run(self):
        if self.id == 1:
            self._set_worker_context(0, "inference_wait")
            self.ready_queue.put(self.id)
            self.processBatch()
            return
        return super().run()


class _FinishFailureAgent(coach_module.SelfPlayAgent):
    def playMoves(self):
        self._set_worker_context(0, "finish_game")
        raise RuntimeError("TEST_FINISH_FAILURE")


class _SuccessfulAgent(mp.Process):
    def __init__(self, *args, **_kwargs):
        super().__init__()
        self.id = int(args[0])
        self.complete_count = args[9]

    def run(self):
        with self.complete_count.get_lock():
            self.complete_count.value += 1


def _hard_exit():
    os._exit(17)


def test_python_worker_exception_reaches_parent_with_context_and_traceback(monkeypatch):
    coach = _runtime_coach()
    monkeypatch.setattr(coach_module, "SelfPlayAgent", _FailureAndInferenceWaitAgent)
    Coach.generateSelfPlayAgents(coach)
    started = time.monotonic()
    with pytest.raises(SelfPlayWorkerError) as caught:
        Coach.processSelfPlayBatches(coach, 37)
    elapsed = time.monotonic() - started

    message = str(caught.value)
    assert elapsed < 5.0
    assert "worker=0" in message
    assert "iteration=37" in message
    assert "game slot=0" in message
    assert "stage=search_generate" in message
    assert "RuntimeError" in message
    assert "TEST_SEARCH_FAILURE" in message
    assert "Traceback" in message
    assert coach.agents == []


def test_sibling_waiting_for_inference_is_released_after_worker_failure(monkeypatch):
    coach = _runtime_coach()
    monkeypatch.setattr(coach_module, "SelfPlayAgent", _FailureAndInferenceWaitAgent)
    Coach.generateSelfPlayAgents(coach)
    with pytest.raises(SelfPlayWorkerError, match="TEST_SEARCH_FAILURE"):
        Coach.processSelfPlayBatches(coach, 37)
    assert coach.agents == []


def test_finish_game_exception_uses_the_same_parent_propagation_contract(monkeypatch):
    coach = _runtime_coach()
    coach.warmup = True
    monkeypatch.setattr(coach_module, "SelfPlayAgent", _FinishFailureAgent)
    Coach.generateSelfPlayAgents(coach)
    with pytest.raises(SelfPlayWorkerError) as caught:
        Coach.processSelfPlayBatches(coach, 37)
    assert "stage=finish_game" in str(caught.value)
    assert "TEST_FINISH_FAILURE" in str(caught.value)
    assert coach.agents == []


def test_parent_inference_failure_is_re_raised_with_iteration_and_batch_context():
    coach = _runtime_coach()
    coach.args.workers = 1
    coach.input_tensors = [torch.zeros(1, *Connect4Game.observation_size())]
    coach.policy_tensors = [torch.zeros(1, Connect4Game.action_size())]
    coach.value_tensors = [torch.zeros(1, Connect4Game.num_players() + 1)]
    coach.batch_ready = [mp.Event()]
    coach.ready_queue.put(0)

    class FailingNetwork:
        def process(self, _batch):
            raise RuntimeError("TEST_PARENT_INFERENCE_FAILURE")

    coach.train_net = FailingNetwork()
    with pytest.raises(RuntimeError) as caught:
        Coach.processSelfPlayBatches(coach, 37)
    assert "TEST_PARENT_INFERENCE_FAILURE" in str(caught.value)
    assert "iteration=37" in str(caught.value)
    assert "worker_ids=[0]" in str(caught.value)


def test_successful_workers_complete_without_false_positive_error(monkeypatch):
    coach = _runtime_coach()
    monkeypatch.setattr(coach_module, "SelfPlayAgent", _SuccessfulAgent)
    Coach.generateSelfPlayAgents(coach)
    Coach.processSelfPlayBatches(coach, 37)
    assert coach.completed.value == coach.args.workers
    assert coach._drain_worker_error_queue() == []
    Coach.killSelfPlayAgents(coach)
    assert coach.agents == []


def test_hard_child_exit_is_detected_without_error_payload():
    coach = _runtime_coach()
    coach.args.workers = 1
    agent = mp.Process(target=_hard_exit)
    agent.id = 0
    coach.agents = [agent]
    coach.batch_ready = [mp.Event()]
    agent.start()
    agent.join(timeout=2.0)
    assert agent.exitcode == 17
    with pytest.raises(SelfPlayWorkerError) as caught:
        coach._check_selfplay_workers(37)
    assert "exited unexpectedly with exit code 17" in str(caught.value)
    coach._abort_selfplay_agents()
    assert coach.agents == []
    assert not agent.is_alive()


def test_worker_error_payload_has_the_fixed_picklable_shape():
    payload = {
        "worker_id": 3,
        "pid": 123,
        "iteration": 37,
        "game_slot": 1,
        "game_id": None,
        "stage": "finish_game",
        "exception_type": "RuntimeError",
        "exception_message": "TEST_FINISH_FAILURE",
        "traceback": "Traceback (most recent call last):",
    }
    error = SelfPlayWorkerError(payload)
    assert error.payload == payload
    assert all(key in error.payload for key in (
        "worker_id", "pid", "iteration", "game_slot", "game_id", "stage",
        "exception_type", "exception_message", "traceback",
    ))
    assert "TEST_FINISH_FAILURE" in str(error)


class _ReplayGame:
    @staticmethod
    def observation_size():
        return (2,)

    @staticmethod
    def action_size():
        return 4

    @staticmethod
    def num_players():
        return 2

    @staticmethod
    def logical_topology():
        return SimpleNamespace(point_count=1)


def test_replay_write_failure_leaves_no_completion_marker_and_preserves_checkpoint(tmp_path, monkeypatch):
    checkpoint_folder = tmp_path / "checkpoint" / "run"
    checkpoint_folder.mkdir(parents=True)
    atomic_torch_save({"state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_folder / "iteration-0000.pkl")

    coach = object.__new__(HardenedKataGoSearchCoach)
    coach.args = SimpleNamespace(
        data=os.fspath(tmp_path / "data"),
        run_name="run",
        symmetricSamples=True,
        gocube_endgame_sample_weight=1,
    )
    coach.file_queue = SimpleNamespace(qsize=lambda: 0)
    coach.game_cls = _ReplayGame
    coach._iteration_telemetry = {
        "base_positions": 0,
        "base_endgame_positions": 0,
        "endgame_extra_samples": 0,
        "saved_total": 0,
    }
    coach.writer = _Writer()

    real_save = torch.save
    calls = {"count": 0}

    def flaky_save(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("TEST_REPLAY_WRITE_FAILURE")
        return real_save(*args, **kwargs)

    import alphazero.envs.gocube.training_common as training_common

    monkeypatch.setattr(training_common.torch, "save", flaky_save)
    with pytest.raises(RuntimeError, match="TEST_REPLAY_WRITE_FAILURE"):
        coach.saveIterationSamples(1)

    final_base = tmp_path / "data" / "run" / "iteration-0001"
    assert not os.path.exists(replay_marker_path(os.fspath(final_base)))
    assert not any(os.path.exists(os.fspath(final_base) + suffix) for suffix in REPLAY_TENSOR_SUFFIXES)
    assert find_last_valid_contiguous_checkpoint(checkpoint_folder)[0] == 0
    assert list((tmp_path / "data").glob(".*.staging-*")) == []
