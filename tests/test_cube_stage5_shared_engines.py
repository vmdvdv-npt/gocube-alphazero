from __future__ import annotations

import math
import multiprocessing as mp
import random

import pytest
import torch

from gocube_golden.cube_family import cube_family_topology, initial_cube_state
from gocube_golden.cube_network_v2 import CubeGraphNetV2
from gocube_golden.cube_observation_v2 import (
    build_cube_observation,
    initial_cube_observation_context,
    make_cube_observation_context,
)
from gocube_golden.cube_search import CubeSearchAdapter, CubeSearchPosition
from gocube_golden.cube_selfplay_contract import CubeSelfPlaySearchContract
from gocube_golden.cube_selfplay_v2 import (
    CubeSelfPlayAdapter,
    _CubeCooperativeGame,
)
from gocube_golden.cube_training_targets import build_cube_training_samples
from gocube_golden.diagnostics import operation_stats
from gocube_golden.rules import prepare_legal_actions
from gocube_golden.search import Evaluation, SearchResult
from gocube_golden.selfplay_engine import (
    GameFinished,
    InferenceNeed,
    SelfPlayEngineConfig,
    SharedMemorySpec,
    run_cooperative_selfplay,
)
from gocube_golden.selfplay_policy import (
    apply_root_dirichlet_noise,
    sample_action_from_search_result,
)
from gocube_golden.state import PASS


class _RunnerGame:
    def __init__(self, _context, game_id, _client):
        self.game_id = str(game_id)
        self.waiting = True

    def advance(self):
        if self.waiting:
            return InferenceNeed(7)
        return GameFinished({"game_id": self.game_id})

    def resume(self, evaluation):
        if evaluation != 7:
            raise AssertionError(f"unexpected fake evaluation: {evaluation!r}")
        self.waiting = False


def _runner_write_input(payload, destination):
    destination.fill_(float(payload))


def _runner_decode_output(policy, _wdl):
    return int(round(float(policy[0])))


def _runner_infer(batch):
    values = batch[:, 0].to(dtype=torch.float32)
    return values[:, None], torch.ones((batch.shape[0], 1), dtype=torch.float32)


def _runner_record_metrics(record):
    return {"moves": 1, "technical": False}


class _RunnerAdapter:
    def __init__(self):
        self.worker_context = {"adapter": "fake"}
        self._shared_memory = SharedMemorySpec(
            observation_shape=(1,),
            policy_size=1,
            wdl_size=1,
            write_input=_runner_write_input,
            decode_output=_runner_decode_output,
        )

    @property
    def shared_memory(self):
        return self._shared_memory

    @property
    def infer_shared_batch(self):
        return _runner_infer

    @property
    def worker_game_factory(self):
        return _RunnerGame

    @property
    def record_metrics(self):
        return _runner_record_metrics


def test_cube_search_position_identity_includes_neural_history():
    state = initial_cube_state(size=2)
    initial = initial_cube_observation_context(state)
    altered = make_cube_observation_context(
        state,
        previous_boards=initial.previous_boards,
        previous_action=state.topology.pass_action,
    )
    assert CubeSearchPosition(state, initial).state_key != CubeSearchPosition(state, altered).state_key


def test_cube_search_adapter_advances_pass_history_without_mutating_parent():
    state = initial_cube_state(size=2)
    context = initial_cube_observation_context(state)
    position = CubeSearchPosition(state, context)
    child = CubeSearchAdapter().apply_action(position, PASS)
    assert position.observation_context.previous_action is None
    assert child.observation_context.previous_action == state.topology.pass_action
    assert child.game_state.consecutive_passes == 1
    assert not child.is_terminal


def test_cube_observation_reuses_prepared_legality_context():
    state = initial_cube_state(size=2)
    context = initial_cube_observation_context(state)
    legality = prepare_legal_actions(state)
    with operation_stats() as stats:
        build_cube_observation(state, context, legal_context=legality)
        build_cube_observation(state, context, legal_context=legality)
    assert stats.legal_calculations == 0


def test_shared_root_noise_is_legal_only_and_deterministic():
    generator_a = torch.Generator(device="cpu").manual_seed(7)
    generator_b = torch.Generator(device="cpu").manual_seed(7)
    first = apply_root_dirichlet_noise(
        (0.1, 0.2, 0.3, 0.4),
        (0, 2),
        action_index=lambda action: int(action),
        epsilon=0.25,
        alpha=0.11,
        generator=generator_a,
    )
    second = apply_root_dirichlet_noise(
        (0.1, 0.2, 0.3, 0.4),
        (0, 2),
        action_index=lambda action: int(action),
        epsilon=0.25,
        alpha=0.11,
        generator=generator_b,
    )
    assert first == second
    assert first[1] == first[3] == 0.0
    assert math.isclose(sum(first), 1.0, abs_tol=1e-12)


def test_shared_temperature_sampler_preserves_canonical_zero_temperature_tie():
    result = SearchResult(
        action=0,
        legal_actions=(2, 0),
        root_visits=(4, 0, 4),
        pi=(0.5, 0.0, 0.5),
        simulations=8,
        evaluator_calls=1,
        legal_action_mask=(True, False, True),
    )
    assert sample_action_from_search_result(
        result,
        temperature=0.0,
        rng=random.Random(3),
        action_index=lambda action: int(action),
    ) == 0


def test_cube_scientific_config_does_not_contain_execution_fields():
    config = CubeSelfPlaySearchContract(simulations=2, technical_move_limit=3)
    config.validate()
    assert not hasattr(config, "workers")
    assert config.puct_settings.simulations == 2


def test_common_cooperative_runner_accepts_structural_adapter_contract():
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("common runner smoke requires fork")
    result = run_cooperative_selfplay(
        ["fake-0"],
        adapter=_RunnerAdapter(),
        engine_config=SelfPlayEngineConfig(
            workers=1,
            inference_batch_cap=1,
            inference_batch_wait_ms=0.0,
            device="cpu",
            process_start_method="fork",
            inference_request_timeout_s=5.0,
            active_games_per_worker=1,
        ),
    )
    assert result.records == ({"game_id": "fake-0"},)
    assert result.telemetry["shared_memory_transport"] is True


def _play_pass_fixture(contract):
    topology = cube_family_topology(2)
    model = CubeGraphNetV2(topology=topology).eval()
    adapter = CubeSelfPlayAdapter(model, size=2, contract=contract)
    game = _CubeCooperativeGame(adapter.worker_context, "pass-fixture", None)
    while True:
        step = game.advance()
        if isinstance(step, GameFinished):
            return step.record
        assert isinstance(step, InferenceNeed)
        policy = [0.0] * topology.action_count
        policy[topology.pass_action] = 1.0
        game.resume(Evaluation(policy=tuple(policy), wdl=(0.0, 1.0, 0.0)))


def test_cube_formal_double_pass_record_projects_targets_and_excludes_technical_games():
    formal = _play_pass_fixture(
        CubeSelfPlaySearchContract(
            simulations=1,
            root_noise=False,
            temperature_plies=(1, 1),
            technical_move_limit=3,
        )
    )
    formal.validate(deep=True)
    samples = build_cube_training_samples(formal)
    assert formal.completion == "FORMAL_DOUBLE_PASS"
    assert formal.final_action_trace == (24, 24)
    assert len(samples) == 2
    assert samples[0].selected_action == 24
    assert samples[0].wdl == (0, 0, 1)
    assert samples[1].wdl == (1, 0, 0)
    assert samples[0].score_exact == -0.5

    technical = _play_pass_fixture(
        CubeSelfPlaySearchContract(
            simulations=1,
            root_noise=False,
            temperature_plies=(1, 1),
            technical_move_limit=1,
        )
    )
    assert technical.formal_result is None
    assert technical.technical_termination == "MOVE_LIMIT"
    assert build_cube_training_samples(technical) == ()
