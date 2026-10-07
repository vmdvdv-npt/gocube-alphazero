"""No production games or transports: exercise durable parent selection."""
import json
from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig
from gocube_golden.orchestrator_v2 import operator_job as job
from gocube_golden.orchestrator_v2.workflow import WorkflowRunner, WorkflowSpec


def parameters():
    return {
        "schema": job.SCHEMA, "run_id": "winner-continuation",
        "winner_selection": {"candidate": "source/M249", "reference": "source/M246",
                             "games": 192, "mcts_simulations": 200},
        "training": {"iterations": 3, "gradient_clip": 8.0, "replay_generations": 3},
        "arena": {"every_iterations": 1, "games": 192, "mcts_simulations": 64},
        "self_play": {"master_seed": 2026100501, "search_mode": "pcr", "pcr": {
            "cheap_simulations": 100, "full_simulations": 500, "full_probability": 0.25}},
    }


@pytest.fixture
def spec(monkeypatch):
    def resolve(selector, *, resolver):
        generation = int(selector.split("/M")[1])
        config = EffectiveConfig(
            topology="torus9", compatibility={"input_channels": 5},
            self_play={"komi": 1.5}, training={"optimizer": "Adam", "batch_size": 64,
                                                "weight_decay": 0.0, "l2_sp": False},
            replay={"cap": None}, execution={"device": "cuda", "workers": 16},
            extensions={"training_driver": job.DRIVER, "parent_marker": generation},
        )
        return SimpleNamespace(ref=CheckpointRef(
            "torus9", "source", f"M{generation}", generation,
            f"checkpoints/M{generation}.pt", "sha256:" + ("a" if generation == 249 else "b") * 64),
            effective_config=SimpleNamespace(config=config))
    monkeypatch.setattr(job, "resolve_checkpoint", resolve)
    return WorkflowSpec.from_dict(job.compile_job(parameters(), resolver=SimpleNamespace())["workflow"])


def arena_result(config, wins=110, losses=82):
    return {"validity": "VALID", "candidate_checkpoint": config["candidate_checkpoint"],
            "reference_checkpoint": config["reference_checkpoint"],
            "evaluation_id": "evaluation-1", "evaluation_fingerprint": "fingerprint-1",
            "metrics": {"candidate_wins": wins, "reference_wins": losses,
                        "draws": 0, "valid_games": 192}, "summary": {"technical_games": 0}}


@pytest.mark.parametrize("wins,losses,parent", [(110, 82, 249), (82, 110, 246)])
def test_both_winners_inherit_their_own_config_and_resume_without_replaying_arena(
    spec, tmp_path, wins, losses, parent,
):
    calls = []
    def arena(*, config, **_):
        calls.append("arena")
        assert "simulations=200" in config["profile"]
        return arena_result(config, wins, losses)
    def training(*, config, **_):
        calls.append("training")
        assert config["parent_checkpoint"]["generation"] == parent
        assert config["effective_config"]["extensions"]["parent_marker"] == parent
        assert config["effective_config"]["training"]["optimizer"] == "Adam"
        assert config["effective_config"]["training"]["gradient_clip"] == 8.0
        assert config["effective_config"]["self_play"]["search_mode"] == "pcr"
        assert config["effective_config"]["execution"]["selfplay_master_seed"] == 2026100501
        assert config["effective_config"]["replay"]["generations"] == 3
        assert "simulations=64" in config["arena_profile"]
        assert config["arena_cadence"] == 1
        assert config["lineage_id"] == "winner-continuation"
        assert config["generations"] == 3
        return {"checkpoint": {**config["parent_checkpoint"], "generation": parent + 3}}
    for _ in range(2):
        result = WorkflowRunner(spec, root=tmp_path, handlers={
            "arena": arena, "continuous_training": training}).run()
        assert result["state"] == "COMPLETED"
    assert calls == ["arena", "training"]
    assert result["steps"]["winner"]["outputs"]["selected"]["generation"] == parent


@pytest.mark.parametrize("change", [
    {"validity": "INVALID"},
    {"metrics": {"candidate_wins": 96, "reference_wins": 96, "draws": 0, "valid_games": 192}},
    {"metrics": {"candidate_wins": 110, "reference_wins": 81, "draws": 0, "valid_games": 191}},
    {"metrics": {"candidate_wins": True, "reference_wins": 191, "draws": 0, "valid_games": 192}},
    {"summary": {"technical_games": 1}},
    {"summary": {}},
    {"candidate_checkpoint": {"checkpoint_id": "wrong-model"}},
])
def test_unsafe_decisions_stop_durably_without_training(spec, tmp_path, change):
    calls = []
    def arena(*, config, **_):
        calls.append("arena")
        return {**arena_result(config), **change}
    def training(**_):
        pytest.fail("training must not run for an unsafe decision")
    for _ in range(2):
        state = WorkflowRunner(spec, root=tmp_path, handlers={
            "arena": arena, "continuous_training": training}).run()
        assert state["state"] == "STOPPED"
        assert state["steps"]["training"]["status"] == "PENDING"
    assert calls == ["arena"]


def test_resume_after_training_interruption_reuses_persisted_winner(spec, tmp_path):
    calls = []
    def arena(*, config, **_):
        calls.append("arena")
        return arena_result(config, 80, 112)
    def training(*, config, **_):
        calls.append("training")
        assert config["parent_checkpoint"]["generation"] == 246
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        WorkflowRunner(spec, root=tmp_path, handlers={
            "arena": arena, "continuous_training": training}).run()
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["steps"]["winner"]["status"] == "COMPLETED"
    def resume(*, config, **_):
        assert config["parent_checkpoint"]["generation"] == 246
        return {"checkpoint": config["parent_checkpoint"]}
    state = WorkflowRunner(spec, root=tmp_path, handlers={
        "arena": arena, "continuous_training": resume}).run()
    assert state["state"] == "COMPLETED"
    assert calls == ["arena", "training"]


@pytest.mark.parametrize("patch", [
    {"parent": "source/M249"}, {"training": {"iterations": 0}},
    {"winner_selection": {"candidate": "source/M249", "reference": "source/M249"}},
    {"winner_selection": {"candidate": "source/M249", "reference": "source/M246", "command": "x"}},
    {"winner_selection": {"candidate": "source/M249", "reference": "source/M246", "games": 193}},
    {"ab_tests": [{"id": "lr", "iterations": 1, "A": {}, "B": {}}]},
])
def test_bad_selection_rejected_before_resolution(monkeypatch, patch):
    monkeypatch.setattr(job, "resolve_checkpoint", lambda *a, **kw: pytest.fail("resolved invalid input"))
    with pytest.raises(ValueError):
        job.compile_job({**parameters(), **patch})


def test_normalization_and_unbounded_selection(monkeypatch, spec):
    raw = parameters()
    normalized = job.parse_job(raw)
    assert job.parse_job(normalized) == normalized
    raw["training"]["iterations"] = None
    compiled = job.compile_job(raw, resolver=SimpleNamespace())
    assert compiled["workflow"]["steps"][-1]["config"]["generations"] is None
