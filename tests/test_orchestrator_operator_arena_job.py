from types import SimpleNamespace

import pytest

from gocube_golden.artifact_graph import CheckpointRef, EffectiveConfig
from gocube_golden.orchestrator_v2 import operator_job as job


def _node(lineage: str, checkpoint_id: str, generation: int, fill: str):
    config = EffectiveConfig(
        topology="torus9",
        compatibility={"input_channels": 5},
        self_play={"komi": 1.5},
        training={"optimizer": "Adam", "batch_size": 64},
        replay={"cap": None},
        execution={"device": "cuda", "workers": 16},
        extensions={"training_driver": job.DRIVER},
    )
    return SimpleNamespace(
        ref=CheckpointRef(
            "torus9",
            lineage,
            checkpoint_id,
            generation,
            f"checkpoints/{checkpoint_id}.pt",
            "sha256:" + fill * 64,
        ),
        effective_config=SimpleNamespace(config=config),
    )


def _ladder():
    lineage = "torus9-m203-learner-recovery-20260929-v3"
    return {
        "schema": job.SCHEMA,
        "run_id": "m204-m214-ladder",
        "arenas": [
            {
                "id": "m206-vs-m204",
                "candidate": f"{lineage}/M206",
                "reference": f"{lineage}/M204",
                "games": 192,
                "mcts_simulations": 64,
            },
            {
                "id": "m208-vs-m206",
                "candidate": f"{lineage}/M208",
                "reference": f"{lineage}/M206",
                "games": 192,
                "mcts_simulations": 64,
            },
            {
                "id": "m210-vs-m208",
                "candidate": f"{lineage}/M210",
                "reference": f"{lineage}/M208",
                "games": 192,
                "mcts_simulations": 64,
            },
        ],
    }


def test_arena_only_job_does_not_implicitly_train():
    normalized = job.parse_job(_ladder())
    assert normalized["parent"] is None
    assert normalized["training"]["iterations"] == 0
    assert normalized["ab_tests"] == []
    assert len(normalized["arenas"]) == 3
    assert job.parse_job(normalized) == normalized


def test_arena_ladder_compiles_to_sequential_v2_arena_steps(monkeypatch):
    raw = _ladder()
    lineage = "torus9-m203-learner-recovery-20260929-v3"
    nodes = {
        f"{lineage}/M204": _node(lineage, "M204", 204, "a"),
        f"{lineage}/M206": _node(lineage, "M206", 206, "b"),
        f"{lineage}/M208": _node(lineage, "M208", 208, "c"),
        f"{lineage}/M210": _node(lineage, "M210", 210, "d"),
    }
    monkeypatch.setattr(
        job,
        "resolve_checkpoint",
        lambda selector, *, resolver: nodes[selector],
    )

    compiled = job.compile_job(raw, resolver=SimpleNamespace())
    steps = compiled["workflow"]["steps"]

    assert [step["action"] for step in steps] == ["arena", "arena", "arena"]
    assert [step["step_id"] for step in steps] == [
        "arena-m206-vs-m204",
        "arena-m208-vs-m206",
        "arena-m210-vs-m208",
    ]
    assert steps[0]["dependencies"] == []
    assert steps[1]["dependencies"] == ["arena-m206-vs-m204"]
    assert steps[2]["dependencies"] == ["arena-m208-vs-m206"]

    for step in steps:
        config = step["config"]
        assert config["arena_config"]["games"] == 192
        assert "simulations=64" in config["profile"]
        assert "komi=1.5" in config["profile"]
        assert config["master_seed"] == job.ARENA_RUN_MASTER_SEED
        assert config["workload"]["paired_starts"] is True
        assert config["workload"]["color_swap"] is True
        assert config["workload"]["gating"] == "off"
        assert config["candidate_checkpoint"]["sha256"].startswith("sha256:")
        assert config["reference_checkpoint"]["sha256"].startswith("sha256:")

    assert steps[0]["config"]["candidate_checkpoint"]["checkpoint_id"] == "M206"
    assert steps[0]["config"]["reference_checkpoint"]["checkpoint_id"] == "M204"


def test_arena_only_job_resolves_every_checkpoint_before_runtime(monkeypatch):
    raw = _ladder()
    seen = []

    def fail_on_second(selector, *, resolver):
        seen.append(selector)
        if selector.endswith("/M204"):
            raise ValueError("missing registered checkpoint")
        return _node("lineage", "M206", 206, "e")

    monkeypatch.setattr(job, "resolve_checkpoint", fail_on_second)
    with pytest.raises(ValueError, match="missing registered checkpoint"):
        job.compile_job(raw, resolver=SimpleNamespace())
    assert seen[:2] == [
        raw["arenas"][0]["candidate"],
        raw["arenas"][0]["reference"],
    ]


@pytest.mark.parametrize(
    "arena_patch",
    [
        {"games": 193},
        {"games": 62},
        {"mcts_simulations": 0},
        {"candidate": "../../M206"},
        {"reference": "lineage/../../M204"},
        {"master_seed": -1},
    ],
)
def test_bad_arena_job_parameters_fail_before_checkpoint_resolution(
    monkeypatch, arena_patch
):
    raw = _ladder()
    raw["arenas"][0].update(arena_patch)
    monkeypatch.setattr(
        job,
        "resolve_checkpoint",
        lambda *args, **kwargs: pytest.fail("resolved before parameter validation"),
    )
    with pytest.raises(ValueError):
        job.compile_job(raw, resolver=SimpleNamespace())


def test_duplicate_arena_ids_are_rejected():
    raw = _ladder()
    raw["arenas"][1]["id"] = raw["arenas"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate Arena id"):
        job.parse_job(raw)


def test_training_still_requires_parent_when_no_arena_only_shape():
    raw = {
        "schema": job.SCHEMA,
        "run_id": "training-without-parent",
        "training": {"iterations": 1},
    }
    with pytest.raises(ValueError, match="parent"):
        job.parse_job(raw)


def test_cross_lineage_arena_keeps_checkpoint_references_not_copies(monkeypatch):
    raw = {
        "schema": job.SCHEMA,
        "run_id": "cross-lineage-eval",
        "arenas": [
            {
                "id": "candidate-vs-reference",
                "candidate": "candidate-line/M214",
                "reference": "reference-line/M208",
                "games": 192,
                "mcts_simulations": 64,
                "master_seed": 12345,
            }
        ],
    }
    nodes = {
        "candidate-line/M214": _node("candidate-line", "M214", 214, "f"),
        "reference-line/M208": _node("reference-line", "M208", 208, "1"),
    }
    monkeypatch.setattr(
        job,
        "resolve_checkpoint",
        lambda selector, *, resolver: nodes[selector],
    )

    step = job.compile_job(raw, resolver=SimpleNamespace())["workflow"]["steps"][0]
    config = step["config"]
    assert config["candidate_checkpoint"]["lineage_id"] == "candidate-line"
    assert config["reference_checkpoint"]["lineage_id"] == "reference-line"
    assert config["master_seed"] == 12345
    assert "output_dir" not in config
