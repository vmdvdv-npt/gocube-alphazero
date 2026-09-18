from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from gocube_golden.orchestrator_v2.contracts import (
    ActiveExecution,
    ArtifactRef,
    BusinessState,
    ChangeClass,
    CheckpointNode,
    CheckpointRef,
    EffectiveConfig,
    EffectiveConfigRef,
    EvaluationIdentity,
    EvaluationValidity,
    ExecutionKind,
    ParameterChange,
    PARAMETER_CHANGEABILITY_V2,
    ResolvedBoundary,
    RunMode,
    RunState,
    RuntimeAmendment,
    StartsetRef,
    reusable_scientific_result,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64
SHA_E = "sha256:" + "e" * 64
SHA_F = "sha256:" + "f" * 64


def cp(lineage: str, generation: int, sha: str = SHA_A) -> CheckpointRef:
    return CheckpointRef("torus9", lineage, f"M{generation}", generation, f"checkpoints/M{generation}.pt", sha)


def artifact(path: str, sha: str = SHA_B) -> ArtifactRef:
    return ArtifactRef(path, sha)


def config_ref(sha: str = SHA_C, fingerprint: str = SHA_D) -> EffectiveConfigRef:
    return EffectiveConfigRef(artifact("effective-configs/config.json", sha), fingerprint)


def node(child: CheckpointRef, parent: CheckpointRef | None) -> CheckpointNode:
    return CheckpointNode(
        checkpoint=child,
        genesis=parent is None,
        parent=parent,
        fresh_replay=None if parent is None else artifact(f"replay/fresh-M{child.generation}.jsonl"),
        effective_config=config_ref(),
        provenance=artifact(f"provenance/generations/M{child.generation}.json", SHA_E),
    )


def test_checkpoint_node_genesis_same_lineage_and_cross_lineage():
    assert node(cp("root", 0), None).parent is None
    assert node(cp("same", 2), cp("same", 1, SHA_B)).parent.lineage_id == "same"
    assert node(cp("child", 83), cp("parent", 82, SHA_B)).parent.lineage_id == "parent"


def test_checkpoint_node_non_genesis_requires_parent_and_valid_identity():
    with pytest.raises(ValueError, match="immediate parent"):
        CheckpointNode(
            checkpoint=cp("x", 1),
            genesis=False,
            parent=None,
            fresh_replay=artifact("replay/fresh-M1.jsonl"),
            effective_config=config_ref(),
            provenance=artifact("provenance/generations/M1.json", SHA_E),
        )
    with pytest.raises(ValueError, match="sha256"):
        CheckpointRef("torus9", "x", "M1", 1, "checkpoints/M1.pt", "bad")


@pytest.mark.parametrize("genesis", ["false", 0, 1, None, [], {}])
def test_checkpoint_node_parser_rejects_non_boolean_genesis(genesis):
    payload = node(cp("x", 1), cp("x", 0, SHA_B)).to_dict()
    payload["genesis"] = genesis
    with pytest.raises(ValueError, match="genesis.*boolean"):
        CheckpointNode.from_dict(payload)


@pytest.mark.parametrize("field", ["parent", "fresh_replay"])
def test_checkpoint_node_parser_rejects_malformed_optional_structured_fields(field):
    payload = node(cp("x", 1), cp("x", 0, SHA_B)).to_dict()
    payload[field] = "not-an-object"
    with pytest.raises(ValueError, match=field):
        CheckpointNode.from_dict(payload)


def test_checkpoint_node_schema_has_no_full_ancestry_or_replay_chain():
    payload = node(cp("x", 1), cp("x", 0, SHA_B)).to_dict()
    assert "ancestors" not in payload
    assert "replay_references" not in payload
    payload["ancestors"] = ["M0"]
    with pytest.raises(ValueError, match="forbids"):
        CheckpointNode.from_dict(payload)


def effective(lr=0.0003, sims=128) -> EffectiveConfig:
    return EffectiveConfig(
        topology="torus9",
        compatibility={"rules": "graph-area", "board": "9x9-torus", "model": "graphnet-v2"},
        self_play={"mcts_simulations": sims, "games_per_iteration": 128},
        training={"optimizer": "Adam", "learning_rate": lr, "optimizer_steps": 160},
        replay={"window": 6, "cap": 40000},
        execution={"workers": 16, "active_contexts": 64},
        arena={"games": 128, "reference_gap": 5},
        supervision={"max_generation_restarts": 2},
    )


def test_effective_config_fingerprint_deterministic_and_sensitive():
    a = effective()
    b = EffectiveConfig.from_dict(a.to_dict())
    assert a.fingerprint == b.fingerprint
    assert a.canonical_json == b.canonical_json
    assert effective(lr=0.0002).fingerprint != a.fingerprint
    assert effective(sims=256).fingerprint != a.fingerprint


def test_effective_config_has_no_golden_whitelist():
    cfg = EffectiveConfig(
        topology="torus9",
        compatibility={"rules": "graph-area", "board": "9x9-torus", "model": "graphnet-v2"},
        self_play={"mcts_simulations": 256, "games_per_iteration": 90},
        training={"learning_rate": 0.0002},
    )
    assert cfg.self_play["games_per_iteration"] == 90
    with pytest.raises(TypeError):
        cfg.self_play["mcts_simulations"] = 128


def test_changeability_policy_is_declarative_and_covers_all_three_classes():
    classes = {rule.change_class for rule in PARAMETER_CHANGEABILITY_V2}
    assert classes == {
        ChangeClass.NEXT_GENERATION,
        ChangeClass.NEXT_EXECUTION_UNIT,
        ChangeClass.INCOMPATIBLE_NEW_RUN,
    }
    assert len({rule.path for rule in PARAMETER_CHANGEABILITY_V2}) == len(PARAMETER_CHANGEABILITY_V2)


def test_runtime_amendment_persists_old_new_boundary_and_is_immutable():
    change = ParameterChange(
        "self_play.mcts_simulations",
        128,
        256,
        ChangeClass.NEXT_GENERATION,
        ResolvedBoundary("generation", generation=122),
    )
    amendment = RuntimeAmendment(
        "amend-0001",
        "run-1",
        "2026-09-18T06:00:00+00:00",
        (change,),
        config_ref(SHA_A, SHA_B),
        config_ref(SHA_C, SHA_D),
    )
    payload = amendment.to_dict()
    assert payload["requested_changes"][0]["old_value"] == 128
    assert payload["requested_changes"][0]["new_value"] == 256
    assert payload["requested_changes"][0]["resolved_boundary"] == {"kind": "generation", "generation": 122}
    assert RuntimeAmendment.from_dict(payload).fingerprint == amendment.fingerprint
    with pytest.raises(FrozenInstanceError):
        amendment.run_id = "other"


def test_runtime_amendment_parser_rejects_malformed_requested_change():
    change = ParameterChange(
        "self_play.mcts_simulations", 128, 256, ChangeClass.NEXT_GENERATION,
        ResolvedBoundary("generation", generation=122),
    )
    payload = RuntimeAmendment(
        "amend-0001", "run-1", "2026-09-18T06:00:00+00:00", (change,),
        config_ref(SHA_A, SHA_B), config_ref(SHA_C, SHA_D),
    ).to_dict()
    payload["requested_changes"].append("not-an-object")
    with pytest.raises(ValueError, match="requested change"):
        RuntimeAmendment.from_dict(payload)


def test_runtime_amendment_rejects_unresolved_or_incompatible_change():
    with pytest.raises(ValueError):
        ParameterChange(
            "topology", "torus9", "cube4", ChangeClass.INCOMPATIBLE_NEW_RUN,
            ResolvedBoundary("generation", generation=122),
        )


def state(**kwargs) -> RunState:
    base = dict(
        run_id="run-1",
        mode=RunMode.CONTINUOUS,
        state=BusinessState.READY,
        lineage_id="lineage-1",
        base_config=config_ref(),
        created_at="2026-09-18T06:00:00+00:00",
        updated_at="2026-09-18T06:01:00+00:00",
    )
    base.update(kwargs)
    return RunState(**base)


def test_run_state_roundtrip_running_generation_arena_and_stopped():
    running = state(
        state=BusinessState.RUNNING_GENERATION,
        active_execution=ActiveExecution(ExecutionKind.GENERATION, "generation-M84", 1, generation=84),
    )
    assert RunState.from_dict(running.to_dict()) == running
    arena = state(
        state=BusinessState.RUNNING_ARENA,
        active_execution=ActiveExecution(ExecutionKind.ARENA, "arena-M84-vs-M79", 2, evaluation_fingerprint=SHA_E),
    )
    assert RunState.from_dict(arena.to_dict()) == arena
    assert RunState.from_dict(state(state=BusinessState.STOPPED).to_dict()).state is BusinessState.STOPPED


def test_run_state_cannot_self_declare_commit_without_commit_reference():
    with pytest.raises(ValueError, match="commit reference"):
        state(state=BusinessState.GENERATION_COMMITTED)
    committed = state(
        state=BusinessState.GENERATION_COMMITTED,
        last_committed_checkpoint=cp("lineage-1", 84),
        generation_commit=artifact("transactions/generation-0084.commit.json", SHA_F),
    )
    assert committed.generation_commit.path.endswith("commit.json")


@pytest.mark.parametrize("soft_stop", ["false", 0, 1, None, [], {}])
def test_run_state_parser_rejects_non_boolean_soft_stop(soft_stop):
    payload = state().to_dict()
    payload["soft_stop_requested"] = soft_stop
    with pytest.raises(ValueError, match="soft_stop_requested.*boolean"):
        RunState.from_dict(payload)


def test_run_state_parser_rejects_malformed_applied_amendment():
    payload = state().to_dict()
    payload["applied_amendments"] = ["not-an-object"]
    with pytest.raises(ValueError, match="applied amendment"):
        RunState.from_dict(payload)


def test_run_state_parser_rejects_malformed_present_queued_transition():
    payload = state().to_dict()
    payload["queued_transition"] = "not-an-object"
    with pytest.raises(ValueError, match="queued_transition"):
        RunState.from_dict(payload)


def identity(**overrides) -> EvaluationIdentity:
    args = dict(
        candidate=cp("candidate", 88, SHA_A),
        reference=cp("reference", 83, SHA_B),
        games=128,
        master_seed=20260918,
        startset=StartsetRef("paired-v1", artifact("startsets/paired-v1.json", SHA_C), SHA_D),
        scientific_contract={"mcts_simulations": 64, "noise": False, "temperature": 0, "komi": 0.5},
        execution_contract={"workers": 16, "inference_batch_cap": 64, "wait_ms": 1},
        workload={"paired_starts": True, "color_swap": True},
    )
    args.update(overrides)
    return EvaluationIdentity(**args)


def test_evaluation_identity_complete_contract_controls_fingerprint():
    base = identity()
    assert EvaluationIdentity.from_dict(base.to_dict()).fingerprint == base.fingerprint
    assert identity(candidate=cp("candidate", 88, SHA_E)).fingerprint != base.fingerprint
    assert identity(reference=cp("reference", 83, SHA_E)).fingerprint != base.fingerprint
    assert identity(master_seed=99).fingerprint != base.fingerprint
    assert identity(startset=StartsetRef("paired-v2", artifact("startsets/paired-v2.json", SHA_C), SHA_E)).fingerprint != base.fingerprint
    assert identity(scientific_contract={"mcts_simulations": 128}).fingerprint != base.fingerprint
    assert identity(execution_contract={"workers": 8, "inference_batch_cap": 32}).fingerprint != base.fingerprint


def test_invalid_technical_critical_results_are_not_reusable():
    assert reusable_scientific_result(EvaluationValidity.VALID)
    assert not reusable_scientific_result(EvaluationValidity.INVALID)
    assert not reusable_scientific_result(EvaluationValidity.TECHNICAL)
    assert not reusable_scientific_result(EvaluationValidity.CRITICAL)
